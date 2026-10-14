"""Telegram commands for the paper accounts and the VM. Read-only: no command changes a file or an order.

    python -m btcbot.telegram_bot run       # long-running; on the VM it runs as a systemd service
    python -m btcbot.telegram_bot try /balance   # print one command's answer here, without Telegram

Uses the same bot and environment as the alerts (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID in ~/.telegram_env).
Only messages from TELEGRAM_CHAT_ID are answered; anything else is ignored and logged. The bot asks
Telegram for new messages (long polling), so the VM opens no port. Setup: docs/gcp-setup.md section 13.

Commands: /balance /positions /trades [n] /signals /health /ai /vm /report /help
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from btcbot import alerts, monitor
from btcbot.paper import paused

JST = timezone(timedelta(hours=9))
POLL_SECONDS = 50
ROOT = Path(__file__).resolve().parent.parent

COMMANDS = {
    "balance": "每個帳戶嘅資金、回報同最大回撤",
    "positions": "而家嘅倉位、入場價、止損同未實現盈虧",
    "trades": "最近幾單交易（例如 /trades 5）",
    "signals": "每個帳戶最近一支 4h 訊號",
    "health": "帳戶有冇準時行、未解決警報、API 通唔通",
    "ai": "AI 帳戶最新決定",
    "vm": "VM 開機時間、CPU、記憶體、硬碟、程式版本",
    "report": "即刻出一份每日監察報告",
    "help": "指令清單",
}


# ---------- reading the account folders ----------
def paper_folders(data: Path) -> list[Path]:
    return monitor.account_folders(data)


def ai_folder(data: Path) -> Path | None:
    f = data / "ai-hl"
    return f if (f / "state.json").exists() else None


def load_state(folder: Path) -> dict:
    return json.loads((folder / "state.json").read_text())


def currency(st: dict) -> str:
    return "USD" if monitor.venue(st.get("config", {})) == "hl" else "JPY"


def money(x: float, cur: str) -> str:
    return f"{x:,.2f} {cur}" if cur == "USD" else f"{x:,.0f} {cur}"


def ago(t, now: datetime) -> str:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return "從未"
    mins = (pd.Timestamp(now) - pd.Timestamp(t)).total_seconds() / 60
    if mins < 90:
        return f"{mins:.0f} 分鐘前"
    if mins < 48 * 60:
        return f"{mins / 60:.1f} 小時前"
    return f"{mins / 1440:.1f} 日前"


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.astype(float)
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


def paper_equity(folder: Path, st: dict) -> tuple[float, float | None]:
    """Equity at the last price the trader saw (state.json has the unrounded collateral)."""
    eq = monitor.read_csv(folder / "equity.csv")
    price = float(eq["price"].iloc[-1]) if not eq.empty else None
    if price is None:
        return float(st["collateral"]), None
    return float(st["collateral"]) + float(st["qty"]) * (price - float(st["entry"])), price


# ---------- commands ----------
def cmd_balance(data: Path, now: datetime, arg: str = "") -> str:
    lines = ["💰 資金"]
    for folder in paper_folders(data):
        st = load_state(folder)
        cur = currency(st)
        start = float(st.get("config", {}).get("initial_jpy", st["collateral"]))
        equity, _ = paper_equity(folder, st)
        eq = monitor.read_csv(folder / "equity.csv")
        dd = max_drawdown(pd.concat([pd.Series([start]), eq["equity"]]) if not eq.empty else pd.Series([start]))
        tr = monitor.read_csv(folder / "trades.csv")
        lines.append(f"[{folder.name}] {money(equity, cur)}（{equity / start - 1:+.1%}），"
                     f"最大回撤 {dd:.1%}，已平倉 {len(tr)} 單")
    ai = ai_folder(data)
    if ai:
        from btcbot.ai_trader import INITIAL_USD
        eq = monitor.read_csv(ai / "equity.csv")
        st = load_state(ai)
        equity = float(eq["equity"].iloc[-1]) if not eq.empty else float(st["cash"])
        dd = max_drawdown(pd.concat([pd.Series([INITIAL_USD]), eq["equity"]]) if not eq.empty else pd.Series([INITIAL_USD]))
        lines.append(f"[ai-hl] {money(equity, 'USD')}（{equity / INITIAL_USD - 1:+.1%}），最大回撤 {dd:.1%}，"
                     f"Claude 費用 {st.get('api_cost', 0.0):.2f} USD")
    if len(lines) == 1:
        lines.append(f"搵唔到任何帳戶（{data}）")
    return "\n".join(lines)


def cmd_positions(data: Path, now: datetime, arg: str = "") -> str:
    lines = ["📈 倉位"]
    for folder in paper_folders(data):
        st = load_state(folder)
        cur = currency(st)
        if not st.get("side"):
            lines.append(f"[{folder.name}] 冇倉" + ("（PAUSE 中）" if paused(folder) else ""))
            continue
        equity, price = paper_equity(folder, st)
        pnl = float(st["qty"]) * ((price or float(st["entry"])) - float(st["entry"]))
        side = "多" if st["side"] > 0 else "空"
        lines.append(f"[{folder.name}] {side} {abs(float(st['qty'])):g} @ {float(st['sig_entry']):.5g}，"
                     f"現價 {price if price is not None else '?'}，止損 {float(st['stop']):.5g}，"
                     f"未實現 {money(pnl, cur)}，入場 {ago(st.get('entry_time'), now)}")
    ai = ai_folder(data)
    if ai:
        pos = load_state(ai).get("positions", [])
        if not pos:
            lines.append("[ai-hl] 冇倉")
        for p in pos:
            side = "多" if p["side"] > 0 else "空"
            lines.append(f"[ai-hl] #{p['id']} {p['coin']} {side} {p['leverage']:g}x 保證金 {p['margin']:g} USD "
                         f"@ {p['entry']:.6g}，止損 {p['stop']:.6g}，止賺 {p.get('take_profit', '-')}，"
                         f"策略 {p.get('strategy', '')}")
    return "\n".join(lines)


def cmd_trades(data: Path, now: datetime, arg: str = "") -> str:
    n = max(1, min(int(arg), 20)) if arg.strip().isdigit() else 3
    lines = [f"🧾 每個帳戶最近 {n} 單"]
    for folder in paper_folders(data):
        cur = currency(load_state(folder))
        tr = monitor.read_csv(folder / "trades.csv")
        if tr.empty:
            lines.append(f"[{folder.name}] 未有交易")
            continue
        for _, r in tr.tail(n).iterrows():
            t = pd.Timestamp(r["exit_time"]).tz_convert(JST)
            lines.append(f"[{folder.name}] {t:%m-%d %H:%M} {'多' if r['side'] > 0 else '空'} "
                         f"{r['entry_px']:.5g}→{r['exit_px']:.5g} {r['return']:+.2%}（{r['reason']}），"
                         f"之後資金 {money(r['equity_after'], cur)}")
    ai = ai_folder(data)
    if ai:
        tr = monitor.read_csv(ai / "trades.csv")
        if tr.empty:
            lines.append("[ai-hl] 未有交易")
        for _, r in tr.tail(n).iterrows():
            t = pd.Timestamp(r["closed"]).tz_convert(JST)
            lines.append(f"[ai-hl] {t:%m-%d %H:%M} {r['coin']} {r['side']} {r['pnl_usd']:+.2f} USD "
                         f"（{r['exit_reason']}，{r['strategy']}）")
    return "\n".join(lines)


def cmd_signals(data: Path, now: datetime, arg: str = "") -> str:
    lines = ["📡 最近 4h 訊號"]
    for folder in paper_folders(data):
        sig = monitor.read_csv(folder / "signals.csv")
        if sig.empty:
            lines.append(f"[{folder.name}] 未有紀錄")
            continue
        r = sig.iloc[-1]
        bar = pd.Timestamp(r["bar"]).tz_convert(JST)
        want = "做多" if r["long_signal"] else "做空" if r["short_signal"] else "冇訊號"
        hits = sig[sig["long_signal"].astype(bool) | sig["short_signal"].astype(bool)]
        last_hit = pd.Timestamp(hits["bar"].iloc[-1]).tz_convert(JST).strftime("%m-%d %H:%M") if len(hits) else "未有"
        lines.append(f"[{folder.name}] {bar:%m-%d %H:%M} K 線：{want}，結果 {r['result']}；上一個訊號 {last_hit}")
    return "\n".join(lines)


def hyperliquid_ok() -> str:
    try:
        t0 = time.time()
        resp = requests.post("https://api.hyperliquid.xyz/info", json={"type": "allMids"}, timeout=10)
        resp.raise_for_status()
        return f"正常（{(time.time() - t0) * 1000:.0f} ms，ADA {float(resp.json()['ADA']):.4f}）"
    except Exception as e:
        return f"失敗：{str(e)[:150]}"


def cmd_health(data: Path, now: datetime, arg: str = "", ping=hyperliquid_ok) -> str:
    lines = ["🩺 健康檢查"]
    folders = paper_folders(data) + ([ai_folder(data)] if ai_folder(data) else [])
    for folder in folders:
        eq = monitor.read_csv(folder / "equity.csv")
        last = eq["time"].iloc[-1] if not eq.empty else load_state(folder).get("last_check")
        late = last is None or pd.Timestamp(now) - pd.Timestamp(last) > pd.Timedelta(minutes=monitor.OFFLINE_MINUTES)
        lines.append(f"{'🔴' if late else '🟢'} [{folder.name}] 上次運行 {ago(last, now)}"
                     + ("，PAUSE 中" if paused(folder) else ""))
    if not folders:
        lines.append(f"🔴 搵唔到任何帳戶（{data}）")
    mon = monitor.MonitorState(data / "monitor" / "state.json").data
    lines.append(f"監察程式上次 watch：{ago(mon.get('last_watch'), now)}")
    active = [v["text"] for v in mon.get("active", {}).values()]
    lines.append("未解決警報：" + ("冇" if not active else ""))
    lines += [f"- {t}" for t in active]
    lines.append(f"Hyperliquid API：{ping()}")
    return "\n".join(lines)


def cmd_ai(data: Path, now: datetime, arg: str = "") -> str:
    ai = ai_folder(data)
    if not ai:
        return "未有 AI 帳戶（data/ai-hl）"
    st = load_state(ai)
    dc = monitor.read_csv(ai / "decisions.csv")
    lines = [f"🤖 AI 帳戶：倉位 {len(st.get('positions', []))} 個，Claude 累計費用 {st.get('api_cost', 0.0):.2f} USD"
             + ("，PAUSE 中" if paused(ai) else "")]
    if dc.empty:
        lines.append("未有決定紀錄")
    else:
        r = dc.iloc[-1]
        t = pd.Timestamp(r["time"]).tz_convert(JST)
        err = r.get("error")
        text = r["summary"] if isinstance(r["summary"], str) and r["summary"] else f"出錯：{err}"
        lines.append(f"最新決定 {t:%m-%d %H:%M}（新開 {int(r['new_trades'])} 單，費用 {r['cost_usd']:.3f} USD）：")
        lines.append(str(text)[:2500])
    return "\n".join(lines)


def run_fixed(args: list[str]) -> str:
    """A fixed command (no shell, no user text in it)."""
    try:
        out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as e:
        return f"?（{e}）"


def cmd_vm(data: Path, now: datetime, arg: str = "") -> str:
    lines = ["🖥 VM"]
    try:
        up = float(Path("/proc/uptime").read_text().split()[0])
        lines.append(f"開機 {up / 86400:.1f} 日")
    except OSError:
        pass
    try:
        l1, l5, l15 = os.getloadavg()
        lines.append(f"負載 {l1:.2f} / {l5:.2f} / {l15:.2f}（{os.cpu_count()} 核）")
    except OSError:
        pass
    try:
        mem = {k: int(v.split()[0]) for k, v in
               (ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)}
        total, avail = mem["MemTotal"] / 1024, mem["MemAvailable"] / 1024
        lines.append(f"記憶體 用咗 {total - avail:.0f} / {total:.0f} MB（{1 - avail / total:.0%}）")
    except (OSError, KeyError, ValueError):
        pass
    du = shutil.disk_usage(ROOT)
    lines.append(f"硬碟 用咗 {du.used / 1e9:.1f} / {du.total / 1e9:.1f} GB（{du.used / du.total:.0%}）")
    data_mb = sum(p.stat().st_size for p in data.rglob("*") if p.is_file()) / 1e6 if data.exists() else 0
    lines.append(f"data 資料夾 {data_mb:.0f} MB")
    lines.append(f"程式版本 {run_fixed(['git', 'log', '-1', '--format=%h %cd %s', '--date=format:%m-%d %H:%M'])}"
                 f"（branch {run_fixed(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])}）")
    lines.append(f"PAUSE 全部帳戶：{'係' if (data / 'PAUSE').exists() else '否'}")
    return "\n".join(lines)


def cmd_report(data: Path, now: datetime, arg: str = "") -> str:
    path = monitor.report(data, now)  # sends the report itself, like the 09:35 one
    return f"報告已經發出，亦儲存喺 {path}"


def cmd_help(data: Path, now: datetime, arg: str = "") -> str:
    return "指令（只係睇，唔會改任何嘢）：\n" + "\n".join(f"/{k} - {v}" for k, v in COMMANDS.items())


HANDLERS = {"balance": cmd_balance, "positions": cmd_positions, "trades": cmd_trades, "signals": cmd_signals,
            "health": cmd_health, "ai": cmd_ai, "vm": cmd_vm, "report": cmd_report, "help": cmd_help,
            "start": cmd_help}


def answer(text: str, data: Path, now: datetime | None = None) -> str:
    """The reply to one message. Never raises."""
    now = now or datetime.now(timezone.utc)
    parts = text.strip().split(maxsplit=1)
    if not parts or not parts[0].startswith("/"):
        return cmd_help(data, now)
    name = parts[0][1:].split("@")[0].lower()  # "/balance@my_bot" in group chats
    handler = HANDLERS.get(name)
    if handler is None:
        return f"唔識 /{name}。\n" + cmd_help(data, now)
    try:
        return handler(data, now, parts[1] if len(parts) > 1 else "")
    except Exception as e:
        return f"/{name} 出錯：{type(e).__name__}: {str(e)[:300]}"


# ---------- Telegram ----------
class Bot:
    def __init__(self, token: str, chat_id: str, data: Path, session=None):
        self.token, self.chat_id, self.data = token, str(chat_id), data
        self.session = session or requests.Session()
        self.offset_path = data / "monitor" / "bot_offset"

    def call(self, method: str, http_timeout: float = 15, **params):
        resp = self.session.post(alerts.API.format(token=self.token, method=method), json=params, timeout=http_timeout)
        body = resp.json()
        if not body.get("ok"):
            raise RuntimeError(f"{method}: {body.get('description', resp.status_code)}")
        return body.get("result")

    def say(self, text: str) -> None:
        for i in range(0, len(text), alerts.MAX_LEN):
            self.call("sendMessage", chat_id=self.chat_id, text=text[i:i + alerts.MAX_LEN],
                      disable_web_page_preview=True)

    def handle(self, update: dict, now: datetime | None = None) -> str | None:
        msg = update.get("message") or update.get("edited_message")
        if not msg or "text" not in msg:
            return None
        chat = str(msg["chat"]["id"])
        if chat != self.chat_id:
            print(f"[{datetime.now(timezone.utc):%F %T}] ignored message from chat {chat}: {msg['text'][:50]!r}",
                  flush=True)
            return None
        print(f"[{datetime.now(timezone.utc):%F %T}] {msg['text'][:50]!r}", flush=True)
        if msg["text"].strip().split("@")[0] == "/report":
            self.say("生成緊報告（要重新下載 K 線，可能要一兩分鐘）…")
        reply = answer(msg["text"], self.data, now)
        self.say(reply)
        return reply

    def poll_once(self) -> int:
        offset = int(self.offset_path.read_text()) if self.offset_path.exists() else None
        params = {"timeout": POLL_SECONDS, "allowed_updates": ["message", "edited_message"]}
        if offset is not None:
            params["offset"] = offset
        updates = self.call("getUpdates", http_timeout=POLL_SECONDS + 15, **params)
        for u in updates:
            # save first: a command that crashes the bot is not repeated after the restart
            self.offset_path.parent.mkdir(parents=True, exist_ok=True)
            self.offset_path.write_text(str(u["update_id"] + 1))
            try:
                self.handle(u)
            except Exception as e:
                print(f"[reply failed: {e}]", flush=True)
        return len(updates)

    def run(self) -> None:
        self.call("setMyCommands", commands=[{"command": k, "description": v} for k, v in COMMANDS.items()])
        print("bot started", flush=True)
        while True:
            try:
                self.poll_once()
            except Exception as e:  # network down: wait and try again, never exit
                print(f"[poll failed: {e}]", flush=True)
                time.sleep(10)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Telegram commands for the paper accounts (read-only)")
    p.add_argument("command", choices=["run", "try"])
    p.add_argument("text", nargs="*", help="try: the command to answer, e.g. /balance")
    p.add_argument("--data", default="data")
    a = p.parse_args(argv)
    data = Path(a.data)
    if a.command == "try":
        print(answer(" ".join(a.text) or "/help", data))
        return
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set (run: . ~/.telegram_env)")
    Bot(token, chat, data).run()


if __name__ == "__main__":
    main()
