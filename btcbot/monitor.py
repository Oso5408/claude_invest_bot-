"""Independent watchdog for the paper accounts. Runs from its own cron line, never inside the trader.

    python -m btcbot.monitor watch     # every 5 min: offline / errors / bad data / leverage over the cap
    python -m btcbot.monitor check     # every hour: missed signals (re-runs the strategy on fresh candles)
    python -m btcbot.monitor report    # once a day: summary of the last 24h, also sent to Telegram

Accounts are found as data/paper-*/state.json; put a file called NO_MONITOR in a folder to skip it.
The cron log of account data/paper-x is expected at data/paper-x.log (as in docs/gcp-setup.md).
Alerts go to Telegram (btcbot/alerts.py). Each problem is sent once when it starts, repeated every
few hours while it lasts, and once more when it clears. Monitor state lives in data/monitor/.

The missed-signal check recomputes the 4h breakout signals from candles it downloads itself and
compares them with signals.csv, which the paper trader writes for every 4h candle it processes.
A signal on a candle the trader never processed, or one the trader did not see, is an alert.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from btcbot import alerts, risk
from btcbot.breakout import BreakoutConfig, compute_features
from btcbot.data import fetch_fng, resample

OFFLINE_MINUTES = 15  # three missed 5-minute runs
DATA_CHECK_RUNS = 3  # consecutive runs that skipped a candle because of bad data
REPEAT_HOURS = 6
GRACE = pd.Timedelta(minutes=20)  # give the trader time to process a closed candle (and retry bad data)
OK_RESULTS = {"entered", "in_position", "no_signal"}
SKIP_RESULTS = {"skipped_size", "claude_skip"}  # sizing or Claude decided against it: reported, not an alert
ERROR_WORDS = ("Traceback", "Error", "Exception")


# ---------- helpers ----------
def account_folders(data_dir: Path) -> list[Path]:
    return sorted(p.parent for p in data_dir.glob("paper*/state.json") if not (p.parent / "NO_MONITOR").exists())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def venue(cfg: dict) -> str:
    return "gmo" if "_" in cfg.get("symbol", "ADA_JPY") else "hl"


class MonitorState:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}
        self.data.setdefault("active", {})
        self.data.setdefault("log_offsets", {})
        self.data.setdefault("seen", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["seen"] = self.data["seen"][-2000:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


def raise_problems(state: MonitorState, problems: dict[str, str], now: datetime, scope: str) -> list[str]:
    """Problems found this run -> messages to send. Keys are stable ids; values are the text.
    Only keys starting with `scope` are cleared when they are no longer found."""
    out = []
    active = state.data["active"]
    for key, text in problems.items():
        prev = active.get(key)
        if prev is None or now - datetime.fromisoformat(prev["sent"]) >= timedelta(hours=REPEAT_HOURS):
            out.append(("🔴 " if prev is None else "🔁 仍然未解決：") + text)
            active[key] = {"since": prev["since"] if prev else now.isoformat(), "sent": now.isoformat(), "text": text}
    for key in [k for k in active if k.startswith(scope) and k not in problems]:
        out.append(f"🟢 已恢復：{active.pop(key)['text']}")
    return out


# ---------- watch: every 5 minutes ----------
def watch_folder(folder: Path, state: MonitorState, now: datetime) -> dict[str, str]:
    name = folder.name
    problems: dict[str, str] = {}
    try:
        st = json.loads((folder / "state.json").read_text())
    except Exception as e:
        return {f"watch:{name}:state": f"[{name}] state.json 讀唔到：{e}。唔好再落新單，先檢查檔案。"}

    eq = read_csv(folder / "equity.csv")
    if eq.empty:
        last_run = pd.Timestamp(st["last_check"]) if st.get("last_check") else None
    else:
        last_run = pd.Timestamp(eq["time"].iloc[-1])
    if last_run is None or pd.Timestamp(now) - last_run > pd.Timedelta(minutes=OFFLINE_MINUTES):
        ago = "從未行過" if last_run is None else f"上次 {last_run.tz_convert('Asia/Tokyo'):%m-%d %H:%M} JST"
        problems[f"watch:{name}:offline"] = f"[{name}] 模擬盤超過 {OFFLINE_MINUTES} 分鐘冇更新（{ago}）。檢查 crontab 同 {name}.log。"

    if len(eq) >= DATA_CHECK_RUNS:
        recent = eq["action"].tail(DATA_CHECK_RUNS).astype(str)
        if recent.str.contains("data check").all():
            problems[f"watch:{name}:data"] = f"[{name}] 連續 {DATA_CHECK_RUNS} 次數據唔完整或過時，暫停入市：{recent.iloc[-1]}"

    if not eq.empty and st.get("qty"):
        price = float(eq["price"].iloc[-1])
        equity = float(eq["equity"].iloc[-1])
        cap = min(float(st.get("config", {}).get("max_leverage", risk.MAX_LEVERAGE)), risk.MAX_LEVERAGE)
        lev = abs(st["qty"]) * price / equity if equity > 0 else float("inf")
        if lev > cap * 1.05:
            problems[f"watch:{name}:leverage"] = f"[{name}] 槓桿 {lev:.2f}x 超過上限 {cap:g}x（倉位 {st['qty']:+g} @ {price:g}）。"

    log = folder.parent / f"{name}.log"
    if log.exists():
        offsets = state.data["log_offsets"]
        start = offsets.get(name, 0)
        size = log.stat().st_size
        if size < start:  # log was rotated or cleared
            start = 0
        with log.open("rb") as fh:
            fh.seek(start)
            new = fh.read().decode("utf-8", "replace")
        offsets[name] = size
        errors = [ln for ln in new.splitlines() if any(w in ln for w in ERROR_WORDS)]
        if errors:
            key = f"watch:{name}:errors:{now:%Y%m%d%H%M}"  # a new event each time, never "recovers"
            problems[key] = f"[{name}] log 有 {len(errors)} 行錯誤，最後一行：{errors[-1][:300]}"
    return problems


def watch(data_dir: Path, now: datetime | None = None, send=alerts.send) -> list[str]:
    now = now or datetime.now(timezone.utc)
    state = MonitorState(data_dir / "monitor" / "state.json")
    problems: dict[str, str] = {}
    folders = account_folders(data_dir)
    if not folders:
        problems["watch:none"] = f"搵唔到任何模擬帳戶（{data_dir}/paper*/state.json）。"
    for folder in folders:
        problems.update(watch_folder(folder, state, now))
    # error lines are one-off events: send them, but do not keep them as active problems
    msgs = raise_problems(state, problems, now, "watch:")
    for key in [k for k in state.data["active"] if ":errors:" in k]:
        state.data["active"].pop(key)
    state.data["last_watch"] = now.isoformat()
    state.save()
    for m in msgs:
        send(m)
    return msgs


# ---------- check: missed signals ----------
def make_feed(v: str, symbol: str):
    from btcbot.paper import GMOFeed, HyperliquidFeed
    return HyperliquidFeed(symbol) if v == "hl" else GMOFeed(symbol)


def expected_signals(hourly: pd.DataFrame, cfg: dict, fng: pd.Series | None, now: datetime) -> pd.DataFrame:
    """The strategy's own signals on closed 4h candles, recomputed from scratch."""
    hourly = hourly[hourly.index + pd.Timedelta(hours=1) <= pd.Timestamp(now)]
    tf = pd.Timedelta(cfg.get("timeframe", "4h"))
    bars = resample(hourly, cfg.get("timeframe", "4h"))
    bars = bars[bars.index + tf <= pd.Timestamp(now)]
    strat = BreakoutConfig(allow_short=cfg.get("allow_short", True),
                           max_leverage=cfg.get("max_leverage", risk.MAX_LEVERAGE),
                           fng_short_max=cfg.get("fng_short_max", 100))
    use_fng = fng if strat.fng_short_max < 100 else None
    f = compute_features(bars, strat, use_fng)
    if use_fng is not None:  # the trader takes no short when the index is unknown
        f.loc[f["fng"].isna(), "short_signal"] = False
    f["direction"] = f["long_signal"].astype(int) - f["short_signal"].astype(int)
    return f


def compare(folder: Path, expected: pd.DataFrame, since: pd.Timestamp, now: datetime) -> list[dict]:
    """One row per closed candle in the window: what the strategy says vs what the trader did."""
    sig = read_csv(folder / "signals.csv")
    if sig.empty:
        return []
    sig["bar"] = pd.to_datetime(sig["bar"], utc=True)
    sig = sig.drop_duplicates("bar", keep="last").set_index("bar")
    first = sig.index.min()
    tf = expected.index[1] - expected.index[0] if len(expected) > 1 else pd.Timedelta(hours=4)
    rows = []
    window = expected[(expected.index >= max(since, first)) & (expected.index + tf + GRACE <= pd.Timestamp(now))]
    for bar, r in window.iterrows():
        want = int(r["direction"])
        if bar not in sig.index:
            status = "missed_unprocessed" if want else "unprocessed"
            rows.append({"bar": bar, "want": want, "got": None, "result": None, "status": status})
            continue
        p = sig.loc[bar]
        got = int(bool(p["long_signal"])) - int(bool(p["short_signal"]))
        result = str(p["result"])
        if got != want:
            status = "mismatch"
        elif not want or result in OK_RESULTS:
            status = "ok"
        elif result in SKIP_RESULTS:
            status = "skipped"
        else:
            status = "not_executed"  # paused, duplicate, no_atr, anything unexpected
        rows.append({"bar": bar, "want": want, "got": got, "result": result, "status": status,
                     "close": r["close"], "entry_px": p.get("entry_px")})
    return rows


def describe(name: str, row: dict) -> str:
    side = {1: "做多", -1: "做空", 0: "冇訊號", None: "?"}
    bar = pd.Timestamp(row["bar"]).tz_convert("Asia/Tokyo")
    what = {
        "missed_unprocessed": f"策略有{side[row['want']]}訊號，但模擬盤冇處理呢條 K 線（停咗機或者數據有問題）",
        "mismatch": f"策略計到{side[row['want']]}，模擬盤當時計到{side[row['got']]}（數據唔一致）",
        "not_executed": f"有{side[row['want']]}訊號但冇入市，原因：{row['result']}",
        "skipped": f"有{side[row['want']]}訊號，因為 {row['result']} 冇入市",
    }[row["status"]]
    return f"[{name}] 漏單檢查 {bar:%m-%d %H:%M} JST 4h K 線：{what}"


ALERT_STATUSES = {"missed_unprocessed", "mismatch", "not_executed"}


def run_check(data_dir: Path, now: datetime | None = None, days: float = 2, send=alerts.send,
              feeds: dict | None = None, fng: pd.Series | None = None) -> dict[str, list[dict]]:
    now = now or datetime.now(timezone.utc)
    state = MonitorState(data_dir / "monitor" / "state.json")
    seen = set(state.data["seen"])
    since = pd.Timestamp(now) - pd.Timedelta(days=days)
    feeds = feeds if feeds is not None else {}
    cache: dict[tuple, pd.DataFrame] = {}
    results: dict[str, list[dict]] = {}
    msgs = []
    for folder in account_folders(data_dir):
        name = folder.name
        try:
            cfg = json.loads((folder / "state.json").read_text()).get("config", {})
            key = (venue(cfg), cfg.get("symbol", "ADA_JPY"))
            if key not in cache:
                feed = feeds.get(key) or make_feed(*key)
                cache[key] = feed.candles("1hour", now, int(days) + 22)
            if cfg.get("fng_short_max", 100) < 100 and fng is None:
                fng = fetch_fng()
            rows = compare(folder, expected_signals(cache[key], cfg, fng, now), since, now)
        except Exception as e:
            msgs.append(f"⚠️ [{name}] 漏單檢查做唔到：{e}")
            continue
        results[name] = rows
        for row in rows:
            k = f"check:{name}:{pd.Timestamp(row['bar']).isoformat()}:{row['status']}"
            if row["status"] in ALERT_STATUSES and k not in seen:
                msgs.append("🔴 " + describe(name, row))
                seen.add(k)
            elif row["status"] == "skipped" and k not in seen:
                msgs.append("ℹ️ " + describe(name, row))
                seen.add(k)
    state.data["seen"] = sorted(seen)
    state.data["last_check"] = now.isoformat()
    state.save()
    for m in msgs:
        send(m)
    return results


# ---------- report: once a day ----------
def report(data_dir: Path, now: datetime | None = None, send=alerts.send, **check_kwargs) -> Path:
    now = now or datetime.now(timezone.utc)
    checks = run_check(data_dir, now, days=1, send=send, **check_kwargs)
    state = MonitorState(data_dir / "monitor" / "state.json")
    day_ago = pd.Timestamp(now) - pd.Timedelta(days=1)
    lines = [f"📋 每日監察報告 {now.astimezone(timezone(timedelta(hours=9))):%Y-%m-%d %H:%M} JST", ""]
    for folder in account_folders(data_dir):
        name = folder.name
        eq = read_csv(folder / "equity.csv")
        if not eq.empty:
            eq["time"] = pd.to_datetime(eq["time"], utc=True, format="ISO8601")
            today = eq[eq["time"] > day_ago]
        else:
            today = eq
        runs = len(today)
        line = f"[{name}] 運行 {runs}/288 次"
        if runs:
            start = eq[eq["time"] <= day_ago]["equity"].iloc[-1] if (eq["time"] <= day_ago).any() else today["equity"].iloc[0]
            end = today["equity"].iloc[-1]
            line += f"，資金 {start:,.0f} → {end:,.0f}（{end / start - 1:+.1%}）"
            bad = int(today["action"].astype(str).str.contains("data check").sum())
            if bad:
                line += f"，數據問題 {bad} 次"
        rows = checks.get(name, [])
        sig = [r for r in rows if r["want"]]
        if rows:
            missed = sum(r["status"] in ALERT_STATUSES for r in rows)
            line += f"，4h K 線 {len(rows)} 條，訊號 {len(sig)} 個，漏單 {missed}"
            for r in rows:
                if r["result"] == "entered" and pd.notna(r.get("entry_px")):
                    slip = r["want"] * (float(r["entry_px"]) / r["close"] - 1) * 1e4
                    line += f"，入市價對收市價 {slip:+.0f} bps"
        elif name not in checks:
            line += "，漏單檢查失敗"
        else:
            line += "，未有可檢查嘅 K 線（signals.csv 未有紀錄）"
        lines.append(line)
    active = [v["text"] for k, v in state.data["active"].items()]
    lines += ["", "未解決問題：" + ("冇" if not active else "")] + [f"- {t}" for t in active]
    text = "\n".join(lines)
    out = data_dir / "monitor" / f"report-{now:%Y-%m-%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    send(text)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Independent watchdog for the paper trading accounts")
    p.add_argument("command", choices=["watch", "check", "report"])
    p.add_argument("--data", default="data", help="folder that holds the paper-* account folders")
    p.add_argument("--days", type=float, default=2, help="check: how far back to compare signals")
    a = p.parse_args(argv)
    data = Path(a.data)
    if a.command == "watch":
        watch(data)
    elif a.command == "check":
        run_check(data, days=a.days)
    else:
        print(report(data))


if __name__ == "__main__":
    main()
