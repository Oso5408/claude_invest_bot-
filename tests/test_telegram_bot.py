import json

import pandas as pd

from btcbot import telegram_bot as tb
from btcbot.paper import PaperConfig
from tests.test_monitor import _run_paper
from tests.test_paper import _minutes


def _accounts(tmp_path):
    m = _minutes()
    _, now = _run_paper(tmp_path / "paper-a", m, PaperConfig(history_days=6, allow_short=False))
    ai = tmp_path / "ai-hl"
    ai.mkdir()
    (ai / "state.json").write_text(json.dumps({
        "cash": 195.0, "next_id": 2, "api_cost": 0.42, "positions": [
            {"id": 1, "coin": "ETH", "side": 1, "qty": 0.02, "entry": 3000.0, "stop": 2900.0, "take_profit": 3300.0,
             "margin": 20.0, "leverage": 3.0, "strategy": "trend", "fees": 0.03, "funding": 0.0}]}))
    pd.DataFrame([{"time": str(now), "equity": 196.5, "cash": 195.0, "open": 1}]).to_csv(ai / "equity.csv", index=False)
    pd.DataFrame([{"time": str(now), "slot": "x", "new_trades": 1, "summary": "long ETH on trend",
                   "error": "", "cost_usd": 0.05}]).to_csv(ai / "decisions.csv", index=False)
    pd.DataFrame([{"id": 0, "coin": "BTC", "side": "short", "strategy": "whale", "closed": str(now),
                   "pnl_usd": -1.2, "exit_reason": "stop"}]).to_csv(ai / "trades.csv", index=False)
    return now


def test_every_command_answers(tmp_path):
    now = _accounts(tmp_path)
    for name in ["balance", "positions", "trades 2", "signals", "ai", "vm", "help"]:
        out = tb.answer("/" + name, tmp_path, now)
        assert "出錯" not in out, out
    assert "[paper-a]" in tb.answer("/balance", tmp_path, now)
    assert "ETH" in tb.answer("/positions", tmp_path, now)
    assert "long ETH on trend" in tb.answer("/ai", tmp_path, now)
    health = tb.cmd_health(tmp_path, now, ping=lambda: "正常")
    assert "🟢 [paper-a]" in health and "🟢 [ai-hl]" in health
    late = tb.cmd_health(tmp_path, now + pd.Timedelta(hours=1), ping=lambda: "正常")
    assert "🔴 [paper-a]" in late
    assert "唔識 /rm" in tb.answer("/rm -rf /", tmp_path, now)
    assert tb.answer("/balance@my_bot", tmp_path, now).startswith("💰")


def test_balance_matches_state(tmp_path):
    _accounts(tmp_path)
    st = json.loads((tmp_path / "paper-a" / "state.json").read_text())
    equity, price = tb.paper_equity(tmp_path / "paper-a", st)
    assert price is not None and abs(equity - st["collateral"] - st["qty"] * (price - st["entry"])) < 1e-9
    assert tb.max_drawdown(pd.Series([100, 120, 90, 130])) == -0.25


class FakeSession:
    def __init__(self, updates):
        self.updates, self.sent = updates, []

    def post(self, url, json, timeout):
        method = url.rsplit("/", 1)[1]
        result = True
        if method == "getUpdates":
            result = [u for u in self.updates if u["update_id"] >= json.get("offset", 0)]
        elif method == "sendMessage":
            self.sent.append((json["chat_id"], json["text"]))
        return type("R", (), {"json": lambda s: {"ok": True, "result": result}, "status_code": 200})()


def test_only_owner_chat_is_answered_and_offset_saved(tmp_path):
    _accounts(tmp_path)
    upd = [{"update_id": 7, "message": {"chat": {"id": 999}, "text": "/balance"}},
           {"update_id": 8, "message": {"chat": {"id": 123}, "text": "/help"}}]
    s = FakeSession(upd)
    bot = tb.Bot("tok", "123", tmp_path, session=s)
    assert bot.poll_once() == 2
    assert [c for c, _ in s.sent] == ["123"] and "/balance" in s.sent[0][1]
    assert (tmp_path / "monitor" / "bot_offset").read_text() == "9"
    assert bot.poll_once() == 0  # nothing answered twice
