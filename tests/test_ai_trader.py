import json
import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from btcbot import ai_trader as ai


class FakeMarket:
    def __init__(self, price=1.0):
        self.price = {"ADA": price, "BTC": 60000.0, "ETH": 3000.0, "SOL": 150.0}
        self.bars = {}  # coin -> 1m DataFrame

    def mids(self):
        return dict(self.price)

    def candles(self, coin, interval, start, end):
        df = self.bars.get(coin)
        if df is None:
            idx = pd.date_range(end - pd.Timedelta(hours=1), end, freq="1h", tz="UTC", name="open_time")
            p = self.price[coin]
            return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)
        return df[(df.index >= start) & (df.index <= end)]

    def funding(self, coin, start, end):
        idx = pd.date_range(start.ceil("1h"), end, freq="1h", tz="UTC")
        return pd.Series(0.0001, index=idx)

    def fear_greed(self):
        return {"2026-10-13": 40.0}

    def leaderboard(self):
        return [{"address": "0xabc", "account_value": 5e6, "pnl_30d": 1e6, "roi_30d": 0.2}]

    def positions(self, address):
        return [{"coin": "ETH", "size": 100.0, "entry": 2900.0, "value_usd": 300000}]


T0 = datetime(2026, 10, 14, 0, 2, tzinfo=timezone.utc)  # 09:02 JST


class ScriptBrain:
    """Calls the handlers like Claude would, without the API."""

    def __init__(self, script):
        self.script = script
        self.cost = 0.0
        self.calls = 0

    def run(self, context, handlers):
        self.calls += 1
        self.cost += 0.3
        self.results = []
        for name, kwargs in self.script:
            try:
                self.results.append(handlers[name](**kwargs))
            except Exception as e:
                self.results.append(e)
        return "done"

    def write(self, system, text):
        self.cost += 0.01
        return "review: idea wrong"


def order(**kw):
    base = dict(coin="ADA", side="long", margin_usd=20, leverage=3, stop_price=0.95, take_profit_price=0,
                max_hold_hours=0, strategy="test", reason="test")
    return ("place_order", {**base, **kw})


def test_limits_are_enforced_in_code(tmp_path):
    brain = ScriptBrain([order(margin_usd=25), order(leverage=4), order(stop_price=1.1), order(stop_price=0.5),
                         order(coin="DOGE"), order(), order(side="short", stop_price=1.05), order()])
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    t.step(T0)
    r = brain.results
    assert all(isinstance(x, Exception) for x in r[:5])
    assert r[5]["opened"] == 1 and r[6]["opened"] == 2
    assert "daily limit" in str(r[7])  # third trade of the day refused
    pos = t.state["positions"][0]
    assert pos["qty"] * pos["entry"] == pytest.approx(60.0)  # 20 USD x 3
    assert len(t.state["positions"]) == 2


def test_decides_once_per_slot_and_new_day_resets(tmp_path):
    brain = ScriptBrain([order()])
    m = FakeMarket()
    t = ai.AITrader(tmp_path, m, brain)
    t.step(T0)
    t.step(T0.replace(minute=7))
    assert brain.calls == 1  # same slot, no second call
    t.step(T0.replace(hour=12, minute=2))  # 21:02 JST, second slot
    assert brain.calls == 2 and t.entries_left() == 0
    t.step(T0.replace(day=15, minute=2))  # next Japan day
    assert brain.calls == 3 and t.entries_left() == 1


def test_stop_loss_closes_and_review_is_logged(tmp_path):
    m = FakeMarket()
    brain = ScriptBrain([order(stop_price=0.95, take_profit_price=1.2)])
    t = ai.AITrader(tmp_path, m, brain)
    t.step(T0)
    idx = pd.date_range(pd.Timestamp(T0).floor("min"), periods=10, freq="1min", name="open_time")
    lows = [1.0] * 5 + [0.94] + [1.0] * 4
    m.bars["ADA"] = pd.DataFrame({"open": 1.0, "high": 1.0, "low": lows, "close": 1.0, "volume": 1.0}, index=idx)
    st = t.step(T0.replace(minute=12))
    assert "stop_loss" in st["action"] and not t.state["positions"]
    tr = pd.read_csv(tmp_path / "trades.csv")
    assert tr["pnl_usd"].iloc[0] < -2.9  # about -5% of 60 USD plus costs
    assert tr["review"].iloc[0] == "review: idea wrong"
    assert t.state["api_cost"] == pytest.approx(0.31)


def test_pause_blocks_new_trades(tmp_path):
    (tmp_path / "PAUSE").write_text("")
    brain = ScriptBrain([order()])
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    st = t.step(T0)
    assert brain.calls == 0 and "PAUSE" in st["action"] and not t.state["positions"]


def test_whales_and_report(tmp_path):
    brain = ScriptBrain([("get_whale_positions", {"top": 3}), order()])
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    t.step(T0)
    assert brain.results[0][0]["positions"][0]["coin"] == "ETH"
    out = ai.daily_report(tmp_path, brain, T0.replace(minute=35))
    assert out.read_text() == "review: idea wrong"
    assert "Open trades" in ai.report_text(tmp_path, T0, tmp_path.parent)


class FakeClient:
    """Two API turns: a place_order call, then finish."""

    def __init__(self):
        self.turn = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.turn += 1
        usage = SimpleNamespace(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0,
                                server_tool_use=SimpleNamespace(web_search_requests=1))
        if self.turn == 1:
            blocks = [SimpleNamespace(type="tool_use", id="t1", name="place_order", input=dict(order()[1]))]
        else:
            blocks = [SimpleNamespace(type="tool_use", id="t2", name="finish", input={"summary": "bought ADA"})]
        return SimpleNamespace(stop_reason="tool_use", content=blocks, usage=usage)


def test_brain_tool_loop(tmp_path):
    brain = ai.Brain(client=FakeClient())
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    st = t.step(T0)
    assert "bought ADA" in st["action"] and len(t.state["positions"]) == 1
    d = pd.read_csv(tmp_path / "decisions.csv")
    assert d["new_trades"].iloc[0] == 1 and d["cost_usd"].iloc[0] > 0


def test_models_sonnet_decides_haiku_writes(tmp_path):
    seen = []

    class Rec(FakeClient):
        def create(self, **kw):
            seen.append(kw)
            if kw["model"] == ai.WRITER_MODEL:
                usage = SimpleNamespace(input_tokens=100, output_tokens=10, cache_read_input_tokens=0)
                return SimpleNamespace(stop_reason="end_turn", usage=usage,
                                       content=[SimpleNamespace(type="text", text="report")])
            return super().create(**kw)

    brain = ai.Brain(client=Rec())
    ai.AITrader(tmp_path, FakeMarket(), brain).step(T0)
    ai.daily_report(tmp_path, brain, T0)
    assert seen[0]["model"] == "claude-sonnet-5-5" and seen[0]["fallbacks"] == "default"
    assert seen[-1]["model"] == "claude-haiku-5-5" and "fallbacks" not in seen[-1]


def test_whale_changes():
    old = {"0xa": {"ETH": 100.0, "BTC": 1.0}, "0xb": {"SOL": 50.0}}
    new = {"0xa": {"ETH": -100.0, "BTC": 2.0, "ADA": 50_000.0}, "0xc": {"SOL": 9.0}}
    ch = ai.whale_changes(old, new, {"ETH": 3000.0, "BTC": 60000.0, "ADA": 1.0})
    what = {(c["coin"], c["change"]) for c in ch}
    assert what == {("ETH", "flipped to short"), ("BTC", "added"), ("ADA", "opened long")}
    assert ch[0]["coin"] == "ETH"  # biggest change first; 0xb/0xc left or joined the list, skipped


class ViewBrain(ScriptBrain):
    def run(self, context, handlers):
        self.contexts = getattr(self, "contexts", []) + [context]
        super().run(context, handlers)
        return {"coins": [{"coin": c, "view": "no trade", "strategy": "range", "reason": "flat"} for c in ai.COINS],
                "whales": "big accounts flipped ETH short", "summary": "no trade"}


def test_every_decision_sees_all_coins_and_whale_moves(tmp_path):
    m = FakeMarket()
    brain = ViewBrain([])
    t = ai.AITrader(tmp_path, m, brain)
    t.step(T0)
    ctx = brain.contexts[0]
    assert set(ctx["market_overview"]) == set(ai.COINS)
    assert {"change_15m", "change_1h", "change_4h"} <= set(ctx["market_overview"]["ADA"])
    assert ctx["big_accounts"]["our_coins"]["ETH"]["accounts_long"] == 1
    m.positions = lambda address: [{"coin": "ETH", "size": -100.0, "entry": 3000.0, "value_usd": 300000}]
    t.step(T0.replace(hour=12))  # 21:02 JST
    assert brain.contexts[1]["big_accounts"]["changes_since_last_decision"][0]["change"] == "flipped to short"
    d = pd.read_csv(tmp_path / "decisions.csv")
    assert len(d) == 2 and "ETH" in d["coin_views"].iloc[0] and d["whales"].iloc[1].startswith("big accounts")
    assert len(pd.read_csv(tmp_path / "whales.csv")) == 2


def test_read_skill_is_logged_and_reported(tmp_path):
    brain = ScriptBrain([("read_skill", {"name": "exit-strategies"}), ("read_skill", {"name": "exit-strategies"}),
                         ("read_skill", {"name": "../state"})])
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    t.step(T0)
    assert "# Exit Strategies" in brain.results[0] and isinstance(brain.results[2], ValueError)
    assert pd.read_csv(tmp_path / "skills.csv")["skills_read"].iloc[0] == "regime-detection,exit-strategies"
    assert "09: regime-detection,exit-strategies" in ai.report_text(tmp_path, T0)


def test_every_skill_file_exists_and_is_listed():
    files = {p.stem for p in ai.SKILLS_DIR.glob("*.md")}
    assert files == set(ai.SKILLS)
    tool = next(x for x in ai.TOOLS if x["name"] == "read_skill")
    assert tool["input_schema"]["properties"]["name"]["enum"] == list(ai.SKILLS)
    assert all(k in ai.TRADER_SYSTEM for k in ai.SKILLS)


def test_old_decisions_csv_gets_new_columns(tmp_path):
    # decisions.csv written before coin_views / whales existed
    pd.DataFrame([{"time": "2026-10-13 12:00:00+00:00", "slot": "2026-10-13T21", "new_trades": 0,
                   "summary": "old", "error": "", "cost_usd": 0.04}]).to_csv(tmp_path / "decisions.csv", index=False)
    ai.AITrader(tmp_path, FakeMarket(), ViewBrain([])).step(T0)
    d = pd.read_csv(tmp_path / "decisions.csv")
    assert list(d["summary"]) == ["old", "no trade"] and "ETH" in d["coin_views"].iloc[1]
    assert "Decisions in the last 24h" in ai.report_text(tmp_path, T0)


def test_default_decides_every_hour(tmp_path):
    brain = ScriptBrain([])
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    for h in range(3):
        for m in (2, 7, 12):
            t.step(T0.replace(hour=h, minute=m))
    assert brain.calls == 3  # once per hour, not every 5 minutes


def test_indicators_on_known_series():
    idx = pd.date_range("2026-01-01", periods=300, freq="1h", tz="UTC")
    close = pd.Series(range(1, 301), index=idx, dtype=float)  # steady uptrend
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0})
    ind = ai.indicators(df)
    assert ind["trend"] == "up" and ind["rsi14"] == 100.0
    assert ind["ema20"] > ind["ema50"] > ind["ema200"]
    assert ind["atr14"] == pytest.approx(2.0, abs=0.01)
    assert ind["donchian20_high"] == 300.0 and ind["volume_ratio20"] == 1.0
    assert ai.completed(df, "1h", idx[-1] + pd.Timedelta("30min")).index[-1] == idx[-2]


def test_overview_has_indicators_per_timeframe(tmp_path):
    brain = ViewBrain([("get_indicators", {"coin": "BTC", "interval": "1d"})])
    ai.AITrader(tmp_path, FakeMarket(), brain).step(T0)
    ov = brain.contexts[0]["market_overview"]["ADA"]["indicators"]
    assert set(ov) == {"4h", "1h", "15m"}
    assert "error" in brain.results[0] or "rsi14" in brain.results[0]


def test_context_has_fear_greed_and_guides_read_today(tmp_path):
    brain = ViewBrain([("read_skill", {"name": "exit-strategies"})])
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    t.step(T0)
    t.step(T0.replace(hour=1, minute=2))
    ctx = brain.contexts[1]
    assert ctx["fear_greed_last_7_days"] == {"2026-10-13": 40.0}
    assert ctx["guides_read_today"] == ["exit-strategies", "regime-detection"]
    assert "regime" in ctx["regime_detection_guide"].lower()
    skills = pd.read_csv(tmp_path / "skills.csv", keep_default_na=False)
    assert all("regime-detection" in x for x in skills["skills_read"])


def test_prompt_says_breakout_entries_are_the_tested_ones():
    assert "never tested" in ai.TRADER_SYSTEM and "fresh 4h breakout" in ai.TRADER_SYSTEM


def test_report_uses_jst_and_shows_equity_after_api_cost(tmp_path):
    t = ai.AITrader(tmp_path, FakeMarket(), ViewBrain([]))
    t.step(T0)
    text = ai.report_text(tmp_path, T0.replace(minute=35))
    assert "times in JST" in text and "10-14 09:02" in text
    assert "equity after API cost: 199.70" in text


def test_market_watches_the_daily_trader_scan(tmp_path, monkeypatch):
    scan = [{"address": "0xsteady", "account_value": 50_000, "pnl_30d": 5998, "roi_30d": 0.12,
             "pnl_by_coin": {"ADA": 5998}}]
    f = tmp_path / "latest.json"
    f.write_text(json.dumps(scan))
    m = ai.HLMarket(cache_dir=tmp_path, traders_file=f)
    assert m.leaderboard() == scan
    old = time.time() - 49 * 3600
    os.utime(f, (old, old))  # stale scan: back to the plain leaderboard
    monkeypatch.setattr(m.session, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("leaderboard")))
    with pytest.raises(RuntimeError, match="leaderboard"):
        m.leaderboard()
