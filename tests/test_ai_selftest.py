from datetime import datetime, timezone

import pandas as pd
import pytest

from btcbot import ai_selftest as st
from btcbot import ai_trader as ai

NOW = datetime(2026, 10, 15, 8, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("name", list(st.SCENARIOS))
def test_rule_follower_passes_every_scenario(name):
    """The made-up markets are labelled right: following the tested rules gets every answer."""
    sc = st.SCENARIOS[name]
    out = st.run_scenario(name, st.ScriptedBrain(), now=NOW)
    assert out["passed"], out
    seen = out.get("market_seen")
    if sc["kind"] == "breakout":
        assert seen["close"] > seen["donchian20_high"] and seen["volume_ratio20"] == pytest.approx(sc["volume"], abs=0.05)
    if sc["kind"] in ("breakout", "breakdown"):  # just past the channel, not an obvious move
        edge = seen["donchian20_high"] if sc["kind"] == "breakout" else seen["donchian10_low"]
        assert abs(seen["close"] / edge - 1) < 0.01 and 30 < seen["rsi14"] < 70
    if sc["kind"] == "wick":
        assert seen["close"] < seen["donchian20_high"]
    if out["orders"]:
        assert out["orders_ok"] and out["closed"]


def test_ai_that_trades_everything_fails_the_traps():
    class Eager(st.ScriptedBrain):
        def run(self, context, handlers):
            px = context["prices"]["SOL"]
            handlers["place_order"](coin="SOL", side="long", margin_usd=20, leverage=2, stop_price=px * 0.97,
                                    take_profit_price=0, max_hold_hours=0, strategy="s", reason="r")
            return {"summary": "", "coins": []}
    assert not st.run_scenario("fake-breakout-wick", Eager(), now=NOW)["passed"]
    out = st.run_scenario("quiet-breakout-long", Eager(), now=NOW)
    assert not out["passed"] and out["other_trades"] == ["SOL"]


def test_stop_closes_when_there_is_no_take_profit():
    class NoTP(st.ScriptedBrain):
        def run(self, context, handlers):
            px = context["prices"]["SOL"]
            handlers["place_order"](coin="SOL", side="long", margin_usd=20, leverage=3, stop_price=px * 0.97,
                                    take_profit_price=0, max_hold_hours=0, strategy="s", reason="r")
            return {"summary": "", "coins": []}
    out = st.run_scenario("quiet-market", NoTP("quiet-market"), now=NOW)
    [c] = out["closed"]
    assert c["exit_reason"] == "stop_loss" and c["pnl_usd"] < 0


def test_over_limit_order_from_the_ai_is_rejected():
    class Greedy(st.ScriptedBrain):
        def run(self, context, handlers):
            px = context["prices"]["SOL"]
            for kw in (dict(margin_usd=50, leverage=3), dict(margin_usd=20, leverage=5)):
                with pytest.raises(ValueError):
                    handlers["place_order"](coin="SOL", side="long", stop_price=px * 0.97, take_profit_price=0,
                                            max_hold_hours=0, strategy="s", reason="r", **kw)
            return {"summary": "", "coins": []}
    out = st.run_scenario("quiet-market", Greedy("quiet-market"), now=NOW)
    assert out["orders"] == [] and out["trades_left_today"] == ai.MAX_ENTRIES_PER_DAY


def test_scenario_market_is_made_up_but_scaled_to_live_prices():
    m = st.ScenarioMarket(pd.Timestamp(NOW), "breakdown-short-greed", {"ETH": 4000.0})
    assert m.mids()["ETH"] == pytest.approx(4000.0)
    assert list(m.fear_greed().values())[-1] == 72.0


def test_trailing_stop_follows_the_trend_then_closes_in_profit(tmp_path):
    now = pd.Timestamp(NOW).floor("4h") + pd.Timedelta(minutes=2)
    market = st.ScenarioMarket(now, "quiet-market")

    class Trail(st.ScriptedBrain):
        def run(self, context, handlers):
            px = context["prices"]["SOL"]
            atr = context["market_overview"]["SOL"]["indicators"]["4h"]["atr14"]
            handlers["place_order"](coin="SOL", side="long", margin_usd=20, leverage=3, stop_price=px - 2 * atr,
                                    take_profit_price=0, max_hold_hours=0, strategy="4h breakout", reason="r",
                                    trailing_atr_mult=2)
            return {"summary": "", "coins": []}
    t = ai.AITrader(tmp_path, market, Trail("quiet-market"))
    t.step(now.to_pydatetime())
    [pos] = t.state["positions"]
    first_stop, entry = pos["stop"], pos["entry"]
    t.brain = None

    market.extend("SOL", entry * 1.10, minutes=12 * 60)  # a 12-hour trend up
    t.step((now + pd.Timedelta(hours=12, minutes=5)).to_pydatetime())
    [pos] = t.state["positions"]
    assert pos["stop"] > first_stop and pos["stop"] > entry  # moved up, now locks in a profit
    moved = pos["stop"]
    t.step((now + pd.Timedelta(hours=12, minutes=10)).to_pydatetime())  # same 4h candle: no change
    assert t.state["positions"][0]["stop"] == moved

    market.extend("SOL", moved * 0.99, minutes=60)
    t.step((now + pd.Timedelta(hours=13, minutes=15)).to_pydatetime())
    assert t.state["positions"] == []
    tr = pd.read_csv(tmp_path / "trades.csv").iloc[-1]
    assert tr["exit_reason"] == "trailing_stop" and tr["pnl_usd"] > 0 and tr["trailing_atr_mult"] == 2


def test_trailing_mult_is_checked():
    class Bad(st.ScriptedBrain):
        def run(self, context, handlers):
            with pytest.raises(ValueError):
                handlers["place_order"](coin="SOL", side="long", margin_usd=20, leverage=3,
                                        stop_price=context["prices"]["SOL"] * 0.97, take_profit_price=0,
                                        max_hold_hours=0, strategy="s", reason="r", trailing_atr_mult=-1)
            return {"summary": "", "coins": []}
    assert st.run_scenario("quiet-market", Bad("quiet-market"), now=NOW)["orders"] == []
