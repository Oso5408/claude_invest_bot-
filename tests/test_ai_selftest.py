from datetime import datetime, timezone

import pandas as pd
import pytest

from btcbot import ai_selftest as st
from btcbot import ai_trader as ai

NOW = datetime(2026, 10, 15, 8, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("name", list(st.SCENARIOS))
def test_signal_reaches_context_order_opens_and_take_profit_closes(name):
    sc = st.SCENARIOS[name]
    out = st.run_scenario(name, st.ScriptedBrain(name), now=NOW)
    sig = out["signal_in_context"]
    if sc["side"] == "long":
        assert sig["close"] > sig["donchian20_high"] and sig["trend"] == "up"
    else:
        assert sig["close"] < sig["donchian10_low"] and sig["trend"] == "down"
    assert sig["volume_ratio20"] >= 2
    [o] = out["orders"]
    assert (o["coin"], o["side"]) == (sc["coin"], sc["side"]) and out["orders_ok"]
    assert out["trades_left_today"] == ai.MAX_ENTRIES_PER_DAY - 1
    [c] = out["closed"]
    assert c["exit_reason"] == "take_profit" and c["strategy"] == "4h Donchian breakout" and c["pnl_usd"] > 0


def test_stop_closes_when_there_is_no_take_profit():
    class NoTP(st.ScriptedBrain):
        def run(self, context, handlers):
            px = context["prices"]["SOL"]
            handlers["place_order"](coin="SOL", side="long", margin_usd=20, leverage=3, stop_price=px * 0.97,
                                    take_profit_price=0, max_hold_hours=0, strategy="s", reason="r")
            return {"summary": "", "coins": []}
    out = st.run_scenario("breakout-long", NoTP("breakout-long"), now=NOW)
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
    out = st.run_scenario("breakout-long", Greedy("breakout-long"), now=NOW)
    assert out["orders"] == [] and out["trades_left_today"] == ai.MAX_ENTRIES_PER_DAY


def test_scenario_market_is_made_up_but_scaled_to_live_prices():
    m = st.ScenarioMarket(pd.Timestamp(NOW), "breakdown-short", {"ETH": 4000.0})
    assert m.mids()["ETH"] == pytest.approx(4000.0)
    assert list(m.fear_greed().values())[-1] == 75.0
