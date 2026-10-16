from datetime import datetime, timezone

import pandas as pd

from btcbot import ai_missed as am
from btcbot import ai_selftest as st
from btcbot import ai_trader as ai

NOW = pd.Timestamp(datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)) + pd.Timedelta(minutes=2)


class Quiet(st.ScriptedBrain):
    def run(self, context, handlers):
        return {"summary": "nothing", "coins": [{"coin": "NEAR", "view": "no trade", "strategy": "wait",
                                                 "reason": "looks weak"}]}


def run(tmp_path, scenario, brain):
    market = st.ScenarioMarket(NOW, scenario)
    t = ai.AITrader(tmp_path, market, brain)
    t.step(NOW.to_pydatetime())
    return am.review(tmp_path, market, NOW + pd.Timedelta(minutes=5), days=2)


def test_missed_breakout_is_listed_with_what_the_ai_said(tmp_path):
    out = run(tmp_path, "quiet-breakout-long", Quiet("quiet-breakout-long"))
    [r] = [r for r in out["rows"] if r["coin"] == "NEAR"]
    assert r["side"] == "long" and r["taken"] == "NO" and "looks weak" in r["ai_said"]
    assert out["missed"] == out["signals"] == 1


def test_taken_breakout_counts_as_taken(tmp_path):
    out = run(tmp_path, "quiet-breakout-long", st.ScriptedBrain())
    [r] = out["rows"]
    assert r["taken"] != "NO" and out["missed"] == 0


def test_short_with_greed_is_not_a_signal(tmp_path):
    assert run(tmp_path, "breakdown-short-greed", Quiet("x"))["signals"] == 0


def test_trailing_result_follows_a_trend():
    idx = pd.date_range("2026-10-01", periods=60, freq="4h", tz="UTC")
    close = pd.Series([1.0] * 30 + [1.0 + 0.02 * i for i in range(1, 21)] + [1.4 - 0.05 * i for i in range(1, 11)], index=idx)
    df = pd.DataFrame({"open": close.shift().fillna(1.0), "high": close * 1.005, "low": close * 0.995,
                       "close": close, "volume": 1.0}, index=idx)
    ret, state = am.trailing_result(df, 30, "long")
    assert state == "stopped" and ret > 0.1
