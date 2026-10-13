import numpy as np
import pandas as pd
import pytest

from btcbot import risk, threer


def _path_bars(seed: int, days: int = 400) -> pd.DataFrame:
    """15m bars built from a 1-minute random walk, so highs and lows are real path extremes."""
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004 / np.sqrt(15), n))), index=idx)
    df = p.resample("15min").ohlc()
    df["volume"] = 1.0
    return df


def test_daily_context_has_no_same_day_values():
    df = _path_bars(0, days=80)
    ctx = threer.daily_context(df, threer.ThreeRConfig())
    t = df.index[60 * 96 + 10]
    today = df[df.index.floor("D") == t.floor("D")]
    yesterday = df[df.index.floor("D") == t.floor("D") - pd.Timedelta(days=1)]
    assert ctx.loc[t, "range_high"] == pytest.approx(yesterday["high"].max())
    assert ctx.loc[t, "range_low"] == pytest.approx(yesterday["low"].min())
    assert ctx.loc[t, "prev_close"] == pytest.approx(yesterday["close"].iloc[-1])
    assert ctx.loc[t, "range_high"] != pytest.approx(today["high"].max())


def test_no_edge_on_a_random_walk_without_costs():
    cfg = threer.ThreeRConfig(taker_fee=0, maker_fee=0, slippage=0, setup="both")
    r = [threer.run(_path_bars(s), cfg).trades["R"].mean() for s in (1, 2)]
    assert abs(np.mean(r)) < 0.2


def test_leverage_cap_and_trades_per_day():
    df = _path_bars(5, days=200)
    res = threer.run(df, threer.ThreeRConfig(risk_frac=0.5, setup="both", max_trades_per_day=2))
    tr = res.trades
    assert len(tr)
    per_day = pd.to_datetime(tr["entry_time"]).dt.floor("D").value_counts()
    assert per_day.max() <= 2
    assert (np.sign(tr["entry"] - tr["stop"]) == tr["side"]).all()
    # a 50% risk budget with sub-2% stops asks for far more than 2x: the cap must bind on every entry
    lev = tr["qty"] * tr["entry"] / tr["eq_before"]
    assert lev.max() <= risk.MAX_LEVERAGE + 1e-9
    assert lev.max() > 1.9
