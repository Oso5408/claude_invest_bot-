import numpy as np
import pandas as pd

from btcbot import breakout
from btcbot.data import resample
from btcbot.execution import ExecutionOptimizer
from btcbot.meanrev import AccountConfig
from btcbot.synthetic import random_walk


def bars15(prices, lows=None, highs=None):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="15min", tz="UTC")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": highs if highs is not None else p,
                         "low": lows if lows is not None else p, "close": p, "volume": 1.0}, index=idx)


def test_limit_fills_on_pullback_with_maker_fee():
    b = bars15([100] * 16, lows=[100, 100, 99, 98] + [100] * 12, highs=[101] * 16)
    ex = ExecutionOptimizer(b, b.index[:1], pullback_atr=0.5, maker_fee=0.0001)
    px, fee, lo, hi = ex(0, 1, signal_close=100, signal_atr=2)  # limit at 99
    assert (px, fee) == (99, 0.0001)
    assert lo == 98 and hi == 101  # range from the fill bar onwards


def test_skip_and_chase_when_never_touched():
    b = bars15(np.linspace(100, 110, 16))
    skip = ExecutionOptimizer(b, b.index[:1], pullback_atr=0.5, on_miss="skip")
    assert skip(0, 1, 100, 2) is None
    chase = ExecutionOptimizer(b, b.index[:1], pullback_atr=0.5, on_miss="chase", slippage=0.001, taker_fee=0.0005)
    px, fee, _, _ = chase(0, 1, 100, 2)
    assert np.isclose(px, 110 * 1.001) and fee == 0.0005


def test_market_style_hook_matches_default_engine():
    df = random_walk(start_price=100, vol=0.01, drift=0.0003)
    df4 = resample(df, "4h")
    acct = AccountConfig()
    o, l, h = df4["open"].to_numpy(), df4["low"].to_numpy(), df4["high"].to_numpy()

    def market(t, d, close, atr):
        return o[t] * (1 + acct.slippage * d), acct.fee_rate, l[t], h[t]

    a = breakout.run(df4, breakout.BreakoutConfig(), acct)
    b = breakout.run(df4, breakout.BreakoutConfig(), acct, entry_fill=market)
    assert np.allclose(a.equity.to_numpy(), b.equity.to_numpy())
