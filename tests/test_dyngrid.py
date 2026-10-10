import numpy as np
import pandas as pd

from btcbot import risk
from btcbot.dyngrid import DynGridConfig, fill_gaps, run


def sine(amp: float, n: int = 3000) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    c = pd.Series(100 + amp * np.sin(np.arange(n) * 2 * np.pi / 40), index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.1, "low": np.minimum(o, c) - 0.1,
                         "close": c, "volume": 1.0})


def test_grid_earns_when_price_stays_in_range():
    for mode in ("rebuild", "follow"):
        res = run(sine(0.3), DynGridConfig(cells=10, mode=mode))
        assert res.stats["range_exits"] == 0
        assert res.stats["total_return"] > 0


def test_grid_loses_when_swings_overrun_the_range():
    res = run(sine(1.0), DynGridConfig(cells=10, mode="rebuild"))
    assert res.stats["range_exits"] > 0
    assert res.stats["total_return"] < 0


def test_position_never_above_2x():
    seen = []
    orig = risk.check_leverage

    def spy(qty, price, equity):
        seen.append(abs(qty) * price / equity if equity > 0 else 0)
        orig(qty, price, equity)

    risk.check_leverage = spy
    try:
        for mode in ("rebuild", "follow"):
            run(sine(1.0), DynGridConfig(cells=20, mode=mode))
    finally:
        risk.check_leverage = orig
    assert seen and max(seen) <= risk.MAX_LEVERAGE + 1e-9


def test_fill_gaps_adds_flat_bars():
    idx = pd.DatetimeIndex(["2025-01-01 00:00", "2025-01-01 00:15", "2025-01-01 00:45"], tz="UTC")
    df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3],
                       "volume": [5, 5, 5]}, index=idx, dtype=float)
    out = fill_gaps(df)
    assert len(out) == 4
    assert out.iloc[2][["open", "high", "low", "close"]].tolist() == [2, 2, 2, 2]
    assert out.iloc[2]["volume"] == 0
