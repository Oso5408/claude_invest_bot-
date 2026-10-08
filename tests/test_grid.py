import numpy as np
import pandas as pd

from btcbot.grid import GridConfig, run


def _path(prices):
    """Bars whose open, low/high and close walk exactly through `prices`."""
    p = np.asarray(prices, dtype=float)
    idx = pd.date_range("2024-06-01", periods=len(p) - 1, freq="h", tz="UTC", name="open_time")
    o, c = p[:-1], p[1:]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                         "volume": 1e4}, index=idx)


def test_oscillation_inside_range_makes_money():
    prices = [100.0] + [98.5, 101.5] * 50 + [100.0]
    res = run(_path(prices))
    assert res.stats["grid_rebuilds"] == 0
    assert res.stats["final_equity_jpy"] > 30_000
    assert res.stats["signal_trades"] > 100


def test_breaking_the_range_closes_and_rebuilds_at_a_loss():
    prices = list(np.linspace(100, 80, 60)) + [80.0] * 5
    res = run(_path(prices))
    assert res.stats["grid_rebuilds"] >= 1
    first = res.trades.iloc[0]
    assert first["reason"] == "range_exit" and first["return"] < 0


def test_position_follows_lines_crossed():
    # three lines down from 100 at 1% spacing -> long three units
    from btcbot import grid
    df = _path([100.0, 99.5, 98.95, 97.9, 97.05])
    res = run(df, GridConfig())
    unit = np.floor(2.0 * 30_000 / (10 * 100.0) / 10) * 10
    # equity change equals 3 units marked to the last close against their fill prices
    fills = [100 / 1.01, 100 / 1.01**2, 100 / 1.01**3]
    expected = sum(unit * (97.05 - f) for f in fills)
    assert abs(res.equity.iloc[-1] - 30_000 - expected) < 1e-6


def test_grid_never_breaks_the_2x_cap():
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    res = run(_path(np.concatenate([[100.0], prices])), GridConfig(edge_leverage=5.0))
    assert res.stats["orders_capped_at_2x"] > 0
    assert (res.equity > 0).all()
