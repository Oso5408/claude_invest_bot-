import numpy as np
import pandas as pd
import pytest

from btcbot import risk
from btcbot.breakout import BreakoutConfig, compute_features, run
from btcbot.meanrev import RiskController
from btcbot.strategy import StrategyConfig
from btcbot.synthetic import random_walk


def _bars(close, volume=None):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-06-01", periods=len(close), freq="h", tz="UTC", name="open_time")
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.full(len(close), 1000.0) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001,
                         "low": np.minimum(open_, close) * 0.999, "close": close, "volume": vol}, index=idx)


def test_seeded_prior_sizes_from_first_trade():
    rc = RiskController(StrategyConfig(kelly_fraction=0.25), cold_start_trades=0, cold_start_leverage=0.0)
    rc.seed_prior(0.35, 2.5, 0.02, 30)
    assert rc.n_trades == 0
    # p = 0.35, b = 2.5, avg loss 2%: full Kelly 4.5x, quarter 1.125x
    assert rc.leverage() == pytest.approx(1.125)
    assert 0 < rc.leverage() <= risk.MAX_LEVERAGE


def test_long_breakout_needs_volume():
    close = [100] * 30 + [105]
    quiet = compute_features(_bars(close), BreakoutConfig())
    loud = compute_features(_bars(close, [1000] * 30 + [5000]), BreakoutConfig())
    assert not quiet["long_signal"].iloc[-1]
    assert loud["long_signal"].iloc[-1]


def test_short_breakout_ignores_volume():
    f = compute_features(_bars([100] * 30 + [95]), BreakoutConfig())
    assert f["short_signal"].iloc[-1]
    assert not compute_features(_bars([100] * 30 + [95]), BreakoutConfig(allow_short=False))["short_signal"].iloc[-1]


def test_trailing_stop_locks_in_profit():
    # flat, breakout with volume, steady climb, then a sharp drop
    close = [100.0] * 40 + [103.0] + list(np.linspace(103, 130, 60)) + list(np.linspace(130, 110, 20))
    vol = [1000] * 40 + [5000] + [1000] * 80
    res = run(_bars(close, vol), BreakoutConfig(allow_short=False))
    tr = res.trades
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["side"] == 1 and t["reason"] == "trailing_stop"
    assert t["exit_px"] > t["entry_px"] * 1.15  # stop trailed up well above entry
    assert t["return"] > 0.15


def test_stop_never_moves_back():
    close = [100.0] * 40 + [103.0] + list(np.linspace(103, 120, 30)) + [119.9, 119.8, 120.1]
    vol = [1000] * 40 + [5000] + [1000] * 33
    res = run(_bars(close, vol), BreakoutConfig(allow_short=False))
    assert len(res.trades) == 0  # small pullback, the raised stop has not been hit
    assert res.equity.iloc[-1] > res.equity.iloc[40]


def test_breakout_no_lookahead():
    df = random_walk(n=3000, seed=1, start_price=100, vol=0.01)
    base = run(df).equity
    changed = df.copy()
    changed.iloc[2000:, :4] *= 1.3
    assert np.allclose(base.iloc[:2000], run(changed).equity.iloc[:2000])


def test_breakout_respects_2x_cap():
    df = random_walk(n=24 * 200, seed=3, start_price=100, vol=0.01, drift=0.0005)
    res = run(df, BreakoutConfig(kelly_fraction=5.0))  # asks for far more than 2x
    assert res.stats["orders_capped_at_2x"] > 0
    assert (res.equity > 0).all()


def test_resample_to_4h():
    from btcbot.data import resample
    df = _bars(np.arange(1.0, 9.0), volume=np.ones(8))
    r = resample(df, "4h")
    assert len(r) == 2
    assert r["open"].iloc[0] == df["open"].iloc[0] and r["close"].iloc[0] == df["close"].iloc[3]
    assert r["high"].iloc[1] == df["high"].iloc[4:].max() and r["volume"].iloc[0] == 4
