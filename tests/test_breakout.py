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


def test_mtf_trend_and_volume_filters():
    from btcbot.breakout import mtf_config
    n = 24 * 60
    idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC", name="open_time")
    close = np.full(n, 100.0)
    close[: n - 2] = np.linspace(200, 100, n - 2)  # long downtrend: price under the 4h EMA50
    close[-2], close[-1] = 100.0, 104.0  # then one big up break with volume
    vol = np.full(n, 1000.0)
    vol[-1] = 5000
    df = _bars(close, vol)
    f = compute_features(df, mtf_config())
    assert not f["long_signal"].iloc[-1]  # blocked: below the trend EMA
    g = compute_features(df, mtf_config(trend_timeframe=None))
    assert g["long_signal"].iloc[-1]
    # a short break without volume is ignored in the MTF preset
    close2 = close.copy()
    close2[-1] = 90.0
    vol2 = np.full(n, 1000.0)
    h = compute_features(_bars(close2, vol2), mtf_config())
    assert not h["short_signal"].iloc[-1]
    vol2[-1] = 5000
    assert compute_features(_bars(close2, vol2), mtf_config())["short_signal"].iloc[-1]


def test_soft_leverage_limit_for_spot():
    df = random_walk(n=24 * 200, seed=3, start_price=100, vol=0.01, drift=0.0005)
    res = run(df, BreakoutConfig(kelly_fraction=5.0, max_leverage=1.0, allow_short=False))
    tr = res.trades[res.trades["qty"] != 0]
    assert len(tr) and (tr["qty"] > 0).all()
    assert (tr["qty"] * tr["entry_px"]).max() <= 1.0 * res.equity.max() * 1.001


def test_fng_value_is_only_used_the_day_after_its_stamp():
    from btcbot.breakout import align_fng
    fng = pd.Series([10.0, 90.0], index=pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True))
    idx = pd.date_range("2025-01-01", periods=12, freq="4h", tz="UTC")
    v = align_fng(idx, fng)
    assert v.iloc[:6].isna().all()  # day 1: the day-1 value may not be out yet
    assert (v.iloc[6:] == 10).all()  # day 2 sees day 1's value, not day 2's


def test_fng_band_blocks_entries():
    df = random_walk(n=3000, start_price=100, vol=0.01, drift=0.0003)
    fng_hi = pd.Series(80.0, index=pd.date_range(df.index[0].floor("D") - pd.Timedelta(days=2),
                                                 df.index[-1], freq="D", tz="UTC"))
    base = compute_features(df, BreakoutConfig(), fng_hi)
    assert base["long_signal"].any()  # default band 0-100 lets everything through
    blocked = compute_features(df, BreakoutConfig(fng_long_max=75, fng_short_min=85), fng_hi)
    assert not blocked["long_signal"].any() and not blocked["short_signal"].any()
    assert len(run(df, BreakoutConfig(fng_long_max=75, fng_short_min=85), fng=fng_hi).trades) == 0


def test_rollover_fee_charged_on_4h_and_15m_bars():
    from btcbot.meanrev import rollover_counts
    for freq, per_day in [("1h", 1), ("4h", 1), ("15min", 1)]:
        idx = pd.date_range("2025-01-01", periods=int(pd.Timedelta("10D") / pd.Timedelta(freq)), freq=freq, tz="UTC")
        n = rollover_counts(idx, 6)
        assert n.sum() in (9, 10) and n.max() == per_day


def test_funding_per_bar_sums_hours_and_charges_longs():
    from btcbot.breakout import funding_per_bar
    idx = pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC")
    hours = pd.date_range("2025-01-01", periods=16, freq="1h", tz="UTC")
    rates = pd.Series(0.0001, index=hours)
    f = funding_per_bar(idx, rates)
    assert f[0] == 0 and np.allclose(f[1:], 0.0004)
    df = random_walk(n=3000, start_price=100, vol=0.01, drift=0.0003)
    paid = pd.Series(0.0005, index=pd.date_range(df.index[0], df.index[-1], freq="1h"))
    free = run(df, BreakoutConfig(allow_short=False), fng=None, funding=paid * 0)
    cost = run(df, BreakoutConfig(allow_short=False), fng=None, funding=paid)
    assert free.stats["leverage_fees_jpy"] == 0 and cost.stats["leverage_fees_jpy"] > 0
