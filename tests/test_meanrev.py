import numpy as np
import pandas as pd
import pytest

from btcbot import risk
from btcbot.meanrev import AccountConfig, MarginAccount, RiskController, MeanRevConfig, run
from btcbot.strategy import StrategyConfig
from btcbot.synthetic import mean_reverting, random_walk


def test_cap_quantity_limits_to_2x():
    assert risk.cap_quantity(1000, 100.0, 30_000) == pytest.approx(600)
    assert risk.cap_quantity(-1000, 100.0, 30_000) == pytest.approx(-600)
    assert risk.cap_quantity(100, 100.0, 30_000) == 100
    assert risk.cap_quantity(100, 100.0, 0) == 0
    q = risk.cap_quantity(1000, 100.0, 30_000, fee_rate=0.0003)
    acct = MarginAccount(30_000)
    acct.set_position(q, 100.0, 0.0003)  # fee taken, still within 2x


def test_account_refuses_more_than_2x():
    acct = MarginAccount(30_000)
    acct.set_position(600, 100.0, 0.0)  # exactly 2x is fine
    with pytest.raises(risk.LeverageLimitError):
        MarginAccount(30_000).set_position(601, 100.0, 0.0)


def test_max_leverage_is_two():
    assert risk.MAX_LEVERAGE == 2.0


@pytest.mark.parametrize("qty", [600, -600, 500, -400])
def test_loss_cut_price_hits_75_percent(qty):
    acct = MarginAccount(30_000)
    acct.set_position(qty, 100.0, 0.0)
    px = risk.loss_cut_price(acct.collateral, acct.qty, acct.entry)
    assert risk.maintenance_ratio(acct.equity(px), acct.qty, px) == pytest.approx(0.75)


def test_kelly_leverage_math():
    k = RiskController(StrategyConfig(prior_alpha=1, prior_beta=1, payoff_min_trades=1, kelly_fraction=0.25), 0, 0.5)
    assert k.leverage() == 0  # no evidence yet
    for r in [0.02, 0.02, -0.01]:
        k.update(r)
    # p = 3/5, b = 2, avg loss 1% -> full Kelly 40x, quarter 10x, capped at 2x
    assert k.leverage() == risk.MAX_LEVERAGE
    k2 = RiskController(StrategyConfig(prior_alpha=1, prior_beta=1, payoff_min_trades=1, kelly_fraction=0.25), 0, 0.5)
    for r in [0.0105, -0.01, 0.0105, -0.01]:
        k2.update(r)
    # p = 3/6, b = 1.05 -> full Kelly (0.5 - 0.5 / 1.05) / 0.01 = 2.38x, quarter = 0.595x
    assert k2.leverage() == pytest.approx(0.25 * (0.5 - 0.5 / 1.05) / 0.01)


def test_never_above_2x_at_any_close():
    df = mean_reverting(n=24 * 120, seed=2)
    res = run(df, MeanRevConfig(trend_filter=False, cold_start_leverage=2.0))
    assert res.stats["signal_trades"] > 20
    # set_position raises on any fill above 2x, so finishing the run is the check;
    # also no equity value can be negative
    assert (res.equity > 0).all()


def test_gap_down_triggers_loss_cut():
    df = mean_reverting(n=24 * 120, seed=2)
    cfg = MeanRevConfig(cold_start_leverage=2.0, trend_filter=False)
    base = run(df, cfg)
    assert base.stats["loss_cuts"] == 0
    long_trade = base.trades[(base.trades["side"] > 0) & (base.trades["qty"] > 0)].iloc[0]
    # the bar after the long fills, price gaps down 40%: at about 2x that is a loss cut at the open
    pos = df.index.get_loc(long_trade["entry_time"]) + 1
    crash = df.copy()
    crash.iloc[pos:, :4] *= 0.6
    out = run(crash, cfg)
    assert out.stats["loss_cuts"] >= 1
    cut = out.trades[out.trades["reason"] == "loss_cut"].iloc[0]
    assert cut["exit_time"] == df.index[pos]
    assert (out.equity > 0).all()


def test_leverage_fee_charged_daily():
    df = mean_reverting(n=24 * 90, seed=5)
    res = run(df)
    assert res.stats["leverage_fees_jpy"] > 0
    free = run(df, acct_cfg=AccountConfig(leverage_fee_per_day=0.0))
    assert free.stats["final_equity_jpy"] > res.stats["final_equity_jpy"]


def test_min_order_blocks_tiny_positions():
    df = mean_reverting(n=24 * 90, seed=5)
    res = run(df, acct_cfg=AccountConfig(min_order=1e9))
    assert res.stats["final_equity_jpy"] == 30_000


def test_no_lookahead():
    df = random_walk(n=3000, seed=1, start_price=100, vol=0.01)
    base = run(df).equity
    changed = df.copy()
    changed.iloc[2000:, :4] *= 1.3
    assert np.allclose(base.iloc[:2000], run(changed).equity.iloc[:2000])


def test_cold_start_then_kelly():
    rc = RiskController(StrategyConfig(kelly_fraction=0.25), cold_start_trades=30, cold_start_leverage=0.5)
    for _ in range(29):
        rc.update(-0.01)  # a terrible record does not matter yet
    assert rc.leverage() == 0.5
    rc.update(-0.01)
    assert rc.leverage() == 0.0  # 30 trades, negative expectancy: Kelly stays locked
    for _ in range(40):
        rc.update(0.02)
    assert 0 < rc.leverage() <= risk.MAX_LEVERAGE


def test_cold_start_leverage_cannot_exceed_cap():
    rc = RiskController(StrategyConfig(), cold_start_trades=30, cold_start_leverage=5.0)
    assert rc.leverage() == risk.MAX_LEVERAGE


def _trend(n, step):
    idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC", name="open_time")
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(step + rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100], close[:-1]])
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                         "low": np.minimum(open_, close) * 0.998, "close": close, "volume": 1e4}, index=idx)


def test_trend_filter_blocks_trades_against_4h_ema():
    up = run(_trend(24 * 200, 0.001)).trades
    down = run(_trend(24 * 200, -0.001)).trades
    assert len(up) and (up["side"] > 0).all()
    assert len(down) and (down["side"] < 0).all()
    both = run(_trend(24 * 200, 0.001), MeanRevConfig(trend_filter=False)).trades
    assert (both["side"] < 0).any()


def test_trend_ema_waits_for_completed_4h_bars():
    from btcbot.meanrev import compute_features
    df = _trend(24 * 60, 0.0)
    f = compute_features(df, MeanRevConfig(trend_ema=10))
    changed = df.copy()
    changed.iloc[1000:, :4] *= 2  # bar 1000 opens 16:00 UTC, inside the 16:00-20:00 4h bar
    g = compute_features(changed, MeanRevConfig(trend_ema=10))
    assert np.allclose(f["trend_ema"].iloc[:1000], g["trend_ema"].iloc[:1000], equal_nan=True)
    # first hourly bar that can see the changed 4h bar is the one closing at 20:00 UTC
    first_diff = np.flatnonzero(~np.isclose(f["trend_ema"], g["trend_ema"], equal_nan=True))[0]
    assert df.index[first_diff] + pd.Timedelta(hours=1) == pd.Timestamp("2024-07-12 20:00", tz="UTC")
