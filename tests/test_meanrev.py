import numpy as np
import pandas as pd
import pytest

from btcbot import risk
from btcbot.meanrev import AccountConfig, LeverageKelly, MarginAccount, MeanRevConfig, run
from btcbot.strategy import StrategyConfig
from btcbot.synthetic import mean_reverting, random_walk


def test_cap_quantity_limits_to_2x():
    assert risk.cap_quantity(1000, 100.0, 30_000) == pytest.approx(600)
    assert risk.cap_quantity(-1000, 100.0, 30_000) == pytest.approx(-600)
    assert risk.cap_quantity(100, 100.0, 30_000) == 100
    assert risk.cap_quantity(100, 100.0, 0) == 0


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
    k = LeverageKelly(StrategyConfig(prior_alpha=1, prior_beta=1, payoff_min_trades=1, kelly_fraction=0.25))
    assert k.leverage() == 0  # no evidence yet
    for r in [0.02, 0.02, -0.01]:
        k.update(r)
    # p = 3/5, b = 2, avg loss 1% -> full Kelly 40x, quarter 10x, capped at 2x
    assert k.leverage() == risk.MAX_LEVERAGE
    k2 = LeverageKelly(StrategyConfig(prior_alpha=1, prior_beta=1, payoff_min_trades=1, kelly_fraction=0.25))
    for r in [0.0105, -0.01, 0.0105, -0.01]:
        k2.update(r)
    # p = 3/6, b = 1.05 -> full Kelly (0.5 - 0.5 / 1.05) / 0.01 = 2.38x, quarter = 0.595x
    assert k2.leverage() == pytest.approx(0.25 * (0.5 - 0.5 / 1.05) / 0.01)


def test_never_above_2x_at_any_close():
    df = mean_reverting(n=24 * 120, seed=2)
    res = run(df)
    assert res.stats["signal_trades"] > 20
    # set_position raises on any fill above 2x, so finishing the run is the check;
    # also no equity value can be negative
    assert (res.equity > 0).all()


def test_gap_down_triggers_loss_cut():
    df = mean_reverting(n=24 * 120, seed=2)
    base = run(df)
    assert base.stats["loss_cuts"] == 0
    long_trade = base.trades[(base.trades["side"] > 0) & (base.trades["qty"] > 0)].iloc[0]
    # the bar after the long fills, price gaps down 40%: at about 2x that is a loss cut at the open
    pos = df.index.get_loc(long_trade["entry_time"]) + 1
    crash = df.copy()
    crash.iloc[pos:, :4] *= 0.6
    out = run(crash)
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
