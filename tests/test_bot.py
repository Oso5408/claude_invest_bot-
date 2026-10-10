from datetime import date

import numpy as np
import pytest

from btcbot.backtest import BacktestConfig, max_drawdown, run_backtest
from btcbot.data import date_params, parse_klines
from btcbot.strategy import BayesKelly, StrategyConfig, compute_features
from btcbot.synthetic import random_walk

SAMPLE = {"status": 0, "data": [
    {"openTime": "1704067200000", "open": "6000000", "high": "6100000", "low": "5900000", "close": "6050000", "volume": "1.5"},
    {"openTime": "1704070800000", "open": "6050000", "high": "6060000", "low": "6000000", "close": "6010000", "volume": "0.7"},
]}


def test_parse_klines():
    df = parse_klines(SAMPLE)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index[0]) == "2024-01-01 00:00:00+00:00"
    assert df["close"].iloc[1] == 6010000.0


def test_parse_klines_error():
    with pytest.raises(RuntimeError):
        parse_klines({"status": 5, "messages": [{"message_code": "ERR-5201"}]})


def test_date_params():
    assert date_params("1hour", date(2024, 2, 28), date(2024, 3, 1)) == ["20240228", "20240229", "20240301"]
    assert date_params("1day", date(2022, 5, 1), date(2024, 1, 1)) == ["2022", "2023", "2024"]


def test_kelly_math():
    bk = BayesKelly(StrategyConfig(prior_alpha=1, prior_beta=1, payoff_min_trades=1))
    assert bk.kelly() == 0  # p=0.5, b=1: no edge
    for r in [0.02, 0.02, -0.01]:
        bk.update(r)
    # posterior p = 3/5, payoff b = 0.02/0.01 = 2 -> f* = 0.6 - 0.4/2 = 0.4
    assert bk.kelly() == pytest.approx(0.4)
    assert bk.position_size() == pytest.approx(0.1)  # quarter Kelly


def test_no_lookahead():
    df = random_walk(n=3000, seed=1)
    base = run_backtest(df).equity
    changed = df.copy()
    changed.iloc[2000:, :4] *= 1.5  # rewrite the future
    after = run_backtest(changed).equity
    assert np.allclose(base.iloc[:2000], after.iloc[:2000])


def test_time_window_uses_jst():
    df = random_walk(n=48)
    f = compute_features(df, StrategyConfig(entry_hours_jst=(9,)))
    # bar opening 23:00 UTC closes 00:00 UTC = 09:00 JST
    assert f["time_ok"].sum() == 2
    assert all(f.index[f["time_ok"]].hour == 23)


def test_fees_cost_money_and_trend_is_caught():
    df = random_walk(n=24 * 200, drift=0.0006, vol=0.004, seed=3)
    cheap = run_backtest(df, bt=BacktestConfig(fee_rate=0.0))
    dear = run_backtest(df, bt=BacktestConfig(fee_rate=0.002))
    assert cheap.stats["trades_taken"] > 0
    assert cheap.stats["total_return"] > 0
    assert dear.stats["final_equity_jpy"] < cheap.stats["final_equity_jpy"]


def test_max_drawdown():
    import pandas as pd
    assert max_drawdown(pd.Series([100, 120, 90, 130])) == pytest.approx(-0.25)
