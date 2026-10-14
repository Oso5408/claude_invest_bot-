import numpy as np
import pandas as pd

from btcbot.swing import Costs, simulate


def _bars(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="1h", tz="UTC")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p * 1.001, "low": p * 0.999, "close": p, "volume": 1.0}, index=idx)


def test_time_exit_and_costs():
    h1 = _bars([100, 101, 102, 103, 104, 105])
    sig = pd.Series([1], index=h1.index[:1])
    stop = pd.Series([0.05], index=h1.index[:1])
    tr = simulate(h1, sig, stop, hold_h=3, costs=Costs(fee=0.0005, slippage=0.0))
    assert len(tr) == 1 and tr.reason[0] == "time"
    assert np.isclose(tr.gross_ret[0], 0.03)
    assert np.isclose(tr.net_ret[0], 0.03 - 0.001)
    assert np.isclose(tr.gross_r[0], 0.6)


def test_stop_and_short_funding():
    h1 = _bars([100, 99, 103, 104, 90])
    sig = pd.Series([-1], index=h1.index[:1])
    stop = pd.Series([0.02], index=h1.index[:1])
    fund = pd.Series(0.001, index=h1.index)
    tr = simulate(h1, sig, stop, hold_h=4, funding=fund, costs=Costs(fee=0.0, slippage=0.0))
    # bar 2 opens at 103, above the 102 stop: the gap fills at the open
    assert tr.reason[0] == "stop" and np.isclose(tr.gross_ret[0], -0.03)
    # short receives positive funding for bars 0..2
    assert np.isclose(tr.funding[0], -0.003) and np.isclose(tr.net_ret[0], -0.03 + 0.003)


def test_signals_ignored_while_in_trade():
    h1 = _bars([100] * 10)
    sig = pd.Series([1, 1, -1], index=h1.index[[0, 1, 4]])
    stop = pd.Series(0.05, index=sig.index)
    tr = simulate(h1, sig, stop, hold_h=3, costs=Costs(0, 0))
    assert list(tr.entry_time) == [h1.index[0], h1.index[4]]
