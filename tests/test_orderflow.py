import io
import zipfile

import numpy as np
import pandas as pd

from btcbot import orderflow as of


def test_build_month_taker_side(tmp_path):
    rows = ["agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
            "1,1.0,10,1,1,0,false",       # taker buy 10
            "2,1.2,5,2,2,30000,true",     # taker sell 6
            "3,0.9,10,3,3,59999,false",   # taker buy 9
            "4,1.1,1,4,4,60000,true"]     # next minute
    path = tmp_path / "x.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.csv", "\n".join(rows) + "\n")
    df = of.build_month(str(path))
    assert list(df.index) == [0, 1]
    r = df.loc[0]
    assert (r.open, r.high, r.low, r.close) == (1.0, 1.2, 0.9, 0.9)
    assert np.isclose(r.buy, 19.0) and np.isclose(r.sell, 6.0)


def test_random_walk_has_no_edge():
    df = of.random_walk(minutes=60_000)
    t = of.run(df, of.Params(fee=0.0, slip=0.0))
    assert len(t) > 500
    assert abs(t["r"].mean()) < 0.06


def test_delta_filter_finds_planted_edge():
    # Taker buying in a minute pushes the next minutes up: the filter must see it.
    rng = np.random.default_rng(1)
    n = 40_000
    delta = rng.uniform(-1, 1, n)
    drift = np.r_[0, delta[:-1]] * 0.0008
    ret = drift + rng.normal(0, 0.001, n)
    close = 0.6 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * 1.0003
    lo = np.minimum(open_, close) * 0.9997
    buy = (1 + delta) * 500
    sell = (1 - delta) * 500
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                       "buy": buy, "sell": sell, "n": 60},
                      index=pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC"))
    plain = of.run(df, of.Params(fee=0.0, slip=0.0))["r"].mean()
    filt = of.run(df, of.Params(fee=0.0, slip=0.0, delta_min=0.5))["r"].mean()
    assert filt > plain + 0.1
