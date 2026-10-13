import io
import zipfile

import numpy as np
import pandas as pd

from btcbot import rs_breakout as rb
from btcbot.binance_klines import parse_zip


def _bars(close, freq, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(close), freq=freq, tz="UTC")
    c = np.asarray(close, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0}, index=idx)


def test_as_of_waits_for_bar_close():
    src = pd.DatetimeIndex(["2024-01-01 00:15", "2024-01-01 00:30"], tz="UTC")
    dst = pd.DatetimeIndex(["2024-01-01 00:10", "2024-01-01 00:15", "2024-01-01 00:25"], tz="UTC")
    out = rb.as_of(np.array([1.0, 2.0]), src, dst)
    assert np.isnan(out[0]) and out[1] == 1.0 and out[2] == 1.0


def test_unfinished_btc_bar_not_used():
    # BTC jumps in the 15m bar 01:00-01:15; only ADA bars closing at or after 01:15 may see it
    btc = _bars([100.0] * 4 + [200.0] + [200.0] * 3, "15min")
    ada = _bars(np.linspace(1, 2, 8 * 3), "5min")
    rs = rb.relative_strength(ada, btc, 15)
    rs_a = rb.as_of(rs.to_numpy(), btc.index + pd.Timedelta("15min"), ada.index + pd.Timedelta("5min"))
    jump = ada.index.get_loc(pd.Timestamp("2024-01-01 01:05", tz="UTC"))  # closes 01:10
    assert rs_a[jump] > -0.5 and rs_a[jump + 1] < -0.5


def test_rs_sign():
    btc = _bars(np.linspace(100, 101, 30), "5min")
    ada = _bars(np.linspace(1, 1.05, 30), "5min")
    rs = rb.relative_strength(ada, btc, 60).dropna()
    assert (rs > 0).all()


def test_entry_next_open_and_stop_first():
    c = [1.0] * 20 + [1.10, 1.10, 1.10]
    ada = _bars(c, "5min")
    ada.iloc[21, ada.columns.get_loc("open")] = 1.10
    ada.iloc[21, ada.columns.get_loc("low")] = 0.5  # hits the stop
    ada.iloc[21, ada.columns.get_loc("high")] = 2.0  # and the target
    btc = _bars([1.0] * 23, "5min")
    r = rb.run(ada, btc, btc, rb.RSConfig(version="C0", rs_minutes=5, taker_fee=0, maker_fee=0, slippage=0))
    t = r.trades.iloc[0]
    assert t.entry_time == ada.index[21] and t.reason == "stop" and t.R < 0


def test_parse_binance_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", "open_time,open,high,low,close,volume,close_time\n1704067200000,1,2,0.5,1.5,10,0\n")
    df = parse_zip(buf.getvalue())
    assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC") and df.close.iloc[0] == 1.5
