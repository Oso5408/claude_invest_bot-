import pandas as pd

from btcbot.breakout import BreakoutConfig, run
from btcbot.hyperliquid import account_config, parse_candles, parse_funding
from btcbot.synthetic import random_walk


def test_parse_candles_and_funding():
    rows = [{"t": 1700000000000, "T": 1700014399999, "o": "0.5", "h": "0.6", "l": "0.4", "c": "0.55", "v": "1000",
             "i": "4h", "s": "ADA", "n": 10},
            {"t": 1700014400000, "T": 1700028799999, "o": "0.55", "h": "0.7", "l": "0.5", "c": "0.6", "v": "800",
             "i": "4h", "s": "ADA", "n": 5}]
    df = parse_candles(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[1] - df.index[0] == pd.Timedelta("4h") and df["close"].iloc[1] == 0.6
    f = parse_funding([{"coin": "ADA", "fundingRate": "0.0000125", "premium": "0", "time": 1700000000076}])
    assert f.iloc[0] == 0.0000125 and str(f.index.tz) == "UTC"


def test_hyperliquid_account_runs_in_usd():
    df = random_walk(n=3000, start_price=0.5, vol=0.01, drift=0.0003)
    funding = pd.Series(0.0005, index=pd.date_range(df.index[0], df.index[-1], freq="1h"))
    res = run(df, BreakoutConfig(), account_config(200), funding=funding)
    assert len(res.trades) > 0 and res.stats["leverage_fees_jpy"] != 0
