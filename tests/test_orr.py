import numpy as np
import pandas as pd

from btcbot import orr

NOCOST = dict(taker_fee=0.0, maker_fee=0.0, slippage=0.0)


def _days(n_days: int) -> pd.DataFrame:
    """Hourly bars, rising 0.5% a day, each day moving 1.0 -> 1.02 -> 1.0 around its base."""
    idx = pd.date_range("2025-01-01", periods=24 * n_days, freq="1h", tz="UTC")
    base = 100 * 1.005 ** (np.arange(len(idx)) // 24)
    c = base * (1 + 0.01 * np.sin(np.arange(len(idx)) % 24 / 24 * 2 * np.pi))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0},
                        index=idx)


def _with_last_day(path: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    df = _days(70)
    last = df.index[-1] + pd.Timedelta(hours=1)
    p = df["close"].iloc[-1]
    rows = [(o * p, h * p, l * p, c * p) for o, h, l, c in path]
    extra = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                         index=pd.date_range(last, periods=len(rows), freq="1h"))
    extra["volume"] = 1.0
    return pd.concat([df, extra])


def test_long_reversal_hits_half_retrace_target():
    # sell-off to 0.94, the next bar takes out the low bar's high (0.955), then rallies
    df = _with_last_day([(1.0, 1.0, 0.955, 0.955), (0.955, 0.955, 0.94, 0.945), (0.945, 0.96, 0.944, 0.958),
                         (0.95, 0.975, 0.95, 0.975), (0.975, 0.98, 0.97, 0.98)])
    tr = orr.run(df, orr.ORRConfig(**NOCOST)).trades
    last = tr.iloc[-1]
    assert last["side"] == 1
    assert last["entry"] == df["high"].iloc[-4]  # high of the bar that made the low
    assert abs(last["stop"] - df["low"].iloc[-4]) < 1e-9
    assert abs(last["target"] - (df["low"].iloc[-4] + 0.5 * (df["open"].iloc[-5] - df["low"].iloc[-4]))) < 1e-9
    assert last["reason"] == "target"


def test_entry_bar_through_the_stop_counts_as_a_loss_or_no_trade():
    path = [(1.0, 1.0, 0.955, 0.955), (0.955, 0.955, 0.94, 0.945), (0.945, 0.96, 0.93, 0.935), (0.935, 0.94, 0.93, 0.94)]
    df = _with_last_day(path)
    day = df.index[-1].floor("D")
    worst = orr.run(df, orr.ORRConfig(**NOCOST)).trades
    worst = worst[pd.to_datetime(worst["entry_time"]) >= day]
    assert len(worst) == 1 and worst.iloc[0]["reason"] == "stop"
    skip = orr.run(df, orr.ORRConfig(ambiguous="skip", **NOCOST)).trades
    assert skip.empty or (pd.to_datetime(skip["entry_time"]) >= day).sum() == 0


def test_us_session_follows_new_york_daylight_saving():
    idx = pd.DatetimeIndex(["2025-07-01 13:30", "2025-07-01 13:15", "2025-01-15 14:30", "2025-01-15 14:15"], tz="UTC")
    start, _, inside = orr.sessions(idx, orr.ORRConfig(session="us"))
    assert list(start.strftime("%m-%d %H:%M")) == ["07-01 13:30", "06-30 13:30", "01-15 14:30", "01-14 14:30"]
    assert list(inside) == [True, False, True, False]


def test_daily_context_uses_only_finished_days():
    df = _days(70)
    starts, _, _ = orr.sessions(df.index, orr.ORRConfig())
    close, _, _ = orr.daily_context(df, orr.ORRConfig(), starts)
    daily_close = df["close"].groupby(df.index.floor("D")).last()
    i = 24 * 60 + 5  # a bar on day 60
    assert close[i] == daily_close.iloc[59]
