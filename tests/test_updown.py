from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from btcbot import telegram_bot as tb
from btcbot import updown


def _minutes(end: datetime, n: int = 60, seed: int = 0) -> pd.DataFrame:
    """n closed-or-live 1m candles ending with the one that opens at `end` (a random walk)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp(end).floor("1min"), periods=n, freq="1min", tz="UTC")
    close = 80_000 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    open_ = np.r_[80_000, close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close),
                         "close": close, "volume": 1.0}, index=idx)


def test_call_times():
    t = datetime(2026, 10, 17, 9, 3, 30, tzinfo=timezone.utc)
    assert updown.window_start(t) == datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)
    assert updown.next_call_time(t) == datetime(2026, 10, 17, 9, 7, 3, tzinfo=timezone.utc)
    assert updown.next_call_time(datetime(2026, 10, 17, 9, 1, tzinfo=timezone.utc)) == \
        datetime(2026, 10, 17, 9, 2, 3, tzinfo=timezone.utc)


def test_predict_follows_first_two_minutes():
    now = datetime(2026, 10, 17, 9, 2, 3, tzinfo=timezone.utc)
    m = _minutes(now)
    w = pd.Timestamp("2026-10-17 09:00", tz="UTC")
    m.loc[w, "open"] = 80_000.0
    m.loc[w + pd.Timedelta(minutes=1), "close"] = 80_200.0  # +0.25% in two minutes
    row = updown.predict(m, w)
    assert row["call"] == "UP" == row["baseline"] and row["p_up"] > 0.9
    m.loc[w + pd.Timedelta(minutes=1), "close"] = 79_800.0
    assert updown.predict(m, w)["call"] == "DOWN"


def test_random_walk_hit_rate_near_72pct():
    hits = []
    for seed in range(400):
        now = datetime(2026, 10, 17, 9, 7, 3, tzinfo=timezone.utc)
        m = _minutes(now, seed=seed)
        w = pd.Timestamp("2026-10-17 09:00", tz="UTC")
        row = updown.predict(m, w)
        hits.append((m.loc[w + pd.Timedelta(minutes=4), "close"] > row["open"]) == (row["call"] == "UP"))
    assert 0.62 < np.mean(hits) < 0.82


def test_step_sends_scores_and_summarises(tmp_path):
    sent = []
    t0 = datetime(2026, 10, 17, 9, 2, 3, tzinfo=timezone.utc)
    full = _minutes(t0 + timedelta(minutes=5), n=70)
    fetch = lambda now: full.loc[:pd.Timestamp(now).floor("1min")]  # noqa: E731
    send = lambda text, silent=False: sent.append((text, silent))  # noqa: E731
    first = updown.step(tmp_path, t0, fetch, send)
    assert first["window"].startswith("2026-10-17T09:00") and sent[0][1] is True
    assert "BTC 18:00-18:05" in sent[0][0]  # shown in JST
    assert updown.step(tmp_path, t0 + timedelta(seconds=20), fetch, send) is None  # same candle: no repeat
    updown.step(tmp_path, t0 + timedelta(minutes=5), fetch, send)
    df = updown.load(tmp_path)
    assert len(df) == 2 and df["result"].notna().sum() == 1
    w = pd.Timestamp("2026-10-17 09:00", tz="UTC")
    assert df.loc[0, "result"] == ("UP" if full.loc[w + pd.Timedelta(minutes=4), "close"] > df.loc[0, "open"] else "DOWN")
    assert "上次 18:00" in sent[-1][0]
    out = tb.answer("/predict", tmp_path, t0 + timedelta(minutes=6))
    assert "1/1" in out or "0/1" in out
    assert "跟頭 2 分鐘" in out


def test_summary_without_data(tmp_path):
    assert "未有紀錄" in updown.summary(tmp_path)
