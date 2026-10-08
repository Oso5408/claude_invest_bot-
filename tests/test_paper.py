import json

import numpy as np
import pandas as pd
import pytest

from btcbot import risk
from btcbot.paper import PaperConfig, PaperTrader, rollovers_between


def test_rollovers_between():
    t = lambda s: pd.Timestamp(s, tz="Asia/Tokyo").tz_convert("UTC")
    assert rollovers_between(t("2026-10-08 05:00"), t("2026-10-08 06:30"), 6) == 1
    assert rollovers_between(t("2026-10-08 06:00"), t("2026-10-08 07:00"), 6) == 0  # already counted
    assert rollovers_between(t("2026-10-08 07:00"), t("2026-10-10 06:00"), 6) == 2
    assert rollovers_between(t("2026-10-08 07:00"), t("2026-10-08 23:00"), 6) == 0


class FakeFeed:
    """Serves candles cut off at `now`, like the live API (the current candle is partial)."""

    def __init__(self, minutes: pd.DataFrame):
        self.m = minutes
        self.now = None

    def candles(self, interval, now, days):
        self.now = pd.Timestamp(now)
        m = self.m[self.m.index < self.now]
        rule = {"1min": "1min", "1hour": "1h"}[interval]
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        return m.resample(rule, label="left", closed="left").agg(agg).dropna()

    def bid_ask(self):
        last = self.m[self.m.index < self.now]["close"].iloc[-1]
        return last * 0.9995, last * 1.0005


def _minutes():
    rng = np.random.default_rng(1)
    n_flat = 60 * 24 * 6
    flat = 100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n_flat)))
    up = np.linspace(flat[-1], flat[-1] * 1.25, 60 * 24)  # strong breakout day
    down = np.linspace(up[-1], up[-1] * 0.85, 60 * 12)  # sharp reversal
    close = np.concatenate([flat, up, down])
    vol = np.full(len(close), 10.0)
    vol[n_flat:n_flat + 240] = 200.0  # volume surge on the breakout
    idx = pd.date_range("2026-09-01", periods=len(close), freq="min", tz="UTC", name="open_time")
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0002,
                         "low": np.minimum(open_, close) * 0.9998, "close": close, "volume": vol}, index=idx)


def test_paper_trader_enters_trails_and_exits(tmp_path):
    m = _minutes()
    feed = FakeFeed(m)
    cfg = PaperConfig(history_days=6, allow_short=False)
    times = pd.date_range(m.index[0] + pd.Timedelta(days=3), m.index[-1], freq="30min")
    actions = []
    for t in times:
        trader = PaperTrader(tmp_path, cfg)  # reload from disk every run, like cron
        actions.append(trader.step(feed, t.to_pydatetime()))
    assert actions[0]["action"].startswith("started")
    assert any(a["action"].startswith("long") for a in actions)
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert len(trades) >= 1
    first = trades.iloc[0]
    assert first["side"] == 1 and first["reason"] == "trailing_stop" and first["return"] > 0.05
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["collateral"] > 30_000
    eq = pd.read_csv(tmp_path / "equity.csv")
    assert len(eq) == len(times)


def test_paper_trader_respects_2x(tmp_path):
    m = _minutes()
    feed = FakeFeed(m)
    cfg = PaperConfig(history_days=6)
    trader = PaperTrader(tmp_path, cfg)
    trader.strat.kelly_fraction = 10.0  # asks for far more than 2x
    times = pd.date_range(m.index[0] + pd.Timedelta(days=3), m.index[-1], freq="1h")
    for t in times:
        st = trader.step(feed, t.to_pydatetime())
        if st["qty"]:
            assert abs(st["qty"]) * st["price"] <= risk.MAX_LEVERAGE * st["equity"] * 1.05


class FngFeed(FakeFeed):
    def __init__(self, minutes, value, fail=False):
        super().__init__(minutes)
        self.value, self.fail = value, fail

    def fng(self):
        if self.fail:
            raise ConnectionError("down")
        days = pd.date_range(self.m.index[0].floor("D") - pd.Timedelta(days=2), self.now.floor("D"), freq="D")
        return pd.Series(self.value, index=days)


def _sides(tmp_path, feed, cfg):
    m = feed.m
    for t in pd.date_range(m.index[0] + pd.Timedelta(days=3), m.index[-1], freq="1h"):
        PaperTrader(tmp_path, cfg).step(feed, t.to_pydatetime())
    tr = tmp_path / "trades.csv"
    return set(pd.read_csv(tr)["side"]) if tr.exists() else set()


def test_paper_fng_filter_blocks_shorts_in_greed(tmp_path):
    m = _minutes()
    cfg = PaperConfig(history_days=6, fng_short_max=50)
    assert -1 in _sides(tmp_path / "fear", FngFeed(m, 20.0), cfg)
    assert -1 not in _sides(tmp_path / "greed", FngFeed(m, 80.0), cfg)
    assert -1 not in _sides(tmp_path / "down", FngFeed(m, 20.0, fail=True), cfg)  # unknown -> no shorts
