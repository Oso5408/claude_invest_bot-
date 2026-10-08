import json
from datetime import datetime, timezone

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


class FakeAdvisor:
    def __init__(self, action, multiplier):
        from btcbot.advisor import Decision
        self.d = Decision(action, "test", multiplier)
        self.calls = []

    def decide(self, ctx):
        self.calls.append(ctx)
        return self.d


def _run_with(tmp_path, advisor):
    m = _minutes()
    feed = FngFeed(m, 20.0)
    cfg = PaperConfig(history_days=6, allow_short=False)
    out = []
    for t in pd.date_range(m.index[0] + pd.Timedelta(days=3), m.index[-1], freq="30min"):
        out.append(PaperTrader(tmp_path, cfg, advisor=advisor).step(feed, t.to_pydatetime()))
    return out


def test_claude_skip_blocks_entry_and_is_logged(tmp_path):
    adv = FakeAdvisor("skip", 0.0)
    out = _run_with(tmp_path, adv)
    assert adv.calls and "rules_leverage" in adv.calls[0] and "fear_greed_last_7_days" in adv.calls[0]
    assert not (tmp_path / "trades.csv").exists()
    assert all(o["qty"] == 0 for o in out)
    log = pd.read_csv(tmp_path / "claude.csv")
    assert (log["action"] == "skip").all()


def test_claude_half_and_never_more_than_rules(tmp_path):
    full = _run_with(tmp_path / "full", FakeAdvisor("go", 1.0))
    half = _run_with(tmp_path / "half", FakeAdvisor("half", 0.5))
    big = _run_with(tmp_path / "big", FakeAdvisor("go", 5.0))  # a bad multiplier is clipped to 1
    q = lambda out: max(abs(o["qty"]) for o in out)
    assert 0 < q(half) < q(full) and q(big) == q(full)


def test_advisor_error_follows_rules():
    from btcbot.advisor import ClaudeAdvisor

    class Boom:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    raise ConnectionError("no network")

    d = ClaudeAdvisor(client=Boom()).decide({"x": 1})
    assert d.action == "go" and d.multiplier == 1.0 and "ConnectionError" in d.error


def test_daily_review_reads_every_account(tmp_path):
    from btcbot.advisor import daily_review
    _run_with(tmp_path / "paper-claude", FakeAdvisor("go", 1.0))
    _run_with(tmp_path / "paper-rules", None)

    class Reviewer:
        def review(self, summary):
            self.summary = summary
            return "ok"

    r = Reviewer()
    out = daily_review(tmp_path, r, now=datetime(2026, 9, 10, tzinfo=timezone.utc))
    assert out.read_text() == "ok"
    assert "## paper-claude" in r.summary and "## paper-rules" in r.summary and "Claude entry decisions" in r.summary


def test_hyperliquid_feed_and_funding(tmp_path, monkeypatch):
    from btcbot import hyperliquid as hl
    from btcbot.paper import HyperliquidFeed

    m = _minutes()
    m[["open", "high", "low", "close"]] /= 100  # USD-sized prices

    def fake_post(url, body, session=None):
        if body["type"] == "candleSnapshot":
            r = body["req"]
            lo, hi = pd.Timestamp(r["startTime"], unit="ms", tz="UTC"), pd.Timestamp(r["endTime"], unit="ms", tz="UTC")
            rule = {"1m": "1min", "1h": "1h"}[r["interval"]]
            agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            part = m[(m.index >= lo) & (m.index < hi)].resample(rule).agg(agg).dropna()
            return [{"t": int(t.timestamp() * 1000), "o": str(x.open), "h": str(x.high), "l": str(x.low),
                     "c": str(x.close), "v": str(x.volume)} for t, x in part.iterrows()]
        if body["type"] == "l2Book":
            return {"levels": [[{"px": "1.0"}], [{"px": "1.001"}]]}
        if body["type"] == "fundingHistory":
            hours = pd.date_range(pd.Timestamp(body["startTime"], unit="ms", tz="UTC").ceil("h"),
                                  pd.Timestamp(body["endTime"], unit="ms", tz="UTC"), freq="1h")
            return [{"coin": "ADA", "fundingRate": "0.0001", "premium": "0", "time": int(h.timestamp() * 1000)}
                    for h in hours]
        raise AssertionError(body)

    monkeypatch.setattr(hl, "_post", fake_post)
    feed = HyperliquidFeed("ADA")
    feed.fng = lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    cfg = PaperConfig(symbol="ADA", history_days=6, allow_short=False, initial_jpy=200)
    out = []
    for t in pd.date_range(m.index[0] + pd.Timedelta(days=3), m.index[-1], freq="1h"):
        out.append(PaperTrader(tmp_path, cfg, hl.account_config(200)).step(feed, t.to_pydatetime()))
    assert any(o["action"].startswith("long") for o in out)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["leverage_fees"] > 0  # longs paid the positive funding
