from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from btcbot import ai_trader as ai


class FakeMarket:
    def __init__(self, price=1.0):
        self.price = {"ADA": price, "BTC": 60000.0, "ETH": 3000.0, "SOL": 150.0}
        self.bars = {}  # coin -> 1m DataFrame

    def mids(self):
        return dict(self.price)

    def candles(self, coin, interval, start, end):
        df = self.bars.get(coin)
        if df is None:
            idx = pd.date_range(end - pd.Timedelta(hours=1), end, freq="1h", tz="UTC", name="open_time")
            p = self.price[coin]
            return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)
        return df[(df.index >= start) & (df.index <= end)]

    def funding(self, coin, start, end):
        idx = pd.date_range(start.ceil("1h"), end, freq="1h", tz="UTC")
        return pd.Series(0.0001, index=idx)

    def fear_greed(self):
        return {"2026-10-13": 40.0}

    def leaderboard(self):
        return [{"address": "0xabc", "account_value": 5e6, "pnl_30d": 1e6, "roi_30d": 0.2}]

    def positions(self, address):
        return [{"coin": "ETH", "size": 100.0, "entry": 2900.0, "value_usd": 300000}]


T0 = datetime(2026, 10, 14, 0, 2, tzinfo=timezone.utc)  # 09:02 JST


class ScriptBrain:
    """Calls the handlers like Claude would, without the API."""

    def __init__(self, script):
        self.script = script
        self.cost = 0.0
        self.calls = 0

    def run(self, context, handlers):
        self.calls += 1
        self.cost += 0.3
        self.results = []
        for name, kwargs in self.script:
            try:
                self.results.append(handlers[name](**kwargs))
            except Exception as e:
                self.results.append(e)
        return "done"

    def write(self, system, text):
        self.cost += 0.01
        return "review: idea wrong"


def order(**kw):
    base = dict(coin="ADA", side="long", margin_usd=20, leverage=3, stop_price=0.95, take_profit_price=0,
                max_hold_hours=0, strategy="test", reason="test")
    return ("place_order", {**base, **kw})


def test_limits_are_enforced_in_code(tmp_path):
    brain = ScriptBrain([order(margin_usd=25), order(leverage=4), order(stop_price=1.1), order(stop_price=0.5),
                         order(coin="DOGE"), order(), order(side="short", stop_price=1.05), order()])
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    t.step(T0)
    r = brain.results
    assert all(isinstance(x, Exception) for x in r[:5])
    assert r[5]["opened"] == 1 and r[6]["opened"] == 2
    assert "daily limit" in str(r[7])  # third trade of the day refused
    pos = t.state["positions"][0]
    assert pos["qty"] * pos["entry"] == pytest.approx(60.0)  # 20 USD x 3
    assert len(t.state["positions"]) == 2


def test_decides_once_per_slot_and_new_day_resets(tmp_path):
    brain = ScriptBrain([order()])
    m = FakeMarket()
    t = ai.AITrader(tmp_path, m, brain)
    t.step(T0)
    t.step(T0.replace(minute=7))
    assert brain.calls == 1  # same slot, no second call
    t.step(T0.replace(hour=12, minute=2))  # 21:02 JST, second slot
    assert brain.calls == 2 and t.entries_left() == 0
    t.step(T0.replace(day=15, minute=2))  # next Japan day
    assert brain.calls == 3 and t.entries_left() == 1


def test_stop_loss_closes_and_review_is_logged(tmp_path):
    m = FakeMarket()
    brain = ScriptBrain([order(stop_price=0.95, take_profit_price=1.2)])
    t = ai.AITrader(tmp_path, m, brain)
    t.step(T0)
    idx = pd.date_range(pd.Timestamp(T0).floor("min"), periods=10, freq="1min", name="open_time")
    lows = [1.0] * 5 + [0.94] + [1.0] * 4
    m.bars["ADA"] = pd.DataFrame({"open": 1.0, "high": 1.0, "low": lows, "close": 1.0, "volume": 1.0}, index=idx)
    st = t.step(T0.replace(minute=12))
    assert "stop_loss" in st["action"] and not t.state["positions"]
    tr = pd.read_csv(tmp_path / "trades.csv")
    assert tr["pnl_usd"].iloc[0] < -2.9  # about -5% of 60 USD plus costs
    assert tr["review"].iloc[0] == "review: idea wrong"
    assert t.state["api_cost"] == pytest.approx(0.31)


def test_pause_blocks_new_trades(tmp_path):
    (tmp_path / "PAUSE").write_text("")
    brain = ScriptBrain([order()])
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    st = t.step(T0)
    assert brain.calls == 0 and "PAUSE" in st["action"] and not t.state["positions"]


def test_whales_and_report(tmp_path):
    brain = ScriptBrain([("get_whale_positions", {"top": 3}), order()])
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    t.step(T0)
    assert brain.results[0][0]["positions"][0]["coin"] == "ETH"
    out = ai.daily_report(tmp_path, brain, T0.replace(minute=35))
    assert out.read_text() == "review: idea wrong"
    assert "Open trades" in ai.report_text(tmp_path, T0, tmp_path.parent)


class FakeClient:
    """Two API turns: a place_order call, then finish."""

    def __init__(self):
        self.turn = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.turn += 1
        usage = SimpleNamespace(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0,
                                server_tool_use=SimpleNamespace(web_search_requests=1))
        if self.turn == 1:
            blocks = [SimpleNamespace(type="tool_use", id="t1", name="place_order", input=dict(order()[1]))]
        else:
            blocks = [SimpleNamespace(type="tool_use", id="t2", name="finish", input={"summary": "bought ADA"})]
        return SimpleNamespace(stop_reason="tool_use", content=blocks, usage=usage)


def test_brain_tool_loop(tmp_path):
    brain = ai.Brain(client=FakeClient())
    t = ai.AITrader(tmp_path, FakeMarket(), brain)
    st = t.step(T0)
    assert "bought ADA" in st["action"] and len(t.state["positions"]) == 1
    d = pd.read_csv(tmp_path / "decisions.csv")
    assert d["new_trades"].iloc[0] == 1 and d["cost_usd"].iloc[0] > 0
