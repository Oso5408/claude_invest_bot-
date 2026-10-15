from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from btcbot import copy_backtest as cb

T0 = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)
H = 3_600_000
M5 = 300_000
COST = 2 * cb.COST_PER_SIDE


def fill(coin, ms, side, px, sz, start, closed=0.0):
    return {"coin": coin, "px": str(px), "sz": str(sz), "side": side, "time": T0 + ms,
            "startPosition": str(start), "closedPnl": str(closed), "fee": "1", "crossed": False}


FILLS = [
    fill("ETH", 1 * H + 60_000, "B", 100, 1, 0),             # long ETH at 01:01
    fill("ETH", 5 * H + 60_000, "A", 110, 1, 1, closed=10),  # flat at 05:01
    fill("BTC", 2 * H + 60_000, "A", 200, 1, 0),             # short BTC at 02:01 ...
    fill("BTC", 2 * H + 20 * 60_000, "B", 190, 1, -1),       # ... closed at 02:20: the hourly check never sees it
    fill("ZEC", 3 * H, "B", 50, 10, 0),                      # not our coin
    fill("ZEC", 4 * H, "A", 60, 10, 10, closed=100),
]


def bars(interval_ms, n=200, start=T0):
    """Price = 100 + hours since T0 (ETH-like); BTC uses 200 - hours."""
    idx = pd.to_datetime(np.arange(n) * interval_ms + start, unit="ms", utc=True)
    hours = (np.arange(n) * interval_ms) / H
    return idx, hours


@pytest.fixture
def fake_post(monkeypatch):
    calls = []

    def post(url, body, session=None):
        calls.append(body["type"])
        if body["type"] == "userFillsByTime":
            return [f for f in FILLS if body["startTime"] <= f["time"] <= body["endTime"]]
        if body["type"] == "fundingHistory":
            if body["startTime"] > T0:
                return []
            return [{"time": T0 + h * H, "fundingRate": "0.0001"} for h in range(1, 24)]
        req = body["req"]
        step = {"5m": M5, "15m": 3 * M5, "1h": H}[req["interval"]]
        idx, hours = bars(step, n=24 * H // step)
        px = 100 + hours if req["coin"] == "ETH" else 200 - hours
        return [{"t": int(t.value // 1_000_000), "o": str(p), "h": str(p), "l": str(p), "c": str(p), "v": "1"}
                for t, p in zip(idx, px)]

    monkeypatch.setattr(cb, "_post", post)
    return calls


def test_copies_with_delay_costs_and_funding(fake_post):
    copied, theirs = cb.backtest(["0xabc"], days=1, now=datetime(2026, 10, 2, tzinfo=timezone.utc), pause=0,
                                 progress=lambda *_: None)
    assert set(theirs["coin"]) == {"ETH", "BTC", "ZEC"} and theirs.loc[theirs["coin"] == "ZEC", "ours"].item() is False
    eth = copied[copied["coin"] == "ETH"].set_index("mode")
    # their prices: 100 -> 110; funding paid by a long for hours 2..5 (4 x 0.01%)
    assert np.isclose(eth.loc["their prices", "net"], 0.10 - COST - 4 * 0.0001)
    # 5m check: next 5m open after 01:01 is 01:05 (101.0833), after 05:01 it is 05:05 (105.0833)
    assert eth.loc["5m check", "bars"] == "5m"
    assert np.isclose(eth.loc["5m check", "gross"], (100 + 5 + 5 / 60) / (100 + 1 + 5 / 60) - 1)
    # 1h check: 02:00 (102) -> 06:00 (106), funding hours 3..6
    assert np.isclose(eth.loc["1h check", "net"], 106 / 102 - 1 - COST - 4 * 0.0001)
    assert np.isclose(eth.loc["1h check", "pnl_usd"], eth.loc["1h check", "net"] * cb.NOTIONAL_USD)
    btc = copied[copied["coin"] == "BTC"].set_index("mode")
    assert bool(btc.loc["1h check", "missed"]) and not bool(btc.loc["5m check", "missed"])
    assert btc.loc["5m check", "gross"] > 0  # short into a falling price
    text = cb.report(copied, theirs, 1)
    assert "other coins 1 trades +98" in text and "1h check  0xabc" in text and "(+1 missed)" in text and "Most positions open at once (1h check): 1" in text


def test_fetch_fills_pages_when_capped(monkeypatch):
    many = [{"time": T0 + i, "coin": "ETH", "px": "1", "sz": "1", "side": "B", "startPosition": str(i)}
            for i in range(4500)]

    def post(url, body, session=None):  # newest 2000 in the range, like a capped answer
        rows = [f for f in many if body["startTime"] <= f["time"] <= body["endTime"]]
        return rows[-2000:]

    monkeypatch.setattr(cb, "_post", post)
    got = cb.fetch_fills("0xabc", T0, T0 + 10_000, pause=0)
    assert len(got) == 4500 and [int(r["time"]) for r in got] == sorted(int(r["time"]) for r in got)


def test_next_open_falls_back_to_coarser_candles():
    idx5 = pd.date_range("2026-10-01 10:00", periods=12, freq="5min", tz="UTC")
    idx1h = pd.date_range("2026-10-01 00:00", periods=24, freq="1h", tz="UTC")
    b = {"5m": pd.DataFrame({"open": np.arange(12.0)}, index=idx5),
         "1h": pd.DataFrame({"open": np.arange(24.0) + 100}, index=idx1h)}
    assert cb.next_open(b, pd.Timestamp("2026-10-01 10:07", tz="UTC"), ("5m", "1h"))[::2] == (2.0, "5m")
    assert cb.next_open(b, pd.Timestamp("2026-10-01 03:30", tz="UTC"), ("5m", "1h"))[::2] == (104.0, "1h")
