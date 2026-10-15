from datetime import datetime, timezone

import numpy as np
import pandas as pd

from btcbot import whale_style as ws

T0 = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)
H = 3_600_000


def fill(coin, h, side, px, sz, start, closed=0.0, crossed=True):
    return {"coin": coin, "px": str(px), "sz": str(sz), "side": side, "time": T0 + h * H,
            "startPosition": str(start), "closedPnl": str(closed), "fee": "1", "crossed": crossed}


FILLS = [
    fill("ETH", 0, "A", 2500, 1, 3),                 # closing part of a trade opened before the window: skipped
    fill("ETH", 1, "A", 2500, 2, 2, closed=50),      # flat: skip ends
    fill("BTC", 2, "B", 100, 1, 0, crossed=False),    # long 1 @100 (limit)
    fill("BTC", 5, "B", 110, 1, 1),                   # add at a worse price (adding to a winner)
    fill("BTC", 8, "A", 120, 1, 2, closed=15),        # partial exit
    fill("BTC", 30, "A", 125, 1, 1, closed=20),       # flat: 28h trade
    fill("SOL", 40, "A", 50, 4, 0),                   # short 4 @50
    fill("SOL", 41, "A", 52, 4, -4),                  # add at a better price for a short (averaging up)
    fill("SOL", 45, "B", 55, 10, -8, closed=-36),     # flip to long 2
    fill("SOL", 50, "A", 56, 2, 2, closed=2),         # flat
    fill("ETH", 60, "B", 2600, 1, 0),                 # still open
]


def test_round_trips_rebuilds_trades_from_fills():
    closed, still_open = ws.round_trips(FILLS)
    assert [(t["coin"], t["side"]) for t in closed] == [("BTC", 1), ("SOL", -1), ("SOL", 1)]
    btc = closed[0]
    assert btc["hours"] == 28 and btc["adds"] == 1 and btc["adds_better"] == 0 and btc["partial_exits"] == 1
    assert btc["avg_entry"] == 105 and btc["avg_exit"] == 122.5 and btc["maker_fills"] == 1
    assert btc["pnl"] == 15 + 20 - 4  # four fees of 1
    sol = closed[1]
    assert sol["adds_better"] == 1 and sol["avg_entry"] == 51 and sol["return"] < 0
    assert [t["coin"] for t in still_open] == ["ETH"]


def test_entry_context_and_profile():
    idx = pd.date_range(pd.Timestamp(T0 - 8 * 24 * H, unit="ms", tz="UTC"), periods=8 * 24 + 60, freq="1h")
    up = pd.DataFrame({"open": 0.0, "high": np.linspace(60, 130, len(idx)) + 1, "low": np.linspace(60, 130, len(idx)) - 1,
                       "close": np.linspace(60, 130, len(idx)), "volume": 1.0}, index=idx)
    candles = {"BTC": up * [1, 1, 1, 1, 1], "SOL": up * [1, 0.5, 0.5, 0.5, 1]}
    closed, still_open = ws.round_trips(FILLS)
    t = ws.entry_context(pd.DataFrame(closed), candles)
    assert t.loc[0, "with_trend"] == 1.0 and t.loc[1, "with_trend"] == 0.0  # long in an uptrend, short against it
    p = ws.profile(t, still_open)
    assert p["trades"] == 3 and p["open_now"] == 1 and p["style"] == "day trader"
    assert p["win_rate"] == round(2 / 3, 2) and p["long_share"] == round(2 / 3, 2)
    text = ws.summary_text({"0xabc": p | {"fills_capped": False}, "0xdef": {"error": "HTTPError: 429"}})
    assert "day trader" in text and "0xdef  HTTPError" in text and "Overall (1 traders)" in text


def test_analyse_pulls_fills_and_candles(monkeypatch):
    calls = []

    def fake_post(url, body, session=None):
        calls.append(body["type"])
        if body["type"] == "userFillsByTime":
            return FILLS
        r = body["req"]
        ts = range(r["startTime"] // H * H, r["endTime"], H)
        return [{"t": t, "o": "100", "h": "101", "l": "99", "c": "100", "v": "1"} for t in ts]

    monkeypatch.setattr(ws, "_post", fake_post)
    out = ws.analyse(["0xabc"], now=datetime(2026, 10, 4, tzinfo=timezone.utc), pause=0, progress=lambda *_: None)
    assert out["0xabc"]["trades"] == 3 and out["0xabc"]["fills"] == len(FILLS)
    assert calls.count("candleSnapshot") == 2  # BTC and SOL, once each
