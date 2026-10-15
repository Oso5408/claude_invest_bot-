"""Hyperliquid API self-test against a fake Hyperliquid that answers in the real response format
(recorded shapes from the info endpoint docs), so the checks run in the cloud without network."""
from datetime import datetime, timezone

import pandas as pd
import pytest
import requests

from btcbot import hl_selftest as hs

NOW = datetime(2026, 10, 15, 12, 7, tzinfo=timezone.utc)
PRICES = {"ADA": 0.62, "BTC": 67_000.0, "ETH": 2_600.0, "SOL": 150.0, "HYPE": 40.0, "NEAR": 2.5, "DOGE": 0.12}
STEP_MS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
WHALE = "0x" + "ab" * 20


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self.data


class FakeHL:
    """Answers like api.hyperliquid.xyz/info and stats-data.hyperliquid.xyz. `breaks` turns on one fault."""

    def __init__(self, breaks=()):
        self.breaks = set(breaks)
        self.now_ms = int(NOW.timestamp() * 1000)

    def info(self, body):
        kind = body["type"]
        if kind == "allMids":
            mids = {k: str(v) for k, v in PRICES.items()}
            if "no_sol" in self.breaks:
                mids.pop("SOL")
            return mids
        if kind == "candleSnapshot":
            r = body["req"]
            step = STEP_MS[r["interval"]]
            end = self.now_ms - (10 * step if "stale" in self.breaks else 0)
            t, rows = r["startTime"] // step * step, []
            while t <= min(r["endTime"], end):
                px = PRICES[r["coin"]]
                row = {"t": t, "T": t + step - 1, "s": r["coin"], "i": r["interval"], "o": str(px), "c": str(px * 1.001),
                       "h": str(px * 1.002), "l": str(px * 0.999), "v": "1234.5", "n": 42}
                if "renamed" in self.breaks:
                    row["close"] = row.pop("c")
                rows.append(row)
                t += step
            return rows
        if kind == "fundingHistory":
            t = body["startTime"] // 3_600_000 * 3_600_000 + 3_600_000
            rows = []
            while t <= body["endTime"] and t <= self.now_ms and len(rows) < 500:
                rows.append({"coin": body["coin"], "fundingRate": "0.0000125", "premium": "0.0001", "time": t + 37})
                t += 3_600_000
            return rows
        if kind == "l2Book":
            px = PRICES[body["coin"]]
            return {"coin": body["coin"], "time": self.now_ms,
                    "levels": [[{"px": str(px * 0.9999), "sz": "1000", "n": 3}],
                               [{"px": str(px * 1.0001), "sz": "900", "n": 2}]]}
        if kind == "clearinghouseState":
            return {"assetPositions": [{"type": "oneWay", "position": {
                "coin": "BTC", "szi": "-2.5", "entryPx": "68000", "positionValue": "167500", "unrealizedPnl": "2500",
                "leverage": {"type": "cross", "value": 5}}}]}
        raise ValueError(kind)

    def leaderboard(self):
        rows = [{"ethAddress": f"0x{i:040x}" if i else WHALE, "accountValue": str(5_000_000 - i * 10_000),
                 "displayName": None, "prize": 0,
                 "windowPerformances": [["day", {"pnl": "1", "roi": "0", "vlm": "0"}],
                                        ["month", {"pnl": str(900_000 - i * 1000), "roi": "0.1", "vlm": "0"}]]}
                for i in range(30)]
        return {"leaderboardRows": rows}


@pytest.fixture
def fake_hl(monkeypatch, request):
    hl = FakeHL(getattr(request, "param", ()))

    class Session:
        def post(self, url, json=None, timeout=None):
            assert "hyperliquid" in url
            return FakeResponse(hl.info(json))

        def get(self, url, params=None, timeout=None):
            if "down_leaderboard" in hl.breaks:
                return FakeResponse({}, status=503)
            assert url.startswith("https://stats-data.hyperliquid.xyz/")
            return FakeResponse(hl.leaderboard())

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: Session().post(url, json, timeout))
    return hl


def run():
    seen = []
    res = hs.run_checks(now=NOW, progress=lambda n, total, r: seen.append((n, total)))
    assert seen == [(i, len(res)) for i in range(1, len(res) + 1)]
    return {r["name"]: r for r in res}


def test_all_endpoints_pass_on_healthy_api(fake_hl):
    res = run()
    assert len(res) == 17 and all(r["ok"] for r in res.values()), [r for r in res.values() if not r["ok"]]
    assert res["allMids (AI prices)"]["detail"] == {c: PRICES[c] for c in ("ADA", "BTC", "ETH", "SOL", "HYPE", "NEAR")}
    assert res["leaderboard (stats-data.hyperliquid.xyz)"]["detail"]["accounts"] == 30
    assert res["clearinghouseState top 3 whales"]["detail"][0]["biggest"] == "BTC"


@pytest.mark.parametrize("fake_hl", [("stale",)], indirect=True)
def test_stale_candles_fail(fake_hl):
    res = run()
    bad = res["candleSnapshot ADA 15m/1h/4h (AI)"]
    assert not bad["ok"] and any("stale" in p for p in bad["problems"])
    assert not res["rule paper feed ADA (1m, 1h, l2Book, funding)"]["ok"]


@pytest.mark.parametrize("fake_hl", [("no_sol",)], indirect=True)
def test_missing_coin_fails(fake_hl):
    assert run()["allMids (AI prices)"]["problems"] == ["SOL missing"]


@pytest.mark.parametrize("fake_hl", [("renamed",)], indirect=True)
def test_changed_response_shape_is_reported_not_crashed(fake_hl):
    res = run()
    assert not res["candleSnapshot BTC 15m/1h/4h (AI)"]["ok"]
    assert "KeyError" in res["candleSnapshot BTC 15m/1h/4h (AI)"]["problems"][0]
    assert res["allMids (AI prices)"]["ok"]


@pytest.mark.parametrize("fake_hl", [("down_leaderboard",)], indirect=True)
def test_leaderboard_down_fails_whales_too(fake_hl):
    res = run()
    assert not res["leaderboard (stats-data.hyperliquid.xyz)"]["ok"]
    assert res["clearinghouseState top 3 whales"]["problems"] == ["no leaderboard, nothing to look up"]


def test_candle_checks_catch_bad_values():
    idx = pd.date_range("2026-10-15 08:00", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": [1.1, 0.9, 1.1, 1.1], "low": 0.95, "close": 1.0, "volume": 1.0}, index=idx)
    p = hs.check_candles(df, "1h", pd.Timestamp("2026-10-15 11:30", tz="UTC"), "BTC")
    assert any("low/high" in x for x in p) and any("outside" in x for x in p)
    f = pd.Series([0.02], index=pd.DatetimeIndex([pd.Timestamp("2026-10-15 11:00", tz="UTC")]))
    assert any("looks wrong" in x for x in hs.check_funding(f, pd.Timestamp("2026-10-15 11:30", tz="UTC")))
