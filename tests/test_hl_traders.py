from datetime import datetime, timezone

from btcbot import hl_traders as ht


def lb_row(addr, equity, day, week, month, all_time, vlm):
    perf = lambda roi: {"pnl": str(roi * equity), "roi": str(roi), "vlm": str(vlm)}
    return {"ethAddress": addr, "accountValue": str(equity), "displayName": None, "prize": 0,
            "windowPerformances": [["day", perf(day)], ["week", perf(week)], ["month", perf(month)],
                                   ["allTime", perf(all_time)]]}


LEADERBOARD = {"leaderboardRows": [
    lb_row("0xsteady", 50_000, 0.01, 0.05, 0.40, 1.2, 500_000),
    lb_row("0xbtcguy", 200_000, 0.0, 0.02, 0.10, 0.5, 2_000_000),
    lb_row("0xtiny", 500, 0.5, 2.0, 9.0, 9.0, 10_000),          # too small
    lb_row("0xoneweek", 80_000, 0.1, -0.05, 0.6, 0.3, 400_000),  # down this week
    lb_row("0xloser", 80_000, 0.0, 0.01, 0.2, -0.4, 400_000),    # down all time
    lb_row("0xmaker", 1_000_000, 0.0, 0.01, 0.05, 0.3, 900_000_000),  # market maker turnover
    lb_row("0xvault", 300_000_000, 0.0, 0.01, 0.05, 0.3, 1e9),   # vault-sized
]}


def fill(coin, px, sz, closed=0.0, fee=0.1):
    return {"coin": coin, "px": str(px), "sz": str(sz), "side": "B", "time": 1, "dir": "Open Long",
            "closedPnl": str(closed), "fee": str(fee), "crossed": True}


FILLS = {
    "0xsteady": [fill("ADA", 0.6, 10_000, 0), fill("ADA", 0.65, 10_000, 500)] * 3 + [fill("DOGE", 0.1, 10_000, 50)],
    "0xbtcguy": [fill("HYPE", 30, 100, 200)] * 20 + [fill("BTC", 80_000, 0.01, -5)],
}
STATE = {"assetPositions": [{"type": "oneWay", "position": {"coin": "ADA", "szi": "-5000", "entryPx": "0.6",
                                                               "positionValue": "3000", "unrealizedPnl": "10"}}]}


class Resp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class Session:
    def get(self, url, timeout=None):
        return Resp(LEADERBOARD)

    def post(self, url, json=None, timeout=None):
        if json["type"] == "userFillsByTime":
            return Resp(FILLS[json["user"]])
        return Resp(STATE if json["user"] == "0xsteady" else {"assetPositions": []})


def test_shortlist_keeps_only_steady_real_traders():
    short = ht.shortlist(ht.parse_leaderboard(LEADERBOARD))
    assert list(short["address"]) == ["0xsteady", "0xbtcguy"]


def test_scan_finds_who_trades_our_coins():
    lb, checked = ht.scan(session=Session(), now=datetime(2026, 10, 15, tzinfo=timezone.utc), pause=0,
                          progress=lambda *_: None)
    assert len(lb) == 7 and len(checked) == 2
    good = ht.ours(checked)
    assert list(good["address"]) == ["0xsteady"]  # BTC is a sliver of 0xbtcguy's volume
    r = good.iloc[0]
    assert r["our_fills"] == 6 and r["our_share"] > 0.9 and r["our_win_rate"] == 1.0
    assert r["pnl_ADA"] == round(3 * 500 - 6 * 0.1) and r["holds_now"] == "ADA short 3,000"
    text = ht.report(lb, checked)
    assert "1 trade our coins" in text and "0xsteady" in text and "0xbtcguy" not in text
