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
    lb_row("0xlucky", 235_000, 0.0, 0.3, 0.3, 0.3, 400_000),     # whole month made in the last week
]}


def fill(coin, px, sz, closed=0.0, fee=0.1):
    return {"coin": coin, "px": str(px), "sz": str(sz), "side": "B", "time": 1, "dir": "Open Long",
            "closedPnl": str(closed), "fee": str(fee), "crossed": True}


FILLS = {
    "0xsteady": [fill("ADA", 0.6, 10_000, 0), fill("ADA", 0.65, 10_000, 500)] * 12 + [fill("DOGE", 0.1, 10_000, 50)],
    "0xbtcguy": [fill("HYPE", 30, 100, 200)] * 20 + [fill("BTC", 80_000, 0.01, -5)],
}
STATE = {"assetPositions": [{"type": "oneWay", "position": {"coin": "ADA", "szi": "-5000", "entryPx": "0.6",
                                                               "positionValue": "3000", "unrealizedPnl": "10"}}]}


class Resp:
    status_code = 200

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
    assert len(lb) == 8 and len(checked) == 2
    good = ht.ours(checked)
    assert list(good["address"]) == ["0xsteady"]  # BTC is a sliver of 0xbtcguy's volume
    r = good.iloc[0]
    assert r["our_fills"] == 24 and r["our_share"] > 0.9 and r["our_win_rate"] == 1.0
    assert r["pnl_ADA"] == round(12 * 500 - 24 * 0.1) and r["holds_now"] == "ADA short 3,000"
    text = ht.report(lb, checked)
    assert "1 trade our coins" in text and "0xsteady" in text and "0xbtcguy" not in text
    watch = ht.watchlist(good, datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert watch == [{"address": "0xsteady", "account_value": 50_000, "pnl_30d": 5998, "roi_30d": 0.12,
                      "our_fills_30d": 24, "our_win_rate": 1.0, "pnl_by_coin": {"ADA": 5998},
                      "scanned": "2026-10-15T00:00+00:00"}]


def test_losing_or_rare_traders_in_our_coins_are_dropped():
    checked = ht.pd.DataFrame([
        {"address": "0xa", "equity": 1e5, "our_share": 0.9, "our_fills": 100, "our_pnl_30d": -5000},
        {"address": "0xb", "equity": 1e5, "our_share": 0.9, "our_fills": 7, "our_pnl_30d": 5000},
        {"address": "0xc", "equity": 1e5, "our_share": 0.9, "our_fills": 30, "our_pnl_30d": 1000},
        {"address": "0xd", "equity": 1e4, "our_share": 0.5, "our_fills": 30, "our_pnl_30d": 1000},
        {"address": "0xe", "equity": 1e6, "our_share": 0.6, "our_fills": 500, "our_pnl_30d": 396}])  # 0.04%
    assert list(ht.ours(checked)["address"]) == ["0xd", "0xc"]


def test_lists_every_coin_the_winners_trade():
    lb, checked = ht.scan(session=Session(), now=datetime(2026, 10, 15, tzinfo=timezone.utc), pause=0,
                          progress=lambda *_: None)
    table = ht.coin_table(checked)
    assert set(table.index) == {"HYPE", "ADA", "BTC", "DOGE"}
    assert table.loc["HYPE", "traders"] == 1 and table.loc["HYPE", "pnl"] == round(20 * (200 - 0.1))
    assert table.loc["BTC", "traders_in_profit"] == 0 and abs(table["volume_share"].sum() - 1) < 0.01
    steady = checked.set_index("address").loc["0xsteady"]
    assert steady["top_coins"].startswith("ADA ") and "DOGE" in steady["top_coins"]
    text = ht.report(lb, checked)
    assert "HYPE" in text and "ADA *" in text and "most traded: ADA" in text
    assert ht.coin_label("@107") == "@107 (spot)"
