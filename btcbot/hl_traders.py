"""Scan the whole Hyperliquid leaderboard for good traders who trade our coins (the AI account's COINS: ADA, BTC, ETH, SOL, HYPE, NEAR).

Step 1, leaderboard (one download, every account): keep accounts that
- hold at least --min-equity USD (default 10,000) so it isn't one lucky bet on a tiny account,
- are up on the week, the month and all time, so it isn't one good day,
- made at least 10% of their 30-day PnL before the last 7 days (not all in one week),
- traded this month but not like a market maker (monthly volume < 100x equity),
- are not huge vault-like accounts (> 50M USD equity),
then rank by 30-day PnL / equity. Leaderboard ROI% is not used: a deposit after a lucky bet on a small
balance makes it huge (the first scan's top ROI, +2264%, was 2,264 USD on a 235k account).

Step 2, for the best --candidates (default 100): their fills of the last 30 days (userFillsByTime) and
open positions (clearinghouseState). Keep those with at least 20% of their traded volume and 20 fills in
our coins and a realized profit in them of at least 1% of equity, ranked by that profit / equity ("our_return").
It also lists what every checked account trades (all coins), per account and summed over all of them.

The best --keep (default 20) go to data/hl-traders/latest.json; the AI paper account watches those
(ai_trader.HLMarket.leaderboard) while the file is under 48 hours old, else the plain leaderboard.

Read-only public data, no key, no orders. Run on the VM (the cloud container can't reach Hyperliquid):

    .venv/bin/python -m btcbot.hl_traders                       # data/hl-traders/YYYY-MM-DD.csv + latest.json
    .venv/bin/python -m btcbot.hl_traders --min-equity 50000 --candidates 150
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from btcbot.ai_trader import COINS, LEADERBOARD_URL
from btcbot.hyperliquid import MAINNET, _post

WINDOWS = ("day", "week", "month", "allTime")
MAX_EQUITY = 50_000_000
MAX_TURNOVER = 100  # monthly volume / equity; above this it is market making / HFT, not copyable
MIN_OUR_SHARE = 0.2
MIN_OUR_FILLS = 20
MIN_OUR_RETURN = 0.01  # realized profit in our coins at least 1% of equity
MIN_EARLIER_PNL = 0.1  # share of 30-day PnL made before the last 7 days
LATEST = Path("data/hl-traders/latest.json")


def parse_leaderboard(data: dict) -> pd.DataFrame:
    """One row per account: address, equity and pnl/roi/vlm for every window."""
    rows = []
    for r in data.get("leaderboardRows", []):
        perf = {w: p for w, p in r.get("windowPerformances", [])}
        row = {"address": r["ethAddress"], "name": r.get("displayName") or "", "equity": float(r.get("accountValue", 0))}
        for w in WINDOWS:
            p = perf.get(w, {})
            for k in ("pnl", "roi", "vlm"):
                row[f"{w}_{k}"] = float(p.get(k, 0) or 0)
        rows.append(row)
    return pd.DataFrame(rows)


def shortlist(lb: pd.DataFrame, min_equity: float = 10_000) -> pd.DataFrame:
    """Steady winners, best 30-day PnL / equity first."""
    if lb.empty:
        return lb
    turnover = lb["month_vlm"] / lb["equity"].where(lb["equity"] > 0)
    keep = ((lb["equity"] >= min_equity) & (lb["equity"] <= MAX_EQUITY)
            & (lb["week_roi"] > 0) & (lb["month_roi"] > 0) & (lb["allTime_roi"] > 0) & (lb["allTime_pnl"] > 0)
            & (lb["month_pnl"] - lb["week_pnl"] >= MIN_EARLIER_PNL * lb["month_pnl"])
            & (lb["month_vlm"] > 0) & (turnover < MAX_TURNOVER))
    out = lb[keep].assign(turnover=turnover[keep].round(1), month_return=(lb["month_pnl"] / lb["equity"])[keep])
    return out.sort_values("month_return", ascending=False)


def coin_stats(fills: list[dict], coins=COINS) -> dict:
    """Share of traded volume in our coins, realized PnL per coin, win rate of closing fills."""
    total = sum(float(f["px"]) * float(f["sz"]) for f in fills)
    ours = [f for f in fills if f["coin"] in coins]
    vol = sum(float(f["px"]) * float(f["sz"]) for f in ours)
    closes = [float(f["closedPnl"]) for f in ours if float(f.get("closedPnl", 0) or 0) != 0]
    pnl = {c: round(sum(float(f.get("closedPnl", 0) or 0) - float(f.get("fee", 0) or 0)
                        for f in ours if f["coin"] == c)) for c in coins}
    every = {}  # every coin they traded: [volume, realized pnl after fees, fills]
    for f in fills:
        v = every.setdefault(f["coin"], [0.0, 0.0, 0])
        v[0] += float(f["px"]) * float(f["sz"])
        v[1] += float(f.get("closedPnl", 0) or 0) - float(f.get("fee", 0) or 0)
        v[2] += 1
    top = sorted(every.items(), key=lambda kv: -kv[1][0])
    return {"top_coins": ", ".join(f"{c} {v[0] / total:.0%}" for c, v in top[:5]) if total else "",
            "coins_json": json.dumps({c: [round(v[0]), round(v[1]), v[2]] for c, v in top}),"fills_30d": len(fills), "our_fills": len(ours), "our_share": round(vol / total, 2) if total else 0.0,
            "our_win_rate": round(sum(x > 0 for x in closes) / len(closes), 2) if closes else None,
            "our_pnl_30d": sum(pnl.values()), **{f"pnl_{c}": v for c, v in pnl.items()},
            "fills_capped": len(fills) >= 2000}


def positions_text(state: dict, coins=COINS) -> str:
    out = []
    for ap in state.get("assetPositions", []):
        p = ap["position"]
        if p["coin"] in coins and float(p["szi"]):
            side = "long" if float(p["szi"]) > 0 else "short"
            out.append(f"{p['coin']} {side} {float(p['positionValue']):,.0f}")
    return "; ".join(out)


def scan(min_equity: float = 10_000, candidates: int = 100, days: int = 30, session=None, now: datetime | None = None,
         pause: float = 1.0, progress=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every account on the leaderboard, the shortlist with fill stats)."""
    s = session or requests.Session()
    now = now or datetime.now(timezone.utc)
    resp = s.get(LEADERBOARD_URL, timeout=120)
    resp.raise_for_status()
    lb = parse_leaderboard(resp.json())
    short = shortlist(lb, min_equity).head(candidates)
    progress(f"leaderboard: {len(lb)} accounts, {len(shortlist(lb, min_equity))} steady winners, checking top {len(short)}")
    end = int(now.timestamp() * 1000)
    rows = []
    for n, r in enumerate(short.itertuples(index=False), 1):
        row = r._asdict()
        try:
            fills = _post(MAINNET, {"type": "userFillsByTime", "user": r.address, "startTime": end - days * 86_400_000,
                                    "endTime": end, "aggregateByTime": True}, s)
            row |= coin_stats(fills)
            row["holds_now"] = positions_text(_post(MAINNET, {"type": "clearinghouseState", "user": r.address}, s))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if n % 10 == 0 or n == len(short):
            progress(f"[{n}/{len(short)}] fills checked")
        time.sleep(pause)  # userFillsByTime is a heavy call; stay well under the rate limit
    return lb, pd.DataFrame(rows)


def ours(checked: pd.DataFrame) -> pd.DataFrame:
    if checked.empty or "our_share" not in checked:
        return checked.iloc[0:0]
    m = ((checked["our_share"] >= MIN_OUR_SHARE) & (checked["our_fills"] >= MIN_OUR_FILLS)
         & (checked["our_pnl_30d"] >= MIN_OUR_RETURN * checked["equity"]))
    good = checked[m].assign(our_return=checked["our_pnl_30d"] / checked["equity"])
    return good.sort_values("our_return", ascending=False)


def watchlist(good: pd.DataFrame, now: datetime, keep: int = 20) -> list[dict]:
    """The rows ai_trader reads: same keys as its leaderboard, plus what they did in our coins."""
    out = []
    for r in good.head(keep).itertuples(index=False):
        out.append({"address": r.address, "account_value": round(r.equity), "pnl_30d": round(r.our_pnl_30d),
                    "roi_30d": round(r.our_return, 3), "our_fills_30d": int(r.our_fills),
                    "our_win_rate": None if pd.isna(r.our_win_rate) else float(r.our_win_rate),
                    "pnl_by_coin": {c: int(getattr(r, "pnl_" + c)) for c in COINS if getattr(r, "pnl_" + c)},
                    "scanned": now.isoformat(timespec="minutes")})
    return out


def coin_table(checked: pd.DataFrame) -> pd.DataFrame:
    """What the checked steady winners trade, all coins: traders, share of their volume, realized PnL."""
    rows = []
    for addr, js in zip(checked.get("address", []), checked.get("coins_json", [])):
        if isinstance(js, str):
            for coin, (vol, pnl, n) in json.loads(js).items():
                rows.append({"coin": coin, "address": addr, "volume": vol, "pnl": pnl, "fills": n})
    if not rows:
        return pd.DataFrame(columns=["traders", "volume_share", "pnl", "traders_in_profit", "fills"])
    df = pd.DataFrame(rows)
    out = df.groupby("coin").agg(traders=("address", "nunique"), volume=("volume", "sum"), pnl=("pnl", "sum"),
                                 traders_in_profit=("pnl", lambda x: int((x > 0).sum())), fills=("fills", "sum"))
    out["volume_share"] = (out["volume"] / out["volume"].sum()).round(3)
    return out.sort_values(["traders", "volume"], ascending=False)[
        ["traders", "volume_share", "pnl", "traders_in_profit", "fills"]]


def coin_label(coin: str) -> str:
    """Spot pairs show up as '@107' or 'PURR/USDC', builder-deployed perps as 'dex:COIN'."""
    if coin.startswith("@") or "/" in coin:
        return f"{coin} (spot)"
    return coin


def report(lb: pd.DataFrame, checked: pd.DataFrame, top: int = 20) -> str:
    good = ours(checked)
    lines = [f"{len(lb)} accounts on the leaderboard, {len(checked)} steady winners checked, "
             f"{len(good)} trade our coins with a profit of >= {MIN_OUR_RETURN:.0%} of equity (>= {MIN_OUR_SHARE:.0%} of volume, >= {MIN_OUR_FILLS} fills "
             "in 30 days), best realized profit / equity first", ""]
    for r in good.head(top).itertuples(index=False):
        coins = ", ".join(f"{c} {int(getattr(r, 'pnl_' + c)):+,}" for c in COINS if getattr(r, "pnl_" + c))
        win = f"{r.our_win_rate:.0%}" if pd.notna(r.our_win_rate) else "-"
        lines.append(f"{r.address}  equity {r.equity:,.0f}  our coins 30d {r.our_return:+.1%} of equity"
                     f"  (all coins 30d {r.month_pnl:+,.0f}, 7d {r.week_pnl:+,.0f})")
        lines.append(f"    our coins {r.our_share:.0%} of volume, {int(r.our_fills)} fills, win {win}, realized 30d {coins or 0}"
                     f"{' (fills capped at 2000)' if r.fills_capped else ''}")
        lines.append(f"    most traded: {r.top_coins}")
        held = "lookup failed" if pd.isna(r.holds_now) else (r.holds_now or "nothing in our coins")
        lines.append(f"    holds now: {held}")
    table = coin_table(checked)
    lines += ["", f"What all {len(checked)} steady winners traded in 30 days (top 25 by number of traders):",
              f"{'coin':<20}{'traders':>8}{'in profit':>10}{'volume':>8}{'realized PnL':>16}"]
    for coin, r in table.head(25).iterrows():
        mark = " *" if coin in COINS else ""
        lines.append(f"{coin_label(coin) + mark:<20}{int(r.traders):>8}{int(r.traders_in_profit):>10}{r.volume_share:>8.1%}"
                     f"{r.pnl:>+16,.0f}")
    lines.append("(* = one of our coins; volume = share of all their volume; PnL is after fees, closed trades only)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Find steady Hyperliquid leaderboard traders who trade the AI account's coins")
    p.add_argument("--min-equity", type=float, default=10_000)
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--top", type=int, default=20, help="how many to print")
    p.add_argument("--keep", type=int, default=20, help="how many go to latest.json for the AI account")
    p.add_argument("--out", default="data/hl-traders")
    a = p.parse_args(argv)
    now = datetime.now(timezone.utc)
    lb, checked = scan(a.min_equity, a.candidates, now=now)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{now:%Y-%m-%d}.csv"
    checked.to_csv(path, index=False)
    coin_table(checked).to_csv(out / f"{now:%Y-%m-%d}-coins.csv")
    watch = watchlist(ours(checked), now, a.keep)
    (out / LATEST.name).write_text(json.dumps(watch, indent=1))
    print(report(lb, checked, a.top))
    print(f"\nsaved {path}; {len(watch)} traders in {out / LATEST.name} for the AI account")


if __name__ == "__main__":
    main()
