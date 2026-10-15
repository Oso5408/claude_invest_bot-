"""Scan the whole Hyperliquid leaderboard for good traders who trade our coins (ADA, BTC, ETH, SOL).

Step 1, leaderboard (one download, every account): keep accounts that
- hold at least --min-equity USD (default 10,000) so ROI isn't one lucky bet on a tiny account,
- are up on the week, the month and all time (ROI > 0 in each), so it isn't one good day,
- traded this month but not like a market maker (monthly volume < 100x equity),
- are not huge vault-like accounts (> 50M USD equity),
then rank by 30-day ROI.

Step 2, for the best --candidates (default 100): their fills of the last 30 days (userFillsByTime) and
open positions (clearinghouseState). Keep those where our coins are at least 20% of their traded volume
and at least 5 fills, and show their realized PnL per coin, win rate on closing fills and what they hold now.

Read-only public data, no key, no orders. Run on the VM (the cloud container can't reach Hyperliquid):

    .venv/bin/python -m btcbot.hl_traders                       # writes data/hl-traders/YYYY-MM-DD.csv
    .venv/bin/python -m btcbot.hl_traders --min-equity 50000 --candidates 150
"""
from __future__ import annotations

import argparse
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
MIN_OUR_FILLS = 5


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
    """Steady winners, best 30-day ROI first."""
    if lb.empty:
        return lb
    turnover = lb["month_vlm"] / lb["equity"].where(lb["equity"] > 0)
    keep = ((lb["equity"] >= min_equity) & (lb["equity"] <= MAX_EQUITY)
            & (lb["week_roi"] > 0) & (lb["month_roi"] > 0) & (lb["allTime_roi"] > 0) & (lb["allTime_pnl"] > 0)
            & (lb["month_vlm"] > 0) & (turnover < MAX_TURNOVER))
    return lb[keep].assign(turnover=turnover[keep].round(1)).sort_values("month_roi", ascending=False)


def coin_stats(fills: list[dict], coins=COINS) -> dict:
    """Share of traded volume in our coins, realized PnL per coin, win rate of closing fills."""
    total = sum(float(f["px"]) * float(f["sz"]) for f in fills)
    ours = [f for f in fills if f["coin"] in coins]
    vol = sum(float(f["px"]) * float(f["sz"]) for f in ours)
    closes = [float(f["closedPnl"]) for f in ours if float(f.get("closedPnl", 0) or 0) != 0]
    pnl = {c: round(sum(float(f.get("closedPnl", 0) or 0) - float(f.get("fee", 0) or 0)
                        for f in ours if f["coin"] == c)) for c in coins}
    return {"fills_30d": len(fills), "our_fills": len(ours), "our_share": round(vol / total, 2) if total else 0.0,
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
    m = (checked["our_share"] >= MIN_OUR_SHARE) & (checked["our_fills"] >= MIN_OUR_FILLS)
    return checked[m].sort_values("month_roi", ascending=False)


def report(lb: pd.DataFrame, checked: pd.DataFrame, top: int = 20) -> str:
    good = ours(checked)
    lines = [f"{len(lb)} accounts on the leaderboard, {len(checked)} steady winners checked, "
             f"{len(good)} trade our coins (>= {MIN_OUR_SHARE:.0%} of volume, >= {MIN_OUR_FILLS} fills in 30 days)", ""]
    for r in good.head(top).itertuples(index=False):
        coins = ", ".join(f"{c} {getattr(r, 'pnl_' + c):+,}" for c in COINS if getattr(r, "pnl_" + c))
        win = f"{r.our_win_rate:.0%}" if pd.notna(r.our_win_rate) else "-"
        lines.append(f"{r.address}  equity {r.equity:,.0f}  ROI 30d {r.month_roi:+.0%} / 7d {r.week_roi:+.0%} / all {r.allTime_roi:+.0%}")
        lines.append(f"    our coins {r.our_share:.0%} of volume, {r.our_fills} fills, win {win}, realized 30d {coins or 0}"
                     f"{' (fills capped at 2000)' if r.fills_capped else ''}")
        lines.append(f"    holds now: {r.holds_now or 'nothing in our coins'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Find steady Hyperliquid leaderboard traders who trade ADA/BTC/ETH/SOL")
    p.add_argument("--min-equity", type=float, default=10_000)
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default="data/hl-traders")
    a = p.parse_args(argv)
    now = datetime.now(timezone.utc)
    lb, checked = scan(a.min_equity, a.candidates, now=now)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{now:%Y-%m-%d}.csv"
    checked.to_csv(path, index=False)
    print(report(lb, checked, a.top))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
