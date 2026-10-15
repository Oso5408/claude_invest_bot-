"""Would our 4h breakout work on another Hyperliquid coin? Same strategy, same settings, no re-tuning.

For each coin: Hyperliquid 4h candles (the last 5000, about 833 days, or since listing) and hourly funding,
then the three account setups we paper trade, with Hyperliquid costs (taker 0.045%, 0.05% slippage, hourly
funding), 200 USD:
- 2x both sides, 2x with shorts only when Fear & Greed <= 50 (paper-hl-2x-fng), 1x long only (paper-hl-1x-long)
Per coin and setup: return, max drawdown, trades, win rate, average trade, share of profit from the best
5 trades, every half-year block, and the same with 0.15% slippage. ADA is run over the same dates as a
yardstick. Order sizes use each coin's Hyperliquid size step.

Read-only public data, no key, no orders. Run on the VM (the cloud container can't reach Hyperliquid):

    .venv/bin/python -m btcbot.coin_check --coins NEAR HYPE      # also saves data/coin-check/<COIN>_4h.csv
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from btcbot import breakout
from btcbot.hyperliquid import account_config, fetch_candles, fetch_funding

SIZE_STEP = {"ADA": 1.0, "NEAR": 0.1, "HYPE": 0.01, "SOL": 0.01, "ETH": 0.0001, "BTC": 0.00001}
SETUPS = {
    "2x both": (breakout.BreakoutConfig(), False),
    "2x F&G<=50 shorts": (breakout.BreakoutConfig(fng_short_max=50), True),
    "1x long": (breakout.BreakoutConfig(allow_short=False, max_leverage=1.0), False),
}


def load(coin: str, out: Path, refresh: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    from btcbot.data import load_csv
    from btcbot.hyperliquid import load_funding
    cpath, fpath = out / f"{coin}_4h.csv", out / f"{coin}_funding.csv"
    fresh = cpath.exists() and fpath.exists() and time.time() - cpath.stat().st_mtime < 12 * 3600
    if not fresh and (refresh or not cpath.exists()):  # a rerun within 12 hours reuses today's download
        candles = fetch_candles(coin, "4h")
        candles.to_csv(cpath)
        fetch_funding(coin, int(candles.index[0].timestamp() * 1000)).to_csv(fpath)
    return load_csv(cpath), load_funding(fpath)


def acct_for(coin: str, slippage: float = 0.0005):
    step = SIZE_STEP.get(coin, 0.01)
    return replace(account_config(200), slippage=slippage, min_order=step, size_step=step)


def run_one(df: pd.DataFrame, funding: pd.Series, fng: pd.Series | None, coin: str, setup: str,
            slippage: float = 0.0005) -> dict:
    cfg, use_fng = SETUPS[setup]
    res = breakout.run(df, cfg, acct_for(coin, slippage), fng if use_fng else None, funding)
    eq, tr, s = res.equity, res.trades, res.stats
    out = {"coin": coin, "setup": setup, "start": f"{df.index[0]:%Y-%m-%d}", "return": s["total_return"],
           "max_dd": s["max_drawdown"], "trades": len(tr), "win_rate": s["win_rate"],
           "avg_trade": float(tr["return"].mean()) if len(tr) else np.nan, "buy_hold": s["buy_and_hold_return"]}
    if len(tr):
        before = [eq[eq.index < t].iloc[-1] if (eq.index < t).any() else 200.0 for t in tr["entry_time"]]
        pnl = eq.reindex(tr["exit_time"]).to_numpy() - np.array(before)
        total = eq.iloc[-1] - 200
        out["top5_share"] = float(np.sort(pnl)[::-1][:5].sum() / total) if total > 0 else np.nan
    blocks = eq.resample("6MS").last()
    starts = pd.concat([pd.Series([200.0], index=[eq.index[0]]), blocks.iloc[:-1]]).to_numpy()
    out["half_years"] = {f"{t:%Y-%m}": float(v / s0 - 1) for t, v, s0 in zip(blocks.index, blocks.to_numpy(), starts)}
    return out


def check(coins: list[str], data: dict, fng: pd.Series, yardstick: str = "ADA") -> list[dict]:
    rows = []
    for coin in coins:
        df, fund = data[coin]
        for setup in SETUPS:
            r = run_one(df, fund, fng, coin, setup)
            r["return_slip_0.15%"] = run_one(df, fund, fng, coin, setup, slippage=0.0015)["return"]
            if yardstick in data and coin != yardstick:
                ydf, yfund = data[yardstick]
                ydf = ydf[ydf.index >= df.index[0]]
                y = run_one(ydf, yfund, fng, yardstick, setup)
                r[f"{yardstick}_same_dates"] = y["return"]
                r[f"{yardstick}_same_dates_dd"] = y["max_dd"]
            rows.append(r)
    return rows


def report(rows: list[dict], yardstick: str = "ADA") -> str:
    pct = lambda x: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.0%}"
    lines = []
    for r in rows:
        lines.append(f"{r['coin']} / {r['setup']} (from {r['start']}): return {pct(r['return'])}, max DD {pct(-abs(r['max_dd']))}, "
                     f"{r['trades']} trades, win {r['win_rate']:.0%}, avg trade {r['avg_trade']:+.2%}, "
                     f"top 5 trades = {pct(r.get('top5_share'))} of profit")
        extra = f"    buy & hold {pct(r['buy_hold'])}; with 0.15% slippage {pct(r['return_slip_0.15%'])}"
        if f"{yardstick}_same_dates" in r:
            extra += (f"; {yardstick} same dates {pct(r[f'{yardstick}_same_dates'])} "
                      f"(DD {pct(-abs(r[f'{yardstick}_same_dates_dd']))})")
        lines.append(extra)
        lines.append("    half years: " + ", ".join(f"{k} {pct(v)}" for k, v in r["half_years"].items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Our 4h breakout on other Hyperliquid coins, unchanged")
    p.add_argument("--coins", nargs="+", default=["NEAR", "HYPE"])
    p.add_argument("--yardstick", default="ADA")
    p.add_argument("--out", default="data/coin-check")
    p.add_argument("--no-refresh", action="store_true", help="reuse the saved CSVs")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from btcbot.data import fetch_fng
    fng = fetch_fng()
    data = {}
    for coin in dict.fromkeys([*a.coins, a.yardstick]):
        print(f"downloading {coin} 4h candles and funding ...", flush=True)
        data[coin] = load(coin, out, refresh=not a.no_refresh)
        print(f"    {len(data[coin][0])} candles from {data[coin][0].index[0]:%Y-%m-%d}", flush=True)
    rows = check(a.coins, data, fng, a.yardstick)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print()
    print(report(rows, a.yardstick))
    print(f"\nsaved {out / 'results.csv'}")


if __name__ == "__main__":
    main()
