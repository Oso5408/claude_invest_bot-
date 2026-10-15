"""Would copying a few Hyperliquid traders have made money for us? A backtest from their real fills.

For each address it pulls the fills of the last --days (userFillsByTime, paged; Hyperliquid only keeps about the
10,000 most recent fills per account), rebuilds their trades with whale_style.round_trips (flat -> position ->
flat, adds and partial exits folded in) and copies each trade in our coins (the AI account's COINS):
- we open the same side when we would first see their position and close when we would first see it gone;
  their adds and partial exits are ignored (one fixed-size position per trade)
- three ways of seeing it:
    "their prices"  their own average entry and exit (the best case, no delay; our costs still apply)
    "5m check"      the open of the first 5-minute candle after their entry / exit (a check every 5 minutes;
                    15m or 1h candles where 5m ones are no longer served, see the report)
    "1h check"      the open of the first 1h candle after it (the AI account's hourly look)
  a trade they open and close between two checks is never seen, so it is counted as missed
- costs as in the paper accounts: taker fee 0.045% + slippage 0.05% per side, plus hourly funding
- size: 20 USD margin x 2x = 40 USD per trade, so the USD totals are what a 200 USD account would have seen

Trades already open when the window starts are skipped (their start isn't visible). Read-only public data, no
key, no orders. Run on the VM (the cloud container can't reach Hyperliquid):

    .venv/bin/python -m btcbot.copy_backtest                      # the five swing traders picked on 2026-10-15
    .venv/bin/python -m btcbot.copy_backtest --address 0xabc... --days 60
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from btcbot.ai_trader import COINS, SLIPPAGE
from btcbot.hyperliquid import MAINNET, TAKER_FEE, _post, parse_candles, parse_funding
from btcbot.whale_style import round_trips

FIVE = ("0x28280399d9afd68a3555c65ca6a75a1bd55693a0", "0x405eafc7534c349d5803b206a7cf5f0ab7686af4",
        "0xb69eec8fea0081343c3b09e652ba1fa7dcf7033a", "0x07eac6dc2af876d2f0402f6804cc6ceaa0cd20cd",
        "0xa2e366c4e2d453cab49cc9c64f0670fd2fd060a3")
NOTIONAL_USD = 40.0  # 20 USD margin x 2x
COST_PER_SIDE = TAKER_FEE + SLIPPAGE
MODES = ("their prices", "5m check", "1h check")
CHECK_BARS = {"5m check": ("5m", "15m", "1h"), "1h check": ("1h",)}  # finest first
H = 3_600_000


def fetch_fills(addr: str, start: int, end: int, session=None, pause: float = 0.5, max_calls: int = 30) -> list[dict]:
    """Every fill in [start, end]. A call returns at most 2000, so uncovered parts are asked for again."""
    seen, out, todo, calls = set(), [], [(start, end)], 0
    while todo and calls < max_calls:
        a, b = todo.pop()
        rows = _post(MAINNET, {"type": "userFillsByTime", "user": addr, "startTime": a, "endTime": b,
                               "aggregateByTime": True}, session)
        calls += 1
        for r in rows:
            key = (r["time"], r["coin"], r["px"], r["sz"], r["side"], r.get("startPosition"))
            if key not in seen:
                seen.add(key)
                out.append(r)
        if len(rows) >= 2000:
            ts = [int(r["time"]) for r in rows]
            if min(ts) > a:
                todo.append((a, min(ts) - 1))
            if max(ts) < b:
                todo.append((max(ts) + 1, b))
        time.sleep(pause)
    return sorted(out, key=lambda r: int(r["time"]))


def next_open(bars: dict[str, pd.DataFrame], t: pd.Timestamp, intervals: tuple[str, ...]) -> tuple[float, pd.Timestamp, str]:
    """Open of the first candle starting after t, from the finest interval that still covers t."""
    for iv in intervals:
        df = bars.get(iv)
        if df is None or not len(df) or df.index[0] > t:
            continue
        later = df.index[df.index > t]
        if len(later):
            return float(df.loc[later[0], "open"]), later[0], iv
    return np.nan, pd.NaT, ""


def copy_trade(tr: dict, mode: str, bars: dict[str, pd.DataFrame], funding: pd.Series) -> dict | None:
    """Our result from copying one of their trades; None when we would never have seen it or lack prices."""
    if mode == "their prices":
        entry, t_in, exit_, t_out, iv = tr["avg_entry"], tr["open"], tr["avg_exit"], tr["close"], "fills"
    else:
        entry, t_in, iv = next_open(bars, tr["open"], CHECK_BARS[mode])
        if np.isnan(entry):
            return None
        if t_in >= tr["close"]:
            return {"missed": True}  # opened and closed between two checks
        exit_, t_out, iv_out = next_open(bars, tr["close"], CHECK_BARS[mode])
        if np.isnan(exit_):
            return None
        iv = iv if iv == iv_out else f"{iv}/{iv_out}"
    side = tr["side"]
    gross = side * (exit_ / entry - 1)
    fund = float(funding[(funding.index > t_in) & (funding.index <= t_out)].sum()) if len(funding) else 0.0
    net = gross - 2 * COST_PER_SIDE - side * fund  # longs pay positive funding
    return {"missed": False, "entry_time": t_in, "exit_time": t_out, "entry": entry, "exit": exit_,
            "gross": gross, "funding": -side * fund, "net": net, "pnl_usd": net * NOTIONAL_USD, "bars": iv}


def backtest(addresses, days: int = 90, coins=COINS, session=None, now: datetime | None = None, pause: float = 0.5,
             progress=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(copied trades, every trade of theirs with whether it is in our coins)."""
    s = session or requests.Session()
    end = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    start = end - days * 86_400_000
    market: dict[str, tuple[dict, pd.Series]] = {}
    rows, theirs = [], []
    for n, addr in enumerate(addresses, 1):
        try:
            fills = fetch_fills(addr, start, end, s, pause)
        except Exception as e:
            progress(f"[{n}/{len(addresses)}] {addr[:10]} fills failed: {type(e).__name__}: {e}")
            continue
        closed, _ = round_trips(fills)
        progress(f"[{n}/{len(addresses)}] {addr[:10]} {len(fills)} fills, {len(closed)} closed trades")
        for tr in closed:
            ours = tr["coin"] in coins
            theirs.append({"address": addr, "coin": tr["coin"], "ours": ours, "pnl_usd": tr["pnl"],
                           "return": tr["return"], "hours": tr["hours"]})
            if not ours:
                continue
            if tr["coin"] not in market:
                market[tr["coin"]] = _market(tr["coin"], start, end, s, pause)
            bars, funding = market[tr["coin"]]
            for mode in MODES:
                r = copy_trade(tr, mode, bars, funding)
                if r is not None:
                    rows.append({"address": addr, "coin": tr["coin"], "side": tr["side"], "mode": mode,
                                 "their_open": tr["open"], "their_close": tr["close"], "their_hours": tr["hours"],
                                 "their_return": tr["return"], **r})
    return pd.DataFrame(rows), pd.DataFrame(theirs)


def _market(coin: str, start: int, end: int, s, pause: float) -> tuple[dict, pd.Series]:
    bars = {}
    for iv in ("5m", "15m", "1h"):
        try:
            bars[iv] = parse_candles(_post(MAINNET, {"type": "candleSnapshot", "req": {
                "coin": coin, "interval": iv, "startTime": start, "endTime": end + 2 * H}}, s))
        except Exception:
            bars[iv] = parse_candles([])
        time.sleep(pause)
    parts, t = [], start
    while t < end:
        got = _post(MAINNET, {"type": "fundingHistory", "coin": coin, "startTime": t, "endTime": end}, s)
        if not got:
            break
        parts.append(parse_funding(got))
        t = int(got[-1]["time"]) + 1
        time.sleep(pause)
        if len(got) < 500:
            break
    funding = pd.concat(parts) if parts else parse_funding([])
    return bars, funding[~funding.index.duplicated()].sort_index()


def summarise(copied: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    copied = copied.assign(mode=pd.Categorical(copied["mode"], MODES, ordered=True))
    done = copied[~copied["missed"]]
    g = done.groupby(by, observed=True)
    out = pd.DataFrame({"trades": g.size(), "win_rate": g["net"].apply(lambda x: (x > 0).mean()),
                        "avg_net": g["net"].mean(), "avg_gross": g["gross"].mean(), "pnl_usd": g["pnl_usd"].sum()})
    missed = copied[copied["missed"]].groupby(by, observed=True).size()
    out["missed"] = missed.reindex(out.index).fillna(0).astype(int)
    return out


def max_open(copied: pd.DataFrame, mode: str) -> int:
    done = copied[(copied["mode"] == mode) & ~copied["missed"]]
    if not len(done):
        return 0
    ev = pd.concat([pd.Series(1, index=done["entry_time"]), pd.Series(-1, index=done["exit_time"])]).sort_index(kind="stable")
    return int(ev.groupby(level=0).sum().cumsum().max())


def report(copied: pd.DataFrame, theirs: pd.DataFrame, days: int) -> str:
    pct = lambda x: f"{x:+.2%}"
    lines = [f"Copying {theirs['address'].nunique() if len(theirs) else 0} traders, last {days} days, "
             f"{NOTIONAL_USD:.0f} USD per trade, Hyperliquid taker fee + slippage + funding", ""]
    if len(theirs):
        lines.append("Their own trades: in our coins vs other coins (their USD profit)")
        for addr, t in theirs.groupby("address", sort=False):
            ours, other = t[t["ours"]], t[~t["ours"]]
            lines.append(f"  {addr[:10]}  our coins {len(ours)} trades {ours['pnl_usd'].sum():+,.0f}  |  "
                         f"other coins {len(other)} trades {other['pnl_usd'].sum():+,.0f}")
        lines.append("")
    if not len(copied):
        return "\n".join(lines + ["No trades in our coins to copy."])
    for title, by in (("By trader", ["mode", "address"]), ("By coin", ["mode", "coin"]), ("All together", ["mode"])):
        lines.append(title)
        for key, r in summarise(copied, by).iterrows():
            key = key if isinstance(key, tuple) else (key,)
            name = "  ".join(k[:10] if k.startswith("0x") else k for k in key)
            lines.append(f"  {name:<28} {int(r['trades']):>3} trades (+{int(r['missed'])} missed)  win {r['win_rate']:.0%}  "
                         f"avg {pct(r['avg_net'])} net ({pct(r['avg_gross'])} before costs)  total {r['pnl_usd']:+.2f} USD")
        lines.append("")
    fallback = copied[(copied["mode"] == "5m check") & ~copied["missed"] & (copied["bars"] != "5m")]
    lines.append(f"Most positions open at once (1h check): {max_open(copied, '1h check')} "
                 f"(each needs 20 USD margin; a 200 USD account holds 10)")
    if len(fallback):
        lines.append(f"Note: {len(fallback)} '5m check' trades used 15m/1h candles (5m only goes back about 17 days)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Backtest copying Hyperliquid traders from their real fills")
    p.add_argument("--address", nargs="*", default=list(FIVE))
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--out", default="data/copy-backtest")
    a = p.parse_args(argv)
    copied, theirs = backtest(a.address, a.days)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    copied.to_csv(out / "trades.csv", index=False)
    print()
    print(report(copied, theirs, a.days))
    print(f"\nsaved {out / 'trades.csv'}")


if __name__ == "__main__":
    main()
