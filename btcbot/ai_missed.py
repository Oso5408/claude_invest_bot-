"""Look back over the AI paper account (data/ai-hl): which tested signals came up in the real market, and
did the AI take them? Same standard as btcbot.ai_selftest:
- 4h breakout: a completed 4h candle closed above the 20-bar high on 1.2x volume (long), or below the 10-bar
  low with Fear & Greed <= 50 (short). Taken = the AI opened that coin on that side within 4 hours.
- copy traders: copy_alerts.csv "opened" or "flipped" alerts. Taken = same coin and side within 1 hour.
For each signal it prints what the AI said about that coin at its next decision, and what the tested exit
would have made (2x ATR trailing stop on 4h candles, still open = marked at the last close; copies: the
move 24 hours later), so a skip that cost money stands out.

Reads Hyperliquid public candles and the Fear & Greed Index; writes nothing.

    .venv/bin/python -m btcbot.ai_missed --days 7
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from btcbot import ai_trader as ai

WARMUP_BARS = 260


def signals(df: pd.DataFrame, start: pd.Timestamp, fear_greed: dict[str, float]) -> list[dict]:
    """Every completed 4h candle from `start` on that gives the tested breakout signal."""
    out = []
    for i in range(len(df)):
        t = df.index[i]
        if t + pd.Timedelta(hours=4) < start:
            continue
        ind = ai.indicators(df.iloc[:i + 1])
        side = ai.breakout_signal(ind)
        if side == "none":
            continue
        close_at = t + pd.Timedelta(hours=4)
        fg = fear_greed.get(close_at.strftime("%Y-%m-%d"))
        if side == "short" and fg is not None and fg > 50:
            continue  # the tested short filter
        out.append({"time": close_at, "side": side, "price": ind["close"], "atr": ind["atr14"],
                    "fear_greed": fg, "bar": i})
    return out


def trailing_result(df: pd.DataFrame, bar: int, side: str, atr_mult: float = 2.0) -> tuple[float, str]:
    """Return of the tested exit from the close of `bar`: 2x ATR trailing stop, moved at each 4h close."""
    d = 1 if side == "long" else -1
    entry = float(df["close"].iloc[bar])
    atr = ai.indicators(df.iloc[:bar + 1])["atr14"]
    stop, best = entry - d * atr_mult * atr, entry
    for j in range(bar + 1, len(df)):
        hi, lo, cl = (float(df[k].iloc[j]) for k in ("high", "low", "close"))
        if (d > 0 and lo <= stop) or (d < 0 and hi >= stop):
            return d * (stop / entry - 1), "stopped"
        best = max(best, hi) if d > 0 else min(best, lo)
        atr = ai.indicators(df.iloc[:j + 1])["atr14"]
        stop = max(stop, best - atr_mult * atr) if d > 0 else min(stop, best + atr_mult * atr)
    return d * (float(df["close"].iloc[-1]) / entry - 1), "open"


def entries(folder: Path) -> pd.DataFrame:
    """Every trade the AI opened: closed ones from trades.csv plus the open ones in state.json."""
    rows = []
    if (folder / "trades.csv").exists():
        t = pd.read_csv(folder / "trades.csv")
        rows += [{"coin": r.coin, "side": r.side, "opened": r.opened, "strategy": r.strategy} for r in t.itertuples()]
    if (folder / "state.json").exists():
        for p in json.loads((folder / "state.json").read_text()).get("positions", []):
            rows.append({"coin": p["coin"], "side": "long" if p["side"] > 0 else "short", "opened": p["opened"],
                         "strategy": p.get("strategy", "")})
    df = pd.DataFrame(rows, columns=["coin", "side", "opened", "strategy"])
    df["opened"] = pd.to_datetime(df["opened"], utc=True, format="mixed")
    return df


def decisions(folder: Path) -> pd.DataFrame:
    if not (folder / "decisions.csv").exists():
        return pd.DataFrame(columns=["time", "trigger", "summary", "coin_views"])
    d = pd.read_csv(folder / "decisions.csv")
    d["time"] = pd.to_datetime(d["time"], utc=True, format="mixed")
    return d.sort_values("time")


def coin_view(dec: pd.DataFrame, coin: str, after: pd.Timestamp, within: pd.Timedelta) -> str:
    """What the AI said about `coin` at its first decision in [after, after + within)."""
    d = dec[(dec["time"] >= after) & (dec["time"] < after + within)]
    if not len(d):
        return "no decision in that window"
    r = d.iloc[0]
    try:
        views = json.loads(r["coin_views"]) if isinstance(r["coin_views"], str) else []
    except ValueError:
        views = []
    v = next((x for x in views if isinstance(x, dict) and x.get("coin") == coin), None)
    when = r["time"].astimezone(ai.JST).strftime("%m-%d %H:%M")
    if v:
        return f"{when} JST {v.get('view', '')} / {v.get('strategy', '')}: {v.get('reason', '')}"
    return f"{when} JST (no view on {coin}) {str(r['summary'])[:200]}"


def taken(ent: pd.DataFrame, coin: str, side: str, after: pd.Timestamp, within: pd.Timedelta) -> str:
    e = ent[(ent["coin"] == coin) & (ent["side"] == side) & (ent["opened"] >= after)
            & (ent["opened"] < after + within)]
    return str(e["strategy"].iloc[0]) if len(e) else ""


def entries_that_day(ent: pd.DataFrame, t: pd.Timestamp) -> int:
    day = t.astimezone(ai.JST).date()
    return int(sum(o.astimezone(ai.JST).date() == day and o < t for o in ent["opened"]))


def review(folder: Path, market, now: pd.Timestamp, days: float) -> dict:
    start = now - pd.Timedelta(days=days)
    ent, dec = entries(folder), decisions(folder)
    if len(dec):
        start = max(start, dec["time"].iloc[0] - pd.Timedelta(hours=4))  # the first decision could act on the candle before it
    try:
        fg = {k: float(v) for k, v in market.fear_greed().items()}
    except Exception:
        fg = {}
    rows = []
    for coin in ai.COINS:
        df = market.candles(coin, "4h", start - WARMUP_BARS * pd.Timedelta(hours=4), now)
        df = ai.completed(df, "4h", now)
        for s in signals(df, start, fg):
            ret, state = trailing_result(df, s["bar"], s["side"])
            got = taken(ent, coin, s["side"], s["time"], pd.Timedelta(hours=4))
            rows.append({"kind": "4h breakout", "time": s["time"], "coin": coin, "side": s["side"],
                         "price": round(s["price"], 6), "fear_greed": s["fear_greed"],
                         "taken": got or "NO", "trades_before_that_day": entries_that_day(ent, s["time"]),
                         "tested_exit_return": round(ret, 4), "tested_exit": state,
                         "ai_said": "" if got else coin_view(dec, coin, s["time"], pd.Timedelta(hours=4))})
    if (folder / "copy_alerts.csv").exists():
        ca = pd.read_csv(folder / "copy_alerts.csv")
        ca["time"] = pd.to_datetime(ca["time"], utc=True, format="mixed")
        ca = ca[(ca["time"] >= start) & ca["change"].str.startswith(("opened", "flipped"))]
        for r in ca.itertuples():
            side = "long" if r.change.endswith("long") else "short"
            got = taken(ent, r.coin, side, r.time, pd.Timedelta(hours=1))
            m = market.candles(r.coin, "1h", r.time - pd.Timedelta(hours=2), min(now, r.time + pd.Timedelta(hours=26)))
            after = m[m.index >= r.time]
            move = None
            if len(after) > 1:
                d = 1 if side == "long" else -1
                move = round(d * (float(after["close"].iloc[min(24, len(after) - 1)]) / float(after["open"].iloc[0]) - 1), 4)
            rows.append({"kind": f"copy {r.account}", "time": r.time, "coin": r.coin, "side": side,
                         "price": None, "fear_greed": None, "taken": got or "NO",
                         "trades_before_that_day": entries_that_day(ent, r.time),
                         "tested_exit_return": move, "tested_exit": "24h move",
                         "ai_said": "" if got else coin_view(dec, r.coin, r.time, pd.Timedelta(hours=1))})
    rows.sort(key=lambda x: x["time"])
    missed = [r for r in rows if r["taken"] == "NO"]
    blocked = [r for r in missed if r["trades_before_that_day"] >= ai.MAX_ENTRIES_PER_DAY]
    return {"from": str(start), "to": str(now), "signals": len(rows), "taken": len(rows) - len(missed),
            "missed": len(missed), "missed_daily_limit_full": len(blocked),
            "missed_tested_return_sum": round(sum(r["tested_exit_return"] or 0 for r in missed), 4),
            "rows": rows}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Signals the AI paper account took or missed")
    p.add_argument("--days", type=float, default=7)
    p.add_argument("--folder", default="data/ai-hl")
    a = p.parse_args(argv)
    now = pd.Timestamp(datetime.now(timezone.utc))
    out = review(Path(a.folder), ai.HLMarket(), now, a.days)
    for r in out["rows"]:
        t = r["time"].astimezone(ai.JST).strftime("%m-%d %H:%M")
        ret = "" if r["tested_exit_return"] is None else f"{r['tested_exit_return']:+.1%} ({r['tested_exit']})"
        print(f"{t} JST  {r['kind']:<18} {r['coin']:<5} {r['side']:<5}  taken: {r['taken']:<20} "
              f"trades that day before: {r['trades_before_that_day']}  tested result: {ret}")
        if r["ai_said"]:
            print(f"    AI said: {r['ai_said'][:400]}")
    print(f"\nFrom {out['from']} to {out['to']}: {out['signals']} signals, {out['taken']} taken, "
          f"{out['missed']} missed ({out['missed_daily_limit_full']} when that day's 2 trades were already used). "
          f"Missed ones would have made {out['missed_tested_return_sum']:+.1%} in total (sum of returns, before leverage).")


if __name__ == "__main__":
    main()
