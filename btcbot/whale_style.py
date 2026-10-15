"""How do the watched Hyperliquid traders trade? Rebuilds every trade from their fills and describes the method.

For each address (default: the AI account's watch list, data/hl-traders/latest.json) it pulls the fills of the
last --days (userFillsByTime, up to 2000) and 1h candles of every coin they traded, then rebuilds round trips
(flat -> position -> flat, or a flip) per coin and measures:
- holding time: scalper (< 1h), day trader (< 24h), swing (1-7 days), position (> 7 days)
- where they enter: for a long, where the price sat in the previous 24h range (top 20% = buying strength /
  breakout, bottom 20% = buying dips), mirrored for shorts; and whether they trade with the 7-day trend
- adding to a trade: at a better price than their average (averaging down) or a worse one (adding to winners)
- limit vs market orders (maker share), long vs short, win rate, average win / loss, profit factor, largest loss

Trades already open when the window starts are skipped (their start isn't visible); open ones are reported apart.
Read-only public data, no key, no orders. Run on the VM (the cloud container can't reach Hyperliquid):

    .venv/bin/python -m btcbot.whale_style                         # the 13-20 watched traders
    .venv/bin/python -m btcbot.whale_style --address 0xabc... 0xdef...
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from btcbot.hyperliquid import MAINNET, _post, parse_candles

LATEST = Path("data/hl-traders/latest.json")


def round_trips(fills: list[dict]) -> tuple[list[dict], list[dict]]:
    """(closed round trips, still-open trades) from one account's fills, oldest first."""
    fills = sorted(fills, key=lambda f: int(f["time"]))
    closed, open_, skipping, seen = [], {}, set(), set()
    for f in fills:
        coin, px, sz = f["coin"], float(f["px"]), float(f["sz"])
        delta = sz if f["side"] == "B" else -sz
        before = float(f.get("startPosition", 0) or 0)
        after = before + delta
        fee = float(f.get("fee", 0) or 0)
        pnl = float(f.get("closedPnl", 0) or 0) - fee
        t = pd.Timestamp(int(f["time"]), unit="ms", tz="UTC")
        maker = not f.get("crossed", True)
        flat_or_flip = abs(after) < 1e-12 or before * after < 0
        if coin not in seen:
            seen.add(coin)
            if abs(before) > 1e-12:
                skipping.add(coin)  # opened before the window: its start isn't visible
        if coin in skipping:
            if flat_or_flip:
                skipping.discard(coin)
                if abs(after) > 1e-12:  # flipped: a new trade starts here
                    open_[coin] = _new_trade(coin, t, px, abs(after), 1 if after > 0 else -1, maker, 0.0)
            continue
        tr = open_.get(coin)
        if tr is None:
            if abs(after) > 1e-12:
                open_[coin] = _new_trade(coin, t, px, abs(after), 1 if after > 0 else -1, maker, -fee)
            continue
        tr["fills"] += 1
        tr["maker_fills"] += maker
        if flat_or_flip:
            tr["pnl"] += pnl
            tr["exit_notional"] += abs(before) * px
            tr["exit_qty"] += abs(before)
            closed.append(_finish(tr, t))
            del open_[coin]
            if abs(after) > 1e-12:
                open_[coin] = _new_trade(coin, t, px, abs(after), 1 if after > 0 else -1, maker, 0.0)
        elif abs(after) > abs(before):  # added
            avg = tr["entry_notional"] / tr["entry_qty"]
            tr["adds"] += 1
            tr["adds_better"] += (px < avg) if tr["side"] > 0 else (px > avg)
            tr["entry_notional"] += abs(delta) * px
            tr["entry_qty"] += abs(delta)
            tr["max_qty"] = max(tr["max_qty"], abs(after))
            tr["pnl"] -= fee
        else:  # partly closed
            tr["pnl"] += pnl
            tr["exit_notional"] += abs(delta) * px
            tr["exit_qty"] += abs(delta)
            tr["partial_exits"] += 1
    return closed, list(open_.values())


def _new_trade(coin, t, px, qty, side, maker, pnl):
    return {"coin": coin, "side": side, "open": t, "entry_px": px, "entry_notional": px * qty, "entry_qty": qty,
            "max_qty": qty, "exit_notional": 0.0, "exit_qty": 0.0, "pnl": pnl, "fills": 1, "maker_fills": int(maker),
            "adds": 0, "adds_better": 0, "partial_exits": 0}


def _finish(tr, t):
    tr = dict(tr)
    tr["close"] = t
    tr["hours"] = (t - tr["open"]).total_seconds() / 3600
    tr["avg_entry"] = tr["entry_notional"] / tr["entry_qty"]
    tr["avg_exit"] = tr["exit_notional"] / tr["exit_qty"] if tr["exit_qty"] else np.nan
    tr["max_notional"] = tr["max_qty"] * tr["avg_entry"]
    tr["return"] = tr["side"] * (tr["avg_exit"] / tr["avg_entry"] - 1)
    return tr


def entry_context(trades: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Where each entry sat in the previous 24h range (1 = at the high for a long / the low for a short)
    and whether it was with the 7-day trend."""
    loc, with_trend = [], []
    for r in trades.itertuples(index=False):
        c = candles.get(r.coin)
        if c is None or c.empty:
            loc.append(np.nan), with_trend.append(np.nan)
            continue
        prev = c[c.index + pd.Timedelta("1h") <= r.open]
        day, week = prev.tail(24), prev.tail(24 * 7)
        if len(day) < 12 or len(week) < 24 * 3:
            loc.append(np.nan), with_trend.append(np.nan)
            continue
        hi, lo = day["high"].max(), day["low"].min()
        x = (r.entry_px - lo) / (hi - lo) if hi > lo else 0.5
        loc.append(float(np.clip(x if r.side > 0 else 1 - x, 0, 1)))
        trend = week["close"].iloc[-1] / week["close"].iloc[0] - 1
        with_trend.append(float(np.sign(trend) == r.side))
    return trades.assign(entry_loc=loc, with_trend=with_trend)


def profile(trades: pd.DataFrame, still_open: list[dict]) -> dict:
    if trades.empty:
        return {"trades": 0, "open_now": len(still_open)}
    t = trades
    wins, losses = t[t["pnl"] > 0], t[t["pnl"] <= 0]
    hours = t["hours"].median()
    style = ("scalper" if hours < 1 else "day trader" if hours < 24 else "swing" if hours < 24 * 7 else "position")
    loc = t["entry_loc"].dropna()
    strong, weak = (loc >= 0.8).mean() if len(loc) else np.nan, (loc <= 0.2).mean() if len(loc) else np.nan
    entry = ("buys strength / breakouts" if strong >= 0.4 and strong > 1.5 * weak else
             "buys dips / fades moves" if weak >= 0.4 and weak > 1.5 * strong else "mixed entries")
    adds = int(t["adds"].sum())
    better = int(t["adds_better"].sum())
    adding = ("rarely adds" if adds < len(t) * 0.3 else
              "averages down (adds at better prices)" if better > 0.6 * adds else
              "adds to winners (adds at worse prices)" if better < 0.4 * adds else "adds both ways")
    long_share = float((t["side"] > 0).mean())
    return {
        "trades": len(t), "open_now": len(still_open), "style": style, "median_hours": round(float(hours), 1),
        "entry": entry, "entry_near_strength": _r(strong), "entry_near_weakness": _r(weak),
        "with_7d_trend": _r(t["with_trend"].dropna().mean()), "adding": adding,
        "partial_exits_per_trade": round(float(t["partial_exits"].mean()), 1),
        "maker_share": _r(t["maker_fills"].sum() / t["fills"].sum()), "long_share": _r(long_share),
        "win_rate": _r(len(wins) / len(t)), "avg_win": round(float(wins["pnl"].mean()), 0) if len(wins) else 0,
        "avg_loss": round(float(losses["pnl"].mean()), 0) if len(losses) else 0,
        "profit_factor": round(float(wins["pnl"].sum() / -losses["pnl"].sum()), 2) if losses["pnl"].sum() < 0 else None,
        "largest_loss": round(float(t["pnl"].min()), 0), "pnl": round(float(t["pnl"].sum()), 0),
        "median_size_usd": round(float(t["max_notional"].median()), 0),
        "coins": t.groupby("coin")["pnl"].agg(["count", "sum"]).sort_values("count", ascending=False)
                  .head(5).round(0).astype(int).apply(lambda r: f"{r['count']}x {r['sum']:+,}", axis=1).to_dict(),
    }


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def analyse(addresses: list[str], days: int = 30, session=None, now: datetime | None = None, pause: float = 1.0,
            progress=print) -> dict[str, dict]:
    s = session or requests.Session()
    end = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    start = end - days * 86_400_000
    candles: dict[str, pd.DataFrame] = {}
    out = {}
    for n, addr in enumerate(addresses, 1):
        try:
            fills = _post(MAINNET, {"type": "userFillsByTime", "user": addr, "startTime": start, "endTime": end,
                                    "aggregateByTime": True}, s)
            closed, still_open = round_trips(fills)
            trades = pd.DataFrame(closed)
            for coin in (set(trades["coin"]) if len(trades) else set()) - set(candles):
                try:
                    candles[coin] = parse_candles(_post(MAINNET, {"type": "candleSnapshot", "req": {
                        "coin": coin, "interval": "1h", "startTime": start - 8 * 86_400_000, "endTime": end}}, s))
                except Exception:
                    candles[coin] = pd.DataFrame()  # spot pairs / delisted: no entry context
                time.sleep(pause / 2)
            if len(trades):
                trades = entry_context(trades, candles)
            out[addr] = profile(trades, still_open) | {"fills": len(fills), "fills_capped": len(fills) >= 2000}
        except Exception as e:
            out[addr] = {"error": f"{type(e).__name__}: {e}"}
        progress(f"[{n}/{len(addresses)}] {addr[:10]} done")
        time.sleep(pause)
    return out


def summary_text(profiles: dict[str, dict]) -> str:
    lines = []
    for addr, p in profiles.items():
        if "error" in p or not p.get("trades"):
            lines.append(f"{addr}  {p.get('error') or 'no complete trades in the window'}")
            continue
        pct = lambda x: "-" if x is None else f"{x:.0%}"
        lines.append(f"{addr}  {p['style']}, holds {p['median_hours']}h (median), {p['entry']}, {p['adding']}")
        lines.append(f"    {p['trades']} trades{' (fills capped at 2000)' if p['fills_capped'] else ''}, "
                     f"win {pct(p['win_rate'])}, avg win {p['avg_win']:+,.0f} / avg loss {p['avg_loss']:+,.0f}, "
                     f"profit factor {p['profit_factor']}, largest loss {p['largest_loss']:+,.0f}, total {p['pnl']:+,.0f}")
        lines.append(f"    entries near 24h strength {pct(p['entry_near_strength'])} / weakness {pct(p['entry_near_weakness'])}, "
                     f"with 7d trend {pct(p['with_7d_trend'])}, long {pct(p['long_share'])}, limit orders {pct(p['maker_share'])}, "
                     f"partial exits {p['partial_exits_per_trade']}/trade, median size {p['median_size_usd']:,.0f} USD")
        lines.append(f"    coins: {', '.join(f'{c} {v}' for c, v in p['coins'].items())}")
    ok = [p for p in profiles.values() if p.get("trades")]
    if ok:
        count = lambda key: pd.Series([p[key] for p in ok]).value_counts().to_dict()
        lines += ["", f"Overall ({len(ok)} traders): style {count('style')}", f"    entries {count('entry')}",
                  f"    adding {count('adding')}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Describe how Hyperliquid traders trade, from their fills")
    p.add_argument("--address", nargs="*", help="default: data/hl-traders/latest.json")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--out", default="data/hl-traders/styles.json")
    a = p.parse_args(argv)
    addresses = a.address or [r["address"] for r in json.loads(LATEST.read_text())]
    profiles = analyse(addresses, a.days)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(profiles, indent=1, default=str))
    print()
    print(summary_text(profiles))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
