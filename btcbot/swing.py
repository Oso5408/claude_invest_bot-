"""1-2 day swing research engine on hourly bars (Hyperliquid costs).

A signal is decided at the close of a bar and filled at the open of the next hourly
bar (index of `signals` = that bar's open time). The trade is held for `hold_h` hours
with a fixed stop `stop_pct` away from the entry; whichever comes first ends it.
The stop is checked on each hourly bar's high/low; a gap through it fills at the open.
One trade at a time per run: signals during an open trade are ignored.

Results are in R (1R = the stop distance):
- gross_r: zero cost (no fee, no slippage, no funding), stop filled exactly at the stop.
- net_r: taker fee both sides, slippage both sides, hourly funding (longs pay positive).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TAKER = 0.00045


@dataclass
class Costs:
    fee: float = TAKER  # per side
    slippage: float = 0.0005  # per side, as a fraction of price


def daily(h1: pd.DataFrame) -> pd.DataFrame:
    """UTC daily bars from hourly bars, labelled by the day's open time."""
    return h1.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                  "volume": "sum"}).dropna()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def simulate(h1: pd.DataFrame, signals: pd.Series, stop_pct: pd.Series, hold_h: int,
             funding: pd.Series | None = None, costs: Costs | None = None) -> pd.DataFrame:
    costs = costs or Costs()
    idx = h1.index
    o, h, l = (h1[k].to_numpy() for k in ("open", "high", "low"))
    fund = np.zeros(len(h1)) if funding is None else funding.reindex(idx).fillna(0.0).to_numpy()
    sig = signals[signals != 0].dropna()
    pos = idx.get_indexer(sig.index)
    stops = stop_pct.reindex(sig.index).to_numpy()
    rows, busy_until = [], -1
    for t, side, sp, when in zip(pos, sig.to_numpy(), stops, sig.index):
        if t < 0 or t <= busy_until or not np.isfinite(sp) or sp <= 0 or t + hold_h >= len(h1):
            continue
        side = int(np.sign(side))
        e = o[t]
        stop = e * (1 - side * sp)
        exit_raw, reason, k = o[t + hold_h], "time", t + hold_h  # k = bar whose open ends the trade
        for j in range(t, t + hold_h):
            if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
                op = o[j]
                exit_raw = min(op, stop) if side > 0 else max(op, stop)
                reason, k = "stop", j + 1
                break
        gross = side * (exit_raw / e - 1)
        ep = e * (1 + side * costs.slippage)
        xp = exit_raw * (1 - side * costs.slippage)
        fsum = fund[t:k].sum()
        net = side * (xp / ep - 1) - 2 * costs.fee - side * fsum
        rows.append({"entry_time": when, "exit_time": idx[min(k, len(idx) - 1)], "side": side, "stop_pct": sp,
                     "gross_ret": gross, "net_ret": net, "funding": side * fsum, "reason": reason,
                     "gross_r": gross / sp, "net_r": net / sp})
        busy_until = k - 1
    return pd.DataFrame(rows)


def summarize(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"n": 0}
    r = tr["net_r"]
    return {"n": len(tr), "win": (r > 0).mean(), "net_r": r.mean(), "gross_r": tr["gross_r"].mean(),
            "cost_r": (tr["gross_r"] - r).mean(), "t": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan,
            "stop_med": tr["stop_pct"].median(), "stop_hits": (tr["reason"] == "stop").mean(),
            "sum_r": r.sum()}
