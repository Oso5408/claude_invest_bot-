"""ExecutionOptimizer: fill 4h breakout entries with a 15-minute pullback limit order.

The 4h strategy (btcbot/breakout.py) decides at a 4h close that it wants in. Normally it
buys at the next 4h open with a market order (taker fee + slippage). Here it instead
places a limit order `pullback_atr` x ATR(4h) better than the signal close and watches
the 15-minute bars of the next 4h bar:

- filled: the first 15m bar that trades through the limit fills it at the limit (or at
  that bar's open if it gapped through), maker fee, no slippage. The trailing stop then
  only sees the price range after the fill.
- not filled within `window` 15m bars: `on_miss="skip"` drops the trade; "chase" buys at
  market at the close of the window (taker fee + slippage).

Only entries change. Exits stay stop orders, filled as in the 4h engine.

    python -m btcbot.execution --csv data/ADA_JPY_15min.csv --hl-funding data/HL_ADA_funding.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from btcbot import breakout
from btcbot.data import load_csv, resample


class ExecutionOptimizer:
    def __init__(self, bars15: pd.DataFrame, index4h: pd.DatetimeIndex, pullback_atr: float = 0.5,
                 window: int = 16, on_miss: str = "skip", maker_fee: float = 0.00015,
                 taker_fee: float = 0.00045, slippage: float = 0.0005):
        if on_miss not in ("skip", "chase"):
            raise ValueError(f"unknown on_miss {on_miss}")
        self.o, self.h, self.l, self.c = (bars15[k].to_numpy() for k in ("open", "high", "low", "close"))
        times = bars15.index.as_unit("ns").asi8
        starts = index4h.as_unit("ns").asi8
        self.first = np.searchsorted(times, starts, side="left")
        self.last = np.searchsorted(times, starts + pd.Timedelta("4h").value, side="left")  # exclusive
        self.pullback_atr, self.window, self.on_miss = pullback_atr, window, on_miss
        self.maker_fee, self.taker_fee, self.slippage = maker_fee, taker_fee, slippage
        self.filled = self.chased = self.skipped = 0
        self.saved = []  # entry price improvement vs a market order at the 4h open, per filled entry

    def __call__(self, t: int, direction: int, signal_close: float, signal_atr: float):
        a, b = self.first[t], self.last[t]
        if a >= b:
            return None
        market = self.o[a] * (1 + self.slippage * direction)
        limit = signal_close - direction * self.pullback_atr * signal_atr
        end = min(b, a + self.window)
        for j in range(a, end):
            if (direction > 0 and self.l[j] <= limit) or (direction < 0 and self.h[j] >= limit):
                px = min(self.o[j], limit) if direction > 0 else max(self.o[j], limit)
                self.filled += 1
                self.saved.append(direction * (market - px) / market)
                return px, self.maker_fee, self.l[j:b].min(), self.h[j:b].max()
        if self.on_miss == "skip":
            self.skipped += 1
            return None
        k = end - 1
        px = self.c[k] * (1 + self.slippage * direction)
        self.chased += 1
        self.saved.append(direction * (market - px) / market)
        rest_lo = self.l[k + 1:b].min() if k + 1 < b else px
        rest_hi = self.h[k + 1:b].max() if k + 1 < b else px
        return px, self.taker_fee, min(px, rest_lo), max(px, rest_hi)


def compare(bars15: pd.DataFrame, cfg: breakout.BreakoutConfig, acct_cfg, funding=None, fng=None,
            variants: list[dict] | None = None) -> pd.DataFrame:
    """Baseline (market at the 4h open) against each pullback setting, same signals and costs."""
    df4 = resample(bars15, "4h")
    rows = []
    base = breakout.run(df4, cfg, acct_cfg, fng, funding)
    rows.append(_row("market at 4h open", base, None))
    for v in variants or [dict(pullback_atr=k, on_miss=m) for m in ("skip", "chase") for k in (0.0, 0.25, 0.5, 1.0)]:
        ex = ExecutionOptimizer(bars15, df4.index, maker_fee=acct_cfg.fee_rate / 3, taker_fee=acct_cfg.fee_rate,
                                slippage=acct_cfg.slippage, **v)
        res = breakout.run(df4, cfg, acct_cfg, fng, funding, entry_fill=ex)
        rows.append(_row(f"limit {v['pullback_atr']} ATR, {v.get('on_miss', 'skip')}", res, ex))
    return pd.DataFrame(rows)


def _row(name, res, ex) -> dict:
    s = res.stats
    row = {"entry": name, "return": s["total_return"], "max_dd": s["max_drawdown"], "sharpe": s["sharpe"],
           "trades": s["signal_trades"], "win_rate": s["win_rate"], "missed": s.get("missed_entries", 0)}
    if ex is not None:
        row.update(limit_filled=ex.filled, chased=ex.chased,
                   avg_saving=float(np.mean(ex.saved)) if ex.saved else float("nan"))
    return row


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="4h breakout: market entry vs 15m pullback limit entry")
    p.add_argument("--csv", required=True, help="15-minute bars")
    p.add_argument("--hl-funding", help="Hyperliquid funding CSV: Hyperliquid fees and funding")
    p.add_argument("--fng-csv")
    p.add_argument("--fng-short-max", type=float, default=100)
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--max-leverage", type=float, default=2.0)
    a = p.parse_args(argv)
    bars15 = load_csv(a.csv)
    cfg = breakout.BreakoutConfig(allow_short=not a.no_short, max_leverage=a.max_leverage,
                                  fng_short_max=a.fng_short_max)
    funding = None
    if a.hl_funding:
        from btcbot.hyperliquid import account_config, load_funding
        acct, funding = account_config(30_000), load_funding(a.hl_funding)
    else:
        from btcbot.meanrev import AccountConfig
        acct = AccountConfig()
    fng = None
    if a.fng_csv:
        from btcbot.data import load_fng
        fng = load_fng(a.fng_csv)
    pd.set_option("display.width", 200)
    print(compare(bars15, cfg, acct, funding, fng).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
