"""Dynamic grid on GMO Coin crypto FX (default ADA_JPY), up to 2x leverage.

The range comes from indicators instead of a fixed percentage:
- centre: EMA(50) of 15-minute closes
- ceiling / floor: centre +/- 2.5 x ATR(14)
- the range is cut into `cells` equal steps (cells + 1 lines)
Indicators are taken from the previous closed bar, so nothing peeks ahead.

Orders: buy limits on every line below the price, sell limits on every line above.
A buy filled on one line puts a sell on the line above, and the other way round, so
each up-down round trip across one step earns one step on one unit. On margin
the sells above the price open shorts, so the grid is neutral.

Unit size: at the edge of the range the position is `cells / 2` units worth
`edge_leverage` x equity (2x by default); risk.py still caps every fill.

What to do when the price runs out of the range (`mode`):
- "rebuild": the grid stays where it was built. When the price goes one step past
  the ceiling or floor, close everything at market and build a new grid from the
  current EMA and ATR, starting flat.
- "follow": the grid is redrawn from the latest EMA and ATR at every bar open and
  the position is kept, never stopped out. The new orders sit around the current
  price (buys below, sells above), so a unit bought before the grid moved down may
  be sold below what it cost. At most `cells / 2` units either way; past the range
  edge no orders are left and the position waits for the price to come back.

Fills: grid limit orders fill at the line price with the maker fee (0 on ADA_JPY)
and no slippage; market fills pay the taker fee plus slippage. Inside each bar the
price is assumed to go open -> low -> high -> close on a rising bar and
open -> high -> low -> close on a falling bar. Real 15-minute bars wiggle more
than that, so the grid's fill count is if anything understated.

    python -m btcbot.dyngrid --csv data/ADA_JPY_15min.csv --cells 10 15 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcbot import risk
from btcbot.data import load_csv
from btcbot.meanrev import AccountConfig, MarginAccount, Result, _stats, rollover_counts


@dataclass
class DynGridConfig:
    cells: int = 10
    ema: int = 50
    atr: int = 14
    width_atr: float = 2.5  # ceiling and floor are this many ATRs from the EMA
    edge_leverage: float = 2.0
    maker_fee: float = 0.0  # ADA_JPY makerFee from the GMO symbols API
    mode: str = "rebuild"  # or "follow"


def indicators(df: pd.DataFrame, cfg: DynGridConfig) -> tuple[np.ndarray, np.ndarray]:
    """EMA and ATR as known at each bar's open (from the previous closed bar)."""
    c = df["close"]
    ema = c.ewm(span=cfg.ema, adjust=False, min_periods=cfg.ema).mean()
    prev = c.shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / cfg.atr, adjust=False, min_periods=cfg.atr).mean()
    return ema.shift(1).to_numpy(), atr.shift(1).to_numpy()


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """GMO leaves out 15-minute bars with no trades. Put them back as flat bars at the last
    close, so the EMA and ATR count time, not trades."""
    step = df.index.to_series().diff().min()
    full = df.reindex(pd.date_range(df.index[0], df.index[-1], freq=step, name=df.index.name))
    close = full["close"].ffill()
    for k in ("open", "high", "low"):
        full[k] = full[k].fillna(close)
    full["close"] = close
    full["volume"] = full["volume"].fillna(0.0)
    return full


def run(df: pd.DataFrame, cfg: DynGridConfig | None = None, acct_cfg: AccountConfig | None = None) -> Result:
    cfg = cfg or DynGridConfig()
    acct_cfg = acct_cfg or AccountConfig()
    if cfg.mode not in ("rebuild", "follow"):
        raise ValueError(f"unknown mode {cfg.mode}")
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    ema, atr = indicators(df, cfg)
    n_roll = rollover_counts(df.index, acct_cfg.rollover_hour_jst)
    half = cfg.cells / 2
    max_units = int(np.ceil(half))
    acct = MarginAccount(acct_cfg.initial_jpy)
    equity = np.full(len(df), acct_cfg.initial_jpy, dtype=float)
    trades, n_limit, n_market, loss_cuts, cap_hits, exits, bars_outside = [], 0, 0, 0, 0, 0, 0

    floor_, step, unit = 0.0, 0.0, 0.0
    e = 0.0  # the empty slot: buys sit on the lines below it, sells on the lines above
    m = 0  # units held (+ long, - short)
    grid_start, grid_equity = None, 0.0

    def line(k: float) -> float:
        return floor_ + k * step

    def fill(new_qty: float, price: float, market: bool) -> bool:
        nonlocal n_limit, n_market, cap_hits
        new_qty = round(new_qty / acct_cfg.size_step) * acct_cfg.size_step
        if new_qty == acct.qty:
            return False
        px = price * (1 + acct_cfg.slippage * np.sign(new_qty - acct.qty)) if market else price
        fee = acct_cfg.fee_rate if market else cfg.maker_fee

        def over_cap(q: float) -> bool:  # equity after this fill's fee must carry the new position
            return abs(q) * px > risk.MAX_LEVERAGE * (acct.equity(px) - abs(q - acct.qty) * px * fee)

        if new_qty and over_cap(new_qty):
            cap_hits += 1
            if not market:
                return False  # a grid order that would break 2x is not placed
            new_qty = np.trunc(risk.cap_quantity(new_qty, px, acct.equity(px), fee) / acct_cfg.size_step) * acct_cfg.size_step
            while new_qty and over_cap(new_qty):
                new_qty -= np.sign(new_qty) * acct_cfg.size_step
            if new_qty == acct.qty:
                return False
        acct.set_position(new_qty, px, fee)
        if market:
            n_market += 1
        else:
            n_limit += 1
        return True

    def place(t: int, price: float) -> None:
        """(Re)draw the lines from this bar's EMA/ATR, empty slot at the price."""
        nonlocal floor_, step, e
        floor_ = ema[t] - cfg.width_atr * atr[t]
        step = 2 * cfg.width_atr * atr[t] / cfg.cells
        e = np.floor((price - floor_) / step) + 0.5  # between the line below and the line above

    def size(price: float) -> None:
        nonlocal unit
        raw = cfg.edge_leverage * acct.equity(price) / (half * price)
        unit = np.floor(raw / acct_cfg.size_step) * acct_cfg.size_step
        if unit < acct_cfg.min_order:
            unit = 0.0

    def start_grid(t: int, price: float) -> None:
        nonlocal m, grid_start, grid_equity
        place(t, price)
        size(price)
        m, grid_start, grid_equity = 0, df.index[t], acct.equity(price)

    def close_grid(t: int, price: float, reason: str) -> None:
        nonlocal m
        if acct.qty:
            fill(0.0, price, market=True)
        m = 0
        eq = acct.equity(price)
        trades.append({"entry_time": grid_start, "exit_time": df.index[t], "exit_px": price,
                       "return": eq / grid_equity - 1 if grid_equity else 0.0, "reason": reason})

    def walk(p: float) -> None:
        """Fill every grid order the price reaches on its way to p. A buy on line k leaves
        k empty, so its paired sell is on line k + 1, and the other way round."""
        nonlocal e, m
        while unit:
            buy, sell = int(np.ceil(e)) - 1, int(np.floor(e)) + 1
            if 0 <= buy <= cfg.cells and p <= line(buy) and m < max_units:
                if not fill(acct.qty + unit, line(buy), market=False):
                    return
                m, e = m + 1, buy
            elif 0 <= sell <= cfg.cells and p >= line(sell) and m > -max_units:
                if not fill(acct.qty - unit, line(sell), market=False):
                    return
                m, e = m - 1, sell
            else:
                return

    start = int(np.argmax(~np.isnan(ema) & ~np.isnan(atr) & (atr > 0)))
    active = False  # a rebuild-mode grid only starts while the price is inside its range
    t = start
    for t in range(start, len(df)):
        if acct.qty and n_roll[t]:
            acct.charge(n_roll[t] * acct_cfg.leverage_fee_per_day * abs(acct.qty) * o[t])

        if not active:
            start_grid(t, o[t])
            active = bool(unit) and (cfg.mode == "follow" or floor_ <= o[t] <= line(cfg.cells))
        elif cfg.mode == "follow" and atr[t] > 0:
            place(t, o[t])  # the grid moves with the EMA/ATR; the position is kept
            if n_roll[t]:
                size(o[t])

        path = (o[t], l[t], h[t], c[t]) if c[t] >= o[t] else (o[t], h[t], l[t], c[t])
        for i, p in enumerate(path):
            if not active:
                break
            if acct.qty and risk.maintenance_ratio(acct.equity(p), acct.qty, p) <= risk.LOSS_CUT_RATIO:
                close_grid(t, risk.loss_cut_price(acct.collateral, acct.qty, acct.entry), "loss_cut")
                loss_cuts += 1
                active = False
                break
            walk(p)
            if cfg.mode == "rebuild":
                top, bottom = line(cfg.cells + 1), line(-1)
                if p >= top or p <= bottom:
                    # one step past the range: stop out there (or at the open if it gapped),
                    # then wait for the next bar's EMA/ATR to build a new grid
                    close_grid(t, p if i == 0 else (top if p >= top else bottom), "range_exit")
                    exits += 1
                    active = False
        if not floor_ <= c[t] <= line(cfg.cells):
            bars_outside += 1

        equity[t] = acct.equity(c[t])
        if equity[t] <= 0:
            equity[t:] = equity[t]
            break

    close_grid(t, c[t], "end")
    eq = pd.Series(equity[start: t + 1], index=df.index[start: t + 1], name="equity")
    tr = pd.DataFrame(trades)
    stats = _stats(eq, tr, df.iloc[start:], acct, acct_cfg, loss_cuts, cap_hits)
    stats.update(grid_fills=n_limit, market_fills=n_market, range_exits=exits,
                 pct_bars_outside_range=bars_outside / max(t + 1 - start, 1))
    if cfg.mode == "follow":
        stats["signal_trades"] = n_limit
        stats["win_rate"] = float("nan")
    return Result(eq, tr, stats)


def main(argv: list[str] | None = None) -> None:
    from btcbot.backtest import format_stats

    p = argparse.ArgumentParser(description="Backtest the EMA/ATR dynamic grid")
    p.add_argument("--csv", required=True)
    p.add_argument("--cells", type=int, nargs="+", default=[10, 15, 20])
    p.add_argument("--mode", nargs="+", default=["rebuild", "follow"], choices=["rebuild", "follow"])
    p.add_argument("--width-atr", type=float, default=DynGridConfig.width_atr)
    p.add_argument("--edge-leverage", type=float, default=DynGridConfig.edge_leverage)
    a = p.parse_args(argv)
    df = fill_gaps(load_csv(a.csv))
    for mode in a.mode:
        for cells in a.cells:
            res = run(df, DynGridConfig(cells=cells, mode=mode, width_atr=a.width_atr, edge_leverage=a.edge_leverage))
            print(f"--- mode={mode} cells={cells}")
            print(format_stats(res.stats))


if __name__ == "__main__":
    main()
