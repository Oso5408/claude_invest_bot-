"""Neutral dynamic grid on GMO Coin crypto FX (default ADA_JPY), up to 2x leverage.

Grid:
- `levels` lines spaced `spacing` apart (geometric), centred on the price when the
  grid is built: with the defaults, 10 lines 1% apart above and 10 below (+/-10%).
- Neutral: each line crossed going down adds one unit long (or closes one unit
  short), each line crossed going up adds one unit short (or closes one unit long).
  So the position is -unit x (grid lines above the centre), and every down-up or
  up-down round trip across one line earns one spacing on one unit.
- Unit size: at the edge of the range the position reaches `edge_leverage` x
  equity (2x by default), never more, because risk.py caps every fill.
- Dynamic: if price leaves the range, close everything at market and rebuild
  the grid around the new price. That is where a grid takes its losses.

Fills: grid orders are limit orders filled at the line price (maker fee, 0 on
ADA_JPY, no slippage). Range-exit closes are market orders (taker fee + slippage).
Inside each bar the price is assumed to go open -> low -> high -> close on a
rising bar and open -> high -> low -> close on a falling bar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcbot import risk
from btcbot.data import load_csv, resample
from btcbot.meanrev import AccountConfig, MarginAccount, Result, _stats


@dataclass
class GridConfig:
    levels: int = 20  # lines in the range, half above and half below the centre
    spacing: float = 0.01
    edge_leverage: float = 2.0
    maker_fee: float = 0.0  # ADA_JPY makerFee from the GMO symbols API


def run(df: pd.DataFrame, cfg: GridConfig | None = None, acct_cfg: AccountConfig | None = None) -> Result:
    cfg = cfg or GridConfig()
    acct_cfg = acct_cfg or AccountConfig()
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    jst_hour = df.index.tz_convert("Asia/Tokyo").hour
    half = cfg.levels // 2
    acct = MarginAccount(acct_cfg.initial_jpy)
    equity = np.empty(len(df))
    fills, resets, loss_cuts, cap_hits = [], 0, 0, 0
    trades = []  # one row per grid lifetime, so the shared stats have something to count

    center, unit, g, grid_start, grid_equity = 0.0, 0.0, 0, None, 0.0

    def line(k: int) -> float:
        return center * (1 + cfg.spacing) ** k

    def build(t: int, price: float) -> None:
        nonlocal center, unit, g, grid_start, grid_equity
        center, g, grid_start = price, 0, df.index[t]
        eq = acct.equity(price)
        grid_equity = eq
        # at the edge the position is `half` units worth about half x edge price
        raw = cfg.edge_leverage * eq / (half * price)
        unit = np.floor(raw / acct_cfg.size_step) * acct_cfg.size_step
        if unit < acct_cfg.min_order:
            unit = 0.0

    def close_grid(t: int, price: float, reason: str) -> None:
        nonlocal resets
        if acct.qty:
            px = price * (1 - acct_cfg.slippage * np.sign(acct.qty))
            acct.set_position(0.0, px, acct_cfg.fee_rate)
        eq = acct.equity(price)
        trades.append({"entry_time": grid_start, "exit_time": df.index[t], "center": center, "exit_px": price,
                       "return": eq / grid_equity - 1 if grid_equity else 0.0, "reason": reason})
        resets += 1

    def step_to(t: int, target: float) -> bool:
        """Walk the price to `target`, filling every line crossed. False if the range broke."""
        nonlocal g, cap_hits
        while True:
            up = line(g + 1)
            down = line(g - 1)
            if target >= up:
                if g + 1 > half:
                    return False
                new_qty = acct.qty - unit
                k = +1
                px = up
            elif target <= down:
                if g - 1 < -half:
                    return False
                new_qty = acct.qty + unit
                k = -1
                px = down
            else:
                return True
            if unit and abs(new_qty) > abs(acct.qty):
                capped = risk.cap_quantity(new_qty, px, acct.equity(px), cfg.maker_fee)
                if abs(capped) < abs(new_qty):
                    cap_hits += 1
                    new_qty = acct.qty  # skip the fill, keep tracking the line
            if new_qty != acct.qty:
                acct.set_position(new_qty, px, cfg.maker_fee)
                fills.append(df.index[t])
            g += k

    build(0, o[0])
    for t in range(len(df)):
        if acct.qty and jst_hour[t] == acct_cfg.rollover_hour_jst:
            acct.charge(acct_cfg.leverage_fee_per_day * abs(acct.qty) * o[t])

        path = (o[t], l[t], h[t], c[t]) if c[t] >= o[t] else (o[t], h[t], l[t], c[t])
        for i, p in enumerate(path):
            if acct.qty:
                if risk.maintenance_ratio(acct.equity(p), acct.qty, p) <= risk.LOSS_CUT_RATIO:
                    close_grid(t, risk.loss_cut_price(acct.collateral, acct.qty, acct.entry), "loss_cut")
                    loss_cuts += 1
                    build(t, p)
                    continue
            if not step_to(t, p):
                # price left the range: stop out one spacing past the last line (or at the
                # open if it gapped past), then rebuild the grid around the current price
                stop = line(half + 1) if p > center else line(-half - 1)
                close_grid(t, p if i == 0 else stop, "range_exit")
                build(t, p)

        equity[t] = acct.equity(c[t])
        if equity[t] <= 0:
            equity[t:] = equity[t]
            break

    close_grid(t, c[t], "end")
    eq = pd.Series(equity[: t + 1], index=df.index[: t + 1], name="equity")
    tr = pd.DataFrame(trades)
    stats = _stats(eq, tr, df, acct, acct_cfg, loss_cuts, cap_hits)
    stats["signal_trades"] = len(fills)
    stats["win_rate"] = float((tr["return"] > 0).mean()) if len(tr) else float("nan")
    stats["grid_rebuilds"] = resets - 1
    return Result(eq, tr, stats)


def main(argv: list[str] | None = None) -> None:
    from btcbot.backtest import format_stats

    p = argparse.ArgumentParser(description="Backtest a neutral dynamic grid with up to 2x leverage")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--levels", type=int, default=GridConfig.levels)
    p.add_argument("--spacing", type=float, default=GridConfig.spacing)
    p.add_argument("--timeframe", default=None)
    a = p.parse_args(argv)

    if a.csv:
        df = load_csv(a.csv)
    else:
        from btcbot.synthetic import mean_reverting
        df = mean_reverting()
    if a.timeframe:
        df = resample(df, a.timeframe)
    res = run(df, GridConfig(levels=a.levels, spacing=a.spacing))
    print(format_stats(res.stats))


if __name__ == "__main__":
    main()
