"""Bar-by-bar backtest for the strategy in strategy.py.

Timing rule (no look-ahead): the decision is made on bar t's close and the
order fills at bar t+1's open, with slippage and a fee on every fill.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from btcbot.data import load_csv
from btcbot.strategy import BayesKelly, StrategyConfig, compute_features


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0  # JPY
    fee_rate: float = 0.0005  # 0.05% per fill (GMO Coin spot taker); maker is a rebate, we stay conservative
    slippage: float = 0.0002  # 0.02% worse price on every fill


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, strat: StrategyConfig | None = None, bt: BacktestConfig | None = None) -> Result:
    strat = strat or StrategyConfig()
    bt = bt or BacktestConfig()
    f = compute_features(df, strat)
    opens, closes = f["open"].to_numpy(), f["close"].to_numpy()
    signal, entry_ok = f["signal"].to_numpy(), f["entry_ok"].to_numpy()

    bk = BayesKelly(strat)
    cash, btc, fees_paid = bt.initial_cash, 0.0, 0.0
    in_trade, entry_px, entry_time, size = False, 0.0, None, 0.0
    pending: tuple[str, float] | None = None
    equity = np.empty(len(f))
    trades = []

    for t in range(len(f)):
        if pending is not None:
            action, pend_size = pending
            pending = None
            if action == "enter":
                entry_px = opens[t] * (1 + bt.slippage)
                entry_time, size, in_trade = f.index[t], pend_size, True
                if size > 0:
                    notional = (cash + btc * opens[t]) * size
                    fee = notional * bt.fee_rate
                    btc += (notional - fee) / entry_px
                    cash -= notional
                    fees_paid += fee
            else:  # exit
                exit_px = opens[t] * (1 - bt.slippage)
                # Return the signal earned per yen, after both fees; feeds the learner even when size was 0.
                trade_ret = exit_px / entry_px * (1 - bt.fee_rate) ** 2 - 1
                bk.update(trade_ret)
                if btc > 0:
                    proceeds = btc * exit_px
                    fee = proceeds * bt.fee_rate
                    cash += proceeds - fee
                    fees_paid += fee
                    btc = 0.0
                trades.append({"entry_time": entry_time, "exit_time": f.index[t], "entry_px": entry_px,
                               "exit_px": exit_px, "size": size, "return": trade_ret,
                               "win_prob_after": bk.win_prob, "kelly_after": bk.kelly()})
                in_trade = False

        equity[t] = cash + btc * closes[t]

        if t == len(f) - 1:
            break
        if in_trade and not signal[t]:
            pending = ("exit", 0.0)
        elif not in_trade and entry_ok[t]:
            pending = ("enter", bk.position_size())

    eq = pd.Series(equity, index=f.index, name="equity")
    trades_df = pd.DataFrame(trades)
    return Result(eq, trades_df, summarize(eq, trades_df, df, bt, fees_paid))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def summarize(eq: pd.Series, trades: pd.DataFrame, df: pd.DataFrame, bt: BacktestConfig, fees_paid: float) -> dict:
    total = eq.iloc[-1] / bt.initial_cash - 1
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    bar_ret = eq.pct_change().dropna()
    bars_per_year = len(eq) / years
    sharpe = float(bar_ret.mean() / bar_ret.std() * np.sqrt(bars_per_year)) if bar_ret.std() > 0 else 0.0
    taken = trades[trades["size"] > 0] if len(trades) else trades
    bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    return {
        "start": str(eq.index[0]),
        "end": str(eq.index[-1]),
        "final_equity_jpy": round(float(eq.iloc[-1])),
        "total_return": float(total),
        "cagr": float((1 + total) ** (1 / years) - 1),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe,
        "fees_paid_jpy": round(fees_paid),
        "signal_trades": len(trades),
        "trades_taken": len(taken),
        "win_rate_taken": float((taken["return"] > 0).mean()) if len(taken) else float("nan"),
        "buy_and_hold_return": float(bh),
        "buy_and_hold_max_drawdown": max_drawdown(df["close"]),
    }


def format_stats(s: dict) -> str:
    pct = {"total_return", "cagr", "max_drawdown", "win_rate_taken", "win_rate", "buy_and_hold_return", "buy_and_hold_max_drawdown"}
    lines = []
    for k, v in s.items():
        lines.append(f"{k:28s} {v:.2%}" if k in pct else f"{k:28s} {v:,.2f}" if isinstance(v, float) else f"{k:28s} {v:,}" if isinstance(v, int) else f"{k:28s} {v}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Backtest the BTC strategy")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV from btcbot.data")
    src.add_argument("--synthetic", action="store_true", help="use random-walk data (for checking the code, not the strategy)")
    p.add_argument("--fee", type=float, default=BacktestConfig.fee_rate)
    p.add_argument("--hours", default=None, help="JST entry hours, e.g. 9-17 or 9,10,21")
    p.add_argument("--trades-out", default=None, help="optional CSV path for the trade list")
    args = p.parse_args(argv)

    if args.csv:
        df = load_csv(args.csv)
    else:
        from btcbot.synthetic import random_walk
        df = random_walk()

    hours = None
    if args.hours:
        if "-" in args.hours:
            a, b = map(int, args.hours.split("-"))
            hours = tuple(range(a, b + 1))
        else:
            hours = tuple(int(h) for h in args.hours.split(","))

    res = run_backtest(df, StrategyConfig(entry_hours_jst=hours), BacktestConfig(fee_rate=args.fee))
    print(format_stats(res.stats))
    if args.trades_out:
        res.trades.to_csv(args.trades_out, index=False)
        print(f"trades saved to {args.trades_out}")


if __name__ == "__main__":
    main()
