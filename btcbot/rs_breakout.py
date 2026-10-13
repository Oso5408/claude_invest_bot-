"""Strategy C "btc_confirmed_ada_rs_v1": ADA breakout filtered by BTC trend and ADA-vs-BTC
relative strength. Backtest only.

Spec (oso01, 2026-10-13), written for 5m ADA bars and 15m BTC bars:
- BTC trend (filter bars, e.g. 15m): up = close > EMA50 and EMA50 rising; down = mirror.
- RS = ADA return - BTC return over the last rs_minutes (60), from synchronous closes.
- Trigger (ADA bars): close above the highest high of the previous `lookback` (12) finished
  bars = long; close below the lowest low = short.
- Versions: C0 breakout only, C1 + BTC trend, C2 + RS sign, C3 + both.
- The signal is known at the bar close; entry at the next bar's open (taker + slippage).
- Stop: lowest low (highest high) of the last `stop_bars` (6) finished bars, incl. the signal bar.
- Target: 1.5R from the fill, a resting limit (maker). Time exit after `hold_bars` bars (60 min).
- Inside a bar the stop is checked before the target, so a bar that hits both is a loss.

No look-ahead: a BTC or RS value is used only from the close of the bar it was computed on;
an ADA bar may use it when that close is at or before the ADA bar's own close.

Sizing: risk_frac of equity at the stop, capped at risk.MAX_LEVERAGE x equity.
Costs default to Hyperliquid (taker 0.045%, maker 0.015%, 0.05% slippage, hourly funding).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcbot import risk
from btcbot.backtest import max_drawdown
from btcbot.breakout import funding_per_bar
from btcbot.hyperliquid import MAKER_FEE, TAKER_FEE

VERSIONS = {"C0": (False, False), "C1": (True, False), "C2": (False, True), "C3": (True, True)}


@dataclass
class RSConfig:
    version: str = "C3"
    ema: int = 50
    rs_minutes: int = 60
    lookback: int = 12
    stop_bars: int = 6
    target_r: float = 1.5
    hold_bars: int = 12
    risk_frac: float = 0.01
    initial_equity: float = 200.0
    taker_fee: float = TAKER_FEE
    maker_fee: float = MAKER_FEE
    slippage: float = 0.0005


@dataclass
class RSResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def _bar_len(index: pd.DatetimeIndex) -> pd.Timedelta:
    return index.to_series().diff().min()


def as_of(values: np.ndarray, src_close: pd.DatetimeIndex, dst_close: pd.DatetimeIndex) -> np.ndarray:
    """For each destination close time, the last source value whose bar had closed by then."""
    pos = np.searchsorted(src_close.asi8, dst_close.asi8, side="right") - 1
    out = np.full(len(dst_close), np.nan)
    ok = pos >= 0
    out[ok] = np.asarray(values, dtype=float)[pos[ok]]
    return out


def btc_trend(btc: pd.DataFrame, ema: int = 50) -> np.ndarray:
    """+1 up, -1 down, 0 unclear, per BTC bar (known at that bar's close)."""
    c = btc["close"].to_numpy(dtype=float)
    e = pd.Series(c).ewm(span=ema, adjust=False, min_periods=ema).mean().to_numpy()
    rising = np.r_[False, e[1:] > e[:-1]]
    falling = np.r_[False, e[1:] < e[:-1]]
    return np.where((c > e) & rising, 1, np.where((c < e) & falling, -1, 0))


def relative_strength(ada: pd.DataFrame, btc_rs: pd.DataFrame, minutes: int) -> pd.Series:
    """RS = ADA return - BTC return over `minutes`, on btc_rs's bar grid (synchronous closes).
    ADA is resampled to that grid with the close of the ADA bar ending at the same time."""
    step = _bar_len(btc_rs.index)
    n = int(pd.Timedelta(minutes=minutes) / step)
    if n < 1:
        raise ValueError("rs_minutes shorter than the BTC bar")
    a = ada["close"].resample(step).last().reindex(btc_rs.index)
    b = btc_rs["close"]
    return (a / a.shift(n) - 1) - (b / b.shift(n) - 1)


def signals(ada: pd.DataFrame, btc_filter: pd.DataFrame, btc_rs: pd.DataFrame, cfg: RSConfig) -> dict:
    """Per ADA bar, everything known at its close."""
    ada_close_t = ada.index + _bar_len(ada.index)
    trend = as_of(btc_trend(btc_filter, cfg.ema), btc_filter.index + _bar_len(btc_filter.index), ada_close_t)
    rs = relative_strength(ada, btc_rs, cfg.rs_minutes)
    rs_a = as_of(rs.to_numpy(), btc_rs.index + _bar_len(btc_rs.index), ada_close_t)
    hi = ada["high"].rolling(cfg.lookback).max().shift().to_numpy()
    lo = ada["low"].rolling(cfg.lookback).min().shift().to_numpy()
    c = ada["close"].to_numpy()
    return {"trend": trend, "rs": rs_a, "brk_long": c > hi, "brk_short": c < lo,
            "stop_long": ada["low"].rolling(cfg.stop_bars).min().to_numpy(),
            "stop_short": ada["high"].rolling(cfg.stop_bars).max().to_numpy()}


def run(ada: pd.DataFrame, btc_filter: pd.DataFrame, btc_rs: pd.DataFrame, cfg: RSConfig | None = None,
        funding: pd.Series | None = None, sig: dict | None = None) -> RSResult:
    cfg = cfg or RSConfig()
    sig = sig or signals(ada, btc_filter, btc_rs, cfg)
    use_trend, use_rs = VERSIONS[cfg.version]
    trend, rs = sig["trend"], sig["rs"]
    want_long = sig["brk_long"] & ~np.isnan(trend) & ~np.isnan(rs)
    want_short = sig["brk_short"] & ~np.isnan(trend) & ~np.isnan(rs)
    if use_trend:
        want_long &= trend == 1
        want_short &= trend == -1
    if use_rs:
        want_long &= rs > 0
        want_short &= rs < 0

    o, h, l, c = (ada[k].to_numpy() for k in ("open", "high", "low", "close"))
    fund = funding_per_bar(ada.index, funding) if funding is not None else np.zeros(len(ada))
    n = len(ada)
    eq = cfg.initial_equity
    curve = np.empty(n)
    trades = []
    pos = None
    pending = 0  # side signalled at the previous bar's close

    def close(t: int, px: float, fee: float, reason: str) -> None:
        nonlocal eq, pos
        side, qty = pos["side"], pos["qty"]
        eq += side * qty * (px - pos["entry"]) - qty * px * fee
        pnl = eq - pos["eq_before"]
        base = cfg.risk_frac * pos["eq_before"]
        trades.append({"signal_time": pos["signal_time"], "entry_time": ada.index[pos["t"]], "exit_time": ada.index[t],
                       "side": side, "entry": pos["entry"], "stop": pos["stop"], "target": pos["target"], "exit": px,
                       "reason": reason, "bars": t - pos["t"] + 1, "stop_pct": pos["risk"] / pos["entry"],
                       "trend": pos["trend"], "rs": pos["rs"],
                       "R": side * qty * (px - pos["entry"]) / base, "net_R": pnl / base,
                       "pnl": pnl, "return": pnl / pos["eq_before"]})
        pos = None

    for t in range(n):
        # 1. enter at this bar's open on the previous close's signal
        if pending and pos is None:
            side, s = pending, t - 1
            fill = o[t] * (1 + cfg.slippage * side)
            stop = sig["stop_long"][s] if side > 0 else sig["stop_short"][s]
            risk_dist = side * (fill - stop)
            if risk_dist > 0:
                qty = abs(risk.cap_quantity(cfg.risk_frac * eq / risk_dist, fill, eq, cfg.taker_fee))
                if qty > 0:
                    eq_before = eq
                    eq -= qty * fill * cfg.taker_fee
                    pos = {"side": side, "qty": qty, "entry": fill, "stop": stop, "risk": risk_dist,
                           "target": fill + side * cfg.target_r * risk_dist, "t": t, "eq_before": eq_before,
                           "signal_time": ada.index[s], "trend": trend[s], "rs": rs[s]}
        pending = 0

        # 2. manage: funding, stop before target, time exit at the close
        if pos is not None:
            side = pos["side"]
            if t > pos["t"]:
                eq -= side * fund[t] * pos["qty"] * o[t]
            if (l[t] <= pos["stop"]) if side > 0 else (h[t] >= pos["stop"]):
                px = min(o[t], pos["stop"]) if side > 0 else max(o[t], pos["stop"])
                close(t, px * (1 - cfg.slippage * side), cfg.taker_fee, "stop")
            elif (h[t] >= pos["target"]) if side > 0 else (l[t] <= pos["target"]):
                px = max(o[t], pos["target"]) if side > 0 else min(o[t], pos["target"])
                close(t, px, cfg.maker_fee, "target")
            elif t - pos["t"] + 1 >= cfg.hold_bars:
                close(t, c[t] * (1 - cfg.slippage * side), cfg.taker_fee, "time")

        # 3. new signal at this close (only when flat; no pyramiding)
        if pos is None and t + 1 < n:
            pending = 1 if want_long[t] else (-1 if want_short[t] else 0)

        curve[t] = eq + (pos["side"] * pos["qty"] * (c[t] - pos["entry"]) if pos else 0.0)
        if curve[t] <= 0:
            curve[t:] = curve[t]
            break

    if pos is not None:
        close(t, c[t] * (1 - cfg.slippage * pos["side"]), cfg.taker_fee, "end")
        curve[t] = eq
    m = t + 1
    eqs = pd.Series(curve[:m], index=ada.index[:m], name="equity")
    tr = pd.DataFrame(trades)
    return RSResult(eqs, tr, stats(eqs, tr, cfg))


def stats(eq: pd.Series, tr: pd.DataFrame, cfg: RSConfig) -> dict:
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    daily = eq.resample("1D").last().pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0
    out = {"total_return": float(eq.iloc[-1] / cfg.initial_equity - 1), "max_drawdown": max_drawdown(eq),
           "sharpe": sharpe, "trades": len(tr), "trades_per_week": len(tr) / days * 7}
    if len(tr):
        gains = tr["pnl"].sort_values(ascending=False)
        net = tr["pnl"].sum()
        out.update(win_rate=float((tr["pnl"] > 0).mean()), avg_R=float(tr["R"].mean()),
                   avg_net_R=float(tr["net_R"].mean()), cost_R=float((tr["R"] - tr["net_R"]).mean()),
                   avg_net_pct=float(tr["return"].mean()), avg_bars=float(tr["bars"].mean()),
                   target_rate=float((tr["reason"] == "target").mean()),
                   median_stop_pct=float(tr["stop_pct"].median()),
                   top5_share=float(gains.head(5).sum() / net) if net > 0 else float("nan"))
    return out
