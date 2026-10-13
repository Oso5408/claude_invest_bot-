"""ADA 5m Trend Pullback v1 (oso01's spec). Backtest only.

Rules (long side; shorts mirrored). Every parameter is fixed before the test:
1. Direction (1H): the last *finished* hourly bar closed above EMA50 and EMA50 is higher than
   `slope_bars` hours earlier = long only; the opposite = short only; anything else = no trade.
   A "trend" is one unbroken run of the same direction. At most `max_entries` per trend.
2. Pullback (trading timeframe): a bar closes below EMA20 (price moving against the 1H trend).
   The pullback low is the lowest low from that bar on.
3. Confirmation: the first bar that closes back above EMA20 *and* closes up (close > open).
   Entry: a buy-stop at the previous bar's high, live for the next `order_bars` bars
   (each bar's trigger is the bar before it). Filled at max(open, trigger) plus slippage, taker fee.
   A close back below EMA20 before the fill re-arms the pullback; a finished order without a fill
   waits for a new pullback.
4. Stop: pullback low minus `buffer_atr` x ATR(14) of the trading timeframe, market exit (taker +
   slippage). Exit "fixed": target `target_r` x the real entry-to-stop distance, resting limit
   (maker fee). Exit "atr_trail": no target; after each closed bar the stop moves up to
   highest close since entry minus `trail_atr` x ATR (never down).
   A bar that reaches both the stop and the target counts as stopped. An entry bar that also
   trades through the stop counts as filled and stopped.

Indicators at bar t use bars up to t only; orders made at the close of t act from bar t+1.
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


@dataclass
class PullbackConfig:
    ema_1h: int = 50
    slope_bars: int = 5  # EMA50 "rising" = above its value this many hours ago
    ema_fast: int = 20
    atr_len: int = 14
    order_bars: int = 3
    buffer_atr: float = 0.2
    max_entries: int = 2  # per trend
    exit: str = "fixed"  # "fixed" or "atr_trail"
    target_r: float = 1.5
    trail_atr: float = 2.0
    long_only: bool = False
    risk_frac: float = 0.01
    initial_equity: float = 200.0
    taker_fee: float = TAKER_FEE
    maker_fee: float = MAKER_FEE
    slippage: float = 0.0005


@dataclass
class PullbackResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def hourly_direction(df: pd.DataFrame, cfg: PullbackConfig) -> np.ndarray:
    """Per bar: the 1H direction known at the bar's close (+1, -1 or 0)."""
    h = df["close"].resample("1h", label="left", closed="left").last().dropna()
    ema = h.ewm(span=cfg.ema_1h, adjust=False, min_periods=cfg.ema_1h).mean()
    up = (h > ema) & (ema > ema.shift(cfg.slope_bars))
    down = (h < ema) & (ema < ema.shift(cfg.slope_bars))
    d = np.where(up, 1, np.where(down, -1, 0))
    known_at = (h.index + pd.Timedelta(hours=1)).as_unit("ns").asi8  # an hourly bar is known when it closes
    step = df.index.to_series().diff().min()
    bar_close = (df.index + step).as_unit("ns").asi8
    i = np.searchsorted(known_at, bar_close, side="right") - 1
    return np.where(i >= 0, d[np.clip(i, 0, None)], 0)


def indicators(df: pd.DataFrame, cfg: PullbackConfig) -> tuple[np.ndarray, np.ndarray]:
    c = df["close"]
    ema = c.ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    prev = c.shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / cfg.atr_len, adjust=False, min_periods=cfg.atr_len).mean()
    return ema.to_numpy(), atr.to_numpy()


def run(df: pd.DataFrame, cfg: PullbackConfig | None = None, funding: pd.Series | None = None) -> PullbackResult:
    cfg = cfg or PullbackConfig()
    direction = hourly_direction(df, cfg)
    ema, atr = indicators(df, cfg)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    fund = funding_per_bar(df.index, funding) if funding is not None else np.zeros(len(df))
    n = len(df)

    eq = cfg.initial_equity
    curve = np.empty(n)
    trades = []
    pos = None
    trend, entries = 0, 0
    phase = "wait"  # wait -> pullback -> armed
    pb_ext = np.nan  # pullback low (long) / high (short)
    order_left = 0

    def close(t: int, px: float, fee: float, reason: str) -> None:
        nonlocal eq, pos
        side, qty = pos["side"], pos["qty"]
        eq += side * qty * (px - pos["entry"]) - qty * px * fee
        pnl = eq - pos["eq_before"]
        planned = cfg.risk_frac * pos["eq_before"]
        trades.append({"entry_time": pos["time"], "exit_time": df.index[t], "side": side, "entry": pos["entry"],
                       "stop0": pos["stop0"], "exit": px, "reason": reason, "stop_pct": pos["risk"] / pos["entry"],
                       "bars": t - pos["t"], "leverage": qty * pos["entry"] / pos["eq_before"],
                       "R": side * qty * (px - pos["entry"]) / planned,  # before fees and funding
                       "net_R": pnl / planned,  # after fees, slippage and funding
                       "pnl": pnl, "return": pnl / pos["eq_before"]})
        pos = None

    for t in range(n):
        # 1. manage an open position (stop first if both are inside one bar)
        if pos is not None:
            side = pos["side"]
            eq -= side * fund[t] * pos["qty"] * o[t]
            if (l[t] <= pos["stop"]) if side > 0 else (h[t] >= pos["stop"]):
                px = min(o[t], pos["stop"]) if side > 0 else max(o[t], pos["stop"])
                close(t, px * (1 - cfg.slippage * side), cfg.taker_fee, "stop")
            elif pos["target"] is not None and ((h[t] >= pos["target"]) if side > 0 else (l[t] <= pos["target"])):
                px = max(o[t], pos["target"]) if side > 0 else min(o[t], pos["target"])
                close(t, px, cfg.maker_fee, "target")

        # 2. a live entry order from earlier closes (trigger = previous bar's high / low)
        if pos is None and phase == "armed" and t > 0:
            side = trend
            trig = h[t - 1] if side > 0 else l[t - 1]
            if (h[t] > trig) if side > 0 else (l[t] < trig):
                fill = (max(o[t], trig) if side > 0 else min(o[t], trig)) * (1 + cfg.slippage * side)
                stop = pb_ext - side * cfg.buffer_atr * atr[t - 1]
                risk_dist = side * (fill - stop)
                phase = "wait"
                qty = abs(risk.cap_quantity(cfg.risk_frac * eq / risk_dist, fill, eq, cfg.taker_fee)) if risk_dist > 0 else 0.0
                if qty > 0:
                    eq_before = eq
                    eq -= qty * fill * cfg.taker_fee
                    target = fill + side * cfg.target_r * risk_dist if cfg.exit == "fixed" else None
                    pos = {"side": side, "qty": qty, "entry": fill, "stop": stop, "stop0": stop, "target": target,
                           "risk": risk_dist, "time": df.index[t], "t": t, "eq_before": eq_before, "best": c[t]}
                    entries += 1
                    # rest of the entry bar: through the stop = filled then stopped (worst case)
                    if (l[t] <= stop) if side > 0 else (h[t] >= stop):
                        close(t, stop * (1 - cfg.slippage * side), cfg.taker_fee, "stop")
                    elif target is not None and ((h[t] >= target) if side > 0 else (l[t] <= target)):
                        close(t, target, cfg.maker_fee, "target")

        # 3. trailing stop update from this closed bar (acts from the next bar)
        if pos is not None and cfg.exit == "atr_trail" and not np.isnan(atr[t]):
            side = pos["side"]
            pos["best"] = max(pos["best"], c[t]) if side > 0 else min(pos["best"], c[t])
            new = pos["best"] - side * cfg.trail_atr * atr[t]
            pos["stop"] = max(pos["stop"], new) if side > 0 else min(pos["stop"], new)

        curve[t] = eq + (pos["side"] * pos["qty"] * (c[t] - pos["entry"]) if pos else 0.0)
        if curve[t] <= 0:
            curve[t:] = curve[t]
            break

        # 4. signal state at this bar's close
        d = int(direction[t])
        if d != trend:
            trend, entries, phase, pb_ext = d, 0, "wait", np.nan
        if cfg.long_only and trend < 0:
            continue
        if trend == 0 or np.isnan(ema[t]) or np.isnan(atr[t]) or entries >= cfg.max_entries:
            phase = "wait"
            continue
        against = c[t] < ema[t] if trend > 0 else c[t] > ema[t]
        if phase == "armed":
            order_left -= 1
            if against:
                phase = "pullback"
                pb_ext = min(pb_ext, l[t]) if trend > 0 else max(pb_ext, h[t])
            elif order_left <= 0:
                phase = "wait"
            continue
        if phase == "wait" and against and pos is None:
            phase, pb_ext = "pullback", (l[t] if trend > 0 else h[t])
            continue
        if phase == "pullback":
            pb_ext = min(pb_ext, l[t]) if trend > 0 else max(pb_ext, h[t])
            confirmed = (c[t] > ema[t] and c[t] > o[t]) if trend > 0 else (c[t] < ema[t] and c[t] < o[t])
            if confirmed and pos is None:
                phase, order_left = "armed", cfg.order_bars
            elif pos is not None:
                phase = "wait"

    m = t + 1
    eqs = pd.Series(curve[:m], index=df.index[:m], name="equity")
    tr = pd.DataFrame(trades)
    return PullbackResult(eqs, tr, _stats(eqs, tr, cfg))


def _stats(eq: pd.Series, tr: pd.DataFrame, cfg: PullbackConfig) -> dict:
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    out = {"total_return": float(eq.iloc[-1] / cfg.initial_equity - 1), "max_drawdown": max_drawdown(eq),
           "final_equity": float(eq.iloc[-1]), "trades": len(tr), "trades_per_week": len(tr) / days * 7}
    if len(tr):
        out.update(win_rate=float((tr["pnl"] > 0).mean()), avg_R=float(tr["R"].mean()),
                   avg_net_R=float(tr["net_R"].mean()), cost_R=float((tr["R"] - tr["net_R"]).mean()),
                   median_stop_pct=float(tr["stop_pct"].median()), median_bars=float(tr["bars"].median()))
    return out
