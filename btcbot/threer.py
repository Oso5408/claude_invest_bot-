"""Intraday "3R / probability factor" method (Doug's video), 1-2 trades a day. Backtest only.

Daily context, from finished UTC days only (a value is used from the day after it closed):
- Trend bias: yesterday's close above its 50-day SMA = longs are with the trend, below = shorts.
- Liquidity zones: range high/low = yesterday's high/low. Swing high = the nearest earlier
  daily high above the range high (swing low mirrored).
- Range: daily ATR(14).

Setups on intraday bars (decided at a bar's close, filled at the next bar's open):
- "reversal" (his main example): with an up bias, price dips into the lower zone (low at or
  below range low + zone_tol x ATR) and turns (a close above the previous bar's high within
  touch_window bars of the touch). Long; structural stop just below the lowest low since the
  touch. Mirrored for shorts in a down bias at the upper zone.
- "breakout": with an up bias, the first close above the range high today. Stop just below the
  lowest low of the last touch_window bars. Mirrored for shorts.

Every trade: target = entry + r_mult x risk. Skipped when the stop is too tight or too wide,
when target - today's low (short: today's high - target) needs more than room_atr daily ATRs,
and, with obstacle_check, when the nearest opposite liquidity level sits before the target.
Counter-trend trades are skipped unless counter_risk_mult > 0 (smaller size, counter_r_mult target).

Exits: target as a resting limit (maker fee, no slippage), stop as a market order (taker +
slippage), and anything still open at the end of the UTC day closes at the market. If the stop
and the target are both inside one bar, the stop is assumed to come first.

Sizing: risk_frac of equity per trade at the stop, capped at risk.MAX_LEVERAGE x equity.
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
class ThreeRConfig:
    setup: str = "reversal"  # "reversal", "breakout" or "both"
    sma_days: int = 50
    atr_days: int = 14
    swing_lookback_days: int = 60
    zone_tol: float = 0.1  # x daily ATR: how close to the range level counts as a touch
    touch_window: int = 8  # bars after a touch in which the turn must come; also the breakout stop lookback
    stop_buffer: float = 0.001  # stop sits this fraction beyond the structural level
    min_stop: float = 0.003  # skip when the stop is closer than this (noise and costs eat it)
    max_stop_atr: float = 0.5  # skip when the stop is wider than this many daily ATRs
    r_mult: float = 3.0
    room_atr: float | None = 1.0  # None turns the range check off
    obstacle_check: bool = False
    counter_risk_mult: float = 0.0  # 0 = no counter-trend trades
    counter_r_mult: float = 1.0
    max_trades_per_day: int = 2
    flat_at_day_end: bool = True
    risk_frac: float = 0.01
    initial_equity: float = 200.0
    taker_fee: float = TAKER_FEE
    maker_fee: float = MAKER_FEE
    slippage: float = 0.0005


@dataclass
class ThreeRResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def daily_context(df: pd.DataFrame, cfg: ThreeRConfig) -> pd.DataFrame:
    """Per intraday bar: SMA, ATR, range and swing levels known at the start of its UTC day."""
    day = df.index.floor("D")
    d = df[["high", "low", "close"]].groupby(day).agg({"high": "max", "low": "min", "close": "last"})
    prev = d["close"].shift()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    hi, lo = d["high"].to_numpy(), d["low"].to_numpy()
    swing_hi, swing_lo = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for i in range(len(d)):  # row i describes day i; levels are used on day i + 1
        for j in range(i - 1, max(-1, i - 1 - cfg.swing_lookback_days), -1):
            if np.isnan(swing_hi[i]) and hi[j] > hi[i]:
                swing_hi[i] = hi[j]
            if np.isnan(swing_lo[i]) and lo[j] < lo[i]:
                swing_lo[i] = lo[j]
            if not (np.isnan(swing_hi[i]) or np.isnan(swing_lo[i])):
                break
    cols = {
        "prev_close": d["close"], "sma": d["close"].rolling(cfg.sma_days).mean(),
        "atr": tr.ewm(alpha=1 / cfg.atr_days, adjust=False, min_periods=cfg.atr_days).mean(),
        "range_high": d["high"], "range_low": d["low"], "swing_high": swing_hi, "swing_low": swing_lo,
    }
    # plain arrays on purpose: Series would re-align to the shifted index and leak the same day's values
    ctx = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=d.index + pd.Timedelta(days=1))
    return ctx.reindex(day).set_axis(df.index)


def run(df: pd.DataFrame, cfg: ThreeRConfig | None = None, funding: pd.Series | None = None) -> ThreeRResult:
    cfg = cfg or ThreeRConfig()
    ctx = daily_context(df, cfg)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    sma, atr = ctx["sma"].to_numpy(), ctx["atr"].to_numpy()
    pclose = ctx["prev_close"].to_numpy()
    rh, rl = ctx["range_high"].to_numpy(), ctx["range_low"].to_numpy()
    sh, sl = ctx["swing_high"].to_numpy(), ctx["swing_low"].to_numpy()
    day = df.index.floor("D").asi8
    fund = funding_per_bar(df.index, funding) if funding is not None else np.zeros(len(df))
    last_bar_of_day = np.r_[day[1:] != day[:-1], True]

    eq = cfg.initial_equity
    curve = np.empty(len(df))
    trades = []
    pos = None  # dict while a trade is open
    pending = None  # entry decided at the previous close
    day_lo = day_hi = np.nan
    trades_today, broke_up, broke_dn = 0, False, False
    touch_lo_bar = touch_hi_bar = -10**9
    low_since_touch, high_since_touch = np.inf, -np.inf

    def close(t: int, px: float, fee: float, reason: str) -> None:
        nonlocal eq, pos
        side, qty = pos["side"], pos["qty"]
        pnl = side * qty * (px - pos["entry"]) - qty * px * fee
        eq += pnl
        trades.append({"entry_time": pos["time"], "exit_time": df.index[t], "side": side, "entry": pos["entry"], "qty": qty,
                       "eq_before": pos["eq_before"], "stop": pos["stop"], "target": pos["target"], "exit": px, "reason": reason,
                       "with_trend": pos["with_trend"], "setup": pos["setup"],
                       "pnl": eq - pos["eq_before"],  # after entry and exit fees and funding
                       "R": side * (px - pos["entry"]) / pos["risk"],  # price move in units of risk, before costs
                       "return": eq / pos["eq_before"] - 1})
        pos = None

    for t in range(len(df)):
        if t == 0 or day[t] != day[t - 1]:
            day_lo, day_hi = l[t], h[t]
            trades_today, broke_up, broke_dn = 0, False, False
            touch_lo_bar = touch_hi_bar = -10**9
        else:
            day_lo, day_hi = min(day_lo, l[t]), max(day_hi, h[t])

        # 1. entry decided at the previous close fills at this open (never carried into a new day)
        if pending is not None and day[t] == pending["day"]:
            side = pending["side"]
            px = o[t] * (1 + cfg.slippage * side)
            risk_dist = side * (px - pending["stop"])
            if risk_dist > 0:
                qty = pending["risk_mult"] * cfg.risk_frac * eq / risk_dist
                qty = abs(risk.cap_quantity(qty, px, eq, cfg.taker_fee))
                if qty > 0:
                    fee = qty * px * cfg.taker_fee
                    eq_before = eq
                    eq -= fee
                    pos = {"side": side, "qty": qty, "entry": px, "stop": pending["stop"],
                           "target": px + side * pending["r_mult"] * risk_dist, "risk": risk_dist,
                           "time": df.index[t], "with_trend": pending["with_trend"], "setup": pending["setup"],
                           "entry_fee": fee, "funding": 0.0, "eq_before": eq_before}
                    trades_today += 1
        pending = None

        # 2. funding, stop, target, end of day
        if pos is not None:
            f_cost = pos["side"] * fund[t] * pos["qty"] * o[t]
            eq -= f_cost
            pos["funding"] += f_cost
            side = pos["side"]
            if (l[t] <= pos["stop"]) if side > 0 else (h[t] >= pos["stop"]):
                px = min(o[t], pos["stop"]) if side > 0 else max(o[t], pos["stop"])
                close(t, px * (1 - cfg.slippage * side), cfg.taker_fee, "stop")
            elif (h[t] >= pos["target"]) if side > 0 else (l[t] <= pos["target"]):
                px = max(o[t], pos["target"]) if side > 0 else min(o[t], pos["target"])
                close(t, px, cfg.maker_fee, "target")
            elif cfg.flat_at_day_end and last_bar_of_day[t]:
                close(t, c[t] * (1 - cfg.slippage * side), cfg.taker_fee, "day_end")

        curve[t] = eq + (pos["side"] * pos["qty"] * (c[t] - pos["entry"]) if pos else 0.0)
        if curve[t] <= 0:
            curve[t:] = curve[t]
            break

        # 3. zone touches and structure, then look for a setup at this close
        a = atr[t]
        if np.isnan(a) or np.isnan(sma[t]):
            continue
        tol = cfg.zone_tol * a
        if l[t] <= rl[t] + tol:
            if t - touch_lo_bar > cfg.touch_window:
                low_since_touch = l[t]
            touch_lo_bar = t
        if h[t] >= rh[t] - tol:
            if t - touch_hi_bar > cfg.touch_window:
                high_since_touch = h[t]
            touch_hi_bar = t
        low_since_touch = min(low_since_touch, l[t])
        high_since_touch = max(high_since_touch, h[t])
        new_up = c[t] > rh[t] and not broke_up
        new_dn = c[t] < rl[t] and not broke_dn
        broke_up |= c[t] > rh[t]
        broke_dn |= c[t] < rl[t]
        if pos is not None or trades_today >= cfg.max_trades_per_day or last_bar_of_day[t] or t == len(df) - 1:
            continue

        bias = 1 if pclose[t] > sma[t] else -1
        cands = []  # (side, structural level, setup)
        if cfg.setup in ("reversal", "both") and t > 0:
            if t - touch_lo_bar <= cfg.touch_window and c[t] > h[t - 1] and c[t] < (day_lo + day_hi) / 2:
                cands.append((1, low_since_touch, "reversal"))
            if t - touch_hi_bar <= cfg.touch_window and c[t] < l[t - 1] and c[t] > (day_lo + day_hi) / 2:
                cands.append((-1, high_since_touch, "reversal"))
        if cfg.setup in ("breakout", "both"):
            w = slice(max(0, t - cfg.touch_window + 1), t + 1)
            if new_up:
                cands.append((1, l[w].min(), "breakout"))
            if new_dn:
                cands.append((-1, h[w].max(), "breakout"))
        for side, level, setup in cands:
            with_trend = side == bias
            if not with_trend and cfg.counter_risk_mult <= 0:
                continue
            r_mult = cfg.r_mult if with_trend else cfg.counter_r_mult
            stop = level * (1 - cfg.stop_buffer * side)
            dist = side * (c[t] - stop)
            if dist <= 0 or dist / c[t] < cfg.min_stop or dist > cfg.max_stop_atr * a:
                continue
            target = c[t] + side * r_mult * dist
            if cfg.room_atr is not None:
                need = (target - day_lo) if side > 0 else (day_hi - target)
                if need > cfg.room_atr * a:
                    continue
            if cfg.obstacle_check:
                levels = np.array([rh[t], sh[t]] if side > 0 else [rl[t], sl[t]])
                ahead = levels[(side * (levels - c[t]) > 0) & ~np.isnan(levels)]
                if len(ahead) and side * (target - (ahead.min() if side > 0 else ahead.max())) > 0:
                    continue
            pending = {"day": day[t], "side": side, "stop": stop, "r_mult": r_mult, "with_trend": with_trend,
                       "setup": setup, "risk_mult": 1.0 if with_trend else cfg.counter_risk_mult}
            break

    n = t + 1
    eqs = pd.Series(curve[:n], index=df.index[:n], name="equity")
    tr = pd.DataFrame(trades)
    return ThreeRResult(eqs, tr, _stats(eqs, tr, df, cfg))


def _stats(eq: pd.Series, tr: pd.DataFrame, df: pd.DataFrame, cfg: ThreeRConfig) -> dict:
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    out = {"total_return": float(eq.iloc[-1] / cfg.initial_equity - 1), "max_drawdown": max_drawdown(eq),
           "final_equity": float(eq.iloc[-1]), "trades": len(tr), "trades_per_week": len(tr) / days * 7,
           "buy_and_hold_return": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)}
    if len(tr):
        out.update(win_rate=float((tr["R"] > 0).mean()), avg_R=float(tr["R"].mean()),
                   target_rate=float((tr["reason"] == "target").mean()),
                   day_end_rate=float((tr["reason"] == "day_end").mean()),
                   avg_return=float(tr["return"].mean()))
    return out
