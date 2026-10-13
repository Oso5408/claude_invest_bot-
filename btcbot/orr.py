"""Opening Range Reversal (Doug's ORR video) on intraday bars. Backtest only.

Daily context from finished UTC days only (a value is used from the day after it closed):
- Trend bias: last close above its 50-day SMA = only longs (buy the opening dip),
  below = only shorts (sell the opening rally).
- Daily range: ATR(14) of the UTC daily candles.

Crypto has no opening bell, so a "session" starts at a chosen time:
- "utc": 00:00 UTC (the daily candle open), session lasts 24 hours.
- "us": 09:30 New York (the US stock open, daylight saving handled), session ends at 16:00 New York.

Within a session (long side, shorts mirrored):
- The opening move = the highest price before the session low (the open counts) minus the
  session low. It must be at least move_atr x daily ATR (video: 20%).
- Entry: a later bar trades above the high of the bar that made the session low (a buy-stop at
  that high; taker fee + slippage). If that bar also trades below the stop, the order inside
  the bar is unknown: ambiguous="worst" assumes filled then stopped, "skip" assumes no trade.
  The truth lies between; report both.
- Stop: the session low (less stop_buffer), market exit (taker + slippage).
- Target: retrace x the opening move above the low (video: 50%, or 100% for the full move),
  as a resting limit (maker fee). Skipped when the entry is already at or above the target.
- Anything still open at the end of the session closes at the market.
- Entries only in the first entry_hours of the session.

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
class ORRConfig:
    session: str = "utc"  # "utc" or "us"
    entry_hours: float | None = None  # None = 8h for "utc", whole session for "us"
    sma_days: int = 50
    atr_days: int = 14
    use_bias: bool = True  # False: trade both sides every day
    against_bias: bool = False  # True: only trade against the bias (control test)
    move_atr: float = 0.2
    retrace: float = 0.5
    stop_buffer: float = 0.0
    min_stop: float = 0.0  # skip when entry-to-stop is under this fraction of price
    max_trades_per_session: int = 1
    ambiguous: str = "worst"  # entry bar that also trades through the stop: "worst" = stopped, "skip" = no trade
    risk_frac: float = 0.01
    initial_equity: float = 200.0
    taker_fee: float = TAKER_FEE
    maker_fee: float = MAKER_FEE
    slippage: float = 0.0005


@dataclass
class ORRResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def sessions(index: pd.DatetimeIndex, cfg: ORRConfig) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Per bar: session start, whether entries are allowed, whether the bar is inside the session."""
    if cfg.session == "utc":
        start = index.floor("D")
        length = pd.Timedelta(hours=24)
        entry = pd.Timedelta(hours=cfg.entry_hours if cfg.entry_hours is not None else 8)
    elif cfg.session == "us":
        ny = index.tz_convert("America/New_York")
        local_day = (ny - pd.Timedelta(hours=9, minutes=30)).tz_localize(None).normalize()
        start = (local_day + pd.Timedelta(hours=9, minutes=30)).tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
        length = pd.Timedelta(hours=6, minutes=30)
        entry = pd.Timedelta(hours=cfg.entry_hours) if cfg.entry_hours is not None else length
    else:
        raise ValueError(cfg.session)
    age = index - start
    return start, np.asarray(age < entry), np.asarray(age < length)


def daily_context(df: pd.DataFrame, cfg: ORRConfig, start: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Last finished UTC day's close, SMA and ATR as known at each bar's session start."""
    day = df.index.floor("D")
    d = df[["high", "low", "close"]].groupby(day).agg({"high": "max", "low": "min", "close": "last"})
    prev = d["close"].shift()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
    cols = {"close": d["close"], "sma": d["close"].rolling(cfg.sma_days).mean(),
            "atr": tr.ewm(alpha=1 / cfg.atr_days, adjust=False, min_periods=cfg.atr_days).mean()}
    # plain arrays: a Series would re-align to the shifted index and leak the same day's values
    ctx = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=d.index + pd.Timedelta(days=1))
    # a session starting at S may use UTC days that finished by S: the row for floor(S)
    ctx = ctx.reindex(start.floor("D"))
    return ctx["close"].to_numpy(), ctx["sma"].to_numpy(), ctx["atr"].to_numpy()


def run(df: pd.DataFrame, cfg: ORRConfig | None = None, funding: pd.Series | None = None) -> ORRResult:
    cfg = cfg or ORRConfig()
    starts, can_enter, in_sess = sessions(df.index, cfg)
    pclose, sma, atr = daily_context(df, cfg, starts)
    start = starts.asi8
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    fund = funding_per_bar(df.index, funding) if funding is not None else np.zeros(len(df))
    n = len(df)
    last_of_sess = np.r_[(start[1:] != start[:-1]) | ~in_sess[1:], True] & in_sess

    eq = cfg.initial_equity
    curve = np.empty(n)
    trades = []
    pos = None
    trades_today = 0
    # session state; *_ref = high of the bar that made the low (low of the bar that made the high)
    s_lo = s_hi = run_hi = run_lo = lo_ref = hi_ref = peak_before_lo = trough_before_hi = np.nan

    def close(t: int, px: float, fee: float, reason: str) -> None:
        nonlocal eq, pos
        side, qty = pos["side"], pos["qty"]
        eq += side * qty * (px - pos["entry"]) - qty * px * fee
        pnl = eq - pos["eq_before"]
        trades.append({"entry_time": pos["time"], "exit_time": df.index[t], "side": side, "entry": pos["entry"],
                       "stop": pos["stop"], "target": pos["target"], "exit": px, "reason": reason,
                       "move_atr": pos["move_atr"], "stop_pct": pos["risk"] / pos["entry"],
                       # in units of the planned risk (risk_frac x equity); the leverage cap can make it smaller
                       "R": side * qty * (px - pos["entry"]) / (cfg.risk_frac * pos["eq_before"]),  # before fees and funding
                       "net_R": pnl / (cfg.risk_frac * pos["eq_before"]),  # after fees, slippage and funding
                       "pnl": pnl, "return": pnl / pos["eq_before"]})
        pos = None

    for t in range(n):
        new_sess = t == 0 or start[t] != start[t - 1]
        if new_sess:
            trades_today = 0
            s_lo = s_hi = run_hi = run_lo = lo_ref = hi_ref = peak_before_lo = trough_before_hi = np.nan

        # 1. manage an open position (stop first if both are inside one bar)
        if pos is not None:
            f_cost = pos["side"] * fund[t] * pos["qty"] * o[t]
            eq -= f_cost
            side = pos["side"]
            if (l[t] <= pos["stop"]) if side > 0 else (h[t] >= pos["stop"]):
                px = min(o[t], pos["stop"]) if side > 0 else max(o[t], pos["stop"])
                close(t, px * (1 - cfg.slippage * side), cfg.taker_fee, "stop")
            elif (h[t] >= pos["target"]) if side > 0 else (l[t] <= pos["target"]):
                px = max(o[t], pos["target"]) if side > 0 else min(o[t], pos["target"])
                close(t, px, cfg.maker_fee, "target")

        # 2. entry: a buy-stop at the high of the low bar (sell-stop at the low of the high bar)
        a = atr[t]
        if (pos is None and in_sess[t] and can_enter[t] and not new_sess and trades_today < cfg.max_trades_per_session
                and not np.isnan(a) and not np.isnan(sma[t])):
            bias = 1 if pclose[t] > sma[t] else -1
            sides = [bias] if cfg.use_bias else [1, -1]
            if cfg.against_bias:
                sides = [-bias]
            for side in sides:
                if side > 0:
                    level, ref, move = s_lo, lo_ref, peak_before_lo - s_lo
                    triggered = h[t] > ref and (cfg.ambiguous == "worst" or l[t] >= s_lo)
                else:
                    level, ref, move = s_hi, hi_ref, s_hi - trough_before_hi
                    triggered = l[t] < ref and (cfg.ambiguous == "worst" or h[t] <= s_hi)
                if not triggered or move < cfg.move_atr * a:
                    continue
                fill = (max(o[t], ref) if side > 0 else min(o[t], ref)) * (1 + cfg.slippage * side)
                target = level + side * cfg.retrace * move
                stop = level * (1 - cfg.stop_buffer * side)
                risk_dist = side * (fill - stop)
                if side * (target - fill) <= 0 or risk_dist <= 0 or risk_dist / fill < cfg.min_stop:
                    continue
                qty = abs(risk.cap_quantity(cfg.risk_frac * eq / risk_dist, fill, eq, cfg.taker_fee))
                if qty <= 0:
                    continue
                eq_before = eq
                eq -= qty * fill * cfg.taker_fee
                pos = {"side": side, "qty": qty, "entry": fill, "stop": stop, "target": target, "risk": risk_dist,
                       "time": df.index[t], "eq_before": eq_before, "move_atr": move / a}
                trades_today += 1
                # rest of the entry bar. If it also traded through the stop the order is unknown:
                # assume the worst (filled, then stopped). Otherwise the target may fill too.
                if (l[t] <= stop) if side > 0 else (h[t] >= stop):
                    close(t, stop * (1 - cfg.slippage * side), cfg.taker_fee, "stop")
                elif (h[t] >= target) if side > 0 else (l[t] <= target):
                    close(t, target, cfg.maker_fee, "target")
                break

        # 3. session end
        if pos is not None and last_of_sess[t]:
            close(t, c[t] * (1 - cfg.slippage * pos["side"]), cfg.taker_fee, "session_end")

        curve[t] = eq + (pos["side"] * pos["qty"] * (c[t] - pos["entry"]) if pos else 0.0)
        if curve[t] <= 0:
            curve[t:] = curve[t]
            break

        # 4. update session extremes with this bar
        if in_sess[t]:
            if np.isnan(s_lo) or l[t] < s_lo:
                peak_before_lo = o[t] if np.isnan(run_hi) else max(run_hi, o[t])
                s_lo, lo_ref = l[t], h[t]
            if np.isnan(s_hi) or h[t] > s_hi:
                trough_before_hi = o[t] if np.isnan(run_lo) else min(run_lo, o[t])
                s_hi, hi_ref = h[t], l[t]
            run_hi = h[t] if np.isnan(run_hi) else max(run_hi, h[t])
            run_lo = l[t] if np.isnan(run_lo) else min(run_lo, l[t])

    m = t + 1
    eqs = pd.Series(curve[:m], index=df.index[:m], name="equity")
    tr = pd.DataFrame(trades)
    return ORRResult(eqs, tr, _stats(eqs, tr, df, cfg))


def _stats(eq: pd.Series, tr: pd.DataFrame, df: pd.DataFrame, cfg: ORRConfig) -> dict:
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    out = {"total_return": float(eq.iloc[-1] / cfg.initial_equity - 1), "max_drawdown": max_drawdown(eq),
           "final_equity": float(eq.iloc[-1]), "trades": len(tr), "trades_per_week": len(tr) / days * 7}
    if len(tr):
        out.update(win_rate=float((tr["pnl"] > 0).mean()), gross_win_rate=float((tr["R"] > 0).mean()),
                   avg_R=float(tr["R"].mean()), avg_net_R=float(tr["net_R"].mean()),
                   target_rate=float((tr["reason"] == "target").mean()),
                   median_stop_pct=float(tr["stop_pct"].median()))
    return out
