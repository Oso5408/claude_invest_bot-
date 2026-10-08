"""Strategy B: Donchian breakout with an ATR trailing stop, up to 2x leverage.

Entries (decided at the bar close, filled at the next open):
- Long: close above the highest high of the previous 20 bars, with volume above
  1.2x the average volume of the previous 20 bars.
- Short: close below the lowest low of the previous 10 bars.

Exit: a trailing stop only. It starts at entry -/+ 2 x ATR(14) and then follows
the best price reached since entry at the same distance, never moving back.
The stop is checked inside each bar; a gap through it fills at the open.

Sizing: RiskController seeded with a 35% win rate, 2.5 payoff ratio and 2% average
loss (about 2 x the median hourly ATR on ADA) as if 30 trades had already
happened, so quarter Kelly works from the first trade. Real trades update it.
The 2x cap, fees, leverage fee and loss cut are the same as in meanrev.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcbot import risk
from btcbot.data import load_csv, resample
from btcbot.meanrev import AccountConfig, MarginAccount, Result, RiskController, _round_qty, _stats, htf_ema
from btcbot.strategy import StrategyConfig


@dataclass
class BreakoutConfig:
    long_lookback: int = 20
    short_lookback: int = 10
    volume_lookback: int = 20
    volume_mult: float = 1.2
    atr_period: int = 14
    atr_mult: float = 2.0
    allow_short: bool = True
    max_leverage: float = risk.MAX_LEVERAGE  # lower it for spot-only venues (e.g. 1.0); can never exceed risk.MAX_LEVERAGE
    short_needs_volume: bool = False
    trend_timeframe: str | None = None  # e.g. "4h": longs only above its EMA, shorts only below
    trend_ema: int = 50
    kelly_fraction: float = 0.25
    prior_win_rate: float = 0.35
    prior_payoff: float = 2.5
    prior_avg_loss: float = 0.02
    prior_weight: int = 30
    # Fear & Greed filter (0 = extreme fear, 100 = extreme greed). A new entry is only
    # taken when the index is inside the band for its side. Defaults let everything through.
    fng_long_min: float = 0
    fng_long_max: float = 100
    fng_short_min: float = 0
    fng_short_max: float = 100


def align_fng(index: pd.DatetimeIndex, fng: pd.Series) -> pd.Series:
    """Fear & Greed value known at each bar. The value stamped day D (00:00 UTC) is only
    used from D + 1 day, so a bar never sees a value that may not have been published yet."""
    known = fng.copy()
    known.index = known.index + pd.Timedelta(days=1)
    return known.reindex(index, method="ffill")


def compute_features(df: pd.DataFrame, cfg: BreakoutConfig, fng: pd.Series | None = None) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift()
    tr = pd.concat([out["high"] - out["low"], (out["high"] - prev_close).abs(),
                    (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / cfg.atr_period, adjust=False, min_periods=cfg.atr_period).mean()
    out["donchian_high"] = out["high"].shift().rolling(cfg.long_lookback).max()
    out["donchian_low"] = out["low"].shift().rolling(cfg.short_lookback).min()
    out["volume_ma"] = out["volume"].shift().rolling(cfg.volume_lookback).mean()
    loud = out["volume"] > cfg.volume_mult * out["volume_ma"]
    out["long_signal"] = (out["close"] > out["donchian_high"]) & loud
    out["short_signal"] = (out["close"] < out["donchian_low"]) & cfg.allow_short
    if cfg.short_needs_volume:
        out["short_signal"] &= loud
    if cfg.trend_timeframe:
        out["trend_ema"] = htf_ema(out, cfg.trend_timeframe, cfg.trend_ema)
        out["long_signal"] &= out["close"] > out["trend_ema"]
        out["short_signal"] &= out["close"] < out["trend_ema"]
    if fng is not None:
        # unknown index (before the history starts) blocks nothing
        out["fng"] = align_fng(out.index, fng)
        v = out["fng"]
        out["long_signal"] &= v.isna() | v.between(cfg.fng_long_min, cfg.fng_long_max)
        out["short_signal"] &= v.isna() | v.between(cfg.fng_short_min, cfg.fng_short_max)
    return out


def mtf_config(**overrides) -> BreakoutConfig:
    """Multi-timeframe volume breakout: 4h EMA50 sets the side, 15m 20-bar breakout
    with volume above 1.5x its 20-bar average triggers, 2 x ATR trailing stop exits."""
    base = dict(long_lookback=20, short_lookback=20, volume_lookback=20, volume_mult=1.5,
                short_needs_volume=True, trend_timeframe="4h", trend_ema=50)
    base.update(overrides)
    return BreakoutConfig(**base)


def run(df: pd.DataFrame, cfg: BreakoutConfig | None = None, acct_cfg: AccountConfig | None = None,
        fng: pd.Series | None = None) -> Result:
    cfg = cfg or BreakoutConfig()
    acct_cfg = acct_cfg or AccountConfig()
    f = compute_features(df, cfg, fng)
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = f["atr"].to_numpy()
    long_sig, short_sig = f["long_signal"].to_numpy(), f["short_signal"].to_numpy()
    jst_hour = f.index.tz_convert("Asia/Tokyo").hour

    rc = RiskController(StrategyConfig(kelly_fraction=cfg.kelly_fraction), cold_start_trades=0, cold_start_leverage=0.0)
    rc.seed_prior(cfg.prior_win_rate, cfg.prior_payoff, cfg.prior_avg_loss, cfg.prior_weight)
    acct = MarginAccount(acct_cfg.initial_jpy)
    side, sig_entry, sig_days, entry_time, sig_qty = 0, 0.0, 0, None, 0.0
    stop, best, entry_atr = 0.0, 0.0, 0.0
    pending: tuple[int, float, float] | None = None  # (direction, qty, atr at signal)
    equity = np.empty(len(f))
    trades, loss_cuts, cap_hits = [], 0, 0

    def fill_price(price: float, delta: float) -> float:
        return price * (1 + acct_cfg.slippage * np.sign(delta))

    def close_trade(t: int, px: float, reason: str) -> None:
        nonlocal side
        acct.set_position(0.0, px, acct_cfg.fee_rate)
        ret = side * (px / sig_entry - 1) - acct_cfg.leverage_fee_per_day * sig_days - 2 * acct_cfg.fee_rate
        rc.update(ret)
        trades.append({"entry_time": entry_time, "exit_time": f.index[t], "side": side, "entry_px": sig_entry,
                       "exit_px": px, "qty": sig_qty, "return": ret, "reason": reason, "lev_after": rc.leverage()})
        side = 0

    for t in range(len(f)):
        # 1. fill the entry decided at the previous close
        if pending is not None:
            direction, target, entry_atr = pending
            pending = None
            px = fill_price(o[t], direction)
            capped = risk.cap_quantity(target, px, acct.equity(px), acct_cfg.fee_rate)
            cap_hits += capped != target
            acct.set_position(_round_qty(capped, acct_cfg), px, acct_cfg.fee_rate)
            side, sig_entry, sig_days, entry_time, sig_qty = direction, px, 0, f.index[t], acct.qty
            best = px
            stop = px - side * cfg.atr_mult * entry_atr

        # 2. leverage fee at the 06:00 JST rollover
        if side and jst_hour[t] == acct_cfg.rollover_hour_jst:
            sig_days += 1
            if acct.qty:
                acct.charge(acct_cfg.leverage_fee_per_day * abs(acct.qty) * o[t])

        # 3. trailing stop, then the exchange loss cut, both inside the bar
        if side:
            hit = l[t] <= stop if side > 0 else h[t] >= stop
            if hit:
                px = min(o[t], stop) if side > 0 else max(o[t], stop)
                close_trade(t, fill_price(px, -side), "trailing_stop")
        if side and acct.qty:
            worst = l[t] if acct.qty > 0 else h[t]
            if risk.maintenance_ratio(acct.equity(worst), acct.qty, worst) <= risk.LOSS_CUT_RATIO:
                lc = risk.loss_cut_price(acct.collateral, acct.qty, acct.entry)
                px = min(o[t], lc) if acct.qty > 0 else max(o[t], lc)
                close_trade(t, fill_price(px, -acct.qty), "loss_cut")
                loss_cuts += 1

        equity[t] = acct.equity(c[t])
        if equity[t] <= 0:
            equity[t:] = equity[t]
            break
        if t == len(f) - 1 or np.isnan(atr[t]):
            continue

        # 4. at the close: move the stop, or look for a new breakout
        if side:
            best = max(best, h[t]) if side > 0 else min(best, l[t])
            trail = best - side * cfg.atr_mult * atr[t]
            stop = max(stop, trail) if side > 0 else min(stop, trail)
            if abs(acct.qty) * c[t] > risk.MAX_LEVERAGE * equity[t]:
                # price moved against us past 2x: trim at the next open (same as meanrev)
                px = fill_price(o[t + 1], -acct.qty)
                acct.set_position(_round_qty(risk.cap_quantity(acct.qty, px, acct.equity(px), acct_cfg.fee_rate),
                                             acct_cfg), px, acct_cfg.fee_rate)
        elif long_sig[t] or short_sig[t]:
            direction = 1 if long_sig[t] else -1
            lev = min(rc.leverage(), cfg.max_leverage, risk.MAX_LEVERAGE)
            pending = (direction, direction * lev * equity[t] / c[t], atr[t])

    eq = pd.Series(equity[: t + 1], index=f.index[: t + 1], name="equity")
    tr = pd.DataFrame(trades)
    return Result(eq, tr, _stats(eq, tr, df, acct, acct_cfg, loss_cuts, cap_hits))


def main(argv: list[str] | None = None) -> None:
    from btcbot.backtest import format_stats

    p = argparse.ArgumentParser(description="Backtest the breakout strategy with up to 2x leverage")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--synthetic", action="store_true", help="trending random data, for checking the code")
    p.add_argument("--jpy", type=float, default=AccountConfig.initial_jpy)
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--max-leverage", type=float, default=risk.MAX_LEVERAGE, help="e.g. 1 for spot-only (Alpaca)")
    p.add_argument("--timeframe", default=None, help="resample the CSV first, e.g. 4h")
    p.add_argument("--mtf", action="store_true", help="multi-timeframe preset: 4h EMA50 + 1.5x volume breakout both ways")
    p.add_argument("--trades-out")
    p.add_argument("--fng-csv", help="data/fear_greed.csv from `python -m btcbot.data --fng`")
    p.add_argument("--fng-long", type=float, nargs=2, metavar=("MIN", "MAX"), default=(0, 100),
                   help="only go long when the Fear & Greed Index is in this band")
    p.add_argument("--fng-short", type=float, nargs=2, metavar=("MIN", "MAX"), default=(0, 100),
                   help="only go short when the Fear & Greed Index is in this band")
    a = p.parse_args(argv)

    if a.csv:
        df = load_csv(a.csv)
    else:
        from btcbot.synthetic import random_walk
        df = random_walk(start_price=100, vol=0.01, drift=0.0003)
    if a.timeframe:
        df = resample(df, a.timeframe)
    opts = dict(allow_short=not a.no_short, max_leverage=a.max_leverage,
                fng_long_min=a.fng_long[0], fng_long_max=a.fng_long[1],
                fng_short_min=a.fng_short[0], fng_short_max=a.fng_short[1])
    cfg = mtf_config(**opts) if a.mtf else BreakoutConfig(**opts)
    fng = None
    if a.fng_csv:
        from btcbot.data import load_fng
        fng = load_fng(a.fng_csv)
    res = run(df, cfg, AccountConfig(initial_jpy=a.jpy), fng)
    print(format_stats(res.stats))
    if a.trades_out:
        res.trades.to_csv(a.trades_out, index=False)


if __name__ == "__main__":
    main()
