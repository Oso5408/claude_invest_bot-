"""Mean reversion on GMO Coin crypto FX (default ADA_JPY) with quarter Kelly sizing.

Strategy:
- z = (close - rolling mean) / rolling std over `window` bars.
- z below -entry_z: go long. z above +entry_z: go short.
- Exit when z returns to +/- exit_z, when |z| passes stop_z (the move kept going),
  or after max_hold bars.
- Skip new entries when volatility is in its top percentile (crashes, news spikes).

Sizing: a Beta posterior on the win rate plus the running average win and loss
give the Kelly leverage f* = (p - (1 - p) / b) / avg_loss. We use a quarter of it,
and risk.py caps every order at 2x equity no matter what.

Account rules (GMO 暗号資産FX): no trading fee, 0.04% of position value per day
for positions held at the 06:00 JST rollover, loss cut at 75% maintenance ratio.
Fills happen at the next bar's open plus slippage. No live orders anywhere.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from btcbot import risk
from btcbot.backtest import max_drawdown
from btcbot.data import load_csv
from btcbot.strategy import BayesKelly, StrategyConfig


@dataclass
class MeanRevConfig:
    window: int = 48
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 4.0
    max_hold: int = 72
    vol_window: int = 24
    vol_rank_window: int = 24 * 30
    vol_max_pct: float = 0.9
    kelly_fraction: float = 0.25
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    payoff_min_trades: int = 10


@dataclass
class AccountConfig:
    initial_jpy: float = 30_000.0  # about 200 USD
    fee_rate: float = 0.0  # GMO crypto FX charges no trading fee
    slippage: float = 0.0005  # spread + slippage per fill; ADA is thinner than BTC
    leverage_fee_per_day: float = 0.0004
    rollover_hour_jst: int = 6
    min_order: float = 10.0  # ADA; check with `python -m btcbot.data --rules ADA_JPY`
    size_step: float = 1.0


class LeverageKelly(BayesKelly):
    """Kelly for a position whose wins and losses are small fractions of its value."""

    def leverage(self) -> float:
        if len(self.wins) + len(self.losses) < self.cfg.payoff_min_trades or not self.losses or not self.wins:
            return 0.0
        avg_loss = float(np.mean(self.losses))
        f = (self.win_prob - (1 - self.win_prob) / self.payoff) / avg_loss
        return float(min(max(f * self.cfg.kelly_fraction, 0.0), risk.MAX_LEVERAGE))


@dataclass
class MarginAccount:
    collateral: float
    qty: float = 0.0
    entry: float = 0.0
    fees: float = 0.0
    leverage_fees: float = 0.0

    def equity(self, price: float) -> float:
        return self.collateral + self.qty * (price - self.entry)

    def set_position(self, new_qty: float, price: float, fee_rate: float) -> None:
        """Move to new_qty at price. The 2x check runs on every fill."""
        if new_qty == self.qty:
            return
        fee = abs(new_qty - self.qty) * price * fee_rate
        self.collateral += self.qty * (price - self.entry) - fee
        self.fees += fee
        self.qty, self.entry = new_qty, price
        risk.check_leverage(self.qty, price, self.equity(price))

    def charge(self, amount: float) -> None:
        self.collateral -= amount
        self.leverage_fees += amount


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict = field(default_factory=dict)


def compute_features(df: pd.DataFrame, cfg: MeanRevConfig) -> pd.DataFrame:
    out = df.copy()
    mean = out["close"].rolling(cfg.window).mean()
    std = out["close"].rolling(cfg.window).std()
    out["z"] = (out["close"] - mean) / std
    vol = np.log(out["close"]).diff().rolling(cfg.vol_window).std()
    vol_pct = vol.rolling(cfg.vol_rank_window, min_periods=cfg.vol_window * 2).rank(pct=True)
    out["vol_ok"] = vol_pct <= cfg.vol_max_pct
    return out


def _round_qty(qty: float, acct: AccountConfig) -> float:
    q = np.floor(abs(qty) / acct.size_step) * acct.size_step
    return float(np.sign(qty) * q) if q >= acct.min_order else 0.0


def run(df: pd.DataFrame, cfg: MeanRevConfig | None = None, acct_cfg: AccountConfig | None = None) -> Result:
    cfg = cfg or MeanRevConfig()
    acct_cfg = acct_cfg or AccountConfig()
    f = compute_features(df, cfg)
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    z, vol_ok = f["z"].to_numpy(), f["vol_ok"].to_numpy()
    jst_hour = f.index.tz_convert("Asia/Tokyo").hour

    kelly = LeverageKelly(StrategyConfig(prior_alpha=cfg.prior_alpha, prior_beta=cfg.prior_beta,
                                         payoff_min_trades=cfg.payoff_min_trades, kelly_fraction=cfg.kelly_fraction))
    acct = MarginAccount(acct_cfg.initial_jpy)
    side, sig_entry, sig_bars, sig_days, entry_time, sig_qty = 0, 0.0, 0, 0, None, 0.0
    pending: tuple[str, float] | None = None
    equity = np.empty(len(f))
    trades, loss_cuts, cap_hits = [], 0, 0

    def close_signal(t: int, px: float, reason: str) -> None:
        nonlocal side
        ret = side * (px / sig_entry - 1) - acct_cfg.leverage_fee_per_day * sig_days
        kelly.update(ret)
        trades.append({"entry_time": entry_time, "exit_time": f.index[t], "side": side, "entry_px": sig_entry,
                       "exit_px": px, "qty": sig_qty, "return": ret, "reason": reason, "kelly_lev_after": kelly.leverage()})
        side = 0

    def fill_price(price: float, delta: float) -> float:
        return price * (1 + acct_cfg.slippage * np.sign(delta))

    for t in range(len(f)):
        # 1. fill yesterday-close decision at this open
        if pending is not None:
            action, target = pending
            pending = None
            if action == "exit":
                px = fill_price(o[t], -acct.qty if acct.qty else -side)
                acct.set_position(0.0, px, acct_cfg.fee_rate)
                close_signal(t, px, reason)
            else:
                px = fill_price(o[t], target if target else (1 if action == "long" else -1))
                capped = risk.cap_quantity(target, px, acct.equity(px))
                cap_hits += capped != target
                acct.set_position(_round_qty(capped, acct_cfg), px, acct_cfg.fee_rate)
                if action in ("long", "short"):
                    side, sig_entry, sig_bars, sig_days, entry_time = (1 if action == "long" else -1), px, 0, 0, f.index[t]
                    sig_qty = acct.qty

        # 2. leverage fee for anything held through the 06:00 JST rollover
        if side and jst_hour[t] == acct_cfg.rollover_hour_jst:
            sig_days += 1
            if acct.qty:
                acct.charge(acct_cfg.leverage_fee_per_day * abs(acct.qty) * o[t])

        # 3. intrabar loss cut at 75% maintenance ratio
        if acct.qty:
            worst = l[t] if acct.qty > 0 else h[t]
            if risk.maintenance_ratio(acct.equity(worst), acct.qty, worst) <= risk.LOSS_CUT_RATIO:
                lc = risk.loss_cut_price(acct.collateral, acct.qty, acct.entry)
                px = min(o[t], lc) if acct.qty > 0 else max(o[t], lc)  # gap through the level fills at the open
                px = fill_price(px, -acct.qty)
                acct.set_position(0.0, px, acct_cfg.fee_rate)
                close_signal(t, px, "loss_cut")
                loss_cuts += 1

        equity[t] = acct.equity(c[t])
        if equity[t] <= 0:
            equity[t:] = equity[t]
            break
        if t == len(f) - 1 or np.isnan(z[t]):
            continue

        # 4. decide at the close
        if side:
            sig_bars += 1
            back = (side > 0 and z[t] >= -cfg.exit_z) or (side < 0 and z[t] <= cfg.exit_z)
            stop = abs(z[t]) >= cfg.stop_z and np.sign(z[t]) == -side
            if back or stop or sig_bars >= cfg.max_hold:
                reason = "revert" if back else "stop" if stop else "timeout"
                pending = ("exit", 0.0)
            elif abs(acct.qty) * c[t] > risk.MAX_LEVERAGE * equity[t]:
                pending = ("trim", acct.qty)  # price moved against us: cut back to 2x
        elif vol_ok[t] and abs(z[t]) >= cfg.entry_z:
            direction = -1 if z[t] > 0 else 1
            qty = direction * kelly.leverage() * equity[t] / c[t]
            pending = ("long" if direction > 0 else "short", qty)

    eq = pd.Series(equity[: t + 1], index=f.index[: t + 1], name="equity")
    tr = pd.DataFrame(trades)
    return Result(eq, tr, _stats(eq, tr, df, acct, acct_cfg, loss_cuts, cap_hits))


def _stats(eq, tr, df, acct, acct_cfg, loss_cuts, cap_hits) -> dict:
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    total = eq.iloc[-1] / acct_cfg.initial_jpy - 1
    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(len(eq) / years)) if r.std() > 0 else 0.0
    return {
        "start": str(eq.index[0]),
        "end": str(eq.index[-1]),
        "final_equity_jpy": round(float(eq.iloc[-1])),
        "total_return": float(total),
        "cagr": float(max(1 + total, 0.0) ** (1 / years) - 1),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe,
        "signal_trades": len(tr),
        "win_rate": float((tr["return"] > 0).mean()) if len(tr) else float("nan"),
        "loss_cuts": loss_cuts,
        "orders_capped_at_2x": int(cap_hits),
        "trading_fees_jpy": round(acct.fees),
        "leverage_fees_jpy": round(acct.leverage_fees),
        "buy_and_hold_return": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1),
    }


def main(argv: list[str] | None = None) -> None:
    from btcbot.backtest import format_stats

    p = argparse.ArgumentParser(description="Backtest mean reversion with up to 2x leverage")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--synthetic", action="store_true", help="mean-reverting random data, for checking the code")
    p.add_argument("--jpy", type=float, default=AccountConfig.initial_jpy)
    p.add_argument("--entry-z", type=float, default=MeanRevConfig.entry_z)
    p.add_argument("--window", type=int, default=MeanRevConfig.window)
    p.add_argument("--min-order", type=float, default=AccountConfig.min_order)
    p.add_argument("--trades-out")
    a = p.parse_args(argv)

    if a.csv:
        df = load_csv(a.csv)
    else:
        from btcbot.synthetic import mean_reverting
        df = mean_reverting()
    res = run(df, MeanRevConfig(entry_z=a.entry_z, window=a.window),
              AccountConfig(initial_jpy=a.jpy, min_order=a.min_order))
    print(format_stats(res.stats))
    if a.trades_out:
        res.trades.to_csv(a.trades_out, index=False)


if __name__ == "__main__":
    main()
