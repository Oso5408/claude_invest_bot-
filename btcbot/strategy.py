"""Long-only trend strategy with entry filters and Bayesian Kelly sizing.

Pieces:
- Signal: fast moving average above slow moving average means "want to be long".
- Time-window filter: new entries only during chosen JST hours. Exits are always allowed.
- Volatility filter: new entries only when recent volatility sits between two
  percentiles of its own history (skip dead markets and panics).
- Bayesian update: a Beta(alpha, beta) posterior on the signal's win rate,
  updated after every completed signal trade (taken or not), plus a running
  estimate of the average win / average loss ratio.
- Kelly sizing: f* = p - (1 - p) / b, scaled by a fraction (quarter Kelly by
  default) and capped. If f* <= 0 the bot stays flat but keeps learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    fast: int = 24
    slow: int = 96
    # JST hours (0-23) in which new entries are allowed; None = any hour
    entry_hours_jst: tuple[int, ...] | None = None
    vol_window: int = 24
    vol_rank_window: int = 24 * 30
    vol_min_pct: float = 0.2
    vol_max_pct: float = 0.9
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    prior_payoff: float = 1.0  # assumed win/loss ratio before any evidence
    payoff_min_trades: int = 5
    kelly_fraction: float = 0.25
    max_position: float = 0.5  # never put more than this share of equity in BTC


def compute_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Add signal and filter columns. Every value at row t uses data up to close t only."""
    out = df.copy()
    close = out["close"]
    out["ma_fast"] = close.rolling(cfg.fast).mean()
    out["ma_slow"] = close.rolling(cfg.slow).mean()
    out["signal"] = (out["ma_fast"] > out["ma_slow"]) & out["ma_slow"].notna()

    log_ret = np.log(close).diff()
    out["vol"] = log_ret.rolling(cfg.vol_window).std()
    out["vol_pct"] = out["vol"].rolling(cfg.vol_rank_window, min_periods=cfg.vol_window * 2).rank(pct=True)
    out["vol_ok"] = out["vol_pct"].between(cfg.vol_min_pct, cfg.vol_max_pct)

    if cfg.entry_hours_jst is None:
        out["time_ok"] = True
    else:
        # The decision is made at the bar's close, so test the close hour.
        idx = out.index if out.index.tz is not None else out.index.tz_localize("UTC")
        step = idx[1] - idx[0] if len(idx) > 1 else pd.Timedelta(hours=1)
        close_hour = (idx + step).tz_convert("Asia/Tokyo").hour
        out["time_ok"] = np.isin(close_hour, cfg.entry_hours_jst)

    out["entry_ok"] = out["signal"] & out["vol_ok"] & out["time_ok"]
    return out


@dataclass
class BayesKelly:
    """Beta posterior on win rate plus running payoff ratio, turned into a Kelly fraction."""

    cfg: StrategyConfig
    alpha: float = field(init=False)
    beta: float = field(init=False)
    wins: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.alpha = self.cfg.prior_alpha
        self.beta = self.cfg.prior_beta

    def update(self, trade_return: float) -> None:
        if trade_return > 0:
            self.alpha += 1
            self.wins.append(trade_return)
        else:
            self.beta += 1
            self.losses.append(-trade_return)

    @property
    def win_prob(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def payoff(self) -> float:
        if len(self.wins) + len(self.losses) < self.cfg.payoff_min_trades or not self.wins or not self.losses:
            return self.cfg.prior_payoff
        return float(np.mean(self.wins) / np.mean(self.losses))

    def kelly(self) -> float:
        p, b = self.win_prob, self.payoff
        return p - (1 - p) / b

    def position_size(self) -> float:
        f = self.kelly() * self.cfg.kelly_fraction
        return float(min(max(f, 0.0), self.cfg.max_position))
