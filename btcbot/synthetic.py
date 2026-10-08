"""Random-walk OHLCV data for testing the code offline. It has no edge by design."""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk(n: int = 24 * 365, start_price: float = 10_000_000.0, vol: float = 0.008,
                drift: float = 0.0, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wiggle = np.abs(rng.normal(0, vol / 2, n))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC", name="open_time")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(1, 50, n)}, index=idx)


def mean_reverting(n: int = 24 * 365, start_price: float = 100.0, vol: float = 0.01, pull: float = 0.05,
                   seed: int = 0, start: str = "2024-06-01") -> pd.DataFrame:
    """Log price pulled back toward a fixed level each bar (an Ornstein-Uhlenbeck walk)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] * (1 - pull) + rng.normal(0, vol)
    close = start_price * np.exp(x)
    open_ = np.concatenate([[start_price], close[:-1]])
    wiggle = np.abs(rng.normal(0, vol / 3, n))
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC", name="open_time")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wiggle),
                         "low": np.minimum(open_, close) * (1 - wiggle), "close": close,
                         "volume": rng.uniform(1e3, 1e5, n)}, index=idx)
