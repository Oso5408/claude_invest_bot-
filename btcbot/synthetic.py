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
