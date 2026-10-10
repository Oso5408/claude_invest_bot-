"""Hyperliquid public market data: perp candles and hourly funding rates. No account or key.

API docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint
- candleSnapshot only serves the most recent 5000 candles per interval (4h = about 833 days).
- fundingHistory returns at most 500 rows per call, so we page forward by time.
Prices are in USD (USDC). Base-tier perp fees: taker 0.045%, maker 0.015%.

    python -m btcbot.hyperliquid --coin ADA                # data/HL_ADA_4h.csv + data/HL_ADA_funding.csv
    python -m btcbot.hyperliquid --coin ADA --interval 1h
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

MAINNET = "https://api.hyperliquid.xyz/info"
TESTNET = "https://api.hyperliquid-testnet.xyz/info"
TAKER_FEE = 0.00045
MAKER_FEE = 0.00015
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _post(url: str, body: dict, session: requests.Session | None = None):
    resp = (session or requests).post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_candles(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"open": df["o"].astype(float), "high": df["h"].astype(float), "low": df["l"].astype(float),
                        "close": df["c"].astype(float), "volume": df["v"].astype(float)})
    out.index = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    out.index.name = "open_time"
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_candles(coin: str, interval: str = "4h", start_ms: int = 0, end_ms: int | None = None,
                  url: str = MAINNET) -> pd.DataFrame:
    end_ms = end_ms or int(time.time() * 1000)
    step = INTERVAL_MS[interval] * 5000
    frames, t = [], max(start_ms, end_ms - step)  # older candles are not served anyway
    with requests.Session() as s:
        while t < end_ms:
            rows = _post(url, {"type": "candleSnapshot",
                               "req": {"coin": coin, "interval": interval, "startTime": t, "endTime": min(t + step, end_ms)}}, s)
            frames.append(parse_candles(rows))
            t += step
            time.sleep(0.2)
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def parse_funding(rows: list[dict]) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC", name="time"), name="funding")
    idx = pd.to_datetime([int(r["time"]) for r in rows], unit="ms", utc=True)
    s = pd.Series([float(r["fundingRate"]) for r in rows], index=idx, name="funding")
    s.index.name = "time"
    return s[~s.index.duplicated()].sort_index()


def fetch_funding(coin: str, start_ms: int, end_ms: int | None = None, url: str = MAINNET) -> pd.Series:
    end_ms = end_ms or int(time.time() * 1000)
    parts, t = [], start_ms
    with requests.Session() as s:
        while t < end_ms:
            rows = _post(url, {"type": "fundingHistory", "coin": coin, "startTime": t, "endTime": end_ms}, s)
            if not rows:
                break
            part = parse_funding(rows)
            parts.append(part)
            nxt = int(rows[-1]["time"]) + 1
            if nxt <= t:
                break
            t = nxt
            time.sleep(0.2)
    if not parts:
        return parse_funding([])
    out = pd.concat(parts)
    return out[~out.index.duplicated()].sort_index()


def account_config(initial_usd: float = 200.0):
    """Hyperliquid perp costs for the backtest: taker fee, no daily fee (funding is passed separately),
    whole-ADA order sizes. Amounts are in USD even though the field is called initial_jpy."""
    from btcbot.meanrev import AccountConfig
    return AccountConfig(initial_jpy=initial_usd, fee_rate=TAKER_FEE, slippage=0.0005, leverage_fee_per_day=0.0,
                         min_order=1, size_step=1)


def load_funding(path: str | Path) -> pd.Series:
    raw = pd.read_csv(path)
    idx = pd.to_datetime(raw["time"], utc=True, format="ISO8601")  # some stamps carry milliseconds, some don't
    return pd.Series(raw["funding"].to_numpy(dtype=float), index=pd.DatetimeIndex(idx, name="time"),
                     name="funding").sort_index()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download Hyperliquid perp candles and funding")
    p.add_argument("--coin", default="ADA")
    p.add_argument("--interval", default="4h", choices=sorted(INTERVAL_MS))
    p.add_argument("--testnet", action="store_true")
    p.add_argument("--out-dir", default="data")
    a = p.parse_args(argv)
    url = TESTNET if a.testnet else MAINNET
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    candles = fetch_candles(a.coin, a.interval, url=url)
    cpath = out / f"HL_{a.coin}_{a.interval}.csv"
    candles.to_csv(cpath)
    print(f"Saved {len(candles)} candles ({candles.index[0]} to {candles.index[-1]}) to {cpath}")

    start = int(candles.index[0].timestamp() * 1000)
    funding = fetch_funding(a.coin, start, url=url)
    fpath = out / f"HL_{a.coin}_funding.csv"
    funding.to_csv(fpath)
    print(f"Saved {len(funding)} hourly funding rates to {fpath}")


if __name__ == "__main__":
    main()
