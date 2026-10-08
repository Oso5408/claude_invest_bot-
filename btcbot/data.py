"""Download BTC (JPY-priced) OHLCV klines from the GMO Coin public API.

No account or API key is needed. Endpoint docs:
https://api.coin.z.com/docs/#klines

For intervals of 1hour and shorter the API returns one calendar day per
request (date=YYYYMMDD). For 4hour and longer it returns one year per
request (date=YYYY). Days are cut at 06:00 JST by GMO, which is fine because
we de-duplicate on open time.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.coin.z.com/public/v1/klines"
DAILY_INTERVALS = {"1min", "5min", "10min", "15min", "30min", "1hour"}
YEARLY_INTERVALS = {"4hour", "8hour", "12hour", "1day", "1week", "1month"}
COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]


def parse_klines(payload: dict) -> pd.DataFrame:
    """Turn one API response into a typed DataFrame indexed by UTC open time."""
    if payload.get("status") != 0:
        raise RuntimeError(f"GMO API error: {payload.get('messages')}")
    rows = payload.get("data") or []
    if not rows:
        return pd.DataFrame(columns=COLUMNS[1:], index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    df = pd.DataFrame(rows).rename(columns={"openTime": "open_time"})
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for col in COLUMNS[1:]:
        df[col] = df[col].astype(float)
    return df.set_index("open_time")[COLUMNS[1:]]


def fetch_one(symbol: str, interval: str, date_param: str, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests
    resp = s.get(BASE_URL, params={"symbol": symbol, "interval": interval, "date": date_param}, timeout=15)
    resp.raise_for_status()
    return parse_klines(resp.json())


def date_params(interval: str, start: date, end: date) -> list[str]:
    if interval in DAILY_INTERVALS:
        days = (end - start).days + 1
        return [(start + timedelta(days=i)).strftime("%Y%m%d") for i in range(days)]
    if interval in YEARLY_INTERVALS:
        return [str(y) for y in range(start.year, end.year + 1)]
    raise ValueError(f"unknown interval: {interval}")


def fetch_range(symbol: str, interval: str, start: date, end: date, pause: float = 0.2) -> pd.DataFrame:
    """Fetch every chunk between start and end (inclusive) and merge them."""
    frames = []
    with requests.Session() as session:
        for i, param in enumerate(date_params(interval, start, end)):
            frames.append(fetch_one(symbol, interval, param, session))
            if pause:
                time.sleep(pause)  # stay well under the public rate limit
            if (i + 1) % 30 == 0:
                print(f"  fetched {i + 1} chunks (up to {param})")
    if not frames:
        return parse_klines({"status": 0, "data": []})
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Combine bars into a longer timeframe, e.g. "4h". Bars are labelled by their open time."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_time"], index_col="open_time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def fetch_rules(symbol: str) -> dict:
    """Order rules (minOrderSize, sizeStep, tickSize, fees) for one symbol."""
    resp = requests.get("https://api.coin.z.com/public/v1/symbols", timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 0:
        raise RuntimeError(f"GMO API error: {payload.get('messages')}")
    for rule in payload["data"]:
        if rule["symbol"] == symbol:
            return rule
    raise ValueError(f"{symbol} not listed by GMO Coin")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download GMO Coin klines to CSV")
    p.add_argument("--symbol", default="BTC", help="BTC, ADA = spot (priced in JPY); BTC_JPY, ADA_JPY = leverage")
    p.add_argument("--interval", default="1hour")
    p.add_argument("--start", help="YYYY-MM-DD (ADA_JPY leverage trading began 2024-05-25)")
    p.add_argument("--rules", action="store_true", help="print the symbol's order rules and exit")
    p.add_argument("--end", default=None, help="YYYY-MM-DD, default yesterday (UTC)")
    p.add_argument("--out", default=None, help="CSV path, default data/<symbol>_<interval>.csv")
    args = p.parse_args(argv)
    if args.rules:
        print(fetch_rules(args.symbol))
        return
    if not args.start:
        p.error("--start is required")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else datetime.now(timezone.utc).date() - timedelta(days=1)
    out = Path(args.out or f"data/{args.symbol}_{args.interval}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {args.symbol} {args.interval} from {start} to {end}")
    df = fetch_range(args.symbol, args.interval, start, end)
    df.to_csv(out)
    print(f"Saved {len(df)} bars to {out}")


if __name__ == "__main__":
    main()
