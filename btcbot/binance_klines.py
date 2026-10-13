"""Download Binance USDT-M futures klines from the public data site (no account or key).
Run on your own PC (the cloud container cannot reach Binance):

    python -m btcbot.binance_klines --symbol ADAUSDT --interval 5m --start 2024-06 --end 2026-09
    python -m btcbot.binance_klines --symbol BTCUSDT --interval 5m --start 2024-06 --end 2026-09

Writes data/BIN_<SYMBOL>_<interval>.csv.gz (open_time,open,high,low,close,volume in UTC).
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume"]


def parse_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    df = pd.read_csv(io.StringIO(raw), header=0 if first.startswith("open_time") else None, usecols=range(6))
    df.columns = COLS
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    return df.set_index("open_time").astype(float)


def download(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    frames = []
    with requests.Session() as s:
        for month in pd.period_range(start, end, freq="M"):
            url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
            r = s.get(url, timeout=60)
            if r.status_code == 404:
                print("missing", url)
                continue
            r.raise_for_status()
            frames.append(parse_zip(r.content))
            print("ok", month, len(frames[-1]))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ADAUSDT")
    p.add_argument("--interval", default="5m")
    p.add_argument("--start", default="2024-06")
    p.add_argument("--end", default="2026-09")
    p.add_argument("--out-dir", default="data")
    a = p.parse_args()
    df = download(a.symbol, a.interval, a.start, a.end)
    out = Path(a.out_dir) / f"BIN_{a.symbol}_{a.interval}.csv.gz"
    df.to_csv(out)
    print(f"wrote {out}: {len(df)} rows {df.index[0]} .. {df.index[-1]}")


if __name__ == "__main__":
    main()
