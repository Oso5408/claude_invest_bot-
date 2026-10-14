"""Download data for the ETH / BTC trend pullback test. Standard library only.

Run on your PC (any Python 3):   python fetch_eth_btc.py
Writes into a folder "hl_data" next to this script:
  HL_<COIN>_<5m|15m|1h>.csv   Hyperliquid candles (API keeps only the latest 5000 per interval)
  HL_<COIN>_funding.csv       Hyperliquid hourly funding, full history
  HL_<COIN>_book.csv          Hyperliquid order book snapshots (spread / depth for slippage)
  BN_<COIN>_5m.csv            Binance USDT-M futures 5m candles from data.binance.vision (long history)
"""
import csv, io, json, os, sys, time, urllib.request, zipfile
from datetime import date, datetime, timezone

COINS = ["ETH", "BTC", "SOL", "ADA"]
HL = "https://api.hyperliquid.xyz/info"
BV = "https://data.binance.vision/data/futures/um"
START = date(2023, 1, 1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hl_data")
os.makedirs(OUT, exist_ok=True)


def get(url, body=None, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                         headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # network hiccup
            err = e
        time.sleep(2 ** k)
    raise RuntimeError(f"{url}: {err}")


def iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")


def hl_candles(coin, interval):
    now = int(time.time() * 1000)
    rows = json.loads(get(HL, {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval,
                                                                  "startTime": 0, "endTime": now}}))
    path = os.path.join(OUT, f"HL_{coin}_{interval}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for r in sorted(rows, key=lambda r: r["t"]):
            w.writerow([iso(r["t"]), r["o"], r["h"], r["l"], r["c"], r["v"]])
    print(f"  {path}: {len(rows)} rows from {iso(rows[0]['t'])[:16] if rows else '-'}")


def hl_funding(coin):
    t, end, out = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000), int(time.time() * 1000), []
    while t < end:
        rows = json.loads(get(HL, {"type": "fundingHistory", "coin": coin, "startTime": t, "endTime": end}))
        if not rows:
            break
        out += rows
        t = rows[-1]["time"] + 1
        if len(rows) < 500:
            break
        time.sleep(0.2)
    path = os.path.join(OUT, f"HL_{coin}_funding.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "funding"])
        for r in out:
            w.writerow([iso(r["time"]), r["fundingRate"]])
    print(f"  {path}: {len(out)} rows from {iso(out[0]['time'])[:16] if out else '-'}")


def hl_book(coin, snaps=10):
    path = os.path.join(OUT, f"HL_{coin}_book.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "bid", "ask", "spread_pct", "bid_usd_10lv", "ask_usd_10lv"])
        for _ in range(snaps):
            b = json.loads(get(HL, {"type": "l2Book", "coin": coin}))
            bids, asks = b["levels"]
            bid, ask = float(bids[0]["px"]), float(asks[0]["px"])
            w.writerow([iso(b["time"]), bid, ask, (ask - bid) / ((ask + bid) / 2),
                        sum(float(x["px"]) * float(x["sz"]) for x in bids[:10]),
                        sum(float(x["px"]) * float(x["sz"]) for x in asks[:10])])
            time.sleep(3)
    print(f"  {path}")


def months(d0, d1):
    y, m = d0.year, d0.month
    while (y, m) < (d1.year, d1.month):
        yield f"{y}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def bn_5m(coin):
    sym, today, rows = f"{coin}USDT", date.today(), []

    def add(blob):
        if not blob:
            return False
        z = zipfile.ZipFile(io.BytesIO(blob))
        for line in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))):
            if line and line[0].isdigit():
                rows.append(line[:6])
        return True

    for ym in months(START, today):
        if not add(get(f"{BV}/monthly/klines/{sym}/5m/{sym}-5m-{ym}.zip")):
            print(f"    missing month {ym}")
    for d in range(1, today.day):
        add(get(f"{BV}/daily/klines/{sym}/5m/{sym}-5m-{today:%Y-%m}-{d:02d}.zip"))
    path = os.path.join(OUT, f"BN_{coin}_5m.csv")
    seen = set()
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for r in sorted(rows, key=lambda r: int(r[0])):
            if r[0] not in seen:
                seen.add(r[0]); w.writerow([iso(r[0])] + r[1:6])
    print(f"  {path}: {len(seen)} rows")


if __name__ == "__main__":
    for coin in COINS:
        print(coin)
        for iv in ("5m", "15m", "1h"):
            hl_candles(coin, iv)
        hl_funding(coin)
        hl_book(coin)
        bn_5m(coin)
    print("Done. Folder:", OUT)
