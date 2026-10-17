"""Download Hyperliquid data for the cross-sectional momentum test. Standard library only.

Run on a machine that can reach Hyperliquid:   python xs_fetch.py [folder]
Writes into [folder] (default "hl_data/xs" next to this script); re-running skips files already done, so give a
new folder (e.g. hl_data/xs-2027-01) to get fresh data:
  meta.json                 every perp in HL meta, including delisted ones
  D_<COIN>.csv              daily candles (full history, API keeps 5000 per interval)
  F_<COIN>.csv              hourly funding, full history, for coins ever in the top 25 by volume
"""
import csv, json, os, sys, time, urllib.request
from datetime import datetime, timezone

HL = "https://api.hyperliquid.xyz/info"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hl_data", "xs")
TOP = 25        # funding is fetched for coins that were ever this high in 30-day volume
PAUSE = 1.0     # seconds between requests (HL info limit is 1200 weight per minute)


def post(body, tries=6):
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(HL, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                time.sleep(PAUSE)
                return json.loads(r.read())
        except Exception as e:  # 429 or network hiccup
            err = e
            time.sleep(5 * 2 ** k)
    raise RuntimeError(f"{body}: {err}")


def iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def daily(coin):
    path = os.path.join(OUT, f"D_{coin}.csv")
    if os.path.exists(path):
        return
    rows = post({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1d", "startTime": 0,
                                                    "endTime": int(time.time() * 1000)}}) or []
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "open", "high", "low", "close", "volume"])
        for r in sorted(rows, key=lambda r: r["t"]):
            w.writerow([iso(r["t"])[:10], r["o"], r["h"], r["l"], r["c"], r["v"]])
    print(f"  D_{coin}: {len(rows)} days")


def funding(coin):
    path = os.path.join(OUT, f"F_{coin}.csv")
    if os.path.exists(path):
        return
    t, end, out = 0, int(time.time() * 1000), []
    while t < end:
        rows = post({"type": "fundingHistory", "coin": coin, "startTime": t, "endTime": end})
        if not rows:
            break
        out += rows
        t = rows[-1]["time"] + 1
        if len(rows) < 500:
            break
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "funding"])
        for r in out:
            w.writerow([iso(r["time"]), r["fundingRate"]])
    os.replace(path + ".tmp", path)
    print(f"  F_{coin}: {len(out)} hours")


def ever_top(coins):
    """Coins that were in the top TOP by 30-day average dollar volume on at least one day."""
    vol = {}
    for c in coins:
        with open(os.path.join(OUT, f"D_{c}.csv")) as f:
            for r in csv.DictReader(f):
                vol.setdefault(r["date"], {})[c] = float(r["volume"]) * float(r["close"])
    days, keep, hist = sorted(vol), set(), {c: [] for c in coins}
    for d in days:
        for c in coins:
            hist[c].append(vol[d].get(c))
        avg = {c: sum(x for x in h[-30:] if x) / 30 for c, h in hist.items() if h[-1] is not None}
        keep.update(sorted(avg, key=avg.get, reverse=True)[:TOP])
    return sorted(keep)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        OUT = os.path.abspath(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    meta = post({"type": "meta"})
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f)
    coins = [u["name"] for u in meta["universe"]]
    print(f"{len(coins)} perps in meta ({sum(1 for u in meta['universe'] if u.get('isDelisted'))} delisted)")
    print("Daily candles ...")
    for i, c in enumerate(coins, 1):
        try:
            daily(c)
        except Exception as e:
            print(f"  D_{c}: FAILED {e}")
        if i % 25 == 0:
            print(f"  [{i}/{len(coins)}]")
    have = [c for c in coins if os.path.exists(os.path.join(OUT, f"D_{c}.csv"))]
    top = ever_top(have)
    print(f"Funding for {len(top)} coins ever in top {TOP} ...")
    for i, c in enumerate(top, 1):
        try:
            funding(c)
        except Exception as e:
            print(f"  F_{c}: FAILED {e}")
        print(f"  [{i}/{len(top)}]")
    print("Done. Folder:", OUT)
