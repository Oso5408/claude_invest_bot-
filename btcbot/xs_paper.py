"""Daily paper accounts for the long-only momentum test (docs/prereg-long-only-momentum.md). No orders, no key.

Two accounts, 200 USD each, same rules and costs as the backtest in btcbot/xs_momentum.py:
    data/daily-xs-long3     long-only top 3 by 28-day momentum among the 20 biggest HL perps, every other
                            Monday, 1x, held only while BTC's previous close > its 50-day average (else cash)
    data/daily-btc-sma50    benchmark: 1x BTC while the same switch is on, else cash

Run once a day just after 00:00 UTC (the daily candle's open is the fill price, as in the backtest). A second run on
the same day does nothing, so a retry cron line is safe; runs after 02:00 UTC are refused because the open would no
longer be a fair fill. A missed day is not replayed. Each account folder gets state.json, trades.csv and
equity.csv; the monitor watches data/daily-*/ (offline after 26 hours) and adds them to the daily report.

    .venv/bin/python -m btcbot.xs_paper run
    .venv/bin/python -m btcbot.xs_paper status
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from btcbot import risk
from btcbot import xs_momentum as xs
from btcbot.hyperliquid import MAINNET, _post, fetch_funding, parse_candles

START_EQUITY = 200.0
HISTORY_DAYS = 110          # enough for the 60-bar minimum, 30-day volume and 50-day BTC average
LATEST_HOUR = 2             # refuse to trade after 02:00 UTC
ACCOUNTS = {"daily-xs-long3": "xs", "daily-btc-sma50": "btc"}


def fetch_daily(now: datetime, pause: float = 0.3) -> dict[str, pd.DataFrame]:
    """Daily candles of every listed (not delisted) HL perp for the last HISTORY_DAYS days, today's open included."""
    end = int(now.timestamp() * 1000)
    start = end - HISTORY_DAYS * 86_400_000
    out = {}
    with requests.Session() as s:
        meta = _post(MAINNET, {"type": "meta"}, s)
        for u in meta["universe"]:
            if u.get("isDelisted"):
                continue
            rows = _post(MAINNET, {"type": "candleSnapshot", "req": {"coin": u["name"], "interval": "1d",
                                                                      "startTime": start, "endTime": end}}, s)
            df = parse_candles(rows)
            df = df[df["volume"] > 0]
            if len(df):
                df.index = df.index.tz_convert(None).normalize()
                out[u["name"]] = df
            time.sleep(pause)
    return out


def load_state(folder: Path) -> dict:
    path = folder / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"equity": START_EQUITY, "positions": {}, "regime_on": False, "last_date": None,
            "fees": 0.0, "funding": 0.0, "started": None}


def save_state(folder: Path, st: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "state.tmp"
    tmp.write_text(json.dumps(st, indent=2))
    tmp.replace(folder / "state.json")


def append(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def slip(coin: str) -> float:
    return xs.TAKER_FEE + xs.LOW_SLIP.get(coin, xs.SLIP)


def step(folder: Path, kind: str, p: xs.Panel, funding: dict[str, float], now: datetime) -> dict:
    """One daily run of one account on panel p (its last row is today). funding: coin -> sum of hourly rates
    since the last run. Returns the new state."""
    st = load_state(folder)
    folder.mkdir(parents=True, exist_ok=True)
    today = p.dates[-1]
    if st["last_date"] == str(today.date()):
        return st
    i = len(p.dates) - 1
    col = {c: j for j, c in enumerate(p.coins)}
    e = st["equity"]
    stamp = now.isoformat(timespec="seconds")

    def trade(coin: str, dq: float, px: float, why: str) -> None:
        nonlocal e
        cost = abs(dq) * px * slip(coin)
        e -= cost
        st["fees"] += cost
        append(folder / "trades.csv", {"time": stamp, "coin": coin, "side": "buy" if dq > 0 else "sell",
                                       "qty": f"{abs(dq):.8g}", "price": f"{px:.8g}", "cost": f"{cost:.4f}",
                                       "reason": why})

    pos = st["positions"]
    for coin in list(pos):                       # mark to today's open, pay funding, drop delisted coins
        q, last = pos[coin]["qty"], pos[coin]["px"]
        j = col.get(coin)
        if j is None or np.isnan(p.o[i, j]):
            trade(coin, -q, last, "no candle today (delisted?)")
            del pos[coin]
            continue
        o = p.o[i, j]
        e += q * (o - last)
        f = q * o * funding.get(coin, 0.0)
        e -= f
        st["funding"] += f
        pos[coin]["px"] = o

    on = bool(xs.regime_on(p)[i])
    if kind == "xs":
        rebal = xs.rebalance_day(xs.LONG_ONLY, today) or (on and not st["regime_on"])
        if rebal or not on:
            tgt = xs.targets(p, i, xs.LONG_ONLY) if on else {}
            want = {p.coins[j]: w for j, w in tgt.items()}
            for coin in sorted(set(pos) | set(want)):
                o = p.o[i, col[coin]]
                q = want.get(coin, 0.0) * e / o
                have = pos.get(coin, {}).get("qty", 0.0)
                if abs(q - have) * o < 1e-9:
                    continue
                trade(coin, q - have, o, "rebalance" if on else "BTC switch off")
                if q == 0:
                    pos.pop(coin, None)
                else:
                    pos[coin] = {"qty": q, "px": o}
    else:
        o = p.o[i, col["BTC"]]
        if on and "BTC" not in pos:
            q = e / o
            trade("BTC", q, o, "BTC switch on")
            pos["BTC"] = {"qty": q, "px": o}
        elif not on and "BTC" in pos:
            trade("BTC", -pos["BTC"]["qty"], o, "BTC switch off")
            del pos["BTC"]

    gross = sum(abs(v["qty"]) * v["px"] for v in pos.values()) / e if e > 0 else 0.0
    if gross > risk.MAX_LEVERAGE:
        raise RuntimeError(f"{folder.name}: leverage {gross:.2f}x over the {risk.MAX_LEVERAGE}x cap")
    st.update(equity=e, regime_on=on, last_date=str(today.date()), last_run=stamp, started=st["started"] or stamp)
    append(folder / "equity.csv", {"time": stamp, "equity": f"{e:.4f}", "btc_switch": int(on),
                                   "gross": f"{gross:.3f}",
                                   "positions": " ".join(f"{c}:{v['qty']:.6g}" for c, v in sorted(pos.items()))})
    save_state(folder, st)
    return st


def run(data_dir: Path, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    folders = {name: data_dir / name for name in ACCOUNTS}
    if all(load_state(f)["last_date"] == str(now.date()) for f in folders.values()):
        print(f"{now:%Y-%m-%d}: already done")
        return
    if now.hour >= LATEST_HOUR:
        raise SystemExit(f"{now:%H:%M} UTC is too late to fill at today's open; next run tomorrow")
    daily = fetch_daily(now)
    p = xs.build_panel(daily, {})
    if p.dates[-1] != pd.Timestamp(now.date()):
        raise SystemExit(f"today's daily candle is not out yet (latest {p.dates[-1].date()}); retry later")
    held = set()
    for f in folders.values():
        held |= set(load_state(f)["positions"])
    funding = {}
    for coin in sorted(held):
        since = min((load_state(f).get("last_run") for f in folders.values()
                     if coin in load_state(f)["positions"]), default=None)
        if since:
            rates = fetch_funding(coin, int(pd.Timestamp(since).timestamp() * 1000), int(now.timestamp() * 1000))
            funding[coin] = float(rates.sum())
    for name, kind in ACCOUNTS.items():
        st = step(folders[name], kind, p, funding, now)
        print(f"{name}: equity {st['equity']:.2f}, BTC switch {'on' if st['regime_on'] else 'off'}, "
              f"positions {sorted(st['positions']) or 'cash'}")


def status(data_dir: Path) -> None:
    for name in ACCOUNTS:
        st = load_state(data_dir / name)
        print(f"{name}: equity {st['equity']:.2f} USD ({st['equity'] / START_EQUITY - 1:+.1%}), "
              f"fees {st['fees']:.2f}, funding paid {st['funding']:+.2f}, last run {st.get('last_run') or '-'}, "
              f"BTC switch {'on' if st['regime_on'] else 'off'}, positions {sorted(st['positions']) or 'cash'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Daily paper accounts for the long-only momentum test")
    ap.add_argument("command", choices=["run", "status"])
    ap.add_argument("--data", default="data")
    a = ap.parse_args(argv)
    run(Path(a.data)) if a.command == "run" else status(Path(a.data))


if __name__ == "__main__":
    main()
