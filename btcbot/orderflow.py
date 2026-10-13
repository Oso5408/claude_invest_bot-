"""Order-Flow Momentum v1 research backtest (strategy B).

Tests whether a 60-second taker Delta filter improves a plain 1-minute
breakout on ADA. Uses Binance USDT-M futures aggTrades as a proxy for
Hyperliquid trade flow (is_buyer_maker gives the taker side). Order-book
imbalance is NOT tested: no historical book data exists for it.

Steps:
  python -m btcbot.orderflow download --start 2026-03 --end 2026-08
  python -m btcbot.orderflow build
  python -m btcbot.orderflow backtest
  python -m btcbot.orderflow randomwalk     # engine sanity check

Rules (orderflow_momentum_v1, long; short mirrors):
  signal  1m close > highest high of the previous 15 completed bars
  filter  Delta = (taker buy - taker sell) / total notional of that bar (last 60s)
  entry   next bar open, plus slippage (signal is only known at bar close)
  stop    lowest low of the last 5 bars (incl. signal bar) minus buffer
  target  1.5R, max hold 30 bars, one position at a time
  a bar that touches both stop and target counts as a stop
Costs: Hyperliquid taker 0.045% per side plus slippage per side.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join("data", "orderflow")
URL = "https://data.binance.vision/data/futures/um/monthly/aggTrades/{s}/{s}-aggTrades-{m}.zip"
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id",
        "last_trade_id", "transact_time", "is_buyer_maker"]


# ---------------------------------------------------------------- data

def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def download(symbol: str, start: str, end: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for m in months(start, end):
        path = os.path.join(DATA_DIR, f"{symbol}-aggTrades-{m}.zip")
        if os.path.exists(path):
            print("have", path)
            continue
        url = URL.format(s=symbol, m=m)
        print("get", url, flush=True)
        tmp = path + ".part"
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(tmp, path)


def _minute_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({"price": float, "quantity": float, "transact_time": np.int64})
    buyer_maker = df["is_buyer_maker"].astype(str).str.lower() == "true"
    notional = df["price"] * df["quantity"]
    g = pd.DataFrame({
        "minute": df["transact_time"] // 60_000,
        "price": df["price"],
        "buy": np.where(buyer_maker, 0.0, notional),   # taker bought
        "sell": np.where(buyer_maker, notional, 0.0),  # taker sold
    }).groupby("minute", sort=True)
    return pd.DataFrame({
        "open": g["price"].first(), "high": g["price"].max(),
        "low": g["price"].min(), "close": g["price"].last(),
        "buy": g["buy"].sum(), "sell": g["sell"].sum(),
        "n": g["price"].size(),
    })


def build_month(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            first = fh.readline().decode()
        header = 0 if first.startswith("agg_trade_id") else None
        parts = []
        with z.open(name) as fh:
            reader = pd.read_csv(io.TextIOWrapper(fh), header=header, names=COLS,
                                 usecols=["price", "quantity", "transact_time",
                                          "is_buyer_maker"], chunksize=2_000_000)
            for chunk in reader:
                parts.append(_minute_agg(chunk))
    df = pd.concat(parts)
    g = df.groupby(level=0, sort=True)  # merge minutes split across chunks
    return pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "buy": g["buy"].sum(), "sell": g["sell"].sum(),
        "n": g["n"].sum(),
    })


def build(symbol: str) -> str:
    zips = sorted(f for f in os.listdir(DATA_DIR)
                  if f.startswith(f"{symbol}-aggTrades-") and f.endswith(".zip"))
    if not zips:
        sys.exit(f"no zips in {DATA_DIR}; run download first")
    frames = []
    for f in zips:
        print("build", f, flush=True)
        frames.append(build_month(os.path.join(DATA_DIR, f)))
    df = pd.concat(frames)
    df = df[~df.index.duplicated()].sort_index()
    out = os.path.join(DATA_DIR, f"{symbol}_1m_flow.csv")
    df.index = pd.to_datetime(df.index * 60_000, unit="ms", utc=True)
    df.index.name = "time"
    df.to_csv(out)
    print("wrote", out, len(df), "minutes")
    return out


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    full = pd.date_range(df.index[0], df.index[-1], freq="1min")
    df = df.reindex(full)
    df["close"] = df["close"].ffill()
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df["close"])
    df[["buy", "sell", "n"]] = df[["buy", "sell", "n"]].fillna(0.0)
    return df


# ---------------------------------------------------------------- backtest

@dataclass
class Params:
    lookback: int = 15
    stop_bars: int = 5
    stop_buffer: float = 0.0005  # 0.05% beyond structure
    target_r: float = 1.5
    max_hold: int = 30
    delta_min: float | None = None  # None = plain breakout
    fee: float = 0.00045            # HL taker per side
    slip: float = 0.0002            # per side
    max_cost_r: float | None = None  # skip if round-trip cost > this many R
    shorts: bool = True


def signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    hh = df["high"].shift(1).rolling(p.lookback).max()
    ll = df["low"].shift(1).rolling(p.lookback).min()
    tot = df["buy"] + df["sell"]
    delta = (df["buy"] - df["sell"]) / tot.where(tot > 0)
    out = pd.DataFrame(index=df.index)
    out["long"] = df["close"] > hh
    out["short"] = (df["close"] < ll) & p.shorts
    out["delta"] = delta
    out["stop_long"] = df["low"].rolling(p.stop_bars).min() * (1 - p.stop_buffer)
    out["stop_short"] = df["high"].rolling(p.stop_bars).max() * (1 + p.stop_buffer)
    return out


def run(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    sig = signals(df, p)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    lg, sh, dl = sig["long"].to_numpy(), sig["short"].to_numpy(), sig["delta"].to_numpy()
    sl_l, sl_s = sig["stop_long"].to_numpy(), sig["stop_short"].to_numpy()
    n = len(df)
    trades = []
    i = p.lookback
    while i < n - 2:
        side = 0
        if lg[i] and (p.delta_min is None or (dl[i] == dl[i] and dl[i] >= p.delta_min)):
            side = 1
        elif sh[i] and (p.delta_min is None or (dl[i] == dl[i] and dl[i] <= -p.delta_min)):
            side = -1
        if side == 0:
            i += 1
            continue
        e = i + 1
        entry = o[e] * (1 + side * p.slip)
        stop = sl_l[i] if side == 1 else sl_s[i]
        risk = (entry - stop) * side
        if not risk > 0:
            i += 1
            continue
        risk_pct = risk / entry
        cost_r = 2 * (p.fee + p.slip) / risk_pct
        if p.max_cost_r is not None and cost_r > p.max_cost_r:
            i += 1
            continue
        target = entry + side * p.target_r * risk
        exit_px, j = None, e
        last = min(e + p.max_hold - 1, n - 1)
        for j in range(e, last + 1):
            hit_stop = (l[j] <= stop) if side == 1 else (h[j] >= stop)
            hit_tgt = (h[j] >= target) if side == 1 else (l[j] <= target)
            if hit_stop:  # both in one bar -> stop
                gap = min(o[j], stop) if side == 1 else max(o[j], stop)
                exit_px = (gap if j > e else stop) * (1 - side * p.slip)
                break
            if hit_tgt:
                exit_px = target  # limit order, no slippage
                break
        if exit_px is None:
            exit_px = c[last] * (1 - side * p.slip)
            j = last
        gross = (exit_px - entry) * side
        fees = p.fee * (entry + exit_px)
        trades.append({
            "time": df.index[e], "side": side, "delta": dl[i], "risk_pct": risk_pct,
            "r": (gross - fees) / risk, "cost_r": cost_r,
            "bars": j - e + 1,
        })
        i = j + 1  # one position at a time; next signal after exit bar
    return pd.DataFrame(trades)


def summary(t: pd.DataFrame, days: float, risk: float = 0.01) -> dict:
    if t.empty:
        return {"trades": 0}
    eq = float(np.prod(1 + risk * t["r"].clip(lower=-3)))
    half = len(t) // 2
    return {
        "trades": len(t), "per_day": round(len(t) / days, 1),
        "win%": round(float(100 * (t["r"] > 0).mean()), 1),
        "avg_R": round(float(t["r"].mean()), 3),
        "R_1st": round(float(t["r"].iloc[:half].mean()), 3),
        "R_2nd": round(float(t["r"].iloc[half:].mean()), 3),
        "stop%_med": round(float(100 * t["risk_pct"].median()), 3),
        "cost_R_med": round(float(t["cost_r"].median()), 2),
        "ret%@1%risk": round(100 * (eq - 1), 1),
    }


def backtest(path: str) -> None:
    df = load(path)
    days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    print(f"data {df.index[0]} -> {df.index[-1]} ({days:.0f} days)\n")
    rows = []
    for cost_name, fee, slip in (("HL cost", 0.00045, 0.0002), ("zero cost", 0.0, 0.0)):
        for dmin in (None, 0.0, 0.2, 0.3, 0.4, 0.5):
            for mcr in (None, 0.25):
                if cost_name == "zero cost" and mcr is not None:
                    continue
                p = Params(delta_min=dmin, fee=fee, slip=slip, max_cost_r=mcr)
                t = run(df, p)
                rows.append({"costs": cost_name,
                             "delta": "none" if dmin is None else f">={dmin}",
                             "cost_cap": "-" if mcr is None else f"<{mcr}R",
                             **summary(t, days)})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))

    # Does Delta sort outcomes at all? Plain breakouts, zero cost, by Delta bucket.
    t = run(df, Params(fee=0.0, slip=0.0))
    t["aligned"] = t["delta"] * t["side"]
    t["bucket"] = pd.cut(t["aligned"], [-1.01, -0.2, 0.0, 0.2, 0.4, 0.6, 1.01])
    print("\nplain breakouts, zero cost, by Delta in trade direction:")
    print(t.groupby("bucket", observed=True)["r"].agg(["count", "mean",
          lambda r: (r > 0).mean()]).rename(columns={"<lambda_0>": "win"}).round(3))
    out = os.path.join(DATA_DIR, "orderflow_results.csv")
    res.to_csv(out, index=False)
    print("\nwrote", out)


# ---------------------------------------------------------------- sanity

def random_walk(minutes: int = 200_000, seed: int = 7) -> pd.DataFrame:
    """Second-level random walk with random taker sides -> 1m bars. No edge exists."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.0002, minutes * 60)
    px = 0.6 * np.exp(np.cumsum(steps))
    m = np.arange(minutes * 60) // 60
    notional = rng.exponential(1000, minutes * 60)
    side = rng.random(minutes * 60) < 0.5
    g = pd.DataFrame({"m": m, "p": px, "b": np.where(side, notional, 0),
                      "s": np.where(side, 0, notional)}).groupby("m")
    df = pd.DataFrame({"open": g["p"].first(), "high": g["p"].max(), "low": g["p"].min(),
                       "close": g["p"].last(), "buy": g["b"].sum(), "sell": g["s"].sum()})
    df["n"] = 60
    df.index = pd.date_range("2025-01-01", periods=minutes, freq="1min", tz="UTC")
    return df


def randomwalk() -> None:
    df = random_walk()
    for dmin in (None, 0.2):
        t = run(df, Params(delta_min=dmin, fee=0.0, slip=0.0))
        print("random walk zero cost, delta", dmin, summary(t, len(df) / 1440))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "build", "backtest", "randomwalk", "all"])
    ap.add_argument("--symbol", default="ADAUSDT")
    ap.add_argument("--start", default="2026-03")
    ap.add_argument("--end", default="2026-08")
    a = ap.parse_args(argv)
    flow = os.path.join(DATA_DIR, f"{a.symbol}_1m_flow.csv")
    if a.cmd in ("download", "all"):
        download(a.symbol, a.start, a.end)
    if a.cmd in ("build", "all"):
        flow = build(a.symbol)
    if a.cmd in ("backtest", "all"):
        backtest(flow)
    if a.cmd == "randomwalk":
        randomwalk()


if __name__ == "__main__":
    main()
