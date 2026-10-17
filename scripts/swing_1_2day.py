"""1-2 day holding strategies on ADA/BTC/ETH/SOL (Binance perp 1h prices, Hyperliquid costs).

Usage: python swing_1_2day.py DATA_DIR [--random]
DATA_DIR holds {COIN}_1h.pkl (Binance USDT-M 5m resampled to 1h) and {COIN}_fund.pkl
(Hyperliquid hourly funding). --random replaces prices with a shuffled-returns walk
(same hourly volatility, no autocorrelation, built from 5m steps) as a sanity check.
All parameters are fixed in advance; nothing here is tuned on results.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/claude_invest_bot-")
from btcbot.swing import Costs, atr, daily, simulate, summarize

COINS = ["ADA", "BTC", "ETH", "SOL"]
SLIP = {"ADA": 0.0005, "SOL": 0.0005, "BTC": 0.0002, "ETH": 0.0002}
DAY = pd.Timedelta("1D")


def shuffled(h1: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Random walk with the coin's own hourly volatility, built from 5-minute steps so
    every high/low is a price the path really traded (path-consistent wicks)."""
    rng = np.random.default_rng(seed)
    vol5 = np.log(h1["close"]).diff().std() / np.sqrt(12)
    steps = rng.standard_normal(len(h1) * 12) * vol5
    p = h1["close"].iloc[0] * np.exp(np.cumsum(steps)).reshape(len(h1), 12)
    op = np.r_[h1["close"].iloc[0], p[:-1, -1]]
    out = pd.DataFrame({"open": op, "close": p[:, -1]}, index=h1.index)
    out["high"] = np.maximum(p.max(axis=1), op)
    out["low"] = np.minimum(p.min(axis=1), op)
    out["volume"] = h1["volume"].to_numpy()
    return out


def stop_from_daily(d: pd.DataFrame, mult: float = 1.5) -> pd.Series:
    """1.5 x ATR14 (daily) as a fraction of price, known at the close of day D -> usable at D+1 00:00."""
    s = mult * atr(d) / d["close"]
    s.index = s.index + DAY
    return s


def at_next_open(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index + DAY
    return s


# ---- strategies: each returns (signals, stop_pct, hold_h)
def s1_daily_breakout(d, h1, fund, btc=None, lookback=20, hold=48):
    hi = d["high"].rolling(lookback).max().shift()
    lo = d["low"].rolling(lookback).min().shift()
    sig = pd.Series(np.where(d["close"] > hi, 1, np.where(d["close"] < lo, -1, 0)), index=d.index)
    return at_next_open(sig), stop_from_daily(d), hold


def s2_tsmom(d, h1, fund, btc=None, look=3, hold=48):
    """Sign of the last 3 days' return, re-decided every 2 days (always in the market)."""
    r = d["close"].pct_change(look)
    sig = np.sign(r)
    sig = sig[(np.arange(len(sig)) % 2) == 0]
    return at_next_open(sig), stop_from_daily(d), hold


def s3_big_move_fade(d, h1, fund, btc=None, z=2.5, hold=24):
    r = d["close"].pct_change()
    sd = r.rolling(20).std().shift()
    sig = pd.Series(np.where(r > z * sd, -1, np.where(r < -z * sd, 1, 0)), index=d.index)
    return at_next_open(sig), stop_from_daily(d), hold


def s4_funding_extreme(d, h1, fund, btc=None, z=2.0, hold=48):
    if fund is None:
        return pd.Series(dtype=float), stop_from_daily(d), hold
    f24 = fund.rolling(24, min_periods=20).mean()
    base_m = f24.rolling(720, min_periods=500).mean()
    base_s = f24.rolling(720, min_periods=500).std()
    zz = (f24 - base_m) / base_s
    # decide at 00:00 UTC each day with the funding known up to then
    zz = zz[zz.index.hour == 23]
    zz.index = zz.index + pd.Timedelta("1h")
    sig = pd.Series(np.where(zz > z, -1, np.where(zz < -z, 1, 0)), index=zz.index)
    return sig, stop_from_daily(d), hold


def s5_btc_lead(d, h1, fund, btc=None, z=2.0, hold=24):
    """BTC 4h move > 2 sigma while the alt moved less than half as much -> trade the alt in BTC's direction."""
    if btc is None:
        return pd.Series(dtype=float), stop_from_daily(d), hold
    b4 = btc["close"].pct_change(4)
    a4 = h1["close"].pct_change(4)
    sd = b4.rolling(720, min_periods=500).std()
    up = (b4 > z * sd) & (a4 < 0.5 * b4)
    dn = (b4 < -z * sd) & (a4 > 0.5 * b4)
    sig = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=h1.index)
    sig = sig[sig.index.hour % 4 == 3]
    sig.index = sig.index + pd.Timedelta("1h")
    stop = stop_from_daily(d).reindex(sig.index, method="ffill")
    return sig, stop, hold


STRATS = {
    "S1 daily breakout 20d, hold 48h": s1_daily_breakout,
    "S2 3-day momentum, hold 48h": s2_tsmom,
    "S3 fade big day (2.5 sigma), hold 24h": s3_big_move_fade,
    "S4 funding extreme fade (z 2), hold 48h": s4_funding_extreme,
    "S5 BTC-led alt catch-up, hold 24h": s5_btc_lead,
}
# fixed neighbours, reported for sensitivity only
NEIGHBOURS = {
    "S1 daily breakout 20d, hold 24h": lambda *a, **k: s1_daily_breakout(*a, **k, hold=24),
    "S1 daily breakout 10d, hold 48h": lambda *a, **k: s1_daily_breakout(*a, **k, lookback=10),
    "S3 fade big day 2 sigma, hold 24h": lambda *a, **k: s3_big_move_fade(*a, **k, z=2.0),
    "S3 follow big day 2.5 sigma, hold 48h": lambda *a, **k: (lambda s: (-s[0], s[1], 48))(s3_big_move_fade(*a, **k)),
}


def half_year(ts: pd.Series) -> pd.Series:
    return ts.dt.year.astype(str) + "H" + np.where(ts.dt.month <= 6, "1", "2")


def run_all(data_dir: str, random: bool = False, strats=None, seeds=(1,)):
    strats = strats or STRATS
    out = []
    for seed in seeds:
        h1s = {c: pd.read_pickle(f"{data_dir}/{c}_1h.pkl") for c in COINS}
        if random:
            h1s = {c: shuffled(h, seed + i) for i, (c, h) in enumerate(h1s.items())}
        funds = {c: pd.read_pickle(f"{data_dir}/{c}_fund.pkl") for c in COINS}
        for name, fn in strats.items():
            for c in COINS:
                if "BTC-led" in name and c == "BTC":
                    continue
                h1, d = h1s[c], daily(h1s[c])
                sig, stop, hold = fn(d, h1, funds[c], btc=h1s["BTC"])
                tr = simulate(h1, sig, stop, hold, funds[c], Costs(slippage=SLIP[c]))
                if tr.empty:
                    continue
                tr["coin"], tr["strategy"], tr["seed"] = c, name, seed
                out.append(tr)
    return pd.concat(out, ignore_index=True)


def table(tr: pd.DataFrame, by) -> pd.DataFrame:
    return pd.DataFrame({k: summarize(g) for k, g in tr.groupby(by)}).T


if __name__ == "__main__":
    data_dir = sys.argv[1]
    rnd = "--random" in sys.argv
    pd.set_option("display.width", 220, "display.max_rows", 500)
    tr = run_all(data_dir, rnd, {**STRATS, **NEIGHBOURS}, seeds=(1, 2, 3) if rnd else (1,))
    tr.to_pickle(f"{data_dir}/trades{'_random' if rnd else ''}.pkl")
    cols = ["n", "win", "gross_r", "cost_r", "net_r", "t", "stop_med", "stop_hits"]
    print(table(tr, ["strategy", "coin"])[cols].astype(float).round(3))
    print(table(tr, ["strategy"])[cols].astype(float).round(3))
    tr["half"] = half_year(tr["entry_time"])
    piv = tr.groupby(["strategy", "coin", "half"])["net_r"].mean().unstack().round(2)
    print(piv)
