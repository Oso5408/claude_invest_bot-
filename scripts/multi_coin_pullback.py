"""Trend Pullback v1 on several coins with the SAME rules and parameters (no per-coin tuning).
Usage: python multi_coin_pullback.py <hl_data folder> [ETH BTC ...]
Prints per coin: whole-period stats, zero-cost stats, net R per half-year block, slippage sensitivity,
15m version, and a Hyperliquid-vs-Binance check on the overlapping recent 5m window."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataclasses import replace
import numpy as np, pandas as pd
from btcbot import pullback as pb
from btcbot.data import load_csv
from btcbot.dyngrid import fill_gaps
from btcbot.hyperliquid import load_funding

D = sys.argv[1].rstrip("/") + "/"
COINS = sys.argv[2:] or ["ETH", "BTC"]
SLIP = {"BTC": 0.0002, "ETH": 0.0002, "SOL": 0.0003, "ADA": 0.0005}  # per side, see report
Z = dict(taker_fee=0, maker_fee=0, slippage=0)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)


def resample(df, rule):
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def row(name, r):
    s = r.stats
    return {"variant": name, "ret": s["total_return"], "dd": s["max_drawdown"], "n": s["trades"],
            "n/wk": s["trades_per_week"], "win": s.get("win_rate"), "R_pre_fee": s.get("avg_R"),
            "netR": s.get("avg_net_R"), "stop%": s.get("median_stop_pct"), "bars": s.get("median_bars")}


def blocks(tr, col="net_R"):
    if not len(tr):
        return {}
    t = pd.to_datetime(tr["entry_time"])
    key = t.dt.year.astype(str) + "H" + np.where(t.dt.month <= 6, "1", "2")
    return tr.groupby(key.values)[col].mean().round(3).to_dict()


for coin in COINS:
    fund = load_funding(D + f"HL_{coin}_funding.csv")
    df = fill_gaps(load_csv(D + f"BN_{coin}_5m.csv"))
    base = pb.PullbackConfig(slippage=SLIP[coin])
    print(f"\n===== {coin}  Binance 5m {df.index[0]:%Y-%m-%d}..{df.index[-1]:%Y-%m-%d}  "
          f"HL funding from {fund.index[0]:%Y-%m-%d}  slippage {SLIP[coin]:.2%}/side")
    V = {"A 1.5R spec": {}, "A zero cost": Z, "A slip 0": dict(slippage=0), "A slip 0.05%": dict(slippage=0.0005),
         "A long only": dict(long_only=True),
         "B ATR trail": dict(exit="atr_trail"), "B zero cost": dict(exit="atr_trail", **Z),
         "B slip 0.05%": dict(exit="atr_trail", slippage=0.0005)}
    rows, blk = [], {}
    for name, kw in V.items():
        r = pb.run(df, replace(base, **kw), fund)
        rows.append(row(name, r))
        if name in ("A 1.5R spec", "A zero cost", "B ATR trail", "B zero cost"):
            blk[name] = blocks(r.trades)
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    print("net R per half-year block:")
    print(pd.DataFrame(blk).T.to_string())
    # 15m version, same parameters
    d15 = resample(df, "15min")
    rows = [row(f"15m {n}", pb.run(d15, replace(base, **kw), fund)) for n, kw in
            {"A spec": {}, "A zero cost": Z, "B trail": dict(exit="atr_trail"), "B zero cost": dict(exit="atr_trail", **Z)}.items()]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # Hyperliquid vs Binance on the overlap
    try:
        hl = fill_gaps(load_csv(D + f"HL_{coin}_5m.csv"))
        lo, hi = hl.index[0], hl.index[-1]
        bn = df[(df.index >= lo) & (df.index <= hi)]
        j = hl.join(bn, rsuffix="_bn", how="inner")
        rc = np.log(j.close).diff().corr(np.log(j.close_bn).diff())
        gap = (j.close / j.close_bn - 1).abs().median()
        hr = pb.run(hl, base, fund).trades; br = pb.run(bn, base, fund).trades
        same = len(set(hr.entry_time) & set(br.entry_time)) if len(hr) and len(br) else 0
        print(f"HL vs Binance 5m overlap {lo:%Y-%m-%d}..{hi:%Y-%m-%d}: return corr {rc:.4f}, median price gap {gap:.4%}, "
              f"trades HL {len(hr)} / BN {len(br)}, same entry bar {same}, "
              f"netR HL {hr.net_R.mean() if len(hr) else float('nan'):.3f} / BN {br.net_R.mean() if len(br) else float('nan'):.3f}")
    except FileNotFoundError:
        pass
    try:
        bk = pd.read_csv(D + f"HL_{coin}_book.csv")
        print(f"HL book: median spread {bk.spread_pct.median():.4%}, 10-level depth bid ${bk.bid_usd_10lv.median():,.0f} ask ${bk.ask_usd_10lv.median():,.0f}")
    except FileNotFoundError:
        pass
