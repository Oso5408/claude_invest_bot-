"""Follow-up checks for swing_1_2day.py: sensitivity grid, concentration, slippage, overlap with the 4h breakout."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/claude_invest_bot-")
sys.path.insert(0, "/mnt/project-files/reports/scripts")
from btcbot.swing import Costs, daily, simulate, summarize
from swing_1_2day import COINS, SLIP, s1_daily_breakout, s3_big_move_fade

D = sys.argv[1]
h1s = {c: pd.read_pickle(f"{D}/{c}_1h.pkl") for c in COINS}
funds = {c: pd.read_pickle(f"{D}/{c}_fund.pkl") for c in COINS}


def run(fn, slip_mult=1.0, **kw):
    out = []
    for c in COINS:
        h1, d = h1s[c], daily(h1s[c])
        sig, stop, hold = fn(d, h1, funds[c], **kw)
        tr = simulate(h1, sig, stop, hold, funds[c], Costs(slippage=SLIP[c] * slip_mult))
        tr["coin"] = c
        out.append(tr)
    return pd.concat(out, ignore_index=True)


def follow(d, h1, fund, z=2.5, hold=48):
    sig, stop, _ = s3_big_move_fade(d, h1, fund, z=z)
    return -sig, stop, hold


def line(tr):
    s = summarize(tr)
    a = tr[tr.entry_time < "2025-01-01"].net_r.mean()
    b = tr[tr.entry_time >= "2025-01-01"].net_r.mean()
    return f"n {s['n']:4d} gross {s['gross_r']:+.3f} net {s['net_r']:+.3f} t {s['t']:+.2f} | 2023-24 {a:+.3f} 2025-26 {b:+.3f}"


print("## sensitivity grid (all reported, none picked)")
for z in (2.0, 2.5, 3.0):
    for hold in (24, 48, 96):
        print(f"follow big day z {z} hold {hold:3d}h:", line(run(follow, z=z, hold=hold)))
for lb in (10, 20, 40):
    for hold in (24, 48, 96):
        print(f"breakout {lb:2d}d hold {hold:3d}h:", line(run(s1_daily_breakout, lookback=lb, hold=hold)))

print("## main configs: slippage x3, concentration, equity at 1% risk")
for name, fn, kw in [("follow 2.5s 48h", follow, {}), ("breakout 20d 48h", s1_daily_breakout, {})]:
    tr = run(fn, **kw)
    print(name, "slip x3:", line(run(fn, slip_mult=3, **kw)))
    r = tr.net_r.sort_values(ascending=False)
    print(f"  sum R {r.sum():+.1f}, top5 {r.head(5).sum():+.1f}, without top5 mean {r.iloc[5:].mean():+.3f}, "
          f"without top10 mean {r.iloc[10:].mean():+.3f}; long {tr[tr.side>0].net_r.mean():+.3f} ({(tr.side>0).sum()}) "
          f"short {tr[tr.side<0].net_r.mean():+.3f} ({(tr.side<0).sum()})")
    tr = tr.sort_values("exit_time")
    eq = (1 + 0.01 * tr.net_r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = (tr.exit_time.max() - tr.entry_time.min()).days / 365
    print(f"  4 coins, 1% risk per trade: total {eq.iloc[-1] - 1:+.0%} over {yrs:.1f}y, max DD {dd:.0%}, "
          f"avg notional {0.01 / tr.stop_pct.mean():.2f}x equity, avg gross move {tr.gross_ret.abs().mean():.2%}")
    print("  per year net R:", tr.groupby(tr.entry_time.dt.year).net_r.agg(['count', 'mean']).round(3).to_dict('index'))

print("## overlap with the 4h breakout (ADA, Hyperliquid 4h, 2024-06 onward)")
from btcbot import breakout
from btcbot.data import load_csv, load_fng
from btcbot.hyperliquid import account_config, load_funding
R = "/home/claude/claude_invest_bot-/data/"
b = breakout.run(load_csv(R + "HL_ADA_4h.csv"), breakout.BreakoutConfig(fng_short_max=50), account_config(200),
                 load_fng(R + "fear_greed.csv"), load_funding(R + "HL_ADA_funding.csv")).trades
for name, fn in [("follow 2.5s 48h", follow), ("breakout 20d 48h", s1_daily_breakout)]:
    tr = run(fn)
    tr = tr[(tr.coin == "ADA") & (tr.entry_time >= "2024-07-01")]
    same = [((b.entry_time <= t) & (b.exit_time > t) & (b.side == s)).any() for t, s in zip(tr.entry_time, tr.side)]
    tr = tr.assign(inside=same)
    print(f"{name}: {len(tr)} ADA trades, {np.mean(same):.0%} start while the 4h breakout already holds the same side;",
          f"net R inside {tr[tr.inside].net_r.mean():+.3f} ({tr.inside.sum()}), outside {tr[~tr.inside].net_r.mean():+.3f} ({(~tr.inside).sum()})")
