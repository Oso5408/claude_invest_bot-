"""Cross-sectional momentum on Hyperliquid perps: long the strongest K, short the weakest K (direction A).

The rules are pre-registered in docs/prereg-xs-momentum.md and are not to be tuned. In short:
- universe on rebalance day d: perps with >= 60 daily bars before d and a bar on d, top 20 by 30-day average
  dollar volume up to d-1 (fewer than 8 eligible -> flat)
- signal: close[d-2] / close[d-2-N] - 1 (skips the latest day), or for the funding variant the 7-day average of
  daily funding up to d-1 (long the lowest, short the highest)
- weights: inverse 30-day daily volatility inside each side, each side = 1.0 x equity (2x gross, hard 2x cap)
- executed at the open of d, held until the next rebalance; each leg has a stop 3 sigma from its entry (checked on
  daily high / low, gap -> open), a stopped leg stays flat until the next rebalance
- costs: taker 0.045% + slippage per side (BTC/ETH 0.02%, others 0.05%, stress 0.15%), real hourly HL funding
- development period 2023-06-01 .. 2026-03-31; the hold-out (2026-04-01 on) is dropped at load time unless
  --holdout is given

Data comes from btcbot/xs_fetch.py (D_<COIN>.csv daily candles, F_<COIN>.csv hourly funding). Research only:

    python -m btcbot.xs_momentum --data C:/Users/osola/Downloads/hl_data/xs
    python -m btcbot.xs_momentum --data ... --perm 1000          # plus the shuffled-pick test on the main variant
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from btcbot.hyperliquid import TAKER_FEE

DEV_START, DEV_END, HOLDOUT_START = "2023-06-01", "2026-03-31", "2026-04-01"
TOP_N, MIN_BARS, MIN_UNIVERSE, VOL_DAYS, STOP_SIGMA = 20, 60, 8, 30, 3.0
MAX_GROSS = 2.0
LOW_SLIP = {"BTC": 0.0002, "ETH": 0.0002}
SLIP = 0.0005


@dataclass(frozen=True)
class Variant:
    name: str
    signal: str = "momentum"      # or "funding"
    lookback: int = 7
    k: int = 2
    rebalance: str = "weekly"     # Mondays, "biweekly" (every other Monday from 2023-01-02) or "daily"
    weights: str = "inv_vol"      # or "equal"
    slip: float | None = None     # one slippage for every coin (stress test); None = LOW_SLIP / SLIP
    long_only: bool = False       # long the top K only, `side_gross` x equity in total
    side_gross: float = MAX_GROSS / 2
    regime: bool = False          # hold only while BTC's previous close > its 50-day average, else cash
    stop: bool = True             # per-leg 3-sigma stop


VARIANTS = [
    Variant("1 main: 7d, K2, weekly"),
    Variant("2 28d lookback", lookback=28),
    Variant("3 daily rebalance", rebalance="daily"),
    Variant("4 K4", k=4),
    Variant("5 equal notional", weights="equal"),
    Variant("6 funding factor", signal="funding"),
]

# docs/prereg-long-only-momentum.md (oso01's method A, 2026-10-17)
LONG_ONLY = Variant("long-only top 3, 28d, biweekly, BTC > SMA50", lookback=28, k=3, rebalance="biweekly",
                    weights="equal", long_only=True, side_gross=1.0, regime=True, stop=False)
REGIME_DAYS = 50


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    coins: list[str]
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    dvol: np.ndarray      # 30-day average dollar volume up to d-1 (missing days count as 0)
    bars: np.ndarray      # number of daily bars before d
    sigma: np.ndarray     # std of the 30 daily log returns up to d-1
    fund: np.ndarray      # sum of the hourly funding rates on day d (0 where unknown)
    fund7: np.ndarray     # 7-day average of daily funding up to d-1 (NaN where unknown)
    has_fund: np.ndarray  # coin has a funding file


def build_panel(daily: dict[str, pd.DataFrame], funding: dict[str, pd.Series], end: str | None = None) -> Panel:
    """daily: coin -> DataFrame indexed by date with open/high/low/close/volume; funding: coin -> hourly rate
    Series indexed by timestamp. Rows after `end` are dropped before anything is computed."""
    frames = {k: v for k, v in daily.items() if len(v)}
    dates = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for v in frames.values()])))
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    coins = sorted(frames)
    col = lambda f: pd.DataFrame({k: frames[k][f] for k in coins}).reindex(dates).astype(float)
    o, h, l, c, v = (col(f) for f in ("open", "high", "low", "close", "volume"))
    dvol = (v * c).fillna(0).rolling(VOL_DAYS, min_periods=1).mean().shift(1)
    bars = c.notna().cumsum().shift(1).fillna(0)
    sigma = np.log(c / c.shift(1)).rolling(VOL_DAYS, min_periods=VOL_DAYS).std().shift(1)
    f = pd.DataFrame(index=dates, columns=coins, dtype=float)
    for k in coins:
        if k in funding and len(funding[k]):
            s = funding[k]
            f[k] = s.groupby(s.index.normalize()).sum().reindex(dates)
    fund7 = f.rolling(7, min_periods=7).mean().shift(1)
    has = np.array([k in funding and len(funding[k]) > 0 for k in coins])
    return Panel(dates, coins, o.values, h.values, l.values, c.values, dvol.values, bars.values,
                 sigma.values, f.fillna(0).values, fund7.values, has)


def targets(p: Panel, i: int, v: Variant, rng: np.random.Generator | None = None) -> dict[int, float]:
    """Target weights (fraction of equity, + long / - short) for the open of day i, from data up to day i-1.
    With rng the K longs and K shorts are drawn at random from the same universe (shuffled-pick test)."""
    if i < 2 + v.lookback:
        return {}
    ok = ~np.isnan(p.o[i]) & (p.bars[i] >= MIN_BARS) & ~np.isnan(p.sigma[i]) & (p.sigma[i] > 0)
    if v.signal == "funding":
        score = p.fund7[i]
        ok &= p.has_fund & ~np.isnan(score)
    else:
        score = p.c[i - 2] / p.c[i - 2 - v.lookback] - 1
        ok &= ~np.isnan(score)
    idx = np.flatnonzero(ok)
    if len(idx) < MIN_UNIVERSE:
        return {}
    idx = idx[np.argsort(-p.dvol[i, idx], kind="stable")][:TOP_N]
    if len(idx) < MIN_UNIVERSE:
        return {}
    if rng is not None:
        pick = rng.permutation(idx)
        top, bottom = pick[:v.k], pick[v.k:2 * v.k]
    else:
        order = idx[np.argsort(-score[idx], kind="stable")]   # strongest first
        if v.signal == "funding":
            order = order[::-1]                                # long the lowest funding
        top, bottom = order[:v.k], order[-v.k:]
    out = {}
    for side, legs in ((1.0, top), (-1.0, bottom))[:1 if v.long_only else 2]:
        raw = np.ones(len(legs)) if v.weights == "equal" else 1 / p.sigma[i, legs]
        for j, w in zip(legs, raw / raw.sum()):
            out[int(j)] = out.get(int(j), 0.0) + side * w * v.side_gross
    return out


def regime_on(p: Panel) -> np.ndarray:
    """True on day i when BTC's close on i-1 is above the average of its last REGIME_DAYS closes up to i-1."""
    btc = pd.Series(p.c[:, p.coins.index("BTC")], index=p.dates)
    return (btc > btc.rolling(REGIME_DAYS, min_periods=REGIME_DAYS).mean()).shift(1, fill_value=False).values


def rebalance_day(v: Variant, day: pd.Timestamp) -> bool:
    if v.rebalance == "daily":
        return True
    if day.weekday() != 0:
        return False
    return v.rebalance == "weekly" or (day - pd.Timestamp("2023-01-02")).days // 7 % 2 == 0


def simulate(p: Panel, v: Variant, start: str, end: str, equity: float = 200.0,
             rng: np.random.Generator | None = None) -> dict:
    slip = np.array([v.slip if v.slip is not None else LOW_SLIP.get(k, SLIP) for k in p.coins])
    rate = TAKER_FEE + slip
    lo, hi = p.dates.searchsorted(pd.Timestamp(start)), p.dates.searchsorted(pd.Timestamp(end), "right")
    qty: dict[int, float] = {}
    entry: dict[int, tuple[float, float]] = {}     # coin -> (entry price, stop price)
    e = equity
    on = regime_on(p) if v.regime else np.ones(len(p.dates), bool)
    eq, fees, fund_paid, traded, stops, gross_max = [], 0.0, 0.0, 0.0, 0, 0.0

    def trade(j: int, dq: float, px: float) -> None:
        nonlocal e, fees, traded
        cost = abs(dq) * px * rate[j]
        e -= cost; fees += cost; traded += abs(dq) * px

    for i in range(lo, hi):
        prev = p.c[i - 1] if i > 0 else p.o[i]
        for j in list(qty):                        # gap from yesterday's close; a missing bar = delisted, exit
            if np.isnan(p.o[i, j]):
                trade(j, -qty[j], prev[j]); del qty[j]; entry.pop(j, None)
            else:
                e += qty[j] * (p.o[i, j] - prev[j])
        rebal = rebalance_day(v, p.dates[i]) or (on[i] and not on[i - 1])
        if rebal or not on[i]:
            tgt = targets(p, i, v, rng) if on[i] else {}
            for j in set(qty) | set(tgt):
                want = tgt.get(j, 0.0) * e / p.o[i, j]
                have = qty.get(j, 0.0)
                if want == 0:
                    trade(j, -have, p.o[i, j]); qty.pop(j, None); entry.pop(j, None)
                    continue
                trade(j, want - have, p.o[i, j])
                if have == 0 or np.sign(have) != np.sign(want):   # new leg: entry and stop fixed here
                    s = STOP_SIGMA * p.sigma[i, j]
                    entry[j] = (p.o[i, j], p.o[i, j] * (1 - s if want > 0 else 1 + s))
                qty[j] = want
        gross = sum(abs(q) * p.o[i, j] for j, q in qty.items()) / e if qty else 0.0
        if gross > MAX_GROSS:                      # hard 2x cap after drift
            for j in qty:
                dq = qty[j] * (MAX_GROSS / gross - 1)
                trade(j, dq, p.o[i, j]); qty[j] += dq
            gross = MAX_GROSS
        gross_max = max(gross_max, gross)
        for j, q in qty.items():
            f = q * p.o[i, j] * p.fund[i, j]
            e -= f; fund_paid += f
        for j in list(qty):
            q, (_, stop) = qty[j], entry[j]
            hit = v.stop and (p.l[i, j] <= stop if q > 0 else p.h[i, j] >= stop)
            if hit:
                fill = min(p.o[i, j], stop) if q > 0 else max(p.o[i, j], stop)
                e += q * (fill - p.o[i, j])
                trade(j, -q, fill); del qty[j]; del entry[j]; stops += 1
            else:
                e += q * (p.c[i, j] - p.o[i, j])
        eq.append(e)
    curve = pd.Series(eq, index=p.dates[lo:hi])
    return {"equity": curve, "fees": fees, "funding": fund_paid, "traded": traded, "stops": stops,
            "gross_max": gross_max, "start_equity": equity}


def btc_benchmark(p: Panel, start: str, end: str, regime: bool, equity: float = 200.0) -> dict:
    """1x BTC, in cash while the regime switch is off (or always held), same costs as the strategy."""
    j = p.coins.index("BTC")
    rate = TAKER_FEE + LOW_SLIP["BTC"]
    on = regime_on(p) if regime else np.ones(len(p.dates), bool)
    lo, hi = p.dates.searchsorted(pd.Timestamp(start)), p.dates.searchsorted(pd.Timestamp(end), "right")
    e, qty, eq, fees, fund = equity, 0.0, [], 0.0, 0.0
    for i in range(lo, hi):
        o = p.o[i, j]
        if qty:
            e += qty * (o - p.c[i - 1, j])
        if on[i] and not qty:
            qty = e / o; fees += e * rate; e -= e * rate
        elif not on[i] and qty:
            fees += qty * o * rate; e -= qty * o * rate; qty = 0.0
        if qty:
            f = qty * o * p.fund[i, j]; e -= f; fund += f
            e += qty * (p.c[i, j] - o)
        eq.append(e)
    return {"equity": pd.Series(eq, index=p.dates[lo:hi]), "fees": fees, "funding": fund, "traded": 0.0,
            "stops": 0, "gross_max": 1.0, "start_equity": equity}


def stats(res: dict, btc: pd.Series | None = None) -> dict:
    curve = res["equity"]
    r = curve.pct_change().fillna(curve.iloc[0] / res["start_equity"] - 1)
    n = len(r)
    sd = r.std(ddof=1)
    t = r.mean() / sd * np.sqrt(n) if sd > 0 else 0.0
    half = n // 2
    trimmed = r.drop(r.nlargest(5).index)
    out = {
        "days": n,
        "total": curve.iloc[-1] / res["start_equity"] - 1,
        "max_dd": (curve / curve.cummax().clip(lower=res["start_equity"]) - 1).min(),
        "sharpe": r.mean() / sd * np.sqrt(365) if sd > 0 else 0.0,
        "t": t,
        "half1": (1 + r.iloc[:half]).prod() - 1,
        "half2": (1 + r.iloc[half:]).prod() - 1,
        "mean_ex_top5": trimmed.mean(),
        "fees": res["fees"], "funding": res["funding"], "stops": res["stops"], "gross_max": res["gross_max"],
        "turnover_x_per_year": res["traded"] / curve.mean() / n * 365,
        "years": {str(y): (1 + g).prod() - 1 for y, g in r.groupby(r.index.year)},
    }
    if btc is not None:
        b = btc.reindex(r.index)
        m = b.notna()
        out["beta_btc"] = float(np.polyfit(b[m], r[m], 1)[0]) if m.sum() > 10 else float("nan")
    return out


def load(folder: str | Path) -> tuple[dict, dict]:
    folder = Path(folder)
    daily, funding = {}, {}
    for f in sorted(folder.glob("D_*.csv")):
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        df = df[df["volume"] > 0]      # HL back-fills years of zero-volume bars before a listing: not tradable
        if len(df):
            daily[f.stem[2:]] = df
    for f in sorted(folder.glob("F_*.csv")):
        df = pd.read_csv(f, parse_dates=["time"])
        funding[f.stem[2:]] = pd.Series(df["funding"].values, index=pd.DatetimeIndex(df["time"]))
    return daily, funding


def gates(s: dict, stress: dict, perm_p: float | None) -> list[tuple[str, bool]]:
    return [("t > 2.5", s["t"] > 2.5), ("Sharpe > 1", s["sharpe"] > 1),
            ("both halves positive", s["half1"] > 0 and s["half2"] > 0),
            ("positive without the best 5 days", s["mean_ex_top5"] > 0),
            ("|beta to BTC| < 0.2", abs(s.get("beta_btc", np.inf)) < 0.2),
            ("positive with 0.15% slippage", stress["total"] > 0),
            ("shuffled-pick p < 0.05", perm_p is not None and perm_p < 0.05)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="folder written by xs_fetch.py")
    ap.add_argument("--holdout", action="store_true", help="evaluate the hold-out period (only after the dev gates)")
    ap.add_argument("--start", help="forward test: only days from this date (after the hold-out was used up), "
                                     "e.g. 2026-10-17; implies the full data set")
    ap.add_argument("--perm", type=int, default=0, help="shuffled-pick runs for the main variant")
    ap.add_argument("--long-only", action="store_true",
                    help="docs/prereg-long-only-momentum.md: long-only top 3 with the BTC switch, vs BTC benchmarks")
    ap.add_argument("--json", help="write all numbers here")
    a = ap.parse_args(argv)
    daily, funding = load(a.data)
    if a.start:
        start, end, panel_end = a.start, None, None
    elif a.holdout:
        start, end, panel_end = HOLDOUT_START, None, None
    else:
        start, end, panel_end = DEV_START, DEV_END, DEV_END
    p = build_panel(daily, funding, end=panel_end)
    end = end or str(p.dates[-1].date())
    btc = pd.Series(p.c[:, p.coins.index("BTC")], index=p.dates).pct_change() if "BTC" in p.coins else None
    print(f"{len(p.coins)} coins, {len(funding)} with funding, {start} .. {end}\n")
    if a.long_only:
        runs = {LONG_ONLY.name: simulate(p, LONG_ONLY, start, end),
                "same, 0.15% slippage": simulate(p, replace(LONG_ONLY, slip=0.0015), start, end),
                "BTC 1x + same switch": btc_benchmark(p, start, end, regime=True),
                "BTC buy and hold": btc_benchmark(p, start, end, regime=False)}
        rows = {k: stats(r, btc) for k, r in runs.items()}
        on = regime_on(p)[p.dates.searchsorted(pd.Timestamp(start)):p.dates.searchsorted(pd.Timestamp(end), "right")]
        print(f"BTC switch on {on.mean():.0%} of days\n")
        print("| run | total | max DD | Sharpe | t | fees | funding | turnover/yr |")
        print("|---|---|---|---|---|---|---|---|")
        for k, s in rows.items():
            print(f"| {k} | {s['total']:+.1%} | {s['max_dd']:.1%} | {s['sharpe']:.2f} | {s['t']:.2f} | "
                  f"{s['fees']:.1f} | {s['funding']:+.1f} | {s['turnover_x_per_year']:.0f}x |")
        print("\nYearly:", {y: f"{x:+.1%}" for y, x in rows[LONG_ONLY.name]["years"].items()})
        m, b = rows[LONG_ONLY.name], rows["BTC 1x + same switch"]
        print("\nGates (judged on the hold-out only):")
        for label, ok in [("total > 0", m["total"] > 0), ("beats BTC 1x + same switch", m["total"] > b["total"]),
                          ("max DD smaller than 30%", m["max_dd"] > -0.30)]:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if a.json:
            Path(a.json).write_text(json.dumps(rows, default=float, indent=1))
        return
    rows = {}
    for v in VARIANTS:
        rows[v.name] = stats(simulate(p, v, start, end), btc)
    main_v = VARIANTS[0]
    stress = stats(simulate(p, replace(main_v, slip=0.0015), start, end), btc)
    print("| variant | total | max DD | Sharpe | t | 1st half | 2nd half | beta BTC | fees | funding | stops | turnover/yr |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in list(rows.items()) + [("main, 0.15% slippage", stress)]:
        print(f"| {name} | {s['total']:+.1%} | {s['max_dd']:.1%} | {s['sharpe']:.2f} | {s['t']:.2f} | "
              f"{s['half1']:+.1%} | {s['half2']:+.1%} | {s.get('beta_btc', float('nan')):+.2f} | "
              f"{s['fees']:.1f} | {s['funding']:+.1f} | {s['stops']} | {s['turnover_x_per_year']:.0f}x |")
    print("\nYearly (main):", {y: f"{x:+.1%}" for y, x in rows[main_v.name]["years"].items()})
    perm_p, perm = None, []
    if a.perm:
        rng = np.random.default_rng(20261017)
        perm = [stats(simulate(p, main_v, start, end, rng=rng))["sharpe"] for _ in range(a.perm)]
        perm_p = float(np.mean(np.array(perm) >= rows[main_v.name]["sharpe"]))
        print(f"\nShuffled picks ({a.perm} runs): Sharpe median {np.median(perm):.2f}, "
              f"95th pct {np.percentile(perm, 95):.2f}; main {rows[main_v.name]['sharpe']:.2f}, p = {perm_p:.3f}")
    if not a.holdout:
        print("\nDev gates (main variant):")
        for label, ok in gates(rows[main_v.name], stress, perm_p):
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "stress": stress, "perm": perm, "perm_p": perm_p},
                                           default=float, indent=1))


if __name__ == "__main__":
    main()
