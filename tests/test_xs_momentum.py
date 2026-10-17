import numpy as np
import pandas as pd

from btcbot import xs_momentum as xs
from btcbot.hyperliquid import TAKER_FEE

DATES = pd.date_range("2023-01-02", periods=400, freq="D")   # 2023-01-02 is a Monday


def frame(close, vol=1e6, dates=DATES):
    close = np.asarray(close, float)
    o = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) * 1.001, "low": np.minimum(o, close) * 0.999,
                         "close": close, "volume": vol}, index=dates)


def walks(n_coins=12, drift=None, seed=0, sd=0.03, dates=DATES):
    rng = np.random.default_rng(seed)
    drift = np.zeros(n_coins) if drift is None else drift
    return {f"C{k:02d}": frame(100 * np.exp(np.cumsum(drift[k] + sd * rng.standard_normal(len(dates)))),
                               dates=dates) for k in range(n_coins)}


def test_persistent_winners_make_money():
    drift = np.linspace(-0.01, 0.01, 12)
    p = xs.build_panel(walks(drift=drift), {})
    s = xs.stats(xs.simulate(p, xs.VARIANTS[0], "2023-03-15", "2024-02-01"))
    assert s["total"] > 0.5 and s["t"] > 2


def test_random_walk_has_no_edge():
    dates = pd.date_range("2020-01-06", periods=1500, freq="D")
    p = xs.build_panel(walks(n_coins=30, seed=3, dates=dates), {})
    s = xs.stats(xs.simulate(p, xs.Variant("rw", slip=0.0), "2020-04-01", "2024-02-01"))
    assert abs(s["t"]) < 3


def test_targets_only_use_past_data():
    data = walks(seed=1)
    p = xs.build_panel(data, {})
    i = 200
    before = xs.targets(p, i, xs.VARIANTS[0])
    changed = {}
    for k, df in data.items():
        df = df.copy()
        df.iloc[i:, df.columns.get_loc("close")] *= np.random.default_rng(9).uniform(0.5, 2, len(df) - i)
        df.iloc[i:, df.columns.get_loc("volume")] *= 1000 if k == "C00" else 1
        changed[k] = df
    after = xs.targets(xs.build_panel(changed, {}), i, xs.VARIANTS[0])
    assert before and before.keys() == after.keys()
    assert all(abs(before[j] - after[j]) < 1e-12 for j in before)
    assert abs(sum(abs(w) for w in before.values()) - xs.MAX_GROSS) < 1e-9


def test_end_drops_holdout_rows():
    p = xs.build_panel(walks(), {}, end="2023-06-30")
    assert p.dates[-1] == pd.Timestamp("2023-06-30")


WIGGLE = 1 + 1e-4 * (np.arange(len(DATES)) % 2)   # almost flat; sigma must be > 0 to be eligible


def test_flat_prices_cost_only_the_entry_fees():
    t = np.arange(len(DATES))
    flat = {f"C{k:02d}": frame((100.0 + k) * WIGGLE * np.exp(k * 1e-7 * t)) for k in range(10)}   # no ties
    for k, df in flat.items():
        df["high"] = df["low"] = df["close"]
    p = xs.build_panel(flat, {})
    res = xs.simulate(p, xs.VARIANTS[0], "2023-04-01", "2023-12-31")
    expected = 200 - 200 * xs.MAX_GROSS * (TAKER_FEE + xs.SLIP)
    assert abs(res["equity"].iloc[-1] - expected) < 0.05
    assert res["stops"] == 0


def test_funding_variant_collects_funding_on_both_sides():
    flat = {f"C{k:02d}": frame(100.0 * WIGGLE) for k in range(10)}
    for df in flat.values():
        df["high"] = df["low"] = df["close"]
    hours = pd.date_range(DATES[0], DATES[-1] + pd.Timedelta(hours=23), freq="h")
    fund = {k: pd.Series((n - 4.5) * 1e-5, index=hours) for n, k in enumerate(flat)}   # C00 lowest .. C09 highest
    p = xs.build_panel(flat, fund)
    tgt = xs.targets(p, 100, xs.VARIANTS[5])
    assert {p.coins[j] for j, w in tgt.items() if w > 0} == {"C00", "C01"}
    assert {p.coins[j] for j, w in tgt.items() if w < 0} == {"C08", "C09"}
    res = xs.simulate(p, xs.VARIANTS[5], "2023-04-01", "2023-12-31")
    assert res["funding"] < 0 and res["equity"].iloc[-1] > 200


def test_short_squeeze_hits_the_stop():
    data = walks(seed=2, sd=0.01)
    df = data["C05"]
    jump = df.index.get_loc(pd.Timestamp("2023-06-07"))
    close = df["close"].values.copy()
    close[:jump] = close[:jump] * np.linspace(1.0, 0.5, jump)   # weakest coin into the short leg ...
    close[jump:] = close[jump - 1] * 2                          # ... then doubles
    data["C05"] = frame(close)
    p = xs.build_panel(data, {})
    res = xs.simulate(p, xs.VARIANTS[0], "2023-04-01", "2023-07-01")
    assert res["stops"] >= 1


def with_btc(data, btc_close):
    data = dict(data)
    data["BTC"] = frame(btc_close)
    return data


def test_long_only_weights_and_biweekly_schedule():
    p = xs.build_panel(with_btc(walks(seed=4), 100 * np.exp(np.linspace(0, 1, len(DATES)))), {})
    tgt = xs.targets(p, 200, xs.LONG_ONLY)
    assert len(tgt) == 3 and all(w > 0 for w in tgt.values())
    assert abs(sum(tgt.values()) - 1.0) < 1e-9
    mondays = [d for d in DATES if xs.rebalance_day(xs.LONG_ONLY, d)]
    assert all(d.weekday() == 0 for d in mondays)
    assert all((b - a).days == 14 for a, b in zip(mondays, mondays[1:]))


def test_regime_off_means_cash():
    falling = 100 * np.exp(np.linspace(0, -1, len(DATES)))      # BTC always below its 50-day average
    p = xs.build_panel(with_btc(walks(seed=5), falling), {})
    res = xs.simulate(p, xs.LONG_ONLY, "2023-04-01", "2023-12-31")
    assert res["traded"] == 0 and res["equity"].iloc[-1] == 200
    rising = 100 * np.exp(np.linspace(0, 1, len(DATES)))
    p = xs.build_panel(with_btc(walks(seed=5), rising), {})
    assert xs.simulate(p, xs.LONG_ONLY, "2023-04-01", "2023-12-31")["traded"] > 0


def test_btc_benchmark_follows_btc():
    rising = 100 * np.exp(np.linspace(0, 1, len(DATES)))
    p = xs.build_panel(with_btc(walks(seed=6), rising), {})
    res = xs.btc_benchmark(p, "2023-04-01", "2023-12-31", regime=False)
    i0, i1 = p.dates.get_loc(pd.Timestamp("2023-04-01")), p.dates.get_loc(pd.Timestamp("2023-12-31"))
    btc = p.coins.index("BTC")
    gross = p.c[i1, btc] / p.o[i0, btc]
    assert abs(res["equity"].iloc[-1] / 200 / gross - 1) < 0.002
