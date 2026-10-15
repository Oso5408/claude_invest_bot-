"""Self-test of the rule-based Hyperliquid paper accounts on made-up ADA prices: does a 4h breakout make
each account enter (or correctly stay out), does the 2x ATR trailing stop follow the trend, and does the
5-minute check close the trade on it.

The accounts use the same settings as the VM crontab (docs/gcp-setup.md):
- paper-hl-2x-fng: both sides, up to 2x, shorts only when Fear & Greed <= 50
- paper-hl-1x-long: long only, 1x
Three made-up markets, each run through the real PaperTrader every 30 minutes, reloaded from disk like cron:
- breakout-long: ADA breaks its 4h 20-bar high on 3x volume, trends up 24h, then drops back
- breakdown-short-fear: ADA breaks its 4h 10-bar low with Fear & Greed 30, falls 24h, then bounces
- breakdown-short-greed: the same fall with Fear & Greed 75 (the 2x-fng account must not short)

No network, no API, temporary folders only; data/ is never touched.

    .venv/bin/python -m btcbot.rule_selftest
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from btcbot.hyperliquid import account_config
from btcbot.paper import PaperConfig, PaperTrader

ACCOUNTS = {
    "paper-hl-2x-fng": dict(allow_short=True, max_leverage=2.0, fng_short_max=50),
    "paper-hl-1x-long": dict(allow_short=False, max_leverage=1.0),
}
SCENARIOS = {
    "breakout-long": {"side": 1, "fear_greed": 40.0},
    "breakdown-short-fear": {"side": -1, "fear_greed": 30.0},
    "breakdown-short-greed": {"side": -1, "fear_greed": 75.0},
}
EXPECTED = {  # which side each account should take in each market (0 = stay out)
    ("paper-hl-2x-fng", "breakout-long"): 1, ("paper-hl-1x-long", "breakout-long"): 1,
    ("paper-hl-2x-fng", "breakdown-short-fear"): -1, ("paper-hl-1x-long", "breakdown-short-fear"): 0,
    ("paper-hl-2x-fng", "breakdown-short-greed"): 0, ("paper-hl-1x-long", "breakdown-short-greed"): 0,
}
START = pd.Timestamp("2026-09-01", tz="UTC")


def scenario_minutes(scenario: str, seed: int = 3) -> pd.DataFrame:
    """6 quiet days around 0.60 USD, then a 4h candle that breaks out (3x volume), a 24h trend the same way
    (+15% / -15%) and a 12h reversal of 12%."""
    d = SCENARIOS[scenario]["side"]
    rng = np.random.default_rng(seed)
    n_flat = 60 * 24 * 6
    logp = np.zeros(n_flat)
    steps = rng.normal(0, 0.0004, n_flat)
    for i in range(1, n_flat):
        logp[i] = 0.995 * logp[i - 1] + steps[i]
    flat = 0.6 * np.exp(logp)
    brk = flat[-1] * np.exp(np.linspace(0, np.log(1 + d * 0.04), 240) + np.cumsum(rng.normal(0, 0.001, 240)) * 0.3)
    trend = brk[-1] * np.exp(np.linspace(0, np.log(1 + d * 0.15), 60 * 24) + rng.normal(0, 0.002, 60 * 24))
    back = trend[-1] * np.exp(np.linspace(0, np.log(1 - d * 0.12), 60 * 12))
    close = np.concatenate([flat, brk, trend, back])
    vol = rng.uniform(8, 12, len(close))
    vol[n_flat:n_flat + 240] *= 3
    idx = pd.date_range(START, periods=len(close), freq="min", name="open_time")
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0002,
                         "low": np.minimum(open_, close) * 0.9998, "close": close, "volume": vol}, index=idx)


class ScenarioFeed:
    """Serves candles cut off at `now` like the live Hyperliquid API, plus funding and Fear & Greed."""

    has_funding = True

    def __init__(self, minutes: pd.DataFrame, fear_greed: float):
        self.m, self.fg, self.now = minutes, fear_greed, None

    def candles(self, interval, now, days):
        self.now = pd.Timestamp(now)
        m = self.m[self.m.index < self.now]
        rule = {"1min": "1min", "1hour": "1h"}[interval]
        return m.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

    def bid_ask(self):
        last = self.m[self.m.index < self.now]["close"].iloc[-1]
        return last * 0.9998, last * 1.0002

    def funding(self, start, end):
        return pd.Series(0.00001, index=pd.date_range(start.ceil("1h"), end, freq="1h"))

    def fng(self):
        days = pd.date_range(START - pd.Timedelta(days=2), self.now.floor("D"), freq="D")
        return pd.Series(self.fg, index=days)


def run_account(account: str, scenario: str) -> dict:
    m = scenario_minutes(scenario)
    feed = ScenarioFeed(m, SCENARIOS[scenario]["fear_greed"])
    cfg = PaperConfig(symbol="ADA", initial_jpy=200.0, history_days=6, **ACCOUNTS[account])
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for t in pd.date_range(START + pd.Timedelta(days=3), m.index[-1], freq="30min"):
            PaperTrader(folder, cfg, account_config(200.0)).step(feed, t.to_pydatetime())  # reload like cron
        sig = pd.read_csv(folder / "signals.csv")
        eq = pd.read_csv(folder / "equity.csv")
        trades = pd.read_csv(folder / "trades.csv") if (folder / "trades.csv").exists() else pd.DataFrame()
    entered = sig[sig["result"] == "entered"]
    side = int(trades["side"].iloc[0]) if len(trades) else 0
    out = {"account": account, "scenario": scenario, "expected_side": EXPECTED[(account, scenario)],
           "taken_side": side,
           "signals_seen": int((sig["long_signal"] | sig["short_signal"]).sum()),
           "entries": len(entered),
           "stop_moves": int(eq["action"].astype(str).str.startswith("trail stop").sum()),
           "max_position_x": round(float((eq["qty"].abs() * eq["price"] / eq["equity"]).max()), 2),
           "trades": trades[["side", "entry_px", "exit_px", "return", "reason"]].round(4).to_dict(orient="records")
           if len(trades) else []}
    ok = side == out["expected_side"] and out["max_position_x"] <= ACCOUNTS[account]["max_leverage"] * 1.05
    if side:
        t = trades.iloc[0]
        ok &= t["reason"] == "trailing_stop" and t["return"] > 0 and out["stop_moves"] > 0
    out["ok"] = bool(ok)
    return out


def main() -> None:
    results = [run_account(a, s) for s in SCENARIOS for a in ACCOUNTS]
    for r in results:
        print(json.dumps(r, ensure_ascii=False, default=str))
    print("ALL OK" if all(r["ok"] for r in results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
