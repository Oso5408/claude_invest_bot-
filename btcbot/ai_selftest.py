"""Exam for the AI paper account on made-up markets: can Claude spot a signal that is not obvious, pick the
strategy that fits, and stay out of traps that only look like signals?

Seven scenarios, one decision each (about 0.05 USD of API each, about 0.35 USD in all). In each one every
coin trades in a quiet range and at most one coin has something going on, close to the edge of the rules:
- quiet-breakout-long: NEAR closes 0.6% above its 4h 20-bar high on 1.6x volume, Fear & Greed 45 -> long
- breakdown-short-fear: ADA closes 0.6% below its 4h 10-bar low on normal volume, Fear & Greed 28 -> short
- breakdown-short-greed: BTC does the same with Fear & Greed 72 -> no trade (the tested short filter)
- fake-breakout-wick: SOL spikes 1% above its 20-bar high on 2x volume but closes back inside -> no trade
- breakout-low-volume: ETH closes 0.6% above its 20-bar high on 0.8x volume -> no trade (needs 1.2x)
- quiet-market: nothing anywhere -> no trade
- copy-trader-opens: one of the five copy traders opens a HYPE long, the market is quiet -> long HYPE (copy)
The scenario name is never shown to Claude; it only sees the market, like a normal hourly decision.

A scenario passes when Claude's order (or no order) on the signal coin matches the expected answer and it
trades nothing else. When it does open a trade, the price is then walked to its take profit or stop and the
5-minute check must close it. Runs in a temporary folder; data/ai-hl and every other account are never touched.
Prices are scaled to the live Hyperliquid prices when reachable so they look normal; every candle is made up.

    .venv/bin/python -m btcbot.ai_selftest              # real Claude (needs ~/.anthropic_env)
    .venv/bin/python -m btcbot.ai_selftest --no-claude  # rule follower instead of Claude (no API cost)
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from btcbot import ai_trader as ai

DEFAULT_PRICES = {"ADA": 0.6, "BTC": 60000.0, "ETH": 3000.0, "SOL": 150.0, "HYPE": 40.0, "NEAR": 2.5}
# kind: what the signal coin's last completed 4h candle does; margin: how far past the channel it closes
SCENARIOS = {
    "quiet-breakout-long": {"coin": "NEAR", "kind": "breakout", "margin": 0.006, "volume": 1.6, "fear_greed": 45.0,
                            "expect": "long"},
    "breakdown-short-fear": {"coin": "ADA", "kind": "breakdown", "margin": 0.006, "volume": 1.1, "fear_greed": 28.0,
                             "expect": "short"},
    "breakdown-short-greed": {"coin": "BTC", "kind": "breakdown", "margin": 0.006, "volume": 1.1, "fear_greed": 72.0,
                              "expect": "no trade"},
    "fake-breakout-wick": {"coin": "SOL", "kind": "wick", "margin": 0.01, "volume": 2.0, "fear_greed": 45.0,
                           "expect": "no trade"},
    "breakout-low-volume": {"coin": "ETH", "kind": "breakout", "margin": 0.006, "volume": 0.8, "fear_greed": 45.0,
                            "expect": "no trade"},
    "quiet-market": {"coin": None, "kind": "none", "fear_greed": 50.0, "expect": "no trade"},
    "copy-trader-opens": {"coin": "HYPE", "kind": "copy", "fear_greed": 50.0, "expect": "long"},
}
COPY_SIZE_USD = 250_000.0


def _bridge(rng, n: int, sd: float) -> np.ndarray:
    """Random wiggle that starts and ends at 0, so a move stays noisy but lands where it should."""
    w = np.cumsum(rng.normal(0, sd, n))
    return w - np.linspace(0, w[-1], n)


class ScenarioMarket:
    """Made-up 1m candles for every coin over the last ~45 days, all in a quiet range. The signal coin's last
    completed 4h candle is rebuilt so it closes just past (or, for the traps, just short of) the channel the
    rules use. Candles of every interval are resampled from the 1m ones."""

    def __init__(self, now: pd.Timestamp, scenario: str, prices: dict[str, float] | None = None, seed: int = 7):
        sc = SCENARIOS[scenario]
        self.sc, self.now = sc, now
        prices = {**DEFAULT_PRICES, **(prices or {})}
        rng = np.random.default_rng(seed)
        bar4 = now.floor("4h") - pd.Timedelta(hours=4)  # the last completed 4h candle
        idx = pd.date_range(now.floor("min") - pd.Timedelta(days=45), now.floor("min"), freq="1min",
                            tz="UTC", name="open_time")
        last = idx >= bar4
        self.bars: dict[str, pd.DataFrame] = {}
        for coin in ai.COINS:
            steps = rng.normal(0, 0.0004, len(idx))
            logp = np.zeros(len(idx))
            for i in range(1, len(idx)):  # quiet, range-bound market (pulls back to the mean)
                logp[i] = 0.995 * logp[i - 1] + steps[i]
            logp[last] *= 0.3  # the other coins end mid-range, so they never show a signal by chance
            vol = rng.uniform(0.8, 1.2, len(idx))
            if coin == sc["coin"] and sc["kind"] in ("breakout", "breakdown", "wick"):
                logp, vol = self._signal_bar(rng, idx, bar4, logp, vol, sc)
            path = np.exp(logp)
            path *= prices[coin] / path[-1]
            close = pd.Series(path, index=idx)
            open_ = close.shift(fill_value=close.iloc[0])
            wick = np.abs(rng.normal(0, 0.0002, len(idx)))
            self.bars[coin] = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wick),
                                            "low": np.minimum(open_, close) * (1 - wick), "close": close,
                                            "volume": vol * 1000}, index=idx)
        self.copy_positions: dict[str, list[dict]] = {}
        if sc["kind"] == "copy":
            px = self.mids()[sc["coin"]]
            self.copy_positions[ai.COPY_TRADERS[0]] = [{"coin": sc["coin"], "size": COPY_SIZE_USD / px, "entry": px,
                                                        "value_usd": COPY_SIZE_USD}]

    @staticmethod
    def _signal_bar(rng, idx, bar4, logp, vol, sc):
        """Rebuild the last completed 4h candle relative to the 20-bar high / 10-bar low before it."""
        prior = pd.Series(logp, index=idx)[idx < bar4]
        hi4 = prior.resample("4h", origin="epoch").max().iloc[-20:].max()
        lo4 = prior.resample("4h", origin="epoch").min().iloc[-10:].min()
        bar = (idx >= bar4) & (idx < bar4 + pd.Timedelta(hours=4))
        after = idx >= bar4 + pd.Timedelta(hours=4)
        n, start = int(bar.sum()), prior.iloc[-1]
        if sc["kind"] == "breakout":
            path = np.linspace(start, hi4 + np.log(1 + sc["margin"]), n)
        elif sc["kind"] == "breakdown":
            path = np.linspace(start, lo4 + np.log(1 - sc["margin"]), n)
        else:  # wick: spike above the high, then close back inside the channel
            top, k = hi4 + np.log(1 + sc["margin"]), int(n * 0.6)
            path = np.concatenate([np.linspace(start, top, k), np.linspace(top, hi4 + np.log(0.996), n - k)])
        logp = logp.copy()
        logp[bar] = path + _bridge(rng, n, 0.0008)
        logp[after] = logp[bar][-1] + np.cumsum(rng.normal(0, 0.0002, int(after.sum())))
        vol = vol.copy()
        vol[bar] *= sc["volume"]
        return logp, vol

    def extend(self, coin: str, to_price: float, minutes: int = 30) -> None:
        """Add 1m candles after now that walk to to_price (to test the stop / take-profit check)."""
        df = self.bars[coin]
        last = df.index[-1]
        idx = pd.date_range(last + pd.Timedelta(minutes=1), periods=minutes, freq="1min", tz="UTC", name="open_time")
        close = np.linspace(df["close"].iloc[-1], to_price, minutes + 1)[1:]
        open_ = np.concatenate([[df["close"].iloc[-1]], close[:-1]])
        new = pd.DataFrame({"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close),
                            "close": close, "volume": 1000.0}, index=idx)
        self.bars[coin] = pd.concat([df, new])

    def mids(self) -> dict[str, float]:
        return {c: float(df["close"].iloc[-1]) for c, df in self.bars.items()}

    def candles(self, coin, interval, start, end):
        df = self.bars[coin]
        if interval != "1m":
            df = df.resample(ai.INTERVAL[interval], origin="epoch", label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        return df[(df.index >= start) & (df.index <= end)]

    def funding(self, coin, start, end):
        return pd.Series(0.00001, index=pd.date_range(start.ceil("1h"), end, freq="1h", tz="UTC"))

    def fear_greed(self):
        day = self.now.normalize()
        return {str((day - pd.Timedelta(days=i)).date()): self.sc["fear_greed"] for i in range(13, -1, -1)}

    def leaderboard(self):
        return []  # no leaderboard moves, so the scenario's signal is the only thing going on

    def positions(self, address):
        return list(self.copy_positions.get(address, []))


class ScriptedBrain:
    """Stands in for Claude in --no-claude mode and follows the tested rules exactly: 4h breakout long (close
    above the 20-bar high on at least 1.2x volume), short below the 10-bar low when Fear & Greed <= 50, and
    copy a copy trader's new position. A correct exam scores 7 of 7 with it."""

    def __init__(self, scenario: str | None = None):
        self.cost = 0.0

    def run(self, context, handlers):
        fg = list(context["fear_greed_last_7_days"].values())[-1]
        views = []
        for coin, ov in context["market_overview"].items():
            ind, px, side = ov["indicators"]["4h"], context["prices"][coin], None
            if ind["close"] > ind["donchian20_high"] and ind["volume_ratio20"] >= 1.2:
                side = "long"
            elif ind["close"] < ind["donchian10_low"] and fg <= 50:
                side = "short"
            for a in context["copy_traders"]["alerts_now"]:
                if a["coin"] == coin and a["change"].startswith("opened"):
                    side = a["change"].split()[-1]
            if side:
                d = 1 if side == "long" else -1
                handlers["place_order"](coin=coin, side=side, margin_usd=20, leverage=2,
                                        stop_price=px - d * 2 * ind["atr14"], take_profit_price=0, max_hold_hours=0,
                                        strategy="rules", reason="scripted", trailing_atr_mult=2)
            views.append({"coin": coin, "view": side or "no trade", "strategy": "rules", "reason": ""})
        return {"summary": "scripted", "coins": views}

    def write(self, system, text):
        return "(no review in scripted mode)"


def run_scenario(scenario: str, brain, prices: dict | None = None, now: datetime | None = None) -> dict:
    """One decision on the made-up market, graded against the expected answer. If a trade opens, the price
    is walked to its take profit (or through its stop) and the 5-minute check must close it."""
    now = pd.Timestamp(now or datetime.now(timezone.utc)).floor("4h") + pd.Timedelta(minutes=2)
    sc = SCENARIOS[scenario]
    market = ScenarioMarket(now, scenario, prices)
    out = {"scenario": scenario, "signal_coin": sc["coin"], "expected": sc["expect"]}
    with tempfile.TemporaryDirectory() as tmp:
        t = ai.AITrader(Path(tmp), market, brain)
        t.now = now.to_pydatetime()
        if sc["kind"] == "copy":  # the copy traders were flat at the previous run
            t.state["copy_snapshot"] = {a: {} for a in t.copy_traders}
        if sc["coin"]:
            ind = t._overview(market.mids())[sc["coin"]]["indicators"]["4h"]
            out["market_seen"] = {k: ind[k] for k in ("close", "donchian20_high", "donchian10_low",
                                                      "volume_ratio20", "trend", "rsi14")} | {"fear_greed": sc["fear_greed"]}
        status = t.step(now.to_pydatetime())
        dec = pd.read_csv(Path(tmp) / "decisions.csv").iloc[-1].to_dict()
        views = json.loads(dec["coin_views"]) if isinstance(dec.get("coin_views"), str) else []
        pos = list(t.state["positions"])
        taken = {p["coin"]: "long" if p["side"] > 0 else "short" for p in pos}
        got = taken.get(sc["coin"], "no trade")
        out["got"] = got
        out["other_trades"] = sorted(c for c in taken if c != sc["coin"])
        out["passed"] = got == sc["expect"] and not out["other_trades"]
        out["ai_view_on_signal_coin"] = next((v for v in views if v.get("coin") == sc["coin"]), None)
        out["ai_summary"] = dec.get("summary") if isinstance(dec.get("summary"), str) else dec.get("error")
        out["orders"] = [{"coin": p["coin"], "side": taken[p["coin"]]} | {k: p[k] for k in (
            "margin", "leverage", "entry", "stop", "take_profit", "trail_atr", "strategy", "reason")} for p in pos]
        out["orders_ok"] = all(p["margin"] <= ai.MAX_MARGIN_USD and p["leverage"] <= ai.MAX_POSITION_LEVERAGE
                               and abs(p["entry"] - p["stop"]) / p["entry"] <= ai.MAX_STOP_PCT for p in pos)
        out["skills_read"] = pd.read_csv(Path(tmp) / "skills.csv", keep_default_na=False).iloc[-1]["skills_read"]
        out["api_cost_usd"] = round(t.state.get("api_cost", 0.0), 4)
        out["trades_left_today"] = status["trades_left_today"]
        if pos:  # walk the price to the take profit, or through the stop, and let the 5-minute check act
            p = pos[0]
            target = p["take_profit"] * (1 + p["side"] * 0.002) if p["take_profit"] else \
                p["stop"] * (1 - p["side"] * 0.002)
            market.extend(p["coin"], target)
            t.brain = None  # the follow-up run only manages trades, like a run between decision hours
            t.step((now + pd.Timedelta(minutes=35)).to_pydatetime())
            trades = Path(tmp) / "trades.csv"
            out["closed"] = pd.read_csv(trades)[["coin", "side", "strategy", "exit_reason", "pnl_usd"]] \
                .to_dict(orient="records") if trades.exists() else []
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Exam the AI paper account on made-up markets")
    p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    p.add_argument("--no-claude", action="store_true", help="rule follower instead of Claude (no API cost)")
    a = p.parse_args(argv)
    try:
        prices = {c: v for c, v in ai.HLMarket().mids().items() if c in ai.COINS}
    except Exception:
        prices = None
    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    results = []
    for n, name in enumerate(names, 1):
        print(f"[{n}/{len(names)}] {name} ...", flush=True)
        brain = ScriptedBrain() if a.no_claude else ai.Brain()
        results.append(run_scenario(name, brain, prices))
        print(json.dumps(results[-1], indent=2, ensure_ascii=False, default=str), flush=True)
    print("\nSCORE")
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['scenario']:<24} expected {r['expected']:<9} got {r['got']}"
              + (f" (also traded {', '.join(r['other_trades'])})" if r["other_trades"] else ""))
    print(f"  {sum(r['passed'] for r in results)}/{len(results)} passed, API cost "
          f"{sum(r['api_cost_usd'] for r in results):.2f} USD")


if __name__ == "__main__":
    main()
