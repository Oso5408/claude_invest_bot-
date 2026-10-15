"""Self-test of the AI paper account on made-up market data: does a clear signal reach Claude, does
Claude pick a fitting strategy, does the order open inside the limits, and does the 5-minute check close it.

Two scenarios, each one decision (about 0.05 USD of API):
- breakout-long: SOL breaks its 4h 20-bar high on 2.5x volume (the tested 4h breakout long), other coins flat.
- breakdown-short: ETH breaks its 4h 10-bar low on high volume while Fear & Greed is 75 (greed), to check
  that the AI no longer treats "no shorts above 50" as a rule for every strategy.

Runs in a temporary folder; data/ai-hl and every other account are never touched. Prices are scaled to the
live Hyperliquid prices when reachable so they look normal, but every candle is made up.

    .venv/bin/python -m btcbot.ai_selftest              # real Claude (needs ~/.anthropic_env)
    .venv/bin/python -m btcbot.ai_selftest --no-claude  # mechanics only, a scripted order instead of Claude
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

DEFAULT_PRICES = {"ADA": 0.6, "BTC": 60000.0, "ETH": 3000.0, "SOL": 150.0}
SCENARIOS = {
    "breakout-long": {"coin": "SOL", "side": "long", "move": 0.035, "fear_greed": 40.0},
    "breakdown-short": {"coin": "ETH", "side": "short", "move": -0.035, "fear_greed": 75.0},
}


class ScenarioMarket:
    """Made-up 1m candles for the four coins over the last ~45 days; the signal coin makes its move in the
    last completed 4h candle. Candles of every interval are resampled from the 1m ones."""

    def __init__(self, now: pd.Timestamp, scenario: str, prices: dict[str, float] | None = None, seed: int = 7):
        sc = SCENARIOS[scenario]
        self.sc, self.now = sc, now
        prices = {**DEFAULT_PRICES, **(prices or {})}
        rng = np.random.default_rng(seed)
        bar4 = now.floor("4h") - pd.Timedelta(hours=4)  # the last completed 4h candle
        idx = pd.date_range(now.floor("min") - pd.Timedelta(days=45), now.floor("min"), freq="1min",
                            tz="UTC", name="open_time")
        self.bars: dict[str, pd.DataFrame] = {}
        for coin in ai.COINS:
            steps = rng.normal(0, 0.0004, len(idx))
            logp = np.zeros(len(idx))
            for i in range(1, len(idx)):  # quiet, range-bound market (pulls back to the mean)
                logp[i] = 0.995 * logp[i - 1] + steps[i]
            path = np.exp(logp)
            vol = rng.uniform(0.8, 1.2, len(idx))
            if coin == sc["coin"]:
                move = (idx >= bar4) & (idx < bar4 + pd.Timedelta(hours=4))
                after = idx >= bar4 + pd.Timedelta(hours=4)
                n = int(move.sum())  # the move is a noisy walk, not a straight line, like a real breakout
                walk = np.cumsum(rng.normal(0, 0.0015, n))
                ramp = np.linspace(0, np.log(1 + sc["move"]), n) + walk - np.linspace(0, walk[-1], n)
                logp = np.log(path)
                logp[move] = logp[move][0] + ramp
                logp[after] = logp[move][-1] + np.cumsum(steps[after]) * 0.3  # holds the new level
                path = np.exp(logp)
                vol[move] *= 2.5
            path *= prices[coin] / path[-1]
            close = pd.Series(path, index=idx)
            open_ = close.shift(fill_value=close.iloc[0])
            wick = np.abs(rng.normal(0, 0.0002, len(idx)))
            self.bars[coin] = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wick),
                                            "low": np.minimum(open_, close) * (1 - wick), "close": close,
                                            "volume": vol * 1000}, index=idx)

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
        return []  # no big-account moves, so the signal is the only reason to trade

    def positions(self, address):
        return []


class ScriptedBrain:
    """Stands in for Claude in --no-claude mode: places the textbook 4h breakout order on the signal."""

    def __init__(self, scenario: str):
        self.sc, self.cost = SCENARIOS[scenario], 0.0

    def run(self, context, handlers):
        coin, d = self.sc["coin"], 1 if self.sc["side"] == "long" else -1
        ind = context["market_overview"][coin]["indicators"]["4h"]
        px = context["prices"][coin]
        handlers["place_order"](coin=coin, side=self.sc["side"], margin_usd=20, leverage=3,
                                stop_price=px - d * 2 * ind["atr14"], take_profit_price=px * (1 + d * 0.06),
                                max_hold_hours=96, strategy="4h Donchian breakout", reason="scripted self-test")
        return {"summary": "scripted order", "coins": []}

    def write(self, system, text):
        return "(no review in scripted mode)"


def run_scenario(scenario: str, brain, prices: dict | None = None, now: datetime | None = None) -> dict:
    """One decision on the made-up market, then push the price to the take profit (or past the stop if there
    is none) and run the 5-minute check. Returns what happened at each stage."""
    now = pd.Timestamp(now or datetime.now(timezone.utc)).floor("4h") + pd.Timedelta(minutes=2)
    sc = SCENARIOS[scenario]
    market = ScenarioMarket(now, scenario, prices)
    out = {"scenario": scenario, "signal": f"{sc['coin']} 4h {'breakout up' if sc['side'] == 'long' else 'breakdown'}"
           f" {sc['move']:+.0%} on 2.5x volume, Fear & Greed {sc['fear_greed']:g}"}
    with tempfile.TemporaryDirectory() as tmp:
        t = ai.AITrader(Path(tmp), market, brain)
        t.now = now.to_pydatetime()
        ind = t._overview(market.mids())[sc["coin"]]["indicators"]["4h"]
        ov = t._overview(market.mids())[sc["coin"]]["indicators"]
        out["signal_in_context"] = {k: ind[k] for k in ("close", "donchian20_high", "donchian10_low",
                                                         "volume_ratio20", "trend", "atr14_pct", "rsi14")} \
            | {"rsi14_1h": ov["1h"]["rsi14"], "rsi14_15m": ov["15m"]["rsi14"]}
        status = t.step(now.to_pydatetime())
        dec = pd.read_csv(Path(tmp) / "decisions.csv").iloc[-1].to_dict()
        views = json.loads(dec["coin_views"]) if isinstance(dec.get("coin_views"), str) else []
        out["ai_summary"] = dec.get("summary") or dec.get("error")
        out["ai_view_on_signal_coin"] = next((v for v in views if v.get("coin") == sc["coin"]), None)
        out["skills_read"] = pd.read_csv(Path(tmp) / "skills.csv", keep_default_na=False).iloc[-1]["skills_read"]
        out["api_cost_usd"] = round(t.state.get("api_cost", 0.0), 4)
        pos = list(t.state["positions"])
        out["orders"] = [{"side": "long" if p["side"] > 0 else "short"} | {k: p[k] for k in ("coin", "margin", "leverage", "entry", "stop", "take_profit",
                                            "deadline", "strategy", "reason")} for p in pos]
        out["orders_ok"] = all(p["margin"] <= ai.MAX_MARGIN_USD and p["leverage"] <= ai.MAX_POSITION_LEVERAGE
                               and abs(p["entry"] - p["stop"]) / p["entry"] <= ai.MAX_STOP_PCT for p in pos)
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
    p = argparse.ArgumentParser(description="Self-test the AI paper account on made-up market data")
    p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    p.add_argument("--no-claude", action="store_true", help="scripted order instead of Claude (no API cost)")
    a = p.parse_args(argv)
    try:
        prices = {c: v for c, v in ai.HLMarket().mids().items() if c in ai.COINS}
    except Exception:
        prices = None
    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    for name in names:
        brain = ScriptedBrain(name) if a.no_claude else ai.Brain()
        print(json.dumps(run_scenario(name, brain, prices), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
