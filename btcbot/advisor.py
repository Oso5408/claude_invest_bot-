"""Claude as a second opinion on the paper trader, plus a daily review.

1. Entry check (`ClaudeAdvisor.decide`): when the rules want to open a trade, Claude sees
   the signal, recent 4h candles, the Fear & Greed Index and the account, may search
   the web for news, and answers through the `decide` tool:
     go   - take the trade as the rules sized it
     half - take it at half size
     skip - don't take it
   Claude can only shrink or cancel a trade. It never opens one on its own, never
   makes one bigger, and the 2x cap in risk.py still applies to whatever it returns.
   If the call fails for any reason the rules are followed unchanged and the error is
   logged, so a network problem never changes what the account does.

2. Daily review (`python -m btcbot.advisor review`): Claude reads every paper account's
   trades and equity and writes a short report to data/reviews/YYYY-MM-DD.md. It
   changes nothing.

Needs ANTHROPIC_API_KEY in the environment (see docs/gcp-setup.md).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODEL = "claude-opus-5-5"
MULTIPLIER = {"go": 1.0, "half": 0.5, "skip": 0.0}

DECIDE_TOOL = {
    "name": "decide",
    "description": "Record your decision on the proposed trade. Call this exactly once, at the end.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["go", "half", "skip"],
                       "description": "go = full size, half = half size, skip = no trade"},
            "reason": {"type": "string", "description": "One or two sentences, plain English."},
        },
        "required": ["action", "reason"],
        "additionalProperties": False,
    },
}
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}

ENTRY_SYSTEM = """You review trades for a small paper-trading account (fake money) trading ADA with up to 2x \
leverage (GMO Coin's ADA_JPY crypto FX, or the ADA perpetual on Hyperliquid in USD; the venue says which). \
A rule-based 4h Donchian breakout strategy has just produced a signal. Your job is to decide \
whether the account should take it at full size, half size, or not at all.

Context on the strategy, from a 2.3-year backtest: it wins about 40% of trades, and its profit comes from \
a few large trend moves caught by a trailing stop, so most skipped trades are small losers but skipping \
the wrong one loses a big winner. Skipping too often is costly. Prefer "go" unless you see a concrete \
reason (for example scheduled news that could reverse the move, a breakout on very thin volume, or the \
move already being exhausted). Use "half" when the case is mixed.

You may search the web for recent ADA or crypto-market news before deciding. Then call the decide tool \
once. Do not invent prices; use the numbers you are given."""

REVIEW_SYSTEM = """You write a short daily review of a paper-trading experiment (fake money) for its owner, \
who reads Traditional Chinese (Hong Kong style). Several accounts trade ADA with the same 4h breakout \
strategy and different settings, so they can be compared: accounts named paper-hl-* use Hyperliquid prices \
in USD (200 USD start), the others GMO Coin prices in JPY (30,000 JPY start). Write in Traditional Chinese \
(Hong Kong style), plain sentences, under 300 words: how each account did since the last review and in \
total, anything unusual (losses bigger than the stop should allow, errors, no runs for hours), and whether \
the results so far differ from the backtest. Be honest that a few trades prove nothing. Do not recommend \
real-money trading."""


@dataclass
class Decision:
    action: str
    reason: str
    multiplier: float
    error: str | None = None


class ClaudeAdvisor:
    def __init__(self, model: str = MODEL, client=None, web_search: bool = True):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.tools = [DECIDE_TOOL] + ([WEB_SEARCH_TOOL] if web_search else [])

    def _create(self, system: str, messages: list, tools: list | None):
        kwargs = dict(model=self.model, max_tokens=16000, system=system, messages=messages,
                      output_config={"effort": "medium"},
                      betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if tools:
            kwargs["tools"] = tools
        return self.client.beta.messages.create(**kwargs)

    def decide(self, context: dict) -> Decision:
        try:
            return self._decide(context)
        except Exception as e:  # any failure: follow the rules unchanged
            return Decision("go", "Claude unavailable, following the rules", 1.0, f"{type(e).__name__}: {e}")

    def _decide(self, context: dict) -> Decision:
        messages = [{"role": "user", "content": "Proposed trade and market context (JSON):\n"
                                                + json.dumps(context, indent=1, default=str)}]
        for _ in range(6):
            resp = self._create(ENTRY_SYSTEM, messages, self.tools)
            if resp.stop_reason == "refusal":
                raise RuntimeError("request declined")
            for block in resp.content:
                if block.type == "tool_use" and block.name == "decide":
                    action = block.input["action"]
                    return Decision(action, block.input["reason"], MULTIPLIER[action])
            if resp.stop_reason != "pause_turn":
                raise RuntimeError(f"no decision (stop_reason={resp.stop_reason})")
            messages.append({"role": "assistant", "content": resp.content})  # resume the paused search turn
        raise RuntimeError("no decision after repeated pauses")

    def review(self, summary: str) -> str:
        messages = [{"role": "user", "content": summary}]
        resp = self._create(REVIEW_SYSTEM, messages, None)
        if resp.stop_reason == "refusal":
            raise RuntimeError("request declined")
        return "".join(b.text for b in resp.content if b.type == "text")


def entry_context(bars: pd.DataFrame, row: pd.Series, direction: int, leverage: float, price: float,
                  equity: float, fng: pd.Series | None, trades_path: Path, symbol: str = "ADA_JPY") -> dict:
    """What Claude sees for one signal. Only data known at the decision time."""
    recent = bars.tail(30)[["open", "high", "low", "close", "volume"]].round(4)
    recent.index = recent.index.strftime("%Y-%m-%d %H:%M UTC")
    ctx = {
        "symbol": symbol, "venue": "Hyperliquid (USD)" if symbol == "ADA" else "GMO Coin (JPY)", "timeframe": "4h",
        "signal": "long breakout" if direction > 0 else "short breakdown",
        "rule_details": {"close": row["close"], "donchian_high_20": row["donchian_high"],
                         "donchian_low_10": row["donchian_low"], "volume": row["volume"],
                         "volume_avg_20": row["volume_ma"], "atr_14": row["atr"],
                         "initial_stop": price - direction * 2 * row["atr"]},
        "entry_price_now": price,
        "rules_leverage": round(leverage, 3),
        "account_equity": round(equity, 2),
        "last_30_candles_4h": recent.to_dict(orient="index"),
    }
    if fng is not None and len(fng):
        ctx["fear_greed_last_7_days"] = {d.strftime("%Y-%m-%d"): v for d, v in fng.tail(7).items()}
    if trades_path.exists():
        t = pd.read_csv(trades_path).tail(5)
        ctx["last_5_trades"] = t[["entry_time", "exit_time", "side", "return", "reason"]].to_dict(orient="records")
    return ctx


def account_summary(folder: Path, since: pd.Timestamp) -> str:
    lines = [f"## {folder.name}"]
    eq_path, tr_path, cl_path = folder / "equity.csv", folder / "trades.csv", folder / "claude.csv"
    if eq_path.exists():
        eq = pd.read_csv(eq_path, parse_dates=["time"])
        last = eq.iloc[-1]
        day = eq[eq["time"] >= since]
        start = day["equity"].iloc[0] if len(day) else last["equity"]
        lines.append(f"runs logged: {len(eq)}, last run {last['time']}, equity now {last['equity']}, "
                     f"24h ago {start}, position qty {last['qty']} side {last['side']} "
                     f"stop {last['stop']}")
        errors = day[day["action"].astype(str).str.contains("unavailable|error", case=False)]
        if len(errors):
            lines.append(f"errors in last 24h: {len(errors)}, e.g. {errors['action'].iloc[-1]}")
    if tr_path.exists():
        t = pd.read_csv(tr_path)
        lines.append(f"closed trades: {len(t)}, win rate {(t['return'] > 0).mean():.0%}, "
                     f"avg return {t['return'].mean():+.2%}")
        lines.append(t.tail(10).to_csv(index=False))
    if cl_path.exists():
        lines.append("Claude entry decisions:\n" + pd.read_csv(cl_path).tail(10).to_csv(index=False))
    return "\n".join(lines)


def daily_review(data_dir: Path, advisor: ClaudeAdvisor, now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    since = pd.Timestamp(now) - pd.Timedelta(days=1)
    folders = sorted(p for p in data_dir.glob("paper-*") if p.is_dir())
    summary = (f"Review time: {now:%Y-%m-%d %H:%M} UTC.\nBacktest reference (2024-05 to 2026-10, 30,000 JPY start): "
               "4h breakout 2x both sides +151%, max drawdown -42%; with shorts only when Fear & Greed <= 50: "
               "+886%, max drawdown -48%; 1x long only +187%, max drawdown -27%.\n\n"
               + "\n\n".join(account_summary(f, since) for f in folders))
    out = data_dir / "reviews" / f"{now:%Y-%m-%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(advisor.review(summary))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Claude daily review of the paper accounts")
    p.add_argument("command", choices=["review"])
    p.add_argument("--data", default="data")
    a = p.parse_args(argv)
    out = daily_review(Path(a.data), ClaudeAdvisor(web_search=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
