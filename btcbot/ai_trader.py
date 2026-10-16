"""AI-run paper account (fake money): Claude picks its own trades on live Hyperliquid prices.

An experiment next to the rule-based paper accounts, to see whether an AI that chooses its own
strategy each day does better or worse than the 4h breakout. Unlike btcbot/advisor.py, here Claude
may open trades by itself. The hard limits below are module constants, checked in code on every
order, so no prompt or answer from Claude can get around them:

- at most MAX_ENTRIES_PER_DAY new trades per Japan day (00:00-24:00 JST)
- at most MAX_MARGIN_USD of margin per trade, at most MAX_POSITION_LEVERAGE on that margin
  (20 USD x 3 = 60 USD position; two of them on a 200 USD account is 0.6x, inside risk.MAX_LEVERAGE)
- every trade needs a stop loss, no more than MAX_STOP_PCT from the entry (well before liquidation)
- coins limited to COINS (Hyperliquid perps: ADA, BTC, ETH, SOL, HYPE, NEAR)

Claude can also read a few trading guides (btcbot/skills/, copied from agiprolabs/claude-trading-skills
at a fixed commit) with the read_skill tool; they are text only and change none of the limits above.
Which guides it read at each decision goes to skills.csv and the daily report.

Each `python -m btcbot.ai_trader step` (cron, every 5 minutes):
1. Checks stop loss / take profit / time limit of open trades against every 1 minute candle since the
   last run, charges the hourly funding and the taker fee, and asks Claude why a closed trade won or lost.
2. Checks the copy traders (COPY_TRADERS, or copy_traders.json in the folder): when one of them opens,
   closes or flips a position of COPY_MIN_USD or more in our coins, Claude is asked at once (logged in
   copy_alerts.csv); otherwise once an hour (the first run after each full hour) Claude is asked what to do. Claude can look at
   candles, funding, the Fear & Greed Index, the positions of the top Hyperliquid accounts ("big
   sharks") and the web, then open or close trades through tools, or do nothing.
3. Appends to equity.csv; trades.csv holds every closed trade with Claude's reasons and the review.

`python -m btcbot.ai_trader report` (cron, once a day) writes reports/YYYY-MM-DD.md in Traditional
Chinese, also sent to Telegram when it is set up. A PAUSE file in the folder blocks new trades.
Needs ANTHROPIC_API_KEY. Nothing here can place a real order.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pandas as pd
import requests

from btcbot import risk
from btcbot.advisor import WEB_SEARCH_TOOL
from btcbot.hyperliquid import MAINNET, TAKER_FEE, _post, parse_candles, parse_funding
from btcbot.paper import paused

MAX_ENTRIES_PER_DAY: Final[int] = 2
MAX_MARGIN_USD: Final[float] = 20.0
MAX_POSITION_LEVERAGE: Final[float] = 3.0
MAX_STOP_PCT: Final[float] = 0.25
COINS: Final[tuple[str, ...]] = ("ADA", "BTC", "ETH", "SOL", "HYPE", "NEAR")
SLIPPAGE = 0.0005
INITIAL_USD = 200.0
DEFAULT_SLOTS_JST = tuple(range(24))  # every hour
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
# swing traders oso01 picked on 2026-10-15; btcbot.copy_backtest: copying them an hour late in our coins made
# +1.65% per trade after costs over the 31-90 days before they were picked (82 trades), 0x405e the most
COPY_TRADERS: Final[tuple[str, ...]] = (
    "0x28280399d9afd68a3555c65ca6a75a1bd55693a0", "0x405eafc7534c349d5803b206a7cf5f0ab7686af4",
    "0xb69eec8fea0081343c3b09e652ba1fa7dcf7033a", "0x07eac6dc2af876d2f0402f6804cc6ceaa0cd20cd",
    "0xa2e366c4e2d453cab49cc9c64f0670fd2fd060a3")
COPY_MIN_USD = 10_000.0
COPY_EVENTS = ("opened long", "opened short", "closed long", "closed short", "flipped to long", "flipped to short")
JST = timezone(timedelta(hours=9))
DECISION_MODEL = "claude-sonnet-5-5"  # trading decisions
WRITER_MODEL = "claude-haiku-5-5"  # trade reviews and the daily report
# list prices (input, output) per token, for the cost log only
PRICES = {DECISION_MODEL: (2.0 / 1e6, 10.0 / 1e6), WRITER_MODEL: (0.10 / 1e6, 0.50 / 1e6)}
PRICE_SEARCH = 10.0 / 1000
SKILLS_DIR = Path(__file__).parent / "skills"
SKILLS: Final[dict[str, str]] = {  # name -> when to read it (shown in the prompt)
    "regime-detection": "is a coin trending or ranging now, and which strategy type fits that",
    "exit-strategies": "where to put the stop, take profit and time limit, trailing stops",
    "volatility-modeling": "ATR and realized volatility, how wide a stop should be",
    "copy-trading": "whether a big account's trades are worth following or just luck",
    "correlation-analysis": "how closely the coins move together, so two trades are not one doubled bet",
    "mean-reversion": "testing whether a move is likely to revert (this project's z-score mean reversion lost after costs)",
    "cointegration-analysis": "pairs trading two coins (needs two orders, so it uses both of the day's trades)",
}

STRATEGY_NOTES = """Strategies this project already backtested (Hyperliquid or GMO costs, ADA unless noted):
- 4h Donchian breakout (20-bar high long with 1.2x volume, 10-bar low short, 2x ATR trailing stop): the only
  one with a positive result. Hyperliquid 2024-06 to 2026-10, 200 USD: 2x both sides +353% (max drawdown -47%),
  2x with shorts only when Fear & Greed <= 50 +1,029% (DD -41%), 1x long only +206% (DD -30%). Wins about 40%,
  profit comes from a few big trend moves; stop is about 3% away.
- Every intraday strategy lost after costs and had about zero edge even without costs: 3R pullback,
  opening range reversal, trend pullback (ADA, ETH, BTC on 5m/15m), order-flow momentum, BTC trend + ADA
  relative strength, z-score mean reversion, grids. Reason: a round trip costs about 0.19% (fees +
  slippage), so with a 0.5% stop each trade starts about 0.4R behind. Rule of thumb from that work: avoid
  stops tighter than 1.5% unless the edge is very strong.
- Fear & Greed: blocking shorts when the index is above 50 helped the breakout; long-side filters hurt.
- How the 4h breakout enters and exits: enter at the close of the 4h candle that breaks the 20-bar high
  (long, volume at least 1.2x) or the 10-bar low (short); the only exit is a 2x ATR trailing stop
  (trailing_atr_mult 2, no take profit, no time limit), because the profit comes from a few long trends. A breakout candle nearly always looks
  overbought or oversold (RSI high or low, price far from the averages): that is what a breakout is, and
  the backtest made its money from exactly those entries. Waiting for a pullback or skipping because RSI
  is extreme was never tested, and every extra entry filter tested here (SMA50 and ATR room filters,
  Fear & Greed long filters, pullback limit entries) made results worse or no better.
- Limit order first, market order if not filled within 1 hour (ADA 4h breakout, 15m bars): worse than a
  plain market entry in every setup (2x with the F&G filter +855% -> +576% to +627%). The limit fills on
  pullbacks, while the trades that run straight away get chased at a worse price, and those are the big ones.
- Same 4h breakout on NEAR and HYPE (Hyperliquid, about 2 years): both worse than ADA over the same dates.
  NEAR 1x long +88%, 2x results fragile; HYPE 1x long +93%, HYPE shorts lost money while the coin rose
  about 500%. NEAR and HYPE are in this account because the steady leaderboard winners trade them a lot,
  not because a strategy was proven on them."""

TRADER_SYSTEM = f"""You run a small paper-trading account (fake money, a research experiment) on Hyperliquid \
perpetual futures. Your owner wants to see whether you can choose good trades yourself. You are called once \
an hour, so there is no need to force a trade: the next look is an hour away. Each time you are called you decide what to do now: open a trade, close an open trade, or do nothing. Doing nothing is a \
normal, often correct answer.

Hard limits, enforced in code (an order outside them is rejected and you get the error back):
- at most {MAX_ENTRIES_PER_DAY} new trades per Japan day; the context says how many are left today
- margin at most {MAX_MARGIN_USD:g} USD per trade, leverage at most {MAX_POSITION_LEVERAGE:g}x on that margin
- every trade needs a stop loss within {MAX_STOP_PCT:.0%} of the entry price
- coins: {", ".join(COINS)}
Costs: taker fee {TAKER_FEE:.3%} per side plus about {SLIPPAGE:.2%} slippage per side, hourly funding.

The point of this account is to be multi-coin, multi-strategy, and to learn from what the big \
accounts do. It must not just repeat the rule-based 4h breakout on ADA (other accounts already do that). \
Every time you are called, work through all of this before deciding:
1. Every coin ({", ".join(COINS)}): look at its candles on 4h, 1h and 15m (the bigger trend, the swing and \
the entry timing) and its funding. The context has a short overview of each to start from, including \
indicators already computed in code on 4h, 1h and 15m (EMA 20/50/200, RSI 14, ATR 14, Bollinger Bands, \
Donchian channels, volume ratio); use these numbers rather than estimating indicators from raw candles, \
and get_indicators for other timeframes. 4h_breakout_signal is the tested 4h breakout rule already checked \
in code on the last completed 4h candle: "long" (closed above the 20-bar high, volume at least 1.2x), \
"short" (closed below the 10-bar low; the short side uses the 10-bar low, not the 20-bar low; the tested \
short also needs Fear & Greed at or below 50) or "none". \
That candle has already closed, so there is no close left to wait for. Remember that \
a round trip costs about 0.2%, so a 15m setup with a tight stop starts far behind; use 15m mainly to time \
entries for ideas that also make sense on 1h and 4h.
2. Several strategy types for each coin, for example trend following / breakout, range or mean reversion, \
following the big accounts, funding or sentiment extremes, or one you find by searching the web. Pick \
the one that fits each coin's current market best, or none.
3. The big accounts: the context has what the watched Hyperliquid accounts hold now and what they \
changed since your last decision (opened, closed, added, cut, flipped). They come from a daily scan of \
the whole leaderboard: steady winners (up on the week, month and all time) with a realized profit in our \
coins over 30 days; their pnl_30d and roi_30d are that profit in our coins and its share of their equity. \
If the scan is missing they are the leaderboard's top 30-day PnL instead (no pnl_by_coin field). Say what \
they seem to be doing and whether it is worth following; they can be hedging or just lucky. get_whale_positions shows more detail.
4. The copy traders (copy_traders in the context): five swing traders your owner picked. They hold 1 to 5 \
days, mostly buy strength with the 7-day trend and add to winners, and use limit orders. A backtest of \
copying their trades in our coins an hour late, with our costs, made about +1.6% per trade over the two \
months before they were picked (82 trades, win rate about 49%, so a few big winners carry it; most of it \
from 0x405e and 0xb69e, but all five were picked and the backtest copied every one of their trades with \
no chart filter, so each of the five counts as tested). Their profit in other coins can't be copied. When copy_traders.alerts_now is not \
empty you were called because one of them just opened, closed or flipped a position: decide whether to \
follow (same side, name the strategy "copy 0x..." with the first 6 characters of the address) or not, and \
say why. They use no stop that we can see, but you must: set it wide enough for a multi-day swing (for \
example 2 to 3 ATR on 4h) and consider a trailing stop. When one of them closes a position you copied, \
consider closing yours. Copying is not required, but following is the tested default: a flat or unclear chart is \
not a reason to skip, since the backtest did not wait for confirmation. Skip only with a concrete reason, \
for example the chart clearly against the trade (a strong opposite trend or a breakdown), funding far \
against it, or it would use the last trade of the day while a tested breakout signal is also live.
Then open a trade only where you see a real edge after costs; doing nothing is fine when nothing stands \
out. Name the strategy and give your reason on every order; your owner reads them in the daily report.

{STRATEGY_NOTES}
These are research results, not rules for this account. The Fear & Greed short filter belongs to the 4h \
breakout only; it does not forbid shorts in other strategies. The current index is in the context. The \
backtests were mostly on ADA; BTC, ETH, SOL, HYPE and NEAR can behave differently, so judge each coin on its own \
market rather than waiting for the same ADA breakout everywhere.

Trading guides you can open with read_skill when they help a decision (general crypto guides written \
for Solana tokens; ignore anything about Solana, PumpFun, DEX liquidity or scripts, and where a guide \
disagrees with this project's backtest results above, the backtests win):
{chr(10).join(f"- {k}: {v}" for k, v in SKILLS.items())}
When a coin shows a fresh 4h breakout, the evidence favours taking it (shorts weaker when Fear & Greed is \
above 50); if you skip one, give a concrete reason with evidence, not only "RSI is high" or "overextended".

The regime-detection guide is in the context every time; use it when you judge which strategy fits a coin. \
Read exit-strategies before placing any order (once per day is enough; you see which guides you already \
read today in the context). \
The guides never change the hard limits. If a guide shaped a trade, say which in the reason.

Do not invent prices. When you are done, call finish once with your view on every coin (even the ones \
you did not trade), what the big accounts are doing, and a one or two sentence summary."""

REVIEW_SYSTEM = """A trade in a paper-trading experiment has just closed. In Traditional Chinese (Hong Kong \
style), in at most 4 short sentences, explain why it won or lost: was the idea right or wrong, was it the \
stop placement, timing, costs or funding, or just noise. Be honest; one trade proves little."""

REPORT_SYSTEM = """You write the daily report of an AI-run paper-trading account (fake money) for its owner, \
who reads Traditional Chinese (Hong Kong style). Plain sentences, under 600 words. Cover: equity now and \
change since yesterday and since the 200 USD start; every trade closed in the last 24 hours with what \
strategy it used, the result in USD, and why it won or lost (use the reviews given); open trades; what the \
AI decided at each decision time (all times are Japan time, JST; never call them Hong Kong time) and why, including its view on each coin (long / short / no trade and \
which strategy fits) and what the big Hyperliquid accounts were doing (opened, closed, flipped); a \
section headed 跟單大戶 from the COPY TRADERS facts: what the five copy traders did in our coins in the last \
24 hours, whether the AI followed each move and why, copied trades still open, and copied trades' results \
so far next to the AI's own trades (say "no moves" if there were none); which \
trading guides (skills) the AI read at each decision; the API cost so far. Compare with the rule-based 4h breakout \
accounts if their numbers are given. Be honest that a few trades prove nothing. Do not recommend \
real-money trading."""


def _tool(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": props,
                             "required": list(props) if required is None else required,
                             "additionalProperties": False}}


COIN = {"type": "string", "enum": list(COINS)}
TOOLS = [
    _tool("get_candles", "Recent OHLCV candles of a Hyperliquid perp, oldest first.",
          {"coin": COIN, "interval": {"type": "string", "enum": ["5m", "15m", "1h", "4h", "1d"]},
           "bars": {"type": "integer", "description": "how many candles, 10 to 300"}}),
    _tool("get_indicators", "EMA 20/50/200, RSI 14, ATR 14, Bollinger Bands (20, 2), Donchian channels and "
          "volume ratio on the latest completed candle, computed exactly in code.",
          {"coin": COIN, "interval": {"type": "string", "enum": ["5m", "15m", "1h", "4h", "1d"]}}),
    _tool("get_funding", "Hourly funding rates of the last 48 hours (positive: longs pay shorts).", {"coin": COIN}),
    _tool("get_fear_greed", "Crypto Fear & Greed Index (0-100) of the last 14 days.", {}),
    _tool("get_whale_positions", "Open positions of the watched Hyperliquid accounts (steady traders in our "
          "coins from the daily leaderboard scan), biggest positions first.",
          {"top": {"type": "integer", "description": "how many accounts, 1 to 20"}}),
    _tool("read_skill", "Read one of the trading guides listed in the instructions (markdown text).",
          {"name": {"type": "string", "enum": list(SKILLS)}}),
    _tool("place_order", "Open a new paper trade at the current price (market order).",
          {"coin": COIN, "side": {"type": "string", "enum": ["long", "short"]},
           "margin_usd": {"type": "number", "description": f"at most {MAX_MARGIN_USD:g}"},
           "leverage": {"type": "number", "description": f"1 to {MAX_POSITION_LEVERAGE:g}"},
           "stop_price": {"type": "number"},
           "take_profit_price": {"type": "number", "description": "0 = none"},
           "max_hold_hours": {"type": "number", "description": "close after this many hours; 0 = no limit"},
           "trailing_atr_mult": {"type": "number", "description": "trailing stop: after each completed 4h "
                                 "candle the stop moves to the best price since entry minus this many 4h ATR(14), "
                                 "only ever towards the price; 0 = fixed stop. The backtested 4h breakout uses 2"},
           "strategy": {"type": "string", "description": "short name of the strategy"},
           "reason": {"type": "string", "description": "why this trade, two or three sentences"}}),
    _tool("close_position", "Close an open paper trade now at the current price.",
          {"id": {"type": "integer"}, "reason": {"type": "string"}}),
    _tool("finish", "Call once at the end, with your view on every coin.",
          {"coins": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["coin", "view", "strategy", "reason"],
              "properties": {"coin": COIN, "view": {"type": "string", "enum": ["long", "short", "no trade"]},
                             "strategy": {"type": "string", "description": "the strategy that fits best now"},
                             "reason": {"type": "string", "description": "one or two sentences"}}}},
           "whales": {"type": "string", "description": "what the big accounts are doing, two or three sentences"},
           "summary": {"type": "string"}}),
]


# ---------- indicators ----------
INTERVAL = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def indicators(df: pd.DataFrame) -> dict:
    """Standard indicators on the latest completed candle, computed in code so Claude never has to
    work them out from raw numbers. Wilder smoothing for RSI and ATR."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    if len(c) < 30:
        return {"error": f"only {len(c)} candles"}
    ema = {n: c.ewm(span=n, adjust=False).mean() for n in (20, 50, 200)}
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss)).fillna(50.0)  # no losses -> 100, flat -> 50
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    px = float(c.iloc[-1])

    def r(x, n=6):
        return None if pd.isna(x) else round(float(x), n)
    out = {"close": px, "candle": str(df.index[-1]),
           "ema20": r(ema[20].iloc[-1]), "ema50": r(ema[50].iloc[-1]),
           "ema200": r(ema[200].iloc[-1]) if len(c) >= 200 else None,
           "rsi14": r(rsi.iloc[-1], 1), "rsi14_prev3": [r(x, 1) for x in rsi.iloc[-4:-1]],
           "atr14": r(atr.iloc[-1]), "atr14_pct": r(atr.iloc[-1] / px, 4),
           "bb_upper": r(mid.iloc[-1] + 2 * sd.iloc[-1]), "bb_mid": r(mid.iloc[-1]),
           "bb_lower": r(mid.iloc[-1] - 2 * sd.iloc[-1]), "bb_width_pct": r(4 * sd.iloc[-1] / mid.iloc[-1], 4),
           "donchian20_high": r(h.iloc[-21:-1].max()), "donchian20_low": r(l.iloc[-21:-1].min()),
           "donchian10_low": r(l.iloc[-11:-1].min()),
           "volume_ratio20": r(v.iloc[-1] / v.iloc[-21:-1].mean(), 2)}
    out["trend"] = ("up" if px > out["ema20"] > out["ema50"] else
                    "down" if px < out["ema20"] < out["ema50"] else "mixed")
    return out


def breakout_signal(ind: dict) -> str:
    """The tested 4h breakout rule on the last completed 4h candle: "long" when it closed above the 20-bar
    high on at least 1.2x volume, "short" when it closed below the 10-bar low, else "none"."""
    if "close" not in ind:
        return "none"
    if ind["donchian20_high"] is not None and ind["close"] > ind["donchian20_high"] and (ind["volume_ratio20"] or 0) >= 1.2:
        return "long"
    if ind["donchian10_low"] is not None and ind["close"] < ind["donchian10_low"]:
        return "short"
    return "none"


def completed(df: pd.DataFrame, interval: str, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the candle that is still forming."""
    return df[df.index + pd.Timedelta(INTERVAL[interval]) <= now]


# ---------- market data ----------
class HLMarket:
    """Public Hyperliquid data plus the Fear & Greed Index. No account or key."""

    def __init__(self, url: str = MAINNET, cache_dir: Path | None = None, traders_file: Path | None = None):
        self.url = url
        self.cache_dir = cache_dir
        self.traders_file = traders_file  # daily btcbot.hl_traders scan; used instead of the leaderboard while fresh
        self.session = requests.Session()

    def mids(self) -> dict[str, float]:
        return {k: float(v) for k, v in _post(self.url, {"type": "allMids"}, self.session).items() if k in COINS}

    def candles(self, coin: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows = _post(self.url, {"type": "candleSnapshot", "req": {
            "coin": coin, "interval": interval, "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000)}}, self.session)
        return parse_candles(rows)

    def funding(self, coin: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        rows = _post(self.url, {"type": "fundingHistory", "coin": coin, "startTime": int(start.timestamp() * 1000),
                                "endTime": int(end.timestamp() * 1000)}, self.session)
        return parse_funding(rows)

    def fear_greed(self) -> dict:
        from btcbot.data import FNG_URL, parse_fng
        resp = self.session.get(FNG_URL, params={"limit": 14, "format": "json"}, timeout=15)
        resp.raise_for_status()
        return {d.strftime("%Y-%m-%d"): v for d, v in parse_fng(resp.json()).items()}

    def leaderboard(self) -> list[dict]:
        """The steady traders in our coins from the daily scan (under 48 hours old), else the plain
        leaderboard's top 30-day PnL."""
        f = self.traders_file
        if f and f.exists() and time.time() - f.stat().st_mtime < 48 * 3600:
            rows = json.loads(f.read_text())
            if rows:
                return rows
        cache = self.cache_dir / "leaderboard.json" if self.cache_dir else None
        if cache and cache.exists() and time.time() - cache.stat().st_mtime < 12 * 3600:
            return json.loads(cache.read_text())
        resp = self.session.get(LEADERBOARD_URL, timeout=60)
        resp.raise_for_status()
        rows = []
        for r in resp.json().get("leaderboardRows", []):
            perf = {w: p for w, p in r.get("windowPerformances", [])}
            month = perf.get("month", {})
            rows.append({"address": r["ethAddress"], "account_value": float(r.get("accountValue", 0)),
                         "pnl_30d": float(month.get("pnl", 0)), "roi_30d": float(month.get("roi", 0))})
        rows = sorted((r for r in rows if r["account_value"] >= 100_000), key=lambda r: -r["pnl_30d"])[:50]
        if cache:
            cache.write_text(json.dumps(rows))
        return rows

    def positions(self, address: str) -> list[dict]:
        st = _post(self.url, {"type": "clearinghouseState", "user": address}, self.session)
        out = []
        for ap in st.get("assetPositions", []):
            p = ap["position"]
            out.append({"coin": p["coin"], "size": float(p["szi"]), "entry": float(p["entryPx"]),
                        "value_usd": round(float(p["positionValue"])), "unrealized_pnl": round(float(p["unrealizedPnl"])),
                        "leverage": p.get("leverage", {}).get("value")})
        return sorted(out, key=lambda p: -p["value_usd"])


# ---------- Claude ----------
class Brain:
    """Claude through the Messages API: a tool loop for decisions, plain text for reviews and reports."""

    def __init__(self, model: str = DECISION_MODEL, writer_model: str = WRITER_MODEL, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.writer_model = writer_model
        self.cost = 0.0  # USD, rough, for the log

    def _create(self, system: str, messages: list, tools: list | None, effort: str, model: str | None = None):
        model = model or self.model
        kwargs = dict(model=model, max_tokens=16000, system=system, messages=messages,
                      output_config={"effort": effort})
        if model != WRITER_MODEL:  # Haiku has no server-side fallback
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if tools:
            kwargs["tools"] = tools
        resp = self.client.beta.messages.create(**kwargs)
        u = resp.usage
        searches = getattr(getattr(u, "server_tool_use", None), "web_search_requests", 0) or 0
        p_in, p_out = PRICES.get(model, PRICES[DECISION_MODEL])
        self.cost += (u.input_tokens + (getattr(u, "cache_read_input_tokens", 0) or 0)) * p_in \
            + u.output_tokens * p_out + searches * PRICE_SEARCH
        if resp.stop_reason == "refusal":
            raise RuntimeError("request declined")
        return resp

    def run(self, context: dict, handlers: dict, max_rounds: int = 20) -> dict:
        """Let Claude call the tools until it calls finish. Returns the finish input (coin views, whales, summary)."""
        messages = [{"role": "user", "content": "Account and market context (JSON):\n"
                                                + json.dumps(context, indent=1, default=str)}]
        for _ in range(max_rounds):
            resp = self._create(TRADER_SYSTEM, messages, TOOLS + [WEB_SEARCH_TOOL], "medium")
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "pause_turn":
                continue
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                return {"summary": "".join(b.text for b in resp.content if b.type == "text") or "(no summary)"}
            results, summary = [], None
            for b in calls:
                if b.name == "finish":
                    summary = dict(b.input)
                    out, err = "ok", False
                else:
                    try:
                        out, err = json.dumps(handlers[b.name](**b.input), default=str), False
                    except Exception as e:  # the error goes back to Claude, never crashes the run
                        out, err = f"{type(e).__name__}: {e}", True
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
            if summary is not None:
                return summary
            messages.append({"role": "user", "content": results})
        return {"summary": "(stopped: too many tool rounds)"}

    def write(self, system: str, text: str) -> str:
        resp = self._create(system, [{"role": "user", "content": text}], None, "medium", self.writer_model)
        return "".join(b.text for b in resp.content if b.type == "text")


# ---------- the account ----------
class AITrader:
    def __init__(self, folder: Path, market, brain=None, slots_jst: tuple[int, ...] = DEFAULT_SLOTS_JST,
                 copy_traders: tuple[str, ...] | None = None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        if copy_traders is None:
            f = self.folder / "copy_traders.json"  # a list of addresses, to change who is copied without code
            copy_traders = tuple(json.loads(f.read_text())) if f.exists() else COPY_TRADERS
        self.copy_traders = tuple(copy_traders)
        self.market = market
        self.brain = brain
        self.slots = tuple(sorted(slots_jst))
        self.state_path = self.folder / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "cash": INITIAL_USD, "positions": [], "entries": {}, "slots_done": [], "next_id": 1,
            "last_check": None, "api_cost": 0.0}
        self.now = datetime.now(timezone.utc)
        self._mids: dict[str, float] = {}
        self._skills_read: list[str] = []
        self._copy_alerts: list[dict] = []

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.state_path)

    def _append(self, name: str, row: dict) -> None:
        path = self.folder / name
        new = pd.DataFrame([row])
        if path.exists() and path.stat().st_size:
            cols = list(pd.read_csv(path, nrows=0).columns)
            if set(row) - set(cols):  # a newer version added columns: rewrite once under the wider header
                pd.concat([pd.read_csv(path), new], ignore_index=True).to_csv(path, index=False)
                return
            new = new.reindex(columns=cols)  # same columns in the file's order, missing ones empty
        new.to_csv(path, mode="a", header=not (path.exists() and path.stat().st_size), index=False)

    def _day(self) -> str:
        return self.now.astimezone(JST).strftime("%Y-%m-%d")

    def entries_left(self) -> int:
        return MAX_ENTRIES_PER_DAY - self.state["entries"].get(self._day(), 0)

    def equity(self, mids: dict[str, float]) -> float:
        return self.state["cash"] + sum(p["side"] * p["qty"] * (mids.get(p["coin"], p["entry"]) - p["entry"])
                                        for p in self.state["positions"])

    # ----- orders (the only way a trade opens; every limit is checked here) -----
    def place_order(self, coin: str, side: str, margin_usd: float, leverage: float, stop_price: float,
                    take_profit_price: float, max_hold_hours: float, strategy: str, reason: str,
                    trailing_atr_mult: float = 0.0) -> dict:
        if paused(self.folder):
            raise ValueError("PAUSE file present: no new trades")
        if self.entries_left() <= 0:
            raise ValueError(f"daily limit reached: {MAX_ENTRIES_PER_DAY} trades today already")
        if coin not in COINS:
            raise ValueError(f"coin must be one of {COINS}")
        if not 0 < margin_usd <= MAX_MARGIN_USD:
            raise ValueError(f"margin_usd must be above 0 and at most {MAX_MARGIN_USD:g}")
        if not 1 <= leverage <= MAX_POSITION_LEVERAGE:
            raise ValueError(f"leverage must be 1 to {MAX_POSITION_LEVERAGE:g}")
        d = 1 if side == "long" else -1 if side == "short" else 0
        if not d:
            raise ValueError("side must be long or short")
        mid = self._mids.get(coin) or self.market.mids()[coin]
        px = mid * (1 + d * SLIPPAGE)
        if d * (px - stop_price) <= 0:
            raise ValueError(f"stop {stop_price} is on the wrong side of the entry {px:.6g}")
        if abs(px - stop_price) / px > MAX_STOP_PCT:
            raise ValueError(f"stop more than {MAX_STOP_PCT:.0%} from the entry {px:.6g}")
        if not 0 <= trailing_atr_mult <= 10:
            raise ValueError("trailing_atr_mult must be 0 (off) to 10")
        if take_profit_price and d * (take_profit_price - px) <= 0:
            raise ValueError(f"take profit {take_profit_price} is on the wrong side of the entry {px:.6g}")
        notional = margin_usd * leverage
        eq = self.equity(self._mids or self.market.mids())
        used = sum(p["qty"] * p["entry"] for p in self.state["positions"])
        if used + notional > risk.MAX_LEVERAGE * eq:  # account-wide cap from risk.py still holds
            raise ValueError(f"account would exceed {risk.MAX_LEVERAGE:g}x of equity {eq:.2f}")
        if sum(p["margin"] for p in self.state["positions"]) + margin_usd > eq:
            raise ValueError(f"not enough free equity ({eq:.2f} USD)")
        fee = notional * TAKER_FEE
        s = self.state
        pos = {"id": s["next_id"], "coin": coin, "side": d, "qty": notional / px, "entry": px, "margin": margin_usd,
               "leverage": leverage, "stop": stop_price, "take_profit": take_profit_price or 0.0,
               "deadline": str(pd.Timestamp(self.now) + pd.Timedelta(hours=max_hold_hours)) if max_hold_hours else None,
               "opened": str(self.now), "fees": fee, "funding": 0.0, "strategy": strategy, "reason": reason,
               "trail_atr": trailing_atr_mult, "best": px, "trail_bar": None}
        s["next_id"] += 1
        s["cash"] -= fee
        s["positions"].append(pos)
        s["entries"] = {k: v for k, v in s["entries"].items() if k >= self._day()}
        s["entries"][self._day()] = s["entries"].get(self._day(), 0) + 1
        self._save()
        return {"opened": pos["id"], "entry": round(px, 6), "qty": round(pos["qty"], 6), "fee_usd": round(fee, 4),
                "trades_left_today": self.entries_left()}

    def close_position(self, id: int, reason: str, px: float | None = None, when=None) -> dict:
        pos = next((p for p in self.state["positions"] if p["id"] == id), None)
        if pos is None:
            raise ValueError(f"no open trade with id {id}")
        if px is None:
            px = (self._mids.get(pos["coin"]) or self.market.mids()[pos["coin"]]) * (1 - pos["side"] * SLIPPAGE)
        fee = pos["qty"] * px * TAKER_FEE
        gross = pos["side"] * pos["qty"] * (px - pos["entry"])
        self.state["cash"] += gross - fee
        self.state["positions"].remove(pos)
        pnl = gross - pos["fees"] - fee - pos["funding"]
        row = {"id": id, "coin": pos["coin"], "side": "long" if pos["side"] > 0 else "short",
               "strategy": pos["strategy"], "opened": pos["opened"], "closed": str(when or self.now),
               "entry": pos["entry"], "exit": px, "stop": pos["stop"], "take_profit": pos["take_profit"],
               "margin": pos["margin"], "leverage": pos["leverage"], "pnl_usd": round(pnl, 4),
               "return_on_margin": round(pnl / pos["margin"], 4), "fees": round(pos["fees"] + fee, 4),
               "funding": round(pos["funding"], 4), "exit_reason": reason, "entry_reason": pos["reason"],
               "trailing_atr_mult": pos.get("trail_atr", 0.0)}
        row["review"] = self._review(row)
        self._append("trades.csv", row)
        self._save()
        return {"closed": id, "exit": round(px, 6), "pnl_usd": row["pnl_usd"]}

    def _review(self, row: dict) -> str:
        if self.brain is None:
            return ""
        try:
            start = pd.Timestamp(row["opened"]) - pd.Timedelta(hours=12)
            bars = self.market.candles(row["coin"], "1h", start, pd.Timestamp(self.now))
            text = json.dumps(row, default=str) + "\n\n1h candles around the trade:\n" + bars.round(6).to_csv()
            return self.brain.write(REVIEW_SYSTEM, text).strip()
        except Exception as e:
            return f"(review failed: {type(e).__name__}: {e})"

    # ----- the 5-minute run -----
    def _manage(self, last: pd.Timestamp | None) -> list[str]:
        notes, now = [], pd.Timestamp(self.now)
        for pos in list(self.state["positions"]):
            since = max(last, pd.Timestamp(pos["opened"])) if last is not None else pd.Timestamp(pos["opened"])
            rates = self.market.funding(pos["coin"], since, now)
            rates = rates[(rates.index > since) & (rates.index <= now)]
            if len(rates):
                cost = float(rates.sum()) * pos["side"] * pos["qty"] * self._mids.get(pos["coin"], pos["entry"])
                pos["funding"] += cost  # positive rate: longs pay
                self.state["cash"] -= cost
            bars = self.market.candles(pos["coin"], "1m", since.floor("min"), now)
            exit_px, why, when = None, None, None
            for t, b in bars.iterrows():
                d = pos["side"]
                if (d > 0 and b["low"] <= pos["stop"]) or (d < 0 and b["high"] >= pos["stop"]):
                    exit_px = (min(b["open"], pos["stop"]) if d > 0 else max(b["open"], pos["stop"])) * (1 - d * SLIPPAGE)
                    why, when = ("trailing_stop" if "first_stop" in pos else "stop_loss"), t  # moved by _trail
                elif pos["take_profit"] and ((d > 0 and b["high"] >= pos["take_profit"])
                                             or (d < 0 and b["low"] <= pos["take_profit"])):
                    exit_px = (max(b["open"], pos["take_profit"]) if d > 0 else min(b["open"], pos["take_profit"])) \
                        * (1 - d * SLIPPAGE)
                    why, when = "take_profit", t
                if why:
                    break
            if why is None and pos["deadline"] and now >= pd.Timestamp(pos["deadline"]):
                why = "time_limit"
            if why:
                r = self.close_position(pos["id"], why, exit_px, when)
                notes.append(f"closed #{pos['id']} {pos['coin']} {why} pnl {r['pnl_usd']:+.2f}")
            elif pos.get("trail_atr"):
                moved = self._trail(pos)
                if moved:
                    notes.append(f"#{pos['id']} {pos['coin']} trailing stop -> {moved:.6g}")
        if any(p.get("trail_atr") for p in self.state["positions"]):
            self._save()
        return notes

    def _trail(self, pos: dict) -> float | None:
        """Like the backtest: at each completed 4h candle, best price since entry minus trail_atr x ATR(14),
        and the stop only moves towards the price. Returns the new stop when it moved."""
        now = pd.Timestamp(self.now)
        if pos.get("trail_bar") and str(now.floor("4h") - pd.Timedelta("4h")) <= pos["trail_bar"]:
            return None  # no new 4h candle since the last move, skip the download
        bars = completed(self._indicator_candles(pos["coin"], "4h"), "4h", now)
        if len(bars) < 30 or (pos.get("trail_bar") and str(bars.index[-1]) <= pos["trail_bar"]):
            return None
        pos["trail_bar"] = str(bars.index[-1])
        held = bars[bars.index + pd.Timedelta("4h") > pd.Timestamp(pos["opened"])]
        d = pos["side"]
        if len(held):
            pos["best"] = max(pos["best"], float(held["high"].max())) if d > 0 else \
                min(pos["best"], float(held["low"].min()))
        atr = indicators(bars)["atr14"]
        if not atr:
            return None
        new = pos["best"] - d * pos["trail_atr"] * atr
        if d * (new - pos["stop"]) <= 0:
            return None
        pos.setdefault("first_stop", pos["stop"])
        pos["stop"] = new
        return new

    def _due_slot(self) -> str | None:
        """The latest decision time of today that has passed and was not handled yet."""
        local = self.now.astimezone(JST)
        passed = [h for h in self.slots if local.hour >= h]
        if not passed:
            return None
        key = f"{self._day()}T{passed[-1]:02d}"
        return None if key in self.state["slots_done"] else key

    def _context(self, mids: dict) -> dict:
        trades = self.folder / "trades.csv"
        last = pd.read_csv(trades).tail(10)[["coin", "side", "strategy", "opened", "closed", "pnl_usd",
                                             "exit_reason"]].to_dict(orient="records") if trades.exists() else []
        return {"time_utc": str(self.now), "time_jst": str(self.now.astimezone(JST)),
                "equity_usd": round(self.equity(mids), 2), "start_usd": INITIAL_USD,
                "trades_left_today": self.entries_left(), "prices": mids,
                "open_trades": [{k: p[k] for k in ("id", "coin", "side", "qty", "entry", "stop", "take_profit",
                                                   "deadline", "opened", "strategy")} | {"trailing_atr_mult": p.get("trail_atr", 0.0)}
                               for p in self.state["positions"]],
                "last_10_closed_trades": last,
                "fear_greed_last_7_days": self._fear_greed(),
                "guides_read_today": self._guides_today(),
                "regime_detection_guide": self._auto_guide("regime-detection"),
                "market_overview": self._overview(mids),
                "big_accounts": self._whale_watch(),
                "copy_traders": {"alerts_now": self._copy_alerts,
                                 "positions_in_our_coins": {a[:10]: {c: v for c, v in pos.items() if v}
                                                            for a, pos in self.state.get("copy_snapshot", {}).items()}}}

    def _fear_greed(self):
        try:
            return dict(list(self.market.fear_greed().items())[-7:])
        except Exception as e:
            return f"unavailable: {e}"

    def _auto_guide(self, name: str) -> str:
        """A guide put in every decision's context, so it is read whether or not Claude asks for it."""
        if name not in self._skills_read:
            self._skills_read.append(name)
        return (SKILLS_DIR / f"{name}.md").read_text()

    def _guides_today(self) -> list[str]:
        path = self.folder / "skills.csv"
        if not path.exists():
            return []
        k = pd.read_csv(path, keep_default_na=False)
        k = k[k["slot"].astype(str).str.startswith(self._day())]
        return sorted({g for row in k["skills_read"] for g in str(row).split(",") if g and g != "none"})

    def _overview(self, mids: dict) -> dict:
        """Per coin: price changes over 15m / 1h / 4h / 24h / 7d, the latest funding and the indicators on
        4h, 1h and 15m, so every coin gets looked at."""
        now, out = pd.Timestamp(self.now), {}
        for coin in COINS:
            try:
                frames = {iv: self._indicator_candles(coin, iv) for iv in ("4h", "1h", "15m")}
                c, q = frames["4h"]["close"], frames["15m"]["close"]
                px = mids.get(coin, float(c.iloc[-1]))

                def chg(closes, bars):
                    return round(px / float(closes.iloc[-1 - bars]) - 1, 4) if len(closes) > bars else None
                f = self.market.funding(coin, now - pd.Timedelta(hours=24), now)
                out[coin] = {"price": px, "change_15m": chg(q, 1), "change_1h": chg(q, 4), "change_4h": chg(c, 1),
                             "change_24h": chg(c, 6), "change_7d": chg(c, 42),
                             "funding_last": float(f.iloc[-1]) if len(f) else None,
                             "funding_24h_sum": round(float(f.sum()), 6) if len(f) else None,
                             "indicators": {iv: indicators(completed(df, iv, now)) for iv, df in frames.items()}}
                out[coin]["4h_breakout_signal"] = breakout_signal(out[coin]["indicators"]["4h"])
            except Exception as e:
                out[coin] = f"unavailable: {e}"
        return out

    def _indicator_candles(self, coin: str, interval: str) -> pd.DataFrame:
        """Enough candles for EMA 200 to settle (about 260)."""
        now = pd.Timestamp(self.now)
        return self.market.candles(coin, interval, now - 260 * pd.Timedelta(INTERVAL[interval]), now)

    def _whale_watch(self, top: int = 10) -> dict:
        """What the top accounts hold in our coins, and what changed since the last decision."""
        try:
            accounts = self._whales(top, keep=None)  # all positions, so a small one isn't read as closed
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        snap, prices = {}, dict(self._mids)
        for a in accounts:
            if isinstance(a["positions"], list):
                snap[a["address"]] = {p["coin"]: p["size"] for p in a["positions"]}
                for p in a["positions"]:
                    if p["size"]:
                        prices.setdefault(p["coin"], p["value_usd"] / abs(p["size"]))
        totals = {}
        for coin in COINS:
            sizes = [v[coin] for v in snap.values() if coin in v]
            px = self._mids.get(coin, 0.0)
            totals[coin] = {"accounts_long": sum(x > 0 for x in sizes), "accounts_short": sum(x < 0 for x in sizes),
                            "net_usd": round(sum(sizes) * px)}
        changes = whale_changes(self.state.get("whale_snapshot", {}), snap, prices)
        self.state["whale_snapshot"] = snap
        self._append("whales.csv", {"time": str(self.now), "totals": json.dumps(totals),
                                    "changes": json.dumps(changes[:20])})
        return {"our_coins": totals, "changes_since_last_decision": changes[:20],
                "accounts": [{k: a[k] for k in ("address", "account_value", "pnl_30d", "roi_30d", "pnl_by_coin") if k in a}
                             | {"biggest_positions": a["positions"][:4] if isinstance(a["positions"], list)
                                else a["positions"]} for a in accounts]}

    def _whales(self, top: int, keep: int | None = 6) -> list[dict]:
        out = []
        for r in self.market.leaderboard()[:max(1, min(int(top), 20))]:
            try:
                pos = self.market.positions(r["address"])[:keep]
            except Exception as e:
                pos = f"unavailable: {e}"
            out.append({**r, "positions": pos})
        return out

    def _copy_check(self) -> list[dict]:
        """Positions the copy traders opened, closed or flipped in our coins since the last run (every 5
        minutes). The first run only takes a snapshot; an address that can't be read keeps its old one."""
        old = self.state.get("copy_snapshot", {})
        new = dict(old)
        for addr in self.copy_traders:
            try:
                new[addr] = {p["coin"]: p["size"] for p in self.market.positions(addr) if p["coin"] in COINS}
            except Exception:
                continue
        new = {a: v for a, v in new.items() if a in self.copy_traders}
        self.state["copy_snapshot"] = new
        alerts = [c for c in whale_changes(old, new, self._mids, COPY_MIN_USD) if c["change"] in COPY_EVENTS]
        for c in alerts:
            self._append("copy_alerts.csv", {"time": str(self.now), **c})
        return alerts

    def _handlers(self) -> dict:
        now = pd.Timestamp(self.now)
        def get_candles(coin, interval, bars):
            n = max(10, min(int(bars), 300))
            df = self.market.candles(coin, interval, now - n * pd.Timedelta(INTERVAL[interval]), now)
            return df.tail(n).round(6).reset_index().astype({"open_time": str}).to_dict(orient="records")

        def get_funding(coin):
            s = self.market.funding(coin, now - pd.Timedelta(hours=48), now)
            return {str(t): float(v) for t, v in s.items()}

        def read_skill(name):
            if name not in SKILLS:
                raise ValueError(f"unknown guide {name}")
            if name not in self._skills_read:
                self._skills_read.append(name)
            return (SKILLS_DIR / f"{name}.md").read_text()

        def get_indicators(coin, interval):
            return indicators(completed(self._indicator_candles(coin, interval), interval, now))

        return {"get_candles": get_candles, "get_indicators": get_indicators, "read_skill": read_skill, "get_funding": get_funding,
                "get_fear_greed": lambda: self.market.fear_greed(),
                "get_whale_positions": self._whales,
                "place_order": self.place_order,
                "close_position": lambda id, reason: self.close_position(id, f"ai: {reason}")}

    def step(self, now: datetime | None = None) -> dict:
        self.now = now or datetime.now(timezone.utc)
        s = self.state
        cost_start = self.brain.cost if self.brain is not None else 0.0
        last = pd.Timestamp(s["last_check"]) if s["last_check"] else None
        self._mids = self.market.mids()
        notes = self._manage(last)

        self._copy_alerts = self._copy_check() if self.copy_traders else []
        slot, trigger = self._due_slot(), "hourly"
        held = {p["coin"] for p in s["positions"]}
        wake = [c for c in self._copy_alerts if not c["change"].startswith("closed") or c["coin"] in held]
        if wake:  # a close of a coin we don't hold is only logged
            trigger = "copy: " + "; ".join(f"{c['account']} {c['change']} {c['coin']}" for c in wake)
            slot = slot or f"{self._day()}T{self.now.astimezone(JST):%H%M}-copy"
        if slot:
            s["slots_done"] = [k for k in s["slots_done"] if k >= self._day()] + [slot]
            self._save()  # a crash mid-decision must not repeat the call on the next run
            if paused(self.folder):
                notes.append(f"{slot}: PAUSE file present, no decision")
            elif self.entries_left() <= 0 and not s["positions"]:
                notes.append(f"{slot}: daily limit reached, no decision")
            elif self.brain is None:
                notes.append(f"{slot}: no Claude, no decision")
            else:
                cost0, before = self.brain.cost, s["next_id"]
                self._skills_read = []
                try:
                    out, err = self.brain.run(self._context(self._mids), self._handlers()), ""
                except Exception as e:
                    out, err = {}, f"{type(e).__name__}: {e}"
                if isinstance(out, str):
                    out = {"summary": out}
                summary = out.get("summary", "")
                spent = self.brain.cost - cost0
                self._append("decisions.csv", {"time": str(self.now), "slot": slot, "trigger": trigger,
                                               "new_trades": s["next_id"] - before, "summary": summary,
                                               "coin_views": json.dumps(out.get("coins", []), ensure_ascii=False),
                                               "whales": out.get("whales", ""),
                                               "error": err, "cost_usd": round(spent, 4)})
                # own file, so decisions.csv keeps its columns
                self._append("skills.csv", {"time": str(self.now), "slot": slot,
                                            "skills_read": ",".join(self._skills_read)})
                notes.append(f"{slot}: {summary or err}")

        if self.brain is not None:  # decisions and trade reviews
            s["api_cost"] = s.get("api_cost", 0.0) + self.brain.cost - cost_start
        s["last_check"] = str(self.now)
        self._save()
        status = {"time": str(self.now), "equity": round(self.equity(self._mids), 2), "cash": round(s["cash"], 2),
                  "open": len(s["positions"]), "trades_left_today": self.entries_left(),
                  "action": "; ".join(notes) or "none"}
        self._append("equity.csv", status)
        return status


def whale_changes(old: dict, new: dict, prices: dict, min_usd: float = 10_000) -> list[dict]:
    """Positions the top accounts opened, closed, added to, cut or flipped between two snapshots,
    biggest first. Accounts missing from either snapshot are skipped (left or joined the top list)."""
    out = []
    for addr in set(old) & set(new):
        for coin in set(old[addr]) | set(new[addr]):
            a, b = old[addr].get(coin, 0.0), new[addr].get(coin, 0.0)
            if a == b:
                continue
            px = prices.get(coin)
            usd = abs(b - a) * px if px else None
            if usd is not None and usd < min_usd:
                continue
            if a == 0:
                what = "opened long" if b > 0 else "opened short"
            elif b == 0:
                what = "closed long" if a > 0 else "closed short"
            elif a * b < 0:
                what = "flipped to long" if b > 0 else "flipped to short"
            else:
                what = "added" if abs(b) > abs(a) else "cut"
            out.append({"account": addr[:10], "coin": coin, "change": what, "size_before": a, "size_now": b,
                        "change_usd": round(usd) if usd is not None else None})
    return sorted(out, key=lambda r: -(r["change_usd"] or 0))


# ---------- daily report ----------
def report_text(folder: Path, now: datetime, data_dir: Path | None = None) -> str:
    since = pd.Timestamp(now) - pd.Timedelta(days=1)
    parts = [f"Report time {now.astimezone(JST):%Y-%m-%d %H:%M} JST. Start balance {INITIAL_USD:g} USD."]
    st = json.loads((folder / "state.json").read_text()) if (folder / "state.json").exists() else {}
    eq_path, tr_path, dc_path = folder / "equity.csv", folder / "trades.csv", folder / "decisions.csv"
    if eq_path.exists():
        eq = pd.read_csv(eq_path, parse_dates=["time"])
        day = eq[eq["time"] >= since]
        parts.append(f"Equity now {eq['equity'].iloc[-1]}, 24h ago {day['equity'].iloc[0] if len(day) else 'n/a'}.")
    parts.append("Open trades: " + json.dumps(st.get("positions", []), default=str))
    rc = folder / "report_cost.csv"
    report_cost = pd.read_csv(rc)["cost_usd"].sum() if rc.exists() else 0.0
    api_cost = st.get("api_cost", 0) + report_cost
    parts.append(f"API cost so far: {api_cost:.2f} USD. Trading equity does not include it; "
                 f"equity after API cost: {(eq['equity'].iloc[-1] if eq_path.exists() else INITIAL_USD) - api_cost:.2f} USD.")
    if tr_path.exists():
        t = pd.read_csv(tr_path)
        recent = t[pd.to_datetime(t["closed"], utc=True, format="ISO8601") >= since]
        parts.append(f"All closed trades: {len(t)}, total pnl {t['pnl_usd'].sum():+.2f} USD, "
                     f"win rate {(t['pnl_usd'] > 0).mean():.0%}.")
        parts.append("Trades closed in the last 24h (with the AI's entry reason and review):\n"
                     + (recent.to_csv(index=False) if len(recent) else "none"))
    if dc_path.exists():
        d = pd.read_csv(dc_path)
        when = pd.to_datetime(d["time"], utc=True, format="ISO8601")
        d = d[when >= since].copy()
        d.insert(0, "time_jst", when[when >= since].dt.tz_convert("Asia/Tokyo").dt.strftime("%m-%d %H:%M"))
        d = d.drop(columns=["time"])
        parts.append("Decisions in the last 24h (times in JST):\n" + (d.to_csv(index=False) if len(d) else "none"))
    parts.append(copy_section(folder, st, since))
    sk_path = folder / "skills.csv"
    if sk_path.exists():
        k = pd.read_csv(sk_path, keep_default_na=False)
        k = k[pd.to_datetime(k["time"], utc=True, format="ISO8601") >= since]
        parts.append("Trading guides (skills) the AI read at each decision in the last 24h:\n"
                     + ("\n".join(f"{r.slot}: {r.skills_read or 'none'}" for r in k.itertuples()) if len(k) else "none"))
    if data_dir is not None:
        for f in sorted(data_dir.glob("paper-hl-*/equity.csv")):
            e = pd.read_csv(f)
            if len(e):
                tf = f.parent / "trades.csv"
                n = len(pd.read_csv(tf)) if tf.exists() else 0
                parts.append(f"Rule-based account {f.parent.name}: equity {e['equity'].iloc[-1]} USD, "
                             f"closed trades so far {n}.")
    return "\n\n".join(parts)


def copy_section(folder: Path, st: dict, since: pd.Timestamp) -> str:
    """Facts for the report's copy-trader section: their moves, the AI's answers, and how copied trades did."""
    out = ["COPY TRADERS (the five swing traders the AI is alerted about):"]
    ca_path, dc_path, tr_path = folder / "copy_alerts.csv", folder / "decisions.csv", folder / "trades.csv"
    moves = pd.read_csv(ca_path) if ca_path.exists() else pd.DataFrame()
    if len(moves):
        moves = moves[pd.to_datetime(moves["time"], utc=True, format="ISO8601") >= since]
    out.append("Their moves in our coins in the last 24h:\n" + (moves.to_csv(index=False) if len(moves) else "none"))
    if dc_path.exists():
        d = pd.read_csv(dc_path, keep_default_na=False)
        if "trigger" in d:
            d = d[d["trigger"].astype(str).str.startswith("copy")
                  & (pd.to_datetime(d["time"], utc=True, format="ISO8601") >= since)]
            out.append("The AI's answer to each alert (new_trades = trades it opened):\n"
                       + (d[["time", "trigger", "new_trades", "summary"]].to_csv(index=False) if len(d) else "none"))
    is_copy = lambda strat: strat.astype(str).str.lower().str.startswith("copy")
    open_copies = [p for p in st.get("positions", []) if str(p.get("strategy", "")).lower().startswith("copy")]
    out.append("Copied trades still open: " + (json.dumps(open_copies, default=str) if open_copies else "none"))
    if tr_path.exists():
        t = pd.read_csv(tr_path)
        mine, copied = t[~is_copy(t["strategy"])], t[is_copy(t["strategy"])]
        closed = pd.to_datetime(t["closed"], utc=True, format="ISO8601")
        stat = lambda x: (f"{len(x)} trades, total {x['pnl_usd'].sum():+.2f} USD, "
                          f"win rate {(x['pnl_usd'] > 0).mean():.0%}" if len(x) else "no trades")
        out.append(f"Copied trades closed so far: {stat(copied)}; in the last 24h: "
                   f"{stat(copied[closed[copied.index] >= since])}. The AI's own (not copied) trades so far: {stat(mine)}.")
    return "\n".join(out)


def daily_report(folder: Path, brain, now: datetime | None = None, data_dir: Path | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    facts = report_text(folder, now, data_dir)
    try:
        cost0 = brain.cost
        body = brain.write(REPORT_SYSTEM, facts)
        # own file, so a trader run at the same minute can't overwrite it in state.json
        pd.DataFrame([{"time": str(now), "cost_usd": round(brain.cost - cost0, 4)}]).to_csv(
            folder / "report_cost.csv", mode="a", header=not (folder / "report_cost.csv").exists(), index=False)
    except Exception as e:  # the numbers still get written
        body = f"(Claude 寫唔到報告：{type(e).__name__}: {e})\n\n{facts}"
    out = folder / "reports" / f"{now.astimezone(JST):%Y-%m-%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="AI-run paper account on Hyperliquid prices (fake money)")
    p.add_argument("command", choices=["step", "report", "status"])
    p.add_argument("--folder", default="data/ai-hl")
    p.add_argument("--slots", default=",".join(map(str, DEFAULT_SLOTS_JST)),
                   help="decision hours in Japan time, e.g. 9,21 (default: every hour)")
    p.add_argument("--no-claude", action="store_true", help="only manage open trades, never ask Claude")
    p.add_argument("--traders", default="data/hl-traders/latest.json",
                   help="accounts to watch from python -m btcbot.hl_traders (falls back to the leaderboard)")
    a = p.parse_args(argv)
    folder = Path(a.folder)
    market = HLMarket(cache_dir=folder, traders_file=Path(a.traders))
    if a.command == "status":
        t = AITrader(folder, market)
        print(json.dumps({**t.state, "equity": round(t.equity(market.mids()), 2)}, indent=2, default=str))
        return
    brain = None if a.no_claude else Brain()
    if a.command == "report":
        out = daily_report(folder, brain, data_dir=folder.parent)
        from btcbot import alerts
        alerts.send(f"🤖 AI 模擬帳戶日報\n\n{out.read_text()}")
        print(out)
        return
    slots = tuple(int(h) for h in a.slots.split(",") if h.strip())
    print(json.dumps(AITrader(folder, market, brain, slots).step(), default=str))


if __name__ == "__main__":
    main()
