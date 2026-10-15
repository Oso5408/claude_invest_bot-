"""Self-test of every Hyperliquid API call the bot makes, through the bot's own code: does each endpoint
answer, is the data in the shape the code expects, is the latest candle / funding fresh, are the prices sane
for ADA, BTC, ETH and SOL, and how long each call takes.

What is checked (read-only public data, no key, no account, no orders):
- allMids                  AI account prices, Telegram /health
- candleSnapshot 1m, 1h    rule paper accounts (paper.HyperliquidFeed)
- candleSnapshot 15m/1h/4h AI account (ai_trader.HLMarket)
- fundingHistory           both
- l2Book                   rule paper accounts (bid/ask)
- leaderboard (stats-data.hyperliquid.xyz) + clearinghouseState   AI account whale watch

Nothing is written; data/ is never touched. Run it on the VM (the cloud container can't reach Hyperliquid):

    .venv/bin/python -m btcbot.hl_selftest
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

from btcbot.ai_trader import COINS, HLMarket
from btcbot.paper import HyperliquidFeed

# Wide bands: only catches broken units / wrong coin / zero prices, not market moves.
PRICE_RANGE = {"ADA": (0.05, 20.0), "BTC": (5_000.0, 1_000_000.0), "ETH": (100.0, 100_000.0), "SOL": (1.0, 5_000.0)}
MAX_FUNDING_PER_HOUR = 0.005  # 0.5%/h; Hyperliquid caps hourly funding at 4%, normal is ~0.00125%
MAX_SPREAD = 0.005
MAX_MID_GAP = 0.03  # last 1m close vs allMids
SLOW_MS = 5_000


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, round((time.perf_counter() - t) * 1000)


def check_candles(df: pd.DataFrame, interval: str, now: pd.Timestamp, coin: str) -> list[str]:
    """Problems with a candle frame from parse_candles, or [] if it looks right."""
    step = pd.Timedelta(interval.replace("m", "min"))
    if df.empty:
        return ["no candles"]
    p = []
    if list(df.columns) != ["open", "high", "low", "close", "volume"]:
        p.append(f"columns {list(df.columns)}")
    if str(df.index.tz) != "UTC":
        p.append("index not UTC")
    bad = (df["low"] > df[["open", "close"]].min(axis=1)) | (df["high"] < df[["open", "close"]].max(axis=1))
    if bad.any():
        p.append(f"{int(bad.sum())} candles with low/high outside open/close")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        p.append("zero/negative price or negative volume")
    gaps = df.index.to_series().diff().dropna()
    if len(gaps) and (gaps % step != pd.Timedelta(0)).any():
        p.append("candle times not on the interval grid")
    missing = int((gaps / step - 1).clip(lower=0).sum()) if len(gaps) else 0
    if missing > max(2, len(df) // 50):
        p.append(f"{missing} missing candles")
    age = now - df.index[-1]
    if age > 2 * step:
        p.append(f"latest candle opened {age} ago (stale)")
    lo, hi = PRICE_RANGE.get(coin, (0, float("inf")))
    if not lo <= df["close"].iloc[-1] <= hi:
        p.append(f"close {df['close'].iloc[-1]} outside {lo}-{hi}")
    return p


def check_funding(s: pd.Series, now: pd.Timestamp) -> list[str]:
    if s.empty:
        return ["no funding rows"]
    p = []
    if str(s.index.tz) != "UTC":
        p.append("index not UTC")
    if now - s.index[-1] > pd.Timedelta(hours=2):
        p.append(f"latest funding {now - s.index[-1]} ago (stale)")
    gaps = s.index.to_series().diff().dropna()
    if len(gaps) and (gaps.dt.round("min") != pd.Timedelta(hours=1)).mean() > 0.1:
        p.append("funding not hourly")
    if s.abs().max() > MAX_FUNDING_PER_HOUR:
        p.append(f"funding {s.abs().max():.5f}/h looks wrong")
    return p


def run_checks(market: HLMarket | None = None, feed_factory=HyperliquidFeed, now: datetime | None = None,
               ping=None, progress=None) -> list[dict]:
    """Every check as {name, ok, ms, detail, problems}; progress(n, total, result) is called after each one.
    market / feed_factory / ping are swappable for tests."""
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    market = market or HLMarket()
    if ping is None:
        from btcbot.telegram_bot import hyperliquid_ok as ping
    checks = []

    def add(name, fn, slow_ms=SLOW_MS):
        checks.append((name, fn, slow_ms))

    mids_holder = {}

    def mids():
        m, ms = _timed(market.mids)
        mids_holder.update(m)
        p = [f"{c} missing" for c in COINS if c not in m]
        p += [f"{c} {v} outside {PRICE_RANGE[c]}" for c, v in m.items() if not PRICE_RANGE[c][0] <= v <= PRICE_RANGE[c][1]]
        return ms, {c: m.get(c) for c in COINS}, p
    add("allMids (AI prices)", mids)

    def health():
        msg, ms = _timed(ping)
        return ms, msg, [] if msg.startswith("正常") else [msg]
    add("allMids (Telegram /health)", health)

    for coin in COINS:
        def ai_candles(coin=coin):
            out, total, p = {}, 0, []
            for iv, bars in (("15m", 260), ("1h", 260), ("4h", 260)):
                df, ms = _timed(lambda: market.candles(coin, iv, now_ts - bars * pd.Timedelta(iv.replace("m", "min")), now_ts))
                total += ms
                out[iv] = {"rows": len(df), "last_open": str(df.index[-1]) if len(df) else None,
                           "last_close": float(df["close"].iloc[-1]) if len(df) else None}
                p += [f"{iv}: {x}" for x in check_candles(df, iv, now_ts, coin)]
                if len(df) < bars * 0.9:
                    p.append(f"{iv}: only {len(df)} of ~{bars} candles")
            return total, out, p
        add(f"candleSnapshot {coin} 15m/1h/4h (AI)", ai_candles)

        def ai_funding(coin=coin):
            s, ms = _timed(lambda: market.funding(coin, now_ts - pd.Timedelta(hours=48), now_ts))
            detail = {"rows": len(s), "last": str(s.index[-1]) if len(s) else None,
                      "last_rate": float(s.iloc[-1]) if len(s) else None}
            return ms, detail, check_funding(s, now_ts)
        add(f"fundingHistory {coin} 48h (AI)", ai_funding)

    def paper_feed():
        feed = feed_factory("ADA")
        total, out, p = 0, {}, []
        for iv, hl_iv, days in (("1hour", "1h", 30), ("1min", "1m", 1)):
            df, ms = _timed(lambda: feed.candles(iv, now_ts.to_pydatetime(), days))
            total += ms
            out[hl_iv] = {"rows": len(df), "last_open": str(df.index[-1]) if len(df) else None}
            p += [f"{hl_iv}: {x}" for x in check_candles(df, hl_iv, now_ts, "ADA")]
            if hl_iv == "1m" and len(df) and mids_holder.get("ADA"):
                gap = abs(df["close"].iloc[-1] / mids_holder["ADA"] - 1)
                out["1m_close_vs_mid"] = f"{gap:.2%}"
                if gap > MAX_MID_GAP:
                    p.append(f"last 1m close {gap:.1%} away from allMids")
        (bid, ask), ms = _timed(feed.bid_ask)
        total += ms
        spread = (ask - bid) / ((ask + bid) / 2) if bid > 0 else float("inf")
        out["bid_ask"] = [bid, ask]
        out["spread"] = f"{spread:.3%}"
        if not 0 < bid < ask or spread > MAX_SPREAD:
            p.append(f"bad book bid {bid} ask {ask}")
        s, ms = _timed(lambda: feed.funding(now_ts - pd.Timedelta(hours=48), now_ts))
        total += ms
        out["funding_rows"] = len(s)
        p += [f"funding: {x}" for x in check_funding(s, now_ts)]
        return total, out, p
    add("rule paper feed ADA (1m, 1h, l2Book, funding)", paper_feed)

    lb_holder = []

    def leaderboard():
        rows, ms = _timed(market.leaderboard)
        lb_holder.extend(rows)
        p = [] if len(rows) >= 10 else [f"only {len(rows)} accounts with >= 100k USD (AI watches top 10)"]
        bad = [r for r in rows if not (str(r["address"]).startswith("0x") and len(r["address"]) == 42)]
        if bad:
            p.append(f"{len(bad)} rows without a 0x address")
        return ms, {"accounts": len(rows), "top_pnl_30d": rows[0]["pnl_30d"] if rows else None}, p
    add("leaderboard (stats-data.hyperliquid.xyz)", leaderboard, slow_ms=30_000)  # a few MB of JSON

    def whale_positions():
        if not lb_holder:
            return 0, None, ["no leaderboard, nothing to look up"]
        total, out, p = 0, [], []
        for r in lb_holder[:3]:
            pos, ms = _timed(lambda: market.positions(r["address"]))
            total += ms
            out.append({"address": r["address"][:10], "positions": len(pos),
                        "biggest": pos[0]["coin"] if pos else None})
            for x in pos:
                if x["size"] and x["value_usd"] < 0:
                    p.append(f"{x['coin']} negative position value")
        return total, out, p
    add("clearinghouseState top 3 whales", whale_positions)

    results = []
    for n, (name, fn, slow_ms) in enumerate(checks, 1):
        try:
            ms, detail, problems = fn()
        except Exception as e:
            ms, detail, problems = None, None, [f"{type(e).__name__}: {e}"]
        if ms is not None and ms > slow_ms:
            problems = [*problems, f"slow: {ms} ms"]
        results.append({"name": name, "ok": not problems, "ms": ms, "detail": detail, "problems": problems})
        if progress:
            progress(n, len(checks), results[-1])
    return results


def main() -> None:
    now = datetime.now(timezone.utc)
    print(f"Hyperliquid API self-test {now:%Y-%m-%d %H:%M} UTC (read-only, no orders)", flush=True)

    def show(n, total, r):
        print(f"[{n}/{total}] {'OK  ' if r['ok'] else 'FAIL'} {r['name']} ({r['ms']} ms)", flush=True)
        print("     " + json.dumps({k: r[k] for k in ("detail", "problems") if r[k]}, ensure_ascii=False, default=str),
              flush=True)

    results = run_checks(now=now, progress=show)
    print("ALL OK" if all(r["ok"] for r in results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
