"""BTC 5-minute up/down calls on Telegram. Predictions only: no orders, no account, no Claude cost.

At minute 2 of every 5-minute candle (UTC :02, :07, :12, ...) it reads Hyperliquid BTC 1-minute candles and
sends UP or DOWN for "this candle closes above its open", with a probability. Every call goes to
data/updown/predictions.csv and is scored once the candle has closed.

How the call is made (checked on Binance BTC 5m, 2023-01..2026-10, ~400k candles):
- Knowing the first 2 of 5 minutes already gives ~72% hits even on a pure random walk, so
  P(up) = Phi((move so far + expected rest) / (1m volatility * sqrt(3))).
- Expected rest = REVERSAL * previous candle's return * 3/5: 5m candles reverse very slightly
  (slope -0.026 overall, -0.005 in 2026), so this almost never changes the call.
So the honest bar is not 50% but the "follow the first 2 minutes" baseline, logged next to every call.

Polymarket (read only, never trades): at the same moment it reads the public price of "Up" in Polymarket's
"Bitcoin Up or Down" 5-minute market (slug btc-updown-5m-<window start, unix s>), and later that market's
own result. Polymarket settles on a Chainlink BTC/USD TWAP, not Hyperliquid, so its result can differ from ours
on tiny moves. A failed Polymarket read only leaves those columns empty.

    python -m btcbot.updown run        # loop forever (systemd service updown-bot)
    python -m btcbot.updown once       # make one call now and print it (sends nothing)
    python -m btcbot.updown status     # hit rates: today, 7 days, all, vs baseline
    python -m btcbot.updown settle     # backfill Polymarket results for every past call
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from btcbot import alerts, hyperliquid

COIN = "BTC"
WINDOW = timedelta(minutes=5)
CALL_AT = timedelta(minutes=2, seconds=3)  # a few seconds after minute 2 so the second 1m candle has closed
REVERSAL = -0.026
VOL_MINUTES = 30
JST = timezone(timedelta(hours=9))
GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com"
COLUMNS = ["window", "sent", "open", "price", "move_pct", "prev_pct", "p_up", "call", "baseline",
           "close", "result", "correct", "baseline_correct", "pm_up_mid", "pm_up_bid", "pm_up_ask", "pm_result"]


def folder(data: Path) -> Path:
    return data / "updown"


def window_start(t: datetime) -> datetime:
    t = t.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return t - timedelta(minutes=t.minute % 5)


def next_call_time(now: datetime) -> datetime:
    t = window_start(now) + CALL_AT
    return t if t > now else t + WINDOW


def fetch_minutes(now: datetime, minutes: int = 60) -> pd.DataFrame:
    end = int(now.timestamp() * 1000)
    rows = hyperliquid._post(hyperliquid.MAINNET, {"type": "candleSnapshot", "req": {
        "coin": COIN, "interval": "1m", "startTime": end - minutes * 60_000, "endTime": end}})
    return hyperliquid.parse_candles(rows)


def predict(m: pd.DataFrame, window: datetime) -> dict:
    """The call for the 5m candle starting at `window`, from 1m candles up to minute 2 of it."""
    w = pd.Timestamp(window)
    if w not in m.index:
        raise ValueError(f"no 1m candle at {w}")
    first2 = m.loc[w:w + pd.Timedelta(minutes=1)]
    o, price = float(m.loc[w, "open"]), float(first2["close"].iloc[-1])
    prev = m.loc[w - pd.Timedelta(minutes=5):w - pd.Timedelta(minutes=1)]
    prev_ret = math.log(prev["close"].iloc[-1] / prev["open"].iloc[0]) if len(prev) == 5 else 0.0
    hist = m.loc[:w + pd.Timedelta(minutes=1)].tail(VOL_MINUTES)
    rets = (hist["close"] / hist["open"]).map(math.log)
    vol = max(float(rets.std()) if len(rets) > 2 else 0.0, 1e-5)
    move = math.log(price / o)
    z = (move + REVERSAL * prev_ret * 3 / 5) / (vol * math.sqrt(3))
    p_up = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    baseline = "UP" if move > 0 else "DOWN" if move < 0 else ("DOWN" if prev_ret > 0 else "UP")
    return {"window": w.isoformat(), "open": o, "price": price, "move_pct": round(move * 100, 4),
            "prev_pct": round(prev_ret * 100, 4), "p_up": round(p_up, 4),
            "call": "UP" if p_up >= 0.5 else "DOWN", "baseline": baseline}


def pm_slug(window) -> str:
    return f"btc-updown-5m-{int(pd.Timestamp(window).timestamp())}"


def pm_market(window, get=requests.get) -> dict | None:
    rows = get(GAMMA, params={"slug": pm_slug(window)}, timeout=10).json()
    return rows[0] if rows else None


def pm_quote(window, get=requests.get) -> dict:
    """Polymarket's price of "Up" right now: CLOB midpoint and best bid/ask. {} when unavailable."""
    try:
        mk = pm_market(window, get)
        if not mk:
            return {}
        up_token = json.loads(mk["clobTokenIds"])[0]
        out = {}
        try:
            out["pm_up_mid"] = float(get(f"{CLOB}/midpoint", params={"token_id": up_token}, timeout=10).json()["mid"])
            book = get(f"{CLOB}/book", params={"token_id": up_token}, timeout=10).json()
            if book.get("bids"):
                out["pm_up_bid"] = max(float(b["price"]) for b in book["bids"])
            if book.get("asks"):
                out["pm_up_ask"] = min(float(a["price"]) for a in book["asks"])
        except Exception:
            out.setdefault("pm_up_mid", float(json.loads(mk["outcomePrices"])[0]))  # Gamma's few-seconds-old price
        return out
    except Exception as e:
        print(f"[polymarket quote failed: {type(e).__name__}: {e}]", flush=True)
        return {}


def pm_result(window, get=requests.get) -> str | None:
    """'UP' / 'DOWN' once Polymarket has settled the market, 'NA' if it has no such market, else None."""
    try:
        mk = pm_market(window, get)
        if not mk:
            return "NA"  # don't keep asking for a market that doesn't exist
        if not mk.get("closed"):
            return None
        up = float(json.loads(mk["outcomePrices"])[0])
        return "UP" if up >= 0.99 else "DOWN" if up <= 0.01 else None
    except Exception:
        return None


def pm_settle(df: pd.DataFrame, now: datetime, get=requests.get, limit: int | None = 12,
              max_age: timedelta | None = timedelta(hours=48), pause: float = 0.0) -> pd.DataFrame:
    """Fill pm_result for up to `limit` unsettled calls, oldest first. Polymarket closes a market some
    minutes after it ends, so the oldest ones are the ones most likely to be settled by now."""
    df = df.copy()
    t = pd.to_datetime(df["window"], utc=True)
    todo = df["pm_result"].isna() & (t + WINDOW + timedelta(minutes=1) <= now)
    if max_age is not None:
        todo &= t >= now - max_age
    for i in list(df.index[todo])[:limit]:
        df.at[i, "pm_result"] = pm_result(df.at[i, "window"], get)
        time.sleep(pause)
    return df


def load(data: Path) -> pd.DataFrame:
    path = folder(data) / "predictions.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"result": object, "call": object, "baseline": object, "pm_result": object})
    for c in COLUMNS:
        if c not in df:
            df[c] = None
    return df


def save(data: Path, df: pd.DataFrame) -> None:
    folder(data).mkdir(parents=True, exist_ok=True)
    tmp = folder(data) / "predictions.csv.tmp"
    df[COLUMNS].to_csv(tmp, index=False)
    tmp.replace(folder(data) / "predictions.csv")


def score(df: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Fill close/result for calls whose candle has fully closed in `m`. A flat candle counts as DOWN."""
    df = df.copy()
    for i in df.index[df["result"].isna()]:
        last = pd.Timestamp(df.at[i, "window"]) + pd.Timedelta(minutes=4)
        if last not in m.index or m.index[-1] <= last:  # the 5th minute is still open
            continue
        close = float(m.loc[last, "close"])
        result = "UP" if close > float(df.at[i, "open"]) else "DOWN"
        df.at[i, "close"], df.at[i, "result"] = close, result
        df.at[i, "correct"] = int(df.at[i, "call"] == result)
        df.at[i, "baseline_correct"] = int(df.at[i, "baseline"] == result)
    return df


def hit_line(df: pd.DataFrame, label: str) -> str:
    done = df[df["result"].notna()]
    if done.empty:
        return f"{label}：未有結果"
    n, k, b = len(done), int(done["correct"].sum()), int(done["baseline_correct"].sum())
    return f"{label}：{k}/{n} = {k / n:.1%}（跟頭 2 分鐘 {b / n:.1%}）"


def pm_line(df: pd.DataFrame) -> str:
    """Our probability vs Polymarket's minute-2 price, on candles Polymarket has settled (lower Brier = better)."""
    done = df[df["pm_result"].isin(["UP", "DOWN"]) & df["pm_up_mid"].notna()]
    if done.empty:
        return "Polymarket 對比：未有結果"
    y = (done["pm_result"] == "UP").astype(float)
    mid, ours = done["pm_up_mid"].astype(float), done["p_up"].astype(float)
    hit_pm = ((mid >= 0.5) == (y == 1)).mean()
    hit_ours = ((ours >= 0.5) == (y == 1)).mean()
    brier_pm, brier_ours = ((mid - y) ** 2).mean(), ((ours - y) ** 2).mean()
    return (f"Polymarket 對比（{len(done)} 支，用佢結算結果）：我哋中 {hit_ours:.1%} vs 盤口 {hit_pm:.1%}；"
            f"Brier 我哋 {brier_ours:.3f} vs 盤口 {brier_pm:.3f}（細啲好啲）")


def summary(data: Path, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    df = load(data)
    if df.empty:
        return "BTC 5 分鐘預測：未有紀錄（updown-bot 有冇行緊？）"
    t = pd.to_datetime(df["window"], utc=True)
    today = now.astimezone(JST).date()
    lines = ["BTC 5 分鐘升跌預測（只係預測，唔落單）",
             hit_line(df[t.dt.tz_convert(JST).dt.date == today], "今日（日本時間）"),
             hit_line(df[t >= now - timedelta(days=7)], "7 日"),
             hit_line(df, "全部"),
             pm_line(df),
             f"最後一次：{t.max().tz_convert(JST):%m-%d %H:%M} JST",
             "隨機估係 50%；第 2 分鐘跟住已行方向估都有約 72%，要贏過後者先算有預測能力。"]
    return "\n".join(lines)


def message(row: dict, df: pd.DataFrame, now: datetime) -> str:
    w = pd.Timestamp(row["window"]).tz_convert(JST)
    arrow = "⬆️ UP" if row["call"] == "UP" else "⬇️ DOWN"
    p = row["p_up"] if row["call"] == "UP" else 1 - row["p_up"]
    lines = [f"BTC {w:%H:%M}-{w + pd.Timedelta(minutes=5):%H:%M}：{arrow}（{p:.0%}）",
             f"開 {row['open']:,.0f} → 而家 {row['price']:,.0f}（{row['move_pct']:+.3f}%），上一支 {row['prev_pct']:+.3f}%"]
    if row.get("pm_up_mid") is not None:
        lines.append(f"Polymarket Up 價 {row['pm_up_mid']:.2f}（只係記錄，唔落注）")
    done = df[df["result"].notna()]
    if not done.empty:
        last = done.iloc[-1]
        lw = pd.Timestamp(last["window"]).tz_convert(JST)
        lines.append(f"上次 {lw:%H:%M} 估 {last['call']}，結果 {last['result']} {'✅' if last['correct'] == 1 else '❌'}")
    t = pd.to_datetime(df["window"], utc=True)
    lines.append(hit_line(df[t.dt.tz_convert(JST).dt.date == now.astimezone(JST).date()], "今日"))
    return "\n".join(lines)


def step(data: Path, now: datetime, fetch=fetch_minutes, send=alerts.send, get=requests.get) -> dict | None:
    """Score finished calls, then make and send the call for the current candle. Returns the new row."""
    m = fetch(now)
    df = score(load(data), m)
    w = window_start(now)
    if (df["window"] == pd.Timestamp(w).isoformat()).any():
        save(data, df)
        return None
    row = predict(m, w)
    row.update(pm_quote(w, get))  # read first: the price at minute 2 is what we compare against
    row["sent"] = now.isoformat(timespec="seconds")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = pm_settle(df, now, get)
    save(data, df)
    send(message(row, df, now), silent=True)
    return row


def run(data: Path) -> None:
    print("updown bot started", flush=True)
    while True:
        now = datetime.now(timezone.utc)
        time.sleep(max(0.0, (next_call_time(now) - now).total_seconds()))
        now = datetime.now(timezone.utc)
        if now - window_start(now) > CALL_AT + timedelta(seconds=50):  # woke up late (VM paused): skip
            continue
        try:
            step(data, now)
        except Exception as e:  # network down: skip this candle, never exit
            print(f"[{now:%F %T}] step failed: {type(e).__name__}: {e}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="BTC 5-minute up/down calls on Telegram (no trading)")
    p.add_argument("command", choices=["run", "once", "status", "settle"])
    p.add_argument("--data", default="data")
    a = p.parse_args(argv)
    data = Path(a.data)
    if a.command == "status":
        print(summary(data))
    elif a.command == "settle":  # backfill every unsettled Polymarket result
        df = pm_settle(load(data), datetime.now(timezone.utc), limit=None, max_age=None, pause=0.2)
        save(data, df)
        print(f"Polymarket 結果：{df['pm_result'].notna().sum()}/{len(df)} 支；有盤口價：{df['pm_up_mid'].notna().sum()} 支")
    elif a.command == "once":
        now = datetime.now(timezone.utc)
        w = window_start(now)
        if now - w < timedelta(minutes=2):
            w -= WINDOW  # minute 2 not reached yet: show the previous candle's call
        row = predict(fetch_minutes(now), w)
        row.update(pm_quote(w))
        print(row)
    else:
        run(data)


if __name__ == "__main__":
    main()
