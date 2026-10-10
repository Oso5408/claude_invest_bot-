"""Paper trading on live GMO Coin prices: the 4h breakout strategy with fake money.

No account, no API key, no real orders. Each run of `python -m btcbot.paper step`:
1. Pulls recent 1h and 1min candles plus the current bid/ask from GMO's public API.
2. Checks the trailing stop and the 75% loss cut against every 1min low/high since
   the last run, and charges the 0.04%/day leverage fee for each 06:00 JST passed.
3. When a new 4h candle has closed: moves the trailing stop, or enters on a
   breakout at the current ask (long) / bid (short) plus the taker fee.
4. Saves the state to data/paper/state.json and appends to trades.csv / equity.csv.

Run it every 5 minutes from cron (see docs/gcp-setup.md). Each processed 4h candle is logged to signals.csv
for the independent monitor (btcbot/monitor.py); a PAUSE file stops new entries. The strategy logic,
sizing (RiskController) and the 2x cap are the same code the backtest uses.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from btcbot import risk
from btcbot.advisor import entry_context
from btcbot.breakout import BreakoutConfig, compute_features
from btcbot.data import BASE_URL, FNG_URL, parse_fng, parse_klines, resample
from btcbot.meanrev import AccountConfig, MarginAccount, RiskController, _round_qty
from btcbot.strategy import StrategyConfig

JST = timezone(timedelta(hours=9))
TICKER_URL = "https://api.coin.z.com/public/v1/ticker"


@dataclass
class PaperConfig:
    symbol: str = "ADA_JPY"
    timeframe: str = "4h"
    allow_short: bool = True
    max_leverage: float = risk.MAX_LEVERAGE
    initial_jpy: float = 30_000.0
    history_days: int = 20  # enough 4h candles for the 20-bar channel and ATR
    fng_short_max: float = 100  # only short when the Fear & Greed Index is at or below this (100 = off)


def rollovers_between(start: pd.Timestamp, end: pd.Timestamp, hour_jst: int) -> int:
    """How many hour_jst:00 JST moments fall in (start, end]."""
    s, e = start.tz_convert("Asia/Tokyo"), end.tz_convert("Asia/Tokyo")
    first = s.normalize() + pd.Timedelta(hours=hour_jst)
    if first <= s:
        first += pd.Timedelta(days=1)
    return 0 if first > e else int((e - first) // pd.Timedelta(days=1)) + 1


def check_bar_data(hourly: pd.DataFrame, bar: pd.Timestamp, tf: pd.Timedelta, now: datetime) -> str | None:
    """Why a just-closed bar must not be traded yet, or None if its data looks complete and fresh."""
    if hourly.empty:
        return "no 1h candles"
    newest = hourly.index.max()
    if newest < pd.Timestamp(now) - pd.Timedelta(hours=2):
        return f"stale data: newest 1h candle {newest}"
    have = int(((hourly.index >= bar) & (hourly.index < bar + tf)).sum())
    need = int(tf / pd.Timedelta(hours=1))
    if have < need:
        return f"incomplete {tf} candle {bar}: {have}/{need} hourly candles"
    return None


def paused(folder: Path) -> bool:
    """A PAUSE file in the account folder or its parent (all accounts) blocks new entries."""
    return (folder / "PAUSE").exists() or (folder.parent / "PAUSE").exists()


def gmo_business_date(t: datetime) -> str:
    """GMO's kline days roll over at 06:00 JST."""
    return (t.astimezone(JST) - timedelta(hours=6)).strftime("%Y%m%d")


class GMOFeed:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.session = requests.Session()

    def _klines(self, interval: str, date: str) -> pd.DataFrame:
        resp = self.session.get(BASE_URL, params={"symbol": self.symbol, "interval": interval, "date": date}, timeout=15)
        resp.raise_for_status()
        return parse_klines(resp.json())

    def candles(self, interval: str, now: datetime, days: int) -> pd.DataFrame:
        frames = [self._klines(interval, gmo_business_date(now - timedelta(days=d))) for d in range(days, -1, -1)]
        df = pd.concat(frames)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def bid_ask(self) -> tuple[float, float]:
        resp = self.session.get(TICKER_URL, params={"symbol": self.symbol}, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != 0:
            raise RuntimeError(f"GMO API error: {payload.get('messages')}")
        row = payload["data"][0]
        return float(row["bid"]), float(row["ask"])

    def fng(self) -> pd.Series:
        resp = self.session.get(FNG_URL, params={"limit": 10, "format": "json"}, timeout=15)
        resp.raise_for_status()
        return parse_fng(resp.json())


class HyperliquidFeed(GMOFeed):
    """Hyperliquid perp prices (USD) and hourly funding, mainnet public data. No account needed."""

    has_funding = True
    INTERVALS = {"1min": "1m", "1hour": "1h"}

    def __init__(self, coin: str = "ADA", url: str | None = None):
        from btcbot import hyperliquid as hl
        self.hl = hl
        self.symbol = coin
        self.url = url or hl.MAINNET
        self.session = requests.Session()

    def candles(self, interval: str, now: datetime, days: int) -> pd.DataFrame:
        end = int(now.timestamp() * 1000)
        start = end - (days + 1) * 86_400_000
        rows = self.hl._post(self.url, {"type": "candleSnapshot", "req": {
            "coin": self.symbol, "interval": self.INTERVALS[interval], "startTime": start, "endTime": end}}, self.session)
        return self.hl.parse_candles(rows)

    def bid_ask(self) -> tuple[float, float]:
        book = self.hl._post(self.url, {"type": "l2Book", "coin": self.symbol}, self.session)
        bids, asks = book["levels"]
        return float(bids[0]["px"]), float(asks[0]["px"])

    def funding(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        return self.hl.fetch_funding(self.symbol, int(start.timestamp() * 1000), int(end.timestamp() * 1000), self.url)


class PaperTrader:
    def __init__(self, folder: Path, cfg: PaperConfig | None = None, acct_cfg: AccountConfig | None = None,
                 strat: BreakoutConfig | None = None, advisor=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or PaperConfig()
        self.acct_cfg = acct_cfg or AccountConfig(initial_jpy=self.cfg.initial_jpy)
        self.strat = strat or BreakoutConfig(allow_short=self.cfg.allow_short, max_leverage=self.cfg.max_leverage,
                                             fng_short_max=self.cfg.fng_short_max)
        self.advisor = advisor  # optional ClaudeAdvisor: may halve or cancel a new entry, never enlarge it
        self.state_path = self.folder / "state.json"
        self.state = self._load()

    # ---------- state ----------
    def _new_rc(self) -> RiskController:
        rc = RiskController(StrategyConfig(kelly_fraction=self.strat.kelly_fraction), 0, 0.0)
        rc.seed_prior(self.strat.prior_win_rate, self.strat.prior_payoff, self.strat.prior_avg_loss, self.strat.prior_weight)
        return rc

    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        rc = self._new_rc()
        return {"collateral": self.cfg.initial_jpy, "qty": 0.0, "entry": 0.0, "fees": 0.0, "leverage_fees": 0.0,
                "side": 0, "sig_entry": 0.0, "entry_time": None, "stop": 0.0, "best": 0.0, "sig_days": 0,
                "last_check": None, "last_bar": None, "loss_cuts": 0,
                "rc": {"alpha": rc.alpha, "beta": rc.beta, "wins": rc.wins, "losses": rc.losses, "seeded": rc._seeded},
                "config": asdict(self.cfg)}

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.state_path)

    def _account(self) -> MarginAccount:
        s = self.state
        return MarginAccount(s["collateral"], s["qty"], s["entry"], s["fees"], s["leverage_fees"])

    def _store_account(self, a: MarginAccount) -> None:
        self.state.update(collateral=a.collateral, qty=a.qty, entry=a.entry, fees=a.fees, leverage_fees=a.leverage_fees)

    def _rc(self) -> RiskController:
        rc = self._new_rc()
        r = self.state["rc"]
        rc.alpha, rc.beta, rc.wins, rc.losses, rc._seeded = r["alpha"], r["beta"], list(r["wins"]), list(r["losses"]), r["seeded"]
        return rc

    def _store_rc(self, rc: RiskController) -> None:
        self.state["rc"] = {"alpha": rc.alpha, "beta": rc.beta, "wins": rc.wins, "losses": rc.losses, "seeded": rc._seeded}

    def _append(self, name: str, row: dict) -> None:
        path = self.folder / name
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

    def _hourly(self, feed, now: datetime) -> pd.DataFrame:
        """Recent 1h candles, cached so each run only downloads the last two GMO days."""
        path = self.folder / f"{self.cfg.symbol}_1hour.csv"
        cached = None
        if path.exists():
            cached = pd.read_csv(path, parse_dates=["open_time"], index_col="open_time")
        fresh = feed.candles("1hour", now, 1 if cached is not None and len(cached) else self.cfg.history_days)
        df = fresh if cached is None else pd.concat([cached, fresh])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df.index >= pd.Timestamp(now) - pd.Timedelta(days=self.cfg.history_days + 2)]
        df.to_csv(path)
        return df

    # ---------- trading ----------
    def _close(self, acct: MarginAccount, rc: RiskController, when, px: float, reason: str) -> None:
        s = self.state
        side = s["side"]
        acct.set_position(0.0, px, self.acct_cfg.fee_rate)
        carry = s.get("sig_funding", 0.0) + self.acct_cfg.leverage_fee_per_day * s["sig_days"]
        ret = side * (px / s["sig_entry"] - 1) - carry - 2 * self.acct_cfg.fee_rate
        rc.update(ret)
        self._append("trades.csv", {"entry_time": s["entry_time"], "exit_time": str(when), "side": side,
                                    "entry_px": s["sig_entry"], "exit_px": px, "qty": s["entry_qty"], "return": ret,
                                    "reason": reason, "equity_after": acct.equity(px)})
        s.update(side=0, stop=0.0, best=0.0, sig_days=0, sig_funding=0.0, entry_time=None)

    def step(self, feed, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        s, acct, rc = self.state, self._account(), self._rc()
        last_check = pd.Timestamp(s["last_check"]) if s["last_check"] else None

        # 1. carrying cost since the last run (GMO: fee per 06:00 JST passed; Hyperliquid: hourly funding),
        #    then stop / loss cut on every minute since the last run
        if s["side"] and last_check is not None and getattr(feed, "has_funding", False):
            rates = feed.funding(last_check, pd.Timestamp(now))
            rates = rates[(rates.index > last_check) & (rates.index <= pd.Timestamp(now))]
            if len(rates) and acct.qty:
                bid, ask = feed.bid_ask()
                acct.charge(float(rates.sum()) * acct.qty * (bid + ask) / 2)  # positive rate: longs pay
                s["sig_funding"] = s.get("sig_funding", 0.0) + s["side"] * float(rates.sum())
        elif s["side"] and last_check is not None:
            n = rollovers_between(last_check, pd.Timestamp(now), self.acct_cfg.rollover_hour_jst)
            if n:
                s["sig_days"] += n
                if acct.qty:
                    bid, ask = feed.bid_ask()
                    acct.charge(n * self.acct_cfg.leverage_fee_per_day * abs(acct.qty) * (bid + ask) / 2)
        if s["side"] and last_check is not None:
            days = min(3, (pd.Timestamp(now) - last_check).days + 1)
            minutes = feed.candles("1min", now, days)
            minutes = minutes[minutes.index >= last_check.floor("min")]
            for t, bar in minutes.iterrows():
                side = s["side"]
                if (side > 0 and bar["low"] <= s["stop"]) or (side < 0 and bar["high"] >= s["stop"]):
                    px = min(bar["open"], s["stop"]) if side > 0 else max(bar["open"], s["stop"])
                    self._close(acct, rc, t, px * (1 - self.acct_cfg.slippage * side), "trailing_stop")
                    break
                if acct.qty:
                    worst = bar["low"] if acct.qty > 0 else bar["high"]
                    if risk.maintenance_ratio(acct.equity(worst), acct.qty, worst) <= risk.LOSS_CUT_RATIO:
                        lc = risk.loss_cut_price(acct.collateral, acct.qty, acct.entry)
                        px = min(bar["open"], lc) if acct.qty > 0 else max(bar["open"], lc)
                        self._close(acct, rc, t, px * (1 - self.acct_cfg.slippage * np.sign(acct.qty)), "loss_cut")
                        s["loss_cuts"] += 1
                        break

        # 2. act on newly closed 4h candles
        hourly = self._hourly(feed, now)
        tf = pd.Timedelta(self.cfg.timeframe)
        bars = resample(hourly, self.cfg.timeframe)
        bars = bars[bars.index + tf <= pd.Timestamp(now)]  # completed candles only
        action = "none"
        fng_now = None
        if len(bars):
            fng = None
            if self.strat.fng_short_max < 100 or self.advisor is not None:
                try:
                    fng = feed.fng()
                except Exception as e:  # no index -> no shorts this time, never a guess
                    fng = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
                    action = f"none (Fear & Greed unavailable: {e})"
            f = compute_features(bars, self.strat, fng)
            if fng is not None:
                fng_now = None if f["fng"].isna().iloc[-1] else float(f["fng"].iloc[-1])
                if fng_now is None and self.strat.fng_short_max < 100:
                    f.loc[f.index[-1], "short_signal"] = False
            last = f.index[-1]
            problem = check_bar_data(hourly, last, tf, now)
            new_bar = s["last_bar"] is None or last > pd.Timestamp(s["last_bar"])
            if s["last_bar"] is None:
                action = "started: waiting for the next 4h candle to close"
            elif new_bar and problem:
                # skip rather than guess; the bar is retried on the next run
                action = f"none (data check: {problem})"
            elif new_bar:
                row = f.iloc[-1]
                direction = 1 if row["long_signal"] else -1 if row["short_signal"] else 0
                key = f"{last.isoformat()}:{'long' if direction > 0 else 'short'}"
                side_before, result = s["side"], "no_signal"
                if s["side"]:
                    best = max(s["best"], row["high"]) if s["side"] > 0 else min(s["best"], row["low"])
                    trail = best - s["side"] * self.strat.atr_mult * row["atr"]
                    s["best"] = best
                    s["stop"] = max(s["stop"], trail) if s["side"] > 0 else min(s["stop"], trail)
                    action = f"trail stop -> {s['stop']:.3f}"
                    result = "in_position" if direction else "no_signal"
                elif direction and np.isnan(row["atr"]):
                    result = "no_atr"
                elif direction and s.get("last_signal_key") == key:
                    result = "duplicate"
                    action = f"duplicate signal {key} ignored"
                elif direction and paused(self.folder):
                    result = "paused"
                    action = f"{'long' if direction > 0 else 'short'} signal not taken: PAUSE file present"
                elif direction:
                    s["last_signal_key"] = key
                    bid, ask = feed.bid_ask()
                    px = ask if direction > 0 else bid
                    lev = min(rc.leverage(), self.strat.max_leverage, risk.MAX_LEVERAGE)
                    eq = acct.equity(px)
                    note = ""
                    if self.advisor is not None:
                        ctx = entry_context(bars, row, direction, lev, px, eq, fng, self.folder / "trades.csv",
                                            symbol=self.cfg.symbol)
                        d = self.advisor.decide(ctx)
                        lev *= min(max(d.multiplier, 0.0), 1.0)  # shrink or cancel only
                        self._append("claude.csv", {"time": str(now), "signal": "long" if direction > 0 else "short",
                                                    "price": px, "rules_leverage": ctx["rules_leverage"],
                                                    "action": d.action, "leverage": round(lev, 3),
                                                    "reason": d.reason, "error": d.error or ""})
                        note = f" (Claude: {d.action})"
                    target = risk.cap_quantity(direction * lev * eq / px, px, eq, self.acct_cfg.fee_rate)
                    qty = _round_qty(target, self.acct_cfg)
                    if qty == 0:
                        result = "claude_skip" if note and lev == 0 else "skipped_size"
                        action = f"{'long' if direction > 0 else 'short'} signal skipped{note}"
                    else:
                        result = "entered"
                        acct.set_position(qty, px, self.acct_cfg.fee_rate)
                        s.update(side=direction, sig_entry=px, entry_time=str(now), sig_days=0, sig_funding=0.0, best=px,
                                 stop=px - direction * self.strat.atr_mult * row["atr"], entry_qty=acct.qty)
                        action = (f"{'long' if direction > 0 else 'short'} {acct.qty:+.0f} @ {px:.5g}, "
                                  f"stop {s['stop']:.5g}{note}")
                self._append("signals.csv", {"bar": str(last), "time": str(now), "close": row["close"],
                                             "long_signal": bool(row["long_signal"]),
                                             "short_signal": bool(row["short_signal"]),
                                             "side_before": side_before, "side_after": s["side"], "result": result,
                                             "entry_px": s["sig_entry"] if result == "entered" else None})
            if not (new_bar and problem and s["last_bar"] is not None):
                s["last_bar"] = str(last)

        mark = float(hourly["close"].iloc[-1]) if len(hourly) else s["entry"]
        s["last_check"] = str(now)
        self._store_account(acct)
        self._store_rc(rc)
        self._save()
        status = {"time": str(now), "price": mark, "equity": round(acct.equity(mark)), "qty": acct.qty,
                  "side": s["side"], "stop": float(f"{s['stop']:.5g}"), "leverage_next": round(rc.leverage(), 3),
                  "action": action}
        if fng_now is not None:
            status["fng"] = fng_now
        self._append("equity.csv", status)
        return status


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Paper trade the 4h breakout on live GMO or Hyperliquid prices")
    p.add_argument("command", choices=["step", "status"])
    p.add_argument("--venue", choices=["gmo", "hl"], default="gmo",
                   help="hl = Hyperliquid ADA perp in USD (200 USD account, hourly funding, 0.045%% fee)")
    p.add_argument("--usd", type=float, default=200.0, help="starting balance for --venue hl")
    p.add_argument("--folder", default="data/paper")
    p.add_argument("--symbol", default=PaperConfig.symbol)
    p.add_argument("--no-short", action="store_true")
    p.add_argument("--max-leverage", type=float, default=risk.MAX_LEVERAGE)
    p.add_argument("--jpy", type=float, default=PaperConfig.initial_jpy)
    p.add_argument("--fng-short-max", type=float, default=PaperConfig.fng_short_max,
                   help="only short when the Fear & Greed Index is at or below this, e.g. 50")
    p.add_argument("--claude", action="store_true",
                   help="ask Claude before each entry (go / half / skip); needs ANTHROPIC_API_KEY")
    a = p.parse_args(argv)

    acct_cfg = None
    if a.venue == "hl":
        from btcbot.hyperliquid import account_config
        symbol = "ADA" if a.symbol == PaperConfig.symbol else a.symbol
        acct_cfg = account_config(a.usd)
        cfg = PaperConfig(symbol=symbol, allow_short=not a.no_short, max_leverage=a.max_leverage, initial_jpy=a.usd,
                          fng_short_max=a.fng_short_max)
    else:
        cfg = PaperConfig(symbol=a.symbol, allow_short=not a.no_short, max_leverage=a.max_leverage,
                          initial_jpy=a.jpy, fng_short_max=a.fng_short_max)
    advisor = None
    if a.claude and a.command == "step":
        from btcbot.advisor import ClaudeAdvisor
        advisor = ClaudeAdvisor()
    trader = PaperTrader(Path(a.folder), cfg, acct_cfg, advisor=advisor)
    if a.command == "status":
        s = trader.state
        print(json.dumps({k: s[k] for k in ("collateral", "qty", "entry", "side", "stop", "last_check", "last_bar",
                                            "fees", "leverage_fees", "loss_cuts")}, indent=2, default=str))
        trades = Path(a.folder) / "trades.csv"
        if trades.exists():
            t = pd.read_csv(trades)
            print(f"trades {len(t)}, win rate {(t['return'] > 0).mean():.0%}, avg {t['return'].mean():+.2%}")
        return
    feed = HyperliquidFeed(cfg.symbol) if a.venue == "hl" else GMOFeed(cfg.symbol)
    print(json.dumps(trader.step(feed), default=str))


if __name__ == "__main__":
    main()
