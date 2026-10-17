import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from btcbot import monitor
from btcbot import xs_momentum as xs
from btcbot import xs_paper

DATES = pd.date_range("2026-07-01", periods=110, freq="D")


def frame(close, dates=DATES):
    close = np.asarray(close, float)
    o = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, close) * 1.001, "low": np.minimum(o, close) * 0.999,
                         "close": close, "volume": 1e6}, index=dates)


def daily(btc_trend=1.0, seed=0, dates=DATES):
    rng = np.random.default_rng(seed)
    out = {f"C{k:02d}": frame(100 * np.exp(np.cumsum(0.001 * k + 0.02 * rng.standard_normal(len(dates)))), dates)
           for k in range(12)}
    out["BTC"] = frame(100 * np.exp(np.linspace(0, btc_trend, len(dates))), dates)
    return out


def now_for(dates=DATES, hour=0, minute=10):
    d = dates[-1]
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=timezone.utc)


def test_first_run_enters_top3_and_btc_when_switch_is_on(tmp_path):
    p = xs.build_panel(daily(), {})
    st = xs_paper.step(tmp_path / "daily-xs-long3", "xs", p, {}, now_for())
    assert st["regime_on"] and len(st["positions"]) == 3
    notional = sum(v["qty"] * v["px"] for v in st["positions"].values())
    assert abs(notional - st["equity"]) / st["equity"] < 0.01          # 1x
    assert st["equity"] < 200 and st["fees"] > 0
    b = xs_paper.step(tmp_path / "daily-btc-sma50", "btc", p, {}, now_for())
    assert list(b["positions"]) == ["BTC"]


def test_same_day_twice_does_nothing(tmp_path):
    p = xs.build_panel(daily(), {})
    f = tmp_path / "daily-xs-long3"
    first = xs_paper.step(f, "xs", p, {}, now_for())
    again = xs_paper.step(f, "xs", p, {"C11": 0.5}, now_for())
    assert again == first
    assert len(pd.read_csv(f / "equity.csv")) == 1


def test_switch_off_goes_to_cash_and_funding_is_paid(tmp_path):
    f = tmp_path / "daily-xs-long3"
    p = xs.build_panel(daily(), {})
    st = xs_paper.step(f, "xs", p, {}, now_for())
    coin = sorted(st["positions"])[0]
    dates2 = pd.date_range(DATES[0], periods=len(DATES) + 1, freq="D")
    data = daily(dates=dates2)
    btc = data["BTC"]["close"].values.copy()
    btc[-2] = btc[-60:-2].min() * 0.5                                   # yesterday's close far below the average
    data["BTC"] = frame(btc, dates2)
    p2 = xs.build_panel(data, {})
    st2 = xs_paper.step(f, "xs", p2, {coin: 0.001}, now_for(dates2))
    assert not st2["regime_on"] and st2["positions"] == {}
    assert st2["funding"] > 0
    trades = pd.read_csv(f / "trades.csv")
    assert (trades["reason"] == "BTC switch off").sum() == 3


def test_run_refuses_late_and_skips_when_done(tmp_path, monkeypatch):
    monkeypatch.setattr(xs_paper, "fetch_daily", lambda now: daily())
    monkeypatch.setattr(xs_paper, "fetch_funding", lambda *a, **k: pd.Series(dtype=float))
    with pytest.raises(SystemExit):
        xs_paper.run(tmp_path, now_for(hour=5))
    xs_paper.run(tmp_path, now_for())
    assert (tmp_path / "daily-xs-long3" / "state.json").exists()
    xs_paper.run(tmp_path, now_for(hour=5))                             # already done today: no error


def test_monitor_sees_daily_accounts(tmp_path, monkeypatch):
    monkeypatch.setattr(xs_paper, "fetch_daily", lambda now: daily())
    xs_paper.run(tmp_path, now_for())
    sent = []
    msgs = monitor.watch(tmp_path, now=now_for(hour=12), send=sent.append)
    assert not any("daily-" in m for m in msgs)
    later = datetime(2026, 12, 31, tzinfo=timezone.utc)
    msgs = monitor.watch(tmp_path, now=later, send=sent.append)
    assert any("daily-xs-long3" in m and "冇行" in m for m in msgs)
    st = json.loads((tmp_path / "daily-btc-sma50" / "state.json").read_text())
    assert st["equity"] < 200
    out = monitor.report(tmp_path, now=now_for(hour=12), send=sent.append)
    assert "daily-xs-long3" in out.read_text() and "BTC 開關開" in out.read_text()
