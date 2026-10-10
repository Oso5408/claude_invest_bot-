import json
from datetime import timedelta

import pandas as pd

from btcbot import monitor
from btcbot.paper import PaperConfig, PaperTrader, check_bar_data
from tests.test_paper import FakeFeed, _minutes


def _run_paper(folder, m, cfg, every="30min", stop=None):
    feed = FakeFeed(m)
    times = pd.date_range(m.index[0] + pd.Timedelta(days=3), stop or m.index[-1], freq=every)
    for t in times:
        PaperTrader(folder, cfg).step(feed, t.to_pydatetime())
    return feed, times[-1].to_pydatetime()


def test_check_bar_data():
    idx = pd.date_range("2026-10-01 00:00", periods=8, freq="h", tz="UTC")
    hourly = pd.DataFrame({"close": 1.0}, index=idx)
    tf = pd.Timedelta("4h")
    now = pd.Timestamp("2026-10-01 08:02", tz="UTC").to_pydatetime()
    assert check_bar_data(hourly, idx[4], tf, now) is None
    assert "incomplete" in check_bar_data(hourly.drop(idx[6]), idx[4], tf, now)
    assert "stale" in check_bar_data(hourly, idx[4], tf, now + timedelta(hours=3))


def test_missed_signal_check_ok_then_alerts_on_missed(tmp_path):
    m = _minutes()
    folder = tmp_path / "paper-a"
    cfg = PaperConfig(history_days=6, allow_short=False)
    feed, now = _run_paper(folder, m, cfg)
    sig = pd.read_csv(folder / "signals.csv")
    assert (sig["result"] == "entered").any()
    sent = []
    res = monitor.run_check(tmp_path, now, days=10, send=sent.append, feeds={("gmo", "ADA_JPY"): feed})
    rows = res["paper-a"]
    assert rows and all(r["status"] == "ok" for r in rows), [r for r in rows if r["status"] != "ok"]
    assert sent == []

    # pretend the trader never processed the candle that gave the entry signal
    entered = sig[sig["result"] == "entered"]["bar"].iloc[0]
    sig[sig["bar"] != entered].to_csv(folder / "signals.csv", index=False)
    res = monitor.run_check(tmp_path, now, days=10, send=sent.append, feeds={("gmo", "ADA_JPY"): feed})
    assert [r["status"] for r in res["paper-a"] if str(r["bar"]) == entered] == ["missed_unprocessed"]
    assert len(sent) == 1 and "漏單" in sent[0]
    sent.clear()  # the same miss is only reported once
    monitor.run_check(tmp_path, now, days=10, send=sent.append, feeds={("gmo", "ADA_JPY"): feed})
    assert sent == []


def test_pause_file_blocks_entries_and_check_flags_it(tmp_path):
    m = _minutes()
    folder = tmp_path / "paper-a"
    (tmp_path / "PAUSE").write_text("")
    feed, now = _run_paper(folder, m, PaperConfig(history_days=6, allow_short=False))
    assert not (folder / "trades.csv").exists()
    assert (pd.read_csv(folder / "signals.csv")["result"] == "paused").any()
    sent = []
    monitor.run_check(tmp_path, now, days=10, send=sent.append, feeds={("gmo", "ADA_JPY"): feed})
    assert sent and all("paused" in s for s in sent)


def test_duplicate_signal_key_is_not_traded_twice(tmp_path):
    m = _minutes()
    folder = tmp_path / "paper-a"
    cfg = PaperConfig(history_days=6, allow_short=False)
    feed, _ = _run_paper(folder, m, cfg)
    sig = pd.read_csv(folder / "signals.csv")
    entered = sig[sig["result"] == "entered"]["bar"].iloc[0]
    # roll the state back to just before that candle was processed, but keep the signal key
    st = json.loads((folder / "state.json").read_text())
    st.update(side=0, qty=0.0, last_bar=str(pd.Timestamp(entered) - pd.Timedelta(hours=4)))
    (folder / "state.json").write_text(json.dumps(st))
    t = pd.Timestamp(entered) + pd.Timedelta(hours=4, minutes=5)
    out = PaperTrader(folder, cfg).step(feed, t.to_pydatetime())
    assert out["action"].startswith("duplicate") and out["side"] == 0


def test_watch_offline_errors_leverage_and_recovery(tmp_path):
    m = _minutes()
    folder = tmp_path / "paper-a"
    feed, now = _run_paper(folder, m, PaperConfig(history_days=6, allow_short=False))
    sent = []
    (tmp_path / "paper-a.log").write_text("Traceback: an old error from before the monitor started\n")
    assert monitor.watch(tmp_path, now + timedelta(minutes=5), send=sent.append) == []  # old errors ignored

    later = now + timedelta(minutes=30)
    msgs = monitor.watch(tmp_path, later, send=sent.append)
    assert len(msgs) == 1 and "冇更新" in msgs[0]
    assert monitor.watch(tmp_path, later + timedelta(minutes=5), send=sent.append) == []  # no spam
    assert len(monitor.watch(tmp_path, later + timedelta(hours=7), send=sent.append)) == 1  # reminder

    with (tmp_path / "paper-a.log").open("a") as fh:
        fh.write('{"ok": 1}\nTraceback (most recent call last):\nValueError: boom\n')
    st = json.loads((folder / "state.json").read_text())
    eq = pd.read_csv(folder / "equity.csv")
    st["qty"] = 3 * eq["equity"].iloc[-1] / eq["price"].iloc[-1]  # 3x, over the cap
    (folder / "state.json").write_text(json.dumps(st))
    PaperTrader(folder, PaperConfig(history_days=6, allow_short=False))  # no-op load
    eq.loc[len(eq)] = eq.iloc[-1]
    eq.loc[len(eq) - 1, "time"] = str(later + timedelta(hours=8))
    eq.to_csv(folder / "equity.csv", index=False)
    msgs = monitor.watch(tmp_path, later + timedelta(hours=8, minutes=1), send=sent.append)
    text = "\n".join(msgs)
    assert "已恢復" in text and "槓桿" in text and "ValueError: boom" in text
    assert monitor.watch(tmp_path, later + timedelta(hours=8, minutes=2), send=sent.append) == []  # log already read


def test_report_writes_summary(tmp_path):
    m = _minutes()
    folder = tmp_path / "paper-a"
    feed, now = _run_paper(folder, m, PaperConfig(history_days=6, allow_short=False), every="5min",
                           stop=m.index[0] + pd.Timedelta(days=7, hours=6))
    sent = []
    out = monitor.report(tmp_path, now, send=sent.append, feeds={("gmo", "ADA_JPY"): feed})
    text = out.read_text()
    assert "[paper-a] 運行" in text and "漏單 0" in text and sent[-1].startswith("📋")


def test_check_failure_is_reported_once(tmp_path):
    m = _minutes()
    folder = tmp_path / "paper-a"
    _, now = _run_paper(folder, m, PaperConfig(history_days=6, allow_short=False))

    class Down:
        def candles(self, *a):
            raise RuntimeError("MAINTENANCE")

    sent = []
    for i in range(3):
        monitor.run_check(tmp_path, now + timedelta(hours=i), send=sent.append, feeds={("gmo", "ADA_JPY"): Down()})
    assert len(sent) == 1 and "MAINTENANCE" in sent[0]
