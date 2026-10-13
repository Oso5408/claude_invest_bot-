import numpy as np
import pandas as pd

from btcbot import risk
from btcbot.pullback import PullbackConfig, hourly_direction, run

ZERO = dict(taker_fee=0.0, maker_fee=0.0, slippage=0.0)


def random_walk(n: int, seed: int, substeps: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    r = rng.normal(0, 0.0018 / np.sqrt(substeps), (n, substeps))
    p = np.exp(np.cumsum(r.ravel())).reshape(n, substeps)
    o = np.r_[1.0, p[:-1, -1]]
    return pd.DataFrame({"open": o, "high": np.maximum(p.max(1), o), "low": np.minimum(p.min(1), o),
                         "close": p[:, -1], "volume": 1.0}, index=idx)


def test_no_lookahead_trades_before_a_cut_are_unchanged():
    df = random_walk(20000, 1)
    cut = df.index[12000]
    full = run(df, PullbackConfig()).trades
    part = run(df[df.index < cut], PullbackConfig()).trades
    done = full[pd.to_datetime(full.exit_time) < cut - pd.Timedelta(hours=1)]
    pd.testing.assert_frame_equal(done.reset_index(drop=True), part.iloc[:len(done)].reset_index(drop=True))


def test_hourly_direction_uses_only_finished_hours():
    df = random_walk(3000, 2)
    d = hourly_direction(df, PullbackConfig())
    # changing the last bar of an hour may change the direction from that bar's close on, never before it
    t = 2000 - (2000 % 12) + 11  # last 5-minute bar of an hour
    bumped = df.copy()
    bumped.iloc[t, bumped.columns.get_loc("close")] *= 1.05
    d2 = hourly_direction(bumped, PullbackConfig())
    assert (d[:t] == d2[:t]).all()


def test_random_walk_has_no_edge_at_zero_cost():
    # a fine price path (fills at the trigger level are then close to reality)
    rs = [run(random_walk(60000, s, substeps=100), PullbackConfig(**ZERO)).stats["avg_R"] for s in range(2)]
    assert abs(np.mean(rs)) < 0.08


def test_position_never_above_2x():
    tr = run(random_walk(20000, 3), PullbackConfig(risk_frac=0.05)).trades
    assert len(tr) > 0
    assert tr.leverage.max() <= risk.MAX_LEVERAGE + 1e-9
    assert tr.leverage.max() > 1.9  # the cap was reached, so the check means something
