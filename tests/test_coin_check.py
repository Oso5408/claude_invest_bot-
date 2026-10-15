import numpy as np
import pandas as pd

from btcbot import coin_check as cc
from btcbot.data import load_csv, load_fng
from btcbot.hyperliquid import load_funding


def test_runs_every_setup_and_compares_with_ada_on_the_same_dates():
    ada = load_csv("data/HL_ADA_4h.csv")
    fund = load_funding("data/HL_ADA_funding.csv")
    fng = load_fng("data/fear_greed.csv")
    near = ada.iloc[2000:] * np.array([4.0, 4.0, 4.0, 4.0, 1.0])  # a made-up "NEAR" listed later
    rows = cc.check(["NEAR"], {"NEAR": (near, fund), "ADA": (ada, fund)}, fng)
    assert [r["setup"] for r in rows] == list(cc.SETUPS)
    for r in rows:
        assert r["start"] == f"{near.index[0]:%Y-%m-%d}" and r["trades"] > 0
        # same prices, same dates: the ADA yardstick should land close (only order rounding differs)
        assert abs(r["return"] - r["ADA_same_dates"]) < 0.1
        assert r["return_slip_0.15%"] < r["return"]
        assert sum(1 for _ in r["half_years"]) >= 3
    text = cc.report(rows)
    assert "NEAR / 1x long" in text and "ADA same dates" in text and "half years:" in text
