import pytest

from btcbot import rule_selftest as rs


@pytest.mark.parametrize("account,scenario", list(rs.EXPECTED))
def test_rule_account_takes_the_right_side_and_trails_out(account, scenario):
    r = rs.run_account(account, scenario)
    assert r["ok"], r
