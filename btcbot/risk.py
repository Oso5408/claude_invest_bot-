"""Hard risk limits. These are module constants on purpose: no config, CLI flag or
strategy can raise them. Every order in the leverage backtest goes through
cap_quantity(), and MarginAccount re-checks the result before filling.

GMO Coin crypto FX (暗号資産FX) rules used here:
- Required margin is 50% of position value (Japanese law caps crypto at 2x).
- Loss cut when equity / required margin falls to 75% or below.
"""

from __future__ import annotations

from typing import Final

MAX_LEVERAGE: Final[float] = 2.0
MARGIN_RATE: Final[float] = 1.0 / MAX_LEVERAGE
LOSS_CUT_RATIO: Final[float] = 0.75


class LeverageLimitError(RuntimeError):
    pass


def cap_quantity(target_qty: float, price: float, equity: float) -> float:
    """Shrink a signed target position so its value is at most MAX_LEVERAGE x equity."""
    if equity <= 0 or price <= 0:
        return 0.0
    max_qty = MAX_LEVERAGE * equity / price
    return max(-max_qty, min(max_qty, target_qty))


def check_leverage(qty: float, price: float, equity: float) -> None:
    """Raise if a position is above the cap. A small tolerance absorbs float rounding."""
    if qty == 0:
        return
    if equity <= 0 or abs(qty) * price > MAX_LEVERAGE * equity * (1 + 1e-9):
        raise LeverageLimitError(f"position {qty} @ {price} exceeds {MAX_LEVERAGE}x of equity {equity}")


def maintenance_ratio(equity: float, qty: float, price: float) -> float:
    """Equity divided by required margin. 1.0 means exactly 2x leverage."""
    if qty == 0:
        return float("inf")
    return equity / (abs(qty) * price * MARGIN_RATE)


def loss_cut_price(collateral: float, qty: float, entry: float) -> float:
    """Price at which the maintenance ratio hits LOSS_CUT_RATIO.

    Equity(P) = collateral + qty * (P - entry); required margin = |qty| * P * MARGIN_RATE.
    """
    k = LOSS_CUT_RATIO * MARGIN_RATE
    s = 1.0 if qty > 0 else -1.0
    return (qty * entry - collateral) / (qty * (1 - k * s))
