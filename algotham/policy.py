from dataclasses import dataclass
from enum import Enum


class BuyPolicy(Enum):
    CALCEL_IF_NOT_ENOUGH = (
        "cancel_if_not_enough"  # Cancel buy asset if not enoygh cash is left
    )
    BUY_AS_MUCH_AS_POSSIBLE = "buy_as_much_as_possible"  # Buy as much as possible asset with current holding cash if not enoygh cash is left


class SellPolicy(Enum):
    CALCEL_IF_NOT_ENOUGH = (
        "cancel_if_not_enough"  # Cancel sell asset if not enoygh asset is left
    )
    SELL_AS_MUCH_AS_POSSIBLE = "sell_as_much_as_possible"  # Sell as much as possible (all) asset with current holding asset if not enoygh asset is left
    SELL_ALWAYS = "sell_always"  # Sell the amount of assets specified by order even if not enough assets to sell (i.e. allow whole/partial short selling)


class CloseShortPolicy(Enum):
    CALCEL_IF_NOT_ENOUGH = (
        "cancel_if_not_enough"  # Cancel buy asset if not enoygh cash is left
    )
    BUY_AS_MUCH_AS_POSSIBLE = "buy_as_much_as_possible"  # Buy as much as possible asset with current holding cash if not enoygh cash is left


class CloseLongPolicy(Enum):
    CALCEL_IF_NOT_ENOUGH = (
        "cancel_if_not_enough"  # Cancel sell asset if not enoygh asset is left
    )
    SELL_AS_MUCH_AS_POSSIBLE = "sell_as_much_as_possible"  # Sell as much as possible (all) asset with current holding asset if not enoygh asset is left
    SELL_ALWAYS = "sell_always"  # Sell the amount of assets specified by order even if not enough assets to sell (i.e. allow whole/partial short selling)


@dataclass
class BuySellPolicy:
    buy_policy: BuyPolicy
    sell_policy: SellPolicy
    close_long_policy: CloseLongPolicy
    close_short_policy: CloseShortPolicy


DEFAULT_BUYSELL_POLICY = BuySellPolicy(
    buy_policy=BuyPolicy.BUY_AS_MUCH_AS_POSSIBLE,
    sell_policy=SellPolicy.SELL_ALWAYS,
    close_long_policy=CloseLongPolicy.SELL_ALWAYS,
    close_short_policy=CloseShortPolicy.BUY_AS_MUCH_AS_POSSIBLE,
)
