from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fastprogress import progress_bar as pb

from .order import Order, OrderStatus, OrderType
from .policy import (
    DEFAULT_BUYSELL_POLICY,
    BuyPolicy,
    BuySellPolicy,
    CloseLongPolicy,
    CloseShortPolicy,
    SellPolicy,
)
from .portfolio import Portfolio
from .strategy import BaseStrategy
from .utils.logger import Logger


class BackTest:
    def __init__(
        self,
        dt_index: pd.DatetimeIndex,
        strategy: BaseStrategy,
        init_portfolio: Portfolio = Portfolio(init_cash=10000),
        execution_lag: int = 1,
        df_ref_data: Optional[Dict[str, pd.DataFrame]] = None,
        buysell_policy: BuySellPolicy = DEFAULT_BUYSELL_POLICY,
    ):
        """_summary_

        Args:
            dt_index (pd.DatetimeIndex): _description_
            strategy (BaseStrategy): _description_
            init_portfolio (Portfolio, optional): _description_. Defaults to Portfolio(init_cash=10000).
            execution_lag (int, optional): _description_. Defaults to 1.
            df_ref_data (Optional[Dict[str, pd.DataFrame]], optional): _description_. Defaults to None.
            buysell_policy (BuySellPolicy, optional): _description_. Defaults to DEFAULT_BUYSELL_POLICY.
        """
        self._dt_index = dt_index
        self._portfolio = init_portfolio
        self._strategy = strategy
        self._execution_lag = execution_lag
        self._orders = []
        self._df_ref_data = df_ref_data
        self._buysell_policy = buysell_policy

    def run(
        self, start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None
    ) -> None:
        """_summary_

        Args:
            start_dt (Optional[datetime], optional): _description_. Defaults to None.
            end_dt (Optional[datetime], optional): _description_. Defaults to None.
        """
        if start_dt is None:
            start_dt = self._dt_index.min()
        else:
            start_dt = pd.to_datetime(start_dt).tz_localize(tz="utc")
        if end_dt is None:
            end_dt = self._dt_index.max()
        else:
            end_dt = pd.to_datetime(end_dt).tz_localize(tz="utc")
        dt_indices = self._dt_index[
            (self._dt_index >= start_dt) & (self._dt_index <= end_dt)  # type: ignore
        ]
        for dt_idx in pb(dt_indices):
            idx = self.dt_idx2idx(dt_idx)
            self._strategy.next(dt_idx=dt_idx, idx=idx, bt=self)
            self._process_orders(dt_idx=dt_idx, idx=idx)
            self._portfolio.record(dt=dt_idx)

    def _buy_and_sell(
        self,
        order_type: OrderType,
        asset_name: str,
        sr_ref_price: pd.Series,
        abs_size: float,
        dt_idx: pd.Timestamp,
        idx: int,
        execution_lag: Optional[int] = None,
    ) -> Order:
        """_summary_

        Args:
            order_type (OrderType): _description_
            asset_name (str): _description_
            sr_ref_price (pd.Series): _description_
            abs_size (float): _description_
            dt_idx (pd.Timestamp): _description_
            idx (int): _description_
            execution_lag (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Order: _description_
        """
        if execution_lag is None:
            execution_lag = self._execution_lag
        order = Order(
            order_type=order_type,
            asset_name=asset_name,
            abs_size=abs_size,
            ordered_timestamp=dt_idx,
            price_at_ordertime=sr_ref_price.loc[dt_idx],
            execution_timestamp=self.idx2dt_idx(idx + execution_lag)
            if idx + execution_lag < len(self._dt_index)
            else None,
            sr_ref_price=sr_ref_price,
        )
        self._orders.append(order)
        self._strategy.on_order_accepted(order, self, dt_idx=dt_idx)
        return order

    def buy(
        self,
        asset_name: str,
        sr_ref_price: pd.Series,
        abs_size: float,
        dt_idx: pd.Timestamp,
        idx: int,
        execution_lag: Optional[int] = None,
    ) -> Order:
        """_summary_

        Args:
            asset_name (str): _description_
            sr_ref_price (pd.Series): _description_
            abs_size (float): _description_
            dt_idx (pd.Timestamp): _description_
            idx (int): _description_
            execution_lag (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Order: _description_
        """
        assert abs_size > 0, "size must be positive"
        return self._buy_and_sell(
            order_type=OrderType.BUY,
            asset_name=asset_name,
            sr_ref_price=sr_ref_price,
            abs_size=abs_size,
            dt_idx=dt_idx,
            idx=idx,
            execution_lag=execution_lag,
        )

    def sell(
        self,
        asset_name: str,
        sr_ref_price: pd.Series,
        abs_size: float,
        dt_idx: pd.Timestamp,
        idx: int,
        execution_lag: Optional[int] = None,
    ) -> Order:
        """_summary_

        Args:
            asset_name (str): _description_
            sr_ref_price (pd.Series): _description_
            abs_size (float): _description_
            dt_idx (pd.Timestamp): _description_
            idx (int): _description_
            execution_lag (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Order: _description_
        """
        assert abs_size > 0, "size must be positive"
        return self._buy_and_sell(
            order_type=OrderType.SELL,
            asset_name=asset_name,
            sr_ref_price=sr_ref_price,
            abs_size=abs_size,
            dt_idx=dt_idx,
            idx=idx,
            execution_lag=execution_lag,
        )

    def _close_position(
        self,
        order_type: OrderType,
        asset_name: str,
        sr_ref_price: pd.Series,
        dt_idx: pd.Timestamp,
        idx: int,
        abs_size: Optional[float] = None,
        execution_lag: Optional[int] = None,
        close_target_order_id: Optional[str] = None,
    ) -> Order:
        """_summary_

        Args:
            order_type (OrderType): _description_
            asset_name (str): _description_
            sr_ref_price (pd.Series): _description_
            abs_size (float): _description_
            dt_idx (pd.Timestamp): _description_
            idx (int): _description_
            execution_lag (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Order: _description_
        """
        if execution_lag is None:
            execution_lag = self._execution_lag
        order = Order(
            order_type=order_type,
            asset_name=asset_name,
            abs_size=abs_size,
            ordered_timestamp=dt_idx,
            price_at_ordertime=sr_ref_price.loc[dt_idx],
            execution_timestamp=self.idx2dt_idx(idx + execution_lag),
            sr_ref_price=sr_ref_price,
            close_target_order_id=close_target_order_id,
        )
        self._orders.append(order)
        self._strategy.on_order_accepted(order, self, dt_idx=dt_idx)
        return order

    def close_long(
        self,
        asset_name: str,
        sr_ref_price: pd.Series,
        dt_idx: pd.Timestamp,
        idx: int,
        abs_size: Optional[float] = None,
        execution_lag: Optional[int] = None,
        close_target_order_id: Optional[str] = None,
    ) -> Order:
        """_summary_

        Args:
            asset_name (str): _description_
            sr_ref_price (pd.Series): _description_
            abs_size (float): _description_
            dt_idx (pd.Timestamp): _description_
            idx (int): _description_
            execution_lag (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Order: _description_
        """
        if abs_size is not None:
            assert abs_size > 0, "size must be positive"
        return self._close_position(
            order_type=OrderType.CLOSE_LONG,
            asset_name=asset_name,
            sr_ref_price=sr_ref_price,
            abs_size=abs_size,
            dt_idx=dt_idx,
            idx=idx,
            execution_lag=execution_lag,
            close_target_order_id=close_target_order_id,
        )

    def close_short(
        self,
        asset_name: str,
        sr_ref_price: pd.Series,
        dt_idx: pd.Timestamp,
        idx: int,
        abs_size: Optional[float] = None,
        execution_lag: Optional[int] = None,
        close_target_order_id: Optional[str] = None,
    ) -> Order:
        """_summary_

        Args:
            asset_name (str): _description_
            sr_ref_price (pd.Series): _description_
            abs_size (float): _description_
            dt_idx (pd.Timestamp): _description_
            idx (int): _description_
            execution_lag (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Order: _description_
        """
        if abs_size is not None:
            assert abs_size > 0, "size must be positive"
        return self._close_position(
            order_type=OrderType.CLOSE_SHORT,
            asset_name=asset_name,
            sr_ref_price=sr_ref_price,
            abs_size=abs_size,
            dt_idx=dt_idx,
            idx=idx,
            execution_lag=execution_lag,
            close_target_order_id=close_target_order_id,
        )

    def _process_orders(self, dt_idx: pd.Timestamp, idx: int) -> None:
        """_summary_

        Args:
            dt_idx (pd.Timestamp): _description_
            idx (int): _description_
        """
        unprocess_orders = [
            order for order in self._orders if order.status == OrderStatus.UNPROCESSED
        ]
        for order in unprocess_orders:
            if (
                order.execution_timestamp is not None
                and order.execution_timestamp <= dt_idx
            ):
                self._process_single_order(order, dt_idx, idx)
                if (
                    order.executed_timestamp is not None
                    and order.execution_timestamp != order.executed_timestamp
                ):
                    Logger.w(
                        "Backtest._process_orders",
                        "order.execution_timestamp does not match order.executed_timestamp. "
                        f"dt_idx :{dt_idx}, order.execution_timestamp : {order.execution_timestamp}, "
                        f"order.executed_timestamp : {order.executed_timestamp}, order_id : {order.order_id}, "
                        f"order.order_type : {order.order_type}",
                    )
                if order.execution_timestamp != dt_idx:
                    assert "this does not happen", (order.execution_timestamp, dt_idx)

    def _process_single_order(
        self, order: Order, dt_idx: pd.Timestamp, idx: int
    ) -> None:
        """_summary_

        Args:
            order (Order): _description_
            dt_idx (pd.Timestamp): _description_
            idx (int): _description_
        """
        if order.status == OrderStatus.UNPROCESSED:
            price = order.sr_ref_price.loc[dt_idx]
            abs_size = order.abs_size
            if order.order_type == OrderType.BUY:
                assert (abs_size is not None) and (abs_size > 0), "this does not happen"
                self._process_single_buy_order(
                    dt_idx=dt_idx, order=order, price=price, abs_size=abs_size
                )
            elif order.order_type == OrderType.SELL:
                assert (abs_size is not None) and (abs_size > 0), "this does not happen"
                self._process_single_sell_order(
                    dt_idx=dt_idx, order=order, price=price, abs_size=abs_size
                )
            elif order.order_type == OrderType.CLOSE_LONG:
                self._process_single_close_long_order(
                    dt_idx=dt_idx, order=order, price=price, abs_size=abs_size
                )
            elif order.order_type == OrderType.CLOSE_SHORT:
                self._process_single_close_short_order(
                    dt_idx=dt_idx, order=order, price=price, abs_size=abs_size
                )

            self._strategy.on_order_processed(order, self, dt_idx=dt_idx)

    def dt_idx2idx(self, dt_idx: pd.Timestamp) -> int:
        """_summary_

        Args:
            dt_idx (pd.Timestamp): _description_

        Returns:
            int: _description_
        """
        return np.where(self._dt_index == dt_idx)[0][0]

    def idx2dt_idx(self, idx: int) -> Optional[pd.Timestamp]:
        """_summary_

        Args:
            idx (int): _description_

        Returns:
            pd.Timestamp: _description_
        """
        return self._dt_index[idx] if idx < len(self._dt_index) else None

    @property
    def order_history(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        order_dicts = [order.__dict__.copy() for order in self._orders]
        [o.pop("sr_ref_price") for o in order_dicts]
        return pd.DataFrame(order_dicts)

    @property
    def df_ref_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """_summary_

        Returns:
            Optional[Dict[str, pd.DataFrame]]: _description_
        """
        return self._df_ref_data

    def find_order_from_id(self, order_id: str) -> Optional[Order]:
        for order in self._orders:
            if order.order_id == order_id:
                return order
        return None

    def _process_single_buy_order(
        self,
        dt_idx: pd.Timestamp,
        order: Order,
        price: float,
        abs_size: float,
    ) -> None:
        if self._portfolio.can_buy(price * abs_size):
            self._portfolio.bought_asset(
                asset_name=order.asset_name,
                size=abs_size,
                paid_cash=int(price * abs_size),
            )
            order.status = OrderStatus.PROCESSED
            order.executed_price = price
            order.executed_size = abs_size
        else:
            # dont have enough cash to buy asset_name
            if self._buysell_policy.buy_policy == BuyPolicy.CALCEL_IF_NOT_ENOUGH:
                # cancel buy
                order.status = OrderStatus.CANCELED
                order.canceled_reason = f"no enough cash to buy {order.asset_name}"
            elif self._buysell_policy.buy_policy == BuyPolicy.BUY_AS_MUCH_AS_POSSIBLE:
                # buy asset_name as much as possible with current cash
                abs_size = 0.99 * (self._portfolio.cash / price)
                self._portfolio.bought_asset(
                    asset_name=order.asset_name,
                    size=abs_size,
                    paid_cash=int(price * abs_size),
                )
                order.status = OrderStatus.PROCESSED
                order.executed_price = price
                order.executed_size = abs_size
        order.executed_timestamp = dt_idx

    def _process_single_sell_order(
        self,
        dt_idx: pd.Timestamp,
        order: Order,
        price: float,
        abs_size: float,
    ) -> None:
        holding_target_asset_amount = self._portfolio.other_asset(name=order.asset_name)
        if holding_target_asset_amount >= abs_size:
            # sell abs_size asset from holding_target_asset_amount asset
            self._portfolio.sold_asset(
                asset_name=order.asset_name,
                size=abs_size,
                gained_cash=int(price * abs_size),
            )
            order.executed_price = price
            order.executed_size = abs_size
            order.status = OrderStatus.PROCESSED
        else:
            if self._buysell_policy.sell_policy == SellPolicy.CALCEL_IF_NOT_ENOUGH:
                # cancel sell
                order.status = OrderStatus.CANCELED
                order.canceled_reason = (
                    f"no enough asset({order.asset_name}) to sell {order.asset_name}"
                )
            elif (
                self._buysell_policy.sell_policy == SellPolicy.SELL_AS_MUCH_AS_POSSIBLE
            ):
                # sell all holding asset
                self._portfolio.sold_asset(
                    asset_name=order.asset_name,
                    size=holding_target_asset_amount,
                    gained_cash=int(price * holding_target_asset_amount),
                )
                order.executed_price = price
                order.executed_size = holding_target_asset_amount
                order.status = OrderStatus.PROCESSED
            elif self._buysell_policy.sell_policy == SellPolicy.SELL_ALWAYS:
                # whole/partial short selling
                self._portfolio.sold_asset(
                    asset_name=order.asset_name,
                    size=abs_size,
                    gained_cash=int(price * abs_size),
                )
                order.executed_price = price
                order.executed_size = abs_size
                order.status = OrderStatus.PROCESSED
            else:
                raise RuntimeError("unexpected")
        order.executed_timestamp = dt_idx

    def _process_single_close_long_order(
        self,
        dt_idx: pd.Timestamp,
        order: Order,
        price: float,
        abs_size: Optional[float] = None,
    ) -> None:
        holding_target_asset_amount = self._portfolio.other_asset(name=order.asset_name)
        if order.close_target_order_id is not None:
            target_order = self.find_order_from_id(order_id=order.close_target_order_id)
            if target_order is not None:
                if target_order.order_type != OrderType.BUY:
                    raise Exception(
                        "target order type for OrderType.CLOSE_LONG must be OrderType.BUY"
                    )
                abs_size = target_order.executed_size
                if abs_size is None:
                    Logger.w(
                        "_process_single_order",
                        f"For close order ({order.order_id}), close target order ({order.close_target_order_id}) "
                        "is specified but executed_size for target order is None, "
                        "maybe smoething wrong.",
                    )
        if abs_size is None:
            raise ValueError("either close target order and abs_size must be specified")
            # abs_size = holding_target_asset_amount

        if holding_target_asset_amount >= abs_size:
            # sell abs_size asset from holding_target_asset_amount asset
            self._portfolio.sold_asset(
                asset_name=order.asset_name,
                size=abs_size,
                gained_cash=int(price * abs_size),
            )
            order.executed_price = price
            order.executed_size = abs_size
            order.status = OrderStatus.PROCESSED
        else:
            if (
                self._buysell_policy.close_long_policy
                == CloseLongPolicy.CALCEL_IF_NOT_ENOUGH
            ):
                # cancel sell
                order.status = OrderStatus.CANCELED
                order.canceled_reason = (
                    f"no enough asset({order.asset_name}) to sell {order.asset_name}"
                )
            elif (
                self._buysell_policy.close_long_policy
                == CloseLongPolicy.SELL_AS_MUCH_AS_POSSIBLE
            ):
                # sell all holding asset
                self._portfolio.sold_asset(
                    asset_name=order.asset_name,
                    size=holding_target_asset_amount,
                    gained_cash=int(price * holding_target_asset_amount),
                )
                order.executed_price = price
                order.executed_size = holding_target_asset_amount
                order.status = OrderStatus.PROCESSED
            elif self._buysell_policy.close_long_policy == CloseLongPolicy.SELL_ALWAYS:
                # whole/partial short selling
                self._portfolio.sold_asset(
                    asset_name=order.asset_name,
                    size=abs_size,
                    gained_cash=int(price * abs_size),
                )
                order.executed_price = price
                order.executed_size = abs_size
                order.status = OrderStatus.PROCESSED
            else:
                raise RuntimeError("unexpected")
        order.executed_timestamp = dt_idx

    def _process_single_close_short_order(
        self,
        dt_idx: pd.Timestamp,
        order: Order,
        price: float,
        abs_size: Optional[float] = None,
    ) -> None:
        # holding_target_asset_amount = self._portfolio.other_asset(name=order.asset_name)
        if order.close_target_order_id is not None:
            target_order = self.find_order_from_id(order_id=order.close_target_order_id)
            if target_order is not None:
                if target_order.order_type != OrderType.SELL:
                    raise Exception(
                        "target order type for OrderType.CLOSE_SHORT must be OrderType.SELL"
                    )
                abs_size = target_order.executed_size
                if abs_size is None:
                    Logger.w(
                        "_process_single_order",
                        f"For close order ({order.order_id}), close target order ({order.close_target_order_id}) "
                        "is specified but executed_size for target order is None, "
                        "maybe smoething wrong.",
                    )
            else:
                Logger.w(
                    "_process_single_order",
                    f"close target order ({order.close_target_order_id}) not found ",
                )
        if abs_size is None:
            raise ValueError("either close target order and abs_size must be specified")
            # abs_size = holding_target_asset_amount

        if self._portfolio.can_buy(price * abs_size):
            self._portfolio.bought_asset(
                asset_name=order.asset_name,
                size=abs_size,
                paid_cash=int(price * abs_size),
            )
            order.status = OrderStatus.PROCESSED
            order.executed_price = price
            order.executed_size = abs_size
        else:
            # dont have enough cash to buy asset_name
            if (
                self._buysell_policy.close_short_policy
                == CloseShortPolicy.CALCEL_IF_NOT_ENOUGH
            ):
                # cancel buy
                order.status = OrderStatus.CANCELED
                order.canceled_reason = f"no enough cash to buy {order.asset_name}"
            elif (
                self._buysell_policy.close_short_policy
                == CloseShortPolicy.BUY_AS_MUCH_AS_POSSIBLE
            ):
                # buy asset_name as much as possible with current cash
                abs_size = 0.99 * (self._portfolio.cash / price)
                if int(price * abs_size) > 0:
                    self._portfolio.bought_asset(
                        asset_name=order.asset_name,
                        size=abs_size,
                        paid_cash=int(price * abs_size),
                    )
                    order.status = OrderStatus.PROCESSED
                    order.executed_price = price
                    order.executed_size = abs_size
                else:
                    order.status = OrderStatus.CANCELED
                    order.canceled_reason = "int(price * abs_size) = 0"
        order.executed_timestamp = dt_idx
