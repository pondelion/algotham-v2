from abc import ABCMeta, abstractmethod
from typing import Optional

import pandas as pd

from .order import Order


class BaseStrategy(metaclass=ABCMeta):
    @abstractmethod
    def next(self, dt_idx: pd.Timestamp, idx: int, bt: "BackTest") -> None:  # type: ignore
        """_summary_

        Args:
            dt_idx (pd.Timestamp): _description_
            idx (int): _description_
            bt (BackTest): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("Must implement")

    def on_order_accepted(
        self, order: Order, bt: "BackTest", dt_idx: Optional[pd.Timestamp] = None  # type: ignore
    ) -> None:
        """Called from backtest when order is accepted. Can be overwriten.

        Args:
            order (Order): _description_
            bt (BackTest): _description_
            dt_idx (_type_, optional): _description_. Defaults to None#type:ignore.
        """
        pass

    def on_order_processed(
        self, order: Order, bt: "BackTest", dt_idx: Optional[pd.Timestamp] = None  # type: ignore
    ) -> None:
        """Called from backtest when order process is completed. Can be overwriten.

        Args:
            order (Order): _description_
            bt (BackTest): _description_
            dt_idx (_type_, optional): _description_. Defaults to None#type:ignore.
        """
        pass
