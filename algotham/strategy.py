from abc import ABCMeta, abstractmethod
from typing import Optional

import pandas as pd

from .order import Order


class BaseStrategy(metaclass=ABCMeta):
    @abstractmethod
    def next(self, dt_idx: pd.Timestamp, idx: int, bt: "BackTest"):  # type: ignore
        raise NotImplementedError("Must implement")

    def on_order_accepted(
        self, order: Order, bt: "BackTest", dt_idx: Optional[pd.Timestamp] = None  # type: ignore
    ):
        pass

    def on_order_processed(
        self, order: Order, bt: "BackTest", dt_idx: Optional[pd.Timestamp] = None  # type: ignore
    ):
        pass
