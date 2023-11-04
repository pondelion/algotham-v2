import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class OrderStatus(Enum):
    UNPROCESSED = "UNPROCESSED"
    PROCESSED = "PROCESSED"
    CANCELED = "CANCELED"


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


@dataclass
class Order:
    order_type: OrderType
    asset_name: str
    ordered_timestamp: pd.Timestamp
    price_at_ordertime: float
    execution_timestamp: Optional[pd.Timestamp]
    sr_ref_price: pd.Series
    abs_size: Optional[float] = None
    executed_timestamp: Optional[pd.Timestamp] = None
    executed_price: Optional[float] = None
    executed_size: Optional[float] = None
    status: OrderStatus = OrderStatus.UNPROCESSED
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    close_target_order_id: Optional[str] = None
