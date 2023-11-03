from typing import Optional

import numpy as np
import pandas as pd

from algotham.backtest import BackTest
from algotham.order import Order
from algotham.portfolio import Portfolio
from algotham.strategy import BaseStrategy


class SampleStrategy(BaseStrategy):
    def next(self, dt_idx: pd.Timestamp, idx: int, bt: "BackTest"):
        print(idx, dt_idx, bt.df_ref_data["btcjpy_buy"].loc[dt_idx]["price_mean"])  # type: ignore
        if idx < 20:
            return
        # using past 20 periods feature data for prediction
        s_idx = idx + 1 - 20
        e_idx = idx + 1
        df_feat = bt.df_ref_data["btcjpy_buy"].iloc[s_idx:e_idx]  # type: ignore
        assert df_feat.index.max() == dt_idx, (
            "index mismatch",
            df_feat.index.max(),
            dt_idx,
        )
        pred = self._mock_daily_prediction(df_feat)
        if pred == 1:
            order = bt.buy(
                asset_name="BTC_JPY",
                sr_ref_price=bt.df_ref_data["btcjpy_buy"]["price_ohlc_open"],  # type: ignore
                abs_size=0.01,
                dt_idx=dt_idx,
                idx=idx,
            )
            order
            # print(order)
        elif pred == -1:
            order = bt.sell(
                asset_name="BTC_JPY",
                sr_ref_price=bt.df_ref_data["btcjpy_sell"]["price_ohlc_open"],  # type: ignore
                abs_size=0.01,
                dt_idx=dt_idx,
                idx=idx,
            )
            order
            # print(order)

    def _mock_daily_prediction(self, df_feat: pd.DataFrame) -> int:
        rnd = np.random.random()
        if rnd <= 0.4:
            return 1  # buy
        elif rnd < 0.6:
            return 0  # do nothing
        else:
            return -1  # sell

    def on_order_accepted(
        self, order: Order, bt: "BackTest", dt_idx: Optional[pd.Timestamp] = None
    ):
        print(f"[{dt_idx}] on_order_accepted {order.order_type}, {order.order_id}")

    def on_order_processed(
        self, order: Order, bt: "BackTest", dt_idx: Optional[pd.Timestamp] = None
    ):
        print(f"[{dt_idx}] on_order_processed {order.order_type}, {order.order_id}")


class TestBacktest:
    def test_backtest_with_sample_strategy(self):
        BTCJPY_BUY_SAMPLE_DATA_URL = "https://github.com/pondelion/algotham_v2/raw/main/data/sample/sample_btcjpy_buy_5min_230601.csv.gzip"
        BTCJPY_SELL_SAMPLE_DATA_URL = "https://github.com/pondelion/algotham_v2/raw/main/data/sample/sample_btcjpy_sell_5min_230601.csv.gzip"
        df_btcjpy_buy = pd.read_csv(BTCJPY_BUY_SAMPLE_DATA_URL, compression="gzip")
        df_btcjpy_sell = pd.read_csv(BTCJPY_SELL_SAMPLE_DATA_URL, compression="gzip")
        df_btcjpy_buy["timestamp"] = pd.to_datetime(
            df_btcjpy_buy["timestamp"], utc=True
        )
        df_btcjpy_sell["timestamp"] = pd.to_datetime(
            df_btcjpy_sell["timestamp"], utc=True
        )
        df_btcjpy_buy = df_btcjpy_buy.set_index("timestamp")
        df_btcjpy_sell = df_btcjpy_sell.set_index("timestamp")
        assert (df_btcjpy_buy.index == df_btcjpy_sell.index).all()

        pf = Portfolio(init_cash=1000 * 10000)
        N_PERIODS = 500
        bt = BackTest(
            dt_index=df_btcjpy_buy[-N_PERIODS:].index,  # type: ignore
            strategy=SampleStrategy(),
            init_portfolio=pf,
            df_ref_data={
                "btcjpy_buy": df_btcjpy_buy[-N_PERIODS:],
                "btcjpy_sell": df_btcjpy_sell[-N_PERIODS:],
            },
        )
        bt.run()
        df_evaluated_assets_hitory = pf.evaluated_assets_hitory(
            ref_prices={"BTC_JPY": df_btcjpy_sell["price_mean"]}
        )
        assert set(
            ["cash", "BTC_JPY", "evaluated_cash_BTC_JPY", "total_evaluated_cash", "pnl"]
        ) == set(df_evaluated_assets_hitory.columns)
        assert len(df_evaluated_assets_hitory) == N_PERIODS
        print(df_evaluated_assets_hitory)
        df_order_history = bt.order_history
        assert set(
            [
                "order_type",
                "asset_name",
                "abs_size",
                "ordered_timestamp",
                "price_at_ordertime",
                "execution_timestamp",
                "executed_timestamp",
                "executed_price",
                "executed_size",
                "status",
                "order_id",
            ]
        ) == set(df_order_history.columns)
        assert len(df_order_history) > 0
        # df_evaluated_assets_hitory['pnl'].plot()
