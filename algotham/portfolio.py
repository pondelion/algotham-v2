from typing import Dict, Union

import pandas as pd


class PortfolioRecorder:
    """_summary_"""

    def __init__(self):
        """_summary_"""
        self._portfolios = []

    def record(self, portfolio: "Portfolio", dt: pd.Timestamp):
        """_summary_

        Args:
            portfolio (Portfolio): _description_
            dt (pd.Timestamp): _description_
        """
        portfolio_dict = portfolio.portfolio
        portfolio_dict["datetime"] = dt
        self._portfolios.append(portfolio_dict)

    @property
    def history(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        if len(self._portfolios) > 0:
            return pd.DataFrame(self._portfolios).set_index("datetime").fillna(0)
        else:
            df_history = pd.DataFrame([])
            df_history.index.name = "datetime"
            return df_history


class Portfolio:
    """_summary_"""

    def __init__(
        self,
        init_cash: int = 0,
        init_other_assets: Dict = {},
    ):
        """_summary_

        Args:
            init_cash (int, optional): _description_. Defaults to 0.
            init_other_assets (Dict, optional): _description_. Defaults to {}.
        """
        self._cash = init_cash
        self._other_assets = init_other_assets.copy()
        self._pf_recorder = PortfolioRecorder()

    @property
    def cash(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return self._cash

    def other_asset(self, name: str) -> Union[int, float]:
        """_summary_

        Args:
            name (str): _description_

        Returns:
            Union[int, float]: _description_
        """
        return self._other_assets.get(name, 0.0)

    @property
    def portfolio(self) -> Dict:
        """_summary_

        Returns:
            Dict: _description_
        """
        portfolio = self._other_assets.copy()
        portfolio["cash"] = self._cash
        return portfolio

    def can_buy(self, cash_to_pay) -> bool:
        """_summary_

        Args:
            cash_to_pay (_type_): _description_

        Returns:
            bool: _description_
        """
        return cash_to_pay < self._cash

    def bought_asset(
        self,
        asset_name: str,
        size: Union[int, float],
        paid_cash: int,
    ) -> None:
        """_summary_

        Args:
            asset_name (str): _description_
            size (Union[int, float]): _description_
            paid_cash (int): _description_

        Raises:
            Exception: _description_
        """
        assert asset_name != "cash", "asset_name must not be cash"
        if paid_cash > self._cash:
            raise Exception("trying to buy asset with over cash capacity")
        self._cash -= paid_cash
        if asset_name in self._other_assets:
            self._other_assets[asset_name] += size
        else:
            self._other_assets[asset_name] = size

    def sold_asset(
        self,
        asset_name: str,
        size: Union[int, float],
        gained_cash: int,
    ) -> None:
        """_summary_

        Args:
            asset_name (str): _description_
            size (Union[int, float]): _description_
            gained_cash (int): _description_
        """
        assert asset_name != "cash", "asset_name must not be cash"
        assert gained_cash > 0, f"gained_cash must be positive => {gained_cash}"
        if asset_name in self._other_assets:
            self._other_assets[asset_name] -= size
        else:
            self._other_assets[asset_name] = -size
        self._cash += gained_cash

    def record(self, dt: pd.Timestamp):
        self._pf_recorder.record(self, dt)

    @property
    def pf_history(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        return self._pf_recorder.history

    def evaluated_assets_hitory(self, ref_prices: Dict[str, pd.Series]):
        """_summary_

        Args:
            ref_prices (Dict[str, pd.Series]): _description_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        df_pf_history = self.pf_history
        for asset_name in df_pf_history.columns:
            if asset_name != "cash" and asset_name not in ref_prices:
                raise Exception(
                    f"price data with asset name {asset_name} not found in ref_prices"
                )
        df_evaluated_cash_history = df_pf_history.copy()
        evaluated_cash_cols = []
        for asset_name in df_pf_history.columns:
            if asset_name == "cash":
                continue
            evaluated_cash_col = f"evaluated_cash_{asset_name}"
            evaluated_cash_cols.append(evaluated_cash_col)
            evaluated_cash_history = []
            for idx, r in df_pf_history.iterrows():
                size = r.loc[asset_name]
                price = ref_prices[asset_name].loc[idx]  # type: ignore
                evaluated_cash = size * price
                evaluated_cash_history.append(evaluated_cash)
            df_evaluated_cash_history[evaluated_cash_col] = evaluated_cash_history
        df_evaluated_cash_history["total_evaluated_cash"] = df_evaluated_cash_history[
            ["cash"] + evaluated_cash_cols
        ].sum(axis=1)
        df_evaluated_cash_history["pnl"] = (
            df_evaluated_cash_history["total_evaluated_cash"]
            - df_evaluated_cash_history["total_evaluated_cash"].iloc[0]
        )
        return df_evaluated_cash_history
