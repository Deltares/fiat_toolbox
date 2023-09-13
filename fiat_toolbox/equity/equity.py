import os
from pathlib import Path
from typing import Union
from delft_fiat.models.calc import calc_rp_coef
import numpy as np
import pandas as pd


class Equity:
    def __init__(self, method):
        self.method = method

    def check_datatype(
        self,
        variable: Union[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Check that inputs for equity are rather .csv files or pd.Dataframes

        Parameters
        ----------
        variable : Union[str, pd.DataFrame]
            _description_

        Returns
        -------
        pd.DataFrame
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        if isinstance(variable, pd.DataFrame):
            variable = variable
        elif os.path.exists(variable):
            variable = pd.read_csv(variable)
        elif isinstance(variable, str) and variable.endswith(".csv"):
            variable = pd.read_csv(variable)
        else:
            raise ValueError(
                "Input variable is neither a pandas DataFrame nor a path to a CSV file."
            )
        return variable

    def get_equity_input(
        self,
        census_table: Union[str, pd.DataFrame, Path],
        damages_table: Union[str, pd.DataFrame, Path],
        aggregation_label: str,
    ) -> pd.DataFrame:
        """Create dataframe with damage and social data used to calculate the equity weights

        Parameters
        ----------
        census_table : Union[str, pd.DataFrame, Path]
            _description_
        damages_table : Union[str, pd.DataFrame, Path]
            _description_
        aggregation_label : str
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        # Check if data inputs are wether .csv files or pd.DataFrame
        census_table = self.check_datatype(census_table)
        damages_table = self.check_datatype(damages_table)

        # Merge census block groups with fiat output (damages estimations per return period)
        df = census_table.merge(damages_table, on=aggregation_label, how="left")
        df = df.dropna().reset_index(drop=True)
        return df

    def calculate_equity_weights(
        self,
        df: pd.DataFrame,
        percapitalincome_label: str,
        totalpopulation_label: str,
        gamma: float,
    ) -> pd.DataFrame:
        # Get elasticity parameter
        # gamma = 1.2

        # Get population and income per capital data
        I_PC = df[percapitalincome_label]  # mean per capita income
        Pop = df[totalpopulation_label]  # population

        # Calculate aggregated annual income
        I_AA = I_PC * Pop

        # Calculate weighted average income per capita
        I_WA = np.average(I_PC, weights=Pop)

        # Calculate equity weights
        EW = (I_PC / I_WA) ** -gamma  # Equity Weight

        # Add annual income to the dataframe
        df["I_AA"] = I_AA
        # Add equity weight calculations into the dataframe
        df["EW"] = EW
        return df

    def calculate_ewced_per_rp(
        self,
        df_ew: pd.DataFrame,
        gamma: float,
    ) -> pd.DataFrame:
        # Get equity weight data
        I_AA = df_ew["I_AA"]
        EW = df_ew["EW"]

        # Retrieve columns with damage per return period data of fiat output
        RP_cols = [name for name in df_ew.columns if "Total Damage" in name]

        # Get weighted expected annual damage per return period
        for col in RP_cols:
            # Damage for return period
            D = df_ew[col]
            # Return period
            RP = int(col.split(" ")[-1][2:])
            # Period of interest in years
            t = 1
            # Probability of exceedance
            P = 1 - np.exp(-t / RP)
            # Social vulnerability
            z = D / I_AA
            # Risk premium
            R = (1 - (1 + P * ((1 - z) ** (1 - gamma) - 1)) ** (1 / (1 - gamma))) / (
                P * z
            )
            # This step is needed to avoid nan value when z is zero
            R[R.isna()] = 0
            # Certainty equivalent damage
            CED = R * D
            # Equity weighted certainty equivalent damage
            EWCED = EW * CED

            # Add risk premium data to dataframes
            df_ew[f"R_RP{RP}"] = R
            # Add ewced to dataframes
            df_ew[f"EWCED_RP{RP}"] = EWCED
        return df_ew, RP_cols

    def calculate_ewced(
        self,
        df_ew_rp: pd.DataFrame,
        RP_cols,
    ) -> pd.DataFrame:
        layers = []
        return_periods = []
        for i in RP_cols:
            RP = int(i.split(" ")[-1][2:])
            return_periods.append(RP)
            layers.append(df_ew_rp.loc[:, f"EWCED_RP{RP}"].values)

        stacked_layers = np.dstack(tuple(layers)).squeeze()
        df_ew_rp["EWCEAD"] = (
            stacked_layers
            @ np.array(calc_rp_coef(return_periods))[:, None]
        )
        return df_ew_rp

    def rank_ewced(
        self,
        df_ewced: pd.DataFrame,
    ) -> pd.DataFrame:
        df_ewced["rank_EAD"] = df_ewced["EAD"].rank(ascending=False)
        df_ewced["rank_EWCEAD"] = df_ewced["EWCEAD"].rank(ascending=False)
        df_ewced["rank_diff"] = df_ewced["rank_EWCEAD"] - df_ewced["rank_EAD"]
        return df_ewced

    def calculate_resilience_index(
        self,
        df_ewced: pd.DataFrame,
    ) -> pd.DataFrame:
        df_ewced["soc_res"] = df_ewced["EAD"] / df_ewced["EWCEAD"]
        df_ewced["soc_res"][df_ewced["soc_res"] == np.inf] = np.nan
        return df_ewced

    def setup_equity_method(
        self,
        census_table: Union[str, pd.DataFrame, Path],
        damages_table: Union[str, pd.DataFrame, Path],
        aggregation_label: str,
        percapitalincome_label: str,
        totalpopulation_label: str,
        gamma: float,
        output_file: Union[str, Path, None] = None,
    ) -> pd.DataFrame:
        df = self.get_equity_input(census_table, damages_table, aggregation_label)
        df_ew = self.calculate_equity_weights(
            df, percapitalincome_label, totalpopulation_label, gamma
        )
        df_ew_rp, RP_cols = self.calculate_ewced_per_rp(df_ew, gamma)
        df_ewced = self.calculate_ewced(df_ew_rp, RP_cols)
        df_ewced = self.rank_ewced(df_ewced)
        df_ewced = self.calculate_resilience_index(df_ewced)
        if output_file is not None:
            df_ewced_filtered = df_ewced[["Census_Bg", "EW", "EWCEAD"]]
            df_ewced_filtered.to_csv(output_file, index=False)
        return df_ewced
