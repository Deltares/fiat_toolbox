import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import parse

from fiat_toolbox.equity.fiat_functions import calc_rp_coef


class Equity:
    def __init__(
        self,
        census_table: Union[str, pd.DataFrame, Path],
        damages_table: Union[str, pd.DataFrame, Path],
        aggregation_label: str,
        percapitaincome_label: str,
        totalpopulation_label: str,
        damage_column_pattern: str = "Total Damage ({rp}Y)",
    ):
        """_summary_

        Parameters
        ----------
        census_table : Union[str, pd.DataFrame, Path]
            Census data
        damages_table : Union[str, pd.DataFrame, Path]
            Damage results
        aggregation_label : str
            column name of aggregation areas
        percapitaincome_label : str
            column name of per capita income
        totalpopulation_label : str
            column name of total population
        """
        # Merge tables
        self.df = self._merge_tables(census_table, damages_table, aggregation_label)
        self.df0 = self.df.copy()  # Keep copy of original
        self.aggregation_label = aggregation_label
        self.percapitaincome_label = percapitaincome_label
        self.totalpopulation_label = totalpopulation_label
        self.damage_column_pattern = damage_column_pattern

    @staticmethod
    def _check_datatype(
        variable: Union[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Check that inputs for equity are rather .csv files or pd.Dataframes

        Parameters
        ----------
        variable : Union[str, pd.DataFrame]
            input

        Returns
        -------
        pd.DataFrame
            input in dataframe format

        Raises
        ------
        ValueError
            Error if input is not in correct format
        """

        if isinstance(variable, pd.DataFrame):
            pass
        elif os.path.exists(variable):
            variable = pd.read_csv(variable)
        elif isinstance(variable, str) and variable.endswith(".csv"):
            variable = pd.read_csv(variable)
        else:
            raise ValueError(
                "Input variable is neither a pandas DataFrame nor a path to a CSV file."
            )
        return variable

    @staticmethod
    def _merge_tables(
        census_table: Union[str, pd.DataFrame, Path],
        damages_table: Union[str, pd.DataFrame, Path],
        aggregation_label: str,
    ) -> pd.DataFrame:
        """Create dataframe with damage and social data used to calculate the equity weights

        Parameters
        ----------
        census_table : Union[str, pd.DataFrame, Path]
            Census data
        damages_table : Union[str, pd.DataFrame, Path]
            Damage results
        aggregation_label : str
            column name used to merge on

        Returns
        -------
        pd.DataFrame
            merged dataframe
        """
        # Check if data inputs are whether .csv files or pd.DataFrame
        census_table = Equity._check_datatype(census_table)
        damages_table = Equity._check_datatype(damages_table)
        # If the aggregated damages format is the fiat_toolbox one make sure columns are interpreted correctly
        if "Unnamed:" in damages_table.columns[0]:
            # Use name from input label
            damages_table = damages_table.rename(
                columns={damages_table.columns[0]: aggregation_label}
            )
            index_name = damages_table.columns[0]
            damages_table = damages_table.set_index(index_name)

            # Drop rows containing other variables
            damages_table = damages_table.drop(
                [
                    "Description",
                    "Show In Metrics Table",
                    "Show In Metrics Map",
                    "Long Name",
                ]
            )
            damages_table = damages_table.apply(pd.to_numeric)
        # Merge census block groups with fiat output (damages estimations per return period)
        df = damages_table.merge(census_table, on=aggregation_label, how="left")
        df = df[~df[aggregation_label].isna()]
        df = df.reset_index(drop=True)
        return df

    def _calculate_equity_weights(self):
        """Calculates equity weights per aggregation area"""
        # Get population and income per capital data
        I_PC = self.df[self.percapitaincome_label]  # mean per capita income
        Pop = self.df[self.totalpopulation_label]  # population

        # Calculate aggregated annual income
        I_AA = I_PC * Pop

        # Calculate weighted average income per capita
        I_PC = np.ma.MaskedArray(I_PC, mask=np.isnan(I_PC))
        I_WA = np.ma.average(I_PC, weights=Pop)

        # Calculate equity weights
        EW = (I_PC / I_WA) ** -self.gamma  # Equity Weight

        # Add annual income to the dataframe
        self.df["I_AA"] = I_AA
        # Add equity weight calculations into the dataframe
        self.df["EW"] = EW

    def _get_rp_from_name(self, name):
        parser = parse.parse(self.damage_column_pattern, name, extra_types={"s": str})
        if parser:
            rp = int(parser.named["rp"])
        else:
            rp = None
        return rp

    def calculate_ewced_per_rp(self):
        """Get equity weighted certainty equivalent damages per return period using a risk prenium"""

        # Get equity weight data
        I_AA = self.df["I_AA"]
        EW = self.df["EW"]

        # Retrieve columns with damage per return period data of fiat output
        RPs = {}
        for name in self.df.columns:
            if self._get_rp_from_name(name):
                rp = self._get_rp_from_name(name)
                RPs[rp] = name
        # Make sure data is sorted
        self.RPs = {}
        for key in sorted(RPs.keys()):
            self.RPs[key] = RPs[key]

        if len(self.RPs) == 0:
            raise ValueError(
                "Columns with damages per return period could not be found."
            )

        # Get Equity weighted certainty equivalent damage per return period
        for rp in self.RPs:
            col = self.RPs[rp]
            # Damage for return period
            D = self.df[col]
            # Period of interest in years
            t = 1
            # Probability of exceedance
            P = 1 - np.exp(-t / rp)
            # Social vulnerability
            z = D / I_AA
            # Risk premium
            R = (
                1
                - (1 + P * ((1 - z) ** (1 - self.gamma) - 1)) ** (1 / (1 - self.gamma))
            ) / (P * z)
            # This step is needed to avoid nan value when z is zero
            R[R.isna()] = 0
            # Equity weighted damage
            EWD = EW * D
            # Equity weighted certainty equivalent damage
            EWCED = R * EWD
            # Add risk premium data to dataframes
            self.df[f"R_RP_{rp}"] = R
            # Add ewd and ewced to dataframes
            self.df[f"EWD_RP_{rp}"] = EWD
            self.df[f"EWCED_RP_{rp}"] = EWCED

    def calculate_ewcead(self):
        """Calculates equity weighted certainty expected annual damages using log linear approach"""
        layers = []
        return_periods = []
        for rp in self.RPs:
            return_periods.append(rp)
            layers.append(self.df.loc[:, f"EWCED_RP_{rp}"].values)

        stacked_layers = np.dstack(tuple(layers)).squeeze()
        self.df["EWCEAD"] = stacked_layers.dot(
            np.array(calc_rp_coef(return_periods))[:, None]
        )

    def calculate_ewead(self):
        """Calculates equity weighted certainty expected annual damages using log linear approach"""
        layers = []
        return_periods = []
        for rp in self.RPs:
            return_periods.append(rp)
            layers.append(self.df.loc[:, f"EWD_RP_{rp}"].values)

        stacked_layers = np.dstack(tuple(layers)).squeeze()
        self.df["EWEAD"] = stacked_layers.dot(
            np.array(calc_rp_coef(return_periods))[:, None]
        )

    def equity_calculation(
        self,
        gamma: float = 1.2,
        output_file: Union[str, Path, None] = None,
    ) -> pd.DataFrame:
        """Calculates equity weighted risk

        Parameters
        ----------
        gamma : float, optional
            elasticity by default 1.2
        output_file : Union[str, Path, None], optional
            output file path, by default None

        Returns
        -------
        pd.DataFrame
            dataframe with the results
        """
        self.gamma = gamma
        # Get equity weights
        self._calculate_equity_weights()
        # Calculate equity weighted damage per return period
        self.calculate_ewced_per_rp()
        # Calculate equity weighted risk
        self.calculate_ewead()
        self.calculate_ewcead()
        # Keep only results
        df_ewced_filtered = self.df[[self.aggregation_label, "EW", "EWEAD", "EWCEAD"]]
        # Save file if requested
        if output_file is not None:
            df_ewced_filtered.to_csv(output_file, index=False)

        return df_ewced_filtered

    def rank_ewced(self, ead_column: str = "Risk (EAD)") -> pd.DataFrame:
        """Ranks areas per EAD EWCEAD and the calculates difference in ranking between 2nd and 1st

        Parameters
        ----------
        ead_column : str, optional
            name of column where the standard EAD calculation exists, by default "Risk (EAD)"

        Returns
        -------
        pd.DataFrame
            ranking results
        """
        if ead_column not in self.df.columns:
            raise ValueError(
                f"EAD column '{ead_column}' not present in provided aggregated file. A different column name can be specified using the 'ead_column' argument."
            )
        self.df["rank_EAD"] = self.df[ead_column].rank(ascending=False).astype(int)
        self.df["rank_EWEAD"] = self.df["EWEAD"].rank(ascending=False).astype(int)
        self.df["rank_EWCEAD"] = self.df["EWCEAD"].rank(ascending=False).astype(int)
        self.df["rank_diff_EWEAD"] = self.df["rank_EWEAD"] - self.df["rank_EAD"]
        self.df["rank_diff_EWCEAD"] = self.df["rank_EWCEAD"] - self.df["rank_EAD"]
        return self.df[
            [
                self.aggregation_label,
                "rank_EAD",
                "rank_EWCEAD",
                "rank_diff_EWCEAD",
                "rank_EWEAD",
                "rank_diff_EWEAD",
            ]
        ]

    def calculate_resilience_index(
        self, ead_column: str = "Risk (EAD)"
    ) -> pd.DataFrame:
        """Calculates a simple socioeconomic resilience indicators by the ratio of the standard EAD to the EWCEAD

        Parameters
        ----------
        ead_column : str, optional
            name of column where the standard EAD calculation exists, by default "Risk (EAD)"

        Returns
        -------
        pd.DataFrame
            index results
        """
        self.df["SRI"] = self.df[ead_column] / self.df["EWCEAD"]
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        return self.df[[self.aggregation_label, "SRI"]]
