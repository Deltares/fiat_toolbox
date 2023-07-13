import pandas as pd
import numpy as np
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
import os
from typing import Union

class Equity:
    def __init__(self, method):
        self.method = method

    def check_datatype(
            self,
            variable: Union[str, pd.DataFrame],
    )->pd.DataFrame:
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
        elif isinstance(variable, str) and variable.endswith('.csv'):
            variable = pd.read_csv(variable)
        else:
            raise ValueError("Input variable is neither a pandas DataFrame nor a path to a CSV file.")
        return variable 

    def get_equity_input(
            self,
            census_table: Union[str, pd.DataFrame, Path], 
            damages_table: Union[str, pd.DataFrame, Path],
            aggregation_label: str,
    )->pd.DataFrame:
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
        damages_table  = self.check_datatype(damages_table)

        # Merge census block groups with fiat output (damages estimations per return period)
        df = pd.merge(census_table, damages_table, on=aggregation_label, how="left")
        df = df.dropna().reset_index(drop=True)
        return df

    def calculate_equity_weights(
            self,
            df: pd.DataFrame, 
            percapitalincome_label: str,
            totalpopulation_label: str,
            gamma: float,
    )->pd.DataFrame:
        # Get elasticity parameter
        # gamma = 1.2 

        # Get population and income per capital data
        I_PC = df[percapitalincome_label] # mean per capita income
        Pop  = df[totalpopulation_label] # population

        # Calculate aggregated annual income
        I_AA = I_PC * Pop

        # Calculate weighted average income per capita
        I_WA = np.average(I_PC, weights=Pop)

        # Calculate equity weights
        EW = (I_PC / I_WA) ** -gamma # Equity Weight
        
        # Add annual income to the dataframe
        df["I_AA"] = I_AA
        # Add equity weight calculations into the dataframe
        df["EW"] = EW 
        return df

    def calculate_ewced_per_rp(
            self,
            df_ew: pd.DataFrame, 
            gamma: float,
    )->pd.DataFrame:
        
        # Get equity weight data
        I_AA = df_ew["I_AA"] 
        EW   = df_ew["EW"]

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
            P = 1 - np.exp(-t/RP) 
            # Social vulnerability
            z = D / I_AA 
            # Risk premium
            R = (1 - (1 + P*((1-z)**(1-gamma)-1))**(1/(1-gamma))) / (P*z) 
            # This step is needed to avoid nan value when z is zero
            R[R.isnull()] = 0 
            # Certainty equivalent damage
            CED = R * D  
            # Equity weighted certainty equivalent damage
            EWCED = EW * CED  

            # Add risk premium data to dataframes
            df_ew[f"R_RP{RP}"] = R
            # Add ewced to dataframes 
            df_ew[f"EWCED_RP{RP}"] = EWCED
        return df_ew, RP_cols 

    # Taken from FIAT for now
    def calculate_coefficients(
            self,
            T
    ):
        """Calculates coefficients used to compute the EAD as a linear function of the known damages
        Args:
            T (list of ints): return periods T1 … Tn for which damages are known
        Returns:
            alpha [list of floats]: coefficients a1, …, an (used to compute the AED as a linear function of the known damages)

        In which D(f) is the damage, D, as a function of the frequency of exceedance, f. In order to compute this EAD,
        function D(f) needs to be known for the entire range of frequencies. Instead, D(f) is only given for the n
        frequencies as mentioned in the table above. So, in order to compute the integral above, some assumptions need
        to be made for function D(h):

        (i)	   For f > f1 the damage is assumed to be equal to 0
        (ii)   For f<fn, the damage is assumed to be equal to Dn
        (iii)  For all other frequencies, the damage is estimated from log-linear interpolation between the known damages and frequencies

        """
        # Step 1: Compute frequencies associated with T-values.
        f = [1 / i for i in T]
        lf = [np.log(1 / i) for i in T]
        # Step 2:
        c = [(1 / (lf[i] - lf[i + 1])) for i in range(len(T[:-1]))]
        # Step 3:
        G = [(f[i] * lf[i] - f[i]) for i in range(len(T))]
        # Step 4:
        a = [
            ((1 + c[i] * lf[i + 1]) * (f[i] - f[i + 1]) + c[i] * (G[i + 1] - G[i]))
            for i in range(len(T[:-1]))
        ]
        b = [
            (c[i] * (G[i] - G[i + 1] + lf[i + 1] * (f[i + 1] - f[i])))
            for i in range(len(T[:-1]))
        ]
        # Step 5:
        if len(T) == 1:
            alpha = f
        else:
            alpha = [
                b[0] if i == 0 else f[i] + a[i - 1] if i == len(T) - 1 else a[i - 1] + b[i]
                for i in range(len(T))
            ]
        return alpha

    def calculate_ewced(
            self,
            df_ew_rp: pd.DataFrame,  
            RP_cols,
    )->pd.DataFrame:
        layers = []
        return_periods = []
        for i in RP_cols:
            RP = int(i.split(" ")[-1][2:])
            return_periods.append(RP)
            layers.append(df_ew_rp.loc[:, f"EWCED_RP{RP}"].values)

        stacked_layers = np.dstack(tuple(layers)).squeeze()
        df_ew_rp[f"EWCEAD"] = stacked_layers @ np.array(self.calculate_coefficients(return_periods))[:, None]
        return df_ew_rp

    def rank_ewced(
            self,
            df_ewced: pd.DataFrame, 
    )->pd.DataFrame:
        df_ewced["rank_EAD"] = df_ewced["EAD"].rank(ascending=False)
        df_ewced["rank_EWCEAD"] = df_ewced["EWCEAD"].rank(ascending=False)
        df_ewced["rank_diff"] = df_ewced["rank_EWCEAD"] - df_ewced["rank_EAD"]
        return df_ewced

    def calculate_resilience_index(
            self,
            df_ewced: pd.DataFrame, 
    )->pd.DataFrame:
        df_ewced["soc_res"] =  df_ewced["EAD"]/df_ewced["EWCEAD"]
        df_ewced["soc_res"][df_ewced["soc_res"] == np.inf] = np.nan
        return df_ewced

    def setup_equity_method(
            self,
            census_table: Union[str, pd.DataFrame, Path], 
            damages_table: Union[str, pd.DataFrame, Path],
            aggregation_label: str,
            percapitalincome_label: str,
            totalpopulation_label:str,
            gamma: float,
            output_file:  Union[str, Path, None] = None,
    )->pd.DataFrame:
        df = self.get_equity_input(census_table, damages_table, aggregation_label)
        df_ew = self.calculate_equity_weights(df, percapitalincome_label, totalpopulation_label, gamma)
        df_ew_rp, RP_cols = self.calculate_ewced_per_rp(df_ew, gamma)
        df_ewced = self.calculate_ewced(df_ew_rp, RP_cols)
        df_ewced = self.rank_ewced(df_ewced)
        df_ewced = self.calculate_resilience_index(df_ewced) 
        if output_file is not None:
            df_ewced_filtered = df_ewced[['Census_Bg', 'EW', 'EWCEAD']]
            df_ewced_filtered.to_csv(output_file, index=False)
        return df_ewced
