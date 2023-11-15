
import numpy as np
import pandas as pd


class ExceedanceProbabilityCalculator:
    def __init__(self, column_prefix):
        self.column_prefix = column_prefix

    def append_probability(self, df: pd.DataFrame, threshold: float, T: float) -> pd.DataFrame:
        """Append exceedance probability to dataframe.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the data.
        threshold : float
            Threshold value.
        T : float
            Time horizon.
            
        Returns
        -------
        pandas.DataFrame
            Dataframe containing the data and the exceedance probability.
        """
        
        # Initialize result dataframe
        result = df.copy()

        # Calculate exceedance probability
        result['Exceedance Probability'] = self.calculate(df, threshold, T)

        return result

    def calculate(self, df: pd.DataFrame, threshold: float, T: float) -> pd.DataFrame:
        """Calculate exceedance probability.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the data.
        threshold : float
            Threshold value.
        T : float
            Time horizon.

        Returns
        -------
        pandas.DataFrame    
            Dataframe containing the exceedance probability.
        """

        # Extract return periods from column names
        return_periods = [int(col.split('(')[1][:-2]) for col in df.columns if col.startswith(self.column_prefix)]
        
        # Calculate exceedance probability
        return self._calculate(df, return_periods, threshold, T).to_frame()
        
    def _calculate(self, df: pd.DataFrame, return_periods: list, threshold: float, T: float) -> pd.Series:
        """Calculate exceedance probability.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the data.
        return_periods : list
            List of return periods.
        threshold : float
            Threshold value.
        T : float
            Time horizon.

        Returns
        -------
        pandas.Series
            Series containing the exceedance probability.
        """

        # Extract values for the selected columns
        values = df.filter(like=self.column_prefix).to_numpy()

        # If all values are nan, return nan
        result = np.where(np.isnan(values).all(axis=1), np.nan, 0)

        # Custom interpolation function
        def custom_interp(x, xp, fp):
            res = np.interp(x, xp, fp)
            if res == fp[-1]:
                return np.nan
            elif res <= fp[0]:
                return fp[0]
            else: 
                return res

        # Interpolate to find the return period for which the threshold is first exceeded
        RP = np.array([custom_interp(threshold, row, return_periods) for row in values])

        # Calculate exceedance probability
        mask = ~np.isnan(RP)
        result[mask] = np.round((1 - np.exp(-T / RP[mask])) * 100, 1)

        return pd.Series(result, name = 'Exceedance Probability')