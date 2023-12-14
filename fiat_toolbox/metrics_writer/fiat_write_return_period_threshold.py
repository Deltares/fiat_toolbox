
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
        return_periods = [float(col.split('(')[1][:-2]) for col in df.columns if col.startswith(self.column_prefix)]
        
        # Calculate exceedance probability
        return self._calculate(df, return_periods, threshold, T).to_frame()
        
    def append_to_file(self, input_file: str, output_file: str, threshold: float, T: float) -> None:
        """Append exceedance probability to file.

        Parameters
        ----------
        input_file : str
            Path to input file.
        output_file : str
            Path to output file.
        threshold : float
            Threshold value.
        T : float
            Time horizon.
        """

        # Read data from file
        df = pd.read_csv(input_file, index_col=0)

        # Append exceedance probability
        result = self.append_probability(df, threshold, T)

        # Write data to file
        result.to_csv(output_file)

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

        # Convert all non-numerical values to nan
        df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

        # Extract values for the selected columns
        values = df.filter(like=self.column_prefix).to_numpy()

        # Create a mask where True indicates a NaN value
        nan_mask = np.isnan(values)

        # Check if there are any NaN values after the first non-NaN value in each row
        invalid_rows = np.any(np.diff(nan_mask.astype(int), axis=1) == 1, axis=1)

        # Add the check if all elements in a row are NaN
        invalid_rows = invalid_rows | np.all(nan_mask, axis=1)

        # Custom interpolation function
        def custom_interp(x, xp, fp):
            if x > xp[-1]:
                return np.nan
            elif x < xp[0]:
                return fp[0]
            else: 
                return np.interp(x, xp, fp)

        # Interpolate to find the return period for which the threshold is first exceeded
        RP = np.array([custom_interp(threshold, row, return_periods) for row in values])

        # Calculate exceedance probability
        mask = ~invalid_rows
        result = np.full(len(df), np.nan)
        result[mask] = np.round((1 - np.exp(-T / RP[mask])) * 100, 1)

        return pd.Series(result, name = 'Exceedance Probability')