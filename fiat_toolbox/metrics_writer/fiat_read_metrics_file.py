import os
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class IMetricsFileReader(ABC):
    """Interface for reading metrics from a file."""

    @abstractmethod
    def read_metrics_from_file(self, metric: str) -> pd.Series:
        """
        Reads metrics from a file.

        Parameters
        ----------
        metric : str
            The metric to read from the file.

        Returns
        -------
        pd.DataFrame
            The metrics read from the file.

        Raises
        ------
        KeyError
            If the metric is not found in the file.
        """

        pass


class MetricsFileReader(IMetricsFileReader):
    """Reads metrics from a file."""

    def __init__(self, metrics_file_path: str):
        """
        Initializes a new instance of the MetricsFileReader class.

        Parameters
        ----------
        metrics_file_path : str
            The path to the file containing the metrics.

        Raises
        ------
        FileNotFoundError
            If the file cannot be found.
        ValueError
            If the file is not a valid metrics file.
        """

        # Check if the file is a csv file
        if not metrics_file_path.endswith(".csv"):
            raise ValueError("The file must be a csv file.")

        # Check if the file exists
        if not os.path.exists(metrics_file_path):
            raise FileNotFoundError("The file does not exist.")

        # Set the metrics file path
        self.metrics_file_path = metrics_file_path

    def read_aggregated_metric_from_file(self, metric: str) -> Dict:
        """Reads metrics from a file. These metrics are aggregated metrics.

        Parameters:
        ----------
        metric: str
            The metric to read from the file.

        Returns:
        -------
        pd.DataFrame
            The metrics read from the file.

        Raises:
        ------
        KeyError
            If the metric is not found in the file.
        """

        # Read the metrics from the file
        df_metrics = pd.read_csv(self.metrics_file_path, index_col=0)

        # Remove the desctioption row
        df_metrics = df_metrics.iloc[1:]

        # Check if the metric is in the dataframe
        if metric not in df_metrics.columns:
            raise KeyError(f"The metric {metric} was not found in the file.")

        # Return the metric
        return df_metrics[metric].to_dict()

    def read_metrics_from_file(self) -> Dict:
        """
        Reads metrics from a file.

        Returns
        -------
        pd.DataFrame
            The metrics read from the file.

        Raises
        ------
        KeyError
            If the metric is not found in the file.
        """

        # Read the metrics from the file
        df_metrics = pd.read_csv(self.metrics_file_path, index_col=0)

        # Remove the desctioption row
        df_metrics = df_metrics.iloc[1:]

        # Return the metric
        return df_metrics.transpose().to_dict()["Value"]
