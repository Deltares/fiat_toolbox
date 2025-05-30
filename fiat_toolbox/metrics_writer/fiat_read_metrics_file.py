import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd

from fiat_toolbox.metrics_writer.fiat_metrics_interface import IMetricsFileReader


class MetricsFileReader(IMetricsFileReader):
    """Reads metrics from a file."""

    def __init__(
        self,
        metrics_file_path: Union[str, Path],
        logger: logging.Logger = logging.getLogger(__name__),
    ):
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

        # Convert the path to a Path object
        if not isinstance(metrics_file_path, Path):
            metrics_file_path = Path(metrics_file_path)

        # Check if the file is a csv file
        if not metrics_file_path.suffix == ".csv":
            raise ValueError("The file must be a csv file.")

        # Check if the file exists
        if not os.path.exists(metrics_file_path):
            raise FileNotFoundError("The file does not exist.")

        # Set the metrics file path
        self.metrics_file_path = metrics_file_path
        self.logger = logger

    def read_aggregated_metric_from_file(self, metric: str) -> pd.DataFrame:
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
        return df_metrics[metric]

    def read_metrics_from_file(self, **kwargs) -> pd.DataFrame:
        """
        Reads metrics from a file.

        Parameters
        ----------
        include_long_names : bool
            Include the long names of the metrics.
        include_metrics_table_selection : bool
            Include the metrics table selection.
        include_metrics_map_selection : bool
            Include the metrics map selection.
        include_description : bool
            Include the description of the metrics.

        Returns
        -------
        pd.DataFrame
            The metrics read from the file.

        Raises
        ------
        KeyError
            If the metric is not found in the file.
        """

        # Set the default values
        include_long_names = kwargs.get("include_long_names", False)
        include_metrics_table_selection = kwargs.get(
            "include_metrics_table_selection", False
        )
        include_metrics_map_selection = kwargs.get(
            "include_metrics_map_selection", False
        )
        include_description = kwargs.get("include_description", False)

        # Read the metrics from the file
        df_metrics = pd.read_csv(self.metrics_file_path, index_col=0)

        # If you can't grab the value, transpose the data
        if "Value" not in df_metrics.columns:
            df_metrics = df_metrics.transpose()

        # If the value is still not one of the columns, the metrics file is aggregated
        if "Value" not in df_metrics.columns:
            aggregations = set(df_metrics.columns) - {
                "Description",
                "Long Name",
                "Show In Metrics Table",
                "Show In Metrics Map",
            }

            # Ensure values are interpreted as numbers
            for aggregation in aggregations:
                df_metrics[aggregation] = pd.to_numeric(df_metrics[aggregation])

            # Remove the desctioption row
            if not include_description:
                df_metrics = df_metrics.drop("Description", axis="columns")

            # Remove the long names row
            if not include_long_names:
                df_metrics = df_metrics.drop("Long Name", axis="columns")

            # Remove the metrics table selection row
            if not include_metrics_table_selection:
                df_metrics = df_metrics.drop("Show In Metrics Table", axis="columns")

            # Remove the metrics map selection row
            if not include_metrics_map_selection:
                df_metrics = df_metrics.drop("Show In Metrics Map", axis="columns")

        else:
            # Ensure values are interpreted as numbers
            df_metrics["Value"] = pd.to_numeric(df_metrics["Value"])

            # Remove the desctioption row
            if not include_description:
                df_metrics = df_metrics.drop("Description", axis="columns")

            # Remove the long names row
            if not include_long_names:
                df_metrics = df_metrics.drop("Long Name", axis="columns")

            # Remove the metrics table selection row
            if not include_metrics_table_selection:
                df_metrics = df_metrics.drop("Show In Metrics Table", axis="columns")

            # Remove the metrics map selection row
            if not include_metrics_map_selection:
                df_metrics = df_metrics.drop("Show In Metrics Map", axis="columns")

        # Return the metric
        return df_metrics
