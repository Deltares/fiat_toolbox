import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd


class IMetricsFileWriter(ABC):
    """Interface for writing metrics to a file."""
    logger: logging.Logger

    @abstractmethod
    def __init__(self, config_file: Union[str, Path], logger: logging.Logger = logging.getLogger(__name__)): ...

    @abstractmethod
    def parse_metrics_to_file(
        self,
        df_results: pd.DataFrame,
        metrics_path: Path,
        write_aggregate: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Parse a metrics file and write the metrics to a file.

        Parameters
        ----------
        df_results : pd.DataFrame
            The results dataframe.
        metrics_path : Path
            The path to where to store the metrics file.
        write_aggregate : str
            The name of the aggregation label to write to the metrics file
            (None for no aggregation label, 'all' for all possible ones).
        overwrite : bool
            Whether to overwrite the existing metrics file if it already exists.
        """
        pass


class IMetricsFileReader(ABC):
    """Interface for reading metrics from a file."""
    logger: logging.Logger

    @abstractmethod
    def __init__(self, metrics_file_path: Union[str, Path], logger: logging.Logger = logging.getLogger(__name__)): ...


    @abstractmethod
    def read_metrics_from_file(self, **kwargs) -> pd.Series:
        """
        Reads metrics from a file.

        Parameters
        ----------
        **kwargs
            Keyword arguments.

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

    @abstractmethod
    def read_aggregated_metric_from_file(self, metric: str) -> pd.Series:
        """
        Reads metrics from a file. These metrics are aggregated metrics.

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
