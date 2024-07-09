import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import duckdb
import pandas as pd
import tomli

from fiat_toolbox.metrics_writer.fiat_metrics_interface import IMetricsFileWriter
from fiat_toolbox.metrics_writer.fiat_read_metrics_file import MetricsFileReader


# sql command struct
@dataclass
class sql_struct:
    name: str
    long_name: str
    show_in_metrics_table: bool
    description: str
    select: str
    filter: str
    groupby: str


class MetricsFileWriter(IMetricsFileWriter):
    """Class to parse metrics and write to a file."""

    def __init__(self, config_file: Union[str, Path]):
        """
        Initialize the class.

        Parameters
        ----------
        config_file : Union[str, Path]
            The path to the metrics file.
        """
        # Convert the path to a Path object
        if isinstance(config_file, str):
            config_file = Path(config_file)

        # Check whether the file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")

        self.config_file = config_file

    def _read_metrics_file(
        self, include_aggregates: bool
    ) -> Union[Dict[str, sql_struct], Dict[str, Dict[str, sql_struct]]]:
        """
        Read a metrics file and return a list of SQL commands.

        Parameters
        ----------
        include_aggregates : bool
            Whether to include aggregation labels in the metrics.

        Returns
        -------
        Union[Dict[str, sql_struct], Dict[str, Dict[str, sql_struct]]]
            A dictionary with the SQL commands.
        """

        # Read the metrics file
        _, extension = os.path.splitext(self.config_file)
        if extension == ".json":
            metrics = json.load(open(self.config_file, "r"))
        elif extension == ".toml":
            metrics = tomli.load(open(self.config_file, "rb"))
        else:
            raise ValueError(
                f"Config file '{self.config_file}' has an invalid extension. Only .json and .toml are supported."
            )

        # Create the sql commands dictionary
        sql_command_set = {}
        if include_aggregates:
            # Check whether the metrics file contains aggregation labels
            if "aggregateBy" not in metrics or len(metrics["aggregateBy"]) == 0:
                raise ValueError(
                    "No aggregation labels specified in the metrics file, but include_aggregates is set to True."
                )
            # Loop over the aggregation labels
            for aggregate in metrics["aggregateBy"]:
                aggregate_command = {}
                # Check whether the metrics file contains metrics
                if "queries" not in metrics or len(metrics["queries"]) == 0:
                    raise ValueError("No queries specified in the metrics file.")
                # Loop over the metrics
                for metric in metrics["queries"]:
                    # Check whether the metric contains all required fields
                    if all(
                        key in metrics
                        for key in [
                            "name",
                            "long_name",
                            "show_in_metrics_table",
                            "description",
                            "select",
                            "filter",
                        ]
                    ):
                        raise ValueError(
                            f"The metrics file for metric {metric['name']} does not contain all required fields."
                        )
                    # Create the sql command
                    sql_command = sql_struct(
                        name=metric["name"],
                        long_name=metric["long_name"],
                        show_in_metrics_table=metric["show_in_metrics_table"],
                        description=metric["description"],
                        select=metric["select"],
                        filter=metric["filter"],
                        groupby="`Aggregation Label: " + aggregate + "`",
                    )

                    # Check whether the metric name is already in the dictionary
                    if metric["name"] in aggregate_command:
                        raise ValueError(
                            f"Duplicate metric name {metric['name']} in metrics file."
                        )

                    # Add the sql command to the dictionary
                    aggregate_command[metric["name"]] = sql_command

                # Check whether the aggregation label is already in the dictionary
                if aggregate in sql_command_set:
                    raise ValueError(
                        f"Duplicate aggregation label {aggregate} in metrics file."
                    )

                # Add the sql command to the dictionary
                sql_command_set[aggregate] = aggregate_command
        else:
            # Check whether the metrics file contains metrics
            if "queries" not in metrics or len(metrics["queries"]) == 0:
                raise ValueError("No queries specified in the metrics file.")

            # Loop over the metrics
            for metric in metrics["queries"]:
                # Check whether the metric contains all required fields
                if all(
                    key in metrics
                    for key in [
                        "name",
                        "long_name",
                        "show_in_metrics_table",
                        "description",
                        "select",
                        "filter",
                    ]
                ):
                    raise ValueError(
                        f"The metrics file for metric {metric['name']} does not contain all required fields."
                    )

                # Create the sql command
                sql_command = sql_struct(
                    name=metric["name"],
                    long_name=metric["long_name"],
                    show_in_metrics_table=metric["show_in_metrics_table"],
                    description=metric["description"],
                    select=metric["select"],
                    filter=metric["filter"],
                    groupby="",
                )

                # Check whether the metric name is already in the dictionary
                if metric["name"] in sql_command_set:
                    raise ValueError(
                        f"Duplicate metric name {metric['name']} in metrics file."
                    )

                # Add the sql command to the dictionary
                sql_command_set[metric["name"]] = sql_command

        # Return the sql commands dictionary
        return sql_command_set

    @staticmethod
    def _create_single_metric(
        df_results: pd.DataFrame, sql_command: sql_struct
    ) -> Union[Tuple[str, object], Tuple[str, Dict[str, object]]]:
        """
        Create a metrics table from the results dataframe based on an SQL command.

        Parameters
        ----------
        df_results : pd.DataFrame
            The results dataframe.
        sql_command : sql_struct
            The SQL command.

        Returns
        -------
        Union[Tuple[str, object], Tuple[str, Dict[str, object]]]
            A tuple with the metric name and value or, in the case of a groupby statement,
            a tuple with the metric name and a dictionary with the groupby variables as keys
            and the metric as value.
        """

        # First add the the groupby variables to the query
        sql_query = "SELECT "
        if sql_command.groupby:
            sql_query += f"{sql_command.groupby}, "

        # Then add the select variables
        if not sql_command.select:
            raise ValueError(
                f"No select statement specified for metric {sql_command.name}."
            )
        sql_query += f"{sql_command.select} AS `{sql_command.name}` FROM df_results"

        # Then add the filter statement
        if sql_command.filter:
            sql_query += f" WHERE {sql_command.filter}"

        # Finally add the groupby statement
        if sql_command.groupby:
            sql_query += f" GROUP BY {sql_command.groupby}"

        # Execute the query. If the query is invalid, an error PandaSQLException will be raised
        sql_query = sql_query.replace("`", '"')
        result = duckdb.query(sql_query).df()

        # If the command contains a groupby statement, return a dictionary with the groupby variables as keys and the metric as value
        if sql_command.groupby:
            # Set the groupby variables as index
            labeled_result = result.set_index(sql_command.groupby.replace("`", ""))
            # Remove rows without index name
            labeled_result = labeled_result[labeled_result.index.notna()]
            # Return the metric name and the dictionary
            return labeled_result.columns[0], dict(
                labeled_result[labeled_result.columns[0]]
            )
        # Otherwise return the metric name and the value
        return result.columns[0], result[result.columns[0]][0]

    @staticmethod
    def _create_metrics_dict(
        df_results: pd.DataFrame, sql_commands: Dict[str, sql_struct]
    ) -> Dict[str, object]:
        """
        Create a metrics table from the results dataframe based on a list of SQL commands.

        Parameters
        ----------
        df_results : pd.DataFrame
            The results dataframe.
        sql_commands : list[sql_struct]
            A list of SQL commands.

        Returns
        -------
        dict
            A dictionary with the metric names and values.
        """

        # Initialize the metrics dictionary
        df_metrics = {}

        # Run the sql commands one by one
        for name, command in sql_commands.items():
            # Create the metric (_create_single_metric is a static method, so no need to instantiate the class)
            _, value = MetricsFileWriter._create_single_metric(df_results, command)

            # Store the metric in the metrics dictionary using the metric name as key
            df_metrics[name] = value

        return df_metrics

    def _parse_metrics(
        self, df_results: pd.DataFrame, include_aggregates: bool
    ) -> Union[dict, Dict[str, dict]]:
        """
        Parse the metrics based on the config file and return a dictionary with the metrics.

        Parameters
        ----------
        df_results : pd.DataFrame
            The results dataframe.
        include_aggregates : bool
            Whether to include aggregation labels in the metrics.

        Returns
        -------
        Union[dict, List[dict]]
            A dictionary with the metrics or, in the case of multiple aggregation labels,
            a list of dictionaries.
        """

        # Read the metrics file
        sql_commands = self._read_metrics_file(include_aggregates)

        # Create the metrics dictionary
        if include_aggregates:
            metrics = {}
            # Loop over the aggregation labels
            for aggregate, commands in sql_commands.items():
                # Create the metrics dictionary for the current aggregation label (the _create_metrics_dict is a static method, so no need to instantiate the class)
                metrics[aggregate] = MetricsFileWriter._create_metrics_dict(
                    df_results, commands
                )
            return metrics
        else:
            # Create the metrics dictionary (the _create_metrics_dict is a static method, so no need to instantiate the class)
            return MetricsFileWriter._create_metrics_dict(df_results, sql_commands)

    @staticmethod
    def _write_metrics_file(
        metrics: Union[dict, Dict[str, dict]],
        config: Union[Dict[str, sql_struct], Dict[str, Dict[str, sql_struct]]],
        metrics_path: Path,
        write_aggregate: str = None,
        overwrite: bool = False,
        aggregations: list = None,
    ) -> None:
        """
        Write a metrics dictionary to a metrics file.

        Parameters
        ----------
        metrics : Union[dict, List[dict]]
            A dictionary with the metrics or, in the case of multiple aggregation labels,
            a list of dictionaries.
        config : Union[Dict[str, sql_struct], Dict[str, Dict[str, sql_struct]]]
            A dictionary with the SQL commands.
        metrics_path : Path
            The path to where to store the metrics file.
        write_aggregate : str
            The name of the aggregation label to write to the metrics file (None for no aggregation label).
        overwrite : bool
            Whether to overwrite the existing metrics file if it already exists. If False, it appends to the file.
        aggregations : list
            A list of aggregation areas. If write_aggregate is None, this is ignored.

        Returns
        -------
        None
        """

        if write_aggregate:
            # Get the metrics for the current aggregation label
            aggregate_metrics = metrics[write_aggregate]

            # Find the names dynamically
            if aggregations is None:
                for value in aggregate_metrics.values():
                    aggregations.update(value.keys())

            # Update all empty metrics with 0
            for key, value in aggregate_metrics.items():
                if value == {}:
                    aggregate_metrics[key] = {name: 0 for name in aggregations}
                    continue
                for name in aggregations:
                    if name not in value:
                        aggregate_metrics[key][name] = 0

            # Create a dataframe from the metrics dictionary
            metricsFrame = (
                pd.DataFrame().from_dict(aggregate_metrics, orient="index").fillna(0)
            )

            # Add the long name to the dataframe
            metricsFrame.insert(
                0,
                "Long Name",
                [
                    config[write_aggregate][name].long_name
                    for name, _ in metricsFrame.iterrows()
                ],
            )

            # Add the metrics table selector to the dataframe
            metricsFrame.insert(
                0,
                "Show In Metrics Table",
                [
                    config[write_aggregate][name].show_in_metrics_table
                    for name, _ in metricsFrame.iterrows()
                ],
            )

            # Add the description to the dataframe
            metricsFrame.insert(
                0,
                "Description",
                [
                    config[write_aggregate][name].description
                    for name, _ in metricsFrame.iterrows()
                ],
            )

            # Check if the file already exists
            if os.path.exists(metrics_path):
                if overwrite:
                    logging.warning(
                        f"Metrics file '{metrics_path}' already exists. Overwriting..."
                    )
                    os.remove(metrics_path)
                else:
                    new_metrics = MetricsFileReader(
                        metrics_path
                    ).read_metrics_from_file(
                        include_long_names=True,
                        include_description=True,
                        include_metrics_table_selection=True,
                    )
                    metricsFrame = new_metrics.append(metricsFrame)

            # Transpose the dataframe
            metricsFrame = metricsFrame.transpose()

            # Write the metrics to a file
            if metrics_path.parent and not metrics_path.parent.exists():
                metrics_path.parent.mkdir(parents=True)
            metricsFrame.to_csv(metrics_path)
        else:
            # Create a dataframe from the metrics dictionary
            metricsFrame = (
                pd.DataFrame()
                .from_dict(metrics, orient="index", columns=["Value"])
                .fillna(0)
            )

            # Add the long name to the dataframe
            metricsFrame.insert(
                0,
                "Long Name",
                [config[name].long_name for name, _ in metricsFrame.iterrows()],
            )

            # Add the metrics table selector to the dataframe
            metricsFrame.insert(
                0,
                "Show In Metrics Table",
                [
                    config[name].show_in_metrics_table
                    for name, _ in metricsFrame.iterrows()
                ],
            )

            # Add the description to the dataframe
            metricsFrame.insert(
                0,
                "Description",
                [config[name].description for name, _ in metricsFrame.iterrows()],
            )

            # Check if the file already exists
            if os.path.exists(metrics_path):
                if overwrite:
                    logging.warning(
                        f"Metrics file '{metrics_path}' already exists. Overwriting..."
                    )
                    os.remove(metrics_path)
                else:
                    new_metrics = MetricsFileReader(
                        metrics_path
                    ).read_metrics_from_file(
                        include_long_names=True,
                        include_description=True,
                        include_metrics_table_selection=True,
                    )
                    metricsFrame = new_metrics.append(metricsFrame)

            # Write the metrics to a file
            if metrics_path.parent and not metrics_path.parent.exists():
                metrics_path.parent.mkdir(parents=True)
            metricsFrame.to_csv(metrics_path)

    def parse_metrics_to_file(
        self,
        df_results: pd.DataFrame,
        metrics_path: Union[str, Path],
        write_aggregate: str = None,
        overwrite: bool = False,
    ) -> Union[str, Dict[str, str]]:
        """
        Parse a metrics file and write the metrics to a file.

        Parameters
        ----------
        df_results : pd.DataFrame
            The results dataframe.
        metrics_path : Union[str, Path]
            The path to where to store the metrics file.
        write_aggregate : str
            The name of the aggregation label to write to the metrics file
            (None for no aggregation label, 'all' for all possible ones).
        overwrite : bool
            Whether to overwrite the existing metrics file if it already exists. If False, it appends to the file.

        Returns
        -------
        Union[str, Dict[str, str]]
            The path to the metrics file or a dictionary with the aggregation labels as keys
            and the paths to the metrics files as values.
        """

        # Convert the path to a Path object
        if isinstance(metrics_path, str):
            metrics_path = Path(metrics_path)

        # Check whether to include aggregation labels
        include_aggregates = True if write_aggregate else False

        # Read the metrics config file
        config = self._read_metrics_file(include_aggregates)

        # Parse the metrics
        metrics = self._parse_metrics(df_results, include_aggregates)

        # Write the metrics to a file
        if write_aggregate == "all":
            # Initialize the return dictionary
            return_files = {}
            for key in config.keys():
                # If using aggregation labels, add the aggregation label to the filename
                directory, filename = os.path.split(metrics_path)
                filename, extension = os.path.splitext(filename)
                new_filename = filename + "_" + key + extension
                new_path = Path(os.path.join(directory, new_filename))
                return_files[key] = new_path

                # Write the metrics to a file
                MetricsFileWriter._write_metrics_file(
                    metrics,
                    config,
                    new_path,
                    write_aggregate=key,
                    overwrite=overwrite,
                    aggregations=df_results["Aggregation Label: " + key].unique(),
                )
        else:
            # Write the metrics to a file
            MetricsFileWriter._write_metrics_file(
                metrics,
                config,
                metrics_path,
                write_aggregate=write_aggregate,
                overwrite=overwrite,
            )
            return_files = metrics_path

        return return_files
