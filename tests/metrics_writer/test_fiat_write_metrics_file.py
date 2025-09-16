import os
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

from fiat_toolbox.metrics_writer.fiat_write_metrics_file import (
    MetricsFileWriter,
    sql_struct,
)
from pydantic import ValidationError  # Add this import at the top if not present


class TestReadMetricsConfigFile(unittest.TestCase):
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_without_aggregates(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "queries": [
                {
                    "name": "Total Damage Sum",
                    "long_name": "Total Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`Object ID` > 2",
                    "groupby": "",
                },
                {
                    "name": "Single Family Damage Sum",
                    "long_name": "Single Family Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event for only single families",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    "groupby": "",
                },
            ]
        }

        # Act
        write_class = MetricsFileWriter(config_file)
        sql_prompts_without_aggregates = write_class._read_metrics_file(
            include_aggregates=False
        )

        # Assert
        sql_prompts_expected = {
            "Total Damage Sum": sql_struct(
                name="Total Damage Sum",
                long_name="Total Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event",
                select="SUM(`Total Damage Event`)",
                filter="`Object ID` > 2",
                groupby="",
            ),
            "Single Family Damage Sum": sql_struct(
                name="Single Family Damage Sum",
                long_name="Single Family Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event for only single families",
                select="SUM(`Total Damage Event`)",
                filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                groupby="",
            ),
        }
        self.assertEqual(sql_prompts_without_aggregates, sql_prompts_expected)

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.tomli.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_without_aggregates_toml(
        self, mock_path_exists, mock_open, mock_toml_load
    ):
        # Arrange
        config_file = Path("config_file.toml")
        mock_path_exists.return_value = True
        mock_toml_load.return_value = {
            "queries": [
                {
                    "name": "Total Damage Sum",
                    "long_name": "Total Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`Object ID` > 2",
                    "groupby": "",
                },
                {
                    "name": "Single Family Damage Sum",
                    "long_name": "Single Family Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event for only single families",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    "groupby": "",
                },
            ]
        }

        # Act
        write_class = MetricsFileWriter(config_file)
        sql_prompts_without_aggregates = write_class._read_metrics_file(
            include_aggregates=False
        )

        # Assert
        sql_prompts_expected = {
            "Total Damage Sum": sql_struct(
                name="Total Damage Sum",
                long_name="Total Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event",
                select="SUM(`Total Damage Event`)",
                filter="`Object ID` > 2",
                groupby="",
            ),
            "Single Family Damage Sum": sql_struct(
                name="Single Family Damage Sum",
                long_name="Single Family Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event for only single families",
                select="SUM(`Total Damage Event`)",
                filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                groupby="",
            ),
        }
        self.assertEqual(sql_prompts_without_aggregates, sql_prompts_expected)

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_with_aggregates(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "aggregateBy": ["Subbasin", "Tax Use"],
            "queries": [
                {
                    "name": "Total Damage Sum",
                    "long_name": "Total Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`Object ID` > 2",
                    "groupby": "",
                },
                {
                    "name": "Single Family Damage Sum",
                    "long_name": "Single Family Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event for only single families",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    "groupby": "",
                },
            ],
        }

        # Act
        write_class = MetricsFileWriter(config_file)
        sql_prompts_list_with_aggregates = write_class._read_metrics_file(
            include_aggregates=True
        )

        # Assert
        sql_prompts_expected = {
            "Subbasin": {
                "Total Damage Sum": sql_struct(
                    name="Total Damage Sum",
                    long_name="Total Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event",
                    select="SUM(`Total Damage Event`)",
                    filter="`Object ID` > 2",
                    groupby="`aggregation_label:Subbasin`",
                ),
                "Single Family Damage Sum": sql_struct(
                    name="Single Family Damage Sum",
                    long_name="Single Family Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event for only single families",
                    select="SUM(`Total Damage Event`)",
                    filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    groupby="`aggregation_label:Subbasin`",
                ),
            },
            "Tax Use": {
                "Total Damage Sum": sql_struct(
                    name="Total Damage Sum",
                    long_name="Total Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event",
                    select="SUM(`Total Damage Event`)",
                    filter="`Object ID` > 2",
                    groupby="`aggregation_label:Tax Use`",
                ),
                "Single Family Damage Sum": sql_struct(
                    name="Single Family Damage Sum",
                    long_name="Single Family Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event for only single families",
                    select="SUM(`Total Damage Event`)",
                    filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    groupby="`aggregation_label:Tax Use`",
                ),
            },
        }

        self.assertEqual(sql_prompts_list_with_aggregates, sql_prompts_expected)

    def test_read_metrics_file_no_file(self):
        # Arrange
        config_file = Path("config_file.json")

        # Assert
        with self.assertRaises(FileNotFoundError) as context:
            MetricsFileWriter(config_file)
            self.assertTrue(
                "Config file config_file.json not found" in str(context.exception)
            )

        with self.assertRaises(FileNotFoundError) as context:
            MetricsFileWriter(config_file)
            self.assertTrue(
                "Config file config_file.json not found" in str(context.exception)
            )

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_unsupported_file_extension(self, mock_path_exists, mock_open):
        # Arrange
        config_file = Path("config_file.txt")

        # Act
        write_class = MetricsFileWriter(config_file)

        # Assert
        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=False)
            self.assertTrue(
                "Config file 'config_file.txt' has an invalid extension. Only .json and .toml are supported."
                in str(context.exception)
            )

        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=True)
            self.assertTrue(
                "Config file 'config_file.txt' has an invalid extension. Only .json and .toml are supported."
                in str(context.exception)
            )

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_no_queries(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {"aggregateBy": ["Subbasin", "Tax Use"]}

        # Act
        write_class = MetricsFileWriter(config_file)

        # Assert
        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=False)
            self.assertTrue(
                "No queries specified in the metrics file." in str(context.exception)
            )

        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=True)
            self.assertTrue(
                "No queries specified in the metrics file." in str(context.exception)
            )

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_no_aggregateBy(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "queries": [
                {
                    "name": "Total Damage Sum",
                    "description": "Total of the damage event",
                    "show_in_metrics_table": True,
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`Object ID` > 2",
                    "groupby": "",
                },
                {
                    "name": "Single Family Damage Sum",
                    "description": "Total of the damage event for only single families",
                    "show_in_metrics_table": True,
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    "groupby": "",
                },
            ]
        }

        # Act
        write_class = MetricsFileWriter(config_file)

        # Assert
        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=True)
            self.assertTrue(
                "No aggregation labels specified in the metrics file, but include_aggregates is set to True."
                in str(context.exception)
            )

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_no_aggregateBy_or_queries(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {}

        # Act
        write_class = MetricsFileWriter(config_file)

        # Assert
        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=True)
            self.assertTrue(
                "No aggregation labels specified in the metrics file, but include_aggregates is set to True."
                in str(context.exception)
            )

        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=False)
            self.assertTrue(
                "No queries specified in the metrics file." in str(context.exception)
            )

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_empty_aggregateBy(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "aggregateBy": [],
            "queries": [
                {
                    "name": "Total Damage Sum",
                    "description": "Total of the damage event",
                    "show_in_metrics_table": True,
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`Object ID` > 2",
                    "groupby": "",
                },
                {
                    "name": "Single Family Damage Sum",
                    "description": "Total of the damage event for only single families",
                    "show_in_metrics_table": True,
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    "groupby": "",
                },
            ],
        }

        # Act
        write_class = MetricsFileWriter(config_file)

        # Assert
        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=True)
            self.assertTrue(
                "No aggregation labels specified in the metrics file, but include_aggregates is set to True."
                in str(context.exception)
            )

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_empty_queries(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "aggregateBy": ["Subbasin", "Tax Use"],
            "queries": [],
        }

        # Act
        write_class = MetricsFileWriter(config_file)

        # Assert
        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=True)
            self.assertTrue(
                "No queries specified in the metrics file." in str(context.exception)
            )

        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=False)
            self.assertTrue(
                "No queries specified in the metrics file." in str(context.exception)
            )

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_key_missing(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "aggregateBy": ["Subbasin", "Tax Use"],
            "queries": [
                {
                    "name": "Total Damage Sum",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`Object ID` > 2",
                    "groupby": "",
                },
                {
                    "description": "Total of the damage event for only single families",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    "groupby": "",
                },
            ],
        }

        # Act
        write_class = MetricsFileWriter(config_file)

        # Assert
        with self.assertRaises(ValidationError) as context:
            write_class._read_metrics_file(include_aggregates=True)
        self.assertIn(
            "Field required", str(context.exception)
        )
        self.assertIn(
            "description", str(context.exception)
        )
        self.assertIn(
            "long_name", str(context.exception)
        )


    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_duplicate_metric_name(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "aggregateBy": ["Subbasin", "Tax Use"],
            "queries": [
                {
                    "name": "Total Damage Sum",
                    "long_name": "Total Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`Object ID` > 2",
                    "groupby": "",
                },
                {
                    "name": "Total Damage Sum",
                    "long_name": "Total Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event for only single families",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    "groupby": "",
                },
            ],
        }

        # Act
        write_class = MetricsFileWriter(config_file)

        # Assert
        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=True)
            self.assertTrue(
                "Duplicate metric name Total Damage Sum." in str(context.exception)
            )

        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=False)
            self.assertTrue(
                "Duplicate metric name Total Damage Sum." in str(context.exception)
            )

    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.json.load")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.open")
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_read_metrics_file_duplicate_aggregate_name(
        self, mock_path_exists, mock_open, mock_json_load
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_path_exists.return_value = True
        mock_json_load.return_value = {
            "aggregateBy": ["Subbasin", "Subbasin"],
            "queries": [
                {
                    "name": "Total Damage Sum",
                    "long_name": "Total Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`Object ID` > 2",
                    "groupby": "",
                },
                {
                    "name": "Single Family Damage Sum",
                    "long_name": "Single Family Damage Sum",
                    "show_in_metrics_table": True,
                    "description": "Total of the damage event for only single families",
                    "select": "SUM(`Total Damage Event`)",
                    "filter": "`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    "groupby": "",
                },
            ],
        }

        # Act
        write_class = MetricsFileWriter(config_file)

        # Assert
        with self.assertRaises(ValueError) as context:
            write_class._read_metrics_file(include_aggregates=True)
            self.assertTrue(
                "Duplicate aggregate name Subbasin." in str(context.exception)
            )


class TestCreateSingleMetric(unittest.TestCase):
    def test_create_single_metric(self):
        # Arrange
        df_results = pd.DataFrame(
            {
                "Object ID": [1, 2, 3, 4, 5],
                "Total Damage Event": [100, 200, 300, 400, 500],
            }
        )
        sql_command = sql_struct(
            name="Total Damage Sum",
            long_name="Total Damage Sum",
            show_in_metrics_table=True,
            description="Total of the damage event",
            select="SUM(`Total Damage Event`)",
            filter="`Object ID` > 2",
            groupby="",
        )

        # Act
        (
            TotalDamageSumName,
            TotalDamageSumValue,
        ) = MetricsFileWriter._create_single_metric(
            df_results=df_results, sql_command=sql_command
        )

        # Assert
        self.assertEqual(TotalDamageSumName, "Total Damage Sum")
        self.assertEqual(TotalDamageSumValue, 1200)

    def test_create_single_metric_no_filter(self):
        # Arrange
        df_results = pd.DataFrame(
            {
                "Object ID": [1, 2, 3, 4, 5],
                "Total Damage Event": [100, 200, 300, 400, 500],
            }
        )
        sql_command = sql_struct(
            name="Total Damage Sum",
            long_name="Total Damage Sum",
            show_in_metrics_table=True,
            description="Total of the damage event",
            select="SUM(`Total Damage Event`)",
            filter="",
            groupby="",
        )

        # Act
        (
            TotalDamageSumName,
            TotalDamageSumValue,
        ) = MetricsFileWriter._create_single_metric(
            df_results=df_results, sql_command=sql_command
        )

        # Assert
        self.assertEqual(TotalDamageSumName, "Total Damage Sum")
        self.assertEqual(TotalDamageSumValue, 1500)

    def test_create_single_metric_aggregated(self):
        # Arrange
        df_results = pd.DataFrame(
            {
                "Object ID": [1, 2, 3, 4, 5],
                "aggregation_label:Tax Use": [
                    "SINGLE FAMILY",
                    "SINGLE FAMILY",
                    "VACANT RESIDENTIAL",
                    "VACANT COMMERCIAL",
                    "VACANT COMMERCIAL",
                ],
                "Total Damage Event": [100, 200, 300, 400, 500],
            }
        )
        sql_command = sql_struct(
            name="Total Damage Sum",
            long_name="Total Damage Sum",
            show_in_metrics_table=True,
            description="Total of the damage event",
            select="SUM(`Total Damage Event`)",
            filter="",
            groupby="`aggregation_label:Tax Use`",
        )

        # Act
        (
            TotalDamageSumName,
            TotalDamageSumValue,
        ) = MetricsFileWriter._create_single_metric(
            df_results=df_results, sql_command=sql_command
        )

        # Assert
        self.assertEqual(TotalDamageSumName, "Total Damage Sum")
        self.assertEqual(
            TotalDamageSumValue,
            {"SINGLE FAMILY": 300, "VACANT RESIDENTIAL": 300, "VACANT COMMERCIAL": 900},
        )

    def test_incorrect_query(self):
        # Arrange
        df_results = pd.DataFrame(
            {
                "Object ID": [1, 2, 3, 4, 5],
                "Total Damage Event": [100, 200, 300, 400, 500],
            }
        )
        sql_command = sql_struct(
            name="TotalDamageSum",
            long_name="Total Damage Sum",
            show_in_metrics_table=True,
            description="Total of the damage event",
            select="SUM(`Total Damage Event`)",
            filter="this filter doesnt make sense",
            groupby="",
        )

        # Act & Assert
        with self.assertRaises(duckdb.ParserException):
            MetricsFileWriter._create_single_metric(
                df_results=df_results, sql_command=sql_command
            )

    def test_missing_select(self):
        # Arrange
        df_results = pd.DataFrame(
            {
                "Object ID": [1, 2, 3, 4, 5],
                "Total Damage Event": [100, 200, 300, 400, 500],
            }
        )
        sql_command = sql_struct(
            name="TotalDamageSum",
            long_name="Total Damage Sum",
            show_in_metrics_table=True,
            description="Total of the damage event",
            select="",
            filter="`Object ID` > 2",
            groupby="",
        )

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            MetricsFileWriter._create_single_metric(
                df_results=df_results, sql_command=sql_command
            )
            self.assertTrue(
                "No select statement specified for metric TotalDamageSum."
                in str(context.exception)
            )


class TestCreateMetricsDict(unittest.TestCase):
    @patch(
        "fiat_toolbox.metrics_writer.fiat_write_metrics_file.MetricsFileWriter._create_single_metric"
    )
    def test_create_metrics_dict(self, mock_create_single_metric):
        # Arrange
        mock_create_single_metric.side_effect = [
            ("Total Damage Sum", 1200),
            (
                "Aggregated Damage Sum",
                {
                    "SINGLE FAMILY": 300,
                    "VACANT RESIDENTIAL": 300,
                    "VACANT COMMERCIAL": 900,
                },
            ),
        ]

        df_results = pd.DataFrame(
            {
                "Object ID": [1, 2, 3, 4, 5],
                "aggregation_label:Tax Use": [
                    "SINGLE FAMILY",
                    "SINGLE FAMILY",
                    "VACANT RESIDENTIAL",
                    "VACANT COMMERCIAL",
                    "VACANT COMMERCIAL",
                ],
                "Total Damage Event": [100, 200, 300, 400, 500],
            }
        )

        sql_prompts = {
            "Total Damage Sum": sql_struct(
                name="Total Damage Sum",
                long_name="Total Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event",
                select="SUM(`Total Damage Event`)",
                filter="`Object ID` > 2",
                groupby="",
            ),
            "Aggregated Damage Sum": sql_struct(
                name="Aggregated Damage Sum",
                long_name="Aggregated Damage Sum",
                show_in_metrics_table=True,
                description="Aggregate of the damage event",
                select="SUM(`Total Damage Event`)",
                filter="",
                groupby="`aggregation_label:Tax Use`",
            ),
        }

        # Act
        metricsTable = MetricsFileWriter._create_metrics_dict(
            df_results=df_results, sql_commands=sql_prompts
        )

        # Assert
        self.assertEqual(
            metricsTable,
            {
                "Total Damage Sum": 1200,
                "Aggregated Damage Sum": {
                    "SINGLE FAMILY": 300,
                    "VACANT RESIDENTIAL": 300,
                    "VACANT COMMERCIAL": 900,
                },
            },
        )


class TestParseMetrics(unittest.TestCase):
    @patch(
        "fiat_toolbox.metrics_writer.fiat_write_metrics_file.MetricsFileWriter._read_metrics_file"
    )
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_parse_metrics_config_file_without_aggregates(
        self, mock_check_exists, mock_read_metrics_file
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_check_exists.return_value = True
        mock_read_metrics_file.return_value = {
            "Total Damage Sum": sql_struct(
                name="Total Damage Sum",
                long_name="Total Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event",
                select="SUM(`Total Damage Event`)",
                filter="`Object ID` > 2",
                groupby="",
            ),
            "Single Family Damage Sum": sql_struct(
                name="Single Family Damage Sum",
                long_name="Single Family Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event for only single families",
                select="SUM(`Total Damage Event`)",
                filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                groupby="",
            ),
        }
        df_results = pd.DataFrame(
            {
                "Object ID": [1, 2, 3, 4, 5],
                "aggregation_label:Tax Use": [
                    "SINGLE FAMILY",
                    "SINGLE FAMILY",
                    "VACANT RESIDENTIAL",
                    "VACANT COMMERCIAL",
                    "VACANT COMMERCIAL",
                ],
                "aggregation_label:Subbasin": [
                    "OAKFOREST",
                    "OAKFOREST",
                    "OAKFOREST",
                    "BAYLURE",
                    "BAYLURE",
                ],
                "Total Damage Event": [100, 200, 300, 400, 500],
            }
        )

        # Act
        write_class = MetricsFileWriter(config_file)
        metrics = write_class._parse_metrics(df_results, include_aggregates=False)

        # Assert
        metrics_expected = {"Total Damage Sum": 1200, "Single Family Damage Sum": 300}

        self.assertEqual(metrics, metrics_expected)

    @patch(
        "fiat_toolbox.metrics_writer.fiat_write_metrics_file.MetricsFileWriter._read_metrics_file"
    )
    @patch("fiat_toolbox.metrics_writer.fiat_write_metrics_file.os.path.exists")
    def test_parse_metrics_config_file_with_aggregates(
        self, mock_check_exists, mock_read_metrics_file
    ):
        # Arrange
        config_file = Path("config_file.json")
        mock_check_exists.return_value = True
        mock_read_metrics_file.return_value = {
            "Subbasin": {
                "Total Damage Sum": sql_struct(
                    name="Total Damage Sum",
                    long_name="Total Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event",
                    select="SUM(`Total Damage Event`)",
                    filter="`Object ID` > 2",
                    groupby="`aggregation_label:Subbasin`",
                ),
                "Single Family Damage Sum": sql_struct(
                    name="Single Family Damage Sum",
                    long_name="Single Family Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event for only single families",
                    select="SUM(`Total Damage Event`)",
                    filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    groupby="`aggregation_label:Subbasin`",
                ),
            },
            "Tax Use": {
                "Total Damage Sum": sql_struct(
                    name="Total Damage Sum",
                    long_name="Total Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event",
                    select="SUM(`Total Damage Event`)",
                    filter="`Object ID` > 2",
                    groupby="`aggregation_label:Tax Use`",
                ),
                "Single Family Damage Sum": sql_struct(
                    name="Single Family Damage Sum",
                    long_name="Single Family Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event for only single families",
                    select="SUM(`Total Damage Event`)",
                    filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    groupby="`aggregation_label:Tax Use`",
                ),
            },
        }
        df_results = pd.DataFrame(
            {
                "Object ID": [1, 2, 3, 4, 5],
                "aggregation_label:Tax Use": [
                    "SINGLE FAMILY",
                    "SINGLE FAMILY",
                    "VACANT RESIDENTIAL",
                    "VACANT COMMERCIAL",
                    "VACANT COMMERCIAL",
                ],
                "aggregation_label:Subbasin": [
                    "OAKFOREST",
                    "OAKFOREST",
                    "OAKFOREST",
                    "BAYLURE",
                    "BAYLURE",
                ],
                "Total Damage Event": [100, 200, 300, 400, 500],
            }
        )

        # Act
        write_class = MetricsFileWriter(config_file)
        metrics = write_class._parse_metrics(df_results, include_aggregates=True)

        # Assert
        metrics_expected = {
            "Subbasin": {
                "Total Damage Sum": {"BAYLURE": 900, "OAKFOREST": 300},
                "Single Family Damage Sum": {"OAKFOREST": 300},
            },
            "Tax Use": {
                "Total Damage Sum": {
                    "VACANT COMMERCIAL": 900,
                    "VACANT RESIDENTIAL": 300,
                },
                "Single Family Damage Sum": {"SINGLE FAMILY": 300},
            },
        }

        self.assertEqual(metrics, metrics_expected)


class TestMetricsFileWriter(unittest.TestCase):
    def test_write_metrics_file_no_aggregation(self):
        # Arrange
        metrics_no_aggregation = {
            "Total Damage Sum": 1200,
            "Single Family Damage Sum": 300,
        }

        sql_prompts_no_aggregation = {
            "Total Damage Sum": sql_struct(
                name="Total Damage Sum",
                long_name="Total Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event",
                select="SUM(`Total Damage Event`)",
                filter="`Object ID` > 2",
                groupby="",
            ),
            "Single Family Damage Sum": sql_struct(
                name="Single Family Damage Sum",
                long_name="Single Family Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event for only single families",
                select="SUM(`Total Damage Event`)",
                filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                groupby="",
            ),
        }

        cwd = Path(os.path.dirname(os.path.abspath(__file__)))

        # Act
        MetricsFileWriter._write_metrics_file(
            metrics_no_aggregation,
            sql_prompts_no_aggregation,
            cwd.joinpath("data", "metrics_no_aggregation.csv"),
        )

        # Assert
        df_metrics_no_aggregation_expected = pd.read_csv(
            cwd.joinpath("data", "test_metrics_no_aggregation.csv")
        )
        df_metrics_no_aggregation = pd.read_csv(
            cwd.joinpath("data", "metrics_no_aggregation.csv")
        )

        self.assertTrue(
            df_metrics_no_aggregation.equals(df_metrics_no_aggregation_expected)
        )
        os.remove(cwd.joinpath("data", "metrics_no_aggregation.csv"))

    def test_write_metrics_file_with_aggregation(self):
        # Arrange
        metrics_with_aggregation = {
            "Subbasin": {
                "Total Damage Sum": {"BAYLURE": 900, "OAKFOREST": 300},
                "Single Family Damage Sum": {"OAKFOREST": 300},
            },
            "Tax Use": {
                "Total Damage Sum": {
                    "VACANT COMMERCIAL": 900,
                    "VACANT RESIDENTIAL": 300,
                },
                "Single Family Damage Sum": {"SINGLE FAMILY": 300},
            },
        }

        sql_prompts_with_aggregation = {
            "Subbasin": {
                "Total Damage Sum": sql_struct(
                    name="Total Damage Sum",
                    long_name="Total Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event",
                    select="SUM(`Total Damage Event`)",
                    filter="`Object ID` > 2",
                    groupby="`aggregation_label:Subbasin`",
                ),
                "Single Family Damage Sum": sql_struct(
                    name="Single Family Damage Sum",
                    long_name="Single Family Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event for only single families",
                    select="SUM(`Total Damage Event`)",
                    filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    groupby="`aggregation_label:Subbasin`",
                ),
            },
            "Tax Use": {
                "Total Damage Sum": sql_struct(
                    name="Total Damage Sum",
                    long_name="Total Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event",
                    select="SUM(`Total Damage Event`)",
                    filter="`Object ID` > 2",
                    groupby="`aggregation_label:Tax Use`",
                ),
                "Single Family Damage Sum": sql_struct(
                    name="Single Family Damage Sum",
                    long_name="Single Family Damage Sum",
                    show_in_metrics_table=True,
                    description="Total of the damage event for only single families",
                    select="SUM(`Total Damage Event`)",
                    filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                    groupby="`aggregation_label:Tax Use`",
                ),
            },
        }

        cwd = Path(os.path.dirname(os.path.abspath(__file__)))

        # Act
        MetricsFileWriter._write_metrics_file(
            metrics_with_aggregation,
            sql_prompts_with_aggregation,
            cwd.joinpath("data", "metrics_subbasin.csv"),
            write_aggregate="Subbasin",
        )
        MetricsFileWriter._write_metrics_file(
            metrics_with_aggregation,
            sql_prompts_with_aggregation,
            cwd.joinpath("data", "metrics_taxuse.csv"),
            write_aggregate="Tax Use",
        )

        # Assert
        df_metrics_subbasin_expected = pd.read_csv(
            cwd.joinpath("data", "test_metrics_subbasin.csv")
        )
        df_metrics_taxuse_expected = pd.read_csv(
            cwd.joinpath("data", "test_metrics_taxuse.csv")
        )

        df_metrics_subbasin = pd.read_csv(cwd.joinpath("data", "metrics_subbasin.csv"))
        df_metrics_taxuse = pd.read_csv(cwd.joinpath("data", "metrics_taxuse.csv"))

        self.assertTrue(df_metrics_subbasin.equals(df_metrics_subbasin_expected))
        self.assertTrue(df_metrics_taxuse.equals(df_metrics_taxuse_expected))

        os.remove(cwd.joinpath("data", "metrics_subbasin.csv"))
        os.remove(cwd.joinpath("data", "metrics_taxuse.csv"))

    def test_write_metrics_file_existing_name(self):
        # Arrange
        metrics_no_aggregation = {
            "Total Damage Sum": 1200,
            "Single Family Damage Sum": 300,
        }

        sql_prompts_no_aggregation = {
            "Total Damage Sum": sql_struct(
                name="Total Damage Sum",
                long_name="Total Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event",
                select="SUM(`Total Damage Event`)",
                filter="`Object ID` > 2",
                groupby="",
            ),
            "Single Family Damage Sum": sql_struct(
                name="Single Family Damage Sum",
                long_name="Single Family Damage Sum",
                show_in_metrics_table=True,
                description="Total of the damage event for only single families",
                select="SUM(`Total Damage Event`)",
                filter="`aggregation_label:Tax Use` == 'SINGLE FAMILY'",
                groupby="",
            ),
        }

        temporary_file_name = Path("temptestfile.txt")
        open(temporary_file_name, "a").close()

        # Act & Assert
        with self.assertLogs(level="WARNING") as cm:
            MetricsFileWriter._write_metrics_file(
                metrics_no_aggregation,
                sql_prompts_no_aggregation,
                temporary_file_name,
                overwrite=True,
            )
            self.assertEqual(
                cm.output,
                [
                    f"WARNING:root:Metrics file '{temporary_file_name}' already exists. Overwriting..."
                ],
            )

        os.remove(temporary_file_name)
