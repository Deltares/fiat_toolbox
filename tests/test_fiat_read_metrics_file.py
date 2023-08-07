import unittest
from unittest.mock import patch

import pandas as pd

from fiat_toolbox.metrics_writer.fiat_read_metrics_file import MetricsFileReader


class TestReadMetricsFile(unittest.TestCase):
    @patch("fiat_toolbox.metrics_writer.fiat_read_metrics_file.pd.read_csv")
    @patch("fiat_toolbox.metrics_writer.fiat_read_metrics_file.os.path.exists")
    def test_read_metrics_file(self, mock_path_exists, mock_read_csv):
        # Arrange
        metrics_file_path = "metrics_file.csv"
        mock_path_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame(
            {
                "": ["Description", 1, 2, 3, 4, 5],
                "Total Damage Event": ["Total of the events", 100, 200, 300, 400, 500],
                "Other metric": ["Just another metric", 0, 0, 0, 0, 0],
            },
            columns=["", "Total Damage Event", "Other metric"],
        ).set_index("")

        # Act
        read_class = MetricsFileReader(metrics_file_path)
        df_results = read_class.read_aggregated_metric_from_file(
            metric="Total Damage Event"
        ).to_dict()

        # Assert
        df_expected = {
            1: 100,
            2: 200,
            3: 300,
            4: 400,
            5: 500,
        }

        self.assertEqual(df_results, df_expected)

    @patch("fiat_toolbox.metrics_writer.fiat_read_metrics_file.os.path.exists")
    def test_read_metrics_file_no_file(self, mock_path_exists):
        # Arrange
        metrics_file_path = "metrics_file.csv"
        mock_path_exists.return_value = False

        # Act & Assert
        with self.assertRaises(FileNotFoundError) as context:
            MetricsFileReader(metrics_file_path)
            self.assertTrue("The file does not exist." in str(context.exception))

    @patch("fiat_toolbox.metrics_writer.fiat_read_metrics_file.os.path.exists")
    def test_read_metrics_file_not_csv(self, mock_path_exists):
        # Arrange
        metrics_file_path = "metrics_file.txt"
        mock_path_exists.return_value = True

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            MetricsFileReader(metrics_file_path)
            self.assertTrue("The file must be a csv file." in str(context.exception))

    @patch("fiat_toolbox.metrics_writer.fiat_read_metrics_file.os.path.exists")
    @patch("fiat_toolbox.metrics_writer.fiat_read_metrics_file.pd.read_csv")
    def test_read_metrics_file_no_metric(self, mock_read_csv, mock_path_exists):
        # Arrange
        metrics_file_path = "metrics_file.csv"
        mock_path_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame(
            {
                "": ["Description", 1, 2, 3, 4, 5],
                "Total Damage Event": ["Total of the events", 100, 200, 300, 400, 500],
                "Other metric": ["Put", "whatever", "you", "want", "to", "hai"],
            },
            columns=["", "Total Damage Event", "Other metric"],
        ).set_index("")

        # Act
        read_class = MetricsFileReader(metrics_file_path)

        # Assert
        with self.assertRaises(KeyError) as context:
            read_class.read_aggregated_metric_from_file(metric="Bullocks metric name")
            self.assertTrue(
                "The metric Bullocks metric name is not found in the file."
                in str(context.exception)
            )

    @patch("fiat_toolbox.metrics_writer.fiat_read_metrics_file.pd.read_csv")
    @patch("fiat_toolbox.metrics_writer.fiat_read_metrics_file.os.path.exists")
    def test_read_metrics_file_no_aggregates(self, mock_path_exists, mock_read_csv):
        # Arrange
        mock_path_exists.return_value = True
        mock_read_csv.return_value = (
            pd.DataFrame(
                {
                    "": ["Name1", "Name2", "Name3", "Name4", "Name5"],
                    "Long Name": [
                        "Long Name1",
                        "Long Name2",
                        "Long Name3",
                        "Long Name4",
                        "Long Name5",
                    ],
                    "Show In Metrics Table": [
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    "Description": [
                        "Description1",
                        "Description2",
                        "Description3",
                        "Description4",
                        "Description5",
                    ],
                    "Value": [1, 2, 3, 4, 5],
                }
            )
            .set_index("")
            .transpose()
        )

        metrics_file_path = "metrics_file.csv"

        # Act
        read_class = MetricsFileReader(metrics_file_path)
        df_results = read_class.read_metrics_from_file().to_dict()["Value"]

        # Assert
        df_expected = {
            "Name1": 1,
            "Name2": 2,
            "Name3": 3,
            "Name4": 4,
            "Name5": 5,
        }
        self.assertEqual(df_results, df_expected)
