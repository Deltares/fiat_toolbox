import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go

from fiat_toolbox.infographics.infographics import InfographicsParser


class TestInfographicsParserGetMetrics(unittest.TestCase):
    # TODO: These tests should be extended with integration tests where you are testing on actual data. Before this can be done, a standard database should be created with all the necessary data.

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("fiat_toolbox.infographics.infographics.MetricsFileReader")
    def test_get_impact_metrics(
        self,
        mock_metrics_file_reader,
        mock_path_exists,
    ):
        # Arrange
        mock_path_exists.return_value = True

        mock_reader = mock_metrics_file_reader.return_value
        mock_reader.read_metrics_from_file.return_value = pd.DataFrame(
            {"Value": [1, 2, 3]}
        )

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="metrics_path.csv",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )
        df_results = parser._get_impact_metrics()

        # Assert
        self.assertEqual(df_results, {0: 1, 1: 2, 2: 3})
        self.assertEqual(mock_path_exists.call_count, 1)
        self.assertEqual(
            str(mock_path_exists.call_args_list[0][0][0]),
            "metrics_path.csv",
        )

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("fiat_toolbox.infographics.infographics.MetricsFileReader")
    def test_get_impact_metrics_no_file(
        self,
        mock_metrics_file_reader,
        mock_path_exists,
    ):
        # Arrange
        mock_path_exists.return_value = False

        mock_reader = mock_metrics_file_reader.return_value
        mock_reader.read_metrics_from_file.return_value = {"test": [1, 2, 3]}

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="metrics_path.csv",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(FileNotFoundError) as context:
            _ = parser._get_impact_metrics()

        self.assertTrue(
            "Metrics file not found at metrics_path.csv" in str(context.exception)
        )
        self.assertEqual(mock_path_exists.call_count, 1)
        self.assertEqual(
            str(mock_path_exists.call_args_list[0][0][0]),
            "metrics_path.csv",
        )


class TestInfographicsParserPiesDictionary(unittest.TestCase):
    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("builtins.open")
    @patch("fiat_toolbox.infographics.infographics.tomli.load")
    def test_get_pies_dict(self, mock_tomli_load, mock_open, mock_path_exists):
        # Arrange
        path = "some_config_path"
        mock_open.return_value.__enter__.return_value = "some_config"
        mock_path_exists.return_value = True
        mock_tomli_load.return_value = {
            "Charts": {
                "testchart": {"Name": "testpie", "Image": "test.png"},
                "testchart2": {"Name": "testpie2", "Image": "test2.png"},
            },
            "Categories": {
                "testcategory": {"Name": "testcat", "Color": "red"},
                "testcategory2": {"Name": "testcat2", "Color": "blue"},
            },
            "Slices": {
                "testslice": {
                    "Name": "test",
                    "Query": "test_query",
                    "Category": "testcat",
                    "Chart": "testpie",
                },
                "testslice2": {
                    "Name": "test2",
                    "Query": "test_query2",
                    "Category": "testcat2",
                    "Chart": "testpie",
                },
                "testslice3": {
                    "Name": "test3",
                    "Query": "test_query3",
                    "Category": "testcat",
                    "Chart": "testpie2",
                },
                "testslice4": {
                    "Name": "test4",
                    "Query": "test_query4",
                    "Category": "testcat2",
                    "Chart": "testpie2",
                },
            },
        }

        metrics = {
            "test_query": 1,
            "test_query2": 2,
            "test_query3": 3,
            "test_query4": 4,
        }

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        pie_dict = parser._get_pies_dictionary(path, metrics)

        # Assert
        expected_dict = {
            "testpie": {
                "Name": "testpie",
                "Image": "test.png",
                "Values": [1, 2],
                "Colors": ["red", "blue"],
                "Labels": ["testcat", "testcat2"],
            },
            "testpie2": {
                "Name": "testpie2",
                "Image": "test2.png",
                "Values": [3, 4],
                "Colors": ["red", "blue"],
                "Labels": ["testcat", "testcat2"],
            },
        }

        self.assertEqual(pie_dict, expected_dict)
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(mock_tomli_load.call_count, 1)
        self.assertEqual(mock_path_exists.call_count, 1)
        self.assertEqual(
            str(mock_path_exists.call_args_list[0][0][0]), "some_config_path"
        )

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("builtins.open")
    @patch("fiat_toolbox.infographics.infographics.tomli.load")
    def test_get_pies_dict_no_config(
        self, mock_tomli_load, mock_open, mock_path_exists
    ):
        # Arrange
        path = "some_config_path"
        mock_open.return_value.__enter__.return_value = "some_config"
        mock_path_exists.return_value = False

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(FileNotFoundError) as context:
            _ = parser._get_pies_dictionary(path, {})

        self.assertTrue(
            "Infographic configuration file not found at some_config_path"
            in str(context.exception)
        )
        self.assertEqual(mock_open.call_count, 0)
        self.assertEqual(mock_tomli_load.call_count, 0)
        self.assertEqual(mock_path_exists.call_count, 1)
        self.assertEqual(
            str(mock_path_exists.call_args_list[0][0][0]), "some_config_path"
        )

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("builtins.open")
    @patch("fiat_toolbox.infographics.infographics.tomli.load")
    def test_get_pies_dict_no_charts(
        self, mock_tomli_load, mock_open, mock_path_exists
    ):
        # Arrange
        path = "some_config_path"
        mock_open.return_value.__enter__.return_value = "some_config"
        mock_path_exists.return_value = True
        mock_tomli_load.return_value = {
            "Categories": {
                "testcategory": {"Name": "testcat", "Color": "red"},
                "testcategory2": {"Name": "testcat2", "Color": "blue"},
            },
            "Slices": {
                "testslice": {
                    "Name": "test",
                    "Query": "test_query",
                    "Category": "testcat",
                    "Chart": "testpie",
                },
                "testslice2": {
                    "Name": "test2",
                    "Query": "test_query2",
                    "Category": "testcat2",
                    "Chart": "testpie",
                },
                "testslice3": {
                    "Name": "test3",
                    "Query": "test_query3",
                    "Category": "testcat",
                    "Chart": "testpie2",
                },
                "testslice4": {
                    "Name": "test4",
                    "Query": "test_query4",
                    "Category": "testcat2",
                    "Chart": "testpie2",
                },
            },
        }

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(KeyError) as context:
            _ = parser._get_pies_dictionary(path, {})

        self.assertTrue(
            "Charts not found in pie chart configuration file" in str(context.exception)
        )
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(mock_tomli_load.call_count, 1)
        self.assertEqual(mock_path_exists.call_count, 1)
        self.assertEqual(
            str(mock_path_exists.call_args_list[0][0][0]), "some_config_path"
        )

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("builtins.open")
    @patch("fiat_toolbox.infographics.infographics.tomli.load")
    def test_get_pies_dict_no_categories(
        self, mock_tomli_load, mock_open, mock_path_exists
    ):
        # Arrange
        path = "some_config_path"
        mock_open.return_value.__enter__.return_value = "some_config"
        mock_path_exists.return_value = True
        mock_tomli_load.return_value = {
            "Charts": {
                "testchart": {"Name": "testpie", "Image": "test.png"},
                "testchart2": {"Name": "testpie2", "Image": "test2.png"},
            },
            "Slices": {
                "testslice": {
                    "Name": "test",
                    "Query": "test_query",
                    "Category": "testcat",
                    "Chart": "testpie",
                },
                "testslice2": {
                    "Name": "test2",
                    "Query": "test_query2",
                    "Category": "testcat2",
                    "Chart": "testpie",
                },
                "testslice3": {
                    "Name": "test3",
                    "Query": "test_query3",
                    "Category": "testcat",
                    "Chart": "testpie2",
                },
                "testslice4": {
                    "Name": "test4",
                    "Query": "test_query4",
                    "Category": "testcat2",
                    "Chart": "testpie2",
                },
            },
        }

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(KeyError) as context:
            _ = parser._get_pies_dictionary(path, {})

        self.assertTrue(
            "Categories not found in pie chart configuration file"
            in str(context.exception)
        )
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(mock_tomli_load.call_count, 1)
        self.assertEqual(mock_path_exists.call_count, 1)
        self.assertEqual(
            str(mock_path_exists.call_args_list[0][0][0]), "some_config_path"
        )

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("builtins.open")
    @patch("fiat_toolbox.infographics.infographics.tomli.load")
    def test_get_pies_dict_no_slices(
        self, mock_tomli_load, mock_open, mock_path_exists
    ):
        # Arrange
        path = "some_config_path"
        mock_open.return_value.__enter__.return_value = "some_config"
        mock_path_exists.return_value = True
        mock_tomli_load.return_value = {
            "Charts": {
                "testchart": {"Name": "testpie", "Image": "test.png"},
                "testchart2": {"Name": "testpie2", "Image": "test2.png"},
            },
            "Categories": {
                "testcategory": {"Name": "testcat", "Color": "red"},
                "testcategory2": {"Name": "testcat2", "Color": "blue"},
            },
        }

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(KeyError) as context:
            _ = parser._get_pies_dictionary(path, {})

        self.assertTrue(
            "Slices not found in pie chart configuration file" in str(context.exception)
        )
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(mock_tomli_load.call_count, 1)
        self.assertEqual(mock_path_exists.call_count, 1)
        self.assertEqual(
            str(mock_path_exists.call_args_list[0][0][0]), "some_config_path"
        )


class TestInfographicsParserChartsFigure(unittest.TestCase):
    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("fiat_toolbox.infographics.infographics.go.Figure.to_html")
    @patch("builtins.open")
    def test_figure_to_html(self, mock_open, mock_to_html, mock_path_exists):
        # Arrange
        figure_path = Path("parent/some_figure.html")
        styles_path = "styles.css"
        mock_file = mock_open.return_value.__enter__.return_value

        def exists_side_effect(path):
            if ".html" in str(path):
                # In case of the html file, we want it to not exist
                return False
            else:
                return True

        mock_path_exists.side_effect = exists_side_effect
        mock_to_html.return_value = "<body>some_figure</body>"
        figs = [go.Figure(), go.Figure(), go.Figure()]

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )
        parser._figures_list_to_html(figs, figure_path, styles_path)

        # Assert
        expected_html = f"""<!DOCTYPE html>
                    <html>
                        <head>
                            <link rel="stylesheet" type="text/css" href="{styles_path}">
                        </head>
                        <body>
                            <div class="container">
                                <div class="top-half">
                                    some_figure
                                </div>
                                <div class="bottom-left">
                                    some_figure
                                </div>
                                <div class="bottom-right">
                                    some_figure
                                </div>
                            </div>
                        </body>
                    </html>"""

        # Tabs and spaces are removed to make the comparison easier
        self.assertEqual(
            mock_file.write.call_args[0][0].replace(" ", ""),
            expected_html.replace(" ", ""),
        )
        self.assertEqual(mock_file.write.call_count, 1)
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(mock_to_html.call_count, 3)
        self.assertEqual(mock_path_exists.call_count, 3)
        self.assertEqual(str(mock_path_exists.call_args_list[0][0][0]), styles_path)
        self.assertEqual(
            str(mock_path_exists.call_args_list[1][0][0]), str(figure_path)
        )
        self.assertEqual(
            str(mock_path_exists.call_args_list[2][0][0]), str(figure_path.parent)
        )

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("fiat_toolbox.infographics.infographics.go.Figure.to_html")
    @patch("builtins.open")
    def test_figure_to_html_no_figures(self, mock_open, mock_to_html, mock_path_exists):
        # Arrange
        figure_path = Path("parent/some_figure.html")
        styles_path = "styles.css"

        mock_file = mock_open.return_value.__enter__.return_value

        def exists_side_effect(path):
            if ".html" in str(path):
                # In case of the html file, we want it to not exist
                return False
            else:
                return True

        mock_path_exists.side_effect = exists_side_effect

        mock_to_html.return_value = "<body>some_figure</body>"

        figs = []

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )
        parser._figures_list_to_html(figs, figure_path, styles_path)

        # Assert
        expected_html = f"""<!DOCTYPE html>
                    <html>
                        <head>
                            <link rel="stylesheet" type="text/css" href="{styles_path}">
                        </head>
                        <body>
                            <div class="container">
                                <div class="top-half">

                                </div>
                                <div class="bottom-left">

                                </div>
                                <div class="bottom-right">

                                </div>
                            </div>
                        </body>
                    </html>"""

        # Tabs and spaces are removed to make the comparison easier
        self.assertEqual(
            mock_file.write.call_args[0][0].replace(" ", ""),
            expected_html.replace(" ", ""),
        )
        self.assertEqual(mock_file.write.call_count, 1)
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(mock_to_html.call_count, 0)
        self.assertEqual(mock_path_exists.call_count, 3)
        self.assertEqual(str(mock_path_exists.call_args_list[0][0][0]), styles_path)
        self.assertEqual(
            str(mock_path_exists.call_args_list[1][0][0]), str(figure_path)
        )
        self.assertEqual(
            str(mock_path_exists.call_args_list[2][0][0]), str(figure_path.parent)
        )

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    def test_figure_to_html_no_styles(self, mock_path_exists):
        # Arrange
        figure_path = "some_figure.html"
        styles_path = "styles.css"
        mock_path_exists.return_value = False

        figs = [go.Figure(), go.Figure(), go.Figure()]

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(FileNotFoundError) as context:
            parser._figures_list_to_html(figs, figure_path, styles_path)

        self.assertTrue("Stylesheet not found at styles.css" in str(context.exception))

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    def test_figure_to_html_wrong_css_suffix(self, mock_path_exists):
        # Arrange
        figure_path = "some_figure.html"
        styles_path = "styles.txt"

        def exists_side_effect(path):
            if ".html" in str(path):
                # In case of the html file, we want it to not exist
                return False
            else:
                return True

        mock_path_exists.side_effect = exists_side_effect
        figs = [go.Figure(), go.Figure(), go.Figure()]

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(ValueError) as context:
            parser._figures_list_to_html(figs, figure_path, styles_path)

        self.assertTrue(
            "Stylesheet must be a .css file, not styles.txt" in str(context.exception)
        )

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    def test_html_already_exists(self, mock_path_exists):
        # Arrange
        figure_path = "some_figure.html"
        styles_path = "styles.css"
        mock_path_exists.return_value = True
        figs = [go.Figure(), go.Figure(), go.Figure()]

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(FileExistsError) as context:
            parser._figures_list_to_html(figs, figure_path, styles_path)

        self.assertTrue(
            "File already exists at some_figure.html" in str(context.exception)
        )

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    def test_html_wrong_suffix(self, mock_path_exists):
        # Arrange
        figure_path = "some_figure.txt"
        styles_path = "styles.css"

        def exists_side_effect(path):
            if ".txt" in str(path):
                # In case of the txt file, we want it to not exist
                return False
            else:
                return True

        mock_path_exists.side_effect = exists_side_effect
        figs = [go.Figure(), go.Figure(), go.Figure()]

        # Act
        parser = InfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(ValueError) as context:
            parser._figures_list_to_html(figs, figure_path, styles_path)

        self.assertTrue(
            "File path must be a .html file, not some_figure.txt"
            in str(context.exception)
        )
