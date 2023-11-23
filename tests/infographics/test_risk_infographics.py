import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go

from fiat_toolbox.infographics.risk_infographics import RiskInfographicsParser

# Setup method to create a RiskInfographicsParser object
# def setup_method(self):
#     cwd = Path(os.path.dirname(os.path.abspath(__file__)))
#     self.metrics_full_path = cwd.joinpath("data", "risk", "test_scenario_metrics.csv")
#     self.config_base_path = cwd.joinpath("data", "risk")
#     self.output_base_path = cwd.joinpath("data", "risk")

#     self.standard_parser = RiskInfographicsParser(
#         scenario_name="test_scenario",
#         metrics_full_path=self.metrics_full_path,
#         config_base_path=self.config_base_path,
#         output_base_path=self.output_base_path,


class TestRiskInfographicsParserGetMetrics(unittest.TestCase):
    # TODO: These tests should be extended with integration tests where you are testing on actual data. Before this can be done, a standard database should be created with all the necessary data.

    @patch("fiat_toolbox.infographics.risk_infographics.Path.exists")
    @patch("fiat_toolbox.infographics.risk_infographics.MetricsFileReader")
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
        parser = RiskInfographicsParser(
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

    @patch("fiat_toolbox.infographics.risk_infographics.Path.exists")
    @patch("fiat_toolbox.infographics.risk_infographics.MetricsFileReader")
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
        parser = RiskInfographicsParser(
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


class TestRiskInfographicsParserChartsFigure(unittest.TestCase):
    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("fiat_toolbox.infographics.infographics.Image.open")
    @patch("fiat_toolbox.infographics.risk_infographics.go.Figure.to_html")
    @patch("builtins.open")
    def test_figure_to_html(self, mock_open, mock_to_html, mock_open_image, mock_path_exists):
        # Arrange
        figure_path = Path("parent/some_figure.html")
        mock_file = mock_open.return_value.__enter__.return_value
        mock_open_image.return_value = "some_image"

        def exists_side_effect(path):
            if ".html" in str(path):
                # In case of the html file, we want it to not exist
                return False
            else:
                return True

        mock_path_exists.side_effect = exists_side_effect
        mock_to_html.return_value = "<body>some_figure</body>"
        figs = go.Figure()

        metrics = {"ExpectedAnnualDamages": 1000000, "FloodedHomes": 1000}
        charts = {
            "Other": {
                "Expected_Damages": {
                    "title": "Expected annual damages",
                    "image": "money.png",
                    "font_size": 30
                },
                "Flooded": {
                    "title": "Number of homes with a high chance of being flooded in a 30-year period",
                    "image": "house.png",
                    "font_size": 30
                },
                "Return_Periods": {
                    "title": "Building damages",
                    "font_size": 30,
                    "image_scale": 0.125,
                    "numbers_font": 15,
                    "subtitle_font": 25,
                    "legend_font": 20
                },
                "Info": {
                    "title": "Building damages",
                    "image": "house.png",
                    "scale": 0.125,
                }
            }
        }

        # Act
        parser = RiskInfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )
        parser._figures_list_to_html(figs, metrics, charts, figure_path)

        # Assert
        expected_html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title></title>
                    <style>
                        .container {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            justify-content: center;  # Center the plots vertically
                            height: 100vh;
                        }
                        .inner-div {
                            text-align: center;
                        }
                        .img-container {
                            max-width: 10%;
                            height: auto;
                            margin: 0 auto;
                        }
                        .chart-container {
                            /* Add your CSS styling for chart container here */
                        }
                        h1 {
                            font-size: 30px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                            font-weight:normal; 
                        }
                        h2 {
                            font-size: 30px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                            font-weight:normal; 
                        }
                        p {
                            font-size: 20px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="inner-div">
                            <h1>Expected annual damages</h1>
                            <img src="money.png" alt="Expected Damage" class="img-container">
                            <p>$1,000,000</p>
                        </div>
                        <div class="inner-div">
                            <h2>Number of homes with a high chance of being flooded in a 30-year period</h2>
                            <img src="house.png" alt="Flooded Homes" class="img-container">
                            <p>1,000</p>
                        </div>
                        <div class="inner-div chart-container">
                            some_figure
                        </div>
                    </div>
                </body>
                </html>
                """

        # Tabs and spaces are removed to make the comparison easier
        self.assertEqual(
            mock_file.write.call_args[0][0].replace(" ", ""),
            expected_html.replace(" ", ""),
        )
        self.assertEqual(mock_file.write.call_count, 1)
        self.assertEqual(mock_open.call_count, 1)
        self.assertEqual(mock_to_html.call_count, 1)
        self.assertEqual(mock_path_exists.call_count, 4)
        self.assertEqual(
            str(mock_path_exists.call_args_list[0][0][0]), str(figure_path)
        )
        self.assertEqual(
            str(mock_path_exists.call_args_list[1][0][0]), str(figure_path.parent)
        )

    @patch("fiat_toolbox.infographics.risk_infographics.Path.exists")
    @patch("fiat_toolbox.infographics.infographics.Image.open")
    @patch("fiat_toolbox.infographics.risk_infographics.go.Figure.to_html")
    @patch("builtins.open")
    def test_figure_to_html_no_figures(self, mock_open, mock_to_html, mock_open_image, mock_path_exists):
        # Arrange
        figure_path = Path("parent/some_figure.html")
        mock_open_image.return_value = "some_image"

        def exists_side_effect(path):
            if ".html" in str(path):
                # In case of the html file, we want it to not exist
                return False
            else:
                return True

        mock_path_exists.side_effect = exists_side_effect

        mock_to_html.return_value = "<body>some_figure</body>"

        figs = []

        metrics = {"ExpectedAnnualDamages": 1000000, "FloodedHomes": 1000}
        charts = {
            "Other": {
                "Expected_Damages": {
                    "title": "Expected annual damages",
                    "image": "money.png",
                    "font_size": 30
                },
                "Flooded": {
                    "title": "Number of homes with a high chance of being flooded in a 30-year period",
                    "image": "house.png",
                    "font_size": 30
                },
                "Return_Periods": {
                    "title": "Building damages",
                    "font_size": 30,
                    "image_scale": 0.125,
                    "numbers_font": 15,
                    "subtitle_font": 25,
                    "legend_font": 20
                },
                "Info": {
                    "title": "Building damages",
                    "image": "house.png",
                    "scale": 0.125,
                }
            }
        }

        # Act
        parser = RiskInfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(AttributeError) as context:
            parser._figures_list_to_html(figs, metrics, charts, figure_path)

        self.assertTrue(
            "'list' object has no attribute 'to_html'" in str(context.exception)
        )

    @patch("fiat_toolbox.infographics.risk_infographics.Path.exists")
    def test_html_already_exists(self, mock_path_exists):
        # Arrange
        figure_path = "some_figure.html"
        mock_path_exists.return_value = True
        figs = [go.Figure(), go.Figure(), go.Figure()]
        metrics = {"ExpectedAnnualDamages": 1000000, "FloodedHomes": 1000}
        charts = {
            "Other": {
                "expected_damage_image": "expected_damage_image.png",
                "flooded_title": "Flooded buildings",
                "flooded_image": "flooded_image.png",
            }
        }

        # Act
        parser = RiskInfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(FileExistsError) as context:
            parser._figures_list_to_html(figs, metrics, charts, figure_path)

        self.assertTrue(
            "File already exists at some_figure.html" in str(context.exception)
        )

    @patch("fiat_toolbox.infographics.risk_infographics.Path.exists")
    def test_html_wrong_suffix(self, mock_path_exists):
        # Arrange
        figure_path = "some_figure.txt"

        def exists_side_effect(path):
            if ".txt" in str(path):
                # In case of the txt file, we want it to not exist
                return False
            else:
                return True

        mock_path_exists.side_effect = exists_side_effect
        figs = [go.Figure(), go.Figure(), go.Figure()]
        metrics = {"ExpectedAnnualDamages": 1000000, "FloodedHomes": 1000}
        charts = {
            "Other": {
                "expected_damage_image": "expected_damage_image.png",
                "flooded_title": "Flooded buildings",
                "flooded_image": "flooded_image.png",
            }
        }

        # Act
        parser = RiskInfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        # Assert
        with self.assertRaises(ValueError) as context:
            parser._figures_list_to_html(figs, metrics, charts, figure_path)

        self.assertTrue(
            "File path must be a .html file, not some_figure.txt"
            in str(context.exception)
        )
