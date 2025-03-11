import base64
import io
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
from plotly.graph_objects import Figure

from fiat_toolbox.infographics.risk_infographics import RiskInfographicsParser


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
    money_bin = b"fake_money_image_data"
    house_bin = b"fake_house_image_data"
    money_path = "money.png"
    house_path = "house.png"

    @patch("fiat_toolbox.infographics.risk_infographics.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_encode_image_from_path(self, mock_open, mock_path_exists):
        # Arrange
        mock_path_exists.return_value = True
        mock_open.return_value.read.return_value = self.money_bin

        # Act
        encoded_image = RiskInfographicsParser._encode_image_from_path(self.money_path)

        # Assert
        expected_encoded_string = (
            f"data:image/png;base64,{base64.b64encode(self.money_bin).decode()}"
        )
        assert encoded_image == expected_encoded_string
        mock_open.assert_called_once_with(Path(self.money_path), "rb")
        mock_path_exists.assert_called_once_with(Path(self.money_path))

    @patch("fiat_toolbox.infographics.infographics.Path.exists")
    @patch("fiat_toolbox.infographics.infographics.Image.open")
    @patch("fiat_toolbox.infographics.risk_infographics.Path.exists")
    @patch("fiat_toolbox.infographics.risk_infographics.Figure.to_html")
    @patch("builtins.open", new_callable=mock_open)
    def test_figure_to_html(
        self,
        mock_open,
        mock_to_html,
        mock_open_image,
        mock_path_exists,
        mock_path_exists_infographics,
    ):
        # Arrange
        figure_path = Path("parent/some_figure.html")
        mock_open_image.return_value = "some_image"

        def exists_side_effect(path):
            if ".html" in str(path):
                # In case of the html file, we want it to not exist
                return False
            else:
                return True

        mock_path_exists_infographics.side_effect = exists_side_effect
        mock_path_exists.side_effect = exists_side_effect
        mock_to_html.return_value = "<body>some_figure</body>"

        def mock_open_side_effect(file_path, mode="r", encoding=None):
            file = str(file_path)
            if "r" in mode:
                if "money.png" in file:
                    return io.BytesIO(self.money_bin)
                elif "house.png" in file:
                    return io.BytesIO(self.house_bin)
                    # return mock_open(read_data=house_bin).return_value
            else:
                return mock_open.return_value

        mock_open.side_effect = mock_open_side_effect
        mock_file = mock_open.return_value.__enter__.return_value

        figs = Figure()

        metrics = {"ExpectedAnnualDamages": 1000000, "FloodedHomes": 1000}
        charts = {
            "Other": {
                "Expected_Damages": {
                    "title": "Expected annual damages",
                    "image": "money.png",
                    "image_scale": 0.125,
                    "title_font_size": 30,
                    "numbers_font_size": 15,
                    "height": 300,
                },
                "Flooded": {
                    "title": "Number of homes with a high chance of being flooded in a 30-year period",
                    "image": "house.png",
                    "image_scale": 0.125,
                    "title_font_size": 30,
                    "numbers_font_size": 15,
                    "height": 300,
                },
                "Return_Periods": {
                    "title": "Building damages",
                    "font_size": 30,
                    "image_scale": 0.125,
                    "numbers_font": 15,
                    "subtitle_font": 25,
                    "legend_font": 20,
                    "plot_height": 300,
                },
                "Info": {
                    "title": "Building damages",
                    "image": "house.png",
                    "scale": 0.125,
                },
            }
        }

        # Act
        parser = RiskInfographicsParser(
            scenario_name="test_scenario",
            metrics_full_path="DontCare",
            config_base_path="DontCare",
            output_base_path="DontCare",
        )

        parser._figures_list_to_html(
            rp_fig=figs, metrics=metrics, charts=charts, file_path=figure_path
        )

        # Assert
        expected_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title></title>
                    <style>
                        .container {{
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            justify-content: center;  # Center the plots vertically
                            height: 100vh;
                        }}
                        .inner-div {{
                            text-align: center;
                            max-height: 300px; /* Add your max height here */
                            overflow: auto; /* Add this to handle content that exceeds the max height */
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                        }}
                        .chart-container {{
                            /* Add your CSS styling for chart container here */
                        }}
                        .img-container1 {{
                            max-width: 10%;
                            height: auto;
                            margin: 0 auto;
                            transform: scale(0.125); /* Add your scale factor here */
                        }}
                        .img-container2 {{
                            max-width: 10%;
                            height: auto;
                            margin: 0 auto;
                            transform: scale(0.125); /* Add your scale factor here */
                        }}
                        h1 {{
                            font-size:  30px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                            font-weight:normal;
                        }}
                        h2 {{
                            font-size: 30px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                            font-weight:normal;
                        }}
                        p1 {{
                            font-size: 15px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                        }}
                        p2 {{
                            font-size: 15px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="inner-div">
                            <h1>Expected annual damages</h1>
                            <img src="data:image/png;base64,{base64.b64encode(self.money_bin).decode()}" alt="Expected Damage" class="img-container1">
                            <p1>$1,000,000</p1>
                        </div>
                        <div class="inner-div">
                            <h2>Number of homes with a high chance of being flooded in a 30-year period</h2>
                            <img src="data:image/png;base64,{base64.b64encode(self.house_bin).decode()}" alt="Flooded Homes" class="img-container2">
                            <p2>1,000</p2>
                        </div>
                        <div class="inner-div chart-container">
                            some_figure
                        </div>
                    </div>
                </body>
                </html>
                """
        self.maxDiff = 10000
        # Tabs and spaces are removed to make the comparison easier
        self.assertEqual(
            mock_file.write.call_args[0][0].replace(" ", ""),
            expected_html.replace(" ", ""),
        )
        self.assertEqual(mock_file.write.call_count, 1)
        self.assertEqual(mock_open.call_count, 3)  # 2 images and 1 html file
        self.assertEqual(mock_to_html.call_count, 1)
        self.assertEqual(mock_path_exists_infographics.call_count, 6)

    @patch("fiat_toolbox.infographics.risk_infographics.Path.exists")
    @patch("fiat_toolbox.infographics.infographics.Image.open")
    @patch("fiat_toolbox.infographics.risk_infographics.Figure.to_html")
    @patch("builtins.open")
    def test_figure_to_html_no_figures(
        self, mock_open, mock_to_html, mock_open_image, mock_path_exists
    ):
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
                    "image_scale": 0.125,
                    "title_font_size": 30,
                    "numbers_font_size": 15,
                    "height": 300,
                },
                "Flooded": {
                    "title": "Number of homes with a high chance of being flooded in a 30-year period",
                    "image": "house.png",
                    "image_scale": 0.125,
                    "title_font_size": 30,
                    "numbers_font_size": 15,
                    "height": 300,
                },
                "Return_Periods": {
                    "title": "Building damages",
                    "font_size": 30,
                    "image_scale": 0.125,
                    "numbers_font": 15,
                    "subtitle_font": 25,
                    "legend_font": 20,
                    "plot_height": 300,
                },
                "Info": {
                    "title": "Building damages",
                    "image": "house.png",
                    "scale": 0.125,
                },
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
        figs = [Figure(), Figure(), Figure()]
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
        figs = [Figure(), Figure(), Figure()]
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
