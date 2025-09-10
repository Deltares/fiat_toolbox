import base64
import logging
from pathlib import Path
from typing import Dict, List, Union

from plotly.graph_objects import Figure

from fiat_toolbox.infographics.infographics import InfographicsParser
from fiat_toolbox.infographics.infographics_interface import IInfographicsParser
from fiat_toolbox.metrics_writer.fiat_read_metrics_file import (
    MetricsFileReader,
)


class RiskInfographicsParser(IInfographicsParser):
    """Class for creating the infographic"""

    logger: logging.Logger = (logging.getLogger(__name__),)

    def __init__(
        self,
        scenario_name: str,
        metrics_full_path: Union[Path, str],
        config_base_path: Union[Path, str],
        output_base_path: Union[Path, str],
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """Initialize the InfographicsParser

        Parameters
        ----------
        scenario_name : str
            The name of the scenario
        metrics_full_path : Union[Path, str]
            The path to the metrics file
        config_base_path : Union[Path, str]
            The path to the config folder
        output_base_path : Union[Path, str]
            The path to the output folder
        """

        # Save the scenario name
        self.scenario_name = scenario_name

        # Convert the metrics path to a Path object
        if isinstance(metrics_full_path, str):
            metrics_full_path = Path(metrics_full_path)
        self.metrics_full_path = metrics_full_path

        # Convert the config path to a Path object
        if isinstance(config_base_path, str):
            config_base_path = Path(config_base_path)
        self.config_base_path = config_base_path

        # Convert the output path to a Path object
        if isinstance(output_base_path, str):
            output_base_path = Path(output_base_path)
        self.output_base_path = output_base_path
        self.logger = logger

    def _get_impact_metrics(self) -> Dict:
        """Get the impact metrics for a scenario

        Returns
        -------
        Dict
            The impact metrics for the scenario
        """

        # Check if the metrics file exists
        if not Path.exists(self.metrics_full_path):
            raise FileNotFoundError(
                f"Metrics file not found at {self.metrics_full_path}"
            )

        # Read configured metrics
        metrics = (
            MetricsFileReader(self.metrics_full_path)
            .read_metrics_from_file()
            .to_dict()["Value"]
        )

        # Return the metrics
        return metrics

    @staticmethod
    def _encode_image_from_path(image_path: str) -> str:
        """Encode an image from a path to a base64 string

        Parameters
        ----------
        image_path : str
            The path to the image

        Returns
        -------
        str
            The base64 encoded image string
        """
        path = Path(image_path)
        if not Path.exists(path):
            RiskInfographicsParser.logger.error(f"Image not found at {path}")
            return
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        return f"data:image/png;base64,{encoded_string}"

    @staticmethod
    def _figures_list_to_html(
        rp_fig: Figure,
        metrics: Dict,
        charts: Dict,
        file_path: Union[str, Path] = "infographics.html",
        image_folder_path: Union[str, Path] = None,
    ):
        """Save a list of plotly figures in an HTML file

        Parameters
        ----------
            rp_fig : Figure
                The plotly figure consisting of the pie charts for multiple return periods
            metrics : Dict
                The impact metrics for the scenario
            file_path : Union[str, Path], optional
                Path to the HTML file, by default "infographics.html"
            image_path : Union[str, Path], optional
                Path to the image folder, by default None
        """
        # Convert the file_path to a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Check if the file_path already exists
        if Path.exists(file_path):
            raise FileExistsError(f"File already exists at {file_path}")

        # Check if the file_path is correct
        if file_path.suffix != ".html":
            raise ValueError(f"File path must be a .html file, not {file_path}")

        # Create the directory if it does not exist
        if not Path.exists(file_path.parent):
            file_path.parent.mkdir(parents=True)

        # Check if the image_path exists
        expected_damage_path = InfographicsParser._check_image_source(
            charts["Other"]["Expected_Damages"]["image"],
            image_folder_path,
            return_image=False,
        )
        flooded_path = InfographicsParser._check_image_source(
            charts["Other"]["Flooded"]["image"], image_folder_path, return_image=False
        )

        # Div height is the max of the chart heights
        div_height = max(
            charts["Other"]["Expected_Damages"]["height"],
            charts["Other"]["Flooded"]["height"],
            charts["Other"]["Return_Periods"]["plot_height"],
        )

        # Write the html to the file
        with open(file_path, mode="w", encoding="utf-8") as infographics:
            rp_charts = (
                rp_fig.to_html(config={"displayModeBar": False})
                .split("<body>")[1]
                .split("</body>")[0]
            )

            infographics.write(
                f"""
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
                            max-height: {div_height}px; /* Add your max height here */
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
                            transform: scale({charts['Other']['Expected_Damages']['image_scale']}); /* Add your scale factor here */
                        }}
                        .img-container2 {{
                            max-width: 10%;
                            height: auto;
                            margin: 0 auto;
                            transform: scale({charts['Other']['Flooded']['image_scale']}); /* Add your scale factor here */
                        }}
                        h1 {{
                            font-size: {charts['Other']['Expected_Damages']['title_font_size']}px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                            font-weight:normal;
                        }}
                        h2 {{
                            font-size: {charts['Other']['Flooded']['title_font_size']}px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                            font-weight:normal;
                        }}
                        p1 {{
                            font-size: {charts['Other']['Flooded']['numbers_font_size']}px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                        }}
                        p2 {{
                            font-size: {charts['Other']['Flooded']['numbers_font_size']}px; /* Adjust the font size as needed */
                            font-family: Verdana; /* Specify the font family as Verdana */
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="inner-div">
                            <h1>{charts['Other']['Expected_Damages']['title']}</h1>
                            <img src="{RiskInfographicsParser._encode_image_from_path(expected_damage_path)}" alt="Expected Damage" class="img-container1">
                            <p1>${'{:,.0f}'.format(metrics[charts['Other']['Expected_Damages']['query']])}</p1>
                        </div>
                        <div class="inner-div">
                            <h2>{charts['Other']['Flooded']['title']}</h2>
                            <img src="{RiskInfographicsParser._encode_image_from_path(flooded_path)}" alt="Flooded Homes" class="img-container2">
                            <p2>{'{:,.0f}'.format(metrics[charts['Other']['Flooded']['query']])}</p2>
                        </div>
                        <div class="inner-div chart-container">
                            {rp_charts}
                        </div>
                    </div>
                </body>
                </html>
                """
            )

    def _get_infographics(
        self,
    ) -> Union[Dict, Dict, Figure]:
        """Get the infographic for a scenario

        Returns
        -------
        Figure
            The infographic for the scenario

        """

        # Get the impact metrics
        metrics = self._get_impact_metrics()

        # Get the infographic configuration
        pie_chart_config_path = self.config_base_path.joinpath(
            "config_risk_charts.toml"
        )

        # Check if the infographic configuration files exist
        if not Path.exists(pie_chart_config_path):
            raise FileNotFoundError(
                f"Infographic configuration file not found at {pie_chart_config_path}"
            )

        # Get the pie chart dictionaries
        charts = InfographicsParser._get_pies_dictionary(pie_chart_config_path, metrics)

        # Create the pie chart figures
        charts_fig = InfographicsParser._get_pie_chart_figure(
            data=charts.copy(),
            legend_orientation="h",
            yanchor="top",
            y=-0.1,
            title=charts["Other"]["Return_Periods"]["title"],
            image_path=self.config_base_path.joinpath("images"),
            title_font_size=charts["Other"]["Return_Periods"]["font_size"],
            subtitle_font_size=charts["Other"]["Return_Periods"]["subtitle_font"],
            image_scale=charts["Other"]["Return_Periods"]["image_scale"],
            numbers_font=charts["Other"]["Return_Periods"]["numbers_font"],
            legend_font_size=charts["Other"]["Return_Periods"]["legend_font"],
            plot_info=charts["Other"]["Info"]["text"],
            plot_info_img=charts["Other"]["Info"]["image"],
            plot_info_scale=charts["Other"]["Info"]["scale"],
            plot_height=charts["Other"]["Return_Periods"]["plot_height"],
        )

        # Return the figure
        return metrics, charts, charts_fig

    def get_infographics(self) -> Union[List[Figure], Figure]:
        """Get the infographic for a scenario

        Returns
        -------
        Union[List[Figure], Figure]
            The infographic for the scenario as a list of figures or a single figure
        """

        # Get the infographic
        _, _, infographic = self._get_infographics()

        # Return the infographic
        return infographic

    def write_infographics_to_file(self) -> str:
        """Write the infographic for a scenario to file

        Returns
        -------
        str
            The path to the infographic file
        """

        # Create the infographic path
        infographic_html = self.output_base_path.joinpath(
            f"{self.scenario_name}_metrics.html"
        )

        # Check if the infographic already exists. If so, return the path
        if Path.exists(infographic_html):
            RiskInfographicsParser.logger.info(
                f"Infographic already exists, skipping creation. Path: {infographic_html}"
            )
            return str(infographic_html)

        # Get the infographic
        metrics, charts, infographic = self._get_infographics()

        # Convert the infographic to html. The default for using relative image paths is to have an images folder in the same directory as the config files
        self._figures_list_to_html(
            infographic,
            metrics,
            charts,
            infographic_html,
            self.config_base_path.joinpath("images"),
        )

        # Return the path to the infographic
        return str(infographic_html)

    def get_infographics_html(self) -> str:
        """Get the path to the infographic html file

        Returns
        -------
        str
            The path to the infographic html file
        """

        # Create the infographic path
        infographic_path = self.output_base_path.joinpath(
            f"{self.scenario_name}_metrics.html"
        )

        return str(infographic_path)
