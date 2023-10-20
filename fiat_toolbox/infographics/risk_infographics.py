from pathlib import Path
from typing import Dict, List, Union

import plotly.graph_objects as go

from fiat_toolbox.infographics.infographics import InfographicsParser
from fiat_toolbox.infographics.infographics_interface import IInfographicsParser
from fiat_toolbox.metrics_writer.fiat_read_metrics_file import (
    MetricsFileReader,
)


class RiskInfographicsParser(IInfographicsParser):
    """Class for creating the infographic"""

    def __init__(
        self,
        scenario_name: str,
        metrics_full_path: Union[Path, str],
        config_base_path: Union[Path, str],
        output_base_path: Union[Path, str],
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
    def _figures_list_to_html(
        rp_fig: go.Figure,
        metrics: Dict,
        charts: Dict,
        file_path: Union[str, Path] = "infographics.html",
    ):
        """Save a list of plotly figures in an HTML file

        Parameters
        ----------
            rp_fig : go.Figure
                The plotly figure consisting of the pie charts for multiple return periods
            metrics : Dict
                The impact metrics for the scenario
            file_path : Union[str, Path], optional
                Path to the HTML file, by default "infographics.html"
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

        # Write the html to the file
        with open(file_path, "w", encoding="utf-8") as infographics:
            rp_charts = rp_fig.to_html().split("<body>")[1].split("</body>")[0]

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
                            justify-content: space-between;
                            height: 100vh;
                        }}
                        .inner-div {{
                            text-align: center;
                        }}
                        .img-container {{
                            max-width: 10%;
                            height: auto;
                            margin: 0 auto;
                        }}
                        .chart-container {{
                            /* Add your CSS styling for chart container here */
                        }}
                        h1 {{
                            font-size: 25px; /* Adjust the font size as needed */
                            font-family: Verdana, sans-serif; /* Specify the font family as Verdana */
                        }}
                        p {{
                            font-size: 20px; /* Adjust the font size as needed */
                            font-family: Verdana, sans-serif; /* Specify the font family as Verdana */
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="inner-div">
                            <h1>Expected annual damages</h1>
                            <img src="{charts['Other']['expected_damage_image']}" alt="Expected Damage" class="img-container">
                            <p>${'{:,.0f}'.format(metrics['ExpectedAnnualDamages'])}</p>
                        </div>
                        <div class="inner-div">
                            <h1>{charts['Other']['flooded_title']}</h1>
                            <img src="{charts['Other']['flooded_image']}" alt="Flooded Homes" class="img-container">
                            <p>{'{:,.0f}'.format(metrics['FloodedHomes'])}</p>
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
    ) -> Union[Dict, Dict, go.Figure]:
        """Get the infographic for a scenario

        Returns
        -------
        go.Figure
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
            title="Building damage",
        )

        # Return the figure
        return metrics, charts, charts_fig

    def get_infographics(self) -> Union[List[go.Figure], go.Figure]:
        """Get the infographic for a scenario

        Returns
        -------
        Union[List[go.Figure], go.Figure]
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
            # TODO: Print logging message
            return str(infographic_html)

        # Get the infographic
        metrics, charts, infographic = self._get_infographics()

        # Convert the infographic to html
        self._figures_list_to_html(infographic, metrics, charts, infographic_html)

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
