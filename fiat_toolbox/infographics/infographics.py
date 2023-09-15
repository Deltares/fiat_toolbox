from pathlib import Path
from typing import Dict, List, Union

import plotly.graph_objects as go
import tomli
from plotly.subplots import make_subplots

from fiat_toolbox.infographics.infographics_interface import IInfographicsParser
from fiat_toolbox.metrics_writer.fiat_read_metrics_file import MetricsFileReader


class InfographicsParser(IInfographicsParser):
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
    def _get_pies_dictionary(
        pie_chart_config_path: Union[str, Path], metrics: Dict
    ) -> Dict:
        """Get a dictionary which contains the configuration and data for the pie charts

        Parameters
        ----------
        pie_chart_config_path : Union[str, Path]
            The path to the pie chart configuration file
        metrics : Dict
            The impact metrics for the scenario

        Returns
        -------
        Dict
            The dictionary which contains the configuration and data for the pie charts
        """

        # Convert the pie chart configuration path to a Path object
        if isinstance(pie_chart_config_path, str):
            pie_chart_config_path = Path(pie_chart_config_path)

        # Check if the pie chart configuration file exists
        if not Path.exists(pie_chart_config_path):
            raise FileNotFoundError(
                f"Infographic configuration file not found at {pie_chart_config_path}"
            )

        # Initialize the pie chart dictionary
        pie_dict = {}
        with open(pie_chart_config_path, mode="rb") as fc:
            # Read the pie chart configuration
            pie_chart_config = tomli.load(fc)

            # Check if the charts are defined
            if "Charts" not in pie_chart_config:
                raise KeyError("Charts not found in pie chart configuration file")

            # Read the charts configuration
            for key, value in pie_chart_config["Charts"].items():
                pie_dict[value["Name"]] = {}
                pie_dict[value["Name"]]["Name"] = value["Name"]
                pie_dict[value["Name"]]["Image"] = value["Image"]
                pie_dict[value["Name"]]["Values"] = []
                pie_dict[value["Name"]]["Colors"] = []
                pie_dict[value["Name"]]["Labels"] = []

            # Check if the categories are defined
            if "Categories" not in pie_chart_config:
                raise KeyError("Categories not found in pie chart configuration file")

            # Read the categories configuration
            categorie_dict = {}
            for key, value in pie_chart_config["Categories"].items():
                categorie_dict[value["Name"]] = {}
                categorie_dict[value["Name"]]["Name"] = value["Name"]
                categorie_dict[value["Name"]]["Color"] = value["Color"]

            # Check if the slices are defined
            if "Slices" not in pie_chart_config:
                raise KeyError("Slices not found in pie chart configuration file")

            # Read the configuration for the seperate pie slices
            for key, value in pie_chart_config["Slices"].items():
                pie_dict[value["Chart"]]["Values"].append(
                    float(metrics[value["Query"]])
                )
                pie_dict[value["Chart"]]["Labels"].append(value["Category"])
                pie_dict[value["Chart"]]["Colors"].append(
                    categorie_dict[value["Category"]]["Color"]
                )

            # Check if the "Other" category is defined
            if "Other" in pie_chart_config:
                pie_dict["Other"] = {}
                for key, value in pie_chart_config["Other"].items():
                    pie_dict["Other"][key] = value

        return pie_dict

    @staticmethod
    def _figures_list_to_html(
        figs,
        file_path: Union[str, Path] = "infographics.html",
        stylesheet: Union[str, Path] = "styles.css",
    ):
        """Save a list of plotly figures in an HTML file

        Parameters
        ----------
            figs : list[plotly.graph_objects.Figure]
                List of plotly figures to be saved. As it is currently implemented,
                the first figure will be the top half of the HTML file, the second
                figure will be the bottom left and the third figure will be the bottom right.
                If the list is shorter than 3, the remaining figures will be empty.
            file_path : Union[str, Path], optional
                Path to the HTML file, by default "infographics.html"
            stylesheet : Union[str, Path], optional
                Path to the stylesheet, by default "styles.css"

        Returns
        -------
            None

        Raises
        ------
            ValueError
                If the number of figures too large
            FileNotFoundError
                If the stylesheet is not found
            ValueError
                If the stylesheet is not a .css file
            FileExistsError
                If the file_path already exists
            ValueError
                If the file_path is not a .html file

        """

        # Check if the number of figures is correct
        if len(figs) > 3:
            raise ValueError("Only 3 figures are allowed")

        # Convert the stylesheet to a Path object
        if isinstance(stylesheet, str):
            stylesheet = Path(stylesheet)

        # Check if the stylesheet exists
        if not Path.exists(stylesheet):
            raise FileNotFoundError(f"Stylesheet not found at {stylesheet}")

        # Check if the stylesheet is a .css file
        if stylesheet.suffix != ".css":
            raise ValueError(f"Stylesheet must be a .css file, not {stylesheet}")

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
            figure1_html = (
                figs[0].to_html().split("<body>")[1].split("</body>")[0]
                if len(figs) > 0
                else ""
            )
            figure2_html = (
                figs[1].to_html().split("<body>")[1].split("</body>")[0]
                if len(figs) > 1
                else ""
            )
            figure3_html = (
                figs[2].to_html().split("<body>")[1].split("</body>")[0]
                if len(figs) > 2
                else ""
            )

            infographics.write(
                f"""<!DOCTYPE html>
                    <html>
                        <head>
                            <link rel="stylesheet" type="text/css" href="{stylesheet}">
                        </head>
                        <body>
                            <div class="container">
                                <div class="top-half">
                                    {figure1_html}
                                </div>
                                <div class="bottom-left">
                                    {figure2_html}
                                </div>
                                <div class="bottom-right">
                                    {figure3_html}
                                </div>
                            </div>
                        </body>
                    </html>"""
            )

    @staticmethod
    def _get_pie_chart_figure(data: Dict, **kwargs):
        """Create a pie chart figure from the pie chart dictionary, usually created by _get_pies_dictionary

        Parameters
        ----------
            data : Dict
                The pie chart dictionary
            **kwargs : dict
                Additional arguments for the figure

        Returns
        -------
            go.Figure
                The pie chart figure
        """

        # Remove the "Other" category if it exists
        if "Other" in data:
            data.pop("Other")

        # Get the title and legend configuration with default values
        title = kwargs.get("title", "")
        legend_orientation = kwargs.get("legend_orientation", "h")
        yanchor = kwargs.get("yanchor", "bottom")
        y = kwargs.get("y", 1)
        xanchor = kwargs.get("xanchor", "center")
        x = kwargs.get("x", 0.5)

        # Create the pie chart figure
        fig = make_subplots(
            rows=1,
            cols=len(data),
            specs=[[{"type": "domain"}] * len(data)],
            horizontal_spacing=0.2 / len(data),
            vertical_spacing=0,
        )

        # Add the pie chart to the figure
        for idx, (key, value) in enumerate(data.items()):
            # Create single pie chart
            trace = go.Pie(
                values=value["Values"],
                labels=value["Labels"],
                hole=0.6,
                title=f"{ value['Name'] } <br> ",
                title_position="top center",
                hoverinfo="label+percent+name+value",
                textinfo="none",
                name=value["Name"],
                direction="clockwise",
                sort=False,
                marker={
                    "line": {"color": "#000000", "width": 2},
                    "colors": value["Colors"],
                },
            )

            # Add the pie chart to the figure
            fig.add_trace(trace, row=1, col=idx + 1)

            # Get the center of the pie chart (domain)
            domain_center_x = sum(fig.get_subplot(row=1, col=idx + 1).x) / 2
            domain_center_y = sum(fig.get_subplot(row=1, col=idx + 1).y) / 2

            # Add the pie chart image
            fig.add_layout_image(
                {
                    "source": value["Image"],
                    "sizex": 0.1,
                    "sizey": 0.1,
                    "x": domain_center_x,
                    "y": domain_center_y - 0.05,
                    "xanchor": "center",
                    "yanchor": "middle",
                    "visible": True,
                }
            )

            # Add the sum of all slices to the pie chart
            fig.add_annotation(
                x=domain_center_x,
                y=domain_center_y - 0.1,
                text="{}".format(sum(value["Values"])),
                font={"size": 20, "family": "Verdana", "color": "black"},
                xanchor="center",
                yanchor="top",
                showarrow=False,
            )

        # Final update for the layout
        fig.update_layout(
            font={"size": 20, "family": "Verdana", "color": "black"},
            title_text=title,
            title_font={"size": 25, "family": "Verdana", "color": "black"},
            title_x=0.5,
            autosize=True,
            legend={
                "orientation": legend_orientation,
                "yanchor": yanchor,
                "y": y,
                "xanchor": xanchor,
                "x": x,
                "itemclick": False,
                "itemdoubleclick": False,
            },
        )

        # Update the layout images
        fig.update_layout_images()

        return fig

    def _get_infographics(
        self,
    ) -> go.Figure:
        """Get the infographic for a scenario

        Returns
        -------
        go.Figure
            The infographic for the scenario

        """

        # Get the impact metrics
        metrics = self._get_impact_metrics()

        # Get the infographic configuration
        pie_chart_config_path = self.config_base_path.joinpath("config_charts.toml")
        pie_people_config_path = self.config_base_path.joinpath("config_people.toml")

        # Check if the infographic configuration files exist
        if not Path.exists(pie_chart_config_path):
            raise FileNotFoundError(
                f"Infographic configuration file not found at {pie_chart_config_path}"
            )
        if not Path.exists(pie_people_config_path):
            raise FileNotFoundError(
                f"Infographic configuration file not found at {pie_people_config_path}"
            )

        # Get the pie chart dictionaries
        charts = InfographicsParser._get_pies_dictionary(pie_chart_config_path, metrics)
        people = InfographicsParser._get_pies_dictionary(
            pie_people_config_path, metrics
        )

        # Create the pie chart figures
        charts_fig = InfographicsParser._get_pie_chart_figure(
            data=charts.copy(),
            legend_orientation="h",
            yanchor="top",
            y=-0.1,
            title="Building damage",
        )

        people_fig = InfographicsParser._get_pie_chart_figure(
            data=people.copy(),
            legend_orientation="h",
            yanchor="top",
            y=-0.1,
            title="People",
        )

        # Return the figure
        return [charts_fig, people_fig]

    def get_infographics(self) -> Union[List[go.Figure], go.Figure]:
        """Get the infographic for a scenario

        Returns
        -------
        Union[List[go.Figure], go.Figure]
            The infographic for the scenario as a list of figures or a single figure
        """

        # Get the infographic
        infographic = self._get_infographics()

        # Return the infographic
        return infographic

    def write_infographics_to_file(self) -> str:
        """Write the infographic for a scenario to file

        Returns
        -------
        str
            The path to the infographic file
        """

        # Get the infographic stylesheet
        infographic_style = self.config_base_path.joinpath("styles.css")

        # Check if the infographic stylesheet exists
        if not Path.exists(infographic_style):
            raise FileNotFoundError(
                f"Infographic stylesheet not found at {infographic_style}"
            )

        # Create the infographic path
        infographic_html = self.output_base_path.joinpath(
            f"{self.scenario_name}_metrics.html"
        )

        # Check if the infographic already exists. If so, return the path
        if Path.exists(infographic_html):
            # TODO: Print logging message
            return str(infographic_html)

        # Get the infographic
        infographic = self._get_infographics()

        # Convert the infographic to html
        self._figures_list_to_html(infographic, infographic_html, infographic_style)

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
