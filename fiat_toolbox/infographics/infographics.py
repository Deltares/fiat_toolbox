import logging
from pathlib import Path
from typing import Dict, List, Union

import tomli
import validators
from PIL import Image
from plotly.graph_objects import Bar, Figure, Pie
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
                if "Image" in value:
                    pie_dict[value["Name"]]["Image"] = value["Image"]
                pie_dict[value["Name"]]["Values"] = []
                pie_dict[value["Name"]]["Colors"] = []
                pie_dict[value["Name"]]["Labels"] = []

            # Check if the categories are defined
            if "Categories" not in pie_chart_config:
                raise KeyError("Categories not found in pie chart configuration file")

            # Read the categories configuration
            category_dict = {}
            for key, value in pie_chart_config["Categories"].items():
                category_dict[value["Name"]] = {}
                category_dict[value["Name"]]["Name"] = value["Name"]
                category_dict[value["Name"]]["Color"] = value["Color"]
                if "Image" in value:
                    category_dict[value["Name"]]["Image"] = value["Image"]

            # Check if the slices are defined
            if "Slices" not in pie_chart_config:
                raise KeyError("Slices not found in pie chart configuration file")

            # Read the configuration for the separate pie slices
            for key, value in pie_chart_config["Slices"].items():
                pie_dict[value["Chart"]]["Values"].append(
                    float(metrics[value["Query"]])
                )
                pie_dict[value["Chart"]]["Labels"].append(value["Category"])
                pie_dict[value["Chart"]]["Colors"].append(
                    category_dict[value["Category"]]["Color"]
                )
                if "Image" in category_dict[value["Category"]]:
                    if "Image" not in pie_dict[value["Chart"]]:
                        pie_dict[value["Chart"]]["Image"] = []
                    pie_dict[value["Chart"]]["Image"].append(
                        category_dict[value["Category"]]["Image"]
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

        Returns
        -------
            None

        Raises
        ------
            ValueError
                If the number of figures too large
            FileExistsError
                If the file_path already exists
            ValueError
                If the file_path is not a .html file

        """

        # Check if the number of figures is correct
        if len(figs) > 3:
            raise ValueError("Only 3 figures are allowed")

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
                figs[0]
                .to_html(config={"displayModeBar": False})
                .split("<body>")[1]
                .split("</body>")[0]
                if len(figs) > 0
                else ""
            )
            figure2_html = (
                figs[1]
                .to_html(config={"displayModeBar": False})
                .split("<body>")[1]
                .split("</body>")[0]
                if len(figs) > 1
                else ""
            )
            figure3_html = (
                figs[2]
                .to_html(config={"displayModeBar": False})
                .split("<body>")[1]
                .split("</body>")[0]
                if len(figs) > 2
                else ""
            )

            infographics.write(
                f"""
                <!DOCTYPE html>
                <html>
                    <head>
                        <style>
                        .container {{
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            justify-content: center;  # Center the plots vertically
                        }}
                        .top-half, .bottom {{
                            display: flex;
                            justify-content: center;
                            align-items: center;  # Center the plots vertically within their divs
                            width: 100%;
                        }}
                        .top-half {{
                            width: 100%;
                        }}
                        .bottom {{
                            flex-direction: row;
                        }}
                        .bottom-left, .bottom-right {{
                            width: 50%;
                            align-items: center;  # Center the plots vertically within their divs
                        }}
                    </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="top-half">
                                {figure1_html}
                            </div>
                            <div class="bottom">
                                <div class="bottom-left">
                                    {figure2_html}
                                </div>
                                <div class="bottom-right">
                                    {figure3_html}
                                </div>
                            </div>
                        </div>
                    </body>
                </html>
                """
            )

    @staticmethod
    def _check_image_source(
        img: str, image_folder_path: str = None, return_image: bool = True
    ) -> Union[str, Image.Image, None]:
        """Check if the image source is a url or a local path. If so, return the image source. If not, return None

        Parameters
        ----------
            img : str
                The image source
            image_folder_path : str, optional
                The path to the image folder, by default None
            return_image : bool, optional
                Whether to return the image or the path to the image, by default True returns the image

        Returns
        -------
            Union[str, Image, None]
                The image source or None if the image source is not a url or a local path
        """
        # Check if the image is a url. If so, add the image to the pie chart
        if validators.url(img):
            # Add the pie chart image
            return img
        elif image_folder_path and "{image_path}" in img:
            path = Path(img.replace("{image_path}", str(image_folder_path)))
            if Path.exists(path):
                if return_image:
                    return Image.open(path)
                else:
                    return str(path)
            else:
                return None
        else:
            path = Path(img)
            # Check if the given path is an absolute path
            if Path.exists(path):
                if return_image:
                    return Image.open(path)
                else:
                    return str(path)
            else:
                return None

    @staticmethod
    def _add_info_button(
        fig: Figure, plot_info: str, img: str, img_path: str, scale: float
    ) -> Figure:
        """Add an info button to a plotly figure

        Parameters
        ----------
            fig : plotly.graph_objects.Figure
                The plotly figure to which the info button should be added
            plot_info : str
                The text that should be shown when the info button is clicked
            img : str
                The image source
            img_path : str
                The path to the image folder
            scale : float
                The scale of the image

        Returns
        -------
            plotly.graph_objects.Figure
                The plotly figure with the info button
        """

        # Check if the image source is a url or a local path
        img_source = InfographicsParser._check_image_source(img, img_path)

        # Add an image
        fig.add_layout_image(
            {
                "source": img_source,
                "xref": "paper",
                "yref": "paper",
                "x": 1,
                "y": 1,
                "sizex": scale,
                "sizey": scale,
                "xanchor": "center",
                "yanchor": "top",
            }
        )

        # Add a hover label to the image
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1,
            y=1,
            text="   ",
            xanchor="center",
            yanchor="top",
            showarrow=False,
            font={"size": 16, "color": "LightSeaGreen"},
            hovertext=plot_info,
        )

        return fig

    @staticmethod
    def _get_pie_chart_figure(data: Dict, **kwargs):
        """Create a pie chart figure from the pie chart dictionary, usually created by _get_pies_dictionary

        Parameters
        ----------
            data : Dict
                The pie chart dictionary
            **title : str, optional
                The title of the pie chart, by default ""
            **title_font_size : int, optional
                The font size of the title, by default 25
            **subtitle_font_size : int, optional
                The font size of the subtitle, by default 20
            **image_scale : float, optional
                The scale of the image, by default 0.2
            **numbers_font : int, optional
                The font size of the numbers, by default 20
            **legend_font_size : int, optional
                The font size of the legend, by default 20
            **legend_orientation : str, optional
                The orientation of the legend, by default "h"
            **yanchor : str, optional
                The y anchor of the legend, by default "bottom"
            **y : float, optional
                The y position of the legend, by default 1
            **xanchor : str, optional
                The x anchor of the legend, by default "center"
            **x : float, optional
                The x position of the legend, by default 0.5
            **image_path : Union[str, Path], optional
                The path to the image folder, by default None
            **plot_width : int, optional
                The width of the plot in pixels, by default len(data)*200
            **plot_height : int, optional
                The height of the plot in pixels, by default 500
            **plot_info : str, optional
                The plot info, by default ""
            **plot_info_img : str, optional
                The plot info image, by default ""
            **plot_info_scale : float, optional
                The scale of the plot info image, by default 0.2

        Returns
        -------
            Figure
                The pie chart figure
        """

        # Remove the "Other" category if it exists
        if "Other" in data:
            data.pop("Other")

        # Get the title and legend configuration with default values
        title = kwargs.get("title", "")
        title_font_size = kwargs.get("title_font_size", 25)
        subtitle_font_size = kwargs.get("subtitle_font_size", 20)
        image_scale = kwargs.get("image_scale", 0.2)
        numbers_font = kwargs.get("numbers_font", 20)
        legend_font_size = kwargs.get("legend_font_size", 20)
        legend_orientation = kwargs.get("legend_orientation", "h")
        yanchor = kwargs.get("yanchor", "bottom")
        y = kwargs.get("y", 1)
        xanchor = kwargs.get("xanchor", "center")
        x = kwargs.get("x", 0.5)
        image_path = kwargs.get("image_path", None)
        plot_width = kwargs.get("plot_width", len(data) * 200)
        plot_height = kwargs.get("plot_height", 500)
        plot_info = kwargs.get("plot_info", "")
        plot_info_img = kwargs.get("plot_info_img", "")
        plot_info_scale = kwargs.get("plot_info_scale", 0.2)

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
            trace = Pie(
                values=value["Values"],
                labels=value["Labels"],
                hole=0.6,
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

            # Add the title annotation
            fig.add_annotation(
                x=domain_center_x,
                y=1,
                text=f"{value['Name']} <br> ",
                font={
                    "size": subtitle_font_size,
                    "family": "Verdana",
                    "color": "black",
                },
                xanchor="center",
                yanchor="middle",
                showarrow=False,
            )

            # Add the image to the pie chart
            img_source = InfographicsParser._check_image_source(
                value["Image"], image_path
            )

            if img_source:
                fig.add_layout_image(
                    {
                        "source": img_source,
                        "sizex": image_scale,
                        "sizey": image_scale,
                        "x": domain_center_x,
                        "y": domain_center_y + 0.05,
                        "xanchor": "center",
                        "yanchor": "middle",
                        "visible": True,
                    }
                )

            # Add the sum of all slices to the pie chart
            fig.add_annotation(
                x=domain_center_x,
                y=domain_center_y - 0.05,
                text="{:,.0f}".format(sum(value["Values"])),
                font={"size": numbers_font, "family": "Verdana", "color": "black"},
                xanchor="center",
                yanchor="top",
                showarrow=False,
            )

        # Final update for the layout
        fig.update_layout(
            title_text=title,
            title_font={"size": title_font_size, "family": "Verdana", "color": "black"},
            title_x=0.5,
            width=plot_width,  # Set the width in pixels
            height=plot_height,  # Set the height in pixels
            legend={
                "orientation": legend_orientation,
                "yanchor": yanchor,
                "y": y,
                "xanchor": xanchor,
                "x": x,
                "itemclick": False,
                "itemdoubleclick": False,
                "font": {
                    "size": legend_font_size,
                    "family": "Verdana",
                    "color": "black",
                },
            },
        )

        # Add an info button
        fig = InfographicsParser._add_info_button(
            fig, plot_info, plot_info_img, image_path, plot_info_scale
        )

        # Update the layout images
        fig.update_layout_images()

        return fig

    @staticmethod
    def _get_bar_chart_figure(data: Dict, **kwargs):
        """Create a bar chart figure from the bar chart dictionary, usually created by _get_pies_dictionary

        Parameters
        ----------
        data : Dict
            The bar chart dictionary
        **title : str, optional
            The title of the bar chart, by default ""
        **yaxis_title : str, optional
            The title of the y axis of the bar chart, by default ""
        **title_font_size : int, optional
            The font size of the title, by default 25
        **subtitle_font_size : int, optional
            The font size of the subtitle, by default 20
        **image_scale : float, optional
            The scale of the image, by default 0.2
        **numbers_font : int, optional
            The font size of the numbers, by default 20
        **image_path : Union[str, Path], optional
            The path to the image folder, by default None
        **plot_width : int, optional
            The width of the plot in pixels, by default len(data)*200
        **plot_height : int, optional
            The height of the plot in pixels, by default 500
        **plot_info : str, optional
            The plot info, by default ""
        **plot_info_img : str, optional
            The plot info image, by default ""
        **plot_info_scale : float, optional
            The scale of the plot info image, by default 0.2

        Returns
        -------
        Figure
            The bar chart figure
        """

        # Get the title and legend configuration with default values
        title = kwargs.get("title", "")
        yaxis_title = kwargs.get("yaxis_title", "")
        title_font_size = kwargs.get("title_font_size", 25)
        subtitle_font_size = kwargs.get("subtitle_font_size", 20)
        image_scale = kwargs.get("image_scale", 0.2)
        numbers_font = kwargs.get("numbers_font", 20)
        image_path = kwargs.get("image_path", None)
        plot_width = kwargs.get("plot_width", 600)
        plot_height = kwargs.get("plot_height", 500)
        plot_info = kwargs.get("plot_info", "")
        plot_info_img = kwargs.get("plot_info_img", "")
        plot_info_scale = kwargs.get("plot_info_scale", 0.2)

        # Remove the "Other" category if it exists
        if "Other" in data:
            data.pop("Other")

        # Create the pie chart figure
        fig = make_subplots(
            rows=1,
            cols=len(data),
            specs=[[{"type": "domain"}] * len(data)],
            horizontal_spacing=0.2 / len(data),
            vertical_spacing=0,
        )

        # Add the bar chart to the figure
        for idx, (key, chart) in enumerate(data.items()):
            # Create single bar chart
            for i, (value, color, label, image) in enumerate(
                zip(chart["Values"], chart["Colors"], chart["Labels"], chart["Image"])
            ):
                # Add bar to the figure
                fig.add_trace(
                    Bar(
                        x=[label],
                        y=[int(value)],
                        marker={
                            "color": color,
                            "line": {"color": "black", "width": 2},
                        },
                        hoverinfo="x+y",
                    )
                )

                # Add annotation with the value on top of the bar
                fig.add_annotation(
                    x=label,
                    y=value + 0.1,
                    text="{:,.0f}".format(value),
                    showarrow=False,
                    font={
                        "color": "black",  # Set text color to white
                        "size": numbers_font,  # Set text size to 14
                    },
                    xref="x",
                    yref="y",
                    yanchor="bottom",
                )

                # Add the image to the bar chart
                img_source = InfographicsParser._check_image_source(image, image_path)

                if img_source:
                    # Calculate paper coordinates for the image
                    x_paper = (i + 0.5) / len(chart["Labels"])

                    # Add image below the bar
                    fig.add_layout_image(
                        {
                            "source": img_source,
                            "xref": "paper",
                            "yref": "paper",
                            "x": x_paper,
                            "y": -0.1 * image_scale,
                            "sizex": image_scale,
                            "sizey": image_scale,
                            "xanchor": "center",
                            "yanchor": "top",
                            "sizing": "contain",
                            "visible": True,
                        }
                    )

            # Add a new annotation to serve as the x-axis title
            fig.add_annotation(
                {
                    "x": 0.5,
                    "y": -image_scale,  # Adjust this value to move the title up or down
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "text": "Road users",  # X-axis title
                    "font": {"size": numbers_font},  # Adjust font size as needed
                    "xanchor": "center",
                    "yanchor": "top",
                }
            )

            # Update layout for better visualization
            fig.update_layout(
                title={
                    "text": title,
                    "x": 0.5,  # Center the main title
                },
                title_font={
                    "size": title_font_size,
                    "family": "Verdana",
                    "color": "black",
                },  # Adjust font size as needed
                xaxis={
                    "showticklabels": False,  # Hide x-axis labels
                },
                yaxis={
                    "title": yaxis_title,
                    "showgrid": False,  # Hide y-axis grid lines
                    "zeroline": False,  # Hide zero line
                    "showticklabels": False,  # Hide y-axis labels
                    "title_font": {
                        "size": subtitle_font_size,
                        "family": "Verdana",
                        "color": "black",
                    },  # Adjust font size as needed
                },
                width=plot_width,  # Set the width in pixels
                height=plot_height,  # Set the height in pixels
                showlegend=False,  # Remove legend
                plot_bgcolor="rgba(0, 0, 0, 0)",  # Set plot background color to transparent
                bargap=0.2,  # Set the gap between bars (adjust as needed)
                bargroupgap=0.5,  # Set the gap between bar groups (adjust as needed)
            )

        # Add an info button
        fig = InfographicsParser._add_info_button(
            fig, plot_info, plot_info_img, image_path, plot_info_scale
        )

        return fig

    def _get_infographics(
        self,
    ) -> Figure:
        """Get the infographic for a scenario

        Returns
        -------
        Figure
            The infographic for the scenario

        """

        # Get the impact metrics
        metrics = self._get_impact_metrics()

        # Get the infographic configuration
        pie_chart_config_path = self.config_base_path.joinpath("config_charts.toml")
        pie_people_config_path = self.config_base_path.joinpath("config_people.toml")
        roads_config_path = self.config_base_path.joinpath("config_roads.toml")

        # Get the pie chart dictionaries from the configuration for charts
        return_fig = []
        try:
            charts = InfographicsParser._get_pies_dictionary(
                pie_chart_config_path, metrics
            )
            charts_fig = InfographicsParser._get_pie_chart_figure(
                data=charts.copy(),
                legend_orientation="h",
                yanchor="top",
                y=-0.1,
                image_path=self.config_base_path.joinpath("images"),
                title=charts["Other"]["Title"]["text"],
                title_font_size=charts["Other"]["Title"]["font"],
                subtitle_font_size=charts["Other"]["Subtitle"]["font"],
                image_scale=charts["Other"]["Plot"]["image_scale"],
                numbers_font=charts["Other"]["Plot"]["numbers_font"],
                legend_font_size=charts["Other"]["Legend"]["font"],
                plot_width=charts["Other"]["Plot"]["width"],
                plot_height=charts["Other"]["Plot"]["height"],
                plot_info=charts["Other"]["Info"]["text"],
                plot_info_img=charts["Other"]["Info"]["image"],
                plot_info_scale=charts["Other"]["Info"]["scale"],
            )
            return_fig.append(charts_fig)
        except FileNotFoundError:
            self.logger.warning("No charts configuration file found")

        # Get the pie chart dictionaries from the configuration for people
        try:
            people = InfographicsParser._get_pies_dictionary(
                pie_people_config_path, metrics
            )
            people_fig = InfographicsParser._get_pie_chart_figure(
                data=people.copy(),
                legend_orientation="h",
                yanchor="top",
                y=-0.1,
                image_path=self.config_base_path.joinpath("images"),
                title=people["Other"]["Title"]["text"],
                title_font_size=people["Other"]["Title"]["font"],
                subtitle_font_size=people["Other"]["Subtitle"]["font"],
                image_scale=people["Other"]["Plot"]["image_scale"],
                numbers_font=people["Other"]["Plot"]["numbers_font"],
                legend_font_size=people["Other"]["Legend"]["font"],
                plot_width=people["Other"]["Plot"]["width"],
                plot_height=people["Other"]["Plot"]["height"],
                plot_info=people["Other"]["Info"]["text"],
                plot_info_img=people["Other"]["Info"]["image"],
                plot_info_scale=people["Other"]["Info"]["scale"],
            )
            return_fig.append(people_fig)
        except FileNotFoundError:
            self.logger.warning("No people configuration file found")

        # Get the bar chart dictionaries from the configuration for roads
        try:
            roads = InfographicsParser._get_pies_dictionary(roads_config_path, metrics)
            roads_fig = InfographicsParser._get_bar_chart_figure(
                data=roads.copy(),
                image_path=self.config_base_path.joinpath("images"),
                title=roads["Other"]["Title"]["text"],
                yaxis_title=roads["Other"]["Y_axis_title"]["text"],
                title_font_size=roads["Other"]["Title"]["font"],
                subtitle_font_size=roads["Other"]["Subtitle"]["font"],
                image_scale=roads["Other"]["Plot"]["image_scale"],
                numbers_font=roads["Other"]["Plot"]["numbers_font"],
                plot_width=roads["Other"]["Plot"]["width"],
                plot_height=roads["Other"]["Plot"]["height"],
                plot_info=roads["Other"]["Info"]["text"],
                plot_info_img=roads["Other"]["Info"]["image"],
                plot_info_scale=roads["Other"]["Info"]["scale"],
            )
            return_fig.append(roads_fig)
        except FileNotFoundError:
            self.logger.warning("No roads configuration file found")

        # Return the figure
        return return_fig

    def get_infographics(self) -> Union[List[Figure], Figure]:
        """Get the infographic for a scenario

        Returns
        -------
        Union[List[Figure], Figure]
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
        self._figures_list_to_html(infographic, infographic_html)

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
