from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

_FORMATS = ["geopackage", "shapefile", "GeoJSON"]


class IAggregationAreas(ABC):
    """Interface for writing an aggregation areas spatial file."""

    @abstractmethod
    def write_spatial_file(
        df_metrics: pd.DataFrame,
        gdf_aggr_areas: gpd.GeoDataFrame,
        out_path: Union[str, Path],
        id_name: Optional[str] = "name",
        file_format: Optional[str] = "geopackage",
    ) -> None:
        """Saves a geospatial file where the aggregation areas are join with metrics from a metric table.

        Parameters
        ----------
        df_metrics : pd.DataFrame
            dataframe containing the metrics
        gdf_aggr_areas : gpd.GeoDataFrame
            geodataframe with the aggregation areas (with an identifier column provided with argument "id_name")
        out_path : Union[str, Path]
            path where the geospatial file should be saved
        id_name : Optional[str], optional
            name of the identified column in gdf_aggr_areas to be used for the join, by default "name"
        file_format : Optional[str], optional
            file format of the output geospatial file, by default "geopackage"

        Raises
        ------
        ValueError
            If the given file format is not implemented.
        """
        pass


class AggregationAreas(IAggregationAreas):
    """Write an aggregation areas spatial file."""

    @staticmethod
    def _check_extension(out_path, ext):
        out_path = Path(out_path)
        if out_path.suffix != ext:
            raise ValueError(
                f"File extension given: '{out_path.suffix}' does not much the file format specified: {ext}."
            )

    @staticmethod
    def write_spatial_file(
        df_metrics: pd.DataFrame,
        gdf_aggr_areas: gpd.GeoDataFrame,
        out_path: Union[str, Path],
        id_name: Optional[str] = "name",
        file_format: Optional[str] = "geopackage",
    ) -> None:
        """Saves a geospatial file where the aggregation areas are join with metrics from a metric table.

        Parameters
        ----------
        df_metrics : pd.DataFrame
            dataframe containing the metrics
        gdf_aggr_areas : gpd.GeoDataFrame
            geodataframe with the aggregation areas (with an identifier column provided with argument "id_name")
        out_path : Union[str, Path]
            path where the geospatial file should be saved
        id_name : Optional[str], optional
            name of the identified column in gdf_aggr_areas to be used for the join, by default "name"
        file_format : Optional[str], optional
            file format of the output geospatial file, by default "geopackage"

        Raises
        ------
        ValueError
            If the given file format is not implemented.
        """
        # Get index as the first column
        index_name = df_metrics.columns[0]
        df_metrics = df_metrics.set_index(index_name)

        # Only keep metrics that are supposed to be in the metrics table
        metrics_to_keep = (
            df_metrics.loc["Show In Metrics Table", :]
            .map(lambda x: True if x == "True" else False)
            .astype(bool)
        )
        df = df_metrics.loc[:, metrics_to_keep]

        # Drop rows containing other variables
        df = df.drop(["Description", "Show In Metrics Table", "Long Name"])
        df = df.apply(pd.to_numeric)

        # Joins based on provided column name
        joined = gdf_aggr_areas.join(df, on=id_name)

        # Save file
        if file_format == "geopackage":
            AggregationAreas._check_extension(out_path, ".gpkg")
            joined.to_file(out_path, driver="GPKG")
        elif file_format == "shapefile":
            AggregationAreas._check_extension(out_path, ".shp")
            joined.to_file(out_path)
        elif file_format == "GeoJSON":
            AggregationAreas._check_extension(out_path, ".geojson")
            joined.to_file(out_path, driver="GeoJSON")
        else:
            raise ValueError(
                f"File format specified: {file_format} not in implemented formats: {*_FORMATS,}."
            )
