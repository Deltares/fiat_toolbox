import math
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as geom

from fiat_toolbox import FiatColumns, get_fiat_columns
from fiat_toolbox.utils import extract_variables, matches_pattern


def generate_polygon(point, shape_type, diameter):
    """
    Generate a polygon of a specified shape and diameter centered at a given point.

    Parameters
    ----------
    point (shapely.geometry.Point): The center point of the polygon.
    shape_type (str): The type of shape to generate. Must be one of 'circle', 'square', or 'triangle'.
    diameter (float): The diameter of the shape.

    Returns
    -------
    shapely.geometry.Polygon: The generated polygon.

    Raises
    ------
    ValueError: If the shape_type is not one of 'circle', 'square', or 'triangle'.
    """
    if shape_type == "circle":
        return point.buffer(diameter / 2)
    elif shape_type == "square":
        half_side = diameter / 2
        return geom.Polygon(
            [
                (point.x - half_side, point.y - half_side),
                (point.x + half_side, point.y - half_side),
                (point.x + half_side, point.y + half_side),
                (point.x - half_side, point.y + half_side),
            ]
        )
    elif shape_type == "triangle":
        height = (math.sqrt(3) / 2) * diameter
        return geom.Polygon(
            [
                (point.x, point.y - height / 2),
                (point.x - diameter / 2, point.y + height / 2),
                (point.x + diameter / 2, point.y + height / 2),
            ]
        )
    else:
        raise ValueError(
            "Invalid shape type. Choose from 'circle', 'square', or 'triangle'."
        )


def check_extension(out_path, ext):
    """
    Checks if the file extension of the given path matches the specified extension.

    Parameters:
    out_path (str or Path): The path to the file.
    ext (str): The expected file extension (including the dot, e.g., '.txt').

    Raises:
    ValueError: If the file extension of out_path does not match the specified ext.
    """
    out_path = Path(out_path)
    if out_path.suffix != ext:
        raise ValueError(
            f"File extention given: '{out_path.suffix}' does not much the file format specified: {ext}."
        )


def mode(my_list):
    """
    Calculate the mode(s) of a list.

    The mode is the value that appears most frequently in a data set. If there are multiple values with the same highest frequency, all of them are returned in a sorted list.

    Parameters:
    my_list (list): A list of elements to find the mode of.

    Returns:
    list: A sorted list of the mode(s) of the input list.
    """
    ct = Counter(my_list)
    max_value = max(ct.values())
    return sorted(key for key, value in ct.items() if value == max_value)


class Footprints:

    def __init__(
        self,
        footprints: Optional[gpd.GeoDataFrame] = gpd.GeoDataFrame(),
        field_name: Optional[str] = None,
        fiat_columns: Optional[FiatColumns] = None,
        fiat_version: Optional[str] = "0.2",
        depth_rounding: Optional[int] = 2,
        damage_rounding: Optional[int] = 0,
    ):
        """
        Initialize the Footprints object.
        Parameters:
        -----------
        footprints : gpd.GeoDataFrame
            A GeoDataFrame containing the footprint geometries.
        field_name : Optional[str], default "BF_FID"
            The name of the field to be used as the index. Must be present in the footprints columns.
        fiat_columns : Optional[FiatColumns], default None
            An object containing the column naming format. If None, the default format for the specified fiat_version will be used.
        fiat_version : Optional[str], default "0.2"
            The version of the FIAT format to use for column naming.
        Raises:
        -------
        AttributeError
            If the specified field_name is not present in the footprints columns.
        ValueError
            If the values in the specified field_name are not unique.
        """
        # Check if field name is present
        if field_name is not None:
            if field_name not in footprints.columns:
                raise AttributeError(
                    f"field_name= '{field_name}' is not in footprints columns."
                )
            # Check if indices are unique
            if not footprints[field_name].is_unique:
                raise ValueError(f"Values in the field '{field_name}' are not unique.")
            # Save attributes
            footprints = footprints.set_index(field_name)
        if (field_name is None) and (not footprints.empty):
            raise AttributeError(
                "'field_name' attribute needs to be provided to define the unique identifier of the given footprints."
            )
        self.footprints = footprints
        self.field_name = field_name
        # Get column naming format
        if fiat_columns is None:
            self.fiat_columns = get_fiat_columns(fiat_version=fiat_version)
        else:
            self.fiat_columns = fiat_columns

        self.depth_rounding = depth_rounding
        self.damage_rounding = damage_rounding

    def set_point_data(
        self,
        objects: Union[gpd.GeoDataFrame, pd.DataFrame],
        no_footprints_shape: str = "triangle",
        no_footprints_diameter: float = 10.0,
    ):
        """
        Sets the point data for the given objects, converting points without footprints to polygons.

        Parameters:
        -----------
        objects : Union[gpd.GeoDataFrame, pd.DataFrame]
            The input data containing the objects. It can be a GeoDataFrame or a DataFrame.
        no_footprints_shape : str, optional
            The shape to use for objects without footprints. Default is "triangle".
        no_footprints_diameter : float, optional
            The diameter to use for the shape of objects without footprints. Default is 10.0.

        Returns:
        --------
        None
        """
        # Get column names per type
        columns = self._get_column_names(objects)
        cols = columns["string"] + columns["depth"] + columns["damage"]
        # Convert points to shapes
        no_footprint_objects_with_shape = self._no_footprint_points_to_polygons(
            objects, no_footprints_shape, no_footprints_diameter
        )
        # Filter columns
        gdf = no_footprint_objects_with_shape[
            [self.fiat_columns.object_id, "geometry"] + cols
        ]
        # Rounding
        for col in columns["depth"]:
            gdf[col] = gdf[col].round(self.depth_rounding)
        for col in columns["damage"]:
            gdf[col] = gdf[col].round(self.damage_rounding).fillna(0)
        # Save to object
        self.results = gdf

    def aggregate(
        self,
        objects: Union[gpd.GeoDataFrame, pd.DataFrame],
        field_name: Optional[str] = None,
        drop_no_footprints: Optional[bool] = False,
        no_footprints_shape: str = "triangle",
        no_footprints_diameter: float = 10.0,
    ):
        """
        Aggregate objects based on a specified field name that connects to the footprints unique id. If objects has
        spatial information, it can be used for objects without footprint connections to make standard shape footprints.
        Parameters:
        -----------
        objects : Union[gpd.GeoDataFrame, pd.DataFrame]
            The spatial objects to be aggregated. Can be a GeoDataFrame or DataFrame.
        field_name : Optional[str], default=None
            The field name to merge and aggregate objects on. If not provided, it defaults to the instance's field_name.
        drop_no_footprints : Optional[bool], default=False
            If True, objects without footprints will be dropped. If False, they will be assigned a default shape.
        no_footprints_shape : str, default="triangle"
            The shape to assign to objects without footprints. Options include "triangle", "circle", etc.
        no_footprints_diameter : float, default=10.0
            The diameter of the shape to assign to objects without footprints.
        Raises:
        -------
        AttributeError
            If the specified field_name is not found in the columns of the provided objects.
        Returns:
        --------
        None
            The aggregated results are stored in the instance's `aggregated_results` attribute.
        """
        # Merge based on "field_name" column
        if (
            field_name is None
        ):  # if field_name is not provided assume it is the same as the footprints one
            field_name = self.field_name
        if field_name not in objects.columns:
            raise AttributeError(
                f"'{field_name}' not found columns of the provided objects."
            )
        gdf = self.footprints.merge(
            objects.drop(columns="geometry", errors="ignore"),
            on=field_name,
            how="outer",
        )

        # Remove the building footprints without any object attached
        gdf = gdf.loc[~gdf[self.fiat_columns.object_id].isna()]
        gdf[self.fiat_columns.object_id] = gdf[self.fiat_columns.object_id].astype(
            int
        )  # ensure that object ids are interpreted correctly as integers

        # Get column names per type
        columns = self._get_column_names(gdf)

        for col in columns["string"]:
            gdf[col] = gdf[col].astype(str)

        # Aggregate objects with the same "field_name"
        count = np.unique(gdf[field_name], return_counts=True)
        multiple_bffid = count[0][count[1] > 1][:-1]

        # First, combine the Primary Object Type and Object ID
        bffid_object_mapping = {}
        bffid_objectid_mapping = {}
        for bffid in multiple_bffid:
            all_objects = gdf.loc[
                gdf[field_name] == bffid, self.fiat_columns.primary_object_type
            ].to_numpy()
            all_object_ids = gdf.loc[
                gdf[field_name] == bffid, self.fiat_columns.object_id
            ].to_numpy()
            bffid_object_mapping.update({bffid: "_".join(mode(all_objects))})
            bffid_objectid_mapping.update(
                {bffid: "_".join([str(x) for x in all_object_ids])}
            )
        gdf.loc[
            gdf[field_name].isin(multiple_bffid), self.fiat_columns.primary_object_type
        ] = gdf[field_name].map(bffid_object_mapping)
        gdf.loc[gdf[field_name].isin(multiple_bffid), self.fiat_columns.object_id] = (
            gdf[field_name].map(bffid_objectid_mapping)
        )

        # Aggregated results using different functions based on type of output
        mapping = {}
        for name in columns["string"]:
            mapping[name] = pd.Series.mode
        for name in columns["depth"]:
            mapping[name] = "mean"
        for name in columns["damage"]:
            mapping[name] = "sum"

        agg_cols = columns["string"] + columns["depth"] + columns["damage"]

        df_groupby = (
            gdf.loc[gdf[field_name].isin(multiple_bffid), [field_name] + agg_cols]
            .groupby(field_name)
            .agg(mapping)
        )

        # Replace values in footprints file
        for agg_col in agg_cols:
            bffid_aggcol_mapping = dict(zip(df_groupby.index, df_groupby[agg_col]))
            gdf.loc[gdf[field_name].isin(multiple_bffid), agg_col] = gdf[
                field_name
            ].map(bffid_aggcol_mapping)

        # Drop duplicates
        gdf = gdf.drop_duplicates(subset=[field_name])
        gdf = gdf.reset_index(drop=True)
        exposure = [self.fiat_columns.object_id, "geometry"] + agg_cols
        gdf = gdf[exposure]

        for col in columns["string"]:
            for ind, val in enumerate(gdf[col]):
                if isinstance(val, np.ndarray):
                    gdf.loc[ind, col] = str(val[0])

        # Add extra footprints
        extra_footprints = []

        # If point object don't have a footprint reference assume a shape
        if not drop_no_footprints and "geometry" in objects.columns:
            no_footprint_objects = objects[
                (objects[self.field_name].isna()) & (objects.geometry.type == "Point")
            ]
            if len(no_footprint_objects) > 1:
                no_footprint_objects_with_shape = self._no_footprint_points_to_polygons(
                    no_footprint_objects, no_footprints_shape, no_footprints_diameter
                )
                no_footprint_objects_with_shape = no_footprint_objects_with_shape[
                    [self.fiat_columns.object_id, "geometry"] + agg_cols
                ].to_crs(gdf.crs)
                extra_footprints.append(no_footprint_objects_with_shape)

        # Add objects which are already described by a polygon
        if "geometry" in objects.columns:
            footprint_objects = self._find_footprint_objects(objects)[
                [self.fiat_columns.object_id, "geometry"] + agg_cols
            ].to_crs(gdf.crs)
            extra_footprints.append(footprint_objects)

        # Combine
        gdf = pd.concat([gdf] + extra_footprints, axis=0)

        # Rounding
        for col in columns["depth"]:
            gdf[col] = gdf[col].round(self.depth_rounding)
        for col in columns["damage"]:
            gdf[col] = gdf[col].round(self.damage_rounding).fillna(0)

        self.results = gdf

    def calc_normalized_damages(self):
        """
        Calculate normalized damages for the aggregated results.
        This method calculates the normalized damages per type and total damage percentage
        for the given aggregated results based on the run type. The results are stored back
        in the `aggregated_results` attribute.
        For "event" run type:
        - Calculates the percentage damage per type and total damage percentage.
        For "risk" run type:
        - Calculates the total damage percentage and risk (Expected Annual Damage) percentage.
        The calculated percentages are rounded to 2 decimal places and stored in new columns
        in the GeoDataFrame.
        Attributes:
            aggregated_results (GeoDataFrame): The aggregated results containing damage data.
            run_type (str): The type of run, either "event" or "risk".
        Returns:
            None
        """
        gdf = self.results.copy()
        # Calculate normalized damages per type
        value_cols = [
            col
            for col in gdf.columns
            if matches_pattern(col, self.fiat_columns.max_potential_damage)
        ]

        # Only for event type calculate % damage per type
        if self.run_type == "event":
            dmg_cols = [
                col
                for col in gdf.columns
                if matches_pattern(col, self.fiat_columns.damage)
            ]
            # Do per type
            for dmg_col in dmg_cols:
                new_name = dmg_col + " %"
                name = extract_variables(dmg_col, self.fiat_columns.damage)["name"]
                gdf[new_name] = (
                    gdf[dmg_col]
                    / gdf[self.fiat_columns.max_potential_damage.format(name=name)]
                    * 100
                )
                gdf[new_name] = gdf[new_name].round(2).fillna(0)

            # Do total
            tot_dmg_per_name = f"{self.fiat_columns.total_damage} %"
            gdf[tot_dmg_per_name] = (
                gdf[self.fiat_columns.total_damage]
                / gdf.loc[:, value_cols].sum(axis=1)
                * 100
            )
            gdf[tot_dmg_per_name] = gdf[tot_dmg_per_name].round(2).fillna(0)

        elif self.run_type == "risk":
            tot_dmg_cols = gdf.columns[
                gdf.columns.str.startswith(self.fiat_columns.total_damage)
            ].tolist()
            for tot_dmg_col in tot_dmg_cols:
                new_name = tot_dmg_col + " %"
                gdf[new_name] = (
                    gdf[tot_dmg_col] / gdf.loc[:, value_cols].sum(axis=1) * 100
                )
                gdf[new_name] = gdf[new_name].round(2)
            risk_ead_per_name = f"{self.fiat_columns.risk_ead} %"
            gdf[risk_ead_per_name] = (
                gdf[self.fiat_columns.risk_ead]
                / gdf.loc[:, value_cols].sum(axis=1)
                * 100
            )
            gdf[risk_ead_per_name] = gdf[risk_ead_per_name].round(2).fillna(0)

        self.aggregated_results = gdf

    def write(self, output_path: Union[str, Path]):
        """
        Writes the aggregated results to a file.

        Parameters:
        output_path (Union[str, Path]): The path where the output file will be saved.
                                        It can be a string or a Path object.

        Returns:
        None
        """
        self.results.to_file(output_path, driver="GPKG")

    def _get_column_names(self, gdf):
        """
        Extracts and categorizes column names from a GeoDataFrame based on predefined criteria.
        Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame containing the columns to be categorized.
        Returns:
        col_dict: A dictionary with keys 'string', 'depth', and 'damage', each containing a list of column names.
            - 'string': Columns that are strings and will be aggregated.
            - 'depth': Columns related to inundation depth (only if total damage is present).
            - 'damage': Columns related to damage, including potential damage and total damage.
        Raises:
        ValueError: If neither 'total_damage' nor 'ead_damage' columns are present in the GeoDataFrame.
        """

        # Get string columns that will be aggregated
        string_columns = [self.fiat_columns.primary_object_type] + [
            col
            for col in gdf.columns
            if matches_pattern(col, self.fiat_columns.aggregation_label)
        ]

        # Get type of run and columns
        if self.fiat_columns.total_damage in gdf.columns:
            self.run_type = "event"
            # If event save inundation depth
            depth_columns = [
                col for col in gdf.columns if self.fiat_columns.inundation_depth in col
            ]
            # And all type of damages
            damage_columns = [
                col
                for col in gdf.columns
                if matches_pattern(col, self.fiat_columns.damage)
                and not matches_pattern(col, self.fiat_columns.max_potential_damage)
                and not matches_pattern(col, self.fiat_columns.damage_function)
            ]
            damage_columns.append(self.fiat_columns.total_damage)
        elif self.fiat_columns.risk_ead in gdf.columns:
            self.run_type = "risk"
            depth_columns = []
            # For risk only save total damage per return period and EAD
            damage_columns = [
                col
                for col in gdf.columns
                if matches_pattern(col, self.fiat_columns.total_damage_rp)
            ]
            damage_columns.append(self.fiat_columns.risk_ead)
        else:
            raise ValueError(
                f"The is no {self.fiat_columns.total_damage} or {self.fiat_columns.risk_ead} column in the results."
            )
        # add the max potential damages
        pot_damage_columns = [
            col
            for col in gdf.columns
            if matches_pattern(col, self.fiat_columns.max_potential_damage)
        ]
        damage_columns = pot_damage_columns + damage_columns

        # create mapping dictionary
        col_dict = {
            "string": string_columns,
            "depth": depth_columns,
            "damage": damage_columns,
        }

        return col_dict

    def _find_footprint_objects(self, objects):
        """
        Identifies and returns objects that have a footprint.

        This method filters the input objects to find those that do not have a
        value in the specified field (self.field_name) and have a geometry type
        of "Polygon".

        Parameters:
        objects (GeoDataFrame): A GeoDataFrame containing spatial objects with
                                geometries and attributes.

        Returns:
        GeoDataFrame: A GeoDataFrame containing objects that have a footprint
                      (i.e., objects with missing values in the specified field
                      and a geometry type of "Polygon").
        """
        buildings_with_footprint = objects[
            (objects[self.field_name].isna())
            & (objects.geometry.type.isin(["Polygon", "MultiPolygon"]))
        ]
        return buildings_with_footprint

    @staticmethod
    def _no_footprint_points_to_polygons(objects, shape, diameter):
        """
        Converts point geometries of buildings without footprints to polygon geometries.
        This method identifies buildings that do not have footprint information and converts their point geometries
        to polygon geometries based on the specified shape and diameter.
        Args:
            objects (GeoDataFrame): A GeoDataFrame containing building geometries and attributes.
            shape (str): The shape of the polygon to generate (e.g., 'circle', 'square').
            diameter (float): The diameter of the polygon to generate.
        Returns:
            GeoDataFrame or None: A GeoDataFrame with updated polygon geometries for buildings without footprints,
                                  or None if there are no such buildings.
        """
        init_crs = objects.crs
        objects = objects.to_crs(objects.estimate_utm_crs())
        shape_type = shape
        diameter = diameter

        # Transform points to shapes
        objects["geometry"] = objects["geometry"].apply(
            lambda point: generate_polygon(point, shape_type, diameter)
        )
        objects = objects.to_crs(init_crs)

        return objects
