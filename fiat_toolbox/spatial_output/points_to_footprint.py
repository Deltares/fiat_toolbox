from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd


class IPointsToFootprints(ABC):
    """Interface for writing a footprints spatial file."""

    @abstractmethod
    def write_footprint_file(
        footprints: gpd.GeoDataFrame, results: pd.DataFrame
    ) -> None:
        pass


class PointsToFootprints(IPointsToFootprints):
    """Write a footprints spatial file."""

    @staticmethod
    def _check_extension(out_path, ext):
        out_path = Path(out_path)
        if out_path.suffix != ext:
            raise ValueError(
                f"File extention given: '{out_path.suffix}' does not much the file format specified: {ext}."
            )

    @staticmethod
    def _mode(my_list):
        ct = Counter(my_list)
        max_value = max(ct.values())
        return sorted(key for key, value in ct.items() if value == max_value)

    @staticmethod
    def write_footprint_file(
        footprints: gpd.GeoDataFrame,
        points: pd.DataFrame,
        out_path: Union[str, Path],
        id: Optional[str] = "BF_FID",
        extra_footprints: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        # Merge based on "id" column
        gdf = pd.merge(footprints[[id, "geometry"]], points, on=id, how="outer")

        # Remove the building footprints without any object attached
        gdf = gdf.loc[~gdf["Object ID"].isnull()]
        gdf["Object ID"] = gdf["Object ID"].astype(int)

        # Get columns that will be used
        strings = ["Primary Object Type"] + [
            col for col in gdf.columns if "Aggregation Label:" in col
        ]

        depths = []

        # Get type of run
        if "Total Damage" in gdf.columns:
            # If event save inundation depth
            depths = depths + [col for col in gdf.columns if "Inundation Depth" in col]
            # And all type of damages
            dmgs = [
                col
                for col in gdf.columns
                if "Damage:" in col and "Max Potential" not in col
            ]
            dmgs.append("Total Damage")
        elif "Risk (EAD)" in gdf.columns:
            # For risk only save total damage per return period and EAD
            dmgs = [col for col in gdf.columns if "Total Damage" in col]
            dmgs.append("Risk (EAD)")
        else:
            raise ValueError(
                "The is no 'Total Damage' or 'Risk (EAD)' column in the results."
            )

        # Aggregate objects with the same "id"
        count = np.unique(gdf[id], return_counts=True)
        multiple_bffid = count[0][count[1] > 1][:-1]

        # First, combine the Primary Object Type and Object ID
        bffid_object_mapping = {}
        bffid_objectid_mapping = {}
        for bffid in multiple_bffid:
            all_objects = gdf.loc[gdf[id] == bffid, "Primary Object Type"].values
            all_object_ids = gdf.loc[gdf[id] == bffid, "Object ID"].values
            bffid_object_mapping.update(
                {bffid: "_".join(PointsToFootprints._mode(all_objects))}
            )
            bffid_objectid_mapping.update(
                {bffid: "_".join([str(x) for x in all_object_ids])}
            )
        gdf.loc[gdf[id].isin(multiple_bffid), "Primary Object Type"] = gdf[id].map(
            bffid_object_mapping
        )
        gdf.loc[gdf[id].isin(multiple_bffid), "Object ID"] = gdf[id].map(
            bffid_objectid_mapping
        )

        # Aggregated results using different functions based on type of output
        mapping = {}
        for name in strings:
            mapping[name] = pd.Series.mode
        for name in depths:
            mapping[name] = "mean"
        for name in dmgs:
            mapping[name] = "sum"

        agg_cols = strings + depths + dmgs

        df_groupby = (
            gdf.loc[gdf[id].isin(multiple_bffid), [id] + agg_cols]
            .groupby(id)
            .agg(mapping)
        )

        # Replace values in footprints file
        for agg_col in agg_cols:
            bffid_aggcol_mapping = dict(zip(df_groupby.index, df_groupby[agg_col]))
            gdf.loc[gdf[id].isin(multiple_bffid), agg_col] = gdf[id].map(
                bffid_aggcol_mapping
            )

        # Drop duplicates
        gdf = gdf.drop_duplicates("BF_FID")
        gdf = gdf.reset_index(drop=True)
        gdf = gdf[["Object ID", "geometry"] + agg_cols]
        gdf.to_file(out_path, driver="GPKG")

        return gdf
