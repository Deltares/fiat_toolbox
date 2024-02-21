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
        gdf = footprints[[id, "geometry"]].merge(points, on=id, how="outer")

        # Remove the building footprints without any object attached
        gdf = gdf.loc[~gdf["Object ID"].isna()]
        gdf["Object ID"] = gdf["Object ID"].astype(int)

        # Get columns that will be used
        strings = ["Primary Object Type"] + [
            col for col in gdf.columns if "Aggregation Label:" in col
        ]

        for col in strings:
            gdf[col] = gdf[col].astype(str)

        depths = []

        # Get type of run
        if "Total Damage" in gdf.columns:
            run_type = "event"
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
            run_type = "risk"
            # For risk only save total damage per return period and EAD
            dmgs = [col for col in gdf.columns if "Total Damage" in col]
            dmgs.append("Risk (EAD)")
        else:
            raise ValueError(
                "The is no 'Total Damage' or 'Risk (EAD)' column in the results."
            )
        pot_dmgs = [col for col in gdf.columns if "Max Potential Damage" in col]
        dmgs = pot_dmgs + dmgs
        # Aggregate objects with the same "id"
        count = np.unique(gdf[id], return_counts=True)
        multiple_bffid = count[0][count[1] > 1][:-1]

        # First, combine the Primary Object Type and Object ID
        bffid_object_mapping = {}
        bffid_objectid_mapping = {}
        for bffid in multiple_bffid:
            all_objects = gdf.loc[gdf[id] == bffid, "Primary Object Type"].to_numpy()
            all_object_ids = gdf.loc[gdf[id] == bffid, "Object ID"].to_numpy()
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
        gdf = gdf.drop_duplicates(subset=[id])
        gdf = gdf.reset_index(drop=True)
        gdf = gdf[["Object ID", "geometry"] + agg_cols]

        for col in strings:
            for ind, val in enumerate(gdf[col]):
                if isinstance(val, np.ndarray):
                    gdf.loc[ind, col] = str(val[0])
        if extra_footprints is not None:
            extra_footprints = extra_footprints.to_crs(gdf.crs)
            # Merge based on "Object ID" column
            extra_footprints = extra_footprints[["Object ID", "geometry"]].merge(points, on="Object ID", how="left")[["Object ID", "geometry"]+agg_cols]
            gdf = pd.concat([gdf, extra_footprints], axis=0)
        
        # Calculate normalized damages per type
        value_cols = gdf.columns[gdf.columns.str.startswith('Max Potential Damage:')].tolist()
        
        # Only for event type calculate % damage per type
        if run_type == "event":
            dmg_cols = gdf.columns[gdf.columns.str.startswith('Damage:')].tolist()
            # Do per type
            for dmg_col in dmg_cols:
                new_name = dmg_col + " %"
                name = dmg_col.split("Damage: ")[1]
                gdf[new_name] = gdf[dmg_col] / gdf["Max Potential Damage: " + name] * 100
                gdf[new_name] = gdf[new_name].round(2)
            # Do total
            gdf["Total Damage %"] = gdf["Total Damage"] / gdf.loc[:, value_cols].sum(axis=1) * 100
            gdf["Total Damage %"] = gdf["Total Damage %"].round(2).fillna(0)
            
        # Calculate total normalized damage
        if run_type == "risk":
            tot_dmg_cols = gdf.columns[gdf.columns.str.startswith('Total Damage')].tolist()
            for tot_dmg_col in tot_dmg_cols:
                new_name = tot_dmg_col + " %"
                gdf[new_name] = gdf[tot_dmg_col] / gdf.loc[:, value_cols].sum(axis=1) * 100
                gdf[new_name] = gdf[new_name].round(2)
            gdf["Risk (EAD) %"] = gdf["Risk (EAD)"] / gdf.loc[:, value_cols].sum(axis=1) * 100
            gdf["Risk (EAD) %"] = gdf["Risk (EAD) %"].round(2).fillna(0)
        
        # Save file
        gdf.to_file(out_path, driver="GPKG")
        return gdf
