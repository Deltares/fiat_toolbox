from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from fiat_toolbox.spatial_output.aggregation_areas import AggregationAreas

file_path = Path(__file__).parent.resolve()


def test_write_aggr_areas():
    # Aggegation levels to test
    names = ["aggr_lvl_1", "aggr_lvl_2"]
    for name in names:
        # Get metrics file
        metrics_path = (
            file_path / "data" / f"current_extreme12ft_comb_test_metrics_{name}.csv"
        )

        # Get areas file
        aggr_areas_path = file_path / "data" / f"{name}.geojson"

        # Define output name
        outpath = file_path / f"aggregation_areas_{name}.gpkg"

        # Read files
        metrics = pd.read_csv(metrics_path)
        aggr_areas = gpd.read_file(aggr_areas_path)

        # Write output
        AggregationAreas.write_spatial_file(metrics, aggr_areas, outpath)

        # Assert
        assert outpath.exists()

        out = gpd.read_file(outpath)
        assert isinstance(out, gpd.GeoDataFrame)
        assert sorted(out["name"]) == sorted(aggr_areas["name"])
        index_name = metrics.columns[0]
        metrics = metrics.set_index(index_name)
        values0 = pd.to_numeric(
            metrics.loc[sorted(aggr_areas["name"]), "TotalDamageEvent"]
        ).tolist()
        out = out.set_index("name")
        values1 = out.loc[sorted(aggr_areas["name"]), "TotalDamageEvent"].tolist()
        assert values0 == values1

        # Delete created files
        for file in list(outpath.parent.glob(outpath.stem + ".*")):
            file.unlink()


def test_error_handling():
    name = "aggr_lvl_2"

    # Get metrics file
    metrics_path = (
        file_path / "data" / f"current_extreme12ft_comb_test_metrics_{name}.csv"
    )

    # Get areas file
    aggr_areas_path = file_path / "data" / f"{name}.geojson"

    # Define output name
    outpath = file_path / f"aggregation_areas_{name}.gpkg"

    # Read files
    metrics = pd.read_csv(metrics_path)
    aggr_areas = gpd.read_file(aggr_areas_path)

    # Assert error when unknown file format is given
    with pytest.raises(ValueError):
        AggregationAreas.write_spatial_file(
            metrics, aggr_areas, outpath, file_format="matlab"
        )

    outpath = file_path / f"aggregation_areas_{name}.shp"

    # Assert error when there is a file_format and extension mismatch
    with pytest.raises(ValueError):
        AggregationAreas.write_spatial_file(
            metrics, aggr_areas, outpath, file_format="geopackage"
        )

    AggregationAreas.write_spatial_file(
        metrics, aggr_areas, outpath, file_format="shapefile"
    )

    assert outpath.exists()
    # Delete created files
    for file in list(outpath.parent.glob(outpath.stem + ".*")):
        file.unlink()

    outpath = file_path / f"aggregation_areas_{name}.geojson"
    AggregationAreas.write_spatial_file(
        metrics, aggr_areas, outpath, file_format="GeoJSON"
    )

    assert outpath.exists()
    # Delete created files
    for file in list(outpath.parent.glob(outpath.stem + ".*")):
        file.unlink()
