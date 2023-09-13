from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from fiat_toolbox.spatial_output.points_to_footprint import PointsToFootprints

file_path = Path(__file__).parent.resolve()


def test_write_footprints_event():
    # Get footprints file
    footprints_path = file_path / "data" / "building_footprints.geojson"
    # Get fiat results file
    results_path = file_path / "data" / "output_event.csv"

    footprints = gpd.read_file(footprints_path)
    results = pd.read_csv(results_path)

    # Define output name
    outpath = file_path / "building_footprints_event.gpkg"
    out = PointsToFootprints.write_footprint_file(footprints, results, outpath)

    out_example = out["Total Damage"][out["Object ID"] == "1393_1394"].to_numpy()[0]
    in_example = (
        results["Total Damage"][results["Object ID"] == 1393].to_numpy()[0]
        + results["Total Damage"][results["Object ID"] == 1394].to_numpy()[0]
    )
    assert out_example == in_example
    # Delete created files
    outpath.unlink()


def test_write_footprints_risk():
    # Get footprints file
    footprints_path = file_path / "data" / "building_footprints.geojson"
    # Get fiat results file
    results_path = file_path / "data" / "output_risk.csv"

    footprints = gpd.read_file(footprints_path)
    results = pd.read_csv(results_path)

    # Define output name
    outpath = file_path / "building_footprints_risk.gpkg"
    out = PointsToFootprints.write_footprint_file(footprints, results, outpath)

    out_example = out["Risk (EAD)"][out["Object ID"] == "1393_1394"].to_numpy()[0]
    in_example = (
        results["Risk (EAD)"][results["Object ID"] == 1393].to_numpy()[0]
        + results["Risk (EAD)"][results["Object ID"] == 1394].to_numpy()[0]
    )
    assert out_example == in_example
    # Delete created files
    outpath.unlink()


def test_error_handling():
    # Get footprints file
    footprints_path = file_path / "data" / "building_footprints.geojson"
    # Get fiat results file
    results_path = file_path / "data" / "output_risk.csv"

    footprints = gpd.read_file(footprints_path)
    results = pd.read_csv(results_path)
    del results["Risk (EAD)"]
    # Define output name
    outpath = file_path / "building_footprints_risk.gpkg"
    with pytest.raises(ValueError):
        PointsToFootprints.write_footprint_file(footprints, results, outpath)
