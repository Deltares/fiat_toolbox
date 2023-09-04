from pathlib import Path

import geopandas as gpd
import pandas as pd

from fiat_toolbox.spatial_output.points_to_footprint import PointsToFootprints

file_path = Path(__file__).parent.resolve()


def test_write_footprints_event():
    # Get footprints file
    footprints_path = file_path / "data" / "Buildings.shp"
    # Get fiat results file
    results_path = file_path / "data" / "output_event" / "output.csv"

    footprints = gpd.read_file(footprints_path)
    results = pd.read_csv(results_path)

    # Define output name
    outpath = file_path / "building_footprints_event.gpkg"
    PointsToFootprints.write_footprint_file(footprints, results, outpath)


def test_write_footprints_risk():
    # Get footprints file
    footprints_path = file_path / "data" / "Buildings.shp"
    # Get fiat results file
    results_path = file_path / "data" / "output_risk" / "output.csv"

    footprints = gpd.read_file(footprints_path)
    results = pd.read_csv(results_path)

    # Define output name
    outpath = file_path / "building_footprints_risk.gpkg"
    PointsToFootprints.write_footprint_file(footprints, results, outpath)


def test_error_handling():
    pass
