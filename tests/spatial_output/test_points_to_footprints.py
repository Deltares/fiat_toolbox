from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from fiat_toolbox.spatial_output.footprints import Footprints
from fiat_toolbox.spatial_output.footprints import Fiat
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
    
    # Aggregate results
    footprints = Footprints(footprints)
    footprints.aggregate(results)
    footprints.calc_normalized_damages()
    footprints.write(outpath)
    
    out = footprints.aggregated_results
    
    out_example = out[Fiat.total_damage][out[Fiat.object_id] == "1393_1394"].to_numpy()[0]
    in_example = (
        results[Fiat.total_damage][results[Fiat.object_id] == 1393].to_numpy()[0]
        + results[Fiat.total_damage][results[Fiat.object_id] == 1394].to_numpy()[0]
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
    
    # Aggregate results
    footprints = Footprints(footprints)
    footprints.aggregate(results)
    footprints.calc_normalized_damages()
    footprints.write(outpath)
    
    out = footprints.aggregated_results


    out_example = out[Fiat.risk_ead][out[Fiat.object_id] == "1393_1394"].to_numpy()[0]
    in_example = (
        results[Fiat.risk_ead][results[Fiat.object_id] == 1393].to_numpy()[0]
        + results[Fiat.risk_ead][results[Fiat.object_id] == 1394].to_numpy()[0]
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
    del results[Fiat.risk_ead]

    with pytest.raises(ValueError):
        footprints = Footprints(footprints)
        footprints.aggregate(results)
