from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from fiat_toolbox import get_fiat_columns
from fiat_toolbox.spatial_output.footprints import Footprints

file_path = Path(__file__).parent.resolve()

_FIAT_VERSION = "0.1.0rc2"
_FIAT_COLUMNS = get_fiat_columns(fiat_version=_FIAT_VERSION)


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
    footprints = Footprints(footprints, field_name="BF_FID", fiat_version=_FIAT_VERSION)
    footprints.aggregate(results)
    footprints.calc_normalized_damages()
    footprints.write(outpath)

    out = footprints.aggregated_results

    out_example = out[_FIAT_COLUMNS.total_damage][
        out[_FIAT_COLUMNS.object_id] == "1393_1394"
    ].to_numpy()[0]
    in_example = (
        results[_FIAT_COLUMNS.total_damage][
            results[_FIAT_COLUMNS.object_id] == 1393
        ].to_numpy()[0]
        + results[_FIAT_COLUMNS.total_damage][
            results[_FIAT_COLUMNS.object_id] == 1394
        ].to_numpy()[0]
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
    footprints = Footprints(footprints, field_name="BF_FID", fiat_version=_FIAT_VERSION)
    footprints.aggregate(results)
    footprints.calc_normalized_damages()
    footprints.write(outpath)

    out = footprints.aggregated_results

    out_example = out[_FIAT_COLUMNS.risk_ead][
        out[_FIAT_COLUMNS.object_id] == "1393_1394"
    ].to_numpy()[0]
    in_example = (
        results[_FIAT_COLUMNS.risk_ead][
            results[_FIAT_COLUMNS.object_id] == 1393
        ].to_numpy()[0]
        + results[_FIAT_COLUMNS.risk_ead][
            results[_FIAT_COLUMNS.object_id] == 1394
        ].to_numpy()[0]
    )
    assert out_example == round(in_example)
    # Delete created files
    outpath.unlink()


def test_error_handling():
    # Get footprints file
    footprints_path = file_path / "data" / "building_footprints.geojson"
    # Get fiat results file
    results_path = file_path / "data" / "output_risk.csv"

    footprints = gpd.read_file(footprints_path)
    results = pd.read_csv(results_path)
    del results[_FIAT_COLUMNS.risk_ead]

    with pytest.raises(ValueError):
        footprints = Footprints(
            footprints, field_name="BF_FID", fiat_version=_FIAT_VERSION
        )
        footprints.aggregate(results)
