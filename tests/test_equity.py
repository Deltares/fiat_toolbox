
from fiat_toolbox.equity.equity import *
import pytest
from pathlib import Path


DATASET = Path().absolute() / "local_test_database" / "data"

_cases = {

    "equity": {
        "census_data": "population_income_data.csv",
        "fiat_data": "aggregated_damage.csv",
        "gamma": 1.2,
        "output_file_equity": "aggregated_ewced.csv",
    }
}

@pytest.mark.parametrize("case", list(_cases.keys()))
def test_equity(case):

    census_data = DATASET.joinpath(_cases[case]["census_data"])
    fiat_data   = DATASET.joinpath(_cases[case]["fiat_data"])
    gamma       = _cases[case]["gamma"]
    output_file_equity       = DATASET.joinpath(_cases[case]["output_file_equity"])

    df_equity = setup_equity_method(census_data, fiat_data, gamma, output_file_equity)

    assert "EWCEAD" in df_equity.columns

    # Delete file
    output_file_equity.unlink()


