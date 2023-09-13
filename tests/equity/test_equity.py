from pathlib import Path

import pytest

from fiat_toolbox.equity.equity import Equity

DATASET = Path(__file__).parent / "data"

_cases = {
    "equity": {
        "census_data": "population_income_data.csv",
        "fiat_data": "aggregated_damage.csv",
        "aggregation_label": "Census_Bg",
        "percapitalincome_label": "PerCapitaIncomeBG",
        "totalpopulation_label": "TotalPopulationBG",
        "gamma": 1.2,
        "output_file_equity": "aggregated_ewced.csv",
    }
}


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_equity(case):
    census_data = DATASET.joinpath(_cases[case]["census_data"])
    fiat_data = DATASET.joinpath(_cases[case]["fiat_data"])
    aggregation_label = _cases[case]["aggregation_label"]
    percapitalincome_label = _cases[case]["percapitalincome_label"]
    totalpopulation_label = _cases[case]["totalpopulation_label"]
    gamma = _cases[case]["gamma"]
    output_file_equity = DATASET.joinpath(_cases[case]["output_file_equity"])

    equity = Equity(method="JarlKind")

    df_equity = equity.setup_equity_method(
        census_data,
        fiat_data,
        aggregation_label,
        percapitalincome_label,
        totalpopulation_label,
        gamma,
        output_file_equity,
    )

    assert "EWCEAD" in df_equity.columns

    # Delete file
    output_file_equity.unlink()
