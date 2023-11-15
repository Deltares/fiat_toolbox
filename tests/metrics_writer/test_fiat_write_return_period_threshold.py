import numpy as np
import pandas as pd

from fiat_toolbox.metrics_writer.fiat_write_return_period_threshold import (
    ExceedanceProbabilityCalculator,  # replace with the actual module name
)

# Test data
df = pd.DataFrame(
    {
        "something (2Y)": [0, 0.1, 0.2],
        "something (5Y)": [0, 0.2, 0.4],
        "something (10Y)": [0, 0.3, 0.6],
        "something (25Y)": [0.4, 0.6, 0.8],
        "something (50Y)": [0.9, 1.0, 1.1],
    }
)

# Expected result for the test data
expected = pd.DataFrame(
    {
        "Exceedance Probability": [82.0, 99.8, 100.0],
    }
)

# Test case for the calculate method
def test_calculate():
    # Arrange
    calculator = ExceedanceProbabilityCalculator("something")

    # Act
    result = calculator.calculate(df, threshold=0.2, T=30)

    # Assert
    pd.testing.assert_frame_equal(result, expected)

def test_append_probability():
    # Arrange
    calculator = ExceedanceProbabilityCalculator("something")

    # Act
    result = calculator.append_probability(df, threshold=0.2, T=30)

    # Assert
    # Result looks like:
    #    something (2Y)  something (5Y)  something (10Y)  something (25Y)  something (50Y)  Exceedance Probability
    # 0             0.0             0.0              0.0              0.4              0.9                    82.0
    # 1             0.1             0.2              0.3              0.6              1.0                    99.8
    # 2             0.2             0.4              0.6              0.8              1.1                   100.0
    pd.testing.assert_frame_equal(result, df.join(expected))

# Test case for the calculate method with some nan values. If only a single value in a row is a nan but a result can still be found, the result should be calculated.
def test_calculate_some_nan():
    # Arrange
    # Arrange input as:
    #    something (2Y)  something (5Y)  something (10Y)  something (25Y)  something (50Y)
    # 0             NaN             NaN              0.0              0.4              0.9
    # 1             0.1             0.2              0.3              0.6              1.0
    # 2             0.2             0.4              0.6              0.8              1.1
    df_nan = df.copy()
    df_nan.iloc[0, 0:2] = np.nan
    calculator = ExceedanceProbabilityCalculator("something")

    # Act
    result = calculator.calculate(df_nan, threshold=0.2, T=30)

    # Assert
    pd.testing.assert_frame_equal(result, expected)

# Test case for the calculate method with a row with only nan's. If all values in a row are nan, the result should be nan.
def test_calculate_full_row_nan():
    # Arrange
    # Arrange input as:
    #    something (2Y)  something (5Y)  something (10Y)  something (25Y)  something (50Y)
    # 0             NaN             NaN              NaN              NaN              NaN
    # 1             0.1             0.2              0.3              0.6              1.0
    # 2             0.2             0.4              0.6              0.8              1.1
    df_nan = df.copy()
    df_nan.iloc[0, :] = np.nan

    # Expected result looks like:
    #    Exceedance Probability
    # 0                    NaN
    # 1                   99.8
    # 2                  100.0
    expected_nan = expected.copy()
    expected_nan.iloc[0] = np.nan
    calculator = ExceedanceProbabilityCalculator("something")

    # Act
    result = calculator.calculate(df_nan, threshold=0.2, T=30)

    # Assert
    pd.testing.assert_frame_equal(result, expected_nan)

# Test case for the calculate method with all nan values
def test_calculate_with_all_nan():
    # Arrange
    # Arrange input as:
    #    something (2Y)  something (5Y)  something (10Y)  something (25Y)  something (50Y)
    # 0             NaN             NaN              NaN              NaN              NaN
    # 1             NaN             NaN              NaN              NaN              NaN
    # 2             NaN             NaN              NaN              NaN              NaN
    df_nan = df.copy()
    df_nan[:] = np.nan

    # Expected result looks like:
    #    Exceedance Probability
    # 0                    NaN
    # 1                    NaN
    # 2                    NaN
    expected_nan = expected.copy()
    expected_nan["Exceedance Probability"] = np.nan
    calculator = ExceedanceProbabilityCalculator("something")

    # Act
    result = calculator.calculate(df_nan, threshold=0.2, T=30)

    # Assert
    pd.testing.assert_frame_equal(result, expected_nan)