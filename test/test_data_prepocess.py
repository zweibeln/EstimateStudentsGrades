from data_preprocess import check_missing_values, encode_nonnumeric_values
import pandas as pd
import pytest


def test_check_missing_values():
    df = {'idx': [1, 2, 3, 4, 5],
          'letter': ['a', 'b', 'c', 'd', 'e'],
          'value': ['red', None, 'blue', 'blue', 'yellow']}

    df = pd.DataFrame(df, columns=['idx', 'letter', 'value', 'converted_tf'])
    assert pytest.raises(ValueError, check_missing_values, df)


def test_encoding_features():
    df = {'idx': [1, 2, 3, 4, 5],
          'letter': ['a', 'b', 'c', 'd', 'e'],
          'value': ['red', 'purple', 'blue', 'green', 'yellow']}
    df = pd.DataFrame(df, columns=['idx', 'letter', 'value'])
    df_encoded = encode_nonnumeric_values(df)
    for dtype in df_encoded.dtypes:
        assert dtype == 'int64' or dtype == 'uint8'
