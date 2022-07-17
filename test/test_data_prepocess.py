from data_preprocess import check_missing_values
import pandas as pd
import pytest


def test_check_missing_values():
    df = {'idx': [1, 2, 3, 4, 5],
          'letter': ['a', 'b', 'c', 'd', 'e'],
          'value': ['red', None, 'blue', 'blue', 'yellow']}

    df = pd.DataFrame(df, columns=['idx', 'letter', 'value', 'converted_tf'])
    assert pytest.raises(ValueError, check_missing_values, df)

