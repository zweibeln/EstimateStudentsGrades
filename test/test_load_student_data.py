from student_data_loader import load_student_data
import os
import pytest
import pandas as pd

os.chdir('..')


def test_load_student_data():
    assert type(load_student_data('mat')) == pd.DataFrame
    assert len(load_student_data('mat')) == 395
    assert pytest.raises(ValueError, load_student_data, 'cmb')
