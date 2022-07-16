from student_data_loader import load_student_data
import os
import pytest

os.chdir('..')


def test_load_student_data():
    assert isinstance(load_student_data('mat'), list)
    assert len(load_student_data('mat')) == 396
    assert len(load_student_data('mat')[0][0]) == 227
    assert pytest.raises(ValueError, load_student_data, 'cmb')
