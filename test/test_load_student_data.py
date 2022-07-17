from student_data_loader import load_student_data
import os
import pytest

os.chdir('..')


def test_load_student_data():
    assert isinstance(load_student_data('mat')[0], list)
    assert (len(load_student_data('mat')[0]) + len(load_student_data('mat')[1]) +
            len(load_student_data('mat')[2])) == 395  # Total length of full data
    assert len(load_student_data('mat')[0][0]) == 33
    assert len(load_student_data('mat')[0]) == 237  # Length of training set
    assert len(load_student_data('mat')[1]) == 79  # Length of validation set
    assert len(load_student_data('mat')[2]) == 79  # Length of test set
    assert pytest.raises(ValueError, load_student_data, 'cmb')
