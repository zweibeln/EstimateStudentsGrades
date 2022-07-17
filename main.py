import argparse
from student_data_loader import load_student_data


def __main__():
    # load the data
    data_train, data_val, data_test = load_student_data('mat')


__main__()
