import pandas as pd


def load_student_data(lecture):
    """
    :param lecture: string->'mat' or 'por'
    :return: list of lists containing students' information
    """
    if lecture not in ['mat', 'por']:
        raise ValueError(f'Lecture {lecture} does not exist. Only mat or por is accepted as lecture')
    return pd.read_csv(f'student-{lecture}.csv', sep=';')

