import csv
import random


def load_student_data(lecture):
    """
    :param lecture: string->'mat' or 'por'
    :return: list of lists containing students' information
    """
    if lecture not in ['mat', 'por']:
        raise ValueError(f'Lecture {lecture} does not exist. Only mat or por is accepted as lecture')
    file = open(f'student-{lecture}.csv')  # Import the data for desired class: mat/por
    csvreader = csv.reader(file, delimiter=';')  # Read the data
    data = []
    for row in csvreader:
        data.append(row)
    file.close()
    # Split data into train/val/test sets  [%60,%20,%20]
    data = data[1:]
    random.shuffle(data)
    data_train = data[0:int(len(data)*0.6)]
    data_val = data[int(len(data)*0.6):int(len(data)*0.6 + len(data)*0.2)]
    data_test = data[int(len(data)*0.6 + len(data)*0.2):]
    return data_train, data_val, data_test
