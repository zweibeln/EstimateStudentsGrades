import csv


def load_student_data(lecture):
    """
    :param lecture: string->'mat' or 'por'
    :return: list of lists containing students' information
    """
    if lecture not in ['mat', 'por']:
        raise ValueError(f'Lecture {lecture} does not exist. Only mat or por is accepted as lecture')
    file = open(f'student-{lecture}.csv')  # Import the data for desired class: mat/por
    csvreader = csv.reader(file)
    data = []
    for row in csvreader:
        data.append(row)
    file.close()
    return data
