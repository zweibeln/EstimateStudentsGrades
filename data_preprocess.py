import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def visualize_correlation_of_features(data):
    if type(data) != pd.DataFrame:
        raise ValueError('Input data is not Pandas DataFrame')
    f = plt.figure(figsize=(10, 10))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14,
               rotation=45)
    plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Feature Correlation Matrix', fontsize=16)


def check_missing_values(data):
    if data.isnull().sum().sum() > 0:
        raise ValueError('Some of the data is missing')
    else:
        pass


def encode_nonnumeric_values(data):
    # get nonnumeric column names
    nonnumeric_cloumns = [data.columns[index] for index,dtype in enumerate(data.dtypes) if dtype == 'object']

    # before encoding make sure not a single value repeats itself in another column so make them uniue wrt columns
    for name in nonnumeric_cloumns:
        data[name] = data[name].apply(lambda x: name[0]+x)

    # one hot encode the nonnumeric values
    dummy_mtx = pd.DataFrame()
    for name in nonnumeric_cloumns:
        dummy_mtx = pd.concat([dummy_mtx, pd.get_dummies(data[name])], axis=1)

    # merge the encoded values with the original data
    data = pd.concat([data, dummy_mtx], axis=1)

    # drop the nonnumeric values from data
    for name in nonnumeric_cloumns:
        data.drop([name], axis=1, inplace=True)

    return data
