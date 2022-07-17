import pandas as pd
import matplotlib.pyplot as plt


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
