from student_data_loader import load_student_data
from data_preprocess import *
from sklearn.linear_model import LinearRegression


def __main__():
    # load the data
    data = load_student_data('mat')

    # visualize the correlation between features
    visualize_correlation_of_features(data)

    # check if exist any missing value in data
    check_missing_values(data)

    # encode nonnumeric values
    data_encoded = encode_nonnumeric_values(data)

    # split the data_set into train and test
    X_train, X_test, y_train, y_test = scale_split_data(data_encoded)

    # select model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # evaluate model by predicting G3 values
    y_pred = model.predict(X_test)
    mse = ((y_test-y_pred)**2).sum()/len(y_pred)
    print(f'Model MSE = {mse}')
    print(f'Model R2 score = {model.score(X_test,y_test)}')

__main__()
