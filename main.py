from student_data_loader import load_student_data
from data_preprocess import *
from regression_models import *

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
    model_name = list(model_dict.keys())
    for name in model_name:
        model = model_dict[name]['model']
        print(f'Current Used Model : {name}')
        model.fit(X_train, y_train)

        # evaluate model by predicting G3 values
        y_pred = model.predict(X_test)
        model_dict[name]['mse'] = ((y_test-y_pred)**2).sum()/len(y_pred)
        model_dict[name]['R2'] = model.score(X_test,y_test)
        print(f'Model {name} MSE = {model_dict[name]["mse"]}')
        print(f'Model {name} R2 score = {model_dict[name]["R2"]}')

    # compare model performances on a plot
    mse = []
    R2 = []
    for name in model_name:
        mse.append(model_dict[name]['mse'])
        R2.append(model_dict[name]['R2'])
    plt.figure(figsize=(12, 12))
    plt.plot(model_dict.keys(), mse, '-*r', markersize=12)
    for name, m in zip(model_name, mse):
        if name == 'ElasticNet':
            plt.text(name, m, f'{name}', rotation=45, style='italic', fontweight='bold',
                     horizontalalignment='right', verticalalignment='top')
        else:
            plt.text(name, m, f'{name}', rotation=45, style='italic', fontweight='bold')
    plt.title('Model Estimation Performances Based On MSE', fontsize=18)
    plt.ylabel('MSE', fontsize=12)
    plt.xlabel('Model Name', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.savefig('figures/MSEcomparison.png')

    plt.figure(figsize=(12, 12))
    plt.plot(model_dict.keys(), R2, '-*b', markersize=12)
    for name, r in zip(model_name, R2):
        if name == 'RnmForest':
            plt.text(name, r, f'{name}', rotation=45, style='italic', fontweight='bold',
                     horizontalalignment='right', verticalalignment='top')
        else:
            plt.text(name, r, f'{name}', rotation=45, style='italic', fontweight='bold')
    plt.title('Model Estimation Performances Based On R2', fontsize=12)
    plt.ylabel('R2', fontsize=12)
    plt.xlabel('Model Name', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.savefig('figures/R2comparison.png')

__main__()
