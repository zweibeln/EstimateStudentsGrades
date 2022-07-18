from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


model_dict = {'LinearRegression': {'model': LinearRegression(),
                                     'mse': 0,
                                     'R2' : 0},
                         'Ridge': {'model': Ridge(alpha=0.1),
                                     'mse': 0,
                                   'R2' : 0},
                         'Lasso': {'model': Lasso(alpha=0.1),
                                     'mse': 0,
                                     'R2' : 0},
                    'ElasticNet': {'model': ElasticNet(),
                                     'mse': 0,
                                     'R2' : 0},
                 'BayesianRidge': {'model': BayesianRidge(),
                                     'mse': 0,
                                     'R2' : 0},
                           'ARD': {'model': ARDRegression(),
                                     'mse': 0,
                                     'R2' : 0},
                           'SGD': {'model': SGDRegressor(),
                                     'mse': 0,
                                     'R2' : 0},
                 'DecisionTree' : {'model': DecisionTreeRegressor(random_state=0),
                                     'mse': 0,
                                     'R2' : 0},
                    'RnmForest' : {'model': RandomForestRegressor(n_estimators=50),
                                     'mse': 0,
                                     'R2' : 0},
}