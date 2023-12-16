from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from optparse import OptionParser
import xgboost as xgb
import pandas as pd
import numpy as np
import copy
import shap
import os


def get_model(param=None):
    if options.model == 'best':
        best_param = {
            'n_estimators': 600,
            'max_depth': 7,
            'min_child_weight': 1,
            'min_split_loss': 0,
            'reg_lambda': 10.0,
            'learning_rate': 0.1,
            'random_state': random_seed
        }
        return xgb.XGBRegressor(**best_param)

    elif options.model == 'xgb':
        if param is not None:
            return xgb.XGBRegressor(**param)
        else:
            return xgb.XGBRegressor(random_state=random_seed)

    elif options.model == 'dt':
        return DecisionTreeRegressor(random_state=random_seed)

    else:
        return LinearRegression()


def get_model_score(param, scoring, num_folder=5):
    local_score = 0
    for train_idx_local, valid_idx_local in KFold(n_splits=num_folder).split(X, y):
        X_train_local, y_train_local = X[train_idx_local], y[train_idx_local]
        X_valid_local, y_valid_local = X[valid_idx_local], y[valid_idx_local]
        model_local = xgb.XGBRegressor(**param)
        model_local.fit(X_train_local, y_train_local)
        pred_local = model_local.predict(X_valid_local)
        local_score += scoring(y_valid_local, pred_local)
    return local_score / num_folder


def find_parameters():
    parameters = {
        'n_estimators': 300,
        'max_depth': 6,
        'min_child_weight': 1,
        'min_split_loss': 0,
        'reg_lambda': 1.0,
        'learning_rate': 0.1,
        'random_state': random_seed
    }
    candidate = {
        'n_estimators': np.arange(25, 1001, 25, dtype=int),
        'max_depth': np.linspace(1, 10, 10, dtype=int),
        'min_child_weight': np.linspace(1, 57, 15, dtype=int),
        'min_split_loss': np.linspace(0, 0.5, 11),
        'reg_lambda': np.linspace(0, 20, 21),
    }

    best_score = 0
    for current_param in candidate.keys():
        print(f"Finding best parameter: {current_param}")
        for value in candidate[current_param]:
            local_param = copy.deepcopy(parameters)
            local_param[current_param] = value
            score = get_model_score(local_param, r2_score)
            print(f"\t{value}: r2_score - {round(score, 4)}")
            if score > best_score:
                best_score = score
                parameters[current_param] = value
        print(f"\tCurrent parameters: {parameters}\n")
    return best_score


if __name__ == '__main__':
    random_seed = 42
    parser = OptionParser()
    parser.add_option('--train', dest='train', type=str, help='train csv path', default=None)
    parser.add_option('--test', dest='test', type=str, help='test csv path', default=None)
    parser.add_option('--model', dest='model', type=str, help='used model', default='best')
    parser.add_option('-g', dest='grid', help='see grid search', action='store_true', default=False)
    parser.add_option('-s', dest='shap', help='show shap value', action='store_true', default=False)
    parser.add_option('-o', dest='out', help='output prediction', action='store_true', default=False)
    options, args = parser.parse_args()
    if options.train is None or options.test is None:
        print('Error: No training file or testing file.')
        exit(1)

    # train
    train_df = pd.read_csv(options.train, encoding='latin-1')
    train_df.drop(['title', 'snippet', 'company'], inplace=True, axis=1)
    X = train_df.drop(['salary_min', 'salary_max'], axis=1).to_numpy(dtype=float)
    y = train_df['salary_min'].to_numpy()
    kf = KFold(n_splits=5)

    if options.grid:
        find_parameters()

    models = []
    MSE, RMSE, MAE, MAPE, R2 = [], [], [], [], []
    for train_idx, valid_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        model = get_model()
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        MSE.append(mean_squared_error(y_valid, pred, squared=True))
        RMSE.append(mean_squared_error(y_valid, pred, squared=False))
        MAE.append(mean_absolute_error(y_valid, pred))
        MAPE.append(mean_absolute_percentage_error(y_valid, pred))
        R2.append(r2_score(y_valid, pred))
        models.append(model)

    show = 4
    print("Valid Score-")
    print(f'\tMSE ({np.mean(MSE).round(show)}):\n\t\t{list(np.round(MSE, show))}\n')
    print(f'\tRMSE ({np.mean(RMSE).round(show)}):\n\t\t{list(np.round(RMSE, show))}\n')
    print(f'\tMAE ({np.mean(MAE).round(show)}):\n\t\t{list(np.round(MAE, show))}\n')
    print(f'\tMAPE ({(np.mean(MAPE)).round(show)}):\n\t\t{list(np.round(MAPE, show))}\n')
    print(f'\tR2 Score ({np.mean(R2).round(show)}):\n\t\t{list(np.round(R2, show))}\n')

    # test
    test_df = pd.read_csv(options.test, encoding='latin-1')
    test_df.drop(['title', 'snippet', 'company'], inplace=True, axis=1)
    X_test = test_df.drop(['salary_min', 'salary_max'], axis=1).to_numpy(dtype=float)
    y_test = test_df['salary_min'].to_numpy()

    test_model = get_model()
    test_model.fit(X, y)
    y_pred = test_model.predict(X_test)

    print("Test Score-")
    print(f'\tMSE: {mean_squared_error(y_test, y_pred, squared=True)}')
    print(f'\tRMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
    print(f'\tMAE: {mean_absolute_error(y_test, y_pred)}')
    print(f'\tMAPE: {mean_absolute_percentage_error(y_test, y_pred)}')
    print(f'\tR2 Score: {r2_score(y_test, y_pred)}\n')

    if options.out:
        head, _ = os.path.split(options.test)
        output_name = os.path.join(head, 'min_salary_predict.csv')
        y_pred_df = pd.DataFrame(y_pred, columns=['min_salary_predict'])
        y_pred_df.to_csv(output_name, index=False, encoding='latin-1')
        print(f'Write prediction file to {output_name}\n')

    if options.shap:
        explainer = shap.Explainer(test_model)
        shap_values = explainer(test_df.drop(['salary_min', 'salary_max'], axis=1))
        shap.summary_plot(shap_values, test_df.drop(['salary_min', 'salary_max'], axis=1))
