from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import shap


if __name__ == '__main__':

    random_seed = 42

    train_df = pd.read_csv("./dataset/train_preprocess.csv", encoding='latin-1')
    train_df.drop(['title', 'snippet', 'company'], inplace=True, axis=1)

    X = train_df.drop(['salary'], axis=1).to_numpy(dtype=float)
    y = train_df['salary'].to_numpy()
    kf = KFold(n_splits=5)

    models = []
    MSE, RMSE, MAE, MAPE, R2 = [], [], [], [], []
    for train_idx, valid_idx in kf.split(X, y):
        X_train, y_train, X_valid, y_valid = X[train_idx], y[train_idx], X[valid_idx], y[valid_idx]
        model = DecisionTreeRegressor(random_state=random_seed)
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
    print(f'\tMSE ({np.mean(MSE).round(show)}):\n\t\t{np.round(MSE, show)}\n')
    print(f'\tRMSE ({np.mean(RMSE).round(show)}):\n\t\t{np.round(RMSE, show)}\n')
    print(f'\tMAE ({np.mean(MAE).round(show)}):\n\t\t{np.round(MAE, show)}\n')
    print(f'\tMAPE ({(np.mean(MAPE)*100).round(show)}%):\n\t\t{np.round(MAPE, show)}\n')
    print(f'\tR2 Score ({np.mean(R2).round(show)}):\n\t\t{np.round(R2, show)}\n')

    # test
    train_df = pd.read_csv("./dataset/test_preprocess.csv", encoding='latin-1')
    train_df.drop(['title', 'snippet', 'company'], inplace=True, axis=1)

    X_test = train_df.drop(['salary'], axis=1).to_numpy(dtype=float)
    y_test = train_df['salary'].to_numpy()
    all_pred = np.array([test_model.predict(X_test) for test_model in models]).transpose()

    pred = []
    for d in all_pred:
        pred.append(sorted(list(d))[2])
    y_pred_df = pd.DataFrame(pred, columns=['min_salary_prediction'])
    y_pred_df.to_csv('./dataset/min_salary_predict.csv', index=False, encoding='latin-1')

    print("Test Score-")
    print(f'\tMSE: {mean_squared_error(y_test, pred, squared=True)}')
    print(f'\tRMSE: {mean_squared_error(y_test, pred, squared=False)}')
    print(f'\tMAE: {mean_absolute_error(y_test, pred)}')
    print(f'\tMAPE: {mean_absolute_percentage_error(y_test, pred)}')
    print(f'\tR2 Score: {r2_score(y_test, pred)}')

    # explainer = shap.Explainer(models[1])
    # shap_values = explainer(train_df.drop(['salary'], axis=1))
    # shap.summary_plot(shap_values, train_df.drop(['salary'], axis=1))
