from lightgbm import LGBMRegressor

import optuna
from sklearn.linear_model import (
    Ridge,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor
from losses import criterion
import numpy as np
import pandas as pd
from utils import *
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
   
class Models:
    def __init__(self):
        self.model = {
            "Linear_5": LGBMRegressor(objective="quantile", alpha=0.05),
            "Linear": LGBMRegressor(),
            "Linear_95": LGBMRegressor(objective="quantile", alpha=0.95),
            "Residuals": Ridge(),
        }
        print(self.model)
        
    def optimize_params(self, X, y):
        def objective(trial):
            param = {
                'metric': 'rmse', 
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 2, 100),
                'max_depth': trial.suggest_int('max_depth', -1, 50),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample_for_bin': 200000,
                'objective': None,
                'class_weight': None,
                'min_split_gain': 0.0,
                'min_child_weight': 0.001,
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
                'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                'subsample_freq': 0,
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, log=True),
                'random_state': 0,
                'n_jobs': -1,
                'silent': True,
                'importance_type': 'split',
                'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
            }
            
            model = LGBMRegressor(**param)
            score = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error').mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        return study.best_trial.params
    
    def fit(self, X, y, testX=None, testY=None, eval=True):

        # Get models
        linear_model_5 = self.model.get("Linear_5")
        linear_model_95 = self.model.get("Linear_95")
        residuals_model = self.model.get("Residuals")

        params = {'alpha': (np.logspace(-8, 8, 100))} 
        grid_search_residuals = GridSearchCV(residuals_model, params, scoring='neg_mean_squared_error', cv=10)

        best_params = self.optimize_params(X, y)

        self.model["Linear"] = LGBMRegressor(**best_params)
        linear_model = self.model["Linear"]

        linear_model_5.fit(X, y)
        linear_model.fit(X, y)
        linear_model_95.fit(X, y)

        residuals = y - linear_model.predict(X)
        
        grid_search_residuals.fit(X, residuals)

        self.model["Residuals"] = grid_search_residuals.best_estimator_

        if eval:
            assert (
                testX is not None and testY is not None
            ), "testX and testY must be provided for evaluation"
            
            residuals_model = self.model["Residuals"]

            pred_5 = linear_model_5.predict(testX)
            pred = linear_model.predict(testX)
            pred_95 = linear_model_95.predict(testX)

            res_pred = residuals_model.predict(testX)

            z = (res_pred - np.mean(res_pred)) / np.std(res_pred)

            norm_quantiles = stats.norm.ppf(np.linspace(0, 1, len(z)))

            plt.plot(norm_quantiles, np.sort(z), "b")
            plt.xlabel("Theoretical Quantiles")
            plt.ylabel("Ordered Values")
            plt.axhline(color="r")
            plt.show()

            plt.figure(figsize=(150, 15))
            plt.fill_between(
                range(len(pred_5)), pred_5, pred_95, color="black", alpha=0.2
            )
            plt.plot(pred, color="red", alpha=1)
            plt.show()

            plt.figure(figsize=(50, 5))
            plt.plot(testY - pred + res_pred, color="red", alpha=1)
            plt.ylabel("Test Residuals with residuals")

            plt.figure(figsize=(50, 5))
            plt.plot(testY - pred, color="blue", alpha=1)
            plt.ylabel("Test Residuals without residuals")

            overall_pred = pred + res_pred

            mse, rmse, mspe = self.eval(overall_pred, testY)
            print(f"With Residual, MSE: {mse}, RMSE: {rmse}, MSPE: {mspe} %")

            mse, rmse, mspe = self.eval(pred, testY)
            print(f"No Residual, MSE: {mse}, RMSE: {rmse}, MSPE: {mspe} %")

    def forecast(
        self,
        X,
        y=None,
        n=12,
        eval=True,
        save_model=True,
        path="models/best_forecasting_model.pkl",
    ):
        best_rmse = np.inf
        best_forecasts = None
        best_model = None
        for model_name, model in self.models.items():
            forecasts = []
            for i in range(n):
                prev_rows = X.iloc[-n:, :]
                last_row = prev_rows.iloc[-1:, :]
                pred = model.predict(last_row)
                new_row = step_transform(prev_rows, pred).iloc[-1:, :]
                forecasts.append(pred)
                X = pd.concat([X, new_row], ignore_index=True)
            forecasts = pd.DataFrame(forecasts, columns=["CGM"])

            if eval:
                assert len(y) > 0, "y must be provided for evaluation"
                assert len(y) == len(
                    forecasts
                ), "y and forecasts must have the same length"
                mse, rmse, mspe = criterion(forecasts, y, plot=True)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_forecasts = forecasts
                    best_model = model
                print(f"{model_name} MSE: {mse}, RMSE: {rmse}, MSPE: {mspe} %")

        if save_model and best_model is not None:
            assert path is not None, "path must be provided to save the model"

            dir_name = os.path.dirname(path)

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            with open(path, "wb") as f:
                pickle.dump(best_model, f)

        X = X.iloc[n - 1 : n * 2 - 1, :]
        X.reset_index(drop=True, inplace=True)
        X.loc[:, "pred"] = best_forecasts
        return X

    def eval(self, pred, y):
        return criterion(pred, y)
