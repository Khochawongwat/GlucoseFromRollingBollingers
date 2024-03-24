import os
import pickle
import time
import warnings
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Ridge

import optuna
from sklearn.model_selection import cross_val_score, learning_curve

import matplotlib.pyplot as plt
from tqdm import tqdm

from losses import criterion

import numpy as np
import pandas as pd

from utils import fit_navigator_model, get_navigator_prediction, step_transform

class HybridModel:
    def __init__(self, use_navigator = True):
        self.model = {
            "Base": LGBMRegressor(verbose = -1),
            "Residuals": Ridge(),
            "Navigator": LGBMClassifier(verbose = -1)
        }
        
        self.quantile = {
            "Lower": LGBMRegressor(objective="quantile", alpha=0.01, verbose = -1),
            "Median": LGBMRegressor(objective="quantile", alpha=0.5, verbose = -1),
            "Upper": LGBMRegressor(objective="quantile", alpha=0.99, verbose = -1)
        }
        self.use_navigator = use_navigator

    def optimize_lgbm_params(self, X, y, prune = True, n_trials = 10, cv = 5, verbose = 1):
        print("Optimizing Base LGBM parameters...")
        warnings.filterwarnings('ignore', category=UserWarning)
        def objective(trial):
            param = {
                'random_state': 0,
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
                'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.2, log = True),
                'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
                'num_leaves' : trial.suggest_int('num_leaves', 1, 500),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100),
            }

            model = LGBMRegressor(**param)
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
            
            if verbose:
                train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
                train_scores_mean = -train_scores.mean(axis=1)
                test_scores_mean = -test_scores.mean(axis=1)

                plt.plot(train_sizes, train_scores_mean, label='Training score')
                plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
                plt.title('Learning Curve')
                plt.xlabel('Training Size')
                plt.ylabel('Score')
                plt.legend(loc='best')
                plt.show()

            if prune:
                for step, score in enumerate(scores):
                    trial.report(score, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    
            return scores.mean()

        study = optuna.create_study(direction='maximize')

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        lgbm_base_params = study.best_trial.params
        
        os.makedirs("params", exist_ok = True)

        with open(f'params/lgbm_params_{time.time()}.pkl', 'wb') as f:
            pickle.dump(lgbm_base_params, f)
            
        return lgbm_base_params

    def optimize_ridge_params(self, X, y, testX = None, testY = None):
        print("Optimizing Ridge parameters...")
        def objective(trial, X, y):
            params = {
                'alpha': trial.suggest_float('alpha', 1e-4, 10, log = True),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [False]),
                'max_iter': trial.suggest_int('max_iter', 1, 1000, log = True),
                'solver': trial.suggest_categorical('solver', ['auto']),
            }

            ridge = Ridge(**params)
            return cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_squared_error').mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X, y), n_trials = 100)

        ridge_best_params = study.best_params

        os.makedirs("params", exist_ok = True)

        with open(f'params/ridge_best_params_{time.time()}.pkl', 'wb') as f:
            pickle.dump(ridge_best_params, f)
        
        print(ridge_best_params)

        ridge_best = Ridge(**ridge_best_params)
        ridge_best.fit(X, y)
        
        if testX is not None and testY is not None:
            print(ridge_best.score(testX, testY))

        return ridge_best_params
    
    def fit(self, X, y, testX = None,  testY = None, eval = True, tune = True) -> None:
        assert not (eval == True and (testX is None and testY is None)), "Testset is required for evaluation."
    
        if self.use_navigator:
            self.model["Navigator"] = fit_navigator_model(self.model["Navigator"], X, y)
            X["direction"] = get_navigator_prediction(self.model["Navigator"], X)
            
        if tune:
            lgbm_base_params = self.optimize_lgbm_params(X, y)
            if os.path.exists("params"):
                with open("params/ridge_best_params_1711182709.153041.pkl", "rb") as f:
                    ridge_base_params = pickle.load(f)
                    print(f"Loaded Ridge params {ridge_base_params} from file.")
            else:
                ridge_base_params = self.optimize_ridge_params(X, y, testX, testY)
            
            self.model["Base"].set_params(**lgbm_base_params)
            self.model["Residuals"].set_params(**ridge_base_params)

        self.model["Base"].fit(X, y)

        self.quantile["Lower"].fit(X, y)
        self.quantile["Median"].fit(X, y)
        self.quantile["Upper"].fit(X, y)

        self.model["Residuals"].fit(X, y - self.model["Base"].predict(X))

        print(f"Base fitted with columns: {X.columns}")

        if eval:
            assert (testX is not None and testY is not None), "testX and testY must be provided for evaluation"

            testX['direction'] = get_navigator_prediction(self.model["Navigator"], testX)
            
            base_pred = self.model["Base"].predict(testX)
            residuals_pred = self.model["Residuals"].predict(testX)

            total_pred = base_pred + residuals_pred
            print(f"Base: {criterion(base_pred, testY)}")
            print(f"Base + Residuals: {criterion(total_pred, testY)} Change: {100 -  abs(criterion(base_pred, testY)[1] - criterion(total_pred, testY)[1] / criterion(base_pred, testY)[1] * 100)}%")

    def forecast(self, X, n_steps = 1, return_X = True, use_confi = True) -> np.array:
        assert n_steps > 0, "n_steps must be greater than 0"
        assert len(X) > 0 or not (X is None), "X must have at least one row and not be None"
        
        forecasts = []
        confi_forecasts = {"Lower": [], "Median": [], "Upper": []}
        
        for _ in tqdm(range(n_steps), "Forecasting"):
            last_row = X.iloc[-1:, :]
            base_pred = self.model["Base"].predict(last_row)
            residuals_pred = self.model["Residuals"].predict(last_row)
            total_pred = base_pred + residuals_pred
            forecasts.append(total_pred)

            if use_confi:
                for quantile in ["Lower", "Median", "Upper"]:
                    confi_pred = self.quantile[quantile].predict(last_row)
                    confi_forecasts[quantile].append(confi_pred)

            new_row = step_transform(X, total_pred, self.use_navigator, self.model["Navigator"]).iloc[-1:, :]
            new_row.index = [X.index[-1] + 1]
            X = pd.concat([X, new_row])
        
        forecasts = np.array(forecasts).flatten()

        if use_confi:
            for quantile in ["Lower", "Median", "Upper"]:
                confi_forecasts[quantile] = np.array(confi_forecasts[quantile]).flatten()

        result = {"X": X, "forecasts": forecasts, "confi_forecasts": confi_forecasts} if return_X else {"forecasts": forecasts, "confi_forecasts": confi_forecasts}

        return result