import os
import pickle
import time
import warnings
from lightgbm import LGBMRegressor
import numpy as np
import optuna
import pandas as pd
from sklearn import pipeline
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score, learning_curve
from processors import (OutlierTransformer, 
                        BollingersTransformer)

class Model:
    def __init__(self, y: str, dataloader: any, window_size = 20, std = 2) -> None:
        assert isinstance(y, str), 'y should be a target name'

        self.model = {
            "Base": LGBMRegressor(verbose = -1),
            "Residual": LGBMRegressor(verbose = -1),
        }

        self.features = None
        self.y = y

        self.window_size = window_size
        self.std = std
    
        self._train = dataloader['train']
        _train = pd.concat(dataloader['train'])

        self.pipe = pipeline.Pipeline([
            ("outlier", OutlierTransformer(_train)),
            ("bb", BollingersTransformer(window_size, std = std)),
        ])
        
    def optimize_lgbm_params(self, X, prune = True, n_trials = 1000, cv = 10):

        X = self.pipe.fit_transform(X)
        y = X[self.y]
        X = X.drop(columns=[self.y])

        print("Optimizing Base LGBM parameters...")
        warnings.filterwarnings('ignore', category=UserWarning)
        def objective(trial):
            param = {
                'random_state': 0,
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log = True),  # Adjusted range
                'max_depth': trial.suggest_categorical('max_depth', [5,10,15,20,25,30]),  # Adjusted values
                'num_leaves' : trial.suggest_int('num_leaves', 2, 1000),  # Adjusted range
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),  # New parameter
                'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1, log=True),  # New parameter
                'verbose': -1,
            }
            model = LGBMRegressor(**param)
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

            if prune:
                for step, score in enumerate(scores):
                    trial.report(score, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    
            return scores.mean()

        study = optuna.create_study(direction='maximize', storage='sqlite:///my_study.db')

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        lgbm_base_params = study.best_trial.params
        
        os.makedirs("params", exist_ok = True)

        with open(f'params/lgbm_params_{time.time()}.pkl', 'wb') as f:
            pickle.dump(lgbm_base_params, f)
        
        self.model["Base"] = LGBMRegressor(**lgbm_base_params)
        print("Optimizing LGBM parameters...")
        return lgbm_base_params
    
    def fit(self) -> None:
        train = [self.pipe.fit_transform(T) for T in self._train]
        train = pd.concat(train)
    
        X, y = train.drop(columns=[self.y]), train[self.y]

        if self.features is None:
            self.features = X.columns.values
            print(f"Fitting with features: {self.features}")

        self._fit_model(X, y)

    def _fit_model(self, X, y):
        try:
            self.model["Base"].fit(X, y, init_model=self.model["Base"])
            self.model["Residual"].fit(X, y - self.model["Base"].predict(X), init_model=self.model["Residual"])
        except NotFittedError:
            self.model["Base"].fit(X, y)
            self.model["Residual"].fit(X, y - self.model["Base"].predict(X))

    def evaluate(self, X, y) -> float:
        assert isinstance(X, pd.DataFrame), 'X should be a DataFrame'
        assert isinstance(y, pd.Series), 'y should be a Series'
        X = self.pipe.transform(X)

        X = X.drop(columns=[self.y])
        return np.mean((self.model["Base"].predict(X) + self.model["Residual"].predict(X) - y) ** 2)
    
    def predict(self, X) -> pd.Series:
        assert isinstance(X, pd.DataFrame), 'X should be a DataFrame'
        X = X.copy()
        X = self.pipe.transform(X)
        X = X.drop(columns=[self.y])
        return pd.Series(self.model["Base"].predict(X), name=self.y) + pd.Series(self.model["Residual"].predict(X), name=self.y + "_residual")

    
    def _create_new_row(self, X, pred, first):
        prev = X.tail(1).copy()
        new_row = prev.copy()

        new_row['prev'] = pred
        if not first:
            new_row.index = pd.to_datetime(new_row.index)
            new_row.index = new_row.index + pd.DateOffset(minutes = 5)
        return new_row

    def forecast(self, X, n_steps = 24) -> pd.Series:
        assert isinstance(X, pd.DataFrame), 'X should be a DataFrame'
        X = X.copy()
        X = self.pipe.transform(X)
        X = X.drop(columns=[self.y])
        
        for i in range(n_steps):
            last_row = X.tail(1)
            pred = self.model["Base"].predict(last_row) + self.model["Residual"].predict(last_row)
            bb_transformer = BollingersTransformer(window_size=self.window_size, std=self.std, attribute='prev')
            new_row = self._create_new_row(X, pred, first = i == 0)
            X = pd.concat([X, new_row])
            X = bb_transformer.transform(X)
        return X.tail(n_steps)