from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from tqdm import tqdm
from xgboost import XGBRegressor
from losses import criterion
import numpy as np
import pandas as pd
from utils import *
import pickle

class Models:
    def __init__(self):
        self.models = {
            #"Ridge": Ridge(max_iter=1000),
            #"Lasso": Lasso(max_iter=2000),
            #"Elastic Net": ElasticNet(max_iter=3000),
            #"Random Forest": RandomForestRegressor(),
            #"Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor(verbose = -1),
            #"CatBoost": AdaBoostRegressor(),
        }
    
    def fit(self, X, y, testX = None, testY = None, eval = True):
        best_rmse = np.inf
        best_preds = None
        for model_name, model in tqdm(self.models.items()):
            model.fit(X, y)
            if eval:
                assert testX is not None and testY is not None, "testX and testY must be provided for evaluation"
                mse, rmse, mspe = self.eval(model, testX, testY)
                print(f"{model_name} MSE: {mse}, RMSE: {rmse}, MSPE: {mspe} %")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_preds = model.predict(testX)
        return pd.DataFrame(best_preds, columns = ["pred"])

    def forecast(self, X, y = None, n = 12, eval = True, save_model = True, path = "models/best_forecasting_model.pkl"):        
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
            forecasts = pd.DataFrame(forecasts, columns = ["CGM"])

            if eval:
                assert len(y) > 0, "y must be provided for evaluation"
                assert len(y) == len(forecasts), "y and forecasts must have the same length"
                mse, rmse, mspe = criterion(forecasts, y, plot = True)
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
            
            with open(path, 'wb') as f:
                pickle.dump(best_model, f)

        X = X.iloc[n-1: n*2 - 1, :]
        X.reset_index(drop=True, inplace=True)
        X.loc[:, "pred"] = best_forecasts
        return X
    
    def eval(self, model, X, y):
        return criterion(model.predict(X), y)