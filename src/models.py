from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from tqdm import tqdm
from xgboost import XGBRegressor
from losses import criterion
import numpy as np
import pandas as pd
from utils import *
 
class Models:
    def __init__(self):
        self.models = {
            "Ridge": Ridge(max_iter=1000),
            "Lasso": Lasso(max_iter=2000),
            "Elastic Net": ElasticNet(max_iter=3000),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor(verbose = -1),
            "CatBoost": AdaBoostRegressor(),
        }
    
    def fit(self, X, y, testX = None, testY = None, eval = True):
        for model_name, model in tqdm(self.models.items()):
            model.fit(X, y)
            if eval:
                assert testX is not None and testY is not None, "testX and testY must be provided for evaluation"
                mse, rmse, mspe = self.eval(model, testX, testY)
                print(f"{model_name} MSE: {mse}, RMSE: {rmse}, MSPE: {mspe} %")

    def forecast(self, X, y, n):
        forecasts = []
        ma_columns = [x for x in X.columns if "CGM" in x]
        for model_name, model in self.models.items():
            for i in range(n):
                prev_rows = X.iloc[-len(ma_columns):, :]
                last_row = prev_rows.tail(1)
                pred = model.predict(last_row)
                new_row = step_transform(prev_rows, pred).iloc[-1:, :]
                forecasts.append(pred)
                X = pd.concat([X, new_row], ignore_index=True)
        return pd.DataFrame(forecasts, columns = ["CGM"])
    
    def eval(self, model, X, y):
        return criterion(model.predict(X), y)