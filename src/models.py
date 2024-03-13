from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from tqdm import tqdm
from xgboost import XGBRegressor
from losses import criterion

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
                print(f"{model_name} MSE: {mse}, RMSE: {rmse}, MSPE: {mspe}%")

    def eval(self, model, X, y):
        return criterion(model.predict(X), y)