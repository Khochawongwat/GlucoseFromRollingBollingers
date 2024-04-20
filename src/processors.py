import numpy as np
from sklearn.base import TransformerMixin
from utils import *

class BaseTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class OutlierTransformer(BaseTransformer):
    def __init__(self, train, threshold: float = 3, attribute: str = "CGM", shift = 1) -> None:
        self.threshold = threshold
        self.attribute = attribute
        self.shift = shift
        self.mean = None
        self.std = None
        self.train = train

    def fit(self, X, y=None):
        self.mean = self.train[self.attribute].mean()
        self.std = self.train[self.attribute].std()
        return self

    def transform(self, X, y=None):
        z = np.abs((X[self.attribute] - self.mean) / self.std)
        X.loc[(z > self.threshold), self.attribute] = (
            X[self.attribute].shift(self.shift).rolling(window=3).mean()
        )
        return X
    
class BollingersTransformer(BaseTransformer):
    def __init__(self, window_size= 24, shift = 1, std = 2, attribute: str = "CGM"):
        self.window_size = window_size
        self.shift = shift
        self.attribute = attribute
        self.std = std

    def transform(self, X, y=None):
        X = X.copy()
        assert self.window_size > 0, "Length must be greater than 0"

        rolling_mean = X[self.attribute].shift(self.shift).rolling(window=self.window_size).mean()
        rolling_std = X[self.attribute].shift(self.shift).rolling(window=self.window_size).std()
        
        X["upper"] = rolling_mean + (self.std * rolling_std)
        X["middle"] = rolling_mean
        X["lower"] = rolling_mean - (self.std * rolling_std)

        X.fillna(0, inplace=True)

        if "prev" not in X.columns:   
            X[f"prev"] = X[self.attribute].shift(self.shift)

        X["prev%"] = X["prev"].pct_change()
    
        for band in ["upper", "middle", "lower"]:
            X[f"{band}%"] = X[band].pct_change()
        
        for window in range(2, 12):
            X[f"rolling_mean_{window}"] = X[self.attribute].shift(self.shift).rolling(window=window).mean()
            X[f"rolling_std_{window}"] = X[self.attribute].shift(self.shift).rolling(window=window).std()

        return X