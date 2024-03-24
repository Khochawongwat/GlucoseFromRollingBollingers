from sklearn.preprocessing import MinMaxScaler
import pandas as pd
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

class DateTransformer(BaseTransformer):
    def __init__(self, attribute: str = "Time"):
        self.attribute = attribute

    def datetime_to_bins(self, X):
        X["month"] = X[self.attribute].dt.month
        X["day"] = X[self.attribute].dt.day
        X["hour"] = X[self.attribute].dt.hour
        X["minute"] = X[self.attribute].dt.minute
        X["weekday"] = X[self.attribute].dt.weekday
        X["year"] = X[self.attribute].dt.year
        return X

    def transform(self, X, y=None) -> pd.DataFrame:
        X[self.attribute] = pd.to_datetime(
            X[self.attribute], format="%d-%b-%Y %H:%M:%S"
        )
        X = self.datetime_to_bins(X)
        return X

class OutlierRemover(BaseTransformer):
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

class MovingAverageTransformer(BaseTransformer):
    def __init__(self, length= 6, shift = 1, weighted = True):
        self.length = length
        self.shift = shift
        self.weighted = weighted

    def transform(self, X, y=None):
        assert self.length > 0, "Length must be greater than 0"
        for m in range(1, self.length + 1):
            if self.weighted:
                weights = np.arange(1, m + 1)
                X[f"wCGM({m})"] = X["CGM"].shift(self.shift).rolling(window=m).apply(lambda x: np.average(x, weights=weights[::-1]), raw=True)
            else:
                X[f"CGM({m})"] = X["CGM"].shift(self.shift).rolling(window=m).mean()
        return X.dropna()

class FeatureTransformer(BaseTransformer):
    def __init__(self, shift = 1, window = 24, std = 3, length = 6):
        self.shift = shift
        self.scaler = MinMaxScaler()
        self.window = window
        self.std = std
        self.length = length

    def calculate_velocity(self, X):
        return X["CGM"].shift(self.shift).diff() / (X["Time"].shift(self.shift).diff().astype("int64") // 1e9)

    def calculate_change(self, X):
        return X['CGM'].shift(self.shift) - X['CGM'].shift(self.shift + 1)

    def calculate_bands(self, X):
        rolling_mean = X["CGM"].shift(self.shift).rolling(window=self.window).mean()
        rolling_std = X["CGM"].shift(self.shift).rolling(window=self.window).std()
        upper_band = rolling_mean + (rolling_std * self.std)
        lower_band = rolling_mean - (rolling_std * self.std)
        return upper_band, lower_band

    def calculate_extreme_CGM(self, X):
        quantiles = [0.80, 0.85, 0.90, 0.95]
        for quantile in quantiles:
            high_quantile = X['CGM'].shift(self.shift).quantile(quantile)
            low_quantile = X['CGM'].shift(self.shift).quantile(1 - quantile)
            X[f'extreme_CGM_{int(quantile * 100)}'] = ((X['CGM'] > high_quantile) | (X['CGM'] < low_quantile)).astype(int)
        return X

    def transform(self, X, y=None):
        X["cgm_velo"] = self.calculate_velocity(X)
        X['change'] = self.calculate_change(X)
        X['upper_band'], X['lower_band'] = self.calculate_bands(X)
        X = self.calculate_extreme_CGM(X)
        return X.drop(["Time"], axis=1)