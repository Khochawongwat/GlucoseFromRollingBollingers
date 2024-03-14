from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from features import create_time_since_last_meal

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
    def __init__(self, length=6, shift = 1):
        self.length = length
        self.shift = shift

    def transform(self, X, y=None):
        assert self.length > 0, "Length must be greater than 0"
        for m in range(1, self.length + 1):
            X[f"CGM({m})"] = X["CGM"].shift(self.shift).rolling(window=m).mean()
        return X.dropna()

class FeatureTransformer(BaseTransformer):
    def __init__(self, has_meal=False, has_velo=True, shift = 1):
        self.has_meal = has_meal
        self.has_velo = has_velo
        self.shift = shift
        self.scaler = MinMaxScaler()

    def transform(self, X, y=None):
        if self.has_velo:
            X["cgm_velo"] = X["CGM"].shift(self.shift).diff() / (
                X["Time"].shift(self.shift).diff().astype("int64") // 1e9
            )

        return X.drop(["Time"], axis=1)
