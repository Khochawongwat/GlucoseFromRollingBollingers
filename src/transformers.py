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
    def __init__(self, has_insulin=False, length=6, shift = 1):
        self.length = length
        self.shift = shift
        self.has_insulin = has_insulin

    def transform(self, X, y=None):
        assert self.length > 0, "Length must be greater than 0"
        for m in range(1, self.length + 1):
            X[f"CGM({m})"] = X["CGM"].shift(self.shift).rolling(window=m).mean()
            if self.has_insulin:
                X[f"insulin({m})"] = X["insulin"].shift(self.shift).rolling(window=m).mean()
        return X.dropna()

class FeatureTransformer(BaseTransformer):
    def __init__(self, train, has_insulin=False, has_meal=False, has_velo=True, shift = 1):
        self.train = train
        self.has_insulin = has_insulin
        self.has_meal = has_meal
        self.has_velo = has_velo
        self.shift = shift
        self.scaler = MinMaxScaler()
        self.cols_to_drop = ["Time", "basal_insulin", "bolus_insulin"]

    def transform(self, X, y=None):
        if self.has_insulin:
            self.scaler.fit(self.train[["basal_insulin", "bolus_insulin"]])
            transformed_data = self.scaler.transform(
                X[["basal_insulin", "bolus_insulin"]]
            )
            X["insulin"] = np.apply_along_axis(np.sum, 1, transformed_data)

        if self.has_insulin:
            X = create_time_since_last_meal(X)
            self.cols_to_drop.extend(["meal_time"])

        if self.has_velo:
            X["cgm_velo"] = X["CGM"].shift(self.shift).diff() / (
                X["Time"].shift(self.shift).diff().astype("int64") // 1e9
            )

        return X.drop(
            self.cols_to_drop + ["is_breakfast", "is_dinner", "is_snack", "is_lunch"],
            axis=1,
        )
