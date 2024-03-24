import os
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from transformers import DateTransformer, OutlierRemover, FeatureTransformer, MovingAverageTransformer

FEATURES = [
    "Time",
    "CGM",
]

def reduce_classes(data: pd.DataFrame) -> pd.DataFrame:
    return data[FEATURES]

def get_pipeline(train_data: pd.DataFrame):
    pipe = Pipeline(
        [
            ("DateTransformer", DateTransformer()),
            ("OutlierRemover", OutlierRemover(train=train_data)),
            ("FeatureTransformer", FeatureTransformer()),
            ("MovingAverageTransformer", MovingAverageTransformer()),
        ]
    )
    return pipe

def load_data(dir: str):
    assert os.path.isdir(dir), f"{dir} is not a valid directory"
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    print(f"Found {len(files)} files in {dir}")
    dataset = {
        "train": [],
        "test": [],
    }
    for file in files:
        try:
            df = pd.read_csv(f"{dir}/{file}", sep=",")
            if "Training" in file:
                dataset["train"].append(df)
            elif "Testing" in file:
                dataset["test"].append(df)
        except Exception as e:
            print(f"Failed to load {file}: {e}")
    dataset["train"] = pd.concat(dataset["train"], ignore_index=True)
    dataset["test"] = pd.concat(dataset["test"], ignore_index=True)
    for key in dataset:
        print(key, dataset[key].shape)
    return dataset

def get_train_dataset(data: pd.DataFrame) -> pd.DataFrame:
    pipe = get_pipeline(data)
    processed_data = pipe.fit_transform(data).dropna()
    print(processed_data.head())
    assert processed_data.isna().sum().sum() == 0, "There are missing values in the dataset"

    return processed_data

def get_any_dataset(data: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    pipe = get_pipeline(train)
    processed_data = pipe.fit_transform(data).dropna()

    assert processed_data.isna().sum().sum() == 0, "There are missing values in the dataset"

    return processed_data

def get_tuning_dataset(data: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([data, train], ignore_index=True)
    pipe = get_pipeline(train)
    processed_data = pipe.fit_transform(merged).dropna()

    assert processed_data.isna().sum().sum() == 0, "There are missing values in the dataset"

    return processed_data

def combine_keys(dataset: dict) -> pd.DataFrame:
    keys = list(dataset.keys())
    combined = pd.concat([dataset[key] for key in keys], ignore_index=True)
    return combined 

def increment_dt(df: pd.DataFrame) -> pd.DataFrame:
    transformer = DateTransformer()

    #Convert binary columns to datetime
    df['Time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

    df.iloc[-1, df.columns.get_loc('Time')] = df.iloc[-2, df.columns.get_loc('Time')] + pd.Timedelta(minutes=5)

    transformer.fit_transform(df)

    #Leave the Time column as datetime because it will be needed later for cgm_velo
    return df

def get_velocity(df: pd.DataFrame, new_value: float) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.loc[df_copy.index[-1], 'cgm_velo'] = (new_value - df_copy.loc[df_copy.index[-2],'CGM(1)']) / (df_copy["Time"].diff().iloc[-1].total_seconds())
    return df_copy

def get_moving_average(df: pd.DataFrame, new_value: float) -> pd.DataFrame:
    df_copy = df.copy()

    cgm_columns = [col for col in df_copy.columns if "CGM" in col]
    cgm_columns.sort(key=lambda x: int(x.split('(')[1].split(')')[0]))

    window_sizes = [i for i in range(1, len(cgm_columns) + 1)]

    for i, col in enumerate(cgm_columns):
        cumulative_sum = df_copy.loc[df_copy.index[-2], col] * (window_sizes[i] - 1) + new_value
        df_copy.loc[df_copy.index[-1], col] = cumulative_sum / window_sizes[i]

    df_copy.loc[df_copy.index[-1], 'CGM(1)'] = new_value

    return df_copy

def step_transform(df: pd.DataFrame, pred: float) -> pd.DataFrame:
    T = df.copy()
    new_row = pd.DataFrame(np.nan, index=[0], columns=T.columns)
    T = pd.concat([T, new_row], ignore_index=True)
    T = increment_dt(T)
    T = get_velocity(T, pred)
    T = get_moving_average(T, pred)
    return T.drop(columns=['Time'])

def fit_navigator_model(model, X, y):
    X_copy = X.copy()
    y_copy = y.copy()

    X_copy = create_navigation_features(X_copy)
    y_copy = calculate_direction(y_copy)
    X_copy = get_previous_direction(X_copy, y_copy)

    model.fit(X_copy, y_copy)
    return model

def get_navigator_prediction(model, X):
    X_copy = X.copy()
    X_copy = create_navigation_features(X_copy)
    if "CGM(1)" in X_copy.columns:
        X_copy = get_previous_direction(X_copy, X_copy["CGM(1)"])
    else:
        X_copy = get_previous_direction(X_copy, X_copy["wCGM(1)"])
    y_pred = model.predict(X_copy)
    return y_pred

def calculate_direction(y):
    y_shifted = y.copy().shift(1)
    y_diff = y_shifted.diff()
    nY = y_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return nY

def create_navigation_features(X):
    
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 23.0)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 23.0)

    X["minute_sin"] = np.sin(2 * np.pi * X["minute"] / 59.0)
    X["minute_cos"] = np.cos(2 * np.pi * X["minute"] / 59.0)

    return X

def get_previous_direction(X, y):
    X["prev_direction"] = y.shift(1)
    X["prev_direction"] = X["prev_direction"].fillna('steady')
    X["prev_direction"] = X["prev_direction"].map({'steady': 0, 'up': 1, 'down': -1})
    return X
