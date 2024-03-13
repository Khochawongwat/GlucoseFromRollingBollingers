import os
import pandas as pd
from sklearn.pipeline import Pipeline
from transformers import *

FEATURES = [
    "Time",
    "CGM",
    "basal_insulin",
    "bolus_insulin",
    "is_breakfast",
    "is_lunch",
    "is_dinner",
    "is_snack",
]

def reduce_classes(data: pd.DataFrame) -> pd.DataFrame:
    return data[FEATURES]

def get_pipeline(train_data: pd.DataFrame):
    pipe = Pipeline(
        [
            ("DateTransformer", DateTransformer()),
            ("OutlierRemover", OutlierRemover(train=train_data)),
            ("FeatureTransformer", FeatureTransformer(train=train_data)),
            ("MovingAverage", MovingAverageTransformer()),
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
    processed_data = pipe.fit_transform(data)

    assert processed_data.isna().sum().sum() == 0, "There are missing values in the dataset"
    
    return processed_data

def get_any_dataset(data: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    pipe = get_pipeline(train)
    processed_data = pipe.fit_transform(data)

    assert processed_data.isna().sum().sum() == 0, "There are missing values in the dataset"

    return processed_data

def get_tuning_dataset(data: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([data, train], ignore_index=True)
    pipe = get_pipeline(train)
    processed_data = pipe.fit_transform(merged)

    assert processed_data.isna().sum().sum() == 0, "There are missing values in the dataset"
    
    return processed_data

def combine_keys(dataset: dict) -> pd.DataFrame:
    keys = list(dataset.keys())
    combined = pd.concat([dataset[key] for key in keys], ignore_index=True)
    return combined
