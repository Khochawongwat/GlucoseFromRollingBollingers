import os
import pandas as pd
from transformers import (
    ClassEqualizer,
    DateTransformer,
    OutlierRemover,
    MovingAverageTransformer,
    FeatureTransformer,
    Splitter,
)
from sklearn.pipeline import Pipeline
import traceback

class DataProvider:
    def __init__(self, dir: str):
        self.data = {}
        self.load_data(dir)

    def load_data(self, dir: str):
        file_names = [
            f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))
        ]
        print(f"Found {len(file_names)} files in {dir}")
        for file_name in file_names:
            try:
                self.data[file_name] = pd.read_csv(f"{dir}/{file_name}", sep=",")
            except Exception as e:
                print(f"Failed to load {file_name}: {e}")
        print(f"Loaded {len(self.data)}/{len(file_names)} files")
        try:
            self._process_()
        except Exception as e:
            print(f"Failed to process data: {e}")
            traceback.print_exc()

        print(f"Data processing complete")

    def _get_data_(self, file_name: str):
        return self.data[file_name]

    def get_all_data(self):
        return self.data
    
    def _process_(self):
        t = {}
        for file_name in self.data.keys():

            selected_features = [
                "Time",
                "basal_insulin",
                "bolus_insulin",
                "CGM",
                "is_breakfast",
                "is_lunch",
                "is_dinner",
                "is_snack",
            ]
            _train_ = self.data[file_name][selected_features]
            _train_ = _train_.iloc[: int(CONFIG.get("TRAIN_SIZE") * _train_.shape[0])]

            pipe = Pipeline(
                [
                    ("ClassEqualizer", ClassEqualizer()),
                    ("DateTransformer", DateTransformer()),
                    ("OutlierRemover", OutlierRemover(train=_train_)),
                    ("FeatureTransformer", FeatureTransformer(train=_train_)),
                    ("MovingAverage", MovingAverageTransformer()),
                ]
            )
            
            name, type = file_name.split('_')
            if name not in t:
                t[name] = {}
            t[name][type.split('.')[0].split('ing')[0].lower()] = pipe.fit_transform(self.data[file_name])
        self.data = t

    def splitXY(self, X):
        splitter = Splitter()
        return splitter.fit_transform(X)
    
    def process_single_file(self, file_name: str):
        try:
            data = pd.read_csv(file_name, sep=",")
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")
            return

        selected_features = [
            "Time",
            "basal_insulin",
            "bolus_insulin",
            "CGM",
            "is_breakfast",
            "is_lunch",
            "is_dinner",
            "is_snack",
        ]
        _train_ = data[selected_features]
        _train_ = _train_.iloc[: int(CONFIG.get("TRAIN_SIZE") * _train_.shape[0])]

        pipe = Pipeline(
            [
                ("ClassEqualizer", ClassEqualizer()),
                ("DateTransformer", DateTransformer()),
                ("OutlierRemover", OutlierRemover(train=_train_)),
                ("FeatureTransformer", FeatureTransformer(train=_train_)),
                ("MovingAverage", MovingAverageTransformer()),
            ]
        )

        processed_data = pipe.fit_transform(data)
        return processed_data
    
def combine_data(data: dict, mode: str):
    all_features = []
    all_labels = []

    for patient in data.keys():
        train_data = data[patient][mode]
        
        features = train_data.drop('CGM', axis=1)
        labels = train_data['CGM']
        
        all_features.append(features)
        all_labels.append(labels)

    all_features = pd.concat(all_features)
    all_labels = pd.concat(all_labels)
    return all_features, all_labels