import os
import pandas as pd
from sklearn import pipeline

class DataLoader:
    def __init__(self, dir):
        assert os.path.isdir(dir), f"{dir} is not a valid directory"
        self.dir = dir
        self.dataset = {}
        self.train = None
        self.test = None
        self.tune = None

    def load(self):
        files = [f for f in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, f))]
        print(f"Found {len(files)} files in {self.dir}")
        for file in files:
            self._load_file(file)
        self._concat_data()

    def _load_file(self, file):
        try:
            df = pd.read_csv(f"{self.dir}/{file}", sep=",")
            person = file.split('_')[0]
            if person not in self.dataset:
                self.dataset[person] = {"train": [], "test": []}
            if "Training" in file:
                self.dataset[person]["train"].append(df)
            elif "Testing" in file:
                self.dataset[person]["test"].append(df)
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    def _concat_data(self):
        for person in self.dataset:
            self.dataset[person]["train"] = pd.concat(self.dataset[person]["train"], ignore_index=True)
            self.dataset[person]["test"] = pd.concat(self.dataset[person]["test"], ignore_index=True)
            print(person, "Train:", self.dataset[person]["train"].shape, "Test:", self.dataset[person]["test"].shape)
    
    def cut(self, time: str, y: str):
        self.attribute = y
        self.time = time
        for person in self.dataset:
            self.dataset[person]["train"] = self.dataset[person]["train"][[time, y]]
            self.dataset[person]["test"] = self.dataset[person]["test"][[time, y]]

    def process(self):
        train_lst = []
        test_lst = []
        
        for key in self.dataset.keys():
            for data_type in ['train', 'test']:
                df = self.dataset[key][data_type]

                mask = df[self.attribute].isna().rolling(25).sum().fillna(0) > 24
                df['group'] = (mask != mask.shift(1)).cumsum().astype(float)

                df_list = []

                for group in df['group'].unique():
                    if df[df['group'] == group].shape[0] < 25:
                        df.loc[df['group'] == group, 'CGM'] = df.loc[df['group'] == group, 'CGM'].interpolate()
                    else:
                        mask = df['group'] == group
                        split_index = df.loc[mask].shape[0] // 2
                        df1 = df.loc[mask].iloc[:split_index]
                        df2 = df.loc[mask].iloc[split_index:]
                        df1.dropna(subset=['CGM'], inplace=True)
                        df2.dropna(subset=['CGM'], inplace=True)

                        df1[self.time] = pd.to_datetime(df1[self.time], format="%d-%b-%Y %H:%M:%S")
                        df2[self.time] = pd.to_datetime(df2[self.time], format="%d-%b-%Y %H:%M:%S")

                        df1.set_index(df1[self.time], inplace=True)
                        df2.set_index(df2[self.time], inplace=True)

                        df1['diff'] = df1[self.time].diff()
                        df2['diff'] = df2[self.time].diff()

                        is_constant_df1 = (df1['diff'].nunique() == 1) and (df1['diff'].dropna().iloc[0] == pd.Timedelta(minutes = 5))
                        is_constant_df2 = (df2['diff'].nunique() == 1) and (df2['diff'].dropna().iloc[0] == pd.Timedelta(minutes = 5))
                        
                        df1.drop(columns=['diff', 'group', "Time"], inplace=True)
                        df2.drop(columns=['diff', 'group', "Time"], inplace=True)

                        if not df1.empty and is_constant_df1 and len(df1) > 6:
                            df_list.append(df1)

                        if not df2.empty and is_constant_df2 and len(df2) > 6:
                            df_list.append(df2)
                
                for i, df in enumerate(df_list):
                    if df.isnull().values.any():
                        print(f"DataFrame at index {i} contains null values.")

                if data_type == 'train':
                    train_lst.extend(df_list)
                else:
                    test_lst.extend(df_list)

        self.train = train_lst[:-2]
        self.test = test_lst[len(test_lst)//2:]
        self.tune = test_lst[:len(test_lst)//2]

    def __getitem__(self, key):
        if key == 'train':
            return self.train
        elif key == 'test':
            return self.test
        elif key == 'tune':
            return self.tune
        else:
            raise KeyError(f"Invalid key: {key}")
    
    def __len__(self):
        return len(self.train) + len(self.test)