import pandas as pd
import numpy as np

def fill_forward(df, column):
    df[column] = df[column].ffill()
    return df

def calculate_time_since_last_meal(df, time_column, meal_time_column):
    since_last_meal = pd.to_datetime(df[time_column]) - pd.to_datetime(df[meal_time_column])
    df['time_since_last_meal(min)'] = since_last_meal.dt.total_seconds() / 60
    df['time_since_last_meal(hour)'] = round(df['time_since_last_meal(min)'] / 60)
    return df

def create_time_since_last_meal(df: pd.DataFrame):
    MEAL_TYPES = ['is_breakfast', 'is_lunch', 'is_dinner', 'is_snack']
    cdf = df.copy()
    
    cdf['meal_time'] = cdf['Time'].where(cdf[MEAL_TYPES].any(axis=1))
    cdf = fill_forward(cdf, 'meal_time')

    cdf['meal_type'] = np.nan
    for i, meal_type in enumerate(MEAL_TYPES):
        cdf.loc[cdf[meal_type] == 1, 'meal_type'] = i
    cdf = fill_forward(cdf, 'meal_type')

    cdf = calculate_time_since_last_meal(cdf, 'Time', 'meal_time')

    return cdf.drop(MEAL_TYPES, axis = 1)