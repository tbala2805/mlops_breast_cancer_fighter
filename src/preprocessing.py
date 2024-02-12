# typing
from typing import List
from typing import Union
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib


def preprocess(data: pd.DataFrame,
               columns_to_drop: List[str] = ['id', 'Unnamed: 32'],
               target_col: str = 'diagnosis',
               assets_path: str = "") -> tuple:
    # Preprocessing a bit
    data.drop(columns=columns_to_drop, inplace=True)

    # dividing the data to X and Y
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # scaling the features to max value 1 and min value of 0
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # saving scaler joblib file
    scaler_filename = os.path.join(assets_path, "minmax_scaler.save")
    joblib.dump(scaler, scaler_filename)

    # encoding the target from categorical to numerical
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y


def load_saved_minmax(assets_path):
    my_scaler = joblib.load(os.path.join(assets_path, "minmax_scaler.save"))
    return my_scaler

