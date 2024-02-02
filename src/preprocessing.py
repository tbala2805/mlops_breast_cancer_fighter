# typing
from typing import List
from typing import Union


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess(data: pd.DataFrame,
               columns_to_drop: List[str] = ['id', 'Unnamed: 32'],
               target_col: str = 'diagnosis') -> tuple:
    ## Preprocessing a bit
    data.drop(columns=columns_to_drop, inplace=True)

    # dividing the data to X and Y
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # scaling the features to max value 1 and min value of 0
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # encoding the target from categorical to numerical
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y

