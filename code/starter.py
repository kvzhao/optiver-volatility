
import os

import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm


DATA_ROOT_DIR = "../input/optiver-realized-volatility-prediction"

config = {'input_trade_path': os.path.join(DATA_ROOT_DIR, "trade_"),
          'input_book_path': os.path.join(DATA_ROOT_DIR, "book_"),
          'train_path': os.path.join(DATA_ROOT_DIR, "train.csv"),
          'test_path' : os.path.join(DATA_ROOT_DIR, "test.csv")}

temp = pd.read_parquet(
    "{}/book_test.parquet/stock_id=0/7832c05caae3489cbcbbb9b02cf61711.parquet".format(DATA_ROOT_DIR))

print(temp)