
import os
import random
import glob
import gc
from tqdm import tqdm

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno

import pyarrow as pa
import pyarrow.parquet as pq

import plotly.express as px
import plotly.graph_objects as go


book_example = pd.read_parquet('../input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0')

print(book_example)