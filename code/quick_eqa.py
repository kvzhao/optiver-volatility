
"""
https://www.kaggle.com/piantic/starter-optiver-quick-eda-automl-wip

"""

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

from utils import calculate_wap, log_return
from utils import realized_volatility

dataset = pq.ParquetDataset('../input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0')
book_example = dataset.read()
book_example = book_example.to_pandas()

stock_id = '5'
time_id = 5
book_example = book_example[book_example['time_id']==time_id]
book_example.loc[:,'stock_id'] = stock_id

msno.matrix(book_example, fontsize = 16)

book_example['wap'] = calculate_wap(book_example)
book_example.loc[:,'log_return'] = log_return(book_example['wap'])
book_example = book_example[~book_example['log_return'].isnull()]

book_example.loc[:,'realized_volatility'] = book_example.groupby(['time_id'])['log_return'].apply(realized_volatility)
new_book_example = book_example[~book_example['realized_volatility'].isnull()].reset_index()

print(book_example)

print(new_book_example)

pd.DataFrame(book_example.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()

temp_aggs = book_example.groupby(['time_id']).agg(
                                            bid_price1_min = ('bid_price1', 'min'),
                                            bid_price2_max = ('bid_price2', 'max'),
                                            bid_price1_mean = ('bid_price1', 'mean'),
                                            bid_price2_mean = ('bid_price2', 'mean'),
                                            bid_price1_median = ('bid_price1', 'median'),
                                            bid_price2_median = ('bid_price2', 'median'),
                                            ask_price1_min = ('ask_price1', 'min'),
                                            ask_price2_max = ('ask_price2', 'max'),
                                            ask_price1_mean = ('ask_price1', 'mean'),
                                            ask_price2_mean = ('ask_price2', 'mean'),
                                            ask_price1_median = ('ask_price1', 'median'),
                                            ask_price2_median = ('ask_price2', 'median'),
)

aggs_book_example = pd.merge(new_book_example, temp_aggs, on=['time_id'], how='left')
print(aggs_book_example)


#fig = px.line(book_example, x="seconds_in_bucket", y="wap", title='WAP of stock_id_5, time_id_5')
#fig.show()
#fig = px.line(book_example, x="seconds_in_bucket", y="log_return", title='Log return of stock_id_5, time_id_5')
#fig.show()


"""

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=book_example["seconds_in_bucket"], 
        y=book_example["bid_price1"], 
        mode='lines', 
        name='bid_price1'
    )
)
fig.add_trace(
    go.Scatter(
        x=book_example["seconds_in_bucket"], 
        y=book_example["bid_price2"], 
        mode='lines', 
        name='bid_price2'
    )
)

fig.show()

"""