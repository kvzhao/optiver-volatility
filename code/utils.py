
import numpy as np

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

import gc


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return ** 2))


def calculate_wap(df):
    '''
    https://www.kaggle.com/konradb/we-need-to-go-deeper
    '''
    a1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    a2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    b = df['bid_size1'] + df['ask_size1'] + df['bid_size2']+ df['ask_size2']

    x = (a1 + a2) / b
    return x


def get_log_return_df_per_time_id(file_path):
    #df_book_data = pd.read_parquet(file_path)
    dataset = pq.ParquetDataset(file_path)
    book_dataset = dataset.read()
    df_book_data = book_dataset.to_pandas()

    df_book_data['wap'] = calculate_wap(df_book_data)
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]

    stock_id = file_path.split('=')[1]
    df_book_data['row_id'] = df_book_data['time_id'].apply(lambda x:f'{stock_id}-{x}')
 
    del dataset, book_dataset
    gc.collect()

    return df_book_data


def get_realized_volatility_df_per_time_id(file_path):
    #df_book_data = pd.read_parquet(file_path)
    dataset = pq.ParquetDataset(file_path)
    book_dataset = dataset.read()
    df_book_data = book_dataset.to_pandas()

    df_book_data['wap'] = calculate_wap(book_example)
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]

    df_book_data['realized_volatility'] = df_book_data.groupby(['time_id'])['log_return'].apply(realized_volatility)
    df_book_data = df_book_data[~df_book_data['realized_volatility'].isnull()]

    stock_id = file_path.split('=')[1]
    df_book_data['row_id'] = df_book_data['time_id'].apply(lambda x:f'{stock_id}-{x}')

    del dataset, book_dataset
    gc.collect()    

    return df_book_data


def realized_volatility_per_time_id(file_path, prediction_column_name):
    df_book = pd.read_parquet(file_path)
    df_book['wap'] = calculate_wap(df_book)
    df_book['log_return'] = df_book.groupby(['time_id'])['wap'].apply(log_return)
    df_book = df_book[~df_book['log_return'].isnull()]
    df_realized_vol_per_stock =  pd.DataFrame(df_book.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()
    df_realized_vol_per_stock = df_realized_vol_per_stock.rename(columns = {'log_return':prediction_column_name})
    stock_id = file_path.split('=')[1]
    df_realized_vol_per_stock['row_id'] = df_realized_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')
    return df_realized_vol_per_stock[['row_id',prediction_column_name]]