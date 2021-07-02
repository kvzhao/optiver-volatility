
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

from sklearn.model_selection import train_test_split, KFold

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


class Config:
    data_dir = '../input/optiver-realized-volatility-prediction/'
    seed = 42
    test_size = 0.2
    draw_feat_importance = False

def get_trade_and_book_by_stock_and_time_id(stock_id, time_id=None, dataType='train'):
    book_example = pd.read_parquet(f'{Config.data_dir}book_{dataType}.parquet/stock_id={stock_id}')
    trade_example = pd.read_parquet(f'{Config.data_dir}trade_{dataType}.parquet/stock_id={stock_id}')
    if time_id:
        book_example = book_example[book_example['time_id'] == time_id]
        trade_example = trade_example[trade_example['time_id'] == time_id]
    book_example.loc[:, 'stock_id'] = stock_id
    trade_example.loc[:, 'stock_id'] = stock_id
    return book_example, trade_example

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))


def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def calculate_wap1(df):
    a1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    b1 = df['bid_size1'] + df['ask_size1']
    a2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    b2 = df['bid_size2'] + df['ask_size2']
    
    x = (a1/b1 + a2/b2)/ 2
    
    return x


def calculate_wap2(df):
        
    a1 = df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']
    a2 = df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']
    b = df['bid_size1'] + df['ask_size1'] + df['bid_size2']+ df['ask_size2']
    
    x = (a1 + a2)/ b
    return x

def realized_volatility_per_time_id(file_path, prediction_column_name):

    stock_id = file_path.split('=')[1]

    df_book = pd.read_parquet(file_path)
    df_book['wap1'] = calculate_wap1(df_book)
    df_book['wap2'] = calculate_wap2(df_book)

    df_book['log_return1'] = df_book.groupby(['time_id'])['wap1'].apply(log_return)
    df_book['log_return2'] = df_book.groupby(['time_id'])['wap2'].apply(log_return)
    df_book = df_book[~df_book['log_return1'].isnull()]

    df_rvps =  pd.DataFrame(df_book.groupby(['time_id'])[['log_return1', 'log_return2']].agg(realized_volatility)).reset_index()
    df_rvps[prediction_column_name] = 0.6 * df_rvps['log_return1'] + 0.4 * df_rvps['log_return2']

    df_rvps['row_id'] = df_rvps['time_id'].apply(lambda x:f'{stock_id}-{x}')
    
    return df_rvps[['row_id',prediction_column_name]]


def get_agg_info(df):
    agg_df = df.groupby(['stock_id', 'time_id']).agg(mean_sec_in_bucket = ('seconds_in_bucket', 'mean'), 
                                                     mean_price = ('price', 'mean'),
                                                     mean_size = ('size', 'mean'),
                                                     mean_order = ('order_count', 'mean'),
                                                     max_sec_in_bucket = ('seconds_in_bucket', 'max'), 
                                                     max_price = ('price', 'max'),
                                                     max_size = ('size', 'max'),
                                                     max_order = ('order_count', 'max'),
                                                     min_sec_in_bucket = ('seconds_in_bucket', 'min'), 
                                                     min_price = ('price', 'min'),
                                                     #min_size = ('size', 'min'),
                                                     #min_order = ('order_count', 'min'),
                                                     median_sec_in_bucket = ('seconds_in_bucket', 'median'), 
                                                     median_price = ('price', 'median'),
                                                     median_size = ('size', 'median'),
                                                     median_order = ('order_count', 'median')
                                                    ).reset_index()
    
    return agg_df

def get_stock_stat(stock_id : int, dataType = 'train'):
    
    book_subset, trade_subset = get_trade_and_book_by_stock_and_time_id(stock_id, dataType=dataType)
    book_subset.sort_values(by=['time_id', 'seconds_in_bucket'])

    ## book data processing
    
    book_subset['bas'] = (book_subset[['ask_price1', 'ask_price2']].min(axis = 1)
                                / book_subset[['bid_price1', 'bid_price2']].max(axis = 1)
                                - 1)                               
    
    book_subset['wap1'] = calculate_wap1(book_subset)
    book_subset['wap2'] = calculate_wap2(book_subset)
    
    book_subset['log_return_bid_price1'] = np.log(book_subset['bid_price1'].pct_change() + 1)
    book_subset['log_return_ask_price1'] = np.log(book_subset['ask_price1'].pct_change() + 1)
    # book_subset['log_return_bid_price2'] = np.log(book_subset['bid_price2'].pct_change() + 1)
    # book_subset['log_return_ask_price2'] = np.log(book_subset['ask_price2'].pct_change() + 1)
    book_subset['log_return_bid_size1'] = np.log(book_subset['bid_size1'].pct_change() + 1)
    book_subset['log_return_ask_size1'] = np.log(book_subset['ask_size1'].pct_change() + 1)
    # book_subset['log_return_bid_size2'] = np.log(book_subset['bid_size2'].pct_change() + 1)
    # book_subset['log_return_ask_size2'] = np.log(book_subset['ask_size2'].pct_change() + 1)
    book_subset['log_ask_1_div_bid_1'] = np.log(book_subset['ask_price1'] / book_subset['bid_price1'])
    book_subset['log_ask_1_div_bid_1_size'] = np.log(book_subset['ask_size1'] / book_subset['bid_size1'])

    book_subset['log_return1'] = (book_subset.groupby(by = ['time_id'])['wap1'].
                                  apply(log_return).
                                  reset_index(drop = True).
                                  fillna(0))

    book_subset['log_return2'] = (book_subset.groupby(by = ['time_id'])['wap2'].
                                  apply(log_return).
                                  reset_index(drop = True).
                                  fillna(0))
    stock_stat = pd.merge(
        book_subset.groupby(by = ['time_id'])['log_return1'].agg(realized_volatility).reset_index(),
        book_subset.groupby(by = ['time_id'], as_index = False)['bas'].mean(),
        on = ['time_id'],
        how = 'left'
    )
    
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return2'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_bid_price1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_ask_price1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_bid_size1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_return_ask_size1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_ask_1_div_bid_1'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )
    stock_stat = pd.merge(
        stock_stat,
        book_subset.groupby(by = ['time_id'])['log_ask_1_div_bid_1_size'].agg(realized_volatility).reset_index(),
        on = ['time_id'],
        how = 'left'
    )


    stock_stat['stock_id'] = stock_id

    return stock_stat

def get_data_set(stock_ids : list, dataType = 'train'):

    stock_stat = Parallel(n_jobs=-1)(
        delayed(get_stock_stat)(stock_id, dataType) 
        for stock_id in stock_ids
    )
    
    stock_stat_df = pd.concat(stock_stat, ignore_index = True)

    return stock_stat_df

def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def plot_feature_importance(df, model):
    feature_importances_df = pd.DataFrame({
        'feature': df.columns,
        'importance_score': model.feature_importances_
    })
    plt.rcParams["figure.figsize"] = [10, 5]
    ax = sns.barplot(x = "feature", y = "importance_score", data = feature_importances_df)
    ax.set(xlabel="Features", ylabel = "Importance Score")
    plt.xticks(rotation=45)
    plt.show()
    return feature_importances_df

train = pd.read_csv(Config.data_dir + 'train.csv')
test = pd.read_csv(Config.data_dir + 'test.csv')
train.stock_id.unique()

if not os.path.exists("train_features_df.pickle"):
    train_stock_stat_df = get_data_set(train.stock_id.unique(), dataType = 'train')
    train_data_set = pd.merge(train, train_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left')
    train_data_set.to_pickle('train_features_df.pickle')
    print(train_data_set.info())
else:
    train_data_set = pd.read_pickle("train_features_df.pickle")
    print(train_data_set)

if not os.path.exists("test_features_df.pickle"):
    test_stock_stat_df = get_data_set(test['stock_id'].unique(), dataType = 'test')
    test_data_set = pd.merge(test, test_stock_stat_df, on = ['stock_id', 'time_id'], how = 'left')
    test_data_set.fillna(-999, inplace=True)
    test_data_set.to_pickle('test_features_df.pickle')
    print(test_data_set.info())
else:
    test_data_set = pd.read_pickle("test_features_df.pickle")
    print(test_data_set.info())

x = gc.collect()

X_display = train_data_set.drop(['stock_id', 'time_id', 'target'], axis = 1)
X = X_display.values
y = train_data_set['target'].values

print("Data & Label:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=Config.test_size,
    random_state=Config.seed, shuffle=False)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

## MODEL

xgb = XGBRegressor(tree_method='hist',
                   random_state = Config.seed,
                   n_jobs= - 1)

xgb.fit(X_train, y_train)

xgb_preds = xgb.predict(X_test)
R2 = round(r2_score(y_true = y_test, y_pred = xgb_preds), 6)
RMSPE = round(rmspe(y_true = y_test, y_pred = xgb_preds), 6)
print(f'Performance of the naive XGBOOST prediction: R2 score: {R2}, RMSPE: {RMSPE}')
if Config.draw_feat_importance:
    plot_feature_importance(X_display, xgb)

lgbm = LGBMRegressor(device='cpu', random_state=Config.seed)
lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict(X_test)
R2 = round(r2_score(y_true = y_test, y_pred = lgbm_preds),6)
RMSPE = round(rmspe(y_true = y_test, y_pred = lgbm_preds),6)
print(f'Performance of the naive LIGHTGBM prediction: R2 score: {R2}, RMSPE: {RMSPE}')
if Config.draw_feat_importance:
    plot_feature_importance(X_display, lgbm)

#np.shape(X_train)

test_data_set_final = test_data_set.drop(['stock_id', 'time_id'], axis = 1)
y_pred = test_data_set_final[['row_id']]
X_test = test_data_set_final.drop(['row_id'], axis = 1)

y_pred = y_pred.assign(target = lgbm.predict(X_test))
y_pred.to_csv('submission.csv', index = False)
print("Done, save prediction to submission.csv")
