
import pandas as pd

trade_testparquet = pd.read_parquet("../input/optiver-realized-volatility-prediction/trade_test.parquet/stock_id=0")
print(trade_testparquet)

train_df = pd.read_csv("../input/optiver-realized-volatility-prediction/train.csv")
print("index and target of train")
print(train_df)

book_sample = pd.read_parquet("../input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0")
print("book sample of stock_id = 0")
print(book_sample)

trade_sample = pd.read_parquet("../input/optiver-realized-volatility-prediction/trade_train.parquet/stock_id=0")
print("trade sample of stock_id = 0")
print(trade_sample)
