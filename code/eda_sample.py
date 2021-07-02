
import pandas as pd

train_df = pd.read_csv("../input/optiver-realized-volatility-prediction/train.csv")

book_example = pd.read_parquet("../input/optiver-realized-volatility-prediction/book_train.parquet/stock_id=0")
trade_example = pd.read_parquet("../input/optiver-realized-volatility-prediction/trade_train.parquet/stock_id=0")

stock = train_df.groupby("stock_id")["target"].agg(
    ["mean", "median", "std", "count", "sum"]).reset_index()

print(stock)


time_id = 5
train_example = train_df[(train_df.stock_id == 0) & (train_df.time_id == time_id)]

book_given_time = book_example[book_example.time_id == time_id]
trade_given_time = trade_example[trade_example.time_id == time_id]

print("train", train_example)
print("book", book_given_time)
print("trade", trade_given_time)

