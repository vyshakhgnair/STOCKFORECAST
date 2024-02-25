from dataprep import *
import pandas as pd
import yfinance as yf


def StockData(ticker):
    df = yf.download(ticker)
    df.drop(['Adj Close'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df = DataGen(df)
    return df