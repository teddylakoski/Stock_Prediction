import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
import os
import sys
import time
import random


def yf_download_with_retry(tickers, start=None, end=None, tries=6):
    """
    Robust yfinance downloader that retries on Yahoo rate limits.
    Works for a list of tickers.
    """
    last_err = None
    for i in range(tries):
        try:
            return yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,        # reduces bursty requests
                group_by="column"
            )
        except Exception as e:
            last_err = e
            msg = str(e)

            # Retry only for rate limit type errors
            if ("RateLimitError" in msg) or ("Too Many Requests" in msg):
                sleep_s = (2 ** i) + random.random()  # exponential backoff + jitter
                time.sleep(sleep_s)
                continue

            # If it's not rate limiting, raise the real error
            raise

    raise RuntimeError(f"yfinance download failed after {tries} tries. Last error: {last_err}")


def extract_features():
    return_period = 5

    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")

    stk_tickers = ['NVDA', 'AVGO', 'TSM', 'ORCL', 'AMD', 'META']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

    # ✅ FIX: use retry wrapper instead of raw yf.download
    stk_data = yf_download_with_retry(stk_tickers, start=START_DATE, end=END_DATE)

    # FRED (usually fine; not the source of your error)
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    # Target: NVDA future log return
    Y = np.log(stk_data.loc[:, ('Adj Close', 'NVDA')]).diff(return_period).shift(-return_period)
    Y.name = Y.name[-1] + '_Future'

    # Features: other stocks + FX + indexes
    X1 = np.log(stk_data.loc[:, ('Adj Close', ('AVGO', 'TSM', 'ORCL', 'AMD', 'META'))]).diff(return_period)
    X1.columns = X1.columns.droplevel()

    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X = pd.concat([X1, X2, X3], axis=1)

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]
    dataset.index.name = 'Date'

    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:, 1:]
    return features


def get_bitcoin_historical_prices(days=60):
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()  # ✅ helpful: raises if API returns an error
    data = response.json()

    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df


