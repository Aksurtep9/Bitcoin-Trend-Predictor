import sqlite3
import pandas as pd
import sqlalchemy
from binance.client import Client
from talib import WILLR, SMA, RSI, MOM, CMO, ADX, TEMA
from datetime import datetime, timedelta
import binance_secrets
import numpy as np
import sklearn as sk
import sklearn.preprocessing as prep




def get_klines_from_api():
    client = Client(binance_secrets.API_KEY, binance_secrets.API_SECRET)

    symbol = 'BTCUSDT'
    start_date = '2018-01-01'
    end_date = '2023-10-01'
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_date, end_date)
    return klines



def calculate_days_halving(date_in):
    date_format =  "%Y-%m-%d"
    halving_dates = [datetime(2012,11,28), datetime(2016,7,9), datetime(2020, 5, 11), datetime(2024, 4, 1)]
    date = datetime.strptime(date_in, date_format)

    days_after, days_before = -1, -1

    for index in range(4):
        days = date - halving_dates[index]

        if days < timedelta(days=0):
            days_after = date - halving_dates[index - 1]
            days_before = halving_dates[index] - date
            break

    return days_after.days, days_before.days


def calculate_trend_pred(closing_mean, sma_mean):
    div = closing_mean/sma_mean

    if div < 0.95:
        return 0
    elif div >= 0.95 and div <= 1.05:
        return 1
    else: return 2


def get_DF_from_DB():
    engine = sqlalchemy.create_engine('sqlite:///DataSource.db')
    engine.connect()
    frame = pd.read_sql_table('BTC_USDT_Historic', 'sqlite:///DataSource.db')
    return frame


def get_DF_from_klines():
    frame = pd.DataFrame(get_klines_from_api())
    if frame is not None:
        frame = frame.iloc[:, :6]
        frame.columns = ['Time','Open','high' ,'low' , 'close', 'Volume' ]
        frame[['Open','high' ,'low' , 'close', 'Volume']] = frame[['Open','high' ,'low' , 'close', 'Volume']].apply(pd.to_numeric)

        frame['WILLR'] = WILLR(frame['high'], frame['low'], frame['close'])
        frame['SMA'] = SMA(frame['close'], timeperiod=20)
        frame['RSI'] = RSI(frame['close'], timeperiod=14)
        frame['MOM'] = MOM(frame['close'], timeperiod=10)
        frame['CMO'] = CMO(frame['close'], timeperiod=14)
        frame['ADX'] = ADX(frame['high'], frame['low'], frame['close'], timeperiod=7)
        frame['TEMA'] = TEMA(frame['close'], timeperiod=10)
        frame['Time'] = pd.to_datetime(frame['Time'], unit='ms')
        frame[['DAYS_AFTER_HALVING', 'DAYS_BEFORE_HALVING']] = frame['Time'].apply(lambda x: calculate_days_halving(x.strftime("%Y-%m-%d"))).tolist()

    else:
        print('Couldn\'t load DataFrame from the API')

    return frame

def normalize_dataframe(frame):
    scaled_features = prep.StandardScaler().drop(['Time', 'index'], axis='columns').values

def load_df_to_DB(frame):
    if frame is not None:
        engine = sqlalchemy.create_engine('sqlite:///DataSource.db')
        engine.connect()
        frame.to_sql('BTC_USDT_Historic', engine, if_exists='replace')


def create_sections(df, seq_length):
    X_seq = []
    y_lab = []
    for i in range(30, len(df) - 7, 3):
        sequence = df.iloc[i - seq_length:i]
        

        pred_seq = df.iloc[i: i+7]

        sma_pred = pred_seq['SMA'].mean()
        close_pred = pred_seq['close'].mean()

        input_val = sequence.drop(['Time', 'index'], axis='columns').values
        X_seq.append(input_val)
        y_lab.append(calculate_trend_pred(close_pred, sma_pred))

    return np.array(X_seq), y_lab

#frame = get_DF_from_klines()
frame = get_DF_from_DB()

X_seq, y_lab = create_sections(frame, 14)


print(len(X_seq[6:15])) 
print(X_seq[0].item(0))
print(type(X_seq[0].item(0)))
print(f"DOWNTREND SECTIONS:  {y_lab.count(0)}")
print(f"SIDETREND SECTIONS:   {y_lab.count(1)}")
print(f"UPTREND SECTIONS:  {y_lab.count(2)}")

#load_df_to_DB(frame)
print(frame[:40])