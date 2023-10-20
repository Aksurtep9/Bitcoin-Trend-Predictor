import sqlite3
import pandas as pd
import sqlalchemy
from binance.client import Client
from talib import WILLR, SMA, RSI, MOM, CMO, ADX, TEMA
from datetime import datetime, timedelta
import binance_secrets




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
        frame['WILLR'] = WILLR(frame['high'], frame['low'], frame['close'])
        frame['SMA'] = SMA(frame['close'], timeperiod=20)
        frame['RSI'] = RSI(frame['close'], timeperiod=14)
        frame['MOM'] = MOM(frame['close'], timeperiod=10)
        frame['CMO'] = CMO(frame['close'], timeperiod=14)
        frame['ADX'] = ADX(frame['high'], frame['low'], frame['close'], timeperiod=14)
        frame['TEMA'] = TEMA(frame['close'], timeperiod=14)
        frame['Time'] = pd.to_datetime(frame['Time'], unit='ms')
        frame[['DAYS_AFTER_HALVING', 'DAYS_BEFORE_HALVING']] = frame['Time'].apply(lambda x: calculate_days_halving(x.strftime("%Y-%m-%d"))).tolist()

    else:
        print('Couldn\'t load DataFrame from the API')

    return frame



def load_df_to_DB(frame):
    if frame is not None:
        engine = sqlalchemy.create_engine('sqlite:///DataSource.db')
        engine.connect()
        frame.to_sql('BTC_USDT_Historic', engine, if_exists='replace')


frame = get_DF_from_klines()
#frame = get_DF_from_DB()
load_df_to_DB(frame)
print(frame.sample(10))