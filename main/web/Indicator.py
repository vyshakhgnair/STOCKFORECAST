import pandas as pd
import ta
from ta import *
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from ta.volume import MFIIndicator
from ta.volume import ease_of_movement
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator
from ta.trend import PSARIndicator
from ta.momentum import StochasticOscillator
from ta.momentum import UltimateOscillator
from ta.momentum import ROCIndicator
from ta.momentum import WilliamsRIndicator
from ta.momentum import PercentagePriceOscillator
from ta.momentum import PercentageVolumeOscillator
from ta.volatility import KeltnerChannel


def MACD(df):
    macd_indicator = ta.trend.MACD(df['Close'],window_slow=12,window_fast=9)
    macd=macd_indicator.macd()
    df['MACD']=macd
    return df

def RSI(df):
    rsi_indicator=ta.momentum.RSIIndicator(df['Close'])
    rsi=rsi_indicator.rsi()
    df['RSI']=rsi
    return df

def BBs(df):
    indicator_bb = BollingerBands(close=df["Close"], window=14, window_dev=2)
    df['BB_BBM'] = indicator_bb.bollinger_mavg()
    df['BB_BBH'] = indicator_bb.bollinger_hband()
    df['BB_BBL'] = indicator_bb.bollinger_lband()
    df['BH']=df['BB_BBH']-df['BB_BBM']
    df['BL']=df['BB_BBM']-df['BB_BBL']
    return df

def EMA(df):
    indicator_ema = EMAIndicator(close=df["Close"], window=14)
    df['EMA']=indicator_ema.ema_indicator()
    return df

def MFI(df):
    indicator_mfi = MFIIndicator(high=df['High'],low=df['Low'],volume=df['Volume'],close=df["Close"], window=14)
    df['MFI']=indicator_mfi.money_flow_index()
    return df

def EOM(df):
    df['EOM']=ease_of_movement(high=df['High'],low=df['Low'],volume=df['Volume'],window=14)
    return df

def GAIN(df):
    df['GAIN']=df['Open']-df['Close']
    return df
    
def volatility(df):
    df['VOLATILITY']=df['High']-df['Low']
    return df

def ATR(df):
    indicator_atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df["Close"], window=14)
    df['ATR'] = indicator_atr.average_true_range()
    return df

def ADX(df):
    indicator_adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = indicator_adx.adx()
    df['ADX_pos'] = indicator_adx.adx_pos()
    df['ADX_neg'] = indicator_adx.adx_neg()
    return df

def PSAR(df):
    indicator_psar = PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2)
    df['PSAR'] = indicator_psar.psar()
    df['PSAR_up'] = indicator_psar.psar_up()
    df['PSAR_down'] = indicator_psar.psar_down()
    return df

def Stochastic(df):
    indicator_stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['stoch_k'] = indicator_stoch.stoch()
    df['stoch_d'] = indicator_stoch.stoch_signal()
    return df


def ROC(df):
    indicator_roc = ROCIndicator(close=df['Close'], window=14)
    df['ROC'] = indicator_roc.roc()
    return df



def PPO(df):
    indicator_ppo = PercentagePriceOscillator(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['PPO'] = indicator_ppo.ppo()
    df['PPO_signal'] = indicator_ppo.ppo_signal()
    df['PPO_hist'] = indicator_ppo.ppo_hist()
    return df

def PVO(df):
    indicator_pvo = PercentageVolumeOscillator(volume=df['Volume'], window_slow=26, window_fast=12, window_sign=9)
    df['PVO'] = indicator_pvo.pvo()
    df['PVO_signal'] = indicator_pvo.pvo_signal()
    df['PVO_hist'] = indicator_pvo.pvo_hist()
    return df

def Keltner(df):
    indicator_kc = KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10)
    df['KC_middle'] = indicator_kc.keltner_channel_mband()
    df['KC_upper'] = indicator_kc.keltner_channel_hband()
    df['KC_lower'] = indicator_kc.keltner_channel_lband()
    return df