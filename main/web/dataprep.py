from Indicator import *
import pandas as pd
from sklearn.impute import SimpleImputer

def DataGen(df):
    df = (MACD(df)
      .pipe(RSI)
      .pipe(BBs)
      .pipe(EMA)
      .pipe(GAIN)
      .pipe(volatility)
      .pipe(ATR)
      .pipe(ADX)
      .pipe(PSAR)
      .pipe(Stochastic)
      .pipe(ROC)
      .pipe(PPO)
      .pipe(PVO)
      .pipe(Keltner))
    
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return(df)

