import pandas as pd

# # # # # # # # # # # # # # # # 
# Technical Indicators
# # # # # # # # # # # # # # # # 

# https://www.babypips.com/learn/forex/elementary#common-chart-indicators
# https://www.babypips.com/learn/forex/summary-common-chart-indicators
# https://github.com/femtotrader/pandas_talib/blob/master/pandas_talib/__init__.py

def EMA(df, n=7, price='close'):
    n = max(int(n),2)
    return (df[price].ewm(span=n).mean(),)

def BBANDS(df, n=20, k=2, price='close'):
    '''
    Bollinger Bands
    '''
    n = max(int(n),2)
    
    ma = df[price].rolling(window=n).mean()
    std = df[price].rolling(window=n).std()
    return (ma-k*std, ma, ma+k*std)

def MACD(df, n_fast=12, n_slow=26, n_signal=9, price='close'):
    '''
    MACD, MACD Signal and MACD difference
    '''
    n_fast = max(int(n_fast),2)
    n_slow = max(int(n_slow),2)
    n_signal = max(int(n_signal),2)

    EMA_fast = df[price].ewm(span=n_fast).mean()
    EMA_slow = df[price].ewm(span=n_slow).mean()
    MACD = EMA_fast - EMA_slow
    MACD_signal = MACD.ewm(span=n_signal).mean()
    MACD_hist = MACD - MACD_signal
    return (MACD, MACD_signal, MACD_hist)

def PSAR(df):
    '''
    Parabolic Stop And Reversal
    '''
    pass

def STOCHASTIC(df, n=14, k=3, price='close'):
    '''
    Stochastic Oscillator
    '''
    n = max(int(n),2)
    k = max(int(k),2)

    L = df[price].rolling(window=n).min()
    H = df[price].rolling(window=n).max()
    D = (df[price] - L) / (H - L)
    D_ma = D.rolling(window=k).mean()
    return (D, D_ma, D-D_ma)

def RSI(df, n=14, price='close'):
    '''
    Relative Strength Index
    '''
    n = max(int(n),2)

    U = df[price].diff()
    D = U.multiply(-1)
    U[U<0]=0
    D[D<0]=0
    U_ema = U.ewm(span=n).mean()
    D_ema = D.ewm(span=n).mean()
    RSI = U_ema / (U_ema+D_ema)
    return (RSI,)

def ATR(df):
    '''
    Average True Range
    '''
    ATR1 = (df['high'] - df['low']).abs()
    ATR2 = (df['high'] - df['close'].shift()).abs()
    ATR3 = (df['low'] - df['close'].shift()).abs()
    ATR = pd.concat([ATR1,ATR2,ATR3],axis=1,join='inner').max(axis=1)
    return (ATR,)

def ADX(df, n=14):
    '''
    Average Directional Index
    '''
    n = max(int(n),2)

    UpMove = df['high'].diff()
    DownMove = df['low'].diff().multiply(-1)
    pDM = UpMove.where((UpMove <= DownMove) | (UpMove < 0), 0) 
    nDM = DownMove.where((DownMove <= UpMove) | (DownMove < 0), 0)
    _ATR = ATR(df)[0]
    pDI = pDM.rolling(window=n).mean() / _ATR
    nDI = nDM.rolling(window=n).mean() / _ATR
    ADX = (pDI - nDI).abs() / (pDI + nDI).abs()
    return (ADX,)

def IKH(df):
    '''
    Ichimoku Kinko Hyo
    '''
    H_52 = df['high'].rolling(window=52).max()
    L_52 = df['low'].rolling(window=52).min()
    H_26 = df['high'].rolling(window=26).max()
    L_26 = df['low'].rolling(window=26).min()
    H_9 = df['high'].rolling(window=9).max()
    L_9 = df['low'].rolling(window=9).min()

    kijun = (H_26+L_26)/2
    tenkan = (H_9+L_9)/2
    chikou = df['close'].shift(-26)
    senkou_1 = ((tenkan+kijun)/2).shift(26)
    senkou_2 = ((H_52+L_52)/2).shift(26)

    return (kijun,tenkan,chikou,senkou_1,senkou_2)