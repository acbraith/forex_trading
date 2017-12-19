import pandas as pd
import numpy as np
import os

filenames = ['DAT_ASCII_EURUSD_M1_'+str(y)+'.csv' for y in range(2000,2005)]

# # # # # # # # # # # # # # # # 
# Data Preprocessing
# # # # # # # # # # # # # # # # 

from preprocess import preprocess

for filename in filenames:
	if not(os.path.isfile('data/preprocessed/'+filename)):
		print("PREPROCESSING:", filename)
		data = pd.read_csv('data/raw/'+filename, sep=';')
		data = preprocess(data)
		data.to_csv('data/preprocessed/'+filename)

# # # # # # # # # # # # # # # # 
# Load Data
# # # # # # # # # # # # # # # # 

data_segments = []

for filename in filenames:
	data_segment = pd.read_csv('data/preprocessed/'+filename, index_col=0)
	data_segments += [data_segment]

data = pd.concat(data_segments)

# # # # # # # # # # # # # # # # 
# Autoregressive Model
# # # # # # # # # # # # # # # # 

def autoregressive_model(data, periods=20, freq=5):
	df = data[['open','high','low','close']]
	df = pd.concat([
		data['open'],
		data['high']-data['open'],
		data['low']-data['open'],
		data['close']-data['open']], axis=1)
	df = data[['open']]
	cols = [data[['open','high','low','close']]]
	for i in range(1,periods+1):
		cols += [df.diff(i*freq)]
	cols += [data['long_profit']]
	return pd.concat(cols, axis=1)


data = autoregressive_model(data)

# # # # # # # # # # # # # # # # 
# Technical Indicators
# # # # # # # # # # # # # # # # 
'''
from indicators import *

print("CALCULATING TECHNICAL INDICATORS")

indicators = [BBANDS, MACD, STOCHASTIC, RSI, ADX, IKH]

for indicator in indicators:
	#print(indicator.__name__)
	res = indicator(data_segment)
	for i,m in enumerate(res):
		data[indicator.__name__+str(i)] = m

def technical_indicator_model(data):
	cols = [
		data['close'] - data['close'].shift(),
		data['close'] - data['open'],
		data['close'] - data['high'],
		data['close'] - data['low'],
		data['close'] - data['BBANDS0'],
		data['close'] - data['BBANDS1'],
		data['close'] - data['BBANDS2'],
		data['STOCHASTIC0'],
		data['STOCHASTIC1'],
		data['STOCHASTIC2'],
		data['RSI0'],
		data['ADX0'],
		data['close'] - data['IKH0'],
		data['close'] - data['IKH1'],
		#IKH2 give info about future; can't use
		data['close'] - data['IKH3'],
		data['close'] - data['IKH4'],
		data['IKH1'] - data['IKH1'].shift(),
		data['long_profit'],
		]
	data = pd.concat(cols, axis=1, join='inner')
	return data

data = technical_indicator_model(data)
'''
# # # # # # # # # # # # # # # # 
# Model Testing
# # # # # # # # # # # # # # # # 

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

print("TESTING MODEL")

import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier

X = data.iloc[:,:-1].as_matrix()
y = data.iloc[:,-1].as_matrix()

# FIXED TRADE DURATION
reg = xgb.XGBRegressor(missing=np.nan)
#reg = LinearRegression()
'''
scores = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error',
	cv=KFold(n_splits=5,shuffle=False), n_jobs=1)
print("Mean Absolute Error:")
print("\tMean  : "+'{0:.2f}'.format(-scores.mean()))
print("\t95% CI: "+'{0:.2f}'.format(-scores.mean()-2*scores.std())+
	" to "+'{0:.2f}'.format(-scores.mean()+2*scores.std()))
'''
from sklearn.metrics import make_scorer
def backtest_score(y, y_pred, **kwargs):
	y_res = np.where(y_pred>0, y, 0)
	return -np.mean(y_res)
scorer = make_scorer(backtest_score)

scores = cross_val_score(reg, X, y, scoring=scorer,
	cv=KFold(n_splits=5,shuffle=False), n_jobs=1)
print("Average Trade Return:")
print("\tMean  : "+'{0:.3e}'.format(-scores.mean()))
print("\t95% CI: "+'{0:.3e}'.format(-scores.mean()-2*scores.std())+
	" to "+'{0:.3e}'.format(-scores.mean()+2*scores.std()))

'''

#clf = xgb.XGBClassifier(missing=np.nan)
#clf = LogisticRegressionCV()
clf = MLPClassifier()

# FIXED STOP LOSS / TAKE PROFIT

# aim to predict LONG opportunities
# absense of LONG opportunity = SHORT opportunity (symmetric take profit / stop loss)
# test cross-validation accuracy
scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5,shuffle=False), n_jobs=1)
print("Accuracy:")
print("\tMean  : "+'{0:.2f}'.format(scores.mean()*100)+"%")
print("\t95% CI: "+'{0:.2f}'.format((scores.mean()-2*scores.std())*100)+
	" to "+'{0:.2f}'.format((scores.mean()+2*scores.std())*100)+"%")

# determine how large a spread would let us still average a profit
# on success, trade makes take_profit - 2*spread
# on fail, trade loses take_profit + 2*spread
# theta*(tp-2s) = (1-theta)*(tp+2s)
# => theta = 1/2 + s/tp
# and s = tp(theta - 1/2)
spreads = [10000 * take_profit*(s - 1/2) \
	for s in [scores.mean()-2*scores.std(),scores.mean(),scores.mean()+2*scores.std()]]
print("Profitable Spread:")
print("\tMean  : "+'{0:.1f}'.format(spreads[1]))
print("\t95% CI: "+'{0:.1f}'.format(spreads[0])+" to "+'{0:.1f}'.format(spreads[2]))

# determine rate of money gain / loss for a given spread
# 1 day has 60*24 trades
# x% success, y% fail (given by accuracy)
# return/day = 60*24 * (x*(tp-2s) - y*(tp+2s))
spread = 1.5/10000
trade_returns = [100*(1+s*(take_profit-2*spread)-(1-s)*(take_profit+2*spread)) \
	for s in [scores.mean()-2*scores.std(),scores.mean(),scores.mean()+2*scores.std()]]
print("Return on 1.5 pip spread:")
print("\tMean  : "+'{0:.2f}'.format(trade_returns[1])+"%")
print("\t95% CI: "+'{0:.2f}'.format(trade_returns[0])+" to "+'{0:.2f}'.format(trade_returns[2])+"%")
'''