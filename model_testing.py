import pandas as pd
import numpy as np
import os

filenames = ['DAT_ASCII_EURUSD_M1_'+str(y)+'.csv' for y in range(2000,2017)]

# # # # # # # # # # # # # # # # 
# Data Preprocessing
# # # # # # # # # # # # # # # # 

from preprocess import preprocess

take_profit = 0.01
for filename in filenames:
	preprocess(filename, take_profit)

# # # # # # # # # # # # # # # # 
# Technical Indicators
# # # # # # # # # # # # # # # # 

from indicators import *

data_segments = []

for filename in filenames:
	print("CALCULATING TECHNICAL INDICATORS:", filename)
	data_segment = pd.read_csv('data/preprocessed/'+filename, index_col=0)

	indicators = [BBANDS, MACD, STOCHASTIC, RSI, ADX, IKH]

	for indicator in indicators:
		print(indicator.__name__)
		res = indicator(data_segment)
		for i,m in enumerate(res):
			data_segment[indicator.__name__+str(i)] = m
	
	data_segments += [data_segment]

data = pd.concat(data_segments)
data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# # # # # # # # # # # # # # # # 
# Model Testing
# # # # # # # # # # # # # # # # 

import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier

# clf = xgb.XGBClassifier(missing=np.nan)
#clf = LogisticRegressionCV()
clf = MLPClassifier()

X = data.drop(['timestamp','long_success','short_success'], axis=1).as_matrix()
y = data['long_success'].as_matrix()

'''print("Feature Importance:")
clf = clf.fit(X,y)
for col,i in zip(
	data.drop(['timestamp','long_success','short_success'], axis=1).columns, 
	clf.feature_importances_):
	print("\t",col,i)'''


# aim to predict LONG opportunities
# absense of LONG opportunity = SHORT opportunity (symmetric take profit / stop loss)
# test cross-validation accuracy
scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5,shuffle=False), n_jobs=-1)
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