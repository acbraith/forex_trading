import pandas as pd
import numpy as np
import os

# # # # # # # # # # # # # # # # 
# Data Preprocessing
# # # # # # # # # # # # # # # # 

def preprocess(data, fixed_time = True, time = 60, take_profit=0.01):

		data.columns = ['timestamp','open','high','low','close','volume']
		data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d %H%M%S')

		# calculate long and short opportunities
		if not(fixed_time):
			data['long_success'] = ''
			data['short_success'] = ''
			long_successes = np.zeros((len(data)))
			short_successes = np.zeros((len(data)))
			for i, row in data.iterrows():
				if (i+1)%1000 == 0:
					print(i+1,"/",len(data))
				long_success = np.where(data.loc[i:]['close'] > row['close'] + take_profit)[0]
				short_success = np.where(data.loc[i:]['close'] <= row['close'] - take_profit)[0]
				if len(long_success) > 0 and (len(short_success) == 0 or long_success[0] < short_success[0]):
					long_successes[i] = True
				else:
					long_successes[i] = False
				if len(short_success) > 0 and (len(long_success) == 0 or short_success[0] < long_success[0]):
					short_successes[i] = True
				else:
					short_successes[i] = False
			data['long_success'] = long_successes
			data['short_success'] = short_successes
		else:
			ask = data['high'] # buy at ask
			bid = data['low']  # sell at bid
			long_profit = -ask + bid.shift(-time)
			short_profit = bid - ask.shift(-time)
			data['long_profit'] = long_profit
			data['short_profit'] = short_profit

		return data
