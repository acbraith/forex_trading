import pandas as pd
import numpy as np
import os

# # # # # # # # # # # # # # # # 
# Data Preprocessing
# # # # # # # # # # # # # # # # 

def preprocess(filename, take_profit=0.01):
	if not(os.path.isfile('data/preprocessed/'+filename)):

		print("PREPROCESSING:", filename)

		data = pd.read_csv('data/raw/'+filename, sep=';')
		data.columns = ['timestamp','open','high','low','close','volume']
		data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d %H%M%S')

		# calculate long and short opportunities
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

		# save preprocessed data
		data.to_csv('data/preprocessed/'+filename)
