'''
TODO
    - Model inputs:
        Autoregressive OHLC
            Try diff instead of just absolute values
            Diff in closes and diff from close
        Add in autoregressive OHLC bars of different time periods
            eg past 10 1min, past 10 15min, past 10 1hour...
'''

import pandas as pd 
import numpy as np

import os
import itertools

from matplotlib import pyplot as plt

from sklearn.model_selection import cross_val_score, KFold

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from multiprocessing.pool import Pool

from indicators import *

DATA_PERIOD = '1H' # M, D, H, Min
TEST_PERIOD = '6M'
AUTOREGRESSION_N = 1

# forex data
FILENAMES = [
    'DAT_ASCII_EURUSD_T_201701.csv',
    'DAT_ASCII_EURUSD_T_201702.csv',
    'DAT_ASCII_EURUSD_T_201703.csv',
    'DAT_ASCII_EURUSD_T_201704.csv',
    'DAT_ASCII_EURUSD_T_201705.csv',
    'DAT_ASCII_EURUSD_T_201706.csv',
    'DAT_ASCII_EURUSD_T_201707.csv',
    'DAT_ASCII_EURUSD_T_201708.csv',
    'DAT_ASCII_EURUSD_T_201709.csv',
    'DAT_ASCII_EURUSD_T_201710.csv',
    'DAT_ASCII_EURUSD_T_201711.csv',
    'DAT_ASCII_EURUSD_T_201712.csv',
]

########################################
# Preprocess Data
########################################

def load_chunk(f):
    print("Loading", f)
    try:
        ohlcv = pd.read_csv("EURUSD/processed/"+DATA_PERIOD+"/"+f, index_col = 0, header=[0,1], parse_dates=True)
        print("Loaded", f, "from cache")
        return ohlcv
    except:
        pass
    chunk = pd.read_csv("EURUSD/"+f, names=['timestamp','bid','ask','volume'])

    print("Formatting dates for", f)

    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='%Y%m%d %H%M%S%f')
    chunk = chunk.set_index('timestamp')

    print("Producing OHLCV for", f)
    # produce ohlcv
    bid = chunk['bid'].resample(DATA_PERIOD).ohlc()
    ask = chunk['ask'].resample(DATA_PERIOD).ohlc()
    # pad nan with close
    closes = bid['close'].fillna(method='pad')
    bid = bid.apply(lambda x: x.fillna(closes))
    closes = ask['close'].fillna(method='pad')
    ask = ask.apply(lambda x: x.fillna(closes))

    v = chunk['volume'].resample(DATA_PERIOD).sum().fillna(value=0)

    ohlcv = pd.concat([bid,ask,v], axis=1, keys=['BID','ASK','VOL'])

    if not os.path.exists("EURUSD/processed/"+DATA_PERIOD+"/"):
        try:
            os.makedirs("EURUSD/processed/"+DATA_PERIOD+"/")
        except OSError as e:
            pass

    ohlcv.to_csv("EURUSD/processed/"+DATA_PERIOD+"/"+f)

    return ohlcv

p = Pool()
chunks = p.map(load_chunk, FILENAMES)
print("Building DataFrame...")
ohlcv = pd.concat(chunks)

########################################
# Define Model, Test, Train
########################################

########################################
# Define Model
m = LogisticRegression

class Signals:
    def __init__(self, dim):
        self.dim = dim
        self.models = {
            'enter_long':m(),
            'exit_long':m(),
            'enter_short':m(),
            'exit_short':m(),
        }
        self.xs = {
            'enter_long':np.random.normal(size=(100,dim)),
            'exit_long':np.random.normal(size=(100,dim)),
            'enter_short':np.random.normal(size=(100,dim)),
            'exit_short':np.random.normal(size=(100,dim)),
        }
        self.indicator_params = {}
        for i in range(2):
            self.indicator_params[i] = {
                EMA: {
                    'n':np.random.randint(1,100),
                },
                BBANDS: {
                    'n':np.random.randint(1,100),
                    'k':np.random.uniform(1,10),
                },
                MACD: {
                    'n_fast':np.random.randint(1,100),
                    'n_slow':np.random.randint(1,100),
                    'n_signal':np.random.randint(1,100),
                },
                STOCHASTIC: {
                    'n':np.random.randint(1,100),
                    'k':np.random.randint(1,100),
                },
                RSI: {
                    'n':np.random.randint(1,100),
                },
                ADX: {
                    'n':np.random.randint(1,100),
                },
            }
        self.ys = [False for _ in range(50)] + [True for _ in range(50)]
        for k in self.models:
            self.models[k].fit(
                self.xs[k],
                self.ys)
    def preprocess(self, X):
        xs = []
        # technical indicators
        for i,ind_params in self.indicator_params.items():
            for ind,params in ind_params.items():
                xs += ind(X['BID'], **params)

        X = pd.concat([X.diff()]+xs, axis=1)

        # autoregressive inputs
        n = AUTOREGRESSION_N
        for i in range(n):
            xs += [X.diff().shift(i)]

        return pd.concat(xs, axis=1).fillna(method='pad').fillna(0)

    def enter_long(self,X):
        return self.models['enter_long'].predict(X)
    def exit_long(self,X):
        return self.models['exit_long'].predict(X)
    def enter_short(self,X):
        return self.models['enter_short'].predict(X)
    def exit_short(self,X):
        return self.models['exit_short'].predict(X)
    def get_neighbour(self):
        other = Signals(self.dim)
        for k in self.models:
            #other.models[k].coef_ = self.models[k].coef_ + \
            #    self.models[k].coef_ * np.random.normal(size=self.xs[k].shape) / 5
            #other.models[k].intercept_ = self.models[k].intercept_ + \
            #    self.models[k].intercept_ * np.random.normal(size=self.xs[k].shape) / 5
            other.xs[k] = self.xs[k] + self.xs[k] * np.random.normal(size=self.xs[k].shape) / 20
            other.models[k].fit(
                other.xs[k],
                other.ys)
        for i in self.indicator_params:
            for ind in self.indicator_params[i]:
                for param in self.indicator_params[i][ind]:
                    other.indicator_params[i][ind][param] = self.indicator_params[i][ind][param] +\
                        self.indicator_params[i][ind][param] * np.random.normal() / 5
        return other

########################################
# Test Given Model
def evaluate_models(data_test, signals, plot=False):

    plot_bid = []
    plot_ask = []
    plot_val = []
    
    entry_px = data_test.iloc[AUTOREGRESSION_N+1]['ASK']['open']
    exit_px = data_test.iloc[-1]['BID']['open']
    buy_and_hold_ret = 100 / entry_px * exit_px

    cash = 100
    ccy = 0
    position = 0

    X_test = signals.preprocess(data_test).shift(1).as_matrix()

    enter_longs  = signals.enter_long (np.nan_to_num(X_test))
    enter_shorts = signals.enter_short(np.nan_to_num(X_test))
    exit_longs   = signals.exit_long  (np.nan_to_num(X_test))
    exit_shorts  = signals.exit_short (np.nan_to_num(X_test))
    
    for i,(datetime,row) in enumerate(data_test.iterrows()):
        if i > AUTOREGRESSION_N:
            enter_long  = enter_longs[i]
            enter_short = enter_shorts[i]
            exit_long   = exit_longs[i]
            exit_short  = exit_shorts[i]

            # determine desired position
            if (position == 1 and exit_long) or \
               (position == -1 and exit_short):
                position = 0
                if ccy > 0:
                    cash += ccy * row['BID']['open']
                elif ccy < 0:
                    cash += ccy * row['ASK']['open']
                ccy = 0
            if position == 0 and enter_long:
                position = 1
                ccy = cash / row['ASK']['open']
                cash = 0
            if position == 0 and enter_short:
                position = -1
                ccy = -cash / row['BID']['open']
                cash += cash

            val = cash + ccy * (row['BID']['open'] if ccy>0 else row['ASK']['open'])
            plot_bid += [row['BID']['open']]
            plot_ask += [row['ASK']['open']]
            plot_val += [val]

    if plot:
        #fig, ax1 = plt.subplots()
        plt.plot(range(len(plot_val)), np.array(plot_val)/100*plot_bid[0], 'k-')
        #plt.set_ylabel('Value', color='b')
        #plt.tick_params('y', colors='b')

        #ax2 = ax1.twinx()
        plt.plot(range(len(plot_bid)), plot_bid, 'r--', alpha=0.6)
        plt.plot(range(len(plot_ask)), plot_ask, 'b--', alpha=0.6)
        plt.fill_between(range(len(plot_bid)), plot_bid, plot_ask, alpha=0.2)
        #ax2.set_ylabel('Price', color='r')
        #ax2.tick_params('y', colors='r')

        #fig.tight_layout()
        plt.show()

    # calculate sharpe ratio
    logR = np.diff(np.log(plot_val))
    if logR.std() == 0:
        sharpe = 0
    else:
        sharpe = logR.mean() / logR.std()
    # annualise
    if 'Min' in DATA_PERIOD:
        n = int(DATA_PERIOD[:-3])
        sharpe *= np.sqrt(252*24*60/n)
    elif 'H' in DATA_PERIOD:
        n = int(DATA_PERIOD[:-1])
        sharpe *= np.sqrt(252*24/n)
    elif 'D' in DATA_PERIOD:
        n = int(DATA_PERIOD[:-1])
        sharpe *= np.sqrt(252/n)
    elif 'M' in DATA_PERIOD:
        n = int(DATA_PERIOD[:-1])
        sharpe *= np.sqrt(12/n)

    return sharpe, val, buy_and_hold_ret


########################################
# Train Model
def get_fitness(m_data_train):
    m,data_train = m_data_train
    f,_,_ = evaluate_models(data_train, m)
    return f

def train_models(data_train):
    p = Pool(processes=6)
    methods = ['random','simulated anneal','evolutionary','fit_models']
    method = methods[2]

    N = 10000

    X_train = Signals(1).preprocess(data_train)
    d = X_train.shape[1]

    s = Signals(d)

    if method == 'random':

        n = N
        candidates = [Signals(d) for _ in range(n)]
        scores = list(
            p.map(
                get_fitness, 
                zip(
                    candidates,
                    itertools.repeat(data_train,n)
                    )
                )
            )
        print("Max Score:", np.max(scores))
        return candidates[np.argmax(scores)]

    elif method == 'simulated anneal':

        iters_per_temp = 3*8 # candidates per iteration
        num_temps = int(N/iters_per_temp) # iterations
        T_max = 10

        initial_ss = [Signals(d) for _ in range(iters_per_temp)]
        initial_scores = list(p.map(get_fitness, zip(
            initial_ss, 
            itertools.repeat(data_train,iters_per_temp))))
        # http://cdn.intechopen.com/pdfs/4631.pdf
        k = -T_max * np.log(0.8) / np.std(initial_scores)

        def get_temp(i): # annealing schedule
            # http://cdn.intechopen.com/pdfs/4631.pdf
            return T_max * np.exp(i*np.log(i/num_temps) / (num_temps+1))
        def P(s1,s2,T): # acceptance probability
            delta = s2-s1
            if delta < 0:
                return 1
            else:
                return np.exp(-k*delta/T)

        s = initial_ss[np.argmax(initial_scores)]
        score_s = np.max(initial_scores)

        s_best = s
        score_best = score_s

        for i in range(num_temps):
            print("Iteration",i,"/",num_temps)
            print("Best Score:",score_best)
            print("Curr Score:",score_s)

            T = get_temp(i+1)

            s_primes = [s.get_neighbour() for _ in range(iters_per_temp)]
            score_s_primes = list(p.map(get_fitness, zip(s_primes, itertools.repeat(data_train,iters_per_temp))))

            for s_prime,score_s_prime in zip(s_primes,score_s_primes):
                if score_s_prime > score_best:
                    s_best = s_prime
                    score_best = score_s_prime
                if P(-score_s, -score_s_prime, T) >= np.random.rand():
                    s = s_prime
                    score_s = score_s_prime
        return s_best

    elif method == 'evolutionary':
        # Evolutionary Programming

        population  = 1000
        generations = int(N/population)
        # populate initial generation
        gen = [Signals(d) for _ in range(population)]
        for g in range(generations):
            # try to avoid overfitting
            # https://stackoverflow.com/questions/27764825/how-to-avoid-overfitting-with-genetic-algorithm
            subset_len = int(len(data_train)/10)
            i_start = np.random.randint(0,len(data_train)-subset_len)
            train_subset = data_train[i_start:i_start+subset_len]
            fs = list(
                p.map(
                    get_fitness, 
                    zip(
                        gen,
                        itertools.repeat(train_subset,len(gen))
                        )
                    )
                )
            ranks = [sorted(fs, reverse=True).index(f) for f in fs]
            print("Generation",g)
            print("  Max Fitness:", np.max(fs))
            print(" Mean Fitness:", np.mean(fs))
            ps = 1/(np.array(ranks)+1)
            ps = ps / ps.sum()
            best = gen[np.argmax(fs)]
            # breed next generation
            gen = [np.random.choice(gen,p=ps).get_neighbour() for _ in range(population)]
            # add some new, random models
            gen[:int(population/10)] = [Signals(d) for _ in range(int(population/10))] 
            # preserve best
            gen[-1] = best
        return best
    else:
        # Fit Models
        signals = Signals(1)
        signals.models['enter_long'].fit(
            X_train.as_matrix()[AUTOREGRESSION_N:-1,:], 
            (data_train['BID']['close'] > data_train['ASK']['open']).as_matrix()[1:-AUTOREGRESSION_N]
            )
        signals.models['exit_long'].fit(
            X_train.as_matrix()[AUTOREGRESSION_N:-1,:], 
            (data_train['BID']['close'] < data_train['BID']['open']).as_matrix()[1:-AUTOREGRESSION_N]
            )
        signals.models['enter_short'].fit(
            X_train.as_matrix()[AUTOREGRESSION_N:-1,:], 
            (data_train['BID']['open'] > data_train['ASK']['close']).as_matrix()[1:-AUTOREGRESSION_N]
            )
        signals.models['exit_short'].fit(
            X_train.as_matrix()[AUTOREGRESSION_N:-1,:], 
            (data_train['BID']['open'] < data_train['BID']['close']).as_matrix()[1:-AUTOREGRESSION_N]
            )
        return signals


########################################
# Backtest
########################################

########################################
# Split Data
chunks = ohlcv.groupby(pd.TimeGrouper(freq=TEST_PERIOD))

ks = sorted(chunks.groups.keys())

test_returns = []
buyhold_returns = []

########################################
# Train/Test on Data

for train,test in zip(ks[:-1],ks[1:]):

    # Train
    print("Train on", train.date(), "to", test.date())
    data_train = chunks.get_group(train)
    signals = train_models(data_train)

    # Test
    data_test = chunks.get_group(test)
    
    print(" Test on", test.date())
    sharpe, returns, buy_and_hold_ret = evaluate_models(data_test, signals, plot=True)

    print("  Buy and Hold Return:", buy_and_hold_ret)
    print("       Trading Return:", returns)
    print("         Sharpe Ratio:", sharpe)
    print()
    test_returns += [returns]
    buyhold_returns += [buy_and_hold_ret]

print("     Trading Return:", np.mean(test_returns),"+-",np.std(test_returns))
print("Buy and Hold Return:", np.mean(buyhold_returns),"+-",np.std(buyhold_returns))
