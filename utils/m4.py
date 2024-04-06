import warnings
warnings.filterwarnings('ignore')

from numpy.random import seed
from sklearn.neural_network import MLPRegressor
# from keras.models import Sequential
# from keras.layers import Dense, SimpleRNN
# from keras.optimizers import RMSprop
# import tensorflow as tf
from math import sqrt
import numpy as np
import pandas as pd
seed(42)
# tf.compat.v1.set_random_seed(42)
def detrend(insample_data):
    """
    Calculates a & b parameters of LRL

    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b


def deseasonalize(original_ts, ppy):
    """
    Calculates and returns seasonal indices

    :param original_ts: original data
    :param ppy: periods per year
    :return:
    """
    """
    # === get in-sample data
    original_ts = original_ts[:-out_of_sample]
    """
    if seasonality_test(original_ts, ppy):
        # print("seasonal")
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        # print("NOT seasonal")
        si = np.full(ppy, 100)

    return si


def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS

    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    """
    As noted by Professor Isidro Lloret Galiana:
    line 82:
    if len(ts_init) % 2 == 0:
    
    should be changed to
    if window % 2 == 0:
    
    This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
    In order for the results to be fully replicable this change is not incorporated into the code below
    """
    
    if len(ts_init) % 2 == 0:
        ts_ma = pd.Series(ts_init).rolling(window=window, center=True).mean()# pd.rolling_mean(ts_init, window, center=True)
        ts_ma = pd.Series(ts_ma).rolling(window=2, center=True).mean() # pd.rolling_mean(ts_ma, 2, center=True)
        ts_ma = np.roll(ts_ma, -1)
    else:
        ts_ma = pd.Series(ts_init).rolling(window=window, center=True).mean() #pd.rolling_mean(ts_init, window, center=True)

    return ts_ma


def seasonality_test(original_ts, ppy):
    """
    Seasonality test

    :param original_ts: time series
    :param ppy: periods per year
    :return: boolean value: whether the TS is seasonal
    """
    
    # Note that the statistical benchmarks, implemented in R, use the same seasonality test, but with ACF1 being squared
    # This difference between the two scripts was mentioned after the end of the competition and, therefore, no changes have been made 
    # to the existing code so that the results of the original submissions are reproducible
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit


def acf(data, k):
    """
    Autocorrelation function

    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)


def split_into_train_test(data, in_num, fh):
    """
    Splits the series into train and test sets. Each step takes multiple points as inputs

    :param data: an individual TS
    :param fh: number of out of sample points
    :param in_num: number of input points for the forecast
    :return:
    """
    train, test = data[:-fh], data[-(fh + in_num):]
    x_train, y_train = train[:-1], np.roll(train, -in_num)[:-in_num]
    x_test, y_test = train[-in_num:], np.roll(test, -in_num)[:-in_num]

    # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
    x_train = np.reshape(x_train, (-1, 1))
    x_test = np.reshape(x_test, (-1, 1))
    temp_test = np.roll(x_test, -1)
    temp_train = np.roll(x_train, -1)
    for x in range(1, in_num):
        x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
        x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
        temp_test = np.roll(temp_test, -1)[:-1]
        temp_train = np.roll(temp_train, -1)[:-1]

    return x_train, y_train, x_test, y_test


# def rnn_bench(x_train, y_train, x_test, fh, input_size):
#     """
#     Forecasts using 6 SimpleRNN nodes in the hidden layer and a Dense output layer

#     :param x_train: train data
#     :param y_train: target values for training
#     :param x_test: test data
#     :param fh: forecasting horizon
#     :param input_size: number of points used as input
#     :return:
#     """
#     # reshape to match expected input
#     x_train = np.reshape(x_train, (-1, input_size, 1))
#     x_test = np.reshape(x_test, (-1, input_size, 1))

#     # create the model
#     model = Sequential([
#         SimpleRNN(6, input_shape=(input_size, 1), activation='linear',
#                   use_bias=False, kernel_initializer='glorot_uniform',
#                   recurrent_initializer='orthogonal', bias_initializer='zeros',
#                   dropout=0.0, recurrent_dropout=0.0),
#         Dense(1, use_bias=True, activation='linear')
#     ])
#     opt = RMSprop(learning_rate=0.001)
#     model.compile(loss='mean_squared_error', optimizer=opt)

#     # fit the model to the training data
#     model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)

#     # make predictions
#     y_hat_test = []
#     last_prediction = model.predict(x_test)[0]
#     for i in range(0, fh):
#         y_hat_test.append(last_prediction)
#         x_test[0] = np.roll(x_test[0], -1)
#         x_test[0, (len(x_test[0]) - 1)] = last_prediction
#         last_prediction = model.predict(x_test)[0]

#     return np.asarray(y_hat_test)

def mlp_bench(x_train, y_train, x_test, fh):
    """
    Forecasts using a simple MLP which 6 nodes in the hidden layer

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []

    model = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                         max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
                         random_state=42)
    model.fit(x_train, y_train)
    

    last_prediction = model.predict(x_test)[0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0]

    return np.asarray(y_hat_test)


def smape(a, b):
    """
    Calculates sMAPE

    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()


def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE

    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep


forecast_horizon = {
        'Hourly':48,
        'Daily':14,
        'Weekly':13,
        'Monthly':18,
        'Quarterly':8,
        'Yearly':6,
    }
frequency = {
        'Hourly':24,
        'Daily':1,
        'Weekly':1,
        'Monthly':12,
        'Quarterly':4,
        'Yearly':1,
    }
def load_m4_data(freq=['Hourly','Daily','Weekly','Monthly','Quarterly','Yearly']):
    df_info = pd.read_csv('./M4-methods/Dataset/M4-info.csv')
    data_dict = {}
    for SP in freq:
        train = pd.read_csv(f'M4-methods/Dataset/Train/{SP}-train.csv')
        test =  pd.read_csv(f'M4-methods/Dataset/Test/{SP}-test.csv')
        data_dict[SP] = {
            'train': train,
            'test': test,
            'SP': SP,
            'freq': frequency[SP],
            'fh': forecast_horizon[SP],
            'num': len(train)
        }
    df_info = df_info[df_info['SP'].isin(freq)]
    return data_dict, df_info

class M4DatasetGenerator():
    def __init__(self, freq=['Hourly','Daily','Weekly','Monthly','Quarterly','Yearly']):
        print('Loading M4 Data...')
        self.data_dict, self.df_info = load_m4_data(freq)
        print('Loaded:')
        for SP in freq:
            print(f"    => {SP} has {self.data_dict[SP]['num']} series")
    
    def generate(self, n_series=None, random=False, seed=7, verbose=False):
        np.random.seed(seed)
        df_info, data_dict = self.df_info, self.data_dict
        if n_series is None:
            n_series = len(df_info)
        if random:
            idx = np.random.randint(low=0, high=len(df_info), size=n_series)
        else:
            idx = range(n_series)
        if verbose: print(f'Generating {len(idx)} series..')

        for serie_index in idx:
            serie_info = df_info.iloc[serie_index]
            serie_id = serie_info.M4id
            serie_sp = serie_info.SP
            fh = data_dict[serie_sp]['fh']
            freq = data_dict[serie_sp]['freq']
            train_df = data_dict[serie_sp]['train']
            test_df = data_dict[serie_sp]['test']
            # the V1 column is the name of the serie
            train_serie = train_df[train_df.V1 == serie_id].dropna(axis=1).values.reshape(-1)[1:]
            test_serie = test_df[test_df.V1 == serie_id].dropna(axis=1).values.reshape(-1)[1:]
            test_serie = test_serie[:fh] # forecast only fh steps
            train_serie = np.asarray(train_serie, dtype=np.float32)
            test_serie = np.asarray(test_serie, dtype=np.float32)
            yield train_serie, test_serie, serie_id, fh, freq, serie_sp
        

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.ml import make_batches, TransformerDataset
import torch

class MultiSerieGenerator():
    def __init__(self, freq, input_len, forecast_horizon, device, verbose=False):
        self.input_len = input_len
        self.forecast_horizon = forecast_horizon
        self.device = device
        self.verbose = verbose
        if verbose: print('Loading M4 Data...')
        self.data_dict, self.df_info = load_m4_data(freq)
        if verbose: print('Loaded:')
        for SP in freq:print(f"    => {SP} has {self.data_dict[SP]['num']} series")
    
    def get_batches(self, n_series=None, random=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        df_info, data_dict = self.df_info, self.data_dict
        if n_series is None:
            n_series = len(df_info)
        if random:
            idx = np.random.randint(low=0, high=len(df_info), size=n_series)
        else:
            idx = range(n_series)
        if self.verbose: print(f'Generating {len(idx)} series..')
        # faz o scalind individual das series completas
        scaler = MinMaxScaler((-1, 1))
        all_enc_x, all_dec_x, all_tgt_y  = [], [], []
        for serie_index in idx:
            serie_info = df_info.iloc[serie_index]
            serie_id = serie_info.M4id
            if self.verbose: print(serie_id, end=', ')
            serie_sp = serie_info.SP
            train_df = data_dict[serie_sp]['train']
            
            # the V1 column is the name of the serie
            train_serie = train_df[train_df.V1 == serie_id].dropna(axis=1).values.reshape(-1)[1:]
            train_serie = scaler.fit_transform(np.asarray(train_serie, dtype=np.float32).reshape(-1, 1)).reshape(-1)
            #
            enc_x, dec_x, tgt_y = make_batches(train_serie, self.input_len, self.forecast_horizon)
            all_enc_x.append(enc_x)
            all_dec_x.append(dec_x)
            all_tgt_y.append(tgt_y)
        # stack
        all_enc_x = torch.vstack(all_enc_x)
        all_dec_x = torch.vstack(all_dec_x)
        all_tgt_y = torch.vstack(all_tgt_y)
        # shuffle
        all_enc_x = all_enc_x.to(self.device)
        all_dec_x = all_dec_x.to(self.device)
        all_tgt_y = all_tgt_y.to(self.device)
        if self.verbose: print(f'Generated {len(all_enc_x)} batches from {len(idx)} series-> shape {all_enc_x.shape}')

        return TransformerDataset(all_enc_x, all_dec_x, all_tgt_y)



    
def get_error_dict(freq=['Hourly','Daily','Weekly','Monthly','Quarterly','Yearly']):
    return {p:{'sMAPE':[],'MASE':[]} for p in freq}

class MultiSerieGenerator():
    def __init__(self, freq, device, verbose=False):
        self.device = device
        self.verbose = verbose
        if verbose: print('Loading M4 Data...')
        self.data_dict, self.df_info = load_m4_data(freq)
        if verbose: print('Loaded:')
        for SP in freq:print(f"    => {SP} has {self.data_dict[SP]['num']} series")
    
    def get_batches(self, block_size, n_series=None, random=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        df_info, data_dict = self.df_info, self.data_dict
        if n_series is None:
            n_series = len(df_info)
        else:
            n_series = min(n_series, len(df_info))
        #
        if random:
            idx = np.random.randint(low=0, high=len(df_info), size=n_series)
        else:
            idx = range(n_series)
        if self.verbose: print(f'Generating {len(idx)} series..')
        # faz o scalind individual das series completas
        scaler = MinMaxScaler((-1, 1))
        batch_x, batch_y, batch_masks = [], [], []
        for serie_index in idx:
            
            serie_info = df_info.iloc[serie_index]
            serie_id = serie_info.M4id
            if self.verbose: print(serie_id, end=', ')
            serie_sp = serie_info.SP
            train_df = data_dict[serie_sp]['train']
            
            # the V1 column is the name of the serie
            train_serie = train_df[train_df.V1 == serie_id].dropna(axis=1).values.reshape(-1)[1:]
            #
            train_serie = scaler.fit_transform(np.asarray(train_serie, dtype=np.float32).reshape(-1, 1)).reshape(-1)
            train_serie = torch.tensor(train_serie, dtype=torch.float32)
            x, y, x_pad = get_x_y(train_serie, block_size=block_size)
            batch_x.append(x), batch_y.append(y), batch_masks.append(x_pad)
        #
        batch_x = torch.vstack(batch_x).unsqueeze(-1).to(self.device)
        batch_y = torch.vstack(batch_y).unsqueeze(-1).to(self.device)
        batch_masks = torch.vstack(batch_masks).to(self.device)

        return batch_x, batch_y, batch_masks#TransformerDataset(all_enc_x, all_dec_x, all_tgt_y)
