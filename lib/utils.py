import logging
import math
import time
from datetime import timedelta
from math import floor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from vectorbt import Portfolio, Trades, MappedArray
from vectorbt.generic.nb import diff_nb
from vectorbt.utils.decorators import cached_property, cached_method


def dropnan(s):
    return s[~np.isnan(s)]

def dropnaninf(s):
    if isinstance(s, float):
        return s
    return s[~s.isin([np.nan, np.inf, -np.inf])]

class ExtendedTrades(Trades):
    @cached_property
    def lr(self) -> MappedArray:
        # @njit
        # def log_nb(col, net_rets):
        #     return np.log(net_rets + 1)
        # return self.returns.to_matrix().vbt.apply_along_axis(log_nb, axis=0).vbt.to_mapped_array()
        @njit
        def map_nb(record, *args):
            # en la posición 10 está el net return
            return np.log(record['return'] + 1)
        return self.map(map_nb)

    @cached_method
    def expected_log_returns(self, min_trades=0, min_lr=None):
        if min_lr is not None:
            # log(net_ret + 1) >=  min_lr
            #    net_ret + 1   >=  e^min_lr
            #      net_ret     >=  e^min_lr - 1
            filter_mask = self.values['return'] >= math.e**min_lr - 1
            trades = self.filter_by_mask(filter_mask)
        else:
            trades = self

        if min_trades > 0:
            filter_mask = trades.count() >= min_trades
            trades = trades[filter_mask]

        return trades.lr.mean()

    @cached_method
    def median_log_returns(self, min_trades=0):
        trades = self
        if min_trades > 0:
            filter_mask = trades.count() >= min_trades
            trades = trades[filter_mask]

        return trades.lr.median()

class ExtendedPortfolio(Portfolio):
    @cached_property
    def trades(self) -> ExtendedTrades:
        """`Portfolio.get_trades` with default arguments."""
        return ExtendedTrades.from_orders(self.orders)

@njit
def lr_nb(price_series):
    c_log = np.log(price_series)
    return diff_nb(c_log)

LR = vbt.IndicatorFactory(
    input_names=['price_series'],
    output_names=['lr']
).from_apply_func(lr_nb, use_ray=True)

def is_notebook():
    import __main__ as main
    return not hasattr(main, '__file__')

def ohlcv_csv_to_df(filepath) -> (str, pd.DataFrame):
    # Se asume que las columnas del CSV están en formato: "Open", "High",..., "Volume USDT", "Volume XXX"
    # No importa el orden de las columnas
    df = pd.read_csv(filepath, index_col=1, parse_dates=True, infer_datetime_format=True)
    symbol = df.iloc[-1]["symbol"][:-5]
    df.drop(columns=['unix', 'symbol'], inplace=True)
    # Volume BTC => Volume BTC && Volume USDT => Volume
    df.rename(columns=lambda col: "Volume" if "Volume" in col and "USDT" in col else col, inplace=True)
    # open => Open && high => High ...
    df.rename(columns=lambda col: col[0].upper() + col[1:], inplace=True)
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
    return symbol, df

def directory_to_data_frame_list(directory) -> List[Tuple[str, pd.DataFrame]]:
    series = []
    path_list = list(Path(directory).glob('*.csv'))
    for path in path_list:
        series.append(ohlcv_csv_to_df(path))
    return series

def create_windows(ohlc: pd.Series, n=5, window_len=0.6, right_set_len=0.4) -> ((), ()):
    split_kwargs = dict(
        n=n,
        window_len=floor(len(ohlc) * window_len),
        set_lens=(right_set_len,),
        left_to_right=False
    )  # n windows, each window_len long, reserve training_set_len days for test/training(nomeacuerdo) #todo revisar

    # (train_price, train_indexes), (test_price, test_indexes)
    windows = ohlc.vbt.rolling_split(**split_kwargs)
    split_kwargs["plot"] = True
    split_kwargs["trace_names"] = ['in-sample', 'out-sample']
    fig = ohlc.vbt.rolling_split(**split_kwargs)
    return fig, windows

def get_best_index(performance, higher_better=True):
    if higher_better:
        return performance[performance.groupby('split_idx').idxmax()].index
    return performance[performance.groupby('split_idx').idxmin()].index

def get_params_by_index(index, level_name):
    return index.get_level_values(level_name).to_numpy()

def get_best_pairs(performance, param_1_name, param_2_name, return_index=False):
    in_best_index = get_best_index(performance)
    in_best_param1 = get_params_by_index(in_best_index, param_1_name)
    in_best_param2 = get_params_by_index(in_best_index, param_2_name)
    if return_index:
        return in_best_index, np.array(list(zip(in_best_param1, in_best_param2)))
    return np.array(list(zip(in_best_param1, in_best_param2)))

@njit
def resample_ohlcv(df, new_frequency, columns=None):
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    if not columns:
        columns = list(filter(lambda col: col in ohlc_dict.keys(), df.columns))

    apply_dict = {k: ohlc_dict[k] for k in columns}
    return df.resample(new_frequency, closed='left', label='left').apply(apply_dict)

def where_true_set_series(series, data):
    data = data.copy()
    data.where(data == True, np.nan, inplace=True)
    data.where(data != 1, series, inplace=True)
    return data

def plot_series_vs_scatters(series_list: list, booleans_list):
    index = None
    series = series_list.pop(0)
    fig = series.vbt.plot()
    while len(series_list):
        series = series_list.pop(0)
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index, copy=True)
        elif index is None:
            index = series.index
        fig = series.vbt.plot(fig=fig)
    i = 1
    for scatter in booleans_list:
        if not isinstance(scatter, pd.Series):
            scatter = pd.Series(scatter, index=index, copy=True)
        elif index is None:
            index = series.index
        scatter = where_true_set_series(series, scatter)
        scatter.name = i
        i += 1
        fig = scatter.vbt.scatterplot(fig=fig)
    return fig

# preallocate empty array and assign slice by chrisaycock
@njit
def shift_np(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

@njit
def positive_return_prob(col, arr, *args):
    indexes = np.where(np.isfinite(arr))[0]
    close = args[0]
    fee = args[1]
    min_trades = args[2]
    n = 1
    adder = 0
    counter = 0
    while n < len(indexes):
        i = indexes[n]
        prev = indexes[n-1]
        #if entry then exit
        if arr[prev] == 1 and arr[i] == -1:
            # if positive return
            if (close[i]/close[prev]) * (1 - fee)**2 > 1:
                adder += 1
            counter += 1
        n += 1
    return adder / counter if counter > min_trades else np.nan

@njit
def trades_count(col, arr):
    indexes = np.where(np.isfinite(arr))[0]
    n = 1
    counter = 0
    while n < len(indexes):
        i = indexes[n]
        prev = indexes[n-1]
        if arr[prev] == 1 and arr[i] == -1:
            counter += 1
        n +=1
    return counter

@njit
def k_mean(col, arr, *args):
    indexes = np.where(np.isfinite(arr))[0]
    close = args[0]
    fee = args[1]
    min_trades = args[2]
    n = 1
    adder = 0
    counter = 0
    while n < len(indexes):
        i = indexes[n]
        prev = indexes[n-1]
        if arr[prev] == 1 and arr[i] == -1:
            adder += math.log(close[i]/close[prev] * (1 - fee)**2)
            counter += 1
        n +=1
    return adder / counter if counter > min_trades else np.nan

def signals_to_ones(entries, exits, columns=None):
    if columns is None:
        columns = entries.columns
    entries = np.where(entries == True, 1, np.nan)
    return pd.DataFrame(np.where(exits == True, -1, entries), columns=columns)

class ElapsedFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.last_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.last_time
        self.last_time = time.time()
        # using timedelta here for convenient default formatting
        elapsed = timedelta(seconds=elapsed_seconds)
        return "{} {}".format(elapsed, record.getMessage())

# https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    resp = []
    # looping till length l
    for i in range(0, len(l), n):
        resp.append(l[i:i + n])
    return resp

# def test():
#     e_all_signals_in_ones = np.array([[np.nan],[np.nan],[np.nan],[1],[np.nan],[-1],[np.nan],[1],[1],[np.nan],[np.nan]])
#     all_signals_in_ones = signals_to_ones(final_entries, lr_exits, columns=[(lag, thld)])
#     assert (np.array_equal(all_signals_in_ones, e_all_signals_in_ones, equal_nan=True))
#
#     e_trade_count = 1
#     e_positive_return = 1
#     e_avg_lr = math.log(math.e**4/math.e**3)
#     avg_lr = k_mean("", all_signals_in_ones.to_numpy().flatten(), p.flatten())
#     positive_return = positive_return_prob("", all_signals_in_ones.to_numpy().flatten(), p.flatten(), fee)
#     trade_count = trades_count("", all_signals_in_ones.to_numpy().flatten())
#     assert (avg_lr == e_avg_lr)
#     assert (positive_return == e_positive_return)
#     assert (trade_count == e_trade_count)