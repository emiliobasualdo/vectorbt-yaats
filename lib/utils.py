import math
from math import floor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from numpy import log, nanmean
from vectorbt import Portfolio
from vectorbt.generic.nb import diff_nb
from vectorbt.utils.decorators import custom_method


def dropnan(s):
    return s[~np.isnan(s)]


class ExtendedPortfolio(Portfolio):
    @custom_method
    def expected_log_returns(self):
        """Get log return mean series per column/group based on portfolio value."""

        @njit
        def log_nb(col, pnl):
            # pnl = pnl[~np.isnan(pnl)]
            return log(pnl / 100 + 1)

        # log_nb = njit(lambda col, pnl: log(pnl/100 + 1))
        mean_nb = njit(lambda col, l_rets: nanmean(l_rets))
        return self.trades.pnl.to_matrix().vbt.apply_and_reduce(log_nb, mean_nb,
                                                                wrap_kwargs=dict(name_or_index="expected_log_returns"))

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


def file_to_data_frame(filepath) -> (str, pd.DataFrame):
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
        series.append(file_to_data_frame(path))
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

def dropnaninf(performance):
    if isinstance(performance, float):
        return performance
    return performance[~performance.isin([np.nan, np.inf, -np.inf])]

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

@njit
def median(col, arr, *args):
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