from math import floor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import njit
from numpy import log, nanmean
from vectorbt import Portfolio
from vectorbt.utils.decorators import custom_method


class ExtendedPortfolio(Portfolio):
    @custom_method
    def expected_log_returns(self):
        """Get log return mean series per column/group based on portfolio value."""
        @njit
        def log_nb(col, pnl):
            pnl = pnl[~np.isnan(pnl)]
            return log(pnl/100 + 1)
        #log_nb = njit(lambda col, pnl: log(pnl/100 + 1))
        mean_nb = njit(lambda col, l_rets: nanmean(l_rets))
        return self.trades.pnl.to_matrix().vbt.apply_and_reduce(log_nb, mean_nb, wrap_kwargs=dict(name_or_index="expected_log_returns"))

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

