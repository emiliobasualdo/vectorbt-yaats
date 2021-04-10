import argparse
import logging
import os
import sys
from datetime import datetime
from math import floor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from numpy import log, nanmean, arange
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


def file_to_data_frame(filepath) -> (str, pd.DataFrame):
    df = pd.read_csv(filepath, index_col=1, parse_dates=True, infer_datetime_format=True)
    symbol = df.iloc[-1]["symbol"][:-5]
    df.drop(columns=['unix', 'symbol'], inplace=True)
    # Volume BTC => Volume && Volume USDT => Volume USDT
    df.rename(columns=lambda col: "Volume" if "Volume" in col and "USDT" not in col else col, inplace=True)
    # open => Open && high => High ...
    df.rename(columns=lambda col: col[0].upper() + col[1:], inplace=True)
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
    return symbol, df


def file_to_data_frame_list(filepath="", directory="") -> List[Tuple[str, pd.DataFrame]]:
    series = []
    if filepath != "":
        series.append(file_to_data_frame(filepath))
    else:
        path_list = list(Path(directory).glob('**/*.csv'))
        for path in path_list:
            series.append(file_to_data_frame(path))
    return series


def expected_log_returns(trades_df: pd.DataFrame) -> float:
    # todo revisar cuentas con los muchachos
    trades_df["Return"] = trades_df["ExitPrice"] / trades_df["EntryPrice"]
    trades_df["Log Returns"] = np.log(trades_df["Return"])
    return 0 if not len(trades_df.index) else trades_df["Log Returns"].sum() / len(trades_df.index)


logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def create_windows(ohlc: pd.Series, n=5, window_len=0.6, right_set_len=0.4) -> ((), ()):
    split_kwargs = dict(
        n=n,
        window_len=floor(len(ohlc) * window_len),
        set_lens=(window_len * right_set_len,),
        left_to_right=False
    )  # n windows, each window_len long, reserve training_set_len days for test/training(nomeacuerdo) #todo revisar

    # (train_price, train_indexes), (test_price, test_indexes)
    windows = ohlc.vbt.rolling_split(**split_kwargs)
    split_kwargs["plot"] = True
    split_kwargs["trace_names"] = ['in-sample', 'out-sample']
    fig = ohlc.vbt.rolling_split(**split_kwargs)
    return fig, windows

