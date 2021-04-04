from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


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
