import argparse
import logging
import os
import sys
from datetime import datetime
from math import floor
from pathlib import Path
from typing import List, Tuple

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
        log_nb = njit(lambda col, pnl: log(pnl/100 + 1))
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


def create_windows(ohlc: pd.Series, n=5, window_len=0.6, training_set_len=0.4) -> ((), ()):
    split_kwargs = dict(
        n=n,
        window_len=floor(len(ohlc) * window_len),
        set_lens=(training_set_len,),
        left_to_right=False
    )  # n windows, each window_len long, reserve training_set_len days for test/training(nomeacuerdo) #todo revisar

    # (train_price, train_indexes), (test_price, test_indexes)
    windows = ohlc.vbt.rolling_split(**split_kwargs)
    split_kwargs["plot"] = True
    split_kwargs["trace_names"] = ['in-sample', 'out-sample']
    fig = ohlc.vbt.rolling_split(**split_kwargs)
    return fig, windows


def window_cross_validation(dirname, ):
    # Parseamos los argumentos de línea de comando
    parser = argparse.ArgumentParser(description='Simulate your stuff')
    parser.add_argument('-f', '--ohlc_file_path', type=str)
    parser.add_argument('-l', '--log_to_file', action="store_true")
    parser.add_argument('-p', '--plot', action="store_true")
    parser.add_argument('-c', '--save_csv', action="store_true")
    args = parser.parse_args()
    ohlc_file_path = args.ohlc_file_path
    log_to_file = args.ohlc_file_path
    plot = args.ohlc_file_path
    save_to_csv = args.save_csv

    start_time = datetime.now()

    if log_to_file or plot or save_to_csv:
        RESULTS_FILE_PREFIX = f"./results/{dirname}/{start_time.strftime('%Y-%m-%d_%H:%M:%S')}"
        # Creamos la carpeta donde se van a guardar los resultados/logs
        # todo guardar en S3 para poder apagar la máquina
        if not os.path.exists(RESULTS_FILE_PREFIX):
            os.makedirs(RESULTS_FILE_PREFIX)
        if log_to_file:
            # Podemos guardar los logs en un archivo
            fh = logging.FileHandler(f"{RESULTS_FILE_PREFIX}/logs.log")
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    # leemos el csv
    logging.info(f"Reading file {ohlc_file_path}")
    (s_name, data) = file_to_data_frame(ohlc_file_path)
    price = data["Close"]

    # creamos las ventanas
    (training_price, train_indexes), (test_price, test_indexes) = create_windows(ohlc=price, plot=True)

    # simulamos Buy&Hold de cada training y test window y tomamo el expected log returns (elr)
    portfolio_kwargs = dict(
        direction='all',  # long and short todo falta ver esto
        freq='d'  # todo falta ver esto
    )
    in_hold_elr = ExtendedPortfolio.from_holding(training_price, **portfolio_kwargs).expected_log_returns()
    out_hold_elr = ExtendedPortfolio.from_holding(test_price, **portfolio_kwargs).expected_log_returns()

    # Simulamos con el training set y el rango de parámetros
    params_range = arange(10, 50)
    training_elr = optimize(training_price, params_range, **portfolio_kwargs).expected_log_returns()

    # Podemos simular el test set para buscar el cúal serían las mejores configuraciónes para cada ventana
    optimize_test_set = False
    if optimize_test_set:
        out_elr = optimize(test_price, params_range, **portfolio_kwargs).expected_log_returns()

    def get_best_index(performance, higher_better=True):
        if higher_better:
            return performance[performance.groupby('split_idx').idxmax()].index
        return performance[performance.groupby('split_idx').idxmin()].index

    def get_best_params(best_index, level_name):
        return best_index.get_level_values(level_name).to_numpy()

    # Buscamos los mejores índices resultado de cada training window
    training_best_index = get_best_index(training_elr)
    # Buscamos los mejores parámetros por ventana
    training_best_fast_windows = get_best_params(training_best_index, 'fast_window')
    training_best_slow_windows = get_best_params(training_best_index, 'slow_window')
    training_best_window_pairs = np.array(list(zip(training_best_fast_windows, training_best_slow_windows)))

    # Simulamos con el test set y los mejores parámetros de cada ventana de training set
    test_elr = simulate(test_price, training_best_fast_windows, training_best_slow_windows,
                        **portfolio_kwargs).expected_log_returns()

    final_result = pd.DataFrame({
        'in_sample_hold': in_hold_elr.values,
        # 'in_sample_median': training_elr.groupby('split_idx').median().values,
        'in_sample_best': training_elr[training_best_index].values,
        'out_sample_hold': out_hold_elr.values,
        # 'out_sample_median': out_elr.groupby('split_idx').median().values,
        'out_sample_test': test_elr.values
    })

    if save_to_csv:
        # guardamos los mejores parámetros por ventana
        pd.DataFrame(training_best_window_pairs, columns=['fast_window', 'slow_window']).to_csv(
            f"{RESULTS_FILE_PREFIX}/best_params_combination.csv")
        # guardamos los resultados finales
        final_result.to_csv(f"{RESULTS_FILE_PREFIX}/final_strat-vs-hold.csv")

    logging.info(f"Time taken {datetime.now() - start_time}")

    if plot:
        final_result.vbt.plot(
            trace_kwargs=[
                dict(line_color=vbt.settings.color_schema['blue']),
                dict(line_color=vbt.settings.color_schema['blue'], line_dash='dash'),
                dict(line_color=vbt.settings.color_schema['blue'], line_dash='dot'),
                dict(line_color=vbt.settings.color_schema['orange']),
                dict(line_color=vbt.settings.color_schema['orange'], line_dash='dash'),
                dict(line_color=vbt.settings.color_schema['orange'], line_dash='dot')
            ]
        )
