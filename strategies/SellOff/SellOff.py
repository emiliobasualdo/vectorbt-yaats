import argparse
import gc
import glob
import shutil
from datetime import timedelta
import time
from functools import partial
from multiprocessing import Pool

import pandas
import vectorbt as vbt
import math
import os
import sys
import numpy as np
import pandas as pd
import json
import logging
from numba import njit
from plotly.subplots import make_subplots
from vectorbt import MappedArray
from vectorbt.generic import nb as generic_nb
from tqdm import tqdm

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from lib.utils import ohlcv_csv_to_df, LR, ExtendedPortfolio, shift_np, ElapsedFormatter, dropnan, divide_chunks, \
    replace_dir


@njit
def signal_calculations(lr, lr_ma, lr_mstd, vol, vol_ma, lr_thld, vol_thld):
    lr_thld_std = lr_thld * lr_mstd
    # lr_thld < 0
    lr_entries = np.where(lr < lr_ma + lr_thld_std, True, False)
    lr_exits = np.where(lr > lr_ma - lr_thld_std, True, False)
    vol_entries = np.where(vol > vol_thld * vol_ma, True, False)
    return lr_entries, lr_exits, vol_entries


@njit
def ma_mstd(shifted_lr, shifted_volume, lag):
    lr_ma = generic_nb.rolling_mean_nb(shifted_lr, lag)
    lr_mstd = generic_nb.rolling_std_nb(shifted_lr, lag)
    vol_ma = generic_nb.rolling_mean_nb(shifted_volume, lag)
    return lr_ma, lr_mstd, vol_ma


@njit
def signals_nb(lr, shifted_lr, vol, shifted_vol, lr_thld, vol_thld, lag):
    """
    Primero calculamos los moving average del volume y los ma y std de lr.
    Se calculan sobre vol y lr shifteados(shifted_lr y shifted_vol) una posición hacia adelante para que, e.i.,
    lr[i+1] > lr_ma[i] - lr_std[i].
    Esta función se corre para cada combinación de lr_thld, vol_thld, lag por ende, recibimos
    shifted_lr y shifted_vol por parámetro para no tener que calcularlo en cada iteración.
    """
    lr_ma, lr_mstd, vol_ma = ma_mstd(shifted_lr, shifted_vol, lag)
    # luego calculamos las 3 señales distintas: LRS+[i], LRS-[i] y VS[i]
    lr_entries, lr_exits, vol_entries = signal_calculations(lr, lr_ma, lr_mstd, vol, vol_ma, lr_thld, vol_thld)
    # por último: entry = LRS-[i] & VS[i]
    final_entries = lr_entries & vol_entries
    return final_entries, lr_exits


ENTRY_SIGNALS = vbt.IndicatorFactory(
    input_names=['lr', 'shifted_lr', 'vol', 'shifted_vol'],
    param_names=['lr_thld', 'vol_thld', 'lag'],
    output_names=['entries', 'exits']
).from_apply_func(signals_nb, use_ray=True)


def merge_intermediate_results(results: [{}]) -> {}:
    merged = {}
    keys = results[0].keys()  # = ["ELR", "ELR_positive_LR", "MLR", ...]
    for k in keys:
        merged[k] = {
            "data": pd.concat(map(lambda r: r[k]['data'], results)),
            "title": results[0][k]["title"]
        }
    return merged


def calculate_metrics(trades_lr: MappedArray, min_trades, min_lr=0) -> {}:
    trade_count = trades_lr.count()
    trades_lr = trades_lr[trade_count >= min_trades]

    results = {}
    # ploteamos elr donde # trades >= min_Trades y elr > 0
    elr = trades_lr.mean()
    results["ELR"] = {
        "data": elr,
        "title": f"AVG(log(p_exit/p_entry * (1-fee)**2)),  #Trades > {min_trades}",
    }

    # filtramos los lr que sean < min_lr
    # ploteamos elr donde lr > 0 la media se toma como la suma  (lr > 0) / (#lr>0)), # trades >= min_Trades y elr > 0
    @njit
    def func(col, arr, *args):
        arr = arr[arr > min_lr]
        return arr.sum() / arr.size if arr.size > 0 else np.nan
    elr_filtered = trades_lr.reduce(func)
    elr_filtered.name = "Positive_ELR"
    results['ELR_positive_LR'] = {
        "data": elr_filtered,
        "title": f"AVG(log(p_exit/p_entry * (1-fee)**2)), log(...) > {min_lr}, #Trades > {min_trades}",
    }

    # ploteamos media donde # trades >= min_Trades y mlr > 0
    results['MLR'] = {
        "data": trades_lr.median(),
        "title": f"Median(log(p_exit/p_entry * (1-fee)**2)), #Trades > {min_trades}",
    }
    # ploteamos # trades > min_trades
    results['Trade_count'] = {
        "data": trade_count,
        "title": f"#Trades > {min_trades}",
    }
    return results

pbar = tqdm()
def simulate_chunk(lag_partition, signals_static_args:dict, close, min_trades, portfolio_kwargs):
    # Calculamos la señal de entrada y salida para cada combinación de lr_thld, vol_thld y lag.
    signals = ENTRY_SIGNALS.run(**signals_static_args, lag=lag_partition,
                                param_product=True, short_name="signals")
    port = ExtendedPortfolio.from_signals(close, signals.entries, signals.exits, **portfolio_kwargs)
    # filtramos todas aquellas corridas que tengan menos de min_trades
    lr = port.trades.lr
    del signals, port
    metrics = calculate_metrics(lr, min_trades)
    #logging.info(f"Chunk {lag_partition[0]} done")
    pbar.update(1)
    return metrics

def simulate_lrs(file, portfolio_kwargs, lr_thld, vol_thld, lag, max_chunk_size, min_trades) -> {}:
    """
    Simulamos un portfolio para optimizar: lr_thld, vol_thld y lag.
    Acá no consideramos ni Stop Loss ni Take Profit.
    Importante: lr_thld < 0

    Calculamos la señal de entrada a partir de subidas anormales de Volumen y caidas anormales del Log Return.
    LRS-[i] = LR[i] < avg(LR[i - lag], ..., LR[i - 1]) - std(LR[i - lag], ..., LR[i - 1]) * T_l
    VS[i] = V[i] > avg(V[i - lag], ..., V[i - 1]) * T_v
    entry = LRS-[i] & VS[i]
    Usamos como señal de salida subidas anormales del Log Return
    LRS+[i] = LR[i] < avg(LR[i - lag], ..., LR[i - 1]) + std(LR[i - lag], ..., LR[i - 1]) * T_l
    exit = LRS+[i]
    """

    # Levantamos un CSV
    logging.info('Loading ohlcv csv')
    _, ohlcv = ohlcv_csv_to_df(file)
    # Tomamos el close como precio.
    close = ohlcv["Close"]
    volume = ohlcv["Volume"]
    del ohlcv

    logging.info('Creating lr indicator')
    # Calculamos el log return de los precios(log return indicator)
    lr_ind = LR.run(close)

    logging.info('Simulating portfolio')

    # Corrermos simulaciones cada chunks de 8GB.
    # Partimos el lag para que forme chunks de 8GB. Obs: usamos float64 => 8 bytes
    total_gbs = close.size * len(lr_thld) * len(vol_thld) * len(lag) * 8 / (1 << 30)
    logging.info(f'Matrix shape={(close.size, (len(lr_thld), len(vol_thld), len(lag)))}, weight={round(total_gbs,2)}GB')
    lags_partition_len = math.floor(max_chunk_size * float(1 << 30) / (close.size * len(lr_thld) * len(vol_thld) * 8))
    lag_chunks = divide_chunks(lag, lags_partition_len)
    logging.info(f'Chunks len={lags_partition_len}, #chunks={len(lag_chunks)}')
    pbar.total = len(lag_chunks)
    # multi-procesamos
    signals_static_args = dict(
        lr=lr_ind.lr, shifted_lr=shift_np(lr_ind.lr.to_numpy(), 1),
        vol=volume, shifted_vol=shift_np(volume.to_numpy(), 1),
        lr_thld=lr_thld, vol_thld=vol_thld
    )
    partial_simulate_chunk = partial(simulate_chunk, signals_static_args=signals_static_args, close=close, min_trades=min_trades, portfolio_kwargs=portfolio_kwargs)
    with Pool() as p:
        metrics = p.map(partial_simulate_chunk, lag_chunks)
    p.close()
    del volume, lr_ind, close
    return merge_intermediate_results(metrics)


def plots_from_metrics(metrics: {}, save_dir):
    logging.info('Saving plots')
    # contador para llevar un orden visual por orden de creación
    plot_counter = 0
    # todo paralelizar
    for metric_name, metric in metrics.items():
        if not metric["data"].size:
            logging.info(f"Empty results for {metric_name}")
            continue
        # guardamos los gráficos 3d
        filepath = f"{save_dir}/{plot_counter}-{metric_name}_volume.html"
        metric["data"].vbt.volume(title=metric["title"]).write_html(filepath)

        # guardamos los box&whiskers
        filepath = f"{save_dir}/{plot_counter}-{metric_name}_bnw.html"
        metric["data"].vbt.boxplot(title=metric["title"]).write_html(filepath)

        # guardamos Plots 2d de la optimización lag , lr_thld ,vol_thld con lag como slider
        filepath = f"{save_dir}/{plot_counter}-{metric_name}_heatmap.html"
        metric["data"].vbt.heatmap(title=metric["title"], slider_level='signals_lag').write_html(filepath)


        plot_counter += 1

    # guardamos los resultados en un csv
    all_metrics_df = pd.concat(map(lambda m: m["data"], metrics.values()), axis=1)
    with open(f"{save_dir}/{plot_counter}-metrics.csv", 'w') as writer:
        writer.write(all_metrics_df.to_csv())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate Sell Off.')
    parser.add_argument('ohlcv_csv', type=str, help="Open high low close volume data in .csv file")
    parser.add_argument('-m', '--min_trades', type=int, default=5,
                        help="Min amount of trades per simulation to consider the simulation as as meaningful")
    parser.add_argument('-c', '--max_chunk_size', type=int, default=8, help="Max chunk size to simulate in Gigabytes")
    args = parser.parse_args()
    filepath = args.ohlcv_csv
    min_trades = args.min_trades
    max_chunk_size = args.max_chunk_size

    # add custom formatter to root logger for simple demonstration
    handler = logging.StreamHandler()
    handler.setFormatter(ElapsedFormatter())
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    # parametros a optimizar en el primer paso
    lr_thld = -np.linspace(0, 3, 30, endpoint=True)
    vol_thld = np.linspace(0, 4, 30, endpoint=True)
    lag = list(range(6, 100, 2))

    portfolio_kwargs = dict(
        direction='longonly',  # Solo long, no se shortea
        freq='m',
        size=np.inf,  # Se compra y vende toda la caja en cada operación.
        fees=0.001,
        max_logs=0,
    )

    metrics = simulate_lrs(filepath, portfolio_kwargs, lr_thld, vol_thld, lag, max_chunk_size, min_trades)

    # guardamos los parámetros para estar al tanto de sus valores al momento de estudiarlos
    _, filename = os.path.split(filepath)
    save_dir = f"./{filename[:-4]}"
    replace_dir(save_dir)
    parameters_to_save = {
        **vars(args),
        **portfolio_kwargs,
        "lr_thld": f"range({lr_thld[0]},{lr_thld[-1]}, steps={len(lr_thld)})",
        "vol_thld": f"range({vol_thld[0]},{vol_thld[-1]}, steps={len(vol_thld)})",
        "lag": f"range({lag[0]},{lag[-1]}, steps={len(lag)})",
    }
    # create or replace dir if exists
    with open(f"{save_dir}/parameters.txt", 'w') as file:
        file.write(json.dumps(parameters_to_save, indent=2))

    # graficamos y guardamos las métricas
    plots_from_metrics(metrics, save_dir=save_dir)
