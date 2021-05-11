import gc
import shutil
from datetime import timedelta
import time
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
from vectorbt.generic import nb as generic_nb

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from lib.utils import ohlcv_csv_to_df, LR, ExtendedPortfolio, shift_np, ElapsedFormatter, dropnan


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


def create_portfolio(file, fee, lr_thld, vol_thld, lag) -> ExtendedPortfolio:
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

    logging.info('Creating indicators')
    # Calculamos el log return de los precios(log return indicator)
    lr_ind = LR.run(close)

    # Calculamos la señal de entrada y salida para cada combinación de lr_thld, vol_thld y lag.
    signals = ENTRY_SIGNALS.run(lr=lr_ind.lr, shifted_lr=shift_np(lr_ind.lr.to_numpy(), 1),
                                vol=volume, shifted_vol=shift_np(volume.to_numpy(), 1),
                                lr_thld=lr_thld, vol_thld=vol_thld, lag=lag,
                                param_product=True, short_name="signals")

    portfolio_kwargs = dict(
        direction='longonly',  # Solo long, no se shortea
        freq='m',
        size=np.inf,  # Se compra y vende toda la caja en cada operación.
        fees=fee,
    )
    del volume, lr_ind
    logging.info('Simulating portfolio')
    port = ExtendedPortfolio.from_signals(close, signals.entries, signals.exits, **portfolio_kwargs, max_logs=0)
    del close, signals
    return port


def plots_from_trades(trades, min_trades=500, min_lr=0.0, save_dir=None, parameters_to_save=None):
    gc.collect()
    results = []
    logging.info('Creating results')
    # ploteamos elr donde # trades >= min_Trades y elr > 0
    elr = trades.expected_log_returns(min_trades=min_trades)
    results.append({
        "data": dropnan(elr[elr > 0]),
        "title": f"AVG(log(p_exit/p_entry * (1-fee)**2)), AVG > 0, #Trades > {min_trades}",
        "name": "ELR"
    })

    # ploteamos elr donde lr > 0 la media se toma como la suma  (lr > 0) / (#lr>0)), # trades >= min_Trades y elr > 0
    elr_filtered = trades.expected_log_returns(min_trades=min_trades, min_lr=min_lr)
    results.append({
        "data": dropnan(elr_filtered[elr_filtered > 0]),
        "title": f"AVG(log(p_exit/p_entry * (1-fee)**2)), log(...) > 0, AVG > 0, #Trades > {min_trades}",
        "name": "ELR_positive_LR"
    })

    # ploteamos media donde # trades >= min_Trades y mlr > 0
    mlr = trades.median_log_returns(min_trades=min_trades)
    results.append({
        "data": dropnan(mlr[mlr > 0]),
        "title": f"Median(log(p_exit/p_entry * (1-fee)**2)), Median > 0, #Trades > {min_trades}",
        "name": "MLR"
    })
    del trades
    if save_dir is not None:
        logging.info('Saving parameters')
        # create or replace dir if exists
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        # contador para llevar un orden visual por orden de creación
        plot_counter = 0
        if parameters_to_save is not None:
            # guardamos los parámetros para estar al tanto de sus valores al momento de estudiarlos
            filepath = f"{save_dir}/{plot_counter}-params.txt"
            plot_counter += 1
            with open(filepath, 'w') as file:
                file.write(json.dumps(parameters_to_save, indent=2))

        logging.info('Saving volumes and heatmaps')
        # todo paralelizar
        for i in range(len(results)):
            if results[i]["data"].size:
                # guardamos los gráficos 3d
                filepath = f"{save_dir}/{plot_counter}-{results[i]['name']}_volume.html"
                results[i]["data"].vbt.volume(title=results[i]["title"]).write_html(filepath)

                # guardamos Plots 2d de la optimización lag , lr_thld ,vol_thld con lag como slider
                filepath = f"{save_dir}/{plot_counter}-{results[i]['name']}_heatmap.html"
                results[i]["data"].vbt.heatmap(title=results[i]["title"], slider_level='signals_lag').write_html(
                    filepath)

                plot_counter += 1

        logging.info('Saving results as csv')
        # guardamos las tablas de los top resultados en 1 csv
        csv_filepath = f"{save_dir}/{plot_counter}-nlargest_results.csv"
        plot_counter += 1
        pd.concat(list(map(lambda df: df['data'].nlargest(20), results))).to_csv(csv_filepath)


if __name__ == '__main__':
    # add custom formatter to root logger for simple demonstration
    handler = logging.StreamHandler()
    handler.setFormatter(ElapsedFormatter())
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    # vbt.settings.caching['enabled'] = False
    symbol = "ADA"
    candles = "_3000"
    file = f"/Users/pilo/development/itba/pf/Binance_Minute_OHLC_CSVs/{candles[1:]}/Binance_{symbol}USDT_minute{candles}.csv"

    lr_thld = -np.linspace(0, 3, 30, endpoint=True)
    vol_thld = np.linspace(0, 4, 30, endpoint=True)
    lag = list(range(6, 100, 2))
    fee = 0.001

    port = create_portfolio(file, fee, lr_thld, vol_thld, lag)
    trades = port.trades
    del port
    min_trades = 5
    min_lr = 0.0
    parameters_to_save = {
        "symbol": symbol,
        "candels": candles,
        "file": file,
        "lr_thld": f"range({lr_thld[0]},{lr_thld[-1]}, steps={len(lr_thld)})",
        "vol_thld": f"range({vol_thld[0]},{vol_thld[-1]}, steps={len(vol_thld)})",
        "lag": f"range({lag[0]},{lag[-1]}, steps={len(lag)})",
        "fee": fee,
        "min_trades": min_trades,
        "min_lr = 0.0": min_lr
    }
    save_dir = f"./{symbol}{candles}"
    plots_from_trades(trades, min_trades=min_trades, min_lr=min_lr, save_dir=save_dir,
                      parameters_to_save=parameters_to_save)