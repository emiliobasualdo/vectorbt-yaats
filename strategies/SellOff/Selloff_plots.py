#%%
import itertools
import p_tqdm
import vectorbt as vbt
import os
import sys
import numpy as np
import pandas as pd
from numba import njit
from plotly.subplots import make_subplots

from strategies.SellOff.SellOff import simulate_lrs, ENTRY_SIGNALS, ma_mstd

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from lib.utils import where_true_set_series, ohlcv_csv_to_df, LR, shift_np


#%%

def plot_series_vs_scatters(series_list: list, booleans_list):
    series = series_list.pop(0)
    fig = series.vbt.plot()
    while len(series_list):
        series = series_list.pop(0)
        series.vbt.plot(fig=fig)
    for scatter in booleans_list:
        name = scatter.name
        scatter = where_true_set_series(series, scatter)
        scatter.name = name
        fig = scatter.vbt.scatterplot(fig=fig, trace_names=name)
    return fig

def add_all_subplots(fig, row, col, list):
    for a in list:
        fig.add_trace(a, row=row, col=col)

#%%

file = f"/Users/pilo/development/itba/pf/Binance_Minute_OHLC_CSVs/3000/Binance_ADAUSDT_minute_3000.csv"
_, ohlcv = ohlcv_csv_to_df(file)
# Tomamos el close como precio.
close = ohlcv["Close"]
volume = ohlcv["Volume"]
del ohlcv

lag_range = [10, 20, 30, 40]
vol_range = [0.5, 0.8, 1, 1.2, 1.5, 1.7, 2]
lr_range =  list(map(lambda x: -x, vol_range))
exit_wait_range = [2, 3, 4]
combinations =list(itertools.product(lag_range, vol_range, lr_range, exit_wait_range))

lr_ind = LR.run(close)
lr_ind.lr.rename("lr_close", inplace=True)
index=lr_ind.lr.index

shifted_lr = shift_np(lr_ind.lr.to_numpy(), 1)
shifted_lr = shifted_lr.reshape((shifted_lr.shape[0],1))
shifted_vol = shift_np(volume.to_numpy(), 1)
shifted_vol = shifted_vol.reshape((shifted_vol.shape[0],1))

#%%

resps = {}
def create_plot(combination):
    lag = combination[0]
    vol_thld = combination[1]
    lr_thld = combination[2]  # lr_thld < 0
    exit_wait = combination[3]

    lr_ma, lr_mstd, vol_ma = ma_mstd(shifted_lr, shifted_vol, lag)
    signal = ENTRY_SIGNALS.run(lr=lr_ind.lr, shifted_lr=shifted_lr,
                                vol=volume, shifted_vol=shifted_vol,
                                lag=lag, lr_thld=lr_thld, vol_thld=vol_thld, exit_wait=exit_wait,
                                short_name="signals")

    lr_mstd_th = lr_ma + lr_thld * lr_mstd  # lr_thld < 0
    lr_mstd_th = lr_mstd_th.reshape((lr_mstd_th.shape[0],))
    lr_mstd_th = pd.Series(lr_mstd_th, index=index, copy=True, name="lr_mstd_th")

    vol_ma_th = vol_thld * vol_ma
    vol_ma_th = vol_ma_th.reshape((vol_ma_th.shape[0],))
    vol_ma_th = pd.Series(vol_ma_th, index=index, copy=True, name="vol_ma_th")

    entries = signal.entries
    entries.name = "entries"
    exits = signal.exits
    exits.name = "exits"

    del signal, lr_ma, lr_mstd, vol_ma
    lr_plot = plot_series_vs_scatters([lr_mstd_th, lr_ind.lr], [entries, exits])
    vol_plot = plot_series_vs_scatters([vol_ma_th, volume], [entries, exits])
    close_plot = plot_series_vs_scatters([close], [entries, exits])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    add_all_subplots(fig, 1, 1, lr_plot.data)
    add_all_subplots(fig, 2, 1, vol_plot.data)
    add_all_subplots(fig, 3, 1, close_plot.data)


    fig.update_layout(height=700, legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        title= f"lag={lag} lr_thld={lr_thld} vol_thld={vol_thld} k={exit_wait}"
    ))
    filename = f"lag_{lag}-lr_thld_{lr_thld}-vol_thld_{vol_thld}-k_{exit_wait}"
    fig.write_html(f"/Users/pilo/development/itba/pf/vectorbt-yaats/strategies/SellOff/plots/k{exit_wait}/{filename}.html")

p_tqdm.p_map(create_plot, combinations)
#create_plot(combinations[0])
#%%


