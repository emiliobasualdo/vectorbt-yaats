#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from math import e

from vectorbt.generic.nb import diff_nb
from lib.utils import directory_to_data_frame_list, ExtendedPortfolio, create_windows, get_best_pairs
from lib.utils import is_notebook


# In[ ]:


current_dir = os.path.dirname(os.path.realpath("__file__"))
directory = "/home/ec2-user/data/long"
ohlcv_series_list = directory_to_data_frame_list(directory)
# concatenamos los dfs
names = list(map(lambda t: t[0], ohlcv_series_list))
dfs = list(map(lambda t: t[1], ohlcv_series_list))

ohlc_dict = {
'Open':'first',
'High':'max',
'Low':'min',
'Close': 'last',
'Volume': 'sum'
}
# Resampleamos la información a candles de cada 5 min para reducir la memora necesaria
# además solo agarramos close y volumne
for i in range(len(dfs)):
    dfs[i] = dfs[i].resample('5T', closed='left', label='left').apply(ohlc_dict).get(["Close", "Volume"])

ov_df = pd.concat(dfs, axis=1, keys=names)
# borramos las filas que tengan nan(parece que algunos pueden estar desalineados)
ov_df.dropna(inplace=True)
ov_df.columns.set_names(["symbol", "value"], inplace=True)
del ohlcv_series_list, names, dfs
ov_df.head()


# In[ ]:


figure, windows = create_windows(ohlc=ov_df, n=10, window_len=0.6, right_set_len=0.3*0.9)
(in_windows, in_windows_index), (out_windows, _) = windows
del _
print("Done creating windows")


# In[ ]:


if is_notebook():
    figure.show()
del figure, ov_df


# In[ ]:


portfolio_kwargs = dict(
    direction='longonly',
    freq='m',
)
# creamos el indicador para el lr y otro para el wlr
# lo hago por separado para poder calcular el mlr
# con data de varios activos #del mercado
# y luego solo utiliza lr con los que me interesa
@njit
def lr_nb(close):
    c_log = np.log(close)
    return diff_nb(c_log)

LR = vbt.IndicatorFactory(
    input_names=['close'],
    output_names=['lr']
).from_apply_func(lr_nb, use_ray=True)

@njit
def wlr_nb(volume, lr):
    mkt_vol = volume.sum(axis=1)
    mkt_ratio = (volume.T / mkt_vol).T
    return lr * mkt_ratio

WLR = vbt.IndicatorFactory(
    input_names=['volume', 'lr'],
    output_names=['wlr']
).from_apply_func(wlr_nb, use_ray=True)

 #creamos el indicador para las bandas
@njit
def mkt_band_nb(mkt_lr, upper_filter, lower_filter):
    filtered = np.where(mkt_lr >= upper_filter, mkt_lr, np.nan)
    filtered = np.where(mkt_lr <= -lower_filter, mkt_lr, filtered)
    return filtered

MKT_BANDS = vbt.IndicatorFactory(
    input_names=['mkt_lr'],
    param_names=['upper_filter', 'lower_filter'],
    output_names=['filtered']
).from_apply_func(mkt_band_nb, use_ray=True)


# In[ ]:


# lr = log_return
# wlr = weighted log_return = lr * (Vi / Vmercado)
# mkt_lr = sum(wlr)
in_close = in_windows.xs('Close', level='value', axis=1)
in_volume = in_windows.xs('Volume', level='value', axis=1)
lr_ind = LR.run(in_close)
wlr_ind = WLR.run(in_volume, lr_ind.lr)
mkt_lr = wlr_ind.wlr.sum(axis=1, level="split_idx", skipna=False)
del in_volume # esto no se usa más
print("Done calculating mkt_lr")
lr_ind.lr.head()


# In[ ]:


mkt_lr.head()


# In[ ]:


# Acá filtramos los thresholds del mkt_lr a partir del cual compramos o vendemos.
upper_fltr = np.linspace(0.00001, 0.003, 60, endpoint=False)
lower_fltr = np.linspace(0.00001, 0.005, 60, endpoint=False)
mkt_bands_ind = MKT_BANDS.run(mkt_lr=mkt_lr, upper_filter=upper_fltr , lower_filter=lower_fltr,
                        per_column=False,
                        param_product=True,
                        short_name="mkt")
del upper_fltr, lower_fltr, mkt_lr
print("Done calculating mkt_bands")


# In[ ]:


if is_notebook():
    # Grafico un resultadoo arbitrario selecionando filtros arbitrarios para ver como ejemplo el funcionamiento de la estategia
    split_index = 5
    _mkt_lr_arb = mkt_lr[split_index]  # agarro el mkt_lr de algúna ventana
    lr_ada = lr_ind.lr[(split_index, "ADA")] # agarro el lr de ADA en esa ventana
    # borramos el mkt cuando está entre 0.0005 y - 0.0005
    filtered = np.where(_mkt_lr_arb >= 0.0005, _mkt_lr_arb, np.nan)
    filtered = np.where(_mkt_lr_arb <= -0.0005, _mkt_lr_arb, filtered)
    fig = pd.DataFrame({
            "lr_ada" : lr_ada,
            "mkt_lr": _mkt_lr_arb,
            "mkt_lr_filtered" : filtered
    }).vbt.plot()
    pd.DataFrame({
            "entries": np.where(filtered >= lr_ada, _mkt_lr_arb, np.nan), # compramos cuando el mercado está por encima de ada
            "exits": np.where(filtered <= lr_ada, _mkt_lr_arb, np.nan)
    }).vbt.scatterplot(fig=fig).show()
    del _mkt_lr_arb, lr_ada, filtered, fig


# In[ ]:


# Ya generamos todos los datos necesarios, ahora vamos a correr todas las simulaciones para cada assets que nos
# interesa testear
# para que no muera por memoria a la mitad y perder todo lo porcesado hasta el momento, me aseguro de que todas
#  las keys existan en el df
test_asset_list = ["ADA", "BTC"]
assert( set(test_asset_list).issubset(in_close.columns.get_level_values(level="symbol").unique()))


# In[ ]:


# Recolectamos el close y el lr de cada uno para poder borrar de memoria el df grande de todos los close y los lrs que no usamos
# puesto que close y lr son varias veces más grandes que el lr y close individual
lrs = {}
close = {}
for asset in test_asset_list:
    lrs[asset] = lr_ind.lr.xs(asset, level='symbol', axis=1)
    close[asset] = in_close.xs(asset, level='symbol', axis=1)
    print(f"Done separating close and lrs for {asset}")
del in_close, lr_ind
gc.collect()


# In[ ]:


# corremos la simulación para cada asset
portfolios = {}
for asset in test_asset_list:
    lr = lrs[asset]
    entries =  mkt_bands_ind.filtered_above(lr, crossover=True)
    exits = mkt_bands_ind.filtered_below(lr, crossover=True)
    portfolios[asset] = ExtendedPortfolio.from_signals(close[asset], entries, exits, **portfolio_kwargs, max_logs=0)
    del  entries, exits, lrs[asset] # el close no se borra porque lo vamos a volver a usar
    gc.collect()
    print(f"Done optimizing {asset}")
params_names = mkt_bands_ind.level_names
del mkt_bands_ind


# In[ ]:


# buscamos la mejor combinación de filtros y ploteamos la performace de todas las combinanciones
def dropnan(s):
    return s[~np.isnan(s)]
in_best_fltr_pairs = {}
for asset in test_asset_list:
    port = portfolios[asset]
    in_best_fltr_pairs[asset] = get_best_pairs(port.expected_log_returns(), *params_names)
    elr_volume = dropnan(port.expected_log_returns()).vbt.volume(title=f"{asset}'s Expected Log Return")
    sharpe_volume = dropnan(port.sharpe_ratio()).vbt.volume(title=f"{asset}'s Sharpe Ratio")
    del portfolios[asset]
    gc.collect()
    # Cuando corremos simulaciones largas, las corremos con python normal y
    # en eso casos queremos guardar los gráficos en un archivo
    if is_notebook():
        elr_volume.show()
        sharpe_volume.show()
    else:
        elr_volume.write_html(f"{current_dir}/{asset}_optimization_exp_log_ret.html")
        sharpe_volume.write_html(f"{current_dir}/{asset}_optimization_sharpe-ratio.html")
    del elr_volume, sharpe_volume
    print(f"Done plotting {asset}")
gc.collect()


# In[ ]:


# acá arranca la parte de correr las simulaciones con los datos del out y los parámetros ya optimizados
out_close = out_windows.xs('Close', level='value', axis=1)
out_volume = out_windows.xs('Volume', level='value', axis=1)
lr_ind = LR.run(out_close)
wlr_ind = WLR.run(out_volume, lr_ind.lr)
mkt_lr = wlr_ind.wlr.sum(axis=1, level="split_idx", skipna=False)
del out_windows, wlr_in
gc.collect()


# In[ ]:


# para cada activo de los que me interesa tradear simulo el resultado de ser corrido con los parámetros optimizados
for asset in test_asset_list:
    in_best_pairs = np.array(in_best_fltr_pairs[asset])
    upper_fltr = in_best_pairs[:,0]
    lower_fltr = in_best_pairs[:,1]
    mkt_bands_ind = MKT_BANDS.run(mkt_lr=mkt_lr, upper_filter=upper_fltr , lower_filter=lower_fltr,
                        per_column=True,
                        param_product=False,
                        short_name="mkt")
    lr = lr_ind.lr.xs(asset, level='symbol', axis=1)
    close = out_close.xs(asset, level='symbol', axis=1)
    entries =  mkt_bands_ind.filtered_above(lr, crossover=True)
    exits = mkt_bands_ind.filtered_below(lr, crossover=True)
    port = ExtendedPortfolio.from_signals(close, entries, exits, **portfolio_kwargs, max_logs=0)
    exp_plot = port.expected_log_returns().vbt.plot(title=f"{asset}'s Expected Log Return")
    sharpe_plot = port.sharpe_ratio().vbt.plot(title=f"{asset}'s Sharpe ratio")
    if is_notebook():
        exp_plot().show()
        sharpe_plot.show()
    else:
        exp_plot.write_html(f"{current_dir}/{asset}_simulation_exp_log_ret.html")
        sharpe_plot.write_html(f"{current_dir}/{asset}_simulation_sharpe-ratio.html")
    print(f"Done simulating {asset}")


# In[ ]:


# un pequeño test para asegurarnos que todas las cuentas den
_py = pd.DataFrame({
    'Close': [1,e,e**2],
    'Volume': [1,2,1]
})
_thon = pd.DataFrame({
    'Close': [e**2,e,1],
    'Volume': [1,4,10]
})
_test_df = pd.concat([_py,_thon], axis=1, keys=["Py", "Thon"])
_test_df.columns.set_names(["asset", "value"], inplace=True)

close = _test_df.xs('Close', level='value', axis=1)
volume = _test_df.xs('Volume', level='value', axis=1)
_test_lrInd = LR.run(close)
_test_wlrInd = WLR.run(volume, _test_lrInd.lr)

exp_py_lr = np.array([np.nan, 1, 1])
exp_thon_lr = np.array([np.nan, -1, -1])
assert (np.allclose(exp_py_lr, _test_lrInd.lr["Py"], equal_nan=True))
assert (np.allclose(exp_thon_lr, _test_lrInd.lr["Thon"], equal_nan=True))
exp_py_vr = np.array([0.5, 1/3, 1/11])
exp_thon_vr = np.array([0.5, 2/3, 10/11])
exp_py_wlr = exp_py_lr * exp_py_vr
exp_thon_wlr = exp_thon_lr * exp_thon_vr
assert (np.allclose(exp_py_wlr, _test_wlrInd.wlr["Py"], equal_nan=True))
assert (np.allclose(exp_thon_wlr, _test_wlrInd.wlr["Thon"], equal_nan=True))
# falta testear el cálculo de mkt_lr
_test_mkt_lr = _test_wlrInd.wlr.sum(axis=1, skipna=False)
exp_mkt_lr = exp_py_wlr + exp_thon_wlr
assert (np.allclose(exp_mkt_lr,_test_mkt_lr, equal_nan=True))


# In[ ]:




