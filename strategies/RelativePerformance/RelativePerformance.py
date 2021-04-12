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
from lib.utils import directory_to_data_frame_list, ExtendedPortfolio


# In[ ]:


current_dir = os.path.dirname(os.path.realpath(__file__))
directory = "/home/ec2-user/data/long/"
ohlcv_series_list = directory_to_data_frame_list(directory)
# concatenamos los dfs
names = list(map(lambda t: t[0], ohlcv_series_list))
dfs = list(map(lambda t: t[1].get(["Close", "Volume"]), ohlcv_series_list))
ov_df = pd.concat(dfs, axis=1, keys=names)
# borramos las filas que tengan nan(parece que algunos pueden estar desalineados)
ov_df.dropna(inplace=True)
ov_df.columns.set_names(["symbol", "value"], inplace=True)
close = ov_df.xs('Close', level='value', axis=1)
volume = ov_df.xs('Volume', level='value', axis=1)
del ohlcv_series_list, names, ov_df
gc.collect()
close.head()


# In[ ]:


# creamos el indicador para el lr y otro para el wlr
# lo hago por separado para poder calcular el mlr
# con data de varios activos del mercado
# y luego solo utiliza lr con los que me interesa
@njit
def lr_nb(close):
    c_log = np.log(close)
    lr = diff_nb(c_log)
    return lr

LR = vbt.IndicatorFactory(
    input_names=['close'],
    output_names=['lr']
).from_apply_func(lr_nb, use_ray=True)

@njit
def wlr_nb(volume, lr):
    mkt_vol = volume.sum(axis=1)
    mkt_ratio = (volume.T / mkt_vol).T
    wrl =  lr * mkt_ratio
    return wrl

WLR = vbt.IndicatorFactory(
    input_names=['volume', 'lr'],
    output_names=['wlr']
).from_apply_func(wlr_nb, use_ray=True)
print("hasta acá todo joya1")


# In[ ]:


lr_ind = LR.run(close)
wlr_ind = WLR.run(volume, lr_ind.lr)
mkt_lr = wlr_ind.wlr.sum(axis=1, skipna=False)


# In[ ]:


# plotear todos los assests hace un gráfico muy feo, entonces solo muestro algunos
#fig = mkt_lr.vbt.plot(trace_names=["MKT_LR"])
#lr_ind.lr[["ADA", "BTC"]].vbt.plot(fig=fig).show()


# In[ ]:


#creamos el indicador para las bandas
@njit
def mkt_band_nb(mkt_lr, upper_filter, lower_filter):
   upper = np.where(mkt_lr >= upper_filter, mkt_lr, np.nan)
   lower = np.where(mkt_lr <= -lower_filter, mkt_lr, np.nan)
   return upper, lower

MKT_BANDS = vbt.IndicatorFactory(
   input_names=['mkt_lr'],
   param_names=['upper_filter', 'lower_filter'],
   output_names=['upper', 'lower']
).from_apply_func(mkt_band_nb, use_ray=True)


# In[ ]:


filters = np.linspace(0.00001, 0.005, 50, endpoint=False)
mkt_bands_ind = MKT_BANDS.run(mkt_lr=mkt_lr, upper_filter=filters , lower_filter=filters,
                        per_column=False,
                        param_product=True,
                        short_name="mkt")
print("hasta acá todo joya2")
del filters


# In[ ]:


portfolio_kwargs = dict(
    direction='longonly',
    freq='m',
)
ada_lr = lr_ind.lr["ADA"]
ada_close = close["ADA"]
btc_lr = lr_ind.lr["BTC"]
btc_close = close["BTC"]
del close
gc.collect()


# In[ ]:


print("hasta acá todo joya3")
entries =  mkt_bands_ind.upper_above(ada_lr, crossover=True)
exits = mkt_bands_ind.lower_below(ada_lr, crossover=True)
ada_port = ExtendedPortfolio.from_signals(ada_close, entries, exits, **portfolio_kwargs,  max_logs=0)
del entries, exits, ada_close
gc.collect()


# In[1]:


print("hasta acá todo joya4")
ada_port.expected_log_returns().vbt.heatmap(title="ADA's Expected Log Return").write_html(f"{current_dir}/ADA_exp_log_ret.html")
ada_port.sharpe_ratio().vbt.heatmap(title="ADA's Sharpe Ratio").write_html(f"{current_dir}/ADA_sharpe-ratio.html")
del ada_port
gc.collect()


# In[ ]:


print("hasta acá todo joya5")
entries =  mkt_bands_ind.upper_above(btc_lr, crossover=True)
exits = mkt_bands_ind.lower_below(btc_lr, crossover=True)
btc_port = ExtendedPortfolio.from_signals(btc_close, entries, exits, **portfolio_kwargs,  max_logs=0)
del entries, exits, btc_close
gc.collect()


# In[ ]:


print("hasta acá todo joya6")
btc_port.expected_log_returns().vbt.heatmap(title="BTC's Expected Log Return").write_html(f"{current_dir}/BTC_exp_log_ret.html")
btc_port.sharpe_ratio().vbt.heatmap(title="BTC's Sharpe Ratio").write_html(f"{current_dir}/BTC_sharpe-ratio.html")
del btc_port
gc.collect()


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




