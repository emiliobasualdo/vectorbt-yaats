#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

from vectorbt import FigureWidget

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from numba import njit
import numpy as np
from lib.utils import create_windows, file_to_data_frame, ExtendedPortfolio
import pandas as pd
import vectorbt as vbt
from vectorbt.utils.config import merge_dicts


# In[ ]:


# leemos el csv
(s_name, ohlcv) = file_to_data_frame(
    "../../Binance_ADAUSDT_minute.csv")
# agarramos solo las columnas que necesitamos
cols = ohlcv.columns
print(cols)
#ohlcv.get(["Open", "High", "Low", "Close", "Volume"]).vbt.ohlcv.plot().show_png()

ohlc = ohlcv.get(["Open", "High", "Low", "Close"])
print(ohlc.head())


# In[ ]:


# creamos las ventanas
figure, windows = create_windows(ohlc=ohlc, n=25, window_len=0.7, right_set_len=0.4)
(in_df, in_df_index), (out_df, _) = windows


# In[ ]:


figure.show()


# In[ ]:


in_df.head()


# In[ ]:


portfolio_kwargs = dict(
    direction='longonly',
    freq='m'
)


# In[ ]:


# creamos el indicador
@njit
def apply_alpha_nb(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, buy_threshold: float,
                  sell_threshold: float):
    aux = (close - open) / (high - low + 0.001)
    aux = np.where(aux >= buy_threshold, 1, aux)
    aux = np.where(aux <= -sell_threshold, -1, aux)
    return aux

Alpha = vbt.IndicatorFactory(
    input_names=['open', 'high', 'low', 'close'],
    param_names=['buy_threshold', 'sell_threshold'],
    output_names=['signal']
).from_apply_func(apply_alpha_nb)
# dir(Alpha)

class _Alpha(Alpha):

    def plot(self,
             column=None,
             signal_trace_kwargs=None,
             add_trace_kwargs=None,
             xref='x', yref='y',
             fig=None,
             **layout_kwargs):  # pragma: no cover

        self_col = self.select_series(column=column)

        if fig is None:
            fig = FigureWidget()
        default_layout = dict()
        default_layout['yaxis' + yref[1:]] = dict(range=[-1.05, 1.05])
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        if signal_trace_kwargs is None:
            signal_trace_kwargs = {}
        signal_trace_kwargs = merge_dicts(dict(
            name='Aplha'
        ), signal_trace_kwargs)

        fig = self_col.signal.vbt.plot(
            trace_kwargs=signal_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        # Fill void between levels
        fig.add_shape(
            type="rect",
            xref=xref,
            yref=yref,
            x0=self_col.signal.index[0],
            y0=self.buy_threshold_array[0],
            x1=self_col.signal.index[-1],
            y1=self.buy_threshold_array[0],
            fillcolor="purple",
            opacity=0.2,
            layer="below",
            line_width=0,
        )

        return fig


# In[ ]:


#def simulate(ohlc_windows, buy_threshold, sell_threshold, param_product):
# creamos las señales
def simulate(ohlc_windows, buy_threshold, sell_threshold, param_product):
    open = ohlc_windows.xs("Open", level=1, axis=1)
    high = ohlc_windows.xs("High", level=1, axis=1)
    low = ohlc_windows.xs("Low", level=1, axis=1)
    close = ohlc_windows.xs("Close", level=1, axis=1)
    momentum = _Alpha.run(open=open, high=high, low=low, close=close,
                            buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                            param_product=param_product,
                            per_column=not param_product,
                            short_name="alpha")
    ones = np.full(1, 1)
    entry_signal = momentum.signal_equal(ones, crossover=True)
    exit_signal = momentum.signal_equal(-ones, crossover=True)
    return ExtendedPortfolio.from_signals(close, entry_signal, exit_signal, **portfolio_kwargs)


# In[ ]:


# Optimizamos para el in
buy_range = np.linspace(0.1, 1, 10, endpoint=False)
sell_range = np.linspace(0.1, 0.5, 10, endpoint=False)
in_port = simulate(in_df, buy_range, sell_range, param_product=True)
in_sharpe = in_port.sharpe_ratio()
in_elr = in_port.expected_log_returns()
#out_elr = simulate_all_params(out_df, params_range).expected_log_returns()


# In[ ]:


def get_best_index(performance, higher_better=True):
    if higher_better:
        return performance[performance.groupby('split_idx').idxmax()].index
    return performance[performance.groupby('split_idx').idxmin()].index

def get_best_params(best_index, level_name):
    return best_index.get_level_values(level_name).to_numpy()


# In[ ]:


#Buscamos el índice de los mejores resultados del in
in_best_index = get_best_index(in_elr)
in_best_buy_thresholds = get_best_params(in_best_index, 'alpha_buy_threshold')
in_best_sell_thresholds = get_best_params(in_best_index, 'alpha_sell_threshold')
in_best_threshold_pairs = np.array(list(zip(in_best_buy_thresholds, -in_best_sell_thresholds)))


# In[ ]:


pd.DataFrame(in_best_threshold_pairs, columns=['buy_threshold', 'sell_threshold']).vbt.plot().show()


# In[ ]:


out_test_port = simulate(out_df, in_best_buy_thresholds, in_best_sell_thresholds, param_product=False)
out_test_elr = out_test_port.expected_log_returns()
out_test_sharpe = out_test_port.sharpe_ratio()


# In[ ]:


# Nos aseguramos que no todos los erls sean iguales
assert (not np.all(in_elr[in_best_index].values == out_test_elr.values))
# chequeamos qeu el in se haya corrido sobre las mismas ventanas qeu el mejor del out
pnl1 = in_port.trades.pnl.to_matrix()[in_best_index]
pnl1 = pnl1[np.isfinite(pnl1).all(axis=1)]
pnl2 = out_test_port.trades.pnl.to_matrix()
pnl2 = pnl2[np.isfinite(pnl2).all(axis=1)]
assert(np.array_equal(pnl1.columns,pnl2.columns))


# In[ ]:


# simulamos Buy&Hold de cada in y out window y tomamos el expected log returns (elr)
in_price = in_df.xs("Close", level=1, axis=1)
out_price = out_df.xs("Close", level=1, axis=1)
in_hold_port = ExtendedPortfolio.from_holding(in_price, **portfolio_kwargs)
out_hold_port = ExtendedPortfolio.from_holding(out_price, **portfolio_kwargs)
assert (in_hold_port.trades.values[0]) # por lo menos 1 trade
assert (out_hold_port.trades.values[0])

in_hold_elr = in_hold_port.expected_log_returns()
in_hold_sharpe = in_hold_port.sharpe_ratio()
out_hold_elr = out_hold_port.expected_log_returns()
out_hold_sharpe = out_hold_port.sharpe_ratio()


# In[ ]:


# ploteamos los elrs
cv_results_df = pd.DataFrame({
    'in_sample_hold': in_hold_elr.values,
    'in_sample_best': in_elr[in_best_index].values,
    'out_sample_hold': out_hold_elr.values,
    'out_sample_test': out_test_elr.values
})

cv_results_df.vbt.plot(
    trace_kwargs=[
        dict(line_color=vbt.settings.color_schema['blue']),
        dict(line_color=vbt.settings.color_schema['blue'], line_dash='dot'),
        dict(line_color=vbt.settings.color_schema['orange']),
        dict(line_color=vbt.settings.color_schema['orange'], line_dash='dot')
    ]
).show()


# In[ ]:


#ploteamos los sharps
cv_results_df = pd.DataFrame({
    'in_sample_hold': in_hold_sharpe.values,
    'in_sample_best': in_sharpe[in_best_index].values,
    'out_sample_hold': out_hold_sharpe.values,
    'out_sample_test': out_test_sharpe.values
})

cv_results_df.vbt.plot(
    trace_kwargs=[
        dict(line_color=vbt.settings.color_schema['blue']),
        dict(line_color=vbt.settings.color_schema['blue'], line_dash='dot'),
        dict(line_color=vbt.settings.color_schema['orange']),
        dict(line_color=vbt.settings.color_schema['orange'], line_dash='dot')
    ]
).show()


# In[ ]:


for col in in_best_index:
    out_test_port.trades.plot(column=col).show()
out_test_port.trades.plot_pnl(column=col).show()

