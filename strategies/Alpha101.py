#%%

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from numba import njit
import numpy as np
from lib.utils import create_windows, file_to_data_frame, ExtendedPortfolio
import pandas as pd
import vectorbt as vbt

#%%

# leemos el csv
(s_name, ohlcv) = file_to_data_frame(
    "/Users/pilo/development/itba/pf/Binance_Minute_OHLC_CSVs/shorts/Binance_ADAUSDT_minute_3000.csv")
# agarramos solo las columnas que necesitamos
cols = ohlcv.columns
print(cols)
ohlc = ohlcv.get(["Open", "High", "Low", "Close"])
print(ohlc.head())

#%%

# creamos las ventanas
figure, windows = create_windows(ohlc=ohlc, n=2, window_len=0.5, right_set_len=0.5)
(in_df, in_df_index), (out_df, _) = windows

#%%


#%%

in_df.head()

#%%

portfolio_kwargs = dict(
    direction='longonly',
    freq='m'
)

#%%

# creamos el indicador
@njit
def apply_alpha_nb(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, buy_threshold: float,
                  sell_threshold: float):
    aux = (close - open) / (high - low + 0.001)
    aux = np.where(aux >= buy_threshold, 1, aux)
    aux = np.where(aux <= -sell_threshold, -1, aux)
    return aux

AlphaInd = vbt.IndicatorFactory(
    input_names=['open', 'high', 'low', 'close'],
    param_names=['buy_threshold', 'sell_threshold'],
    output_names=['signal']
).from_apply_func(apply_alpha_nb)
# dir(AlphaInd)

#%%

def simulate(ohlc_windows, buy_threshold, sell_threshold, param_product):
    # creamos las señales
    open = ohlc_windows.xs("Open", level=1, axis=1)
    high = ohlc_windows.xs("High", level=1, axis=1)
    low = ohlc_windows.xs("Low", level=1, axis=1)
    close = ohlc_windows.xs("Close", level=1, axis=1)
    momentum = AlphaInd.run(open=open, high=high, low=low, close=close,
                            buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                            param_product=param_product,
                            per_column=not param_product,
                            short_name="alpha")
    ones = np.full(momentum.signal.shape[-1], 1)
    entry_signal = momentum.signal_equal(ones, crossover=True)
    exit_signal = momentum.signal_equal(-ones, crossover=True)
    return ExtendedPortfolio.from_signals(close, entry_signal, exit_signal, **portfolio_kwargs)

#%%

# Optimizamos para el in
buy_range = np.linspace(0.1, 0.9, 10, endpoint=False)
sell_range = np.linspace(0.7, 1, 10, endpoint=False)
in_port = simulate(in_df, buy_range, sell_range, param_product=True)
in_sharpe = in_port.sharpe_ratio()
in_elr = in_port.expected_log_returns()
#out_elr = simulate_all_params(out_df, params_range).expected_log_returns()

#%%

def get_best_index(performance, higher_better=True):
    if higher_better:
        return performance[performance.groupby('split_idx').idxmax()].index
    return performance[performance.groupby('split_idx').idxmin()].index

def get_best_params(best_index, level_name):
    return best_index.get_level_values(level_name).to_numpy()

#%%

#Buscamos el índice de los mejores resultados del in
in_best_index = get_best_index(in_elr)
in_best_buy_thresholds = get_best_params(in_best_index, 'alpha_buy_threshold')
in_best_sell_thresholds = get_best_params(in_best_index, 'alpha_sell_threshold')
in_best_threshold_pairs = np.array(list(zip(in_best_buy_thresholds, -in_best_sell_thresholds)))

#%%


#%%

out_test_port = simulate(out_df, in_best_buy_thresholds, in_best_sell_thresholds, param_product=False)
out_test_elr = out_test_port.expected_log_returns()
out_test_sharpe = out_test_port.sharpe_ratio()

#%%

pnl1 = in_port.trades.pnl.to_matrix()[in_best_index]
pnl1 = pnl1[np.isfinite(pnl1).all(axis=1)]
pnl2 = out_test_port.trades.pnl.to_matrix()
pnl2 = pnl2[np.isfinite(pnl2).all(axis=1)]
assert(np.array_equal(pnl1.columns,pnl2.columns))

#%%

# simulamos Buy&Hold de cada in y out window y tomamos el expected log returns (elr)
close_columns = list(filter(lambda col: "Close" in col[1], in_df.columns))
in_hold_port = ExtendedPortfolio.from_holding(in_df[close_columns], **portfolio_kwargs)
out_hold_port = ExtendedPortfolio.from_holding(out_df[close_columns], **portfolio_kwargs)
assert (in_hold_port.trades.values[0]) # por lo menos 1 trade
assert (out_hold_port.trades.values[0])

in_hold_elr = in_hold_port.expected_log_returns()
in_hold_sharpe = in_hold_port.sharpe_ratio()
out_hold_elr = out_hold_port.expected_log_returns()
out_hold_sharpe = out_hold_port.sharpe_ratio()

#%%

print(f"¿Todos los elrs son iguales?:{in_elr[in_best_index].values == out_test_elr.values}")
#%%
# ploteamos los elrs
cv_results_df = pd.DataFrame({
    'in_sample_hold': in_hold_elr.values,
    'in_sample_best': in_elr[in_best_index].values,
    'out_sample_hold': out_hold_elr.values,
    'out_sample_test': out_test_elr.values
})


#%%

#ploteamos los sharps
cv_results_df = pd.DataFrame({
    'in_sample_hold': in_hold_sharpe.values,
    'in_sample_best': in_sharpe[in_best_index].values,
    'out_sample_hold': out_hold_sharpe.values,
    'out_sample_test': out_test_sharpe.values
})
