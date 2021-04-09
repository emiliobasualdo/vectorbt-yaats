#%%

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
ohlcv.get(["Open", "High", "Low", "Close", "Volume"]).vbt.ohlcv.plot().show_png()
ohlc = ohlcv.get(["Open", "High", "Low", "Close"])
print(ohlc.head())

#%%

# creamos las ventanas
figure, windows = create_windows(ohlc=ohlc, n=5, window_len=0.6, training_set_len=0.4)
(in_df, in_indexes), (out_df, out_indexes) = windows
figure.show()

#%%

portfolio_kwargs = dict(
    direction='longonly',
    freq='m'
)


#%%

# creamos el indicador
@njit
def apply_func_nb(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, buy_threshold: float,
                  sell_threshold: float):
    aux = (close - open) / (high - low + 0.001)
    aux = np.where(aux >= buy_threshold, 1, aux)
    aux = np.where(aux <= -sell_threshold, -1, aux)
    return aux

AlphaInd = vbt.IndicatorFactory(
    input_names=['open', 'high', 'low', 'close'],
    param_names=['buy_threshold', 'sell_threshold'],
    output_names=['signal']
).from_apply_func(apply_func_nb)
# dir(AlphaInd)

#%%

# Corremos con todas las combinaciones
def simulate_all_params(ohlc_windows, params_range):
    # creamos las señales
    open = ohlc_windows.xs("Open", level=1, axis=1)
    high = ohlc_windows.xs("High", level=1, axis=1)
    low = ohlc_windows.xs("Low", level=1, axis=1)
    close = ohlc_windows.xs("Close", level=1, axis=1)
    momentum = AlphaInd.run(open=open, high=high, low=low, close=close,
                            buy_threshold=params_range, sell_threshold=params_range,
                            param_product=True,
                            short_name="alpha")
    ones = np.full(momentum.signal.shape, 1)
    entry_signal = momentum.signal_equal(ones)
    exit_signal = momentum.signal_equal(-ones)
    return ExtendedPortfolio.from_signals(close, entry_signal, exit_signal, **portfolio_kwargs)

#%%

# Optimizamos par el in y el out
params_range = np.linspace(0.1, 1, 9, endpoint=False)
in_elr = simulate_all_params(in_df, params_range).expected_log_returns()
out_elr = simulate_all_params(out_df, params_range).expected_log_returns()

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
print(in_best_index)

#%%

in_best_buy_thresholds = get_best_params(in_best_index, 'alpha_buy_threshold')
in_best_sell_thresholds = get_best_params(in_best_index, 'alpha_sell_threshold')
in_best_threshold_pairs = np.array(list(zip(in_best_buy_thresholds, -in_best_sell_thresholds)))

print(in_best_threshold_pairs)

#%%

pd.DataFrame(in_best_threshold_pairs, columns=['buy_threshold', 'sell_threshold']).vbt.plot().show()

#%%

in_df.head()

#%%

# Corremos el out con los mejores parámetros de in
#close, high, low, open = list(map(lambda tu: tu[1], in_df.groupby(level=1, axis=1)))
open = in_df.xs("Open", level=1, axis=1)
high = in_df.xs("High", level=1, axis=1)
low = in_df.xs("Low", level=1, axis=1)
close = in_df.xs("Close", level=1, axis=1)
momentum = AlphaInd.run(open=open, high=high, low=low, close=close,
                        buy_threshold=in_best_buy_thresholds, sell_threshold=in_best_sell_thresholds,
                        short_name="alpha", per_column=True)
ones = np.full(momentum.signal.shape, 1)
entry_signal = momentum.signal_equal(ones, crossover=True)
exit_signal = momentum.signal_equal(-ones, crossover=True)
# imprimo para confirmar que haya algún true
entry_signal.loc[entry_signal[in_best_buy_thresholds[0], in_best_sell_thresholds[0], 0]==True].head()

#%%

trade_price = close
out_test_elr = ExtendedPortfolio.from_signals(trade_price, entry_signal, exit_signal, **portfolio_kwargs).expected_log_returns()
print(out_test_elr)

#%%

# simulamos Buy&Hold de cada in y out window y tomamos el expected log returns (elr)
close_columns = list(filter(lambda col: "Close" in col[1], in_df.columns))
in_hold_port = ExtendedPortfolio.from_holding(in_df[close_columns], **portfolio_kwargs)
out_hold_port = ExtendedPortfolio.from_holding(out_df[close_columns], **portfolio_kwargs)
print(in_hold_port.trades.values[:4])
print(out_hold_port.trades.values[:4])
in_hold_elr = in_hold_port.expected_log_returns()
out_hold_elr = out_hold_port.expected_log_returns()
print(in_hold_elr, out_hold_elr)

#%%

cv_results_df = pd.DataFrame({
    'in_sample_hold': in_hold_elr.values,
    'in_sample_median': in_elr.groupby('split_idx').median().values,
    'in_sample_best': in_elr[in_best_index].values,
    'out_sample_hold': out_hold_elr.values,
    'out_sample_median': out_elr.groupby('split_idx').median().values,
    'out_sample_test': out_test_elr.values
})

cv_results_df.vbt.plot(
    trace_kwargs=[
        dict(line_color=vbt.settings.color_schema['blue']),
        dict(line_color=vbt.settings.color_schema['blue'], line_dash='dash'),
        dict(line_color=vbt.settings.color_schema['blue'], line_dash='dot'),
        dict(line_color=vbt.settings.color_schema['orange']),
        dict(line_color=vbt.settings.color_schema['orange'], line_dash='dash'),
        dict(line_color=vbt.settings.color_schema['orange'], line_dash='dot')
    ]
).show()

