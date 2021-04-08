#%%
from numba import njit
import numpy as np
from lib.utils import create_windows, file_to_data_frame, ExtendedProtfolio
import pandas as pd
import vectorbt as vbt

#%%
# leemos el csv
(s_name, ohlcv) = file_to_data_frame(
    "/Users/pilo/development/itba/pf/Binance_Minute_OHLC_CSVs/shorts/Binance_ADAUSDT_minute_3000.csv")
# agarramos solo las columnas que necesitamos
cols = ohlcv.columns
print(cols)
# ohlcv.get(["Open", "High", "Low", "Close", "Volume"]).vbt.ohlcv.plot().show_png()
ohlc = ohlcv.get(["Open", "High", "Low", "Close"])
print(ohlc.head())

#%%
# creamos las ventanas
figure, windows = create_windows(ohlc=ohlc, n=5, window_len=0.6, training_set_len=0.4)
(training_df, train_indexes), (test_df, test_indexes) = windows
# figure.show_png()

#%%
portfolio_kwargs = dict(
    direction='all',  # long and short todo falta ver esto
    freq='d'  # todo falta ver esto
)
#%%
# simulamos Buy&Hold de cada training y test window y tomamo el expected log returns (elr)
close_columns = list(filter(lambda col: "Close" in col[1], training_df.columns))
in_hold_elr = ExtendedProtfolio.from_holding(training_df[close_columns], **portfolio_kwargs).expected_log_returns()
out_hold_elr = ExtendedProtfolio.from_holding(test_df[close_columns], **portfolio_kwargs).expected_log_returns()
print(in_hold_elr, out_hold_elr)


#%%
# Creamos el alpha
def alpha_from_df(ohlc_windows):
    alpha = pd.DataFrame()
    for w, new_df in ohlc_windows.groupby(level=0, axis=1):
        alpha[w] = (new_df[w, "Close"] - new_df[w, "Open"]) / (new_df[w, "High"] - new_df[w, "Low"] + 0.001)
    return alpha


#%%
# creamos el indicador
@njit
def apply_func_nb_org(alpha: np.ndarray):
    return alpha


@njit
def apply_func_nb(open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, buy_threshold: float,
                  sell_threshold: float):
    aux = (close - open) / (high - low + 0.001)
    aux = np.where(aux >= buy_threshold, 1, aux)
    aux = np.where(aux <= sell_threshold, -1, aux)
    return aux


AlphaInd = vbt.IndicatorFactory(
    input_names=['open', 'high', 'low', 'close'],
    param_names=['buy_threshold', 'sell_threshold'],
    output_names=['signal']
).from_apply_func(apply_func_nb)

# dir(AlphaInd)

def simulate_all_params(ohlc_windows, params_range):
    # creamos las señales
    open, high, low, close = list(map(lambda tu: tu[1].to_numpy(), ohlc_windows.groupby(level=1, axis=1)))
    momentum = AlphaInd.run(open=open, high=high, low=low, close=close,
                            buy_threshold=params_range, sell_threshold=params_range,
                            param_product=True,
                            short_name="Momentum")
    ones = np.full(momentum.signal.shape, 1)
    entry_signal = momentum.signal_equal(ones, crossover=True, multiple=True)
    exit_signal = momentum.signal_equal(-ones, crossover=True, multiple=True)

    trade_price = ohlc_windows.xs("Close", level=1, axis=1)
    return ExtendedProtfolio.from_signals(trade_price.to_numpy(), entry_signal, exit_signal, **portfolio_kwargs)


# Simulamos
params_range = np.linspace(0.1, 0.8, 8)
portfolio = simulate_all_params(training_df, params_range)
portfolio.expected_log_returns()
