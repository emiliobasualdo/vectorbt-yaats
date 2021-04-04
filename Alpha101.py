from math import floor

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.base.reshape_fns import to_2d, to_1d
from vectorbt.utils.config import merge_dicts
from numba import njit
from vectorbt.utils.decorators import cached_property, custom_method
from utils import file_to_data_frame

if __name__ == '__main__':

    (s_name, data) = file_to_data_frame(
        "/Users/pilo/development/itba/pf/Binance_Minute_OHLC_CSVs/Binance_BNBUSDT_minute.csv")
    price = data["Close"]


    split_kwargs = dict(
        n=5,
        window_len=floor(len(price) * 0.6),
        set_lens=(0.4, ),
        left_to_right=False
    )  # 30 windows, each 2 years long, reserve 180 days for test
    portfolio_kwargs = dict(
        direction='all',  # long and short
        freq='d'
    )
    windows = np.arange(10, 50)

    def roll_in_and_out_samples(price, **kwargs):
        return price.vbt.rolling_split(**kwargs)

    #roll_in_and_out_samples(price, **split_kwargs, plot=True, trace_names=['in-sample', 'out-sample']).show()

    (in_price, in_indexes), (out_price, out_indexes) = roll_in_and_out_samples(price, **split_kwargs)

    class ExtendedProtfolio(vbt.Portfolio):
        @custom_method
        def expected_log_returns(self):
            """Get log return mean series per column/group based on portfolio value."""
            log_nb = njit(lambda col, returns: np.log(returns + 1))
            mean_nb = njit(lambda col, l_rets: np.nanmean(l_rets))
            return self.returns().vbt.apply_and_reduce(log_nb, mean_nb)

    def simulate_holding(price, **kwargs):
        portfolio = ExtendedProtfolio.from_holding(price, **kwargs)
        return portfolio.expected_log_returns()

    in_hold_elr = simulate_holding(in_price, **portfolio_kwargs)


    def simulate_all_params(price, windows, **kwargs):
        fast_ma, slow_ma = vbt.MA.run_combs(price, windows, r=2, short_names=['fast', 'slow'])
        entries = fast_ma.ma_above(slow_ma, crossover=True)
        exits = fast_ma.ma_below(slow_ma, crossover=True)
        portfolio = ExtendedProtfolio.from_signals(price, entries, exits, **kwargs)
        return portfolio.expected_log_returns()

    # Simulate all params for in-sample ranges
    in_elr = simulate_all_params(in_price, windows, **portfolio_kwargs)

    print(in_elr)

    def get_best_index(performance, higher_better=True):
        if higher_better:
            return performance[performance.groupby('split_idx').idxmax()].index
        return performance[performance.groupby('split_idx').idxmin()].index

    in_best_index = get_best_index(in_elr)

    print(in_best_index)

    def get_best_params(best_index, level_name):
        return best_index.get_level_values(level_name).to_numpy()

    in_best_fast_windows = get_best_params(in_best_index, 'fast_window')
    in_best_slow_windows = get_best_params(in_best_index, 'slow_window')
    in_best_window_pairs = np.array(list(zip(in_best_fast_windows, in_best_slow_windows)))

    print(in_best_window_pairs)

    pd.DataFrame(in_best_window_pairs, columns=['fast_window', 'slow_window']).vbt.plot().show()

    out_hold_elr = simulate_holding(out_price, **portfolio_kwargs)

    print(out_hold_elr)

    # Simulate all params for out-sample ranges
    out_elr = simulate_all_params(out_price, windows, **portfolio_kwargs)

    print(out_elr)

    def simulate_best_params(price, best_fast_windows, best_slow_windows, **kwargs):
        fast_ma = vbt.MA.run(out_price, window=best_fast_windows, per_column=True)
        slow_ma = vbt.MA.run(out_price, window=best_slow_windows, per_column=True)
        entries = fast_ma.ma_above(slow_ma, crossover=True)
        exits = fast_ma.ma_below(slow_ma, crossover=True)
        portfolio = ExtendedProtfolio.from_signals(price, entries, exits, **kwargs)
        return portfolio.expected_log_returns()

    # Use best params from in-sample ranges and simulate them for out-sample ranges
    out_test_elr = simulate_best_params(out_price, in_best_fast_windows, in_best_slow_windows, **portfolio_kwargs)

    print(out_test_elr)

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