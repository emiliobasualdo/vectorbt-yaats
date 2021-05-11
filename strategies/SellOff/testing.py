import math
import numpy as np
import vectorbt as vbt
from lib.utils import LR, ExtendedPortfolio, shift_np
from strategies.SellOff.SellOff import signals_nb, signal_calculations, ma_mstd

####### Debugging #######
if __name__ == '__main__':
    start = '2019-01-01 UTC'  # crypto is in UTC
    end = '2020-01-01 UTC'
    btc_price = vbt.YFData.download('BTC-USD', start=start, end=end).get('Close')
    fast_ma = vbt.MA.run(btc_price, [10, 20], short_name='fast')
    slow_ma = vbt.MA.run(btc_price, [20, 30], short_name='slow')
    dummy_ma = vbt.MA.run(btc_price, [50, 60], short_name='dummy')
    entries = fast_ma.ma_above(slow_ma, crossover=True)
    exits_1 = fast_ma.ma_below(slow_ma, crossover=True)
    exits_2 = fast_ma.ma_below(dummy_ma, crossover=True)
    exits = exits_1 | exits_2

    port = ExtendedPortfolio.from_signals(btc_price, entries, exits)
    port.trades.median_log_returns().vbt.heatmap(slider_level='dummy_window').show()


####### TESTING #######
def test():
    lag = 2
    p = math.e ** np.array([[5], [6], [6], [3], [2], [4], [7], [8], [6], [5], [4]])
    v = np.array([[5.], [6.], [6.], [3.], [2.], [4.], [7.], [8.], [6.], [5.], [4.]])
    shifted_vol = shift_np(v, 1)
    e_shifted_vol_ma = np.array([[np.nan], [np.nan], [5.5], [6.], [4.5], [2.5], [3.], [5.5], [7.5], [7.], [5.5]])
    e_shifted_vol_mstd = np.array([[np.nan], [np.nan], [np.nan], [0.5], [0.], [1.5], [0.5], [1.], [1.5], [0.5], [1.]])
    e_lr = np.array([[np.nan], [1], [0], [-3], [-1], [2], [3], [1], [-2], [-1], [-1]])
    e_shifted_lr = np.array([[np.nan], [np.nan], [1], [0], [-3], [-1], [2], [3], [1], [-2], [-1]])
    e_lr_ma = np.array(
        [[np.nan], [np.nan], [np.nan], [1 / 2], [-3 / 2], [-2], [1 / 2], [5 / 2], [2], [-1 / 2], [-3 / 2]])
    e_lr_mstd = np.array([[np.nan], [np.nan], [np.nan], [0.5], [1.5], [1.], [1.5], [0.5], [1.], [1.5], [0.5]])
    lr = LR.run(p).lr.to_numpy()
    shifted_lr = shift_np(lr, 1)
    lr_ma, lr_mstd, vol_ma = ma_mstd(shifted_lr, shifted_vol, 2)
    assert (np.isclose(lr, e_lr, equal_nan=True).all())
    assert (np.isclose(shifted_lr, e_shifted_lr, equal_nan=True).all())
    assert (np.isclose(lr_ma, e_lr_ma, equal_nan=True).all())
    assert (np.isclose(lr_mstd, e_lr_mstd, equal_nan=True).all())
    assert (np.isclose(vol_ma, e_shifted_vol_ma, equal_nan=True).all())

    thld = -0.1
    e_lr_entries = e_lr < e_lr_ma + thld * e_lr_mstd
    e_lr_exits = e_lr > e_lr_ma - thld * e_lr_mstd
    e_vol_entries = v > -thld * vol_ma
    e_final_entries = e_lr_entries & e_vol_entries
    lr_entries, lr_exits, vol_entries = signal_calculations(lr, lr_ma, lr_mstd, v, vol_ma, thld, -thld)
    final_entries, _ = signals_nb(lr, shifted_lr, v, shifted_vol, thld, -thld, lag)
    assert (np.array_equal(lr_entries, e_lr_entries))
    assert (np.array_equal(lr_exits, e_lr_exits))
    assert (np.array_equal(vol_entries, e_vol_entries))
    assert (np.array_equal(final_entries, e_final_entries))
    _portfolio_kwargs = dict(
        direction='longonly',
        size=np.inf,
        freq='m',
        fees=0.001,
    )
    port = ExtendedPortfolio.from_signals(p, e_final_entries, lr_exits, **_portfolio_kwargs)
    e_elr = (np.log(-0.63348844 + 1) + np.log(-0.98270268 + 1)) / 2
    assert (np.isclose(e_elr, port.trades.expected_log_returns()[0], equal_nan=True))
    e_mlr = (np.log(-0.63348844 + 1) + np.log(-0.98270268 + 1)) / 2
    assert (np.isclose(e_mlr, port.trades.median_log_returns()[0], equal_nan=True))
    e_filtered_elr = np.log(-0.63348844 + 1)
    assert (np.isclose(e_filtered_elr, port.trades.expected_log_returns(min_lr=-2)[0], equal_nan=True))
