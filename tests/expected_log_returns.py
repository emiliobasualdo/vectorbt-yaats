
import numpy as np
import pandas as pd

from lib.utils import ExtendedPortfolio

# voy a testear que un simple buy&hold
# setup
close = pd.DataFrame([[0.5, 1.5], [1, 2]])
print(close.head())

buy_price = close.iloc[0]
sell_price = close.iloc[-1]
gross_returns = sell_price/buy_price
log_gross_return = np.log(gross_returns)
print("buy_price", buy_price)
print("sell_price", sell_price)
print("gross_returns", gross_returns)
print("log_gross_return", log_gross_return)

# run
portfolio_kwargs = dict(
    direction='longonly',
    freq='m'
)
hold_elr = ExtendedPortfolio.from_holding(close, **portfolio_kwargs).expected_log_returns()
print("hold_elr", hold_elr)

#test
assert (np.allclose(hold_elr,log_gross_return)) # equals and epsilon