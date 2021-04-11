import numpy as np
from ipynb.fs.defs.Alpha101 import apply_alpha_nb


def _test_apply_func():
    ## setup       -0.3  1  -1
    #              -0.3,0.9,-0.9
    high =  np.array([4, 9, 9])
    open =  np.array([3, 1, 9])
    close = np.array([2, 9, 1])
    low =   np.array([1, 1, 1])
    buy_threshold = 0.5
    sell_threshold = 0.5
    # expected values
    alpha = (close - open) / (high - low + 0.001)
    for i in range(len(alpha)):
        if alpha[i] >= buy_threshold:
            alpha[i] = 1
        elif alpha[i] <= -sell_threshold:
            alpha[i] = -1
    ## run
    resp = apply_alpha_nb(open,high,low,close,buy_threshold,sell_threshold)
    ## assert
    assert (np.allclose(alpha, resp))


_test_apply_func()
