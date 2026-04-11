import numpy as np
import pandas as pd
from scipy.signal import argrelmin, argrelmax
# %%
def get_shape_ratio(ret_ser, type = 'day'):
    return_mean = ret_ser.mean()
    return_std = ret_ser.std()
    if type == 'day':
        return return_mean / return_std * np.sqrt(252)
    if type == 'month':
        return return_mean / return_std * np.sqrt(12)

def get_max_drawdown(cum_ser):
    Roll_Max = cum_ser.cummax()
    Daily_Drawdown = cum_ser / Roll_Max - 1.0
    return -Daily_Drawdown.min()

def get_calmar_ratio(cum_ser):
    max_draw = get_max_drawdown(cum_ser)
    return (cum_ser.iloc[-1] - 1) / max_draw



