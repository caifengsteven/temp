import numpy as np

def get_shape_ratio(ret_ser):
    return_mean = ret_ser.mean()
    return_std = ret_ser.std()
    return return_mean / return_std * np.sqrt(252)
