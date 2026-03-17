import pandas as pd
import numpy as np
from scipy import fft, ifft
from scipy.signal import hilbert
import matplotlib.pyplot as plt
# %%
def hilbert_from_scratch(signal):
    fast_ft = fft(signal) #scipy fft
    for i in range(6,len(signal),4):
        fast_ft[i] = 0
        fast_ft[i+1] = 0

    fast_ft[0] = fast_ft[0]*.5
    fast_ft[1] = fast_ft[1]*.5

    if(len(fast_ft) > 1):
        fast_ft[2] = fast_ft[2]*.5
        fast_ft[3] = fast_ft[3]*.5

    inverse_fft = ifft(fast_ft) #scipy ifft

    x = 2 / len(signal)

    for i in range(0,len(signal),1):
        inverse_fft[i] = inverse_fft[i]*x

    return inverse_fft

# %%
x = np.sin(np.linspace(0, 7, 100))
plt.plot(x)
plt.show()

# %%
xh = hilbert(x)
x_ang = np.angle(xh, deg = False)
plt.plot(np.cos(x_ang), np.sin(x_ang))
plt.show()


