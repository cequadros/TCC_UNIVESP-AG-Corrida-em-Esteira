# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import scipy as sp
# from scipy import signal
# from scipy.signal import medfilt

print(55*'#')
print('Prof. PAULO R. P. SANTIAGO'.center(50))
print(55*'#')


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.median(y, axis=1)


# nome = input('Digite o nome do arquivo com o no final .txt  ')
nome = 'datatestePeso6kmCansado.txt'

regex = '\s+'
data = pd.read_csv(nome, sep=regex, header=None)
datm = data.values

nlin = np.size(datm, 0)  # numero de linhas
ncol = np.size(datm, 1)  # numero de colunas
fs = 1000  # frequency sample

# tempo = np.linspace(0,nlin-1,nlin) / fs
tempo = datm[:, 0]
channel1 = datm[:, 1]
channel2 = datm[:, 2]
channel3 = datm[:, 3]
channel4 = datm[:, 4]

# channelf = sp.signal.medfilt(channel1, 7) # remove de noise of the channel 1
# channe2f = sp.signal.medfilt(channel2, 7) # remove de noise of the channel 2
# channe3f = sp.signal.medfilt(channel3, 7) # remove de noise of the channel 3
# channe4f = sp.signal.medfilt(channel4, 7) # remove de noise of the channel 4

window_len = int(input('What the window size?  '))

lcell1f = medfilt(channel1, window_len)  # remove de noise of the channel 1
lcell2f = medfilt(channel2, window_len)  # remove de noise of the channel 2
lcell3f = medfilt(channel3, window_len)  # remove de noise of the channel 3
lcell4f = medfilt(channel4, window_len)  # remove de noise of the channel 4

lcellfilt = np.matrix([lcell1f, lcell2f, lcell3f, lcell4f])

lcellfilter = lcellfilt.transpose()

np.savetxt('filt_'+nome, lcellfilter, fmt='%.4f')

plt.subplot(2, 2, 1)
# plt.hold()
plt.plot(tempo, channel1, 'r-')
plt.plot(tempo, lcell1f, 'k-')
plt.title('Load Cell 1')
# plt.xlabel('time [s]')
plt.ylabel('Voltage')

plt.subplot(2, 2, 2)
plt.plot(tempo, channel2, 'r-')
plt.plot(tempo, lcell2f, 'k-')
plt.title('Load Cell 2')
# plt.xlabel('time [s]')
plt.ylabel('Voltage')

plt.subplot(2, 2, 3)
plt.plot(tempo, channel3, 'r-')
plt.plot(tempo, lcell3f, 'k-')
plt.title('Load Cell 3')
plt.xlabel('time [s]')
plt.ylabel('Voltage')

plt.subplot(2, 2, 4)
plt.plot(tempo, channel4, 'r-')
plt.plot(tempo, lcell4f, 'k-')
plt.title('Load Cell 4')
plt.xlabel('time [s]')
plt.ylabel('Voltage')
plt.show()
