# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
# import scipy as sp
# from scipy import signal
# from scipy.signal import medfilt

print(55*'#')
print('Prof. PAULO R. P. SANTIAGO'.center(50))
print(55*'#')


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

fc = 30  # Cut-off frequency of the filter
w = fc / (fs / 2)  # Normalize the frequency
b, a = signal.butter(5, w, 'low')

lcell1f = signal.filtfilt(b, a, channel1)
lcell2f = signal.filtfilt(b, a, channel2)
lcell3f = signal.filtfilt(b, a, channel3)
lcell4f = signal.filtfilt(b, a, channel4)

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
