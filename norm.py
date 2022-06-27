import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
from pandas import DataFrame
from scipy import signal
import os

dir_atual = os.getcwd()        
os.chdir(dir_atual)
#from Filt_butterworth import filt_butterworth
#from PreSignalProcessing import preSignalProcessing
#from AnalysisOfSignals import analysisOfSignals




# Responsavel por selecionar o ponto de inicio do sinal no eixo y, 
# retirando assim as variações apresentadas perto do ponto zero
def delimitRegionOfSignalsOnTheY_axis(df):
    for i in range(0, len(df)):     
        if( df[i] < 0):
            df[i] = 0
    df = np.array(df)

    return df
        
# Responsavel por separar os sinais 1 a 1, isto é, cada elemento da lista sera um sinal
def separateSignalsOnTheX_axis(df):
    finalList = []
    auxiliaryList = []

    for i in range(1, len(df) - 1):
        aux0 = True
        aux1 = False
        
        if(df[i] == 0):
            aux0 = False
            if((df[i - 1] != 0 and df[i + 1] == 0)):
                aux1 = True

        if(i == len(df) - 2):
            aux1 = True
        
        if(aux0):
            auxiliaryList.append(df[i])
                
        if(aux1):
            if(len(auxiliaryList) > 0):

                auxiliaryList = DataFrame(auxiliaryList)
                finalList.append(auxiliaryList)
                auxiliaryList = []
    
    listResult = []

    for i in range(len(finalList), len(finalList) - 42, -1 ):
        listResult.append(finalList[i - 1])

    listResult.pop(0)
    listResult.pop(len(listResult) - 1)

    print(len(listResult))
    return listResult


def run(df):
    data = delimitRegionOfSignalsOnTheY_axis(df)
    data = separateSignalsOnTheX_axis(data)

    return data

    
def filterButterworth(nameData):
    lcelldata = pd.read_csv(nameData, header=None)
    datm = lcelldata.values

    fs = 1000  # frequency sample

    channel1 = datm[:, 1]
    channel2 = datm[:, 2]
    channel3 = datm[:, 3]
    channel4 = datm[:, 4]

    fc = 59  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(5, w, 'low')

    lcell1f = signal.filtfilt(b, a, channel1)
    lcell2f = signal.filtfilt(b, a, channel2)
    lcell3f = signal.filtfilt(b, a, channel3)
    lcell4f = signal.filtfilt(b, a, channel4)

    lcellfilt = np.matrix([lcell1f, lcell2f, lcell3f, lcell4f])

    lcellfilter = lcellfilt.transpose()

    #salva novo arquivo
    np.savetxt('DadosFiltrados/filt_' + nameData, lcellfilter, fmt='%.10f')
    nome = "filt_" + nameData

    return nome



#nome arquivo
fileName = 'hiit01_15kmh_08042020.csv'


fileName = filterButterworth(fileName)


# Abrir arquivo depois de filtrado
dfFinal = pd.read_csv('DadosFiltrados/' + fileName, sep=' ', header=None)





channel1 = ( ((dfFinal[0].loc[30000:70000, ]) - dfFinal[0].loc[0:30000,].mean()) * -1 )
channel1.reset_index(inplace = True, drop=True) # resetar index
channel1 = run(channel1)


channel2 = ( ((dfFinal[1].loc[30000:70000, ]) - dfFinal[1].loc[0:30000,].mean()) * -1 )
channel2.reset_index(inplace = True, drop=True) # resetar index
channel2 = run(channel2)


channel3 = ( ((dfFinal[2].loc[30000:70000, ]) - dfFinal[2].loc[0:30000,].mean()) * -1 )
channel3.reset_index(inplace = True, drop=True) # resetar index
channel3 = run(channel3)


channel4 = ( ((dfFinal[3].loc[30000:70000, ]) - dfFinal[3].loc[0:30000,].mean()) * -1 )
channel4.reset_index(inplace = True, drop=True) # resetar index
channel4 = run(channel4)


#plotando os 5 primeiro sinais da primeira celula
for i in range(0, 5):
    plt.plot(channel1[i])
    plt.show()