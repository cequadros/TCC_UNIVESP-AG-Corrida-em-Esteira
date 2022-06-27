import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pandas import DataFrame

# constantes
NmediaMovel = 7
# Quantidade de amostra considerada para media
quantAmostras = 2

# funcoes

#Calcula quantidade de picos
def numPicos(df):
    df.reset_index(inplace=True)
    valoresPicos = []
    posicaoPicos = []
    auxValor = df.loc[0, 'Fz.1']
    auxPosc = df.loc[0, 'Frame']
    numeroPicos = 0
    tamanho = df.shape
    oldDist = 0

    for i in range(2, tamanho[0] - 1):

        if(df.loc[i - 2, 'Fz.1'] > auxValor):
            newDist = i
            if(newDist - oldDist > 4):

                if((df.loc[i, 'Fz.1'] > auxValor)):

                    numeroPicos += 1
                    posicaoPicos.append(auxPosc)
                    valoresPicos.append(auxValor)

                    oldDist = newDist

        auxValor = df.loc[i, 'Fz.1']
        auxPosc = df.loc[i, 'Frame']

    return numeroPicos, valoresPicos, posicaoPicos


#Trata os dados e aplica media movel
def tratemandoDados(arr):
    listaResult = []
    auxLista = []
    aux2 = False
    aux = False

    for i in range(1, len(arr) - 1):
        if(arr[i][1] != 0):
            aux2 = False
            aux = True

        if(arr[i][1] == 0 or i == len(arr) - 1):
            aux = False
            aux2 = False
            if(arr[i + 1][1] == 0 and arr[i - 1][1] != 0):
                aux2 = True

        if(aux):
            auxLista.append(arr[i])

        if(aux2):
            # aplicando media movel
            for _ in range(0, NmediaMovel):
                for i in range(1, len(auxLista) - 1):
                    if(auxLista[i][1] != 0):
                        auxLista[i][1] = (
                            auxLista[i - 1][1] + auxLista[i][1] + auxLista[i + 1][1]) / 3

            auxLista = DataFrame(auxLista)
            auxLista.rename(columns={0: 'Frame', 1: 'Fz.1'}, inplace=True)
            listaResult.append(auxLista)
            auxLista = []

    return listaResult

#Calcula a distancia entre os picos
def distanciaPicos(valores):
    distancia = 0
    tamanho = len(valores)
    distancia = valores[tamanho - 1] - valores[0]

    return distancia


#Faz a media da quantidade de picos, da distancia entre os picos x1 e x2
def media(list):
    listMediaQuantPicos = []
    listMediaValoresX1 = []
    listMediaValoresX2 = []
    listMediaDistanciaPicos = []

    cont = 0
    totalPicos = 0
    distanciaTotal = 0
    x1Total = 0
    x2Total = 0

    for i in range(0, len(list)):
        valoresPicos = []
        posicaoPicos = []
        distancia = 0
        quant = 0

        quant, valoresPicos, posicaoPicos = numPicos(list[i])
        distancia = distanciaPicos(posicaoPicos)

        cont = cont + 1
        totalPicos = totalPicos + quant
        x1Total = x1Total + valoresPicos[0]
        x2Total = x2Total + valoresPicos[len(valoresPicos) - 1]
        distanciaTotal = distanciaTotal + distancia

        if(cont == quantAmostras):
            listMediaQuantPicos.append(totalPicos/quantAmostras)
            listMediaValoresX1.append(x1Total/quantAmostras)
            listMediaValoresX2.append(x2Total/quantAmostras)
            listMediaDistanciaPicos.append(distanciaTotal/quantAmostras)
            cont = 0
            totalPicos = 0
            x1Total = 0
            x2Total = 0
            distanciaTotal = 0

    return listMediaDistanciaPicos, listMediaQuantPicos, listMediaValoresX1, listMediaValoresX2


#Distancia de cada amostra
def distanciaAmostras(list):
    distanciaTotal = []

    for i in range(0, 30, 2):
        distancia = list[i + 1].loc[0, 'Frame'] - \
            list[i].loc[len(list[i]) - 1, 'Frame']
        distanciaTotal.append(distancia)

    return distanciaTotal


# Carrega dados
meia = pd.read_csv('meia01.csv')
tenis = pd.read_csv('tenis01.csv')


# Seleciona só dados de força
meiaForce = meia.loc[:, ['Frame', 'Fz.1']]
tenisForce = tenis.loc[:, ['Frame', 'Fz.1']]


# Seleciona os valores
tenisForce = tenisForce.values
meiaForce = meiaForce.values


# Tratamento dos dados
for i in range(0, len(tenisForce)):
    if(tenisForce[i][1] > -500):
        tenisForce[i][1] = 0

for i in range(0, len(meiaForce)):
    if(meiaForce[i][1] > -500):
        meiaForce[i][1] = 0


# Transformando lista em array
listaTenis = np.array(tenisForce)
tenisForce = listaTenis

listaMeia = np.array(meiaForce)
meiaForce = listaMeia


# Tratamento dados
listaTenis = []
listaTenis = tratemandoDados(tenisForce)

listaMeia = []
listaMeia = tratemandoDados(meiaForce)


# Analisando
# Para dados 1
listMediaDistanciaPicosMeia = []
listMediaQuantPicosMeia = []
listMediaValoresX1Meia = []
listMediaValoresX2Meia = []
periodoAmostraMeia = []

listMediaDistanciaPicosMeia, listMediaQuantPicosMeia, listMediaValoresX1Meia, listMediaValoresX2Meia = media(listaMeia)
distanciaAmostraMeia = distanciaAmostras(listaMeia)

# Para dados 2
listMediaDistanciaPicosTenis = []
listMediaQuantPicosTenis = []
listMediaValoresX1Tenis = []
listMediaValoresX2Tenis = []
periodoAmostraTenis = []

listMediaDistanciaPicosTenis, listMediaQuantPicosTenis, listMediaValoresX1Tenis, listMediaValoresX2Tenis = media(listaTenis)
distanciaAmostraTenis = distanciaAmostras(listaTenis)


#Essas duas partes sao para salvar os dados e a de baixo para gerar os graficos 1 por 1. 
#'''
#Escrevendo em dois arquivos os resultados
for i in range(0, 15):

    arq1 = open('arquivoArfMeia1.txt', 'a')
    texto = str(listMediaQuantPicosMeia[i]) + "," + str(listMediaValoresX1Meia[i]) + "," + str(listMediaValoresX2Meia[i]
                                                                                               ) + "," + str(listMediaDistanciaPicosMeia[i]) + "," + str(distanciaAmostraMeia[i]) + "," + "Com-meia" + "\n"
    arq1.write(texto)
    arq1.close()

    arq2 = open('arquivoArfTenis1.txt', 'a')
    texto = str(listMediaQuantPicosTenis[i]) + "," + str(listMediaValoresX1Tenis[i]) + "," + str(listMediaValoresX2Tenis[i]
                                                                                                 ) + "," + str(listMediaDistanciaPicosTenis[i]) + "," + str(distanciaAmostraTenis[i]) + "," + "Com-Tenis" + "\n"
    arq2.write(texto)
    arq2.close()
#'''

#'''
for i in range(0, 30):
    plt.xlabel('Tempo')
    plt.ylabel('Força')
    plt.title(i + 1)

    #Grafico listaMeia
    #plt.plot(listaMeia[i]['Fz.1'], color= 'red')

    #Grafico listaTenis
    plt.plot(listaTenis[i]['Fz.1'], color= 'red')
    plt.show()
#'''