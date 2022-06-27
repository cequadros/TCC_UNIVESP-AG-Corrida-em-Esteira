# -*- coding: utf-8 -*-
# %% [markdown]
# ### Apresentacao do script: Sergio Baldo & Paulo Santiago
# %%
# print('\n')
# print(58*'#')
# print('IA_TREADMILL.PY'.center(58))
# print('Análise das forças de contato no solo em esteira ergométrica'.center(58))
## print(58*'#')
# print('\n')

# %% [markdown]
# ### Importando bibliotecas necessárias 
# %%
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import scipy as sp
import matplotlib.pyplot as plt
import sys


# %% [markdow]
# ### Função para realizar o filtro de butterworth
# %%
def filtro(dat, fc=59, fs=1000, filtorder=4, typefilt='low'):
    import numpy as np
    from scipy import signal
    
    nl, nc = dat.shape
    # fc=59  # Cut-off frequency of the filter
    w = fc/(fs/2)  # Normalize the frequency
    b, a = signal.butter(filtorder, w, typefilt)
    
    datf = np.zeros([nl, nc], dtype=float)
    for i in range(nc):
        datf[:,i] = signal.filtfilt(b, a, dat[:,i])
   
    return datf

# %% [markdow]
# ### Função para carregar o arquivo das células de carga
# %%
def readcell(dat, freq=1000, valtara=None, peso=None, filtrar=None, sumcell=None, salvar=None):
    # dfcell = pd.read_csv('hiit01_15kmh_08042020.csv', sep=',', header=None)
    # import pandas as pd
    # import matplotlib.pyplot as plt
    
    dfcell = pd.read_csv(dat, sep=',', header=None)
    cell = -1 * (dfcell[[1,2,3,4]].to_numpy())
    
    # plt.close('all')
    # plt.plot(cell)
    
    if valtara is None:
        # import numpy as np
        i1 = int(np.round(freq/10))
        f1 = int(np.round(freq+freq/10))
        tarar = np.mean(cell[i1+10:f1+10,:], axis=0)
        res = cell - tarar
        vtara = tarar
    else:
        dfvtara = pd.read_csv(valtara, sep=',', header=None)
        dftara = -1 * dfvtara[[1,2,3,4]].to_numpy()
        vtara = np.mean(dftara, 0)
        res = cell - vtara
    # res = cell
   
    print(f'valor de tara = {vtara}')
    if filtrar is not None:
        from ia_treadmill import filtro
        res1 = filtro(res, fc=float(filtrar))
    else:
        res1 = res

    if sumcell is None:
        # import numpy as np
        res2a = np.sum(res1, 1)
    else:
        res2a = res1
    
    if peso is not None:
        dfnormpeso = pd.read_csv(peso, sep=',', header=None)
        normpesoa = -1 * dfnormpeso[[1,2,3,4]].to_numpy()
        normpeso = normpesoa - vtara
        normpeso1 = np.mean(normpeso, 0)
        normpeso = np.sum(normpeso1)
        res2 = res2a / normpeso
    else:
        res2 = res2a
    
    
    # plt.figure()
    # plt.plot(res2)
   
    if salvar is not None:
        if salvar is True:
            nome = input('Digite um nome do arquivo salvo: ')
        else:
            nome = salvar
        np.savetxt(str(nome)+'.txt', res2, fmt='%.21f')
        
    return res2

# %% [markdown] 
# ### Função para definir o número de foot-strikes no sinal
# %%
def selectstrikes(dat, freq=1000, limiar=10):
    import numpy as np
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt
    
    if limiar%2 != 0:
        limiar = limiar+1
    
    dat = np.array(dat)
    
    plt.close('all')
    plt.figure()
    plt.title('Escolha o início e fim do sinal que deseja analisar (click esquerdo do mouse)')
    plt.plot(dat)
    plt.ylabel('Vertical GRF (BW)')
    
    x = plt.ginput(2, timeout=120)
    xinicio = int(np.round(x[0][0]))
    xfim = int(np.round(x[1][0]))
    datc = dat[xinicio:xfim]
    plt.plot(list(range(xinicio,xfim)), dat[xinicio:xfim], '--r')
    

    datmin = min(datc)
    posmin = np.argmin(datc)
    
    dat1 = datc - datmin
    limiar_corte = max(dat1[posmin-limiar:posmin+limiar]) +  2 * np.std(dat1[posmin-limiar:posmin+limiar])
    
    print(f'Cut-off threshold = {limiar_corte}')
    plt.title(f'Select frames: Start = {xinicio} | End = {xfim}')

    plt.figure()
    plt.plot(dat1)
    plt.plot(list(range(len(dat1))), np.linspace(limiar_corte,limiar_corte,len(dat1)), '--r')
    plt.title(f'Cut-off threshold = ({limiar_corte})')

    uplimiar = dat1 > limiar_corte
    downlimiar = dat1 < limiar_corte
    dat2 = dat1[uplimiar]
    datzeros = dat1
    datzeros[downlimiar] = 0
    
    dat2inv = -1*dat2
    peaks, _ = find_peaks(dat2inv, height=[np.mean(dat2inv) + np.std(dat2inv)], distance=np.round(freq/limiar))
   
  
    if len(peaks)%2 != 0:
        cortes1 = peaks[:-1]
    else:
        cortes1 = peaks
    
    

    plt.figure()
    plt.plot(dat2)
    plt.plot(cortes1[::2], dat2[cortes1[::2]], 'gv')
    plt.plot(cortes1[1::2], dat2[cortes1[1::2]], 'r^')
    plt.ylabel('Vertical GRF (BW)')
    
    datcorte = dat2[cortes1[0]:cortes1[-1]]
    cortes2 = cortes1 - cortes1[0]
    nstrikes = int(len(cortes2)-1)

    plt.title(f'Selected for the start and end of foot-strikes (n strikes = {nstrikes})')
    
    plt.figure()
    plt.plot(datcorte)
    plt.plot(cortes2[::2], datcorte[cortes2[::2]], 'gv')
    vecorteplot2 = cortes2[1::2] 
    vecorteplot2[-1] = vecorteplot2[-1]-1
    plt.plot(vecorteplot2, datcorte[vecorteplot2], 'r^')
    
    plt.title(f'Signal adjustment.s (n strikes = {nstrikes})')
    plt.ylabel('Vertical GRF (BW)')
    
    plt.figure()
    plt.plot(datzeros)
    
    loczeros = np.argwhere(datzeros == 0)
    locnzeros = np.argwhere(datzeros != 0)
    
    plt.plot(loczeros[:,0], datzeros[loczeros[:,0]], '.r')
    plt.title(f'Selected signal with values below the threshold replaced by zero: {limiar_corte}')
    plt.ylabel('Vertical GRF (BW)')
    
    ### Seleção de pico baseado em devivadas e picos
    
    # dat1der = np.diff(dat1, axis=0)
    # dat1 = dat1[:,0]
    # dat1der = dat1der[:,0]
    
    # ax1 = plt.subplot(2,1,1)
    # ax1.plot(dat1)
    # ax2 = plt.subplot(2,1,2, sharex=ax1)
    # ax2.plot(dat1der)
    
    # limiar = 10
    # peaks, properties = find_peaks(dat1, height=[np.mean(dat1) + np.std(dat1)], distance=np.round(freq/limiar))
    # ax1.plot(peaks, dat1[peaks], 'rv')

    # mindist = min(np.diff(peaks))-limiar
    # peaks1, properties1 = find_peaks(dat1der, distance=mindist, height=max(dat1der)-3*np.std(dat1der))
    # inicios_dat = peaks1-limiar

    # ax2.plot(peaks1, dat1der[peaks1], 'rv')
    # ax1.plot(inicios_dat, dat1[inicios_dat], 'gv')
    
    # dat2der = np.diff(-1*dat1, axis=0)
    # # dat2der = dat2der[:,0]
    # peaks2, properties2 = find_peaks(dat2der, distance=mindist, height=max(dat2der)-3*np.std(dat2der))
    # finais_dat = peaks2 + 4*limiar
    # ax2.plot(peaks2, dat1der[peaks2], 'm^')
    # ax1.plot(finais_dat, dat1[finais_dat], 'm^')
        
    # dat1inve = dat1[::-1]
    datcorte1 = dat2
    datcorte2 = datcorte
    print(f'Number of foot-strikes = {nstrikes}')
    return datcorte1, cortes1, datcorte2, cortes2, datzeros, loczeros

# %% [markdown] 
# ### Função para extrair os atributos de um único foot-strikes
# %%
def strikeattr(dat, corte=None, showfig=False, numgraph=None):
    # dat = pd.read_csv('strike4channel.txt', sep=' ', header=None)
    
    if corte is not None:
        dat = dat[corte[0]:corte[1]]
    else:
        dat = dat
    
    datres = np.array(dat)
    
    pos_peaks_datres, _ = find_peaks(datres) # número de picos
    pos_peakmax = np.argmax(datres) # posicao do pico máximo
    val_peakmax = max(datres) # valor do pico máximo
    
    der_datres = np.diff(datres) # derivando o sinal
    
    pos_maxdiff = np.argmax(der_datres)
    val_maxdiff = max(der_datres)
    
    peaks_derdatres, _ = find_peaks(-1*der_datres[pos_maxdiff:pos_peakmax])
    pos_itransient = peaks_derdatres[0]+1+pos_maxdiff
    val_itransient = datres[peaks_derdatres[0]+1+pos_maxdiff]
    
    pos_maxdiff = np.argmax(der_datres)
    val_maxdiff = max(der_datres)
    
    # impacttrasient2 = np.trapz(datres[pos_itransient:pos_peakmax])
    
    # retirada = np.trapz(datres[pos_peakmax:-1])
    
    # plt.close('all')
    # plt.figure()
    if showfig is True:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        ax1.plot(datres, 'k--')
        ax1.plot(pos_peakmax, val_peakmax, 'yv', markersize=10)
        ax1.plot(pos_itransient, val_itransient, 'rv', markersize=10)
        ax1.fill_between(range(len(datres)), datres[0],datres, color='b', alpha=0.2)
        ax1.fill_between(range(pos_itransient+1), datres[0],datres[0:pos_itransient+1], color='red', alpha=0.4)
        ax1.plot(pos_maxdiff+1, datres[pos_maxdiff+1], 'gv', markersize=10)
        ax1.fill_between(range(pos_itransient,pos_peakmax+1), datres[0], datres[pos_itransient:pos_peakmax+1], color='yellow', alpha=0.4)
        ax1.fill_between(range(pos_maxdiff+2), datres[0], datres[0:pos_maxdiff+2], color='green', alpha=0.6)
        
        
        ax2.plot(der_datres, 'k--')
        ax2.plot(pos_peakmax-1, der_datres[pos_peakmax-1], 'yv', markersize=10)
        ax2.plot(pos_itransient-1, der_datres[pos_itransient-1], 'rv', markersize=10)
        ax2.plot(pos_maxdiff, val_maxdiff, 'gv', markersize=10)
        ax2.fill_between(range(len(der_datres)), der_datres[0], der_datres, color='b', alpha=0.2)
        ax2.fill_between(range(pos_itransient), der_datres[0], der_datres[0:pos_itransient], color='red', alpha=0.4)
        ax2.fill_between(range(pos_maxdiff+1), der_datres[0], der_datres[0:pos_maxdiff+1], color='green', alpha=0.6)
        ax2.fill_between(range(pos_itransient-1,pos_peakmax), der_datres[pos_itransient-1], der_datres[pos_itransient-1:pos_peakmax], color='yellow', alpha=0.4)
        
        # ax2.fill_between(range(pos_itransient), 0, datres[0:pos_itransient], color='red', alpha=0.4)
        
        ax1.set_title('Attributes Foot-Strike')
        ax2.set_title('Derivative (first-order)')
        ax1.set_ylabel('Vertical GRF (BW)')
        
        if numgraph is not None:
            fig.suptitle(f'Strike Attributes: {numgraph}')
        else:
            fig.suptitle('Strike Attributes')
        
        plt.show()
    
    attr1 = val_peakmax # magnitude do pico máximo (ponto amarelo Y)
    attr2 = pos_peakmax # tempo para o pico máximo (ponto amarelo X)
    attr3 = len(pos_peaks_datres) #  Número de picos
    attr4 = len(datres) # duração de toda fase de apoio (tudo X)
    attr5 = val_itransient # Valor do 1 transient impact (ponto vermelho Y)
    attr6 = pos_itransient # Posição (tempo) do 1 transient impact (ponto vermelho X)
    attr7 = datres[pos_maxdiff+1] # Valor do ponto de maior inclinação do início do contato (ponto verde Y)
    attr8 = pos_maxdiff+1 # Posição (tempo) do ponto de maior inclinação do início do contato (ponto verde X)
    attr9 = np.trapz(datres) # Integral trapezoidal da curva (áreas: verde + vermelha + amarela + azul)
    attr10 = np.trapz(datres[0:pos_peakmax]) # Integral trapezoidal da curva até o ponto de pico max (áreas: verde + vermelha + amarela)
    attr11 = np.trapz(datres[0:pos_itransient]) # Integral trapezoidal do 1 impacto transient (área verde + vermelha)
    attr12 =  attr8 * datres[pos_maxdiff+1] # Produto da maior inclição pelo tempo (aproximadamente área verde)
    # attr12 = np.trapz(datres[0:attr8]) # Integral da área verde
    attr13 = np.trapz(datres[attr6:attr2]) # Integral trapezoidal da area amarela
    attr14 = np.trapz(datres[attr8:attr6]) # Integral trapezoidal da area vermelha
    attr15 = np.trapz(datres[attr2:-1]) # Integral da area apos o pico de força (área azul)
    
    res2 = np.matrix([attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10, attr11, attr12, attr13, attr14, attr15])
    
    res1 = [attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10, attr11, attr12, attr13, attr14, attr15]

    res3 = {'max':attr1,
           'tmax':attr2,
            'npeaks':attr3,
            'ttsupport':attr4,
            'itransient1':attr5,
            'titransient1':attr6,
            'itransient2':attr7,
            'titransient2':attr8,
            'impall-g+r+y+b':attr9,
            'imp2max-g+r+y':attr10,
            'imp2itransient1-g+r':attr11,
            'imp2itransient2-g':attr12,
            'imp2itransient3-y':attr13,
            'imp2itransient4-r':attr14,
            'imp2itransient5-r':attr15} 

    return res1, res2, res3


# %% [markdown]
# ### Para rodar em IDE Spyder, Pycharm etc.
# %% [markdown]
# ### Apresentacao do script: Sergio Baldo & Paulo Santiago
# %%
def iatreadmill_ide(dat_corrida, dattara, datpeso, limiarc=16, graficos=False):
    print('\n')
    print(58*'#')
    print('IA_TREADMILL.PY'.center(58))
    print('Análise das forças de contato no solo em esteira ergométrica'.center(58))
    print(58*'#')
    print('\n')
    
    datread = readcell(dat_corrida, valtara=dattara, peso=datpeso, filtrar=59)
    datcorte1, cortes1, datcorte2, cortes2, datzeros, loczeros = selectstrikes(datread, limiar=limiarc)
    numstrikes = int(len(cortes2)-1)
  
    res_attrmat = np.zeros([numstrikes,15])   
    
    # plt.close('all')
    for i in range(numstrikes):
        _ , attrmat, _ = strikeattr(datcorte2, corte=cortes2[[i,i+1]], showfig=graficos, numgraph=i+1)
        res_attrmat[i,:] = attrmat
    np.savetxt('mat_attr_'+str(dat_corrida), res_attrmat, fmt='%.10f')

    return res_attrmat  


# %% [markdown]
# ### Para rodar no Terminal de comando: Shell Linux, BSD e Mac ou CMD Windows.
# ### Como rodar o script no Terminal ou CMD
# python ia_treadmill arquivo_corrida.csv arquivo_tara.csv 14
# %%
if __name__ == '__main__':
# %% [markdown]
# ### Apresentacao do script: Sergio Baldo & Paulo Santiago
# ### Para rodar digite no terminal dentro da pasta do código e arquivos
# ### python ia_treadmill.py hiit12_13kmh_roberta_15042020.csv esteira_roberta_15042020.csv peso_roberta_15042020.csv 16 0
# ### No final digite 0 para não apresentar gráfico e 1 para apresentar gráfico
# %%
    print('\n')
    print(58*'#')
    print('IA_TREADMILL.PY'.center(58))
    print('Análise das forças de contato no solo em esteira ergométrica'.center(58))
    print('Bolsista SERGIO BALDO'.center(58))
    print('Prof. RENATO TINÓS'.center(58))
    print('Prof. PAULO R. P. SANTIAGO'.center(58))
    print('sergiobaldo@usp.br , paulosantiago@usp.br & rtinos@ffclrp.usp.br'.center(58))
    print('LaBioCoM-EEFERP-USP'.center(58))
    print('Created on 12/04/2020 - Update on 30/04/2020'.center(58))
    print(58*'#')
    print('\n')
        
    #import sys
    datread = readcell(str(sys.argv[1]), valtara=str(sys.argv[2]), peso=str(sys.argv[3]), filtrar=59)
    datcorte1, cortes1, datcorte2, cortes2, datzeros, loczeros = selectstrikes(datread, limiar=int(sys.argv[4]))
    
    numstrikes = int(len(cortes2)-1)
 
    res_attrmat = np.zeros([numstrikes,15])   
    
    mostragraficos = int(sys.argv[5])
    if mostragraficos == 0:
        graficos = False
    else:
        graficos = True
    
    
    # plt.close('all')
    for i in range(numstrikes):
        _ , attrmat, _ = strikeattr(datcorte2, corte=cortes2[[i,i+1]], showfig=graficos,  numgraph=i+1)
        res_attrmat[i,:] = attrmat
    np.savetxt('mat_attr_'+str(sys.argv[1]), res_attrmat, fmt='%.8f')

    
