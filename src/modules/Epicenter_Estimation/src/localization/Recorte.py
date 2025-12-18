# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:57:11 2022

@author: Niko
"""
import os
import os.path
from obspy import read, read_inventory
import pandas as pd
from obspy.signal.trigger import classic_sta_lta,trigger_onset
from obspy.core import UTCDateTime

import sys
from Localization.Funciones import Funciones_features


def Recorte(SAC,CTM,nro_del_evento,t_extra, guardar_trazas_recortadas=False, nombre_sac=None):
    fs_real = SAC[0].stats.sampling_rate
    sac_res = SAC.copy()
    sac_k = sac_res.copy()
    fs = int(sac_k[0].stats.sampling_rate)
    cantidad_sismos = CTM['Inicio'].size #Cantidad de sismos detectados
    #Definir tiempo inicial y final del corte
    if cantidad_sismos>1:
        tin_deteccion = int(CTM['Inicio'][nro_del_evento]*fs  + t_extra*fs)
        tfin_deteccion = int(CTM['Inicio'][nro_del_evento]*fs + CTM['Duracion'][nro_del_evento]*fs)
    else:
        tin_deteccion = int(CTM['Inicio']*fs  + t_extra*fs)
        tfin_deteccion = int(CTM['Inicio']*fs + CTM['Duracion']*fs)

    #Recorte
    sac_k[0] = sac_res[0].slice(UTCDateTime(sac_res[0].stats.starttime + tin_deteccion/fs), UTCDateTime(sac_res[0].stats.starttime + tfin_deteccion/fs))
 
    if guardar_trazas_recortadas:
        
        sac_k.write('data/sac_cortados/'+nombre_sac[:-3]+sac_k[0].stats.channel+ '_'+str(nro_del_evento) +'.sac', format='SAC') 
        
        
    return sac_k

def Picar_P(traza):
    fs=traza[0].stats['sampling_rate']
    #sac_filt_s= Funciones_features.butter_bandpass_lfilter(traza[0].data, lowcut=1, highcut= 2, fs=fs, order=3)
    canal_sac_Z = traza[0].copy()
    canal_sac_E = traza[1].copy()
    canal_sac_N = traza[2].copy()

    sta = canal_sac_Z.stats.station
    inv = read_inventory('Back_azimuth/data/xml/'+sta+'.xml')
    canal_sac_Z.remove_response(inventory=inv, output="VEL")
    canal_sac_E.remove_response(inventory=inv, output="VEL")
    canal_sac_N.remove_response(inventory=inv, output="VEL")

    canal_sac_Z.filter('bandpass', freqmin = 1, freqmax = 10.0, corners=4, zerophase=True)
    canal_sac_E.filter('bandpass', freqmin = 1, freqmax = 10.0, corners=4, zerophase=True)
    canal_sac_N.filter('bandpass', freqmin = 1, freqmax = 10.0, corners=4, zerophase=True)
    
    #cft = classic_sta_lta(sac_filt_s, int(3 * fs), int(6 * fs)) #9 32
    #try:
    #    frame_p =trigger_onset(cft, 1.8, 0.5)[0][0]
    #except:
    #    print('fallo sta/lta y por lo que se fijo que la P se ubica al inicio de la traza')
    #    frame_p = 0
    #return frame_p

    try:
        cft= classic_sta_lta(canal_sac_Z.data, int(5 * fs), int(10 * fs)) #3 6
        P_count =trigger_onset(cft, 1.85, 0.5)[0][0] #1.8 0.5 
        #print('Cuenta de llegada de la P a 40[hz]: ',P_count)
    except:
            
        try:
            cft= classic_sta_lta(canal_sac_Z.data, int(5 * fs), int(10 * fs)) #3 6
            P_count =trigger_onset(cft, 1.8, 0.5)[0][0] #1.8 0.5 
             
        except:
            
            print('Fallo sta-lta, probando canal E')

            try:
                cft= classic_sta_lta(canal_sac_E.data, int(5 * fs), int(10 * fs)) #3 6
                P_count =trigger_onset(cft, 1.85, 0.5)[0][0] #1.8 0.5 

            except:

                try:
                    print('Probando canal N')
                    cft= classic_sta_lta(canal_sac_N.data, int(5 * fs), int(10 * fs)) #3 6
                    P_count =trigger_onset(cft, 1.85, 0.5)[0][0] #1.8 0.5 


                except:
                    print('Relajando criterio sobre Z')
                    cft= classic_sta_lta(canal_sac_Z.data, int(10 * fs), int(20 * fs)) #3 6
                    P_count =trigger_onset(cft, 1.75, 0.5)[0][0] #1.8 0.5 

    return P_count



if False:
    from obspy.signal.trigger import plot_trigger
    traza =sac_corte
    fs=traza[0].stats['sampling_rate']
    sac_filt_s= Funciones_features.butter_bandpass_lfilter(traza[0].data, lowcut=1, highcut= 2, fs=fs, order=3)
    cft = classic_sta_lta(sac_filt_s, int(9 * fs), int(32 * fs))
    
    traza[0].data = sac_filt_s
    
    plot_trigger(traza[0], cft, 1.8, 0.5)
  
    
    plt.plot()    
    plt.plot(sac_filt_s)
    
    frame_p =trigger_onset(cft, 1.8, 0.5)[0][0]












