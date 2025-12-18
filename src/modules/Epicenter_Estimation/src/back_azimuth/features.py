#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os.path
import numpy as np
from obspy import read
from sklearn.preprocessing import OneHotEncoder
import scipy as sp
from src.utils import az
import math
import pandas as pd


class FeatureExtractor:
    def __init__(
        self, stations_index_path="./data/set/stations_index.csv", response="VEL", window_limit=[1,3], filter=['bandpass', [0.5, 10]]
    ):
        self.response = response
        self.stations_index_path = stations_index_path
        self.nro_segundos_ventana_izq = window_limit[0]
        self.nro_segundos_ventana_der = window_limit[1]
        self._setup_encoder()
        self.filter = filter
        self.correlative_index()

    def _setup_encoder(self):
        self.enc = OneHotEncoder()
        df = pd.read_csv(self.stations_index_path)
        ch_station = df['station'].values
        self.enc.fit(np.array(ch_station).reshape(-1, 1))

        # self.enc = OneHotEncoder()
        # id_estacion = np.array(['PB09','PB06','AC02','CO02','PB14','CO01','GO01','GO03', 'MT16', 'PB18','AC04','AC05',
        #                         'AP01','CO03','GO04','HMBCX','MNMCX','MT02','MT03','MT05','PATCX','PB01','PB02','PB03',
        #                         'PB04','PB05','PB07','PB10','PB11','PB12','PB15','PSGCX','TA01','TA02','VA03',
        #                         'GO02','GO05','PB16','PB08','CO04','VA01','AC01','CO05','CO06','VA06','CO10',
        #                         'BO03','AC07','PX06']).reshape(-1,1)

        # self.enc.fit(id_estacion)


    def correlative_index(self):
        #self.enc = OneHotEncoder()
        df = pd.read_csv(self.stations_index_path)

        ch_station = df['station'].values
        nro_station = np.arange(0, len(ch_station))
        id_estacion = dict(zip(ch_station, nro_station))
        self.id_estacion = id_estacion

    def filter_builder(self, trace, filter, options):
        if filter == "bandpass":
            print(f"Applying bandpass filter from {options[0]} to {options[1]} Hz")
            trace.filter(filter, freqmin=options[0], freqmax=options[1], corners=2, zerophase=True)
        else:
            print(f"Applying {filter} filter with frequency {options} Hz")
            trace.filter(filter, freq=options, corners=2, zerophase=True)

    def get_features(self, trace, frame_p, inv, one_hot_encoding = False ,fix_p = False, use_cnn = False):

        trace_copy = trace.copy()
        canal_sac_Z = trace_copy.select(channel="*Z")
        canal_sac_E = trace_copy.select(channel="*E")
        canal_sac_N = trace_copy.select(channel="*N")

        inicio_maximo = max(
            canal_sac_Z[0].stats.starttime,
            canal_sac_E[0].stats.starttime,
            canal_sac_N[0].stats.starttime,
        )
        finales = np.array(
            [
                canal_sac_Z[0].stats.endtime,
                canal_sac_E[0].stats.endtime,
                canal_sac_N[0].stats.endtime,
            ]
        )
        fin_minimo = np.argmin(finales)
        tiempo_inicio = canal_sac_Z[0].stats.starttime
        diferencia_tiempos = inicio_maximo - tiempo_inicio
        tiempo_fin = finales[fin_minimo]
        ###

        ### Downsampling a 40Hz
        fs = int(canal_sac_Z[0].stats.sampling_rate)
        fs_real = int(canal_sac_Z[0].stats.sampling_rate)
        if fs == 100:
            canal_sac_Z[0].data = sp.signal.resample(
                canal_sac_Z[0].data, int(len(canal_sac_Z[0].data) * 40 / 100)
            )
            canal_sac_Z[0].stats.sampling_rate = 40
            canal_sac_Z = canal_sac_Z.slice(inicio_maximo, tiempo_fin)
            canal_sac_E[0].data = sp.signal.resample(
                canal_sac_E[0].data, int(len(canal_sac_E[0].data) * 40 / 100)
            )
            canal_sac_E[0].stats.sampling_rate = 40
            canal_sac_E = canal_sac_E.slice(inicio_maximo, tiempo_fin)
            canal_sac_N[0].data = sp.signal.resample(
                canal_sac_N[0].data, int(len(canal_sac_N[0].data) * 40 / 100)
            )
            canal_sac_N[0].stats.sampling_rate = 40
            canal_sac_N = canal_sac_N.slice(inicio_maximo, tiempo_fin)
            #frame_p = int(frame_p * 40 / fs) if fix_p else frame_p
            fs = 40
        elif fs == 40:
            canal_sac_Z = canal_sac_Z.slice(inicio_maximo, tiempo_fin)
            canal_sac_E = canal_sac_E.slice(inicio_maximo, tiempo_fin)
            canal_sac_N = canal_sac_N.slice(inicio_maximo, tiempo_fin)
        elif fs != 40:
            print("Hay sampling rate distinto a 40, revisar!!")
        ###
        frame_p = int((frame_p-tiempo_inicio) * fs)  
        frame_p -= int(diferencia_tiempos * 40)

        ### remover resp instrumental
        sta = canal_sac_Z[0].stats.station
        cha = canal_sac_Z[0].stats.channel

        canal_sac_Z.remove_response(inventory=inv, output=self.response)
        canal_sac_E.remove_response(inventory=inv, output=self.response)
        canal_sac_N.remove_response(inventory=inv, output=self.response)
        # except:
        # print('Problema con el .xml, no se puede remover resp. instrumental')
        ###

        ### Filtrado Pasabanda
        self.filter_builder(canal_sac_Z, self.filter[0], self.filter[1])
        self.filter_builder(canal_sac_E, self.filter[0], self.filter[1])
        self.filter_builder(canal_sac_N, self.filter[0], self.filter[1])

        data_z = canal_sac_Z[0].data
        data_e = canal_sac_E[0].data
        data_n = canal_sac_N[0].data

        data_z = data_z[
            int(frame_p - fs * self.nro_segundos_ventana_izq) : int(
                frame_p + fs * self.nro_segundos_ventana_der
            )
        ]
        data_e = data_e[
            int(frame_p - fs * self.nro_segundos_ventana_izq) : int(
                frame_p + fs * self.nro_segundos_ventana_der
            )
        ]
        data_n = data_n[
            int(frame_p - fs * self.nro_segundos_ventana_izq) : int(
                frame_p + fs * self.nro_segundos_ventana_der
            )
        ]

        feat_por_evento_temporal, feat_por_evento_global = [], []

        ### Caracteristicas temporales para CNN
        feat_canales_temporal = np.squeeze(
            np.dstack(np.array([data_z, data_e, data_n]))
        )
        feat_por_evento_temporal.append(feat_canales_temporal)

        if one_hot_encoding:
            # one hot encoding
            encoding = np.squeeze(
                self.enc.transform(np.array(sta).reshape(-1, 1)).toarray()
            )
            feat_por_evento_global.append(encoding)
        else:
            # STATION INDEX
            features_canales_global = []
            features_canales_global.append(np.hstack(([self.id_estacion[sta]])))
            feat_canales_global = np.hstack(features_canales_global)

            feat_por_evento_global.append(feat_canales_global)

        return feat_por_evento_temporal, feat_por_evento_global
