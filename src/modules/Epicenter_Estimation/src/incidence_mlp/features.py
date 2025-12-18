

import numpy as np

import scipy as sp
from src.utils import (
    downsample_to40hz,
)


class FeatureExtractor():
    """
    Extractor de caracteristicas para una señal sismica.

    Esta clase se encarga de extraer diversas características de una señal sísmica para su posterior análisis, 
    como características temporales y globales. Las características son extraídas de las componentes de movimiento 
    sísmico (E, N, Z), y se pueden procesar de diferentes maneras según los parámetros configurados en la clase.

    Parameters
    ----------
        response:   str 
            Señal que se utiliza, puede ser "VEL" o "ACC"     
        window_limit:   list of int 
            Lista de largo 2. El primer valor corresponde a los segundos antes del frame p
            que se abordan y el segundo los segundos despues.
        filter: list of objects 
            Lista de largo 2 o 3. El primer valor corresponde al filtro que se utiliza (str) y puede ser 
            "highpass" o "bandpass". El segundo valor corresponde a la frecuencia minima (float) y el tercero la frecuencia máxima
            (solo se incluye si se usa bandpass) 
        path_st:    str 
            Path de un archivo .csv que contenga una columna "station" con los nombres de las estaciones. 

    Attributes
    ----------
        segundos_previo_P:  int
            Segundos de recorte previos al comienzo de onda P.
        segundos_post_p:    int
            Segundos de recorte posteriores al comienzo de onda P.

    Methods
    -------
        filter_builder: 
            Aplica un filtro a la señal.
        get_features: 
            Extrae las características de la señal sísmica, tanto temporales como globales, basadas en los parámetros 
            definidos al crear la instancia de la clase.
    """
    def __init__(self, response="VEL", 
                umbral_corte=0.03,
                window_limit=[20, 120], 
                filter=['highpass', 0.1], ):
        self.response = response
        self.segundos_previo_P = window_limit[0]
        self.segundos_post_p = window_limit[1] 
        self.umbral_corte = umbral_corte  
        self.filter = filter

    
    def filter_builder(self, trace, filter, options):
        """
        Aplica un filtro de tipo bandpass o highpass a la señal.

        Parameters
        ----------
        trace : obspy Trace
            La señal a la cual se le aplicará el filtro.
        filter : str
            El tipo de filtro a aplicar ("highpass" o "bandpass").
        options : list of float
            Frecuencias del filtro. Si es "bandpass", debe contener dos valores (frecuencia mínima y máxima).
        """

        if filter == "bandpass":
            print(f"Applying bandpass filter from {options[0]} to {options[1]} Hz")
            trace.filter(filter, freqmin=options[0], freqmax=options[1], corners=2, zerophase=True)
        else:
            print(f"Applying {filter} filter with frequency {options} Hz")
            trace.filter(filter, freq=options, corners=2, zerophase=True)


    def get_features(self, 
                    trace, 
                    frame_p, 
                    inv, 
                    use_horizontal_mean = False,):
        """
        Extrae las características de la señal sísmica en base a los parámetros de la clase.

        Parameters
        ----------
        trace : obspy Trace
            La señal sísmica de la cual se extraerán las características.
        frame_p : float
            El tiempo de la onda P para la señal.
        inv : obspy Inventory
            El inventario de las estaciones sísmicas.
        Returns
        -------
        feat_por_evento_mlp : list of numpy array
            Las características  extraídas de la señal.
        """

        trace_copy = trace.copy()
        canal_sac_Z = trace_copy.select(channel='*Z')
        canal_sac_E = trace_copy.select(channel='*E')
        canal_sac_N = trace_copy.select(channel='*N')

        inicio_maximo = max(canal_sac_Z[0].stats.starttime,canal_sac_E[0].stats.starttime,canal_sac_N[0].stats.starttime)
        finales = np.array([canal_sac_Z[0].stats.endtime,canal_sac_E[0].stats.endtime,canal_sac_N[0].stats.endtime])
        fin_minimo = np.argmin(finales)
        tiempo_inicio = canal_sac_Z[0].stats.starttime
        diferencia_tiempos = inicio_maximo - tiempo_inicio
        tiempo_fin = finales[fin_minimo]
        
        ###
        
        fs = int(canal_sac_Z[0].stats.sampling_rate)
        fs_real = int(canal_sac_Z[0].stats.sampling_rate)
        if fs == 100:
            print("APLICANDO DOWNSAMPLING")
            canal_sac_Z = downsample_to40hz(canal_sac_Z)
            canal_sac_E = downsample_to40hz(canal_sac_E)
            canal_sac_N = downsample_to40hz(canal_sac_N)
            canal_sac_Z = canal_sac_Z.slice(inicio_maximo, tiempo_fin)
            canal_sac_E = canal_sac_E.slice(inicio_maximo, tiempo_fin)
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
        #p_starttime = trace.select(channel="*Z")[0].stats.starttime + row["frame_p"]/40
        frame_p = int((frame_p - tiempo_inicio)*fs)
        frame_p-=int(diferencia_tiempos*40)
        #print("FS FINAL: ", fs)
        #print("FRAME P: ", frame_p)
        ### remover resp instrumental
        station = canal_sac_Z[0].stats.station
        cha = canal_sac_Z[0].stats.channel
        
        canal_sac_Z.remove_response(inventory=inv, output=self.response)
        canal_sac_E.remove_response(inventory=inv, output=self.response)
        canal_sac_N.remove_response(inventory=inv, output=self.response)

        self.filter_builder(canal_sac_Z, self.filter[0], self.filter[1])
        self.filter_builder(canal_sac_E, self.filter[0], self.filter[1])
        self.filter_builder(canal_sac_N, self.filter[0], self.filter[1])
        
        start_count = abs(frame_p-self.segundos_previo_P*fs)
        end_count = abs(frame_p+self.segundos_post_p*fs)
        
        
        
    
        data_z = canal_sac_Z[0].data
        data_e = canal_sac_E[0].data
        data_n = canal_sac_N[0].data
 

        
        data_z = data_z[start_count: end_count]
        data_e = data_e[start_count: end_count]
        data_n = data_n[start_count: end_count]



        feat_por_evento_mlp = []
        if use_horizontal_mean:
            mean_z = np.mean(data_z)
            mean_e = np.mean(data_e)
            mean_n = np.mean(data_n)
            
            feat_por_evento_mlp = np.hstack((mean_z, mean_e, mean_n))
        else:
            feat_por_evento_mlp = np.hstack((data_z, data_e, data_n)) # 40*4*3 = 480


        

        print(f"SHAPE MLP: {feat_por_evento_mlp.shape}")

        return feat_por_evento_mlp
