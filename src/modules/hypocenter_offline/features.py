import numpy as np
from .utils import (
    downsample_to40hz,
    parametrizador,
    parametrizador2,
    E3,
    E2)
import math

#np.seterr(divide='raise', invalid='raise')

class FeatureExtractor:
    """
    Extractor de caracteristicas para una señal sismica.

    Esta clase se encarga de extraer diversas características de una señal sísmica para su posterior análisis, 
    como características temporales y globales. Las características son extraídas de las componentes de movimiento 
    sísmico (E, N, Z), y se pueden procesar de diferentes maneras según los parámetros configurados en la clase.

    Parameters
    ----------
        response:   str 
            Señal que se utiliza, puede ser "VEL" o "ACC"     
        frame_length:   int 
            Segundos que contiene un frame
        frame_shift:    int 
            Segundos de separación entre frames
        umbral_corte:   float
            Porcentaje de la energia maxima alcanzada que se utiliza para cortar la señal 
        window_limit:   list of int 
            Lista de largo 2. El primer valor corresponde a los segundos antes del frame p
            que se abordan y el segundo los segundos despues.
        filter: list of objects 
            Lista de largo 2 o 3. El primer valor corresponde al filtro que se utiliza (str) y puede ser 
            "highpass" o "bandpass". El segundo valor corresponde a la frecuencia minima (float) y el tercero la frecuencia máxima
            (solo se incluye si se usa bandpass) 


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
        correlative_index: 
            Crea un índice correlativo para las estaciones sísmicas.
        get_features: 
            Extrae las características de la señal sísmica, tanto temporales como globales, basadas en los parámetros 
            definidos al crear la instancia de la clase.
    """
    def __init__(self, 
                response="VEL", 
                frame_length=4, 
                frame_shift=2, 
                window_limit=[20, 120], 
                filter=['highpass', 0.1],
                norm_energy = False,
                square_fft = False,
                log_scale=False):
        
        self.response = response
        self.segundos_previo_P = window_limit[0]
        self.segundos_post_p = window_limit[1] 
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter = filter
        self.norm_energy = norm_energy
        self.square_fft = square_fft
        self.log_scale = log_scale
    
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
            #print(f"Applying bandpass filter from {options[0]} to {options[1]} Hz")
            trace.filter(filter, freqmin=options[0], freqmax=options[1], corners=2, zerophase=True)
        else:
            #print(f"Applying {filter} filter with frequency {options} Hz")
            trace.filter(filter, freq=options, corners=2, zerophase=True)


    # Se modifica para que entregue (C,T,N)

    def get_features(self, 
                    trace, 
                    frame_p, 
                    inv):
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
        one_hot_encoding : bool, opcional
            Si se debe usar codificación one-hot para las estaciones, por defecto es False.
        fix_p : bool, opcional
            Si se debe fijar el tiempo de la onda P, por defecto es False.
        use_cnn : bool, opcional
            Si se deben usar características para redes neuronales convolucionales (CNN), por defecto es False.
        include_index : bool, opcional
            Si se debe incluir el índice de la estación en las características globales, por defecto es False.
        include_env : bool, opcional
            Si se debe incluir la envolvente, por defecto es False.
        include_lat_lon : bool, opcional
            Si se deben incluir las coordenadas latitud y longitud de la estación, por defecto es False.
        norm_energy : bool, opcional
            Si se deben normalizar las características de energía, por defecto es False.
        concat_features : bool, opcional
            Si se deben concatenar las características de los diferentes canales (E, N, Z), por defecto es False (se usa promedio).
        include_p_vector: bool, opcional
            Si se debe incluir un vector de energía para una cantidad limitada de frames después de la p, por defecto es False.
        Returns
        -------
        feat_por_evento_temporal : numpy array (C,T,N)
            Las características temporales extraídas de la señal.
        feat_por_evento_mlp : list of numpy array
            Las características globales extraídas de la señal.
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
        
        if fs !=40:
            #print("APLICANDO DOWNSAMPLING")
            canal_sac_Z = downsample_to40hz(canal_sac_Z,fs)
            canal_sac_E = downsample_to40hz(canal_sac_E,fs)
            canal_sac_N = downsample_to40hz(canal_sac_N,fs)
            canal_sac_Z = canal_sac_Z.slice(inicio_maximo, tiempo_fin)
            canal_sac_E = canal_sac_E.slice(inicio_maximo, tiempo_fin)
            canal_sac_N = canal_sac_N.slice(inicio_maximo, tiempo_fin)
            #frame_p = int(frame_p * 40 / fs) if fix_p else frame_p
            fs = 40
        elif fs == 40:
            canal_sac_Z = canal_sac_Z.slice(inicio_maximo, tiempo_fin)
            canal_sac_E = canal_sac_E.slice(inicio_maximo, tiempo_fin)
            canal_sac_N = canal_sac_N.slice(inicio_maximo, tiempo_fin)

        frame_p = int((frame_p - tiempo_inicio)*fs)
        frame_p-=int(diferencia_tiempos*40)

        
        canal_sac_Z.remove_response(inventory=inv, output=self.response)
        canal_sac_E.remove_response(inventory=inv, output=self.response)
        canal_sac_N.remove_response(inventory=inv, output=self.response)

        self.filter_builder(canal_sac_Z, self.filter[0], self.filter[1])
        self.filter_builder(canal_sac_E, self.filter[0], self.filter[1])
        self.filter_builder(canal_sac_N, self.filter[0], self.filter[1])
        
        frame_len = self.frame_length*fs
        frame_shi = self.frame_shift*fs
        nfft = pow(2,math.ceil(np.log2(frame_len)))
        
        data_z = canal_sac_Z[0]
        data_e = canal_sac_E[0]
        data_n = canal_sac_N[0]

        start_count = abs(frame_p-self.segundos_previo_P*fs)


        #print("CORTANDO POR VENTANA FIJA    : {}, {}".for|(self.segundos_previo_P, self.segundos_post_p))
        data_z = data_z[start_count: start_count + 40*self.segundos_post_p]
        data_e = data_e[start_count: start_count + 40*self.segundos_post_p]
        data_n = data_n[start_count: start_count + 40*self.segundos_post_p]
        muestra_corte_coda = len(data_z)
        #print("MUESTRA CORTE CODA: ", muestra_corte_coda)  


        #asegurarse que recorte de coda deja cantidad suficiente de muestras para calcular características%.
        data_z = data_z[:muestra_corte_coda] #A la traza se le corta la coda
        data_e = data_e[:muestra_corte_coda]
        data_n = data_n[:muestra_corte_coda]
        
   
        scale = 'logaritmica' if self.log_scale else 'lineal'
        
        #print(f'Using {scale} scale')
        if self.square_fft:
            feat_k_z = parametrizador2(data_z, frame_len, frame_shi,nfft, escala = scale)
            feat_k_e = parametrizador2(data_e, frame_len, frame_shi,nfft, escala = scale)
            feat_k_n = parametrizador2(data_n, frame_len, frame_shi,nfft, escala = scale)
        
        else:
            feat_k_z = parametrizador(data_z, frame_len, frame_shi,nfft, escala = scale)
            feat_k_e = parametrizador(data_e, frame_len, frame_shi,nfft, escala = scale)
            feat_k_n = parametrizador(data_n, frame_len, frame_shi,nfft, escala = scale) 
                
        #Distintos tipos de energia por ventana
        if self.norm_energy:
            feat_Energy_z = E3(data_z, frame_len,frame_shi,escala = scale)
            feat_k_z = np.hstack((feat_k_z, np.array([feat_Energy_z]).T))
            feat_Energy_e = E3(data_e, frame_len,frame_shi,escala = scale)
            feat_k_e = np.hstack((feat_k_e, np.array([feat_Energy_e]).T))
            feat_Energy_n = E3(data_n, frame_len,frame_shi,escala = scale)
            feat_k_n = np.hstack((feat_k_n, np.array([feat_Energy_n]).T))
        else:
            
            feat_Energy_z = E2(feat_k_z,escala = scale)
            feat_k_z = np.hstack((feat_k_z, np.array([feat_Energy_z]).T))
            feat_Energy_e = E2(feat_k_e,escala = scale)
            feat_k_e = np.hstack((feat_k_e, np.array([feat_Energy_e]).T))
            feat_Energy_n = E2(feat_k_n,escala = scale)
            feat_k_n = np.hstack((feat_k_n, np.array([feat_Energy_n]).T))

        
        feat_canales_temporal = np.stack([feat_k_z, feat_k_e, feat_k_n], axis=0).astype(np.float32)  # (C,T,N)
        feat_canales_temporal = feat_canales_temporal / np.max(np.abs(feat_canales_temporal), axis=0) #Normalizar feat_canales_temporal
        feat_canales_temporal = feat_canales_temporal.transpose(0, 2, 1)  # (T,N,C)

                
        feat_por_evento_temporal = []
        feat_por_evento_temporal.append(feat_canales_temporal)
            
        return feat_por_evento_temporal