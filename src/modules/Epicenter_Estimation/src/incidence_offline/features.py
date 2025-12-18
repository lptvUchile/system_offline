import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import scipy as sp
from src.utils import (
    downsample_to40hz,
    parametrizador,
    parametrizador2,
    E3,
    up_level_idx,
    down_level_idx,
    E2,
    p_vector_extractor
)
import math
import time

Coordenadas_estaciones ={
    'PB09':[-21.7964, -69.2419, 1.530],'PB06':[-22.7058, -69.5719, 1.440],'AC02':[-26.8355,-69.1291,3.980],
    'CO02':[-31.2037, -71.0003, 1.190],'PB14':[-24.6260, -70.4038, 2.630],'CO01':[-29.9773,-70.0939,2.157],
    'GO01':[-19.6685,-69.1942,3.809],'GO03':[-27.5937,-70.2347,0.730],'PB18':[-17.5895,-69.480, 4.155],
    'MT16':[-33.4285,-70.5234,0.780],'AC04':[-28.2046,-71.0739,0.228],'AC05':[-28.8364,-70.2738,1.227],
    'AP01':[-18.3708,-70.342,0.031],'CO03':[-30.8389,-70.6891,1.003],'GO04':[-30.1727,-70.7993,2.076],
    'HMBCX':[-20.2782,-69.8879,1.152],'MNMCX':[-19.1311,-69.5955,2.304],'MT02':[-33.2591,-71.1377,0.323],
    'MT03':[-33.4936,-70.5102,1.087],'MT05':[-33.3919,-70.7381,0.765],'PATCX':[-20.8207,-70.1529,0.832],
    'PB01':[-21.0432,-69.4874,0.9],'PB02':[-21.3197,-69.896,1.015],'PB03':[-22.0485,-69.7531,1.46],
    'PB04':[-22.3337,-70.1492,1.52],'PB05':[-22.8528,-70.2024,1.15],'PB07':[-21.7267,-69.8862,1.57],
    'PB10':[-23.5134,-70.5541,0.25],'PB11':[-19.761,-69.6558,1.4],'PB12':[-18.6141,-70.3281,0.908],
    'PB15':[-23.2083,-69.4709,1.83],'PSGCX':[-19.5972,-70.1231,0.966],'TA01':[-20.5656,-70.1807,0.075],
    'TA02':[-20.2705,-70.1311,0.0865],'VA03':[-32.7637,-70.5508,1.053], 'GO02':[-25.1626,-69.5904,2.550],
    'GO05':[-35.0099,-71.9303,0.488], 'PB16':[-18.3351,-69.5077,4.480], 'PB08':[-20.1411,-69.1534,3.060],
    'CO04':[-32.0433,-70.9747,2.401],'VA01':[-33.0228,-71.6475,0.0756],'AC01':[-26.1479,-70.5987,0.346],
    'CO05':[-29.9186,-71.2384,0.101],'CO06':[-30.6738,-71.6350,0.2466], 'VA06':[-32.5612,-71.2977,0.080],
    'CO10':[-29.2406,-71.4609,0.035], 'BO03': [-34.4961,-71.9612,0.128],'AC07':[-27.1297,-70.8602,0.072],
    'PX06':[-23.5115,-70.2495,0.700]}

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
        _setup_encoder: 
            Configura el codificador OneHotEncoder para las estaciones.
        correlative_index: 
            Crea un índice correlativo para las estaciones sísmicas.
        get_features: 
            Extrae las características de la señal sísmica, tanto temporales como globales, basadas en los parámetros 
            definidos al crear la instancia de la clase.
    """
    def __init__(self, response="VEL", 
                frame_length=4, 
                frame_shift=2, 
                umbral_corte=0.03,
                window_limit=[20, 120], 
                filter=['highpass', 0.1], 
                path_st = './data/set/stations_index.csv'):
        self.response = response
        self.segundos_previo_P = window_limit[0]
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.segundos_post_p = window_limit[1] 
        self.umbral_corte = umbral_corte  
        self.filter = filter
        self.path_st = path_st
        self._setup_encoder()
        self.correlative_index()
    
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

    def _setup_encoder(self):
        """
        Configura el codificador OneHotEncoder para las estaciones sísmicas.

        Este método inicializa el codificador para realizar una codificación one-hot de las estaciones sísmicas 
        basándose en los códigos de estación predefinidos.
        """
        self.enc = OneHotEncoder()
        id_estacion = np.array(['PB09','PB06','AC02','CO02','PB14','CO01','GO01','GO03', 'MT16', 'PB18','AC04','AC05',
                                'AP01','CO03','GO04','HMBCX','MNMCX','MT02','MT03','MT05','PATCX','PB01','PB02','PB03',
                                'PB04','PB05','PB07','PB10','PB11','PB12','PB15','PSGCX','TA01','TA02','VA03',
                                'GO02','GO05','PB16','PB08','CO04','VA01','AC01','CO05','CO06','VA06','CO10',
                                'BO03','AC07','PX06']).reshape(-1,1)

        self.enc.fit(id_estacion)

    def correlative_index(self):
        """
        Crea un índice correlativo de estaciones basado en el archivo CSV de estaciones.

        Este método lee un archivo CSV con los nombres de las estaciones y asigna un índice numérico único a cada estación.
        """
        ch_station = pd.read_csv(
                    self.path_st,
                    delimiter=",", 
                    index_col=False)
        ch_station = ch_station['station'].values
        nro_station = np.arange(0, len(ch_station))
        self.id_estacion = dict(zip(ch_station, nro_station))

    def get_features(self, 
                    trace, 
                    frame_p, 
                    inv, 
                    one_hot_encoding = False,
                    fix_p = False, 
                    use_cnn = False,
                    include_index = False,
                    include_env = False,
                    include_lat_lon=False,
                    norm_energy = False,
                    concat_features = False,
                    square_fft = False,
                    log_scale=False,
                    include_p_vector=False,
                    n_energy_frames = 0,
                    how_to_include_p = 0,
                    use_module_p = False,
                    hv_duration = 1):
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
        feat_por_evento_temporal : list of numpy array
            Las características temporales extraídas de la señal.
        feat_por_evento_mlp : list of numpy array
            Las características globales extraídas de la señal.
        """
        p_starttime = frame_p
        trace_copy = trace.copy()
        trace_ccopy = trace.copy()
        canal_sac_Z = trace_copy.select(channel='*Z')
        canal_sac_E = trace_copy.select(channel='*E')
        canal_sac_N = trace_copy.select(channel='*N')

        canal_sac_Z_hv = trace_ccopy.select(channel="*Z")
        canal_sac_E_hv = trace_ccopy.select(channel='*E')
        canal_sac_N_hv = trace_ccopy.select(channel='*N')

        inicio_maximo = max(canal_sac_Z[0].stats.starttime,canal_sac_E[0].stats.starttime,canal_sac_N[0].stats.starttime)
        finales = np.array([canal_sac_Z[0].stats.endtime,canal_sac_E[0].stats.endtime,canal_sac_N[0].stats.endtime])
        fin_minimo = np.argmin(finales)
        tiempo_inicio = canal_sac_Z[0].stats.starttime
        diferencia_tiempos = inicio_maximo - tiempo_inicio
        tiempo_fin = finales[fin_minimo]
        
        ###
        
        fs = int(canal_sac_Z[0].stats.sampling_rate)
        fs_real = int(canal_sac_Z[0].stats.sampling_rate)
        if fs !=40:
            print("APLICANDO DOWNSAMPLING")
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
        #elif fs != 40:
        #    print("Hay sampling rate distinto a 40, revisar!!")
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
        
        frame_len = self.frame_length*fs
        frame_shi = self.frame_shift*fs
        nfft = pow(2,math.ceil(np.log2(frame_len)))
        
        data_z = canal_sac_Z[0].data*10**10
        data_e = canal_sac_E[0].data*10**10
        data_n = canal_sac_N[0].data*10**10

        start_count = abs(frame_p-self.segundos_previo_P*fs)

        if self.umbral_corte is not 0:
            print("CORTANDO POR UMBRAL DE ENERGIA: ", self.umbral_corte)

            umbral_corte = self.umbral_corte
            Energia_Z_ref = E3(data_z, frame_len,frame_shi,escala = 'logaritmica')
            # arg_amp_maxima = np.argmax(Energia_Z_ref[1:]) #Assumption: The maximun energy is in S-wave
            # arg_amp_minima = np.argmin(Energia_Z_ref[:arg_amp_maxima]) # take the minimum energy between the start of the signal and the S-wave
            # delta_energia = Energia_Z_ref[arg_amp_maxima]-Energia_Z_ref[arg_amp_minima] 
            # energia_umbral_corte = delta_energia*umbral_corte+Energia_Z_ref[arg_amp_minima] #energy threshold
            # arg_fin_nueva_coda = arg_amp_maxima + np.argmin(np.abs(Energia_Z_ref[arg_amp_maxima:]-energia_umbral_corte))                      
            # muestra_corte_coda = int(fs*frame_len*arg_fin_nueva_coda/frame_shi)   
            
            arg_amp_maxima = np.argmax(Energia_Z_ref[1:]) + 1 #Assumption: The maximun energy is in S-wave
            arg_amp_minima = np.argmin(Energia_Z_ref[:arg_amp_maxima]) # take the minimum energy between the start of the signal and the S-wave
            delta_energia = Energia_Z_ref[arg_amp_maxima] - Energia_Z_ref[arg_amp_minima] 
            energia_umbral_corte = delta_energia * umbral_corte + Energia_Z_ref[arg_amp_minima] #energy threshold

            energy_condition = np.where(Energia_Z_ref < energia_umbral_corte)[0]
            energy_condition = energy_condition[energy_condition > arg_amp_maxima]
            if len(energy_condition) > 0:
                muestra_corte_coda = int(fs * frame_len * energy_condition[0] / frame_shi)
            else:
                muestra_corte_coda = len(data_z)

            print("MUESTRA CORTE CODA: ", muestra_corte_coda)  
        else:
            print("CORTANDO POR VENTANA FIJA    : {}, {}".format(self.segundos_previo_P, self.segundos_post_p))
            data_z = data_z[start_count: start_count + 40*self.segundos_post_p]
            data_e = data_e[start_count: start_count + 40*self.segundos_post_p]
            data_n = data_n[start_count: start_count + 40*self.segundos_post_p]
            muestra_corte_coda = len(data_z)
            print("MUESTRA CORTE CODA: ", muestra_corte_coda)  


        #asegurarse que recorte de coda deja cantidad suficiente de muestras para calcular características%.
        data_z = data_z[:muestra_corte_coda] #A la traza se le corta la coda
        data_e = data_e[:muestra_corte_coda]
        data_n = data_n[:muestra_corte_coda]
        

        if use_cnn:
            feat_por_evento_temporal = []
            ### Caracteristicas temporales para CNN
            feat_canales_temporal = np.squeeze(np.dstack(np.array([data_z, data_e, data_n])))
            feat_por_evento_temporal.append(feat_canales_temporal)
        
        else:
            print('EN parametrizador')
            if log_scale:
                print('Using log scale')
                if square_fft:
                    print('Using square at ffts')
                    feat_k_z = parametrizador2(data_z, frame_len, frame_shi,nfft, escala = 'logaritmica')
                    feat_k_e = parametrizador2(data_e, frame_len, frame_shi,nfft, escala = 'logaritmica')
                    feat_k_n = parametrizador2(data_n, frame_len, frame_shi,nfft, escala = 'logaritmica')
                
                else:
                    print('Using no squares at ffts')
                    #A la traza enventanadada se le obtiene abs(FFT//2), puede ir con o sin logaritmo
                    feat_k_z = parametrizador(data_z, frame_len, frame_shi,nfft, escala = 'logaritmica')
                    feat_k_e = parametrizador(data_e, frame_len, frame_shi,nfft, escala = 'logaritmica')
                    feat_k_n = parametrizador(data_n, frame_len, frame_shi,nfft, escala = 'logaritmica')
                
                #print(feat_k_z.shape)

                #Distintos tipos de energia por ventana
                if norm_energy:
                    feat_Energy_z = E3(data_z, frame_len,frame_shi,escala = 'logaritmica')
                    feat_k_z = np.hstack((feat_k_z, np.array([feat_Energy_z]).T))
                    feat_Energy_e = E3(data_e, frame_len,frame_shi,escala = 'logaritmica')
                    feat_k_e = np.hstack((feat_k_e, np.array([feat_Energy_e]).T))
                    feat_Energy_n = E3(data_n, frame_len,frame_shi,escala = 'logaritmica')
                    feat_k_n = np.hstack((feat_k_n, np.array([feat_Energy_n]).T))
                else:
                    feat_Energy_z = E2(data_z, frame_len,frame_shi,escala = 'logaritmica')
                    feat_k_z = np.hstack((feat_k_z, np.array([feat_Energy_z]).T))
                    feat_Energy_e = E2(data_e, frame_len,frame_shi,escala = 'logaritmica')
                    feat_k_e = np.hstack((feat_k_e, np.array([feat_Energy_e]).T))
                    feat_Energy_n = E2(data_n, frame_len,frame_shi,escala = 'logaritmica')
                    feat_k_n = np.hstack((feat_k_n, np.array([feat_Energy_n]).T))

            else:
                print('Using linear scale')
                if square_fft:
                    print('Using square at ffts')
                    feat_k_z = parametrizador2(data_z, frame_len, frame_shi,nfft, escala = 'lineal')
                    feat_k_e = parametrizador2(data_e, frame_len, frame_shi,nfft, escala = 'lineal')
                    feat_k_n = parametrizador2(data_n, frame_len, frame_shi,nfft, escala = 'lineal')
                
                else:
                    print('Using no squares at ffts')
                    #A la traza enventanadada se le obtiene abs(FFT//2), puede ir con o sin logaritmo
                    feat_k_z = parametrizador(data_z, frame_len, frame_shi,nfft, escala = 'lineal')
                    feat_k_e = parametrizador(data_e, frame_len, frame_shi,nfft, escala = 'lineal')
                    feat_k_n = parametrizador(data_n, frame_len, frame_shi,nfft, escala = 'lineal')
                
                #print(feat_k_z.shape)

                #Distintos tipos de energia por ventana
                if norm_energy:
                    feat_Energy_z = E3(data_z, frame_len,frame_shi,escala = 'lineal')
                    feat_k_z = np.hstack((feat_k_z, np.array([feat_Energy_z]).T))
                    feat_Energy_e = E3(data_e, frame_len,frame_shi,escala = 'lineal')
                    feat_k_e = np.hstack((feat_k_e, np.array([feat_Energy_e]).T))
                    feat_Energy_n = E3(data_n, frame_len,frame_shi,escala = 'lineal')
                    feat_k_n = np.hstack((feat_k_n, np.array([feat_Energy_n]).T))
                else:
                    feat_Energy_z = E2(data_z, frame_len,frame_shi,escala = 'lineal')
                    feat_k_z = np.hstack((feat_k_z, np.array([feat_Energy_z]).T))
                    feat_Energy_e = E2(data_e, frame_len,frame_shi,escala = 'lineal')
                    feat_k_e = np.hstack((feat_k_e, np.array([feat_Energy_e]).T))
                    feat_Energy_n = E2(data_n, frame_len,frame_shi,escala = 'lineal')
                    feat_k_n = np.hstack((feat_k_n, np.array([feat_Energy_n]).T))

            #Features temporales de la traza de un canal
            if concat_features:
                feat_canales_temporal = np.concatenate((feat_k_z,feat_k_e,feat_k_n),axis=1)
            else:
                #TODO: CAMBIAR A PROMEDIO
                #feat_canales_temporal=np.sqrt((feat_k_e**2+feat_k_n**2))/(feat_k_z+1e-12)
                feat_canales_temporal = (feat_k_e+feat_k_n+feat_k_z)/3
                
            feat_por_evento_temporal = []
            feat_por_evento_temporal.append(feat_canales_temporal)
            

        ### Caracteristicas globales
            
        feat_por_evento_mlp = []
        if  include_p_vector:    # DESDE EL 9no FRAME 4,6,8,10,12,14,16,18,20,22,24 [20,24]
            last_frame = 10+n_energy_frames
            #feat_k_z: 59,130 (ffts+energia)
            z_chann, e_chann,n_chann = feat_k_z[10:last_frame,:], feat_k_e[10:last_frame,:], feat_k_n[10:last_frame,:] #shapes (n_energy_frames,130)
            if use_module_p:
                print("Using p-Vector Module")
                p_vector = np.sqrt(e_chann**2+n_chann**2)/(z_chann+1e-12)
            else:
                print("Using p-Vector Mean")
                p_vector = (e_chann+n_chann)/(2*(z_chann+1e-12))
            p_vector = np.sum(p_vector,axis=0)/n_energy_frames #shape (1,130)
            p_vector = p_vector.reshape(-1).tolist()
            if how_to_include_p ==0:    # GLOBALES 130
                feat_por_evento_mlp += p_vector
            elif how_to_include_p==1:   # NUEVA VENTANA 60,130
                feat_por_evento_temporal[0] = np.concatenate((feat_por_evento_temporal[0],np.array(p_vector).reshape(1,-1)),axis=0)
            elif how_to_include_p==2:   #ENTRA AL INICIO USANDO PADDING 59,260
                l,w = np.shape(feat_por_evento_temporal[0])
                padded_entry = np.concatenate((np.array(p_vector).reshape(1,-1),np.zeros((l-1,w))),axis=0)

                feat_por_evento_temporal[0] = np.concatenate((feat_por_evento_temporal[0],padded_entry),axis=1)
            elif how_to_include_p==3:   #ENTRA ENERGIA MEDIA
                feat_por_evento_mlp += [p_vector[-1]]
            elif how_to_include_p==4:
                hv = p_vector_extractor(z_chann=canal_sac_Z_hv,
                                        e_chann=canal_sac_E_hv,
                                        n_chann=canal_sac_N_hv,
                                        p_starttime =p_starttime,
                                        inventory=inv,
                                        dur_sec=hv_duration, 
                                        start_sec=0, 
                                        low_coff=4, 
                                        high_coff=12,
                                        use_module=True)
                feat_por_evento_mlp+= [hv]
            elif how_to_include_p ==5:
                hv = p_vector_extractor(z_chann=canal_sac_Z_hv,
                                        e_chann=canal_sac_E_hv,
                                        n_chann=canal_sac_N_hv,
                                        p_starttime =p_starttime,
                                        inventory=inv,
                                        dur_sec=hv_duration, 
                                        start_sec=0, 
                                        low_coff=4, 
                                        high_coff=12,
                                        use_module=False)
                feat_por_evento_mlp+= [hv]
            
        if include_lat_lon: 
            info_lat_lon = Coordenadas_estaciones[station]
            feat_por_evento_mlp+=info_lat_lon

        if include_env:
            envC = (up_level_idx(data_z, 200, 50, 10)/fs)/120
            envC_ = (down_level_idx(data_z, 200, 50, 10)/fs)/120
            for feat in envC:
                feat_por_evento_mlp.append(feat)
            for feat_ in envC_:
                feat_por_evento_mlp.append(feat_)

        if include_index:
            if one_hot_encoding:
                encoding = np.squeeze(self.enc.transform(np.array(station).reshape(-1,1)).toarray())
                feat_por_evento_mlp.append(encoding)

            else:
                # STATION INDEX
                id_estacion = self.id_estacion[station]
                feat_por_evento_mlp.append(id_estacion)

        feat_por_evento_mlp = [np.array(feat_por_evento_mlp).reshape(-1)]
        print(f"SHAPE FEAT TEMPORAL: {feat_por_evento_temporal[0].shape}")
        print(f"LARGO DE FEAT_GLOBAL: {len(feat_por_evento_mlp[0])}")

        return feat_por_evento_temporal, feat_por_evento_mlp
