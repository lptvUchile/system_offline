from keras.utils import Sequence
import numpy as np
class MyBatchGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=1, shuffle=False):
        'Initialization'
        self.X = X
        # self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        # xb = np.empty((self.batch_size, *self.x[index].shape))
        yb = np.empty((self.batch_size, *self.y[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            # xb[s] = self.x[index]
            yb[s] = self.y[index]

        return Xb, yb

def function_features_extration(sac_conc, inv): 
    sac_k = sac_conc.copy()
    import numpy.matlib
    import os
    import numpy as np
    from obspy import read_inventory
    import math
    import scipy as sp
    import sys
    from .src.utils import utils_magnitude
    
    sys.path.insert(1, "/Magnitude/data/xml/")

    ##Los siguientes parametros no deberian modificarse, ya que fueron los con que se entreno la red
    path_root =  os.getcwd()

    umbral_corte = 0.03 # Threshold, based on the decay of the signal #0.1 is 10% # if is setted in 1 preserves the entire trace
    frame_length = 4 #size in seconds of the frame length
    frame_shift = 2  #frame shift
    Energy = 'E3' # There are different types of energy used 
    Vel = True #True: the traces are converted to velocity, False the traces are kept in count
    escala_features_fft = 'logaritmica' # 'lineal' o 'logaritmica' 
    escala_features_energia = 'logaritmica'
    escala_signal = 1e+10 #Scale to amplify the signal, especially useful when the signal is in velocity
    version = 'original'  #'original', 'fourier_completa' o 'raw'

    
    
    features_canales_temporal,features_canales_global = [], []        
    feat_por_evento_temporal, feat_por_evento_global = [], []
    for ch in range(len(sac_k)):
        sac_k[ch].data = sac_k[ch].data*escala_signal
        estoy_en_Z = 0
        sta = sac_k[ch].stats.channel
        
        fs = int(sac_k[ch].stats.sampling_rate)
        fs_real = int(sac_k[ch].stats.sampling_rate)
        if fs ==100:
            sac_k[ch].data = sp.signal.resample(sac_k[ch].data,int(len(sac_k[ch].data)*40/100))
            sac_k[ch].stats.sampling_rate = 40
            sac_k[ch] = sac_k[ch].slice(sac_k[ch].stats.starttime+1, sac_k[ch].stats.endtime-1)
            fs = 40
        elif fs==40:
            sac_k[ch] = sac_k[ch].slice(sac_k[ch].stats.starttime+1, sac_k[ch].stats.endtime-1)
        elif fs==20:
            sac_k[0].data = sp.signal.resample(sac_k[0].data,int(len(sac_k[0].data)*40/20))
            sac_k[0].stats.sampling_rate = 40
            sac_k = sac_k.slice(sac_k[0].stats.starttime+1, sac_k[0].stats.endtime-1)
            fs = 40
        else:
            print('Hay sampling rate distinto a 40, revisar!!')
        
        frame_len = frame_length*fs
        frame_shi = frame_shift*fs                
        nfft = pow(2,math.ceil(np.log2(frame_len)))
      
        if Vel == True: # Signal are converted to velocity
                
            sta = sac_k[ch].stats.station
            cha = sac_k[ch].stats.channel
            network = sac_k[ch].stats.network

            
            sac_k[ch].remove_response(inventory=inv, output="VEL")
    
        data_k = utils_magnitude.butter_highpass_lfilter(sac_k[ch].data, cutoff=1, fs=fs, order=3)   
    
        if sac_k[ch].stats.channel[-1]=='Z':

            Energia_Z_ref = utils_magnitude.E3(data_k, frame_len,frame_shi,escala = 'lineal')
    
            arg_amp_maxima = np.argmax(Energia_Z_ref) #Assumption: The maximum energy is in S-wave
            if arg_amp_maxima ==0:
                muestra_corte_coda = len(data_k)
            else:
                arg_amp_minima = np.argmin(Energia_Z_ref[:arg_amp_maxima]) # take the minimum energy between the start of the signal and the S-wave
                delta_energia = Energia_Z_ref[arg_amp_maxima]-Energia_Z_ref[arg_amp_minima] 
                energia_umbral_corte = delta_energia*umbral_corte+Energia_Z_ref[arg_amp_minima] #energy threshold
        
                arg_fin_nueva_coda = arg_amp_maxima + np.argmin(np.abs(Energia_Z_ref[arg_amp_maxima:]-energia_umbral_corte))                      
                muestra_corte_coda = int(fs*frame_len*arg_fin_nueva_coda/frame_shi)      
                #data_k_or = data_k
    
        data_k = data_k[:muestra_corte_coda] #The signal is cut       
    
        feat_k = utils_magnitude.parametrizador(data_k, frame_len, frame_shi,nfft, escala = escala_features_fft)
    
        #The signal is windowed
        feat_t = utils_magnitude.get_frames(data_k,frame_len, frame_shi)      
        #FFT
        feat_fourier_completa = np.fft.fft(feat_t, nfft, axis=1)  
        #Real and imaginary part are concatenated
        feat_fourier_completa = np.hstack((feat_fourier_completa.real[:,:feat_fourier_completa.shape[1]//2 +1],
                   feat_fourier_completa.imag[:,:feat_fourier_completa.shape[1]//2 +1]))   
        feat_fourier_completa = np.delete(feat_fourier_completa,[129,257],1)
        #Energy


        feat_Energy = utils_magnitude.E3(data_k, frame_len,frame_shi,escala = escala_features_energia)
        feat_k = np.hstack((feat_k, np.array([feat_Energy]).T))         
            
    

        features_canales_temporal.append(feat_k)

     
        feat_in_test_temporal = np.hstack(features_canales_temporal)   
      
        feat_por_evento_temporal.append(feat_in_test_temporal)
                
    return feat_in_test_temporal
    


class MagnitudeEstimator:
    def __init__(self, model_path_mayores4m, normalization_path_mayores4m, model_path_menores4m, normalization_path_menores4m):
        self.model_path_mayores4m = model_path_mayores4m
        self.normalization_path_mayores4m = normalization_path_mayores4m
        self.model_path_menores4m = model_path_menores4m
        self.normalization_path_menores4m = normalization_path_menores4m

    def DNN_magnitude_todos(self,feat_in_test_temporal):
        from keras.models import load_model
        import numpy as np
        from keras.utils import Sequence
        import os
        import json
        
        global modelMagnitude, dictParametersMagnitude_todos

        try: modelMagnitude, dictParametersMagnitude_todos
        
        except:
            #Path where the model was saved
            path_model = self.model_path_mayores4m
            path_normalization = self.normalization_path_mayores4m
            modelMagnitude = load_model(path_model)
            dictParametersMagnitude_todos = json.load(open(path_normalization, "r", encoding='utf-8'))
        
        #Features are translated into a range between 0 and 1  
        min_f_train_temporal = np.array(dictParametersMagnitude_todos['min_temporal'])
        max_f_train_temporal = np.array(dictParametersMagnitude_todos['max_temporal'])

        
        
        feat_norm_test_temporal = np.array([(feat_in_test_temporal[i]-min_f_train_temporal)/(max_f_train_temporal-min_f_train_temporal)
                                    for i in range(len(feat_in_test_temporal))],dtype=object)
    

    
    

        X_test_temporal, y_test = feat_norm_test_temporal, np.zeros(len(feat_norm_test_temporal))
        x_test = MyBatchGenerator(
        X_test_temporal, 
        np.zeros(len(y_test)), 
        batch_size=1, 
        shuffle=False)
        
        y_estimada_test = np.hstack(modelMagnitude.predict(x_test,verbose=0))
        
        
        return np.round(y_estimada_test[0],2)  

    def DNN_magnitude_menores4m(self, feat_in_test_temporal):
        from keras.models import load_model
        import numpy as np
        
        import os
        import json
        
        global modelMagnitude_menores4m, dictParametersMagnitude

        try: modelMagnitude_menores4m, dictParametersMagnitude
        
        except:
            #Path where the model was saved
            path_model = self.model_path_menores4m
            path_normalization = self.normalization_path_menores4m
            modelMagnitude_menores4m = load_model(path_model)
            dictParametersMagnitude = json.load(open(path_normalization, "r", encoding='utf-8'))
        
        
        #Features are translated into a range between 0 and 1  
        min_f_train_temporal = np.array(dictParametersMagnitude['min_temporal'])
        max_f_train_temporal = np.array(dictParametersMagnitude['max_temporal'])
        
        
        feat_norm_test_temporal = np.array([(feat_in_test_temporal[i]-min_f_train_temporal)/(max_f_train_temporal-min_f_train_temporal)
                                    for i in range(len(feat_in_test_temporal))],dtype=object)
    

    
    

        X_test_temporal, y_test = feat_norm_test_temporal, np.zeros(len(feat_norm_test_temporal))
        x_test = MyBatchGenerator(
        X_test_temporal, 
        np.zeros(len(y_test)), 
        batch_size=1, 
        shuffle=False)
        y_estimada_test = np.hstack(modelMagnitude_menores4m.predict(x_test,verbose=0))
        
        
        return np.round(y_estimada_test[0],2)  
            
    def magnitude_estimation(self, sac_conc, inv):
        feat_in_test_temporal =function_features_extration(sac_conc, inv)
        print(f"feat_in_test_temporal.shape: {feat_in_test_temporal.shape}")
        
        magnitude = self.DNN_magnitude_todos( [feat_in_test_temporal] )
        magnitud_menores4m = self.DNN_magnitude_menores4m( [feat_in_test_temporal] )
        
        return magnitude, magnitud_menores4m