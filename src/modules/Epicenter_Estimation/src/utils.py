#class to use
import numpy.matlib
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.utils import Sequence
import numpy as np
import obspy
import os
import pandas as pd
import scipy as sp

class MyBatchGenerator_lstm_mlp(Sequence):
    'Generates data for Keras'
    def __init__(self, X, x, y, batch_size=1, shuffle=False, global_feat=True):
        'Initialization'
        self.X = X
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.global_feat = global_feat
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
        xb = np.empty((self.batch_size, *self.x[index].shape))
        yb = np.empty((self.batch_size, *self.y[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            xb[s] = self.x[index]
            yb[s] = self.y[index]

        if self.global_feat:
            return [Xb, xb], yb
        else:
            return [Xb], yb
    

def pad_and_convert_to_tensor(arrays):
    max_length = max(arr.shape[0] for arr in arrays)
    
    padded_arrays = [
        np.pad(arr, ((0, max_length - arr.shape[0]), (0, 0)), mode='constant')
        for arr in arrays
    ]
    return tf.convert_to_tensor(padded_arrays, dtype=tf.float32)

def get_frames(signal, frame_length, frame_shift, window=None):
    if window is None:
        window=np.hamming(frame_length) 

    L = len(signal)
    N = int(np.fix((L-frame_length)/frame_shift + 1)) #number of frames

    Index = (np.matlib.repmat(np.arange(frame_length),N,1)+np.matlib.repmat(np.expand_dims(np.arange(N)*frame_shift,1),1,frame_length)).T
    hw=np.matlib.repmat(np.expand_dims(window,1),1,N)
    Seg=signal[Index]*hw

    return Seg.T

def parametrizador(senial, frame_length, frame_shift, nfft, window=None, escala='logaritmica'):
    #pdb.set_trace()
    y = get_frames(senial,frame_length,frame_shift)
    Y = np.abs(np.fft.fft(y, nfft, axis=1))
    if escala == 'logaritmica':
        Y = np.log10(Y[:, :int(np.fix(Y.shape[1]/2))+1] )
    elif escala == 'lineal':
        Y = Y[:, :int(np.fix(Y.shape[1]/2))+1] 
    return Y

def parametrizador2(senial, frame_length, frame_shift, nfft, window=None, escala='logaritmica'):
    #pdb.set_trace()
    y = get_frames(senial,frame_length,frame_shift)
    Y = np.abs(np.fft.fft(y, nfft, axis=1))**2
    if escala == 'logaritmica':
        Y = np.log10(Y[:, :int(np.fix(Y.shape[1]/2))+1] )
    elif escala == 'lineal':
        Y = Y[:, :int(np.fix(Y.shape[1]/2))+1] 
    return Y


def E2(senial, frame_length,frame_shift,escala='logaritmica'):
    y = get_frames(senial,frame_length,frame_shift)        

    if escala =='logaritmica':
        Y = np.log10(np.sum(y**2,1))
    elif escala == 'lineal':
        Y = np.sum(y**2,1)
    return Y

def E3(senial, frame_length,frame_shift,escala='logaritmica'):
    Edos = E2(senial, frame_length,frame_shift,escala)
    if escala =='logaritmica':
        salida = Edos/np.max(Edos) #Edos-np.max(Edos)
    elif escala == 'lineal':
        salida = Edos-np.max(Edos)
    
    return salida

def up_level_idx(y,n,p,l): #Subida
    nn=round((len(y)-n)/p)
    yy=np.zeros(nn)
    for i in range(nn):
        subarray=y[i*p:i*p+n]
        yy[i]=sum(abs(subarray))/n*p
    max_yy = max(yy)
    yy=yy/max_yy
    ix=np.zeros(l)
    vx=np.zeros(l)
    for i in range(l):
        i1=i+1
        ixi=np.argwhere(np.array(yy)>=i1/l)[0]
        ix[i]=ixi
        vx[i]=yy[ixi]
    return ix*p

def down_level_idx(y,n,p,l): #Bajada
    nn=round((len(y)-n)/p)
    yy=np.zeros(nn)
    for i in range(nn):
        subarray=y[i*p:i*p+n]
        yy[i]=sum(abs(subarray))/n*p
    max_yy = max(yy)
    yy = yy/max(yy)
    ix = np.zeros(l)
    vx = np.zeros(l)
    i1 = l
    for i in range(l):
        ixi=np.argwhere(np.array(yy)>=i1/l)[-1]
        ix[i]=ixi
        vx[i]=yy[ixi]
        i1  -= 1
    return ix*p

def az(stla, stlo, evla, evlo, rad = True):
    #client = Client()
    #result = client.distaz(stalat=stla, stalon=stlo, evtlat=evla, evtlon=evlo)
    
    #cos_o = (evla - stla)/result['distance']
    #sen_o = (evlo - stlo)/result['distance']
    #az_o  = result['backazimuth']
    
    distance_m, az_o, baz_o = obspy.geodetics.base.gps2dist_azimuth(float(evla), 
                                                                        float(evlo),
                                                                        float(stla), 
                                                                        float(stlo), 
                                                                        a=6378137.0, 
                                                                        f=0.0033528106647474805)
    
    cos_o = np.cos(np.deg2rad(baz_o)) ### NUEVO
    sen_o = np.sin(np.deg2rad(baz_o)) ### NUEVO

    if rad:
        baz_o = np.deg2rad(baz_o)

    return cos_o, sen_o, baz_o #devuelve baz



import numpy as np
import tensorflow as tf
import obspy

#### some functions to use ######
def to_angle(seno,coseno, corregir = True, rad = False):
    
    angle = np.arctan2(seno,coseno)
    
    ### corregir corrects negative angles to [pi,2pi]
    if corregir == False:
        return angle
    else:
        for i in range(len(angle)):
            if angle[i]<=0:
                angle[i] = angle[i]+2*np.pi
    
    # if rad is False, the angles are returned between 0-360 degrees
    if rad == False:
        return angle*180/np.pi
    # if rad is not false, the angle is returned in radians
    else:
        return angle
    
def estimar_error_abs(pred,tar, rad = False):
    #if the angles are in radian units, use rad = True. If in degrees, use rad = False.
    #this function return the mean absolute error in degrees between the prediction and the target.
    prediction = pred.copy()
    target = tar.copy()
    ### recibe en [-pi,pi]
    output_error = []
    if rad == False:
        for i in range(len(prediction)):
            error = np.min([np.abs(target[i]-prediction[i]),np.abs(target[i]-prediction[i]+360),np.abs(target[i]-prediction[i]-360)])
            output_error.append(error)
        output_error = np.array(output_error)
        output_error = np.mean(output_error)
        return output_error
    
    else:
        for i in range(len(prediction)):
            error = np.min([np.abs(target[i]-prediction[i]),np.abs(target[i]-prediction[i]+2*np.pi),np.abs(target[i]-prediction[i]-2*np.pi)])
            output_error.append(error)
        output_error = np.array(output_error)
        output_error = np.mean(output_error)
        return output_error*180/np.pi

def sec_div_max(secuencia):
    for i in range(len(secuencia)):
        maximo = np.max([np.abs(secuencia[i].max()),np.abs(secuencia[i].min())])
        secuencia[i] = secuencia[i]/maximo
    return secuencia


class VanillaPositionalEncoding(tf.keras.layers.Layer):

    def get_angles(self,pos, i, d_model):
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles # (seq_length, d_model)

    def PositionalEncoding(self,inputs_layer):
        # input shape batch_size, seq_length, d_model
        seq_length = inputs_layer.shape.as_list()[-2]
        d_model = inputs_layer.shape.as_list()[-1]
        # Calculate the angles given the input
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        # Calculate the positional encodings
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Expand the encodings with a new dimension
        pos_encoding = angles[np.newaxis, ...]
        #plt.pcolormesh(pos_encoding[0], cmap='viridis')
        return inputs_layer + tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return self.PositionalEncoding(inputs)
    
    def get_config(self):
        config = super(VanillaPositionalEncoding, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def az(stla, stlo, evla, evlo, rad = True):
    #client = Client()
    #result = client.distaz(stalat=stla, stalon=stlo, evtlat=evla, evtlon=evlo)
    
    #cos_o = (evla - stla)/result['distance']
    #sen_o = (evlo - stlo)/result['distance']
    #az_o  = result['backazimuth']
    
    distance_m, az_o, baz_o = obspy.geodetics.base.gps2dist_azimuth(float(evla), 
                                                                        float(evlo),
                                                                        float(stla), 
                                                                        float(stlo), 
                                                                        a=6378137.0, 
                                                                        f=0.0033528106647474805)
    
    cos_o = np.cos(np.deg2rad(baz_o)) ### NUEVO
    sen_o = np.sin(np.deg2rad(baz_o)) ### NUEVO

    if rad:
        baz_o = np.deg2rad(baz_o)

    return cos_o, sen_o, baz_o #devuelve baz


def compute_binary_metrics(pred, real, label):
    pred = (pred >= 0.5).astype(int)

    # Compute confusion matrix for train, validation, and test sets
    conf_matrix_train = confusion_matrix(real, pred)
    accuracy = np.sum(np.diag(conf_matrix_train)) / np.sum(conf_matrix_train)
    f1 = f1_score(real, pred)
    precision = precision_score(real, pred)
    recall = recall_score(real, pred)
    
    # Print confusion matrices
    print(f"{label} Confusion Matrix:\n", conf_matrix_train)
    print(f"Accuracy {label}: ", accuracy)
    print(f"F1-Score {label}: ", f1)
    print(f"Precision {label}: ", precision)
    print(f"Recall {label}: ", recall)
    return conf_matrix_train, accuracy, f1, precision, recall

def save_binary_metrics(tests_csv, seed, conf_matrix, accuracy, f1, precision, recall, label):
    if seed not in tests_csv.index:
        tests_csv.loc[seed] = {}
    
    tests_csv.at[seed, f"conf_matrix_{label}"] = str(conf_matrix)
    tests_csv.at[seed, f"accuracy_{label}"] = accuracy
    tests_csv.at[seed, f"f1_{label}"] = f1
    tests_csv.at[seed, f"precision_{label}"] = precision
    tests_csv.at[seed, f"recall_{label}"] = recall
    
    return tests_csv

def save_regression_metrics(tests_csv, seed, mape_dist, mae_baz, label):

    if seed not in tests_csv.index:
        tests_csv.loc[seed] = {}

    tests_csv.at[seed, f"mae_dist_{label}"] = mape_dist
    tests_csv.at[seed, f"mae_baz_{label}"] = mae_baz

    return tests_csv


def save_results(id_event, stations_names, real_value, prediction_value, prefix, test_name, seed, label, model_name):
    df = pd.DataFrame(
        data=np.transpose(
            np.vstack((id_event, stations_names, real_value, prediction_value.reshape(-1)))
        ),
        columns=["id_evento", "Estacion", "Real", "Estimacion"],
    )
    #os.makedirs(f"results/{model_name}/{prefix}/{test_name}", exist_ok=True)
    #df.to_csv(f"results/{model_name}/{prefix}/{test_name}/{label}_{seed}.csv", index=False)
    return df



def save_multitask_results(id_train, stations_train, real_values, prediction_values, prefix, test_name, seed, label, model_name):
    data_dict = {
        "id_evento": id_train,
        "Estacion": stations_train,
    }

    for i, real_value in enumerate(real_values):
        data_dict[f"Real_{i}"] = real_value.reshape(-1)
    
    for i, prediction_value in enumerate(prediction_values):
        data_dict[f"Estimacion_{i}"] = prediction_value.reshape(-1)
    

    df = pd.DataFrame(data_dict)
    return df

# def downsample_to40hz(st_channel):
#     st_channel[0].data = sp.signal.resample(
#         st_channel[0].data, int(len(st_channel[0].data) * 40 / 100)
#     )
#     st_channel[0].stats.sampling_rate = 40
#     return st_channel
def downsample_to40hz(st_channel,fs=None):
    if fs is None:
        fs = st_channel[0].stats.sampling_rate
    st_channel[0].data = sp.signal.resample(
        st_channel[0].data, int(len(st_channel[0].data) * 40 / fs)
    )
    st_channel[0].stats.sampling_rate = 40
    return st_channel

def save_target_metrics(tests_csv, seed, label, **kwargs):
    if seed not in tests_csv.index:
        tests_csv.loc[seed] = {}
    
    for key, value in kwargs.items():
        tests_csv.at[seed, f"{key}_{label}"] = value
    
    return tests_csv

def _optional_filter(st, bandpass=[0.1, 8], highpass=None):
    """
    Filtra la señal en base a los parámetros bandpass y highpass.
    Si bandpass es None, no se aplica filtro.
    Si bandpass es una lista de 2 elementos, se aplica filtro bandpass.
    Si highpass es None, no se aplica filtro.
    Si highpass es un número, se aplica filtro highpass.
    """
    if bandpass and len(bandpass) == 2:
        st.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1], corners=2, zerophase=True)
    elif highpass:
        st.filter("highpass", freq=highpass, corners=2, zerophase=True)

def _align_and_resample(st, target_fs=40):
    z = st.select(channel="*Z")[0]
    e = st.select(channel="*E")[0]
    n = st.select(channel="*N")[0]
    t0 = max(z.stats.starttime, e.stats.starttime, n.stats.starttime)
    t1 = min(z.stats.endtime,   e.stats.endtime,   n.stats.endtime)
    st = st.slice(t0, t1)

    fs = z.stats.sampling_rate
    if fs > target_fs:
        st = downsample_to40hz(st, fs)
        fs = st.select(channel="*Z")[0].stats.sampling_rate
    return st, fs, t0

def _compute_p_vector(st, fs, p_starttime_t0, use_module=True, start_sec=0, dur_sec=4):

    z = st.select(channel="*Z")[0].data.astype(float)
    e = st.select(channel="*E")[0].data.astype(float)
    n = st.select(channel="*N")[0].data.astype(float)

    p_idx = int(round((p_starttime_t0) * fs))  # segundos desde t0 * fs
    
    start = p_idx + int(round(start_sec * fs))
    end   = start  + int(round(dur_sec * fs))

    if end > len(z):
        end = len(z)

    z_win = z[start:end]
    e_win = e[start:end]
    n_win = n[start:end]

    # Potencia definida como sum(x^2)/N
    P_E = np.sum(e_win**2) / e_win.shape[0]
    P_N = np.sum(n_win**2) / n_win.shape[0]
    P_Z = np.sum(z_win**2) / z_win.shape[0]


    if use_module:
        H_over_V = np.sqrt(P_E**2 + P_N**2) / P_Z
    else:
        H_over_V = (P_E + P_N) / (2.0 * P_Z)

    return H_over_V 

def p_vector_extractor(z_chann,e_chann,n_chann,p_starttime,inventory,dur_sec=10, start_sec=0, low_coff=None, high_coff=None, use_module=True):
    """
    """

    p_start = p_starttime #UTC
   
    st_raw = z_chann+e_chann+n_chann
    #print(f"canal_z_raw:{st_raw.select(component='Z')[0].data}")

    # alinear/decimar a TARGET_FS
    st_aligned, fs, t0 = _align_and_resample(st_raw, 40)
    #print(f"aligned: {st_aligned.select(component='Z')[0].data}")
    # remover respuesta + filtro (opcionales, no rompen si faltan)
    st_aligned.remove_response(inventory=inventory, output="VEL")
    #print(f"response_removed: {st_aligned.select(component='Z')[0].data}")
    _optional_filter(st_aligned, bandpass=[low_coff, high_coff], highpass=None)
    #print(f"filtered: {st_aligned.select(component='Z')[0].data}")
    

    # P relativa al inicio alineado
    p_start_rel_s = float(p_start - t0)

    # calcular p_vector (como en tu flujo)
    if use_module:
        p_vec = _compute_p_vector(st=st_aligned, fs=40, p_starttime_t0=p_start_rel_s, use_module=True, start_sec=start_sec, dur_sec=dur_sec)
    else:
        p_vec = _compute_p_vector(st=st_aligned, fs=40, p_starttime_t0=p_start_rel_s, use_module=False, start_sec=start_sec, dur_sec=dur_sec)
    p_vec = np.log(p_vec)
    return p_vec