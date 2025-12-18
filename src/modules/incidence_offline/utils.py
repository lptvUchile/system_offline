import numpy as np
import obspy
import pandas as pd
import scipy as sp
import numpy.matlib
from scipy import signal
from typing import List
from obspy import UTCDateTime, read, read_inventory
import os


def get_frames(signal, frame_length, frame_shift, window=None):
    """
    Obtiene las ventanas de la señal.
    Args:
        signal (numpy array): Señal de entrada.
        frame_length (int): Longitud de la ventana.
        frame_shift (int): Desplazamiento de la ventana.
        window (numpy array, optional): Ventana de ponderación.

    Returns:
        numpy array: Ventanas de la señal.
    """
    if window is None:
        window=np.hamming(frame_length) 

    L = len(signal)
    N = int(np.fix((L-frame_length)/frame_shift + 1)) #number of frames

    Index = (np.matlib.repmat(np.arange(frame_length),N,1)+np.matlib.repmat(np.expand_dims(np.arange(N)*frame_shift,1),1,frame_length)).T
    hw=np.matlib.repmat(np.expand_dims(window,1),1,N)
    Seg=signal[Index]*hw

    return Seg.T

def parametrizador(senial, frame_length, frame_shift, nfft, window=None, escala='logaritmica'):
    """
    Parametrizador de la señal.
    Args:
        senial (numpy array): Señal de entrada.
        frame_length (int): Longitud de la ventana.
        frame_shift (int): Desplazamiento de la ventana.
        nfft (int): Longitud de la FFT.
        window (numpy array, optional): Ventana de ponderación.
        escala (str, optional): Escala de la señal.
    Returns:
        numpy array: Espectro de la señal.
    """
    y = get_frames(senial,frame_length,frame_shift)
    Y = np.abs(np.fft.fft(y, nfft, axis=1))
    if escala == 'logaritmica':
        Y = np.log10(Y[:, :int(np.fix(Y.shape[1]/2))+1] )
    elif escala == 'lineal':
        Y = Y[:, :int(np.fix(Y.shape[1]/2))+1] 
    return Y

def parametrizador2(senial, frame_length, frame_shift, nfft, window=None, escala='logaritmica'):
    """
    Parametrizador de la señal al cuadrado.
    Args:
        senial (numpy array): Señal de entrada.
        frame_length (int): Longitud de la ventana.
        frame_shift (int): Desplazamiento de la ventana.
        nfft (int): Longitud de la FFT.
        window (numpy array, optional): Ventana de ponderación.
        escala (str, optional): Escala de la señal.
    Returns:
        numpy array: Espectro de la señal al cuadrado.
    """
    y = get_frames(senial,frame_length,frame_shift)
    Y = np.abs(np.fft.fft(y, nfft, axis=1))**2
    if escala == 'logaritmica':
        Y = np.log10(Y[:, :int(np.fix(Y.shape[1]/2))+1] )
    elif escala == 'lineal':
        Y = Y[:, :int(np.fix(Y.shape[1]/2))+1] 
    return Y


def parametrizador_k(senial, frame_length, frame_shift, nfft, k_in,delta_k,window=None, escala='logaritmica'):
    """
    Parametrizador de la señal para una banda de bines.
    Args:
        senial (numpy array): Señal de entrada.
        frame_length (int): Longitud de la ventana.
        frame_shift (int): Desplazamiento de la ventana.
        nfft (int): Longitud de la FFT.
        k_in (int): Bin inicial.
        delta_k (int): Numero de bins.
        window (numpy array, optional): Ventana de ponderación.
        escala (str, optional): Escala de la señal.
    Returns:
        numpy array: Espectro de la señal para la banda de bins.
    """
    assert delta_k>0
    y = get_frames(senial,frame_length,frame_shift)
    Y = np.abs(np.fft.fft(y, nfft, axis=1))
    if escala == 'logaritmica':
        Y = np.log10(Y[:, :int(np.fix(Y.shape[1]/2))+1] )
    elif escala == 'lineal':
        Y = Y[:, :int(np.fix(Y.shape[1]/2))+1] 
    Y = Y[:, k_in:k_in+delta_k+1]    

    return Y

def E2_k(spectrum,escala='logaritmica'):
    """
    Energia de la señal para una banda de bins.
    Args:
        spectrum (numpy array): Espectro de la señal.
        escala (str, optional): Escala de la señal.
    Returns:
        numpy array: Energia de la señal para la banda de bins.
    """
    if escala =='logaritmica':
        salida = np.log10(np.sum(spectrum**2,1))
    elif escala == 'lineal':
        salida = np.sum(spectrum**2,1)
    return salida



def E2(senial, frame_length,frame_shift,escala='logaritmica'):
    """
    Energia de la señal.
    Args:
        senial (numpy array): Señal de entrada.
        frame_length (int): Longitud de la ventana.
        frame_shift (int): Desplazamiento de la ventana.
        escala (str, optional): Escala de la señal.
    Returns:
        numpy array: Energia de la señal.
    """
    y = get_frames(senial,frame_length,frame_shift)        

    if escala =='logaritmica':
        Y = np.log10(np.sum(y**2,1))
    elif escala == 'lineal':
        Y = np.sum(y**2,1)
    return Y

def E3(senial, frame_length,frame_shift,escala='logaritmica'):
    """
    Energia de la señal normalizada.
    Args:
        senial (numpy array): Señal de entrada.
        frame_length (int): Longitud de la ventana.
        frame_shift (int): Desplazamiento de la ventana.
        escala (str, optional): Escala de la señal.
    Returns:
        numpy array: Energia de la señal normalizada.
    """
    Edos = E2(senial, frame_length,frame_shift,escala)
    if escala =='logaritmica':
        salida = Edos/np.max(Edos) #Edos-np.max(Edos)
    elif escala == 'lineal':
        salida = Edos-np.max(Edos)
    
    return salida


def downsample_to40hz(st_channel,fs):
    """
    Downsample de la señal a 40 Hz.
    Args:
        st_channel (obspy.Stream): Stream de la señal.
        fs (int): Frecuencia de muestreo de la señal.
    Returns:
        obspy.Stream: Stream de la señal downsampled.
    """
    st_channel[0].data = sp.signal.resample(
        st_channel[0].data, int(len(st_channel[0].data) * 40 / fs)
    )
    st_channel[0].stats.sampling_rate = 40
    return st_channel

def safe_collate(batch):
    """
    Collate de la señal.
    Args:
        batch (list): Lista de items.
    Returns:
        list: Lista de items.
    """
    batch = [item for item in batch if item is not None and item["temporal_feat"] is not None]
    if not batch:
        return None  # or handle dummy batch
    return batch

def calc_mae(df,cols_real,cols_pred):
    """
    Calcula el error medio absoluto.
    Args:
        df (pandas.DataFrame): DataFrame con las columnas reales y predichas.
        cols_real (str): Columna real.
        cols_pred (str): Columna predicha.
    Returns:
        float: Error medio absoluto.
    """
    list_errors = np.abs(df[cols_real]-df[cols_pred]).values.tolist()
    mae = np.mean(list_errors)
    return np.round(mae,3)

def calc_std(df,cols_real,cols_pred):
    """
    Calcula la desviacion estandar.
    Args:
        df (pandas.DataFrame): DataFrame con las columnas reales y predichas.
        cols_real (str): Columna real.
        cols_pred (str): Columna predicha.
    Returns:
        float: Desviacion estandar.
    """
    list_errors = (df[cols_real]-df[cols_pred]).values.tolist()
    std = np.std(list_errors)
    return np.round(std,3)

def calc_max_error(df,cols_real,cols_pred):
    """
    Calcula el error maximo.
    Args:
        df (pandas.DataFrame): DataFrame con las columnas reales y predichas.
        cols_real (str): Columna real.
        cols_pred (str): Columna predicha.
    Returns:
        float: Error maximo.
    """
    list_errors = np.abs(df[cols_real]-df[cols_pred]).values.tolist()
    max_val = np.max(list_errors)
    return np.round(max_val,3)

def calc_mae_norm(df,cols_real,cols_pred):
    """
    Calcula el error medio absoluto normalizado.
    Args:
        df (pandas.DataFrame): DataFrame con las columnas reales y predichas.
        cols_real (str): Columna real.
        cols_pred (str): Columna predicha.
    Returns:
        float: Error medio absoluto normalizado.
    """
    mae = calc_mae(df,cols_real,cols_pred)
    list_reals = df[cols_real].values.tolist()
    mae_norm = mae/np.mean(list_reals)
    return np.round(mae_norm,3)

def calc_metrics_per_seed_angle(seed,preds_train,preds_val,preds_test):
    """
    Calcula las metricas por seed y angulo.
    Args:
        seed (int): Seed.
        preds_train (list): Predicciones de la muestra de entrenamiento.
        preds_val (list): Predicciones de la muestra de validacion.
        preds_test (list): Predicciones de la muestra de prueba.
    Returns:
        dict: Metricas por seed y angulo.
    """

    if preds_train is not None and preds_val is not None:
        df_train,df_val,df_test = pd.DataFrame(preds_train),pd.DataFrame(preds_val),pd.DataFrame(preds_test)
        df_train[["real","prediction"]]=np.rad2deg(df_train[["real","prediction"]])
        df_val[["real","prediction"]]=np.rad2deg(df_val[["real","prediction"]])
        df_test[["real","prediction"]]=np.rad2deg(df_test[["real","prediction"]])

        dict_metrics = {"seed":seed,
                        "mae_train": calc_mae(df_train,"real","prediction"),
                        "error_std_train": calc_std(df_train,"real","prediction"),
                        "max_error_train": calc_max_error(df_train,"real","prediction"),
                        "mae_norm_train": calc_mae_norm(df_train,"real","prediction"),
                        "mae_val": calc_mae(df_val,"real","prediction"),
                        "error_std_val": calc_std(df_val,"real","prediction"),
                        "max_error_val": calc_max_error(df_val,"real","prediction"),
                        "mae_norm_val": calc_mae_norm(df_val,"real","prediction"),
                        "mae_test": calc_mae(df_test,"real","prediction"),
                        "error_std_test": calc_std(df_test,"real","prediction"),
                        "max_error_test": calc_max_error(df_test,"real","prediction"),
                        "mae_norm_test": calc_mae_norm(df_test,"real","prediction")}
    else:
        df_test = pd.DataFrame(preds_test)
        df_test[["real","prediction"]]=np.rad2deg(df_test[["real","prediction"]])
        dict_metrics = {"seed": seed,
                        "mae_test": calc_mae(df_test,"real","prediction"),
                        "error_std_test": calc_std(df_test,"real","prediction"),
                        "max_error_test": calc_max_error(df_test,"real","prediction"),
                        "mae_norm_test": calc_mae_norm(df_test,"real","prediction")}
    return dict_metrics


def summarize_results(df,name_experiment,choose_by="mae_val"):
    """
    Resume los resultados de la experimentacion.
    Args:
        df (pandas.DataFrame): DataFrame con las metricas.
        name_experiment (str): Nombre de la experimentacion.
        choose_by (str, optional): Columna por la cual se elige el mejor seed.
    Returns:
        dict: Resultados de la experimentacion.
    """
    best_seed = df[df[choose_by]==df[choose_by].min()]["seed"].item()
    dict_results={"experiment":name_experiment,
                  "best_seed":best_seed,
                  "mean_mae_test":df["mae_test"].mean(),
                  "mean_error_std_test":df["error_std_test"].mean(),
                  "mean_max_error_test":df["max_error_test"].mean(),
                  "mean_mae_norm_test":df["mae_norm_test"].mean(),
                  "best_seed_mae_test":df[df.seed==best_seed]["mae_test"].values.item(),
                  "best_seed_error_std_test":df[df.seed==best_seed]["error_std_test"].values.item(),
                  "best_seed_max_error_test":df[df.seed==best_seed]["max_error_test"].values.item(),
                  "best_seed_mae_norm_test":df[df.seed==best_seed]["mae_norm_test"].values.item()}
    return dict_results

def read_event(sacs_folder:str, event:str, station:str):
    """
    Lee el evento de la carpeta de sacs.
    Args:
        sacs_folder (str): Carpeta de los sacs.
        event (str): Nombre del evento.
        station (str): Nombre de la estacion.
    Returns:
        obspy.Stream: Stream de la señal.
        str: Network de la señal.
    """
    signal_path = os.path.join(sacs_folder,event,station)
    try:
        canal_sac_Z = read(os.path.join(signal_path, station+"_H" + "*Z.sac"))
        canal_sac_E = read(os.path.join(signal_path, station+"_H" + "*E.sac"))
        canal_sac_N = read(os.path.join(signal_path, station+"_H" + "*N.sac"))
    except:
        try:
            canal_sac_Z = read(os.path.join(signal_path, station+"_" + "*Z.sac"))
            canal_sac_E = read(os.path.join(signal_path, station+"_" + "*E.sac"))
            canal_sac_N = read(os.path.join(signal_path, station+"_" + "*N.sac"))
        except Exception as e:
            print(f"Error en el evento {event}, estacion {station}, error: {e}")
            return None
            
    trace = canal_sac_Z+canal_sac_E+canal_sac_N
    network = trace[0].stats.network

    return trace,network

        


