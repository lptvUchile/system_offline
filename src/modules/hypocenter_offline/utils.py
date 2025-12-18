import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
import torch
import torch.nn as nn
import numpy.matlib as matlib

def butter_highpass(cutoff, fs, order=3):
    """
    Filtro Butterworth de paso alto.

    Parameters
    ----------
    cutoff : float
        Frecuencia de corte.
    fs : float
        Frecuencia de muestreo.
    order : int, optional
        Orden del filtro.

    Returns
    -------
    b : array
        Coeficientes del filtro.
    a : array
        Coeficientes del filtro.
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
def butter_highpass_lfilter(data, cutoff, fs, order=2):
    """
    Filtro Butterworth de paso alto.

    Parameters
    ----------
    data : array
        Señal a filtrar.
    cutoff : float
        Frecuencia de corte.
    fs : float
        Frecuencia de muestreo.
    order : int, optional
        Orden del filtro.

    Returns
    -------
    y : array
        Señal filtrada.
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    #hago padding hacia la izquierda
    len_padding = 20 * order    #numero de muestras que se agregan
    data_ = np.pad(data, (len_padding, 0), 'symmetric', reflect_type='odd') 
    y = signal.lfilter(b, a, data_)
    y = y[len_padding:]   #elimino el padding
    return y

def get_frames(signal, frame_length, frame_shift, window=None):
    """
    Obtiene los frames de una señal.

    Parameters
    ----------
    signal : array
        Señal a obtener los frames.
    frame_length : int
        Longitud de los frames.
    frame_shift : int
        Separación entre frames.
    window : array, optional
        Ventana a aplicar a los frames.

    Returns
    -------
    Seg : array
        Frames de la señal.
    """
    if window is None:
        window=np.hamming(frame_length) 

    L = len(signal)
    N = int(np.fix((L-frame_length)/frame_shift + 1)) 

    Index = (matlib.repmat(np.arange(frame_length),N,1)+matlib.repmat(np.expand_dims(np.arange(N)*frame_shift,1),1,frame_length)).T
    hw=matlib.repmat(np.expand_dims(window,1),1,N)
    Seg=signal[Index]*hw

    return Seg.T

def parametrizador(senial, frame_length, frame_shift, nfft, window=None, escala='logaritmica'):
    """
    Parametrizador de una señal.

    Parameters
    ----------
    senial : array
        Señal a parametrizar.
    frame_length : int
        Longitud de los frames.
    frame_shift : int
        Separación entre frames.
    nfft : int
        Longitud de la FFT.
    window : array, optional
        Ventana a aplicar a los frames.
    escala : str, optional
        Escala a utilizar.

    Returns
    -------
    Y : array
        Señal parametrizada.
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
    Parametrizador de una señal.

    Parameters
    ----------
    senial : array
        Señal a parametrizar.
    frame_length : int
        Longitud de los frames.
    frame_shift : int
        Separación entre frames.
    nfft : int
        Longitud de la FFT.
    window : array, optional
        Ventana a aplicar a los frames.
    escala : str, optional
        Escala a utilizar.

    Returns
    -------
    Y : array
        Señal parametrizada.
    """
    
    y = get_frames(senial,frame_length,frame_shift)
    Y = np.abs(np.fft.fft(y, nfft, axis=1))**2
    if escala == 'logaritmica':
        Y = np.log10(Y[:, :int(np.fix(Y.shape[1]/2))+1] )
    elif escala == 'lineal':
        Y = Y[:, :int(np.fix(Y.shape[1]/2))+1] 
    return Y


def E3(senial, frame_length,frame_shift,escala='logaritmica'):
    """
    Energía de una señal.

    Parameters
    ----------
    senial : array
        Señal a obtener la energía.
    frame_length : int
        Longitud de los frames.
    frame_shift : int
        Separación entre frames.
    escala : str, optional
        Escala a utilizar.

    Returns
    -------
    salida : array
        Energía de la señal.
    """

    Edos = E2(senial, frame_length,frame_shift,escala)
    if escala =='logaritmica':
        salida = Edos/np.max(Edos) #Edos-np.max(Edos)
    elif escala == 'lineal':
        salida = Edos-np.max(Edos)
    
    return salida

def E2(spectrum,escala='logaritmica'):
    """
    Energía de una señal.

    Parameters
    ----------
    spectrum : array
        Espectro de la señal.
    escala : str, optional
        Escala a utilizar.

    Returns
    -------
    salida : array
        Energía de la señal.
    """
    
    if escala =='logaritmica':
        salida = np.log10(np.sum(spectrum**2,1))
    elif escala == 'lineal':
        salida = np.sum(spectrum**2,1)
    return salida

def sec_div_max(secuencia):
    """
    Secuencia dividida por el máximo.

    Parameters
    ----------
    secuencia : array
        Secuencia a dividir por el máximo.

    Returns
    -------
    secuencia : array
        Secuencia dividida por el máximo.
    """
    
    for i in range(len(secuencia)):
        maximo = np.max([np.abs(secuencia[i].max()),np.abs(secuencia[i].min())])
        secuencia[i] = secuencia[i]/maximo
    return secuencia


def downsample_to40hz(st_channel,fs):
    """
    Downsample a 40 Hz.

    Parameters
    ----------
    st_channel : array
        Señal a downsample.
    fs : float
        Frecuencia de muestreo.

    Returns
    -------
    st_channel : array
        Señal downsampled.
    """
    
    st_channel[0].data = sp.signal.resample(
        st_channel[0].data, int(len(st_channel[0].data) * 40 / fs)
    )
    st_channel[0].stats.sampling_rate = 40
    return st_channel

def save_target_metrics(tests_csv, seed, label, **kwargs):
    """
    Guarda las métricas de un experimento.

    Parameters
    ----------
    tests_csv : pandas DataFrame
        DataFrame con las métricas de los experimentos.
    seed : int
        Seed del experimento.
    label : str
        Label del experimento.
    **kwargs : dict
        Métricas a guardar.

    Returns
    -------
    tests_csv : pandas DataFrame
        DataFrame con las métricas de los experimentos.
    """
    
    if seed not in tests_csv.index:
        tests_csv.loc[seed] = {}
    
    for key, value in kwargs.items():
        tests_csv.at[seed, f"{key}_{label}"] = value
    
    return tests_csv
    
def safe_collate(batch):
    """
    Cola segura.

    Parameters
    ----------
    batch : list
        Batch a colar.

    Returns
    -------
    batch : list
        Batch colado.
    """
    
    batch = [item for item in batch if item is not None and item["temporal_feat"] is not None]
    if not batch:
        return None  
    return batch

def calc_mae(df,cols_real,cols_pred):
    """
    Calcula el MAE de un DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame con las columnas reales y predichas.
    cols_real : str
        Columna real.
    cols_pred : str
        Columna predicha.

    Returns
    -------
    mae : float
        MAE.
    """
    
    list_errors = np.abs(df[cols_real]-df[cols_pred]).values.tolist()
    mae = np.mean(list_errors)
    return np.round(mae,3)

def calc_std(df,cols_real,cols_pred):
    """
    Calcula la desviación estándar de un DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame con las columnas reales y predichas.
    cols_real : str
        Columna real.
    cols_pred : str
        Columna predicha.

    Returns
    -------
    std : float
    """

    list_errors = (df[cols_real]-df[cols_pred]).values.tolist()
    std = np.std(list_errors)
    return np.round(std,3)

def calc_max_error(df,cols_real,cols_pred):
    """
    Calcula el máximo error de un DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame con las columnas reales y predichas.
    cols_real : str
        Columna real.
    cols_pred : str
        Columna predicha.

    Returns
    -------
    max_val : float
    """

    list_errors = np.abs(df[cols_real]-df[cols_pred]).values.tolist()
    max_val = np.max(list_errors)
    return np.round(max_val,3)

def calc_mae_norm(df,cols_real,cols_pred):
    """
    Calcula el MAE normalizado de un DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame con las columnas reales y predichas.
    cols_real : str
        Columna real.
    cols_pred : str
        Columna predicha.

    Returns
    -------
    mae_norm : float
    """

    mae = calc_mae(df,cols_real,cols_pred)
    list_reals = df[cols_real].values.tolist()
    mae_norm = mae/np.mean(list_reals)
    return np.round(mae_norm,3)

def calc_metrics_per_seed(seed,preds_train,preds_val,preds_test,label="hypocenter"):
    """
    Calcula las métricas por seed.

    Parameters
    ----------
    seed : int
        Seed del experimento.
    preds_train : list
        Predicciones de los datos de entrenamiento.
    preds_val : list
        Predicciones de los datos de validación.
    preds_test : list
        Predicciones de los datos de test.
    label : str, optional
        Label del experimento.

    Returns
    -------
    dict_metrics : dict
        Diccionario con las métricas.
    """
    
    assert label in ["hypocenter","incidence"], "Label debe ser 'hypocenter' o 'incidence'"
    if preds_train is not None and preds_val is not None:
        df_train,df_val,df_test = pd.DataFrame(preds_train),pd.DataFrame(preds_val),pd.DataFrame(preds_test)
        if label == "incidence":
            df_train[["real","prediction"]] = np.rad2deg(df_train[["real","prediction"]])
            df_val[["real","prediction"]] = np.rad2deg(df_val[["real","prediction"]])
            df_test[["real","prediction"]] = np.rad2deg(df_test[["real","prediction"]])
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
        if label == "incidence":
            df_test[["real","prediction"]] = np.rad2deg(df_test[["real","prediction"]])
        dict_metrics = {"seed": seed,
                        "mae_test": calc_mae(df_test,"real","prediction"),
                        "error_std_test": calc_std(df_test,"real","prediction"),
                        "max_error_test": calc_max_error(df_test,"real","prediction"),
                        "mae_norm_test": calc_mae_norm(df_test,"real","prediction")}
    return dict_metrics

def summarize_results(df,name_experiment,choose_by="mae_val"):
    """
    Resume los resultados de un experimento.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame con las métricas de los experimentos.
    name_experiment : str
        Nombre del experimento.
    choose_by : str, optional
        Columna a elegir.

    Returns
    -------
    dict_results : dict
        Diccionario con las métricas.
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

def set_optimizer(model,optimizer_selection,learning_rate):
    """
    Establece el optimizador.

    Parameters
    ----------
    model : torch model
        Modelo a optimizar.
    optimizer_selection : str
        Nombre del optimizador.
    learning_rate : float
        Tasa de aprendizaje.

    Returns
    -------
    optimizer : torch optimizer
        Optimizador.
    """
    
    if optimizer_selection == "adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    elif optimizer_selection == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
    elif optimizer_selection == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(),lr=learning_rate)
    elif optimizer_selection == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(),lr=learning_rate)
    elif optimizer_selection == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(),lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_selection} not supported or void")
    return optimizer

def set_criterion(criterion_selection):
    """
    Establece la función de pérdida.

    Parameters
    ----------
    criterion_selection : str
        Nombre de la función de pérdida.

    Returns
    -------
    criterion : torch criterion
        Función de pérdida.
    """
    
    if criterion_selection == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif criterion_selection == "mse":
        criterion = nn.MSELoss()
    elif criterion_selection == "mae":
        criterion = nn.L1Loss()
    elif criterion_selection == "huber":
        criterion = nn.HuberLoss()
    elif criterion_selection == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Criterion {criterion_selection} not supported or void")
    return criterion