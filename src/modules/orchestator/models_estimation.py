from obspy import UTCDateTime, read_inventory, read
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import argparse
import pandas as pd
import yaml
from tqdm import tqdm

from src.modules.Magnitude.magnitude_estimation import MagnitudeEstimator
from src.modules.hypocenter_offline.reduced_model import HypocenterModelTorch
from src.modules.hypocenter_offline.offline_compute import OfflineDistancePreprocessing


# Carga de modelos


cwd = os.getcwd()

# se inicializa el modelo de magnitud
try:
    magnitude_estimator = MagnitudeEstimator(
        model_path_mayores4m=os.path.join(os.getcwd(), "src/models/norte_extendido_allMag/magnitude_30seeds/DNN_magnitude_1761681833_conGcorr_seed147_Norte_extendido_paper_sist_offline_vel_Toda_repeat1_distance.h5"),
    normalization_path_mayores4m=os.path.join(os.getcwd(), "src/models/norte_extendido_allMag/magnitude_30seeds/normalization_parameters_magnitude_1761681833_conGcorr_seed147_Norte_extendido_paper_sist_offline_vel_Toda_repeat1_distance.json"),
    model_path_menores4m=os.path.join(os.getcwd(), "src/models/magnitud_menor_a_4_vel/magnitude_menor4_20250916_seed132_reTraining_vel.h5"),
    normalization_path_menores4m=os.path.join(os.getcwd(), "src/models/magnitud_menor_a_4_vel/normalization_parameters_magnitude_menor4_20250916_seed132_reTraining_vel.json")
    )
    print("Modelo de magnitud inicializado correctamente")
except Exception as e:
    print(f"Error al inicializar el modelo de magnitud: {e}")



stations_path = os.path.join(cwd, "src/models/stations_index.csv")


#===========================================================
#=========MODELO HYPOCENTRO NUEVO ==========================
#===========================================================



YAML_PATH = os.path.join(os.getcwd(), "src/models/hypocenter_offline_transformer/Config_137_heads3_tam1_cnn4_bs2.yaml")
MODEL_PATH = os.path.join(os.getcwd(), "src/models/hypocenter_offline_transformer/588_weights.pth")



def read_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        test_config = yaml.safe_load(file)
    return test_config
params_dict = read_yaml(YAML_PATH)

hypocenter_offline_preprocessing = OfflineDistancePreprocessing(**params_dict['preprocess_config'])
hypocenter_offline_model = HypocenterModelTorch(device="cpu",model_cfg=params_dict['model_config'])
hypocenter_offline_model.load_model(path=MODEL_PATH)
print("Modelo de torch-hipocentro cargado correctamente")



parser = argparse.ArgumentParser()
parser.add_argument("--sac_test_name", type=str, default="CO10")
parser.add_argument("--detection_dataframe_path", type=str, default="results")
parser.add_argument("--inventory_path", type=str, default="src/models/inventory/C1_CO10.xml")

args = parser.parse_args()
sac_test_name = args.sac_test_name
detection_dataframe_path = args.detection_dataframe_path
inventory_path = args.inventory_path
trace_vel = read(os.path.join(os.getcwd(), f"{sac_test_name}","*_BH*.sac"))

df = pd.read_csv(detection_dataframe_path)
print("df:", df)
error_description_file = "results/error_description.txt"

with open(error_description_file, "w") as f:
    f.write("Error log:\n")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Procesando eventos", unit="evento"):
    
    try:
        inv = read_inventory(inventory_path)
    except Exception as e:
        print(f"Error al leer el inventario: {e}")
        continue
    
    
    # Se recorta la señal usando el segmentado por envolvente y SIN filtrar, se le añaden segundos extras para el modelo de distancia que usa segmentación de 20 antes y 120 después
    trace_sliced_vel= trace_vel.copy().slice(starttime=UTCDateTime(row['time'])- 20, endtime=UTCDateTime(row['time'])+120)


    # se crea la señal concetenada para la estimación de magnitud
    sac_Z_v = trace_sliced_vel.copy().select(component="Z")
    sac_E_v = trace_sliced_vel.copy().select(component="E")
    sac_N_v = trace_sliced_vel.copy().select(component="N")
    sac_conc_v = sac_Z_v.copy()
    sac_conc_v += sac_E_v.copy()
    sac_conc_v += sac_N_v.copy()
    

    
    # se estima la magnitud
    try:
        mag, mag_menores4m = magnitude_estimator.magnitude_estimation(sac_conc_v.copy(), inv)
        df.at[idx, 'pred_magnitud_todos'] = mag
        df.at[idx, 'pred_magnitud_menores4m'] = mag_menores4m
    except Exception as e:
        print(e)
        print("Fallo en el calculo de magnitud")
        print("idx", idx)
        df.at[idx, 'descartado'] = True
        df.at[idx, 'razon_descarte'] = "Fallo en el calculo de magnitud"
        df.at[idx, 'cat_error'] = 2
        with open(error_description_file, "a") as f:
            f.write(f"Evento {idx} fallo en el calculo de magnitud: {e}\n")
        continue

    
    #================================================================================================================
    #===========================================Modelo de HYPOCENTRO PYTORCH ========================================
    #================================================================================================================
    try:
        input_tensor = hypocenter_offline_preprocessing.preprocess_data(trace = trace_sliced_vel, inv = inv, frame_p = UTCDateTime(row['time']))
        dst = hypocenter_offline_model.evaluate_wrapper_mode(input_tensor)
        df.at[idx, 'pred_hipocentro'] = dst
    except Exception as e:
        print(e)
        print("Fallo en el calculo de hipocentro")
        print("idx", idx)
        df.at[idx, 'descartado'] = True
        df.at[idx, 'razon_descarte'] = "Fallo en el calculo de hipocentro"
        df.at[idx, 'cat_error'] = 7
        with open(error_description_file, "a") as f:
            f.write(f"Evento {idx} fallo en el calculo de hipocentro: {e}\n")
        continue
    df.at[idx, 'descartado'] = False




df.to_csv(f"results/models_estimation_{sac_test_name.split('/')[-1].split('.')[0]}.csv", index=False)
print("models_estimation dataframe saved at:", f"results/models_estimation_{sac_test_name.split('/')[-1].split('.')[0]}.csv")
print("Total de eventos procesados:", len(df))