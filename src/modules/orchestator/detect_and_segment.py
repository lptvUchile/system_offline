import sys
import os
import numpy as np
from obspy import read
import obspy
import scipy
import pandas as pd
import argparse

from src.modules.Detection.src.models.Deteccion_sismica import Deteccion_sismica
from src.modules.Detection.src.utils.segment_event import segment_event

parser = argparse.ArgumentParser()
parser.add_argument("--sac_test_name", type=str, default="CO10")
parser.add_argument("--detection_output_path", type=str, default="results")
args = parser.parse_args()

sac_test_name = args.sac_test_name
detection_output_path = args.detection_output_path


    
    
print("sacs:", os.path.join(os.getcwd(), "sacs"))
print("sac_test_name:", sac_test_name)
print("Full path:", os.path.join(os.getcwd(), "sacs", sac_test_name))
os.makedirs(os.path.join(os.getcwd(), "results"), exist_ok=True)

# === Deteccion ================================================================

Deteccion_sismica(
        os.path.join(os.getcwd(), f"{sac_test_name}","*BH*.sac"),
        models={
            "path_probPrior_train": os.path.join(os.getcwd(), "src/models/detection_2025/Probs_Prior_Train.npy"),
            "path_modelo": os.path.join(os.getcwd(), "src/models/detection_2025/model_MLP_HMM_NC_M4_v2.pt"),
            "path_3estados": os.path.join(os.getcwd(), "src/models/detection_2025/phones_3estados.txt"),
            "path_transitions_file": os.path.join(os.getcwd(), "src/models/detection_2025/final_16_layers3_s1_lr001_NorthChile.mdl"),
        },
        results_path=detection_output_path,
    )

# === Segmentacion ============================================================


# === Leer detecciones ========================================================
pre_str = sac_test_name.split('/')[-1].split('.')[0]
print("pre_str:", pre_str)
ctm_file = "Detection_"+pre_str+"_BH*.ctm"
print("ctm_file:", ctm_file)
with open(os.path.join("results", ctm_file), "r") as f:
        detections = f.readlines()

df = pd.DataFrame([line.split() for line in detections if line.strip()], 
                columns=['time', 'duration', 'label'])
df['time'] = df['time'].astype(float)
df['duration'] = df['duration'].astype(float)

trace_vel = read(os.path.join(os.getcwd(), f"{sac_test_name}","*_BH*.sac"))


for idx, row in df.iterrows():
    start_time = trace_vel[0].stats.starttime + row['time']
    end_time = start_time + row['duration']
    
    trace_sliced_and_filtered = trace_vel.copy().filter("bandpass", freqmin=1, freqmax=15).slice(starttime=start_time-20, endtime=end_time + 20)
    fs = trace_sliced_and_filtered[0].stats.sampling_rate

    # se segmenta la señal usando envolvente, como un schmitt trigger, porcentajes 5% y 1%
    start_time_segment, end_time_segment, t, sxx, min_umbral_start, min_umbral_end = segment_event(trace_sliced_and_filtered, 0.05, 0.01)
    
    
    df.at[idx, 'time'] =  trace_sliced_and_filtered[0].stats.starttime + start_time_segment
    if type(end_time_segment) == np.float64:
        print(f"start_time_segment: {start_time_segment}, end_time_segment: {end_time_segment}, lenght: {end_time_segment - start_time_segment}")
    if type(end_time_segment) == np.float64:
        df.at[idx, 'duration'] = end_time_segment
    else:
        df.at[idx, 'duration'] = 120       


# TODO: guardar en CSV?
df.to_csv(os.path.join(detection_output_path, f"Detection_{pre_str}_BH*.csv"), index=False)
print("Detection dataframe saved at:", os.path.join(detection_output_path, f"Detection_{pre_str}_BH*.csv"))


