import numpy as np
from obspy import UTCDateTime, read, read_inventory
import math
import time
import os
import pandas as pd
from obspy import UTCDateTime
from src.distance.features import FeatureExtractor
import glob


def function_features_extraction(
    BDtype="todaBD", 
    test=False, 
    use_accel=False, 
    response="VEL", 
    window_limit=[20, 120], 
    umbral_corte=0.03, 
    filter=None, 
    test_name=None,
    test_folder_name = None,
    ind_random_train=None,
    ind_random_val=None,
    ind_random_test=None,
    use_cnn = False,
    one_hot_encoding = False,
    include_index=False,
    include_env=False,
    include_lat_lon=False,
    include_magnitude=False,
    norm_energy=False,
    concat_features=False,
    square_fft = False,
    log_scale = False,
    include_p_vector = False,
    n_energy_frames = 0,
    how_to_include_p = 0,
    use_module_p = False,
    conjuntos = ["train", "val", "test"]):

    features = FeatureExtractor(response=response, 
    window_limit=window_limit, 
    umbral_corte=umbral_corte, 
    filter=filter)

    start_time = time.time()
    prefix = "acc" if use_accel else "vel"
    for conjunto in conjuntos:

        # weekly_data_dfs = []
        # weekly_df = None
        # if conjunto == "train":
        #     weekly_data_path = glob.glob("./data/weekly/**/events.csv")
        #     for path in weekly_data_path:
        #         weekly_data_dfs.append(pd.read_csv(path))
        #     weekly_df = pd.concat(weekly_data_dfs)
        #     weekly_df.rename(columns={"event": "Evento", "station": "Estacion", "date": "Fecha", "distance": "distancia",}, inplace=True)
        #     weekly_df['costero'] = 0
        #     weekly_df['weekly'] = True
        #     weekly_df = weekly_df[weekly_df['magnitude'] >= 4]

        path_carpeta_conjunto = (
             f"./data/set/{prefix}/" + conjunto + "_" + BDtype + ".csv".replace("\\", "/")
         )

        feat_out = f"./data/features/distance/{prefix}/{test_folder_name}/{test_name}/"
        if not os.path.exists(feat_out):
            os.makedirs(feat_out)

        lista_conjunto = pd.read_csv(
            path_carpeta_conjunto, delimiter=",", index_col=False
        )
        
        #lista_conjunto['weekly'] = False
        # if conjunto == "train":
        #     lista_conjunto = pd.concat([weekly_df, lista_conjunto])

        lista_eventos = lista_conjunto["event"].values
        estaciones = lista_conjunto["station"].values
        #lista_frames_P = lista_conjunto["frame_p"].values
        lista_distancia = lista_conjunto["distance"].values
        #lista_costero  = lista_conjunto["costero"].values
        #lista_izquierda = lista_conjunto["evento_izquierda_estacion"].values
        #lista_derecha = lista_conjunto["evento_derecha_estacion"].values
        p_starttime = lista_conjunto["p_starttime"].values
        lista_magnitude = lista_conjunto["magnitude"].values
        # if conjunto == "train":
        #     weekly = lista_conjunto["weekly"].values
        # else:
        #     weekly = None

        feat_por_evento_temporal, feat_por_evento_global = [], []
        dist_por_evento = {"event": [], "station": [], "distance": []} #"costero": [],"evento_izquierda_estacion": []} #, "evento_izquierda_estacion": [], "evento_derecha_estacion": []}

        for i in range(len(lista_conjunto)):  # iteracion sobre cada evento
            if test:
                if i == 10:
                    break
            print("Leyendo evento:", lista_eventos[i])

            dist_por_evento["event"].append(lista_eventos[i])
            dist_por_evento["station"].append(estaciones[i])
            dist_por_evento["distance"].append(lista_distancia[i])
            event_magnitude = lista_magnitude[i]
            #dist_por_evento["costero"].append(lista_costero[i])
            #dist_por_evento["evento_izquierda_estacion"].append(lista_izquierda[i])
            #dist_por_evento["evento_derecha_estacion"].append(lista_derecha[i])

  
            path_carpeta_evento = f"./data/sacs/{prefix}/merge_intraplaca/" + lista_eventos[i]
            path_evento = path_carpeta_evento + "/" + estaciones[i] + "/"
            try:
                canal_sac_Z = read(path_evento + estaciones[i] + "_" + "*Z.sac")
                canal_sac_E = read(path_evento + estaciones[i] + "_" + "*E.sac")
                canal_sac_N = read(path_evento + estaciones[i] + "_" + "*N.sac")
            except Exception as e:
                print(f"Error en el evento {lista_eventos[i]}, conjunto: {conjunto}, {e}")
                time.sleep(3)
                continue

            # except Exception as e:
            #     print(f"Error en el evento {lista_eventos[i]}, conjunto: {conjunto}, weekly: {weekly[i]}: {e}")
            #     time.sleep(3)
            #     continue
            
            trace = canal_sac_Z
            trace += canal_sac_E
            trace += canal_sac_N

            frame_p = p_starttime[i]
            network = trace[0].stats.network

            try:
                inv = read_inventory(f"./data/inventory/{network}_{estaciones[i]}.xml")
                
                feat_canales_temporal, feat_por_evento_mlp = features.get_features(
                    trace, UTCDateTime(frame_p), inv, one_hot_encoding = one_hot_encoding ,use_cnn = use_cnn,
                    include_index=include_index, include_env=include_env, include_lat_lon=include_lat_lon,
                    norm_energy=norm_energy, concat_features=False, square_fft=square_fft, log_scale=log_scale,
                    include_p_vector = include_p_vector, n_energy_frames=n_energy_frames, how_to_include_p = how_to_include_p,
                    use_module_p = use_module_p
                )


                if include_magnitude: 
                    
                    feat_por_evento_mlp[0] = np.concatenate((feat_por_evento_mlp[0],np.array(event_magnitude).reshape(-1)),axis=0)

            except Exception as e:
                print(f"Error en el evento {lista_eventos[i]}: {e}")
                time.sleep(1)
                continue
            print(f"LARGO DE FEAT_GLOBAL: {len(feat_por_evento_mlp[0])}")

            feat_canales_temporal = feat_canales_temporal[0]
            feat_por_evento_mlp = feat_por_evento_mlp[0]

            feat_por_evento_global.append(feat_por_evento_mlp)
            feat_por_evento_temporal.append(feat_canales_temporal)

            # print(
            #     "Dimension features LSTM: {:d}x{:d}".format(
            #         feat_canales_temporal.shape[0], feat_canales_temporal.shape[1]
            #     )
            # )
            # print("Dimension features MLP: {:d}".format(feat_por_evento_mlp.shape[0]))

            # print("***********************")

        print("Conjunto:", conjunto)
        print(
            "Se leyeron {} de un total de {} eventos".format(
                len(feat_por_evento_temporal), len(lista_conjunto)
            )
        )

        #ind_random = np.random.permutation(np.arange(0, len(feat_por_evento_temporal)))
        if conjunto == "train":
            ind_random = ind_random_train
        elif conjunto == "val":
            ind_random = ind_random_val
        elif conjunto == "test":
            ind_random = ind_random_test

        if ind_random is None:
            ind_random = np.random.permutation(np.arange(0, len(feat_por_evento_temporal)))
        
        feat_por_evento_temporal = np.array(feat_por_evento_temporal, dtype=object)[ind_random]
        feat_por_evento_global = np.array(feat_por_evento_global, dtype=object)[ind_random]
        dist_por_evento["event"] = list(
            np.array(dist_por_evento["event"])[ind_random]
        )
        dist_por_evento["station"] = list(
            np.array(dist_por_evento["station"])[ind_random]
        )
        dist_por_evento["distance"] = list(
            np.array(dist_por_evento["distance"])[ind_random]
        )

        #dist_por_evento["costero"] = list(
        #    np.array(dist_por_evento["costero"])[ind_random]
        #)

        # dist_por_evento["evento_izquierda_estacion"] = list(
        #     np.array(dist_por_evento["evento_izquierda_estacion"])[ind_random]
        # )

        # dist_por_evento["evento_derecha_estacion"] = list(
        #     np.array(dist_por_evento["evento_derecha_estacion"])[ind_random]
        # )

        ### GUARDAR CARACTERISTICAS
        np.save(
            feat_out + "feat_lstm_raw_" + conjunto + "_" + BDtype + ".npy",
            feat_por_evento_temporal,
        )
        np.save(
            feat_out + "feat_mlp_raw_" + conjunto + "_" + BDtype + ".npy",
            feat_por_evento_global,
        )
        np.save(
            feat_out + "distance_raw_" + conjunto + "_" + BDtype + ".npy",
            dist_por_evento,
        )
        print("--- %s seconds ---" % int((time.time() - start_time)))
