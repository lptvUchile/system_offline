import os
import os.path
import numpy as np
from obspy import read, read_inventory
import time
import scipy as sp
import pandas as pd
from src.utils import az
from src.back_azimuth.features import FeatureExtractor as FeatureExtractorBaz
from src.localization import EpicenterEstimator

import math
import time

from src.distance.features import FeatureExtractor as FeatureExtractorDist


def function_features_extraction(
    BDtype="todaBD", 
    test=False, 
    use_accel=False, 
    response="VEL", 
    window_limit_dist=[20, 120],
    options=[1, 10],
    umbral_corte=0.03, 
    filter_dist=None, 
    filter_baz=None,
    test_name=None,
    ind_random_train=None,
    ind_random_val=None,
    ind_random_test=None):

    features_dist = FeatureExtractorDist(response=response, window_limit=window_limit_dist, umbral_corte=umbral_corte, filter=filter_dist)
    features_baz = FeatureExtractorBaz(response=response.upper(), window_limit=options, filter=filter_baz)
    epicenter_estimator = EpicenterEstimator("./data/datos_estaciones.txt")


    if test_name is None:
        raise Exception("test_name is required")
    start_time = time.time()
    path_root = os.getcwd()

    prefix = "acc" if use_accel else "vel"
    conjuntos = ["train", "val", "test"]

    Coordenadas_estaciones = {
        "PB09": [-21.7964, -69.2419, 1.530],
        "PB06": [-22.7058, -69.5719, 1.440],
        "AC02": [-26.8355, -69.1291, 3.980],
        "CO02": [-31.2037, -71.0003, 1.190],
        "PB14": [-24.6260, -70.4038, 2.630],
        "CO01": [-29.9773, -70.0939, 2.157],
        "GO01": [-19.6685, -69.1942, 3.809],
        "GO03": [-27.5937, -70.2347, 0.730],
        "PB18": [-17.5895, -69.480, 4.155],
        "MT16": [-33.4285, -70.5234, 0.780],
        "AC04": [-28.2046, -71.0739, 0.228],
        "AC05": [-28.8364, -70.2738, 1.227],
        "AP01": [-18.3708, -70.342, 0.031],
        "CO03": [-30.8389, -70.6891, 1.003],
        "GO04": [-30.1727, -70.7993, 2.076],
        "HMBCX": [-20.2782, -69.8879, 1.152],
        "MNMCX": [-19.1311, -69.5955, 2.304],
        "MT02": [-33.2591, -71.1377, 0.323],
        "MT03": [-33.4936, -70.5102, 1.087],
        "MT05": [-33.3919, -70.7381, 0.765],
        "PATCX": [-20.8207, -70.1529, 0.832],
        "PB01": [-21.0432, -69.4874, 0.9],
        "PB02": [-21.3197, -69.896, 1.015],
        "PB03": [-22.0485, -69.7531, 1.46],
        "PB04": [-22.3337, -70.1492, 1.52],
        "PB05": [-22.8528, -70.2024, 1.15],
        "PB07": [-21.7267, -69.8862, 1.57],
        "PB10": [-23.5134, -70.5541, 0.25],
        "PB11": [-19.761, -69.6558, 1.4],
        "PB12": [-18.6141, -70.3281, 0.908],
        "PB15": [-23.2083, -69.4709, 1.83],
        "PSGCX": [-19.5972, -70.1231, 0.966],
        "TA01": [-20.5656, -70.1807, 0.075],
        "TA02": [-20.2705, -70.1311, 0.0865],
        "VA03": [-32.7637, -70.5508, 1.053],
        "GO02": [-25.1626, -69.5904, 2.550],
        "GO05": [-35.0099, -71.9303, 0.488],
        "PB16": [-18.3351, -69.5077, 4.480],
        "PB08": [-20.1411, -69.1534, 3.060],
        "CO04": [-32.0433, -70.9747, 2.401],
        "VA01": [-33.0228, -71.6475, 0.0756],
        "AC01": [-26.1479, -70.5987, 0.346],
        "CO05": [-29.9186, -71.2384, 0.101],
        "CO06": [-30.6738, -71.6350, 0.2466],
        "VA06": [-32.5612, -71.2977, 0.080],
        "CO10": [-29.2406, -71.4609, 0.035],
        "BO03": [-34.4961, -71.9612, 0.128],
        "AC07": [-27.1297, -70.8602, 0.072],
        "PX06": [-23.5115, -70.2495, 0.700],
    }

    for conjunto in conjuntos:
        path_carpeta_conjunto = (
            f"./data/set/{prefix}/"
            + conjunto
            + "_20220927_"
            + BDtype
            + ".csv".replace("\\", "/")
        )
        path_carpeta_sac = (
            f"./data/sacs/{prefix}/Events_refstations_and_microseismicity/".replace(
                "\\", "/"
            )
        )
        path_carpeta_sac_sec = f"./data/sacs/{prefix}/Events_secstations/".replace(
            "\\", "/"
        )        

        feat_out_dist = f"./data/features/distance/{prefix}/{test_name}/"
        feat_out_baz = f"./data/features/back_azimuth/{prefix}/{test_name}/"
        if not os.path.exists(feat_out_dist) or not os.path.exists(feat_out_baz):
            os.makedirs(feat_out_dist)
            os.makedirs(feat_out_baz)

        lista_conjunto = pd.read_csv(
            path_carpeta_conjunto, delimiter=",", index_col=False
        )
        
        lista_eventos = lista_conjunto["Evento"].values
        estaciones = lista_conjunto["Estacion"].values
        lista_frames_P = lista_conjunto["frame_p"].values
        origen = lista_conjunto["Catalina?"].values
        lista_distancia = lista_conjunto["distancia"].values
        lista_costero  = lista_conjunto["costero"].values
        lista_izquierda = lista_conjunto["evento_izquierda_estacion"].values
        lista_distancia = lista_conjunto["distancia"].values
        lista_backazimuth = lista_conjunto["backazimuth"].values

        feat_por_evento_temporal_dist, feat_por_evento_global_dist = [], []
        dist_por_evento = {"Evento": [], "Estacion": [], "distancia": [], "costero": [], "evento_izquierda_estacion": []}

        feat_por_evento_temporal_baz, feat_por_evento_global_baz = [], []
        angulo_por_evento = {"Evento": [], "Estacion": [], "Coseno": [], "Seno": [], "costero": [], "evento_izquierda_estacion": []}

        for i in range(len(lista_conjunto)):  # iteracion sobre cada evento
            if test:
                if i == 10:
                    break
            print("Leyendo evento:", lista_eventos[i])

            if origen[i] == "Marcelo":
                path_carpeta_evento = path_carpeta_sac_sec + lista_eventos[i]
            else:
                path_carpeta_evento = path_carpeta_sac + lista_eventos[i]

            path_evento = path_carpeta_evento + "/" + estaciones[i] + "/"

            dist_por_evento["Evento"].append(lista_eventos[i])
            dist_por_evento["Estacion"].append(estaciones[i])
            dist_por_evento["distancia"].append(lista_distancia[i])
            dist_por_evento["costero"].append(lista_costero[i])
            dist_por_evento["evento_izquierda_estacion"].append(lista_izquierda[i])



            ### emparejar canales para que inicien y terminen en el mismo tiempo
            canal_sac_Z = read(path_evento + estaciones[i] + "_" + "*Z.sac")
            canal_sac_E = read(path_evento + estaciones[i] + "_" + "*E.sac")
            canal_sac_N = read(path_evento + estaciones[i] + "_" + "*N.sac")

            trace = canal_sac_Z
            trace += canal_sac_E
            trace += canal_sac_N

            ### calcular back-azimuth
            coords = epicenter_estimator.get_epicenter(
                lista_distancia[i], lista_backazimuth[i], estaciones[i]
            )
            lat, lon = coords[0], coords[1]
            stla, stlo, stal = Coordenadas_estaciones[canal_sac_Z[0].stats.station]
            cos_k, sen_k, az_k = az(stla, stlo, lat, lon, rad=True)
            angulo_por_evento["Coseno"].append(cos_k)
            angulo_por_evento["Seno"].append(sen_k)
            angulo_por_evento["Evento"].append(lista_eventos[i])
            angulo_por_evento["Estacion"].append(estaciones[i])
            angulo_por_evento["costero"].append(lista_costero[i])
            angulo_por_evento["evento_izquierda_estacion"].append(lista_izquierda[i])


            network = trace[0].stats.network

            inv = read_inventory(f"./data/inventory/{network}_{estaciones[i]}.xml")

            feat_temp_dist, feat_global_dist = features_dist.get_features(
                trace, lista_frames_P[i], inv
            )
            feat_temp_dist = feat_temp_dist[0]
            feat_global_dist = feat_global_dist[0]
            feat_por_evento_global_dist.append(feat_global_dist)
            feat_por_evento_temporal_dist.append(feat_temp_dist) 


            feat_temp_baz, feat_global_baz = features_baz.get_features(
                trace, lista_frames_P[i], inv
            )
            feat_temp_baz = feat_temp_baz[0]
            feat_global_baz = feat_global_baz[0]
            feat_por_evento_temporal_baz.append(feat_temp_baz)
            feat_por_evento_global_baz.append(feat_global_baz)

            print(
                "Dimension features LSTM: {:d}x{:d}".format(
                    feat_temp_dist.shape[0], feat_temp_dist.shape[1]
                )
            )
            print("Dimension features MLP: {:d}".format(feat_global_dist.shape[0]))

            print("***********************")


            print(
                "Dimension features CNN: {:d}x{:d}".format(
                    feat_temp_baz.shape[0], feat_temp_baz.shape[1]
                )
            )
            print("Dimension features MLP: {:d}".format(feat_global_baz.shape[0]))

            print("***********************")

        print("Conjunto:", conjunto)
        print(
            "Se leyeron {} de un total de {} eventos".format(
                len(feat_por_evento_temporal_dist), len(lista_conjunto)
            )
        )

        print("Conjunto:", conjunto)
        print(
            "Se leyeron {} de un total de {} eventos".format(
                len(feat_por_evento_temporal_baz), len(lista_conjunto)
            )
        )

        #ind_random = np.random.permutation(np.arange(0, len(feat_por_evento_temporal)))
        if conjunto == "train":
            ind_random = ind_random_train
        elif conjunto == "val":
            ind_random = ind_random_val
        else:
            ind_random = ind_random_test
        
        # Guardar DIST
        feat_por_evento_temporal_dist = np.array(feat_por_evento_temporal_dist, dtype=object)#[ind_random]
        feat_por_evento_global_dist = np.array(feat_por_evento_global_dist, dtype=object)#[ind_random]
        dist_por_evento["Evento"] = list(
            np.array(dist_por_evento["Evento"])#[ind_random]
        )
        dist_por_evento["Estacion"] = list(
            np.array(dist_por_evento["Estacion"])#[ind_random]
        )
        dist_por_evento["distancia"] = list(
            np.array(dist_por_evento["distancia"])#[ind_random]
        )
        dist_por_evento["costero"] = list(
            np.array(dist_por_evento["costero"])#[ind_random]
        )

        dist_por_evento["evento_izquierda_estacion"] = list(
            np.array(dist_por_evento["evento_izquierda_estacion"])#[ind_random]
        )

        ### GUARDAR CARACTERISTICAS
        np.save(
            feat_out_dist + "feat_lstm_raw_" + conjunto + "_" + BDtype + ".npy",
            feat_por_evento_temporal_dist,
        )
        np.save(
            feat_out_dist + "feat_mlp_raw_" + conjunto + "_" + BDtype + ".npy",
            feat_por_evento_global_dist,
        )
        np.save(
            feat_out_dist + "distance_raw_" + conjunto + "_" + BDtype + ".npy",
            dist_por_evento,
        )


        # Guardar BAZ
        feat_por_evento_temporal_baz = np.array(feat_por_evento_temporal_baz, dtype=object)#[ind_random]
        feat_por_evento_global_baz = np.array(feat_por_evento_global_baz, dtype=int)#[ind_random]
        angulo_por_evento["Evento"] = list(
            np.array(angulo_por_evento["Evento"])#[ind_random]
        )
        angulo_por_evento["Estacion"] = list(
            np.array(angulo_por_evento["Estacion"])#[ind_random]
        )
        angulo_por_evento["Coseno"] = list(
            np.array(angulo_por_evento["Coseno"])#[ind_random]
        )
        angulo_por_evento["Seno"] = list(
            np.array(angulo_por_evento["Seno"])#[ind_random]
        )
        angulo_por_evento["costero"] = list(
            np.array(angulo_por_evento["costero"])#[ind_random]
        )
        angulo_por_evento["evento_izquierda_estacion"] = list(
            np.array(angulo_por_evento["evento_izquierda_estacion"])#[ind_random]
        )

        ### GUARDAR CARACTERISTICAS
        np.save(
            feat_out_baz + "feat_cnn_raw_" + conjunto + "_" + BDtype + ".npy",
            feat_por_evento_temporal_baz,
        )
        np.save(
            feat_out_baz + "feat_mlp_raw_" + conjunto + "_" + BDtype + ".npy",
            feat_por_evento_global_baz,
        )
        np.save(
            feat_out_baz + "angulo_raw_" + conjunto + "_" + BDtype + ".npy",
            angulo_por_evento,
        )
    print("--- %s seconds ---" % int((time.time() - start_time)))
