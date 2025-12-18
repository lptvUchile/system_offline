import os
import os.path
import numpy as np
from obspy import UTCDateTime, read, read_inventory
import time
import scipy as sp
import pandas as pd
from src.utils import az
from src.back_azimuth.features import FeatureExtractor
from src.localization import EpicenterEstimator


# todaBD = full database, microsismicidad = microseismicity database, mayores4M = only M>=4 database
def function_features_extraction(
    BDtype="todaBD",
    test=False,
    use_accel=False,
    response="VEL",
    test_name=None,
    test_folder_name = None,
    options=[1, 3],
    filter=None,
    ind_random_train=None,
    ind_random_val=None,
    ind_random_test=None,
    one_hot_encoding=False
):
    if test_name is None:
        raise Exception("test_name is required")
    start_time = time.time()
    path_root = os.getcwd()

    conjuntos = ["train", "val", "test"]
    target = "Angulo"

    nro_segundos_ventana_izq = 1
    nro_segundos_ventana_der = 3

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

    features = FeatureExtractor(
        response=response.upper(),
        window_limit=options,
        filter=filter
    )
    epicenter_estimator = EpicenterEstimator("./data/datos_estaciones.yaml")
    prefix = "acc" if use_accel else "vel"
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
        feat_out = f"./data/features/back_azimuth/{prefix}/{test_folder_name}/{test_name}/"
        
        if not os.path.exists(feat_out):
            os.makedirs(feat_out)

        lista_conjunto = pd.read_csv(
            path_carpeta_conjunto, delimiter=",", index_col=False
        )
        lista_conjunto = lista_conjunto.drop_duplicates()
        lista_eventos = lista_conjunto["Evento"].values
        estaciones = lista_conjunto["Estacion"].values
        backazimuth = lista_conjunto["backazimuth"].values
        distancia = lista_conjunto["distancia"].values
        costero = lista_conjunto["costero"].values
        izquierda = lista_conjunto["evento_izquierda_estacion"].values
        derecha = lista_conjunto["evento_derecha_estacion"].values
        lista_frames_P = lista_conjunto[
            "frame_p"
        ].values  # Manually labelled P-wave arrival count considering the waveform in 40 Hz.

        p_starttime = lista_conjunto["p_starttime"].values

        feat_por_evento_temporal, feat_por_evento_global = [], []
        angulo_por_evento = {"Evento": [], "Estacion": [], "Coseno": [], "Seno": [], "costero": [], "evento_izquierda_estacion": [], "evento_derecha_estacion": []}

        for i in range(len(lista_conjunto)):  # iteracion sobre cada evento
            if test:
                if i == 10:
                    break
            print("Leyendo evento:", lista_eventos[i])

            feat_estaciones = []

            path_carpeta_evento = f"./data/sacs/{prefix}/merge/" + lista_eventos[i]
            path_evento = path_carpeta_evento + "/" + estaciones[i] + "/"

            ### emparejar canales para que inicien y terminen en el mismo tiempo
            canal_sac_Z = read(path_evento + estaciones[i] + "_" + "*Z.sac")
            canal_sac_E = read(path_evento + estaciones[i] + "_" + "*E.sac")
            canal_sac_N = read(path_evento + estaciones[i] + "_" + "*N.sac")

            trace = canal_sac_Z
            trace += canal_sac_E
            trace += canal_sac_N

            ### calcular back-azimuth
            coords = epicenter_estimator.get_epicenter(
                distancia[i], backazimuth[i], estaciones[i]
            )
            lat, lon = coords[0], coords[1]
            stla, stlo, stal = Coordenadas_estaciones[canal_sac_Z[0].stats.station]
            cos_k, sen_k, az_k = az(stla, stlo, lat, lon, rad=True)
            angulo_por_evento["Coseno"].append(cos_k)
            angulo_por_evento["Seno"].append(sen_k)
            angulo_por_evento["Evento"].append(lista_eventos[i])
            angulo_por_evento["Estacion"].append(estaciones[i])
            angulo_por_evento["costero"].append(costero[i])
            angulo_por_evento["evento_izquierda_estacion"].append(izquierda[i])
            angulo_por_evento["evento_derecha_estacion"].append(lista_eventos[i])


            network = trace[0].stats.network

            inv = read_inventory(f"./data/inventory/{network}_{estaciones[i]}.xml")
            # lista_frames_P[i]
            feat_temp, feat_global = features.get_features(
                trace, UTCDateTime(p_starttime[i]), inv, one_hot_encoding=one_hot_encoding
            )
            feat_temp = feat_temp[0]
            feat_global = feat_global[0]
            feat_por_evento_temporal.append(feat_temp)
            feat_por_evento_global.append(feat_global)

            print(
                "Dimension features CNN: {:d}x{:d}".format(
                    feat_temp.shape[0], feat_temp.shape[1]
                )
            )
            print("Dimension features MLP: {:d}".format(feat_global.shape[0]))

            print("***********************")

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
        else:
            ind_random = ind_random_test
        feat_por_evento_temporal = np.array(feat_por_evento_temporal, dtype=object)[ind_random]
        feat_por_evento_global = np.array(feat_por_evento_global, dtype=int)[ind_random]
        angulo_por_evento["Evento"] = list(
            np.array(angulo_por_evento["Evento"])[ind_random]
        )
        angulo_por_evento["Estacion"] = list(
            np.array(angulo_por_evento["Estacion"])[ind_random]
        )
        angulo_por_evento["Coseno"] = list(
            np.array(angulo_por_evento["Coseno"])[ind_random]
        )
        angulo_por_evento["Seno"] = list(
            np.array(angulo_por_evento["Seno"])[ind_random]
        )
        angulo_por_evento["costero"] = list(
            np.array(angulo_por_evento["costero"])[ind_random]
        )
        angulo_por_evento["evento_izquierda_estacion"] = list(
            np.array(angulo_por_evento["evento_izquierda_estacion"])[ind_random]
        )
        angulo_por_evento["evento_derecha_estacion"] = list(
            np.array(angulo_por_evento["evento_derecha_estacion"])[ind_random]
        )

        ### GUARDAR CARACTERISTICAS
        np.save(
            feat_out + "feat_cnn_raw_" + conjunto + "_" + BDtype + ".npy",
            feat_por_evento_temporal,
        )
        np.save(
            feat_out + "feat_mlp_raw_" + conjunto + "_" + BDtype + ".npy",
            feat_por_evento_global,
        )
        np.save(
            feat_out + "angulo_raw_" + conjunto + "_" + BDtype + ".npy",
            angulo_por_evento,
        )
    print("--- %s seconds ---" % int((time.time() - start_time)))
