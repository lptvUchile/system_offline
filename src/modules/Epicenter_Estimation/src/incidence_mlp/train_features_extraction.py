import numpy as np
from obspy import UTCDateTime, read, read_inventory
import math
import time
import os
import pandas as pd
from obspy import UTCDateTime
from src.incidence_mlp.features import FeatureExtractor



def function_features_extraction(
    BDtype="todaBD",
    test=False,
    use_accel=False,
    response="VEL",
    window_limit=[20, 120],
    filter=None,
    test_name=None,
    test_folder_name=None,
    conjuntos=["train", "val", "test"],
    use_horizontal_mean=False
):


    features = FeatureExtractor(
        response=response, window_limit=window_limit, filter=filter
    )

    start_time = time.time()
    prefix = "acc" if use_accel else "vel"
    for conjunto in conjuntos:

        path_carpeta_conjunto = (
            f"./data/set/{prefix}/"
            + conjunto
            + "_"
            + BDtype
            + ".csv".replace("\\", "/")
        )

        feat_out = f"./data/features/incidence_mlp/{prefix}/{test_folder_name}/{test_name}/"
        if not os.path.exists(feat_out):
            os.makedirs(feat_out)

        lista_conjunto = pd.read_csv(
            path_carpeta_conjunto, delimiter=",", index_col=False
        )

        lista_eventos = lista_conjunto["event"].values
        estaciones = lista_conjunto["station"].values
        lista_incidence = lista_conjunto["incidence"].values
        p_starttime = lista_conjunto["p_starttime"].values

        feat_eventos_mlp = []
        dist_por_evento = {"event": [], "station": [], "incidence": []}


        for i in range(len(lista_conjunto)):  # iteracion sobre cada evento
            if test:
                if i == 10:
                    break
            print("Leyendo evento:", lista_eventos[i])

            dist_por_evento["event"].append(lista_eventos[i])
            dist_por_evento["station"].append(estaciones[i])
            dist_por_evento["incidence"].append(lista_incidence[i])

            path_carpeta_evento = (
                f"./data/sacs/{prefix}/merge_intraplaca/" + lista_eventos[i]
            )
            path_evento = path_carpeta_evento + "/" + estaciones[i] + "/"
            try:
                canal_sac_Z = read(path_evento + estaciones[i] + "_" + "*Z.sac")
                canal_sac_E = read(path_evento + estaciones[i] + "_" + "*E.sac")
                canal_sac_N = read(path_evento + estaciones[i] + "_" + "*N.sac")
            except Exception as e:
                print(
                    f"Error en el evento {lista_eventos[i]}, conjunto: {conjunto}, {e}"
                )
                time.sleep(3)
                continue

            trace = canal_sac_Z
            trace += canal_sac_E
            trace += canal_sac_N

            frame_p = p_starttime[i]
            network = trace[0].stats.network

            
            inv = read_inventory(f"./data/inventory/{network}_{estaciones[i]}.xml")

            feat_por_evento_mlp = features.get_features(
                trace,
                UTCDateTime(frame_p),
                inv,
                use_horizontal_mean=use_horizontal_mean
                )


       
            print(f"LARGO DE FEAT_MLP: {len(feat_por_evento_mlp)}")

            feat_eventos_mlp.append(feat_por_evento_mlp)

        print("Conjunto:", conjunto)
        print(
            "Se leyeron {} de un total de {} eventos".format(
                len(feat_eventos_mlp), len(lista_conjunto)
            )
        )


        ind_random = np.random.permutation(
            np.arange(0, len(feat_eventos_mlp))
        )



        feat_eventos_mlp = np.array(feat_eventos_mlp, dtype=object)[
            ind_random
        ]

        dist_por_evento["event"] = list(np.array(dist_por_evento["event"])[ind_random])
        dist_por_evento["station"] = list(
            np.array(dist_por_evento["station"])[ind_random]
        )
        dist_por_evento["incidence"] = list(
            np.array(dist_por_evento["incidence"])[ind_random]
        )

        np.save(
            feat_out + "feat_mlp_raw_" + conjunto + "_" + BDtype + ".npy",
            feat_eventos_mlp,
        )
        np.save(
            feat_out + "incidence_raw_" + conjunto + "_" + BDtype + ".npy",
            dist_por_evento,
        )
        print("--- %s seconds ---" % int((time.time() - start_time)))
