import numpy as np
import pandas as pd
import tensorflow as tf
from shapely.geometry import Point, Polygon, LinearRing, LineString
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
# Project Utilities and Custom Modules
import sys
sys.path.append('././')
from src.utils import (VanillaPositionalEncoding, to_angle, sec_div_max, 
                       pad_and_convert_to_tensor)
import os  

# En la función load_features
def load_features(prefix, test_name, feat_type, feature_kind, dataset_type, category):
    # Ajusta el path base al de la raíz del proyecto usando `os.getenv`
    project_base = os.getenv("PROJECT_BASE", "./Epicenter_Estimation/")
    path = f"{project_base}data/features/{category}/{prefix}/{test_name}/feat_{feature_kind}_raw_{dataset_type}_{feat_type}.npy".replace("\\", "/")
    return np.load(path, allow_pickle=True)

def normalize_sec_div_max(features):
    """Normaliza los features usando sec_div_max."""
    return sec_div_max(features.astype("float"))

def normalize_min_max(features, min_values, max_values):
    """Normaliza los features con Min-Max Scaling de acuerdo con cada dimensión individual."""
    return np.array([
        (x - min_values) / (max_values - min_values) for x in features
    ], dtype=object)


# Función para cargar las etiquetas y datos adicionales
def load_labels(prefix, test_name, feat_type, dataset_type):
    """Cargar las etiquetas y datos adicionales como id de eventos y estaciones."""
    path = f"../data/features/distance/{prefix}/{test_name}/distance_raw_{dataset_type}_{feat_type}.npy".replace("\\", "/")
    dist_real = np.load(path, allow_pickle=True)
    dist_real_df = pd.DataFrame(dist_real[()])
    
    event_ids = dist_real_df['Evento']
    stations = dist_real_df['Estacion']
    distance = dist_real_df['distancia']
    labels = dist_real_df["costero"].values
    izq = dist_real_df["evento_izquierda_estacion"].values
    der = dist_real_df["evento_derecha_estacion"].values
    
    return event_ids, stations, distance, labels, izq, der

def load_all_data(prefix, test_name, feat_type):
    datasets = ['train', 'val', 'test']
    feature_kinds_dist = ['lstm', 'mlp']
    feature_kinds_baz = ['cnn', 'mlp']  # Cambiamos "lstm" por "cnn" para back azimuth
    features_dist = {}
    features_baz = {}
    labels_data = {}

    # Cargar features para distance y back_azimuth
    for dataset in datasets:
        for kind in feature_kinds_dist:
            dist_key = f"{dataset}_{kind}_dist"
            features_dist[dist_key] = load_features(prefix, test_name, feat_type, kind, dataset, category="distance")
        
        for kind in feature_kinds_baz:
            baz_key = f"{dataset}_{kind}_baz"
            features_baz[baz_key] = load_features(prefix, test_name, feat_type, kind, dataset, category="back_azimuth")
    
    # Normalización de los features LSTM y CNN
    # Calcula el mínimo y máximo para normalización de Min-Max solo una vez
    train_lstm_dist_key = 'train_lstm_dist'
    train_cnn_baz_key = 'train_cnn_baz'
    
    min_f_train_lstm_dist = np.min([np.min(x, 0) for x in features_dist[train_lstm_dist_key]], 0)
    max_f_train_lstm_dist = np.max([np.max(x, 0) for x in features_dist[train_lstm_dist_key]], 0)
    
    for dataset in datasets:
        dist_key = f"{dataset}_lstm_dist"
        baz_key = f"{dataset}_cnn_baz"  # Aquí cambiamos a "cnn" en lugar de "lstm" para back azimuth
        features_dist[dist_key] = normalize_min_max(features_dist[dist_key], min_f_train_lstm_dist, max_f_train_lstm_dist)
        features_baz[baz_key] = normalize_sec_div_max(features_baz[baz_key])

    # Convertir MLP features a float32
    for dataset in datasets:
        dist_key = f"{dataset}_mlp_dist"
        baz_key = f"{dataset}_mlp_baz"
        features_dist[dist_key] = np.array([features_dist[dist_key][i] for i in range(len(features_dist[dist_key]))], dtype="float32")
        features_baz[baz_key] = np.array([features_baz[baz_key][i] for i in range(len(features_baz[baz_key]))], dtype="float32")

    # Cargar etiquetas (labels), eventos y estaciones
    for dataset in datasets:
        event_ids, stations, distance, labels, izq, der = load_labels(prefix, test_name, feat_type, dataset)
        labels_data[dataset] = {
            'Evento': event_ids,
            'Estacion': stations,
            'distancia': distance,
            'costero': labels,
            'izq': izq,
            'der': der
        }

    # Extraer tamaños de los features
    largo_cota_dist = None #features_dist['train_lstm_dist'].shape[1]
    tam_feat_mlp_dist = features_dist['train_mlp_dist'].shape[1]
    tam_feat_lstm_dist = features_dist['train_lstm_dist'][0].shape[1]

    largo_cota_baz = features_baz['train_cnn_baz'].shape[1]  # Ajustado para "cnn"
    tam_feat_mlp_baz = features_baz['train_mlp_baz'].shape[-1]
    tam_feat_lstm_baz = features_baz['test_cnn_baz'].shape[-1]  # Ajustado para "cnn"

    return (features_dist, features_baz, labels_data, 
            largo_cota_dist, tam_feat_mlp_dist, tam_feat_lstm_dist, 
            largo_cota_baz, tam_feat_mlp_baz, tam_feat_lstm_baz)



def evaluate_model(model, X_test_lstm_dist, X_test_attention_baz, X_test_mlp_dist, 
                   X_test_mlp_baz, dist_real_test: pd.DataFrame, type = "combined"):
    if type == "combined":
        y_prob = model.predict([X_test_lstm_dist, X_test_attention_baz, X_test_mlp_dist])
        y_prob = np.hstack(y_prob)
    elif type == "dist":
        y_prob = model.predict([X_test_lstm_dist, X_test_mlp_dist])
        y_prob = np.hstack(y_prob)
    elif type == "baz":
        y_prob = model.predict([X_test_attention_baz, X_test_mlp_baz])
        y_prob = np.hstack(y_prob)

    y_pred = np.where(y_prob >= 0.5, 1, 0)
    y_true = dist_real_test["costero"].values

    df_pred_test = dist_real_test.copy()
    df_pred_test['prediccion'] = y_pred

    # Filtrar eventos donde las predicciones no coinciden con los valores reales
    df_filtered = df_pred_test[df_pred_test['costero'] != df_pred_test['prediccion']]

    # Leer el DataFrame de test con eventos
    df_test_path = "../data/set/acc/test_20220927_mayores4M.csv"
    df_test = pd.read_csv(df_test_path)

    # Realizar el merge considerando tanto 'Evento' como 'Estacion' para evitar duplicados incorrectos
    df_merged_test = df_pred_test.merge(df_test[['Evento', 'Estacion', 'lat', 'lon', 'Magnitud', 'backazimuth']],
                                        on=['Evento', 'Estacion'],
                                        how='left')
    df_merged_test['estimacion'] = y_prob
    # Retornar los eventos filtrados y el DataFrame merged final
    return df_filtered["Evento"].values, df_merged_test



'''
def plot_station_events_with_errors(df_model, df_stations, station_list=None, polygon_gdf=None, buffered_polygon_gdf=None, plot_errors=True):
    """
    Plots seismic events and stations with options to filter by stations and to plot only error events.
    
    Args:
    df_model (DataFrame): DataFrame containing event data with latitude and longitude.
    df_stations (DataFrame): DataFrame containing station data with latitude, longitude, and elevation.
    station_list (list): List of station names to filter by. If None, all stations are considered.
    polygon_gdf (GeoDataFrame): GeoDataFrame containing the coastal polygon in EPSG:4326.
    buffered_polygon_gdf (GeoDataFrame): GeoDataFrame containing the buffered coastal polygon in EPSG:4326.
    plot_errors (bool): If True, only plot events where the prediction differs from the real label.
    """
    # Filtrar por eventos con errores si se especifica
    if plot_errors:
        df_filtered = df_model[df_model['costero'] != df_model['prediccion']]
    else:
        df_filtered = df_model.copy()
    
    # Filtrar por estaciones si se proporciona una lista de estaciones
    if station_list is not None:
        df_filtered = df_filtered[df_filtered['Estacion'].isin(station_list)]
    
    # Hacer merge con los datos de las estaciones
    df_filtered = df_filtered.merge(df_stations, left_on='Estacion', right_on='Station')
    
    # Convertir los eventos filtrados a un GeoDataFrame
    event_geometry = [Point(xy) for xy in zip(df_filtered['lon'], df_filtered['lat'])]
    gdf_events = gpd.GeoDataFrame(df_filtered, geometry=event_geometry, crs="EPSG:4326")
    
    # Convertir las estaciones a un GeoDataFrame
    station_geometry = [Point(xy) for xy in zip(df_filtered['Longitude'], df_filtered['Latitude'])]
    gdf_stations = gpd.GeoDataFrame(df_filtered, geometry=station_geometry, crs="EPSG:4326")
    
    # Preparar la configuración del gráfico
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Trazar el mapa base
    try:
        ctx.add_basemap(ax, crs=gdf_events.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        print(f"Error al agregar el mapa base: {e}")
    
    # Trazar el polígono costero
    if polygon_gdf is not None:
        polygon_gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2, zorder=3, label='Polígono Costero')
    
    # Trazar el buffer de 15 km
    if buffered_polygon_gdf is not None:
        buffered_polygon_gdf.plot(ax=ax, edgecolor='orange', facecolor='none', linewidth=1, linestyle='--', zorder=4, label='Buffer 15 km')
    
    # Trazar las estaciones como puntos morados
    gdf_stations.plot(ax=ax, color='purple', marker='*', markersize=40, label='Estaciones', zorder=5)
    
    # Trazar los eventos (costero en verde, no costero en rojo)
    gdf_events[gdf_events['costero'] == 1].plot(ax=ax, color='green', markersize=10, label='Predicho No Costero', zorder=2)
    gdf_events[gdf_events['costero'] == 0].plot(ax=ax, color='red', markersize=10, label='Predicho Costero', zorder=2)
    
    # Añadir nombres de estaciones
    for _, row in df_filtered.iterrows():
        ax.text(row['Longitude'], row['Latitude'], row['Station'], fontsize=8, ha='right', color='purple')
    
    # Trazar líneas conectando estaciones y eventos
    for _, row in df_filtered.iterrows():
        station_point = Point(row['Longitude'], row['Latitude'])
        event_point = Point(row['lon'], row['lat'])
        line = LineString([station_point, event_point])
        gpd.GeoSeries([line], crs="EPSG:4326").plot(ax=ax, color='black', linestyle='--', linewidth=0.5, zorder=1)

    for _, row in df_filtered.iterrows():
        ax.text(row['lon'], row['lat'], str(round(row['estimacion_mean'], 3)),
                fontsize=8, ha='left', color='black', weight='bold', rotation=45)


    
    # Ajustar los límites del gráfico para incluir todos los datos
    bounds = gdf_events.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_xlim([bounds[0] - 0.5, bounds[2] + 1])  # Añadir un pequeño margen en X
    ax.set_ylim([bounds[1] - 0.5, bounds[3] + 1])  # Añadir un pequeño margen en Y
    
    # Eliminar los ejes para una visualización más limpia
    #ax.set_axis_off()

    # Añadir la leyenda
    plt.legend(fontsize=8, loc='upper right')
    
    # Ajustar la presentación
    plt.tight_layout()
    plt.show()
'''

def plot_station_events_with_errors(df_model, df_stations, station_list=None, polygon_gdf=None, buffered_polygon_gdf=None, plot_errors=True):
    """
    Plots seismic events and stations with options to filter by stations and to plot only error events.
    
    Args:
    df_model (DataFrame): DataFrame containing event data with latitude and longitude.
    df_stations (DataFrame): DataFrame containing station data with latitude, longitude, and elevation.
    station_list (list): List of station names to filter by. If None, all stations are considered.
    polygon_gdf (GeoDataFrame): GeoDataFrame containing the coastal polygon in EPSG:4326.
    buffered_polygon_gdf (GeoDataFrame): GeoDataFrame containing the buffered coastal polygon in EPSG:4326.
    plot_errors (bool): If True, only plot events where the prediction differs from the real label.
    """
    # Filtrar por eventos con errores si se especifica
    if plot_errors:
        df_filtered = df_model[df_model['costero'] != df_model['prediccion']]
    else:
        df_filtered = df_model.copy()
    
    # Filtrar por estaciones si se proporciona una lista de estaciones
    if station_list is not None:
        df_filtered = df_filtered[df_filtered['Estacion'].isin(station_list)]
    
    # Hacer merge con los datos de las estaciones
    df_filtered = df_filtered.merge(df_stations, left_on='Estacion', right_on='Station')
    
    # Convertir los eventos filtrados a un GeoDataFrame
    event_geometry = [Point(xy) for xy in zip(df_filtered['lon'], df_filtered['lat'])]
    gdf_events = gpd.GeoDataFrame(df_filtered, geometry=event_geometry, crs="EPSG:4326")
    
    # Convertir las estaciones a un GeoDataFrame
    station_geometry = [Point(xy) for xy in zip(df_filtered['Longitude'], df_filtered['Latitude'])]
    gdf_stations = gpd.GeoDataFrame(df_filtered, geometry=station_geometry, crs="EPSG:4326")
    
    # Preparar la configuración del gráfico
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Trazar el mapa base
    try:
        ctx.add_basemap(ax, crs=gdf_events.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        print(f"Error al agregar el mapa base: {e}")
    
    # Trazar el polígono costero
    if polygon_gdf is not None:
        polygon_gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2, zorder=3, label='Polígono Costero')
    
    # Trazar el buffer de 15 km
    if buffered_polygon_gdf is not None:
        buffered_polygon_gdf.plot(ax=ax, edgecolor='orange', facecolor='none', linewidth=1, linestyle='--', zorder=4, label='Buffer 15 km')
    
    # Trazar las estaciones como puntos morados
    gdf_stations.plot(ax=ax, color='purple', marker='*', markersize=40, label='Estaciones', zorder=5)
    
    # Trazar los eventos (costero en verde, no costero en rojo)
    gdf_events[gdf_events['costero'] == 1].plot(ax=ax, color='green', markersize=10, label='Predicho No Costero', zorder=2)
    gdf_events[gdf_events['costero'] == 0].plot(ax=ax, color='red', markersize=10, label='Predicho Costero', zorder=2)
    
    # Añadir nombres de estaciones
    for _, row in df_filtered.iterrows():
        ax.text(row['Longitude'], row['Latitude'], row['Station'], fontsize=8, ha='right', color='purple')
    
    # Trazar líneas conectando estaciones y eventos
    for _, row in df_filtered.iterrows():
        station_point = Point(row['Longitude'], row['Latitude'])
        event_point = Point(row['lon'], row['lat'])
        line = LineString([station_point, event_point])
        gpd.GeoSeries([line], crs="EPSG:4326").plot(ax=ax, color='black', linestyle='--', linewidth=0.5, zorder=1)

    # Añadir los textos de las estimaciones al costado de las flechas, con un desplazamiento
    for _, row in df_filtered.iterrows():
        ax.text(row['lon'] + 0.01, row['lat'] + 0.01,  # Desplazamiento en coordenadas
                str(round(row['estimacion'], 3)),
                fontsize=6, ha='left', color='black', weight='bold')
        
        # Dibujar una pequeña línea que conecte el evento con el texto de la estimación
        line_to_text = LineString([Point(row['lon'], row['lat']), Point(row['lon'] + 0.1, row['lat'] + 0.1)])
        gpd.GeoSeries([line_to_text], crs="EPSG:4326").plot(ax=ax, color='gray', linestyle=':', linewidth=0.5, zorder=1)

    
    # Ajustar los límites del gráfico para incluir todos los datos
    bounds = gdf_events.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_xlim([bounds[0] - 0.5, bounds[2] + 1])  # Añadir un pequeño margen en X
    ax.set_ylim([bounds[1] - 0.5, bounds[3] + 1])  # Añadir un pequeño margen en Y
    
    # Añadir la leyenda
    plt.legend(fontsize=8, loc='upper right')
    
    # Ajustar la presentación
    plt.tight_layout()
    plt.show()

