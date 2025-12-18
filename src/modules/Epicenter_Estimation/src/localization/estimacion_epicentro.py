# -*- coding: utf-8 -*-+
from numpy import pi
import yaml 
from yaml.loader import SafeLoader
from geographiclib.geodesic import Geodesic
import os

def estimacion_epicentro(sta_lat, sta_lon,dist,baz,rad=None):
    """
    from numpy import pi
    import yaml 
    from yaml.loader import SafeLoader
    from geographiclib.geodesic import Geodesic
    station(str): nombre estacion
    dist(float): distancia en km
    baz(float): backazimuth
    rad(bool): baz en radianes
    """ 
   

    if rad == True:
        baz = baz*(180/pi)
    dist_m = dist*1000

    geo_dict = Geodesic.WGS84.Direct(
        lat1=sta_lat,
        lon1=sta_lon,
        azi1=baz,
        s12=dist_m
        )

    lat = geo_dict['lat2']
    lon = geo_dict['lon2']

    return [lat, lon]