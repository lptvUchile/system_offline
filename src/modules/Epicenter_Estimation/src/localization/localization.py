from src.distance import DistanceModel, DistanceFeatureExtractor
from src.back_azimuth import BackAzimuthModel, BackAzimuthFeatureExtractor
import yaml 
from yaml import SafeLoader
from geographiclib.geodesic import Geodesic
class EpicenterEstimator():
    def __init__(self, stations_path, dist_model_path=None, dist_max_params_path=None, dist_min_params_path=None, dist_max_target_path=None, ba_model_path=None, options=None):
        self.options= {
            "back-azimuth": {
                "global_feat": True,
                "response": "VEL",
                "fix_p": True
            },
            "distance": {
                "global_feat": True,
                "response": "VEL",
                "fix_p": True
            }
        }
        if options is not None:
            if options.get('back-azimuth', None) is not None:
                self.options['back-azimuth'].update(options['back-azimuth'])
            if options.get('distance', None) is not None:
                self.options['distance'].update(options['distance'])
            print(self.options)
        with open(stations_path) as data_st_file:
            self.data_st = yaml.load(data_st_file,Loader=SafeLoader)
        # check if any of the models is None
        if not (dist_model_path is None or dist_max_params_path is None or dist_min_params_path is None or dist_max_target_path is None or ba_model_path is None):
            self.model_enabled = True
            self.dist_model_path = dist_model_path
            self.dist_max_params_path = dist_max_params_path
            self.dist_min_params_path = dist_min_params_path
            self.dist_max_target_path = dist_max_target_path
            self.ba_model_path = ba_model_path
            self.setup_ba_model()
            self.setup_dist_model()
        else:
            self.model_enabled = False
            print("Inicializando sin modelos")
    
    def setup_dist_model(self):
        if not self.model_enabled:
            raise Exception("Modelos no habilitados")
        self.dist_model = DistanceModel()
        self.dist_model.load_model(self.dist_model_path, self.dist_max_params_path, self.dist_min_params_path, self.dist_max_target_path)
    
    def setup_ba_model(self):
        if not self.model_enabled:
            raise Exception("Modelos no habilitados")
        self.ba_model = BackAzimuthModel()
        self.ba_model.load_model(self.ba_model_path)

    def get_epicenter(self, dist, baz, station):
        dist_m = dist*1000

        geo_dict = Geodesic.WGS84.Direct(
                lat1=self.data_st[station][0],
                lon1=self.data_st[station][1],
                azi1=baz,
                s12=dist_m
            )

        lat = geo_dict['lat2']
        lon = geo_dict['lon2']

        return [lat, lon]

    def estimate_epicenter(self, trace, frame_p, inv):
        if not self.model_enabled:
            raise Exception("Modelos no habilitados")
        station = trace[0].stats.station
        dist_feature_extractor = DistanceFeatureExtractor(response=self.options["distance"]["response"] )
        temporal_feat, mlp_feat = dist_feature_extractor.get_features(trace, frame_p=frame_p, inv=inv, fix_p=self.options["distance"]["fix_p"])
        try: 
            if self.options["distance"]["global_feat"]:
                dist = self.dist_model.predict(temporal_feat, mlp_feat)
            else:
                dist = self.dist_model.predict(temporal_feat)
        except ValueError as e:
            print("Error en la predicción de la distancia: ¿Está bien configurado el parámetro del modelo?")
            print(self.options["distance"])
            raise(e)
        
        ba_feature_extractor = BackAzimuthFeatureExtractor(response=self.options["back-azimuth"]["response"])
        temporal_feat, mlp_feat = ba_feature_extractor.get_features(trace, frame_p=frame_p, inv=inv, fix_p=self.options["back-azimuth"]["fix_p"])
        try:
            if self.options["back-azimuth"]["global_feat"]:
                ba = self.ba_model.predict(temporal_feat, mlp_feat)
            else:
                ba = self.ba_model.predict(temporal_feat)
        except ValueError as e:
            print("Error en la predicción del backazimuth: ¿Está bien configurado el parámetro del modelo?")
            print(self.options["back-azimuth"])
            raise(e)
        lat, lon = self.get_epicenter(dist, ba, station)
        
        return [lat, lon], dist, ba

