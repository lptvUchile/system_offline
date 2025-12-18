# Script para realizar prueba del sistema completo

# 1.- Deteccion y segmentacíon los eventos de la traza

#python -m src.modules.orchestator.detect_and_segment --sac_test_name example/sacs/CO10/CO10

# 2.- Estimacione de modelos de magnitud, epicentro y profundidad

#python -m src.modules.orchestator.models_estimation --sac_test_name example/sacs/CO10/CO10 --detection_dataframe_path results/Detection_CO10_BH*.csv --inventory_path example/inventory