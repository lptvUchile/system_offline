# system_offline

Pipeline offline para **detección/segmentación** de eventos y **estimación** (magnitud + hipocentro + ángulo de íncidencia).

## Requisitos

- **Python**: 3.10.13
- **Dependencias**: instalar primero `requirements.txt` y luego `requirements_part_2.txt` (En ese orden)

Instalación (en un venv/conda activo):

```bash
pip install -r requirements.txt 
```

```bash
pip install -r requirements_part_2.txt
```

## Estructura de datos (demo)

El repo incluye una carpeta `example/` con una estructura mínima esperada:

```text
example/
  inventory/
    <inv_file>.xml
  sacs/
    <STA>/
      <STA>_BH*.sac
```

Notas:
- **SACs**: se espera un “prefijo” de estación (p. ej. `example/sacs/CO10/CO10`) y el código busca `*_BH*.sac`.
- **Inventarios**: el script de estimación recibe un **archivo XML** (ruta directa) vía `--inventory_path` y lo carga con `obspy.read_inventory(...)`.

## Ejecución (demo completa)

Desde la raíz del repo:

```bash
python example.py
```

Esto ejecuta, en orden:

1) `src.modules.orchestator.detect_and_segment`
2) `src.modules.orchestator.models_estimation`

## Ejecución por etapas

### 1) Detección + segmentación

```bash
python -m src.modules.orchestator.detect_and_segment \
  --sac_test_name example/sacs/CO10/CO10 \
  --detection_output_path results
```

Outputs típicos en `results/`:
- `Detection_<prefijo>_BH*.ctm`
- `Detection_<prefijo>_BH*.csv`

### 2) Estimación (magnitud, hipocentro y ángulo)

```bash
python -m src.modules.orchestator.models_estimation \
  --sac_test_name example/sacs/CO10/CO10 \
  --detection_dataframe_path "results/Detection_CO10_BH*.csv" \
  --inventory_path example/inventory/C1_CO10.xml
```

Output:
- `results/models_estimation_<prefijo>_BH*.csv`

## Modelos usados (paths actuales)

Los scripts de orquestación cargan modelos desde `src/models/`.


## Troubleshooting rápido

- **`ModuleNotFoundError`**: ejecuta los módulos con `python -m ...` desde la raíz del repo (como en los ejemplos).
- **Mensajes de TensorFlow sobre GPU**: si no tienes CUDA/TensorRT instalados, son warnings esperables; el código corre en CPU.
