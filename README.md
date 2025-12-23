# system_offline

Offline pipeline for **event detection/segmentation** and **estimation** (magnitude + hypocenter).

The App-Segmentation tool was developed to support detection, segmentation, envelope detection, and parameter estimation tasks. In particular, this offline pipeline is designed to estimate magnitude and hypocenter. The system is implemented in Python 3.10.13 and processes seismic data in SAC format together with XML inventory files.

The source code will be available for free at: [`https://github.com/lptvUchile/system_offline`](https://github.com/lptvUchile/system_offline). For more information, you can contact Prof. Néstor Becerra Yoma at: `nbecerra@ing.uchile.cl`

## Requirements

- **Python**: 3.10.13
- **Dependencies**: install `requirements.txt` first and then `requirements_part_2.txt` (in that order)

Installation (in an active venv/conda environment):

```bash
pip install -r requirements.txt 
```

```bash
pip install -r requirements_part_2.txt
```

## Data structure (demo)

The repo includes an `example/` folder with the minimal expected structure:

```text
example/
  inventory/
    <inv_file>.xml
  sacs/
    <STA>/
      <STA>_BH*.sac
```

Notes:
- **SACs**: a station “prefix” is expected (e.g., `example/sacs/CO10/CO10`) and the code searches for `*_BH*.sac`.
- **Inventories**: the estimation script receives an **XML file** (direct path) via `--inventory_path` and loads it with `obspy.read_inventory(...)`.

## Run (full demo)

From the repo root:

```bash
python example.py
```

This runs, in order:

1) `src.modules.orchestator.detect_and_segment`
2) `src.modules.orchestator.models_estimation`

## Run by stages

### 1) Detection + segmentation

```bash
python -m src.modules.orchestator.detect_and_segment \
  --sac_test_name example/sacs/CO10 \
  --detection_output_path results
```

Typical outputs in `results/`:
- `Detection_<prefijo>.ctm`
- `Detection_<prefijo>.csv`

### 2) Estimation (magnitude and hypocenter)

```bash
python -m src.modules.orchestator.models_estimation \
  --sac_test_name example/sacs/CO10 \
  --detection_dataframe_path "results/Detection_CO10.csv" \
  --inventory_path example/inventory/C1_CO10.xml
```

Output:
- `results/models_estimation_<prefijo>.csv`

## Models used (current paths)

The orchestration scripts load models from `src/models/`.


## Quick troubleshooting

- **`ModuleNotFoundError`**: run modules with `python -m ...` from the repo root (as in the examples).
- **TensorFlow GPU messages**: if you don’t have CUDA/TensorRT installed, these are expected warnings; the code runs on CPU.
