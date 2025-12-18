
"""
Script para ejecutar la demo del sistema completo.
"""

import subprocess


def main() -> None:
    subprocess.run(
        [
            "python",
            "-m",
            "src.modules.orchestator.detect_and_segment",
            "--sac_test_name",
            "example/sacs/CO10/CO10",
        ],
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "src.modules.orchestator.models_estimation",
            "--sac_test_name",
            "example/sacs/CO10/CO10",
            "--detection_dataframe_path",
            "results/Detection_CO10_BH*.csv",
            "--inventory_path",
            "example/inventory",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()