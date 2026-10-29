"""Comparación de los modelos con sus predicciones de validación.

Uso:
    python src/run_comparison.py

Lee todos los archivos reports/predicciones_<modelo>_v1_validacion.csv
(uno por modelo), comprueba que tengan los mismos registros con
consolidar_predicciones() y calcula las métricas con src/evaluation.py.

Genera:
- reports/comparacion_modelos_validacion.csv
- reports/errores_por_confianza_validacion.csv
- reports/composicion_errores_por_grupo_validacion.csv

No entrena modelos y no usa adult.test.
"""

from pathlib import Path

import pandas as pd

from evaluation import compare_models, error_composition_by_group, errors_by_confidence
from export_predictions import consolidar_predicciones

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

PREDICTIONS_PATTERN = "predicciones_*_v1_validacion.csv"
GROUP_COLUMNS = ["sex", "race", "age_group"]


def load_predictions(reports_dir=REPORTS_DIR):
    """Lee los archivos de predicciones de cada modelo."""
    paths = sorted(reports_dir.glob(PREDICTIONS_PATTERN))
    if len(paths) < 2:
        raise FileNotFoundError(
            f"Se necesitan al menos dos archivos {PREDICTIONS_PATTERN} en {reports_dir}."
        )
    return paths, [pd.read_csv(path) for path in paths]


def run_comparison():
    paths, predicciones = load_predictions()

    # consolidar_predicciones() falla si los modelos no tienen los mismos registros.
    consolidado = consolidar_predicciones(predicciones)

    comparison = compare_models(consolidado)
    by_confidence = errors_by_confidence(consolidado)
    by_group = pd.concat(
        [error_composition_by_group(consolidado, column) for column in GROUP_COLUMNS],
        ignore_index=True,
    )

    comparison.to_csv(REPORTS_DIR / "comparacion_modelos_validacion.csv", index=False)
    by_confidence.to_csv(REPORTS_DIR / "errores_por_confianza_validacion.csv", index=False)
    by_group.to_csv(REPORTS_DIR / "composicion_errores_por_grupo_validacion.csv", index=False)

    print("\n=== ARCHIVOS DE PREDICCIONES ===")
    for path in paths:
        print(path.name)
    print("Registros por modelo:", consolidado.groupby("modelo").size().to_dict())

    print("\n=== COMPARACIÓN DE MODELOS (validación) ===")
    print(comparison.round(4).to_string(index=False))

    print("\n=== ERRORES POR INTERVALO DE PROBABILIDAD ===")
    print(by_confidence.round(4).to_string(index=False))

    print("\n=== COMPOSICIÓN DE ERRORES POR GRUPO ===")
    print(by_group.round(4).to_string(index=False))

    print("\nResultados guardados en:", REPORTS_DIR)


if __name__ == "__main__":
    run_comparison()
