"""Entrenamiento de Random Forest con la partición y el preprocesamiento compartidos.

Uso:
    python src/random_forest.py

Requiere haber descargado antes los datos con:
    python src/download_data.py

Sigue el acuerdo metodológico (Issue #6):
- Partición 80/20 estratificada de adult.data con get_data_splits().
- Preprocesamiento común sin escalado (build_preprocessor(scale_numeric=False)).
- RandomForestClassifier con valores predeterminados, random_state=42
  y class_weight=None.
- adult.test no se utiliza.

Genera:
- reports/predicciones_random_forest_v1_validacion.csv

Los tiempos que se imprimen son solo informativos. La medición de Green AI
se realiza para todos los modelos en la misma máquina (Issue #9).
"""

import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from export_predictions import construir_predicciones, guardar_predicciones
from preprocessing import build_preprocessor
from split_data import RANDOM_STATE, get_data_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_NAME = "random_forest"
CONFIGURATION = "v1"


def build_random_forest(random_state=RANDOM_STATE):
    """Pipeline de preprocesamiento común + Random Forest con valores predeterminados."""
    return Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=False)),
        (
            "model",
            RandomForestClassifier(
                random_state=random_state,
                class_weight=None,
            ),
        ),
    ])


def train_and_predict(X_train, y_train, X_val):
    """Entrena el modelo y devuelve el pipeline, las probabilidades y los tiempos."""
    model = build_random_forest()

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    probabilities = model.predict_proba(X_val)[:, 1]
    prediction_time = time.perf_counter() - start

    return model, probabilities, training_time, prediction_time


def run_random_forest():
    X_train, X_val, y_train, y_val = get_data_splits()

    model, probabilities, training_time, prediction_time = train_and_predict(
        X_train, y_train, X_val
    )

    predicciones = construir_predicciones(
        X_val, y_val, probabilities, MODEL_NAME, CONFIGURATION
    )
    predictions_path = guardar_predicciones(
        predicciones,
        REPORTS_DIR / f"predicciones_{MODEL_NAME}_{CONFIGURATION}_validacion.csv",
    )

    print("\n=== RANDOM FOREST ===")
    print("Entrenamiento:", len(X_train), "registros")
    print("Validación:", len(X_val), "registros")
    print("Columnas tras preprocesar:",
          len(model.named_steps["preprocessor"].get_feature_names_out()))
    print("Número de árboles:", model.named_steps["model"].n_estimators)
    print(f"Tiempo de entrenamiento (informativo): {training_time:.3f} s")
    print(f"Tiempo de predicción (informativo): {prediction_time:.3f} s")
    print("\nPredicciones guardadas en:", predictions_path)


if __name__ == "__main__":
    run_random_forest()
