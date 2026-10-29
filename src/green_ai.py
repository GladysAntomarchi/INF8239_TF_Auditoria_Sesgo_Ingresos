"""Medición de Green AI para los tres modelos en la misma computadora.

Uso:
    python src/green_ai.py

Como acordamos en el equipo, los tiempos solo se pueden comparar si se
miden en la misma máquina. Este script entrena cada modelo varias veces
con la partición compartida y guarda la mediana de los tiempos de
entrenamiento (fit) y de predicción (predict_proba).

Configuración de los modelos (acuerdo metodológico, Issue #6):
- dummy: DummyClassifier, referencia mínima.
- logistic: preprocesamiento con escalado + LogisticRegression
  (max_iter=2000, random_state=42, class_weight=None), igual que
  src/train_logistic.py.
- random_forest: preprocesamiento sin escalado + RandomForestClassifier
  con valores predeterminados y random_state=42, igual que
  src/random_forest.py.

Genera:
- reports/green_ai_validacion.csv

No usa adult.test.
"""

import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor
from split_data import RANDOM_STATE, get_data_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

CONFIGURATION = "v1"
REPETITIONS = 3

# Estrategia del baseline: se debe confirmar con la configuración que use Gladys.
DUMMY_STRATEGY = "most_frequent"


def build_models():
    """Crea los tres modelos con la configuración acordada."""
    return {
        "dummy": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("model", DummyClassifier(strategy=DUMMY_STRATEGY)),
        ]),
        "logistic": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE, class_weight=None,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight=None,
            )),
        ]),
    }


def measure_times(model, X_train, y_train, X_val, repetitions=REPETITIONS):
    """Entrena y predice varias veces y devuelve los tiempos de cada repetición."""
    fit_times, predict_times = [], []
    for _ in range(repetitions):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        model.predict_proba(X_val)
        predict_times.append(time.perf_counter() - start)
    return fit_times, predict_times


def _windows_value(command):
    """Lee un dato del sistema en Windows; devuelve None si no está disponible."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info():
    """Características de la computadora donde se hizo la medición."""
    processor = platform.processor() or platform.machine()
    ram_gb = None
    if platform.system() == "Windows":
        processor = _windows_value("(Get-CimInstance Win32_Processor).Name") or processor
        ram_bytes = _windows_value("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
        if ram_bytes and ram_bytes.isdigit():
            ram_gb = round(int(ram_bytes) / 1024**3, 2)
    return {
        "sistema": f"{platform.system()} {platform.release()}",
        "procesador": processor,
        "nucleos_logicos": os.cpu_count(),
        "ram_gb": ram_gb,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }


def run_green_ai():
    X_train, X_val, y_train, _ = get_data_splits()
    environment = environment_info()

    rows = []
    for name, model in build_models().items():
        print(f"Midiendo {name} ({REPETITIONS} repeticiones)...")
        fit_times, predict_times = measure_times(model, X_train, y_train, X_val)
        rows.append({
            "modelo": name,
            "configuracion": CONFIGURATION,
            "repeticiones": REPETITIONS,
            "tiempo_entrenamiento_s": statistics.median(fit_times),
            "tiempo_prediccion_s": statistics.median(predict_times),
            "entrenamiento_min_s": min(fit_times),
            "entrenamiento_max_s": max(fit_times),
            **environment,
        })

    result = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "green_ai_validacion.csv"
    result.to_csv(path, index=False)

    print("\n=== GREEN AI (mediana de", REPETITIONS, "repeticiones) ===")
    columns = ["modelo", "tiempo_entrenamiento_s", "tiempo_prediccion_s",
               "entrenamiento_min_s", "entrenamiento_max_s"]
    print(result[columns].round(4).to_string(index=False))
    print("\nEntorno:", environment)
    print("Resultados guardados en:", path)


if __name__ == "__main__":
    run_green_ai()
