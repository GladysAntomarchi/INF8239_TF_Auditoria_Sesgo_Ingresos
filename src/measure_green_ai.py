
import platform
import time
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor
from split_data import get_data_splits


def medir_modelo(repeticiones=3):
    X_train, X_val, y_train, _ = get_data_splits()

    tiempos_entrenamiento = []
    tiempos_prediccion = []

    for intento in range(1, repeticiones + 1):
        modelo = Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight=None,
            )),
        ])

        inicio = time.perf_counter()
        modelo.fit(X_train, y_train)
        tiempo_fit = time.perf_counter() - inicio

        inicio = time.perf_counter()
        probabilidades = modelo.predict_proba(X_val)
        tiempo_predict = time.perf_counter() - inicio

        assert probabilidades.shape == (len(X_val), 2)

        tiempos_entrenamiento.append(tiempo_fit)
        tiempos_prediccion.append(tiempo_predict)

        print(
            f"Repeticion {intento}: "
            f"fit={tiempo_fit:.4f}s, "
            f"predict_proba={tiempo_predict:.4f}s"
        )

    entorno = (
        "Intel Core i7-12700H; "
        "RAM 15.63 GB; "
        "Windows 11 Home; "
        f"Python {platform.python_version()}; "
        f"scikit-learn {sklearn.__version__}"
    )

    resultado = pd.DataFrame([{
        "modelo": "logistic",
        "configuracion": "v1",
        "tiempo_entrenamiento_s": pd.Series(
            tiempos_entrenamiento
        ).median(),
        "tiempo_prediccion_s": pd.Series(
            tiempos_prediccion
        ).median(),
        "entorno": entorno,
    }])

    destino = Path("reports/green_ai_logistic_v1.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(destino, index=False)

    print("\n=== RESUMEN GREEN AI ===")
    print(resultado.to_string(index=False))
    print("\nArchivo guardado en:", destino)


if __name__ == "__main__":
    medir_modelo()
