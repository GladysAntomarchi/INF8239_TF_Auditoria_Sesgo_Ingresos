
import argparse
from pathlib import Path

import pandas as pd
from export_predictions import clasificar_edad
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from inspect_data import load_data
from preprocessing import build_preprocessor


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def construir_modelo_final():
    """Reutiliza la configuración seleccionada en validación."""
    return Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight=None,
        )),
    ])


def evaluar_final():
    """Entrena con adult.data y evalúa una vez en adult.test."""
    ruta_metricas = OUTPUT_DIR / "metricas_logistic_test.csv"
    ruta_predicciones = OUTPUT_DIR / "predicciones_logistic_test.csv"

    if ruta_metricas.exists() or ruta_predicciones.exists():
        raise FileExistsError(
            "Ya existen resultados finales. No se sobrescribirán."
        )

    temporal_predicciones = ruta_predicciones.with_suffix(".tmp")
    temporal_metricas = ruta_metricas.with_suffix(".tmp")

    if temporal_predicciones.exists() or temporal_metricas.exists():
        raise FileExistsError(
            "Existen archivos temporales de una ejecución anterior."
        )

    train, test = load_data()

    categorias_validas = {"<=50K", ">50K"}

    for nombre, datos in [("entrenamiento", train), ("prueba", test)]:
        if datos["income"].isna().any():
            raise ValueError(
                f"Existen etiquetas nulas en {nombre}."
            )

        if set(datos["income"].unique()) != categorias_validas:
            raise ValueError(
                f"Categorías de ingresos inesperadas en {nombre}."
            )

    X_train = train.drop(columns=["income"])
    y_train = (train["income"] == ">50K").astype(int)

    X_test = test.drop(columns=["income"])
    y_test = (test["income"] == ">50K").astype(int)

    if len(train) != 32561 or len(test) != 16281:
        raise ValueError("Dimensiones inesperadas del dataset.")

    if not set(train["income"].dropna().unique()).issubset(
        {"<=50K", ">50K"}
    ):
        raise ValueError("Etiquetas inesperadas en entrenamiento.")

    if not set(test["income"].dropna().unique()).issubset(
        {"<=50K", ">50K"}
    ):
        raise ValueError("Etiquetas inesperadas en prueba.")

    modelo = construir_modelo_final()
    modelo.fit(X_train, y_train)

    probabilidades = modelo.predict_proba(X_test)[:, 1]
    predicciones = (probabilidades >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predicciones, labels=[0, 1]
    ).ravel()

    metricas = {
        "modelo": "logistic",
        "particion": "test",
        "registros": len(test),
        "accuracy": accuracy_score(y_test, predicciones),
        "precision": precision_score(
            y_test, predicciones, zero_division=0
        ),
        "recall": recall_score(
            y_test, predicciones, zero_division=0
        ),
        "f1": f1_score(y_test, predicciones, zero_division=0),
        "tpr": tp / (tp + fn) if (tp + fn) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


    resultados = pd.DataFrame({
        "record_id": X_test.index,
        "particion": "test",
        "modelo": "logistic",
        "configuracion": "v1_final",
        "y_true": y_test.to_numpy(),
        "y_proba": probabilidades,
        "y_pred": predicciones,
        "sex": X_test["sex"].to_numpy(),
        "race": X_test["race"].to_numpy(),
        "age_group": X_test["age"].map(clasificar_edad).to_numpy(),
    })



    temporal_predicciones = ruta_predicciones.with_suffix(".tmp")
    temporal_metricas = ruta_metricas.with_suffix(".tmp")

    if temporal_predicciones.exists() or temporal_metricas.exists():
        raise FileExistsError(
            "Existen archivos temporales de una ejecución anterior."
        )

    try:
        resultados.to_csv(temporal_predicciones, index=False)
        pd.DataFrame([metricas]).to_csv(temporal_metricas, index=False)

        temporal_predicciones.rename(ruta_predicciones)
        temporal_metricas.rename(ruta_metricas)

    finally:
        for temporal in [temporal_predicciones, temporal_metricas]:
            if temporal.exists():
                temporal.unlink()


    print("\n=== EVALUACIÓN EXTERNA FINAL ===")

    print("Entrenamiento:", len(train))
    print("Prueba:", len(test))
    print("\nMétricas:")
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor}")

    print("\nResultados guardados en:")
    print(ruta_metricas)
    print(ruta_predicciones)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmar-evaluacion-final",
        action="store_true",
        help="Autoriza ejecutar la evaluación única en adult.test.",
    )
    args = parser.parse_args()

    if not args.confirmar_evaluacion_final:
        parser.error(
            "Evaluación bloqueada. "
            "Se requiere --confirmar-evaluacion-final."
        )

    evaluar_final()
