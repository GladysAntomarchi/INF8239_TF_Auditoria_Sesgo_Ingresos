"""Comparación de modelos y análisis de errores sobre predicciones de validación.

Trabaja con el formato común de predicciones definido en
src/export_predictions.py (columnas record_id, particion, modelo,
configuracion, y_true, y_proba, y_pred, sex, race, age_group).

No entrena modelos ni utiliza adult.test.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

MODEL_KEYS = ["modelo", "configuracion"]

# Intervalos de probabilidad para el análisis de errores por confianza.
CONFIDENCE_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Misma etiqueta que en src/eda.py para los grupos faltantes.
MISSING_LABEL = "(faltante)"


def classification_metrics(y_true, y_pred):
    """Métricas de clasificación para la clase positiva (>50K = 1)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compare_models(predicciones):
    """Tabla con las métricas globales de cada modelo y configuración."""
    rows = []
    for (modelo, configuracion), group in predicciones.groupby(MODEL_KEYS, sort=False):
        metrics = classification_metrics(group["y_true"], group["y_pred"])
        rows.append({"modelo": modelo, "configuracion": configuracion, **metrics})
    return pd.DataFrame(rows)


def label_outcomes(predicciones):
    """Devuelve una copia con el tipo de resultado de cada registro (TP, FP, TN, FN)."""
    result = predicciones.copy()
    y_true = result["y_true"].astype(int)
    y_pred = result["y_pred"].astype(int)

    result["resultado"] = np.select(
        [
            (y_true == 1) & (y_pred == 1),
            (y_true == 0) & (y_pred == 1),
            (y_true == 0) & (y_pred == 0),
        ],
        ["TP", "FP", "TN"],
        default="FN",
    )
    result["error"] = result["resultado"].isin(["FP", "FN"])
    return result


def errors_by_confidence(predicciones, bins=CONFIDENCE_BINS):
    """Errores por intervalo de probabilidad estimada, para cada modelo.

    Permite distinguir los errores cercanos al umbral (0.5) de los errores
    cometidos con alta confianza.
    """
    labeled = label_outcomes(predicciones)
    labeled["intervalo_proba"] = pd.cut(
        labeled["y_proba"], bins=bins, include_lowest=True
    )

    grouped = labeled.groupby(MODEL_KEYS + ["intervalo_proba"], observed=True, sort=True)
    table = grouped.agg(
        n=("error", "size"),
        n_errores=("error", "sum"),
        fp=("resultado", lambda s: int((s == "FP").sum())),
        fn=("resultado", lambda s: int((s == "FN").sum())),
    ).reset_index()

    table["tasa_error"] = table["n_errores"] / table["n"]
    table["intervalo_proba"] = table["intervalo_proba"].astype(str)
    return table


def error_composition_by_group(predicciones, group_column):
    """Cantidad de FP y FN por grupo y su peso dentro de los errores del modelo.

    Los registros con el grupo faltante se reportan como "(faltante)" en lugar
    de excluirse, de modo que la suma por grupo coincide con los totales.
    Complementa (no reemplaza) la auditoría de TPR/FPR por grupo.
    """
    labeled = label_outcomes(predicciones)
    labeled[group_column] = (
        labeled[group_column]
        .astype(object)
        .where(labeled[group_column].notna(), MISSING_LABEL)
    )

    rows = []
    for keys, model_df in labeled.groupby(MODEL_KEYS, sort=False):
        total_errors = int(model_df["error"].sum())
        for grupo, group_df in model_df.groupby(group_column, sort=True):
            n_errores = int(group_df["error"].sum())
            rows.append({
                "modelo": keys[0],
                "configuracion": keys[1],
                "variable": group_column,
                "grupo": grupo,
                "n": len(group_df),
                "fp": int((group_df["resultado"] == "FP").sum()),
                "fn": int((group_df["resultado"] == "FN").sum()),
                "n_errores": n_errores,
                "tasa_error": n_errores / len(group_df),
                "proporcion_de_errores": n_errores / total_errors if total_errors else 0.0,
            })
    return pd.DataFrame(rows)
