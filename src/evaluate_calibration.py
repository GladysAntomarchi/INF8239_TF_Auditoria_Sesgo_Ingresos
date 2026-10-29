
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

from split_data import get_data_splits
from train_logistic import train_logistic
import pandas as pd
import numpy as np

from export_predictions import clasificar_edad

def obtener_grupos_calibracion(X_val):
    """Prepara los grupos para evaluar la calibración."""

    return {
        "sex": X_val["sex"].fillna("Missing"),
        "race": X_val["race"].fillna("Missing"),
        "age_group": X_val["age"].map(clasificar_edad),
    }

def calcular_calibracion_por_grupo(y_true, probabilidades, grupos):
    """Calcula Brier Score y Log Loss para cada grupo."""

    y = pd.Series(y_true).reset_index(drop=True)
    p = pd.Series(probabilidades, dtype=float).reset_index(drop=True)
    g = pd.Series(grupos).reset_index(drop=True).fillna("Missing")

    if not (len(y) == len(p) == len(g)):
        raise ValueError("Las entradas deben tener la misma longitud.")

    if not y.isin([0, 1]).all():
        raise ValueError("Las etiquetas deben ser binarias: 0 o 1.")

    if not np.isfinite(p.to_numpy()).all() or not p.between(0, 1).all():
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")

    resultados = []

    for grupo in sorted(g.unique()):
        mascara = (g == grupo)

        y_grupo = y[mascara]
        p_grupo = p[mascara]

        resultados.append({
            "grupo": grupo,
            "n": len(y_grupo),
            "positivos": int(y_grupo.sum()),
            "prevalencia_real": y_grupo.mean(),
            "probabilidad_media": p_grupo.mean(),
            "brier_score": brier_score_loss(y_grupo, p_grupo),
            "log_loss": log_loss(
                y_grupo, p_grupo, labels=[0, 1]
            ),
        })

    return pd.DataFrame(resultados)

def evaluar_calibracion_atributos(y_true, probabilidades, X_val):
    """Calcula calibración por sexo, raza y grupo de edad."""

    atributos = obtener_grupos_calibracion(X_val)
    resultados = []

    for atributo, grupos in atributos.items():
        tabla = calcular_calibracion_por_grupo(
            y_true=y_true,
            probabilidades=probabilidades,
            grupos=grupos,
        )

        tabla.insert(0, "atributo", atributo)
        resultados.append(tabla)

    return pd.concat(resultados, ignore_index=True)

def evaluate_calibration():
    """Evalúa y exporta la calibración logística en validación."""

    model, _ = train_logistic()

    _, X_val, _, y_val = get_data_splits()


    probabilidades = model.predict_proba(X_val)[:, 1]

    brier = brier_score_loss(y_val, probabilidades)
    logloss = log_loss(y_val, probabilidades, labels=[0, 1])

    print("\n=== CALIBRACIÓN GLOBAL: REGRESIÓN LOGÍSTICA ===")
    print(f"Registros de validación: {len(y_val)}")
    print(f"Brier Score: {brier:.6f}")
    print(f"Log Loss: {logloss:.6f}")

    resultados = evaluar_calibracion_atributos(
        y_true=y_val,
        probabilidades=probabilidades,
        X_val=X_val,
    )

    for atributo in ["sex", "race", "age_group"]:
        tabla = resultados[resultados["atributo"] == atributo]

        assert tabla["n"].sum() == len(y_val)
        assert tabla["positivos"].sum() == int(y_val.sum())

        print(f"\n=== CALIBRACIÓN POR {atributo.upper()} ===")
        print(tabla.to_string(index=False))

    ruta = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "calibracion_logistica_validacion.csv"
    )

    ruta.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(ruta, index=False)

    print(f"\nResultados guardados en: {ruta}")

    return brier, logloss

if __name__ == "__main__":
    evaluate_calibration()
