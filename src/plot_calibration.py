
"""Genera evidencias de calibración desde predicciones de validación."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss, log_loss


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def generar_calibracion():
    ruta = REPORTS / "predicciones_logistic_v1_validacion.csv"
    df = pd.read_csv(ruta)

    if df.empty:
        raise ValueError("El archivo de predicciones está vacío.")

    if not df["modelo"].eq("logistic").all():
        raise ValueError("Se esperaban predicciones de Regresión Logística.")

    if not df["particion"].eq("validacion").all():
        raise ValueError("Solo se admiten predicciones de validación.")

    if df["record_id"].duplicated().any():
        raise ValueError("Existen identificadores duplicados.")

    y = df["y_true"]
    p = df["y_proba"]

    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Las etiquetas reales no son válidas.")

    if p.isna().any() or not np.isfinite(p).all():
        raise ValueError("Existen probabilidades faltantes o no finitas.")

    if not p.between(0, 1).all():
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")

    # Intervalos de igual amplitud: [0, 0.1], ..., (0.9, 1].
    intervalos = pd.cut(
        p,
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )

    tabla = (
        df.assign(intervalo=intervalos)
        .groupby("intervalo", observed=True)
        .agg(
            n=("y_true", "size"),
            probabilidad_media=("y_proba", "mean"),
            frecuencia_observada=("y_true", "mean"),
            positivos=("y_true", "sum"),
        )
        .reset_index()
    )

    tabla.insert(0, "modelo", "logistic")
    tabla.insert(1, "configuracion", "v1")
    tabla["intervalo"] = tabla["intervalo"].astype(str)

    assert int(tabla["n"].sum()) == len(df)

    brier = brier_score_loss(y, p)
    perdida_logaritmica = log_loss(y, p, labels=[0, 1])

    metricas = pd.DataFrame([{
        "modelo": "logistic",
        "configuracion": "v1",
        "particion": "validacion",
        "n": len(df),
        "brier_score": brier,
        "log_loss": perdida_logaritmica,
    }])

    FIGURES.mkdir(parents=True, exist_ok=True)

    tabla.to_csv(
        REPORTS / "curva_calibracion_logistic_v1.csv",
        index=False,
    )

    metricas.to_csv(
        REPORTS / "metricas_calibracion_logistic_v1.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        label="Calibración perfecta",
    )

    ax.plot(
        tabla["probabilidad_media"],
        tabla["frecuencia_observada"],
        marker="o",
        label="Regresión Logística",
    )

    ax.set(
        xlabel="Probabilidad media predicha",
        ylabel="Frecuencia positiva observada",
        title="Curva de calibración — Regresión Logística",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    figura = FIGURES / "curva_calibracion_logistic_v1.png"
    fig.savefig(figura, dpi=180)
    plt.close(fig)

    print("Registros:", len(df))
    print("Intervalos utilizados:", len(tabla))
    print("Registros contabilizados:", int(tabla["n"].sum()))
    print(f"Brier Score: {brier:.6f}")
    print(f"Log Loss: {perdida_logaritmica:.6f}")
    print("Tabla de calibración guardada.")
    print("Métricas globales guardadas.")
    print("Gráfica guardada en:", figura)

    return tabla, metricas


if __name__ == "__main__":
    generar_calibracion()
