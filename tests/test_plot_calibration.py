
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

import plot_calibration


def test_calibration_outputs(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    figures = reports / "figures"
    reports.mkdir()

    datos = pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5, 6],
        "particion": ["validacion"] * 6,
        "modelo": ["logistic"] * 6,
        "configuracion": ["v1"] * 6,
        "y_true": [0, 0, 0, 1, 1, 1],
        "y_proba": [0.05, 0.15, 0.35, 0.65, 0.85, 0.95],
        "y_pred": [0, 0, 0, 1, 1, 1],
        "sex": ["Female", "Male"] * 3,
        "race": ["White", "Black"] * 3,
        "age_group": ["25-44"] * 6,
    })

    datos.to_csv(
        reports / "predicciones_logistic_v1_validacion.csv",
        index=False,
    )

    monkeypatch.setattr(plot_calibration, "REPORTS", reports)
    monkeypatch.setattr(plot_calibration, "FIGURES", figures)

    tabla, metricas = plot_calibration.generar_calibracion()

    assert tabla["n"].sum() == len(datos)
    assert tabla["positivos"].sum() == datos["y_true"].sum()
    assert metricas.loc[0, "n"] == len(datos)

    assert np.isfinite(metricas.loc[0, "brier_score"])
    assert np.isfinite(metricas.loc[0, "log_loss"])

    assert (reports / "curva_calibracion_logistic_v1.csv").exists()
    assert (reports / "metricas_calibracion_logistic_v1.csv").exists()

    figura = figures / "curva_calibracion_logistic_v1.png"
    assert figura.exists()
    assert figura.stat().st_size > 0


def test_calibration_rejects_duplicate_ids(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()

    datos = pd.DataFrame({
        "record_id": [1, 1],
        "particion": ["validacion", "validacion"],
        "modelo": ["logistic", "logistic"],
        "configuracion": ["v1", "v1"],
        "y_true": [0, 1],
        "y_proba": [0.2, 0.8],
    })

    datos.to_csv(
        reports / "predicciones_logistic_v1_validacion.csv",
        index=False,
    )

    monkeypatch.setattr(plot_calibration, "REPORTS", reports)

    with pytest.raises(ValueError, match="duplicados"):
        plot_calibration.generar_calibracion()
