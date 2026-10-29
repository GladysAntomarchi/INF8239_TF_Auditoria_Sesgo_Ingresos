
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from evaluate_calibration import (
    calcular_calibracion_por_grupo,
    obtener_grupos_calibracion,
    evaluar_calibracion_atributos,
)



def test_calibration_metrics_by_group():
    y_true = pd.Series([0, 1, 0, 1])
    probabilidades = np.array([0.1, 0.9, 0.2, 0.8])
    grupos = pd.Series(["A", "A", "B", "B"])

    resultado = calcular_calibracion_por_grupo(
        y_true, probabilidades, grupos
    ).set_index("grupo")

    assert resultado.loc["A", "n"] == 2
    assert resultado.loc["B", "n"] == 2
    assert resultado.loc["A", "positivos"] == 1
    assert resultado.loc["B", "positivos"] == 1

    assert np.isclose(resultado.loc["A", "brier_score"], 0.01)
    assert np.isclose(resultado.loc["B", "brier_score"], 0.04)


def test_calibration_handles_single_class_group():
    y_true = pd.Series([0, 0, 1, 1])
    probabilidades = np.array([0.1, 0.2, 0.8, 0.9])
    grupos = pd.Series(["A", "A", "B", "B"])

    resultado = calcular_calibracion_por_grupo(
        y_true, probabilidades, grupos
    )

    assert len(resultado) == 2
    assert np.isfinite(resultado["log_loss"]).all()


def test_calibration_rejects_invalid_probabilities():
    y_true = pd.Series([0, 1])
    grupos = pd.Series(["A", "B"])

    with pytest.raises(ValueError):
        calcular_calibracion_por_grupo(
            y_true, [0.2, 1.5], grupos
        )


def test_calibration_preserves_missing_groups():
    y_true = pd.Series([0, 1, 0])
    probabilidades = [0.1, 0.8, 0.2]
    grupos = pd.Series(["Female", None, "Male"])

    resultado = calcular_calibracion_por_grupo(
        y_true, probabilidades, grupos
    )

    assert "Missing" in resultado["grupo"].values
    assert resultado["n"].sum() == 3


def test_calibration_age_groups():
    X = pd.DataFrame({
        "sex": ["Female", "Male", "Female", "Male"],
        "race": ["White", "Black", "White", "Black"],
        "age": [22, 35, 55, 70],
    })

    grupos = obtener_grupos_calibracion(X)

    assert grupos["age_group"].tolist() == [
        "<25", "25-44", "45-64", "65+"
    ]
    assert set(grupos) == {"sex", "race", "age_group"}

def test_calibration_audits_three_attributes():
    X = pd.DataFrame({
        "sex": ["Female", "Male", "Female", "Male"],
        "race": ["White", "Black", "White", "Black"],
        "age": [22, 35, 55, 70],
    })

    y_true = pd.Series([0, 1, 0, 1])
    probabilidades = np.array([0.1, 0.8, 0.2, 0.9])

    resultado = evaluar_calibracion_atributos(
        y_true, probabilidades, X
    )

    assert set(resultado["atributo"]) == {
        "sex", "race", "age_group"
    }

    for atributo in ["sex", "race", "age_group"]:
        tabla = resultado[resultado["atributo"] == atributo]

        assert tabla["n"].sum() == len(y_true)
        assert tabla["positivos"].sum() == int(y_true.sum())
        assert tabla["brier_score"].between(0, 1).all()
        assert np.isfinite(tabla["log_loss"]).all()
