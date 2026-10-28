
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_predictions import (
    COLUMNAS_PREDICCIONES,
    clasificar_edad,
    construir_predicciones,
    consolidar_predicciones,
    guardar_predicciones,
)



@pytest.fixture
def datos_ejemplo():
    X = pd.DataFrame(
        {
            "age": [22, 35, 55, 70],
            "sex": ["Female", "Male", "Female", "Male"],
            "race": ["White", "Black", "White", "Black"],
        },
        index=[10, 20, 30, 40],
    )

    y = pd.Series([0, 1, 1, 0], index=X.index)
    probabilidades = np.array([0.1, 0.8, 0.5, 0.3])

    return X, y, probabilidades


def test_age_groups():
    assert clasificar_edad(22) == "<25"
    assert clasificar_edad(35) == "25-44"
    assert clasificar_edad(55) == "45-64"
    assert clasificar_edad(70) == "65+"
    assert clasificar_edad(np.nan) == "Missing"


def test_predictions_keep_original_ids(datos_ejemplo):
    X, y, probabilidades = datos_ejemplo

    resultado = construir_predicciones(
        X, y, probabilidades, "logistic", "v1"
    )

    assert resultado["record_id"].tolist() == [10, 20, 30, 40]
    assert resultado["y_true"].tolist() == [0, 1, 1, 0]
    assert resultado["y_pred"].tolist() == [0, 1, 1, 0]
    assert resultado["age_group"].tolist() == [
        "<25", "25-44", "45-64", "65+"
    ]
    assert list(resultado.columns) == COLUMNAS_PREDICCIONES


def test_predictions_reject_misaligned_labels(datos_ejemplo):
    X, y, probabilidades = datos_ejemplo

    y_desalineado = y.copy()
    y_desalineado.index = [40, 30, 20, 10]

    with pytest.raises(ValueError, match="desalineados"):
        construir_predicciones(
            X, y_desalineado, probabilidades, "logistic", "v1"
        )


@pytest.mark.parametrize(
    "probabilidades",
    [
        [0.1, 0.8, 1.2, 0.3],
        [0.1, -0.1, 0.5, 0.3],
        [0.1, np.nan, 0.5, 0.3],
        [0.1, np.inf, 0.5, 0.3],
    ],
)
def test_predictions_reject_invalid_probabilities(
    datos_ejemplo, probabilidades
):
    X, y, _ = datos_ejemplo

    with pytest.raises(ValueError):
        construir_predicciones(
            X, y, probabilidades, "logistic", "v1"
        )


def test_predictions_csv_roundtrip(datos_ejemplo, tmp_path):
    X, y, probabilidades = datos_ejemplo

    resultado = construir_predicciones(
        X, y, probabilidades, "logistic", "v1"
    )

    ruta = tmp_path / "predicciones_validacion.csv"
    guardar_predicciones(resultado, ruta)

    recuperado = pd.read_csv(ruta)

    pd.testing.assert_frame_equal(resultado, recuperado)

def test_consolidation_combines_models(datos_ejemplo):
    X, y, probabilidades = datos_ejemplo

    logistic = construir_predicciones(
        X, y, probabilidades, "logistic", "v1"
    )

    random_forest = construir_predicciones(
        X, y, [0.2, 0.9, 0.6, 0.1], "random_forest", "v1"
    )

    resultado = consolidar_predicciones([logistic, random_forest])

    assert len(resultado) == 8
    assert resultado["modelo"].nunique() == 2
    assert resultado.groupby("modelo")["record_id"].nunique().eq(4).all()


def test_consolidation_rejects_different_records(datos_ejemplo):
    X, y, probabilidades = datos_ejemplo

    logistic = construir_predicciones(
        X, y, probabilidades, "logistic", "v1"
    )

    random_forest = construir_predicciones(
        X, y, probabilidades, "random_forest", "v1"
    )

    random_forest.loc[0, "record_id"] = 999

    with pytest.raises(ValueError):
        consolidar_predicciones([logistic, random_forest])


def test_consolidation_rejects_different_attributes(datos_ejemplo):
    X, y, probabilidades = datos_ejemplo

    logistic = construir_predicciones(
        X, y, probabilidades, "logistic", "v1"
    )

    random_forest = construir_predicciones(
        X, y, probabilidades, "random_forest", "v1"
    )

    valor_original = random_forest.loc[0, "sex"]

    random_forest.loc[0, "sex"] = (
        "Female" if valor_original == "Male" else "Male"
    )

    with pytest.raises(ValueError):
        consolidar_predicciones([logistic, random_forest])


def test_consolidation_rejects_repeated_configuration(datos_ejemplo):
    X, y, probabilidades = datos_ejemplo

    logistic = construir_predicciones(
        X, y, probabilidades, "logistic", "v1"
    )

    with pytest.raises(ValueError, match="predicciones duplicadas"):
        consolidar_predicciones([logistic, logistic.copy()])
