
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_predictions import (
    COLUMNAS_PREDICCIONES,
    construir_predicciones,
    consolidar_predicciones,
)


def test_dummy_most_frequent_strategy():
    X_train = pd.DataFrame({"age": [22, 35, 42, 51, 60]})
    y_train = pd.Series([0, 0, 1, 0, 1])

    modelo = DummyClassifier(strategy="most_frequent")
    modelo.fit(X_train, y_train)

    X_val = pd.DataFrame({"age": [25, 45, 65]})

    predicciones = modelo.predict(X_val)
    indice_positivo = list(modelo.classes_).index(1)
    probabilidades = modelo.predict_proba(X_val)[:, indice_positivo]

    assert modelo.strategy == "most_frequent"
    assert predicciones.tolist() == [0, 0, 0]
    assert probabilidades.tolist() == [0.0, 0.0, 0.0]


def test_dummy_predictions_shared_format():
    X_train = pd.DataFrame(
        {
            "age": [22, 35, 42, 51, 60],
            "sex": ["Female", "Male", "Male", "Female", "Male"],
            "race": ["White", "Black", "White", "White", "Black"],
        }
    )
    y_train = pd.Series([0, 0, 1, 0, 1])

    X_val = pd.DataFrame(
        {
            "age": [24, 39, 68],
            "sex": ["Female", "Male", "Female"],
            "race": ["White", "Black", "White"],
        },
        index=[12, 45, 78],
    )
    y_val = pd.Series([0, 1, 1], index=X_val.index)

    modelo = DummyClassifier(strategy="most_frequent")
    modelo.fit(X_train, y_train)

    indice_positivo = list(modelo.classes_).index(1)
    probabilidades = modelo.predict_proba(X_val)[:, indice_positivo]

    dummy = construir_predicciones(
        X=X_val,
        y_true=y_val,
        y_proba=probabilidades,
        modelo="dummy",
        configuracion="v1",
    )

    assert len(dummy) == 3
    assert list(dummy.columns) == COLUMNAS_PREDICCIONES
    assert dummy["record_id"].tolist() == [12, 45, 78]
    assert dummy["y_pred"].tolist() == [0, 0, 0]
    assert dummy["modelo"].eq("dummy").all()
    assert dummy["age_group"].tolist() == ["<25", "25-44", "65+"]

    logistic = construir_predicciones(
        X=X_val,
        y_true=y_val,
        y_proba=[0.2, 0.8, 0.7],
        modelo="logistic",
        configuracion="v1",
    )

    consolidado = consolidar_predicciones([dummy, logistic])

    assert len(consolidado) == 6
    assert consolidado["modelo"].nunique() == 2
