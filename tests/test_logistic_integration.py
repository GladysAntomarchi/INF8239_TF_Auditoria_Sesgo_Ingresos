
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_predictions import construir_predicciones


def test_logistic_predictions_integration():
    X_val = pd.DataFrame(
        {
            "age": [23, 38, 67],
            "sex": ["Female", "Male", "Female"],
            "race": ["White", "Black", "White"],
        },
        index=[12, 45, 78],
    )

    y_val = pd.Series([0, 1, 1], index=X_val.index)
    probabilidades = np.array([0.2, 0.85, 0.6])

    predicciones = construir_predicciones(
        X=X_val,
        y_true=y_val,
        y_proba=probabilidades,
        modelo="logistic",
        configuracion="logistic_v1_10_features",
        particion="validacion",
    )

    assert len(predicciones) == 3
    assert predicciones["record_id"].tolist() == [12, 45, 78]
    assert predicciones["y_pred"].tolist() == [0, 1, 1]
    assert predicciones["modelo"].eq("logistic").all()
    assert predicciones["age_group"].tolist() == [
        "<25", "25-44", "65+"
    ]

def test_logistic_pipeline_with_shared_preprocessing():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from preprocessing import build_preprocessor

    X = pd.DataFrame(
        {
            "age": [22, 35, 48, 61, 27, 54],
            "education_num": [9, 13, 10, 16, 12, 14],
            "capital_gain": [0, 5000, 0, 10000, 0, 8000],
            "capital_loss": [0, 0, 0, 0, 0, 0],
            "hours_per_week": [25, 40, 35, 50, 30, 45],
            "workclass": ["Private"] * 6,
            "marital_status": ["Never-married", "Married"] * 3,
            "occupation": ["Sales", "Exec-managerial"] * 3,
            "relationship": ["Not-in-family", "Husband"] * 3,
            "native_country": ["United-States"] * 6,
            "sex": ["Female", "Male"] * 3,
            "race": ["White", "Black"] * 3,
        }
    )

    y = pd.Series([0, 1, 0, 1, 0, 1])

    modelo = Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight=None,
        )),
    ])

    modelo.fit(X, y)

    probabilidades = modelo.predict_proba(X)[:, 1]

    assert len(probabilidades) == len(X)
    assert np.isfinite(probabilidades).all()
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()

    transformador = modelo.named_steps["preprocessor"]

    columnas_utilizadas = set()
    for nombre, _, columnas in transformador.transformers_:
        if nombre != "remainder":
            columnas_utilizadas.update(columnas)

    assert "sex" not in columnas_utilizadas
    assert "race" not in columnas_utilizadas
    assert "fnlwgt" not in columnas_utilizadas
    assert "education" not in columnas_utilizadas
