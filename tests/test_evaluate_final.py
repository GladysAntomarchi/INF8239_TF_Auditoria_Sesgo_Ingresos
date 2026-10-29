
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from evaluate_final import construir_modelo_final
from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES



def test_modelo_final_conserva_configuracion():
    modelo = construir_modelo_final()

    assert isinstance(modelo, Pipeline)

    clasificador = modelo.named_steps["classifier"]

    assert isinstance(clasificador, LogisticRegression)
    assert clasificador.max_iter == 2000
    assert clasificador.random_state == 42
    assert clasificador.class_weight is None

    preprocesador = modelo.named_steps["preprocessor"]

    variables = []
    for _, _, columnas in preprocesador.transformers:
        variables.extend(columnas)

    assert set(variables) == set(
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    assert len(variables) == 10

    for excluida in ["sex", "race", "fnlwgt", "education"]:
        assert excluida not in variables

import numpy as np
import pandas as pd


def test_modelo_final_con_datos_sinteticos():
    datos = pd.DataFrame({
        "age": [24, 35, 46, 57, 28, 39, 51, 63],
        "education_num": [9, 13, 10, 14, 9, 12, 11, 16],
        "capital_gain": [0, 5000, 0, 10000, 0, 3000, 0, 15000],
        "capital_loss": [0] * 8,
        "hours_per_week": [30, 40, 45, 50, 35, 40, 48, 55],
        "workclass": ["Private"] * 8,
        "marital_status": ["Never-married", "Married-civ-spouse"] * 4,
        "occupation": ["Sales", "Exec-managerial"] * 4,
        "relationship": ["Not-in-family", "Husband"] * 4,
        "native_country": ["United-States"] * 8,
    })

    etiquetas = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    modelo = construir_modelo_final()
    modelo.fit(datos, etiquetas)

    probabilidades = modelo.predict_proba(datos)[:, 1]

    assert len(probabilidades) == len(datos)
    assert np.isfinite(probabilidades).all()
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
    assert set(modelo.predict(datos)).issubset({0, 1})
