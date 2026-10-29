import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from export_predictions import construir_predicciones
from random_forest import build_random_forest, train_and_predict


@pytest.fixture
def datos_controlados():
    """Datos pequeños con las columnas del proyecto y una relación simple con la clase."""
    rng = np.random.default_rng(0)
    n = 200
    education_num = rng.integers(1, 17, n)
    X = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "workclass": rng.choice(["Private", "Self-emp", None], n),
        "fnlwgt": rng.integers(10_000, 500_000, n),
        "education": rng.choice(["HS-grad", "Bachelors"], n),
        "education_num": education_num,
        "marital_status": rng.choice(["Married-civ-spouse", "Never-married"], n),
        "occupation": rng.choice(["Sales", "Exec-managerial", None], n),
        "relationship": rng.choice(["Husband", "Wife", "Own-child"], n),
        "race": rng.choice(["White", "Black"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "capital_gain": rng.choice([0, 0, 0, 5000], n),
        "capital_loss": 0,
        "hours_per_week": rng.integers(10, 60, n),
        "native_country": rng.choice(["United-States", "Mexico", None], n),
    })
    # La clase depende de education_num para que el modelo tenga algo que aprender.
    y = pd.Series((education_num >= 12).astype(int), index=X.index)
    return X.iloc[:150], X.iloc[150:], y.iloc[:150], y.iloc[150:]


def test_random_forest_trains_and_predicts(datos_controlados):
    """Entrena, predice de forma reproducible e ignora los atributos excluidos."""
    X_train, X_val, y_train, y_val = datos_controlados

    model, proba_1, training_time, prediction_time = train_and_predict(X_train, y_train, X_val)
    _, proba_2, _, _ = train_and_predict(X_train, y_train, X_val)

    # Configuración acordada.
    rf = model.named_steps["model"]
    assert rf.random_state == 42
    assert rf.class_weight is None

    # Una probabilidad por registro de validación y resultados reproducibles.
    assert len(proba_1) == len(X_val)
    assert np.array_equal(proba_1, proba_2)
    assert training_time > 0
    assert prediction_time > 0

    # El modelo aprende la relación simple de los datos controlados.
    accuracy = ((proba_1 >= 0.5).astype(int) == y_val.to_numpy()).mean()
    assert accuracy >= 0.9

    # Cambiar sex, race, fnlwgt o education no cambia las predicciones.
    X_val_changed = X_val.copy()
    X_val_changed["sex"] = "Female"
    X_val_changed["race"] = "Other"
    X_val_changed["fnlwgt"] = 1
    X_val_changed["education"] = "Doctorate"
    assert np.array_equal(model.predict_proba(X_val_changed)[:, 1], proba_1)


def test_random_forest_probabilities_are_valid(datos_controlados):
    """Las probabilidades sirven para evaluar la calibración y se exportan correctamente."""
    X_train, X_val, y_train, y_val = datos_controlados

    model = build_random_forest().fit(X_train, y_train)
    proba = model.predict_proba(X_val)

    # Dos columnas (clases 0 y 1), valores en [0, 1] y suma 1 por registro.
    assert list(model.classes_) == [0, 1]
    assert proba.shape == (len(X_val), 2)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1.0)

    # La exportación conserva los registros y aplica el umbral de 0.5.
    predicciones = construir_predicciones(X_val, y_val, proba[:, 1], "random_forest", "v1")
    assert predicciones["record_id"].tolist() == X_val.index.tolist()
    assert predicciones["y_pred"].tolist() == (proba[:, 1] >= 0.5).astype(int).tolist()
