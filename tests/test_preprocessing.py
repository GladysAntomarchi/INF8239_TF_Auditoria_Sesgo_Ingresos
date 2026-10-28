
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    build_preprocessor,
)


def test_preprocessing_excludes_sensitive_features():
    """Comprueba que las variables excluidas no entren al modelo."""

    selected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    assert len(selected) == 10
    assert not selected.intersection({
        "sex", "race", "fnlwgt", "education", "income"
    })


def test_preprocessing_handles_missing_values():
    """Comprueba imputación y transformación de datos incompletos."""

    data = pd.DataFrame({
        "age": [25, 40, 55],
        "education_num": [10, 13, np.nan],
        "capital_gain": [0, 0, 1000],
        "capital_loss": [0, 0, 0],
        "hours_per_week": [40, 50, 35],
        "workclass": ["Private", None, "Self-emp"],
        "marital_status": ["Never-married", "Married", "Divorced"],
        "occupation": ["Sales", "Exec-managerial", None],
        "relationship": ["Not-in-family", "Husband", "Unmarried"],
        "native_country": ["United-States", None, "Mexico"],
        "sex": ["Female", "Male", "Female"],
        "race": ["White", "Black", "White"],
        "fnlwgt": [100000, 200000, 150000],
    })

    preprocessor = build_preprocessor(scale_numeric=True)
    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == 3
    assert np.isfinite(
        transformed.toarray()
        if hasattr(transformed, "toarray")
        else transformed
    ).all()

def test_preprocessing_uses_training_statistics_only():
    """La mediana debe aprenderse solo de entrenamiento."""

    train = pd.DataFrame({
        "age": [20, 30, 40],
        "education_num": [9, 10, 11],
        "capital_gain": [0, 0, 0],
        "capital_loss": [0, 0, 0],
        "hours_per_week": [30, 40, 50],
        "workclass": ["Private"] * 3,
        "marital_status": ["Never-married"] * 3,
        "occupation": ["Sales"] * 3,
        "relationship": ["Not-in-family"] * 3,
        "native_country": ["United-States"] * 3,
    })


    validation = train.iloc[[0]].copy()
    validation["age"] = validation["age"].astype(float)
    validation.loc[:, "age"] = np.nan

    preprocessor = build_preprocessor(scale_numeric=False)

    preprocessor.fit(train)
    transformed = preprocessor.transform(validation)

    # Mediana de edad aprendida del entrenamiento: 30.
    assert transformed[0, 0] == 30
