
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_fairness import evaluate_by_group, evaluar_atributos


def test_fairness_confusion_counts():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    grupos = pd.Series(["A", "A", "B", "B"])

    resultado = evaluate_by_group(y_true, y_pred, grupos)

    grupo_a = resultado.set_index("grupo").loc["A"]
    grupo_b = resultado.set_index("grupo").loc["B"]

    assert grupo_a["TP"] == 1
    assert grupo_a["FN"] == 1
    assert grupo_a["n_positivos"] == 2
    assert grupo_a["TPR"] == 0.5

    assert grupo_b["FP"] == 1
    assert grupo_b["TN"] == 1
    assert grupo_b["n_negativos"] == 2
    assert grupo_b["FPR"] == 0.5


def test_fairness_undefined_rates():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    grupos = pd.Series(["A", "A", "B", "B"])

    resultado = evaluate_by_group(y_true, y_pred, grupos)
    tabla = resultado.set_index("grupo")

    assert np.isnan(tabla.loc["A", "FPR"])
    assert np.isnan(tabla.loc["B", "TPR"])


def test_fairness_audits_three_attributes():
    X = pd.DataFrame(
        {
            "age": [22, 35, 55, 70],
            "sex": ["Female", "Male", "Female", "Male"],
            "race": ["White", "Black", "White", "Black"],
        }
    )

    y_true = pd.Series([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])

    resultado = evaluar_atributos(y_true, y_pred, X)

    assert set(resultado["atributo"]) == {
        "sex", "race", "age_group"
    }

    for atributo in ["sex", "race", "age_group"]:
        tabla = resultado[resultado["atributo"] == atributo]
        assert tabla["n"].sum() == len(y_true)
        assert tabla[["TP", "FP", "TN", "FN"]].sum().sum() == len(y_true)

def test_fairness_wilson_interval_small_group():
    """Comprueba incertidumbre con pocos positivos reales."""

    y_true = pd.Series([1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0])
    grupos = pd.Series(["A"] * 6)

    resultado = evaluate_by_group(y_true, y_pred, grupos)
    grupo = resultado.iloc[0]

    assert grupo["n_positivos"] == 4
    assert grupo["n_negativos"] == 2
    assert grupo["TPR"] == 1.0
    assert 0.50 < grupo["TPR_IC_inf"] < 0.52
    assert np.isclose(grupo["TPR_IC_sup"], 1.0)
    assert 0.0 <= grupo["FPR_IC_inf"] <= grupo["FPR_IC_sup"]

def test_fairness_preserves_missing_groups():
    """Los grupos faltantes no deben excluir registros."""

    y_true = pd.Series([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    grupos = pd.Series(["Female", None, "Male", None])

    resultado = evaluate_by_group(y_true, y_pred, grupos)

    assert "Missing" in resultado["grupo"].values
    assert resultado["n"].sum() == 4
    assert resultado[["TP", "FP", "TN", "FN"]].sum().sum() == 4

    grupo_missing = resultado.set_index("grupo").loc["Missing"]
    assert grupo_missing["n"] == 2
