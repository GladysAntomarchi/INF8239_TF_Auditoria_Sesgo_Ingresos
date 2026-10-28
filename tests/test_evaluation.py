import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation import (
    classification_metrics,
    compare_models,
    error_composition_by_group,
    errors_by_confidence,
)


@pytest.fixture
def predicciones():
    """Dos modelos sobre los mismos 8 registros, con resultados calculables a mano."""
    base = {
        "record_id": list(range(8)),
        "particion": ["validacion"] * 8,
        "y_true": [1, 1, 1, 0, 0, 0, 0, 1],
        "sex": ["Female", "Male", "Male", "Female", "Male", "Female", "Male", "Female"],
        "race": ["White"] * 8,
        "age_group": ["25-44"] * 8,
    }
    modelo_a = pd.DataFrame({
        **base,
        "modelo": "modelo_a",
        "configuracion": "v1",
        # TP, TP, FN, FP, TN, TN, TN, FN
        "y_proba": [0.9, 0.8, 0.4, 0.95, 0.2, 0.1, 0.05, 0.45],
        "y_pred": [1, 1, 0, 1, 0, 0, 0, 0],
    })
    modelo_b = pd.DataFrame({
        **base,
        "modelo": "modelo_b",
        "configuracion": "v1",
        # Todos correctos.
        "y_proba": [0.9, 0.8, 0.7, 0.2, 0.2, 0.1, 0.05, 0.6],
        "y_pred": [1, 1, 1, 0, 0, 0, 0, 1],
    })
    return pd.concat([modelo_a, modelo_b], ignore_index=True)


def test_metrics_and_error_analysis_with_controlled_data(predicciones):
    """Verifica métricas, comparación y análisis de errores con valores calculados a mano."""
    # modelo_a: TP=2, FP=1, TN=3, FN=2.
    metrics = classification_metrics(
        predicciones.loc[predicciones["modelo"] == "modelo_a", "y_true"],
        predicciones.loc[predicciones["modelo"] == "modelo_a", "y_pred"],
    )
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (2, 1, 3, 2)
    assert metrics["accuracy"] == pytest.approx(5 / 8)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 4)
    assert metrics["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))

    # Comparación: una fila por modelo; modelo_b no comete errores.
    comparison = compare_models(predicciones).set_index("modelo")
    assert comparison.index.tolist() == ["modelo_a", "modelo_b"]
    assert comparison.loc["modelo_b", "f1"] == pytest.approx(1.0)
    assert comparison.loc["modelo_b", ["fp", "fn"]].sum() == 0

    # Errores por confianza (modelo_a): el único registro en (0.9, 1.0] es el FP
    # con probabilidad 0.95, un error de alta confianza.
    by_conf = errors_by_confidence(predicciones)
    a = by_conf[by_conf["modelo"] == "modelo_a"]
    assert a["n"].sum() == 8
    assert a["n_errores"].sum() == 3
    high = a[a["intervalo_proba"] == "(0.9, 1.0]"].iloc[0]
    assert (high["n"], high["fp"], high["fn"]) == (1, 1, 0)

    # Composición por sexo (modelo_a): Female tiene 2 errores (FP id 3, FN id 7)
    # y Male 1 error (FN id 2).
    by_sex = error_composition_by_group(predicciones, "sex")
    a_sex = by_sex[by_sex["modelo"] == "modelo_a"].set_index("grupo")
    assert a_sex.loc["Female", "n_errores"] == 2
    assert a_sex.loc["Male", "n_errores"] == 1
    assert a_sex["proporcion_de_errores"].sum() == pytest.approx(1.0)
    assert a_sex["n"].sum() == 8
