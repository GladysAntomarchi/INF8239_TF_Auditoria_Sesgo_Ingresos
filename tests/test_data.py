
import sys
from pathlib import Path

# Permite importar los módulos desde src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inspect_data import COLUMNS, load_data


def test_dataset_columns():
    """Verifica la estructura de ambos conjuntos."""
    train, test = load_data()

    assert train.columns.tolist() == COLUMNS
    assert test.columns.tolist() == COLUMNS

    assert train.shape[1] == 15
    assert test.shape[1] == 15


def test_income_categories():
    """Verifica las categorías válidas de ingresos."""
    train, test = load_data()

    expected = {"<=50K", ">50K"}

    assert set(train["income"].dropna().unique()) == expected
    assert set(test["income"].dropna().unique()) == expected

    assert train["income"].notna().all()
    assert test["income"].notna().all()

def test_missing_values_detection():
    """Verifica los valores faltantes de UCI Adult."""
    train, test = load_data()

    missing_columns = {
        "workclass",
        "occupation",
        "native_country",
    }

    train_missing = set(
        train.columns[train.isna().any()].tolist()
    )

    test_missing = set(
        test.columns[test.isna().any()].tolist()
    )

    assert train_missing == missing_columns
    assert test_missing == missing_columns

    assert not train["income"].isna().any()
    assert not test["income"].isna().any()


def test_feature_overlap():
    """Verifica la detección de coincidencias con datos controlados."""
    import pandas as pd
    from inspect_data import count_feature_overlap

    train = pd.DataFrame({
        "age": [25, 30, 40],
        "occupation": ["A", "B", "C"],
        "income": ["<=50K", ">50K", "<=50K"],
    })

    test = pd.DataFrame({
        "age": [25, 25, 50],
        "occupation": ["A", "A", "D"],
        "income": [">50K", "<=50K", ">50K"],
    })

    unique_matches, occurrences = count_feature_overlap(
        train, test
    )

    assert unique_matches == 1
    assert occurrences == 2
