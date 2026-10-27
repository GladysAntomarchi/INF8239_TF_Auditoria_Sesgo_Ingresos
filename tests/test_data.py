
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
