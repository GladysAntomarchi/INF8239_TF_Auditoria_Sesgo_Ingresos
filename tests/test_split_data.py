
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from split_data import get_data_splits


def test_split_is_reproducible():
    """Comprueba tamaños, estratificación e índices reproducibles."""

    primera = get_data_splits()
    segunda = get_data_splits()

    X_train_1, X_val_1, y_train_1, y_val_1 = primera
    X_train_2, X_val_2, y_train_2, y_val_2 = segunda

    assert len(X_train_1) == 26048
    assert len(X_val_1) == 6513

    assert X_train_1.index.equals(X_train_2.index)
    assert X_val_1.index.equals(X_val_2.index)

    assert y_train_1.equals(y_train_2)
    assert y_val_1.equals(y_val_2)

    assert set(X_train_1.index).isdisjoint(
        set(X_val_1.index)
    )

    assert set(X_train_1.index).union(
        set(X_val_1.index)
    ) == set(range(32561))

    assert abs(y_train_1.mean() - y_val_1.mean()) < 0.01

def test_split_does_not_load_official_test(monkeypatch):
    """Comprueba que la partición no consulta adult.test."""

    import split_data

    def bloquear_carga_completa():
        raise AssertionError(
            "No se debe leer adult.test durante el desarrollo."
        )

    monkeypatch.setattr(
        split_data,
        "load_data",
        bloquear_carga_completa,
    )

    X_train, X_val, y_train, y_val = (
        split_data.get_data_splits()
    )

    assert len(X_train) == 26048
    assert len(X_val) == 6513
    assert len(y_train) == 26048
    assert len(y_val) == 6513
