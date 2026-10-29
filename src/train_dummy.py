
"""Entrenamiento y exportación del baseline DummyClassifier."""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

from split_data import get_data_splits
from export_predictions import (
    construir_predicciones,
    guardar_predicciones,
)


def train_dummy():
    """Entrena el baseline sobre la partición compartida."""

    X_train, X_val, y_train, y_val = get_data_splits()

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    clases = list(model.classes_)
    indice_positivo = clases.index(1)

    probabilidades = model.predict_proba(X_val)[:, indice_positivo]

    predicciones_validacion = construir_predicciones(
        X=X_val,
        y_true=y_val,
        y_proba=probabilidades,
        modelo="dummy",
        configuracion="v1",
        particion="validacion",
    )

    ruta = guardar_predicciones(
        predicciones_validacion,
        ruta="reports/predicciones_dummy_v1_validacion.csv",
    )

    y_pred = predicciones_validacion["y_pred"]

    print("=== BASELINE DUMMY ===")
    print("Estrategia:", model.strategy)
    print("Clase mayoritaria:", model.classes_[model.class_prior_.argmax()])
    print("Entrenamiento:", len(X_train))
    print("Validación:", len(X_val))
    print("Accuracy:", round(accuracy_score(y_val, y_pred), 4))
    print("F1:", round(f1_score(y_val, y_pred, zero_division=0), 4))
    print("Predicciones guardadas en:", ruta)

    return model, predicciones_validacion


if __name__ == "__main__":
    train_dummy()
