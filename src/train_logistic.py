
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocessing import build_preprocessor
from split_data import get_data_splits

from export_predictions import (
    construir_predicciones,
    guardar_predicciones,
)


def train_logistic():
    """Entrena y evalúa Regresión Logística en validación."""

    X_train, X_val, y_train, y_val = get_data_splits()

    model = Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight=None,
        )),
    ])

    model.fit(X_train, y_train)


    predictions = model.predict(X_val)

    probabilidades = model.predict_proba(X_val)[:, 1]


    predicciones_validacion = construir_predicciones(
        X=X_val,
        y_true=y_val,
        y_proba=probabilidades,
        modelo="logistic",
        configuracion="v1",
        particion="validacion",
    )

    ruta_predicciones = guardar_predicciones(
        predicciones_validacion,
        ruta="reports/predicciones_logistic_v1_validacion.csv",
    )

    print("Predicciones guardadas en:", ruta_predicciones)

    print("\n=== PREDICCIONES DE VALIDACIÓN ===")
    print("Registros:", len(predicciones_validacion))
    print("Columnas:", predicciones_validacion.columns.tolist())


    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
    }

    print("\n=== REGRESIÓN LOGÍSTICA ===")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


    print("\n=== MATRIZ DE CONFUSIÓN ===")

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    print([[tn, fp], [fn, tp]])

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    print("\n=== MÉTRICAS DE AUDITORÍA GLOBAL ===")
    print(f"TPR: {tpr:.4f}")
    print(f"FPR: {fpr:.4f}")


    print("\n=== ITERACIONES DEL MODELO ===")
    print(model.named_steps["classifier"].n_iter_)

    return model, metrics


if __name__ == "__main__":
    train_logistic()
