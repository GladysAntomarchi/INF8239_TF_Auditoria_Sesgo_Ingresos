
from sklearn.model_selection import train_test_split

from inspect_data import load_data

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def get_data_splits():
    """Genera entrenamiento y validación estratificados."""

    train, test = load_data()

    X = train.drop(columns=["income"])
    y = (train["income"] == ">50K").astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_test = test.drop(columns=["income"])
    y_test = (test["income"] == ">50K").astype(int)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    splits = get_data_splits()

    X_train, X_val, X_test, y_train, y_val, y_test = splits

    print("\n=== PARTICIONES ===")
    print("Entrenamiento interno:", X_train.shape)
    print("Validación:", X_val.shape)
    print("Prueba oficial:", X_test.shape)

    print("\n=== PROPORCIÓN DE CLASE POSITIVA (>50K) ===")
    print("Entrenamiento:", round(y_train.mean(), 4))
    print("Validación:", round(y_val.mean(), 4))
    print("Prueba:", round(y_test.mean(), 4))

    print("\n=== VERIFICACIÓN DE ÍNDICES ===")
    assert set(X_train.index).isdisjoint(set(X_val.index))
    print("Entrenamiento y validación sin índices compartidos.")
