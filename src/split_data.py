
from sklearn.model_selection import train_test_split

from inspect_data import load_training_data, load_data

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def get_data_splits(include_test=False):
    """Genera una partición estratificada y reproducible.

    Por defecto devuelve únicamente entrenamiento y validación.
    El conjunto oficial de prueba se reserva para la evaluación final.
    """

    train = load_training_data()

    X = train.drop(columns=["income"])
    y = (train["income"] == ">50K").astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


    if not include_test:
        return X_train, X_val, y_train, y_val

    _, test = load_data()

    X_test = test.drop(columns=["income"])
    y_test = (test["income"] == ">50K").astype(int)

    return X_train, X_val, X_test, y_train, y_val, y_test



if __name__ == "__main__":
    X_train, X_val, y_train, y_val = get_data_splits()

    print("\n=== PARTICIONES COMPARTIDAS ===")
    print("Entrenamiento:", X_train.shape)
    print("Validación:", X_val.shape)

    print("\n=== PROPORCIÓN DE CLASE POSITIVA ===")
    print("Entrenamiento:", round(y_train.mean(), 4))
    print("Validación:", round(y_val.mean(), 4))

    assert set(X_train.index).isdisjoint(set(X_val.index))
    print("\nEntrenamiento y validación sin índices compartidos.")
