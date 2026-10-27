
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def load_data():
    train = pd.read_csv(
        DATA_DIR / "adult.data",
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?"],
    )

    test = pd.read_csv(
        DATA_DIR / "adult.test",
        names=COLUMNS,
        skiprows=1,
        skipinitialspace=True,
        na_values=["?"],
    )

    test["income"] = test["income"].str.rstrip(".")

    return train, test


def count_feature_overlap(train, test, target="income"):
    """Cuenta combinaciones y ocurrencias compartidas por características."""
    features = [column for column in train.columns if column != target]

    train_unique = train[features].drop_duplicates()
    test_unique = test[features].drop_duplicates()

    shared = train_unique.merge(
        test_unique,
        on=features,
        how="inner",
    )

    occurrences = test[features].merge(
        shared,
        on=features,
        how="inner",
    )

    return len(shared), len(occurrences)


if __name__ == "__main__":
    train, test = load_data()

    print("\n=== DIMENSIONES ===")
    print("Entrenamiento:", train.shape)
    print("Prueba:", test.shape)

    print("\n=== VARIABLES ===")
    print(train.dtypes)

    print("\n=== PRIMEROS REGISTROS ===")
    print(train.head(3).to_string(index=False))

    print("\n=== CLASES DE INGRESOS ===")
    print("Entrenamiento:")
    print(train["income"].value_counts())

    print("\nPrueba:")
    print(test["income"].value_counts())

    print("\n=== VALORES FALTANTES: ENTRENAMIENTO ===")
    print(train.isna().sum().to_string())

    print("\n=== VALORES FALTANTES: PRUEBA ===")
    print(test.isna().sum().to_string())

    print("\n=== DUPLICADOS EXACTOS ===")
    print("Entrenamiento:", train.duplicated().sum())
    print("Prueba:", test.duplicated().sum())

    print("\n=== DISTRIBUCIÓN DE SEXO ===")
    print(train["sex"].value_counts(dropna=False).to_string())

    print("\n=== DISTRIBUCIÓN DE RAZA ===")
    print(train["race"].value_counts(dropna=False).to_string())

    print("\n=== COINCIDENCIAS ENTRE ENTRENAMIENTO Y PRUEBA ===")

    # Comparamos filas completas y distinguimos coincidencias
    # únicas de ocurrencias repetidas.
    train_unique = train.drop_duplicates()
    test_unique = test.drop_duplicates()

    overlap = train_unique.merge(
        test_unique,
        how="inner",
        on=COLUMNS,
    )

    print("Filas distintas coincidentes:", len(overlap))

    train_overlap = train.merge(
        overlap,
        how="inner",
        on=COLUMNS,
    )

    test_overlap = test.merge(
        overlap,
        how="inner",
        on=COLUMNS,
    )

    print("Ocurrencias en entrenamiento:", len(train_overlap))

    print("Ocurrencias en prueba:", len(test_overlap))

    print("\n=== COINCIDENCIAS SIN VARIABLE OBJETIVO ===")

    shared_features, test_occurrences = count_feature_overlap(
        train,
        test,
        target="income",
    )

    print(
        "Combinaciones de características coincidentes:",
        shared_features,
    )

    print(
        "Ocurrencias coincidentes en prueba:",
        test_occurrences,
    )

    print("\n=== RESUMEN GENERAL DE AUDITORÍA ===")

    summary = pd.DataFrame({
        "conjunto": ["train", "test"],
        "registros": [len(train), len(test)],
        "columnas": [train.shape[1], test.shape[1]],
        "duplicados_exactos": [
            int(train.duplicated().sum()),
            int(test.duplicated().sum()),
        ],
        "registros_con_faltantes": [
            int(train.isna().any(axis=1).sum()),
            int(test.isna().any(axis=1).sum()),
        ],
    })

    reports_dir = DATA_DIR.parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "auditoria_inicial.csv"

    summary.to_csv(output_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nResumen guardado en: {output_path}")
