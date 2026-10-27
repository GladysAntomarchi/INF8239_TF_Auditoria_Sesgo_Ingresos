
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "income"


NUMERIC_FEATURES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


CATEGORICAL_FEATURES = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "native_country",
]


# education se excluye por redundancia con education_num.
# sex y race se conservan en los datos originales para la auditoría,
# pero no se utilizan como variables predictoras.
# fnlwgt se excluye del modelo principal.


def build_preprocessor(scale_numeric=True):
    """Construye un preprocesador sin ajustarlo a los datos."""

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
