"""Análisis exploratorio y estadístico del dataset UCI Adult.

Uso:
    python src/eda.py

Requiere haber descargado antes los datos con:
    python src/download_data.py

Las estadísticas se calculan sobre adult.data. El archivo adult.test
solo se utiliza para comparar la distribución de la variable objetivo.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Genera imágenes sin abrir ventanas.

import matplotlib.pyplot as plt
import pandas as pd

from inspect_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "eda"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "income"
POSITIVE_CLASS = ">50K"

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# Grupos etarios provisionales (pendientes de confirmar con el equipo).
# Intervalos cerrados a la izquierda: [0, 25), [25, 45), [45, 65), [65, 120).
AGE_BINS = [0, 25, 45, 65, 120]
AGE_LABELS = ["<25", "25-44", "45-64", "65+"]


def add_age_group(df, bins=AGE_BINS, labels=AGE_LABELS):
    """Devuelve una copia del DataFrame con la columna age_group."""
    result = df.copy()
    result["age_group"] = pd.cut(
        result["age"],
        bins=bins,
        labels=labels,
        right=False,
    )
    return result


def target_distribution(df, target=TARGET):
    """Frecuencia absoluta y relativa de cada clase de la variable objetivo."""
    counts = df[target].value_counts()
    table = pd.DataFrame({
        "n": counts,
        "proporcion": counts / counts.sum(),
    })
    table.index.name = target
    return table.reset_index()


def numeric_summary(df, columns=NUMERIC_COLUMNS):
    """Estadísticas descriptivas y proporción de ceros de variables numéricas."""
    summary = df[columns].describe().T
    summary["proporcion_ceros"] = (df[columns] == 0).mean()
    return summary.reset_index(names="variable")


def categorical_distribution(df, column):
    """Frecuencia de cada categoría, incluyendo los valores faltantes."""
    counts = df[column].fillna("(faltante)").value_counts()
    table = pd.DataFrame({
        "n": counts,
        "proporcion": counts / len(df),
    })
    table.index.name = "categoria"
    table = table.reset_index()
    table.insert(0, "variable", column)
    return table


def income_rate_by_group(df, group_column, target=TARGET, positive=POSITIVE_CLASS):
    """Tamaño, casos positivos y tasa de ingresos >50K por grupo."""
    is_positive = df[target] == positive
    grouped = is_positive.groupby(df[group_column], observed=True)

    table = pd.DataFrame({
        "n": grouped.size(),
        "n_positivos": grouped.sum(),
    })
    table["tasa_positiva"] = table["n_positivos"] / table["n"]
    table["proporcion_muestra"] = table["n"] / table["n"].sum()
    table.index.name = "grupo"

    table = table.reset_index()
    table.insert(0, "variable", group_column)
    return table


def plot_target_distribution(train_dist, test_dist, path):
    classes = train_dist[TARGET].tolist()
    test_props = test_dist.set_index(TARGET).loc[classes, "proporcion"].values
    positions = range(len(classes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([p - width / 2 for p in positions], train_dist["proporcion"], width,
           label="Entrenamiento (adult.data)")
    ax.bar([p + width / 2 for p in positions], test_props, width,
           label="Prueba (adult.test)")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(classes)
    ax.set_ylabel("Proporción")
    ax.set_title("Distribución de la variable objetivo income")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_age_distribution(df, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in df.groupby(TARGET):
        ax.hist(group["age"], bins=range(15, 95, 5), alpha=0.6, label=label)
    ax.set_xlabel("Edad")
    ax.set_ylabel("Número de personas")
    ax.set_title("Distribución de edades por clase de ingresos (adult.data)")
    ax.legend(title="income")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_rate_by_group(table, title, overall_rate, path):
    labels = table["grupo"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, table["tasa_positiva"])
    ax.axhline(overall_rate, color="gray", linestyle="--",
               label=f"Tasa global ({overall_rate:.1%})")

    for i, (rate, n) in enumerate(zip(table["tasa_positiva"], table["n"])):
        ax.text(i, rate, f"{rate:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Proporción con ingresos >50K")
    ax.set_title(title)
    ax.set_ylim(0, max(table["tasa_positiva"].max(), overall_rate) * 1.35)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_numeric_by_income(df, columns, path):
    classes = sorted(df[TARGET].unique())

    fig, axes = plt.subplots(1, len(columns), figsize=(4.5 * len(columns), 4.5))
    for ax, column in zip(axes, columns):
        data = [df.loc[df[TARGET] == c, column] for c in classes]
        ax.boxplot(data, tick_labels=classes)
        ax.set_title(column)
        ax.set_xlabel("income")
    fig.suptitle("Variables numéricas por clase de ingresos (adult.data)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_eda():
    train, test = load_data()
    train = add_age_group(train)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Variable objetivo
    train_target = target_distribution(train)
    test_target = target_distribution(test)
    target_table = pd.concat([
        train_target.assign(conjunto="train"),
        test_target.assign(conjunto="test"),
    ])
    target_table.round(4).to_csv(REPORTS_DIR / "distribucion_income.csv", index=False)

    # 2. Variables numéricas
    numeric = numeric_summary(train)
    numeric.round(4).to_csv(REPORTS_DIR / "resumen_numericas.csv", index=False)

    # 3. Variables categóricas
    categorical = pd.concat(
        [categorical_distribution(train, c) for c in CATEGORICAL_COLUMNS]
    )
    categorical.round(4).to_csv(REPORTS_DIR / "distribucion_categoricas.csv", index=False)

    # 4. Tasas de ingresos >50K por sexo, raza y grupo etario
    rates = {
        "sex": income_rate_by_group(train, "sex"),
        "race": income_rate_by_group(train, "race").sort_values("n", ascending=False),
        "age_group": income_rate_by_group(train, "age_group"),
    }
    pd.concat(rates.values()).round(4).to_csv(
        REPORTS_DIR / "tasa_ingresos_por_grupo.csv", index=False
    )

    overall_rate = (train[TARGET] == POSITIVE_CLASS).mean()

    # 5. Gráficos
    plot_target_distribution(train_target, test_target, FIGURES_DIR / "eda_distribucion_income.png")
    plot_age_distribution(train, FIGURES_DIR / "eda_edad_por_income.png")
    plot_rate_by_group(rates["sex"], "Tasa de ingresos >50K por sexo",
                       overall_rate, FIGURES_DIR / "eda_tasa_por_sexo.png")
    plot_rate_by_group(rates["race"], "Tasa de ingresos >50K por raza",
                       overall_rate, FIGURES_DIR / "eda_tasa_por_raza.png")
    plot_rate_by_group(rates["age_group"], "Tasa de ingresos >50K por grupo etario",
                       overall_rate, FIGURES_DIR / "eda_tasa_por_edad.png")
    plot_numeric_by_income(train, ["education_num", "hours_per_week"],
                           FIGURES_DIR / "eda_numericas_por_income.png")

    # 6. Resumen en consola
    print("\n=== DISTRIBUCIÓN DE INCOME ===")
    print(target_table.round(4).to_string(index=False))

    print(f"\nTasa global de ingresos >50K (adult.data): {overall_rate:.4f}")

    print("\n=== RESUMEN DE VARIABLES NUMÉRICAS (adult.data) ===")
    print(numeric.round(2).to_string(index=False))

    for name, table in rates.items():
        print(f"\n=== TASA DE INGRESOS >50K POR {name.upper()} ===")
        print(table.round(4).to_string(index=False))

    print(f"\nTablas guardadas en: {REPORTS_DIR}")
    print(f"Gráficos guardados en: {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()