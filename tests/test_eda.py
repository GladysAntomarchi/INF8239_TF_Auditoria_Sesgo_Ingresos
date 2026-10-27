import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eda import add_age_group, income_rate_by_group, target_distribution


def test_eda_with_controlled_data():
    """Verifica grupos etarios, tasas por grupo y distribución objetivo con datos controlados."""
    df = pd.DataFrame({
        "age": [20, 24, 25, 44, 45, 64, 65, 90],
        "sex": ["Female", "Female", "Male", "Male", "Female", "Male", "Female", "Male"],
        "income": ["<=50K", "<=50K", ">50K", "<=50K", ">50K", ">50K", "<=50K", ">50K"],
    })

    # Grupos etarios: los límites 25, 45 y 65 pertenecen al grupo superior.
    with_groups = add_age_group(df)
    assert with_groups["age_group"].astype(str).tolist() == [
        "<25", "<25", "25-44", "25-44", "45-64", "45-64", "65+", "65+",
    ]
    assert "age_group" not in df.columns  # No modifica el DataFrame original.

    # Tasas por sexo: Female 1/4 = 0.25, Male 3/4 = 0.75.
    by_sex = income_rate_by_group(with_groups, "sex").set_index("grupo")
    assert by_sex.loc["Female", "n"] == 4
    assert by_sex.loc["Female", "n_positivos"] == 1
    assert by_sex.loc["Female", "tasa_positiva"] == pytest.approx(0.25)
    assert by_sex.loc["Male", "tasa_positiva"] == pytest.approx(0.75)
    assert by_sex["proporcion_muestra"].sum() == pytest.approx(1.0)

    # Tasas por grupo etario: <25 0/2, 25-44 1/2, 45-64 2/2, 65+ 1/2.
    by_age = income_rate_by_group(with_groups, "age_group").set_index("grupo")
    assert by_age["tasa_positiva"].tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5])

    # Distribución objetivo: 4 de 8 registros con >50K.
    dist = target_distribution(df).set_index("income")
    assert dist.loc[">50K", "n"] == 4
    assert dist["proporcion"].sum() == pytest.approx(1.0)