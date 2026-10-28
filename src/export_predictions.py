
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNAS_PREDICCIONES = [
    "record_id",
    "particion",
    "modelo",
    "configuracion",
    "y_true",
    "y_proba",
    "y_pred",
    "sex",
    "race",
    "age_group",
]


def clasificar_edad(edad):
    """Agrupa edades según el acuerdo metodológico."""
    if pd.isna(edad):
        return "Missing"
    if edad < 25:
        return "<25"
    if edad < 45:
        return "25-44"
    if edad < 65:
        return "45-64"
    return "65+"


def construir_predicciones(
    X,
    y_true,
    y_proba,
    modelo,
    configuracion,
    particion="validacion",
):
    """Construye predicciones alineadas con los registros originales."""

    if particion != "validacion":
        raise ValueError(
            "Este módulo se limita a la partición de validación."
        )

    if not X.index.is_unique:
        raise ValueError("Los índices originales deben ser únicos.")

    if not X.index.equals(y_true.index):
        raise ValueError("X e y_true tienen índices desalineados.")

    if not {"age", "sex", "race"}.issubset(X.columns):
        raise ValueError("Faltan atributos necesarios para auditoría.")

    if len(y_proba) != len(X):
        raise ValueError("La cantidad de probabilidades no coincide.")

    probabilidades = np.asarray(y_proba, dtype=float)

    if probabilidades.ndim != 1:
        raise ValueError("Las probabilidades deben ser un vector.")

    if not np.isfinite(probabilidades).all():
        raise ValueError("Existen probabilidades no válidas.")

    if ((probabilidades < 0) | (probabilidades > 1)).any():
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")

    if not y_true.isin([0, 1]).all():
        raise ValueError("Las clases reales deben ser 0 o 1.")

    if not modelo or not configuracion:
        raise ValueError("Modelo y configuración son obligatorios.")

    resultado = pd.DataFrame(
        {
            "record_id": X.index.to_numpy(),
            "particion": particion,
            "modelo": modelo,
            "configuracion": configuracion,
            "y_true": y_true.to_numpy(),
            "y_proba": probabilidades,
            "y_pred": (probabilidades >= 0.5).astype(int),
            "sex": X["sex"].to_numpy(copy=True),
            "race": X["race"].to_numpy(copy=True),
            "age_group": X["age"].map(clasificar_edad).to_numpy(copy=True),


        }
    )

    return resultado[COLUMNAS_PREDICCIONES]


def guardar_predicciones(
    predicciones,
    ruta="reports/predicciones_validacion.csv",
):
    """Guarda las predicciones previamente validadas."""
    if list(predicciones.columns) != COLUMNAS_PREDICCIONES:
        raise ValueError("El archivo no tiene las columnas esperadas.")

    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    predicciones.to_csv(destino, index=False)

    return destino

def consolidar_predicciones(predicciones_por_modelo):
    """Consolida modelos y verifica correspondencia por registro."""

    if len(predicciones_por_modelo) < 2:
        raise ValueError("Se requieren al menos dos modelos.")

    columnas_contexto = [
        "record_id",
        "y_true",
        "sex",
        "race",
        "age_group",
    ]

    conjuntos = []
    referencia = None

    for predicciones in predicciones_por_modelo:
        if list(predicciones.columns) != COLUMNAS_PREDICCIONES:
            raise ValueError("Columnas de predicciones incorrectas.")

        if predicciones.empty:
            raise ValueError("No se permiten predicciones vacías.")

        if predicciones["record_id"].isna().any():
            raise ValueError("Existen identificadores nulos.")

        if predicciones["record_id"].duplicated().any():
            raise ValueError("Existen identificadores duplicados.")

        if not predicciones["particion"].eq("validacion").all():
            raise ValueError("Solo se admite la partición de validación.")

        if predicciones["modelo"].nunique() != 1:
            raise ValueError("Cada conjunto debe contener un modelo.")

        if predicciones["configuracion"].nunique() != 1:
            raise ValueError("Cada conjunto debe contener una configuración.")

        actual = (
            predicciones[columnas_contexto]
            .sort_values("record_id")
            .reset_index(drop=True)
        )

        if referencia is None:
            referencia = actual
        elif not actual.equals(referencia):
            raise ValueError(
                "Los modelos no contienen los mismos registros "
                "o presentan diferencias en sus atributos."
            )

        conjuntos.append(predicciones)

    consolidado = pd.concat(conjuntos, ignore_index=True)

    if consolidado.duplicated(
        subset=["record_id", "particion", "modelo", "configuracion"]
    ).any():
        raise ValueError("Existen predicciones duplicadas.")

    combinaciones = consolidado[
        ["modelo", "configuracion"]
    ].drop_duplicates()

    if len(combinaciones) != len(predicciones_por_modelo):
        raise ValueError(
            "Se repite una combinación de modelo y configuración."
        )

    return consolidado[COLUMNAS_PREDICCIONES]
