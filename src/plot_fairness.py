
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "auditoria_equidad_logistica_validacion.csv"
OUTPUT = ROOT / "reports" / "figures"

ATRIBUTOS = {
    "sex": "Sexo",
    "race": "Raza",
    "age_group": "Grupo de edad",
}


def graficar_atributo(tabla, atributo):
    """Grafica TPR y FPR con intervalos de Wilson al 95 %."""

    datos = tabla.loc[tabla["atributo"] == atributo].copy()

    if datos.empty:
        raise ValueError(f"No hay resultados para {atributo}")

    if atributo == "age_group":
        orden = ["<25", "25-44", "45-64", "65+"]
        datos["grupo"] = pd.Categorical(
            datos["grupo"], categories=orden, ordered=True
        )
        datos = datos.sort_values("grupo")

    nombres_cortos = {
        "Amer-Indian-Eskimo": "Amer-Indian\n-Eskimo",
        "Asian-Pac-Islander": "Asian-Pac\n-Islander",
    }

    etiquetas = [
        f"{nombres_cortos.get(grupo, grupo)}\nTotal: {n:,}"
        for grupo, n in zip(datos["grupo"], datos["n"])
    ]


    posiciones = np.arange(len(datos))
    fig, axes = plt.subplots(
        2, 1, figsize=(max(9, len(datos) * 1.8), 9)
    )


    for ax, metrica, denominador, color in [
        (axes[0], "TPR", "n_positivos", "steelblue"),
        (axes[1], "FPR", "n_negativos", "darkorange"),
    ]:
        valores = datos[metrica].to_numpy(dtype=float)

        limite_inf = datos[f"{metrica}_IC_inf"].to_numpy(dtype=float)
        limite_sup = datos[f"{metrica}_IC_sup"].to_numpy(dtype=float)

        errores = np.vstack([
            np.maximum(0, valores - limite_inf),
            np.maximum(0, limite_sup - valores),
        ])

        validos = np.isfinite(valores) & np.isfinite(errores).all(axis=0)

        ax.bar(posiciones[validos], valores[validos], color=color, alpha=0.75)
        ax.errorbar(
            posiciones[validos],
            valores[validos],
            yerr=errores[:, validos],
            fmt="none",
            ecolor="black",
            capsize=5,
        )

        ax.set_xticks(posiciones)
        ax.set_xticklabels(etiquetas)

        from matplotlib.ticker import PercentFormatter

        if metrica == "TPR":
            ax.set_ylim(0, 1.05)
        else:
            limite_superior = np.nanmax(
                datos["FPR_IC_sup"].to_numpy(dtype=float)
            )
            ax.set_ylim(0, min(1.05, max(0.15, limite_superior * 1.15)))

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        ax.set_ylabel(metrica)
        ax.set_title(f"{metrica} por {ATRIBUTOS[atributo]}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

        for i, (_, fila) in enumerate(datos.iterrows()):
            n = int(fila[denominador])
            texto = f"Base: {n}"

            if n == 0:
                texto += "\nNo definido"

            ax.text(
                i, -0.24, texto,
                ha="center",
                va="top",
                fontsize=8,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

    fig.suptitle(
        f"Auditoría de equidad — {ATRIBUTOS[atributo]}\n"
        "Regresión Logística · Validación · IC Wilson 95 %",
        fontsize=14,
        fontweight="bold",
    )

    fig.subplots_adjust(hspace=0.75, top=0.85, bottom=0.14)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    destino = OUTPUT / f"equidad_logistica_{atributo}.png"

    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return destino


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {INPUT}")

    tabla = pd.read_csv(INPUT)

    columnas = {
        "atributo", "grupo", "n", "n_positivos", "n_negativos",
        "TPR", "FPR", "TPR_IC_inf", "TPR_IC_sup",
        "FPR_IC_inf", "FPR_IC_sup",
    }

    if not columnas.issubset(tabla.columns):
        raise ValueError("El CSV no contiene las columnas necesarias.")

    for atributo in ATRIBUTOS:
        destino = graficar_atributo(tabla, atributo)
        print(f"Gráfico generado: {destino}")


if __name__ == "__main__":
    main()
