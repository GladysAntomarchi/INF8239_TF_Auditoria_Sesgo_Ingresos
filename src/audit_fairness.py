
from statsmodels.stats.proportion import proportion_confint

import pandas as pd

from export_predictions import clasificar_edad

from sklearn.metrics import confusion_matrix

from train_logistic import train_logistic
from split_data import get_data_splits

def evaluate_by_group(y_true, y_pred, groups):
    """Calcula métricas de clasificación por grupo."""

    results = []

    grupos_limpios = groups.fillna("Missing")

    for group in sorted(grupos_limpios.unique()):
        mask = (grupos_limpios == group)


        tn, fp, fn, tp = confusion_matrix(
            y_true[mask],
            y_pred[mask],
            labels=[0, 1],
        ).ravel()

        tpr = tp / (tp + fn) if tp + fn > 0 else float("nan")
        fpr = fp / (fp + tn) if fp + tn > 0 else float("nan")


        # Intervalos de confianza de Wilson al 95 %
        positive_total = tp + fn
        negative_total = fp + tn

        if positive_total > 0:
            tpr_low, tpr_high = proportion_confint(
                count=tp,
                nobs=positive_total,
                alpha=0.05,
                method="wilson",
            )
        else:
            tpr_low, tpr_high = float("nan"), float("nan")

        if negative_total > 0:
            fpr_low, fpr_high = proportion_confint(
                count=fp,
                nobs=negative_total,
                alpha=0.05,
                method="wilson",
            )
        else:
            fpr_low, fpr_high = float("nan"), float("nan")

        results.append({
            "grupo": group,
            "n": int(mask.sum()),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "n_positivos": int(positive_total),
            "n_negativos": int(negative_total),
            "TPR": tpr,
            "FPR": fpr,
            "TPR_IC_inf": tpr_low,
            "TPR_IC_sup": tpr_high,
            "FPR_IC_inf": fpr_low,
            "FPR_IC_sup": fpr_high,
        })

    return pd.DataFrame(results)

def evaluar_atributos(y_true, y_pred, X_val):
    """Audita sexo, raza y grupos de edad en validación."""

    atributos = {
        "sex": X_val["sex"],
        "race": X_val["race"],
        "age_group": X_val["age"].map(clasificar_edad),
    }

    resultados = []

    for atributo, grupos in atributos.items():
        tabla = evaluate_by_group(
            y_true=y_true,
            y_pred=y_pred,
            groups=grupos,
        )

        tabla.insert(0, "atributo", atributo)
        resultados.append(tabla)

    return pd.concat(resultados, ignore_index=True)



if __name__ == "__main__":
    from pathlib import Path

    model, _ = train_logistic()

    _, X_val, _, y_val = get_data_splits()

    predictions = model.predict(X_val)

    resultados = evaluar_atributos(
        y_true=y_val,
        y_pred=predictions,
        X_val=X_val,
    )

    for atributo in ["sex", "race", "age_group"]:
        tabla = resultados[resultados["atributo"] == atributo]

        print(f"\n=== AUDITORÍA POR {atributo.upper()} ===")
        print(tabla.to_string(index=False))

        global_tn, global_fp, global_fn, global_tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

        totales = tabla[["TN", "FP", "FN", "TP"]].sum()

        assert totales["TN"] == global_tn
        assert totales["FP"] == global_fp
        assert totales["FN"] == global_fn
        assert totales["TP"] == global_tp
        assert tabla["n"].sum() == len(y_val)

        print(f"Conciliación de {atributo}: OK")

    destino = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "auditoria_equidad_logistica_validacion.csv"
    )

    destino.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(destino, index=False, encoding="utf-8")

    print(f"\nResultados guardados en: {destino}")
