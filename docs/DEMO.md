# Demostración reproducible

Esta guía reproduce todo el proyecto desde cero, en orden: datos, EDA, modelos, comparación, equidad, calibración, Green AI, evaluación final y pruebas. Cada paso indica el comando, qué archivos genera y qué resultado se espera.

Los comandos son para Windows PowerShell y se ejecutan desde la raíz del repositorio. Todo el proceso tarda unos 3 a 5 minutos (la mayor parte es Random Forest y Green AI).

## 0. Preparación

```powershell
git clone https://github.com/GladysAntomarchi/INF8239_TF_Auditoria_Sesgo_Ingresos.git
cd INF8239_TF_Auditoria_Sesgo_Ingresos
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 1. Datos

```powershell
python src/download_data.py
python src/inspect_data.py
```

- Descarga `adult.data`, `adult.test` y `adult.names` en `data/raw/` (no se suben al repositorio).
- Genera `reports/auditoria_inicial.csv`.
- Resultado esperado: 32,561 registros en `adult.data` y 16,281 en `adult.test`.

## 2. Análisis exploratorio (solo `adult.data`)

```powershell
python src/eda.py
```

- Genera las tablas de `reports/eda/` y los gráficos `reports/figures/eda_*.png`.
- Resultado esperado: 75.9 % `<=50K` y 24.1 % `>50K`; tasa `>50K` de 10.9 % en mujeres y 30.6 % en hombres.

## 3. Modelos en la partición de validación

Los tres modelos usan la misma partición 80/20 estratificada de `adult.data` (26,048 registros de entrenamiento y 6,513 de validación, `random_state=42`). Ninguno usa `adult.test`.

```powershell
python src/train_dummy.py
python src/train_logistic.py
python src/random_forest.py
```

Cada script guarda sus predicciones en el formato común:

- `reports/predicciones_dummy_v1_validacion.csv`
- `reports/predicciones_logistic_v1_validacion.csv`
- `reports/predicciones_random_forest_v1_validacion.csv`

Random Forest tarda unos 20 segundos.

## 4. Comparación de modelos y análisis de errores

```powershell
python src/run_comparison.py
```

- Genera `reports/comparacion_modelos_validacion.csv`, `reports/errores_por_confianza_validacion.csv` y `reports/composicion_errores_por_grupo_validacion.csv`.
- Resultado esperado:

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|
| dummy | 0.7593 | 0.0000 | 0.0000 | 0.0000 |
| logistic | 0.8548 | 0.7389 | 0.6135 | 0.6704 |
| random_forest | 0.8474 | 0.7013 | 0.6378 | 0.6680 |

La interpretación está en `docs/COMPARACION_MODELOS.md`.

## 5. Equidad y calibración de la Regresión Logística

```powershell
python src/audit_fairness.py
python src/plot_fairness.py
python src/evaluate_calibration.py
python src/plot_calibration.py
```

- Equidad: `reports/auditoria_equidad_logistica_validacion.csv` y `reports/figures/equidad_logistica_*.png` (TPR y FPR por sexo, raza y grupo de edad, con intervalos de Wilson al 95 %).
- Calibración: `reports/calibracion_logistica_validacion.csv`, `reports/curva_calibracion_logistic_v1.csv`, `reports/metricas_calibracion_logistic_v1.csv` y `reports/figures/curva_calibracion_logistic_v1.png`.
- Resultado esperado: TPR de 0.576 en mujeres y 0.621 en hombres; Brier Score de 0.1004 y Log Loss de 0.3152.
- `audit_fairness.py` y `evaluate_calibration.py` vuelven a entrenar la Regresión Logística internamente; sus predicciones salen idénticas a las del paso 3.

## 6. Green AI

```powershell
python src/green_ai.py
```

- Mide los tres modelos en la misma computadora (mediana de 3 repeticiones) y guarda `reports/green_ai_validacion.csv`.
- Los tiempos dependen del equipo y cambian un poco en cada ejecución, así que no serán exactamente los del repositorio. Lo que se debe mantener es la proporción: Random Forest tarda mucho más que la Regresión Logística (en la medición oficial, 20.9 s frente a 0.22 s).
- Para conservar la medición oficial del repositorio después de la demostración:

```powershell
git restore reports/green_ai_validacion.csv
```

## 7. Evaluación final en `adult.test` (sin sobrescribir los resultados oficiales)

La evaluación final se ejecutó una sola vez y sus resultados oficiales están en `reports/final/`. El script `src/evaluate_final.py` está protegido: no se ejecuta sin `--confirmar-evaluacion-final` y no sobrescribe resultados existentes.

Para reproducirla sin tocar los archivos oficiales, se guardan los resultados en una carpeta fuera del repositorio y se comparan:

```powershell
Remove-Item ..\verificacion_demo -Recurse -ErrorAction SilentlyContinue
python -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; import evaluate_final as ef; ef.OUTPUT_DIR=Path('..')/'verificacion_demo'; ef.OUTPUT_DIR.mkdir(exist_ok=True); ef.evaluar_final()"
python -c "import pandas as pd; a=pd.read_csv('reports/final/predicciones_logistic_test.csv'); b=pd.read_csv('../verificacion_demo/predicciones_logistic_test.csv'); print('Predicciones identicas:', (a.y_pred==b.y_pred).all(), '| Dif. max. y_proba:', (a.y_proba-b.y_proba).abs().max()); print('Metricas identicas:', pd.read_csv('reports/final/metricas_logistic_test.csv').equals(pd.read_csv('../verificacion_demo/metricas_logistic_test.csv')))"
```

Resultado esperado: modelo entrenado con los 32,561 registros de `adult.data` y evaluado en los 16,281 de `adult.test`.

| Conjunto | Accuracy | Precisión | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Prueba (`adult.test`) | 0.8525 | 0.7311 | 0.5944 | 0.6557 | 0.0676 |

Y en la comparación: `Predicciones identicas: True`, diferencia máxima de `0.0` y `Metricas identicas: True`.

## 8. Pruebas automatizadas

```powershell
python -m pytest -q
```

Resultado esperado: 47 pruebas aprobadas.

## 9. Comprobación final

```powershell
git status
```

Al probar esta demostración, el único archivo que cambió fue `reports/green_ai_validacion.csv`, por los tiempos. Todas las demás tablas, predicciones y figuras se regeneraron idénticas a las del repositorio. Después de ejecutar `git restore` (paso 6), `git status` debe mostrar el repositorio sin cambios.
