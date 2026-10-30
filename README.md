# Auditoría de sesgo en ingresos mediante Machine Learning

**Asignatura:** INF-8239 - Ciencia de Datos II  
**Tema:** 43 - Auditoría de sesgo en ingresos  
**Modalidad:** A - Proyecto aplicado  
**Equipo:** 2 integrantes

## Objetivo

Desarrollar y evaluar modelos de clasificación binaria de ingresos utilizando el dataset UCI Adult, comparando su desempeño global y por grupos poblacionales para identificar posibles disparidades y formular recomendaciones fundamentadas.

## Dataset

**Nombre:** Adult (Census Income)
**Fuente:** UCI Machine Learning Repository
**Enlace:** https://archive.ics.uci.edu/dataset/2/adult

El dataset contiene 48,842 registros distribuidos en:

- Entrenamiento original: 32,561 registros.
- Prueba oficial: 16,281 registros.
- Variables predictoras originales: 14.
- Variable objetivo: `income`.

El problema consiste en clasificar los ingresos en dos categorías: `<=50K` y `>50K`.

Los archivos originales se descargan desde UCI y no se incluyen en el repositorio Git.

## Requisitos

- Python 3.14.
- Git.
- Dependencias especificadas en `requirements.txt`.

## Instalación

Clonar el repositorio:

```powershell
git clone https://github.com/GladysAntomarchi/INF8239_TF_Auditoria_Sesgo_Ingresos.git
cd INF8239_TF_Auditoria_Sesgo_Ingresos
```

Crear y activar el entorno virtual en Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

## Descarga del dataset

Antes de ejecutar la auditoría y las pruebas automatizadas, descargar los archivos originales:

```powershell
python src/download_data.py
```

Los archivos se almacenan en `data/raw/`.

**Importante:** varias pruebas utilizan los datos originales, por lo que es necesario completar la descarga antes de ejecutar Pytest.

## Auditoría inicial de calidad de datos

Ejecutar:

```powershell
python src/inspect_data.py
```

Este procedimiento permite verificar:

- Dimensiones y estructura de los datos.
- Distribución de la variable objetivo.
- Valores faltantes.
- Registros duplicados.
- Distribución de variables demográficas.
- Coincidencias entre entrenamiento y prueba.

El resumen general de auditoría se guarda en:

`reports/auditoria_inicial.csv`

La documentación del dataset se encuentra en:

`docs/DATASET_CARD.md`

## Demostración reproducible

La guía paso a paso para reproducir todo el proyecto (datos, EDA, modelos, comparación, equidad, calibración, Green AI, evaluación final y pruebas) está en [docs/DEMO.md](docs/DEMO.md).

## Modelos

Los tres modelos usan la misma partición 80/20 estratificada de `adult.data` (26,048 registros de entrenamiento y 6,513 de validación, `random_state=42`) y el mismo preprocesamiento, según el acuerdo metodológico del equipo (Issue #6).

| Modelo | Script | Configuración |
|---|---|---|
| Baseline | `src/train_dummy.py` | `DummyClassifier(strategy="most_frequent")` |
| Regresión Logística | `src/train_logistic.py` | Con escalado; `max_iter=2000`, `random_state=42`, `class_weight=None` |
| Random Forest | `src/random_forest.py` | Sin escalado; valores predeterminados (100 árboles), `random_state=42`, `class_weight=None` |

## Evaluación

| Análisis | Script | Resultados |
|---|---|---|
| Análisis exploratorio (solo `adult.data`) | `src/eda.py` | `reports/eda/`, `reports/figures/eda_*.png`, `docs/EDA_RESUMEN.md` |
| Comparación de modelos y análisis de errores | `src/run_comparison.py` | `reports/comparacion_modelos_validacion.csv` y tablas de errores; `docs/COMPARACION_MODELOS.md` |
| Equidad (TPR y FPR por grupo, con intervalos de Wilson) | `src/audit_fairness.py`, `src/plot_fairness.py` | `reports/auditoria_equidad_logistica_validacion.csv`, `reports/figures/equidad_logistica_*.png` |
| Calibración (Brier Score y Log Loss) | `src/evaluate_calibration.py`, `src/plot_calibration.py` | `reports/calibracion_logistica_validacion.csv`, `reports/figures/curva_calibracion_logistic_v1.png` |
| Green AI (tres modelos en la misma computadora) | `src/green_ai.py` | `reports/green_ai_validacion.csv` |
| Evaluación final en `adult.test` (una sola vez) | `src/evaluate_final.py` | `reports/final/` |

## Resultados principales

Validación (6,513 registros, umbral 0.5):

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline (Dummy) | 0.7593 | 0.0000 | 0.0000 | 0.0000 |
| Regresión Logística | 0.8548 | 0.7389 | 0.6135 | 0.6704 |
| Random Forest | 0.8474 | 0.7013 | 0.6378 | 0.6680 |

Se seleccionó la **Regresión Logística**: su F1 es prácticamente igual al de Random Forest, comete menos errores con alta confianza y su entrenamiento es unas 93 veces más rápido.

Evaluación final en `adult.test` (16,281 registros), con el modelo entrenado en los 32,561 registros de `adult.data`:

| Modelo | Accuracy | Precisión | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Regresión Logística | 0.8525 | 0.7311 | 0.5944 | 0.6557 | 0.0676 |

## Pruebas automatizadas

Después de descargar el dataset, ejecutar todas las pruebas desde la raíz del proyecto:

```powershell
python -m pytest tests/ -v
```

El proyecto incluye **47 pruebas automatizadas**, que verifican:

- La descarga y la carga de los datos, las columnas, las categorías de ingresos, los valores faltantes y las coincidencias entre particiones.
- El preprocesamiento, la partición compartida y el formato común de predicciones.
- El análisis exploratorio, incluidos los grupos con valores faltantes.
- El entrenamiento y las predicciones del baseline, la Regresión Logística y Random Forest.
- Las métricas, el análisis de errores, la equidad y la calibración.
- La evaluación final en `adult.test`.

Las pruebas de los componentes de análisis usan datos pequeños y controlados; las de auditoría de datos usan el dataset original descargado.

El proceso de descarga incluye comprobaciones básicas de existencia y contenido, así como el uso de archivos temporales. Estas comprobaciones no constituyen una verificación criptográfica de integridad.

## Estructura del proyecto

- `data/`: datos originales (no se incluyen en Git).
- `src/`: código fuente.
- `tests/`: pruebas automatizadas.
- `reports/`: resultados, tablas y figuras.
- `docs/`: documentación (dataset, EDA, comparación de modelos y demostración).

## Uso de inteligencia artificial

Las declaraciones de uso de inteligencia artificial de cada integrante se incluyen como anexos del informe técnico.

## Estado del proyecto

El proyecto está completo: auditoría de datos, análisis exploratorio, tres modelos, comparación y análisis de errores, equidad, calibración, Green AI y evaluación final en `adult.test`. Todos los cambios se integraron mediante Pull Requests con revisión cruzada, y pasan las 47 pruebas automatizadas.
