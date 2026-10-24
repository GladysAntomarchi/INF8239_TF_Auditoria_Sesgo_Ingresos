
# Auditoría de sesgo en ingresos mediante Machine Learning

**Asignatura:** INF-8239 - Ciencia de Datos II  
**Tema:** 43 - Auditoría de sesgo en ingresos  
**Modalidad:** A - Proyecto aplicado  
**Equipo:** 2 integrantes

## Objetivo

Desarrollar y evaluar modelos de clasificación binaria de ingresos, comparando su desempeño global y por grupos poblacionales para identificar posibles disparidades y formular recomendaciones fundamentadas.

## Dataset

Adult (Census Income) - UCI Machine Learning Repository.

https://archive.ics.uci.edu/dataset/2/adult

## Modelos previstos

- Baseline: DummyClassifier.
- Modelo 1: Regresión Logística.
- Modelo 2: Random Forest.

## Evaluación

- F1-score.
- TPR y FPR globales y por grupos.
- Calibración.
- Análisis de errores, reproducibilidad y Green AI.

## Estructura

- `data/`: datos originales y procesados.
- `src/`: código fuente.
- `tests/`: pruebas automatizadas.
- `notebooks/`: análisis exploratorio.
- `reports/`: resultados y visualizaciones.
- `models/`: modelos entrenados.
- `docs/`: documentación.

## Estado

Proyecto en fase inicial de configuración. Los modelos y resultados todavía no han sido desarrollados.
