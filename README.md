
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

**Importante:** las pruebas de auditoría utilizan los datos originales, por lo que es necesario completar la descarga antes de ejecutar Pytest.

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


## Pruebas automatizadas

Para ejecutar las pruebas de auditoría que utilizan los datos originales,
primero descargar el dataset:

```powershell
python src/download_data.py
```

Después, ejecutar todas las pruebas desde la raíz del proyecto:

```powershell
python -m pytest tests/ -v
```

Se incluyen cinco pruebas automatizadas:

1. Verificación de columnas esperadas.
2. Validación de categorías de ingresos.
3. Detección de valores faltantes.
4. Verificación de coincidencias entre particiones.
5. Sustitución de archivos vacíos mediante una descarga simulada.

La quinta prueba utiliza archivos temporales y no requiere conexión a
Internet. Las cuatro primeras utilizan el dataset original descargado.

El proceso de descarga incluye comprobaciones básicas de existencia y
contenido, así como el uso de archivos temporales. Estas comprobaciones
no constituyen una verificación criptográfica de integridad.
```

## Modelos previstos

- Baseline: `DummyClassifier`.
- Modelo 1: Regresión Logística.
- Modelo 2: Random Forest.

## Evaluación prevista

- Accuracy, Precision, Recall y F1-score.
- TPR y FPR globales y por grupos.
- Calibración probabilística.
- Análisis de errores.
- Reproducibilidad y Green AI.

## Estructura del proyecto

- `data/`: datos originales y procesados.
- `src/`: código fuente.
- `tests/`: pruebas automatizadas.
- `notebooks/`: análisis exploratorio.
- `reports/`: resultados y visualizaciones.
- `models/`: modelos entrenados.
- `docs/`: documentación.

## Estado del proyecto

La adquisición y auditoría inicial del dataset se encuentran desarrolladas, con cuatro pruebas automatizadas verificadas.

El entrenamiento, la comparación de modelos y las evaluaciones de equidad se integrarán en etapas posteriores.
