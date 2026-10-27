
# Dataset Card — UCI Adult

## 1. Identificación

- Nombre: Adult (Census Income).
- Fuente: UCI Machine Learning Repository.
- URL: https://archive.ics.uci.edu/dataset/2/adult
- Origen: información del censo de Estados Unidos de 1994.
- Licencia: Creative Commons Attribution 4.0 (CC BY 4.0).
- Tipo de problema: clasificación binaria.

## 2. Objetivo

Clasificar si los ingresos anuales de una persona superan los USD 50,000, utilizando características demográficas, educativas y laborales.

## 3. Estructura

- Registros totales: 48,842.
- Entrenamiento: 32,561.
- Prueba: 16,281.
- Características predictoras: 14.
- Variable objetivo: income.

Clases:
- <=50K
- >50K

## 4. Calidad de datos

Resultados obtenidos durante la inspección inicial:

| Indicador | Entrenamiento | Prueba |
|---|---:|---:|
| Registros | 32,561 | 16,281 |
| Registros con faltantes | 2,399 | 1,221 |
| Duplicados exactos | 24 | 5 |

Variables con valores faltantes:
- workclass
- occupation
- native_country

Se identificaron 26 ocurrencias en el conjunto de prueba cuyas características coinciden exactamente con registros de entrenamiento.

## 5. Atributos para la auditoría

Se consideran inicialmente:
- sex
- race
- age

Su utilización permitirá examinar diferencias de desempeño entre grupos poblacionales.

La inclusión de atributos sensibles en el entrenamiento será una decisión metodológica independiente de su utilización para evaluar disparidades.

## 6. Riesgos y limitaciones

- Antigüedad del conjunto de datos.
- Representatividad limitada respecto de poblaciones actuales.
- Diferencias en el tamaño de los grupos poblacionales.
- Valores faltantes en variables categóricas.
- Coincidencias entre entrenamiento y prueba.
- Posibilidad de que variables aparentemente neutrales actúen como indicadores indirectos de características sensibles.

Las disparidades estadísticas identificadas no constituyen, por sí solas, evidencia concluyente de discriminación.

## 7. Uso previsto

Dataset destinado exclusivamente a investigación académica, entrenamiento, comparación y auditoría experimental de modelos de clasificación.

No se considera adecuado para fundamentar decisiones reales sobre personas sin validaciones adicionales.

## 8. Reproducibilidad

Los archivos originales se obtienen mediante:

python src/download_data.py

El análisis inicial se reproduce mediante:

python src/inspect_data.py

Los archivos descargados se conservan localmente en data/raw/ y no se incluyen en el repositorio Git.
