# Comparación de modelos, análisis de errores y Green AI

Issue relacionado: #9
Autora: Emma Cabrera

> **Estado: resultados preliminares.** Faltan el baseline (`DummyClassifier`) en la comparación de métricas y la evaluación final en `adult.test`. Los valores de Regresión Logística y Random Forest no van a cambiar, porque sus predicciones de validación ya están fijas.

## 1. Cómo se obtuvieron los resultados

- Todos los modelos usan la misma partición de validación (6,513 registros de `adult.data`) y el mismo preprocesamiento, según el acuerdo metodológico (#6).
- Las predicciones de cada modelo se guardan en `reports/predicciones_<modelo>_v1_validacion.csv`. Antes de calcular las métricas, `consolidar_predicciones()` comprueba que todos los modelos tengan los mismos registros, las mismas etiquetas reales y los mismos atributos de auditoría.
- Las métricas se calculan con `src/evaluation.py`, con el umbral de 0.5 para todos los modelos.
- Reproducción:
  - `python src/run_comparison.py` (métricas y análisis de errores)
  - `python src/green_ai.py` (tiempos)

## 2. Desempeño global en validación

| Modelo | Accuracy | Precisión | Recall | F1 (>50K) | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (Dummy) | pendiente | | | | | | | |
| Regresión Logística | 0.8548 | 0.7389 | 0.6135 | **0.6704** | 962 | 340 | 4,605 | 606 |
| Random Forest | 0.8474 | 0.7013 | 0.6378 | **0.6680** | 1,000 | 426 | 4,519 | 568 |

Interpretación:

- Los dos modelos tienen casi el mismo F1, que es el criterio principal del acuerdo. La diferencia es de 0.0024, demasiado pequeña para decir que uno es mejor que el otro con una sola partición de validación.
- Tienen perfiles de error distintos. Random Forest encuentra más personas con ingresos `>50K` (recall 0.638 frente a 0.614, es decir, 38 falsos negativos menos), pero a cambio comete 86 falsos positivos más (precisión 0.701 frente a 0.739).
- El accuracy de los dos modelos (0.85) parece alto, pero hay que compararlo con el 75.9 % que obtendría un modelo que siempre predice `<=50K` (ver `docs/EDA_RESUMEN.md`). Por eso no usamos el accuracy para elegir el modelo.

## 3. Análisis de errores

### 3.1 Errores según la probabilidad estimada

| Intervalo de probabilidad | LR: n | LR: errores | LR: tasa | RF: n | RF: errores | RF: tasa |
|---|---:|---:|---:|---:|---:|---:|
| [0.0, 0.1] | 3,210 | 73 | 2.3 % | 3,550 | 140 | 3.9 % |
| (0.1, 0.3] | 1,235 | 239 | 19.4 % | 985 | 223 | 22.6 % |
| (0.3, 0.5] | 766 | 294 | 38.4 % | 562 | 209 | 37.2 % |
| (0.5, 0.7] | 560 | 234 | 41.8 % | 469 | 237 | 50.5 % |
| (0.7, 0.9] | 477 | 96 | 20.1 % | 444 | 126 | 28.4 % |
| (0.9, 1.0] | 265 | 10 | 3.8 % | 503 | 59 | 11.7 % |

Interpretación:

- En los dos modelos, la mayoría de los errores están cerca del umbral (entre 0.3 y 0.7). Es lo esperado: son los casos más difíciles de separar.
- La diferencia importante está en los extremos. Random Forest asigna probabilidades muy altas (mayores que 0.9) a casi el doble de registros que la Regresión Logística (503 frente a 265), y en ese intervalo se equivoca en el 11.7 % de los casos, frente al 3.8 % de la Regresión Logística. En el intervalo más bajo pasa lo mismo: 3.9 % de errores frente a 2.3 %.
- Esto indica que Random Forest comete más **errores con alta confianza**: cuando se equivoca, a veces lo hace muy seguro. Para un uso real esto es más grave que un error cerca del umbral, porque una probabilidad de 0.95 no avisa de que hay que revisar el caso. La evaluación de calibración de Random Forest podrá confirmarlo.

### 3.2 Composición de los errores por grupo

| Grupo | n | LR: FP | LR: FN | LR: tasa de error | RF: FP | RF: FN | RF: tasa de error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Female | 2,158 | 46 | 104 | 7.0 % | 59 | 98 | 7.3 % |
| Male | 4,355 | 294 | 502 | 18.3 % | 367 | 470 | 19.2 % |
| White | 5,533 | 302 | 532 | 15.1 % | 374 | 509 | 16.0 % |
| Black | 662 | 15 | 47 | 9.4 % | 29 | 36 | 9.8 % |
| Asian-Pac-Islander | 200 | 20 | 21 | 20.5 % | 15 | 18 | 16.5 % |
| Amer-Indian-Eskimo | 73 | 1 | 6 | 9.6 % | 8 | 4 | 16.4 % |
| Other | 45 | 2 | 0 | 4.4 % | 0 | 1 | 2.2 % |
| <25 | 1,129 | 8 | 8 | 1.4 % | 4 | 7 | 1.0 % |
| 25-44 | 3,304 | 168 | 347 | 15.6 % | 222 | 317 | 16.3 % |
| 45-64 | 1,829 | 139 | 244 | 20.9 % | 181 | 226 | 22.3 % |
| 65+ | 251 | 25 | 7 | 12.7 % | 19 | 18 | 14.7 % |

Interpretación:

- La tasa de error más alta en hombres y en el grupo de 45-64 años no significa por sí sola que los modelos traten peor a esos grupos. En esos grupos hay más personas con ingresos `>50K` (ver el EDA), y por eso hay más casos difíciles cerca del umbral. Los grupos con muy pocos casos positivos, como `<25`, tienen pocos errores porque casi todos se clasifican como `<=50K`.
- En mujeres, la mayoría de los errores son falsos negativos (104 de 150 en Regresión Logística): el modelo tiende a no reconocer a las mujeres que sí tienen ingresos `>50K`. La auditoría de TPR por grupo, a cargo de Gladys, mide esta diferencia con intervalos de confianza.
- En los grupos pequeños (Amer-Indian-Eskimo con 73 registros, Other con 45) las tasas cambian mucho entre modelos con solo unos pocos errores. Por ejemplo, Amer-Indian-Eskimo pasa de 7 a 12 errores. Con estos tamaños las diferencias no son concluyentes.
- Este análisis complementa la auditoría de equidad, pero no la reemplaza: aquí se describe cómo se reparten los errores, y la auditoría compara las tasas TPR y FPR entre grupos.

## 4. Green AI

Los tres modelos se midieron en la misma computadora, con la mediana de tres repeticiones (`reports/green_ai_validacion.csv`).

Entorno: Windows 11, Intel Core i5-10310U (1.70 GHz, 8 núcleos lógicos), 31.6 GB de RAM, Python 3.14.5, scikit-learn 1.9.1.

| Modelo | Entrenamiento (s) | Predicción (s) | Entrenamiento mín.–máx. (s) |
|---|---:|---:|---:|
| Baseline (Dummy) | 0.070 | 0.016 | 0.061 – 0.087 |
| Regresión Logística | 0.224 | 0.015 | 0.212 – 0.227 |
| Random Forest | 20.889 | 0.186 | 20.771 – 23.727 |

Interpretación:

- Random Forest tarda unas 93 veces más que la Regresión Logística en entrenar y unas 12 veces más en predecir. Este costo extra no se traduce en un mejor F1.
- Random Forest usa los valores predeterminados de scikit-learn (100 árboles y un solo núcleo). Con más núcleos el tiempo bajaría, pero el consumo total de cómputo sería parecido, porque los 100 árboles se construyen igual.
- Los tiempos dependen de la computadora; por eso comparamos los tres modelos en la misma máquina y no usamos mediciones de equipos distintos.

## 5. Conclusión preliminar

Con los criterios del acuerdo (F1 como criterio principal, disparidades, calibración y costo computacional), la **Regresión Logística es la opción preferida por ahora**:

- Tiene un F1 prácticamente igual al de Random Forest.
- Comete menos errores con alta confianza.
- Es unas 93 veces más barata de entrenar.
- Además es más fácil de interpretar, lo que ayuda en una auditoría de sesgo.

Random Forest solo sería preferible si se diera más importancia al recall (encontrar más personas con ingresos `>50K`) que a la precisión y al costo.

Esta conclusión se debe confirmar con:

1. El baseline en la tabla de desempeño.
2. La auditoría de equidad y la calibración de Random Forest, con los mismos métodos que se usaron para la Regresión Logística.
3. La evaluación final en `adult.test`, una sola vez, con la configuración elegida.
