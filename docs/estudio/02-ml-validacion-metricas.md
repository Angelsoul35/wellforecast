# 2. ML, validación y métricas

## Por qué el ML corrige la curva (y no predice la tasa directa)

La idea ingenua sería: entrena un modelo (gradient boosting) para predecir la tasa de
petróleo a partir de features. **No funciona en un pozo en declive**, y la razón es
instructiva:

> **Concepto — los árboles no extrapolan.** Un modelo de árboles (gradient boosting,
> random forest) nunca predice un valor por debajo del mínimo que vio en
> entrenamiento. En un pozo que declina, *todas* las tasas futuras son más bajas que
> las pasadas, así que el modelo se "aplana" en su mínimo de entrenamiento y falla.

La solución, que además es exactamente lo que pide el problema ("corregir la tendencia
de Arps"), es predecir la **razón de corrección** a la curva de Arps:

```
ρ(t) = tasa(t) / arps(t)            (estacionaria, ronda 1)
pronóstico = arps(t) · ρ̂(t)
```

`ρ` es **estacionaria**: vale ≈ 1 donde el pozo sigue la curva y se aleja de 1 donde
hay una intervención. Un árbol maneja bien un objetivo estacionario, y la curva de
Arps —que sí extrapola la caída por construcción— aporta la tendencia. El ML solo
aprende las *desviaciones*. Ese es el sentido preciso de "corregir la tendencia de
Arps".

## Las features (sin fuga de datos)

El modelo de corrección usa, para el día `t`, solo información **anterior** a `t`
(todas las features llevan `shift`):

* `rho_lag_{1,7,14}` — la razón de corrección hace 1, 7 y 14 días.
* `rho_roll_{7,30}` — sus medias móviles de 7 y 30 días (el nivel de la desviación).
* `rho_slope` — el cambio reciente de la desviación (¿se aleja o vuelve a la curva?).
* `t` — días de producción desde el pico (en qué punto de la vida estamos).

El pronóstico multi-paso es **recursivo**: se predice `ρ` de mañana, se agrega a la
historia, y con esa historia se construyen las features del día siguiente. Como `ρ` es
estacionaria, la recursión no se dispara (a diferencia de predecir la tasa directa).

> **Concepto — fuga de datos (leakage).** Una feature del día `t` nunca puede usar el
> valor de `t` ni del futuro. Lo garantizamos con `shift(1)` en todas las móviles y
> con lags del pasado. Una prueba lo demuestra de forma fuerte: perturbamos el valor
> de un día y comprobamos que **ninguna fila de features anterior** cambia.

## La partición temporal (el corazón de la honradez)

Partimos cada serie **en el tiempo**, no al azar: el 70 % inicial entrena, el 30 %
final queda retenido. Nunca se baraja el tiempo y los modelos nunca ven una tasa del
futuro al entrenar.

> **Por qué nunca al azar.** Barajar y hacer un split aleatorio dejaría días del
> futuro en entrenamiento, el modelo "vería" lo que debe predecir, y el error de
> prueba sería una mentira optimista. En series de tiempo, la partición es siempre
> cronológica. La prueba `test_temporal_split_never_leaks_future_into_past` verifica
> que `train.date.max() < test.date.min()`.

## Las métricas

* **MAE** (error absoluto medio, en Sm³/día): `MAE = (1/n) · Σ |yᵢ − ŷᵢ|`.
* **RMSE** (raíz del error cuadrático medio): `RMSE = √( (1/n) · Σ (yᵢ − ŷᵢ)² )`.
  Castiga más los errores grandes.
* **MAPE** (error porcentual absoluto medio): `MAPE = (1/n) · Σ |yᵢ − ŷᵢ| / |yᵢ|`.
  Es adimensional, así que permite **agrupar** pozos de tasas muy distintas.

## Arps vs ML vs persistencia

Comparamos tres pronósticos de la cola retenida:

1. **Arps**: la curva ajustada en entrenamiento, extrapolada. Interpretable; no usa
   datos recientes; da la forma y la EUR. Pero una sola curva no sigue las
   intervenciones.
2. **ML**: la corrección de gradient boosting sobre Arps.
3. **Persistencia (baseline ingenuo "último valor")**: la última tasa de
   entrenamiento, mantenida plana. El suelo que el ML debe superar.

## Sobre/subajuste y la brecha

Reportar solo el error no basta; hay que saber si el modelo generaliza o memoriza.

> **Concepto — la brecha, bien medida.** En un proceso **no estacionario** (el pozo
> declina), comparar el error una-paso del entrenamiento contra la cola lejana
> confundiría el sobreajuste con la propia declinación (las tasas futuras son
> simplemente más bajas). Por eso medimos la brecha en una sub-partición temporal
> *dentro* del entrenamiento (misma época, sin cambio de distribución) y **nunca
> tocamos la cola retenida real**. Una brecha pequeña dice: el modelo generaliza.

Elegimos árboles poco profundos y con hojas grandes a propósito: un modelo más pesado
baja un poco el error de prueba pero abre una brecha grande (memoriza el ruido diario).
La elección del modelo se hace por la brecha de validación interna, no mirando la
prueba.

## El control de barajado

Para probar que el ML aprende **estructura temporal real** y no ruido: reentrenamos
con las tasas de entrenamiento **barajadas en el tiempo**. Si el modelo aprendía algo
real, su error en la cola debe dispararse por encima de la persistencia real. Lo hace
(ver más abajo). Un modelo que "acertara" con datos barajados estaría midiendo fuga, no
aprendizaje.

## Ejemplo con los números reales (Volve)

```
    pozo   train  test     b      régimen   Arps MAPE  ML MAPE  naive MAPE  ML MAE
  F-14 H    1807   775  0.00  exponencial      0.869    0.640       1.348     112
  F-12 H    1701   730  0.01  exponencial      0.594    0.618       0.385     274
  F-11 H     347   149  0.00  exponencial      0.333    0.143       0.188      67
  F-15 D     522   224  1.00    armónico       0.316    0.198       0.222      30
   F-1 C     289   124  1.00    armónico       0.488    0.606       0.521     106
  --------------------------------------------------------------------------------
  FLOTA (días retenidos, agrupados)            0.643    0.543       0.733     158

  MAE flota (Sm³/día): Arps 183  ML 158  naive 175
  brecha una-paso (dentro de train): 0.0676 / 0.1258  (gap 0.0582)
  control de barajado: ML 6.879 vs persistencia real 0.733  -> el borde colapsa
  veredicto: bien ajustado (well-fit)
```

Lectura:

* Sobre una cola exigente de varios años, el ML **recorta el MAPE agrupado un ~25 %**
  frente a la persistencia (0.543 vs 0.733) y le gana a la extrapolación cruda de Arps
  (0.643). Son errores **reales** de decenas de por ciento, no un 1.0 falso: un
  pronóstico diario a dos años es difícil y los pozos son ruidosos.
* La **brecha** una-paso (dentro del entrenamiento) es apenas 0.058: el modelo
  generaliza, no memoriza. De ahí el veredicto *well-fit*.
* Por pozo, el ML gana donde la cola sigue cayendo (`F-14`, `F-11`, `F-15`) y pierde
  donde se aplana (`F-12`, y el corto y ruidoso `F-1 C`) — el mejor caso de la
  persistencia. Se reporta, no se esconde.
* El **control de barajado** dispara el error del ML a 6.879 (muy por encima de la
  persistencia real 0.733): el borde venía de estructura temporal real.

## La lección

- Reporta el error **fuera de muestra** y **contra baselines**, nunca el de
  entrenamiento.
- En series de tiempo, parte **cronológicamente**; mide la brecha donde no se confunda
  con la no estacionariedad.
- Arps y ML son **complementarios**: Arps aporta la física y la forma interpretable;
  el ML corrige las desviaciones con la dinámica reciente. Juntos ganan.
