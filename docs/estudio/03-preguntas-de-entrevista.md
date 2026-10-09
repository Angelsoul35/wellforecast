# 3. Preguntas de entrevista

## Nivel junior / media

**¿Qué hace WellForecast en una frase?**
Pronostica la tasa de producción de petróleo de pozos en declive combinando la curva
de declinación de Arps (el modelo del dominio, interpretable) con una corrección de
gradient boosting, y lo valida con una partición temporal sin fuga de datos contra un
baseline ingenuo. Todo sobre datos reales del campo Volve.

**¿Qué es la ecuación de Arps y qué significa `b`?**
`q(t) = q_i / (1 + b·D_i·t)^(1/b)`. `q_i` es la tasa inicial, `D_i` la declinación
inicial, y `b` el exponente que define la forma: `b = 0` exponencial (caída más
rápida, fracción constante por día), `0 < b < 1` hiperbólica (lo común), `b = 1`
armónica (la más lenta). Lo acotamos a `[0, 1]` para que sea físico.

**¿Por qué normalizas la tasa por horas en línea?**
Porque el volumen diario mezcla la tasa del yacimiento con cuántas horas estuvo
abierto el pozo. Un día de 6 horas produce poco volumen sin que la tasa sea baja. Uso
`volumen / horas · 24` y me quedo con días de suficiente tiempo en línea.

**¿Qué es fuga de datos y cómo la evitas?**
Usar información del futuro (o del presente) en las features. La evito con `shift(1)`
en las móviles, lags del pasado, y sobre todo con una partición **temporal**: entreno
con el 70 % inicial y pruebo con el 30 % final, nunca al azar.

## Nivel media / senior

**¿Por qué el ML predice una razón de corrección y no la tasa directa?**
Porque un modelo de árboles no extrapola por debajo del mínimo que vio en
entrenamiento, y en un pozo en declive todas las tasas futuras son más bajas: se
aplanaría. Predigo `ρ = tasa/arps`, que es estacionaria (ronda 1), y el pronóstico es
`arps(t)·ρ̂(t)`: Arps aporta la caída, el ML la corrección.

**¿Por qué ajustas Arps en error relativo (log) y no absoluto?**
Porque la tasa abarca un orden de magnitud a lo largo de la vida del pozo. En error
absoluto, los días de tasa alta dominan y la cola queda mal. El error relativo pesa a
cada punto por su error porcentual, que es lo que importa.

**¿Cómo mides sobre/subajuste si el proceso no es estacionario?**
No comparo el error de entrenamiento contra la cola lejana, porque eso mezclaría el
sobreajuste con la declinación misma (las tasas futuras son más bajas por naturaleza).
Mido la brecha en una sub-partición temporal *dentro* del entrenamiento (misma época)
y dejo la cola real intacta. Una brecha pequeña (aquí 0.058) dice que generaliza.

**¿Por qué un modelo "más pequeño" si uno más grande baja el error?**
Un modelo más pesado recorta un poco la prueba pero abre una brecha grande: memoriza
el ruido diario. Elijo los hiperparámetros por la brecha de validación interna, no
mirando la prueba, y prefiero el que generaliza.

**Explica el control de barajado.**
Reentreno con las tasas de entrenamiento barajadas en el tiempo. Si el modelo
aprendía estructura real, su error en la cola se dispara (aquí a 6.9 MAPE, muy por
encima de la persistencia real 0.73). Prueba que el borde venía de la estructura
temporal y no de ruido o de fuga.

## Nivel senior / diseño

**¿Arps o ML? ¿Cuál usarías?**
Los dos: son complementarios. Arps es interpretable, no necesita datos recientes, da
la forma de la declinación y la EUR — es lo que lee un ingeniero de yacimientos. El ML
corrige las desviaciones de esa curva con la dinámica reciente y, en la flota, le gana
tanto a Arps crudo como a la persistencia. El ML sin Arps no extrapolaría la caída; el
Arps sin ML no sigue las intervenciones.

**El ML gana en la flota pero pierde en dos pozos. ¿Problema?**
No, y lo reporto. Gana donde la cola sigue cayendo y pierde donde se aplana, que es el
mejor caso de la persistencia (`F-12`, y el corto y ruidoso `F-1 C`). Esconder eso
estaría mal ocultarlo; mostrarlo es parte del rigor. La métrica de la flota agrupada es la
que resume el valor.

**¿Qué le falta para producción?**
Ajustes de declinación por segmentos (para seguir intervenciones), un modelo
multi-horizonte directo para adelantos largos (el recursivo acumula error), intervalos
de predicción, pronóstico de la presión futura como feature exógena, y un panel para
seguir el error por pozo en el tiempo.

## Reto para ti

* Añade un ajuste de Arps por segmentos (re-anclado a una ventana reciente) y compara
  el error a horizontes de 90 y 180 días.
* Cambia el pronóstico recursivo por uno directo (un modelo por horizonte) y mide si
  mejora la cola larga.
* Agrega la presión, la GOR y la WOR como features de la corrección y cuantifica
  cuánto aportan.
* Añade intervalos de predicción (cuantiles) con gradient boosting en modo cuantil.
