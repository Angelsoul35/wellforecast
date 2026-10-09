# 1. Curvas de declinación (Arps)

## Qué es la declinación de un pozo

Cuando un pozo de petróleo empieza a producir, la presión del yacimiento es alta y la
tasa de producción (barriles o Sm³ por día) es grande. A medida que se extrae
petróleo, la presión baja y la tasa **declina**. Esa caída es tan universal que en
1945 J. J. Arps la describió con una fórmula empírica de una sola línea que la
industria sigue usando hoy para estimar reservas y pronosticar producción.

> **Concepto — tasa vs volumen.** El dato crudo de Volve es `BORE_OIL_VOL`, el
> *volumen* de petróleo producido en un día. Pero un día que solo fluyó 6 horas
> reporta poco volumen sin que eso sea una tasa baja. Por eso usamos la **tasa de
> producción** normalizada por horas en línea:
> `tasa = volumen / horas_en_línea · 24` (Sm³/día), y solo en días con suficiente
> tiempo en línea. Esa es la tasa que la curva de declinación modela.

## La ecuación de Arps

```
q(t) = q_i / (1 + b · D_i · t) ^ (1 / b)
```

* `q(t)` es la tasa en el día `t`.
* `q_i` es la tasa inicial (al empezar la declinación, `t = 0`).
* `D_i` es la tasa de declinación nominal inicial (por día), `D_i > 0`.
* `b` es el **exponente de declinación**, acotado a `0 ≤ b ≤ 1`.

La declinación describe la caída **después** del pico. El arranque inicial (cuando la
tasa aún sube) no es declinación, así que localizamos el pico (sobre la tasa suavizada
a 30 días) y medimos `t` como los días desde el pico.

## El factor `b`: la forma de la caída

El exponente `b` elige la *forma* de la curva. Es el parámetro que un ingeniero de
yacimientos mira primero:

* **`b = 0` → declinación exponencial.** Es el límite de la fórmula cuando `b → 0`:
  `q(t) = q_i · e^(−D_i · t)`. Se pierde una **fracción constante** por unidad de
  tiempo. Es la caída más rápida y suele darse en yacimientos sin soporte de presión.
* **`0 < b < 1` → declinación hiperbólica.** El caso común. La tasa de declinación
  *misma* se desacelera con el tiempo (por soporte de presión, cambios de
  permeabilidad relativa, etc.). La curva cae rápido al principio y se aplana.
* **`b = 1` → declinación armónica.** `q(t) = q_i / (1 + D_i · t)`. La caída más lenta
  de las tres.

> **Por qué se acota `b` a `[0, 1]`.** Fuera de ese rango la curva deja de tener
> sentido físico (un `b > 1` implica una cola de producción que no se agota, lo cual
> sobrestima reservas). En el código, `scipy.optimize.curve_fit` recibe esos límites,
> así el ajuste se queda en la zona física aunque el ruido empuje hacia afuera.

En los pozos reales de Volve vemos las tres formas: `F-14 H` y `F-12 H` salen casi
exponenciales (`b ≈ 0`), mientras `F-15 D` y `F-1 C` salen armónicos (`b = 1`).

## Cómo ajustamos la curva (y por qué en error relativo)

El ajuste es **mínimos cuadrados no lineales** con `scipy.optimize.curve_fit`. Pero no
minimizamos el error absoluto sino el **error relativo**: regresamos `log q` en vez de
`q`. La razón es importante:

> **Concepto — error relativo.** La tasa de un pozo cae de miles a decenas de Sm³/día
> a lo largo de su vida. Si minimizamos el error absoluto, los días de tasa alta
> (cerca del pico) dominan el ajuste y la cola queda mal modelada. Un error de 50
> Sm³/día importa poco cuando produces 4000, pero muchísimo cuando produces 100. El
> error relativo (log) pesa a cada punto por su error *porcentual*, que es lo que
> queremos en una magnitud que abarca un orden de magnitud.

Verificamos que el ajuste es correcto con una prueba decisiva: generamos una curva de
Arps con parámetros conocidos (`q_i`, `D_i`, `b`) y comprobamos que el ajuste los
**recupera casi exactos** (ver `tests/test_arps.py`). Si la fórmula estuviera mal, esa
prueba fallaría — y lo verificamos rompiéndola a propósito.

## Producción acumulada y EUR

Integrando la tasa se obtiene la **producción acumulada** `Q(t)`, base de la EUR
(*Estimated Ultimate Recovery*, recuperación final estimada). Para el caso hiperbólico:

```
Q(t) = q_i / (D_i · (1 − b)) · (1 − (1 + b · D_i · t) ^ (1 − 1/b))
```

con los límites exponencial (`Q = q_i/D_i · (1 − e^(−D_i t))`) y armónico
(`Q = q_i/D_i · ln(1 + D_i t)`) resueltos aparte para que sean numéricamente estables.
Extrapolando la curva hacia el futuro y acumulando, Arps da una estimación de cuánto
petróleo queda por producir — el número que mueve decisiones de inversión y abandono.

## El límite de una sola curva

Una sola curva de Arps es suave: no puede seguir las **intervenciones** que sufre un
pozo real (cambios de choke, estimulaciones, soporte por inyección de agua). Por eso,
extrapolar una sola curva a dos años vista da un error apreciable, y por eso añadimos
una capa de ML que *corrige* la curva usando la dinámica reciente. Eso es el tema del
siguiente capítulo. Arps no deja de ser valioso: es el modelo interpretable que aporta
la *forma* de la declinación y la EUR, y la referencia que lee un ingeniero.
