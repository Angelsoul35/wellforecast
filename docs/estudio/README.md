# Guía de estudio — WellForecast

Esta guía está en español y explica el proyecto para que lo estudies y lo puedas
sustentar en una entrevista. El código y la documentación técnica están en inglés;
aquí va el *porqué*: la física de las curvas de declinación, el modelo de machine
learning, y por qué la forma de validar es la que hace que los números sean creíbles.

## Orden de lectura

1. [Curvas de declinación (Arps)](01-curvas-de-declinacion.md) — qué es la
   declinación de un pozo, la ecuación de Arps, el factor `b`, y los casos
   exponencial, hiperbólico y armónico. Cómo se ajusta y cómo da la EUR.
2. [ML, validación y métricas](02-ml-validacion-metricas.md) — por qué el ML corrige
   la curva (y no predice la tasa directa), las features sin fuga de datos, la
   partición temporal, las métricas, sobre/subajuste con la brecha, el control de
   barajado, y un ejemplo con los números reales de Volve.
3. [Preguntas de entrevista](03-preguntas-de-entrevista.md) — preguntas típicas, de
   nivel junior a senior, con respuestas.

## Cómo correrlo mientras estudias

```bash
make install
make benchmark   # el benchmark completo sobre los pozos reales de Volve
make demo        # ajusta y pronostica un pozo, y luego el benchmark de la flota
make test        # 44 pruebas con cobertura
```

El concepto central: **pronosticar la tasa de producción de petróleo de un pozo en
declive**, combinando la curva de declinación de Arps (el modelo del dominio,
interpretable) con una corrección de machine learning, y midiendo ambos de forma
limpia (partición temporal, sin fuga de datos) contra un baseline ingenuo. Los datos
son reales: el campo Volve de Equinor (Mar del Norte, 2008–2016).
