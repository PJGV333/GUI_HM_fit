# Etapa 3.1 — informe provisional

**Estado: incompleta.** El núcleo LM y un Jacobiano espectral para columnas
independientes están implementados y probados de forma aislada. No están
conectados a Espectroscopía ni a RMN; Powell sigue siendo el optimizador
existente. No se ha ejecutado la comparación ni los tiempos exigidos y no hay
evidencia para cambiar el valor por defecto. No interpretar este informe como
aprobación de la etapa.

## Verificación

En la base `8f72278`, `QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 .venv/bin/pytest -q`: 547 aprobadas, 3 omitidas, 5
seleccionadas fuera y 17 xfail. Con el núcleo y sus pruebas: 556 aprobadas,
3 omitidas, 5 seleccionadas fuera y 17 xfail. Las nuevas pruebas contrastan
el Jacobiano con diferencias centrales del residuo real y la norma del
residuo con el objetivo existente, para NNLS y penalización suave, en ambos
modos, con y sin restricciones activas. Son pruebas sintéticas acotadas; no
validan todavía el flujo integrado, suavizado ni cotas inferiores de ε.

## Decisiones tomadas por el agente

- **Objetivo legacy:** respetar el residuo de datos de Powell, en vez de
  añadirle la penalización interior. En `absolute` se incluye la penalización
  exterior existente. Alternativa descartada: unificar los objetivos, porque
  cambiaría el estimador. Evidencia: `_spectral_profile_objective` y pruebas
  que contrastan norma del residuo y objetivo en ambos modos.
- **Alcance del Jacobiano inicial:** admitir explícitamente solo columnas
  independientes `soft_penalty` y `nnls_hard`, y rechazar matrices interiores
  de rango deficiente; no aproximar silenciosamente configuraciones de
  suavizado o cotas generales. Alternativa descartada: reutilizar la fórmula
  sin comprobar sus hipótesis. Evidencia: el solver existente tiene rutas
  distintas para estos otros casos. Falta resolverlas antes de integrar LM.
- **Valor por defecto:** conservar Powell. Alternativa descartada: cambiarlo
  sin comparación. Evidencia: todavía no existen los resultados exigidos.

## Pendiente

Integración completa en Espectroscopía y RMN (incluido IRLS sin modificar los
pesos), tratamiento y aviso de parámetros en límites y sus SE, información de
parada en los informes, pruebas de todos los modos admitidos, comparación del
banco y `tests/phase1_cases.py` con cinco repeticiones y evaluación del
criterio de cambio de valor por defecto. La prueba xfail de grupos de
absortividad no ha pasado a XPASS en la suite anterior.
