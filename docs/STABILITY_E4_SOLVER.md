# Etapa E.4 — reproducción previa a la corrección

Captura realizada en `3ab56d8`, Python 3.14.10, NumPy 2.4.2, SciPy 1.17.1, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. `validation.stage_e4_failure_trace` instrumenta por traza de Python los dos ajustes 1:2 de E.3 sin cambiar el solver; el resultado y los sistemas capturados están en `validation/results/stage_e4_failure_trace.json`. Los números de línea instrumentados corresponden al solver de `3ab56d8`; ejecutar la traza contra otra revisión requiere ajustarlos. `tests/test_stage_e4_solver_reproduction.py` comprueba que ambos sistemas guardados tienen referencias factibles independientes con gradiente proyectado absoluto menor que `1e−9`.

| Sistema histórico | Resultado | Observación final |
|---|---|---|
| μ=α=0 | búsqueda lineal fallida, iteración 23 | distancia mínima a cota `5.08e−11`, umbral de activación fijo `1e−12`; norma de dirección `9097.84`; pasos aceptados previos `~5.68e−14`, con paso proyectado `5.18e−10` y gradiente proyectado máximo `5.96e−8` |
| μ=.01, α=.02 | agotamiento de 300 iteraciones | últimos pasos aceptados `1.455e−11`, norma del cambio `5.09e−14`; gradiente proyectado máximo `3.66e−7`; descenso direccional previsto a paso completo `3.71e−11` |

En el primer sistema, un coeficiente permanece muy cerca de la cota pero fuera del umbral fijo usado para decidir el conjunto activo; la dirección Newton es grande y la proyección la bloquea, con búsquedas de paso extremadamente pequeñas y pendiente proyectada que llega incluso a ser positiva por redondeo. La discrepancia entre el gradiente calculado mediante ecuaciones normales y el gradiente directo es solo `3.60e−19` en ese estado: no atribuir este caso a esa diferencia sin más evidencia.

En el segundo, el término `A L Aᵀ` se evalúa con coeficientes de hasta `11325.5`; la evaluación como suma de cuadrados de segundas diferencias difiere en `1.44e−9` del valor cuadrático, mayor que el descenso estimado a paso completo. Se observan pasos minúsculos reiterados, compatibles con pérdida de precisión en la comparación Armijo; la traza demuestra el estancamiento y la diferencia de evaluación, no prueba que dicha cancelación sea su única causa. El criterio existente `1e−10 max(1, ||b||∞)` es absoluto en estas escalas y por sí solo no demuestra optimalidad relativa.

Las referencias y la factibilidad están documentadas en el informe E.3. Esta captura no repara el solver, no valida la incertidumbre ni demuestra que β₂ sea identificable. No se ha cambiado `legacy`, el objetivo, las cotas ni los pesos.
