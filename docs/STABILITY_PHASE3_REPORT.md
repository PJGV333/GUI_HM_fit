# HM fit — Fase 3: informe de cierre

## Estado de avance

ESTADO: ETAPA 3.6 COMPLETA

- 3.0 aprobada por el autor. 3.1: núcleo LM y Jacobiano espectral independiente
  en `4647f58`, posteriormente integrados; Powell sigue por defecto. Suite
  posterior al constructor de residuo perfilado y diferencias centradas:
  566 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas
  (591 recogidas; 586 seleccionadas), 190.94 s. Antes: 556 aprobadas,
  0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas (581 recogidas).
- 3.1.e: LM disponible como `explicit_lm` en Espectroscopía (API y selector),
  sin alterar el predeterminado Powell. Diagnósticos y notas de límite están en
  el resultado y exportación. Validación inicial sintética en
  `tests/test_phase31_spectro_lm.py`; comparación extensa más abajo. Suite:
  602 recogidas, 577 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5
  deseleccionadas, 185.11 s; antes 598 recogidas y 573 aprobadas.
- 3.1.g–h: comparación Powell/LM realizada con cinco repeticiones por caso;
  Powell permanece por defecto porque LM empeora la SSQ en varios casos.
  El xfail de grupos sigue fallando; no se ha retirado. Suite tras comparación:
  607 recogidas, 582 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5
  deseleccionadas, 184.53 s (mismas cifras antes). Refits GUI de LM y SE en
  límites se verifican también en `tests/test_phase31_refit_lm.py`. Suite de
  este bloque: 625 recogidas, 600 aprobadas, 0 fallidas, 17 xfail, 3 omitidas
  y 5 deseleccionadas, 172.09 s; antes 621 recogidas y 596 aprobadas.
- 3.1.f: LM optativo en RMN dentro del IRLS anterior, sin cambiar sus pesos.
  Probado sintéticamente con Powell, límite activo, parámetro fijo,
  no convergencia y multistart paralelo en `tests/test_phase31_nmr_lm.py`.
  La comparación de literatura confirma que debe conservarse Powell.
  Suite posterior: 607 recogidas, 582 aprobadas, 0 fallidas, 17 xfail,
  3 omitidas y 5 deseleccionadas, 186.40 s; antes 602 recogidas y 577 aprobadas.
- 3.1.c–d: núcleo LM reforzado para cotas iguales, controles inválidos,
  paso que entra desde un límite y agotamiento de iteraciones; diagnósticos y
  SE en límites también propagados a Errors. Suite: 598 recogidas,
  573 aprobadas, 0 fallidas, 17 xfail, 3 omitidas, 5 deseleccionadas,
  191.94 s (antes: 591 recogidas, 566 aprobadas, las demás cifras iguales).
- 3.2: pesos unitarios y por señal disponibles en GUI, API y configuración;
  IRLS sigue por defecto. Pesos finales y ciclos se muestran y exportan;
  tabla de evidencia del banco en este informe. Suite: 610 recogidas,
  585 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas,
  186.40 s; antes 607 recogidas y 582 aprobadas. El full-refit histórico de
  Errors sigue con pesos unitarios; la máscara de §6.4(c) se corrigió en 3.6.
- 3.3: resuelto vectorizado de pH con Brent de referencia y respaldo;
  comparaciones extremas y full-refit sintético añadidas. Suite: 621 recogidas,
  596 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas,
  172.50 s; antes 610 recogidas y 585 aprobadas. Banco y una réplica de
  bootstrap full-refit por técnica perfilados; resultados más abajo.
- 3.4: auditoría del workflow completada; la suite completa incluye el banco.
- 3.5: informe consolidado, cuestiones abiertas y decisiones documentadas;
  cierre técnico sujeto a revisión del autor, no a una validación clínica.

## Etapa 3.6 — diagnóstico antes de modificar LM

Suite final local sin JAX: 606 aprobadas, 0 fallidas, 16 xfail, 3 omitidas y
5 deseleccionadas (176.25 s). Con JAX aislado: 608 aprobadas, 0 fallidas,
16 xfail, 1 omitida y 5 deseleccionadas (179.47 s). El CI del commit de la
máscara, con cinco jobs, terminó en verde:
<https://github.com/PJGV333/GUI_HM_fit/actions/runs/43481798557>.

Los ocho casos de la tabla 3.1.g donde la SSQ de datos de LM superó a la de
Powell se reprodujeron con `validation/phase36_diagnosis.py`, antes de cambiar
el algoritmo. El JSON `validation/results/phase36_diagnosis_before.json`
contiene todos los arranques, resultados, pesos y diagnósticos. En RMN se fijan
para ambos métodos los pesos usados por Powell en su último ciclo IRLS; por
tanto las SSQ de esta tabla son comparables entre sí, a diferencia de las
ponderadas con pesos distintos citadas en 3.1.g. Cada celda contiene SSQ
(gradiente escalado, razón); «initial» significa punto recibido de 3.1.g.

| Caso | Powell inicial | LM inicial | LM desde Powell | Powell desde LM | Error relativo Jacobiano |
|---|---|---|---|---|---:|
| r09a-p2-106 | 258.97512 (0.0074, initial) | 258.90912 (0.00015, initial) | 258.90911 (5.7e-07, ftol) | 258.90912 (6.4e-05, éxito) | 2.9e-08 |
| r09a-r1-5sp | 115.93282 (0.0056, initial) | 115.92046 (0.001, initial) | 115.91988 (3e-07, ftol) | 115.92015 (0.00045, éxito) | 2.2e-08 |
| r09b-r-qx | 115.93282 (0.0056, initial) | 115.92046 (0.001, initial) | 115.91988 (3e-07, ftol) | 115.92015 (0.00045, éxito) | 2.2e-08 |
| phase1 espectro sin ruido | 2.5496872e-29 (0.0018, initial) | 4.8857986e-17 (0.097, initial) | 2.5496872e-29 (0.0018, xtol) | 1.2856975e-13 (1, éxito) | 0.279 |
| phase1 RMN 1:1 | 23.999747 (0.00017, initial) | 23.999746 (4.7e-07, initial) | 23.999746 (9.1e-09, ftol) | 23.999746 (5.3e-05, éxito) | 4.94e-09 |
| golden soft-bound 1 | 1.2450464 (1.5e-06, initial) | 1.3373802 (0.00014, initial) | 1.2450464 (1.1e-07, ftol) | 1.2450464 (1.1e-06, éxito) | 0.000383 |
| golden soft-bound 2 | 1.2431419 (3.3e-05, initial) | 1.3373184 (0.00038, initial) | 1.2431419 (1e-07, ftol) | 1.2431419 (4e-06, éxito) | 9.81e-05 |
| golden soft-bound 3 | 1.247918 (5.7e-05, initial) | 1.3378538 (1.7e-05, initial) | 1.247918 (2.9e-07, ftol) | 1.247918 (7.6e-06, éxito) | 2.56e-06 |

Con pesos fijos, en los cuatro casos RMN el punto Powell no era estacionario:
LM mejora su SSQ desde ese arranque; la mayor SSQ sin ponderar previa no prueba
que LM minimizara peor el objetivo ponderado. En los tres golden Powell desde
LM encuentra la cuenca inferior y LM desde Powell permanece en ella; LM declaró
parada en la cuenca superior con gradiente no nulo. Esto evidencia dependencia
del arranque/parada, pero no demuestra por sí solo que el punto superior sea
un mínimo local exacto. En el caso sin ruido el gradiente escalado es inestable
por el residuo casi nulo y LM agotó iteraciones; Powell arrancado en LM incluso
empeoró la SSQ. La comparación de Jacobianos usa pasos relativos
ε^(1/3) y 1e-5: en el caso sin ruido discrepan 27.9 %, en golden entre
2.56e-6 y 3.83e-4; no se debe afirmar coherencia exacta en esos puntos. En
RMN la discrepancia relativa es ≤ 2.91e-8. Esto es sensibilidad a paso/solver
o cambios de conjunto activo, no prueba de cuál aproximación es la correcta.

| Caso | Clasificación sustentada por el arranque cruzado |
|---|---|
| r09a-p2-106, r09a-r1-5sp, r09b-r-qx | Powell reportó éxito lejos de estacionariedad con los pesos comunes; LM desde Powell reduce SSQ. No son evidencia de que LM haya perdido el objetivo ponderado común. |
| phase1 RMN 1:1 | Powell parte con gradiente mayor; LM desde Powell mejora ligeramente la SSQ común. La diferencia es pequeña. |
| phase1 espectro sin ruido | LM se estancó y agotó iteraciones; gradiente mal condicionado y Jacobianos sensibles al paso. No se acredita un mínimo distinto. |
| golden 1 y 2 | LM declaró xtol en la cuenca superior; después de la corrección declara no_progress. Powell desde LM llega a menor SSQ y LM desde Powell conserva esa cuenca. Hay dependencia de arranque, sin prueba de mínimos locales exactos. |
| golden 3 | LM declara ftol en la cuenca superior con gradiente no nulo; Powell desde LM alcanza la cuenca inferior. No se ha demostrado si el punto superior es mínimo local. |

### 3.6 — aislamiento de JAX

[Defecto comprobado] `hmfit_core/utils/np_backend.py` seleccionaba JAX por
mera disponibilidad y suministraba `xp` a `nr_conc.py`, `lm_conc.py`,
`nnls_utils.py` y `errors.py`. Al instalar JAX, `NewtonRaphson.concentraciones`
devolvía `jaxlib` en vez de NumPy, provocando las cinco fallas documentadas en
Evidencia de JAX. Ahora el motor compartido usa NumPy siempre; JAX se selecciona
exclusivamente en cinética. La variable histórica `HMFIT_USE_JAX` ya no cambia
el backend de equilibrio. La conversión de arrays en errores ya no importa JAX
implícitamente; conserva la conversión NumPy de arrays concretos.
`tests/test_phase36_backend_isolation.py` comprueba tipo y ausencia de importación
incluso con esa variable a 0 o 1. Antes, con JAX importable, el
código ejecutaba `jax_enable_x64=True`: el dtype observado fue **float64**, no
float32; por ello no procede cuantificar un efecto float32 en el banco. El
backend JAX de cinética habilita ahora explícitamente x64 al seleccionarse;
`test_jax_rk4_fine_and_coarse_grid` comprueba `float64`. JAX sigue visible y
experimental.

Con JAX 0.12.0.1 aislado, NumPy 2.4.2 y SciPy 1.17.1: suite completa
605 aprobadas, 0 fallidas, 17 xfail, 1 omitida y 5 deseleccionadas (179.10 s),
incluidas las cinco pruebas que antes fallaban. Sin JAX: 603 aprobadas,
0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas (178.36 s). Se añadió un
job Linux/Python 3.14 separado con JAX CPU; los jobs principales siguen sin JAX.
El primer CI con los cinco jobs, incluido JAX, terminó en verde:
<https://github.com/PJGV333/GUI_HM_fit/actions/runs/43480306927>.

### 3.6 — máscara RMN §6.4(c), corrección aprobada

[Defecto comprobado] La GUI filtraba observaciones por la concentración del
progenitor antes de aplicar la estequiometría. Se sustituyó ese prefiltrado
por el usado por el ajuste: finitud de observación y columna, seguido del
filtro real por señal dentro del proyector (`C·S_j` con estequiometría, o la
columna del progenitor sin ella, con umbral 1e-12). Se corrigieron tanto Errors
como el full-refit y la reproducción del banco. Los grados de libertad, SE y
correlaciones de casos afectados pueden cambiar; el ajuste principal no cambia.
El caso publicado `r09a-r1-11-me` provocó XPASS estricto en
`test_errors_tab_uses_every_observed_value`; se retiró el marcador xfail, no la
prueba, en el mismo commit. Las pruebas enfocadas dieron 37 aprobadas y 9 xfail;
la suite completa sin JAX dio 604 aprobadas, 16 xfail, 3 omitidas y 5
 deseleccionadas (178.32 s); con JAX aislado, 606 aprobadas, 16 xfail,
1 omitida y 5 deseleccionadas (180.59 s).

### 3.6 — parada LM y repetición del banco

LM reconoce `zero_residual` cuando ‖r‖ ≤ 10 ε_mach ‖Y_ponderada‖
(con la misma normalización que r). `no_progress` se informa cuando pasos
rechazados consecutivos llevan λ al máximo 1e30; no se declara éxito si el
residuo no es nulo y el gradiente proyectado no cumple gtol. En vez de
aceptar `xtol` por un paso diminuto causado por los rechazos, se continúa
hasta ese máximo. Las pruebas `test_roundoff_residual_stops_without_undefined_scaled_gradient`
y `test_repeated_rejections_at_maximum_damping_are_not_success` cubren ambos
casos. Las pruebas sintéticas y el diagnóstico previo están arriba.

Se repitió el banco 3.1.g con cinco ejecuciones y los mismos criterios; datos
completos, log β, razones, iteraciones y medianas en
`validation/results/phase36_comparison_after.json`. Se cita la tabla 3.1.g
para los valores anteriores, sin duplicarla: el caso sin ruido sigue agotando
200 iteraciones (SSQ 4.89e-17 frente a 2.55e-29 de Powell); golden 1 y 2 ahora
se marcan **no convergidos** `no_progress` (1.33738 frente a 1.24505, y
1.33732 frente a 1.24314); golden 3 termina `ftol` en 1.33785 frente a
1.24792. El criterio de SSQ no se cumple; Powell sigue por defecto. El resto
de las comparaciones IRLS se conserva solo como información: la SSQ sin
ponderar no es el objetivo de ninguno de los dos ajustes IRLS. La tabla
siguiente reproduce la comparación completa: parejas Powell; LM, vectores
log β en orden de especie, éxito 1/0, iteraciones del último ciclo IRLS y
mediana de tiempo en segundos. SSQ espectral es el objetivo perfilado; SSQ
RMN aquí es solo descriptiva sin ponderar, no el objetivo IRLS.

| Caso | log β P; LM | SSQ P; LM | éxito P; LM (razón LM) | iter. P; LM | s P; LM |
|---|---|---|---|---|---|
| r09a-p2-058 | 7.17371/3.59686/4.06043/0.1/6.23871/9.51271; 7.17387/3.59671/4.06044/0.1/6.23884/9.5128 | 6.407475e-05; 6.4060517e-05 | 1; 1 (ftol) | 1; 3 | 3.763; 0.888 |
| r09a-p2-106 | 7.1914/3.60734/4.09372/0.1/6.12458/9.53348; 7.18772/3.60725/4.09341/0.1/6.11906/9.52748 | 0.00017731567; 0.00017818503 | 1; 1 (ftol) | 1; 4 | 5.979; 1.766 |
| r09a-qx-dil | 4.05787; 4.05787 | 2.5597423e-05; 2.5597423e-05 | 1; 1 (ftol) | 1; 1 | 0.129; 0.070 |
| r09a-r1-11-3n | 2.56992; 2.56992 | 0.002613785; 0.002613785 | 1; 1 (ftol) | 1; 3 | 0.155; 0.124 |
| r09a-r1-11-me | 2.56998; 2.56998 | 6.5471253e-07; 6.5471253e-07 | 1; 1 (xtol) | 1; 1 | 0.122; 0.044 |
| r09a-r1-3sp | 6.17778/2.94243/4.112; 6.17806/2.94204/4.1119 | 0.010209861; 0.010207934 | 1; 1 (ftol) | 5; 5 | 4.483; 3.917 |
| r09a-r1-5sp | 6.1225/2.94633/0.766986/4.48633/4.09053; 6.12193/2.94736/0.765473/4.48486/4.09092 | 7.226111e-06; 7.2272347e-06 | 1; 1 (ftol) | 2; 5 | 4.841; 0.955 |
| r09b-b-qx | 4.76675/1.37332/4.0897; 4.76675/1.37332/4.0897 | 1.1867311e-05; 1.1867311e-05 | 1; 1 (ftol) | 1; 3 | 1.185; 0.306 |
| r09b-d-qx | no admitido | — | — | — | — |
| r09b-r-qx | 6.1225/2.94633/0.766986/4.48633/4.09053; 6.12193/2.94736/0.765473/4.48486/4.09092 | 7.226111e-06; 7.2272347e-06 | 1; 1 (ftol) | 2; 5 | 4.738; 0.968 |
| phase1 espectro ruido | 3.02788; 3.02788 | 8.3928447e-06; 8.3928447e-06 | 1; 1 (ftol) | 2; 5 | 0.091; 0.039 |
| phase1 espectro sin ruido | 3; 3 | 2.5496872e-29; 4.8857986e-17 | 1; 0 (maxiter) | 4; 200 | 0.152; 1.080 |
| phase1 espectro fijo | 3.99956/7; 3.99956/7 | 8.0159197e-06; 8.0159197e-06 | 1; 1 (ftol) | 2; 6 | 0.108; 0.074 |
| phase1 RMN 1:1 | 2.99868; 2.99868 | 0.00010294114; 0.00010294125 | 1; 1 (ftol) | 1; 3 | 0.052; 0.042 |
| phase1 RMN 1:2 fijo | 4.02852/7; 4.02854/7 | 0.00012983397; 0.00012983358 | 1; 1 (ftol) | 1; 4 | 0.094; 0.118 |
| golden 1 | 2.68871; 3.13206 | 1.2450464; 1.3373802 | 1; 0 (no_progress) | 2; 46 | 0.069; 0.210 |
| golden 2 | 2.67167; 3.13148 | 1.2431419; 1.3373184 | 1; 0 (no_progress) | 2; 51 | 0.088; 0.255 |
| golden 3 | 2.68968; 3.13297 | 1.247918; 1.3378538 | 1; 1 (ftol) | 2; 29 | 0.080; 0.276 |

### 3.6 — RMN con pesos unitarios (objetivo fijo)

`validation/phase36_nmr_unit.py` repite cada caso soportado cinco veces, con
pesos exactamente unitarios para ambos métodos; las SSQ de esta tabla sí son
el mismo objetivo. Parejas Powell; LM. Las medianas incluyen lectura, ajuste
y exportación, sin gráficos. Los gradientes se calcularon de forma centrada
con pesos comunes; éxito=1/0. Archivo completo y arranques cruzados:
`validation/results/phase36_nmr_unit.json`. Con el criterio L2 del banco,
Powell unitario reproduce la referencia en todos los casos soportados salvo
`r09a-qx-dil`; en cuatro de los que reproduce, LM supera Δ log β=1e-4:
`r09a-p2-058` (1.60e-4), `r09a-p2-106` (7.60e-4),
`r09a-r1-5sp` y `r09b-r-qx` (1.132e-3). Por tanto tampoco con objetivo
unitario se satisface el primer criterio para cambiar el defecto.

| Caso | log β Powell; LM | SSQ P; LM | gradiente P; LM | razón P; LM | éxito P/LM | iteraciones P; LM | segundos P; LM |
|---|---|---|---|---|---|---|---|
| r09a-p2-058 | 7.18871/3.57783/4.06325/0.1/6.25584/9.46555; 7.1888/3.57767/4.06317/0.1/6.25581/9.46565 | 6.2606638e-05; 6.2606556e-05 | 0.000591; 3.72e-07 | éxito; ftol | 1/1 | 6; 6 | 1.648; 0.379 |
| r09a-p2-106 | 7.14197/3.60924/4.07217/0.1/6.08095/9.59777; 7.14188/3.61/4.07242/0.1/6.08071/9.59753 | 0.00014628047; 0.00014627949 | 0.00193; 5.69e-07 | éxito; ftol | 1/1 | 8; 10 | 2.400; 0.632 |
| r09a-qx-dil | 4.05787; 4.05787 | 2.5597423e-05; 2.5597423e-05 | 7.24e-09; 1.01e-07 | éxito; ftol | 1/1 | 2; 5 | 0.060; 0.048 |
| r09a-r1-11-3n | 2.37814; 2.37814 | 0.0025037175; 0.0025037175 | 1.45e-08; 1.59e-06 | éxito; ftol | 1/1 | 2; 10 | 0.051; 0.052 |
| r09a-r1-11-me | 2.56998; 2.56998 | 6.5471253e-07; 6.5471253e-07 | 1.37e-06; 1.39e-08 | éxito; ftol | 1/1 | 2; 5 | 0.081; 0.033 |
| r09a-r1-3sp | 6.52962/2.14276/4.256; 6.52963/2.14279/4.25599 | 0.0062336351; 0.0062336351 | 1.84e-05; 1.04e-05 | éxito; ftol | 1/1 | 5; 48 | 0.925; 1.650 |
| r09a-r1-5sp | 6.12176/2.94633/0.781379/4.51136/4.09002; 6.12128/2.94607/0.782511/4.51249/4.08973 | 7.1894245e-06; 7.1889949e-06 | 0.00433; 1.52e-07 | éxito; ftol | 1/1 | 12; 6 | 3.211; 0.352 |
| r09b-b-qx | 4.76676/1.37332/4.0897; 4.76675/1.37332/4.0897 | 1.1867312e-05; 1.1867311e-05 | 0.000221; 1.8e-08 | éxito; ftol | 1/1 | 7; 5 | 0.984; 0.179 |
| r09b-d-qx | No admitido: desplazamientos límite fijos | — | — | — | — | — | — |
| r09b-r-qx | 6.12176/2.94633/0.781379/4.51136/4.09002; 6.12128/2.94607/0.782511/4.51249/4.08973 | 7.1894245e-06; 7.1889949e-06 | 0.00433; 1.52e-07 | éxito; ftol | 1/1 | 12; 6 | 3.206; 0.352 |
| phase1 RMN 1:1 | 2.99882; 2.99881 | 0.00010294029; 0.00010294028 | 0.000155; 2.46e-09 | éxito; ftol | 1/1 | 2; 5 | 0.035; 0.026 |
| phase1 RMN 1:2 fijo | 4.02986/7; 4.02986/7 | 0.00012981818; 0.00012981818 | 2.45e-05; 6.25e-07 | éxito; no_progress | 1/0 | 2; 40 | 0.063; 0.084 |

### 3.6 — suavidad y convergencia interior de los golden

En cada solución LM golden se evaluó r(θ±h) para pasos relativos 1e-8,
1e-7, …, 1e-2, y Jacobianos centrados con pasos relativos 1e-8, 1e-5 y 1e-2.
Los datos completos y norma de cada r están en
`validation/results/phase36_golden_smoothness.json`. Las diferencias relativas
entre Jacobianos con pasos 1e-8 y 1e-5 son 0.000928, 0.00104 y 0.00150;
entre 1e-2 y 1e-5 son 0.311, 0.309 y 0.313. La diferencia relativa entre
pendientes hacia delante y atrás crece hasta 1.13, 1.12 y 1.14 en h=1e-2;
en el caso 3 ya alcanza 0.00222 en h=1e-7. Hay sensibilidad al paso/cambios
locales, por lo que no se demuestra suavidad global ni Jacobiano exacto.

El gradiente proyectado interior (norma) a 300/600/3000 iteraciones fue
0.46467/0.23911/0.06524, 0.46481/0.23915/0.06525 y
0.46377/0.23884/0.06517, respectivamente. Las ε cambian en norma ≈294 al
pasar de 300 a 600 y ≈1098–1099 al pasar de 300 a 3000; no hay evidencia de
convergencia a 300. **Pregunta para el autor:** ¿se autoriza cambiar el
resolvedor interior o su criterio para hacerlo converger, sabiendo que cambia
el estimador `legacy`? No se ha cambiado ese resolvedor.

### 3.6 — influencia y razones de pesos IRLS

Razón máximo/mínimo de pesos finales positivos por señal, con Powell e IRLS:
`r09a-p2-058` 124.08; `r09a-p2-106` 130.90; `r09a-qx-dil` 1;
`r09a-r1-11-3n` 2565.79; `r09a-r1-11-me` 1; `r09a-r1-3sp` 1678.17;
`r09a-r1-5sp` 1.532; `r09b-b-qx` 1; `r09b-r-qx` 1.532.
`r09b-d-qx` no se admite. Datos y SE completos en
`validation/results/phase36_irls_influence.json`.

| Caso / señal omitida | log β IRLS | log β unitarios | razón pesos restantes |
|---|---|---|---:|
| 11-3n / NH-2 | 2.56997 | 2.56620 | 1635.49 |
| 11-3n / NH-1 | 2.56992 | 2.39373 | 2565.80 |
| 11-3n / Me | −1.94675 | −1.94654 | 2.44 |
| 3sp / NH-2 | 6.17633/2.94427/4.11240 | 6.72340/2.06535/4.36497 | 887.96 |
| 3sp / NH-1 | 6.17632/2.94391/4.11243 | 6.38239/2.04525/4.21092 | 1682.66 |
| 3sp / Me | 6.79498/2.06065/4.56073 | 6.78370/2.05169/4.56130 | 1.22 |

La señal Me domina fuertemente la ponderación en ambos casos; al omitirla la
razón entre las restantes cae a 2.44 y 1.22, y los SE de algunos parámetros
crecen mucho. Esto es compatible con que IRLS destaque una señal y se aparte
de la referencia unitariamente ponderada, pero no prueba por sí solo la
causa física ni una calibración válida de sus SE. No se cambia el defecto IRLS.

## Decisiones del agente

| N.º | Contexto | Alternativas | Elección | Justificación | Evidencia | Cómo revertirla |
|---|---|---|---|---|---|---|
| D-1 | Objetivo espectral | Añadir penalización a legacy; respetar el objetivo de Powell | Respetar cada modo | No cambiar estimador | `_spectral_profile_objective`; `test_general_profile_residual_matches_existing_objective` | Retirar el constructor de residuo, sin cambiar Powell |
| D-2 | Jacobiano para modos generales | Fórmula analítica supuesta; diferencias centradas del residuo real | Diferencias centradas con paso relativo | Evita aplicar fórmulas no verificadas a modos acoplados | `centered_residual_jacobian`; prueba sintética con grupos, pesos, cotas y suavidad | Seleccionar Powell o revertir la ruta LM |
| D-3 | Humo del banco en CI | Añadir subconjunto; conservar suite completa | Conservar suite completa | Ya corre en cada push y no necesita selección adicional | `.github/workflows/tests.yml`, `pytest.ini`, ejecución local del banco 57 aprobadas y 14 xfail en 50.84 s | No hubo cambio de workflow |
| D-4 | Cota inferior igual a superior en LM | Contabilizar el gradiente de un parámetro inmóvil; excluirlo | Excluirlo del gradiente proyectado | Ningún paso factible puede mover ese parámetro | `test_fixed_bound_is_excluded_from_projected_gradient` | Revertir el cambio en `_gradient` |
| D-5 | Derivadas del ajuste espectral | Perfil analítico en todas las rutas; perfil analítico solo con hipótesis verificadas y diferencias centradas en las demás | Analítico para columnas independientes; centradas para el resto y sistemas de rango deficiente | No usar una fórmula no validada con regularización acoplada | `tests/test_explicit_lm.py` y `tests/test_phase31_spectro_lm.py` | Seleccionar Powell o revertir la rama `explicit_lm` |
| D-6 | Jacobiano RMN en IRLS | Derivar la proyección con ridge; diferenciar el residuo real | Diferencias centradas | Reproduce la proyección, máscara y pesos por señal existentes sin suponer un Jacobiano de proyección inexacto | `test_nmr_lm_with_irls_matches_powell`, `test_nmr_lm_parallel_multistart` | Seleccionar Powell |
| D-7 | Optimización predeterminada | LM; Powell | Powell | LM empeora SSQ en casos de regresión y agota iteraciones en el sintético sin ruido | Tabla 3.1.g y `validation/results/phase31_comparison.json` | Elegir LM explícitamente en la configuración |
| D-8 | Pesos para comparar IRLS | Comparar solo SSQ ponderada; mostrar también SSQ sin ponderar | Mostrar ambas y los pesos usados | Los métodos pueden terminar con pesos distintos, por lo que las SSQ ponderadas no comparten escala | `weights_used_for_fit` y `weighted_ssq_fit` en los resultados | Ignorar campos adicionales, sin cambiar el ajuste |
| D-9 | Configuraciones RMN sin pesos | Unitarios; IRLS previo | IRLS | Compatibilidad y decisión expresa del autor | `test_nmr_weight_selection_roundtrips_and_legacy_defaults_to_irls` y pruebas de regresión | Seleccionar `unit` expresamente |
| D-10 | Resolver pH sin intervalo válido o modelo no monótono | Forzar Newton; usar Brent y respaldo previos | Mantener referencia escalar | Preservar comportamiento en casos sin garantía de monotonía o sin bracket | `test_unbracketed_points_use_reference_fallback` y pruebas extremas | Sustituir llamada vectorizada por `_simulate_pH_titration_reference` |
| D-11 | Otros cuellos observados | Optimizar speciation/render/ODE; no alterar rutas sin prueba de equivalencia | No cambiarlos | Son compartidos y pueden alterar resultados o interfaz; no se ha demostrado una optimización equivalente | `validation/results/phase33_profiles.json` | No hay cambio que revertir |
| D-12 | Bootstrap de GUI con LM | Dejar fallback silencioso a Nelder–Mead; ejecutar LM sobre el residuo que ya usa el refit | Ejecutar LM y reportar fallback cuando proceda | Respeta la elección de optimizador sin modificar el estimador de errores predeterminado | `test_lm_full_refit_does_not_silently_switch_optimizer` | Seleccionar Powell o revertir las ramas de dispatch LM |
| D-13 | SE en límite en pestaña Errors | Mostrar SE sin restricciones; marcar indefinido | NaN, nota y código `bound` | Un intervalo simétrico sin restricciones no es válido en el límite | `test_error_tab_marks_bound_se_undefined` | No seleccionar LM o revertir el tratamiento de `active_bound_mask` |
| D-14 | Nivel numérico de residuo nulo | Umbral absoluto; múltiplo de ε relativo a datos | 10 ε ‖Y ponderada‖ | Mantiene independencia de unidades y evita 0/0 | `test_roundoff_residual_stops_without_undefined_scaled_gradient` | Ajustar el multiplicador en `explicit_lm` |
| D-15 | Rechazos sucesivos y paso mínimo | Marcar xtol como éxito; seguir hasta λ máximo y verificar gradiente | `no_progress` no exitoso salvo gradiente pequeño o residuo nulo | Evita éxito espurio observado en golden | Diagnóstico cruzado y `test_repeated_rejections_at_maximum_damping_are_not_success` | Revertir la rama de rechazos en `explicit_lm` |
| D-16 | Solver interior golden no convergido | Cambiar estimador; conservar y preguntar | Conservar 300 pasos | El autor prohíbe cambiarlo sin autorización | `phase36_golden_smoothness.json` | Sin cambio que revertir |

### Inventario del residuo y Jacobiano espectral

| Modo | Residuo | Jacobiano | Prueba |
|---|---|---|---|
| legacy, soft_penalty/nnls_hard independientes | Datos ponderados; ε resuelta por el solver existente | Perfil analítico; diferencias centradas para C | `test_profile_jacobian_matches_actual_residual` |
| absolute, soft_penalty/nnls_hard independientes | Datos y penalización exterior | Perfil analítico; diferencias centradas para C | `test_profile_jacobian_matches_actual_residual` |
| Grupos, pesos por punto, soft_bound, delta, suavidad o rango deficiente | Datos en legacy; datos y regularización en absolute | Diferencias centradas del residuo real | `test_general_profile_residual_matches_existing_objective` |

Los modos `legacy` y `absolute` difieren en el objetivo exterior: solo el
segundo añade penalización de ε y suavidad. Esa diferencia preexistente no se
modifica. Cada Jacobiano centrado requiere 2p+1 evaluaciones del residuo para
p parámetros, frente a una evaluación del problema interior y 2p+1 cálculos de
concentraciones en la ruta analítica. Los umbrales LM son ftol=xtol=1e-10,
gtol=1e-8, máximo 200 iteraciones y amortiguamiento inicial 1e-3; las normas
son relativas y el gradiente es proyectado.


Rama `fix/stable-phase3-optimizers`, base `0634b93`. La etapa 3.0 fue aprobada
por el autor; sus cambios se probaron con datos sintéticos y no se validaron con
referencias publicadas. Este informe integra y sustituye el informe provisional 3.1.

## Entorno y línea base

Python 3.14.9, NumPy 2.4.2, SciPy 1.17.1; Linux, Qt offscreen. Suite base:
543 recolectadas, 520 aprobadas, 0 fallidas, 17 xfail, 1 omitida y 5 deseleccionadas
(538 seleccionadas), 336.89 s. `pytest.ini` excluye `slow` por defecto: son los cinco
`tests/test_coverage_campaign_smoke.py::test_full_campaign_tier[1,2,3,4,9]`, correspondientes
a la campaña de cobertura aplazada. JAX 0.12.0.1 se instaló posteriormente solo en el
entorno local para las pruebas opcionales; no es una dependencia del proyecto. Con JAX
instalado, la suite dio 517 aprobadas, 5 fallidas, 17 xfail, 1 omitida y 5 deseleccionadas:
las cinco fallas eran asignaciones in-place a arreglos JAX inmutables en pruebas de RMN,
espectroscopía y caché. La causa observada es el cambio de comportamiento al importar:
la disponibilidad de JAX activa el backend JAX de esos módulos y produce arreglos inmutables;
no se identificó un cambio de versiones de NumPy/SciPy como causa. Se retira JAX del
entorno para mantener la línea base de CI; la
comparación opcional de cinética se ejecutó después con JAX aislado mediante `PYTHONPATH`,
sin instalarlo en el entorno normal.

Validación de constantes compartidas, sin JAX: 545 recolectadas, 522 aprobadas,
0 fallidas, 17 xfail, 1 omitida y 5 deseleccionadas; 443.85 s. Etapa 1.1,
sin JAX: 548 recolectadas, 523 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5
 deseleccionadas; 366.01 s. Validación de unidades: 550 recolectadas, 525 aprobadas,
0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas; 326.27 s. Etapa 1.2:
568 recolectadas, 543 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas;
329.81 s. Pruebas focalizadas con JAX aislado: 21 aprobadas. Etapa 1.3: 570
recolectadas, 545 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas;
335.45 s. Revisión previa a 3.1: suite normal 547 aprobadas, 3 omitidas, 17 xfail y
5 deseleccionadas (361.46 s); 9 pruebas del núcleo aprobadas también con importaciones Qt
bloqueadas.

## Evidencia y pruebas por etapa

Las cifras son de la suite principal, con `QT_QPA_PLATFORM=offscreen`,
`OPENBLAS_NUM_THREADS=1` y `OMP_NUM_THREADS=1`; `slow` queda deseleccionado
por `pytest.ini`. Recogidas incluye deseleccionadas.

| Etapa | Recogidas | Aprobadas | Fallidas | xfail | Omitidas | Deseleccionadas |
|---|---:|---:|---:|---:|---:|---:|
| Base 3.0, antes de 3.1 | 572 | 547 | 0 | 17 | 3 | 5 |
| Núcleo LM/Jacobiano inicial | 581 | 556 | 0 | 17 | 3 | 5 |
| Residuo general | 591 | 566 | 0 | 17 | 3 | 5 |
| LM y límites | 598 | 573 | 0 | 17 | 3 | 5 |
| Espectroscopía | 602 | 577 | 0 | 17 | 3 | 5 |
| RMN y comparación | 607 | 582 | 0 | 17 | 3 | 5 |
| Pesos RMN | 610 | 585 | 0 | 17 | 3 | 5 |
| Potenciometría vectorizada y perfiles | 621 | 596 | 0 | 17 | 3 | 5 |
| Refits y SE en límites | 625 | 600 | 0 | 17 | 3 | 5 |

## Problemas y estado

| Etiqueta | Problema | Estado y evidencia |
|---|---|---|
| [Defecto comprobado por inspección] | Una constante de velocidad compartida entre reacciones de distintos órdenes requiere unidades incompatibles | Rechazada en la validación de la GUI y en `fit_global` del núcleo con el mismo mensaje; `tests/test_kinetics_shared_rate_order.py` y `tests/test_phase3_core_rates.py`. Configuraciones guardadas se abren y muestran el error. Implementado y probado con ejemplos sintéticos; no validado con datos publicados. |
| [Limitación metodológica] | JAX integra un solo paso RK4 por intervalo sin control de error | Marcado experimental, aviso en informe y registro; SciPy sigue como referencia. Implementado y probado con datos sintéticos, no validado con datos publicados. |
| [Defecto comprobado] | JAX devolvía solo dos filas por intercambiar las salidas de `lax.scan` | Corregido con autorización del autor; `test_jax_rk4_fine_and_coarse_grid` fallaba por forma (2, 3) frente a (401, 3). |
| [Defecto comprobado] | JAX usaba el corte predeterminado de `pinv` | Corregido con `RANK_RCOND`; una matriz de rango casi deficiente daba coeficientes ~10¹² con el corte predeterminado en float64. Prueba con y sin pesos. |
| [Defecto comprobado] | Ajustes globales mezclaban unidades temporales sin convertir | Reproducido con datos sintéticos: k real 0.025 s⁻¹ (1.5 min⁻¹), ajuste global sin conversión k = 0.02550756, SSQ = 0.20123 con datos sin ruido. Ahora se rechazan unidades conocidas distintas y se avisa si faltan o no se reconocen; prueba `tests/test_phase3_units.py`. Implementado y probado con datos sintéticos, no validado con datos publicados. |
| [Limitación metodológica] | Valores iniciales de cinética y escala | Propuesta automática y Log10 por defecto para k y A de Arrhenius; se conservan las celdas y escalas del usuario. Implementado y probado con datos sintéticos, no validado con datos publicados. |
| [Defecto comprobado] | Eyring aplicaba kBT/h sin convertir unidades de tiempo o concentración | Corregido en Eyring para unidades reconocidas; pruebas de equivalencia y de factores de unidad en `tests/test_phase3_auto_rates.py`. |
| [Limitación metodológica] | Presentación de resultados de potenciometría | Informe y hoja `Fit_summary`: cada pKa con equilibrio y SE analítico o `fixed`, RMS en unidades de señal, N y g.l., autocorrelación y avisos; estadísticos adicionales separados. Probado con pH y mV sintéticos; no validado con datos publicados. |
| [Limitación metodológica] | LM puede converger a SSQ mayor o agotar iteraciones | Implementado y probado con datos sintéticos; evaluado en banco publicado pero no supera los criterios para defecto. Powell se conserva. Tabla 3.1.g. |
| [Limitación metodológica] | Pesos unitarios e IRLS producen estimaciones distintas | Selector implementado y probado con datos sintéticos; comparado con HypNMR publicado en tabla 3.2, sin cambiar el defecto IRLS. |
| [Limitación metodológica] | El solver escalar de pH limita el rendimiento | Vectorizado con respaldo; implementado y probado frente a Brent en casos sintéticos, sin validación con datos publicados. 8.96× en una réplica de bootstrap, no 10×. |
| [Defecto comprobado] | Máscara de Errors RMN y comportamiento inmutable al importar JAX | Ambos corregidos en 3.6: JAX con suite aislada; máscara con regresión publicada y xfail retirado. |
| [Sospecha] | Corte de rango JAX en precisión simple | Riesgo identificado, aún no validado en matrices mal condicionadas ni con datos publicados. |

## Comparación 3.1.g (cinco ejecuciones)

Comando: `QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m validation.phase31_comparison --repeats 5 --output
validation/results/phase31_comparison.json`. Misma máquina y proceso, tiempo de
pared mediano; generación de gráficos desactivada por igual para ambos, pero se
incluyen lectura, ajuste, errores y preparación de exportación. SSQ espectral
es la del objetivo perfilado; RMN muestra aquí SSQ de datos sin ponderar. El
JSON conserva además SSQ con los pesos usados en el último ciclo de cada método:
como IRLS puede producir pesos diferentes, comparar esas SSQ ponderadas entre
métodos no equivale a comparar un único objetivo con pesos fijos. Los valores
individuales de log β se muestran en el orden del modelo, separados por `/`;
la pareja es Powell ; LM. Éxito e iteraciones también están en ese orden;
en RMN corresponden al último ciclo IRLS, no a la suma de todos los ciclos.
Los únicos casos publicados que Powell reproduce en todas las constantes dentro
de la tolerancia L2 del banco son `r09a-r1-11-me` y `r09b-b-qx` (y Powell
reporta éxito); en ambos Δ máximo < 1e-4. El segundo criterio falla en los
casos golden y en el sintético sin ruido. **Se conserva Powell por defecto**.

| Caso | log β Powell ; LM | Δ máximo | SSQ Powell ; LM | Éxito P/LM | Iter. P/LM | segundos P/LM |
|---|---|---:|---:|---:|---:|---:|
| r09a-p2-058 | 7.17371 / 3.59686 / 4.06043 / 0.1 / 6.23871 / 9.51271 ; 7.17387 / 3.59671 / 4.06044 / 0.1 / 6.23884 / 9.5128 | 0.000161 | 6.40748e-05 ; 6.40605e-05 | 1/1 | 1/3 | 3.837/0.959 |
| r09a-p2-106 | 7.1914 / 3.60734 / 4.09372 / 0.1 / 6.12458 / 9.53348 ; 7.18772 / 3.60725 / 4.09341 / 0.1 / 6.11906 / 9.52748 | 0.006 | 0.000177316 ; 0.000178185 | 1/1 | 1/4 | 6.185/1.884 |
| r09a-qx-dil | 4.05787 ; 4.05787 | 4.99e-10 | 2.55974e-05 ; 2.55974e-05 | 1/1 | 1/1 | 0.127/0.070 |
| r09a-r1-11-3n | 2.56992 ; 2.56992 | 7.11e-08 | 0.00261379 ; 0.00261378 | 1/1 | 1/3 | 0.152/0.131 |
| r09a-r1-11-me | 2.56998 ; 2.56998 | 1.97e-09 | 6.54713e-07 ; 6.54713e-07 | 1/1 | 1/1 | 0.121/0.045 |
| r09a-r1-3sp | 6.17778 / 2.94243 / 4.112 ; 6.17806 / 2.94204 / 4.1119 | 0.000396 | 0.0102099 ; 0.0102079 | 1/1 | 5/5 | 4.444/4.175 |
| r09a-r1-5sp | 6.1225 / 2.94633 / 0.766986 / 4.48633 / 4.09053 ; 6.12193 / 2.94736 / 0.765473 / 4.48486 / 4.09092 | 0.00151 | 7.22611e-06 ; 7.22723e-06 | 1/1 | 2/5 | 4.789/1.040 |
| r09b-b-qx | 4.76675 / 1.37332 / 4.0897 ; 4.76675 / 1.37332 / 4.0897 | 1.9e-06 | 1.18673e-05 ; 1.18673e-05 | 1/1 | 1/3 | 1.194/0.327 |
| r09b-d-qx | No admitido: desplazamientos límite fijos | — | — | — | — | — |
| r09b-r-qx | 6.1225 / 2.94633 / 0.766986 / 4.48633 / 4.09053 ; 6.12193 / 2.94736 / 0.765473 / 4.48486 / 4.09092 | 0.00151 | 7.22611e-06 ; 7.22723e-06 | 1/1 | 2/5 | 4.838/1.036 |
| phase1 espectro ruido | 3.02788 ; 3.02788 | 4.02e-09 | 8.39284e-06 ; 8.39284e-06 | 1/1 | 2/5 | 0.092/0.040 |
| phase1 espectro sin ruido | 3 ; 3 | 5.2e-07 | 2.54969e-29 ; 4.8858e-17 | 1/0 | 4/200 | 0.156/1.093 |
| phase1 espectro fijo | 3.99956 / 7 ; 3.99956 / 7 | 6.73e-11 | 8.01592e-06 ; 8.01592e-06 | 1/1 | 2/6 | 0.111/0.078 |
| phase1 RMN 1:1 | 2.99868 ; 2.99868 | 7.41e-06 | 0.000102941 ; 0.000102941 | 1/1 | 1/3 | 0.053/0.046 |
| phase1 RMN 1:2 fijo | 4.02852 / 7 ; 4.02854 / 7 | 1.65e-05 | 0.000129834 ; 0.000129834 | 1/1 | 1/4 | 0.098/0.127 |
| golden soft-bound 1 | 2.68871 ; 3.13206 | 0.443 | 1.24505 ; 1.33738 | 1/1 | 2/22 | 0.071/0.215 |
| golden soft-bound 2 | 2.67167 ; 3.13148 | 0.46 | 1.24314 ; 1.33732 | 1/1 | 2/27 | 0.090/0.259 |
| golden soft-bound 3 | 2.68968 ; 3.13297 | 0.443 | 1.24792 ; 1.33785 | 1/1 | 2/29 | 0.083/0.287 |

La SSQ menor en cada caso se identifica por la comparación de las dos cifras;
el JSON contiene precisión completa cuando la tabla redondea a igualdad.
`test_absgrp_shared_groups_reduce_optical_components` sigue en XFAIL estricto:
con Powell, el RMS agrupado incumple el límite del 5 % respecto al no agrupado;
no ha pasado a XPASS y no se toca la prueba. La unificación de motores sugerida
por su razón histórica está fuera de alcance.

## Pesos RMN: evidencia 3.2

Se usó Powell y el mismo banco con las dos opciones; comando reproducible
`QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m validation.phase32_weights --output
validation/results/phase32_weights.json`. Las columnas contienen vectores en
orden de especies; Δ es el valor absoluto frente a HypNMR, y SE es el reportado
por el procesador. Los pesos finales completos se conservan en el JSON. Con
pesos unitarios se ejecuta un solo ciclo; IRLS sin clave conserva el comportamiento
anterior, incluida la actualización de pesos al final del último ciclo.

| Caso y pesos | log β | SE | Δ frente a referencia | Ciclos |
|---|---|---|---|---:|
| r09a-p2-058 IRLS | 7.1737/3.5969/4.0604/0.1/6.2387/9.5127 | 0.015917/0.010006/0.010409/0/0.025444/0.026936 | 0.01509/0.019161/0.0027695/0/0.017089/0.047109 | 3 |
| r09a-p2-058 unit | 7.1887/3.5778/4.0633/0.1/6.2558/9.4655 | 0.020505/0.013652/0.014407/0/0.035265/0.037652 | 8.64e-05/0.000127/5.41e-05/0/3.99e-05/5.21e-05 | 1 |
| r09a-p2-106 IRLS | 7.1914/3.6073/4.0937/0.1/6.1246/9.5335 | 0.019259/0.015899/0.0081936/0/0.031487/0.031713 | 0.049503/0.0026589/0.021318/0/0.043884/0.064016 | 4 |
| r09a-p2-106 unit | 7.142/3.6092/4.0722/0.1/6.0809/9.5978 | 0.024431/0.021569/0.017307/0/0.049122/0.027452 | 6.9e-05/0.0007574/0.00023049/0/0.00024838/0.00027081 | 1 |
| r09a-qx-dil IRLS | 4.0579 | 0.013346 | 0.0070727 | 2 |
| r09a-qx-dil unit | 4.0579 | 0.013346 | 0.0070727 | 1 |
| r09a-r1-11-3n IRLS | 2.5699 | 0.0030759 | 0.19182 | 4 |
| r09a-r1-11-3n unit | 2.3781 | 0.11788 | 4.14e-05 | 1 |
| r09a-r1-11-me IRLS | 2.57 | 0.0031284 | 2.33e-05 | 2 |
| r09a-r1-11-me unit | 2.57 | 0.0031284 | 2.33e-05 | 1 |
| r09a-r1-3sp IRLS | 6.1778/2.9424/4.112 | 0.012481/0.010709/0.005647 | 0.35182/0.79953/0.144 | 5 |
| r09a-r1-3sp unit | 6.5296/2.1428/4.256 | 0.18388/0.11798/0.18692 | 2.08e-05/0.00013908/3.82e-07 | 1 |
| r09a-r1-5sp IRLS | 6.1225/2.9463/0.76699/4.4863/4.0905 | 0.010509/0.0087755/0.024987/0.044217/0.0051932 | 0.0011965/0.00023231/0.015514/0.026173/0.00083274 | 3 |
| r09a-r1-5sp unit | 6.1218/2.9463/0.78138/4.5114/4.09 | 0.010116/0.0085307/0.026704/0.045776/0.0050635 | 0.00045652/0.00022964/0.001121/0.0011361/0.00031606 | 1 |
| r09b-b-qx IRLS | 4.7668/1.3733/4.0897 | 0.020847/0.0070935/0 | 4.85e-05/1.83e-05/0 | 2 |
| r09b-b-qx unit | 4.7668/1.3733/4.0897 | 0.020847/0.0070936/0 | 3.56e-05/1.84e-05/0 | 1 |
| r09b-d-qx | No admitido: desplazamientos límite fijos | — | — | — |
| r09b-r-qx IRLS | 6.1225/2.9463/0.76699/4.4863/4.0905 | 0.010509/0.0087755/0.024987/0.044217/0.0051932 | 0.0011965/0.00023231/0.015514/0.026173/0.00083274 | 3 |
| r09b-r-qx unit | 6.1218/2.9463/0.78138/4.5114/4.09 | 0.010116/0.0085307/0.026704/0.045776/0.0050635 | 0.00045652/0.00022964/0.001121/0.0011361/0.00031606 | 1 |

La pregunta original sobre §6.4(c) fue contestada por el autor en 3.6 y la
máscara se corrigió; véase la sección «3.6 — máscara RMN §6.4(c)».

## Potenciometría vectorizada 3.3

El balance monótono de los componentes ordinarios se resuelve en paralelo sobre
pH (= −log[H]), con Newton protegido por el intervalo y bisección. Se conserva
`_simulate_pH_titration_reference`; los extremos con balance < 1e-14 preservan
la elección previa, y los puntos sin intervalo o modelos no monótonos recurren
al solver previo. `tests/test_phase33_potentiometry_vectorized.py` compara con
Brent a 1e-10 absoluto: soluciones desde pH < 2 hasta > 12, tres pKa, cuatro
concentraciones (1e-12 a 1 M), cargas fuertes positivas/negativas y entorno de
los puntos de equivalencia; también verifica puntos sin bracket y parámetros de
un refit dentro de 1e-3·SE. La suite previa de potenciometría se conserva.

Una réplica full-refit de bootstrap sintético (semilla 20261113, 60 puntos,
dos pKa), cinco ejecuciones en la misma máquina, mediana: Brent 0.28321 s,
vectorizado 0.03161 s, **8.96×**, por debajo del objetivo 10×. θ fue
[6.99987442939, 4.00033862087] frente a [6.99987442927, 4.00033862340];
SE de referencia [0.00047441, 0.00052385]. El cambio máximo es 4.82e-6·SE.
Las cinco mediciones y semillas están en
`validation/results/phase33_potentiometry_benchmark.json`; reproducir con
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m
validation.phase33_potentiometry_benchmark --output RUTA`. La mejora medida
no garantiza 10× para otros experimentos ni es una campaña de cobertura.

### Perfiles cProfile 3.3

`validation/phase33_profiles.py` perfila una ejecución del banco completo y
una réplica wild bootstrap full-refit por técnica, semilla 20261113. Los datos
son sintéticos para bootstrap; se llama al ajuste completo y se comprueba que
la réplica haya tenido éxito. En Espectroscopía/RMN se incluyen lectura y
escritura XLSX; la medición es perfilada y no equivale a las medianas sin
perfilador de la tabla anterior. Detalle por función, llamadas y muestras en
`validation/results/phase33_profiles.json`. Los tiempos por función son
acumulados y se solapan: no deben sumarse.

| Carga | Antes (s perfilados) | Después (s perfilados) | Punto caliente observado | Cambio |
|---|---:|---:|---|---|
| Banco de literatura (10 casos) | 46.115 | 46.245 | Newton-Raphson concentraciones 39.40 s | Sin cambio del solver |
| Bootstrap potenciometría | 0.502 | 0.040 | balance escalar 0.471 s → vectorial 0.0184 s | Newton vectorizado; misma semilla |
| Bootstrap ácido-base RMN | 0.0052 | 0.0052 | predicción 0.0019 s | Sin cambio |
| Bootstrap ácido-base espectral | 0.0062 | 0.0062 | predicción 0.0030 s | Sin cambio |
| Bootstrap cinética | 0.2053 | 0.2023 | integración | Sin cambio |
| Bootstrap espectroscopia | 0.1424 | 0.1409 | concentraciones 0.0769 s | Sin cambio |
| Bootstrap RMN | 0.4603 | 0.4613 | concentraciones 0.181 s; render 0.164 s | Sin cambio |

Antes y después son ejecuciones emparejadas del mismo código actual, cambiando
únicamente el resolvedor de pH entre referencia escalar y vectorizado; las
rutas no potenciométricas no se modificaron y sus muestras/estimaciones fueron
idénticas a precisión completa. Los tiempos individuales perfilados reflejan
ruido de medición, no una optimización de esas rutas. Se deja
el solver Newton-Raphson compartido sin optimizar: aunque es un cuello medido,
no se ha demostrado una alternativa con la equivalencia numérica exigida.
Igualmente no se altera el render ni el integrador de cinética en esta etapa.

## Evidencia de JAX

Con JAX 0.12.0.1 en CPU y datos sintéticos A → B → C (k1 = 1 s⁻¹,
k2 = 0.2 s⁻¹), diferencia máxima de concentraciones JAX/SciPy: 5.1·10⁻⁷
para 401 puntos y 0.2474 para 5 puntos en [0, 8] s. La comparación usa
`tests/test_phase3_jax.py` (3 aprobadas con JAX aislado); en el entorno normal
sin JAX se omiten los dos casos JAX mediante `importorskip`.

Diagnóstico controlado adicional: `jax==jaxlib==0.12.0.1`, `ml_dtypes==0.6.0`
y `opt_einsum==3.4.0` instalados con `pip --no-deps --target` en un directorio
externo, añadido solo por `PYTHONPATH`; entorno principal sin JAX. Con y sin
ese directorio se usan exactamente NumPy 2.4.2 y SciPy 1.17.1. El mismo
`NewtonRaphson.concentraciones` devuelve `numpy.ndarray` sin JAX y
`jaxlib._jax.ArrayImpl` con JAX. En la suite aislada: 593 aprobadas, 5
fallidas, 17 xfail, 1 omitida, 5 deseleccionadas (298.98 s). Las cinco fallas
son las tres pruebas de coeficientes/orientación RMN, la de caché de
especiación y golden espectral seed 3: todas intentan asignar in-place a
arreglos JAX inmutables. Esto reproduce la causa por cambio de comportamiento
al importar, sin cambiar versiones NumPy/SciPy. El primer ensayo aislado dio
además siete fallos en la nueva fixture LM RMN por multiplicar un array JAX por
una lista; se corrigió la fixture convirtiendo explícitamente a NumPy y las
siete pruebas pasan. No se instala JAX en el entorno ni CI principal.

## Valores iniciales y Eyring (1.2)

Regla implementada: mayor duración entre conjuntos; para cada constante distinta, en orden de
primera aparición, `k0 = 1/(duración · c_típica^(orden−1) · 3^índice)`. La concentración
típica es la mayor positiva de los reactivos de todas sus reacciones; si no hay ninguna, la
mediana de todas las iniciales positivas. Se usan las unidades numéricas de los datos. Para
Arrhenius se propone Ea = 0 y A se obtiene evaluando la fórmula del modelo a T_ref; para
Eyring ΔH = 0 y ΔS se calcula invirtiendo la propia fórmula a T_ref, mediana de temperaturas.
La tabla e informe muestran unidades, o las marcan no especificadas. Las celdas editadas o
cargadas se conservan. El límite inferior cero se representa por −∞ en escala logarítmica.

En el caso A → B → C probado antes del cambio, el arranque igual 1/600 produjo k =
(0.0040568, 0.0493764), SSQ = 0.00088187: **no** quedó en la diagonal en ese caso;
se adopta la separación por 3 según la instrucción posterior del autor, no como efecto
universal demostrado. Las nueve plantillas (ocho existentes más Eyring) pasan el criterio
SSQ ≤ 1.05 × SSQ del ruido en el diseño sintético de `tests/test_phase3_templates.py`;
esto no demuestra convergencia global para datos arbitrarios. Reejecución de la misma
prueba, razón SSQ/SSQ_ruido por plantilla (semilla 73):

| Plantilla | Razón |
|---|---:|
| Primer orden | 0.970712 |
| Consecutiva | 0.931732 |
| Reversible | 0.971327 |
| Segundo orden | 0.956339 |
| Pseudo-primer orden | 0.970712 |
| Unión y reacción | 0.905562 |
| Michaelis-Menten | 0.911581 |
| Arrhenius | 0.943746 |
| Eyring | 0.943754 |

Comparación Arrhenius sintética, tres temperaturas 285/300/315 K, semilla 73, mismo
inicio automático: con A en Log10, éxito `gtol`, 11 evaluaciones, SSQ/ruido = 0.943746;
con A lineal, éxito `gtol`, 4 evaluaciones, SSQ/ruido = 0.962217. El optimizador expone
`nfev`, no iteraciones. Las evaluaciones terminan en puntos diferentes; esto es evidencia
sintética, no validación con datos publicados.

Eyring antes usaba la misma k numérica en segundos y minutos o M y mM, sin factor de unidad.
Para el mismo experimento eso desplazaría ΔS en +R ln 60 = 34.042 J mol⁻¹ K⁻¹ para
minutos, y −R ln 1000 = −57.434 J mol⁻¹ K⁻¹ para mM bimolecular. Ahora el factor de la
plantilla es segundos por unidad temporal × (M por unidad de concentración)^(orden−1);
trayectorias equivalentes en s/M y min/mM se prueban numéricamente. Texto vacío supone
s/M con aviso; texto no reconocido se rechaza. Datos publicados: no validados.

## Informe de potenciometría (1.3)

`tests/test_phase3_potentiometry_report.py` prueba, en datos sintéticos pH y mV con
σ = 0.1, pKa libre con ± y fijo con `fixed`, RMS no ponderado en unidades de señal,
N, g.l. y hoja `Fit_summary`. El aviso de primer volumen cero se conserva con texto
explícito: se refiere al titulante y es normal, no al volumen inicial de la celda. La
presentación resuelve el abierto 6 de la Fase 1. La modificación de una línea en
`tests/test_acid_base.py` actualiza la expectativa del aviso: el volumen cero es el primer
volumen de titulante, normal al comenzar, no el volumen inicial de la celda; no cambia el ajuste.
La diferencia de terminología `fixed`
frente a `const` en otros módulos queda para la Fase 4.

## Pendientes de 3.0 respondidos

1. **Estancamiento k1 = k2:** no se reprodujo en el caso consecutivo probado;
   el arranque igual convergió a (0.0040568, 0.0493764). El factor 3 se adoptó
   por la decisión explícita del autor, no como corrección universal demostrada.
2. **Cinco fallos JAX:** reproducidos en entorno aislado con versiones NumPy y
   SciPy idénticas, debidos a arrays JAX inmutables al importar el backend; véase
   el diagnóstico controlado en Evidencia de JAX.
3. **Estadísticas JAX:** SSQ, RMS, residuos del optimizador, Jacobiano y
   covarianza/errores proceden del objetivo JAX minimizado. Las curvas y el
   diagnóstico de autocorrelación se generan con SciPy; el recuento de
   parámetros lineales para grados de libertad también usa la matriz SciPy.

## Cambios esperados en resultados

El commit `3caecff` mezcla la función nueva de valores iniciales y la corrección de unidades
Eyring; son cambios independientes, separados a continuación para su revisión.

- Potenciometría: presentación y exportación de resumen cambian. El solver de pH
  vectorizado puede producir diferencias de redondeo; en el full-refit probado
  el desplazamiento fue < 1e-3·SE. El RMS se calcula en unidades de señal,
  deshaciendo la división por σ.
- LM es optativo y puede conducir a otros mínimos, en particular en las tres
  regresiones soft-bound; Powell sigue por defecto. SE de parámetros en límites
  activos se marcan indefinidos al elegir LM.
- RMN con pesos unitarios optativos puede cambiar log β y SE; al faltar la
  clave de configuración se conserva IRLS.
- Arranques automáticos y Log10 pueden cambiar mínimos o convergencia respecto a los valores
  anteriores; parámetros y escala escritos por el usuario o cargados no se sustituyen.
- Eyring con unidades reconocidas distintas de s y M cambia constantes numéricas y valores
  ajustados de ΔS‡ por los desplazamientos indicados arriba, sin cambiar ΔH‡ por los factores
  constantes de unidad; con s/M la fórmula no cambia. Texto desconocido ya no permite ajustar.
- Ajustes globales con unidades temporales o de concentración reconocidas y distintas se
  rechazan en vez de entregar constantes inconsistentes. No se convierten los datos.
- JAX: la trayectoria tiene ahora todos los tiempos; antes los ajustes ordinarios fallaban por
  dimensiones. El corte de rango cambia los coeficientes y puede cambiar parámetros y SE en
  mecanismos con matrices casi deficientes.
- Un mecanismo con la misma constante en reacciones de distinto orden ya no se ajusta desde
  la GUI ni desde `fit_global`: antes se aceptaba pese a la incompatibilidad dimensional. Los casos válidos no
  cambian por esta validación.

## CI

El código de la etapa 3.0 en `e41a018` pasó el workflow `tests` en Linux y Windows,
Python 3.11 y 3.14: <https://github.com/PJGV333/GUI_HM_fit/actions/runs/43306960036>.
Windows ejecuta el subconjunto no GUI; Linux ejecuta la suite normal y el banco de literatura.
El workflow se dispara en todos los pushes; `test_literature_benchmark.py`,
`test_literature_speciation.py`, `test_literature_bc50.py` y la prueba GUI se
incluyen en la suite normal. Los cuatro archivos tardaron 50.84 s localmente
(57 aprobadas, 14 xfail), dentro del presupuesto de cinco minutos. JAX no
figura en los requisitos instalados por el workflow. La ejecución de `4647f58`
fue verde: <https://github.com/PJGV333/GUI_HM_fit/actions/runs/43446398678>.
También las ejecuciones de `de66038` y `613fa18` terminaron en verde en los
cuatro trabajos: <https://github.com/PJGV333/GUI_HM_fit/actions/runs/43449185373>
y <https://github.com/PJGV333/GUI_HM_fit/actions/runs/43459085101>. El push de
`613fa18` canceló el CI de `e79343b` por concurrencia; se reejecutó y quedó
verde: <https://github.com/PJGV333/GUI_HM_fit/actions/runs/43458138060>.
El CI de `f0afeca` también terminó en verde:
<https://github.com/PJGV333/GUI_HM_fit/actions/runs/43463381745>.
El CI de `8e5342c` terminó en verde:
<https://github.com/PJGV333/GUI_HM_fit/actions/runs/43464692714>.
El CI del resolvedor vectorizado `2571a66` también terminó en verde:
<https://github.com/PJGV333/GUI_HM_fit/actions/runs/43466430821>.
El CI de los perfiles `c4fd734` terminó en verde:
<https://github.com/PJGV333/GUI_HM_fit/actions/runs/43468016440>.
El CI de los refits y avisos `b785345` también terminó en verde:
<https://github.com/PJGV333/GUI_HM_fit/actions/runs/43471180920>.

## Riesgos y cuestiones abiertas

- Las concentraciones no tenían campo de unidades; se añade un campo opcional vacío por
  defecto para conservar configuraciones existentes. Una unidad vacía o desconocida en un
  ajuste global solo produce aviso y no garantiza compatibilidad de unidades.
- Configuraciones antiguas que compartan una constante entre órdenes distintos se seguirán
  abriendo, pero no podrán ajustarse hasta corregir el mecanismo, tampoco mediante el núcleo.
- [Limitación metodológica] El full-refit de la pestaña Errors de RMN mantiene
  su objetivo histórico de pesos unitarios incluso si el ajuste principal usa
  IRLS. El dispatch LM respeta ese objetivo; la máscara sí se corrigió en 3.6.
  Sus SE no deben interpretarse como un full-refit IRLS idéntico al ajuste.
- [Sospecha] En JAX con precisión de 32 bits, un corte de rango de 10⁻¹⁰ está por debajo de
  la precisión de máquina; no se ha validado el comportamiento para matrices mal condicionadas.
- [Defecto comprobado por inspección] Tras optimizar con JAX, `_run_kinetics_fit` genera las
  curvas y residuos mostrados mediante `objective.predict_dataset`, que integra con SciPy;
  pueden no coincidir con el objetivo JAX en rejillas gruesas. No corregido ni cuantificado
  en esta etapa: SciPy sigue siendo la referencia y JAX queda marcado experimental. SSQ,
  RMS, residuos del optimizador, Jacobiano y covarianza/errores se calculan con el objetivo
  JAX minimizado; el número de parámetros lineales para los grados de libertad se determina
  con la matriz SciPy. Las curvas y el diagnóstico de autocorrelación usan residuos SciPy.
- `fixed` frente a `const` se deja para unificar en Fase 4; no se cambian métodos de error.

## Preguntas para el autor

1. ¿Se mantiene JAX como experimental o se oculta en la versión estable? No se
   ha ocultado ni retirado y no se incorpora al entorno principal.
2. La corrección de la máscara RMN del banco §6.4(c) fue aprobada por el autor
   y se ha implementado; ya no es pregunta pendiente.
3. ¿Conviene abordar en una fase posterior el full-refit RMN de la pestaña
   Errors, actualmente de pesos unitarios incluso cuando el ajuste usa IRLS?
   No se ha cambiado el método de error por defecto en esta fase.
