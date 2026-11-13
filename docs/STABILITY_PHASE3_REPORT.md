# HM fit — Fase 3: informe intermedio

## Estado de avance

- 3.0 aprobada por el autor. 3.1 en curso: núcleo LM y Jacobiano
  espectral independiente en `4647f58`; se añade un constructor de residuo
  perfilado y diferencias centradas para los modos generales. Aún no están
  integrados en los procesadores, y Powell sigue por defecto. Suite posterior:
  566 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas
  (591 recogidas; 586 seleccionadas), 190.94 s. Antes: 556 aprobadas,
  0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas (581 recogidas).
- 3.1.e: LM disponible como `explicit_lm` en Espectroscopía (API y selector),
  sin alterar el predeterminado Powell. Diagnósticos y notas de límite están en
  el resultado y exportación. Validación inicial sintética en
  `tests/test_phase31_spectro_lm.py`; comparación extensa pendiente. Suite:
  602 recogidas, 577 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5
  deseleccionadas, 185.11 s; antes 598 recogidas y 573 aprobadas.
- 3.1.g–h: comparación Powell/LM realizada con cinco repeticiones por caso;
  Powell permanece por defecto porque LM empeora la SSQ en varios casos.
  El xfail de grupos sigue fallando; no se ha retirado. Suite tras comparación:
  607 recogidas, 582 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5
  deseleccionadas, 184.53 s (mismas cifras antes). Falta revisar otras rutas
  de incertidumbre y completar las etapas 3.2–3.5.
- 3.1.f: LM optativo en RMN dentro del IRLS anterior, sin cambiar sus pesos.
  Probado sintéticamente con Powell, límite activo, parámetro fijo,
  no convergencia y multistart paralelo en `tests/test_phase31_nmr_lm.py`.
  La comparación de literatura y la decisión sobre defecto siguen pendientes.
  Suite posterior: 607 recogidas, 582 aprobadas, 0 fallidas, 17 xfail,
  3 omitidas y 5 deseleccionadas, 186.40 s; antes 602 recogidas y 577 aprobadas.
- 3.1.c–d: núcleo LM reforzado para cotas iguales, controles inválidos,
  paso que entra desde un límite y agotamiento de iteraciones; sigue pendiente
  propagar diagnósticos y SE a todas las rutas de análisis adicionales. Suite: 598 recogidas,
  573 aprobadas, 0 fallidas, 17 xfail, 3 omitidas, 5 deseleccionadas,
  191.94 s (antes: 591 recogidas, 566 aprobadas, las demás cifras iguales).
- 3.2, 3.3 y 3.5 pendientes. 3.4: auditoría del workflow completada;
  no requiere cambio de CI, pues la suite completa ya incluye el banco.
  Este documento es provisional; no constituye validación de la fase completa.

## Decisiones del agente

| N.º | Contexto | Alternativas | Elección | Justificación | Evidencia | Cómo revertirla |
|---|---|---|---|---|---|---|
| D-1 | Objetivo espectral | Añadir penalización a legacy; respetar el objetivo de Powell | Respetar cada modo | No cambiar estimador | `_spectral_profile_objective`; `test_general_profile_residual_matches_existing_objective` | Retirar el constructor de residuo, sin cambiar Powell |
| D-2 | Jacobiano para modos generales | Fórmula analítica supuesta; diferencias centradas del residuo real | Diferencias centradas con paso relativo | Evita aplicar fórmulas no verificadas a modos acoplados | `centered_residual_jacobian`; prueba sintética con grupos, pesos, cotas y suavidad | Retirar la función auxiliar; no está integrada |
| D-3 | Humo del banco en CI | Añadir subconjunto; conservar suite completa | Conservar suite completa | Ya corre en cada push y no necesita selección adicional | `.github/workflows/tests.yml`, `pytest.ini`, ejecución local del banco 57 aprobadas y 14 xfail en 50.84 s | No hubo cambio de workflow |
| D-4 | Cota inferior igual a superior en LM | Contabilizar el gradiente de un parámetro inmóvil; excluirlo | Excluirlo del gradiente proyectado | Ningún paso factible puede mover ese parámetro | `test_fixed_bound_is_excluded_from_projected_gradient` | Revertir el cambio en `_gradient` |
| D-5 | Derivadas del ajuste espectral | Perfil analítico en todas las rutas; perfil analítico solo con hipótesis verificadas y diferencias centradas en las demás | Analítico para columnas independientes; centradas para el resto y sistemas de rango deficiente | No usar una fórmula no validada con regularización acoplada | `tests/test_explicit_lm.py` y `tests/test_phase31_spectro_lm.py` | Seleccionar Powell o revertir la rama `explicit_lm` |
| D-6 | Jacobiano RMN en IRLS | Derivar la proyección con ridge; diferenciar el residuo real | Diferencias centradas | Reproduce la proyección, máscara y pesos por señal existentes sin suponer un Jacobiano de proyección inexacto | `test_nmr_lm_with_irls_matches_powell`, `test_nmr_lm_parallel_multistart` | Seleccionar Powell |
| D-7 | Optimización predeterminada | LM; Powell | Powell | LM empeora SSQ en casos de regresión y agota iteraciones en el sintético sin ruido | Tabla 3.1.g y `validation/results/phase31_comparison.json` | Elegir LM explícitamente en la configuración |
| D-8 | Pesos para comparar IRLS | Comparar solo SSQ ponderada; mostrar también SSQ sin ponderar | Mostrar ambas y los pesos usados | Los métodos pueden terminar con pesos distintos, por lo que las SSQ ponderadas no comparten escala | `weights_used_for_fit` y `weighted_ssq_fit` en los resultados | Ignorar campos adicionales, sin cambiar el ajuste |

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


Rama `fix/stable-phase3-optimizers`, base `0634b93`. Estado: etapa 3.0 implementada,
probada con datos sintéticos y pendiente de revisión del autor; no validada con datos publicados.

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
la pareja es Powell ; LM. Éxito e iteraciones también están en ese orden.
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

## Evidencia de JAX

Con JAX 0.12.0.1 en CPU y datos sintéticos A → B → C (k1 = 1 s⁻¹,
k2 = 0.2 s⁻¹), diferencia máxima de concentraciones JAX/SciPy: 5.1·10⁻⁷
para 401 puntos y 0.2474 para 5 puntos en [0, 8] s. La comparación usa
`tests/test_phase3_jax.py` (3 aprobadas con JAX aislado); en el entorno normal
sin JAX se omiten los dos casos JAX mediante `importorskip`.

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

## Cambios esperados en resultados

El commit `3caecff` mezcla la función nueva de valores iniciales y la corrección de unidades
Eyring; son cambios independientes, separados a continuación para su revisión.

- Potenciometría: presentación y exportación de resumen cambian; no cambia el ajuste ni los
  errores numéricos. El RMS se calcula en unidades de señal, deshaciendo la división por σ.
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

## Riesgos y cuestiones abiertas

- Las concentraciones no tenían campo de unidades; se añade un campo opcional vacío por
  defecto para conservar configuraciones existentes. Una unidad vacía o desconocida en un
  ajuste global solo produce aviso y no garantiza compatibilidad de unidades.
- Configuraciones antiguas que compartan una constante entre órdenes distintos se seguirán
  abriendo, pero no podrán ajustarse hasta corregir el mecanismo, tampoco mediante el núcleo.
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
