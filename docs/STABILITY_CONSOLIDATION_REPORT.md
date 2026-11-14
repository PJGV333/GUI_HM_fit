# Consolidación de estabilidad: checkpoint de auditoría

**Estado: incompleta.** No se ha iniciado la Fase 4. Este informe no certifica las etapas pendientes.

## Línea base

- Rama de partida `origin/fix/stable-phase3-optimizers`, tras `git fetch`: `0ed96a3911ac14fbe07a7e0ea0115c06aa93072b`. Árbol limpio. Rama de trabajo `fix/stability-consolidation-phase1-3`.
- Linux, `.venv/bin/python` 3.14.9, NumPy 2.4.2, SciPy 1.17.1, pytest 9.0.2; `pip check`: sin incompatibilidades declaradas. JAX y jaxlib no están instalados; sus rutas no se han validado aquí.
- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`: 606 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail, 283 advertencias, 173.04 s. Las advertencias incluyen opciones desconocidas del optimizador y desbordamientos cinéticos. Es una prueba local, no CI multiplataforma.
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m validation.literature_benchmark run --json /tmp/hmfit-consolidation-literature.json`: terminó con código 0. El código de salida no significa que todos los casos reproduzcan la referencia: por ejemplo `r09a-r1-5sp`, R2: 0.7670 frente a 0.7825, diferencia -0.0155, tolerancia 0.0067. Los criterios y referencias no se modificaron. La salida completa es reproducible con ese comando; `/tmp` no forma parte del checkpoint.
- La suite incluye las pruebas sintéticas y de literatura seleccionadas por defecto. No se han repetido aún todas las campañas Monte Carlo ni las comparaciones por optimizador; las tablas históricas están en los informes anteriores y `validation/results/`.

## Inventario inicial y decisiones

| Ruta | Situación de partida | Decisión solicitada / estado |
|---|---|---|
| `explicit_lm` | `hmfit_core/utils/explicit_lm.py`; procesadores de RMN y espectroscopía y full-refit GUI; pruebas y scripts de comparación | Retirado de selectores y full-refit GUI con rechazo explícito; los procesadores aún lo ejecutan: retirada de producción **pendiente** |
| SciPy/Powell | En producción | Conservar; regresión por completar |
| JAX | Cinética: `hmfit/kinetics/model/kinetics_model.py`, `hmfit_gui_qt/tabs/kinetics_tab.py` y selector; importaciones opcionales en `hmfit_core/utils/errors.py`; job CI separado | Selector retirado y ajuste de cinética rechaza backend JAX; código JAX y dependencias indirectas **pendientes de retirar o aislar**. No está instalado aquí |
| IRLS RMN | Histórico; selección de pesos y diagnósticos parciales en Fase 3 | Conservar para configuraciones antiguas; coherencia con Errors y último ciclo **pendiente de auditoría** |
| HC2/HC3 experimental | `validation/uncertainty_methods.py` usaba suma de cuadrados de concentraciones como leverage | Cálculo inválido desactivado en la campaña: HC2/HC3 se declaran no aplicables para OLS, NNLS y penalización; la corrección fundamentada y repetición de campañas **pendientes** |
| Grados de libertad | Truncamientos a 1 en `kinetics_tab.py` y `hmfit_core/utils/errors.py` | Eliminados en esas rutas; auditoría de todos los usos y convenciones **pendiente** |
| Solver interior espectral | Diferencia 300/3000 iteraciones documentada en Fase 3 | Diagnóstico independiente y KKT **pendientes** |
| Sensibilidades implícitas y cinéticas | Véase `DERIVATIVES_DESIGN.md` | Sensibilidades experimentales 1:1 y cinética irreversible de primer orden probadas; otros modelos **pendientes**, sin integración productiva |

`hmfit_confignmrT2.json` configura Powell; la auditoría de migración de todos los formatos antiguos no está completada. La búsqueda de rutas usa `rg 'explicit_lm|\\bjax\\b|jaxlib'` sobre código, GUI, pruebas, configuración y CI; los resultados históricos no deben borrarse.

## Repetición local de la línea base y auditoría de xfail

En `0a4c33d`, `QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -rxfEs` terminó con 606 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 283 advertencias en 175,04 s. Las omitidas son el fichero experimental RMN ausente y dos pruebas JAX sin dependencia instalada. No es CI multiplataforma. El banco de literatura se ejecutó de nuevo con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m validation.literature_benchmark run --json "$TMPDIR/hmfit-consolidation-literature-current.json"`: código 0, dos casos reproducidos, cuatro parciales, tres no reproducidos y uno no soportado; código 0 no equivale a superar todas las referencias. El JSON temporal no es un artefacto versionado.

Clasificación individual provisional de los 16 xfail, basada en sus motivos y en `docs/LITERATURE_BENCHMARK.md`; no se retiró ninguno ni se afirma haber resuelto sus causas:

- Defecto pendiente de GUI: `test_custom_model_dimensions_start_without_forced_roles` y `test_gui_potentiometry_validation_warns_for_operational_ph` (`tests/test_acid_base.py`).
- Discrepancia de referencia/redondeo, limitación científica conocida: `test_r09a_ratio_at_fixed_cation_is_15_4` y los tres parámetros `r09b-spec-r`, `r09b-spec-b`, `r09b-spec-d-x` de `test_bc50_tables_reproduce_preregistered` (`tests/test_literature_bc50.py`).
- Diferencias ligadas al estimador IRLS, pendiente contrastar tras consolidar pesos: parámetros `r09a-p2-058`, `r09a-p2-106`, `r09a-r1-11-3n`, `r09a-r1-3sp`, `r09a-r1-5sp`, `r09b-r-qx` de `test_literature_reproduces_published_fit` (`tests/test_literature_benchmark.py`).
- Discrepancia no resuelta entre datos/modelo y referencia: `r09a-qx-dil` tanto en `test_literature_reproduces_published_fit` como en `test_literature_forward_model_at_published_constants` (`tests/test_literature_benchmark.py`); no se presupone cuál es su causa definitiva.
- Característica no soportada: `test_literature_reproduces_published_fit[r09b-d-qx]`, desplazamientos límite fijos.
- Defecto pendiente o limitación del modelo espectral: `test_absgrp_shared_groups_reduce_optical_components` (`tests/test_spectro_absgrp_equivalence.py`); causa raíz aún sin verificar.

## Salvaguarda HC2/HC3

La suma de cuadrados de concentraciones usada anteriormente depende de unidades y no es la diagonal del proyector estadístico. Se ha retirado ese cálculo de la campaña experimental, sin sustituirlo por un leverage lineal que ignore parámetros de molestia, sensibilidad no lineal o restricciones. Las tres clases de estimador (OLS, NNLS y penalización suave) devuelven `not_applicable`, razón explícita y valores no disponibles. Un resumen conserva ese estado y una reanudación rechaza filas históricas afectadas antes de escribir. Las pruebas dirigidas cubren estas condiciones, pero no demuestran todavía una fórmula HC2/HC3 válida.

Tras este cambio, la suite local con el mismo entorno terminó con 608 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 284 advertencias en 179,38 s; después se agregó y ejecutó por separado una prueba adicional de reanudación (14 pruebas dirigidas aprobadas). La advertencia adicional observada provino del integrador BDF de SciPy. No se ejecutó CI multiplataforma.

El generador `validation.make_unc_report` identifica como invalidados los HC2/HC3 numéricos históricos, no los grafica y conserva intactos los resúmenes fuente. Una ejecución sobre copias temporales de 21 resúmenes produjo 14 filas HC invalidadas y 21 figuras; no regeneró ni sobrescribió los informes históricos del repositorio. La etiqueta `no aplicable` se reserva para los resultados nuevos así declarados. Otros consumidores directos de los archivos históricos todavía deben tratar esos valores como invalidados. La suite local posterior a esta salvaguarda terminó con 610 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 282 advertencias en 176,74 s.

## Retirada parcial de JAX del ajuste de cinética

El selector normal solo ofrece SciPy. Los archivos de configuración que solicitan JAX u otro backend no reconocido se rechazan antes de modificar el mecanismo o parámetros, indicando que la migración a SciPy debe ser explícita; los archivos sin clave de backend mantienen SciPy. La entrada programática del ajuste también rechaza cualquier backend distinto de SciPy antes de cargar el modelo. Se retiró la implementación JAX previamente inaccesible del ajuste; permanece en la historia Git. La proyección experimental de JAX se trasladó de la pestaña cinética a `validation/jax_kinetics.py`, conservando su prueba de comparación; el árbol sintáctico del cuerpo coincide con el original. También se retiraron de la pestaña las funciones de covarianza y conversión usadas exclusivamente por la antigua ruta JAX. Permanecen `solve_concentrations_jax` en el modelo y referencias opcionales en otras rutas: su aislamiento/eliminación y la auditoría de dependencias están pendientes. No se han repetido las comparaciones numéricas JAX, pues JAX no está instalado.

Tras retirar el código inaccesible del ajuste, las pruebas específicas de cinética dieron 27 aprobadas y 2 omitidas; la suite local completa dio 613 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 283 advertencias en 179,11 s. No es CI multiplataforma.

Tras aislar la proyección y retirar las funciones muertas de la pestaña, las pruebas dirigidas dieron 25 aprobadas y 2 omitidas (las comparaciones JAX). La suite completa dio 666 aprobadas, 6 fallidas, 3 omitidas, 5 deseleccionadas y 16 xfail; los fallos son los seis conocidos de cobertura RMN y baselines históricos, descritos más abajo.

## Retirada parcial de `explicit_lm` de los selectores

Las pestañas RMN y espectroscopía ya no ofrecen `explicit_lm`. Al importar un proyecto que lo solicite, ambas rechazan la configuración antes de modificar el estado e indican Powell como alternativa que el usuario debe elegir explícitamente. Las API `run_nmr_fit` y `run_spectroscopy_fit` también rechazan la selección antes de ejecutar el procesamiento, tanto con diccionarios como con dataclasses. Los dos full-refit rechazan ese optimizador con `unsupported_optimizer`, sin reajuste ni alternativa automática, y se han eliminado sus ramas de ejecución LM; no se alteró el objetivo de los métodos conservados. Se conserva la prueba de persistencia de pesos RMN con Powell. Las entradas directas `run_nmr` y `run_spectroscopy` también rechazan el optimizador antes de modificar el callback global o leer datos; se probaron diccionarios, dataclasses y variantes de mayúsculas/espacios. Esto no constituye la retirada completa: ambos procesadores todavía admiten `explicit_lm`. Las pruebas espectroscópicas experimentales preservan sus aserciones y ejecutan el procesador directamente; deben aislarse fuera del núcleo productivo al retirarse esa rama. No se ha demostrado aún la equivalencia completa de resultados con los optimizadores conservados.

Tras añadir la validación en las API, la suite local dio 619 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 283 advertencias en 179,06 s. No se ha ejecutado CI multiplataforma.

Tras proteger además `run_nmr`, las pruebas dirigidas de API y consistencia RMN dieron 17 aprobadas. La suite completa dio 662 aprobadas, 6 fallidas, 3 omitidas, 5 deseleccionadas y 16 xfail: los seis fallos son los ya documentados de la campaña de cobertura y los cinco baselines históricos de SE.

Tras proteger también `run_spectroscopy`, las pruebas dirigidas de API, experimento histórico, consistencia y regresión espectroscópica dieron 21 aprobadas. La suite completa dio 665 aprobadas, las mismas 6 fallidas, 3 omitidas, 5 deseleccionadas y 16 xfail.

## Varianza residual y grados de libertad en cinética

La pestaña cinética ya no trunca los grados de libertad a uno. El resultado de ajuste reutiliza la varianza calculada por `fit_global` mediante `linearized_covariance`; Errors utiliza los grados de libertad del resultado y, cuando son insuficientes, no presenta una varianza estimable. En el caso sintético de dos tiempos con parámetros lineales y no lineales, antes llegaba a Errors `sigma2=1.9721522630525293e-31` pese a tener grados de libertad negativos; ahora queda no disponible. Las pruebas cubren además casos con parámetros libres, fijos y mixtos, y comprueban SSE/dof positivo. Esto no certifica aún todos los cálculos de incertidumbre ni los demás truncamientos del programa.

Suite local: 624 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail, 283 advertencias en 175,88 s. No es CI multiplataforma.

## Identificabilidad con Jacobianos anchos

`linearized_covariance` usaba SVD reducida incluso con menos observaciones que parámetros libres, omitiendo vectores del espacio nulo derecho. Para `J=[[1,0,1],[0,1,1]]`, antes informaba `[False, False, False]` como máscara de no identificabilidad; ahora devuelve `[True, True, True]`. Se conserva la SVD reducida en matrices altas para evitar un coste cuadrático en observaciones y se usa la base derecha completa solo en matrices anchas. También se verificó un caso con un parámetro identificable y dos no identificables bajo cambios de escala. La función auxiliar `sigma2_reduced`, sin usos internos encontrados, devuelve NaN para grados de libertad no positivos en lugar de truncarlos a uno. La auditoría estadística global sigue pendiente.

Suite local posterior: 627 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail, 283 advertencias en 179,26 s. No es CI multiplataforma.

## Corrección de `param_errors_cs`

La búsqueda adicional detectó otro truncamiento `max(m-p,1)` en la utilidad `param_errors_cs`, sin llamadas internas encontradas. Con dos residuos `[1,-1]` y dos parámetros, antes devolvía varianza 2; ahora devuelve NaN. También utilizaba una pseudoinversa que atribuía precisión finita a parámetros confundidos: para un Jacobiano con dos columnas idénticas devolvía errores estándar `0.32732684`, y ahora ambos se marcan no disponibles. La utilidad comparte `linearized_covariance` y requiere residuos compatibles con diferenciación por paso complejo. Una referencia lineal independiente de rango completo verifica varianza, covarianza y errores estándar; no se afirma que el método sea válido para residuos no holomorfos ni para estimadores con restricciones.

Suite local posterior: 630 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 283 advertencias en 179,28 s. CI multiplataforma pendiente.

## Sensibilidad de equilibrio 1:1

Se añadió una implementación experimental de diferenciación implícita para asociación 1:1, sin cambiar Jacobianos productivos. Se ha contrastado con una solución escalar independiente, una derivada analítica y diferencias finitas en tres escalas, además de un caso del solver existente y un rechazo por mal condicionamiento. Alcance, tolerancias y medición exploratoria de rendimiento en `DERIVATIVES_DESIGN.md`. Quedan pendientes la ampliación a otros modelos, la integración en producción y una validación más amplia.

Suite local: 636 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 284 advertencias en 177,16 s. No es CI multiplataforma.

## Sensibilidad cinética experimental

Se añadió la integración ampliada de sensibilidades directas para A → B de primer orden. Coincide en los casos probados con referencias analíticas y diferencias centradas usando el integrador BDF del modelo cinético existente; incluye una condición inicial que depende del parámetro. Las limitaciones, resultados y medición exploratoria de tiempo figuran en `DERIVATIVES_DESIGN.md`. No se ha integrado en producción ni validado para otros mecanismos o ajustes globales.

Suite local: 641 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 283 advertencias en 181,52 s; CI multiplataforma pendiente.

## RMN: pesos realmente empleados en el último ciclo

En IRLS, el último ajuste usa `weights_used_for_fit`, pero tras ese ajuste se actualizaba `weights_per_signal` y se utilizaba para calcular errores y rotular los pesos finales, sin que el ajuste hubiera utilizado esos pesos nuevos. En el caso sintético reproducido, el ajuste usó `[253038.3577822667, 216151.3936238925]` y la versión anterior calculaba errores con `[253101.93610348302, 216101.5188580443]`. El cálculo analítico ahora usa los pesos efectivamente empleados y se identifica expresamente como condicional a ellos. La exportación `irls_weights_final` y la tabla muestran los pesos empleados; la antigua actualización posterior se conserva como `irls_weights_proposed_next`. Esto cambia deliberadamente los errores estándar y la semántica de ese campo, no las constantes ajustadas.

El banco de literatura se ejecutó sin alterar referencias ni tolerancias: los nueve casos soportados mantuvieron sus log β (diferencia máxima cero en esta ejecución); los errores estándar cambiaron, hasta un 24,72 % relativo en `r09a-r1-3sp`. Clasificaciones publicadas sin cambio. La suite completa ejecutada antes de añadir la nota textual al resultado terminó con 637 aprobadas, 5 fallidas, 3 omitidas, 5 deseleccionadas y 16 xfail: las cinco fallidas son `test_literature_baseline_regression` para `r09a-p2-058`, `r09a-p2-106`, `r09a-r1-3sp`, `r09a-r1-5sp` y `r09b-r-qx`, porque el test compara los SE corregidos con el baseline interno histórico. La prueba dirigida posterior a la nota aprobó; no se han actualizado los baselines ni cambiado criterios para ocultar esta diferencia. Es una regresión de la suite pendiente de resolver mediante una validación y política de baseline explícitas, no una etapa completada.

**Salvaguarda provisional:** la pestaña Errors de RMN calcula con pesos unitarios y el full-refit no repite IRLS. Hasta implementar y validar un tratamiento coherente, esas rutas rechazan el modo IRLS y cualquier modo de pesos desconocido; la interfaz explica la indisponibilidad y no inicia el cálculo automáticamente. Los pesos y el modo utilizados se propagan desde el resultado al contexto de Errors. El método analítico principal del ajuste es distinto y continúa disponible, condicionado a sus pesos finales. Los análisis con pesos unitarios permanecen habilitados. Esta salvaguarda no implementa incertidumbre IRLS ni demuestra su calibración.

Un ajuste real IRLS del caso `r09a-r1-5sp`, conectado al constructor de contexto y al widget Qt en modo offscreen, confirmó la propagación de pesos y el bloqueo de Errors; también se verificó el rechazo del full-refit directo y la continuidad de las pruebas con pesos unitarios. Las pruebas dirigidas dieron 133 aprobadas. La suite completa anterior a los últimos cambios limitados a pruebas dio 649 aprobadas, 6 fallidas, 3 omitidas, 5 deseleccionadas y 16 xfail: cinco fallos de baselines históricos de SE ya descritos y un fallo nuevo de `test_smoke_campaign_runs_every_technique_and_resumes`, que exige éxito para los métodos RMN de una campaña que ajusta con IRLS. La campaña ahora recibe una indisponibilidad explícita en lugar de incertidumbres con otro esquema de pesos; no se ha modificado su diseño ni su criterio de aceptación para hacerla pasar. Es necesario diseñar y validar una campaña coherente antes de recuperar ese requisito.

Las filas RMN históricas de `validation/coverage_campaign.py` producidas con ajuste IRLS y Errors de pesos unitarios no justifican conclusiones sobre la cobertura del estimador IRLS. El agregador y los informes principal y secundario rechazan ahora filas RMN con resultados numéricos, en lugar de presentarlos como evidencia. El ejecutor rechaza campañas nuevas o reanudadas que incluyan RMN antes de escribir el manifiesto o réplicas; se pueden seleccionar otras técnicas. Las celdas históricas no registradas sin técnica identificable se rechazan y las nuevas réplicas guardan explícitamente la técnica. Los archivos crudos existentes no se modifican. Esta medida conservadora también rechaza resultados RMN para los que el formato histórico no permite demostrar un tratamiento coherente; no valida ni repara esos resultados. Los informes ya generados y consumidores externos no quedan invalidados automáticamente y requieren auditoría individual.

Se verificaron diez pruebas de esta salvaguarda, incluidas las rutas de resumen, informes y reanudación sin escrituras, y un ajuste real UV-Vis con resumen y reanudación. La suite completa posterior obtuvo 659 aprobadas, 6 fallidas, 3 omitidas, 5 deseleccionadas y 16 xfail: permanecen los mismos cinco fallos de baselines y el smoke de todas las técnicas, ahora bloqueado antes de iniciar la campaña por la indisponibilidad RMN. No se cambiaron los criterios de aceptación de la campaña ni las referencias.

## Pendiente y límites

Faltan las etapas B–H, la verificación de las causas de los xfail clasificados provisionalmente, los análisis antes/después y las pruebas de CI en todas las plataformas. Aparte de la retirada parcial de JAX descrita arriba, no se han cambiado objetivos ni umbrales productivos. Los informes de fases anteriores son históricos: los resultados HC2/HC3 calculados mediante el leverage incorrecto quedan **invalidados** y no deben utilizarse como evidencia, aunque se conservan los datos. La campaña de incertidumbre rechaza resumir o reanudar filas históricas que contengan estos resultados, antes de agregar nuevas filas; los generadores que leen resúmenes históricos directamente siguen requiriendo cautela. No se ha implementado ni validado aún una alternativa HC2/HC3 ni se han repetido las campañas afectadas. El dictamen actual es **C — consolidación incompleta**; no se cumplen los criterios obligatorios para A.
