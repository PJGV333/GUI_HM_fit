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
| Grados de libertad | Truncamientos a 1 en `kinetics_tab.py` y `hmfit_core/utils/errors.py` | Eliminados los dos de la pestaña cinética; auditoría y corrección del núcleo y otras rutas **pendientes** |
| Solver interior espectral | Diferencia 300/3000 iteraciones documentada en Fase 3 | Diagnóstico independiente y KKT **pendientes** |
| Sensibilidades implícitas y cinéticas | Véase `DERIVATIVES_DESIGN.md` | Implementación y validación **pendientes** |

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

El selector normal solo ofrece SciPy. Los archivos de configuración que solicitan JAX u otro backend no reconocido se rechazan antes de modificar el mecanismo o parámetros, indicando que la migración a SciPy debe ser explícita; los archivos sin clave de backend mantienen SciPy. La entrada programática del ajuste también rechaza cualquier backend distinto de SciPy antes de cargar el modelo. Se retiró la implementación JAX previamente inaccesible del ajuste; permanece en la historia Git. No se han eliminado todavía `solve_concentrations_jax`, la proyección JAX de las pruebas ni las importaciones opcionales en otras rutas: su aislamiento/eliminación y la auditoría de dependencias están pendientes. No se han repetido las comparaciones JAX, pues JAX no está instalado.

Tras retirar el código inaccesible del ajuste, las pruebas específicas de cinética dieron 27 aprobadas y 2 omitidas; la suite local completa dio 613 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 283 advertencias en 179,11 s. No es CI multiplataforma.

## Retirada parcial de `explicit_lm` de los selectores

Las pestañas RMN y espectroscopía ya no ofrecen `explicit_lm`. Al importar un proyecto que lo solicite, ambas rechazan la configuración antes de modificar el estado e indican Powell como alternativa que el usuario debe elegir explícitamente. Las API `run_nmr_fit` y `run_spectroscopy_fit` también rechazan la selección antes de ejecutar el procesamiento, tanto con diccionarios como con dataclasses. Los dos full-refit rechazan ese optimizador con `unsupported_optimizer`, sin reajuste ni alternativa automática, y se han eliminado sus ramas de ejecución LM; no se alteró el objetivo de los métodos conservados. Se conserva la prueba de persistencia de pesos RMN con Powell. Esto no constituye la retirada completa: los procesadores y las entradas directas `run_nmr`/`run_spectroscopy` todavía admiten `explicit_lm`, y las pruebas experimentales se mantienen. No se ha demostrado aún la equivalencia completa de resultados con los optimizadores conservados.

Tras añadir la validación en las API, la suite local dio 619 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail y 283 advertencias en 179,06 s. No se ha ejecutado CI multiplataforma.

## Varianza residual y grados de libertad en cinética

La pestaña cinética ya no trunca los grados de libertad a uno. El resultado de ajuste reutiliza la varianza calculada por `fit_global` mediante `linearized_covariance`; Errors utiliza los grados de libertad del resultado y, cuando son insuficientes, no presenta una varianza estimable. En el caso sintético de dos tiempos con parámetros lineales y no lineales, antes llegaba a Errors `sigma2=1.9721522630525293e-31` pese a tener grados de libertad negativos; ahora queda no disponible. Las pruebas cubren además casos con parámetros libres, fijos y mixtos, y comprueban SSE/dof positivo. Esto no certifica aún todos los cálculos de incertidumbre ni los demás truncamientos del programa.

Suite local: 624 aprobadas, 3 omitidas, 5 deseleccionadas, 16 xfail, 283 advertencias en 175,88 s. No es CI multiplataforma.

## Pendiente y límites

Faltan las etapas B–H, la verificación de las causas de los xfail clasificados provisionalmente, los análisis antes/después y las pruebas de CI en todas las plataformas. Aparte de la retirada parcial de JAX descrita arriba, no se han cambiado objetivos ni umbrales productivos. Los informes de fases anteriores son históricos: los resultados HC2/HC3 calculados mediante el leverage incorrecto quedan **invalidados** y no deben utilizarse como evidencia, aunque se conservan los datos. La campaña de incertidumbre rechaza resumir o reanudar filas históricas que contengan estos resultados, antes de agregar nuevas filas; los generadores que leen resúmenes históricos directamente siguen requiriendo cautela. No se ha implementado ni validado aún una alternativa HC2/HC3 ni se han repetido las campañas afectadas. El dictamen actual es **C — consolidación incompleta**; no se cumplen los criterios obligatorios para A.
