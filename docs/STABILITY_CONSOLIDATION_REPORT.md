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
| `explicit_lm` | `hmfit_core/utils/explicit_lm.py`; procesadores de RMN y espectroscopía, selectores y full-refit GUI; pruebas y scripts de comparación | Retirar de producción sin sustitución silenciosa; **pendiente** |
| SciPy/Powell | En producción | Conservar; regresión por completar |
| JAX | Cinética: `hmfit/kinetics/model/kinetics_model.py`, `hmfit_gui_qt/tabs/kinetics_tab.py` y selector; importaciones opcionales en `hmfit_core/utils/errors.py`; job CI separado | Retirar de producción; **pendiente**. No es dependencia declarada en requirements inspeccionados |
| IRLS RMN | Histórico; selección de pesos y diagnósticos parciales en Fase 3 | Conservar para configuraciones antiguas; coherencia con Errors y último ciclo **pendiente de auditoría** |
| HC2/HC3 experimental | `validation/uncertainty_methods.py` usaba suma de cuadrados de concentraciones como leverage | Cálculo inválido desactivado en la campaña: HC2/HC3 se declaran no aplicables para OLS, NNLS y penalización; la corrección fundamentada y repetición de campañas **pendientes** |
| Grados de libertad | Truncamientos a 1 en `kinetics_tab.py:779,1624` y `hmfit_core/utils/errors.py:198` | Revisar contexto y sustituir solo con pruebas; **pendiente** |
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

## Pendiente y límites

Faltan las etapas B–H, la verificación de las causas de los xfail clasificados provisionalmente, los análisis antes/después y las pruebas de CI en todas las plataformas. No se han cambiado los métodos, umbrales, ni resultados productivos. Los informes de fases anteriores son históricos: los resultados HC2/HC3 calculados mediante el leverage incorrecto quedan **invalidados** y no deben utilizarse como evidencia, aunque se conservan los datos. La campaña de incertidumbre rechaza resumir o reanudar filas históricas que contengan estos resultados, antes de agregar nuevas filas; los generadores que leen resúmenes históricos directamente siguen requiriendo cautela. No se ha implementado ni validado aún una alternativa HC2/HC3 ni se han repetido las campañas afectadas. El dictamen actual es **C — consolidación incompleta**; no se cumplen los criterios obligatorios para A.
