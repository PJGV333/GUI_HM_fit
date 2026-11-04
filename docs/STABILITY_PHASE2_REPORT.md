# HM fit — Fase 2: validación estadística de incertidumbres

> **Estado: PARCIALMENTE VALIDADA (dictamen provisional B).** La investigación del
> sesgo UV-Vis (Etapas A–D) está completada y documentada en
> `docs/UVVIS_BIAS_INVESTIGATION.md`: causa establecida, sin defecto de
> implementación en el núcleo. Permanecen abiertas la ampliación del arnés
> (Etapas E–G), la comparación de métodos de incertidumbre (Etapa F), las
> campañas intermedia/confirmatoria y la Etapa H de rendimiento.

## 1. Base y entorno

La rama `fix/stable-phase2-statistical-validation` parte de `f36d6c8`. El informe de Fase 1 está disponible en el commit `14a6f10`, pero no está presente en el árbol del commit base `f36d6c8`. La línea base se obtuvo con `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q --tb=line -p no:cacheprovider`, desde la raíz del repositorio. Se instaló `odfpy` en el entorno virtual antes de esta ejecución.

| Entorno | Versión |
|---|---:|
| Python | 3.14.8 |
| NumPy | 2.4.2 |
| SciPy | 1.17.1 |
| PySide6 | 6.10.2 |

## 2. Suite

| Ejecución | Recolectadas | Aprobadas | Fallidas | Omitidas | Tiempo |
|---|---:|---:|---:|---:|---:|
| Línea base | 204 | 199 | 4 | 1 | 41.27 s |
| Estado anterior, pasada 1 | 219 | 214 | 4 | 1 | 34.01 s |
| Estado anterior, pasada 2 | 219 | 214 | 4 | 1 | 34.55 s |
| Estado actual, pasada 1 | 225 | 220 | 4 | 1 | 74.62 s |
| Estado actual, pasada 2 | 225 | 220 | 4 | 1 | 72.71 s |
| Rama de finalización, baseline tras integración | 225 | 220 | 4 | 1 | 42.44 s |
| Rama de finalización, correcciones parciales | 232 | 227 | 4 | 1 | 56.31 s |
| Modo absoluto y nuevos tests, suite completa | 255 | 250 | 4 | 1 | 37.20 s |
| Arnés piloto provisional, suite completa | 260 | 255 | 4 | 1 | 42.79 s |
| Resolución separable y piloto, suite completa | 261 | 256 | 4 | 1 | 49.78 s |
| Investigación de sesgo A–D (nueve pruebas nuevas), suite completa | 270 | 265 | 4 | 1 | 39.38 s |

En las dos pasadas del estado de trabajo, los identificadores de los cuatro fallos coinciden exactamente con los de la línea base:

- `tests/test_acid_base.py::test_custom_model_dimensions_start_without_forced_roles`
- `tests/test_acid_base.py::test_gui_equation_editor_generates_canonical_model`
- `tests/test_acid_base.py::test_gui_potentiometry_validation_warns_for_operational_ph`
- `tests/test_spectro_absgrp_equivalence.py::test_absgrp_shared_groups_reduce_optical_components`

Ejecutar únicamente `pytest tests/` no reproduce esa línea base porque excluye pruebas descubiertas desde la raíz.

## 3. Pesos por punto: estado parcial

[Implementado y probado con casos sintéticos; **no validado por campaña**] La nueva opción de configuración `weights_matrix` acepta pesos estadísticos no negativos por punto y longitud de onda. Su raíz cuadrada multiplica el residuo; los pesos preexistentes por longitud de onda siguen siendo multiplicadores del residuo. Las rutas modificadas incluyen el estimador por columnas, el Jacobiano VarPro proyectado, el recuento de grados de libertad y la transferencia de pesos al contexto de incertidumbres y al refit. No se han establecido aún todas las garantías de compatibilidad.

[Defecto comprobado, corregido en el estado de trabajo] El descenso por gradiente ponderado con un número fijo de iteraciones no aseguraba la minimización en casos mal escalados: en un experimento determinista con concentraciones del orden de 10⁻³, el objetivo obtenido con 300 iteraciones fue `17.014161726420298`, frente a `13.254238295371566` de una minimización independiente. La ruta ponderada utiliza ahora un Newton semisuave proyectado con comprobación del gradiente proyectado y búsqueda de paso. Superó tres pruebas sintéticas frente a una referencia independiente y una exploración de 50 semillas con cotas, suavizado y pesos cero. Esto no equivale a validación de cobertura; el sistema denso tiene un límite explícito de 4000 coeficientes.

[Limitación metodológica] La covarianza analítica existente utiliza el estimador de absortividades sin restricciones; no debe interpretarse como una validación de la incertidumbre del estimador restringido. El Jacobiano proyectado de VarPro no es, en general, toda la derivada del residuo perfilado: la derivada completa también contiene un término en el espacio de las columnas lineales.

[Limitación de pruebas] En el entorno de esta ejecución no está instalado JAX; no se ha verificado la ruta con ese backend. Tampoco se ha completado una validación integral de todos los caminos de remuestreo y refit con pesos por punto.

## 4. Auditoría de la rama de finalización

Se integró `fix/uncertainty-reliability` en la rama de finalización mediante un merge ordinario; se conservan los historiales del checkpoint `fb9ea9e` y del merge de Fase 1 `14a6f10`.

[Defecto comprobado, corrección parcial] El refit espectroscópico aceptaba parámetros finitos incluso si el optimizador informaba falta de convergencia, y las excepciones del solver interno se convertían en `1e50`, considerado finito. Ahora las excepciones del solver químico y lineal se clasifican, la cancelación se propaga, se exige éxito del optimizador y se reevalúa el resultado verificando parámetros, cotas y residuo finitos. Se han probado fallos de convergencia, fallo lineal y cancelación, así como un refit sintético válido; faltan pruebas específicas para todos los modos de fallo y optimizadores. El contador full-refit ahora registra intentos y causas y rechaza parámetros no finitos. Las réplicas descartadas pueden concentrarse en regiones extremas: los intervalos condicionados al éxito pueden resultar sesgados o estrechos; aún no se ha cuantificado este efecto.

[Defecto comprobado, corregido en la API numérica] En `solve_A_nnls_pgd`, multiplicadores constantes por longitud de onda distintos de uno se ignoraban aunque `mu` es una penalización absoluta. Se corrigió en esa API y se contrastó un caso analítico. En el procesador se conserva deliberadamente la ruta histórica para pesos que no varían entre puntos, pues el cambio inmediato produjo tres regresiones en las pruebas doradas de `soft_bound`. Por tanto la coherencia general pesos/regularización del procesador **sigue pendiente**; no se afirma corregida en todos los modos. En `_exact_soft_penalty_columns` la penalización se reescala relativamente a los pesos: es una convención distinta de la penalización absoluta y requiere una opción explícita y validación.

[Defectos comprobados, corregidos] El bootstrap one-step de RMN usa ahora el Jacobiano del ajuste original para clasificar identificabilidad, no el de la última réplica. Para parámetros ácido-base todos fijos, la correlación se expresa como no definida (`NaN`), manteniendo SE cero. Ambos cambios tienen regresiones específicas.

[Límite de ejecución] La suite se ejecutó con 227 aprobadas antes de añadir dos pruebas adicionales de clasificación de fallos; las nueve pruebas focalizadas de `tests/test_phase2_completion_critical.py` pasaron después. Los dos intentos posteriores de la suite completa agotaron los límites de 180 s y 300 s sin producir salida. Una prueba acotada mostró que Python arrancó pero la importación de `pytest` no terminó dentro del límite; no se ha determinado la causa. Por ello no se considera verificada la suite completa del último estado. No se detuvieron procesos ajenos.

[Diagnóstico posterior de pytest] En el checkpoint `85b901a`, Python y la importación de `pytest` volvieron a funcionar: `pytest 9.0.2`, importación medida con `-X importtime` ≈81 ms; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest --version` y una prueba aislada también funcionaron. `pip check` no encontró dependencias rotas. En la medición de recuperación había 51 GiB de memoria disponible y `vmstat` no mostraba espera apreciable de I/O en la muestra; estos datos no demuestran la causa de los bloqueos anteriores. No se modificaron ni terminaron procesos ajenos. Con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 QT_QPA_PLATFORM=offscreen`, la suite completa del checkpoint terminó en 35.41 s: **229 aprobadas, 4 fallos preexistentes y 1 omitida**. JAX no está instalado.

[Implementado y probado con casos sintéticos, no validado por campaña] Se añadió la opción explícita `weight_regularization_mode="absolute"`; `"legacy"` continúa como valor por defecto para preservar los proyectos existentes. En modo absoluto se minimiza el residuo ponderado más penalizaciones absolutas `mu` y `alpha`; el objetivo perfilado del optimizador y el full-refit usan los mismos términos. Se compararon 16 combinaciones de pesos uniformes, por λ, por punto y cero, con/sin cotas y suavizado, frente a SciPy y condiciones KKT; también se probó el caso por λ con suavizado y la propagación a la API/refit. La opción absoluta **no admite NNLS duro con suavizado** y falla explícitamente; la covarianza analítica existente no incorpora las penalizaciones. La ruta `legacy` conserva la incompatibilidad histórica descrita arriba y no debe confundirse con la función objetivo absoluta. No se han validado todos los casos de rango deficiente ni el backend JAX.

[Benchmark preliminar, no campaña] `validation/benchmark_weighted.py` ejecuta el Newton denso en procesos independientes con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; resultados reales en `validation/results/summary/weighted_benchmark.csv`. Para 12 puntos: 3×100, 0.00377 s y 81 820 KiB RSS; 5×300, 0.0632 s y 188 380 KiB; 8×500, 0.719 s y 594 120 KiB. La norma infinita del gradiente fue inferior a 1.9×10⁻¹⁴ en esos tres casos. 10×1000 se omitió por el límite de 4000 coeficientes; su Hessiana densa sola requeriría 800 000 000 bytes. Son mediciones únicas con ruido sintético, no benchmarks estadísticos; el solver no expone iteraciones y no se ejecutó una referencia independiente a esas dimensiones.

[Infraestructura parcial] El paquete `validation/` ya contiene un ejecutor reproducible, checkpoints por réplica con checksum, reanudación y comprobación de commit, huella de código, configuración y versiones; se probaron ejecución serial/paralela, corrupción y cambio de fuente. **Solo admite UV-Vis 1:1 con ruido i.i.d. y error analítico**, por lo que no satisface aún la cobertura de técnicas ni de métodos de la campaña solicitada. Véase `validation/README.md`. No iniciar dos coordinadores sobre el mismo directorio: aún no hay bloqueo entre coordinadores. El archivo de configuración `validation/configs/uvvis_11_pilot.json` define 20 conjuntos exploratorios; no se ha completado todavía una campaña confirmatoria.

[Piloto exploratorio UV-Vis 1:1] Con `validation/configs/uvvis_11_pilot.json`, código `dc88860ed0d5256d14785210785b6b5cafe87753`, 8 puntos y ruido normal σ=0.0002, se ejecutaron 20 conjuntos independientes; 20 ajustes convergieron según el procesador. La cobertura del IC t95 para log K verdadero fue **19/20 = 0.95** (error Monte Carlo 0.0487; Wilson 95 % [0.7639, 0.9911]), sesgo 0.00962, RMSE 0.01415, SD 0.01065 y SE medio 0.01625 (SE/SD 1.525). Tiempo sumado de trabajadores 3.19 s; esto no es tiempo de pared. Los resultados están condicionados al éxito; el piloto no demuestra calibración ni sustenta recomendaciones. Se comprobó la reanudación sin duplicados y la igualdad SHA-256 del CSV antes y después de reanudar. Resumen en `validation/results/summary/uvvis_11_pilot_dc88860.json` y datos completos del pequeño piloto y manifiesto en `validation/results/pilot/uvvis_11_dc88860/`; los checkpoints de ejecución permanecen fuera del repositorio. Para reproducir, usar el código del commit indicado y ejecutar el comando del README.

[Escalabilidad parcial] En el modo absoluto, cuando `alpha_smooth=0`, el problema es separable por longitud de onda y ya se resuelve por columnas en lugar de construir una Hessiana global. Una prueba sintética con 3 especies y 1400 longitudes de onda (4200 coeficientes) recuperó los coeficientes de referencia; el caso con suavizado sigue usando la Hessiana densa y conserva el límite de 4000. No se han implementado ni comparado aún solvers dispersos.

[Auditoría `legacy` vs `absolute`, sonda puntual] `validation/legacy_vs_absolute_probe.py` (resultado commitado en `validation/results/legacy_absolute_probe.json`) ajusta un dataset del piloto con tres ponderaciones (unidad, rampa por longitud de onda, rampa por punto) bajo ambas convenciones mediante la objetivo perfilada de producción. En este régimen (μ=10⁻², sin suavizado) la diferencia de log K entre convenciones es ≤1.5·10⁻⁸ y las objetivo cruzadas coinciden a 10⁻⁸: la divergencia de convenciones (la ruta `legacy` reescala μ respecto a los pesos; `absolute` añade la penalización explícitamente a la objetivo) es numéricamente irrelevante cuando los coeficientes penalmente activos son pocos y pequeños. La auditoría completa (regímenes de penalización fuerte, cotas activas, pesos extremos, suavizado `absolute` con solver disperso) sigue pendiente; una sonda de un dataset no la sustituye.

## 5. Investigación del sesgo UV-Vis (Etapas A–D, completadas)

Documento completo con tablas, figuras y datos: `docs/UVVIS_BIAS_INVESTIGATION.md`.
Resultados decisivos, todos reproducibles desde los artefactos commitados:

- Reproducción bit a bit del piloto en `dc88860` y en HEAD (20/20 filas idénticas); la diferencia de `source_sha256` se explicó por finales de línea CRLF/LF del checkout original.
- Ruido cero: el procesador recupera log K = 3 con desviación −2.5·10⁻¹⁴; argmin exacto de la objetivo perfilada idéntico a 10 dígitos para OLS, penalización suave y NNLS. Sin defecto generador–ajuste.
- Comparación pareada (mismos datos para los tres proyectores, argmin escalar exacto, referencia independiente de equilibrio cerrado): el estimador sin restricciones es insesgado (−0.0003 ± 0.0044); el penalizado/NNLS tiene sesgo +0.0096 ± 0.0024 y 20/20 réplicas sobre el verdadero. NNLS y penalización μ=10⁻² coinciden a 10⁻⁶; la saturación ocurre desde μ≈10⁻⁴ (CᵀC~10⁻⁵).
- Mecanismo: E[RSS_pen(θ) − RSS_ols(θ)] del recorte de coeficientes negativos crece ~5400× al alejarse θ por debajo del verdadero; el mínimo de E[RSS_pen] queda por encima del valor verdadero y el de E[RSS_ols] en él. La asimetría no persiste sin restricciones.
- Error estándar: el Jacobiano implementado es la derivada proyectada de VarPro y coincide con su identidad a 6.5·10⁻⁶; el SE describe el estimador no restringido (SE/SD = 0.83; tiende a 1.02 con σ→0), mientras el ajuste usa el restringido (SD 0.0107 ⇒ SE/SD = 1.53). La cobertura 19/20 del piloto procede de intervalos ~3× más anchos que la dispersión real del estimador. No se modificó el núcleo: no hay defecto que corregir; hay acoplamiento estimador↔covarianza por documentar y cuantificar con bootstrap.
- Nueve pruebas de regresión en `tests/test_uvvis_bias_reference.py` fijan: recuperación sin ruido (tres proyectores y producción), química cerrada vs NewtonRaphson, argmin exacto = Powell archivado, referencia independiente, identidad del Jacobiano, regímenes de μ y reproducibilidad de estimaciones archivadas del piloto.
- Matriz de 21 escenarios pareados (20–40 réplicas) con configuraciones en `validation/configs/bias/` y resúmenes en `validation/results/bias/`; figuras regenerables con `validation.make_bias_figures`.

## 6. Trabajo pendiente y dictamen

- Completar y validar el estimador ponderado para cotas suaves y suavizado, sin alterar los resultados por defecto.
- Crear y verificar el arnés `validation/` para todas las técnicas y métodos solicitados, incluidos checkpoints y manifiesto reproducible.
- Implementar y contrastar las opciones explícitas para ε ≥ 0 y las correcciones HC2/HC3.
- Ejecutar la campaña con los tamaños, escenarios y criterios de aceptación fijados antes de iniciarla; analizar por separado los casos con error correlacionado no modelado.
- Elaborar las tablas con errores Monte Carlo, el resumen CSV, las recomendaciones respaldadas por datos, los tiempos y el dictamen final.
- Realizar los commits por etapa; reservar la declaración de fase terminada para cuando se cumplan todos los criterios. Un eventual push del checkpoint WIP no equivale a la publicación de una fase validada.

**Dictamen provisional: no apto para declarar terminada la Fase 2 ni para extraer conclusiones de cobertura.** Cualquier commit que incluya este informe es un checkpoint de trabajo parcial, no una certificación de la fase.
