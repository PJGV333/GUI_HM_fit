# HM fit — Fase 2: validación estadística de incertidumbres

> **Estado: INCOMPLETO.** Este documento recoge trabajo en curso, no es el dictamen solicitado para la Fase 2. No se ha ejecutado la campaña de cobertura y no hay recomendaciones cuantitativas ni validación estadística de los métodos.

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

## 5. Trabajo pendiente y dictamen

- Completar y validar el estimador ponderado para cotas suaves y suavizado, sin alterar los resultados por defecto.
- Crear y verificar el arnés `validation/` para todas las técnicas y métodos solicitados, incluidos checkpoints y manifiesto reproducible.
- Implementar y contrastar las opciones explícitas para ε ≥ 0 y las correcciones HC2/HC3.
- Ejecutar la campaña con los tamaños, escenarios y criterios de aceptación fijados antes de iniciarla; analizar por separado los casos con error correlacionado no modelado.
- Elaborar las tablas con errores Monte Carlo, el resumen CSV, las recomendaciones respaldadas por datos, los tiempos y el dictamen final.
- Realizar los commits por etapa; reservar la declaración de fase terminada para cuando se cumplan todos los criterios. Un eventual push del checkpoint WIP no equivale a la publicación de una fase validada.

**Dictamen provisional: no apto para declarar terminada la Fase 2 ni para extraer conclusiones de cobertura.** Cualquier commit que incluya este informe es un checkpoint de trabajo parcial, no una certificación de la fase.
