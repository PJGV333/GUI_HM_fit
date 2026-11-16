# Etapa E.2 — piloto del conflicto entre objetivos espectroscópicos

**Dictamen: E2-B, evidencia parcial.** No se propone integrar un objetivo nuevo ni cambiar `legacy`. Este piloto no valida incertidumbres, selecciona hiperparámetros definitivos ni demuestra recuperación general. La formulación `absolute` es matemáticamente coherente en las rutas estudiadas, pero las penalizaciones originales no son fiables para estos ejemplos y el solver registra fallos en 1:2.

## Contratos matemáticos observados

Sea `Y` la matriz de espectros **después** de EFA opcional y de la corrección de línea base; `C(k)` las concentraciones de especies, `G` el mapa de grupos, `A` las absortividades agrupadas y `R=C(k)GA−Y`. `L=D₂ᵀD₂`, con segundas diferencias entre canales consecutivos (no normalizadas por espaciamiento en nm). El multiplicador residual es `M_ij=w_j sqrt(v_ij)` si hay pesos por punto `v`; si no, `M_ij=w_j`. Los pesos estadísticos efectivos son `M²`, no `M`. Con `std`, `w` es la desviación estándar entre espectros de cada longitud de onda, normalizada por el máximo y después elevada a `power`; no es por sí misma una estimación de precisión inversa.

En la ruta regularizada estudiada (`soft_bound`, `delta_rel=0.005`):

* **Interior legacy, pesos solo por longitud de onda:** minimiza aproximadamente `ΦL=½||R||²+½ μ||min(A,0)||²+½ α tr(ALAᵀ)`, sujeto a `A≥b(k,Y)`; el PGD histórico se detiene tras hasta 300 pasos. Con pesos por punto, esta descripción cambia: se pasan multiplicadores al interior. Las rutas sin suavizado de `soft_penalty` y `nnls_hard` tienen resolutores propios y no se deben confundir con este PGD.
* **Exterior legacy:** minimiza `sqrt(||M⊙R||²/N)` evaluado en la solución interior aproximada. Es un problema jerárquico con objetivos potencialmente diferentes: omite penalizaciones y, con los pesos espectrales, tampoco usa la misma norma de residuos.
* **Interior absolute:** minimiza `ΦA=½||M⊙R||²+½ μ||min(A,0)||²+½ α tr(ALAᵀ)` con las mismas cotas relativas calculadas para esa modalidad. El Newton ponderado se usa en las rutas blandas; sin acoplamiento espectral se resuelve por columna.
* **Exterior absolute:** minimiza `sqrt(2ΦA/N)` con `A` interior; es una optimización regularizada consistente **si el interior converge**. Las cotas relativas se calculan a partir de una estimación dependiente de `C(k)`, `Y` y los pesos: el conjunto factible puede cambiar con `k`. `nnls_hard` con suavizado no está soportado en `absolute`.

`N` cuenta únicamente observaciones espectrales, no residuos adicionales de penalización. El factor `½` no modifica el minimizador. No hay normalización general de concentraciones ni señales que haga invariantes `μ` y `α`: la escala de `CᵀC`, de los multiplicadores y de los datos determina la fuerza relativa de las penalizaciones. Cambiar unidades requiere transformar parámetros y cotas de manera coherente. La corrección de línea base resta una media por espectro en el rango elegido; modifica `Y` y por ello también los pesos y cotas. La comparación de absortividades frente a la verdad cruda del generador incluye ese efecto y **no** es un error de recuperación ajustado por línea base.

Fuentes: `hmfit_core/processors/spectroscopy_processor.py`, funciones `_solve_absorptivities`, `_spectral_profile_objective`, `compute_spectral_weights` y `baseline_correct`; `validation/stage_e2_formulation_pilot.py`.

## Experimento reproducible

Datos, versiones, semilla, parámetros, métricas y fallos completos: `validation/results/stage_e2_formulation_pilot.json`; generador: `validation/stage_e2_formulation_pilot.py`. Se ejecutó con BLAS de un hilo, Python 3.14.10, NumPy 2.4.2 y SciPy 1.17.1, desde `e93f3bc`; el JSON contiene el SHA-256 del script. Se incluyen tres golden 1:1 (`log β=3`), un factorial línea base/pesos para golden 1, y escenarios sintéticos 1:1 y 1:2 adicionales con y sin ruido. Son **84 intentos**, con **13 fallos explícitos** de solver; no se excluyen silenciosamente. El escenario 1:2 usa `log β=(3,2.5)`. No es un conjunto externo de laboratorio ni una evaluación estadística de cobertura.

Las cinco variantes principales se fijaron antes del primer piloto; las dos variantes aisladas y los casos de escalamiento se añadieron después como controles:

A. `legacy_300`: PGD histórico.
B. `legacy_exact`: Newton ponderado con multiplicadores unitarios, exclusivamente inyectado en el experimento.
C. `absolute_original`: `μ=0.01, α=0.02`, ajustes históricos de los golden.
D. `absolute_scaled`: piloto no ajustado a golden: `μ=0.01 ||C(k₀)||₂²` y `α=0.01 ||C(k₀)||₂²/||L||₂`; es una escala heurística reproducible, **no** una selección óptima ni independiente de unidades/pesos. En golden 1 resulta `μ=1.0512e−7, α=6.6053e−9`.
E. `absolute_unregularized`: `μ=α=0`, manteniendo cotas relativas. No es un ajuste totalmente sin restricciones.
F. `absolute_penalty_only`: `μ=0.01, α=0`.
G. `absolute_smooth_only`: `μ=0, α=0.02`.

La tabla usa RMS **sin multiplicar por pesos** sobre la señal ya corregida, y KKT como máximo gradiente proyectado dividido por la máxima magnitud de los términos del gradiente. Estos RMS no son los RMS ponderados previamente publicados; no deben compararse como si fueran la misma magnitud. Los tiempos son de la ejecución preliminar anterior a una edición de tipado del script; el JSON contiene los de la repetición final. Incluyen el ajuste completo y son orientativos, dependientes del equipo. La optimalidad interior no garantiza que Powell alcance el óptimo exterior global.

| Caso | Variante | log β estimado | RMS sin ponderar | KKT normalizado | Tiempo s | Estado |
|---|---|---:|---:|---:|---:|---|
| Golden 1 | A | 2.68871 | 0.15227 | 0.629 | 0.071 | interior no convergido |
| Golden 1 | B | ~0 | 0.24387 | 1.25e−12 | 0.323 | cota inferior exterior |
| Golden 1 | C | 4.30256 | 0.74004 | 5.73e−10 | 0.225 | sesgo +1.303 |
| Golden 1 | D | 3.32728 | 0.00468 | 6.57e−9 | 0.117 | sesgo +0.327 |
| Golden 1 | E | 3.02423 | 0.000277 | 9.57e−10 | 0.071 | sesgo +0.024 |
| Golden 1 | F | 3.02460 | 0.000278 | 9.57e−10 | 0.071 | sin suavizado |
| Golden 1 | G | 4.29103 | 0.73994 | 8.07e−10 | 0.236 | sin penalización negativa |
| Golden 2 | A | 2.67167 | 0.15608 | 0.627 | 0.094 | interior no convergido |
| Golden 2 | B | 0 | 0.24386 | 1.25e−12 | 0.296 | cota inferior exterior |
| Golden 2 | C | 4.30290 | 0.73990 | 5.73e−10 | 0.262 | sesgo +1.303 |
| Golden 2 | D | 3.32521 | 0.00508 | 7.38e−9 | 0.210 | sesgo +0.325 |
| Golden 2 | E | 2.98935 | 0.000275 | 8.06e−10 | 0.127 | sesgo −0.011 |
| Golden 2 | F | 2.99754 | 0.000276 | 8.06e−10 | 0.103 | sin suavizado |
| Golden 2 | G | 4.29128 | 0.73980 | 8.06e−10 | 0.220 | sin penalización negativa |
| Golden 3 | A | 2.68968 | 0.15213 | 0.629 | 0.080 | interior no convergido |
| Golden 3 | B | 0 | 0.24388 | 9.37e−13 | 0.292 | cota inferior exterior |
| Golden 3 | C | 4.30282 | 0.74049 | 5.74e−10 | 0.244 | sesgo +1.303 |
| Golden 3 | D | 3.32747 | 0.00288 | 9.97e−17 | 0.141 | sesgo +0.327 |
| Golden 3 | E | 3.08997 | 0.001005 | 7.55e−9 | 0.077 | sesgo +0.090; ruido en cola |
| Golden 3 | F | 3.07179 | 0.001006 | 7.54e−9 | 0.070 | sin suavizado |
| Golden 3 | G | 4.29125 | 0.74039 | 8.08e−10 | 0.186 | sin penalización negativa |

La diferencia entre exterior `legacy` y `absolute` no explica por sí sola el resultado: sin pesos espectrales, B y C tienen objetivo interior igual en estos casos y ambos obtienen ~0, mostrando que la regularización también es decisiva. Con pesos `std`, C cambia a ~4.30. En golden 1, E da 3.12169 sin línea base ni pesos, 3.02160 con línea base y pesos unitarios, 2.98908 sin línea base y con pesos, y 3.02423 con ambos: estos cuatro ajustes muestran interacciones; no identifican el efecto causal de una línea base real distinta del generador.

En el caso 1:1 adicional con ruido, A da 4.72388 con KKT 0.994, B y C dan ~0, D da 3.17399 y E 2.99666. Sin ruido, E da 3.00000000003; esto comprueba recuperación en ese ejemplo, no universalmente. En 1:2 con ruido, A devuelve `(2.99941, 6)` con KKT 0.998, B y C fallan por no convergencia, D devuelve `(3.40926, ~0)` y E falla en búsqueda lineal. Sin ruido, B y C vuelven a fallar, D devuelve `(3.41106, ~0)` y E devuelve `(3.00007, 2.60302)` con RMS `3.26e−9`. El segundo parámetro y los espectros muestran sensibilidad o posible debilidad de identificabilidad; no se ha calculado un perfil ni demostrado formalmente su falta de identificabilidad.

Como ejemplo de distorsión de absortividades, el RMSE frente a los coeficientes crudos verdaderos en golden 1 es A 188.19, B 392.44, C 751.48, D 106.23 y E 11.93; la línea base aplicada limita su interpretación. Las rugosidades, ambos objetivos y límites se conservan por fila en el JSON. Los controles F y G muestran que, en estos tres golden, el suavizado original domina la degradación; esto no generaliza a otras formas de espectro. Ocho ajustes G fallan explícitamente por no convergencia.

En golden 1, multiplicar por 10 toda la señal mantiene log β de C en 4.30255 y de E en 3.02423, pero el control G falla. Multiplicar concentraciones y señal por 10, ajustando la verdad de log β de 3 a 2 para conservar el equilibrio 1:1, da 5.84944 para C, 2.32728 para D y 2.02423 para E. Se mantuvieron parámetros C fijos y D se recalculó según su regla: esto evidencia sensibilidad de los parámetros fijos a la escala, no una conversión de unidades completamente validada. Faltan un análisis de condicionamiento del 1:2 y una validación de invariancia para todas las convenciones y tolerancias del solver.

## Decisión y salvaguarda pendiente

1. Conservar `legacy` para reproducibilidad, sin presentar automáticamente el interior histórico como verificado. Por ahora no se ha implementado ninguna advertencia; la salvaguarda recomendada es calcular factibilidad y gradiente proyectado del objetivo **efectivamente usado**, con escala documentada, sin modificar `A` ni los resultados. Debe informar `no verificado` cuando no pueda evaluar correctamente el contrato, y probar igualdad exacta de los resultados históricos con y sin diagnóstico.
2. No recomendar `absolute` con parámetros originales; el modo sin regularización es prometedor en estos ejemplos 1:1, pero falla en 1:2 con ruido. La heurística D tampoco es aceptable como ajuste general por su sesgo y las cotas alcanzadas.
3. Antes de solicitar autorización de integración, comparar escalamiento de señal y concentración, aislar `μ` y `α`, ampliar datos independientes y réplicas, examinar perfiles/identificabilidad y resolver o informar los fallos del interior. Cualquier nueva convención debe versionarse explícitamente; no modificar proyectos históricos silenciosamente.
4. Los SE actuales no están validados para el estimador restringido y suavizado. No se han modificado baselines golden ni código productivo en este piloto.
