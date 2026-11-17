# Etapa E.3 — investigación focalizada del 1:2

## Alcance y métodos

Estudio exploratorio del diseño 1:2 de E.2: 10 puntos, `[H]ₜ=10⁻³`, `[G]ₜ=0…2×10⁻³`, 31 canales, valores verdaderos `(log β₁, log β₂)=(3, 2.5)` y espectros gaussianos conocidos. Se examinan señal sin ruido y una realización independiente con ruido normal de desviación `2×10⁻⁴`, semilla 32. Sin línea base, multiplicadores unitarios y cota inferior relativa `−0.005 max|C⁺Y|` por especie, recalculada en cada punto. No se consideran estos resultados un estudio Monte Carlo ni una estimación de cobertura.

`validation/stage_e3_identifiability.py` resuelve el equilibrio mediante una ecuación escalar de balance de masa y Brent, independientemente de Newton–Raphson de producción; las pruebas contrastan concentraciones y diferencias centrales respecto a ambas constantes. Las diferencias relativas observadas entre derivadas de concentraciones fueron `1.13×10⁻⁷` y `6.50×10⁻⁸` con paso `10⁻³`. La proyección interior independiente usa `scipy.optimize.lsq_linear`/BVLS sobre la matriz de datos y, si corresponde, segundas diferencias, con `μ=0` y la misma cota relativa. En cada punto de perfil se resuelven las absortividades y se reoptimiza la otra constante: malla inicial de siete puntos en `[0,6]` seguida de minimización acotada local alrededor del mejor punto de malla. No se garantiza un mínimo global entre puntos de la malla. Se registran objetivo total (suma de cuadrados, **sin factor ½**), gradiente proyectado absoluto, estados y tiempos en `validation/results/stage_e3_identifiability.json`.

Se comparan `α=0` y `α=7.541875008433269×10⁻⁹`, el valor piloto E.2 para este diseño; no se modifica producción ni se supone que dicho valor sea óptimo. El gradiente proyectado del BVLS se comprueba en el problema original; es una magnitud **absoluta**, no el indicador normalizado del procesador. El Jacobiano perfilado se estima por diferencias centrales con pasos `10⁻³` y `5×10⁻⁴`; los límites cambian con los parámetros y pueden cambiar el conjunto activo. Los valores singulares y la correlación de columnas describen **ese Jacobiano perfilado**, no la matriz de covarianza ni el Jacobiano de concentraciones.

## Perfiles y sensibilidad

| Caso | α | log β₂ fijado | log β₁ reoptimizado | suma de cuadrados |
|---|---:|---:|---:|---:|
| Sin ruido | 0 | 0 | 2.99987 | 3.49×10⁻¹² |
| Sin ruido | 0 | 2.5 | 3.00000 | 7.56×10⁻²⁸ |
| Sin ruido | 0 | 4 | 3.00818 | 3.64×10⁻¹³ |
| Sin ruido | 0 | 6 | 3.33241 | 6.34×10⁻⁸ |
| Ruido, semilla 32 | 0 | 0 | 3.00000 | 8.39×10⁻⁶ |
| Ruido, semilla 32 | 0 | 2.5 | 2.99065 | 8.30×10⁻⁶ |
| Ruido, semilla 32 | 0 | 6 | 3.32654 | 7.63×10⁻⁶ |
| Ruido, semilla 32 | 7.54×10⁻⁹ | 0 | 3.45442 | 0.00285615 |
| Ruido, semilla 32 | 7.54×10⁻⁹ | 2.5 | 3.45430 | 0.00285635 |
| Ruido, semilla 32 | 7.54×10⁻⁹ | 6 | 3.63937 | 0.00347922 |

Los perfiles de ambos parámetros y todos los puntos están en el JSON. Por ejemplo, sin ruido ni suavizado, fijar log β₁ en 2 en lugar de 3 aumenta la suma de cuadrados hasta `5.03×10⁻⁴`, después de reoptimizar log β₂. En cambio, el perfil de β₂ es casi plano en una región extensa: con una sola realización de ruido, la malla llega al límite superior 6 con un objetivo inferior al de 2.5. Esto respalda **identificación débil de β₂ en este diseño**, no falta estructural de identificabilidad en todos los diseños o valores.

En la verdad, los valores singulares de `C` son `[3.2417×10⁻³, 1.4136×10⁻³, 1.4143×10⁻⁴, 1.3154×10⁻¹¹]`, condición aproximada `2.46×10⁸`: la cuarta columna aporta muy poca información independiente. Sin ruido ni suavizado, el Jacobiano perfilado tiene valores singulares `0.0123491` y `6.68×10⁻⁷`, correlación de columnas `−0.9963` y diferencia relativa entre pasos `0.00148`. Con ruido y sin suavizado, los valores cambian a `0.0128991` y `0.000177`, correlación `0.153` y discrepancia entre pasos `0.00376`; la sensibilidad depende de los datos y del conjunto activo. Para el caso sin ruido con suavizado, la diferencia relativa entre pasos es `0.466`: su correlación y valores singulares locales **no deben tratarse como estables**. Con ruido y suavizado, la discrepancia es `1.21×10⁻⁶`. Estas cifras describen sensibilidad numérica local, no intervalos de confianza.

## Fallos del solver observados

`validation/stage_e3_solver_failure_probe.py` repite únicamente los dos ajustes 1:2 con ruido y captura los sistemas que lanzan excepción, sin cambiar el algoritmo productivo. Se resolvieron esos sistemas fijos con referencias BVLS independientes; `validation/results/stage_e3_solver_failures.json` conserva los valores y la procedencia.

| Ruta | Fallo real | Condición de C en el fallo | KKT absoluto de referencia | Conclusión acotada |
|---|---|---:|---:|---|
| `absolute`, μ=α=0 | búsqueda lineal fallida, una longitud de onda | 7.75×10⁵ | 1.79×10⁻¹⁹ | existe una solución factible de referencia para ese sistema; no se identificó la causa exacta de la búsqueda fallida |
| `absolute`, μ=0.01, α=0.02 | agotamiento de 300 iteraciones, 31 canales | 1.18×10⁷ | 1.31×10⁻¹² | existe una solución factible de referencia para ese sistema; no se identificó el mecanismo exacto del agotamiento |

La condición anterior corresponde a `C`, no al Hessiano activo completo. En ambos casos la referencia respeta las cotas. Se verificó por diferencia central una derivada direccional analítica de la función interior en un punto perturbado: diferencias absolutas `5.21×10⁻¹⁹` sin regularización y `2.02×10⁻⁸` con las penalizaciones para paso `0.01`; el caso penalizado es sensible al tamaño de paso y presenta cancelación numérica. Esto no demuestra que todos los Jacobianos o búsquedas del solver estén libres de errores. La débil identificación y el mal condicionamiento son evidencia independiente de la existencia de ambos fallos; **no se ha demostrado que por sí solos los causen**.

El diagnóstico productivo E.3 no cambia ni repara el algoritmo interior. No se han validado los SE del estimador restringido o suavizado, ni se han generado nuevos baselines. La variación de ruido es limitada a una semilla; conviene estudiar más diseños y réplicas antes de conclusiones poblacionales.
