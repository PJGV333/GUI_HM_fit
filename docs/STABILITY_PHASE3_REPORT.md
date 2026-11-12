# HM fit — Fase 3: informe intermedio

Rama `fix/stable-phase3-optimizers`, base `0634b93`. Estado: etapa 3.0 en curso.

## Entorno y línea base

Python 3.14.9, NumPy 2.4.2, SciPy 1.17.1; Linux, Qt offscreen. Suite base:
543 recolectadas, 520 aprobadas, 0 fallidas, 17 xfail, 1 omitida y 5 deseleccionadas
(538 seleccionadas), 336.89 s. `pytest.ini` excluye `slow` por defecto: son los cinco
`tests/test_coverage_campaign_smoke.py::test_full_campaign_tier[1,2,3,4,9]`, correspondientes
a la campaña de cobertura aplazada. JAX 0.12.0.1 se instaló posteriormente solo en el
entorno local para las pruebas opcionales; no es una dependencia del proyecto. Con JAX
instalado, la suite dio 517 aprobadas, 5 fallidas, 17 xfail, 1 omitida y 5 deseleccionadas:
las cinco fallas eran asignaciones in-place a arreglos JAX inmutables en pruebas de RMN,
espectroscopía y caché. Se retira JAX del entorno para mantener la línea base de CI; la
comparación opcional de cinética se ejecutó después con JAX aislado mediante `PYTHONPATH`,
sin instalarlo en el entorno normal.

Validación de constantes compartidas, sin JAX: 545 recolectadas, 522 aprobadas,
0 fallidas, 17 xfail, 1 omitida y 5 deseleccionadas; 443.85 s. Etapa 1.1,
sin JAX: 548 recolectadas, 523 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5
 deseleccionadas; 366.01 s. Validación de unidades: 550 recolectadas, 525 aprobadas,
0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas; 326.27 s. Etapa 1.2:
568 recolectadas, 543 aprobadas, 0 fallidas, 17 xfail, 3 omitidas y 5 deseleccionadas;
329.81 s. Pruebas focalizadas con JAX aislado: 21 aprobadas.

## Problemas y estado

| Etiqueta | Problema | Estado y evidencia |
|---|---|---|
| [Defecto comprobado por inspección] | Una constante de velocidad compartida entre reacciones de distintos órdenes requiere unidades incompatibles | Rechazada en la validación de la GUI; `tests/test_kinetics_shared_rate_order.py`. Configuraciones guardadas se abren y muestran el error. Implementado y probado con ejemplos sintéticos; no validado con datos publicados. |
| [Limitación metodológica] | JAX integra un solo paso RK4 por intervalo sin control de error | Marcado experimental, aviso en informe y registro; SciPy sigue como referencia. Implementado y probado con datos sintéticos, no validado con datos publicados. |
| [Defecto comprobado] | JAX devolvía solo dos filas por intercambiar las salidas de `lax.scan` | Corregido con autorización del autor; `test_jax_rk4_fine_and_coarse_grid` fallaba por forma (2, 3) frente a (401, 3). |
| [Defecto comprobado] | JAX usaba el corte predeterminado de `pinv` | Corregido con `RANK_RCOND`; una matriz de rango casi deficiente daba coeficientes ~10¹² con el corte predeterminado en float64. Prueba con y sin pesos. |
| [Defecto comprobado] | Ajustes globales mezclaban unidades temporales sin convertir | Reproducido con datos sintéticos: k real 0.025 s⁻¹ (1.5 min⁻¹), ajuste global sin conversión k = 0.02550756, SSQ = 0.20123 con datos sin ruido. Ahora se rechazan unidades conocidas distintas y se avisa si faltan o no se reconocen; prueba `tests/test_phase3_units.py`. Implementado y probado con datos sintéticos, no validado con datos publicados. |
| [Limitación metodológica] | Valores iniciales de cinética y escala | Propuesta automática y Log10 por defecto para k y A de Arrhenius; se conservan las celdas y escalas del usuario. Implementado y probado con datos sintéticos, no validado con datos publicados. |
| [Defecto comprobado] | Eyring aplicaba kBT/h sin convertir unidades de tiempo o concentración | Corregido en Eyring para unidades reconocidas; pruebas de equivalencia y de factores de unidad en `tests/test_phase3_auto_rates.py`. |
| [Limitación metodológica] | Presentación de resultados de potenciometría | Pendiente 1.3. |

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
esto no demuestra convergencia global para datos arbitrarios.

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

## Cambios esperados en resultados

- Arranques automáticos y Log10 pueden cambiar mínimos o convergencia respecto a los valores
  anteriores; parámetros y escala escritos por el usuario o cargados no se sustituyen.
- Eyring con unidades reconocidas distintas de s y M cambia constantes numéricas y valores
  ajustados de ΔH/ΔS; con s/M la fórmula no cambia. Texto desconocido ya no permite ajustar.
- Ajustes globales con unidades temporales o de concentración reconocidas y distintas se
  rechazan en vez de entregar constantes inconsistentes. No se convierten los datos.
- JAX: la trayectoria tiene ahora todos los tiempos; antes los ajustes ordinarios fallaban por
  dimensiones. El corte de rango cambia los coeficientes y puede cambiar parámetros y SE en
  mecanismos con matrices casi deficientes.
- Un mecanismo con la misma constante en reacciones de distinto orden ya no se ajusta desde
  la GUI: antes se aceptaba pese a la incompatibilidad dimensional. Los casos válidos no
  cambian por esta validación.

## Riesgos y cuestiones abiertas

- Las concentraciones no tenían campo de unidades; se añade un campo opcional vacío por
  defecto para conservar configuraciones existentes. Una unidad vacía o desconocida en un
  ajuste global solo produce aviso y no garantiza compatibilidad de unidades.
- Configuraciones antiguas que compartan una constante entre órdenes distintos se seguirán
  abriendo, pero no podrán ajustarse hasta corregir el mecanismo. Las APIs del núcleo no se
  cambian por esta validación de interfaz.
- [Sospecha] En JAX con precisión de 32 bits, un corte de rango de 10⁻¹⁰ está por debajo de
  la precisión de máquina; no se ha validado el comportamiento para matrices mal condicionadas.
- La etapa 1.3 y su validación siguen pendientes.
