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
comparación opcional de cinética no se ha ejecutado todavía.

Validación de constantes compartidas, sin JAX: 545 recolectadas, 522 aprobadas,
0 fallidas, 17 xfail, 1 omitida y 5 deseleccionadas; 443.85 s.

## Problemas y estado

| Etiqueta | Problema | Estado y evidencia |
|---|---|---|
| [Defecto comprobado por inspección] | Una constante de velocidad compartida entre reacciones de distintos órdenes requiere unidades incompatibles | Rechazada en la validación de la GUI; `tests/test_kinetics_shared_rate_order.py`. Configuraciones guardadas se abren y muestran el error. Implementado y probado con ejemplos sintéticos; no validado con datos publicados. |
| [Defecto comprobado por inspección] | JAX integra un solo paso RK4 por intervalo y usa otro corte de rango | Pendiente 1.1. |
| [Limitación metodológica] | Valores iniciales de cinética y escala | Pendiente 1.2. |
| [Limitación metodológica] | Presentación de resultados de potenciometría | Pendiente 1.3. |

## Cambios esperados en resultados

- Un mecanismo con la misma constante en reacciones de distinto orden ya no se ajusta desde
  la GUI: antes se aceptaba pese a la incompatibilidad dimensional. Los casos válidos no
  cambian por esta validación.

## Riesgos y cuestiones abiertas

- Configuraciones antiguas que compartan una constante entre órdenes distintos se seguirán
  abriendo, pero no podrán ajustarse hasta corregir el mecanismo. Las APIs del núcleo no se
  cambian por esta validación de interfaz.
- Las etapas 1.1, 1.2 y 1.3 y su validación siguen pendientes.
