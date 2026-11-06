# BORRADOR — Problemas conocidos en HM Fit v0.2.0

> **Borrador para revisión del autor. No publicado.** Se basa en reproducciones sobre el código
> fuente de v0.2.0 (`247c4df`, binarios construidos con Python 3.11):
> `validation/regressions/repro_published_defects.py`, resultados en
> `validation/results/regressions/repro_v020.json`. Detalle técnico en
> `docs/STABILITY_PHASE2_REPORT.md`, §6.

Los **ajustes** de v0.2.0 suelen recuperar bien las constantes. Los problemas siguientes afectan
sobre todo a las **incertidumbres** (SE, IC, correlaciones). Uno de ellos afecta al valor ajustado
cuando la tabla de parámetros contiene texto no numérico. Ninguno corrompe archivos ni datos.

## 1. Pueden cambiar resultados sin aviso

| Módulo | Problema | Qué revisar | Cómo |
|---|---|---|---|
| Espectroscopía, RMN y cinética: tabla de parámetros | Un valor inicial no numérico, por ejemplo `0,5` con coma decimal o una errata, se sustituye por **1.0** (espectroscopía/RMN) o **0.0** (cinética). Un límite no numérico se trata como **sin límite** | Ajustes en los que se escribió con coma decimal o se editó la tabla a mano | Revisar que la tabla use punto decimal; repetir el ajuste. El registro guarda los valores iniciales usados |
| Cinética: límites | Límite inferior > superior convierte el parámetro en **fijo** sin aviso | Parámetros con límites editados | Comprobar que Min < Max en todas las filas |
| UV-Vis: absortividades (`soft_penalty`, valor por defecto) | El solver de ε con penalización no converge a escala molar: log K sesgado (3.043 frente a 3.000 en un 1:1 de prueba, RELIABILITY_REVIEW #2) y SE demasiado pequeño | Constantes UV-Vis obtenidas con el estimador por defecto | Repetir el ajuste con `nnls_hard` o sin restricción y comparar |
| Todos: IC analíticos | Usan z = 1.96 en lugar de t de Student; los grados de libertad no restan los parámetros lineales (espectros, desplazamientos límite) | IC de titulaciones con pocos puntos | Ensanchar con t(ν), con ν = n_obs − p − p_lineales, o validar con bootstrap |
| Cinética, espectroscopía, RMN: bootstrap one-step | Amortiguamiento λ·I sin escalar: con k en M⁻¹ s⁻¹ el SD del bootstrap puede ser **miles de veces** menor que el real | Cualquier SE de "Bootstrap (one-step LM)" | No usar one-step en v0.2.0; usar el analítico o el full-refit |
| Cinética y espectroscopía: identificabilidad | Un parámetro sin efecto en los datos (columna nula del Jacobiano) se reporta con **SE = 0**, como si estuviera fijo | SE exactamente 0 en parámetros no fijados | Tratar SE = 0 de un parámetro libre como "no identificable" |
| Bootstrap linealizado / one-step | Parámetros colineales o sin sensibilidad reciben un SD finito o 0 | Correlaciones cercanas a ±1 | Interpretar sus SE como no definidos |
| Espectroscopía: full-refit | Las réplicas no convergidas se aceptan como válidas: los IC pueden ser demasiado estrechos | IC del full-refit | Comparar con el analítico; desconfiar si son mucho más estrechos |
| RMN 1:2 | Un fallo del optimizador se reporta como éxito (la constante queda en su valor inicial) | Ajustes 1:2 cuya K₂ coincide con el valor inicial | Cambiar el valor inicial y repetir; si K₂ no cambia, no está determinada |
| Todas las técnicas: parámetros fijos | La correlación con un parámetro fijo se muestra como 0 (no definida); en espectroscopía, un log β fijo puede recibir SE ≠ 0 e inflar los SE libres | SE de ajustes con parámetros fijos | Recalcular sin el fijo o con el fijo eliminado del modelo |

## 2. Fallan de forma visible (sin cambiar números)

| Módulo | Problema |
|---|---|
| Cinética | Fijar un parámetro puede lanzar `ValueError: Each lower bound must be strictly less than each upper bound` |
| Espectroscopía / RMN, modelos 1:1 | Todos los bootstrap fallan con un solo parámetro |
| RMN | El análisis de errores falla con `shape mismatch` si una constante no afecta a los datos |
| API de grafos cinéticos (no usada por la GUI) | `KineticGraph.add_kinetic_reaction` falla con Python < 3.14 |

## 3. Qué hacer

- Los resultados ya publicados con v0.2.0 que usen **SE o IC del one-step**, **UV-Vis con el
  estimador por defecto**, o una **tabla de parámetros escrita con coma decimal**, deberían
  recalcularse con una versión corregida.
- Versión correctora propuesta: **v0.2.1** (rama `fix/v0.2.1`, aún sin publicar). Solo incluye la
  validación de la tabla de parámetros (punto 1, primera y segunda fila) y la infraestructura de
  pruebas. El resto de las correcciones está en la línea principal (Fases 1 y 2) y llegará en la
  siguiente versión mayor.
