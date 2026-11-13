# HM fit — Encargo de la Fase 3 (versión 2): etapas restantes y reglas de autonomía

*Autor del encargo: Dr. Pedro Jancarlo Gómez Vega · noviembre de 2026*

Este documento completa y, donde haya conflicto, **sustituye** al encargo original de la Fase 3
(`ENCARGO_FASE3.md`). Lo que no se modifica aquí sigue vigente. Las decisiones 1.1–1.3 (etapa 3.0)
ya están implementadas y aprobadas.

---

## 1. Mapa del programa (contexto; no implementar fases futuras)

| Fase | Contenido | Estado |
|---|---|---|
| Fase 1 | Integridad matemática: parámetros fijos y límites, grados de libertad, identificabilidad, bootstrap | Cerrada (`docs/STABILITY_PHASE1_REPORT.md`) |
| Fase 2 | Calibración de incertidumbres (campaña de cobertura) y decisiones sobre métodos de error y absortividades | Cerrada (`docs/STABILITY_PHASE2_REPORT.md`) |
| Banco de literatura | Comparación con referencias publicadas | Cerrado (`docs/LITERATURE_BENCHMARK.md`) |
| Revisión de cinética y potenciometría | Defectos K1–K7, A1–A2, diagnóstico de residuos | Cerrada (`docs/REVIEW_KINETICS_POTENTIOMETRY.md`) |
| **Fase 3** | Decisiones de la revisión (3.0), LM explícito (3.1), pesos RMN (3.2), rendimiento (3.3), CI (3.4), cierre (3.5) | **En curso: este encargo** |
| Después de la Fase 3 | Campaña final de cobertura sobre el motor resultante; evaluación de la unificación de los tres motores de error; fase de interfaz; preparación de la versión 1.0 | Pendiente de decisión del autor. **Fuera de alcance aquí** |

## 2. Estado al iniciar

- **3.0:** cerrada y aprobada (hasta `8f72278`).
- **3.1:** parcial (hasta `4647f58`):
  - núcleo LM explícito;
  - Jacobiano espectral para columnas independientes;
  - pruebas de diferencias finitas para `soft_penalty` y `nnls_hard`, en modos `legacy` y `absolute`, con restricciones activas e inactivas;
  - informe parcial en `docs/STABILITY_PHASE31_REPORT.md`.
- **Suite:** 556 aprobadas, 17 xfail, 3 omitidas, 5 deseleccionadas (marcador `slow`).

**Primer paso obligatorio.** El autor trabaja desde dos equipos con el mismo repositorio. Ejecuta
`git fetch` y confirma que la rama local `fix/stable-phase3-optimizers` coincide con su remota
(`4647f58` o un descendiente). Si no coincide, actualízala antes de tocar nada.

**Entorno.**
- Usa siempre el entorno virtual del proyecto, nunca el Python del sistema.
- JAX no se instala en el entorno principal (rompe 5 pruebas ajenas a cinética). Si hace falta, usa un entorno aislado.
- Las dependencias de desarrollo que falten se instalan en el entorno virtual y se registran en el informe.

---

## 3. Reglas de trabajo y autonomía

Estas reglas sustituyen a la regla 7 del encargo original ("detenerse y preguntar").

### 3.1 Principio

Trabaja hasta **terminar la Fase 3 completa (etapas 3.1 a 3.5)** en una sola ejecución. Ante
cualquier ambigüedad, **decide, registra y continúa**. El autor audita y corrige al final.
Un informe de "etapa incompleta" **no** es un punto de parada válido.

### 3.2 Cómo decidir

1. Aplica este orden de prioridad cuando dos requisitos choquen:
   1. reglas de seguridad del repositorio (§3.4);
   2. ningún cambio numérico sin caso que lo reproduzca, prueba y entrada en el CHANGELOG (regla 4 del encargo original);
   3. fuera de alcance (§5);
   4. decisiones explícitas del autor (este documento y el original);
   5. tu criterio técnico.
2. Entre opciones válidas, elige la que, por este orden:
   1. no cambia los resultados por defecto;
   2. es reversible;
   3. es más fácil de auditar.
3. Registra la decisión en la tabla **"Decisiones del agente"** del informe, con estas columnas: n.º (D-1, D-2…), contexto, alternativas, elección, justificación, evidencia, cómo revertirla.
4. Continúa en el mismo turno.

### 3.3 Lo que **no** es motivo para detenerse

- Nombres, ubicación del código o estructura de archivos y de commits.
- Diseño de pruebas y tolerancias numéricas: elígelas y justifícalas.
- Dependencias que falten en el entorno virtual.
- Elegir entre implementaciones equivalentes.
- Un modo sin Jacobiano analítico coherente: usa la regla de §4.1.b.
- Un criterio que no se cumple: documenta el resultado y sigue.
- Contradicciones o huecos del encargo: aplica §3.2.

### 3.4 Prohibiciones (no se hacen nunca, pero tampoco detienen el trabajo)

- Modificar `main`, fusionar, crear tags o releases.
- Reescribir historia ya empujada (nada de `push --force`).
- Eliminar o desactivar pruebas existentes. Un `xfail(strict=True)` que pase a XPASS se retira en el mismo commit y se explica.
- Cambiar algo de §5.

Si una tarea parece exigir una de estas acciones, **no la hagas**: anótala como pregunta para el
autor en el informe y continúa con lo demás.

### 3.5 Bloqueos técnicos

Si una subetapa no queda con la suite en verde después de tres enfoques distintos:

1. vuelve al último commit en verde;
2. documenta el bloqueo con la evidencia;
3. continúa con la siguiente subetapa que no dependa de ella.

Dependencias entre subetapas:
- 3.1.e–g dependen de 3.1.a–d;
- 3.2 funciona con LM o con Powell, así que no depende del resultado de 3.1;
- 3.3 y 3.4 son independientes.

### 3.6 Continuidad

Al terminar cada subetapa:
1. suite completa en verde (`python -m pytest -q`);
2. commit acotado;
3. `git push` de la rama;
4. actualizar la sección **"Estado de avance"** del informe (qué está hecho, qué sigue).

Así otra sesión, u otro equipo, puede retomar el trabajo. Si la sesión se acerca a su límite,
actualiza "Estado de avance" y haz push antes de que termine.

**Único punto de parada:** el final de la etapa 3.5.

---

## 4. Etapas

### 4.1 Etapa 3.1 — Completar Levenberg–Marquardt explícito (Espectroscopía y RMN)

**a. El objetivo es idéntico al de Powell en cada modo.** No se cambia el estimador.
- `legacy` (predeterminado): el residuo es **solo** el de datos, con ε(θ) obtenida del problema interno penalizado, igual que `_spectral_profile_objective`.
- `absolute`: el residuo incluye la penalización, igual que su objetivo actual.
- Resto de modos (`soft_bound`, `delta_mode`, penalización de suavidad, grupos de absortividad compartidos, pesos por punto, y cualquier otro que exista): replica exactamente lo que hace hoy el objetivo de Powell en cada uno.
- Anota en el informe, como observación y **sin cambiarla**, la diferencia de objetivo entre `legacy` y `absolute`.
- Incluye en el informe una tabla de inventario con estas columnas: modo, definición del residuo, tipo de Jacobiano (analítico o por diferencias finitas), prueba que lo verifica.

**b. Jacobiano coherente con ese objetivo.**
- Usa un Jacobiano analítico solo donde sea exacto. Cada uno debe verificarse contra diferencias finitas del **objetivo real**, con y sin restricciones activas.
- Caso `legacy` con `soft_penalty`: con el conjunto de ε negativas fijo, ε = (CᵀC + μ·D)⁻¹·CᵀY para cada λ, donde D es la diagonal que indica qué ε son negativas. Se deriva respecto a θ a través de C.
- En los modos acoplados (suavidad, grupos compartidos, pesos por punto), o donde no logres coherencia exacta, usa **diferencias finitas centradas del objetivo real**, con paso relativo h_j = ε_mach^(1/3)·max(|θ_j|, 1). Documenta el coste.
- **Nunca** uses un Jacobiano incoherente con el objetivo.

**c. Algoritmo** (ya iniciado; completar y documentar).
- Amortiguamiento de Marquardt escalado: λ·diag(JᵀJ).
- Subproblema lineal resuelto por QR o SVD de [J; √λ·D], no por ecuaciones normales.
- Criterios de parada relativos, independientes de las unidades:
  - reducción relativa del coste < ftol;
  - paso relativo ‖δ‖ / (‖θ‖ + xtol) < xtol;
  - gradiente escalado max_j |J_jᵀ r| / (‖J_j‖·‖r‖) < gtol, al estilo MINPACK.
- Elige y documenta los valores por defecto de ftol, xtol, gtol y del máximo de iteraciones.
- El resultado reporta siempre: éxito, iteraciones, razón de parada y gradiente escalado final.
- Un ajuste no convergido o que agota las iteraciones se marca **`>>> FIT NOT RELIABLE`** con la causa, igual que en cinética.

**d. Límites.**
- Proyección con conjunto activo: un parámetro en su límite, con el gradiente empujándolo hacia fuera, se mantiene fijo durante ese paso.
- Si al final un parámetro queda en un límite: aviso en el informe y SE = NaN con la nota "parámetro en el límite" (en la convención de la Fase 1, NaN significa "indefinido").
- Los parámetros fijados por máscara siguen las reglas de la Fase 1, sin cambios.

**e. Integración en Espectroscopía.**
- Selector de optimizador (LM / Powell, más los que ya existan) en la GUI, la API y la configuración.
- La exportación incluye el optimizador usado y sus diagnósticos de convergencia.
- Compatibilidad: una configuración guardada que indique optimizador lo conserva; una sin esa clave usa **Powell** (comportamiento previo).

**f. Integración en RMN.** LM sustituye a Powell **dentro** del ciclo IRLS existente, sin cambiar
los pesos (eso es la 3.2). La compatibilidad de configuraciones es la misma que en (e).

**g. Comparación y valor por defecto.**
- Casos: banco de literatura, `tests/phase1_cases.py` y línea base de regresión.
- Tabla por caso: log β con Powell y con LM, |Δ log β|, SSQ, éxito, iteraciones y tiempo (mediana de 5 ejecuciones, en la misma máquina).
- Se considera que Powell "resuelve bien" un caso cuando reporta éxito y reproduce la referencia publicada dentro de la tolerancia del banco.
- **LM pasa a ser el valor por defecto (solo para configuraciones nuevas)** si se cumplen las dos condiciones:
  1. |Δ log β| ≤ 1e-4 en todos los casos que Powell resuelve bien;
  2. SSQ(LM) ≤ SSQ(Powell)·(1 + 1e-8) en todos los casos.
- Si no se cumplen, Powell sigue como valor por defecto, LM queda disponible como opción y la tabla explica por qué. En ambos casos, **continúa**.

**h.** Si `test_absgrp_shared_groups_reduce_optical_components` pasa a XPASS, retira el `xfail` en
el mismo commit. Si sigue fallando, explica por qué en el informe.

**Commits sugeridos:**
1. modos y Jacobianos (a, b);
2. algoritmo y límites (c, d);
3. Espectroscopía (e);
4. RMN (f);
5. comparación y valor por defecto (g, h).

### 4.2 Etapa 3.2 — Pesos en RMN

- Opción visible *Pesos: unitarios / por señal (IRLS)* en la GUI, la API y la configuración.
- El resultado y la exportación muestran los pesos finales por señal y el número de ciclos IRLS.
- **El valor por defecto no cambia (IRLS).** Una configuración sin esa clave usa IRLS.
- Tabla de evidencia con los casos RMN del banco de literatura, con ambas opciones y el optimizador por defecto que resulte de 3.1: log β, SE, |Δ| frente a la referencia (HypNMR) y ciclos.
- **§6.4 (c) de `LITERATURE_BENCHMARK.md`:** requiere aprobación explícita del autor. **No lo corrijas.** Describe el problema y la corrección propuesta como pregunta en el informe.

### 4.3 Etapa 3.3 — Rendimiento

**Potenciometría: vectorizar el cálculo de pH.** Sustituye `brentq` punto a punto por un
resolvedor vectorizado y protegido: intervalo en log[H], Newton con respaldo de bisección, todos
los puntos a la vez. Requisitos:
- No debe fallar donde `brentq` acierta.
- |ΔpH| ≤ 1e-10 frente al resolvedor actual en todos los casos de prueba y en casos extremos: pH < 2 y > 12, ácidos y bases fuertes, cerca de los puntos de equivalencia y soluciones muy diluidas.
- Parámetros ajustados: diferencia ≤ 1e-3·SE (no cambia ninguna cifra reportada).
- Velocidad ≥ ×10 por réplica del bootstrap full-refit (mediana de 5). Si no se alcanza, entrega la mejor mejora lograda con sus mediciones y documéntalo.
- Conserva el resolvedor actual como implementación de referencia para las pruebas de equivalencia.

**Otros cuellos de botella.** Mide con `cProfile` una ejecución del banco y un bootstrap
full-refit por técnica. Optimiza solo los puntos calientes medidos, cada uno con su prueba de
equivalencia (mismas semillas; mismos números a 1e-12 relativo, o con el criterio anterior si
interviene un optimizador). Incluye en el informe la tabla de perfiles antes y después.

### 4.4 Etapa 3.4 — CI

- El CI debe quedar en verde en cada push.
- Si la prueba de humo del banco de literatura no corre completa en cada push, añádela. Elige un subconjunto representativo que no alargue el CI más de unos 5 minutos y justifica la elección.
- JAX no entra en el entorno principal del CI.

### 4.5 Etapa 3.5 — Cierre

**Un solo informe:** `docs/STABILITY_PHASE3_REPORT.md`. Integra en él el contenido de
`STABILITY_PHASE31_REPORT.md`. Secciones obligatorias:

1. Estado de avance.
2. Tabla de problemas y estado, con las etiquetas **[Defecto comprobado]**, **[Limitación metodológica]** y **[Sospecha]**, distinguiendo siempre: implementado, probado con datos sintéticos, validado con datos publicados.
3. Decisiones del agente (D-n).
4. Evidencia de cada cambio.
5. Pruebas antes y después de cada etapa: recolectadas, aprobadas, fallidas, xfail, omitidas y deseleccionadas.
6. Tiempos (3.1.g y 3.3).
7. Cambios esperados en resultados.
8. **Pendientes de 3.0**, que deben quedar respondidos:
   - si se reprodujo el estancamiento con k1 = k2 antes de aplicar el factor 3;
   - la causa de los cinco fallos al instalar JAX: diagnóstico en un entorno aislado, distinguiendo entre cambio de versiones de dependencias y cambio de comportamiento al importar;
   - si las estadísticas y los errores tras un ajuste con JAX se calculan con el objetivo minimizado o con SciPy.
9. Problemas abiertos.
10. Preguntas para el autor, como mínimo:
    - JAX: ¿se mantiene como experimental o se oculta en la versión estable?
    - §6.4 (c) del banco de literatura;
    - todo lo anotado según §3.4.

**Además:**
- `CHANGELOG.md` actualizado.
- Push final y enlace a la ejecución de CI en verde.
- **Después, detente para revisión.**

---

## 5. Fuera de alcance

- La campaña final de cobertura.
- Cambiar los métodos de error por defecto o el estimador de absortividades.
- La unificación de los tres motores de error.
- Rediseños de interfaz, más allá del selector de optimizador (3.1) y de la opción de pesos (3.2).
- Cambiar el valor por defecto de los pesos RMN.
- Corregir §6.4 (c).
- Ocultar o retirar el motor JAX.
- Publicar versiones o modificar `main`.

## 6. Entregables

1. Rama `fix/stable-phase3-optimizers` empujada, con commits acotados y la suite en verde en cada uno.
2. `docs/STABILITY_PHASE3_REPORT.md` completo (§4.5).
3. `CHANGELOG.md` actualizado.
4. Enlace a la ejecución final de CI en verde.
