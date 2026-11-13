# Derivadas especializadas: diseño inicial

**Implementación experimental parcial:** existen sensibilidades probadas para equilibrio 1:1 y cinética irreversible de primer orden; no están integradas en los ajustes productivos. La cobertura de los demás modelos sigue pendiente. La Fase 4 está fuera de alcance.

## Equilibrio

Para `F(c, θ)=0`, resolver `F_c S = -F_θ` mediante un sistema lineal, sin inversión explícita. Especificar orden y unidades de parámetros, incluidos log10 y su factor `ln(10)`. Comprobar residuo del balance, condicionamiento y singularidad antes de aceptar sensibilidades. Ensayar primero 1:1, después otros modelos y ácido-base; comparar con soluciones analíticas y diferencias centradas en varias escalas y pasos.

### Asociación 1:1 implementada

`hmfit_core.derivatives.equilibrium.association_11_logk_sensitivity` recibe totales y concentraciones libres de H y G, y log10(K), con unidades coherentes. Devuelve derivadas de `[H, G, HG]` respecto a log10(K), manteniendo totales fijos, y el condicionamiento y error relativo de balance. Resuelve el sistema lineal de los dos balances sin invertir matrices; rechaza balances relativos superiores a 1e-9, entradas no físicas/no finitas y número de condición superior a 1e12. Estos umbrales son salvaguardas, no una demostración universal de precisión.

`tests/test_derivatives_equilibrium.py` compara tres escalas con una raíz escalar Brent independiente, una fórmula analítica y diferencias centradas, comprueba conservación, rechazos y una solución del Newton–Raphson existente con tolerancia ajustada. Para H_total=0.001, G_total=0.002 y log10(K)=3, diferencia absoluta máxima frente a diferencias centradas con paso 1e-4: 1.18023e-12; condición 3.2255. Medición exploratoria local de 1000 llamadas: 31.07 µs para la sensibilidad con equilibrio ya suministrado frente a 15.63 µs para dos raíces Brent independientes; no es una comparación de ajuste completo ni muestra una mejora de rendimiento. No se ha validado aún para 1:2, cooperatividad, ácido-base, otras singularidades o Jacobianos VarPro.

## Cinética

Integrar con SciPy el sistema aumentado `c'=f(c,θ)`, `S'=f_c S+f_θ`, con `S(0)=∂c(0)/∂θ`. Documentar tolerancias, condiciones iniciales, parámetros compartidos y fijos. Contrastar contra diferencias centradas de **la misma** integración y revisar convergencia al reducir paso y tolerancias. Incluir modelos de primer y segundo orden, reversibles y consecutivos antes de proponer integración en producción.

### Primer orden implementado

`hmfit_core.derivatives.kinetics.solve_first_order_sensitivity` integra con SciPy `A'=-kA`, `B'=kA` y `S_A'=-A-kS_A`, `S_B'=A+kS_A`, respecto a la constante física k (no log10 k). Admite sensibilidades iniciales no nulas y reporta método, tolerancias y evaluaciones; rechaza datos físicos, tiempos y tolerancias inválidos. Las pruebas contrastan soluciones analíticas y diferencias centradas usando BDF del modelo cinético existente, para k=0, 0.8 y 10, incluyendo condiciones iniciales dependientes de k. En k=0 la diferencia centrada usa una extensión matemática al valor negativo de k para referencia, no un ajuste físico negativo.

En un caso exploratorio con k=0.7, 25 tiempos y BDF (rtol=1e-10, atol=1e-13), el error absoluto máximo fue 4.00e-10 frente a la solución analítica y 6.43e-9 frente a diferencias centradas con paso 1e-4. En 20 llamadas locales, el sistema ampliado tomó 9.58 ms/llamada frente a 12.21 ms para dos integraciones de referencia: medición puntual, no evidencia de rendimiento en un ajuste. No se ha validado aún segundo orden, reacciones reversibles o consecutivas, parámetros compartidos/fijos ni múltiples experimentos. No se sustituye ningún Jacobiano productivo.

## Proyección variable

La derivada debe corresponder al objetivo efectivo y a sus pesos, penalizaciones y suavizado. Para NNLS o conjuntos activos cambiantes no presupone diferenciabilidad; informar el cambio y usar respaldo verificado. Comparar exactitud, tiempo, escalamiento y singularidad; no sustituir ninguna ruta productiva antes de demostrar equivalencia o mejora.
