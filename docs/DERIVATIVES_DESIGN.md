# Derivadas especializadas: diseño inicial

**Diseño únicamente; ninguna ruta nueva ha sido implementada ni validada.** La Fase 4 está fuera de alcance.

## Equilibrio

Para `F(c, θ)=0`, resolver `F_c S = -F_θ` mediante un sistema lineal, sin inversión explícita. Especificar orden y unidades de parámetros, incluidos log10 y su factor `ln(10)`. Comprobar residuo del balance, condicionamiento y singularidad antes de aceptar sensibilidades. Ensayar primero 1:1, después otros modelos y ácido-base; comparar con soluciones analíticas y diferencias centradas en varias escalas y pasos.

## Cinética

Integrar con SciPy el sistema aumentado `c'=f(c,θ)`, `S'=f_c S+f_θ`, con `S(0)=∂c(0)/∂θ`. Documentar tolerancias, condiciones iniciales, parámetros compartidos y fijos. Contrastar contra diferencias centradas de **la misma** integración y revisar convergencia al reducir paso y tolerancias. Incluir modelos de primer y segundo orden, reversibles y consecutivos antes de proponer integración en producción.

## Proyección variable

La derivada debe corresponder al objetivo efectivo y a sus pesos, penalizaciones y suavizado. Para NNLS o conjuntos activos cambiantes no presupone diferenciabilidad; informar el cambio y usar respaldo verificado. Comparar exactitud, tiempo, escalamiento y singularidad; no sustituir ninguna ruta productiva antes de demostrar equivalencia o mejora.
