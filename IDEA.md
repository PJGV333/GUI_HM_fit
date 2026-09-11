# HM fit

HM fit es una plataforma científica para modelar y ajustar datos experimentales de sistemas químicos de equilibrio, especialmente titulaciones ácido-base y potenciometría, con extensión a espectroscopía, NMR y cinética química.

El objetivo principal es proporcionar una interfaz gráfica robusta, intuitiva y científicamente rigurosa que permita pasar desde los datos experimentales hasta un modelo químico ajustado sin obligar al usuario a construir manualmente toda la formulación matemática.

## Flujo principal

Para potenciometría y equilibrio ácido-base, el flujo de trabajo debe ser natural:

1. Cargar la curva experimental.
2. Identificar las columnas de volumen y señal/potencial.
3. Definir analito, concentración y volumen inicial.
4. Definir el titulante experimental.
5. Definir las especies químicas presentes.
6. Introducir estimaciones iniciales de pKa / constantes.
7. Construir automáticamente el modelo químico.
8. Ajustar el modelo a los datos.
9. Mostrar parámetros, incertidumbres, residuales, especies y calidad del ajuste.

La interfaz debe ocultar complejidad matemática innecesaria en el modo normal, pero permitir acceso avanzado a los detalles del modelo.

## Fundamento matemático

HM fit representa los sistemas de equilibrio mediante balances de masa y una matriz estequiométrica.

Las concentraciones de especies se obtienen mediante la ley de acción de masas:

    c_species = beta * exp(M^T * ln(c_components))

Las constantes de equilibrio se manejan preferentemente en escala log10.

Las concentraciones libres se determinan resolviendo los balances de masa mediante métodos numéricos robustos, incluyendo Newton-Raphson amortiguado y métodos tipo Levenberg-Marquardt en espacio logarítmico.

La formulación debe preservar:

- positividad de concentraciones;
- conservación de masa;
- consistencia de carga cuando corresponda;
- estabilidad numérica;
- restricciones físicas de los parámetros.

Siempre que sea posible, HM fit separa:

- parámetros no lineales: constantes de equilibrio, pKa, constantes cinéticas, etc.;
- parámetros lineales de observación.

Los parámetros lineales deben eliminarse analíticamente mediante proyección variable cuando esto reduzca la dimensionalidad y mejore la estabilidad del ajuste.

## Representación química

En los sistemas ácido-base:

- H representa internamente H+.
- H tiene carga +1.
- `h_count` representa el número de protones asociados a una especie.
- `charge` representa su carga.
- `include` controla su participación en el modelo.
- `observable` indica si contribuye a la señal experimental.
- `fixed` permite fijar parámetros cuando corresponda.

El titulante es una entidad experimental separada y NO debe convertirse artificialmente en una especie o componente H.

La matriz química debe generarse a partir de la composición real de las especies, utilizando la representación interna de componentes y estequiometría.

## Teoría de grafos

Los sistemas químicos deben poder interpretarse como un grafo:

- nodos = especies químicas;
- aristas = transformaciones/equilibrios entre especies;
- atributos de nodo = composición, carga, protonación y propiedades observables;
- atributos de arista = constantes de equilibrio o parámetros de transformación.

Esta representación debe permitir comprobar conectividad, detectar especies redundantes o desconectadas y facilitar en el futuro la construcción automática de modelos a partir de redes químicas.

Para cinética, el mismo principio puede extenderse a un grafo dirigido de reacciones a partir del cual se construyen las ecuaciones diferenciales del mecanismo.

## Arquitectura

Mantener una separación estricta entre:

- `hmfit_core`: matemáticas, química, optimización y modelos;
- GUI PySide6: presentación e interacción con el usuario;
- módulos especializados como equilibrio, espectroscopía, NMR y cinética.

La GUI nunca debe convertirse en el lugar donde se implemente la lógica científica.

El core debe poder utilizarse y probarse independientemente de la interfaz gráfica.

## Filosofía de la interfaz

HM fit debe ser:

- científicamente correcto;
- sencillo para tareas comunes;
- potente para usuarios avanzados;
- reproducible;
- transparente respecto a los resultados del ajuste.

La interfaz debe priorizar el flujo experimental sobre la estructura interna del programa.

Debe existir una separación conceptual entre un modo básico y opciones avanzadas.

El usuario normal no debería necesitar editar matrices estequiométricas manualmente para realizar un ajuste ácido-base típico.

Las opciones avanzadas pueden exponer matriz/componentes, pKw, parámetros del electrodo, offsets, ponderaciones y parámetros internos.

## Reglas para modificar HM fit

Antes de realizar cambios importantes:

1. Entender el flujo científico afectado.
2. Revisar el código existente antes de reemplazarlo.
3. No duplicar algoritmos que ya existan en `hmfit_core`.
4. Mantener compatibilidad con los modelos existentes siempre que sea razonable.
5. No introducir comportamiento específico para un único dataset como solución general.
6. Mantener separada la lógica científica de la GUI.
7. Validar dimensiones, transformaciones y restricciones de parámetros.
8. Añadir o actualizar pruebas cuando se modifica comportamiento científico.
9. Corregir la causa raíz de los errores en lugar de añadir parches visuales.
10. No simplificar una formulación matemática si se pierde rigor químico.

## Objetivo final

HM fit debe permitir que un químico describa el sistema que realmente tiene en el laboratorio y que el programa construya, resuelva y ajuste automáticamente el modelo matemático correspondiente.

La complejidad debe estar en el motor de HM fit, no en el usuario.
