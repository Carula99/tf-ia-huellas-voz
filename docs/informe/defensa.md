# Defensa: lista de la entrega, ensayos de la demo y coloquio

Fase 6 del Plan (sección 6.11). Lo escribió Claude Code en la nube el 25/10/2026 con el estado de `main` en `f09fd73`. Las distancias de las huellas se midieron en la nube con el dataset y el modelo regenerados con las semillas de `config.yaml`: son las mismas que en la notebook (las tres de la verificación manual coinciden).

## 1. Lista de la entrega (sección 2 del Plan)

### 1.1 Del enunciado

| Requisito | Dónde está | Estado |
|---|---|---|
| Trabajo y defensa individuales; sin plagio | El uso de IA está declarado entero en la sección 9 del informe y en el Documento 2 (decisiones, bitácora y prompts P1 a P33) | Listo |
| 3 identidades autorizadas sintéticas con un modelo generativo | H1: `src/huellas/generador.py`, tag `v0.1.2-huellas-datos`; informe 6.1 y figura 10 | Listo |
| Variantes: traslación, rotación, ruido gaussiano y sal y pimienta, presión y corte parcial | `src/huellas/variantes.py`; informe 6.1 (rangos) | Listo |
| Muestras de validación y lote de intrusos | 114 de val y 114 de test, con 30 identidades intrusas distintas en cada una; informe tabla 6 | Listo |
| Preprocesamiento: ROI, contraste, filtrado, binarización y esqueleto | H2: máscara y ventana de 250 × 250, CLAHE, mediana y Gabor, signo y esqueleto de Lee; informe 3.1 y figura 3 | Listo |
| Vector de características (orientación, textura, minucias) | H3: 374 columnas; informe 3.2 | Listo |
| K-means con k = 3 | H4: `src/core_ml/kmeans.py`, propio; informe 3.3 | Listo |
| Métrica de distancia y umbral justificados experimentalmente | Distancia normalizada y τ\* = 1,561 en la mitad del intervalo sin errores de val; muestra ampliada de 600 huellas nuevas; informe 3.3, 6.4 y figura 12 | Listo |
| K-NN sobre MFCC y energía para «Reconocer», «Identificar» y «Salir» | V3 y V4: `src/core_ml/knn.py`, propio; informe 3.5, 3.6 y 6.6 | Listo (metas de voz en test no cumplidas: ver 6.6) |
| Investigación de las características que separan bien las clases | Tabla 1 del Documento 2 (bloques de H3 y Hu descartado), tabla 5 (pesos de V3), validación cruzada de V4; informe 3.2 y 6.5 | Listo |
| Informe PDF con título exacto, resumen ≤ 200 palabras y las secciones pedidas | `docs/informe/informe.md` (borrador, 27 páginas; resumen de 187 palabras) | Falta completar corchetes, leerlo entero y exportar el PDF final |
| Demostración completa del agente | Sección 2 de este documento | Faltan los 3 ensayos |

### 1.2 De las clases

| Requisito | Dónde está | Estado |
|---|---|---|
| Hitos mostrados en clase y anotados por el JTP | Planilla del JTP | **Pendiente:** H1 a H4, V1 a V4 y la Fase 5 todavía no se mostraron (traspaso, "Pendiente de Santiago") |
| Los profesores dicen los comandos con su voz | K-NN probado con 5 familiares que nunca grabaron (v01 a v05; v05 el 26/10 con la app final, 11 de 11 comandos) | Listo; riesgo en sección 2.4 |
| Semillas y muestras reservadas («mostrame una de la Identidad 1») | Semillas 101, 202 y 303; variantes 39 y 40 de test reservadas; variantes e intrusos nuevos desde 1000 en la app | Listo |
| Variantes por código, no por prompt de imagen | H1: todas las variantes salen de la base por transformaciones | Listo |
| Cero fuga de datos | Intrusos de val y de test distintos; voz partida por persona; test mirado una sola vez en H4 y V4; las reservadas no calibran; huellas nuevas desde 1000 | Listo |
| PCA con la separación de clases | Figuras 11 (huellas) y 13 (voz) del informe | Listo |
| Tests congelados antes del código, escritos en otro chat | 10 archivos, 1797 tests (1722 rápidos y 75 de aceptación) | Listo |
| Nada de huellas reales | Todo sintético | Listo |
| Licencias y librerías citadas | Bibliografía del informe (sección 8) | Listo; revisar las referencias nuevas (ver `LEEME.md`) |
| README, `.gitignore`, tags y arquitectura modular | Repo; tags `v0.0-setup` a `v1.0-rc` | Listo; falta el tag `v1.0` |
| Interfaz cuidada e informe sin relleno | App de la Fase 5; informe | Falta la lectura completa del informe |
| Después de presentar: repo público, sin audios de terceros | `data/` y `models/` no están en Git; los nombres solo por código | Al final |

## 2. Demo

### 2.1 Antes de empezar (en la notebook)

1. No tocar la configuración de sonido de Windows (supresión de ruido y control automático de volumen activos, como en todas las grabaciones).
2. Cargador enchufado; cerrar lo pesado (OneDrive sincronizando, navegador con muchas pestañas).
3. `git status` limpio en `main` y los modelos en su lugar: `models/modelo_huellas.npz` y `models/modelo_voz.npz`.
4. Prueba corta: `venv\Scripts\python -m pytest tests\test_setup.py tests\test_f5.py -q` (426 pasan).
5. `venv\Scripts\python -m src.app --listar` y, si el micrófono de la notebook no es el predeterminado, abrir con `--dispositivo N`.
6. Tener a mano una imagen que no sea una huella (para la regla 10) y, en una carpeta, tres WAV de Santiago diciendo «reconocer», «identificar» y «salir» (para el plan B; de `data/voz/`, de p01).

### 2.2 Guion (unos 5 minutos)

| Paso | Qué se hace | Qué tiene que pasar |
|---|---|---|
| 1 | Abrir la app y mostrar los tres paneles | Estado ESPERANDO_MUESTRA |
| 2 | Decir «Reconocer» sin huella | «Primero elegí una huella» (regla 2) |
| 3 | Elegir `id1_v040` (reservada) | Cruda y esqueleto; HUELLA_CARGADA |
| 4 | Decir «Identificar» | «Primero decí Reconocer» (regla 4) |
| 5 | Decir «Reconocer» | AUTORIZADO, distancia 0,601 contra τ\* = 1,561 (regla 3) |
| 6 | Decir «Identificar» | «Identidad Sintética #1» (regla 5) |
| 7 | Repetir 3 a 6 con `id3_v039` (0,991) y `id2_v039` (1,035) | AUTORIZADO e «Identidad Sintética #3» y «#2» |
| 8 | Elegir `intruso31_v1` y decir «Reconocer» e «Identificar» | NO AUTORIZADO (1,841) y «Huella no autorizada: no se muestra ninguna identidad» (regla 6). Explicar que en el PCA 2D cae cerca de la identidad 3 y aun así se rechaza: τ\* vale en las 374 dimensiones |
| 9 | «Mostrame otra»: «Variante nueva» de la identidad que pidan, e «Intruso nuevo» | La variante, AUTORIZADO con su identidad (en la verificación, 0,66 a 0,73); el intruso, NO AUTORIZADO (1,86 a 7,23) |
| 10 | Decir «hola» o «salida» | «No entendí, repetí el comando» (regla 8) |
| 11 | «Abrir archivo…» con la imagen que no es una huella | «No se pudo leer la huella: elegí otra» (regla 10) |
| 12 | Los profesores dicen los comandos (ver 2.3) | Igual que los pasos 5 y 6 |
| 13 | Decir «Salir» | La app se cierra (regla 9) |

Reservadas: con τ\* = 1,561, las 6 se aceptan, pero `id2_v040` (1,518), `id3_v040` (1,487) e `id1_v039` (1,287) quedan con menos margen: para mostrar, `id1_v040`, `id3_v039` e `id2_v039`. Intrusos de test (variante 1): ninguno se acepta; los más cercanos son `intruso49_v1` (1,731, presilla), `intruso31_v1` (1,841, arco en tienda) e `intruso58_v1` (1,877, verticilo), y los más lejanos pasan de 6.

### 2.3 Cuando hablan los profesores

- **El operador maneja el botón.** Se aprieta antes de que la persona empiece a hablar y se suelta después de que termina (con v04, cuando ella soltaba la barra antes de terminar, «reconocer» no se tomaba). El pre-roll guarda 250 ms antes de apretar y el post-roll, 150 ms después de soltar; el tope es de 3 s.
- Pedir que digan la palabra sola, con voz normal y a unos 30–50 cm de la notebook.
- Si no se toma: repetir una vez, un poco más cerca. Es un rechazo («no entendí»), no un error del agente: en test, ningún comando se confundió con otro.

### 2.4 Planes B

| Si pasa esto | Se hace esto |
|---|---|
| El micrófono no abre o no hay nivel | Cerrar y abrir con `--dispositivo N` (`--listar` para ver cuál); si no, `--sin-microfono` y dar los comandos con «Abrir audio…» y los WAV de Santiago |
| Un comando se rechaza dos veces seguidas | Lo dice Santiago (está en train: 18 de 18 en la prueba en vivo) y se sigue |
| «salida» u otra palabra cierra la app (riesgo conocido, decisión 98) | Volver a abrirla (tarda unos segundos) y explicar por qué se aceptó ese riesgo |
| La app se cuelga | Cerrar la ventana y volver a abrir; el registro de la sesión queda en `data/app/` |
| Preguntan por una huella con poco margen (`id2_v040`) | Mostrarla: se acepta (1,518 contra 1,561) y sirve para explicar el umbral |

### 2.5 Ensayos (3 seguidos, con ruido de fondo)

Cada ensayo es el guion completo de 2.2 sin cortes. Vale si no hay ningún cierre inesperado ni un paso que no se pueda resolver con 2.4.

| Ensayo | Fecha | Lugar y ruido | Quiénes dijeron los comandos | Lo que falló | ¿Vale? |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

Con los 3 completos: la fila "Demo sin cierres inesperados" de la tabla 9 del informe, la ficha 3.10 del Documento 2 y el tag `v1.0`.

## 3. Preguntas probables del coloquio

**Agente**

- *¿Por qué un agente reactivo basado en modelos?* Porque la respuesta a «Identificar» depende de lo que dio el último «Reconocer» de la huella cargada: hace falta un estado interno. No planifica (no es basado en objetivos) ni aprende durante la operación.
- *¿Por qué el entorno es episódico si «Identificar» depende de «Reconocer»?* El episodio es una huella: lo que se decide con una no cambia la siguiente. La dependencia corta dentro del episodio la resuelve el estado interno.
- *¿Qué pasa si cargo otra huella después de una autorizada y digo «Identificar»?* La regla 1 borra el resultado anterior: responde «Primero decí Reconocer». Nunca muestra la identidad de otra huella.

**Huellas**

- *¿Cómo generaste las huellas?* Campo de orientación con el modelo de Sherlock y Monro (núcleos y deltas según el tipo) y crestas por filtrado de Gabor iterativo desde ruido, como SFinGe. Una semilla por identidad; las variantes salen por código de la misma base.
- *¿Por qué no pedirle a una IA otra foto de la misma huella?* Cambiaría las minucias: sería otra huella.
- *¿Para qué alinear en H2?* Para que dos capturas del mismo dedo queden superpuestas y las características por celda o por sector comparen la misma zona. El punto de referencia es el núcleo y el ángulo, θ∞ del ajuste de Sherlock y Monro.
- *¿Qué es el índice de Poincaré?* La suma de las variaciones de la orientación en un círculo, dividida por 2π: +½ en un núcleo, −½ en un delta y 0 en el resto.
- *¿Por qué esas características y no los momentos de Hu?* Se midió la separación (silhouette) de cada bloque: orientación, Gabor y minucias separan; Hu separaba poco (0,13) y empeoraba el total.
- *¿Por qué dividir cada bloque por √d_b?* Para que el bloque de Gabor (300 columnas) no domine la distancia sobre la orientación (50) y las minucias (24).
- *¿Cómo funciona tu K-means?* k-means++ para los centroides iniciales, después Lloyd (asignar al más cercano, mover a la media) hasta que no se mueven; 15 arranques y el de menor inercia. Es no supervisado: las identidades solo nombran los grupos al final (pureza 100 %).
- *¿Cómo justificaste el umbral?* Curvas FAR y FRR en validación: hay un intervalo sin errores y τ\* va en su mitad (1,561). Con 600 huellas nuevas del generador se simularon 2000 pares val/test: con la distancia normalizada y el umbral en la mitad, se cumplen las metas en test el 99,6 % de las veces.
- *¿Usar esas 600 huellas no es fuga de datos?* No: no están en el dataset y no se usaron para entrenar ni para fijar τ\*; solo para elegir el criterio, sin mirar test.
- *¿Por qué hay un FRR de 4,2 %?* Es 1 de 24: `id1_v036`, con un corte del 9 % y la escala reducida un 7 %, queda por encima de τ\*.
- *¿Por qué en el PCA un intruso cae al lado de la identidad 3 y se rechaza?* El PCA 2D muestra el 85 % de la varianza; la decisión se toma con las 374 columnas y la proyección achica las distancias.

**Voz**

- *¿Qué son los MFCC?* El espectro de cada trama de 25 ms pasado por 40 filtros en escala mel (parecida a la del oído), el logaritmo de esas energías y la transformada coseno; se quedan los 13 primeros. La resta de la media (CMN) saca el efecto del micrófono.
- *¿Por qué 5 tramos con media y desvío?* Para tener un vector de largo fijo que conserve el orden temporal de la palabra; el K-NN necesita vectores del mismo largo.
- *¿Cómo funciona tu K-NN?* Busca los 5 vecinos más cercanos (distancia euclídea, después de PCA a 20 y normalizar cada fila), cada uno vota por su clase con peso 1/(d + ε) y gana la de más voto.
- *¿Cómo elegiste K y el PCA?* Validación cruzada dejando una persona afuera en train: siempre se evalúa con una voz que el K-NN no vio, como pasará con los profesores.
- *¿Cómo rechazás lo que no es un comando?* Hay una clase «otro» con palabras parecidas, toses y ruido, y además se exige confianza ≥ 0,60 y distancia ≤ τ\* = 0,774 (calibrado en validación).
- *¿Por qué no se cumplen las metas de voz en test?* Con 4 personas en validación y 4 en test, el umbral calibrado con unas no se trasladó bien a las otras. Sin rechazo, el K-NN elige bien 70 de 71 comandos: los errores son de rechazo, no de confusión. En vivo, con la app, 3 personas nuevas llegaron al 91,2 % y al 3,2 %. Mejorar: más tomas reales de «otro» y más personas (V4.1).
- *¿Por qué no recalibraste τ\* con test?* Sería fuga de datos: test se mira una sola vez y se reporta.
- *¿Y si digo «salida»?* Puede tomarlo como «salir» (pasó 1 vez en 31 en la prueba en vivo). Se decidió no pedir confirmación (decisión 98): el costo es reabrir la app.
- *¿Para qué el pre-roll?* Para no cortar el principio de la palabra si se aprieta el botón tarde: el micrófono está siempre abierto y se guardan los 250 ms anteriores.

**Código e IA**

- *¿Qué hiciste vos y qué la IA?* La IA escribió la mayor parte del código; las decisiones (algoritmos, umbrales, qué hacer cuando una meta no se cumplía) las tomó Santiago con alternativas medidas, y cada etapa se verificó con tests escritos por otra sesión que solo veía el contrato, con scikit-learn como oráculo y con verificación manual.
- *¿Cómo sabés que el PCA, el K-means y el K-NN están bien?* Los tests los comparan con scikit-learn: el PCA coincide hasta 1e-11 y el K-means hasta 1e-14.
- *¿Qué tan rápido es?* Una huella, unos 0,13 s (H2 al cargarla y H3 + H4 al reconocer); un comando, menos de 1 s desde que se suelta el botón.
