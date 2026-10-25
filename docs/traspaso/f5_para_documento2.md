# Fase 5 · Material para el Documento 2 (v0.7.5)

Lo escribió Claude Code en la nube el 25/10/2026 al cerrar la Fase 5 (tag `v1.0-rc`). Es el contenido para sumar al Documento 2 a partir de la v0.7.4: se copia `Fuentes de los Word\doc2_v074.js` como `doc2_v075.js`, se pasa esto al formato de cada sección y se regenera (ver el `LEEME.txt` de `Fuentes de los Word`). La numeración sigue la del Documento 2: decisiones desde la 98, prompts desde P32, figuras desde la 13 y tablas desde la 8. Lo que está entre corchetes lo completa Santiago.

## Ficha 3.9 · Fase 5 · Integración: agente y app

| Campo | Contenido |
|---|---|
| Fechas y hito | 25/10/2026 · tag `v1.0-rc` (commit `06912c0`) · hito "Integración e interfaz": [fecha en que el JTP lo anote] |
| Qué se hizo | Se integraron los dos carriles en el agente de la sección 2: la tabla de reglas condición–acción de la 2.5 como una máquina de estados pura (`src/agente/agente.py`), la percepción de la huella (H2 al cargarla, para mostrar el esqueleto; H3 y H4 al decir "Reconocer"), la captura de voz con el micrófono abierto todo el tiempo sobre un búfer circular y push-to-talk (250 ms antes de apretar, tope de 3 s y 150 ms después de soltar), las huellas que ofrece la app (las 6 reservadas para la demo, los 30 intrusos de test, variantes e intrusos nuevos generados en el momento y "Abrir archivo…"), el visor PCA 2D de entrenamiento y un controlador sin Tk que, después de cada evento, devuelve lo que hay que dibujar. La ventana de CustomTkinter solo dibuja eso. Se agregó la regla 10 (una huella que no se puede leer ni preparar) y un registro de cada sesión en CSV, sin audio. Mismo método que las etapas anteriores: prototipo, contrato con preguntas a Santiago, tests escritos en otra sesión solo a partir del contrato, implementación hasta pytest verde sin tocar los tests y verificación manual. |
| Cómo y con qué | Python 3.12, NumPy, CustomTkinter 6.0.0 (ventana), Matplotlib (PCA dentro de la ventana), sounddevice (micrófono, con una "fábrica" de stream que en los tests se reemplaza por una falsa), OpenCV (lectura de imágenes con rutas con tildes). Claves nuevas en `config.yaml`: `app.pre_roll_s` 0.25, `app.post_roll_s` 0.15, `app.bufer_s` 5.0, `app.medidor_ms` 50, `app.variante_nueva_desde` 1000, `app.intruso_nuevo_desde` 1000 y `metas.respuesta_max_s` 1.0. La app se abre con `venv\Scripts\python -m src.app [--dispositivo N] [--listar] [--sin-microfono]`. Contrato `docs/contratos/f5_integracion.md` versión 1.1; tests `tests/test_f5.py` (136 casos). |
| Conceptos de la materia | Agente reactivo basado en modelos: percepción, estado interno y acción (sección 2). Reglas condición–acción y máquina de estados (tabla 2.5 y regla 10). Arquitectura modular: el agente no sabe nada de la ventana ni del micrófono; la percepción se inyecta. PCA como visualización (los mismos ejes de la figura 7) y por qué el umbral no se dibuja en 2D. |
| Resultados y evidencias | En la notebook: `test_setup.py` y `test_f5.py`, 426 pasan; aceptación de F5, 10 de 10 (huellas reales, voz real, push-to-talk con audio real y tiempos de respuesta por debajo de la meta de 1 s). Verificación manual (25/10): tabla 8 y figuras 13 a 18 (`docs/figuras/app_1_autorizada.png` a `app_6_intruso.png`). Prueba con un familiar que nunca grabó (v04). Prototipo en `Prototipos\f5_2026-10-25`. |
| Problemas y decisiones | Decisiones 98 a 107. En Windows con escalado de pantalla, Matplotlib no seguía el escalado de CustomTkinter y el PCA salía con letra chica: se corrigió en la ventana (`06912c0`). En la prueba con v04, a veces se soltaba la barra espaciadora antes de terminar "reconocer" y el comando no se tomaba; con Santiago manteniendo el botón mientras ella hablaba, anduvo bien. |

## Decisiones (sección 5)

| N.° | Decisión | Por qué | Alternativas descartadas | Fecha |
|---|---|---|---|---|
| 98 | "Salir" cierra directo, sin confirmación: la regla 9 de la tabla 2.5 no cambia. | Es la tabla especificada en la Fase 0 y el enunciado pide que "Salir" concluya la ejecución. El riesgo conocido ("salida" tomada como "salir", 1 de 31 `otro` en la prueba en vivo de V4) se acepta: la app se vuelve a abrir en segundos. En la verificación, "salida" se rechazó. | Pedir "Salir" dos veces dentro de 5 s. | 25/10 |
| 99 | La app ofrece las 6 reservadas para la demo, un intruso de test por identidad intrusa (30), "Abrir archivo…", variantes nuevas de cada identidad y, como complemento, intrusos nuevos, generados en el momento con la receta de H1 desde la variante 1000 y el intruso 1000. | Ante el "mostrame otra" del profesor se puede mostrar una huella que nunca existió. Desde 1000 no se pisan el dataset (variantes 1–40, intrusos 1–60) ni la muestra ampliada del prototipo de H4 (41–140 y 61–160), con la que se eligió el criterio del umbral: ninguna huella nueva se usó para entrenar, calibrar ni elegir nada. | Solo la lista del dataset. | 25/10 |
| 100 | Un solo contrato para la Fase 5, con un controlador sin Tk: el agente, la captura, las huellas, el visor y el controlador se prueban con pytest y dobles de prueba; la ventana se verifica a mano. | Los escenarios del plan y el manejo de errores se prueban sin abrir ventanas ni micrófono; la ventana no tiene lógica propia. | Dos contratos (lógica y ventana por separado). | 25/10 |
| 101 | Regla 10: una huella que no se puede leer o que H2 rechaza (no es `uint8`, mide menos de 64 px o no tiene crestas) da el aviso "No se pudo leer la huella: elegí otra" y lleva a `ESPERANDO_MUESTRA`, descartando también la huella anterior y su resultado. | La tabla 2.5 no cubría un archivo dañado (pedido en "Robustez" del plan). Descartar la anterior mantiene la garantía de la regla 1: nunca se muestra la identidad de otra huella. | Dejar la huella anterior cargada. | 25/10 |
| 102 | H2 corre al cargar la huella (para mostrar el esqueleto) y "Reconocer" corre solo H3 y H4 sobre esa huella ya preparada; `evaluar_huella` da exactamente lo mismo que `reconocer` de H4 y además el vector `z` para el visor. | La huella se ve apenas se elige y "Reconocer" responde más rápido; los tests comparan contra `reconocer` de H4. | Correr H2 a H4 juntos al decir "Reconocer". | 25/10 |
| 103 | El micrófono queda abierto todo el tiempo sobre un búfer circular de 5 s; el clip va de 250 ms antes de apretar a 150 ms después de soltar, con un tope de 3 s de voz, en índices absolutos de muestras. | En la prueba en vivo de V4 todas las tomas arrancaban con unos 50 ms casi en silencio porque el micrófono se abría después del aviso; con el pre-roll, lo que se dijo justo antes de apretar ya está grabado (riesgo b de la revisión de Gemini, adoptado en el plan final de voz). | Abrir el micrófono al apretar. | 25/10 |
| 104 | El visor PCA muestra train proyectado en 2D (los ejes de la figura 7) con los centroides y la huella evaluada, sin círculos de umbral; la comparación con τ\* va en una barra aparte. | τ\* se aplica en el espacio completo con la distancia normalizada de H4 y la proyección achica las distancias: en la verificación, intrusos rechazados caen en 2D cerca del grupo de una identidad. | Dibujar círculos de radio τ\* en el PCA. | 25/10 |
| 105 | Meta de tiempo de respuesta (tabla REAS): menos de 1 s desde que se suelta el botón hasta la respuesta (`metas.respuesta_max_s`). | Con el post-roll (0,15 s), V2 a V4 y H3 + H4 queda con margen (en la nube, H2 tarda 339 ms y H3 + H4, 67 ms; en la notebook, unos 110 y 21 ms). La aceptación lo mide en la notebook. | — | 25/10 |
| 106 | Contrato 1.1 con las respuestas de Santiago a las cuatro preguntas del chat de tests: `huellas.identidades` es una lista de mapeos que se busca por `id`; en `metadatos.csv`, `intruso` es el número de intruso (0 en las identidades); en `FIN` la pantalla no muestra la identidad; el chat de tests puede probar su archivo contra una implementación descartable propia, que no se entrega. | La tercera era una contradicción real del contrato (4.6 contra 6.9). La cuarta repite lo que se le permitió al chat de tests de H4. | — | 25/10 |
| 107 | La app guarda un registro de cada sesión (`data/app/registro_<fecha>.csv`, fuera de Git) con cada acción y aviso, pero no guarda audio. | Sirve de evidencia de la demo y de la prueba en vivo sin guardar voces de terceros (los profesores o los familiares). | Guardar cada toma como en `scripts/probar_voz.py`. | 25/10 |

## Tabla 8 · Verificación manual de la app (25/10)

| Caso | Qué se hizo | Resultado |
|---|---|---|
| Reservadas | `id1_v040`, `id2_v039` e `id3_v039`: "Reconocer" e "Identificar" | AUTORIZADO e "Identidad Sintética #1", "#2" y "#3"; distancias 0,601, 1,035 y 0,991 contra τ\* = 1,561 (las de la sección 8 del contrato) |
| Variantes nuevas | `id1_v1000`, `id2_v1000`, `id3_v1000` e `id2_v1001` | AUTORIZADO con su identidad; distancias 0,722, 0,734, 0,659 y 0,711 (las mismas que el prototipo) |
| Intrusos nuevos | `intruso1000_v1` a `intruso1008_v1` | NO AUTORIZADO los 9; distancias de 1,858 a 7,233 (las mismas que el prototipo) |
| Intruso de test | `intruso31_v1`: "Reconocer" e "Identificar" | NO AUTORIZADO (1,841 contra τ\* = 1,561) y la regla 6 |
| Regla 6 | "Identificar" sobre un intruso rechazado | "Huella no autorizada: no se muestra ninguna identidad" |
| Regla 4 | "Identificar" antes de "Reconocer" | "Primero decí Reconocer" |
| Regla 2 | "Reconocer" e "Identificar" sin huella | "Primero elegí una huella" |
| Regla 8 | "hola", "salida", otras palabras y silencio | "No entendí, repetí el comando" (el K-NN votó `otro`, o el audio no fue válido para V2) |
| Regla 10 | "Abrir archivo…" con un archivo que no es una huella | "No se pudo leer la huella: elegí otra" |
| Pre-roll | Decir "salir" apretando justo al empezar a hablar | Se reconoció |
| Regla 9 | "Salir" | La app se cerró sin errores |
| Persona nueva | v04, familiar que nunca grabó | Anduvo bastante bien; cuando soltaba la barra espaciadora antes de terminar, "reconocer" no se tomaba; con Santiago manteniendo el botón, anduvo bien |
| Comandos aceptados | Los que se ven en la grabación y las capturas | Todos con confianza 1,00 y distancias de 0,37 a 0,61 contra τ\* = 0,774; ninguno confundido con otro comando |

## Figuras 13 a 18

- **Figura 13** (`docs/figuras/app_1_autorizada.png`): `id1_v040` después de "Reconocer": cartel AUTORIZADO, la estrella en el grupo de la identidad 1 y la barra con 0,601 contra τ\*.
- **Figura 14** (`app_2_identificada.png`): `id3_v039` después de "Identificar": "Identidad Sintética #3".
- **Figura 15** (`app_3_sin_huella.png`): "Reconocer" sin huella cargada (regla 2).
- **Figura 16** (`app_4_no_entendi.png`): una palabra que no es un comando (regla 8; el K-NN votó `otro`).
- **Figura 17** (`app_5_huella_ilegible.png`): un archivo que no es una huella (regla 10).
- **Figura 18** (`app_6_intruso.png`): `intruso31_v1` (un intruso de test) después de "Reconocer" e "Identificar": NO AUTORIZADO y el aviso de la regla 6, sin ninguna identidad. En el PCA 2D la estrella cae cerca del grupo de la identidad 3, pero en el espacio completo su distancia normalizada es 1,841, mayor que τ\* = 1,561: es el ejemplo de la decisión 104 (por qué no se dibujan círculos de umbral en la proyección).

## Bitácora (sección 6)

| Fecha | Qué se hizo |
|---|---|
| 25/10 | Fase 5 con Claude Code en la nube (rama `claude/gallant-allen-nnf0iy`, pasada a `main`). Prototipo de las huellas en la app con el dataset regenerado en la nube (90 variantes nuevas aceptadas, 30 intrusos nuevos rechazados). Preguntas a Santiago (decisiones 98 a 100) y contrato `f5_integracion.md` versión 1. Tests escritos por Claude Code en otra sesión, sin el repo: cuatro preguntas, contrato 1.1 y tests congelados (`ef5a41c`). Implementación (`35508c8`): pasan los 126 rápidos en la primera corrida; la ventana se probó en la nube con una pantalla virtual. En la notebook: 426 tests y la aceptación de 10 de 10. Ajuste de la ventana para el escalado de Windows (`06912c0`). Verificación manual con grabación de pantalla y capturas, y prueba con un familiar (v04). Tag `v1.0-rc`. |

## Uso de IA generativa (sección 7)

- **Herramientas:** Claude Code en la nube (claude.ai/code), modelo [completar], para el prototipo, el contrato, la implementación, la revisión de la verificación manual y este material; otra sesión de Claude Code, sin el repo, modelo [completar], para los tests.
- **Cómo se verificó:** el contrato se armó con valores medidos en el prototipo; los tests los escribió otra sesión solo con el contrato y los probó contra una implementación descartable propia con 65 errores plantados (detectó 64); la implementación pasó los tests congelados sin tocarlos; la ventana se revisó con capturas en una pantalla virtual y después en la notebook; la verificación manual (grabación de pantalla y capturas) se revisó cuadro por cuadro contra los valores del prototipo.
- **P32** (arranque de la Fase 5, Claude Code en la nube): «Seguimos con el TF de IA: Fase 5 (integración, agente y app). Leé docs/traspaso/tf-ia-estado.md del repo y AGENTS.md, mirá git log y git status, y arrancamos con lo que dice "Siguiente → Fase 5".»
- **P33** (tests de la Fase 5): el bloque de texto de `docs/contratos/f5_prompt_tests.md`, con `f5_integracion.md` adjunto.

## Mapa de conceptos (sección 4)

- Agente reactivo basado en modelos, estado interno, reglas condición–acción y máquina de estados: explicados en la ficha 3.9 (antes, solo en la sección 2).
- Arquitectura modular del sistema: ficha 3.9.
