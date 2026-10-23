// Documento 1 · Plan de desarrollo v0.3 (parte de la v0.2 + correcciones de la revisión)
const fs = require("fs");
const { Packer } = require("docx");
const L = require("./lib");
const { p, h1, h2, label, bullets, checks, callout, code, table, image, caption, titleBlock, makeDoc } = L;

const S = __dirname;
const MMD = `${S}/mmd`;

function etapa(titulo, { objetivo, pasos, tests, entregable, testLabel = "Test antes del código", extra = [] }) {
  return [
    h2(titulo),
    label("Objetivo"), p(objetivo),
    label("Pasos"), ...bullets(pasos),
    ...extra,
    label(testLabel), ...checks(tests),
    label("Entregable"), p(entregable, { after: 200 }),
  ];
}

const children = [
  ...titleBlock({
    kicker: "TRABAJO FINAL · INTELIGENCIA ARTIFICIAL I · 2026",
    title: "Plan de desarrollo",
    subtitle: "Sistema de reconocimiento de huellas dactilares controlado por reconocimiento de voz",
    meta: [
      "Santiago · Ingeniería en Mecatrónica · Facultad de Ingeniería, UNCuyo",
      "Versión 0.3.2 · 16 de octubre de 2026 · documento vivo: se actualiza a medida que avanza el proyecto",
    ],
  }),
  ...callout([
    "**Cómo usar este documento y qué cambió**",
    "Acá está **qué hay que hacer, en qué orden y con qué criterios de aceptación** (tests antes del código). Lo que efectivamente se hizo, las decisiones y los prompts van en el Documento 2 (Documentación del desarrollo).",
    "**v0.2:** PCA, K-means y K-NN propios en NumPy con scikit-learn como oráculo en los tests; alineación por núcleo; ponderación por bloques 1/√d_b; push-to-talk y K-NN ponderado.",
    "**v0.3:** regla de alineación para cada tipo de huella e identidades sin ambigüedad; tests contra scikit-learn con las mismas convenciones; “dejar una persona afuera” programado en src/; 60 intrusos para que FAR y FRR sean medibles; ajustes de voz (banda de los MFCC, PCA antes del K-NN, grabar mientras se mantiene apretado); lo opcional marcado como tal; checklist, reglas y convenciones recuperados; figuras generadas desde el código del Anexo A.",
    "**v0.3.1:** un único estimador de ángulo para enderezar (anillo que encierra los deltas); posición real del núcleo calculada con la misma función que deforma la imagen; bloque prosódico en V3 (ningún bloque de menos de 5 columnas); tests de H2 medidos en grados y píxeles en lugar de correlación de píxeles.",
    "**v0.3.2:** Fase 0 cerrada (tag `v0.0-setup`): Python 3.12; cada etapa arranca con un contrato en `docs/contratos/`, commiteado antes de los tests; `src/config.py` y `pyproject.toml` en la estructura; ffmpeg solo si llega un audio en otro formato.",
  ]),

  // 1. OBJETIVOS
  h1("1. Objetivos"),
  h2("1.1 Objetivo general"),
  p("Desarrollar en Python un agente de control de acceso con arquitectura modular e interfaz gráfica que, comandado por voz (“Reconocer”, “Identificar” y “Salir”, clasificados con K-NN), determine si una huella dactilar sintética pertenece a una de tres identidades autorizadas o a un intruso y, en el primer caso, la identifique (K-means con k = 3, umbral de distancia calibrado experimentalmente y análisis PCA). Tiene que funcionar en vivo con voces de personas que no participaron del entrenamiento, con cada etapa verificada por tests escritos antes del código y todo documentado en el informe final."),
  h2("1.2 Objetivos específicos"),
  table(["Código", "Objetivo", "Se cumple cuando…"], [
    ["**OE1** Datos de huellas", "Generar con un generador propio 3 identidades sintéticas de tipos distintos (presilla, verticilo y arco en tienda) y 60 intrusos, reproducibles por semilla, con 40 variantes por identidad autorizada.", "Pasa el test de H1."],
    ["**OE2** Preprocesamiento de huellas", "Llevar cualquier huella al mismo formato (250 × 250 px, CLAHE, Gabor, esqueleto de 1 px), alineada respecto de su punto de referencia para resistir rotaciones y cortes parciales.", "Pasa el test de H2."],
    ["**OE3** Características y PCA", "Representar cada huella con un vector por bloques (orientación, Gabor, minucias, Hu) ponderado por 1/√d_b y proyectarlo con un PCA propio en NumPy.", "Pasa el test de H3 (silhouette ≥ 0,50 y 3 grupos separados)."],
    ["**OE4** Clasificación de huellas", "Agrupar con un K-means propio (k = 3), validado contra scikit-learn, y decidir el acceso con un umbral calibrado en validación con la curva FAR–FRR.", "Pasa el test de H4 (metas de identificación, FAR y FRR en test)."],
    ["**OE5** Datos de voz", "Reunir los 3 comandos dichos por al menos 10 personas reales, más voces TTS y la clase “otro”, separados por persona.", "Pasa el test de V1."],
    ["**OE6** Reconocimiento de voz", "Clasificar los comandos con un K-NN propio ponderado por distancia sobre vectores por tramos con CMN, rechazando lo que no es un comando y generalizando a voces nuevas.", "Pasa el test de V4 (validación dejando una persona afuera)."],
    ["**OE7** Integración e interfaz", "Integrar el agente como máquina de estados en una app CustomTkinter con push-to-talk, visor de huella y PCA, y manejo de errores.", "Pasa los escenarios de la Fase 5."],
    ["**OE8** Documentación y entrega", "Informe PDF con las secciones de la consigna, bitácora de IA verificada y todos los hitos aprobados en la planilla del JTP.", "Checklist de la sección 2 completo y 3 demos seguidas sin fallas."],
  ], [2200, 4870, 2000]),
  p("", { after: 60 }),
  h2("1.3 Metas de desempeño (propuesta inicial)"),
  p("Permiten decidir “cuándo está bien” cada etapa y frenar a tiempo, como recomendó el ayudante. Si una meta resulta irreal o demasiado holgada, se ajusta y se registra el motivo en el Documento 2.", { keepNext: true }),
  table(["Componente", "Métrica", "Meta"], [
    ["Huellas · identificación", "Huellas autorizadas de test asignadas a la identidad correcta", "≥ 95 %"],
    ["Huellas · intrusos", "FAR (tasa de falsa aceptación): intrusos de test aceptados", "≤ 5 %"],
    ["Huellas · autorizados", "FRR (tasa de falso rechazo): autorizados de test rechazados", "≤ 10 %"],
    ["Huellas · separación", "Silhouette de los vectores de train con las 3 identidades", "≥ 0,50"],
    ["Voz · voces nuevas", "Acierto con personas de test que no están en el entrenamiento", "≥ 90 %"],
    ["Voz · rechazo", "Ruido u otras palabras tomados como comando válido", "≤ 5 %"],
    ["Demo", "Recorridos completos en vivo sin fallas ni cierres inesperados", "3 de 3"],
  ], [2300, 5370, 1400]),
  p("Con 90 muestras de intrusos en test, cada intruso aceptado suma 1,1 % de FAR; con 24 autorizados, cada rechazo suma 4,2 % de FRR.", { before: 80, run: { size: 19, color: "5F6B7A" } }),

  // 2. REQUISITOS
  h1("2. Requisitos de la cátedra"),
  h2("2.1 Del enunciado"),
  ...checks([
    "**Modalidad:** trabajo y defensa individuales; el plagio implica no aprobar la materia.",
    "**Datos sintéticos de huellas:** 3 identidades autorizadas generadas con modelos generativos; variantes (traslación, rotación, ruido gaussiano y sal y pimienta, presión, corte parcial); muestras de validación y un lote de intrusos.",
    "**Preprocesamiento:** ROI, normalización de contraste, filtrado espacial (Gabor o mediana), binarización y/o esqueletizado.",
    "**Extracción y agrupamiento:** vector de características (orientación de crestas, textura, momentos o densidad de minucias); K-means con k = 3; métrica de distancia y umbral de aceptación justificados experimentalmente.",
    "**Control por voz:** K-NN sobre características acústicas (MFCC, energía o espectrogramas) para “Reconocer” (AUTORIZADO / NO AUTORIZADO), “Identificar” (si fue autorizada, la identidad, ej. “Identidad Sintética #1”) y “Salir” (termina la ejecución).",
    "**Investigación:** buscar los algoritmos de extracción de características que logren una buena separación de clases.",
    "**Informe PDF** entregado el día de la presentación, con título exacto, resumen (máximo 200 palabras), introducción, especificación del agente (tipo, tabla REAS, propiedades del entorno), diseño del agente, código y uso de IA, ejemplo de aplicación, resultados (datos, generación sintética, estadísticas, PCA con separación de clases), conclusiones, bibliografía e IA generativa (prompts de ejemplo y cómo se verificaron).",
    "**Evaluación:** informe + programa + coloquio; se evalúa la eficiencia en el reconocimiento de imágenes y de voz, con demostración completa del agente.",
  ]),
  p("Título exacto del informe: *Trabajo Final Inteligencia Artificial I – año 2026: Sistema de reconocimiento de huellas dactilares controlado por reconocimiento de voz.*", { before: 80 }),
  h2("2.2 De las clases (presentación del TF y clase del 15/10)"),
  ...checks([
    "**Promoción por planilla de hitos:** cada etapa terminada se muestra en clase y el JTP la anota. Con todos los hitos, en la semana del 10 de noviembre se muestra el programa funcionando y se promociona.",
    "**Prueba en vivo:** los profesores van a decir los comandos con su propia voz; el K-NN tiene que funcionar con cualquier hablante.",
    "**Semillas y muestras reservadas:** fijar semillas asociadas a un nombre o ID y guardar muestras que nunca entren al entrenamiento (te van a pedir “mostrame una de la Identidad 1”).",
    "**Variantes por código, no por prompt de imagen:** pedirle a una IA otra toma de la misma huella cambia las minucias; se genera un patrón base por identidad y las variantes se aplican por código.",
    "**Cero fuga de datos:** evaluar con datos usados para entrenar o calibrar es un “error grave”.",
    "**El PCA es decisivo:** la separación de las clases en el PCA dice si las características sirven o si hay que volver al preprocesamiento.",
    "**Tests congelados antes del código:** escritos en otro chat o con otra IA, sin modificarlos después.",
    "**Nada de huellas reales:** como extra se puede probar con la propia, sin entregarla.",
    "**Licencias:** citar las librerías y bases externas (ej. FVC2002 si se usa el plan B) y respetar sus licencias.",
    "**Documentación y arquitectura:** documentar desde el día 1; arquitectura modular; repo con README, .gitignore y tags.",
    "**Interfaz cuidada e informe limpio:** la usabilidad suma puntos y el informe se lee completo para sacar el relleno de IA.",
  ]),

  // 3. DECISIONES
  h1("3. Decisiones de base (actualizadas en v0.3)", { pageBreak: true }),
  p("El porqué y las alternativas descartadas de cada decisión están en el Registro de decisiones del Documento 2.", { keepNext: true }),
  table(["Tema", "Decisión", "Fecha"], [
    ["Alcance", "Proyecto completo, modular y defendible; lo opcional queda marcado para no comprometer la fecha.", "16/10"],
    ["Algoritmos centrales", "PCA, K-means y K-NN propios en NumPy (`src/core_ml/`), igual que la validación “dejando una persona afuera”. scikit-learn solo en `tests/` como oráculo.", "16/10"],
    ["Generador de huellas", "Propio, estilo SFinGe: campo de orientación con núcleos y deltas + filtrado de Gabor iterativo + distorsión elástica. Opcional: fase espiral. Plan B: base sintética FVC2002 DB4.", "16/10"],
    ["Identidades", "Identidad 1 = presilla · Identidad 2 = verticilo · Identidad 3 = arco en tienda.", "16/10"],
    ["Dataset de huellas", "3 × 40 variantes (24 train / 8 val / 8 test) + 60 intrusos × 3 variantes (30 en val, 30 en test).", "16/10"],
    ["Alineación y características", "Centro: punto de referencia (1 núcleo → ese; 2 núcleos → punto medio; ninguno → máxima curvatura). Ángulo: un único estimador para todas las muestras (orientación media en un anillo que encierra los deltas). Vector por bloques ponderado por 1/√d_b, sin bloques de menos de 5 columnas.", "16/10"],
    ["Dataset de voz", "WhatsApp + micrófono de la demo (mínimo 10 personas) + edge-tts + clase “otro” con palabras parecidas a los comandos.", "15/10"],
    ["Captura y clasificación de voz", "K-NN ponderado 1/(d+ε) con rechazo; push-to-talk manteniendo apretado, con tope de 3 s.", "16/10"],
    ["Tests", "Criterios en este plan; tests escritos en otro chat y congelados en Git antes del código.", "15/10"],
    ["Entorno y repositorio", "Windows + VS Code con Python 3.12 en un venv; módulos .py y notebooks para explorar; repo local fuera de OneDrive y GitHub privado con tags por hito, público después de presentar.", "16/10"],
  ], [2000, 6070, 1000], { vcenter: true }),

  // 4. MAPA
  h1("4. Mapa del proyecto", { pageBreak: true }),
  p("Los carriles de huellas (H1–H4) y voz (V1–V4) avanzan en paralelo después del setup: mientras llegan los audios por WhatsApp se construye el carril de huellas. Los dos usan los algoritmos propios de `src/core_ml/` (PCA, K-means, K-NN). Cada etapa sigue el ciclo de la sección 5.", { after: 60 }),
  image(`${MMD}/flujo_v3.png`, 13.2),
  caption("Figura 1. Flujo general del proyecto (v0.3.1)."),

  // 5. METODOLOGÍA
  h1("5. Cómo se trabaja cada etapa", { pageBreak: true }),
  p("Es el ciclo que propuso el ayudante (investigar → planificar → hacer → probar y documentar → verificar), con los tests escritos **antes** del código:", { after: 60 }),
  image(`${MMD}/ciclo_v3.png`, 16),
  caption("Figura 2. Ciclo de trabajo de cada etapa."),
  ...bullets([
    "**1 · Investigar:** qué técnicas se usan en la etapa, con ventajas, limitaciones y fuentes.",
    "**2 · Definir entradas, salidas y criterios:** un contrato en `docs/contratos/` con las firmas exactas de las funciones (tipos, dimensiones de los arrays, qué devuelven) y los criterios de aceptación de este plan. Se commitea antes de los tests.",
    "**3 · Escribir los tests en otro chat y congelarlos:** `tests/test_<etapa>.py` en una sesión limpia que solo ve el contrato, usando scikit-learn como oráculo cuando corresponda, y commit antes de programar. Correrlos antes de implementar: tienen que fallar.",
    "**4 · Implementar en src/** hasta que todos los tests pasen.",
    "**5 · Verificar sin trampas:** si un test falla se corrige el código, nunca el test. Si un criterio estaba mal planteado, se cambia de forma explícita y se anota en el Documento 2.",
    "**6 · Tag, bitácora y hito:** tag en Git, resultados y prompts en el Documento 2 y mostrar el avance en clase.",
  ]),
  h2("Prompt modelo para el chat de los tests"),
  p("Se pega en un chat nuevo, seguido del contrato completo de la etapa. Es la versión que se usó en la Fase 0 (`docs/contratos/fase0_prompt_tests.md`).", { keepNext: true }),
  ...code([
    "Actuá como un ingeniero de testing en Python. Voy a implementar la etapa",
    "[nombre] del trabajo final de IA I (Python 3.12, Windows, VS Code). Abajo va",
    "el contrato completo: firmas exactas, tipos y criterios de aceptación.",
    "Escribí un único archivo, tests/test_<etapa>.py, con pytest. Reglas:",
    "- No implementes nada de src/ ni propongas cómo hacerlo: solo los tests.",
    "- Raíz del repo: Path(__file__).resolve().parents[1].",
    "- Importá src dentro de cada test: si algo no existe, que falle ese test.",
    "- Archivos de texto en UTF-8; lo que escriba archivos va en tmp_path.",
    "- Datos chicos generados dentro del test y, cuando aplique, comparación",
    "  contra scikit-learn con la misma convención (mismos pesos, misma",
    "  inicialización, PCA salvo el signo).",
    "- Un test por criterio, con nombres en español y mensajes claros.",
    "- Si algo del contrato es ambiguo, preguntame antes de escribir.",
    "=== CONTRATO === [contenido de docs/contratos/<etapa>.md]",
  ], { size: 18 }),
  h2("Reglas generales"),
  ...bullets([
    "Nunca evaluar con datos usados para entrenar o para calibrar (umbrales, K, escaladores, pesos de bloques).",
    "Todo lo aleatorio lleva semilla, guardada en `config.yaml` o en el CSV de metadatos.",
    "Dos tipos de tests: **unitarios** (rápidos, con datos chicos) y **de aceptación** (metas de la sección 1.3 sobre el dataset completo, marcados con `@pytest.mark.aceptacion`).",
    "Las comparaciones contra scikit-learn usan la misma convención: K-NN con `weights=lambda d: 1/(d+eps)`; K-means con la misma inicialización (`init=centroides, n_init=1`) o con los centroides emparejados; PCA salvo el signo de cada componente.",
    "Un tag por hito (ej. `v0.1-huellas-datos`), como figura en cada etapa.",
    "A la IA se le piden tareas puntuales y de a una; revisar que no agregue lo que no se pidió.",
    "Cada sesión de trabajo deja una entrada en la bitácora del Documento 2, con los prompts relevantes.",
  ]),

  // 6. ETAPAS
  h1("6. Etapas"),
  ...etapa("6.1 Fase 0 · Setup y especificación del agente", {
    objetivo: "Dejar listos el entorno en Windows + VS Code, el repo privado y la especificación del agente, y mandar el pedido de audios por WhatsApp para que avance en paralelo.",
    pasos: [
      "Completar en el Documento 2 la especificación del agente: **agente reactivo basado en modelos** (estado interno: la última huella evaluada y su resultado), tabla REAS, propiedades del entorno (parcialmente observable, estocástico, episódico por huella, semidinámico, percepciones continuas y acciones discretas, agente individual) y reglas condición–acción.",
      "Escribir el contrato de la fase (`docs/contratos/fase0_setup.md`) y commitearlo antes que los tests.",
      "Inicializar el repo con README, .gitignore (`venv/`, `__pycache__/` y el contenido de `data/` y `models/`, que quedan con un `.gitkeep`) y la estructura de la sección 9.",
      "Crear el venv con Python 3.12, instalar las librerías de la sección 8 y fijar las versiones en `requirements.txt`.",
      "Crear `config.yaml` con rutas, semillas de las identidades e intrusos, tamaño 250 × 250 e hiperparámetros iniciales, y `src/config.py` para leerlo.",
      "Mandar por WhatsApp el instructivo de grabación de V1 (lo hace Santiago).",
    ],
    tests: [
      "`pytest` corre sin errores y verifica el entorno, las carpetas, `requirements.txt`, `.gitignore`, `pyproject.toml`, el README, las claves y valores de `config.yaml` y `src/config.py`.",
      "Se importan numpy, scipy, cv2, skimage, librosa, sklearn, customtkinter y sounddevice.",
      "Clonando el repo en otra carpeta, se instala siguiendo solo el README.",
    ],
    entregable: "Tag `v0.0-setup` · especificación del agente lista. **Hecho el 16/10** (300 tests).",
  }),
  ...etapa("6.2 H1 · Generador de huellas y dataset", {
    objetivo: "3 identidades autorizadas de tipos distintos y 60 intrusos, reproducibles por semilla, con 40 variantes por identidad autorizada.",
    pasos: [
      "**Campo de orientación:** modelo de Sherlock y Monro (1993), como en SFinGe (Cappelli et al., 2000), a partir de la posición de núcleos y deltas. Identidad 1 = presilla (1 núcleo + 1 delta); Identidad 2 = verticilo (2 núcleos + 2 deltas); Identidad 3 = arco en tienda (núcleo y delta cercanos). Guardar la posición real de núcleos y deltas: es la referencia del test de H2.",
      "**Crestas y minucias:** partir de ruido fijado por la semilla y aplicar iteraciones de filtrado de Gabor orientado según el campo (una cresta cada 8–10 px) dentro de una máscara elíptica suave. Las minucias aparecen solas. Opcional, si sobra tiempo: fase espiral (Larkin y Fletcher, 2007).",
      "**Intrusos:** 60 identidades con semillas y singularidades distintas, de todos los tipos (también arcos sin núcleo), con 3 variantes cada una.",
      "**Variantes:** deformación elástica suave, presión (erosión para dedo seco, dilatación para presión alta), rotación ±15°, traslación ±15 px, escala ±8 %, corte parcial periférico de hasta 15–20 % sin tocar el núcleo, ruido gaussiano y sal y pimienta. Guardar los parámetros en `metadatos.csv`.",
      "**Posición real del núcleo en cada variante:** pasar un mapa auxiliar con un punto gaussiano en el núcleo por la **misma función** que deforma la imagen (afín + elástica) y guardar su máximo en `metadatos.csv`. Así se incluye el desplazamiento elástico sin invertir el campo a mano (`cv2.remap` usa el mapeo inverso: sumar el desplazamiento en el punto lo aplica al revés) y sin depender del orden de las transformaciones.",
      "**Particiones sin fuga:** por identidad autorizada, 24 train / 8 val / 8 test (2 de test reservadas para la demo). Intrusos: 30 identidades (90 muestras) solo en val para calibrar el umbral y otras 30 (90 muestras) solo en test.",
      "**Plan B** (si el generador pasa el tope de 2 días): la base sintética FVC2002 DB4, generada con SFinGe. El set B se descarga directo y ya trae varias impresiones por dedo. Usar solo DB4 (DB1–DB3 son huellas reales) y citarla.",
    ],
    tests: [
      "Determinismo: misma semilla ⇒ arrays idénticos; semillas distintas ⇒ correlación cruzada baja (< 0,35).",
      "Particiones: ningún archivo ni identidad intrusa en dos particiones; conteos exactos: 72 train, 114 val (24 + 90) y 114 test (24 + 90).",
      "Metadatos: cada imagen tiene semilla, parámetros de la variante y posición real del núcleo (o “sin núcleo”).",
      "Posición real del núcleo: en una variante con solo rotación y traslación, el máximo del mapa auxiliar coincide con la fórmula afín (≤ 1 px).",
      "Revisión visual: lámina automática (`docs/figuras/grilla_h1.png`) con las 3 bases y sus variantes.",
    ],
    entregable: "Tag `v0.1-huellas-datos` · hito “Generación de huellas”.",
  }),
  ...etapa("6.3 H2 · Preprocesamiento y alineación por punto de referencia", {
    objetivo: "Llevar cualquier huella a 250 × 250 px alineada respecto de su punto de referencia, con iluminación uniforme, crestas realzadas y esqueleto de 1 px.",
    pasos: [
      "**Normalización y segmentación:** escala de grises, media y varianza fijas, y máscara de la huella por varianza local por bloques.",
      "**Punto de referencia (regla por tipo):** con el campo de orientación por bloques suavizado, buscar singularidades con el índice de Poincaré dentro de la máscara. 1 núcleo → ese punto; 2 núcleos (verticilo) → el punto medio; ningún núcleo (arco) → punto de máxima curvatura de las crestas (punto de referencia de FingerCode, Jain et al., 2000). Descartar singularidades pegadas al borde de la máscara, que suelen ser falsas.",
      "**Enderezar (mismo estimador para todas):** el ángulo es la orientación media (ángulo doble, ponderada por coherencia) en un anillo centrado en el punto de referencia y lo bastante ancho para encerrar los deltas, calculada sobre la imagen completa antes de recortar. No usar el eje núcleo–delta (si un delta queda cerca del borde y no se detecta, esa variante cambiaría de método), ni el eje de la máscara (se mueve con los cortes), ni un anillo chico alrededor del núcleo (ahí el ángulo doble da una vuelta completa y el promedio se anula). Un corte que se lleva parte del anillo sesga el ángulo unos 4–7°: no afecta a características con celdas de ~60 px y sectores de 45°.",
      "**Recortar:** el punto de referencia solo centra la ventana de 250 × 250 px, que se recorta sin reescalar.",
      "**Realce:** CLAHE y filtrado de Gabor sintonizado a la orientación y frecuencia local de las crestas (Hong, Wan y Jain, 1998).",
      "**Binarización y esqueletizado:** umbral adaptativo local y adelgazamiento hasta crestas de 1 px.",
    ],
    tests: [
      "Formato: entradas de otro tamaño o en color devuelven siempre la imagen realzada y el esqueleto de 250 × 250 px, no vacíos.",
      "Punto de referencia contra el valor real del generador: error ≤ 8 px en al menos el 95 % de las variantes, probado por separado para presilla, verticilo y arco en tienda.",
      "Arcos sin núcleo (intrusos): se usa el respaldo de máxima curvatura y el pipeline no falla.",
      "Ángulo contra los metadatos: error ≤ 5° en al menos el 95 % de las variantes, incluidas las que tienen corte.",
      "Estabilidad: la misma base rotada 12°, trasladada y con un corte periférico del 15 % da, después de alinear, campos de orientación parecidos en la región central (media de cos 2Δθ ≥ 0,9, a ajustar). No se compara píxel a píxel: con crestas de ~9 px de período, 2–3 px de corrimiento (los produce la distorsión elástica sola) llevan la correlación a cero aunque la alineación esté bien.",
      "El esqueleto tiene 1 px de ancho.",
    ],
    entregable: "Tag `v0.2-huellas-prepro` · hito “Normalización del dataset”.",
  }),
  ...etapa("6.4 H3 · Características ponderadas y PCA propio", {
    objetivo: "Un vector por huella, con bloques equilibrados, que separe las 3 identidades en el PCA.",
    pasos: [
      "**Bloque 1 · Orientación en grilla** (4 × 4 o 5 × 5) sobre la ventana alineada, codificada como (cos 2θ, sin 2θ) para evitar el salto entre 0° y 180°.",
      "**Bloque 2 · Energía de Gabor:** banco de 6 orientaciones × 2 frecuencias; energía por anillos y sectores alrededor del punto de referencia (FingerCode simplificado) e histograma global de respuestas.",
      "**Bloque 3 · Minucias:** *crossing number* sobre el esqueleto (CN = 1 terminación, CN = 3 bifurcación), filtrando las falsas (borde de la máscara, ramas cortas y cortes); cantidad, proporción y densidad radial alrededor del punto de referencia.",
      "**Bloque 4 · Momentos de Hu:** los 7 momentos en escala logarítmica, sobre el mapa de energía de crestas.",
      "**Estandarizar y ponderar:** media 0 y desvío 1 con estadísticas de train, y cada bloque dividido por √d_b. Con características estandarizadas, un bloque de d_b columnas aporta en promedio 2·d_b a la distancia al cuadrado; dividido por √d_b aporta 2 y todos pesan igual. Por eso ningún bloque puede tener menos de 5 columnas: una variable suelta (ej. la proporción de minucias) va dentro de su bloque, si no pesaría lo mismo que la grilla entera.",
      "**Revisar cada bloque:** medir en validación cuánto separa cada bloque por separado (silhouette o razón de Fisher). Si uno no aporta (puede pasar con Hu), bajarle el peso o sacarlo y anotarlo en el Documento 2.",
      "**PCA propio en NumPy:** con X centrada con la media de train, Σ = XᵀX/(n − 1); autovalores y autovectores con `np.linalg.eigh`, ordenados de mayor a menor; proyección 2D y 3D de train, val e intrusos y varianza explicada.",
    ],
    tests: [
      "Vectores de largo fijo, sin NaN ni Inf; después de ponderar, la contribución media de cada bloque a la distancia es parecida (±20 %).",
      "Oráculo: varianza explicada y proyecciones iguales a `sklearn.decomposition.PCA` salvo el signo de cada componente (tolerancia 1e-6).",
      "Escalador y pesos ajustados sin usar test.",
      "Separación: silhouette ≥ 0,50 con las 3 identidades en train.",
    ],
    entregable: "Tag `v0.3-huellas-features` · hito “Vectorización y PCA”.",
  }),
  ...etapa("6.5 H4 · K-means propio (k = 3) y umbral de rechazo", {
    objetivo: "Encontrar los 3 centroides con un K-means propio y calibrar en validación el umbral de distancia que rechaza intrusos.",
    pasos: [
      "**K-means en NumPy:** inicialización k-means++, 15 arranques (`n_init = 15`) y parada por tolerancia de movimiento de los centroides.",
      "**Asignar identidades:** a cada cluster, la identidad mayoritaria del train, verificando que sea una correspondencia uno a uno.",
      "**Distancia:** euclídea al centroide más cercano (opcional: normalizada por la dispersión de cada cluster).",
      "**Umbral:** barrer τ en validación (24 autorizados + 90 intrusos), trazar FAR y FRR y elegir τ* en el punto de igual error (EER) o donde FAR ≤ 5 %.",
      "**Evaluación a ciegas en test** con centroides y τ* congelados: matriz de confusión 4 × 4 (Identidad 1, 2, 3 y No autorizado), identificación, FAR y FRR.",
      "Guardar centroides, escalador, pesos de bloque, τ* y la correspondencia cluster → identidad.",
    ],
    tests: [
      "Oráculo: con la misma inicialización (`init=centroides, n_init=1`), centroides e inercia iguales a `sklearn.cluster.KMeans` (tolerancia 1e-6); con k-means++, centroides emparejados e inercia dentro del 1 %.",
      "Pureza de los clusters en train ≥ 95 % y correspondencia uno a uno.",
      "Lógica de decisión: distancias de ejemplo por encima, por debajo e igual a τ dan el resultado esperado.",
      "Sin fuga: ningún ID de test entra en la calibración del umbral.",
      "Aceptación: metas de la sección 1.3 cumplidas en test.",
    ],
    entregable: "Tag `v0.4-huellas-clasif` · hito “Clasificación K-means y métricas”.",
  }),
  ...etapa("6.6 V1 · Dataset de voz multihablante y clase “otro”", {
    objetivo: "Los 3 comandos dichos por al menos 10 personas reales, más voces TTS y la clase “otro”, separados por persona.",
    pasos: [
      "**WhatsApp:** 4 audios por persona: uno por comando, repitiendo la palabra 8–10 veces con 1 s de pausa, y un cuarto con 10 palabras cualesquiera, incluyendo algunas parecidas a los comandos (“conocer”, “salida”, “identidad”) para la clase “otro”. Avisar que es para un trabajo de la facultad.",
      "**Micrófono de la demo:** grabarse con la notebook que se va a usar en la defensa y, si se puede, grabar a compañeros en clase.",
      "**TTS y ruido:** voces en español de distintos países y géneros con edge-tts; ruido de aula y silencios para la clase “otro”.",
      "**Conversión y cortes:** pasar los .opus/.ogg a WAV mono de 16 kHz con ffmpeg, cortar cada palabra por los valles de energía (0,3–2,5 s) y registrar persona, comando y origen en `metadatos_voz.csv`.",
      "**Partición por persona:** las personas de test nunca aparecen en train ni en val.",
    ],
    tests: [
      "Formato: 16 kHz, mono, 0,3–2,5 s y sin saturación.",
      "Independencia de hablantes: ninguna persona en dos particiones.",
      "Mínimos: 10 personas reales o más y 8 repeticiones o más por comando y persona.",
      "Cada corte contiene una sola palabra (revisión por muestreo).",
    ],
    entregable: "Tag `v0.1-voz-datos` · hito “Dataset de audio”.",
  }),
  ...etapa("6.7 V2 · Preprocesamiento de voz y aumento de datos", {
    objetivo: "Audios comparables entre sí, vengan del celular, de la notebook o del TTS.",
    pasos: [
      "**Recorte de silencios** por energía de corto plazo con umbral adaptativo, dejando 40 ms de margen.",
      "**Pasabanda 80–4000 Hz** (banda que comparten todos los micrófonos), pre-énfasis (α = 0,97) y normalización RMS.",
      "**Aumento solo en train:** murmullo de fondo (SNR 15–20 dB), pitch ±1,5 semitonos y velocidad 0,92–1,08×. Cada copia aumentada conserva el ID de su persona, así queda en el mismo grupo en la validación cruzada.",
    ],
    tests: [
      "Silencio o ruido bajo: no lanza excepción ni divide por cero (devuelve `is_valid_speech = False`).",
      "Un comando precedido por 1 s de silencio sale recortado a la palabra y con RMS normalizado.",
      "El aumento nunca se aplica a val ni a test, y las copias heredan el ID de la persona.",
    ],
    entregable: "Tag `v0.2-voz-prepro` · hito “Preprocesamiento de audio”.",
  }),
  ...etapa("6.8 V3 · Vectorización por tramos (MFCC + CMN) y PCA", {
    objetivo: "Un vector de largo fijo por audio que capture la secuencia de sílabas de cada comando y dependa poco del micrófono y del tono de voz.",
    pasos: [
      "**MFCC:** 13 coeficientes y sus deltas, con ventanas de 25 ms y avance de 10 ms, y el banco Mel limitado a la banda de V2 (`fmax = 4000`, 26–40 filtros); si no, las bandas altas quedan vacías después del filtro.",
      "**CMN:** restar a cada coeficiente su media en el audio para cancelar la respuesta de cada micrófono.",
      "**5 tramos iguales:** media y desvío de MFCC, ΔMFCC, energía RMS y ZCR por tramo, más la duración de la palabra (“Salir” es más corta). En total, 281 valores.",
      "**Bloques:** MFCC (130), ΔMFCC (130) y prosódico = energía RMS + ZCR + duración (21). La duración no va como bloque propio: con 1/√d_b, una sola columna pesaría lo mismo que los 130 MFCC y alguien que dice “Salir” despacio podría salir como “Identificar”.",
      "**Estandarizar y ponderar** con estadísticas de train (cada bloque dividido por √d_b) y proyectar con el PCA propio en 2D y 3D.",
    ],
    tests: [
      "Un audio de 0,5 s y otro de 1,8 s dan vectores del mismo largo, sin NaN ni Inf.",
      "El mismo audio da siempre el mismo vector.",
      "CMN: una ecualización suave de agudos cambia poco el vector.",
      "Pesos: después de ponderar, la duración aporta menos del 5 % de la distancia promedio entre audios.",
    ],
    entregable: "Tag `v0.3-voz-features` · hito “Vectorización de audio”.",
  }),
  ...etapa("6.9 V4 · K-NN propio ponderado y rechazo", {
    objetivo: "Reconocer el comando de una persona nueva y rechazar lo que no es un comando.",
    pasos: [
      "**K-NN en NumPy** con voto ponderado w_i = 1/(d_i + ε).",
      "**Dejar una persona afuera (en src/):** un loop propio que saca todos los audios de una persona por vez, con sus copias aumentadas, para elegir K ∈ {3, 5, 7, 9} y decidir si conviene aplicar el PCA antes del K-NN (unas 20–30 componentes): con ~280 valores y pocos cientos de audios, el K-NN puede andar mejor en menos dimensiones.",
      "**Rechazo (“no entendí”)** si: 1) gana la clase “otro”; 2) la distancia media a los K vecinos supera τ_voz, calibrado en val; o 3) la confianza del voto ponderado es menor a 60 %.",
      "**Evaluación:** matriz de confusión 4 × 4 (Reconocer, Identificar, Salir, Otro / no entendí) con las personas de test.",
      "Guardar el modelo: vectores de referencia, escalador, pesos, PCA si se usa, K, ε y τ_voz.",
    ],
    tests: [
      "Oráculo: predicciones y probabilidades iguales a `KNeighborsClassifier(weights=lambda d: 1/(d + eps))` con el mismo K (salvo empates exactos de distancia).",
      "Oráculo del split: los grupos del loop propio coinciden con `LeaveOneGroupOut` de scikit-learn.",
      "Las personas de test no aparecen en train ni en val.",
      "Aceptación: acierto ≥ 90 % con personas de test y ≤ 5 % de “otro” tomado como comando.",
    ],
    entregable: "Tag `v0.4-voz-clasif` · hito “Clasificación K-NN y métricas”.",
  }),
  ...etapa("6.10 Fase 5 · Integración: máquina de estados y app", {
    objetivo: "Integrar los dos modelos en una app de escritorio cómoda y robusta para la demo en vivo.",
    pasos: [
      "**Máquina de estados:** ESPERANDO_MUESTRA → (elegir huella) HUELLA_CARGADA → “Reconocer” corre H2 → H3 → H4 y pasa a EVALUADA_AUTORIZADA (cartel verde) o EVALUADA_RECHAZADA (cartel rojo) → “Identificar” muestra “Identidad Sintética #N” y su distancia solo en EVALUADA_AUTORIZADA; si no, avisa. **Elegir otra huella vuelve a HUELLA_CARGADA.** “Salir” cierra limpio.",
      "**Push-to-talk:** botón “Hablar comando” (o barra espaciadora) que graba mientras se mantiene apretado, con tope de 3 s, aviso visual al empezar, medidor de nivel y recorte de silencios. Selector de .wav pregrabados como respaldo documentado.",
      "**Ventana en 3 paneles:** 1) huella de prueba (cruda y esqueleto); 2) control por voz, estado del agente y cartel grande; 3) PCA 2D de entrenamiento con la muestra evaluada marcada y coloreada según la decisión, y una barra con su distancia al centroide contra τ*. No se dibujan esferas de umbral: el umbral se aplica en el espacio completo y la proyección achica las distancias. Opcional si sobra tiempo: vista 3D.",
      "**Robustez:** cargar los modelos guardados y manejar errores (sin micrófono, sin huella seleccionada, archivo dañado).",
    ],
    testLabel: "Test antes del código (escenarios)",
    tests: [
      "Huella autorizada → “Reconocer” = AUTORIZADO → “Identificar” = identidad correcta.",
      "Huella intrusa → “Reconocer” = NO AUTORIZADO → “Identificar” no revela ninguna identidad.",
      "Reconocer una autorizada, cambiar a una intrusa y decir “Identificar” → no muestra la identidad anterior.",
      "Comandos sin huella seleccionada, o ruido (“no entendí”) → aviso en pantalla, sin excepciones.",
      "“Salir” cierra la app sin errores.",
      "Prueba con alguien que no está en el dataset.",
    ],
    entregable: "Tag `v1.0-rc` · hito “Integración e interfaz”.",
  }),
  ...etapa("6.11 Fase 6 · Informe final y ensayo de la defensa", {
    objetivo: "Informe PDF y demo listos para la semana del 10/11.",
    pasos: [
      "Armar el informe PDF a partir del Documento 2, con las secciones de la consigna y un resumen de hasta 200 palabras.",
      "Figuras: lámina de variantes y preprocesamiento (H1/H2), PCA de huellas y de voz (H3/V3), curva FAR–FRR y matrices de confusión (H4/V4) y capturas de cada caso en la app (Fase 5).",
      "IA generativa: prompts de ejemplo (diseño, tests y depuración) y cómo se verificó cada salida (pytest, oráculo de scikit-learn y revisión manual).",
      "Leer el informe completo para sacar el relleno y las frases genéricas, y hacer 3 ensayos completos de la demo con ruido de fondo.",
      "Después de presentar: repo público, sin audios de terceros.",
    ],
    testLabel: "Test",
    tests: [
      "Checklist de la sección 2 completo.",
      "Demo de punta a punta 3 veces seguidas sin fallas.",
    ],
    entregable: "Tag `v1.0` · informe y defensa.",
  }),

  // 7. CRONOGRAMA
  h1("7. Cronograma tentativo"),
  p("Alineado con las clases para ir sumando hitos en la planilla. Si un hito sale antes, se adelanta el siguiente.", { keepNext: true }),
  table(["Clase", "Hito a mostrar", "En paralelo"], [
    ["Hasta el mar 20/10", "Fase 0 (lista el 16/10) · H1 (generador y dataset de huellas)", "Pedido de audios enviado (V1)"],
    ["Jue 22/10", "H2 (punto de referencia y normalización) · V1 (dataset de voz)", "V2 · PCA propio en `src/core_ml/`"],
    ["Mar 27/10", "H3 + V2 + V3 (vectorización y PCA de huellas y de voz)", "K-means propio"],
    ["Jue 29/10", "H4 + V4 (K-means y K-NN propios, umbrales y matrices de confusión)", "Máquina de estados"],
    ["Mar 3/11", "Fase 5 (app con push-to-talk y visor PCA)", "Informe"],
    ["Jue 5/11", "Fase 6 (informe completo y ensayo en el aula)", "Ajuste de umbrales con el micrófono del aula"],
    ["Semana del 10/11", "Presentación", "Repo público (sin audios de terceros)"],
  ], [1900, 4570, 2600], { zebra: true }),
  p("Opcional si sobra tiempo: fase espiral en el generador (H1) y vista PCA 3D en la app (Fase 5).", { before: 80, run: { size: 19, color: "5F6B7A" } }),

  // 8. HERRAMIENTAS
  h1("8. Herramientas y librerías"),
  table(["Uso", "Herramienta"], [
    ["Lenguaje y entorno", "Python 3.12 · VS Code · venv en Windows (versiones fijadas en `requirements.txt`)"],
    ["Algoritmos centrales (`src/core_ml/`)", "NumPy: PCA, K-means++, K-NN ponderado y validación dejando una persona afuera, programados desde cero"],
    ["Métricas", "Propias en NumPy (matriz de confusión, FAR, FRR); silhouette de scikit-learn en los tests"],
    ["Oráculo de tests (`tests/`)", "scikit-learn: PCA, KMeans, KNeighborsClassifier, LeaveOneGroupOut"],
    ["Imágenes", "OpenCV · SciPy (ndimage, signal) · scikit-image (skeletonize, CLAHE, Gabor)"],
    ["Audio", "librosa · soundfile (lee Opus y MP3) · sounddevice (push-to-talk) · ffmpeg solo si llega otro formato (ej. .m4a)"],
    ["Voces sintéticas", "edge-tts"],
    ["Interfaz y gráficos", "CustomTkinter · Matplotlib"],
    ["Persistencia, configuración y tests", "joblib · PyYAML (`config.yaml` leído con `src/config.py`) · pytest (configurado en `pyproject.toml`)"],
    ["Plan B de huellas", "FVC2002 DB4 (sintética, SFinGe), solo si hace falta"],
    ["Versionado", "Git + GitHub (repo privado)"],
  ], [3000, 6070], { zebra: true }),

  // 9. ESTRUCTURA
  h1("9. Estructura del proyecto"),
  p("Arquitectura modular, como pidió la cátedra: cada etapa es un módulo con su test congelado, y la app solo los usa.", { keepNext: true }),
  ...code([
    "tf-ia-huellas-voz/",
    "├── README.md          instalación, uso, arquitectura y licencias",
    "├── requirements.txt   dependencias con versiones fijadas",
    "├── pyproject.toml     configuración de pytest",
    "├── config.yaml        semillas, rutas, tamaños e hiperparámetros",
    "├── src/",
    "│   ├── config.py      lectura de config.yaml",
    "│   ├── core_ml/       PCA, K-means++, K-NN y persona afuera (NumPy)",
    "│   ├── huellas/       H1 generador · H2 prepro · H3 features · H4 umbral",
    "│   ├── voz/           V1 ingesta y cortes · V2 prepro · V3 MFCC · V4 rechazo",
    "│   ├── agente/        máquina de estados",
    "│   └── app/           interfaz CustomTkinter",
    "├── tests/             test_setup, test_h1..h4, test_v1..v4, test_f5",
    "├── notebooks/         exploración y figuras del informe",
    "├── data/              fuera del repo (regenerable por semilla)",
    "│   ├── huellas/       train/ val/ test/ metadatos.csv",
    "│   └── voz/           raw_opus/ wav_16k/ cortes/ metadatos_voz.csv",
    "├── models/            .joblib: centroides, escaladores, umbrales",
    "└── docs/",
    "    ├── contratos/     contrato y prompt de los tests de cada etapa",
    "    └── figuras/       diagramas, láminas y figuras del informe",
  ]),
  h2("Convenciones"),
  ...bullets([
    "Nombres de archivo: `id1_v007.png`, `intruso31_v2.png`, `p05_reconocer_03.wav` (persona 05, repetición 3).",
    "`data/` y `models/` quedan fuera del repo: se regeneran con los scripts y las semillas.",
    "Las voces de terceros no se suben al repo público.",
    "Commits chicos y un tag por hito.",
  ]),

  // 10. RIESGOS
  h1("10. Riesgos y cómo cubrirlos"),
  table(["Riesgo", "Cómo cubrirlo"], [
    ["Arcos o verticilos confunden la detección del núcleo", "Regla por cantidad de núcleos con respaldo de máxima curvatura; test contra la posición real que guarda el generador."],
    ["Un corte parcial desplaza la alineación", "Alineación por punto de referencia, cortes acotados al 15–20 % periférico y ventana fija sin reescalar."],
    ["El ángulo de enderezado salta entre variantes de una misma identidad", "Un único estimador para todas (anillo que encierra los deltas) y test en grados contra los metadatos."],
    ["Un bloque largo, corto o poco útil distorsiona la distancia", "Ponderación 1/√d_b, ningún bloque de menos de 5 columnas y revisión de cada bloque en validación."],
    ["Pocos intrusos vuelven inestable el FAR", "60 identidades intrusas (30 en val, 30 en test)."],
    ["El generador propio tarda más de 2 días", "Tope de 2 días; plan B: FVC2002 DB4."],
    ["El ruido del aula dispara “Salir”", "Push-to-talk + clase “otro” con palabras parecidas + umbrales de distancia y de confianza."],
    ["Diferencias entre WhatsApp y el micrófono de la demo", "CMN, pasabanda común, MFCC con `fmax = 4000` y grabaciones con el micrófono de la demo."],
    ["K-NN en alta dimensión con pocos datos", "Comparar con K-NN sobre PCA en la validación dejando una persona afuera."],
    ["Error sutil en los algoritmos propios", "Tests contra scikit-learn con las mismas convenciones."],
    ["Falla el micrófono el día de la defensa", "Probarlo antes en el aula; selector de .wav y respaldo documentado."],
    ["Falta de tiempo", "Un hito por clase; lo opcional queda marcado como tal."],
  ], [3600, 5470], { zebra: true }),

  // ANEXO
  h1("Anexo A. Código de los diagramas (Mermaid)", { pageBreak: true }),
  p("Las figuras 1 y 2 se generan con este código. Para editarlas: pegarlo en mermaid.live o en VS Code con una extensión de Mermaid."),
  h2("Figura 1 · Flujo general"),
  ...code(fs.readFileSync(`${MMD}/flujo_v3.mmd`, "utf8").replace(/\s+$/, "").split("\n"), { size: 15, allowBreak: true }),
  h2("Figura 2 · Ciclo de cada etapa"),
  ...code(fs.readFileSync(`${MMD}/ciclo_v3.mmd`, "utf8").replace(/\s+$/, "").split("\n"), { size: 15, allowBreak: true }),
];

const doc = makeDoc({
  title: "TF IA I 2026 – Plan de desarrollo v0.3.2",
  description: "Plan de desarrollo del Trabajo Final de Inteligencia Artificial I (UNCuyo, 2026), versión 0.3.2",
  headerText: "TF Inteligencia Artificial I · 2026 · Plan de desarrollo · v0.3.2",
  children,
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(process.argv[2], buf); console.log("ok", process.argv[2], buf.length); });
