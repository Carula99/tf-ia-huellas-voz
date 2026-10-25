# Contrato · Fase 5 · Integración: agente con máquina de estados, push-to-talk con búfer circular y app

**Versión 1 · 25/10/2026.**

Es lo que verifica `tests/test_f5.py`. Los tests se escriben en otro chat o ventana aislada solo a partir de este archivo y se congelan (commit) antes de implementar. Se apoya en los contratos de la Fase 0 (`config.yaml`, `src/config.cargar_config` y `RAIZ`), de H1 (la receta de las variantes), de H2 (`preprocesar`), de H3 (`caracteristicas`, `Escalador`, `ajustar_escalador`, `PCA` y `caracteristicas.npz`), de H4 (`ModeloHuellas`, `cargar_modelo` y `reconocer`), de V1 (`leer_audio` y el dataset de voz) y de V4 (`ModeloVoz`, `cargar_modelo` y `reconocer`). Lo que se usa de cada uno está copiado en la sección 0.3, así que no hace falta leer esos contratos. Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

Esta etapa crea `src/agente/agente.py`, `src/agente/percepcion.py` y `src/agente/captura.py`, y `src/app/muestras.py`, `src/app/visor.py`, `src/app/controlador.py`, `src/app/ventana.py` y `src/app/__main__.py`. No cambia nada de `src/huellas/`, `src/voz/` ni `src/core_ml/`.

**Qué hace la Fase 5:** une los dos carriles en el agente de la sección 2 del Documento 2 (reactivo basado en modelos). El agente es la tabla de reglas condición–acción de la sección 2.5 del Documento 2, programada como una máquina de estados pura: recibe percepciones (una huella elegida o un comando de voz ya clasificado) y devuelve una acción, sin saber nada de la ventana ni del micrófono. Alrededor del agente: la percepción de la huella (H2 al cargarla, H3 y H4 al decir "Reconocer"), la captura de voz (micrófono abierto todo el tiempo sobre un búfer circular, push-to-talk de hasta 3 s con 250 ms antes de apretar y 150 ms después de soltar), las huellas que ofrece la app (las reservadas para la demo, los intrusos de test, variantes e intrusos nuevos generados en el momento y un archivo cualquiera), el visor PCA 2D de entrenamiento y un **controlador sin Tk** que arma todo y devuelve, después de cada evento, una `Pantalla` con lo que hay que mostrar. La ventana de CustomTkinter solo dibuja esa `Pantalla`: se verifica a mano (sección 7).

**Decisiones de Santiago (25/10) que este contrato cumple:** (1) "Salir" cierra directo, sin confirmación: la regla 9 de la tabla 2.5 no cambia; (2) la app ofrece las 6 reservadas para la demo, los intrusos de test, "Abrir archivo…" y **variantes nuevas** de cada identidad generadas en el momento (este contrato suma, como complemento, el **intruso nuevo**); (3) un solo contrato para la Fase 5: la lógica (agente, captura, muestras, visor y controlador) con tests, y la ventana con verificación manual. El tag de la etapa es `v1.0-rc`.

## 0. Convenciones

### 0.1 Nombres

- **Estados:** `ESPERANDO_MUESTRA`, `HUELLA_CARGADA`, `EVALUADA_AUTORIZADA` y `EVALUADA_RECHAZADA` (los de la tabla 2.5) más `FIN`, el estado al que lleva "Salir" (en la tabla, "Fin").
- **Comandos:** `voz.comandos` = `["reconocer", "identificar", "salir"]` y la clase de rechazo `voz.clase_rechazo` = `"otro"` ("no entendí"). Son los textos que devuelve `reconocer` de V4 en `comando`.
- **Identidades:** las de `huellas.identidades` (`id` 1, 2 y 3, con `nombre` "Identidad Sintética #1", "#2" y "#3"). `0` es "no autorizada" (`NO_AUTORIZADO` de H4).
- **Huella preparada:** la `HuellaPreprocesada` de H2 (tiene el campo `esqueleto`, `bool` de 250 × 250).
- **Muestras del micrófono:** los índices del búfer circular son **absolutos**: la muestra `i` es la `i`-ésima que se escribió desde que se creó el búfer (desde 0). Los largos y tiempos en muestras se calculan con `round(segundos · fs)` (el `round` de Python) a la frecuencia nativa `fs` del micrófono.

### 0.2 Reglas generales

- **Sin aleatoriedad nueva:** el agente, la captura, el visor y el controlador son deterministas. Las variantes e intrusos nuevos usan las semillas fijas de la receta de H1 (sección 4.4): dos sesiones de la app generan la misma secuencia.
- **Sin scikit-learn:** el texto de ningún archivo de `src/agente/` ni de `src/app/` contiene `sklearn`.
- **Lo que importa cada módulo:** `src/agente/agente.py` no importa `tkinter`, `customtkinter`, `sounddevice` ni `matplotlib`; `src/agente/captura.py` importa `sounddevice` solo dentro de las funciones que lo usan (el módulo se importa sin PortAudio); `src/app/controlador.py`, `muestras.py` y `visor.py` no importan `tkinter` ni `customtkinter`. Importar `src/app/ventana.py` o `src/app/__main__.py` no crea ninguna ventana.
- **Tipos:** `float` es `isinstance(x, float)` (un `np.float64` vale), "int" es un `int` de Python (o `np.integer`) que no sea `bool`; los arrays se controlan por su dtype. Las comparaciones de reales son absolutas a 1e-12 salvo donde dice otra cosa; "idéntico" es `np.array_equal`.
- **Textos exactos:** los de `TEXTOS` (2.1) y `AVISOS` (3.4) se comparan carácter por carácter. Llevan tildes, «» (U+00AB y U+00BB) y … (U+2026) donde se indica.

### 0.3 Lo que se usa de etapas anteriores (copiado de sus contratos)

```python
# Fase 0 · src/config.py
RAIZ: Path                                             # raíz del repo
def cargar_config(ruta=None) -> dict: ...              # ValueError que nombra cada clave obligatoria que falta

# H1 · src/huellas/generador.py y variantes.py
TIPOS = ("presilla", "verticilo", "arco_tienda", "arco")
def generar_base(tipo: str, semilla: int, cfg: dict) -> HuellaBase: ...      # ValueError si tipo no está en TIPOS
def sortear_parametros(rng: np.random.Generator, cfg: dict) -> ParametrosVariante: ...
def aplicar_variante(base: HuellaBase, params: ParametrosVariante, rng: np.random.Generator) -> Variante: ...
# Variante.imagen: uint8 (400, 320). Receta de una variante (sección 4 de H1):
#   rng = np.random.default_rng([semilla, variante]); imagen = aplicar_variante(base, sortear_parametros(rng, cfg), rng).imagen
# El dataset: identidad k con semilla huellas.identidades[k].semilla y variantes 1 a 40 (39 y 40 reservadas para la demo);
# intruso i (1 a 60) con tipo huellas.intrusos.tipos[(i - 1) % 4], semilla huellas.intrusos.semilla_base + i y variantes 1 a 3.

# H2 · src/huellas/preprocesamiento.py
def preprocesar(imagen: np.ndarray, cfg: dict) -> HuellaPreprocesada: ...
# ValueError si la imagen no es uint8 de (alto, ancho), (alto, ancho, 3) o (alto, ancho, 4), si mide menos de 64 px
# de lado o si no tiene una huella. HuellaPreprocesada.esqueleto: bool (250, 250).

# H3 · src/huellas/caracteristicas.py y src/core_ml/pca.py
BLOQUES = ("orientacion", "gabor", "minucias")
def caracteristicas(huella: HuellaPreprocesada, cfg: dict) -> dict[str, np.ndarray]: ...   # un vector (d_b,) por bloque
def ajustar_escalador(bloques_train: dict[str, np.ndarray]) -> Escalador: ...              # (n, d_b) por bloque
# Escalador.transformar(bloques) -> (d,) para un vector por bloque, (n, d) para (n, d_b) por bloque; d = 374 con los datos reales.
class PCA:  # PCA(n_componentes).ajustar(X) -> self; transformar(X) = (X - media) @ componentes.T;
            # atributos media (d,), componentes (k, d), proporcion_varianza (k,); signo: la entrada de mayor |valor| de cada componente es positiva
# caracteristicas.npz (data/huellas/): claves X (n, 374) float64, particion (n,) str ("train"/"val"/"test"), identidad (n,) int64 (0 = intruso), archivo (n,) str, ...

# H4 · src/huellas/clasificacion.py
NO_AUTORIZADO = 0
@dataclass(frozen=True, eq=False)
class ModeloHuellas:
    identidades: np.ndarray    # int64 (k,): 1 … k
    centroides: np.ndarray     # float64 (k, d)
    dispersion: np.ndarray     # float64 (k,)
    tau: float                 # τ*
    pureza: float
    inercia: float
    escalador: Escalador | None
    def puntajes(self, X): ...   # vector (d,) -> (puntaje float, identidad int): la menor ‖x − c_k‖ / dispersion_k y su identidad
@dataclass(frozen=True)
class Reconocimiento:            # el de H4
    identidad: int; mas_cercana: int; distancia: float; tau: float
def cargar_modelo(ruta) -> ModeloHuellas: ...      # FileNotFoundError si no existe
def reconocer(imagen, modelo, cfg) -> Reconocimiento: ...
# reconocer: z = modelo.escalador.transformar(caracteristicas(preprocesar(imagen, cfg), cfg)); distancia, mas_cercana = modelo.puntajes(z);
# identidad = mas_cercana si distancia <= modelo.tau, si no 0. ValueError si el modelo no tiene escalador.

# V1 · src/voz/audio.py
def leer_audio(ruta) -> tuple[np.ndarray, int]: ...   # float32 mono en [-1, 1] y la frecuencia del archivo; lanza si no lo puede leer
# data/voz/dataset/metadatos.csv: columnas archivo (relativo a data/voz/dataset, p. ej. "test/nb_p05_reconocer_01.wav"),
# particion, origen ("notebook", "whatsapp", "tts" o "ruido"), hablante, clase (un comando u "otro"), …

# V4 · src/voz/clasificacion.py
@dataclass(frozen=True)
class Reconocimiento:            # el de V4
    comando: str; prediccion: str | None; confianza: float; distancia: float; tau: float; valido: bool; motivo: str
def cargar_modelo(ruta) -> ModeloVoz: ...           # FileNotFoundError si no existe
def reconocer(audio: np.ndarray, fs: int, modelo: ModeloVoz, cfg: dict) -> Reconocimiento: ...
```

Modelos guardados: `models/modelo_huellas.npz` (se regenera con `venv\Scripts\python -m src.huellas.clasificacion --sobrescribir`) y `models/modelo_voz.npz` (con `venv\Scripts\python -m src.voz.clasificacion --sobrescribir`).

## 1. config.yaml

Claves nuevas, que también van en `CLAVES_OBLIGATORIAS` de `src/config.py` (cambio chico en el lugar):

```yaml
app:                        # Fase 5 (contrato f5_integracion.md)
  pre_roll_s: 0.25          # lo que se toma de antes de apretar el botón
  post_roll_s: 0.15         # lo que se sigue grabando después de soltarlo
  bufer_s: 5.0              # largo del búfer circular del micrófono
  medidor_ms: 50            # ventana del medidor de nivel
  variante_nueva_desde: 1000   # primera variante de "Variante nueva" (el dataset usa 1–40 y el prototipo de H4, 41–140)
  intruso_nuevo_desde: 1000    # primer intruso de "Intruso nuevo" (el dataset usa 1–60 y el prototipo de H4, 61–160)

metas:
  respuesta_max_s: 1.0      # de soltar el botón a la respuesta en pantalla (la meta de la tabla REAS)
```

Usa además las que ya existen: `voz.push_to_talk_max_s` (3.0), `voz.comandos`, `voz.clase_rechazo`, `huellas.identidades`, `huellas.intrusos.semilla_base`, `huellas.intrusos.tipos`, `rutas.huellas`, `rutas.voz`, `rutas.modelos` y `rutas.datos`.

- `bufer_s` tiene que alcanzar para el clip más largo con margen: `bufer_s ≥ pre_roll_s + push_to_talk_max_s + post_roll_s + 1.0` (5,0 ≥ 4,4).
- `variante_nueva_desde > 140` e `intruso_nuevo_desde > 160`: las huellas nuevas no son ninguna de las del dataset ni de la muestra ampliada del prototipo de H4.

## 2. src/agente/

### 2.1 src/agente/agente.py

```python
class Estado(str, Enum):
    ESPERANDO_MUESTRA = "ESPERANDO_MUESTRA"
    HUELLA_CARGADA = "HUELLA_CARGADA"
    EVALUADA_AUTORIZADA = "EVALUADA_AUTORIZADA"
    EVALUADA_RECHAZADA = "EVALUADA_RECHAZADA"
    FIN = "FIN"

TIPOS_ACCION: tuple[str, ...] = ("mostrar_huella", "resultado", "identidad", "aviso", "cerrar", "ninguna")
AUTORIZADO: str = "AUTORIZADO"
NO_AUTORIZADO: str = "NO AUTORIZADO"
REGLAS_AVISO: frozenset[int] = frozenset({2, 4, 6, 8, 10})
TEXTOS: dict[int, str] = {
    1: "Huella cargada: decí Reconocer",
    2: "Primero elegí una huella",
    4: "Primero decí Reconocer",
    6: "Huella no autorizada: no se muestra ninguna identidad",
    8: "No entendí, repetí el comando",
    9: "Cerrando la aplicación",
    10: "No se pudo leer la huella: elegí otra",
}

@dataclass(frozen=True)
class Accion:
    regla: int                     # 0 a 10 (la fila de la tabla de 4.1; 0 = ninguna)
    tipo: str                      # uno de TIPOS_ACCION
    texto: str                     # lo que se muestra
    estado: Estado                 # el estado después de la acción
    autorizado: bool | None = None # solo reglas 3 y 7
    identidad: int | None = None   # solo regla 5
    nombre: str | None = None      # solo regla 5: "Identidad Sintética #N"
    distancia: float | None = None # reglas 3, 5 y 7
    tau: float | None = None       # reglas 3, 5 y 7

def nombres_identidades(cfg: dict) -> dict[int, str]: ...     # {id: nombre} de huellas.identidades
def etiquetas_voz(cfg: dict) -> tuple[str, ...]: ...           # tuple(voz.comandos) + (voz.clase_rechazo,)

class Agente:
    def __init__(self, cfg: dict, preparar: Callable[[np.ndarray], Any],
                 evaluar: Callable[[Any], Any]) -> None: ...
    # Atributos (se leen; no se escriben desde afuera):
    #   estado: Estado              ESPERANDO_MUESTRA al crearse
    #   huella: str                 nombre de la huella cargada ("" si no hay)
    #   imagen: np.ndarray | None   la huella cruda cargada
    #   preparada: Any | None       lo que devolvió preparar(imagen)
    #   resultado: Any | None       lo que devolvió evaluar(preparada) en el último Reconocer
    #   identificada: bool          True después de la regla 5, hasta que cambie la huella o se vuelva a evaluar
    def percibir_huella(self, imagen: np.ndarray | None, nombre: str) -> Accion: ...
    def percibir_comando(self, comando: str) -> Accion: ...
```

- `preparar(imagen)` es la percepción de la huella al cargarla (en la app, `preparar_huella` de 2.2: H2); puede lanzar `ValueError` si la imagen no es una huella.
- `evaluar(preparada)` es la percepción al decir "Reconocer" (en la app, `evaluar_huella` de 2.2: H3 y H4). Devuelve un objeto con al menos los atributos `identidad` (`int`: una de `huellas.identidades` o `0`), `distancia` (`float`) y `tau` (`float`). El agente no lee ningún otro atributo (en particular, nunca usa `mas_cercana`).

### 2.2 src/agente/percepcion.py

```python
def leer_imagen(ruta: str | os.PathLike) -> np.ndarray | None: ...
def preparar_huella(imagen: np.ndarray, cfg: dict) -> HuellaPreprocesada: ...

@dataclass(frozen=True, eq=False)
class EvaluacionHuella:
    identidad: int        # la decisión: 1 a k, o 0 si se rechaza
    mas_cercana: int      # identidad del centroide más cercano (no se muestra nunca)
    distancia: float      # distancia normalizada a ese centroide (el puntaje de H4)
    tau: float            # τ* del modelo
    z: np.ndarray         # (d,) float64: la huella en el espacio de X (para el visor PCA)

def evaluar_huella(huella: HuellaPreprocesada, modelo: ModeloHuellas, cfg: dict) -> EvaluacionHuella: ...
```

### 2.3 src/agente/captura.py

```python
class SinMicrofono(RuntimeError): ...

def nivel_db(bloque: np.ndarray, piso: float = 1e-10) -> float: ...
def listar_microfonos() -> list[tuple[int, str, int]]: ...   # (índice, nombre, fs nativa) de cada entrada de sounddevice

class BuferCircular:
    def __init__(self, capacidad: int) -> None: ...
    capacidad: int
    @property
    def escritas(self) -> int: ...          # muestras escritas desde que se creó
    def escribir(self, bloque: np.ndarray) -> None: ...
    def leer(self, desde: int, hasta: int) -> np.ndarray: ...
    def ultimas(self, n: int) -> np.ndarray: ...

class PushToTalk:
    def __init__(self, bufer: BuferCircular, fs: int, cfg: dict) -> None: ...
    pre: int          # round(app.pre_roll_s · fs)
    post: int         # round(app.post_roll_s · fs)
    maximo: int       # round(voz.push_to_talk_max_s · fs)
    estado: str       # "reposo", "grabando" o "esperando"
    def apretar(self) -> bool: ...
    def soltar(self) -> bool: ...
    def tope(self) -> bool: ...
    def ventana(self) -> tuple[int, int] | None: ...
    def clip(self) -> np.ndarray | None: ...
    def cancelar(self) -> None: ...

class Microfono:
    def __init__(self, cfg: dict, dispositivo: int | None = None, fs: int | None = None,
                 fabrica: Callable[..., Any] | None = None) -> None: ...
    fs: int | None                  # frecuencia nativa (se resuelve en abrir si es None)
    bufer: BuferCircular | None     # se crea en abrir
    abierto: bool
    def abrir(self) -> None: ...
    def cerrar(self) -> None: ...
    def nivel_db(self) -> float | None: ...
```

## 3. src/app/

### 3.1 src/app/muestras.py

```python
ORIGENES: tuple[str, ...] = ("reservada", "intruso_test", "variante_nueva", "intruso_nuevo", "archivo")

@dataclass(frozen=True)
class EntradaHuella:
    nombre: str              # el rótulo de la lista (4.4)
    origen: str              # uno de ORIGENES
    verdad: int | None       # identidad real (1 a k) o 0 si es intruso; None si es un archivo cualquiera
    ruta: Path | None        # el PNG (reservada, intruso_test, archivo); None si se genera
    variante: int | None     # variante de H1; None si es un archivo cualquiera

def catalogo(cfg: dict, carpeta: str | os.PathLike | None = None) -> list[EntradaHuella]: ...
def entrada_archivo(ruta: str | os.PathLike) -> EntradaHuella: ...

class GeneradorDemo:
    def __init__(self, cfg: dict) -> None: ...
    def variante_nueva(self, identidad: int) -> tuple[EntradaHuella, np.ndarray]: ...
    def intruso_nuevo(self) -> tuple[EntradaHuella, np.ndarray]: ...
```

Por defecto, `carpeta` es `RAIZ / rutas.huellas`.

### 3.2 src/app/visor.py

```python
@dataclass(frozen=True, eq=False)
class VisorPCA:
    media: np.ndarray          # (d,) float64
    componentes: np.ndarray    # (2, d) float64
    proporcion: np.ndarray     # (2,) float64: varianza explicada de PC1 y PC2
    puntos: np.ndarray         # (n, 2) float64: las filas de train proyectadas
    identidades: np.ndarray    # (n,) int64: la identidad de cada fila de train
    centroides: np.ndarray     # (k, 2) float64: los centroides del modelo proyectados
    def proyectar(self, z: np.ndarray) -> np.ndarray: ...

def armar_visor(cfg: dict, modelo: ModeloHuellas, carpeta: str | os.PathLike | None = None) -> VisorPCA: ...
```

Por defecto, `carpeta` es `RAIZ / rutas.huellas` (donde está `caracteristicas.npz`).

### 3.3 src/app/controlador.py

```python
COLORES: tuple[str, ...] = ("neutro", "verde", "rojo")
AVISOS: dict[str, str] = {
    "inicio": "Elegí una huella para empezar",
    "audio_ilegible": "No se pudo leer el audio: elegí otro archivo",
    "sin_microfono": "No hay micrófono disponible: usá «Abrir audio…»",
    "error": "Error inesperado: {tipo}: {mensaje}",
}

@dataclass(frozen=True, eq=False)
class Pantalla:
    estado: str                        # el valor del Estado del agente
    huella: str                        # nombre de la huella cargada, o ""
    imagen: np.ndarray | None          # la huella cruda cargada
    esqueleto: np.ndarray | None       # bool (250, 250) de la huella preparada
    cartel: str                        # AUTORIZADO, NO_AUTORIZADO o ""
    color: str                         # uno de COLORES
    identificacion: str                # "Identidad Sintética #N" solo después de la regla 5; si no, ""
    distancia: float | None            # puntaje de la última evaluación (la barra contra τ*)
    tau: float | None
    punto_pca: tuple[float, float] | None
    mensaje: str                       # el texto de la última acción o aviso
    aviso: bool                        # True si lo último fue un aviso
    regla: int                         # regla de la última acción del agente; 0 si fue del controlador
    comando: str                       # el último comando decidido por la voz ("" si todavía no hubo)
    detalle_voz: str                   # diagnóstico del último audio (texto libre; "" si todavía no hubo)
    microfono: bool                    # hay un micrófono abierto
    grabando: bool                     # el push-to-talk está grabando
    nivel_db: float | None             # medidor de nivel (None sin micrófono)
    cerrar: bool                       # la app tiene que cerrarse

class Controlador:
    def __init__(self, cfg: dict, preparar: Callable, evaluar: Callable, reconocer_voz: Callable, *,
                 visor: VisorPCA | None = None, generador: GeneradorDemo | None = None,
                 entradas: list[EntradaHuella] | None = None, microfono: Microfono | None = None,
                 registro: str | os.PathLike | None = None) -> None: ...
    @classmethod
    def desde_archivos(cls, cfg: dict, *, modelos: str | os.PathLike | None = None,
                       huellas: str | os.PathLike | None = None, microfono: bool = True,
                       dispositivo: int | None = None,
                       registro: str | os.PathLike | None = None) -> "Controlador": ...
    agente: Agente
    entradas: list[EntradaHuella]
    def pantalla(self) -> Pantalla: ...
    def elegir(self, entrada: EntradaHuella) -> Pantalla: ...
    def abrir_archivo(self, ruta: str | os.PathLike) -> Pantalla: ...
    def variante_nueva(self, identidad: int) -> Pantalla: ...
    def intruso_nuevo(self) -> Pantalla: ...
    def comando_audio(self, y: np.ndarray, fs: int) -> Pantalla: ...
    def comando_wav(self, ruta: str | os.PathLike) -> Pantalla: ...
    def abrir_microfono(self) -> Pantalla: ...
    def apretar(self) -> Pantalla: ...
    def soltar(self) -> Pantalla: ...
    def actualizar(self) -> Pantalla: ...
    def cerrar(self) -> Pantalla: ...
```

- `reconocer_voz(y, fs)` devuelve un objeto con los atributos del `Reconocimiento` de V4 (`comando`, `prediccion`, `confianza`, `distancia`, `tau`, `valido` y `motivo`).

### 3.4 src/app/ventana.py y src/app/__main__.py

```python
# src/app/ventana.py
def ejecutar(controlador: Controlador) -> None: ...      # crea la ventana CustomTkinter y corre su mainloop

# src/app/__main__.py
def main(argv: list[str] | None = None) -> int: ...
```

**Línea de comandos:** `venv\Scripts\python -m src.app [--dispositivo N] [--listar] [--sin-microfono]`. `--listar` imprime los micrófonos de `listar_microfonos()` y sale con 0; `--sin-microfono` arranca sin abrir el micrófono (los comandos se dan con "Abrir audio…"). Si falta un modelo, imprime el error de `desde_archivos` (con el comando para regenerarlo) y sale con 1, sin abrir la ventana.

## 4. Definiciones exactas

### 4.1 La tabla de reglas (`Agente`)

Percepciones: **H**, una huella legible (`imagen` no es `None` y `preparar(imagen)` no lanza `ValueError`); **HI**, una huella ilegible (`imagen is None`, o `preparar(imagen)` lanza `ValueError`); y los comandos **R** (`"reconocer"`), **I** (`"identificar"`), **S** (`"salir"`) y **O** (`"otro"`). La celda es la regla que se aplica y, después de la flecha, el estado siguiente:

| Percepción | `ESPERANDO_MUESTRA` | `HUELLA_CARGADA` | `EVALUADA_AUTORIZADA` | `EVALUADA_RECHAZADA` | `FIN` |
|---|---|---|---|---|---|
| H | 1 → `HUELLA_CARGADA` | 1 → `HUELLA_CARGADA` | 1 → `HUELLA_CARGADA` | 1 → `HUELLA_CARGADA` | 0 → `FIN` |
| HI | 10 → `ESPERANDO_MUESTRA` | 10 → `ESPERANDO_MUESTRA` | 10 → `ESPERANDO_MUESTRA` | 10 → `ESPERANDO_MUESTRA` | 0 → `FIN` |
| R | 2 → `ESPERANDO_MUESTRA` | 3 → `EVALUADA_*` | 7 → `EVALUADA_*` | 7 → `EVALUADA_*` | 0 → `FIN` |
| I | 2 → `ESPERANDO_MUESTRA` | 4 → `HUELLA_CARGADA` | 5 → `EVALUADA_AUTORIZADA` | 6 → `EVALUADA_RECHAZADA` | 0 → `FIN` |
| S | 9 → `FIN` | 9 → `FIN` | 9 → `FIN` | 9 → `FIN` | 0 → `FIN` |
| O | 8 → sin cambio | 8 → sin cambio | 8 → sin cambio | 8 → sin cambio | 0 → `FIN` |

`EVALUADA_*` es `EVALUADA_AUTORIZADA` si `evaluar(preparada).identidad != 0` y `EVALUADA_RECHAZADA` si es `0`. Las reglas 1 a 9 son las de la tabla 2.5 del Documento 2; la 10 es nueva (una huella que no se puede leer ni preparar) y la 0 es la de `FIN`.

**Qué hace cada regla** (estado interno y `Accion` que devuelve; los campos que no se nombran quedan en `None`):

- **1:** llama a `preparar(imagen)`; `huella = nombre`, `imagen = imagen`, `preparada` = lo que devolvió, `resultado = None`, `identificada = False`. `Accion(1, "mostrar_huella", TEXTOS[1], HUELLA_CARGADA)`.
- **10:** `huella = ""`, `imagen = preparada = resultado = None`, `identificada = False` (se descarta también la huella anterior y su resultado). `Accion(10, "aviso", TEXTOS[10], ESPERANDO_MUESTRA)`. Con `imagen is None` no llama a `preparar`.
- **3 y 7:** llama a `evaluar(preparada)` (en la 7 vuelve a llamarlo: con H4, que es determinista, da lo mismo); `resultado` = lo que devolvió, `identificada = False`. `Accion(r, "resultado", AUTORIZADO o NO_AUTORIZADO, EVALUADA_AUTORIZADA o EVALUADA_RECHAZADA, autorizado=identidad != 0, distancia=float(resultado.distancia), tau=float(resultado.tau))`. **No lleva `identidad` ni `nombre`, ni siquiera si se autorizó:** la identidad se muestra solo con "Identificar". `ValueError` si `resultado.identidad` no es `0` ni una de `huellas.identidades` (sin cambiar el estado).
- **5:** `identificada = True`. `Accion(5, "identidad", nombre, EVALUADA_AUTORIZADA, identidad=resultado.identidad, nombre=nombre, distancia=…, tau=…)`, con `nombre = nombres_identidades(cfg)[resultado.identidad]`.
- **2, 4, 6 y 8:** no cambian nada. `Accion(r, "aviso", TEXTOS[r], estado)`.
- **9:** `estado = FIN`. `Accion(9, "cerrar", TEXTOS[9], FIN)`. Liberar el micrófono y cerrar la ventana es tarea del controlador y de la app.
- **0:** en `FIN`, cualquier percepción devuelve `Accion(0, "ninguna", "", FIN)` sin llamar a `preparar` ni a `evaluar`.

**Garantías:**

- `identidad` y `nombre` de una `Accion` son distintos de `None` **solo** en la regla 5, que solo se aplica en `EVALUADA_AUTORIZADA`. Ningún `texto` de una acción que no sea la regla 5 contiene alguno de los nombres de `huellas.identidades`.
- La regla 1 (y la 10) borran el resultado anterior: después de cargar otra huella, "Identificar" da la regla 4, nunca la identidad de la huella anterior (escenario 3 del plan).
- Si `preparar` lanza una excepción que no es `ValueError`, o `evaluar` lanza cualquier excepción, la excepción sale de `percibir_huella` o `percibir_comando` y el estado interno (todos los atributos) queda como estaba.
- `percibir_comando` lanza `ValueError` (sin cambiar nada) si `comando` no está en `etiquetas_voz(cfg)`. El agente no modifica la `imagen` que recibe.

### 4.2 Percepción de la huella

- **`leer_imagen(ruta)`:** `cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_UNCHANGED)` (así se leen rutas con tildes en Windows). Devuelve `None`, sin lanzar, si el archivo no existe, no se puede leer o no se puede decodificar (por ejemplo, un `.png` con bytes basura). Un PNG de 16 bits vuelve como `uint16`: `preparar_huella` lo rechaza con `ValueError` y el agente aplica la regla 10.
- **`preparar_huella(imagen, cfg)`:** es `preprocesar(imagen, cfg)` de H2 (los mismos errores).
- **`evaluar_huella(huella, modelo, cfg)`:** `z = modelo.escalador.transformar(caracteristicas(huella, cfg))`, `distancia, mas_cercana = modelo.puntajes(z)` e `identidad = mas_cercana` si `distancia ≤ modelo.tau`, si no `0`. Es la segunda mitad de `reconocer` de H4: para cualquier imagen `im`, `evaluar_huella(preparar_huella(im, cfg), modelo, cfg)` da los mismos `identidad`, `mas_cercana`, `distancia` y `tau` que `reconocer(im, modelo, cfg)` (idénticos). `z` es `float64` de `(d,)`. `ValueError` si el modelo no tiene escalador.
- Así, en la app, H2 corre una sola vez al cargar la huella (para mostrar el esqueleto) y "Reconocer" corre solo H3 y H4.

### 4.3 Captura de voz

- **`nivel_db(bloque, piso)`:** `20 · log10(max(rms, piso))` con `rms = √(media de bloque²)` en `float64`; un bloque vacío da `20 · log10(piso)` (−200 dB con el piso por defecto). `ValueError` si el bloque no es de 1 dimensión, si tiene `NaN` o `Inf`, o si `piso ≤ 0`.
- **`BuferCircular(capacidad)`:** `capacidad` es "int" ≥ 1; si no, `ValueError`. Guarda las últimas `capacidad` muestras en `float32`. `escribir(bloque)` acepta un array de 1 dimensión (si no, `ValueError`), lo agrega al final y suma su largo a `escritas`; un bloque vacío no hace nada; si el bloque es más largo que la capacidad, quedan sus últimas `capacidad` muestras (y `escritas` suma el largo entero). `leer(desde, hasta)` devuelve una **copia** `float32` de largo `hasta − desde` con las muestras absolutas `[desde, hasta)`; `ValueError` si `desde > hasta`, si `hasta > escritas` (todavía no se escribió) o si `desde < escritas − capacidad` (ya se pisó) o `desde < 0`. `ultimas(n)` devuelve las últimas `min(n, escritas, capacidad)` muestras. `escribir` y `leer` se pueden llamar desde hilos distintos (el callback del micrófono escribe en otro hilo): se protegen con un `threading.Lock`.
- **`PushToTalk(bufer, fs, cfg)`:** `fs` es "int" ≥ 1; `ValueError` si no, o si `bufer.capacidad < pre + maximo + post + fs` (el búfer tiene que guardar el clip más largo con un segundo de margen). Empieza en `"reposo"`.
  - `apretar()`: en `"reposo"` guarda `inicio_voz = bufer.escritas`, pasa a `"grabando"` y devuelve `True`; en otro estado no hace nada y devuelve `False` (la repetición de teclas de Windows manda varias pulsaciones de la barra espaciadora mientras se la mantiene apretada).
  - `soltar()`: en `"grabando"` guarda `fin_voz = min(bufer.escritas, inicio_voz + maximo)` (el tope de 3 s), pasa a `"esperando"` y devuelve `True`; en otro estado no hace nada y devuelve `False`.
  - `tope()`: `True` si está `"grabando"` y `bufer.escritas − inicio_voz ≥ maximo`.
  - `ventana()`: en `"esperando"`, `(a, b)` con `a = max(0, inicio_voz − pre)` y `b = fin_voz + post`; en otro estado, `None`.
  - `clip()`: en `"esperando"` y con `bufer.escritas ≥ b`, devuelve `bufer.leer(max(a, bufer.escritas − bufer.capacidad), b)`, pasa a `"reposo"` (y la ventana se olvida); si no, `None`. Con la llamada a tiempo, el clip es exactamente `[a, b)`: hasta `pre` muestras de antes de apretar, lo que se habló (a lo sumo `maximo`) y `post` muestras de después de soltar.
  - `cancelar()`: vuelve a `"reposo"` desde cualquier estado.
- **`Microfono(cfg, dispositivo, fs, fabrica)`:** `fabrica` es lo que crea el stream de entrada, con la firma de `sounddevice.InputStream`: se llama como `fabrica(samplerate=fs, channels=1, dtype="float32", device=dispositivo, callback=cb)` y devuelve un objeto con `start()`, `stop()` y `close()`. Por defecto es `sounddevice.InputStream` (importado dentro de `abrir`).
  - `abrir()`: si `fs` es `None`, lo toma de `int(round(sounddevice.query_devices(dispositivo, "input")["default_samplerate"]))`; crea `bufer = BuferCircular(round(app.bufer_s · fs))`; crea el stream con un callback `cb(indata, frames, time, status)` que hace `bufer.escribir(indata[:, 0])` (una copia del canal) y llama a `start()`; `abierto = True`. Si ya está abierto, no hace nada. **Cualquier** excepción al consultar el dispositivo, crear el stream o arrancarlo se convierte en `SinMicrofono` (con el mensaje de la original), después de cerrar lo que se haya abierto; `abierto` queda en `False`.
  - `cerrar()`: si hay stream, `stop()` y `close()` (las excepciones de esas dos llamadas se ignoran); `abierto = False`. Se puede llamar varias veces y también sin haber abierto.
  - `nivel_db()`: `None` si no está abierto; si no, `nivel_db(bufer.ultimas(round(app.medidor_ms / 1000 · fs)))`.
- **`listar_microfonos()`:** los dispositivos de `sounddevice.query_devices()` con `max_input_channels > 0`, como `(índice, nombre, int(round(default_samplerate)))`. No se prueba en pytest (depende del hardware).

### 4.4 Huellas que ofrece la app (`src/app/muestras.py`)

- **`catalogo(cfg, carpeta)`:** lee `carpeta / "metadatos.csv"` de H1 (columnas que usa: `archivo`, relativo a `carpeta`; `particion`; `identidad`; `intruso`; `variante`; `reservada_demo`). Devuelve, en este orden: (1) las filas con `reservada_demo == 1`, en el orden del CSV, como `EntradaHuella(nombre=f"{stem} (reservada)", origen="reservada", verdad=identidad, ruta=carpeta / archivo, variante=variante)`; (2) las filas con `particion == "test"`, `identidad == 0` y `variante == 1` (un intruso de test por identidad intrusa), en el orden del CSV, como `EntradaHuella(nombre=stem, origen="intruso_test", verdad=0, ruta=…, variante=1)`. `stem` es el nombre del archivo sin carpeta ni extensión (`"id1_v039"`, `"intruso31_v1"`). Con el dataset real son 6 + 30 = 36 entradas. `FileNotFoundError` si no está el CSV. No lee las imágenes.
- **`entrada_archivo(ruta)`:** `EntradaHuella(nombre=Path(ruta).name, origen="archivo", verdad=None, ruta=Path(ruta), variante=None)`.
- **`GeneradorDemo(cfg)`:** genera huellas que no están en ningún lado, con la receta de H1 (sección 0.3). Guarda las bases que ya generó (por `(tipo, semilla)`) para no volver a generarlas: la primera variante de una identidad tarda lo que tarda su base (1–2 s); las siguientes, unos 20 ms.
  - `variante_nueva(identidad)`: la `n`-ésima llamada para esa identidad (`n = 0, 1, 2, …`, un contador por identidad) usa `v = app.variante_nueva_desde + n`, la base `generar_base(tipo, semilla, cfg)` de esa identidad y `rng = np.random.default_rng([semilla, v])`; la imagen es `aplicar_variante(base, sortear_parametros(rng, cfg), rng).imagen`. Devuelve `(EntradaHuella(nombre=f"id{identidad}_v{v} (nueva)", origen="variante_nueva", verdad=identidad, ruta=None, variante=v), imagen)`. `ValueError` si `identidad` no es una de `huellas.identidades`.
  - `intruso_nuevo()`: la `n`-ésima llamada usa `i = app.intruso_nuevo_desde + n`, `tipo = huellas.intrusos.tipos[(i − 1) % len(tipos)]`, `semilla = huellas.intrusos.semilla_base + i` y la variante `1` con la misma receta. Devuelve `(EntradaHuella(nombre=f"intruso{i}_v1 (nuevo)", origen="intruso_nuevo", verdad=0, ruta=None, variante=1), imagen)`.
  - Con una copia de `cfg` con `app.variante_nueva_desde = 39`, la primera `variante_nueva(1)` es, píxel por píxel, `data/huellas/test/id1_v039.png`: es la misma receta que el dataset (en el prototipo dio idéntica).

### 4.5 Visor PCA (`src/app/visor.py`)

- **`armar_visor(cfg, modelo, carpeta)`:** con `X` y `particion` de `carpeta / "caracteristicas.npz"` y `T = X[particion == "train"]`: `pca = PCA(2).ajustar(T)` (el PCA propio de H3); `media`, `componentes` y `proporcion` son `pca.media`, `pca.componentes` y `pca.proporcion_varianza`; `puntos = pca.transformar(T)`; `identidades` = la columna `identidad` de esas filas (`int64`); `centroides = pca.transformar(modelo.centroides)`. Son los mismos ejes PC1 y PC2 de la figura de H3 (`pca_h3.png`), que también ajusta el PCA con train. `FileNotFoundError` si no está el `.npz`; `ValueError` si hay menos de 3 filas de train o si el ancho de `X` no es el de `modelo.centroides`.
- **`VisorPCA.proyectar(z)`:** `(z − media) @ componentes.T`; un vector `(d,)` da `(2,)` y una matriz `(n, d)` da `(n, 2)`. `ValueError` con otro `d` o con valores no finitos.
- El visor no dibuja esferas ni círculos de umbral: τ\* se aplica en el espacio completo de `d` columnas (con la distancia normalizada de H4) y la proyección achica las distancias. La comparación con τ\* se muestra con la barra (`Pantalla.distancia` contra `Pantalla.tau`).

### 4.6 Controlador y `Pantalla`

- **Construcción:** `Controlador(cfg, preparar, evaluar, reconocer_voz, …)` crea `agente = Agente(cfg, preparar, evaluar)`; `entradas` es la lista que se pasa (o `[]`). Si se pasa `registro`, crea la carpeta y escribe el encabezado del registro (abajo). No abre el micrófono. La pantalla inicial tiene `mensaje = AVISOS["inicio"]`, `aviso = False` y `regla = 0`.
- **`desde_archivos(cfg, modelos, huellas, microfono, dispositivo, registro)`:** con `modelos` (por defecto `RAIZ / rutas.modelos`) y `huellas` (por defecto `RAIZ / rutas.huellas`): carga `modelo_huellas.npz` y `modelo_voz.npz` con los `cargar_modelo` de H4 y V4; si falta alguno, `FileNotFoundError` con un mensaje que nombra el archivo y el comando que lo regenera (sección 0.3). Arma `preparar = lambda im: preparar_huella(im, cfg)`, `evaluar = lambda h: evaluar_huella(h, modelo_h, cfg)`, `reconocer_voz = lambda y, fs: reconocer(y, fs, modelo_v, cfg)` (el de V4), `visor = armar_visor(cfg, modelo_h, huellas)` (`None` si falta `caracteristicas.npz`), `generador = GeneradorDemo(cfg)`, `entradas = catalogo(cfg, huellas)` (`[]` si falta `metadatos.csv`) y, si `microfono` es `True`, `Microfono(cfg, dispositivo)` sin abrir.
- **`pantalla()`** se arma a partir del agente y del controlador:
  - `estado = agente.estado.value`; `huella = agente.huella`; `imagen = agente.imagen`; `esqueleto = agente.preparada.esqueleto` si hay huella preparada, si no `None`.
  - `cartel` y `color`: `(AUTORIZADO, "verde")` en `EVALUADA_AUTORIZADA`, `(NO_AUTORIZADO, "rojo")` en `EVALUADA_RECHAZADA` y `("", "neutro")` en los demás.
  - `identificacion = nombres_identidades(cfg)[agente.resultado.identidad]` si `agente.identificada`; si no, `""`.
  - `distancia` y `tau`: los de `agente.resultado` en `EVALUADA_AUTORIZADA` y `EVALUADA_RECHAZADA`; si no, `None`. `punto_pca = tuple(float(c) for c in visor.proyectar(agente.resultado.z))` en esos dos estados si hay visor y el resultado tiene `z`; si no, `None`.
  - `mensaje`, `aviso` y `regla`: los de la última acción del agente (`aviso = regla in REGLAS_AVISO`) o del último aviso del controlador (`regla = 0`, `aviso = True`).
  - `comando` y `detalle_voz`: los del último audio procesado. `detalle_voz` es texto libre, pero con un audio válido nombra la predicción, la confianza y la distancia, y con uno no válido, el `motivo` de V2.
  - `microfono` y `grabando`: si el micrófono está abierto y si el push-to-talk está `"grabando"`; `nivel_db = microfono.nivel_db()` (o `None`).
  - `cerrar`: `True` si el agente está en `FIN` o si se llamó a `cerrar()`.
- **Eventos** (cada uno devuelve `pantalla()` al final y **ninguno lanza excepciones**):
  - `elegir(entrada)`: `imagen = leer_imagen(entrada.ruta)` y `agente.percibir_huella(imagen, entrada.nombre)`. `abrir_archivo(ruta)` es `elegir(entrada_archivo(ruta))`.
  - `variante_nueva(identidad)` e `intruso_nuevo()`: piden la imagen al `generador` y la pasan a `agente.percibir_huella(imagen, entrada.nombre)`.
  - `comando_audio(y, fs)`: `rec = reconocer_voz(y, fs)`, guarda `comando = rec.comando` y `detalle_voz`, y llama a `agente.percibir_comando(rec.comando)`. Si la acción es la regla 9, cierra el micrófono.
  - `comando_wav(ruta)`: `y, fs = leer_audio(ruta)` y después `comando_audio(y, fs)`. Si `leer_audio` lanza cualquier excepción: aviso `AVISOS["audio_ilegible"]`, sin llamar a `reconocer_voz` ni al agente.
  - `abrir_microfono()`: si hay micrófono, lo abre y crea `PushToTalk(microfono.bufer, microfono.fs, cfg)`; con `SinMicrofono` (o si no hay micrófono): aviso `AVISOS["sin_microfono"]`.
  - `apretar()` y `soltar()`: llaman a los del push-to-talk; sin micrófono abierto, aviso `AVISOS["sin_microfono"]`.
  - `actualizar()` (la app lo llama cada 30 ms): si el push-to-talk está `"grabando"` y `tope()`, llama a `soltar()`; después, si `clip()` devuelve un clip, hace `comando_audio(clip, microfono.fs)`. Sin micrófono no hace nada.
  - `cerrar()`: cierra el micrófono (se puede llamar varias veces); desde ahí `pantalla().cerrar` es `True`.
  - Después de la regla 9 o de `cerrar()`, todos los eventos devuelven la pantalla sin hacer nada (no llaman a `preparar`, `evaluar`, `reconocer_voz`, `leer_imagen`, `leer_audio` ni al generador).
  - **Errores:** cualquier excepción dentro de un evento (por ejemplo, de `evaluar`, `reconocer_voz` o el generador) se atrapa y deja el aviso `AVISOS["error"].format(tipo=type(e).__name__, mensaje=str(e))`; el estado del agente queda como estaba (4.1) y los eventos siguientes funcionan.
- **Registro** (si se pasó `registro`): un CSV UTF-8 con el encabezado `momento,percepcion,detalle,regla,estado_antes,estado_despues,mensaje` y una fila por cada acción del agente y por cada aviso del controlador, en el orden en que pasan. `momento` es la fecha y hora en ISO 8601; `percepcion` es `"huella"`, `"comando"` o `"aviso"` (los del controlador); `detalle` es el nombre de la huella, el comando con lo que dijo V4, o el texto del aviso; `regla`, el número; y los estados, sus valores. La app lo escribe en `RAIZ / rutas.datos / "app" / "registro_<AAAAMMDD_HHMMSS>.csv"` (fuera de Git). No se guarda audio.

### 4.7 La ventana (verificación manual, sección 7)

`ejecutar(controlador)` arma una ventana de CustomTkinter con tres paneles, que dibuja después de cada evento la `Pantalla` que devuelve el controlador:

1. **Huella:** la lista de `controlador.entradas`, los botones "Variante nueva" (con la identidad 1, 2 o 3) e "Intruso nuevo", y "Abrir archivo…" (PNG). Muestra la huella cruda y el esqueleto lado a lado y el nombre de la huella.
2. **Voz y agente:** el botón "Mantener para hablar" (graba mientras se lo mantiene apretado, igual que la barra espaciadora: `<ButtonPress>`/`<ButtonRelease>` y `<KeyPress-space>`/`<KeyRelease-space>` en toda la ventana), un indicador "● Grabando" mientras `grabando`, el medidor de nivel (`nivel_db` de −60 a 0 dBFS), "Abrir audio…" (WAV, el respaldo sin micrófono), el estado del agente, el cartel grande (`cartel` en verde o rojo), la `identificacion`, el `mensaje` (resaltado si es un aviso) y el `detalle_voz` en letra chica.
3. **PCA:** Matplotlib dentro de la ventana: los puntos de train del visor coloreados por identidad, los centroides marcados, la proporción de varianza de cada eje y, después de "Reconocer", la huella evaluada como una estrella verde (autorizada) o roja (rechazada); abajo, una barra con `distancia` y una línea en `tau`. Sin visor, el panel dice "Sin datos para el PCA".

Llama a `controlador.actualizar()` cada 30 ms con `after`. Con `cerrar` en `True` (la regla 9), muestra el mensaje medio segundo y destruye la ventana; la cruz de la ventana llama a `controlador.cerrar()` antes de destruirla. Generar una huella nueva puede bloquear la ventana hasta 2 s (con el mensaje "Generando…" antes); el micrófono sigue grabando en su hilo.

## 5. Datos de prueba

- **Dobles de prueba** (6.2, 6.3 y 6.9): `preparar` falso (devuelve un objeto con `esqueleto`, o lanza `ValueError` para ciertas imágenes), `evaluar` falso (devuelve un objeto con `identidad`, `distancia`, `tau` y `z` según la huella, y cuenta las llamadas), `reconocer_voz` falso (devuelve objetos con los campos del `Reconocimiento` de V4, por ejemplo de una cola de comandos) y una **fábrica de stream falsa** (guarda el callback para que el test le pase bloques conocidos y anota las llamadas a `start`, `stop` y `close`; otra versión lanza `OSError` al crearse o al arrancar).
- **Huellas chicas** (6.4 y 6.7), generadas por el test con la receta de H1: algunas variantes de las identidades 1 a 3 (por ejemplo, 2 por identidad) y un intruso. Para `evaluar_huella` el test arma un `ModeloHuellas` a mano: `escalador = ajustar_escalador` de los bloques de esas huellas, `centroides` = la media de sus `z` por identidad, `dispersion` = unos, `tau` elegido por el test, `pureza = 1.0` e `inercia = 0.0`.
- **Dataset chico de H3 y H1** (6.7 y 6.8), escrito por el test en `tmp_path`: un `metadatos.csv` con las columnas de 4.4 (y PNG en las rutas que nombra, si el test los lee) y un `caracteristicas.npz` con `X` (gaussianas con semilla, de `(n, d)`), `particion` e `identidad`.
- **Audio sintético** (6.5 y 6.9): rampas `np.arange` en `float32` (para saber qué muestra es cada una) y ruido con semilla.
- **Datos reales** (6.11): `models/modelo_huellas.npz`, `models/modelo_voz.npz`, `data/huellas/` (PNG, `metadatos.csv` y `caracteristicas.npz`) y `data/voz/dataset/` (WAV de V1 y `metadatos.csv`).

## 6. Criterios que verifican los tests

### 6.1 Configuración

- Las claves de la sección 1 están en `config.yaml` con esos valores (`pre_roll_s == 0.25`, `post_roll_s == 0.15`, `bufer_s == 5.0`, `medidor_ms == 50`, `variante_nueva_desde == 1000`, `intruso_nuevo_desde == 1000` y `metas.respuesta_max_s == 1.0`), y `cargar_config` de una copia de `config.yaml` en `tmp_path` sin cualquiera de ellas lanza `ValueError` que la nombra.
- `bufer_s ≥ pre_roll_s + voz.push_to_talk_max_s + post_roll_s + 1.0`, `variante_nueva_desde > 140` e `intruso_nuevo_desde > 160`.
- `nombres_identidades(cfg) == {1: "Identidad Sintética #1", 2: "Identidad Sintética #2", 3: "Identidad Sintética #3"}` y `etiquetas_voz(cfg) == ("reconocer", "identificar", "salir", "otro")`.

### 6.2 Agente (dobles de prueba)

- **La tabla completa:** para cada uno de los 5 estados y cada una de las 6 percepciones de 4.1 (30 casos; `EVALUADA_*` con un `evaluar` que autoriza y con uno que rechaza), llevando al agente a ese estado con percepciones de la tabla, la `Accion` tiene la `regla`, el `tipo`, el `texto` y el `estado` de 4.1, y `agente.estado` es ese estado.
- **Campos de la acción:** reglas 3 y 7 con `autorizado`, `distancia` y `tau` del resultado e `identidad` y `nombre` en `None` (también si se autorizó); regla 5 con `identidad`, `nombre`, `distancia` y `tau`; en las demás, todos esos campos en `None`. Los textos son exactamente los de `TEXTOS`, `AUTORIZADO`, `NO_AUTORIZADO` y los nombres de `huellas.identidades`.
- **Nunca se revela una identidad:** en secuencias de percepciones al azar (con semilla, por ejemplo 2000 pasos con huellas autorizadas, rechazadas e ilegibles y todos los comandos), `identidad` o `nombre` no son `None` solo con la regla 5 y en `EVALUADA_AUTORIZADA`, y ningún `texto` de otra regla contiene un nombre de `huellas.identidades`. Un oráculo de la tabla escrito en el test da la misma regla y el mismo estado en cada paso.
- **Llamadas:** la regla 1 llama a `preparar` una vez con la imagen; la 10 con `imagen=None` no lo llama; las reglas 3 y 7 llaman a `evaluar` una vez con lo que devolvió `preparar`; las reglas 2, 4, 5, 6, 8, 9 y 0 no llaman a ninguno; en `FIN` no se llama a nada.
- **Regla 7 con otro resultado:** si un `evaluar` falso cambia de decisión entre dos "Reconocer", el estado sigue al último resultado.
- **Excepciones:** con un `preparar` que lanza `RuntimeError`, o un `evaluar` que lanza, la excepción sale y `estado`, `huella`, `imagen`, `preparada`, `resultado` e `identificada` quedan como estaban; con un `preparar` que lanza `ValueError` se aplica la regla 10. `percibir_comando` con `"Reconocer"`, `"hola"` o `""` lanza `ValueError` sin cambiar nada. Una `identidad` del resultado fuera de `{0, 1, 2, 3}` lanza `ValueError`.
- El agente no modifica la imagen que recibe (se compara con una copia).

### 6.3 Escenarios de la Fase 5 del plan (con el controlador y dobles de prueba)

1. **Huella autorizada:** `elegir` → comando `"reconocer"` → `cartel == AUTORIZADO`, `color == "verde"` → `"identificar"` → `identificacion` es el nombre de la identidad del resultado.
2. **Huella intrusa:** `elegir` → `"reconocer"` → `NO_AUTORIZADO`, `"rojo"` → `"identificar"` → regla 6, `identificacion == ""` y ningún campo de texto de la `Pantalla` contiene un nombre de identidad.
3. **Cambio de huella:** autorizada → `"reconocer"` → `"identificar"` (se ve la identidad) → `elegir` una intrusa → `identificacion == ""`, `cartel == ""` → `"identificar"` → regla 4 y sigue sin identidad; después de `"reconocer"` → `NO_AUTORIZADO` → `"identificar"` → regla 6.
4. **Sin huella o ruido:** `"reconocer"` e `"identificar"` sin huella → regla 2; `"otro"` (también un `Reconocimiento` no válido, con `motivo` `"sin_voz"`) en cada uno de los cuatro estados que no son `FIN` → regla 8, el estado no cambia y el `detalle_voz` nombra el motivo; ninguno lanza.
5. **Salir:** `"salir"` desde cada uno de los cuatro estados que no son `FIN` → regla 9, `cerrar == True` y la fábrica falsa registra `stop()` y `close()` del micrófono una vez; los eventos siguientes no llaman a nada.
6. **Alguien que no está en el dataset:** es la prueba en vivo de la sección 7.

### 6.4 Percepción de la huella

- `evaluar_huella(preparar_huella(im, cfg), modelo, cfg)` da los mismos `identidad`, `mas_cercana`, `distancia` y `tau` que `reconocer(im, modelo, cfg)` de H4 (idénticos) para las huellas chicas de la sección 5, con un `tau` que acepta unas y rechaza otras; `z` es el `escalador.transformar(caracteristicas(…))` (idéntico) y es `float64` de `(d,)`. `ValueError` con un modelo sin escalador.
- `leer_imagen` de un PNG escrito por el test (también en una carpeta con tildes, como `tmp_path / "huellas ñandú"`) da la imagen idéntica; de un archivo que no existe, de uno vacío y de uno con bytes basura, `None` sin lanzar.
- `preparar_huella` de una imagen de fondo liso (todo 245), de una de 50 × 50 y de una `float64` lanza `ValueError` (en el agente, regla 10).

### 6.5 Búfer circular y push-to-talk (rampas)

- **Búfer:** escribiendo una rampa en bloques de largos variados (incluidos bloques vacíos y uno más largo que la capacidad), `escritas` es la suma de los largos, `leer(desde, hasta)` devuelve exactamente las muestras de la rampa en `[desde, hasta)` (`float32`, una copia: modificarla no cambia el búfer) y `ultimas(n)`, las últimas. `ValueError` con `desde > hasta`, `hasta > escritas`, `desde < escritas − capacidad`, `desde < 0`, un bloque de 2 dimensiones y `capacidad` en 0, −1, 2.5 o `True`.
- **Hilos:** un hilo que escribe 200 bloques mientras el principal lee la ventana más reciente no da errores ni muestras fuera de la rampa.
- **Push-to-talk** (con `fs = 1000` para que las cuentas sean enteras: `pre = 250`, `post = 150`, `maximo = 3000`): con la rampa, `apretar()` en `escritas = 5000` y `soltar()` en `escritas = 6200` dan `ventana() == (4750, 6350)`; `clip()` es `None` hasta que `escritas ≥ 6350` y después devuelve exactamente la rampa en `[4750, 6350)` y vuelve a `"reposo"`. Apretado en `escritas = 100` (menos que el pre-roll), `a = 0`. Mantenido más de 3 s: `tope()` es `True` desde `escritas − inicio_voz = 3000`, y `soltar()` en `escritas = 9000` (con `inicio_voz = 5000`) da `fin_voz = 8000` y la ventana `(4750, 8150)`. `apretar` dos veces seguidas no mueve `inicio_voz` (devuelve `False` la segunda vez); `soltar` sin `apretar` devuelve `False` y no cambia nada; `cancelar` vuelve a `"reposo"`. `ValueError` con un búfer más chico que `pre + maximo + post + fs`.
- **Nivel:** `nivel_db` de un seno de amplitud 1 da `20 · log10(1/√2)` (1e-9), el de ceros da `20 · log10(piso)`, y el de un bloque vacío también; `ValueError` con `NaN`, 2 dimensiones o `piso = 0`.

### 6.6 Micrófono (con la fábrica falsa)

- `Microfono(cfg, fs=48000, fabrica=falsa).abrir()` llama a la fábrica con `samplerate=48000`, `channels=1`, `dtype="float32"` y un `callback`, llama a `start()` y deja `abierto`; el búfer tiene `round(5.0 · 48000)` de capacidad. Pasarle al callback bloques `(n, 1)` de una rampa los deja en el búfer en orden. `nivel_db()` es el de las últimas 2400 muestras (50 ms); sin abrir, `None`.
- `cerrar()` llama a `stop()` y `close()` una vez y se puede repetir sin error; `abrir()` dos veces crea un solo stream.
- Con una fábrica que lanza `OSError` (al crearse o en `start()`), `abrir()` lanza `SinMicrofono` y `abierto` es `False`.
- `src/agente/captura.py` se importa sin importar `sounddevice` (se comprueba en un subproceso de Python que importa el módulo y mira `sys.modules`).

### 6.7 Huellas de la app

- `catalogo` sobre un `metadatos.csv` chico escrito por el test devuelve primero las reservadas (con ` (reservada)`, `origen`, `verdad`, `ruta` y `variante`) y después un intruso de test por identidad intrusa (`variante == 1`), en el orden del CSV y con los nombres de 4.4; no incluye train, val, ni los intrusos de val ni las variantes 2 y 3. `FileNotFoundError` sin el CSV.
- `GeneradorDemo.variante_nueva(k)`: la primera llamada usa `v = 1000`, la segunda `1001`, con un contador por identidad; la imagen es idéntica a la receta de H1 escrita en el test (`generar_base`, `default_rng([semilla, v])`, `sortear_parametros`, `aplicar_variante`), es `uint8` de `(400, 320)`, y `EntradaHuella` tiene el `nombre`, `origen`, `verdad` y `variante` de 4.4. Con `app.variante_nueva_desde = 39` en una copia de `cfg`, la primera de la identidad 1 es idéntica a la de la receta con `v = 39`. `ValueError` con la identidad 4 o 0. Dos `GeneradorDemo` dan la misma secuencia.
- `GeneradorDemo.intruso_nuevo()`: la primera usa `i = 1000` (tipo `tipos[999 % 4]`, semilla `semilla_base + 1000`, variante 1) y la segunda `i = 1001`; imágenes idénticas a la receta.
- La base de cada `(tipo, semilla)` se genera una sola vez (se comprueba envolviendo `generar_base` con un contador mediante `monkeypatch` sobre el módulo de `muestras`).

### 6.8 Visor PCA

- Con el `caracteristicas.npz` chico: `componentes`, `media` y `proporcion` son los de `PCA(2).ajustar(X_train)` del `src.core_ml.pca` (1e-12); los valores absolutos de `componentes` coinciden con los de `sklearn.decomposition.PCA(2)` (1e-9); `puntos` y `centroides` son sus proyecciones; `identidades` son las de train; `proyectar` de un vector da su fila, y de una matriz, todas. `FileNotFoundError` sin el `.npz`; `ValueError` con menos de 3 filas de train, con un modelo de otro `d` y en `proyectar` con otro `d` o con `NaN`.

### 6.9 Controlador (dobles de prueba)

- **`Pantalla`:** después de cada evento de los escenarios de 6.3, `estado`, `cartel`, `color`, `identificacion`, `distancia`, `tau`, `mensaje`, `aviso` y `regla` son los de 4.6; `esqueleto` es el del `preparar` falso; `punto_pca` es `visor.proyectar(z)` del resultado después de "Reconocer" y `None` antes y después de cargar otra huella. Con `estado` distinto de `EVALUADA_AUTORIZADA`, `identificacion == ""`.
- **Errores sin excepciones:** un `evaluar` que lanza `RuntimeError("falla")` en "Reconocer" deja `mensaje == "Error inesperado: RuntimeError: falla"`, `aviso == True`, `regla == 0` y el estado en `HUELLA_CARGADA`, y el evento siguiente funciona; `comando_wav` de un archivo que no existe o con bytes basura deja `AVISOS["audio_ilegible"]` sin llamar a `reconocer_voz`; `abrir_archivo` de un PNG con bytes basura aplica la regla 10; `variante_nueva` sin generador deja un aviso de error.
- **Audio:** `comando_wav` de un WAV escrito por el test con `soundfile` (a 48 kHz) le pasa a `reconocer_voz` lo mismo que `leer_audio` (idéntico) y su `fs`.
- **Micrófono:** sin micrófono o con la fábrica que falla, `abrir_microfono()` deja `AVISOS["sin_microfono"]` y `microfono == False`, y `apretar()` también avisa. Con la fábrica falsa y `fs = 1000` en el `Microfono`: rampa al callback, `apretar()`, más rampa, `soltar()`, más rampa; `actualizar()` antes del post-roll no llama a `reconocer_voz`, y después lo llama una vez con exactamente la ventana de 4.3 y `fs = 1000`; `grabando` es `True` entre `apretar` y `soltar`. Mantenido más de 3 s, `actualizar()` corta solo en el tope.
- **Salir:** la regla 9 cierra el micrófono (la fábrica falsa registra `stop` y `close`), `cerrar == True`, y después `elegir`, `comando_audio`, `comando_wav`, `apretar` y `actualizar` no llaman a nada. `cerrar()` también deja `cerrar == True` y se puede repetir.
- **Registro:** con `registro = tmp_path / "app" / "registro.csv"`, después de un escenario el archivo tiene el encabezado de 4.6 y una fila por acción y aviso, con `regla`, `estado_antes` y `estado_despues` que coinciden con las pantallas.
- **`desde_archivos`** con una carpeta de modelos vacía en `tmp_path`: `FileNotFoundError` cuyo mensaje nombra `modelo_huellas.npz` (o `modelo_voz.npz`) y `--sobrescribir`.

### 6.10 La app se puede importar

- `import src.app.ventana` y `import src.app.__main__` (en un subproceso, sin pantalla interactiva) no crean ninguna ventana ni fallan; `ejecutar` y `main` existen y son llamables.
- El texto de los archivos de `src/agente/` y `src/app/` no contiene `sklearn`; `src/agente/agente.py` no importa `tkinter`, `customtkinter`, `sounddevice` ni `matplotlib`, y `controlador.py`, `muestras.py` y `visor.py` no importan `tkinter` ni `customtkinter` (se comprueba en un subproceso mirando `sys.modules` después de importarlos).

### 6.11 Aceptación (`@pytest.mark.aceptacion`, con los modelos y datos reales)

Cada test se saltea con `pytest.skip` si falta lo que usa (`models/modelo_huellas.npz`, `models/modelo_voz.npz`, `data/huellas/metadatos.csv`, `data/huellas/caracteristicas.npz` o `data/voz/dataset/metadatos.csv`).

- **Huellas reales** (con el `Controlador` armado con `preparar_huella`, `evaluar_huella` y el modelo de H4, y un `reconocer_voz` de prueba que devuelve el comando que pide el test): `catalogo` tiene 36 entradas (6 reservadas y 30 intrusos de test). Las 6 reservadas → `AUTORIZADO` → `"Identidad Sintética #k"` de su identidad, con `distancia` igual a la de `reconocer` de H4 (1e-12) y a la de la sección 8 (1e-3). Los 30 intrusos de test → `NO_AUTORIZADO` → regla 6. El escenario 3 con `id1_v040` y `intruso31_v1`. Las 10 primeras `variante_nueva` de cada identidad → `AUTORIZADO` con su identidad, y los 10 primeros `intruso_nuevo` → `NO_AUTORIZADO` (sección 8: en el prototipo, las 30 primeras de cada identidad y los 30 primeros intrusos).
- **Visor real:** `proporcion` es `[0.4764, 0.3727]` (1e-3) y `centroides` es `[[0.252, -1.423], [1.271, 0.903], [-1.523, 0.520]]` (1e-3).
- **Voz real** (con `desde_archivos(cfg, microfono=False)`): para cada clase de `etiquetas_voz(cfg)`, el primer clip (en el orden de `archivo`) de las filas de `data/voz/dataset/metadatos.csv` con `particion == "test"`, `origen == "notebook"` y esa `clase` para el que `reconocer` de V4 decide esa misma clase. Con esos cuatro clips, por `comando_audio`: `elegir` `id2_v039 (reservada)` → reconocer → `AUTORIZADO` → identificar → `"Identidad Sintética #2"` → el de `otro` → regla 8 → salir → `cerrar == True`. Cada `Pantalla.comando` es la clase del clip. Falla si para alguna clase no hay ningún clip así.
- **Push-to-talk con audio real:** el clip de "reconocer" de arriba, remuestreado a 48 kHz, se pasa al callback de un `Microfono` con la fábrica falsa en bloques de 480 muestras, entre 1 s antes y 1 s después de ruido gaussiano de σ = 1e-4 (semilla fija), con `apretar()` justo antes del clip y `soltar()` justo después; con `actualizar()` después del post-roll, `reconocer_voz` recibe la ventana de 4.3 y `Pantalla.comando` es igual al `comando` de `reconocer` de V4 aplicado directamente a esa ventana (el cableado no cambia nada).
- **Tiempos** (en la notebook): `elegir` de una reservada (H2) tarda menos de `metas.respuesta_max_s`, y `comando_audio` en `HUELLA_CARGADA` con el clip de "reconocer" de arriba, remuestreado a 48 kHz y completado con el mismo ruido hasta 3,4 s (el clip más largo del push-to-talk), tarda menos de `metas.respuesta_max_s − app.post_roll_s` (la mediana de 5 corridas, cargando la huella otra vez antes de cada una).

## 7. Verificación manual (no va en pytest; la hace Santiago y se anota en el Documento 2)

- Correr `venv\Scripts\python -m src.app` con los modelos regenerados y revisar: los tres paneles de 4.7, la lista de 36 huellas, que el medidor de nivel se mueva al hablar, el aviso "● Grabando" al apretar el botón o la barra espaciadora, y el corte solo a los 3 s si se la mantiene apretada.
- **Pre-roll:** decir "salir" apretando el botón justo al empezar a hablar (o un poco tarde): la "s" no se corta y el comando se reconoce. Comparar con `scripts/probar_voz.py`, que abría el micrófono después del aviso.
- **Los seis escenarios del plan** en la app, con la voz: autorizada, intrusa, cambio de huella después de identificar, comandos sin huella y ruido o una palabra cualquiera ("no entendí"), "Salir", y la prueba con alguien que no está en el dataset (un familiar, solo con código). Además: una variante nueva de cada identidad y un intruso nuevo; "Abrir archivo…" con un PNG roto (regla 10); `--sin-microfono` con "Abrir audio…".
- Capturas de cada caso para el informe (`docs/figuras/app_*.png`) y el registro de la sesión (`data/app/registro_*.csv`).
- Un recorrido completo de la demo sin fallas (los 3 seguidos son de la Fase 6).
- Anotar en la ficha 3.9 del Documento 2 los tiempos medidos, lo que se vio y lo que falló.

## 8. Aclaraciones y valores medidos en el prototipo (25/10/2026)

Medidos en la nube de Claude Code (Linux, Python 3.12, las versiones de `requirements.txt`), con el dataset de huellas regenerado con `src.huellas.dataset`, `src.huellas.caracteristicas` y `src.huellas.clasificacion`: la lámina `grilla_h1.png` salió idéntica byte a byte a la del repo y H4 imprimió lo mismo que en la notebook (τ\* = 1,5613, pureza 1, test con identificación 100 %, FAR 0 % y FRR 4,2 %). Los scripts y sus salidas están en `docs/respaldo/prototipos/f5_2026-10-25/`. **La voz no se pudo medir ahí** (los audios crudos están solo en la notebook): la aceptación de voz se escribió sin valores medidos y la mide la primera corrida en la notebook.

- **Reservadas para la demo** (distancia normalizada contra τ\* = 1,5613): `id1_v039` 1,2867, `id1_v040` 0,6012, `id2_v039` 1,0351, `id2_v040` 1,5177, `id3_v039` 0,9906 e `id3_v040` 1,4872. `id2_v040` e `id3_v040` quedan con poco margen (0,044 y 0,074): para la demo conviene mostrar `id1_v040`, `id2_v039` e `id3_v039`.
- **Variantes nuevas** (v1000 a v1029 de cada identidad, 90 en total): las 90 se aceptan con su identidad. Distancia mediana 0,782 / 0,921 / 0,893 y máxima 1,456 / 1,393 / 1,554; 2 de la identidad 3 quedan a menos de 0,1 de τ\*. Las 10 primeras: identidad 1, `0.722 1.231 1.205 0.823 1.456 0.772 1.198 0.796 0.709 0.886`; identidad 2, `0.734 0.711 0.840 1.005 0.988 0.894 0.743 0.700 0.815 0.947`; identidad 3, `0.659 0.799 1.345 0.509 0.674 1.391 0.740 0.852 1.170 1.085`.
- **Intrusos nuevos** (1000 a 1029, variante 1): ninguno aceptado; el más cercano a 1,797 (mediana 2,559). Los 10 primeros: `2.872 2.538 2.973 2.396 3.094 7.233 1.858 2.405 3.512 2.244`.
- **Imágenes que no son huellas:** fondo liso (245) y negra: `ValueError` "No se encontró una huella en la imagen (no hay crestas)"; 50 × 50: `ValueError` por el tamaño; `float64`: `ValueError` por el tipo (las cuatro van a la regla 10). Ruido uniforme: H2 la acepta y H4 la rechaza (distancia 65,0): `NO_AUTORIZADO`, que es lo correcto.
- **Receta:** la variante 39 de la identidad 1 rehecha con la receta de 0.3 es idéntica a `test/id1_v039.png`.
- **Tiempos en la nube** (más lenta que la notebook): `generar_base` 1,1–1,5 s; una variante, 17 ms; H2 (`preprocesar`) 339 ms; H3 + H4, 67 ms; `reconocer` de H4 completo, 380–400 ms. En la notebook, H2 tardaba 110 ms y H3 21 ms (traspaso). Con H2 al cargar la huella, la respuesta a "Reconocer" queda en el post-roll (150 ms) más V2 a V4 (unos 10 ms en V4) más H3 y H4 (unos 25 ms): bastante menos que el segundo de `metas.respuesta_max_s`.
- **Visor PCA 2D de train** (72 filas, 374 columnas): PC1 47,64 % y PC2 37,27 % (84,91 % entre las dos); centroides proyectados `[[0.252, -1.423], [1.271, 0.903], [-1.523, 0.520]]` (identidades 1, 2 y 3).
- **El SHA256 de `caracteristicas.npz`** que da la nube no es el de la notebook (`c3233066…`), aunque los valores que imprimen H3 y H4 son iguales: diferencias de redondeo de la plataforma. Por eso la aceptación compara números con tolerancia y no archivos.
- **Por qué un controlador sin Tk:** los escenarios del plan y el manejo de errores se prueban con pytest sin abrir ventanas ni micrófono; la ventana solo dibuja `Pantalla` y se revisa a mano. El micrófono se prueba con una fábrica falsa que imita `sounddevice.InputStream`.
- **Por qué el búfer circular:** con `probar_voz.py` todas las tomas arrancaban con unos 50 ms casi en silencio, porque el micrófono se abría después del aviso. Con el micrófono abierto todo el tiempo y el pre-roll de 250 ms, lo que se dijo justo antes de apretar ya está grabado; el post-roll de 150 ms evita cortar el final al soltar apurado (riesgo b de la revisión de Gemini, adoptado en el plan final de voz).
- **Por qué "Salir" no pide confirmación** (decisión de Santiago, 25/10): la tabla 2.5 no cambia. El riesgo conocido es "salida" tomada como "salir" (1 de 31 `otro` en la prueba en vivo); si pasa en la demo, la app se vuelve a abrir en unos segundos con los modelos guardados.
- **Por qué arrancan en 1000** las variantes e intrusos nuevos: el dataset usa las variantes 1 a 40 y los intrusos 1 a 60, y la muestra ampliada del prototipo de H4 (con la que se eligió el criterio del umbral) usó las variantes 41 a 140 y los intrusos 61 a 160. Desde 1000, ninguna huella nueva se usó para entrenar, calibrar ni elegir nada.
