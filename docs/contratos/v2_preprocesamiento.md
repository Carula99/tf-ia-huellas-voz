# Contrato · V2 · Preprocesamiento de voz

**Versión 1 · 23/10/2026.** Es lo que verifica `tests/test_v2.py`. Los tests se escriben en otro chat solo a partir de este archivo y se congelan (commit) antes de implementar. Se apoya en el contrato de la Fase 0 (`config.yaml`, `src/config.cargar_config` y las claves `voz.*`) y en el de V1 (`remuestrear`, `leer_audio`, `envolvente_db`, `detectar_palabras` y el dataset de `data/voz/dataset`). Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

**Qué hace V2:** de cada audio deja solo la palabra. La busca con el detector de V1, la recorta con un margen, la filtra (pasabanda y pre-énfasis) y la lleva a un nivel fijo. Si no hay una palabra de 0,3 a 2,5 s, marca el audio como no válido. A cada clip de train le suma copias aumentadas: cambio de velocidad y ruido real de train sumado antes de buscar la palabra, como pasa en vivo. La misma función `preprocesar` se usa en la demo, sin aumento. No calcula características: eso es V3.

## 0. Convenciones

- **Audio en memoria:** `np.ndarray` de 1 dimensión. Lo que entra viene de V1 o del micrófono, en [−1, 1]. Lo que sale es `float32` y puede pasar de ±1 por el pre-énfasis y la normalización: no se recorta. Los cálculos intermedios van en `float64`.
- **Frecuencia de trabajo:** `fs = voz.frecuencia_muestreo` (16000). `preprocesar` acepta cualquier frecuencia: si no es `fs`, primero remuestrea con `remuestrear` de V1. Los tiempos están en segundos del audio a `fs` y se pasan a muestras con `round(t·fs)`.
- **Palabra principal:** `detectar_palabras` de V1 (con `voz.dataset.deteccion`) sobre el audio **tal como llega, sin filtrar**. Si da más de un segmento, la palabra es el que tiene la trama más fuerte: los niveles son los de `envolvente_db` con `trama_ms` y `avance_ms` del detector, y una trama `k` es de un segmento si su inicio `k·H/fs` cae en `[inicio_s, fin_s)`. Si dos segmentos empatan, gana el primero. En vivo, un golpe o una voz de fondo que pasó la regla de dominancia no se suma a la palabra.
- **Válido:** hay palabra principal y dura (`fin_s − inicio_s`) dentro de `voz.duracion_palabra_s` (intervalo cerrado). Si no, el motivo es `sin_voz` (ningún segmento) o `duracion` (fuera del rango). No hay regla de borde: una palabra pegada al principio o al final del audio es válida (en vivo es común soltar el botón apenas termina la palabra).
- **Acondicionamiento**, sobre el audio entero y antes de recortar:
  1. Pasabanda: `scipy.signal.butter(orden_filtro, voz.pasabanda_hz, btype="bandpass", fs=fs, output="sos")` aplicado con `scipy.signal.sosfiltfilt` (hacia adelante y hacia atrás: fase cero, sin corrimiento en el tiempo; con los parámetros por defecto).
  2. Pre-énfasis con `a = voz.preenfasis`: `z[0] = x[0]` y `z[n] = x[n] − a·x[n−1]`.
- **Recorte:** con `m0 = round(inicio_s·fs)`, `m1 = round(fin_s·fs)` y `M = round(margen_ms·fs/1000)` (640 con 40 ms), el clip es `x[m0 − M : m1 + M]`, con ceros donde se sale del audio. Siempre tiene `m1 − m0 + 2M` muestras y la palabra ocupa `[M, M + m1 − m0)`.
- **Normalización:** el clip se multiplica por `10^(nivel_rms_db/20) / rms`, con `rms` medido solo sobre la palabra (`clip[M : M + m1 − m0]`, sin los márgenes). Así el nivel no depende del margen ni de los ceros.
- **Aumento** (solo train): cambio de velocidad como el de una cinta (más rápido y más agudo a la vez, con un solo remuestreo) y ruido real de los clips de ruido de train, con una relación señal-ruido medida sobre la palabra. El ruido se suma **antes** de buscar la palabra: la copia pasa por `preprocesar` igual que un audio en vivo, así que el detector puede recortarla distinto que al original, o declararla no válida.
- **Semillas:** cada copia tiene su generador, `np.random.default_rng([semilla_global, zlib.crc32(archivo.encode("utf-8")), copia])`, con `archivo` el del clip de V1 (ej. `train/nb_p01_identificar_01.wav`). No depende del orden en que se procesan los clips.

## 1. Entrada (V2 la lee y nunca la modifica)

El dataset de V1 en `origen` (por defecto `RAIZ / rutas.voz / "dataset"`): `metadatos.csv` y los WAV de 16 kHz que nombra. De cada fila V2 usa `archivo`, `particion`, `origen`, `codigo`, `hablante`, `clase` y `detalle`; las demás columnas se ignoran (pueden faltar). Los ruidos para el aumento son los clips con `origen = ruido` y `particion = train`, ordenados por `archivo` (los de val y test nunca se usan).

## 2. Claves nuevas de config.yaml

Se suman a las de la Fase 0 y de V1. `cargar_config` también las exige: si falta alguna, lanza `ValueError` con su camino completo.

```yaml
voz:
  preprocesamiento:
    margen_ms: <float en [20, 100]>            # silencio que se deja a cada lado de la palabra
    orden_filtro: 4                            # fijo
    nivel_rms_db: <float en [-35, -15]>        # nivel de la palabra, en dBFS
    aumento:
      copias: <entero en [0, 5]>               # copias aumentadas por clip de train
      velocidad: [<mín>, <máx>]                # 0.8 ≤ mín ≤ 1 ≤ máx ≤ 1.2
      snr_db: [<mín>, <máx>]                   # 0 ≤ mín ≤ máx ≤ 40
```

**Valores de config.yaml real:** `margen_ms: 40`, `nivel_rms_db: -26.0`, `copias: 2`, `velocidad: [0.92, 1.08]` y `snr_db: [10.0, 25.0]`. −26 dBFS es el nivel al que se suele llevar la voz en telefonía (medido según UIT-T P.56) y deja lugar a los picos: medido en el prototipo, el pico de la mediana de los clips queda en −6,5 dBFS y el 1 % pasa apenas de 0. Con 0,92 a 1,08, el tono se mueve entre −1,4 y +1,3 semitonos.

## 3. src/voz/preprocesamiento.py

```python
MOTIVOS_V2: tuple[str, ...] = ("sin_voz", "duracion")

@dataclass(frozen=True)
class Preprocesado:
    y: np.ndarray            # el clip: palabra con margen, acondicionada y normalizada (float32);
                             # 0 muestras si no es válido
    valido: bool
    motivo: str              # "" si es válido; si no, uno de MOTIVOS_V2
    inicio_s: float | None   # la palabra principal en el audio de entrada (a fs);
    fin_s: float | None      # None si no hay ningún segmento
    n_segmentos: int         # cuántos segmentos dio detectar_palabras

def palabra_principal(y: np.ndarray, fs: int, cfg: dict) -> tuple[tuple[float, float] | None, int]:
    # (inicio_s, fin_s) de la palabra principal (sección 0), o None, y la cantidad de segmentos.

def pasabanda(y: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    # Paso 1 del acondicionamiento. Mismo largo; float64.

def preenfasis(y: np.ndarray, a: float) -> np.ndarray:
    # Paso 2. Mismo largo; float64.

def recortar(x: np.ndarray, fs: int, inicio_s: float, fin_s: float, cfg: dict) -> np.ndarray:
    # Recorte de la sección 0, con ceros afuera del audio; float64.

def normalizar_rms(clip: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    # Normalización de la sección 0: la palabra es clip[M : len(clip) − M]. float32.
    # ValueError si la palabra es toda ceros.

def preprocesar(y: np.ndarray, fs: int, cfg: dict) -> Preprocesado:
    # 1. Si fs no es voz.frecuencia_muestreo, remuestrea (V1).
    # 2. Palabra principal sobre el audio sin filtrar. Si no hay: no válido, "sin_voz".
    #    Si dura fuera de voz.duracion_palabra_s: no válido, "duracion" (con inicio_s y fin_s).
    # 3. preenfasis(pasabanda(audio entero)), recortar y normalizar_rms.
    # Un audio a fs vacío, en silencio o más corto que una trama no da error: es "sin_voz".

def generar_prepro(cfg: dict, origen: str | os.PathLike | None = None,
                   destino: str | os.PathLike | None = None,
                   sobrescribir: bool = False) -> list[dict]:
    # Escribe los WAV y metadatos.csv del dataset preprocesado en destino (sección 5) y
    # devuelve sus filas como dicts. destino por defecto: RAIZ / rutas.voz / "prepro".
```

## 4. src/voz/aumento.py

```python
def semilla_copia(cfg: dict, archivo: str, copia: int) -> np.random.Generator:
    # np.random.default_rng([semilla_global, zlib.crc32(archivo.encode("utf-8")), copia])

def cambiar_velocidad(y: np.ndarray, factor: float) -> np.ndarray:
    # Como una cinta: con factor > 1 dura factor veces menos y suena factor veces más agudo.
    # factor se redondea a 2 decimales; d = round(100·factor), g = gcd(100, d) y
    # scipy.signal.resample_poly(y, 100 // g, d // g) en float64. float32.
    # Con factor 1 devuelve una copia.

def sumar_ruido(y: np.ndarray, ruido: np.ndarray, snr_db: float,
                palabra: tuple[float, float], fs: int) -> np.ndarray:
    # y + g·ruido[:len(y)], con g tal que 10·log10(Ps / (g²·Pn)) = snr_db.
    # Ps: media de y² en la palabra, [round(inicio·fs), round(fin·fs)); Pn: media de ruido[:len(y)]².
    # float32. ValueError si len(ruido) < len(y) o si Ps o Pn es 0.

def sortear_aumento(cfg: dict, rng: np.random.Generator, n_ruidos: int) -> dict:
    # Sortea en este orden, con los rangos de voz.preprocesamiento.aumento:
    #   "factor": round(rng.uniform(*velocidad), 2)
    #   "snr_db": rng.uniform(*snr_db)
    #   "ruido": int(rng.integers(n_ruidos))      índice en la lista de ruidos
    #   "desplazamiento": rng.random()            dónde empieza el tramo de ruido, en [0, 1)

def aumentar(y: np.ndarray, fs: int, palabra: tuple[float, float],
             ruidos: list[np.ndarray], params: dict, cfg: dict) -> np.ndarray:
    # La copia aumentada de un clip de V1, sin preprocesar (float32):
    # 1. z = cambiar_velocidad(y, factor); la palabra pasa a (inicio/factor, fin/factor).
    # 2. Si z tiene más de C = round(voz.dataset.clip_max_s·fs) muestras, se queda con C centradas en la palabra:
    #    c = round((inicio + fin)/2·fs), i0 = mín(máx(0, c − C // 2), len(z) − C), z = z[i0 : i0 + C],
    #    y la palabra se corre i0/fs.
    # 3. r = ruidos[ruido], off = floor(desplazamiento·(len(r) − len(z) + 1)) y
    #    sumar_ruido(z, r[off:], snr_db, palabra, fs).
    # ValueError si el ruido elegido es más corto que z.
```

`palabra` es la de `preprocesar` sobre el clip original (copia 0). El ruido se mide contra esa palabra, ya movida por el cambio de velocidad; después la copia pasa entera por `preprocesar`, que busca la palabra de nuevo sobre el audio con ruido.

## 5. Dataset preprocesado (`generar_prepro`)

**Qué se procesa:** cada fila de `metadatos.csv` de V1, en su orden, da la copia 0 (`preprocesar` del clip, sin aumento). Si la fila es de train, no es de `origen = ruido` y su copia 0 es válida, da además las copias `1 … copias`: `params = sortear_aumento(cfg, semilla_copia(cfg, archivo, k), len(ruidos))`, `aumentar(...)` con la palabra de la copia 0 y `preprocesar` del resultado. val y test no tienen copias, y los clips de ruido tampoco (ruido sobre ruido).

**Archivos** (relativos a `destino`): solo los clips válidos, en `{particion}/{nombre}_k{copia}.wav`, con `{nombre}` el del WAV de V1 sin extensión (ej. `train/nb_p01_identificar_01_k0.wav`, `train/nb_p01_identificar_01_k2.wav`). WAV mono de `fs`, con muestras `float32` (`subtype="FLOAT"`): se lee de vuelta exactamente igual, sin recortar los picos que pasan de ±1.

**metadatos.csv** en `destino`: UTF-8 sin BOM, separado por comas, fin de línea `\n`, encabezado igual a `COLUMNAS_V2` y una fila por clip y copia, válidos y no válidos, en el orden de V1 y después por copia. Las columnas son las de `COLUMNAS_V2`, en `src/voz/preprocesamiento.py`:

```python
COLUMNAS_V2: tuple[str, ...] = (
    "archivo", "particion", "origen", "codigo", "hablante", "clase", "detalle", "fuente",
    "copia", "valido", "motivo", "n_segmentos", "palabra_inicio_s", "palabra_fin_s",
    "duracion_palabra_s", "muestras", "factor", "snr_db", "ruido",
)
```

- `archivo`: el WAV de V2, vacío si no es válido. De `particion` a `detalle`: copiados de V1. `fuente`: el `archivo` de V1.
- `copia`: 0 sin aumento, de 1 a `copias` las aumentadas. `valido`: `1` o `0`. `motivo`: vacío si es válido.
- `n_segmentos`, `palabra_inicio_s` y `palabra_fin_s`: los de `preprocesar` (la palabra en el clip de V1 o en la copia aumentada, antes de recortar); vacíos si no hay palabra. `duracion_palabra_s = fin − inicio`, también vacía si no hay palabra. En las copias es la duración de la palabra ya acelerada o frenada, la que va a usar V3.
- `muestras`: largo del WAV (0 si no es válido).
- `factor`, `snr_db` y `ruido` (el `archivo` de V1 del ruido usado): solo en las copias; vacíos en la copia 0.
- Redondeo: `palabra_inicio_s`, `palabra_fin_s` y `duracion_palabra_s` con 3 decimales (la duración, redondeando la resta sin redondear), `factor` y `snr_db` con 2.
- En lo que devuelve la función: `copia`, `n_segmentos` y `muestras` son `int`, `valido` es `bool`, las columnas con decimales son `float` con el mismo redondeo, el resto `str`, y lo vacío del CSV es `None`.

**Otras reglas:**

- `ValueError` si falta `origen/metadatos.csv`, si falta un WAV que nombra, si un WAV no es de `fs`, o si hay clips que llevan copias (`copias > 0`) y no hay ningún ruido de train.
- Determinista: generar dos veces da los mismos WAV byte a byte y el mismo CSV.
- Si `destino` ya tiene algún WAV en `train/`, `val/` o `test/`, o un `metadatos.csv`, y `sobrescribir` es `False`: `FileExistsError` sin tocar nada (ni crear carpetas). Con `sobrescribir=True` borra solo esos archivos y vuelve a generar.
- **Línea de comandos:** `venv\Scripts\python -m src.voz.preprocesamiento [--sobrescribir]` genera el dataset con `config.yaml` e imprime los clips válidos y no válidos por partición, clase y copia (0 o aumentada), los no válidos por motivo y cuántos vectores tiene train.

## 6. Qué significa `valido` en V3, V4 y la demo

- **V3** calcula características solo de los clips válidos.
- **V4** entrena el K-NN solo con las filas de train con `valido = 1` (las copias incluidas; en la validación dejando una persona afuera, las copias salen con su hablante).
- **Evaluación (val, test y en vivo):** un audio no válido se clasifica directamente como `otro` ("no entendí"), sin pasar por el K-NN. Cuenta como acierto si la clase real es `otro` y como error si es un comando. En la demo se responde "no entendí".

Esto se repite en los contratos de V3 y V4.

## 7. Criterios que verifican los tests

### 7.1 Configuración

- Las claves de la sección 2 existen, con su tipo y rango; `orden_filtro` vale 4. `cargar_config` lanza `ValueError` que nombra el camino completo si falta cualquiera.
- En `config.yaml` real, los valores de la sección 2.

### 7.2 Palabra principal y validez (`palabra_principal`, `preprocesar`)

Con palabras sintéticas (por ejemplo, un tono con armónicos y rampas de 20 ms) dentro de 3 s a 16 kHz:

- Una palabra de 0,8 s: válida, un segmento, `inicio_s` y `fin_s` iguales a los de `detectar_palabras`.
- Dos palabras separadas por 0,6 s, la segunda 3 dB más fuerte: la principal es la segunda y `n_segmentos = 2`. Dos palabras iguales (la misma señal, con la segunda corrida un múltiplo exacto del avance del detector, para que sus tramas den los mismos niveles): la primera.
- Silencio (ceros), un audio vacío y uno de 100 muestras: no válidos, `sin_voz`, `y` con 0 muestras, `inicio_s` y `fin_s` en `None`, sin errores.
- Una palabra de 0,2 s y otra de 2,8 s con cierres internos de 0,1 s (como en V1): no válidas, `duracion`, con `inicio_s` y `fin_s`.
- Una palabra que termina en la última muestra del audio, y otra que empieza en la primera: válidas; las `M` muestras de afuera del audio son ceros en el clip.
- **Sin filtrar antes de detectar:** una "s" (ruido por encima de 4,5 kHz, 150 ms) seguida sin pausa de un tono con armónicos de 300 ms, con el mismo pico: `inicio_s` a ≤ 30 ms del inicio de la "s".
- La misma palabra a 44,1 o a 48 kHz da lo mismo que remuestreada antes con `remuestrear`.

### 7.3 Acondicionamiento, recorte y nivel

- `pasabanda`: igual a `sosfiltfilt` con el `butter` de la sección 0 (tolerancia 1e-9). Con senos de 3 s (medidos lejos de los bordes): entre 300 y 2500 Hz, ganancia a ±0,3 dB de 0 (en 3000 Hz la teórica ya es −0,29 dB: no sirve como punto de prueba); en 80 y 4000 Hz, −6 ± 0,5 dB; en 40 Hz y en 6 kHz, menos de −40 dB. Fase cero: un pulso de un tono de 1 kHz queda en el mismo lugar (el máximo de la correlación cruzada con el original, a desplazamiento 0 ± 1 muestra).
- `preenfasis`: la fórmula de la sección 0; con `a = 0` devuelve lo mismo.
- `recortar`: `m1 − m0 + 2M` muestras, con la palabra en `[M, M + m1 − m0)` y ceros fuera del audio.
- `normalizar_rms`: el RMS de la palabra queda en `nivel_rms_db` ± 0,01 dB, para clips de cualquier nivel; los márgenes no cambian el factor. `ValueError` con una palabra toda ceros.
- `preprocesar` de un audio válido es `normalizar_rms(recortar(preenfasis(pasabanda(audio)), …))` (tolerancia 1e-6) y es `float32`.
- Multiplicar el audio de entrada por 0,1 o por 3 da el mismo clip (tolerancia 1e-5), mientras la palabra siga por encima de `voz.dataset.deteccion.nivel_min_db`.

### 7.4 Aumento (`aumento.py`)

- `cambiar_velocidad`: un seno de 440 Hz de 1 s con factor 0,92 y 1,08 queda en `440·factor` ± 2 Hz (pico de la FFT) y con `len(y)/factor` ± 2 muestras; coincide con `resample_poly` con los factores de la sección 4 (tolerancia 1e-6); 1,083 se trata como 1,08; con factor 1, una copia igual.
- `sumar_ruido`: la relación medida sobre la palabra (Ps contra la potencia del ruido sumado) es `snr_db` ± 0,01 dB; fuera de la palabra el audio es `y + g·ruido`. `ValueError` con un ruido más corto que `y`, con un ruido de ceros y con una palabra de ceros.
- `sortear_aumento`: con el mismo generador da lo mismo; en 1000 sorteos, `factor` tiene 2 decimales y queda en `velocidad`, `snr_db` en `snr_db`, `ruido` entre 0 y `n_ruidos − 1` (todos aparecen) y `desplazamiento` en [0, 1).
- `semilla_copia`: misma semilla para el mismo archivo y copia; distinta para otra copia o para otro archivo.
- `aumentar`: igual a componer a mano los pasos de la sección 4; con factor < 1 sobre un clip de 3 s la salida tiene exactamente `C` muestras y contiene la palabra; con factor ≥ 1, `len(cambiar_velocidad(y, factor))` muestras. `ValueError` con un ruido más corto que la copia.

### 7.5 Dataset preprocesado (`generar_prepro`, con un dataset de V1 armado en `tmp_path`)

El dataset de prueba tiene un `metadatos.csv` con las 7 columnas de la sección 1 y WAV de 16 kHz: comandos y `otro` (con palabras sintéticas) en train, val y test, un `otro` sin palabra (silencio) y uno con una palabra de 2,8 s, y dos o tres clips de ruido de train y uno de val, con una configuración con `copias = 2`.

- Una fila por clip y copia, en el orden de la sección 5, con el encabezado `COLUMNAS_V2`. El CSV coincide con lo que devuelve la función (con los tipos de la sección 5).
- Copias: exactamente `copias` por clip de train válido que no es de ruido, y ninguna en val, en test, en los ruidos ni en los clips no válidos. Los `ruido` de las copias son clips de ruido de train; nunca el de val.
- Se escribe exactamente un WAV por fila válida, con el nombre de la sección 5, y ningún otro archivo salvo `metadatos.csv`. Cada WAV es de 16 kHz, mono y `FLOAT` (`soundfile.info`), con `muestras` muestras, y es igual a `preprocesar` del clip de V1 (copia 0) o de `aumentar` con los `params` de su semilla (copias), con tolerancia 1e-7.
- Las filas no válidas tienen `archivo` vacío, `muestras = 0` y su motivo; el silencio es `sin_voz` y la palabra de 2,8 s, `duracion`.
- `duracion_palabra_s` coincide con `palabra_fin_s − palabra_inicio_s` (± 0,001).
- Reproducible: generar dos veces en carpetas distintas da WAV idénticos byte a byte y el mismo CSV. Cambiar el orden de las filas del CSV de V1 no cambia ningún WAV.
- `ValueError` con un WAV que falta, con un WAV de 44,1 kHz y sin ruidos de train (con `copias = 2`); con `copias = 0` no hacen falta ruidos.
- Si ya hay datos: `FileExistsError` sin `sobrescribir` y sin cambiar nada; con `sobrescribir=True` vuelve a generar y no borra nada que no sea un WAV de `train/`, `val/` o `test/` o `metadatos.csv`.

### 7.6 Aceptación (`@pytest.mark.aceptacion`, con los datos reales)

Se saltea con `pytest.skip` si no existe `data/voz/dataset/metadatos.csv`. Genera el dataset preprocesado completo con el `cfg` real en `tmp_path`, en menos de 5 minutos en la notebook:

- Cada clip de V1 tiene su copia 0, y los de train que no son de ruido y son válidos, sus 2 copias.
- Todos los clips de comandos son válidos en la copia 0. En al menos el 97 % de ellos, `palabra_inicio_s` y `palabra_fin_s` están a ≤ 0,011 s de los de V1, que están en las columnas del mismo nombre de `data/voz/dataset/metadatos.csv` (en segundos, relativos al clip; en los comandos V1 garantiza un solo segmento). V1 midió antes de guardar el WAV de 16 bits; leído de vuelta, el redondeo puede mover una trama.
- Los no válidos de la copia 0 son todos de `otro`.
- Al menos el 90 % de las copias aumentadas son válidas, y al menos el 85 % de las de cada clase.
- Ninguna copia en val ni en test; todos los `ruido` son clips de ruido de train.
- train tiene al menos 1500 filas válidas.

## 8. Verificación manual (no va en pytest; se anota en el Documento 2)

- Escuchar 20 pares de train (la copia 0 y una aumentada del mismo clip, elegidos al azar con `semilla_global` entre los comandos): la copia 0 tiene la palabra entera y sin silencios largos; la aumentada suena natural (sin sonido metálico) y se entiende. Anotar en cuántas el recorte le pegó un ruido a la palabra.
- Escuchar los clips no válidos de val y test: que no tengan una palabra o que dure fuera de 0,3 a 2,5 s.
- Tiempo de `venv\Scripts\python -m src.voz.preprocesamiento` en la notebook.

## 9. Fuera de V2

- Características (MFCC, Δ, normalización cepstral, tramos, duración) y PCA: V3. V3 usa la palabra sin los márgenes o con ellos según su contrato; la palabra ocupa `[M, len − M)` del clip.
- Experimentos de V4: pasabanda hasta 7500 Hz, CMVN, ΔΔ, pasar por Opus lo que no viene de WhatsApp, con y sin voces sintéticas.
- **Fase 5 (la app):** push-to-talk de hasta 3 s con 250 ms de pre-roll y 150 ms de post-roll; la cadena en vivo es `preprocesar` → V3 → V4, y un audio no válido responde "no entendí".

## 10. Cambios respecto del plan v0.3.2 y la especificación de Gemini (para el Documento 2)

Todo lo de esta sección se midió con un prototipo sobre los 828 clips de V1 (23/10). Para comparar variantes del aumento se usó un borrador de las características de V3 (13 MFCC con librosa, Δ, CMN y 5 tramos) y del K-NN de V4 (K = 5, voto 1/d, confianza 60 %), con validación cruzada dejando afuera a los 10 hablantes de la notebook de train (6 pliegues; 231 comandos y 78 `otro` evaluados) y con ruido de val sumado antes del recorte a 25 y 15 dB para imitar un aula. Ese borrador no es V3 ni V4: solo sirve para comparar.

- **El detector va sobre el audio sin filtrar.** Con el pasabanda antes de detectar, la "s" de "salir" (casi toda por encima de 4 kHz) desaparece: 14 clips de "salir" quedaban por debajo de 0,3 s y la superposición con la palabra de V1 bajaba hasta 0,47. Sin filtrar, todos los comandos son válidos y el 99 % da la misma palabra que V1 (los otros 6 de 580 se mueven por el redondeo a 16 bits del WAV de V1). El pasabanda se aplica después, al audio que va a V3.
- **Palabra principal:** si hay varios segmentos, el más fuerte. En vivo, un golpe o una voz de fondo no se suman a la palabra.
- **Sin regla de borde** (riesgo que marcó Gemini): la palabra pegada al final es válida y el margen se completa con ceros.
- **Ruido antes del recorte, confirmado.** El detector sufre con el ruido sumado: a 20 dB, solo el 27 al 52 % de las copias conserva la palabra entera sin pegarle ruido (con el ruido casi estacionario de la grabación del 17/10 se come los bordes; con el del 20/10, que tiene golpes, le pega hasta 1 s de ruido). Aun así, en la validación cruzada gana sumarlo antes: con ruido de 25 dB en la evaluación, 89,6 % de acierto contra 84,8 % sumándolo después del recorte y 81,8 % sin aumento; con 15 dB, 68,0 % contra 61,9 % y 58,0 %. Sin ruido en la evaluación, 90,5 %, 90,0 % y 87,4 %. El aumento enseña al K-NN las palabras recortadas como las va a recortar el detector en vivo.
- **SNR de 10 a 25 dB** (lo que recomendó Gemini; sin bajar de 8). Con 15 a 30 dB y con 20 a 35 dB los resultados fueron iguales dentro del ruido de la medición (90 a 92 % sin ruido; 88 a 90 % con 25 dB).
- **Sin micrófono ni sala simulados** (el plan y Gemini los proponían: ecualización de ±6 dB por bandas y un eco leve). Se probaron con tres semillas: con la sala, el acierto con ruido de 25 dB bajó en promedio de 89,5 a 87,6 % y el `otro` aceptado subió de 19 a 24 %; sin ruido, igual o peor. Las grabaciones ya traen la sala de cada persona.
- **Velocidad y tono con un solo remuestreo**, como acelerar una cinta (la "perturbación de velocidad" de Ko et al., 2015): sin el phase vocoder, que deja sonido metálico (otro riesgo que marcó Gemini). 0,92 a 1,08 equivale a −1,4 a +1,3 semitonos.
- **2 copias por clip** (decisión de Santiago): con el prototipo, 1121 de 1172 copias válidas (95,6 %; "salir", la más corta, 92,2 %) y 1722 vectores en train, contra 601 sin aumento.
- **Nivel −26 dBFS** en lugar de −20: con −20, el 42 % de los clips pasaba de 0 dBFS por el pre-énfasis. Los WAV de V2 se guardan en `float32`, así que los picos no se recortan.
- **Duración en las copias:** V2 busca la palabra de nuevo en cada copia, así que `duracion_palabra_s` ya es la de la palabra acelerada o frenada (lo que pedía Gemini para V3).
- **Qué significa `valido`** en V3, V4 y la demo (sección 6), como pidió Gemini.
- **Lo que queda en el contrato** (ruido antes del recorte, sin sala), con tres semillas: 91,0 % de acierto sin ruido, 89,5 % con ruido de 25 dB y 69,7 % con 15 dB, con 13, 19 y 31 % de `otro` aceptado.
- **El borrador ya da una idea de V4:** 90 a 91 % de acierto con personas nuevas sin ruido, pero 10 a 15 % de `otro` aceptado sin umbral de distancia. La meta del 5 % depende del umbral de rechazo de V4.
