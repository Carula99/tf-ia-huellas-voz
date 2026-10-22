# Contrato · V1 · Dataset de voz

**Versión 1.1 · 22/10/2026** (la 1 es del mismo día). **Cambios en la 1.1:** se suman las aclaraciones de cómo leen el contrato los tests (sección 11) y los puntos marcados con *(1.1)* en las secciones 6 y 7: el nombre de los clips de ruido, cómo se mide `larga`, la palabra de 2,8 s con cierres, una nota de WhatsApp cortada, la reproducción contra el corte recortado a [−1, 1] y el mínimo por comando contado solo con la notebook. Los pidió el chat que escribió los tests y el cambio se anota en el Documento 2. Es lo que verifica `tests/test_v1.py`. Los tests se escriben en otro chat solo a partir de este archivo y se congelan (commit) antes de implementar. Se apoya en el contrato de la Fase 0 (`config.yaml`, `src/config.cargar_config` y las claves `voz.*`) y en las grabaciones de `data/voz/notebook` (decisiones 33 a 37). Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

**Qué hace V1:** junta los audios de las cuatro fuentes (micrófono de la notebook, audios de celular mandados por WhatsApp, voces sintéticas y ruido ambiente), los pasa a un formato común (WAV de 16 kHz, mono, PCM de 16 bits) con una palabra por archivo, descarta solo lo que está mal (con el motivo anotado) y reparte todo en train, val y test por hablante. No normaliza, no filtra y no recorta la palabra: eso es V2.

## 0. Convenciones

- **Audio en memoria:** `np.ndarray` de `float32` y 1 dimensión, con valores en [−1, 1]. Un audio estéreo se pasa a mono con el promedio de los canales.
- **Frecuencia de trabajo:** `fs = voz.frecuencia_muestreo` (16000). Todo lo que mide V1 (detección, duraciones, picos y niveles) se mide después de remuestrear.
- **Remuestreo:** `scipy.signal.resample_poly(y, up, down)` en `float64`, con `g = gcd(fs_origen, fs_destino)`, `up = fs_destino // g` y `down = fs_origen // g` (44,1 kHz → 160/441; 48 kHz → 1/3; 24 kHz → 2/3), y el resultado pasado a `float32`. Si las dos frecuencias son iguales, devuelve una copia. Siempre se remuestrea el archivo fuente entero y después se corta.
- **Envolvente:** tramas de `trama_ms` con avance de `avance_ms`; en muestras, `N = round(trama_ms·fs/1000)` y `H = round(avance_ms·fs/1000)`. La trama `k` cubre `y[k·H : k·H + N]`, con `k = 0 … K − 1` y `K = 1 + (len(y) − N) // H` (`K = 0` si `len(y) < N`). Su nivel es `20·log10(máx(rms_k, 1e-6))` dBFS, así que el piso es −120 dB.
- **Detector de palabras** (`detectar_palabras`), con las claves de `voz.dataset.deteccion`:
  1. Si no hay tramas, o si `tope = máx(niveles) < nivel_min_db`, no hay palabras.
  2. `piso` = percentil 10 de los niveles (`np.percentile`, interpolación lineal) y `umbral = máx(tope − rango_db, piso + margen_piso_db)`.
  3. Una trama está activa si su nivel es `> umbral`. Dos corridas seguidas de tramas activas se unen si el hueco entre el final de una (`k_fin·H + N`) y el inicio de la otra (`k_ini·H`) dura menos que `pausa_min_s`.
  4. Cada corrida da un segmento `(k_ini·H/fs, mín(k_fin·H + N, len(y))/fs)`, en segundos. Se descartan los que duran menos que `segmento_min_s`.
  5. Se descartan también los segmentos cuya trama más fuerte no llega a `tope − dif_palabra_db`: un ruido de fondo, una voz lejana o un golpe suave no cuentan como palabra.

  Los umbrales son relativos al tope: multiplicar el audio por una constante no cambia los segmentos mientras el tope siga por encima de `nivel_min_db`.
- **Saturación:** un clip está saturado si más de una fracción `saturacion_frac` de sus muestras tiene `|y| ≥ saturacion`. No alcanza con el pico: los audios de WhatsApp vienen normalizados por el celular y muchos llegan a 1,0 en unas pocas muestras sin estar recortados.
- **Hablante:** la persona real detrás de un código. Los códigos de un grupo de `mismo_hablante` son la misma persona, y su hablante es el primer código del grupo (p01 y p02 → p01; p09 y p13 → p09; p14 y p17 → p14). En las voces sintéticas, el hablante es el nombre de la voz (ej. `es-AR-ElenaNeural`); en el ruido, `ruido:<nombre del archivo sin extensión>`. Es el grupo que usa V4 para "dejar una persona afuera".
- **Clases:** los comandos son `voz.comandos` y la clase de rechazo es `voz.clase_rechazo` (`otro`). El ruido ambiente es de la clase `otro`.
- **Carpetas:** `fuente` es la carpeta de los audios crudos (por defecto `RAIZ / rutas.voz`) y `destino` la del dataset (por defecto `RAIZ / rutas.voz / "dataset"`). En los CSV, las rutas son relativas y van con `/`.

## 1. Fuentes (V1 las lee y nunca las modifica)

```text
<fuente>/
├── notebook/                          lo graba scripts/grabar_voces.py
│   ├── sesiones.csv
│   ├── notas.txt                      exclusiones (formato abajo)
│   ├── pXX/pXX_<comando>_<toma:02d>.wav
│   ├── pXX/pXX_otro_<toma:02d>_<detalle>.wav
│   └── ruido/ruido_<fecha>_<hora>.wav
├── whatsapp/                          lo arma scripts/ordenar_whatsapp.py; puede no existir
│   ├── notas.txt                      opcional, mismo formato
│   ├── origen.csv                     de qué archivo original sale cada uno
│   └── pXX/pXX_<clase>_<nn>.<ext>     ext: ogg, opus o m4a
└── tts/                               lo escribe src/voz/tts.py
    └── <voz>/<palabra>_r<k>.mp3
```

- **notebook:** `sesiones.csv` (columnas de `scripts/grabar_voces.py`: archivo, persona, clase, detalle, estilo, distancia_cm, toma, fs, dispositivo, fecha, pico, rms, avisos) es la fuente de las etiquetas, con `archivo` relativo a `notebook/`. Si un archivo aparece en más de una fila, vale la última. Las filas con `clase = ruido` son las grabaciones de ruido ambiente. `ValueError`, nombrando los archivos, si una fila apunta a un archivo que no existe o si hay un `.wav` en `notebook/pXX/` o en `notebook/ruido/` sin su fila.
- **whatsapp:** los audios llegaron como notas de voz de WhatsApp (OGG/Opus a 16 o 48 kHz, de 0,9 a 3,6 s, una palabra cada una) y como archivos de una grabadora del celular (`.m4a`, AAC a 48 kHz, también una palabra cada uno), guardados en una carpeta por comando y sin el nombre de quién los mandó. `scripts/ordenar_whatsapp.py` (una herramienta sin tests, como el grabador) los copia sin modificarlos a `whatsapp/pXX/pXX_<clase>_<nn>.<ext>`: la clase sale de la carpeta del comando y el código, de la hora del nombre de cada nota de voz (una tabla de tandas con horas y códigos, sin nombres) o, en los `.m4a`, del código de quien los grabó. Anota en `whatsapp/origen.csv` el comando y el nombre del archivo original de cada copia; los nombres de las carpetas de cada persona no se escriben en ningún archivo. Si la persona también grabó en la notebook, usa el mismo código. V1 exige que cada `.ogg`, `.opus` o `.m4a` se llame `pXX_<clase>_<nn>.<ext>` (clase: un comando u `otro`; `nn`, dos dígitos) y esté en la carpeta de su código; si no, `ValueError` que lo nombra. Los demás archivos se ignoran. Un audio puede tener una palabra o varias repeticiones (el formato del plan): V1 lo corta igual.
- **tts:** los MP3 de `plan_tts` (sección 5). Si `tts.voces` no está vacía y falta alguno, `ValueError` con la cantidad que falta y la indicación de correr `python -m src.voz.tts`.
- **notas.txt:** cada línea `excluir: <archivo> | <motivo>` excluye ese archivo (relativo a la carpeta del `notas.txt`). En WhatsApp, `<archivo>#<k>` excluye solo el segmento `k` (desde 1). Las demás líneas son comentarios y se ignoran (las notas que ya tiene el archivo quedan como están, y se suma la línea `excluir:` de `p03/p03_identificar_02.wav`). `ValueError` si una exclusión nombra un archivo que no existe.

## 2. Claves nuevas de config.yaml

Se suman a las de la Fase 0. `cargar_config` también las exige: si falta alguna, lanza `ValueError` con su camino completo.

```yaml
voz:
  dataset:
    clip_max_s: 3.0                  # fijo: igual a voz.push_to_talk_max_s
    saturacion: 0.98                 # fijo: nivel que cuenta como tope
    saturacion_frac: <float en (0, 0.05]>   # fracción de muestras en el tope desde la que un clip está saturado
    mismo_hablante: [[p01, p02], [p09, p13], [p14, p17]]   # fijo: códigos que son la misma persona
    particiones:                     # códigos de personas reales (p01 … p99), cada uno en una sola lista
      train: [<códigos>]
      val: [<códigos>]
      test: [<códigos>]
    min_por_comando: <entero ≥ 1>    # clips por comando de cada hablante de la notebook (aceptación)
    deteccion:
      trama_ms: 20                   # fijo
      avance_ms: 10                  # fijo
      rango_db: <float en [20, 40]>
      margen_piso_db: <float en [3, 15]>
      dif_palabra_db: <float en [6, 20]>
      nivel_min_db: <float en [-70, -40]>
      pausa_min_s: <float en [0.15, 0.5]>
      segmento_min_s: <float en [0.05, 0.2]>
      margen_borde_s: <float en [0, 0.1]>
    whatsapp:
      margen_s: <float en [0.1, 0.5]>       # silencio que se deja a cada lado de la palabra
      factor_doble: <float en [1.3, 2.5]>
    ruido:
      clips_por_archivo: <entero ≥ 1>
      particiones:                   # nombres de los archivos de ruido, sin extensión
        train: [<nombres>]
        val: [<nombres>]
        test: [<nombres>]
    tts:
      voces: [<voces de edge-tts, sin repetir>]
      velocidades: [<texto como "-10%" o "+10%">]
      otros: [<palabras de la clase otro; ninguna es un comando>]
```

**Valores de config.yaml real:** detección con `rango_db: 25.0`, `margen_piso_db: 10.0`, `dif_palabra_db: 10.0`, `nivel_min_db: -55.0`, `pausa_min_s: 0.20`, `segmento_min_s: 0.10` y `margen_borde_s: 0.03`; `saturacion_frac: 0.01`, `margen_s: 0.40`, `factor_doble: 1.6`, `clips_por_archivo: 10` y `min_por_comando: 5`; `velocidades: ["-10%", "+10%"]`, `otros: [conocer, salida, identidad]` y 12 voces (mujer y varón de Argentina, Uruguay, Chile, México, España y Colombia): `es-AR-ElenaNeural`, `es-AR-TomasNeural`, `es-UY-ValentinaNeural`, `es-UY-MateoNeural`, `es-CL-CatalinaNeural`, `es-CL-LorenzoNeural`, `es-MX-DaliaNeural`, `es-MX-JorgeNeural`, `es-ES-ElviraNeural`, `es-ES-AlvaroNeural`, `es-CO-SalomeNeural` y `es-CO-GonzaloNeural` (12 × 6 palabras × 2 velocidades = 144 clips, menos que las ~300 tomas reales de train).

**Calibración (22/10, con los audios reales y un barrido de 108 combinaciones):** con los valores de arriba se excluyen solas 8 de las 377 tomas de comandos de la notebook (2,1 %; sin contar la vacía, que va por `notas.txt`), entre ellas dos de las cuatro dudosas, y 6 de los 147 audios de WhatsApp (5 con la palabra pegada al borde, porque la persona soltó el botón antes de terminar, y una de 0,17 s). Con los primeros valores del borrador (`rango_db` 30, `margen_piso_db` 6 y sin `dif_palabra_db`) se excluían 62 tomas, casi todas de la tanda del 20/10, que tiene más ruido de fondo. La fracción más alta de muestras en el tope es 0,23 % en WhatsApp y 0 en la notebook.

**Particiones de config.yaml real:** las 21 sesiones de la notebook son de 18 personas (p01 y p02, p09 y p13, y p14 y p17 son la misma persona). Van 10 a train, 4 a val y 4 a test, con hombres y mujeres y personas de las dos tandas de grabación (17/10 y 20/10) en cada partición. Quedan fijos en train p01 y p02 (Santiago, que hace la demo), los otros dos pares y p08 y p15, que además mandaron WhatsApp. El resto se sorteó con `semilla_global` entre las combinaciones que cumplen esas condiciones:

```yaml
      train: [p01, p02, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p17,
              p22, p23, p24, p25, p26, p27, p28, p29]
      val: [p03, p04, p20, p21]
      test: [p05, p16, p18, p19]
```

Las 8 personas que solo mandaron WhatsApp (p22 a p29) van a train. Ruido: `ruido_20261020_093611` en val, `ruido_20261020_094933` en test y las otras tres en train. La lista de quién es hombre o mujer queda fuera del proyecto (decisión 34): en el repo solo están los códigos.

## 3. src/voz/audio.py

```python
def leer_audio(ruta: str | os.PathLike) -> tuple[np.ndarray, int]:
    # WAV, OGG/Opus (.ogg u .opus) y MP3 con soundfile. Si soundfile no lo puede abrir
    # (los .m4a y algunas notas de voz de WhatsApp que libsndfile rechaza), con PyAV (av):
    # todos los cuadros del primer flujo de audio, decodificados en orden.
    # Devuelve float32 mono en [−1, 1] y la frecuencia del archivo.

def remuestrear(y: np.ndarray, fs_origen: int, fs_destino: int) -> np.ndarray:
    # Sección 0. float32.

def guardar_wav(ruta: str | os.PathLike, y: np.ndarray, fs: int) -> None:
    # WAV mono PCM de 16 bits. Recorta a [−1, 1] antes de escribir y crea las carpetas.
```

## 4. src/voz/segmentacion.py

```python
def envolvente_db(y: np.ndarray, fs: int, trama_ms: float, avance_ms: float) -> np.ndarray:
    # Niveles de la sección 0, float64, uno por trama.

def detectar_palabras(y: np.ndarray, fs: int, cfg: dict) -> list[tuple[float, float]]:
    # Detector de la sección 0, con cfg["voz"]["dataset"]["deteccion"].
    # Segmentos (inicio_s, fin_s) en orden y sin solaparse.

@dataclass(frozen=True)
class Corte:
    inicio_s: float          # la palabra, en el audio entero
    fin_s: float
    muestra_inicio: int      # el clip que se guarda, en muestras del audio entero
    muestra_fin: int

def cortar_repeticiones(y: np.ndarray, fs: int, cfg: dict) -> list[Corte]:
    # Un corte por segmento de detectar_palabras(y, fs, cfg), en orden.
```

**Clip de cada corte**, con `m = whatsapp.margen_s` y `c = clip_max_s`:

1. Se arranca de `[inicio_s − m, fin_s + m]`.
2. No pasa del principio ni del final del audio, ni del punto medio del silencio con la palabra anterior o la siguiente (`(fin_anterior + inicio_s)/2` y `(fin_s + inicio_siguiente)/2`).
3. Si así dura más que `c`, se achica igual de los dos lados: margen `máx(0, (c − (fin_s − inicio_s))/2)`; si la palabra sola dura más que `c`, el clip es la palabra.
4. `muestra_inicio = round(t_inicio·fs)` y `muestra_fin = round(t_fin·fs)`.

Los clips de un mismo audio no se solapan.

## 5. src/voz/tts.py

```python
def plan_tts(cfg: dict) -> list[dict]:
    # Un MP3 por voz, palabra y velocidad, sin sintetizar nada. Claves: voz, clase,
    # palabra, texto, velocidad, k y archivo.

def sintetizar_tts(cfg: dict, destino: str | os.PathLike | None = None,
                   sintetizador: Callable[[str, str, str, Path], None] | None = None,
                   sobrescribir: bool = False) -> list[Path]:
    # Escribe los MP3 del plan que falten en destino (por defecto RAIZ / rutas.voz / "tts").
    # sintetizador(texto, voz, velocidad, ruta) escribe un MP3 en ruta; por defecto usa
    # edge_tts.Communicate(texto, voz, rate=velocidad).save(ruta). Devuelve las rutas escritas.
```

**Reglas:**

- Orden del plan: por voz (como en `tts.voces`), después por palabra (los comandos en el orden de `voz.comandos` y después `tts.otros`) y después por velocidad. `k` es la posición de la velocidad, desde 1.
- `clase` es la palabra si es un comando y `otro` si no. `texto` es la palabra con la primera letra en mayúscula (`"Reconocer"`). `archivo` es `<voz>/<palabra>_r<k>.mp3`, relativo a la carpeta `tts`.
- `ValueError` si una palabra de `tts.otros` es un comando o si hay voces, palabras o velocidades repetidas.
- `sintetizar_tts` llama al sintetizador solo por los MP3 que no existen; con `sobrescribir=True`, por todos. Crea las carpetas.
- **Línea de comandos:** `venv\Scripts\python -m src.voz.tts [--sobrescribir]` baja las voces con internet y antes verifica con `edge_tts.list_voices()` que existan todas las de `config.yaml`.

## 6. src/voz/dataset.py

```python
COLUMNAS: tuple[str, ...] = (
    "archivo", "particion", "origen", "codigo", "hablante", "clase", "detalle",
    "estilo", "distancia_cm", "toma", "velocidad_tts", "fuente", "muestra_inicio", "muestra_fin",
    "fs_fuente", "duracion_s", "n_segmentos", "palabra_inicio_s", "palabra_fin_s", "pico", "rms_db",
)
COLUMNAS_EXCLUIDAS: tuple[str, ...] = ("fuente", "toma", "origen", "codigo", "clase", "motivo", "nota")
ORIGENES: tuple[str, ...] = ("notebook", "whatsapp", "tts", "ruido")
MOTIVOS: tuple[str, ...] = ("whatsapp_val_test", "notas", "saturada", "sin_palabra",
                            "varias_palabras", "cortada", "duracion", "doble", "larga")

def inventario(cfg: dict, fuente: str | os.PathLike | None = None) -> list[dict]:
    # Un candidato por clip posible de las cuatro fuentes, sin escribir nada. Cada uno tiene
    # las claves de COLUMNAS (archivo y particion vacíos si se excluye) más "motivo" y "nota"
    # (vacíos si entra).

def generar_dataset(cfg: dict, fuente: str | os.PathLike | None = None,
                    destino: str | os.PathLike | None = None,
                    sobrescribir: bool = False) -> tuple[list[dict], list[dict]]:
    # Escribe los WAV, metadatos.csv y excluidas.csv en destino. Devuelve las filas de los
    # dos CSV como dicts.

def muestra_revision(filas: list[dict], n: int, semilla: int) -> list[str]:
    # Clips para escuchar a mano (sección 8).
```

**Clips de cada origen** (todos a 16 kHz, cortados del archivo fuente entero ya remuestreado):

- `notebook`: un clip por toma, la toma entera (3 s).
- `whatsapp`: `cortar_repeticiones` sobre el audio entero; un clip por corte.
- `tts`: un clip por MP3, el archivo entero.
- `ruido`: `n = ruido.clips_por_archivo` clips de `C = round(clip_max_s·fs)` muestras que no se solapan, con inicio en `round(k·(L − C)/(n − 1))` para `k = 0 … n − 1` (0 si `n = 1`), con `L` el largo del archivo. `ValueError` si `L < n·C`.

**Control de calidad.** A cada clip se le corre `detectar_palabras` (de ahí salen `n_segmentos`, `palabra_inicio_s` = inicio del primer segmento y `palabra_fin_s` = fin del último). Se excluye con el primer motivo que corresponda, en este orden:

1. `whatsapp_val_test`: es de WhatsApp y su hablante está en val o en test (ver Particiones).
2. `notas`: figura en un `notas.txt`.
3. `saturada`: el clip está saturado (sección 0).
4. Solo los comandos:
   - `sin_palabra`: ningún segmento.
   - `varias_palabras`: más de un segmento.
   - `cortada` (notebook y WhatsApp): la palabra queda a menos de `margen_borde_s` del principio o del final del archivo fuente.
   - `duracion`: la palabra dura fuera de `voz.duracion_palabra_s` (intervalo cerrado).
   - `doble` (WhatsApp, en un audio con 3 cortes o más): la palabra dura más que `factor_doble` veces la mediana de las palabras de ese audio. Es la forma de detectar dos repeticiones pegadas.
5. Solo `otro` de WhatsApp: `larga`, si la palabra sola dura más que `clip_max_s`. La palabra es la del corte (`fin_s − inicio_s` de `cortar_repeticiones`, como en la regla 3 de la sección 4): en ese caso el clip es la palabra entera y el detector no encuentra nada sobre él (sección 11) *(1.1)*.

En `otro` no se exige palabra: una tos, un golpe o el ruido ambiente pueden dar 0 segmentos y entran igual.

**Particiones:**

- Notebook y WhatsApp: la del hablante según `particiones`. `ValueError`, nombrando los códigos, si un código con audios no está en ninguna lista, si está en dos, o si dos códigos de un grupo de `mismo_hablante` quedan en particiones distintas. Puede haber códigos en las listas que todavía no tienen audios.
- val y test son solo del micrófono de la demo. `ValueError` si un hablante sin sesiones en la notebook está en val o en test. Los audios de WhatsApp de un hablante de val o test no se usan en ninguna partición: quedan en `excluidas.csv` con motivo `whatsapp_val_test`. Ponerlos en train dejaría su voz en dos particiones.
- TTS: todas las voces van a train.
- Ruido: la del archivo según `ruido.particiones`. `ValueError` si una grabación de ruido no está en ninguna lista o está en dos.

**Archivos** (relativos a `destino`): `{particion}/nb_{nombre}.wav` (notebook, con el nombre de la toma sin extensión), `{particion}/wa_{nombre}_s{k:02d}.wav` (WhatsApp, corte `k` desde 1), `{particion}/tts_{voz}_{palabra}_r{k}.wav` y `{particion}/ruido_{nombre}_c{k:02d}.wav` (clip `k` desde 1; `{nombre}` es el nombre del archivo sin extensión, así que de `ruido_20261020_093611.wav` sale `ruido_ruido_20261020_093611_c01.wav` *(1.1)*).

**metadatos.csv** en `destino`: UTF-8 sin BOM, separado por comas, encabezado igual a `COLUMNAS` y una fila por WAV. Orden: por origen (como en `ORIGENES`), después por `fuente` (orden de texto) y después por `toma`.

- `origen`: `notebook`, `whatsapp`, `tts` o `ruido`. `codigo`: el código de la fuente (p02); `hablante`: según la sección 0 (p01).
- `clase`: un comando u `otro`. `detalle`: en notebook, el de `sesiones.csv`; en WhatsApp, el comando o `libre`; en TTS, la palabra; en el ruido, `ruido`.
- `estilo` y `distancia_cm`: los de `sesiones.csv` en notebook y vacíos en el resto. `toma`: la de `sesiones.csv` en notebook, `k` en el resto. `velocidad_tts`: solo en TTS.
- `fuente`: el archivo crudo, relativo a `fuente` (ej. `notebook/p05/p05_reconocer_03.wav`). `muestra_inicio` y `muestra_fin`: el clip, en muestras de 16 kHz del archivo fuente remuestreado. `fs_fuente`: la frecuencia del archivo crudo.
- `duracion_s = (muestra_fin − muestra_inicio)/fs`. `palabra_inicio_s` y `palabra_fin_s`, relativos al clip y vacíos si no hay segmentos. `pico` = máx |y| del clip; `rms_db` = `20·log10(máx(rms, 1e-6))` del clip.
- Redondeo: `duracion_s` y `pico` con 4 decimales, `palabra_inicio_s` y `palabra_fin_s` con 3, `rms_db` con 2.
- En lo que devuelve `generar_dataset`: las columnas enteras (`distancia_cm`, `toma`, `muestra_inicio`, `muestra_fin`, `fs_fuente`, `n_segmentos`) son `int`, las de números con decimales son `float` con el mismo redondeo, el resto `str`, y lo vacío del CSV es `None`.

**excluidas.csv** en `destino`: mismo formato, encabezado igual a `COLUMNAS_EXCLUIDAS`, una fila por clip excluido y en el mismo orden. `motivo` es uno de `MOTIVOS`; `nota` es el texto de `notas.txt` o el valor medido (ej. `duración 2,71 s`).

**Otras reglas:**

- Determinista: generar dos veces da los mismos WAV byte a byte y los mismos CSV. V1 no sortea nada.
- Si `destino` ya tiene algún WAV en `train/`, `val/` o `test/`, o un `metadatos.csv` o un `excluidas.csv`, y `sobrescribir` es `False`: `FileExistsError` sin tocar nada (ni crear carpetas). Con `sobrescribir=True` borra solo esos archivos y vuelve a generar.
- `muestra_revision`: si hay al menos `n` clips de comandos de WhatsApp, elige entre esos; si no, entre todos los clips de comandos. Es `random.Random(semilla).sample(sorted(archivos), mín(n, len(archivos)))`.
- **Línea de comandos:** `venv\Scripts\python -m src.voz.dataset [--sobrescribir]` genera el dataset con `config.yaml` e imprime los conteos por partición, clase y origen, las exclusiones por motivo y 30 clips para escuchar (`semilla_global`).

## 7. Criterios que verifican los tests

### 7.1 Configuración

- Las claves de la sección 2 existen, con su tipo y rango; las `fijo`, con su valor exacto. `cargar_config` lanza `ValueError` que nombra el camino completo si falta cualquiera.
- En `config.yaml` real: las tres listas de `particiones` no tienen códigos repetidos ni en común; los tres grupos de `mismo_hablante`, p08 y p15 están en train; de los códigos p01 a p21 (la notebook) hay 10 hablantes en train, 4 en val y 4 en test; los códigos desde p22 (solo WhatsApp) están en train. Las listas de `ruido.particiones` no tienen nombres en común y ninguna está vacía. `tts.otros` no tiene comandos, y el plan de TTS tiene a lo sumo 300 entradas.

### 7.2 Audio (`leer_audio`, `remuestrear`, `guardar_wav`)

- Lee WAV PCM de 16 bits, OGG/Opus con extensión `.ogg` u `.opus` y MP3 (escritos con soundfile en `tmp_path`) y M4A/AAC (escrito con PyAV): `float32`, 1 dimensión y la frecuencia del archivo. Un WAV estéreo da el promedio de los canales. Un seno de 440 Hz guardado como M4A se lee con la misma frecuencia y una duración a ±0,05 s de la original.
- `remuestrear`: 3 s a 44,1, 48 o 24 kHz dan exactamente 48000 muestras a 16 kHz; coincide con `resample_poly` con los factores de la sección 0 (tolerancia 1e-6); un seno de 440 Hz mantiene su frecuencia (pico de la FFT a ±2 Hz) y su amplitud (±2 %); con frecuencias iguales devuelve una copia igual.
- `guardar_wav`: se lee de vuelta como 16 kHz, mono, `PCM_16`, con diferencia ≤ 1/32767; recorta lo que pasa de ±1 y crea las carpetas.

### 7.3 Detector (`envolvente_db`, `detectar_palabras`)

Con palabras sintéticas (por ejemplo, un tono con armónicos de 0,3 a 1,5 s y rampas de 20 ms) dentro de 3 s a 16 kHz:

- `envolvente_db`: `K` tramas según la sección 0; un seno de amplitud `A` da `20·log10(A/√2)` ± 0,1 dB; ceros dan −120.
- Silencio (ceros), ruido gaussiano de −60 dBFS y ruido estacionario de −35 dBFS: ningún segmento.
- Una palabra con pico entre 0,02 y 0,9, en ceros o sobre un ruido al menos 25 dB más bajo que la palabra: un segmento, con inicio y fin a ≤ 30 ms de los reales.
- Dos palabras separadas por `pausa_min_s − 0,08` s dan un segmento; separadas por `pausa_min_s + 0,08` s, dos. Un cierre interno de 80 ms dentro de una palabra no la parte.
- Un golpe aislado de 50 ms, a más de `pausa_min_s` de todo lo demás, no da segmento.
- Dominancia: con una palabra y, más de 0,5 s después, otra 6 dB más baja, hay dos segmentos; si la segunda está 15 dB más baja, uno solo.
- Escala: una palabra de pico 0,3 multiplicada por 0,1 o por 3 da los mismos segmentos.
- Un audio tipo WhatsApp (10 palabras de 0,45 a 1,0 s con pausas de 0,6 a 1,2 s, niveles que varían 6 dB y ruido de fondo de −45 dBFS) guardado como OGG/Opus y leído de nuevo: 10 segmentos, cada borde a ≤ 30 ms del real.

### 7.4 Cortes (`cortar_repeticiones`)

- Un corte por segmento, en orden; `inicio_s` y `fin_s` iguales a los del detector.
- Cada clip contiene su palabra, está dentro del audio, dura a lo sumo `clip_max_s` (con palabras más cortas que `clip_max_s`; si la palabra es más larga, el clip es la palabra, regla 3 de la sección 4 *(1.1)*) y no se solapa con los otros.
- Con pausas largas y lejos de los bordes, el margen es `margen_s` de cada lado (±1 muestra). Con una pausa menor que `2·margen_s`, el clip termina en el punto medio del silencio. Con una palabra de 2,5 s y `margen_s = 0,4`, el clip dura `clip_max_s` y la palabra queda centrada.

### 7.5 Voces sintéticas (`plan_tts`, `sintetizar_tts`)

- Con el `cfg` real, el plan tiene `voces × (comandos + otros) × velocidades` entradas, en el orden de la sección 5, con `archivo`, `clase`, `texto` y `k` según las reglas y sin archivos repetidos.
- `ValueError` con una palabra de `otros` que es un comando y con voces repetidas.
- Con un sintetizador falso (escribe un MP3 corto con soundfile y anota la llamada), en `tmp_path`: una llamada por entrada con `(texto, voz, velocidad, ruta)`, la ruta igual a `destino / archivo`, y devuelve las rutas escritas. Una segunda llamada no sintetiza nada; si se borra un MP3, rehace solo ese; con `sobrescribir=True`, rehace todos. Los tests nunca usan internet.

### 7.6 Inventario y particiones (`inventario`, con una fuente armada en `tmp_path`)

La fuente de prueba tiene notebook (tomas a 44,1 kHz con `sesiones.csv` y `notas.txt`, de un grupo de `mismo_hablante` y de otras dos personas), WhatsApp (notas de una palabra en OGG/Opus a 16 y a 48 kHz, un M4A y un audio con varias repeticiones), TTS (MP3 a 24 kHz del sintetizador falso) y dos grabaciones de ruido, con una configuración reducida.

- Cada toma de `sesiones.csv` da un candidato; cada audio de WhatsApp, uno por palabra; cada MP3, uno; cada ruido, `clips_por_archivo`. Las etiquetas salen de `sesiones.csv` y del nombre del archivo.
- `hablante`: el primer código del grupo para los dos códigos de `mismo_hablante`; la voz en TTS; `ruido:<nombre>` en el ruido.
- Particiones según las listas, TTS en train y ruido según `ruido.particiones`.
- Una persona con sesión en la notebook y audios de WhatsApp, puesta en val: sus tomas de la notebook van a val y todos sus audios de WhatsApp quedan excluidos con motivo `whatsapp_val_test`. En train, los dos van a train.
- `ValueError` (nombrando lo que falla) con: un código con audios que no está en ninguna lista o está en dos; los dos códigos de un grupo en particiones distintas; un código que solo tiene WhatsApp puesto en val o en test; un ruido sin partición; una fila de `sesiones.csv` sin archivo o un `.wav` sin fila; un audio de WhatsApp con un nombre que no sigue el formato; una exclusión de `notas.txt` que nombra un archivo que no existe; un MP3 del plan que falta.
- Motivos: cada uno de `MOTIVOS` aparece en un clip armado para provocarlo (una toma anotada en `notas.txt`, una con el 2 % de las muestras en ±1, una vacía, una con dos palabras del mismo nivel, una con la palabra pegada al final, una nota de WhatsApp con la palabra pegada al final *(1.1)*, una con una palabra de 2,8 s con cierres internos de 0,1 s *(1.1)*, dos repeticiones pegadas en un audio de WhatsApp y un `otro` de WhatsApp de 3,5 s), y los clips buenos no tienen motivo. Una toma con pico 1,0 en el 0,1 % de las muestras no está saturada, y una con un ruido 15 dB más bajo después de la palabra no tiene dos palabras. Con `#k` en `notas.txt` se excluye solo el corte `k`.
- Un `otro` sin segmentos (una tos corta o ruido) entra igual.

### 7.7 Generación (`generar_dataset` con la misma fuente, en `tmp_path`)

- Crea exactamente los WAV de los candidatos que entran, en la carpeta de su partición y con los nombres de la sección 6, más `metadatos.csv` y `excluidas.csv`: en `destino` no queda ningún otro archivo.
- Cada WAV es de 16 kHz, mono y `PCM_16` (`soundfile.info`), dura `duracion_s` y a lo sumo `clip_max_s`.
- `metadatos.csv` tiene el encabezado igual a `COLUMNAS`, una fila por WAV en el orden de la sección 6, y coincide con lo que devuelve la función (con los tipos de la sección 6); lo mismo `excluidas.csv` con `COLUMNAS_EXCLUIDAS`. Entre los dos está cada candidato del inventario exactamente una vez.
- En los comandos, la palabra dura dentro de `voz.duracion_palabra_s` y `n_segmentos = 1`.
- Reproducible: rehacer una fila con `leer_audio(fuente)`, `remuestrear` y el corte `[muestra_inicio, muestra_fin)`, recortado a [−1, 1] como hace `guardar_wav` *(1.1)*, da el WAV con diferencia ≤ 1/32767. Generar dos veces en carpetas distintas da WAV idénticos byte a byte y los mismos CSV.
- Ningún hablante en dos particiones.
- Si ya hay datos: `FileExistsError` sin `sobrescribir` y sin cambiar nada; con `sobrescribir=True` vuelve a generar y no borra nada que no sea un WAV de `train/`, `val/` o `test/`, `metadatos.csv` o `excluidas.csv`.
- `muestra_revision`: determinista con la misma semilla, sin repetidos, solo comandos, y de WhatsApp cuando alcanzan.

### 7.8 Aceptación (`@pytest.mark.aceptacion`, con los datos reales)

Se saltea con `pytest.skip` si no existe `data/voz/notebook/sesiones.csv` o si faltan MP3 del plan. Genera el dataset completo con el `cfg` real en `tmp_path`, en menos de 2 minutos en la notebook:

- Cada toma de `sesiones.csv` y cada audio de `whatsapp/` está en `metadatos.csv` o en `excluidas.csv`. `p03/p03_identificar_02.wav` está excluida con motivo `notas`.
- Las exclusiones automáticas (todos los motivos menos `notas`) son a lo sumo el 5 % de los clips de comandos de la notebook y a lo sumo el 5 % de los de WhatsApp.
- Al menos 10 hablantes reales; los de cada partición, según las listas de `config.yaml`. Ningún clip de WhatsApp en val ni en test, y ninguno excluido con motivo `whatsapp_val_test` (con las particiones reales no se pierde ninguno).
- Cada hablante que grabó en la notebook tiene al menos `min_por_comando` clips de la notebook de cada comando (sus audios de WhatsApp no cuentan) *(1.1)*.
- En cada partición hay hablantes de las dos fechas de grabación de la notebook (columna `fecha` de `sesiones.csv`).
- Cada partición tiene clips de las cuatro clases; TTS solo en train, con menos clips que los reales de train (notebook y WhatsApp).

## 8. Verificación manual (no va en pytest; se anota en el Documento 2)

- Hombres y mujeres en cada partición, con la lista privada.
- Escuchar los 30 clips de `muestra_revision` y anotar cuántos tienen exactamente una palabra (meta: todos). Los que estén mal van a `notas.txt`.
- Revisar `excluidas.csv`: que lo descartado esté realmente mal.
- Escuchar las cuatro tomas dudosas (`p15_identificar_03`, `p21_identificar_02`, `p21_reconocer_04` y `p17_reconocer_02`) y anotar en `notas.txt` las que estén mal.
- Tiempo de `venv\Scripts\python -m src.voz.dataset` en la notebook.

## 9. Fuera de V1

- Recorte de la palabra, pasabanda, pre-énfasis, normalización y aumento de datos: V2. V2 toma el ruido para el aumento solo de los clips de ruido de train.
- Pasar por Opus lo que no viene de WhatsApp (el audio en vivo, la notebook y TTS): experimento de V4.
- Las ideas de la especificación de Gemini que se adopten van en los contratos de V2 a V4.

## 10. Cambios respecto del plan v0.3.2 (para el Documento 2)

- **16 kHz en V1:** el plan lo pone en V1 (paso y test de formato); el diagrama v3 y el docstring de `grabar_voces.py` lo ponían en V2. Queda en V1 y se corrige el docstring.
- **soundfile, y PyAV donde no alcanza, en lugar de ffmpeg:** soundfile lee Opus y MP3 (verificado en la Fase 0), pero no abre los 13 `.m4a` ni 12 notas de voz de una misma tanda. PyAV (`av`, que trae FFmpeg adentro) los decodifica sin instalar un programa aparte; se suma a `requirements.txt`.
- **Los audios de WhatsApp llegaron distintos de lo pedido:** una nota de voz por palabra (en tandas de 3 a 6 por comando) y algunas grabaciones `.m4a`, ordenados por comando y no por persona. Se asignan a códigos por tanda con una planilla y se copian sin tocarlos; V1 corta igual un audio de una palabra o de varias. No hay audios de `otro` por WhatsApp.
- **Detector calibrado con los audios reales:** con los valores de la primera versión se excluían 62 tomas del 20/10 por el ruido de fondo. Se sumó la regla de dominancia (`dif_palabra_db`) y se subió el margen sobre el piso.
- **Saturación por fracción de muestras en el tope**, no por el pico: los audios de WhatsApp vienen normalizados y muchos tocan 1,0 sin estar recortados.
- **metadatos.csv en `data/voz/dataset/`** (el plan decía `metadatos_voz.csv`), igual que en huellas.
- **Mínimo de 5 clips por comando y hablante** (el plan decía 8): cada sesión de la notebook tiene 6 por comando (uno por estilo) y alguna toma se excluye. En total quedan unos 120 por comando, contra 80 del plan (10 personas × 8).
- **Particiones como listas de códigos en `config.yaml`:** se eligen a mano para tener hombres y mujeres en cada una, sin que el sexo de nadie entre al repo.
- **TTS solo en train y ruido repartido por archivo:** las metas de voz se miden con personas reales, y el ruido de val y test nunca se usa en el aumento de datos. Las voces sintéticas no superan a las tomas reales de train.
- **18 personas en la notebook, no 20:** p09 y p13, y p14 y p17, son la misma persona (como p01 y p02). Reparto de 10, 4 y 4 personas (decisión de Santiago), con hombres y mujeres y las dos tandas en cada partición.
- **val y test solo con personas de la notebook**, de las dos tandas: es el micrófono de la demo. Las 8 personas que solo mandaron WhatsApp suman variedad de voces en train. Los WhatsApp de alguien de val o test no se usan (con el reparto elegido, p08, p09 y p15 quedan en train y no se pierde nada).
- **Control de calidad automático con motivo** (la idea del 17/10): se descarta solo lo saturado, vacío, cortado, con dos palabras o de duración fuera de rango, y queda anotado en `excluidas.csv`.

## 11. Cómo leen el contrato los tests (1.1)

Aclaraciones que sumó el chat que escribió `tests/test_v1.py` (22/10/2026). Lo marcado *(1.1)* en las secciones 6 y 7 sale de acá.

- **Límite del detector:** una palabra pareja que ocupa más del 90 % del audio no se detecta: el percentil 10 de los niveles cae dentro de ella y el umbral (`piso + margen_piso_db`) queda por encima del tope. Por eso `larga` se mide con el corte, y la toma de 2,8 s que provoca `duracion` tiene cierres internos de 0,1 s, como el habla. Con los datos reales, una toma hablada casi entera se excluye igual, pero como `sin_palabra`.
- **`cortada` y `doble` en WhatsApp:** los casos de los tests dan lo mismo midiendo la palabra con el corte o con la detección sobre el clip; el contrato no fija cuál.
- **Clips de ruido:** `ruido_{nombre}_c{k:02d}.wav` con el nombre del archivo sin extensión (queda `ruido_ruido_…`).
- **Reproducción (7.7):** el WAV se compara con el corte del audio remuestreado recortado a [−1, 1], porque `guardar_wav` recorta y el remuestreo puede pasar apenas de 1. Al comparar `pico` y `rms_db` con el WAV, el pico se toma recortado a 1 y `rms_db` solo se compara por encima de −60 dBFS (más abajo la cuantización de 16 bits lo mueve unas décimas).
- **Mínimo por comando (7.8):** se cuentan solo los clips de la notebook de cada hablante.
- **Configuración:** donde dice `<float …>` se acepta también un entero (no `bool`). La falta de una clave se prueba con `cargar_config(ruta)` de la Fase 0 sobre una copia de `config.yaml` sin esa clave.
- **No se fijan:** el `codigo` en TTS y en el ruido, la forma de "vacío" en lo que devuelve `inventario` (`""` o `None`), el texto de `nota` en los motivos automáticos y el mensaje del `ValueError` de un ruido más corto que sus clips.
- **Sin definir:** un código sin ningún audio puesto en val o en test (el contrato admite códigos sin audios en las listas y a la vez prohíbe en val y test a un hablante sin sesiones de la notebook). No se prueba.
- **PyAV:** los tests de M4A y la fuente de prueba de 7.6 y 7.7 lo necesitan; sin `av` instalado, fallan.
