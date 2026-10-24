# Contrato · V3 · Características acústicas por bloques y PCA propio

**Versión 1 · 24/10/2026.** Es lo que verifica `tests/test_v3.py`. Los tests se escriben en otro chat o ventana aislada solo a partir de este archivo y se congelan (commit) antes de implementar. Se apoya en los contratos de la Fase 0 (`config.yaml`, `src/config.cargar_config` y `RAIZ`), de H3 (`src/core_ml/pca.py` con `PCA` y `src/core_ml/metricas.py` con `silhouette`) y de V2 (`Preprocesado`, `preprocesar`, `generar_prepro`, `COLUMNAS_V2` y el dataset preprocesado de `data/voz/prepro/`). Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

**Qué hace V3:** toma cada clip válido de V2 (la palabra recortada con margen, filtrada entre 80 y 4000 Hz con pre-énfasis y normalizada a −26 dBFS), descarta las tramas del margen, aplica normalización cepstral de media (CMN) sobre la palabra y resume la señal en un vector de largo fijo (`d = 281`) dividido en 5 bloques: estadísticas en 5 tramos iguales de los 13 MFCC (`130`), de sus derivadas temporales Δ (`130`), de la energía RMS en dB (`10`) y de la tasa de cruces por cero (`10`), más la duración de la palabra en segundos (`1`). Ajusta el escalador solo con `train`, pondera cada bloque por `1/√(columnas usadas)` para que todos aporten lo mismo a la distancia euclídea y proyecta con el `PCA` propio de `src/core_ml/pca.py`. No entrena ni evalúa el K-NN: eso es V4.

## 0. Convenciones

- **Audio de entrada a V3:** un clip válido de V2 (`np.ndarray` de 1 dimensión a `fs = voz.frecuencia_muestreo = 16000` Hz, con muestras finitas). Con `M = round(voz.preprocesamiento.margen_ms · fs / 1000)` (`640` muestras para 40 ms), el clip tiene `L + 2·M` muestras y la palabra ocupa `[M, M + L)`, por lo que la duración de la palabra en segundos es exactamente `L / fs = (len(clip) − 2·M) / fs`.
- **Tramas:** ventana de `W = round(voz.mfcc.ventana_ms · fs / 1000)` muestras (`400` para 25 ms), avance de `H = round(voz.mfcc.avance_ms · fs / 1000)` muestras (`160` para 10 ms) y `n_fft = voz.mfcc.n_fft` (`512`). Sin centrado (`center=False`). El margen a cada lado equivale a `m = M // H` tramas enteras (`4` tramas para 40 ms).
- **Normalización cepstral (CMN):** a cada uno de los `n_mfcc` coeficientes (sobre las tramas de la palabra, ya sin los márgenes) se le resta su media temporal en esa palabra. Como en el dominio log-espectral un filtro lineal fijo es una constante aditiva, CMN cancela la diferencia lineal entre micrófonos y canales sin aplastar la dinámica entre tramos (a diferencia de CMVN por clip, que queda como experimento de V4).
- **Tramos proporcionales (`K = voz.tramos = 5`):** dadas `N ≥ 1` tramas de la palabra, el tramo `i` (`i = 0, …, K − 1`) abarca las columnas `a ≤ t < b` con `a = floor(i · N / K)` y `b = máx(a + 1, floor((i + 1) · N / K))`. En cada tramo se calcula la media y el desvío estándar (`ddof=0`) por fila. Así cualquier palabra válida (`0,3` a `2,5` s, `27` a `243` tramas en el dataset) produce siempre un vector del mismo largo sin rellenar con silencio.
- **Sólo clips válidos (sección 6 de V2):** V3 procesa únicamente las filas de `metadatos.csv` de V2 con `valido == 1` (`True`). Las filas con `valido == 0` no tienen WAV ni entran a `caracteristicas.npz` (en V4 y en la demo se clasifican directo como `otro`).

## 1. Claves nuevas de config.yaml

Las claves `voz.frecuencia_muestreo`, `voz.pasabanda_hz`, `voz.tramos`, `voz.preprocesamiento.margen_ms` y `voz.mfcc` (`n_mfcc`, `n_mels`, `fmax`, `ventana_ms`, `avance_ms`) ya existen desde la Fase 0 y V2. Esta etapa agrega tres hojas en `voz.mfcc` para que ningún hiperparámetro del extractor quede cableado; `cargar_config` también las exige (si falta alguna, lanza `ValueError` con su camino completo):

```yaml
voz:
  mfcc:
    n_mfcc: 13                        # fijo (Fase 0)
    n_mels: <entero en [26, 40]>      # Fase 0 (en config.yaml real: 40)
    fmax: 4000                        # fijo (Fase 0)
    ventana_ms: 25                    # fijo (Fase 0)
    avance_ms: 10                     # fijo (Fase 0)
    n_fft: 512                        # V3: fijo, puntos de la FFT (≥ ventana en muestras)
    delta_ventana: 5                  # V3: entero impar en [3, 9], ancho de Savitzky-Golay para Δ
    piso_rms: 1.0e-6                  # V3: float en (0, 1e-3], piso de amplitud antes de 20·log10
```

**Valores de `config.yaml` real:** `n_mels: 40`, `n_fft: 512`, `delta_ventana: 5` y `piso_rms: 1.0e-6`. La frecuencia mínima del banco Mel es `fmin = voz.pasabanda_hz[0]` (`80` Hz).

## 2. src/core_ml/pca.py y src/core_ml/metricas.py

Creados en H3 (contrato `docs/contratos/h3_caracteristicas.md`, sección 2); V3 los usa tal cual, sin modificarlos:

```python
# src/core_ml/pca.py
class PCA:
    def __init__(self, n_componentes: int | None = None) -> None: ...
    def ajustar(self, X: np.ndarray) -> "PCA": ...          # devuelve self
    def transformar(self, X: np.ndarray) -> np.ndarray: ...
    def ajustar_transformar(self, X: np.ndarray) -> np.ndarray: ...
    # Atributos después de ajustar (float64):
    #   media: (d,)               media de cada columna de X
    #   componentes: (k, d)       una componente por fila, ortonormales (entrada de mayor |valor| > 0)
    #   varianza: (k,)            autovalores de Σ = Xcᵀ·Xc / (n − 1), de mayor a menor
    #   proporcion_varianza: (k,) varianza / traza(Σ)

# src/core_ml/metricas.py
def silhouette(X: np.ndarray, etiquetas) -> float: ...
```

## 3. src/voz/caracteristicas.py

```python
BLOQUES: tuple[str, ...] = ("mfcc", "delta", "energia", "zcr", "duracion")

def largos_de_bloque(cfg: dict) -> dict[str, int]: ...
    # {"mfcc": 2·K·n_mfcc, "delta": 2·K·n_mfcc, "energia": 2·K, "zcr": 2·K, "duracion": 1}
    # Con K = voz.tramos = 5 y n_mfcc = 13: 130, 130, 10, 10 y 1 (suma 281).

def estadisticas_tramos(matriz: np.ndarray, k: int) -> np.ndarray: ...
    # Parte matriz (2D: filas × N tramas) en k tramos y devuelve en float64 de 1D (2·k·filas,)
    # la concatenación por tramo de [media por fila, desvío por fila con ddof=0].

def caracteristicas(clip: np.ndarray, cfg: dict) -> dict[str, np.ndarray]: ...
    # Los 5 bloques crudos de un clip válido de V2: claves BLOQUES, cada una float64 de 1D.

@dataclass(frozen=True)
class Escalador:
    medias: dict[str, np.ndarray]     # (d_b,) por bloque
    desvios: dict[str, np.ndarray]    # (d_b,) por bloque
    columnas: dict[str, np.ndarray]   # bool (d_b,): columnas que se usan
    pesos: dict[str, float]           # 1 / √(columnas usadas del bloque)
    def transformar(self, bloques: dict[str, np.ndarray]) -> np.ndarray: ...

def ajustar_escalador(bloques_train: dict[str, np.ndarray]) -> Escalador: ...

def generar_caracteristicas(cfg: dict, carpeta: str | os.PathLike | None = None,
                            salida: str | os.PathLike | None = None,
                            sobrescribir: bool = False) -> Path: ...
    # Lee metadatos.csv y los WAV válidos de carpeta (por defecto RAIZ / rutas.voz / "prepro")
    # y escribe caracteristicas.npz en salida (por defecto carpeta / "caracteristicas.npz").
    # Devuelve la ruta escrita.

def generar_figura(cfg: dict, carpeta: str | os.PathLike | None = None,
                   salida: str | os.PathLike | None = None) -> Path: ...
    # PCA de carpeta / "caracteristicas.npz" (o de carpeta si ya es un .npz).
    # Por defecto carpeta = RAIZ / rutas.voz / "prepro" y salida = RAIZ / rutas.figuras / "pca_v3.png".
```

**Línea de comandos:** `venv\Scripts\python -m src.voz.caracteristicas [--sobrescribir]` escribe `data/voz/prepro/caracteristicas.npz` y `docs/figuras/pca_v3.png` con `config.yaml`, e imprime las dimensiones, la varianza explicada del PCA, la silhouette de `train` y de `val` (en los 3 comandos y en las 4 clases, en 281d, PCA 30, PCA 20 y por bloque), la contribución de cada bloque a la distancia y el tiempo medio por clip.

## 4. Definiciones exactas

Con los parámetros de `cfg`: `fs = int(voz.frecuencia_muestreo)`, `M = int(round(voz.preprocesamiento.margen_ms * fs / 1000))`, `W = int(round(voz.mfcc.ventana_ms * fs / 1000))`, `H = int(round(voz.mfcc.avance_ms * fs / 1000))`, `n_fft = int(voz.mfcc.n_fft)`, `n_mfcc = int(voz.mfcc.n_mfcc)`, `n_mels = int(voz.mfcc.n_mels)`, `fmin = float(voz.pasabanda_hz[0])`, `fmax = float(voz.mfcc.fmax)`, `ancho_delta = int(voz.mfcc.delta_ventana)`, `piso_rms = float(voz.mfcc.piso_rms)`, `K = int(voz.tramos)` y `m = M // H`.

### 4.1 Estadísticas por tramo (`estadisticas_tramos(matriz, k)`)

- `matriz` es un array de 2 dimensiones `(F, N)` con `F ≥ 1`, `N ≥ 1` y valores finitos, y `k` es un `int ≥ 1` (no `bool`); si no, lanza `ValueError`.
- Para cada `i = 0, …, k − 1`: `a = int(np.floor(i * N / k))`, `b = max(a + 1, int(np.floor((i + 1) * N / k)))` y `s = np.asarray(matriz, dtype=np.float64)[:, a:b]`. El vector del tramo `i` es `np.concatenate([s.mean(axis=1), s.std(axis=1, ddof=0)])` (de largo `2·F`: primero las `F` medias y después los `F` desvíos).
- Devuelve la concatenación de los `k` tramos en orden, como `np.ndarray` de 1 dimensión y `dtype=float64`, de largo `2 · k · F`.

### 4.2 Extracción de los 5 bloques (`caracteristicas(clip, cfg)`)

- **Validación:** `clip` tiene que ser un `np.ndarray` de 1 dimensión, con valores finitos (sin `NaN` ni `±Inf`) y largo `len(clip) >= 2 * M + n_fft` (al menos una trama STFT dentro de la palabra y duración positiva). Si no, lanza `ValueError`. No modifica `clip` y es determinista.
- Se convierte a `y = np.asarray(clip, dtype=np.float32)` para las funciones de `librosa`:
  1. **MFCC con CMN (`w`):**
     ```python
     S = librosa.feature.mfcc(
         y=y, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, win_length=W, hop_length=H,
         n_mels=n_mels, fmin=fmin, fmax=fmax, center=False,
     ).astype(np.float64)
     ```
     Si `S.shape[1] > 2 * m + K`, se quitan las tramas del margen: `w = S[:, m : S.shape[1] - m]`; si no, `w = S`. Luego se resta la media temporal de cada coeficiente sobre la palabra:
     `w = w - w.mean(axis=1, keepdims=True)`.
  2. **Derivada temporal Δ (`d`):**
     Si `w.shape[1] >= ancho_delta`, `d = librosa.feature.delta(w, width=ancho_delta, mode="nearest").astype(np.float64)`; si no, `d = np.zeros_like(w)`.
  3. **Tramas temporales, energía RMS y cruces por cero (`rms`, `zcr`):**
     `fr = librosa.util.frame(y, frame_length=W, hop_length=H).astype(np.float64)`.
     Si `fr.shape[1] > 2 * m + K`, `fr = fr[:, m : fr.shape[1] - m]`.
     - `rms = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(fr ** 2, axis=0)), piso_rms))[None, :]` (forma `(1, T_f)`).
     - `zcr = (np.abs(np.diff(np.sign(fr), axis=0)) > 0).mean(axis=0)[None, :]` (forma `(1, T_f)`).
  4. **Alineación y bloques por tramos:**
     Con `N = min(w.shape[1], rms.shape[1])`, se toman las primeras `N` columnas de cada matriz:
     - `"mfcc"`: `estadisticas_tramos(w[:, :N], K)` (`2 · K · n_mfcc` = 130 valores).
     - `"delta"`: `estadisticas_tramos(d[:, :N], K)` (`2 · K · n_mfcc` = 130 valores).
     - `"energia"`: `estadisticas_tramos(rms[:, :N], K)` (`2 · K` = 10 valores).
     - `"zcr"`: `estadisticas_tramos(zcr[:, :N], K)` (`2 · K` = 10 valores).
  5. **Duración de la palabra:**
     - `"duracion"`: `np.array([(len(clip) - 2 * M) / float(fs)], dtype=np.float64)` (1 valor).

### 4.3 Escalador, vector y archivo de características

- **`ajustar_escalador(bloques_train)`:** recibe un `dict` con exactamente las claves `BLOQUES`, cada una con un array 2D `(n, d_b)` de las mismas `n ≥ 2` filas (las de `train`). Para cada columna, sobre sus valores no `NaN`: `medias` es la media y `desvios` el desvío estándar muestral (`ddof=1`). Una columna se usa (`columnas[b][j] = True`) si tiene al menos 2 valores no `NaN` y `desvio > 1e-9`. `pesos[b] = 1 / √(columnas[b].sum())`. Lanza `ValueError` si faltan o sobran claves, si algún bloque no es 2D, si las cantidades de filas no coinciden, si `n < 2`, si hay `±Inf` o si algún bloque se queda sin columnas usadas.
- **`Escalador.transformar(bloques)`:** recibe un `dict` con exactamente las claves `BLOQUES`, todas con vectores 1D `(d_b,)` o todas con matrices 2D `(n, d_b)` de igual `n`. Para cada bloque `b` en el orden de `BLOQUES` y solo sobre sus columnas usadas: `z = (x − media) / desvio`, los `NaN` pasan a `0.0` (imputación por la media de `train`) y se multiplica por `pesos[b]`. Devuelve la concatenación horizontal: `(d,)` o `(n, d)` en `float64` con `d = Σ_b columnas[b].sum()`, siempre finito. Lanza `ValueError` si faltan o sobran claves, si las dimensiones o largos `d_b` no coinciden o si hay `±Inf`.
- **Contribución de un bloque** sobre un conjunto de `n ≥ 2` vectores escalados: la media, sobre los pares ordenados `i ≠ j`, de `‖z_{i,b} − z_{j,b}‖²`, con `z_{\cdot,b}` las columnas del bloque `b` en el vector escalado (equivale a `2 · Σ_c var(z_{\cdot, c}, ddof=1)`). Con columnas sin `NaN` vale exactamente `2.0` en `train`.
- **`generar_caracteristicas(cfg, carpeta=None, salida=None, sobrescribir=False)`:**
  - `carpeta`: por defecto `RAIZ / rutas.voz / "prepro"`.
  - `salida`: por defecto `Path(carpeta) / "caracteristicas.npz"`.
  - Lee `carpeta / "metadatos.csv"` (exige al menos las columnas `archivo`, `particion`, `origen`, `hablante`, `clase`, `copia`, `valido`), toma **únicamente** las filas con `valido` en `{"1", "True", "true", 1, True}` en su mismo orden (las filas con `valido == 0` se ignoran y no se busca su WAV), lee cada WAV de `carpeta / archivo` con `soundfile.read(..., dtype="float32")`, verifica que su frecuencia de muestreo sea `voz.frecuencia_muestreo` y calcula `caracteristicas(wav, cfg)`.
  - Ajusta el escalador solo con las filas válidas de `particion == "train"` y escribe `salida` (`np.savez`, sin objetos de Python, legible con `allow_pickle=False`) con una fila por cada fila válida de `metadatos.csv`, en el mismo orden:
    - `archivo`, `particion`, `origen`, `hablante`, `clase`: arrays 1D de texto (`str`);
    - `copia`: array 1D de enteros (`int64`);
    - `mfcc`, `delta`, `energia`, `zcr`, `duracion`: los 5 bloques crudos `(n, d_b)` en `float64`;
    - `X`: `(n, d)` en `float64`, los vectores escalados con el escalador de `train`.
  - No escribe ningún otro archivo. Si `salida` ya existe y `sobrescribir` es `False`: lanza `FileExistsError` sin tocar nada. Si faltan `metadatos.csv` o algún WAV de una fila válida: `FileNotFoundError`. Si faltan columnas requeridas, si un WAV no está a `voz.frecuencia_muestreo` o si hay menos de 2 filas válidas de `train`: `ValueError` sin escribir `salida`.
- **`generar_figura(cfg, carpeta=None, salida=None)`:**
  - `carpeta`: por defecto `RAIZ / rutas.voz / "prepro"`. Si `Path(carpeta)` es un directorio lee `Path(carpeta) / "caracteristicas.npz"`; si es un archivo, lo lee directamente.
  - `salida`: por defecto `RAIZ / rutas.figuras / "pca_v3.png"`.
  - Ajusta el `PCA` propio (`n_componentes=3`, o `mín(3, n_train − 1, d)`) con las filas de `particion == "train"` de `X`, proyecta todas las filas y dibuja con Matplotlib (backend `Agg`) un PNG con dos paneles: la proyección 2D (`PC1–PC2`) y 3D (`PC1–PC2–PC3`), distinguiendo las clases (`reconocer`, `identificar`, `salir` y `otro`) y las particiones (`train`, `val`, `test`), indicando en los ejes el porcentaje de varianza explicada y en el título la silhouette de los 3 comandos en `train` (originales `k0`) y `val`. Escribe el PNG en `salida` y devuelve esa ruta.

## 5. Muestra sintética determinista para los tests

Para que los tests de los puntos 6.4 y 6.5 sean rápidos e independientes de los archivos de `data/`, se definen sobre señales sintéticas deterministas a `fs = 16000` Hz con `M = 640` muestras de margen por lado:

- **Generador de un clip sintético** `clip_sintetico(f0_hz, mod_hz, dur_palabra_s, amp, semilla)`: con `L = round(dur_palabra_s · fs)` muestras de palabra y `N_tot = L + 2·M`, construye `t = np.arange(N_tot, dtype=np.float64) / fs`, una envolvente por tramos que modula la amplitud a lo largo del clip `(1.0 + 0.4 · np.sin(2π · mod_hz · t))` multiplicada por una portadora con tres armónicos `0.6·sin(2π·f0·t) + 0.3·sin(2π·2·f0·t) + 0.1·sin(2π·3·f0·t)` más ruido gaussiano de `np.random.default_rng(semilla)` con desvío `0.02`, escalada por `amp` en `float32`.
- **Muestra de 3 clases de comandos sintéticos** (18 clips de `train`, 6 por clase, y 9 de `val`, 3 por clase):
  - Clase `"reconocer"`: `f0_hz` en `220 ± 15` Hz, `mod_hz = 3.0`, `dur_palabra_s` en `0.75 ± 0.05` s.
  - Clase `"identificar"`: `f0_hz` en `520 ± 25` Hz, `mod_hz = 6.0`, `dur_palabra_s` en `1.15 ± 0.06` s.
  - Clase `"salir"`: `f0_hz` en `1200 ± 40` Hz, `mod_hz = 1.5`, `dur_palabra_s` en `0.42 ± 0.03` s.

## 6. Criterios que verifican los tests

### 6.1 Configuración

- Las claves de la sección 1 existen en `config.yaml` con su tipo y rango (`n_fft == 512`, `delta_ventana` entero impar en `[3, 9]`, `piso_rms` float en `(0, 1e-3]`, `n_mels` entero en `[26, 40]`), y en el `config.yaml` real tienen los valores de la sección 1.
- `cargar_config` lanza `ValueError` que nombra el camino completo si falta cualquiera de las claves nuevas (`voz.mfcc.n_fft`, `voz.mfcc.delta_ventana`, `voz.mfcc.piso_rms`).

### 6.2 PCA propio y silhouette propia (`src/core_ml/`)

- Con matrices de dimensión `(60, 281)` (como los vectores de V3) y `k` en `{2, 3, 20, 30, None}`: `PCA` de `src.core_ml.pca` coincide con `sklearn.decomposition.PCA(n_components=k, svd_solver="full")` en `media`, `varianza`, `proporcion_varianza`, `componentes` y `transformar` (salvo el signo de cada componente, con la entrada de mayor valor absoluto positiva), con tolerancia `1e-6`.
- `silhouette` de `src.core_ml.metricas` coincide con `sklearn.metrics.silhouette_score` (tolerancia `1e-9`) con etiquetas de texto (`"reconocer"`, `"identificar"`, `"salir"`, `"otro"`).
- Ni `src/core_ml/pca.py`, ni `src/core_ml/metricas.py`, ni `src/voz/caracteristicas.py` importan `scikit-learn` (el texto de los tres archivos no contiene `sklearn`).

### 6.3 Estadísticas por tramo y formato de bloques (`estadisticas_tramos`, `largos_de_bloque`, `caracteristicas`)

- `largos_de_bloque(cfg) == {"mfcc": 130, "delta": 130, "energia": 10, "zcr": 10, "duracion": 1}` con el `cfg` real (suma 281), y responde dinámicamente si en una copia de `cfg` se cambian `voz.tramos` o `voz.mfcc.n_mfcc`.
- `estadisticas_tramos(matriz, k)`: en cada tramo `i = 0, …, k − 1` devuelve exactamente `[media por fila, desvío ddof=0 por fila]` sobre las columnas `floor(i·N/k) : máx(floor(i·N/k)+1, floor((i+1)·N/k))`, incluso cuando `N < k` (tramos de 1 columna con desvío 0). Lanza `ValueError` con matrices 1D o 3D, con `N == 0`, con `NaN`/`Inf` o con `k < 1` o `bool`.
- **Oráculo exacto de `caracteristicas(clip, cfg)`:** sobre al menos 6 clips sintéticos de distintas duraciones (entre `0.30` s y `2.0` s), los 5 bloques devueltos coinciden con el oráculo escrito en el test a partir de la sección 4.2 (tolerancia `1e-9`), son `float64` de 1 dimensión sin `NaN` ni `±Inf`, no modifican `clip` y dos llamadas seguidas dan arrays idénticos (`np.array_equal`).
- **Propiedades acústicas (invarianza de escala y duración):**
  - Multiplicar un clip sintético por una constante positiva `c` (por ejemplo `c = 0.25` y `c = 4.0`, manteniéndolo muy por encima de `piso_rms`) deja invariantes `mfcc` (por el logaritmo y CMN), `delta`, `zcr`, `duracion` y los `K` desvíos de `energia` (tolerancia `1e-5`), mientras que las `K` medias de `energia` aumentan exactamente en `20 · log10(c)` dB (tolerancia `1e-4`).
  - `caracteristicas(clip, cfg)["duracion"][0]` es exactamente `(len(clip) − 2·M) / fs` (tolerancia `1e-12`).
- **Errores de `caracteristicas`:** lanza `ValueError` si `clip` no es 1D, si tiene menos de `2·M + n_fft` muestras (`1792` muestras con el `cfg` real) o si contiene `NaN` o `±Inf`.

### 6.4 Escalador, contribución por bloque y separación en la muestra sintética

- `ajustar_escalador` y `Escalador.transformar` sobre los bloques de la muestra sintética:
  - En las columnas usadas, `medias` y `desvios` coinciden con la media y el desvío (`ddof=1`) de los valores no `NaN` de `train` (tolerancia `1e-9`). En `Z_train`, cada columna usada tiene media `0` (`1e-9`).
  - Cada entrada `NaN` en una columna usada se imputa a `0.0` exacto en `Z`, y las columnas con menos de 2 datos no `NaN` o desvío `≤ 1e-9` en `train` quedan con `columnas[b][j] == False` y se descartan.
  - `pesos[b] == 1 / √(columnas[b].sum())`, y la contribución a la distancia cuadrática de cada uno de los 5 bloques en `train` (sin `NaN`) es `2.0` (tolerancia `1e-9`).
  - `transformar` de un clip individual `(d_b,)` coincide con la fila correspondiente de `transformar` por lote `(n, d_b)` (`1e-12`), y una fila no depende de las demás filas del lote.
  - En la muestra sintética de 3 clases (sección 5), la `silhouette` sobre `Z` es `≥ 0.40` tanto en `train` como en `val`.
  - Lanza `ValueError` si faltan o sobran claves de `BLOQUES`, si las dimensiones o `d_b` no coinciden, si hay `< 2` filas en `ajustar_escalador`, si un bloque queda con todas las columnas constantes en `train`, o si hay `±Inf`.

### 6.5 Archivo de características y figura (`generar_caracteristicas`, `generar_figura` en `tmp_path`)

Con un dataset preprocesado de V2 armado en `tmp_path` (con un `metadatos.csv` de `COLUMNAS_V2` que incluya al menos 12 filas válidas de `train`, 6 válidas de `val`, 4 válidas de `test` de las 4 clases, más al menos 2 filas no válidas con `valido=0`, `archivo=""` y sin archivo WAV en disco):

- `generar_caracteristicas(cfg, carpeta)` devuelve `carpeta / "caracteristicas.npz"` y no crea otros archivos.
- El `.npz` se lee con `allow_pickle=False`, tiene exactamente las claves de la sección 4.3 (`archivo`, `particion`, `origen`, `hablante`, `clase`, `copia`, `mfcc`, `delta`, `energia`, `zcr`, `duracion`, `X`) y contiene **solo** las filas con `valido == 1` en el mismo orden de `metadatos.csv` (las filas con `valido == 0` no están).
- Cada fila de los 5 bloques crudos coincide con `caracteristicas(wav_leido, cfg)` (`1e-9`), y `X` coincide con `ajustar_escalador` de las filas válidas de `train` aplicado a todas las filas válidas (`1e-12`).
- **Sin fuga entre particiones:** si en otra carpeta se mantienen idénticos los WAV de `train` y se cambian los WAV de `val` y `test`, las filas de `train` en `X` quedan idénticas (`np.array_equal`).
- Si el `.npz` ya existe y `sobrescribir=False`: lanza `FileExistsError` sin modificar el archivo; con `sobrescribir=True` lo sobrescribe. Si falta `metadatos.csv` o el WAV de una fila válida: `FileNotFoundError`. Si un WAV tiene otra frecuencia (ej. 8000 Hz) o hay menos de 2 filas válidas de `train`: `ValueError` sin dejar un `.npz` escrito.
- `generar_figura(cfg, carpeta, tmp_path / "pca_v3.png")` escribe un PNG no vacío legible como imagen y devuelve su `Path`.

### 6.6 Aceptación (`@pytest.mark.aceptacion`, con los datos reales de `data/voz/prepro/`)

Se saltea con `pytest.skip` si no existe `data/voz/prepro/metadatos.csv`:

- `generar_caracteristicas(cfg, carpeta=RAIZ / "data/voz/prepro", salida=tmp_path / "caracteristicas.npz")` tarda menos de 60 segundos en la notebook.
- El `.npz` tiene exactamente las filas con `valido == 1` de `data/voz/prepro/metadatos.csv` (1908 filas: 1722 de `train`, 87 de `val` y 99 de `test`), sin `NaN` ni `±Inf` en los bloques ni en `X`, y con `X.shape == (1908, 281)`.
- El bloque `duracion` coincide con la columna `duracion_palabra_s` de `metadatos.csv` en todas las filas (tolerancia `1e-3` s).
- La contribución de cada uno de los 5 bloques en `train` es `2.0 ± 1e-6`, y en `val` está en `[1.3, 2.7]`.
- El `PCA` propio ajustado con `X[train]` (`n_componentes=30` y `n_componentes=20`) coincide con `sklearn.decomposition.PCA(svd_solver="full")` en las proyecciones de las 1908 filas salvo el signo (tolerancia `1e-6`), y las primeras 30 componentes explican al menos el `70 %` de la varianza de `train`.
- La `silhouette` de los 3 comandos en los clips originales (`copia == 0`) de `train` y en `val` es positiva tanto en `281d` como tras proyectar con `PCA(20)` y `PCA(30)`, y con `PCA(20)` supera a la de `281d`.
- `generar_figura(cfg, carpeta=tmp_path, salida=tmp_path / "pca_v3.png")` escribe el PNG en `tmp_path`.

## 7. Verificación manual (no va en pytest; se anota en el Documento 2)

- Correr `venv\Scripts\python -m src.voz.caracteristicas --sobrescribir`, revisar `docs/figuras/pca_v3.png` (que se distingan los tres comandos y cómo se distribuyen `train`, `val`, `test` y `otro` en 2D y 3D) y anotar en la ficha de V3 del Documento 2 los tiempos, la varianza explicada y la silhouette por bloque.

## 8. Valores medidos en el prototipo (24/10/2026)

Medidos con `Prototipos\v3_2026-10-24\proto_v3.py` sobre los 1908 clips válidos de `data/voz/prepro/` (`1722` en `train`, `87` en `val` y `99` en `test`):

- **Tiempo:** `10,56` s para los 1908 clips (`4,81` ms por clip en promedio; mediana `1,27` ms, percentil 99 `7,75` ms).
- **Tramas por palabra (`N`, sin los márgenes):** mínimo `27` (en una palabra de `0,30` s), mediana `72`, máximo `243`. Tramas por tramo (`K = 5`): mínimo `5`, mediana `14`, máximo `48`. Ningún `NaN` ni `±Inf`.
- **Duración:** `(len(clip) − 2·M) / fs` difiere en `0,0` s de `duracion_palabra_s` de `metadatos.csv` en los 1908 clips (rango `[0,300, 2,460]` s).
- **Escalador (`d = 281`):** se usan las 281 columnas (`130 + 130 + 10 + 10 + 1`). Contribución a la distancia cuadrática en `train`: `2,0000` en los 5 bloques; en `val`: `mfcc` `1,924`, `delta` `1,827`, `energia` `2,060`, `zcr` `1,570` y `duracion` `2,468`.
- **PCA propio vs. `scikit-learn`:** diferencia máxima `1,78e-15` en `varianza` y `8,20e-13` en las proyecciones. Varianza explicada acumulada en `train`: `PC1` `25,95 %`, `k = 2` `32,46 %`, `k = 3` `36,95 %`, `k = 10` `56,99 %`, `k = 20` `69,83 %`, `k = 30` `75,92 %`.
- **Silhouette (3 comandos: `281d` / `PCA 30` / `PCA 20`):** en `train` completo (`n = 1289`): `0,040` / `0,066` / `0,075`; en `train` originales `k0` (`n = 442`): `0,130` / `0,198` / `0,222`; en `train` notebook `k0` (`n = 231`): `0,145` / `0,225` / `0,249`; en `val` (`n = 67`): `0,070` / `0,113` / `0,124`; en `test` (`n = 71`): `0,087` / `0,150` / `0,167`.
- **Chequeo con el K-NN del borrador (validación cruzada dejando 1 hablante afuera en los 10 hablantes de notebook de `train`, 231 comandos `k0`):** con `K = 3`, acierto `91,8 %` en `281d`, `93,5 %` en `PCA 30` y `92,2 %` en `PCA 20`; con `K = 5`, `90,0 %` en `281d` y `91,3 %` en `PCA 30`. La elección final de `K`, PCA y el umbral de distancia corresponde a V4.

