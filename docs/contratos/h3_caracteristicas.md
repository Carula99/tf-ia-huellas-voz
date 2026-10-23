
# Contrato · H3 · Características ponderadas y PCA propio

**Versión 1.1 · 23/10/2026** (la 1 es del 22/10). **Cambio en la 1.1:** el PCA rechaza "todas las filas iguales" comparando las filas (`np.all(X == X[0])`), además de `traza(Σ) = 0`: con copias de una fila cualquiera, la media puede diferir de la fila en el último decimal y la traza queda en ~1e-30, no en 0 (lo encontró el chat de tests). Cambia la regla 3 del PCA (sección 2). Además, se suman a la sección 8 las lecturas del contrato que dejó anotadas el chat que escribe `tests/test_h3.py`, confirmadas por Santiago. Ningún criterio cambia de umbral. Se anota en el Documento 2 junto con los tests.

Es lo que verifica `tests/test_h3.py`. Los tests se escriben en otro chat solo a partir de este archivo y se congelan (commit) antes de implementar. Se apoya en los contratos de la Fase 0 (`config.yaml` y `src/config.cargar_config`), de H1 (convenciones, `generar_base`, `ParametrosVariante`, `sortear_parametros`, `aplicar_variante`, `plan_dataset`, `generar_dataset`, `metadatos.csv` y la receta de la sección 4) y de H2 (`preprocesar` y `HuellaPreprocesada`). Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

Esta etapa crea también los dos primeros módulos de `src/core_ml/`: el PCA propio (`pca.py`) y la silhouette propia (`metricas.py`). Los usa después el carril de voz (V3), por eso el PCA admite cualquier cantidad de componentes y vectores sueltos.

**Cambios respecto del plan v0.3.2** (medidos con un prototipo sobre las salidas de H2 de las 300 imágenes del dataset; quedan en el Documento 2):

- **Sin el bloque de momentos de Hu (decisión 46, aprobada por Santiago):** solo separaba 0,13 (silhouette en train) y bajaba el total de 0,67 a 0,51; con Hu, el adelanto de H4 daba FAR 8,9 % en test, y sin Hu, 0 %. El vector queda con tres bloques: orientación, Gabor y minucias. La comparación está en la tabla 1 del Documento 2.
- **Cortes y celdas sin huella:** cada celda, sector o anillo con menos de `cobertura_min` (50 %) de huella vale NaN en el bloque crudo y el escalador lo imputa con la media de train (estandarizado vale 0). Las minucias a menos de `borde_minucia_px` (14 px) del borde de la máscara (el borde recto de los cortes y el de la ventana cuentan) se descartan. Por eso "sin NaN ni Inf" del plan vale para el vector escalado, no para los bloques crudos.
- **Gabor:** en lugar del histograma global de respuestas, la energía global de cada filtro normalizada (suma 1). En cada sector va la media del valor absoluto de la respuesta, que con filtros de media cero es el desvío absoluto medio de FingerCode (Jain et al., 2000), simplificado.
- **Minucias:** además de la cantidad, la proporción y la densidad por anillos, la densidad en una grilla de 4 × 4.
- **Columnas sin datos:** el escalador descarta las columnas que en train tienen menos de 2 valores o desvío ≤ 1e-9 (en el dataset, la celda de la esquina superior izquierda de la grilla de minucias), y el peso de cada bloque es 1/√(columnas que quedan).
- **Silhouette propia** en `src/core_ml/metricas.py`, para que la línea de comandos informe la separación sin scikit-learn; los tests la comparan con `sklearn.metrics.silhouette_score`.

## 0. Convenciones

- Las de la sección 0 de H1 (coordenadas `(x, y)` con `imagen[i, j]` en `(x, y) = (j, i)`, orientación en `[0, π)` desde el eje x hacia el eje y) y las de la ventana de H2: `(alto, ancho) = huellas.ventana_px = (250, 250)` y centro `c_w = (124.5, 124.5)`. H2 deja el punto de referencia en `c_w`, así que los anillos y sectores "alrededor del punto de referencia" están centrados en `c_w`.
- **Coordenadas polares de un píxel** `(x, y)` de la ventana: `r = hypot(x − 124.5, y − 124.5)` y `φ = atan2(y − 124.5, x − 124.5) mod 2π`, en `[0, 2π)` (con y hacia abajo, φ crece de x hacia y). Con NumPy, para toda la ventana:

  ```python
  yy, xx = np.mgrid[0:250, 0:250]
  r = np.hypot(xx - 124.5, yy - 124.5)
  fi = np.mod(np.arctan2(yy - 124.5, xx - 124.5), 2 * np.pi)
  ```
- **Anillo** `[r_a, r_{a+1})` de una lista creciente de radios `(r_0, …, r_R)`: los píxeles con `r_a ≤ r < r_{a+1}`; hay `R` anillos.
- **Sector** `s` de `S` (0 a `S − 1`): los píxeles con `np.minimum(np.floor(fi * S / (2 * np.pi)).astype(int), S - 1) == s`. Se fija la expresión porque 500 píxeles de la ventana caen justo sobre las diagonales.
- **Grilla de `g × g`:** bordes `b = np.round(np.linspace(0, 250, g + 1)).astype(int)` (con `g = 5`: 0, 50, 100, 150, 200, 250; con `g = 4`: 0, 62, 125, 188, 250). La celda `(i, j)` son las filas `b[i] ≤ y < b[i+1]` y las columnas `b[j] ≤ x < b[j+1]`; las celdas se recorren por filas (`i` afuera, `j` adentro).
- **Distancia al borde** de una máscara `M` (bool de 250 × 250): `D = scipy.ndimage.distance_transform_edt(np.pad(M, 1))[1:-1, 1:-1]`, la distancia euclídea de cada píxel al píxel más cercano fuera de la máscara; lo que está fuera de la ventana cuenta como fuera (un píxel del borde de la ventana tiene `D ≤ 1`).
- **Crossing number** de un píxel `p` de un esqueleto `E` (bool): con sus 8 vecinos en orden circular `P_1 … P_8` (arriba, arriba-derecha, derecha, abajo-derecha, abajo, abajo-izquierda, izquierda, arriba-izquierda; 1 si el vecino está en `E`, 0 si no o si cae fuera de la ventana), `CN(p) = ½ Σ_{k=1..8} |P_k − P_{k+1}|` con `P_9 = P_1`. CN = 1 es una terminación y CN = 3, una bifurcación.
- **Cobertura** de una zona `A` (celda, sector o anillo) respecto de una máscara `W`: `|A ∩ W| / |A|`. **"Menos de `cobertura_min`"**: `|A ∩ W| < cobertura_min · |A|` (con 0,5 justo, la zona se usa).
- **Densidad:** cantidad de minucias por cada 1000 px² de zona válida.

## 1. Claves nuevas de config.yaml

Se suman a las de la Fase 0, H1 y H2. `cargar_config` también las exige: si falta alguna, lanza `ValueError` con su camino completo.

```yaml
huellas:
  caracteristicas:                      # H3 (contrato h3_caracteristicas.md)
    grilla_orientacion: <entero en [3, 8]>            # celdas por lado de la grilla de orientación
    cobertura_min: 0.5                                # fijo: menos huella que esto → NaN (se imputa)
    gabor_periodos_px: <lista creciente de 1 a 3 floats en [6, 12]>
    gabor_orientaciones: <entero en [4, 12]>
    anillos_gabor_px: <lista creciente de 2 a 6 floats en [0, 124.5]>
    sectores_gabor: <entero en [4, 16]>
    erosion_gabor_px: <float en [0, 20]>              # Gabor solo a esta distancia del borde de la máscara
    borde_minucia_px: 14.0                            # fijo: minucias más cerca del borde → se descartan
    componente_min_px: <entero en [0, 50]>            # componentes del esqueleto más chicas → afuera
    espuela_px: <entero en [0, 20]>                   # largo máximo de una espuela
    par_terminaciones_px: <float en [0, 20]>
    par_bifurcaciones_px: <float en [0, 20]>
    anillos_minucias_px: <lista creciente de 2 a 6 floats en [0, 124.5]>
    grilla_minucias: <entero en [2, 6]>
```

Valores con los que se midió el prototipo (son los que van en `config.yaml`): `grilla_orientacion: 5`, `gabor_periodos_px: [8.0, 10.0]`, `gabor_orientaciones: 6`, `anillos_gabor_px: [20.0, 50.0, 80.0, 110.0]`, `sectores_gabor: 8`, `erosion_gabor_px: 10.0`, `componente_min_px: 20`, `espuela_px: 12`, `par_terminaciones_px: 10.0`, `par_bifurcaciones_px: 8.0`, `anillos_minucias_px: [0.0, 40.0, 75.0, 110.0]` y `grilla_minucias: 4`.

## 2. src/core_ml/pca.py y src/core_ml/metricas.py

```python
# src/core_ml/pca.py
class PCA:
    def __init__(self, n_componentes: int | None = None) -> None: ...
    def ajustar(self, X: np.ndarray) -> "PCA": ...          # devuelve self
    def transformar(self, X: np.ndarray) -> np.ndarray: ...
    def ajustar_transformar(self, X: np.ndarray) -> np.ndarray: ...
    # Atributos después de ajustar (float64):
    #   media: (d,)               media de cada columna de X
    #   componentes: (k, d)       una componente por fila, ortonormales
    #   varianza: (k,)            autovalores de Σ, de mayor a menor
    #   proporcion_varianza: (k,) varianza / traza(Σ)

# src/core_ml/metricas.py
def silhouette(X: np.ndarray, etiquetas) -> float: ...
```

**PCA** (plan, sección 6.4):

1. `n_componentes` es `None` o un `int` ≥ 1 (no `bool`); otra cosa: `ValueError` en el constructor.
2. `ajustar(X)`: `X` es un array de 2 dimensiones `(n, d)` con `n ≥ 2`, `d ≥ 1` y valores finitos; si no, `ValueError`. Se trabaja en `float64` y no se modifica `X`.
3. `media` = media de cada columna; `Xc = X − media`; `Σ = Xcᵀ·Xc / (n − 1)`. Si todas las filas de `X` son iguales (`np.all(X == X[0])`) o si `traza(Σ) = 0`: `ValueError`. *(1.1)* Se comparan las filas porque la traza sola depende del redondeo.
4. Autovalores y autovectores de Σ con `np.linalg.eigh`, ordenados de mayor a menor autovalor. `k = n_componentes`, o `min(n − 1, d)` si es `None`; si `k > min(n − 1, d)`: `ValueError`.
5. `componentes` son los `k` primeros autovectores, uno por fila. **Signo:** en cada componente, la entrada de mayor valor absoluto (la primera, si hay empate) es positiva.
6. `varianza` son sus autovalores (los negativos por redondeo valen 0) y `proporcion_varianza = varianza / traza(Σ)`.
7. `transformar(X)`: `(X − media) · componentesᵀ`. Acepta `(n, d)` y devuelve `(n, k)`, o un vector `(d,)` y devuelve `(k,)`. Antes de `ajustar`: `RuntimeError`. Otro `d`, otra cantidad de dimensiones o valores no finitos: `ValueError`.
8. `ajustar_transformar(X)` es `ajustar(X).transformar(X)`.
9. Solo NumPy: el archivo no importa scikit-learn.

**silhouette(X, etiquetas)**: el coeficiente de silhouette medio con distancia euclídea (Rousseeuw, 1987), igual que `sklearn.metrics.silhouette_score(X, etiquetas)`. Para cada fila `i`: `a(i)` es la distancia media a las otras filas de su grupo y `b(i)` el mínimo, entre los otros grupos, de la distancia media a sus filas; `s(i) = (b − a) / max(a, b)`, y `s(i) = 0` si su grupo tiene una sola fila (o si `max(a, b) = 0`). Devuelve el promedio de `s(i)` como `float`. `etiquetas` es una secuencia de largo `n` de valores comparables (enteros o textos). `ValueError` si `X` no es de 2 dimensiones o tiene valores no finitos, si los largos no coinciden o si la cantidad de etiquetas distintas no está entre 2 y `n − 1`. Solo NumPy.

## 3. src/huellas/caracteristicas.py

```python
BLOQUES: tuple[str, ...] = ("orientacion", "gabor", "minucias")

@dataclass(frozen=True)
class Minucias:
    terminaciones: np.ndarray    # float64 (k_t, 2): (x, y) en la ventana
    bifurcaciones: np.ndarray    # float64 (k_b, 2): (x, y) en la ventana

def largos_de_bloque(cfg: dict) -> dict[str, int]: ...
def extraer_minucias(huella: HuellaPreprocesada, cfg: dict) -> Minucias: ...
def caracteristicas(huella: HuellaPreprocesada, cfg: dict) -> dict[str, np.ndarray]: ...
    # Los 3 bloques crudos: claves BLOQUES, float64 de 1 dimensión, con NaN donde falta huella.

@dataclass(frozen=True)
class Escalador:
    medias: dict[str, np.ndarray]     # (d_b,) por bloque
    desvios: dict[str, np.ndarray]    # (d_b,) por bloque
    columnas: dict[str, np.ndarray]   # bool (d_b,): columnas que se usan
    pesos: dict[str, float]           # 1 / √(columnas usadas del bloque)
    def transformar(self, bloques: dict[str, np.ndarray]) -> np.ndarray: ...

def ajustar_escalador(bloques_train: dict[str, np.ndarray]) -> Escalador: ...

def generar_caracteristicas(cfg: dict, carpeta: str | os.PathLike | None = None,
                            sobrescribir: bool = False) -> Path: ...
    # Lee el dataset de H1 de carpeta (por defecto RAIZ / rutas.huellas) y escribe ahí
    # caracteristicas.npz (sección 4.5). Devuelve la ruta escrita.

def generar_figura(cfg: dict, carpeta: str | os.PathLike | None = None,
                   salida: str | os.PathLike | None = None) -> Path: ...
    # PCA de carpeta / caracteristicas.npz. Por defecto RAIZ / rutas.figuras / "pca_h3.png".
```

**Línea de comandos:** `venv\Scripts\python -m src.huellas.caracteristicas [--sobrescribir]` escribe `data/huellas/caracteristicas.npz` y `docs/figuras/pca_h3.png` con `config.yaml`, e imprime la silhouette de train y de val (total y por bloque) y la contribución de cada bloque a la distancia.

## 4. Definiciones exactas

Con `h` una `HuellaPreprocesada` y los valores de `huellas.caracteristicas`. `caracteristicas(h, cfg)` lanza `ValueError` si `realzada`, `mascara`, `esqueleto` u `orientacion` no son de 250 × 250; no modifica `h` y es determinista.

### 4.1 Bloque de orientación (`2 · g²` valores, `g = grilla_orientacion`)

Para cada celda de la grilla de `g × g`, en orden: si tiene menos de `cobertura_min` de `mascara`, `[NaN, NaN]`; si no, `[media de cos 2θ, media de sen 2θ]` sobre los píxeles de la celda dentro de `mascara`, con `θ = orientacion` en `float64`. Con `g = 5` son 50 valores.

### 4.2 Bloque de Gabor (`F · R · S + F` valores)

- **Filtros:** para cada período `P` de `gabor_periodos_px` y cada `k = 0 … K − 1` (`K = gabor_orientaciones`), en ese orden (`F = len(gabor_periodos_px) · K` filtros): `σ = P / 2`, `L = 2 · round(3σ) + 1`, `g = cv2.getGaborKernel((L, L), σ, π·k/K + π/2, P, 1.0, 0.0, ktype=cv2.CV_64F)`; después `g = g − media(g)` y `g = g / Σ|g|`. El filtro `k` responde a crestas de orientación `π·k/K`.
- **Imagen:** `f = 255 − realzada` en `float64`; con `μ` la media de `f` en `mascara`, `f = μ` fuera de `mascara` y después `f = f − μ`. Respuesta del filtro: `|cv2.filter2D(f, -1, g, borderType=cv2.BORDER_REFLECT)|`.
- **Zona útil:** `G = D(mascara) ≥ erosion_gabor_px`.
- **Valores:** para cada filtro, en orden, y para cada anillo `a` de `anillos_gabor_px` (`R` anillos) y cada sector `s` de `S = sectores_gabor` (el anillo afuera, el sector adentro): si `anillo ∩ sector` tiene menos de `cobertura_min` de `G`, NaN; si no, la media de la respuesta sobre `anillo ∩ sector ∩ G`. Después, los `F` valores globales: la media de la respuesta de cada filtro sobre `G`, dividida por la suma de las `F` medias (suman 1); si esa suma es 0 (una imagen sin textura), los `F` globales son NaN.
- Si `mascara` o `G` están vacías, todo el bloque es NaN. Con los valores de la sección 1: `F = 12`, `R = 3`, `S = 8`, 300 valores.

### 4.3 Minucias (`extraer_minucias`)

1. **Esqueleto limpio** `E`: `esqueleto` sin sus componentes conexas (vecinos-8) de menos de `componente_min_px` píxeles. El CN se calcula sobre `E`.
2. **Candidatas:** terminaciones `T0` = píxeles de `E` con CN = 1 y bifurcaciones `B0` = píxeles de `E` con CN = 3, en los dos casos con `D(mascara) ≥ borde_minucia_px`.
3. **Espuelas:** una unión es un píxel de `E` con CN ≥ 3. Una terminación `t` de `T0` es una espuela si hay una unión `j` a la que se llega desde `t` por un camino de vecinos-8 dentro de `E`, `t = p_0, p_1, …, p_m = j`, con `m ≤ espuela_px` y sin uniones entre `p_1` y `p_{m−1}`. Se descartan las espuelas y las bifurcaciones de `B0` a distancia de Chebyshev ≤ 2 (`|Δx| ≤ 2` y `|Δy| ≤ 2`) de alguna de las uniones así alcanzadas desde una espuela. Quedan `T1` y `B1`.
4. **Pares:** se descartan las terminaciones de `T1` que tienen otra terminación de `T1` a distancia euclídea `< par_terminaciones_px` (las dos del par) y las bifurcaciones de `B1` con otra de `B1` a `< par_bifurcaciones_px`.
5. **Salida:** las que quedan, como `(x, y) = (j, i)` en `float64`, ordenadas por `y` y, a igual `y`, por `x` (el orden de `np.argwhere`). Sin minucias, arrays de forma `(0, 2)`.

### 4.4 Bloque de minucias (`3 + 2 · R_m + g_m²` valores)

Con `V = D(mascara) ≥ borde_minucia_px` (la zona válida), `n_t` y `n_b` la cantidad de terminaciones y bifurcaciones de `extraer_minucias(h, cfg)`, `R_m` los anillos de `anillos_minucias_px` y `g_m = grilla_minucias`:

1. **Globales:** `[1000 · n_t / |V|, 1000 · n_b / |V|, n_b / (n_t + n_b)]` (la proporción vale 0 sin minucias); si `V` está vacía, `[NaN, NaN, NaN]`.
2. **Por anillo**, primero las terminaciones en los `R_m` anillos y después las bifurcaciones: si el anillo tiene menos de `cobertura_min` de `V`, NaN; si no, `1000 · (minucias de ese tipo con r del punto en el anillo) / |anillo ∩ V|`.
3. **Grilla de `g_m × g_m`**, con las terminaciones y las bifurcaciones juntas: si la celda tiene menos de `cobertura_min` de `V`, NaN; si no, `1000 · (minucias en la celda) / |celda ∩ V|`.

Con los valores de la sección 1: 3 + 6 + 16 = 25 valores. `largos_de_bloque(cfg)` devuelve `{"orientacion": 2·g², "gabor": F·R·S + F, "minucias": 3 + 2·R_m + g_m²}` (50, 300 y 25).

### 4.5 Escalador, vector y archivo de características

- **`ajustar_escalador(bloques_train)`:** un dict con exactamente las claves `BLOQUES`, cada una con un array `(n, d_b)` de las mismas `n ≥ 2` filas (las de train). Para cada columna, con sus valores no NaN: `medias` es su media y `desvios` su desvío con `ddof=1`. La columna se usa (`columnas`) si tiene al menos 2 valores no NaN y `desvio > 1e-9`; en las otras, `medias` y `desvios` pueden valer cualquier cosa (también NaN). `pesos[b] = 1 / √(columnas[b].sum())`. `ValueError` si faltan o sobran claves, si las formas no coinciden, si `n < 2`, si hay ±Inf o si un bloque se queda sin columnas.
- **`transformar(bloques)`:** el mismo dict con un vector `(d_b,)` por bloque, o un array `(n, d_b)`. Para cada bloque, en el orden de `BLOQUES` y solo con sus columnas usadas: `z = (x − media) / desvio`, los NaN pasan a 0 (la media de train) y se multiplica por `pesos[b]`. Devuelve los bloques uno al lado del otro: `(d,)` o `(n, d)` con `d = Σ_b columnas[b].sum()`, siempre finito. `ValueError` si faltan o sobran claves, si `d_b` no coincide o si hay ±Inf.
- **Contribución de un bloque** sobre un conjunto de vectores escalados: la media, sobre los pares ordenados `i ≠ j`, de `‖z_i,b − z_j,b‖²`, con `z_·,b` las columnas del bloque `b` en el vector. Con columnas completas vale exactamente 2 en train; las imputadas la bajan un poco.
- **`generar_caracteristicas(cfg, carpeta, sobrescribir)`:** lee `carpeta / "metadatos.csv"` (de H1; usa solo las columnas `archivo`, `particion`, `identidad`, `intruso` y `tipo`), lee cada PNG de `carpeta / archivo` con `cv2.imdecode(np.fromfile(ruta, np.uint8), cv2.IMREAD_UNCHANGED)`, y calcula `caracteristicas(preprocesar(imagen, cfg), cfg)`. Ajusta el escalador solo con las filas de `particion == "train"` y escribe `carpeta / "caracteristicas.npz"` (`np.savez`, sin objetos de Python, se lee con `allow_pickle=False`) con una fila por fila de `metadatos.csv`, en el mismo orden:
  - `archivo`, `particion`, `tipo`: arrays de texto; `identidad`, `intruso`: enteros;
  - `orientacion`, `gabor`, `minucias`: los bloques crudos `(n, d_b)` en `float64`, con sus NaN;
  - `X`: `(n, d)` en `float64`, los vectores escalados con el escalador de train.

  No escribe ningún otro archivo. Si `caracteristicas.npz` ya existe y `sobrescribir` es `False`: `FileExistsError` sin tocar nada. Si no hay filas de train: `ValueError`, sin escribir nada. Si falta `metadatos.csv` o un PNG: `FileNotFoundError`.
- **`generar_figura(cfg, carpeta, salida)`:** lee `caracteristicas.npz`, ajusta el `PCA` propio con las filas de train de `X` y dibuja con matplotlib (backend Agg) la proyección 2D (PC1–PC2) y 3D (PC1–PC2–PC3) de todas las filas, las autorizadas con un color por identidad (train relleno, val y test sin relleno) y los intrusos en gris, con la varianza explicada de cada eje y la silhouette de train y de las autorizadas de val (cuando se puede calcular: al menos 2 identidades y más filas que identidades). Escribe un PNG en `salida` y devuelve esa ruta.

## 5. Muestra de prueba

Los criterios 6.4 a 6.7 se miden sobre la **muestra**: las 120 entradas de `plan_dataset(cfg)` de las identidades 1 a 3 (40 variantes cada una: 24 de train, 8 de val y 8 de test), generadas con la receta de la sección 4 de H1 (`base = generar_base(tipo, semilla, cfg)`, `rng = np.random.default_rng([semilla, v])`, `params = sortear_parametros(rng, cfg)`, `variante = aplicar_variante(base, params, rng)`) y llevadas a la ventana con `h = preprocesar(variante.imagen, cfg)`. De las 120, 30 tienen corte (11, 9 y 10 por identidad). El escalador de la muestra es `ajustar_escalador` con los bloques de sus 72 variantes de train, y `Z` son los 120 vectores escalados. El **centroide** de una identidad es la media de los `Z` de sus 24 variantes de train.

## 6. Criterios que verifican los tests

### 6.1 Configuración

- Las claves de la sección 1 existen, con su tipo y rango (listas estrictamente crecientes), y `cobertura_min` y `borde_minucia_px` tienen sus valores fijos (0.5 y 14.0).
- `cargar_config` lanza `ValueError` que nombra el camino completo si falta cualquiera de las claves de la sección 1.

### 6.2 PCA propio (datos aleatorios con semilla, de rango completo)

- **Oráculo:** con `X` de formas `(40, 5)`, `(20, 60)` y `(100, 30)` (columnas correlacionadas y media distinta de 0) y `k` en {1, 2, `min(n − 1, d)`}: `media`, `varianza`, `proporcion_varianza`, `componentes` y `transformar` de `X` y de filas nuevas son iguales a los de `sklearn.decomposition.PCA(n_components=k, svd_solver="full")` (`mean_`, `explained_variance_`, `explained_variance_ratio_`, `components_`, `transform`) salvo el signo de cada componente, con tolerancia 1e-6 (relativa en `varianza`).
- `componentes` tiene filas ortonormales (`componentes · componentesᵀ = I`, 1e-9), `varianza` es no creciente y ≥ 0, y en cada componente la entrada de mayor valor absoluto es positiva.
- Las proyecciones de las filas de ajuste tienen media 0 (1e-9) y varianza con `ddof=1` igual a `varianza` (relativa 1e-9).
- `ajustar` devuelve el mismo objeto; `ajustar_transformar(X)` es igual a `ajustar(X).transformar(X)`; `transformar` de un vector `(d,)` da `(k,)`, igual a la fila correspondiente; con `n_componentes=None`, `k = min(n − 1, d)`. No modifica `X` y es determinista.
- **Errores:** `ValueError` con `n_componentes` 0, −1, 2.5 o `True`; con `X` de 1 o 3 dimensiones, con una sola fila, con NaN o Inf, con todas las filas iguales o con `k > min(n − 1, d)`; con `transformar` de otro `d`. `RuntimeError` con `transformar` antes de `ajustar`.

### 6.3 Silhouette propia

- Igual a `sklearn.metrics.silhouette_score(X, etiquetas)` (tolerancia 1e-9) con datos aleatorios de 2 a 5 grupos, en escalas de 1e-3 a 1e3, con etiquetas enteras y de texto, y con un grupo de una sola fila.
- `ValueError` con una sola etiqueta distinta, con tantas etiquetas distintas como filas, con largos distintos y con NaN.
- `src/core_ml/pca.py`, `src/core_ml/metricas.py` y `src/huellas/caracteristicas.py` no importan scikit-learn (el texto de los archivos no contiene `sklearn`).

### 6.4 Formato y definiciones exactas (muestra)

- `largos_de_bloque(cfg) == {"orientacion": 50, "gabor": 300, "minucias": 25}` con el `cfg` real. `caracteristicas` devuelve un dict con exactamente las claves de `BLOQUES`, cada una `float64` de 1 dimensión con ese largo, sin ±Inf. No modifica `h` y dos llamadas dan arrays iguales (`np.array_equal`, con `equal_nan=True`).
- **Oráculo de los bloques:** en cada variante de la muestra, el bloque de orientación, el de minucias y `extraer_minucias` son exactamente los de la sección 4 (mismos NaN; valores con tolerancia 1e-9; mismas minucias en el mismo orden). El bloque de Gabor, igual a la sección 4.2 con tolerancia relativa 1e-6, en al menos 6 variantes (2 por identidad, al menos 3 con corte).
- **Minucias:** en cada variante, toda terminación está en `esqueleto` con CN = 1 en el esqueleto limpio y toda bifurcación con CN = 3; ninguna minucia está a menos de `borde_minucia_px` del borde de la máscara (`D ≥ 14`); no hay dos terminaciones a menos de `par_terminaciones_px` ni dos bifurcaciones a menos de `par_bifurcaciones_px`.
- **Celdas sin huella:** en cada variante, una celda (o un sector o un anillo) es NaN si y solo si tiene menos de `cobertura_min` de su máscara (`mascara`, `G` o `V`, según el bloque); los 3 valores globales de minucias y los `F` globales de Gabor no son NaN, y los `F` globales suman 1 (1e-9).
- **Caso sintético de corte:** una `HuellaPreprocesada` con `mascara` en `True` solo en las columnas `x < 180` (un borde recto vertical, como el de un corte), `esqueleto` en `True` en las filas `y = 24, 36, …, 228` (cada 12 px, 18 filas) entre `x = 20` y `x = 179`, `realzada` en 255 y `orientacion` en 0: `extraer_minucias` da exactamente las 18 terminaciones `(20, y)` (a 21 px del borde de la ventana) y ninguna en `x = 179` (a 1 px del borde de la máscara), y ninguna bifurcación.
- **Caso sintético de cobertura:** con `mascara` en `True` solo en `x < 175`, el bloque de orientación tiene NaN exactamente en las 5 celdas de la columna `j = 4` (`x ≥ 200`); las de `j = 3` (`150 ≤ x < 200`), con cobertura 0,5 justa, no son NaN.
- **Sin huella:** con `mascara` vacía (y `esqueleto` vacío), los tres bloques tienen el largo correcto y son todos NaN, sin excepción.
- `ValueError` si `realzada`, `mascara`, `esqueleto` u `orientacion` no son de 250 × 250.

### 6.5 Escalador y vector (muestra)

- `Z` tiene `d = Σ columnas usadas` columnas, sin NaN ni Inf, y en las columnas usadas `medias` y `desvios` son la media y el desvío (`ddof=1`) de los valores no NaN de train (1e-9). La media de cada columna de `Z` sobre las 72 de train es 0 (1e-9).
- **Imputación:** cada entrada NaN de una columna usada vale exactamente 0 en `Z`. Las columnas descartadas son exactamente las que tienen menos de 2 valores o desvío ≤ 1e-9 en train, y `pesos[b] = 1/√(columnas[b].sum())`.
- `transformar` de un vector da la fila correspondiente de `transformar` del lote; `ValueError` con una clave de menos o de más, con un bloque de otro largo y con un Inf; `ajustar_escalador` da `ValueError` con una sola fila.
- `transformar` trabaja fila por fila: la fila de una variante no depende de las otras filas del lote.
- **Bloques equilibrados (plan):** la contribución de cada bloque sobre las 72 de train está en [1,6, 2,4] (2 ± 20 %).

### 6.6 Separación (muestra; silhouette con las identidades como etiquetas)

- **Plan:** la silhouette de `Z` en las 72 de train es ≥ 0,50, y la de `silhouette` propia es igual a la de scikit-learn (1e-9).
- La silhouette de `Z` en las 24 autorizadas de val es ≥ 0,50.
- **Cada bloque por separado** (sus columnas de `Z`): silhouette ≥ 0,30 en las 24 de val.
- **PCA de la muestra:** el `PCA` propio ajustado con las 72 de train de `Z` da las mismas proyecciones de las 120 que scikit-learn, salvo el signo de cada componente (1e-6).

### 6.7 Cortes parciales (muestra y bases)

- **Densidad de minucias** (`dens_t + dens_b`, los dos primeros valores del bloque): por identidad, la media de las variantes con corte dividida por la de las variantes sin corte está en [0,85, 1,15]. Muestra que el filtro del borde saca las minucias falsas del corte sin sacar de más.
- **Distancia al centroide propio** (`‖Z_i − centroide de su identidad‖`): por identidad, la media de las variantes con corte es ≤ 1,5 veces la de las variantes sin corte.
- **Corte directo:** para cada identidad y cada `corte_angulo_grados` `a` en {0, 90, 180, 270}, la variante `ParametrosVariante(12.0, 1.0, 10.0, −8.0, 0.0, 0.0, f, a, 0.0, 0.0)` de su base, con `f` el máximo de `corte_periferico` (0,10) y `np.random.default_rng(0)` (la de 5.7 de H2), preprocesada y escalada con el escalador de la muestra, está más cerca del centroide de su identidad que de los otros dos, a una distancia no mayor que la de la variante de train de esa identidad más lejana de su centroide.

### 6.8 Archivo de características y figura (dataset chico en `tmp_path`)

Un dataset chico escrito por el test en `tmp_path`: los PNG (`cv2.imwrite`) de 12 variantes de train (las 4 primeras de cada identidad), 6 de val (2 por identidad) y las 3 variantes de los intrusos 1 y 2 (val), en las carpetas y con los nombres de `archivo` de `plan_dataset`, y un `metadatos.csv` con al menos las columnas `archivo, particion, identidad, intruso, tipo`.

- `generar_caracteristicas(cfg, carpeta)` devuelve `carpeta / "caracteristicas.npz"` y no crea otros archivos. El npz tiene exactamente las claves de 4.5, con una fila por fila de `metadatos.csv` en el mismo orden y los valores de las columnas leídas.
- Cada fila de los bloques crudos es `caracteristicas(preprocesar(png leído), cfg)` (igual con NaN), y `X` es `ajustar_escalador` de las filas de train aplicado a todas las filas (1e-12).
- **Sin fuga (plan: escalador y pesos ajustados sin usar test):** en otra carpeta con los mismos PNG de train y otros PNG de val (otras variantes de las mismas identidades, con los mismos nombres de archivo), las filas de train de `X` son idénticas.
- Con el npz ya escrito y `sobrescribir=False`: `FileExistsError` y el archivo no cambia (mismo contenido y fecha de modificación); con `sobrescribir=True` lo vuelve a escribir. Sin filas de train: `ValueError` y no se escribe el npz.
- `generar_figura(cfg, carpeta, tmp_path / "pca_h3.png")` escribe un PNG que se puede leer con `cv2.imread` y devuelve esa ruta.

### 6.9 Aceptación (`@pytest.mark.aceptacion`)

- Con el dataset completo generado por `generar_dataset(cfg, tmp)` de H1: `generar_caracteristicas(cfg, tmp)` tarda menos de 3 minutos en la notebook; el npz tiene 300 filas y `X` es finito; la silhouette de `X` es ≥ 0,50 en las 72 de train y en las 24 autorizadas de val; la contribución de cada bloque en train está en [1,6, 2,4]; el `PCA` propio ajustado con las filas de train de `X` da las mismas proyecciones de las 300 filas que scikit-learn salvo el signo (1e-6); y `generar_figura` escribe la figura en `tmp`.

## 7. Verificación manual (no va en pytest; se anota en el Documento 2)

- En `docs/figuras/pca_h3.png` se ven tres grupos separados en train; las variantes de val y de test caen en el grupo de su identidad, también las que tienen corte; los intrusos quedan entre los grupos o afuera.
- La silhouette por bloque y la contribución de cada bloque que imprime la línea de comandos se anotan en la ficha de H3, con el tiempo por huella.

## 8. Aclaraciones para los tests

- **Valores del prototipo** (las definiciones de la sección 4 sobre las salidas de H2 de la muestra): silhouette 0,672 en train y 0,700 en val; por bloque en val, orientación 0,865, Gabor 0,832 y minucias 0,512; contribuciones en train 1,851, 1,990 y 1,852; densidad con corte / sin corte 1,027, 1,059 y 1,033; distancia al centroide con corte / sin corte 1,250, 0,980 y 1,085; corte directo: las 12 asignadas a su identidad, a ≤ 0,83 de la distancia máxima de train (la peor, la identidad 3 con el corte a 180°); PCA igual a scikit-learn hasta 1e-11. En train, la columna 9 del bloque de minucias (la celda de la esquina superior izquierda de la grilla de 4 × 4) no tiene datos y se descarta: `d = 50 + 300 + 24 = 374`.
- **Tiempo:** `caracteristicas` tarda unos 50 ms por huella y `preprocesar`, unos 0,15 s. La muestra (3 bases y 120 variantes), sus resultados de H2 y sus bloques se calculan una sola vez (fixtures con `scope="module"`): menos de un minuto. El oráculo del bloque de Gabor es el más lento (12 filtros de 250 × 250 por variante); por eso alcanza con 6 variantes.
- El oráculo se escribe en el test a partir de la sección 4, sin importar nada de `src/huellas/caracteristicas.py`; puede usar `cv2`, `scipy.ndimage` y NumPy. Las coordenadas de las minucias son enteras (en `float64`), así que se comparan con `==`.
- **Recorrido de las espuelas (4.3, paso 3):** es una búsqueda en anchura desde `t` por los píxeles de `E` con vecinos-8, que no sigue desde las uniones (las alcanza, pero no pasa a través de ellas) y llega hasta `espuela_px` pasos. Las otras terminaciones sí se atraviesan.
- **Variantes con corte:** las de `params.corte_fraccion > 0`.
- **Contribución** (6.5 y 6.9): las columnas del bloque `b` en `Z` son las `columnas[b].sum()` que siguen a las de los bloques anteriores, en el orden de `BLOQUES`.
- **Silhouette por bloque** (6.6): con las columnas del bloque en `Z`; como la silhouette no cambia al multiplicar por una constante, el peso no influye.
- **Dataset chico de 6.8:** los PNG salen de la receta de H1 (`variante.imagen`) y se escriben con `cv2.imwrite` en `carpeta / archivo` (creando `train/` y `val/`). El `metadatos.csv` puede tener solo esas 5 columnas, UTF-8 y separado por comas; `identidad` e `intruso` como en H1 (0 en lo que no corresponde).
- **Sin fuga (6.8):** por ejemplo, la segunda carpeta tiene los mismos 12 PNG de train y, con los nombres de las 6 de val, las imágenes de las variantes 27 y 28 de cada identidad (sin los intrusos, o con los mismos); `X` se compara con `np.array_equal`.
- **`HuellaPreprocesada` armada a mano** (casos sintéticos de 6.4): los campos que H3 no usa (`binaria`, `punto_referencia`, `metodo`, `nucleos`, `angulo` y `matriz`) pueden tener cualquier valor del tipo de la sección 2 de H2 (por ejemplo `binaria = esqueleto`, `(124.5, 124.5)`, `"curvatura"`, `()`, `0.0` y la identidad de 2 × 3). Con `realzada` uniforme, los globales de Gabor son NaN (4.2); esos casos solo miran `extraer_minucias` y el bloque de orientación.
- **Sección 1**, como en H2: los `<float …>` exigen `float` de Python (`8` no sirve, `8.0` sí), también dentro de las listas; los enteros, `int` que no sea `bool`; `cobertura_min` y `borde_minucia_px` se comparan por tipo y valor; si falta la sección `huellas.caracteristicas` entera, el `ValueError` nombra sus 14 claves.
- **Errores de `ajustar_escalador` y `transformar`:** con dicts armados a mano (por ejemplo, con los bloques de la muestra).

### Sumadas en la versión 1.1 (lecturas del chat de tests)

El chat que escribe `tests/test_h3.py` no hizo preguntas: armó los oráculos de la sección 4, dieron los valores del prototipo de esta sección y dejó estas lecturas en el docstring del test. Santiago las confirmó el 23/10.

- **Oráculo de Gabor (6.4):** se compara en las 120 variantes de la muestra (el contrato pide al menos 6).
- **PCA de la muestra y del dataset (6.6 y 6.9):** "las mismas proyecciones que scikit-learn" se prueba con `n_componentes=None` (con las 72 filas de train, `k = 71`) y con `n_componentes=3` (el de la figura), contra `sklearn.decomposition.PCA(n_components=k, svd_solver="full")`. Con `k = 71`, el oráculo difiere de scikit-learn en 2e-12.
- **6.7:** la densidad de minucias y la distancia al centroide se promedian, por identidad, sobre sus 40 variantes de la muestra (train, val y test), separadas en con corte y sin corte. Con esa lectura salen los valores de arriba; con solo val y test, las distancias darían 1,065, 1,051 y 0,970.
- **Umbrales de silhouette (6.6 y 6.9):** se miden con `sklearn.metrics.silhouette_score`; la silhouette propia se compara aparte con la de scikit-learn (6.3 y 6.6).
- **Tolerancias:** son absolutas salvo donde dice "relativa". "Es igual" sin tolerancia (6.2: `ajustar_transformar`, `transformar` de un vector y el determinismo; 6.5: `transformar` de un vector y fila por fila) se compara a 1e-12; `np.array_equal` (6.4 y 6.8), exacto.
- **Dataset chico (6.8):** los PNG se escriben con `cv2.imencode` y `tofile` (los mismos bytes que `cv2.imwrite`, sin problemas con tildes en la ruta). Las 2 de val de cada identidad son las dos primeras de val del plan (variantes 25 y 26); la carpeta sin fuga tiene, con esos nombres, las variantes 27 y 28, y los mismos intrusos. "Con `sobrescribir=True` lo vuelve a escribir": se reemplaza el npz por bytes que no son un npz y tiene que volver a quedar el mismo npz.
- **Todas las filas iguales (6.2):** el test usa 8 copias de una fila con valores exactos en binario (`[3.0, -1.0, 2.0, 0.5, 10.0]`), que da `ValueError` con las dos condiciones de la regla 3.
- **`FileNotFoundError` de 4.5:** no se prueban, porque no están en la sección 6; el código los lanza igual.
