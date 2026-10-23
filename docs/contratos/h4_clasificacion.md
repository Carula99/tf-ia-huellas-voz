# Contrato · H4 · K-means propio, distancia normalizada y umbral de rechazo

**Versión 1.1 · 23/10/2026** (la 1 es del mismo día). **Cambio en la 1.1:** se suman a la sección 8 las lecturas del contrato que dejó anotadas el chat que escribe `tests/test_h4.py`, con las respuestas de Santiago a sus preguntas (los tipos de los escalares y `far_max_val` con el criterio `eer`). Ninguna definición ni criterio cambia de umbral: se precisan. Se anota en el Documento 2 junto con los tests.

Es lo que verifica `tests/test_h4.py`. Los tests se escriben en otro chat solo a partir de este archivo y se congelan (commit) antes de implementar. Se apoya en los contratos de la Fase 0 (`config.yaml`, `src/config.cargar_config` y `RAIZ`), de H1 (`plan_dataset`, `generar_dataset`, `metadatos.csv` y la receta de la sección 4), de H2 (`preprocesar`) y de H3 (`BLOQUES`, `caracteristicas`, `Escalador`, `ajustar_escalador`, `generar_caracteristicas` y el archivo `caracteristicas.npz` de su sección 4.5). Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

Esta etapa suma a `src/core_ml/` el K-means propio (`kmeans.py`) y la matriz de confusión (en `metricas.py`, que también va a usar V4), y crea `src/huellas/clasificacion.py`. Con H4 el carril de huellas queda listo para la integración (Fase 5): `reconocer` lleva una imagen hasta la decisión.

**Cambios respecto del plan v0.3.2** (medidos con un prototipo sobre `caracteristicas.npz` y sobre una muestra ampliada de 600 huellas nuevas del generador que no son parte del dataset; quedan en el Documento 2, con la figura del prototipo):

- **Distancia normalizada** (la opción del plan): el puntaje de una huella es su distancia euclídea al centroide de cada cluster dividida por la dispersión de ese cluster en train, `r_k = √(inercia del cluster / filas del cluster)`, y se toma el menor. Separa mejor que la euclídea: en la muestra ampliada, error de igual tasa 0,3 % contra 0,7 %; en test, 1,1 % contra 2,2 %. **Piso de la dispersión** (pedido de Santiago): `r_k` no baja de `piso_dispersion · r̄` (0,5 de la dispersión de todo train), así un cluster de una sola fila o sin dispersión no deja la distancia dividida por 0. En el dataset no actúa (el menor `r_k` es 0,898 · r̄).
- **Umbral:** criterio `far_max` con `far_max_val` 0,01 (con 90 intrusos de val, ninguno aceptado), y **τ\* en la mitad del intervalo** de τ que cumple el criterio, no en su borde. El plan decía "EER o FAR ≤ 5 %" sin decir dónde, dentro del intervalo, va τ\*. El adelanto de H4 que se midió en H3 lo ponía en el borde de abajo, pegado a la autorizada de val más lejana: FRR 8,3 % en test. En 2000 pares val/test del tamaño del dataset sorteados de la muestra ampliada, la probabilidad de cumplir FRR ≤ 10 % y FAR ≤ 5 % en test pasa del 85,8 % (euclídea, borde de abajo) al 99,6 % (normalizada, `far_max` 0,01, mitad). Con el EER en la mitad da 97,1 %: en el dataset, los dos dan el mismo τ\*; difieren cuando val no queda separable, porque el EER deja entrar intrusos para emparejar con el FRR. El criterio `eer` sigue definido (sección 4.4) porque la Fase 0 lo admite.
- **El modelo se guarda en `.npz`** (sin objetos de Python, se lee con `allow_pickle=False`), no en `.joblib`: no ejecuta código al cargarlo y sigue el formato de `caracteristicas.npz`.
- **Matriz de confusión propia** en `src/core_ml/metricas.py`, comparada en los tests con `sklearn.metrics.confusion_matrix`.
- **Identificación y FRR:** la identificación se mide sin umbral (la identidad del centroide más cercano), y el FRR cuenta como error toda autorizada que no queda aceptada con su propia identidad (rechazada o aceptada como otra).

## 0. Convenciones

- Las identidades autorizadas son `1, …, k` (`k = huellas.kmeans.k = 3`); **`NO_AUTORIZADO = 0`**, el mismo valor que tienen los intrusos en la columna `identidad` de H1 y H3. `verdad` es la identidad real (0 en los intrusos), `prediccion` la identidad del centroide más cercano y `decision` lo que decide el sistema (una identidad o 0).
- Distancia euclídea en el espacio de `X` de H3 (las `d = 374` columnas escaladas). Los empates de un mínimo (`argmin`) se resuelven con el primero (el índice menor).
- **Autorizada bien identificada:** una fila con `verdad ≠ 0` y `prediccion == verdad`.
- **Se acepta** una huella si su puntaje es `≤ τ` (con el puntaje justo igual a τ, se acepta).
- **FAR(τ)** = fracción de los intrusos con puntaje `≤ τ`. **FRR(τ)** = fracción de las autorizadas que no quedan aceptadas con su propia identidad: las de puntaje `> τ` y las de `prediccion ≠ verdad` (estas, con cualquier τ). Las dos son funciones escalonadas: FAR no decrece con τ, FRR no crece, y solo cambian en los puntajes observados.
- `float` de Python para los escalares que devuelven las funciones (`tau`, `inercia`, `pureza`, FAR, FRR); `int64` para etiquetas e identidades en arrays.

## 1. config.yaml

Las claves de `huellas.kmeans` y `huellas.umbral` ya existen (Fase 0). Esta etapa fija estos valores y suma `piso_dispersion`, que `cargar_config` también exige (si falta, `ValueError` con su camino completo):

```yaml
huellas:
  kmeans:
    k: 3                    # fijo (Fase 0)
    n_init: 15              # fijo (Fase 0)
    max_iter: 300
    tol: 1.0e-4
  umbral:
    criterio: far_max       # H4: antes eer
    far_max_val: 0.01       # H4: antes 0.05; con 90 intrusos de val, ninguno aceptado
    piso_dispersion: 0.5    # fijo: r_k no baja de esta fracción de r̄ (sección 4.2)
```

La semilla del K-means es `semilla_global` (2026). Las metas de `metas` (identificación ≥ 0,95, FAR ≤ 0,05, FRR ≤ 0,10) no cambian.

## 2. src/core_ml/kmeans.py y src/core_ml/metricas.py

```python
# src/core_ml/kmeans.py
class KMeans:
    def __init__(self, k: int, n_init: int = 15, max_iter: int = 300, tol: float = 1e-4,
                 semilla: int | None = None) -> None: ...
    def ajustar(self, X: np.ndarray, init: np.ndarray | None = None) -> "KMeans": ...   # devuelve self
    def predecir(self, X: np.ndarray) -> np.ndarray: ...     # índice del centroide más cercano
    def distancias(self, X: np.ndarray) -> np.ndarray: ...   # euclídeas a cada centroide
    # Atributos después de ajustar:
    #   centroides: (k, d) float64
    #   etiquetas: (n,) int64           cluster de cada fila de X (0 a k − 1)
    #   inercia: float                  Σ de las distancias² de cada fila a su centroide
    #   inercia_cluster: (k,) float64   la misma suma, por cluster
    #   n_iter: int                     iteraciones de Lloyd del arranque elegido

# src/core_ml/metricas.py (se suma a silhouette, que no cambia)
def matriz_confusion(verdad, prediccion, etiquetas) -> np.ndarray: ...
```

**KMeans:**

1. `k`, `n_init` y `max_iter` son `int` ≥ 1 (no `bool`); `tol` es un número real finito ≥ 0; `semilla` es `None` o un entero. Otra cosa: `ValueError` en el constructor.
2. `ajustar(X, init)`: `X` es un array de 2 dimensiones `(n, d)` con `n ≥ k`, `d ≥ 1`, valores finitos y al menos `k` filas distintas; `init`, si se da, es `(k, d)` y finito. Si no: `ValueError`. Se trabaja en `float64` y no se modifica `X`. El algoritmo es el de la sección 4.1.
3. `predecir(X)` y `distancias(X)` aceptan `(n, d)` y devuelven `(n,)` y `(n, k)`, o un vector `(d,)` y devuelven un entero y `(k,)`. Antes de `ajustar`: `RuntimeError`. Otro `d`, otra cantidad de dimensiones o valores no finitos: `ValueError`. `predecir(X)` es el `argmin` de `distancias(X)`.
4. Con la misma `semilla` y los mismos datos, el resultado es el mismo. Solo NumPy: el archivo no importa scikit-learn.

**matriz_confusion(verdad, prediccion, etiquetas):** `M[i, j]` es la cantidad de posiciones con `verdad == etiquetas[i]` y `prediccion == etiquetas[j]` (filas: lo real; columnas: lo predicho), en `int64`, de `len(etiquetas) × len(etiquetas)`. Igual a `sklearn.metrics.confusion_matrix(verdad, prediccion, labels=etiquetas)`. `ValueError` si los largos no coinciden, si `etiquetas` está vacía o repite valores, o si algún valor de `verdad` o `prediccion` no está en `etiquetas`. Acepta enteros y textos. Solo NumPy.

## 3. src/huellas/clasificacion.py

```python
NO_AUTORIZADO: int = 0

@dataclass(frozen=True)
class ModeloHuellas:
    identidades: np.ndarray    # int64 (k,): 1, …, k
    centroides: np.ndarray     # float64 (k, d), en el orden de identidades
    dispersion: np.ndarray     # float64 (k,): r_k, con el piso
    tau: float                 # τ*
    pureza: float              # pureza de los clusters en train
    inercia: float             # inercia del K-means en train
    escalador: Escalador | None    # el de H3, ajustado con train (None solo en modelos armados sin él)
    def puntajes(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def decidir(self, X: np.ndarray) -> np.ndarray: ...

@dataclass(frozen=True)
class Reconocimiento:
    identidad: int         # la decisión: 1 a k, o 0 si se rechaza
    mas_cercana: int       # identidad del centroide más cercano (con distancia normalizada)
    distancia: float       # el puntaje: distancia normalizada a ese centroide
    tau: float             # τ* del modelo

def correspondencia(etiquetas, identidades, k: int) -> tuple[dict[int, int], float]: ...
def dispersiones(X, etiquetas, centroides, piso: float) -> np.ndarray: ...
def far_frr(puntaje, prediccion, verdad, tau: float) -> tuple[float, float]: ...
def calibrar_umbral(puntaje, prediccion, verdad, criterio: str, far_max_val: float) -> float: ...
def decidir(puntaje, prediccion, tau: float) -> np.ndarray: ...
def evaluar(decision, prediccion, verdad, k: int) -> dict: ...
def ajustar_modelo(X_train, identidades_train, X_val, verdad_val, cfg: dict,
                   escalador: Escalador | None = None) -> ModeloHuellas: ...
def guardar_modelo(modelo: ModeloHuellas, ruta: str | os.PathLike) -> Path: ...
def cargar_modelo(ruta: str | os.PathLike) -> ModeloHuellas: ...
def entrenar(cfg: dict, carpeta: str | os.PathLike | None = None,
             salida: str | os.PathLike | None = None, sobrescribir: bool = False) -> Path: ...
def evaluar_particion(modelo: ModeloHuellas, cfg: dict, carpeta: str | os.PathLike | None = None,
                      particion: str = "test") -> dict: ...
def reconocer(imagen: np.ndarray, modelo: ModeloHuellas, cfg: dict) -> Reconocimiento: ...
def generar_figura(cfg: dict, carpeta: str | os.PathLike | None = None,
                   modelo: str | os.PathLike | None = None,
                   salida: str | os.PathLike | None = None) -> Path: ...
```

Por defecto, `carpeta` es `RAIZ / rutas.huellas` (donde está `caracteristicas.npz`), el modelo es `RAIZ / rutas.modelos / "modelo_huellas.npz"` y la figura, `RAIZ / rutas.figuras / "far_frr_h4.png"`.

**Línea de comandos:** `venv\Scripts\python -m src.huellas.clasificacion [--sobrescribir]` corre `entrenar` y `generar_figura` con `config.yaml` e imprime la pureza, la inercia, las tres `r_k`, τ\*, FAR y FRR de val, identificación, FAR y FRR de test, la matriz de confusión de test y el puntaje de las 6 variantes reservadas para la demo (39 y 40 de cada identidad) contra τ\*.

## 4. Definiciones exactas

### 4.1 K-means (`KMeans.ajustar`)

1. **Tolerancia:** `tol_abs = tol · media(np.var(X, axis=0))` (varianza con `ddof = 0`; la convención de scikit-learn).
2. **Arranques:** sin `init`, se crea un único generador `rng = np.random.default_rng(semilla)` al empezar `ajustar` y se hacen `n_init` arranques en orden, todos con ese generador. Cada arranque empieza con **k-means++ "greedy"** (Arthur y Vassilvitskii, 2007; la variante de scikit-learn): el primer centroide es una fila de `X` elegida al azar con probabilidad uniforme; para cada uno de los siguientes, con `D²(x)` la distancia² de cada fila al centroide ya elegido más cercano, se sortean `2 + ⌊ln k⌋` filas candidatas con probabilidad proporcional a `D²`, y queda la candidata que hace menor `Σ_x min(D²(x), ‖x − candidata‖²)` (la primera, si empatan). Con `init`, hay un solo arranque desde `init` (`n_init` no se usa y `rng` tampoco).
3. **Lloyd** desde los centroides iniciales `C`, hasta `max_iter` iteraciones: cada fila va al centroide más cercano (`argmin`); cada centroide pasa a ser la media de sus filas (un cluster que se queda sin filas conserva su centroide); `mov = Σ_j ‖C_nuevo_j − C_j‖²` y `C = C_nuevo`. Se para si las etiquetas de esta iteración son iguales a las de la anterior o si `mov ≤ tol_abs`. `n_iter` es la cantidad de iteraciones hechas.
4. **Final de cada arranque:** `etiquetas` = centroide más cercano con los centroides finales (si se paró por `max_iter` o por `tol_abs`, pueden diferir de las de la última iteración); `inercia = Σ_i ‖x_i − C[etiquetas_i]‖²` e `inercia_cluster[j]` la misma suma sobre las filas de `j`.
5. Queda el arranque de **menor inercia** (el primero, si empatan), con sus `centroides`, `etiquetas`, `inercia`, `inercia_cluster` y `n_iter`.

### 4.2 Correspondencia y dispersión

- **`correspondencia(etiquetas, identidades, k)`:** para cada cluster `j = 0 … k − 1`, la identidad con más filas de train en el cluster (si empatan, la menor). Devuelve `(mapa, pureza)`, con `mapa[j]` esa identidad y `pureza = Σ_j (filas de la identidad mayoritaria de j) / n`. `ValueError` si los largos no coinciden, si un cluster no tiene filas o si dos clusters van a la misma identidad (la correspondencia tiene que ser uno a uno).
- **`dispersiones(X, etiquetas, centroides, piso)`:** con `I_j = Σ_{etiquetas = j} ‖x − c_j‖²`, `n_j` las filas del cluster `j` y `r̄ = √(Σ_j I_j / n)` (la dispersión de todo el conjunto): `r_j = max(√(I_j / n_j), piso · r̄)`. Devuelve `(k,)` en `float64`. `ValueError` si un cluster no tiene filas, si `r̄ = 0` o si `piso` no está en `(0, 1]`. Un cluster de una sola fila tiene `I_j = 0` y queda con `r_j = piso · r̄`.

### 4.3 Puntaje y decisión

- **Puntaje** de una fila `x` con un modelo: `d_j(x) = ‖x − centroides[j]‖ / dispersion[j]`; `j* = argmin_j d_j(x)`; el puntaje es `d_{j*}(x)` y la predicción, `identidades[j*]`. `ModeloHuellas.puntajes(X)` devuelve `(puntaje (n,), prediccion (n,) int64)` para `(n, d)`, o `(float, int)` para un vector `(d,)`; `ValueError` con otro `d` o valores no finitos.
- **`decidir(puntaje, prediccion, tau)`:** `prediccion` donde `puntaje ≤ tau` y `NO_AUTORIZADO` donde no, en `int64`. `ModeloHuellas.decidir(X)` es `decidir(*self.puntajes(X), self.tau)` (un entero para un vector).

### 4.4 Calibración del umbral (`calibrar_umbral`)

Con los puntajes, predicciones y verdades de val: `n_a` autorizadas y `n_i` intrusos. FAR y FRR son las de la sección 0 (`far_frr(puntaje, prediccion, verdad, tau)` devuelve `(FAR, FRR)` en un τ). En los dos criterios, τ\* es la **mitad de un intervalo `[a, b)`** de valores de τ:

- **`far_max`:** `m = ⌊far_max_val · n_i + 1e-9⌋` intrusos se pueden aceptar (el 1e-9 evita que el redondeo baje un entero exacto, como 0.3 · 10). `b` es el `(m + 1)`-ésimo menor puntaje de los intrusos (con `far_max_val < 1`, siempre existe): para `τ < b`, FAR ≤ `far_max_val`. Dentro de `τ < b`, el menor FRR se alcanza desde `a` = el mayor puntaje de una autorizada bien identificada que sea `< b`. **τ\* = (a + b) / 2.** Si no hay ninguna autorizada bien identificada con puntaje `< b`: `ValueError`.
- **`eer`:** con `E(τ) = max(FAR(τ), FRR(τ))`, el conjunto de τ donde `E` es mínimo es un intervalo `[a, b)` (FAR no decrece y FRR no crece): `a` es el puntaje observado (o −∞) donde empieza y `b`, el siguiente puntaje observado después del último donde vale ese mínimo. **τ\* = (a + b) / 2.** Si `a = −∞` o `b = +∞`: `ValueError`.
- **Caso separable** (todas las autorizadas bien identificadas por debajo de todos los intrusos): con los dos criterios y `far_max_val < 1 / n_i`, `a` es el mayor puntaje de las autorizadas, `b` el menor de los intrusos y τ\* queda justo en el medio.
- `ValueError` si los largos no coinciden, si hay puntajes no finitos, si no hay autorizadas o no hay intrusos, si `criterio` no es `"eer"` ni `"far_max"` o si `far_max_val` no está en `(0, 1)`.

### 4.5 Evaluación (`evaluar`)

Con `decision`, `prediccion` y `verdad` de un conjunto: `identificacion` = fracción de las autorizadas con `prediccion == verdad`; `far` = fracción de los intrusos con `decision ≠ 0`; `frr` = fracción de las autorizadas con `decision ≠ verdad`; `matriz = matriz_confusion(verdad, decision, [1, …, k, 0])`, de `(k + 1) × (k + 1)` (filas: identidad real 1 a k y "no autorizado"; columnas: la decisión). Devuelve un dict con exactamente esas 4 claves. `ValueError` si los largos no coinciden, si no hay autorizadas o no hay intrusos.

### 4.6 Modelo, archivo, reconocimiento y figura

- **`ajustar_modelo(X_train, identidades_train, X_val, verdad_val, cfg, escalador)`:** `KMeans(k, n_init, max_iter, tol, semilla=cfg["semilla_global"]).ajustar(X_train)` con los valores de `huellas.kmeans`; `mapa, pureza = correspondencia(etiquetas, identidades_train, k)`; los centroides se ordenan por identidad (`identidades = [1, …, k]`, `centroides[i]` el del cluster que va a la identidad `i + 1`); `dispersion = dispersiones(X_train, etiquetas reordenadas, centroides, huellas.umbral.piso_dispersion)`; con los puntajes y predicciones de `X_val`, `tau = calibrar_umbral(…, verdad_val, huellas.umbral.criterio, huellas.umbral.far_max_val)`. No recibe nada de test. `ValueError` si `X_train` o `X_val` no son de 2 dimensiones con el mismo `d` y valores finitos, si los largos no coinciden, si `identidades_train` tiene valores fuera de `1 … k` o `verdad_val` fuera de `0 … k`, y los de las funciones que usa.
- **`guardar_modelo(modelo, ruta)`:** escribe un `.npz` (`np.savez`, sin objetos de Python) con exactamente estas claves: `identidades`, `centroides`, `dispersion`, `tau`, `pureza`, `inercia` y, del escalador, `pesos` (`(3,)`, en el orden de `BLOQUES`) y, para cada bloque `b`, `medias_<b>`, `desvios_<b>` y `columnas_<b>` (16 claves). Crea la carpeta si no existe. Devuelve la ruta. `ValueError` si el modelo no tiene escalador.
- **`cargar_modelo(ruta)`:** lee ese archivo con `allow_pickle=False` y devuelve un `ModeloHuellas` igual al guardado (arrays iguales, con NaN en el mismo lugar, y `Escalador` con los mismos dicts). `FileNotFoundError` si no existe.
- **`entrenar(cfg, carpeta, salida, sobrescribir)`:** lee `carpeta / "caracteristicas.npz"` (de H3); con `tr` las filas de `particion == "train"` y `va` las de `"val"`, arma `escalador = ajustar_escalador({b: bloque[tr] for b in BLOQUES})` (el mismo con el que H3 escaló `X`), `modelo = ajustar_modelo(X[tr], identidad[tr], X[va], identidad[va], cfg, escalador)` y lo guarda en `salida` con `guardar_modelo`. No usa las filas de test. Si `salida` ya existe y `sobrescribir` es `False`: `FileExistsError` sin tocar nada. `FileNotFoundError` si falta `caracteristicas.npz`; `ValueError` si no hay filas de train o de val.
- **`evaluar_particion(modelo, cfg, carpeta, particion)`:** con las filas de esa partición de `caracteristicas.npz`, `evaluar(modelo.decidir(X), prediccion, identidad, k)`.
- **`reconocer(imagen, modelo, cfg)`:** `z = modelo.escalador.transformar(caracteristicas(preprocesar(imagen, cfg), cfg))`, `distancia, mas_cercana = modelo.puntajes(z)`, `identidad = mas_cercana` si `distancia ≤ modelo.tau` y 0 si no. `ValueError` si el modelo no tiene escalador.
- **`generar_figura(cfg, carpeta, modelo, salida)`:** con `caracteristicas.npz` y el modelo guardado, dibuja con matplotlib (backend Agg) dos paneles: FAR y FRR de val en función de τ, con τ\* y el intervalo `[a, b)` marcados, y la matriz de confusión de test de `(k + 1) × (k + 1)` con los números en cada celda. Escribe un PNG en `salida` y devuelve esa ruta.

## 5. Datos de prueba

- **Sintéticos** (6.2 a 6.7): grupos gaussianos con semilla, armados por el test.
- **Dataset chico** (6.8), escrito por el test en `tmp_path` como el de 6.8 de H3 (PNG de `variante.imagen` con `cv2.imencode` y `tofile`, en las carpetas y con los nombres de `archivo` de `plan_dataset`, y un `metadatos.csv` con al menos `archivo, particion, identidad, intruso, tipo`): las variantes 1 a 4 de cada identidad (train), la 25 y la 26 de cada identidad y las 3 variantes de los intrusos 1 y 2 (val), y la 33 y la 34 de cada identidad y las 3 variantes del intruso 31 (test). Después, `generar_caracteristicas(cfg, carpeta)` de H3 escribe ahí `caracteristicas.npz`.
- **Dataset completo** (6.9): `generar_dataset(cfg, tmp)` de H1 y `generar_caracteristicas(cfg, tmp)` de H3.

## 6. Criterios que verifican los tests

### 6.1 Configuración

- `huellas.umbral.criterio` es `"far_max"`, `huellas.umbral.far_max_val` es `0.01` (`float`) y `huellas.umbral.piso_dispersion` es `0.5` (`float`); `huellas.kmeans` tiene `k = 3`, `n_init = 15`, `max_iter = 300` y `tol = 1.0e-4`.
- `cargar_config` lanza `ValueError` que nombra `huellas.umbral.piso_dispersion` si falta esa clave.

### 6.2 K-means propio (datos sintéticos)

- **Oráculo con la misma inicialización:** con varias formas de `X` (por ejemplo `(300, 2)` con `k = 4`, `(60, 40)` con `k = 3` y `(500, 10)` con `k = 5`; grupos juntos y separados) e `init` = `k` filas de `X` elegidas por el test: `centroides`, `etiquetas` e `inercia` iguales a los de `sklearn.cluster.KMeans(n_clusters=k, init=init, n_init=1, max_iter=…, tol=…, algorithm="lloyd")` (`cluster_centers_`, `labels_`, `inertia_`): centroides a 1e-6, inercia relativa 1e-6 y etiquetas idénticas. También con `max_iter` 1 y 2.
- **Oráculo con k-means++:** con `n_init = 15`, la inercia no supera en más del 1 % a la de `sklearn.cluster.KMeans(n_clusters=k, n_init=15)` (cualquier `random_state`; si queda más baja, también vale); con grupos bien separados, además, los centroides emparejados (la permutación que mejor los alinea) son iguales a 1e-6.
- **Propiedades:** `etiquetas` es el `argmin` de `distancias(X)`; `inercia` es la suma de las distancias² de cada fila a su centroide (relativa 1e-9) e `inercia_cluster` la misma suma por cluster; con grupos separados, cada centroide es la media de sus filas (1e-9); `1 ≤ n_iter ≤ max_iter`; `predecir` de un vector es el elemento correspondiente de `predecir` del lote; `ajustar` devuelve el mismo objeto y no modifica `X`.
- **Semilla:** dos ajustes con la misma `semilla` dan lo mismo (a 1e-12); con `n_init = 15` la inercia es ≤ la de `n_init = 1` con la misma `semilla` (el primer arranque es el mismo).
- **Errores:** `ValueError` con `k`, `n_init` o `max_iter` en 0, −1, 2.5 o `True`; con `tol` negativa; con `X` de 1 o 3 dimensiones, con NaN o Inf, con `n < k` o con menos de `k` filas distintas; con `init` de otra forma o con NaN; con `predecir` o `distancias` de otro `d`. `RuntimeError` con `predecir` o `distancias` antes de `ajustar`.

### 6.3 Matriz de confusión

- Igual a `sklearn.metrics.confusion_matrix(verdad, prediccion, labels=etiquetas)` con etiquetas enteras (por ejemplo `[1, 2, 3, 0]`) y de texto, con clases sin ninguna fila y con el orden de `etiquetas` cambiado; `int64`; la suma es `n`.
- `ValueError` con largos distintos, con `etiquetas` vacía o repetida y con un valor que no está en `etiquetas`.
- `src/core_ml/kmeans.py`, `src/core_ml/metricas.py` y `src/huellas/clasificacion.py` no importan scikit-learn (el texto de los archivos no contiene `sklearn`).

### 6.4 Correspondencia y dispersión (datos sintéticos)

- Con 3 grupos separados con identidades conocidas (por ejemplo, el cluster 0 con la identidad 3), `correspondencia` da el mapa correcto y pureza 1; con un cluster que tiene filas de dos identidades, la mayoritaria y la pureza exacta; con un empate, la identidad menor.
- `ValueError` si dos clusters van a la misma identidad, si un cluster no tiene filas o si los largos no coinciden.
- `dispersiones` es exactamente la fórmula de 4.2 (1e-12): en grupos donde el piso no actúa, `r_j = √(I_j / n_j)`; un cluster de una sola fila (o con todas sus filas iguales) queda en `piso · r̄`; con todas las filas de cada cluster iguales a su centroide (`r̄ = 0`), `ValueError`; `ValueError` con `piso` 0 o 1.5.

### 6.5 Puntaje, decisión y umbral (casos armados a mano)

- **Lógica de decisión (plan):** con τ = 1.0, puntajes 0.5, 1.0 y 1.5 dan la predicción, la predicción y 0; `decidir` devuelve `int64` y no cambia la predicción de las aceptadas.
- **Puntaje:** con un modelo armado a mano (`ModeloHuellas` con centroides, dispersiones y τ elegidos por el test y `escalador=None`), `puntajes` es exactamente `min_j ‖x − c_j‖ / r_j` y su `argmin` (1e-12), también cuando la distancia normalizada elige un centroide que no es el más cercano en distancia euclídea; un vector da lo mismo que su fila en el lote.
- **`far_frr`:** en los ejemplos de la sección 8, los valores exactos, contando como error de FRR una autorizada mal identificada aunque su puntaje sea bajo.
- **`calibrar_umbral`:** los τ\* de los tres ejemplos de la sección 8 con los dos criterios (1e-12); en el caso separable, τ\* es el punto medio entre la autorizada más lejana y el intruso más cercano, con los dos criterios; el resultado no cambia al reordenar las filas.
- **Errores de `calibrar_umbral`:** `ValueError` sin intrusos, sin autorizadas, con largos distintos, con NaN, con `criterio` `"otro"`, con `far_max_val` 0 o 1, y con `far_max` cuando ninguna autorizada bien identificada queda por debajo de `b`.

### 6.6 Evaluación (casos armados a mano)

- `evaluar` da `identificacion`, `far`, `frr` y `matriz` exactos en un caso con una autorizada rechazada, una aceptada con otra identidad y un intruso aceptado; la matriz es `(k + 1) × (k + 1)` con el orden `[1, …, k, 0]`. `ValueError` sin autorizadas o sin intrusos.

### 6.7 Modelo (datos sintéticos)

- `ajustar_modelo` con 3 grupos separados en train (con las identidades desordenadas respecto de los clusters) y un val con autorizadas e intrusos lejanos: `identidades == [1, 2, 3]`; cada fila de `centroides` es la media de las filas de train de esa identidad (1e-9); `dispersion` es la de 4.2 con `piso_dispersion` de `cfg`; `tau` es `calibrar_umbral` de los puntajes de val con `criterio` y `far_max_val` de `cfg` (1e-12); `pureza` 1; `inercia` igual a la de `KMeans(…, semilla=cfg["semilla_global"])` con los valores de `huellas.kmeans`.
- **De qué depende cada parte:** con el mismo train, acercar a un centroide una fila de intruso de val (que pase a ser el intruso más cercano) cambia τ y deja iguales los centroides y las dispersiones; cambiar val no toca el K-means. La separación de test está en 6.8 y 6.9.
- `ValueError` si `identidades_train` tiene un 0 o un 4, si `X_val` tiene otro `d` y si los grupos de train no permiten una correspondencia uno a uno (por ejemplo, dos identidades mezcladas en el mismo grupo y un grupo partido en dos).

### 6.8 Archivo del modelo y reconocimiento (dataset chico)

- `entrenar(cfg, carpeta, salida=tmp_path / "modelo.npz")` devuelve esa ruta y el archivo tiene exactamente las 16 claves de 4.6; `cargar_modelo` da un modelo con los mismos valores; `tau` es igual a `calibrar_umbral` de las filas de val del npz con ese modelo (1e-12), y el escalador es igual a `ajustar_escalador` de los bloques de train del npz.
- **Sin fuga:** con las filas de test del npz cambiadas por otras (por ejemplo, otros PNG de test con los mismos nombres, o el npz reescrito con otras filas de test), el modelo guardado es idéntico (`np.array_equal`).
- Con el archivo ya escrito y `sobrescribir=False`: `FileExistsError` y el archivo no cambia (mismo contenido y fecha de modificación); con `sobrescribir=True` se vuelve a escribir. `FileNotFoundError` si falta `caracteristicas.npz`.
- **Reconocimiento:** para cada PNG del dataset chico, `reconocer(imagen leída, modelo, cfg)` da `distancia` y `mas_cercana` iguales a `modelo.puntajes` de su fila de `X` (1e-9) e `identidad` igual a `modelo.decidir` de esa fila; `tau` es el del modelo. `ValueError` con un modelo sin escalador.
- `evaluar_particion(modelo, cfg, carpeta, "test")` es `evaluar` de las filas de test del npz.
- `generar_figura(cfg, carpeta, modelo, tmp_path / "far_frr_h4.png")` escribe un PNG que se puede leer con `cv2.imread` y devuelve esa ruta.

### 6.9 Aceptación (`@pytest.mark.aceptacion`)

Con el dataset completo en `tmp` (sección 5) y `entrenar(cfg, tmp, tmp / "modelo_huellas.npz")`:

- **Pureza (plan):** la de los clusters en train es ≥ 0,95, con correspondencia uno a uno.
- **Oráculo en el dataset (plan):** `KMeans` con `init` = los centroides de `sklearn.cluster.KMeans(n_clusters=3, n_init=15, random_state=0)` ajustado con las filas de train de `X`, y `n_init = 1`, da los mismos centroides e inercia que ese scikit-learn (1e-6, la inercia relativa); los centroides del modelo, emparejados, también son esos (1e-6).
- **Metas en test (plan, sección 1.3):** con el modelo congelado, `evaluar_particion(…, "test")` da identificación ≥ `metas.identificacion_min`, FAR ≤ `metas.far_max` y FRR ≤ `metas.frr_max`.
- **Sin fuga:** τ\* es igual a `calibrar_umbral` con solo las filas de val (1e-12).
- Todo (dataset, características y modelo) tarda menos de 5 minutos en la notebook.

## 7. Verificación manual (no va en pytest; se anota en el Documento 2)

- En `docs/figuras/far_frr_h4.png`, τ\* queda en el medio del intervalo sin errores de val, y la matriz de confusión de test coincide con lo que imprime la línea de comandos.
- La pureza, las `r_k`, τ\*, las métricas de val y de test y la matriz se anotan en la ficha de H4, con el puntaje de las variantes reservadas para la demo y su margen respecto de τ\*.

## 8. Aclaraciones para los tests

- **Valores del prototipo** (las definiciones de la sección 4 sobre `caracteristicas.npz`; `Prototipos\h4_2026-10-23` en OneDrive): K-means en las 72 de train, 30 ms, inercia 31,4493, 2 iteraciones, pureza 1 (los 3 clusters son las 3 identidades y los centroides, sus medias); el mismo óptimo con las semillas 0, 1, 7 y 2026; contra scikit-learn con `init` = centroides, 1e-16. `r_k` = 0,6636, 0,7196 y 0,5934 (`r̄` = 0,6609; el piso de 0,5 no actúa). En val, la autorizada más lejana está en 1,4205 y el intruso más cercano en 1,7021: τ\* = 1,5613 (con `eer` da lo mismo; con `far_max` 0,05, 1,7005). En test: identificación 100 %, FAR 0 % (0 de 90), FRR 4,2 % (1 de 24, `test/id1_v036.png` en 1,721); matriz `[[7,0,0,1],[0,8,0,0],[0,0,8,0],[0,0,0,90]]`. El intruso de test más cercano es `intruso49` (presilla), con sus 3 variantes entre 1,720 y 1,748. Reservadas para la demo: todas aceptadas, `id2_v040` en 1,518 e `id3_v040` en 1,487.
- **K-means contra scikit-learn en el prototipo:** en 10 juegos de datos aleatorios (las formas de 6.2, más `(200, 5)` con `k = 2` y `(120, 8)` con `k = 6`, con grupos juntos y separados), con `init` dado: centroides a ≤ 1,1e-14, inercia relativa ≤ 5,6e-16, etiquetas idénticas e incluso el mismo `n_iter_`, también con `max_iter` 1, 2 y 3 (el `n_iter` no se pide en 6.2). Con k-means++, la inercia queda a ≤ 1,1e-4 relativa; en un caso con grupos juntos los dos llegan a óptimos locales distintos (centroides a 3,6e-2), por eso los centroides emparejados se piden solo con grupos separados. Si un cluster se queda sin filas, scikit-learn lo reubica y este K-means lo deja quieto (4.1, paso 3): los oráculos no tienen que usar datos donde pase eso.
- **Ejemplos de la calibración** (`k = 3`; cada lista tiene 4 autorizadas y 4 intrusos, en el orden `verdad`, `prediccion`, `puntaje`):
  1. **Separable:** `verdad = [1, 2, 3, 1, 0, 0, 0, 0]`, `prediccion = [1, 2, 3, 1, 1, 2, 3, 1]`, `puntaje = [0.5, 0.8, 1.0, 0.7, 1.4, 2.0, 2.5, 3.0]`. Con `far_max` 0.01 y con `eer`: τ\* = 1.2. En τ = 1.2: FAR 0, FRR 0.
  2. **Con solape:** igual, con `puntaje = [0.5, 0.9, 1.3, 0.7, 1.1, 1.6, 2.0, 3.0]`. `far_max` 0.01: `b` = 1.1, `a` = 0.9, τ\* = 1.0 (FAR 0, FRR 0.25). `far_max` 0.30 (`m = 1`): `b` = 1.6, `a` = 1.3, τ\* = 1.45. `eer`: `E` vale 0.25, su mínimo, en `[0.9, 1.6)`, τ\* = 1.25.
  3. **Una autorizada mal identificada:** `verdad = [1, 2, 3, 1, 0, 0, 0, 0]`, `prediccion = [1, 3, 3, 1, 1, 2, 3, 1]`, `puntaje = [0.5, 0.6, 1.0, 0.7, 1.4, 2.0, 2.5, 3.0]`. La segunda nunca cuenta como aceptada bien (FRR ≥ 0.25 en todo τ). `far_max` 0.01: τ\* = 1.2. `eer`: el mínimo 0.25 vale en `[1.0, 2.0)`, τ\* = 1.5 (el EER deja entrar un intruso para emparejar con el FRR).
- **Tiempo:** `KMeans` y `ajustar_modelo` tardan milisegundos. El dataset chico (33 huellas: generación, H2 y H3) se arma una sola vez (fixture con `scope="module"`), en unos 10 s. El de aceptación tarda lo que `generar_dataset` y `generar_caracteristicas` (en la notebook, 57 s y 72 s).
- **Dataset chico:** con solo 4 variantes de train por identidad el K-means separa igual las 3 identidades (pureza 1 en el prototipo) y τ\* sale 2,33; en su test, una de las 6 autorizadas queda rechazada. Los tests de 6.8 no le piden metas.
- **`ModeloHuellas` armado a mano** (6.5): `pureza` e `inercia` pueden valer cualquier `float` y `escalador=None`.
- **Comparaciones:** las tolerancias son absolutas salvo donde dice "relativa"; "igual" sin tolerancia se compara a 1e-12, y "idéntico" o `np.array_equal`, exacto.
- **Sección 1:** como en H2 y H3, `far_max_val`, `piso_dispersion` y `tol` tienen que ser `float` de Python (`0.01`, `0.5` y `1.0e-4`; un entero no sirve) y `k`, `n_init` y `max_iter`, `int` que no sea `bool`.
- El oráculo de la calibración, de la evaluación y de las dispersiones se escribe en el test a partir de las secciones 0 y 4, sin importar nada de `src/huellas/clasificacion.py`.

### Sumadas en la versión 1.1 (lecturas del chat de tests)

El chat que escribe `tests/test_h4.py` hizo tres preguntas, que Santiago respondió el 23/10, y dejó sus lecturas del contrato en el docstring del test:

- **Tipos de los escalares** (respuesta de Santiago): "float" es `isinstance(x, float)`, así que `np.float64` vale y un array de 0 dimensiones no. "int" o "un entero" es un `int` de Python que no sea `bool` (`np.int64` no vale): `predecir`, `ModeloHuellas.decidir` y `ModeloHuellas.puntajes` de un vector, `identidad` y `mas_cercana` de `Reconocimiento`, `n_iter` del K-means y las claves y los valores del dict de `correspondencia`. Los arrays en `int64` o `float64` se controlan por su dtype.
- **`far_max_val` fuera de (0, 1)** (respuesta de Santiago): `calibrar_umbral` da `ValueError` con los dos criterios, también con `eer`, donde no se usa (la Fase 0 también lo valida siempre).
- **Implementación descartable** (respuesta de Santiago): el chat de tests corrió el archivo en la nube contra una implementación que no se entrega (las definiciones del prototipo más lo que faltaba), con las mismas versiones de NumPy, SciPy, OpenCV y scikit-learn que el venv: pasan los 169 tests normales en 34 s y los 5 de aceptación en 2 min 56 s, y de 36 errores plantados detectan 35 (el que no, la falta de `caracteristicas.npz`, ya da `FileNotFoundError` en `np.load`).
- **6.2:** grupos gaussianos con semilla: "separados", centros a 30 o más entre sí; "juntos", centros N(0, 2²). Los datos evitan que scikit-learn tenga que reubicar clusters vacíos. El oráculo con `init` dado se corre con `max_iter` 300, 1 y 2 (`tol = 1e-4`) y también con `tol = 1e-2`, donde la parada por `mov ≤ tol · media(var(X))` corta antes y cambia el resultado (muestra que la tolerancia se escala por la varianza). El k-means++ propio se prueba con las semillas 0 y 2026 contra `random_state=0`. Se controlan además los dtypes de la sección 2 y que `distancias` sea la euclídea.
- **6.5:** el modelo armado a mano tiene un punto que la distancia normalizada lleva a un centroide que no es el más cercano en distancia euclídea; `far_frr` se compara con el oráculo en cada puntaje observado, en los puntos medios y fuera del rango.
- **6.7:** además del `cfg` real (con val separable, donde el piso no actúa y los dos criterios dan el mismo τ\*), copias del `cfg` en memoria con `piso_dispersion` 1.0, `criterio` `"eer"` y `far_max_val` 0.30, con un val con solape: muestran que esos valores salen de `cfg`.
- **6.8:** "`cargar_modelo` da los mismos valores": igual al archivo (exacto) y a `ajustar_modelo` de las filas de train y val del npz con `ajustar_escalador` de los bloques de train (1e-12); además, `guardar_modelo` → `cargar_modelo` devuelve exactamente lo guardado. "Sin fuga": el npz se reescribe con otras filas de test (en `X` y en los bloques) y los dos modelos guardados se comparan clave por clave con `np.array_equal` (NaN en el mismo lugar cuentan como iguales). "Se vuelve a escribir": el modelo se reemplaza por bytes que no son un npz y tiene que volver a quedar el mismo. El modelo de los tests de `reconocer` y de `evaluar_particion` es el que devuelve `cargar_modelo` del archivo que escribió `entrenar`.
- **Errores:** como en H3, solo los que nombra la sección 6. No se prueban los `ValueError` de `puntajes` ni de `guardar_modelo`, ni el `FileNotFoundError` de `cargar_modelo` (el código los lanza igual).
- **Dataset chico:** con 12 filas de train el escalador deja `d = 372`; τ\* = 2,327 y la autorizada de test rechazada es `id3_v033` (2,63).
