# Contrato · V4 · K-NN propio, selección de hiperparámetros por validación cruzada por persona y umbral de rechazo

**Versión 1.1 · 24/10/2026** (la 1 es del mismo día). **Cambio en la 1.1:** se suman a las secciones 3, 4, 5, 6 y 8 las precisiones del chat que escribe `tests/test_v4.py`, con las respuestas de Santiago: el desempate en K-NN usa el mínimo de las distancias a los vecinos de cada clase; `ModeloVoz.transformar`, `votar` y `decidir` reciben `cfg`; `archivo` en el dataset chico es relativo `{particion}/{nombre}_k{copia}.wav`; corrección del ejemplo 5 y suma del ejemplo 6 de calibración; y corrección de la cuenta de comandos rechazados en test (16, no 20). Ninguna definición ni umbral cambia de valor. Se anota en el Documento 2 junto con los tests.

Es lo que verifica `tests/test_v4.py`. Los tests se escriben en otro chat o ventana aislada (con un modelo distinto de Gemini) solo a partir de este archivo y se congelan (commit) antes de implementar. Se apoya en los contratos de la Fase 0 (`config.yaml`, `src/config.cargar_config` y `RAIZ`), de H3 (`src/core_ml/pca.py` con `PCA`), de H4 (`matriz_confusion` en `src/core_ml/metricas.py`, firma en la sección 2 de `h4_clasificacion.md`), de V2 (`Preprocesado`, `preprocesar` y `metadatos.csv`) y de V3 (`BLOQUES`, `largos_de_bloque`, `caracteristicas`, `Escalador`, `ajustar_escalador`, `generar_caracteristicas` y `caracteristicas.npz`). Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

Esta etapa suma `src/core_ml/knn.py` (el K-NN propio) y crea `src/voz/clasificacion.py`. Con V4 el carril de voz queda listo para la integración (Fase 5): `reconocer` lleva un audio crudo hasta el comando o `"otro"` ("no entendí").

**Qué hace V4:** parte de `X` de V3 (281 columnas estandarizadas con `train` y con peso `1/√d_b` por bloque), la **repondera por bloque** con uno de tres esquemas, la **proyecta** con el `PCA` propio (20 o 30 componentes, o sin PCA), **normaliza cada fila a norma L2 unitaria** y clasifica con un K-NN propio de voto `1/(d + ε)` en las cuatro clases (`reconocer`, `identificar`, `salir` y `otro`). El esquema de bloques, la cantidad de componentes y `K` se eligen con **validación cruzada dejando una persona afuera en `train`** (sin mirar `val` ni `test`). El **umbral de distancia τ\*** se calibra en `val`. Un comando se acepta solo si el K-NN no dice `"otro"`, la confianza del voto llega a `voz.knn.confianza_min` y la distancia no pasa de τ\*; en cualquier otro caso la decisión es `"otro"`. Un audio no válido de V2 es `"otro"` directamente, sin pasar por el K-NN.

**Reglas fijadas por Santiago al cerrar V3 y cumplidas acá:** (1) el factor por bloque sobre `X` es un hiperparámetro elegido por validación cruzada por persona en `train` (sin tocar V3); (2) `test` no se usa para elegir nada (ni esquema de bloques, ni PCA, ni `K`, ni τ\*); (3) los tests los escribe otro modelo.

**Lo que mide y lo que no cumple (se dice acá para que no quede escondido en la sección 8):** con este contrato, el prototipo da en `val` un acierto de comandos de 85,07 % con `"otro"` aceptado 2,94 %, y en `test` 77,46 % con `"otro"` aceptado 14,71 %. Las metas del plan (`metas.voz_acierto_min` 0,90 y `metas.voz_otro_aceptado_max` 0,05) **no se cumplen en `test`** (y la de acierto tampoco en `val`). La sección 6.9 no las exige; las reporta. Qué hacer con eso es una decisión de Santiago (ver 8).

## 0. Convenciones

- **Clases:** `ETIQUETAS = voz.comandos + [voz.clase_rechazo]` = `["reconocer", "identificar", "salir", "otro"]`, textos (`str`), en ese orden para la matriz de confusión. `comandos` son los tres primeros; `"otro"` es el rechazo.
- **Fuentes:** `caracteristicas.npz` de V3 (solo filas válidas: claves `archivo`, `particion`, `origen`, `hablante`, `clase`, `copia`, `mfcc`, `delta`, `energia`, `zcr`, `duracion` y `X`) y `metadatos.csv` de V2 (todas las filas, con la columna `valido`). Las filas con `valido == 0` no están en el `.npz`: son siempre `"otro"` por decisión (no pasan por el K-NN) y se suman a los conjuntos de evaluación como filas con `prediccion = decision = "otro"` y `verdad = clase` de su fila.
- **Persona:** el código de `hablante` pasado por `voz.dataset.mismo_hablante` (p01 y p02 son la misma persona; p09 y p13; p14 y p17): cada grupo se reemplaza por su primer código. Una persona entera sale junta de un pliegue, con todas sus copias aumentadas y sus filas de WhatsApp.
- **Fila evaluable en la validación cruzada:** `origen == "notebook"` y `copia == 0` (el micrófono de la demo, sin aumento). Las personas de la validación cruzada son las que tienen al menos una fila evaluable de comando.
- **Voto, confianza y distancia del K-NN:** definidos en 4.1. `confianza` es la fracción del voto total que se lleva la clase ganadora; `distancia` es la media de las distancias a los vecinos de la clase ganadora.
- **Se acepta** una fila como comando si `prediccion != "otro"` **y** `confianza ≥ voz.knn.confianza_min` **y** `distancia ≤ τ`. Con la distancia igual a τ, se acepta.
- **Métricas sobre un conjunto** (con las filas no válidas ya sumadas como `"otro"`; `verdad`, `prediccion` y `decision` son arrays de textos de igual largo):
  - `identificacion` = fracción de las filas de comando (`verdad ∈ comandos`) con `prediccion == verdad` (sin rechazo).
  - `acierto` = fracción de las filas de comando con `decision == verdad`. `frr = 1 − acierto`.
  - `otro_aceptado` = fracción de las filas con `verdad == "otro"` cuya `decision != "otro"`.
  - `matriz` = `matriz_confusion(verdad, decision, ETIQUETAS)` (filas: lo real; columnas: la decisión), de 4 × 4.
- **Sin aleatoriedad:** nada de V4 usa semillas. El K-NN, el PCA (`eigh`) y la validación cruzada son deterministas; dos corridas con los mismos datos dan lo mismo, bit a bit.
- Tipos: `float` de Python para los escalares que devuelven las funciones (`tau`, métricas, `confianza`, `distancia`), `int` de Python que no sea `bool` para `k` y los conteos, y `str` de Python para una etiqueta suelta; los arrays de textos son de `str` de NumPy.

## 1. config.yaml

**Sin claves nuevas.** V4 usa las que ya existen: `voz.comandos`, `voz.clase_rechazo`, `voz.dataset.mismo_hablante`, `voz.knn.k_candidatos` (`[3, 5, 7, 9]`), `voz.knn.eps` (`1.0e-9`), `voz.knn.confianza_min` (`0.60`), `voz.knn.pca_componentes_candidatas` (`[20, 30]`), `metas.voz_acierto_min` (`0.90`) y `metas.voz_otro_aceptado_max` (`0.05`), más `rutas.voz`, `rutas.modelos` y `rutas.figuras`.

- La opción "**sin PCA**" está siempre en la grilla (no es una clave): `n_componentes = None`.
- Los **esquemas de bloques** (sección 3) son parte del contrato, como `BLOQUES` en V3: no son claves de `config.yaml`.
- `pca_componentes_candidatas` sigue siendo una lista de enteros mayores que 1 (lo exige `tests/test_setup.py`).

## 2. src/core_ml/knn.py

```python
class KNN:
    def __init__(self, k: int = 5, eps: float = 1e-9) -> None: ...
    def ajustar(self, X: np.ndarray, etiquetas) -> "KNN": ...            # devuelve self
    def vecinos(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
        # (distancias (n, k) float64, indices (n, k) int64), de menor a mayor distancia
    def votar(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
        # (prediccion (n,), confianza (n,) float64, distancia (n,) float64)
    def predecir(self, X: np.ndarray) -> np.ndarray: ...                  # prediccion de votar
    # Atributos después de ajustar:
    #   X_entrenamiento: (n, d) float64      copia de X
    #   etiquetas: (n,)                      copia de etiquetas
    #   clases: (c,)                         np.unique(etiquetas)
```

1. `k` es `int` ≥ 1 (no `bool`) y `eps`, un número real finito en `(0, 1e-3]`; si no, `ValueError` en el constructor.
2. `ajustar(X, etiquetas)`: `X` de 2 dimensiones `(n, d)` con `n ≥ k`, `d ≥ 1` y valores finitos; `etiquetas` de 1 dimensión con `n` elementos. Si no: `ValueError`. Se trabaja en `float64` y no se modifica nada de lo que se pasa.
3. `vecinos`, `votar` y `predecir` aceptan `(n, d)` y devuelven arrays con `n` filas, o un vector `(d,)` y devuelven `(k,)` y `(k,)` para `vecinos`, y un trío `(etiqueta, float, float)` (escalares de Python) para `votar` y la etiqueta para `predecir`. Antes de `ajustar`: `RuntimeError`. Otro `d`, otra cantidad de dimensiones o valores no finitos: `ValueError`.
4. Solo NumPy: `src/core_ml/knn.py` no importa scikit-learn (el texto del archivo no contiene `sklearn`).

## 3. src/voz/clasificacion.py

```python
ESQUEMAS_BLOQUES: tuple[str, ...] = ("contrato", "sin_dur_zcr", "z_columna")
PCA_SIN: None = None     # "sin PCA" en la grilla

def etiquetas_clasificacion(cfg: dict) -> list[str]: ...
def factores_esquema(esquema: str, cfg: dict) -> np.ndarray: ...                 # (5,) float64, orden de BLOQUES
def reponderar_bloques(X: np.ndarray, factores, cfg: dict) -> np.ndarray: ...
def normalizar_l2(X: np.ndarray, piso: float = 1e-12) -> np.ndarray: ...

@dataclass(frozen=True)
class ModeloVoz:
    esquema: str                  # uno de ESQUEMAS_BLOQUES
    factores: np.ndarray          # (5,) float64
    n_componentes: int | None     # 20, 30 o None (sin PCA)
    pca_media: np.ndarray         # (d_r,) float64; forma (0,) si no hay PCA
    pca_componentes: np.ndarray   # (k, d_r) float64; forma (0, 0) si no hay PCA
    k: int
    eps: float
    confianza_min: float
    tau: float                    # τ*
    entrenamiento: np.ndarray     # (n, d') float64: filas de train ya transformadas
    etiquetas: np.ndarray         # (n,) de str
    escalador: Escalador | None   # el de V3, ajustado con train (None solo en modelos armados sin él)
    def transformar(self, X: np.ndarray, cfg: dict) -> np.ndarray: ...
    def votar(self, X: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def decidir(self, X: np.ndarray, cfg: dict) -> np.ndarray: ...

@dataclass(frozen=True)
class Reconocimiento:
    comando: str            # la decisión: un comando o "otro"
    prediccion: str | None  # lo que votó el K-NN (None si el audio no es válido)
    confianza: float        # 0.0 si el audio no es válido
    distancia: float        # float("inf") si el audio no es válido
    tau: float              # τ* del modelo
    valido: bool            # lo que dice V2
    motivo: str             # "" si es válido; si no, el motivo de V2 ("sin_voz" o "duracion")

def elegir_mejor(tabla: list[dict]) -> dict: ...   # dict con exactamente esquema, n_componentes y k
def seleccionar_hiperparametros(X, etiquetas, personas, evaluable, n_otro_no_validos: int,
                                cfg: dict) -> tuple[dict, list[dict]]: ...
def calibrar_umbral(distancia, confianza, prediccion, verdad, cfg: dict,
                    n_otro_no_validos: int = 0) -> float: ...
def decidir(prediccion, confianza, distancia, tau: float, cfg: dict) -> np.ndarray: ...
def evaluar(decision, prediccion, verdad, cfg: dict) -> dict: ...
def ajustar_modelo(X_train, etiquetas_train, personas_train, evaluable_train,
                   n_otro_no_validos_train: int, X_val, etiquetas_val, n_otro_no_validos_val: int,
                   cfg: dict, escalador: Escalador | None = None) -> ModeloVoz: ...
def guardar_modelo(modelo: ModeloVoz, ruta: str | os.PathLike) -> Path: ...
def cargar_modelo(ruta: str | os.PathLike) -> ModeloVoz: ...
def entrenar(cfg: dict, carpeta: str | os.PathLike | None = None,
             salida: str | os.PathLike | None = None, sobrescribir: bool = False) -> Path: ...
def evaluar_particion(modelo: ModeloVoz, cfg: dict, carpeta: str | os.PathLike | None = None,
                      particion: str = "test") -> dict: ...
def reconocer(audio: np.ndarray, fs: int, modelo: ModeloVoz, cfg: dict) -> Reconocimiento: ...
def generar_figura(cfg: dict, carpeta: str | os.PathLike | None = None,
                   modelo: str | os.PathLike | None = None,
                   salida: str | os.PathLike | None = None) -> Path: ...
```

Por defecto, `carpeta` es `RAIZ / rutas.voz / "prepro"` (donde están `caracteristicas.npz` y `metadatos.csv`), el modelo es `RAIZ / rutas.modelos / "modelo_voz.npz"` y la figura, `RAIZ / rutas.figuras / "rechazo_v4.png"`.

**Línea de comandos:** `venv\Scripts\python -m src.voz.clasificacion [--sobrescribir]` corre `entrenar` y `generar_figura` con `config.yaml` e imprime: los hiperparámetros elegidos (esquema, `n_componentes`, `K`), las 5 mejores filas de la tabla de la validación cruzada (con `n_tau5`, `n_c60`, `n_4cl` y `tau_cv`), τ\*, la `identificacion`, el `acierto`, la `frr` y el `otro_aceptado` de `val` y de `test` (cada uno con la decisión sin umbral de distancia, solo con la confianza, y con τ\*), y las dos matrices de confusión de 4 × 4.

## 4. Definiciones exactas

Con `comandos`, `clase_rechazo` y `ETIQUETAS` de la sección 0, `cmin = float(voz.knn.confianza_min)`, `eps = float(voz.knn.eps)`, `otro_max = float(metas.voz_otro_aceptado_max)` y `largos = largos_de_bloque(cfg)`.

### 4.1 K-NN (`KNN`)

- **Distancia:** euclídea `d_ij = ‖x_i − t_j‖₂` entre cada fila de consulta `x_i` y cada fila de entrenamiento `t_j` (se compara con `scipy.spatial.distance.cdist` a 1e-9).
- **Vecinos:** los `k` de menor distancia, ordenados de menor a mayor. Un empate de distancias se resuelve con el índice de entrenamiento menor.
- **Voto:** cada vecino `j` aporta `w_j = 1 / (d_j + eps)` a su clase. `voto(c) = Σ w_j` de los vecinos de la clase `c`.
- **Ganador:** la clase de mayor voto. Si dos clases tienen votos que difieren en 1e-12 o menos (en valor absoluto), gana la que tenga el **mínimo** de las distancias a sus vecinos entre los `k` vecinos considerados (la menor distancia al vecino más cercano de esa clase); y si todavía empatan, la que aparezca primero en `clases` (orden alfabético de `np.unique`).
- `prediccion` es el ganador; `confianza = voto(ganador) / Σ_j w_j`; `distancia` es la **media simple** (no ponderada) de las distancias a los vecinos de la clase ganadora.
- Con `k = 1`: `prediccion` es la etiqueta del vecino más cercano, `confianza = 1.0` y `distancia = d_1`.
- **Equivalencia con scikit-learn:** en datos sin empates, `prediccion` es la de `sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, weights="distance")` y `vecinos` da las mismas distancias que su `kneighbors` (1e-9). (scikit-learn pesa `1/d` sin `eps`: coincide mientras ninguna distancia sea cercana a 0.)

### 4.2 Esquemas de bloques, reponderación y normalización

- **`factores_esquema(esquema, cfg)`**: devuelve un array `(5,)` en `float64`, en el orden de `BLOQUES = ("mfcc", "delta", "energia", "zcr", "duracion")`:
  - `"contrato"`: `[1, 1, 1, 1, 1]` (el peso `1/√d_b` de V3, tal cual).
  - `"sin_dur_zcr"`: `[1, 1, 1, 0, 0]` (se descartan los bloques `zcr` y `duracion`: quedan 270 columnas).
  - `"z_columna"`: `[√d_mfcc, √d_delta, √d_energia, √d_zcr, √d_duracion]` con `d_b = largos[b]` (`√130, √130, √10, √10, 1`): como cada bloque de `X` ya está dividido por `√d_b`, esto lo deja como la estandarización `z` por columna, sin peso por bloque (todas las columnas pesan igual).
  - `ValueError` si `esquema` no está en `ESQUEMAS_BLOQUES`.
- **`reponderar_bloques(X, factores, cfg)`**: `X` es `(n, d)` o `(d,)` con `d = Σ largos = 281` (si no: `ValueError`), valores finitos, y `factores` un array `(5,)` de reales ≥ 0. Multiplica el bloque `b` (las columnas contiguas según `largos`, en el orden de `BLOQUES`) por `factores[b]` y **descarta** las columnas de los bloques con factor `0`. Devuelve `float64` de `(n, d_r)` o `(d_r,)` y no modifica `X`.
- **`normalizar_l2(X, piso)`**: divide cada fila (o el vector) por `max(‖fila‖₂, piso)`. Una fila de ceros queda en ceros (sin `NaN`). `ValueError` con `NaN` o `Inf`, con más de 2 dimensiones o con `piso ≤ 0`.
- **Proyección:** con `n_componentes = k` (`int`), el ajuste es `PCA(k).ajustar(reponderar_bloques(X_train, factores, cfg))`, y la transformación de una fila reponderada `r` es `(r − media) @ componentes.T`, que es lo mismo que `PCA.transformar`. Sin PCA (`None`), no se proyecta.
- **`ModeloVoz.transformar(X, cfg)`**: `normalizar_l2(proyección(reponderar_bloques(X, factores, cfg)))`, con `X` en el espacio de V3 (281 columnas); acepta `(n, 281)` o `(281,)` y devuelve `(n, d')` o `(d',)`. La normalización va **después** de la proyección.
- **`ModeloVoz.votar(X, cfg)`**: es `KNN(k, eps).ajustar(entrenamiento, etiquetas).votar(transformar(X, cfg))` (devuelve predicción, confianza y distancia).
- **`ModeloVoz.decidir(X, cfg)`**: es `decidir(*votar(X, cfg), tau, cfg)` (array de `str` para matriz `(n, 281)` o un `str` para un vector).
- `confianza_min` del modelo es el valor de `cfg` al momento de ajustar y queda guardado como registro.

### 4.3 Selección de hiperparámetros (`seleccionar_hiperparametros`)

Con `X` `(n, 281)` y `etiquetas`, `personas` y `evaluable` de `n` elementos (las filas de `train`; `evaluable` es `bool`), y `n_otro_no_validos` (cuántas filas no válidas de `"otro"` evaluables hay: se suman al denominador de `"otro"`):

1. **Grilla:** `esquema` en `ESQUEMAS_BLOQUES` × `n_componentes` en `voz.knn.pca_componentes_candidatas + [None]` (en ese orden: `[20, 30, None]`) × `k` en `voz.knn.k_candidatos`: 3 × 3 × 4 = 36 combinaciones.
2. **Pliegues:** una persona de `personas[evaluable & (etiquetas ∈ comandos)]` por pliegue, en orden alfabético. En el pliegue de la persona `p`, se entrena con **todas las filas con `personas != p`** (de cualquier origen y copia) y se consultan las filas con `evaluable & personas == p` (comandos y `"otro"`). Para cada esquema y cada `n_componentes`, dentro del pliegue: `reponderar_bloques`, `PCA` ajustado **solo con las filas de entrenamiento del pliegue** (si hay `n_componentes`), proyección de entrenamiento y consulta, `normalizar_l2` y `KNN(k, eps)`. El escalador de V3 está ajustado con todo `train` (incluida la persona que sale del pliegue): es una fuga chica que se acepta y queda anotada.
3. **Descarte:** una combinación se descarta si en algún pliegue `n_componentes > mín(n_entrenamiento − 1, d_r)` o `k > n_entrenamiento`, o si `calibrar_umbral` lanza `ValueError` sobre las predicciones juntas. Una descartada queda en la tabla con `descartada = True`, `tau_cv = float("nan")` y los conteos en `0`.
4. **Conteos por combinación** (sobre las predicciones de todos los pliegues juntas, con `verdad` = la etiqueta de la fila consultada):
   - `tau_cv = calibrar_umbral(distancia, confianza, prediccion, verdad, cfg, n_otro_no_validos)` (4.4): es el τ que deja el `"otro"` aceptado en `otro_max` **dentro de la propia validación cruzada**.
   - `n_4cl` = comandos con `prediccion == verdad` (sin rechazo).
   - `n_c60` = comandos con `prediccion == verdad` y `confianza ≥ cmin`.
   - `n_tau5` = comandos con `prediccion == verdad`, `confianza ≥ cmin` y `distancia ≤ tau_cv`.
   - Se cumple siempre `n_tau5 ≤ n_c60 ≤ n_4cl`.
5. **Criterio de elección (`elegir_mejor`)**: entre las no descartadas, se ordena por: (1) **mayor `n_tau5 + n_c60`**; (2) mayor `n_4cl`; (3) el esquema que aparece antes en `ESQUEMAS_BLOQUES`; (4) `n_componentes` más a la izquierda en `[20, 30, None]`; (5) menor `k`. Gana la primera. Devuelve un `dict` con exactamente las claves `esquema`, `n_componentes` y `k`. `ValueError` si la tabla está vacía o todas están descartadas. Con enteros, no hay empates por redondeo.
   - Por qué la suma: `n_tau5` mide el rechazo (cuántos comandos sobreviven cuando el umbral deja entrar a lo sumo el 5 % de `"otro"`) y `n_c60` mide la clasificación sin umbral de distancia; `n_tau5` solo depende de una distancia puntual (la del `(m + 1)`-ésimo `"otro"` más cercano) y es más ruidoso. La suma exige las dos cosas a la vez. Sin PCA, en la esfera L2 de 281 dimensiones las distancias se concentran y `n_tau5` cae solo (65 a 76 %), así que la propia validación descarta esa opción sin mirar `val`.
6. **Devuelve** `(mejor, tabla)`: `mejor` es un `dict` con exactamente las claves `esquema`, `n_componentes` y `k`; `tabla` es una lista de `dict` (uno por combinación, en el orden de la grilla: esquema, después `n_componentes`, después `k`) con exactamente las claves `esquema`, `n_componentes`, `k`, `n_tau5`, `n_c60`, `n_4cl`, `tau_cv` y `descartada`. No modifica los arrays que recibe, no usa `val` ni `test` y no importa scikit-learn.
7. `ValueError` si los largos de `X`, `etiquetas`, `personas` y `evaluable` no coinciden, si `X` no es `(n, 281)` finito, si `n_otro_no_validos` es negativo o no es `int`, si `etiquetas` tiene valores fuera de `ETIQUETAS` o si no hay ninguna persona con filas evaluables de comando.
8. **`elegir_mejor(tabla)`** es la función pura del punto 5 (recibe la lista de filas ya armada): devuelve un `dict` con exactamente las claves `esquema`, `n_componentes` y `k`, y se prueba sola con tablas hechas a mano.

### 4.4 Calibración del umbral (`calibrar_umbral`)

Con las predicciones de un conjunto (val, o los pliegues de la validación cruzada): `distancia`, `confianza`, `prediccion` y `verdad` de igual largo (solo filas válidas; las no válidas entran por `n_otro_no_validos`). `es_cmd = verdad ∈ comandos`, `es_otro = verdad == "otro"`, `acep = (prediccion != "otro") & (confianza ≥ cmin)`.

1. `n_otro = (es_otro).sum() + n_otro_no_validos` y `m = ⌊otro_max · n_otro + 1e-9⌋` (cuántos `"otro"` se pueden aceptar; el `1e-9` evita que el redondeo baje un entero exacto).
2. `B` = las distancias de las filas `es_otro & acep`, ordenadas de menor a mayor. `C` = las distancias de las filas `es_cmd & acep & (prediccion == verdad)`, ordenadas.
3. **Si `len(B) > m`:** `b = B[m]` (el `(m + 1)`-ésimo `"otro"` aceptable más cercano); `a` = el mayor valor de `C` que sea **menor que** `b`. **τ\* = (a + b) / 2.**
4. **Si `len(B) ≤ m`** (ya entran a lo sumo `m` sin umbral): `a = máx(C)`; `b` = la menor distancia entre **todas** las filas `es_otro` (aceptables o no) que sea mayor que `a`, o `1.2 · a` si no hay ninguna. **τ\* = (a + b) / 2.**
5. `cmin` y `otro_max` salen de `cfg`. `ValueError` si los largos no coinciden, si hay valores no finitos, si `verdad` o `prediccion` tienen valores fuera de `ETIQUETAS`, si `otro_max` no está en `(0, 1)`, si `n_otro_no_validos` es negativo o no es `int`, si no hay filas de comando, o si `C` queda vacío (en el punto 3, vacío después de quedarse con las menores que `b`).
6. El resultado no depende del orden de las filas.

### 4.5 Decisión y evaluación

- **`decidir(prediccion, confianza, distancia, tau, cfg)`:** `prediccion` donde `(prediccion != "otro") & (confianza ≥ cmin) & (distancia ≤ tau)` y `"otro"` donde no; array de textos de `str`. Con un solo elemento (escalares), un `str`. `ValueError` si los largos no coinciden.
- **`evaluar(decision, prediccion, verdad, cfg)`:** con las filas ya sumadas (válidas y no válidas, estas con `prediccion = decision = "otro"`), devuelve un `dict` con **exactamente** las claves `identificacion`, `acierto`, `frr`, `otro_aceptado` y `matriz` (sección 0; `matriz` de 4 × 4 en `int64`, con el orden de `ETIQUETAS`). `ValueError` si los largos no coinciden, si hay valores fuera de `ETIQUETAS`, si no hay filas de comando o si no hay filas de `"otro"`.

### 4.6 Modelo, archivo, evaluación, reconocimiento y figura

- **`ajustar_modelo(...)`:** `(esquema, n_componentes, k) = seleccionar_hiperparametros(X_train, etiquetas_train, personas_train, evaluable_train, n_otro_no_validos_train, cfg)[0]`; con **todas** las filas de `X_train`: `factores = factores_esquema(...)`, `reponderar_bloques`, `PCA` (si `n_componentes`), `normalizar_l2` → `entrenamiento`; `ModeloVoz` con `eps` y `confianza_min` de `cfg`; con `X_val`: `distancia, confianza, prediccion` de `votar(X_val, cfg)` y `tau = calibrar_umbral(distancia, confianza, prediccion, etiquetas_val, cfg, n_otro_no_validos_val)`. **No recibe nada de `test`.** `ValueError` si `X_train` o `X_val` no son `(n, 281)`, si los largos no coinciden, si hay etiquetas fuera de `ETIQUETAS`, y los de las funciones que usa.
- **`guardar_modelo(modelo, ruta)`:** escribe un `.npz` (`np.savez`, sin objetos de Python) con **exactamente** estas 27 claves: `esquema`, `factores`, `n_componentes` (`int64`; `0` si no hay PCA), `k`, `eps`, `confianza_min`, `tau`, `entrenamiento`, `etiquetas`, `pca_media`, `pca_componentes` y, del escalador, `pesos` (`(5,)`, en el orden de `BLOQUES`) y, para cada bloque `b`, `medias_<b>`, `desvios_<b>` y `columnas_<b>` (15 claves). Crea la carpeta si no existe y devuelve la ruta. `ValueError` si el modelo no tiene escalador.
- **`cargar_modelo(ruta)`:** lee ese archivo con `allow_pickle=False` y devuelve un `ModeloVoz` igual al guardado (arrays iguales y `Escalador` con los mismos dicts; `n_componentes` vuelve a ser `None` si estaba en `0`). `FileNotFoundError` si no existe.
- **`entrenar(cfg, carpeta, salida, sobrescribir)`:** lee `carpeta / "caracteristicas.npz"` (de V3) y `carpeta / "metadatos.csv"` (de V2). Con `tr` las filas del `.npz` de `particion == "train"` y `va` las de `"val"`: `personas` (con `mismo_hablante`), `evaluable = (origen == "notebook") & (copia == 0)`, `n_otro_no_validos_train` = las filas de `metadatos.csv` de `particion == "train"`, `valido == 0`, `origen == "notebook"`, `copia == 0` y `clase == "otro"`, y `n_otro_no_validos_val` = las de `particion == "val"`, `valido == 0` y `clase == "otro"`; `escalador = ajustar_escalador({b: bloque[tr] for b in BLOQUES})` (el mismo con el que V3 escaló `X`); `modelo = ajustar_modelo(...)` y lo guarda en `salida` (por defecto la ruta del modelo). **No usa las filas de `test`.** Si `salida` ya existe y `sobrescribir` es `False`: `FileExistsError` sin tocar nada. `FileNotFoundError` si falta `caracteristicas.npz` o `metadatos.csv`; `ValueError` si no hay filas de train o de val.
- **`evaluar_particion(modelo, cfg, carpeta, particion)`:** con las filas de esa partición de `caracteristicas.npz` y las filas de `metadatos.csv` de esa partición con `valido == 0` (cada una con `verdad` = su `clase` y `prediccion = decision = "otro"`): `decision = modelo.decidir(X, cfg)`, `prediccion = modelo.votar(X, cfg)[0]` y `evaluar(...)`. `ValueError` si `particion` no es `train`, `val` ni `test`.
- **`reconocer(audio, fs, modelo, cfg)`:** `p = preprocesar(audio, fs, cfg)` (V2). Si `p.valido` es `False`: `Reconocimiento(comando="otro", prediccion=None, confianza=0.0, distancia=float("inf"), tau=modelo.tau, valido=False, motivo=p.motivo)`, sin pasar por el K-NN. Si no: `z = modelo.escalador.transformar(caracteristicas(p.y, cfg))`, `prediccion, confianza, distancia = modelo.votar(z, cfg)` y `comando = decidir(prediccion, confianza, distancia, modelo.tau, cfg)`, con `valido=True` y `motivo=""`. `ValueError` si el modelo no tiene escalador.
- **`generar_figura(cfg, carpeta, modelo, salida)`:** con `caracteristicas.npz`, `metadatos.csv` y el modelo guardado, dibuja con Matplotlib (backend `Agg`) dos paneles: a la izquierda, el acierto de comandos y el `"otro"` aceptado de `val` en función de τ (con las filas no válidas sumadas), con τ\* marcado; a la derecha, la matriz de confusión de `test` de 4 × 4 con los números en cada celda. Escribe un PNG en `salida` y devuelve esa ruta.

## 5. Datos de prueba

- **Sintéticos** (6.2 a 6.7): grupos gaussianos con semilla, armados por el test (para `seleccionar_hiperparametros` y `ajustar_modelo`, matrices de `(n, 281)`).
- **Clips sintéticos** (6.8): `clip_sintetico(f0_hz, mod_hz, dur_palabra_s, amp, semilla)` de la sección 5 de `v3_caracteristicas.md`, a `fs = 16000` y `M = 640`, con las tres clases de comandos de esa sección y una cuarta, `"otro"`: `f0_hz` en `800 ± 40` Hz, `mod_hz = 4.5`, `dur_palabra_s` en `0.95 ± 0.06` s (los valores dentro del rango se sortean con `np.random.default_rng(semilla)`).
- **Dataset chico** (6.8), escrito por el test en `tmp_path` como el de 6.5 de V3: WAV `float32` (clips sintéticos) y un `metadatos.csv` con al menos las columnas `archivo, particion, origen, hablante, clase, copia, valido` (más, si quiere, el resto de `COLUMNAS_V2` vacías), con la columna `archivo` relativa a la carpeta con el formato `{particion}/{nombre}_k{copia}.wav` (por ejemplo `train/q01_reconocer_01_k0.wav`; para las filas no válidas, `archivo` vacío): `train`: 3 personas (`q01` a `q03`, `origen = "notebook"`, `copia = 0`) con 4 clips de cada una de las 4 clases (48 filas válidas); `val`: 2 personas (`q04` y `q05`) con 3 clips por clase (24 válidas) y 2 filas no válidas de `"otro"` (`valido = 0`, `archivo` vacío, sin WAV); `test`: 2 personas (`q06` y `q07`) con 2 clips por clase (16 válidas) y 1 fila no válida de `"otro"`. Después, `generar_caracteristicas(cfg, carpeta)` de V3 escribe ahí `caracteristicas.npz`.
- **Dataset real** (6.9): `data/voz/prepro/` (`caracteristicas.npz` y `metadatos.csv`).

## 6. Criterios que verifican los tests

### 6.1 Configuración

- Las claves de la sección 1 existen en `config.yaml` con sus valores: `voz.knn.k_candidatos == [3, 5, 7, 9]`, `voz.knn.eps == 1.0e-9` (`float`), `voz.knn.confianza_min == 0.60` (`float`), `voz.knn.pca_componentes_candidatas == [20, 30]`, `metas.voz_acierto_min == 0.90` y `metas.voz_otro_aceptado_max == 0.05` (`float`).
- `etiquetas_clasificacion(cfg) == ["reconocer", "identificar", "salir", "otro"]`.

### 6.2 K-NN propio (datos sintéticos)

- **Oráculo de la sección 4.1 escrito en el test:** con varias formas de datos (por ejemplo `(120, 5)` con 3 clases, `(300, 20)` con 4 clases y `(60, 2)`), `k` en `{1, 3, 5, 9}`: `vecinos` (distancias e índices), `prediccion`, `confianza` y `distancia` iguales al oráculo (1e-12; las predicciones, idénticas).
- **Contra scikit-learn:** en datos gaussianos sin empates, `predecir` es igual a `KNeighborsClassifier(n_neighbors=k, weights="distance")`, y las distancias de `vecinos`, a su `kneighbors` (1e-9).
- **Propiedades:** `confianza` en `(0, 1]`; `confianza == 1.0` si los `k` vecinos son de la misma clase; con `k = 1`, `prediccion` es la etiqueta del más cercano, `confianza == 1.0` y `distancia == d_1`; el resultado no depende del orden de las filas de entrenamiento cuando no hay empates; un vector da lo mismo que su fila en el lote; `ajustar` devuelve el mismo objeto y no modifica sus argumentos; dos llamadas dan resultados idénticos.
- **Empates:** con `X_entrenamiento = [[-1.0], [1.0]]`, etiquetas `["a", "b"]`, `k = 2` y la consulta `[0.0]`, los dos votos son iguales y gana `"a"` (el primero de `clases`); con `[[-2.0], [1.0]]`, gana `"b"`. Con `eps = 1e-15` y `k = 4`: clase `"a"` con vecinos a distancias 1.0 y 3.0 (voto = 1/1 + 1/3 = 4/3), clase `"b"` con dos vecinos a distancia 1.5 (voto = 1/1.5 + 1/1.5 = 4/3); al empatar en voto, el mínimo de las distancias de `"a"` es 1.0 y el de `"b"` es 1.5 → gana `"a"`.
- **Errores:** `ValueError` con `k` en 0, −1, 2.5 o `True`, con `eps` en 0, negativo, `NaN` o mayor que `1e-3`, con `X` de 1 o 3 dimensiones, con `NaN` o `Inf`, con `n < k`, con `etiquetas` de otro largo, y con `vecinos`, `votar` o `predecir` de otro `d`; `RuntimeError` antes de `ajustar`.
- `src/core_ml/knn.py` y `src/voz/clasificacion.py` no importan scikit-learn (el texto de los dos archivos no contiene `sklearn`).

### 6.3 Esquemas de bloques, reponderación y normalización L2

- `factores_esquema(esquema, cfg)`: los tres arrays de 4.2 (1e-12), también con una copia de `cfg` de otra `voz.tramos` o `voz.mfcc.n_mfcc` (a través de `largos_de_bloque`); `ValueError` con un esquema desconocido.
- `reponderar_bloques`: con `X` de `(n, 281)` aleatoria, el esquema `"contrato"` la devuelve igual, `"sin_dur_zcr"` da `(n, 270)` con las primeras 270 columnas intactas, y `"z_columna"` da `(n, 281)` con cada bloque multiplicado por su factor (1e-12); un vector da lo mismo que su fila; no modifica `X`; `ValueError` con otro `d`, con `NaN` o con factores negativos.
- **Propiedad de `z_columna`:** con bloques sintéticos de `train` y `ajustar_escalador` de V3, la varianza (`ddof = 1`) de cada columna de `reponderar_bloques(X_train, factores_esquema("z_columna", cfg), cfg)` es 1 (1e-9), y con `"contrato"` es `1 / d_b` por bloque.
- `normalizar_l2`: cada fila queda con norma 1 (1e-12); un vector, igual; una fila de ceros queda en ceros; `ValueError` con `NaN` o `Inf`, con 3 dimensiones y con `piso = 0`.
- `ModeloVoz.transformar(X, cfg)` de un modelo armado a mano (con PCA y sin PCA) es la cadena de 4.2 escrita en el test (1e-9; el PCA de oráculo es `src.core_ml.pca.PCA`); un vector da lo mismo que su fila.

### 6.4 Selección de hiperparámetros y `elegir_mejor`

- **`elegir_mejor` con tablas armadas a mano:** gana la mayor `n_tau5 + n_c60`; con la misma suma, la mayor `n_4cl`; con lo mismo, el esquema que viene antes en `ESQUEMAS_BLOQUES`, después `n_componentes` en el orden `[20, 30, None]`, y después el `k` menor; devuelve un `dict` con exactamente `esquema`, `n_componentes` y `k`; las descartadas no cuentan; `ValueError` con la tabla vacía o con todas descartadas. El caso de la sección 8 (suma 411 empatada, gana la de mayor `n_4cl`) está entre los de la prueba.
- **`seleccionar_hiperparametros` con datos sintéticos** (por ejemplo, 5 personas con 4 clases y 18 filas por persona y clase, de `(n, 281)`):
  - `tabla` tiene una fila por combinación de la grilla (en el orden de 4.3, con exactamente las claves de 4.3); `mejor` tiene exactamente `esquema`, `n_componentes` y `k`, y es igual a `elegir_mejor(tabla)`.
  - En toda fila no descartada, `n_tau5 ≤ n_c60 ≤ n_4cl`; en las descartadas, `descartada` es `True` y `tau_cv` es `NaN`.
  - **Oráculo de una combinación** (por ejemplo `"sin_dur_zcr"`, `None`, `k = 3`, y `"z_columna"`, 20, `k = 5`): el test arma los pliegues de 4.3 con `sklearn.decomposition.PCA`, `KNeighborsClassifier` (o su propio cálculo de voto) y la calibración de 4.4, y `n_4cl`, `n_c60` y `n_tau5` son iguales a los de la tabla (enteros exactos).
  - **Pliegues por persona:** si a una persona se le cambian todas las filas (de `X`) que no son evaluables, `n_4cl` de las demás personas no se ve afectado salvo por sus entrenamientos (se prueba con una persona de la que solo hay filas no evaluables: sale de los pliegues y su presencia agrega filas de entrenamiento a los demás, sin generar pliegue propio); el número de pliegues es el de personas con filas evaluables de comando.
  - **Descarte:** con `n_otro_no_validos = 0` y una `n_entrenamiento` chica (por ejemplo, 25 filas por pliegue), las combinaciones con `n_componentes = 30` quedan descartadas y la elegida no las usa; si todas quedan descartadas, `ValueError`.
  - Determinista (dos llamadas dan listas iguales) y no modifica los arrays que recibe.
  - `ValueError` con largos distintos, con `X` de otro ancho, con `n_otro_no_validos` negativo, con etiquetas fuera de `ETIQUETAS` y sin ninguna persona evaluable.

### 6.5 Calibración del umbral y decisión (casos armados a mano)

- **`calibrar_umbral`:** los τ\* de los ejemplos de la sección 8 (1e-12), con el `cfg` real (`confianza_min` 0.60, `metas.voz_otro_aceptado_max` 0.05) y con una copia de `cfg` de `metas.voz_otro_aceptado_max` 0.30 donde lo indica el ejemplo; el resultado no cambia al reordenar las filas.
- **`ValueError`:** con largos distintos, con `NaN`, con valores fuera de `ETIQUETAS`, con `metas.voz_otro_aceptado_max` 0 o 1, con `n_otro_no_validos` negativo, sin filas de comando, y con ningún comando bien clasificado con confianza suficiente por debajo de `b`.
- **`decidir`:** con τ = 0.85, `cmin` 0.60 y las filas del ejemplo A de la sección 8, las decisiones indicadas; una confianza justo en `cmin` y una distancia justo en τ se aceptan; una `prediccion` `"otro"` nunca se convierte en comando; `decidir` devuelve textos y no cambia la predicción de las aceptadas; con escalares, un `str`.

### 6.6 Evaluación (casos armados a mano)

- `evaluar` da `identificacion`, `acierto`, `frr`, `otro_aceptado` y `matriz` exactos en el ejemplo de la sección 8 (una comando rechazado, uno aceptado con otra identidad y un `"otro"` aceptado); la matriz es de 4 × 4 en `int64` con el orden de `ETIQUETAS`; `frr == 1 − acierto`; exactamente las 5 claves. `ValueError` sin filas de comando o sin filas de `"otro"`, con largos distintos y con valores fuera de `ETIQUETAS`.

### 6.7 Modelo (datos sintéticos)

- `ajustar_modelo` con 4 clases separadas en `train` (de `(n, 281)`, con personas y filas evaluables) y un `val` con comandos y `"otro"`: `(esquema, n_componentes, k)` iguales a los de `seleccionar_hiperparametros` con esos mismos datos; `entrenamiento` es `transformar(X_train)` (1e-9) y `etiquetas`, las de `etiquetas_train`; `pca_media` y `pca_componentes` son los de `PCA(n_componentes).ajustar(reponderar_bloques(X_train, factores, cfg))` (1e-9, o las formas vacías si no hay PCA); `eps` y `confianza_min` son los de `cfg`; `tau` es `calibrar_umbral` de `votar(X_val, cfg)` (1e-12).
- **De qué depende cada parte:** con el mismo `train`, cambiar `val` (por ejemplo, acercar un `"otro"` de `val` a un centro de clase) cambia `tau` y deja iguales `esquema`, `n_componentes`, `k`, `entrenamiento` y la proyección; cambiar `val` no toca los hiperparámetros. La separación de `test` está en 6.8 y 6.9.
- `ValueError` si `X_train` o `X_val` no son `(n, 281)`, si los largos no coinciden y si hay etiquetas fuera de `ETIQUETAS`.

### 6.8 Archivo del modelo, evaluación y reconocimiento (dataset chico)

- `entrenar(cfg, carpeta, salida=tmp_path / "modelo.npz")` sobre el dataset chico (con `archivo = {particion}/{nombre}_k{copia}.wav` relativo a la carpeta) devuelve esa ruta y el archivo tiene exactamente las 27 claves de 4.6; `cargar_modelo` da un modelo con los mismos valores; `(esquema, n_componentes, k)` son los de `seleccionar_hiperparametros` aplicada a las filas de train de `caracteristicas.npz` (con `n_otro_no_validos_train` de `metadatos.csv`), `tau` es `calibrar_umbral` de las filas de val con ese modelo y `n_otro_no_validos_val = 2` (1e-12), y el escalador es igual a `ajustar_escalador` de los bloques de train.
- **Sin fuga:** con las filas de `test` del `.npz` y del `metadatos.csv` cambiadas por otras, el modelo guardado es idéntico (`np.array_equal` clave por clave).
- **Reproducible:** dos `entrenar` a archivos distintos dan modelos idénticos.
- Con el archivo ya escrito y `sobrescribir=False`: `FileExistsError` y el archivo no cambia (mismo contenido y fecha de modificación); con `sobrescribir=True` se vuelve a escribir. `FileNotFoundError` si falta `caracteristicas.npz` o `metadatos.csv`.
- `guardar_modelo` → `cargar_modelo` devuelve exactamente lo guardado.
- `evaluar_particion(modelo, cfg, carpeta, "test")` es `evaluar` de las filas de test del `.npz` más las no válidas del `.csv` (la matriz de 4 × 4 suma `16 + 1` filas de test); también con `"val"` (`24 + 2`). `ValueError` con otra partición.
- **Reconocimiento:** para un audio de silencio, `reconocer` da `comando = "otro"`, `valido = False`, `motivo = "sin_voz"`, `prediccion = None`, `confianza = 0.0` y `distancia = inf`. Para un audio armado con un clip sintético de una clase rodeado de silencio, `reconocer` coincide con la cadena `preprocesar` → `caracteristicas` → `escalador.transformar` → `modelo.votar(z, cfg)` → `decidir` escrita en el test (si V2 lo da por no válido, `motivo` es el de `preprocesar`). `ValueError` con un modelo sin escalador.
- `generar_figura(cfg, carpeta, modelo, tmp_path / "rechazo_v4.png")` escribe un PNG que se puede leer con `matplotlib.pyplot.imread` y devuelve esa ruta.

### 6.9 Aceptación (`@pytest.mark.aceptacion`, con los datos reales de `data/voz/prepro/`)

Se saltea con `pytest.skip` si no existe `data/voz/prepro/caracteristicas.npz`. Con `entrenar(cfg, carpeta=RAIZ / "data/voz/prepro", salida=tmp_path / "modelo_voz.npz")`:

- **Tiempo:** todo (selección, ajuste y calibración) tarda menos de 5 minutos en la notebook.
- **Hiperparámetros** (reproducen el prototipo, sección 8): el modelo es `esquema = "z_columna"`, `n_componentes = 20` y `k = 5`. Con `seleccionar_hiperparametros` sobre las filas de train de los datos reales, esa combinación tiene `(n_tau5, n_c60, n_4cl) = (190, 221, 224)` y la de `"sin_dur_zcr"`, 30 y `k = 5`, `(195, 216, 221)`: las dos suman 411 y gana la primera por `n_4cl`; 231 comandos evaluados y 10 personas.
- **Sin fuga:** `tau` es igual a `calibrar_umbral` con solo las filas de `val` (1e-12) y está en `0.7738 ± 1e-3`.
- **`val`** (`evaluar_particion(…, "val")`, 101 filas = 87 válidas y 14 no válidas): `matriz == [[20, 0, 0, 3], [0, 19, 1, 2], [0, 0, 18, 4], [0, 0, 1, 33]]`, `otro_aceptado ≤ 1 / 34` (la regla de calibración con `m = 1`) y `acierto == 57 / 67` (1e-12).
- **`test`** (`evaluar_particion(…, "test")`, 105 filas = 99 válidas y 6 no válidas): `identificacion ≥ metas.voz_acierto_min` (en el prototipo, 70 de 71: 98,59 %) y `matriz == [[18, 0, 0, 6], [0, 17, 0, 7], [0, 0, 20, 3], [1, 2, 2, 29]]` (reproduce el prototipo: **no se usa para elegir nada**; si cambia, algo de lo anterior cambió). **No se exige** `acierto ≥ metas.voz_acierto_min` ni `otro_aceptado ≤ metas.voz_otro_aceptado_max` en `test`: el prototipo da 77,46 % y 14,71 %, y no las cumple (ver 8).
- **Reproducible:** dos corridas dan modelos idénticos.

## 7. Verificación manual (no va en pytest; se anota en el Documento 2)

- Correr `venv\Scripts\python -m src.voz.clasificacion --sobrescribir` y revisar `docs/figuras/rechazo_v4.png`: que τ\* quede donde marca el criterio (en la zona donde el `"otro"` aceptado de `val` baja a ≤ 1 de 34) y que la matriz de confusión de `test` coincida con lo que imprime la línea de comandos.
- Anotar en la ficha de V4 del Documento 2: los hiperparámetros elegidos y la tabla de la validación cruzada, τ\*, las métricas de `val` y de `test` (con y sin umbral de distancia) y las dos matrices, con **la diferencia entre lo medido y las metas del plan**.
- Con la cadena completa (`reconocer`), probar unos audios en vivo con la notebook (comandos, palabras parecidas como "salida" e "identidad", una tos, silencio) antes de la Fase 5.

## 8. Aclaraciones y valores medidos en el prototipo (24/10/2026)

Medidos con `Prototipos\v4_2026-10-24\` (`proto_v4.py`, `proto_v4b.py`, `proto_v4c.py`, `verificar_contrato_v4.py` y `verificar_criterio_cv.py`, con sus `resultados*.txt`) sobre `data/voz/prepro/caracteristicas.npz` y `metadatos.csv`.

- **K-NN propio contra scikit-learn** (`weights="distance"`, 87 consultas de `val` contra 1722 vecinos): diferencia máxima de distancias 1,1e-15 y predicciones idénticas (87 de 87) para `k` = 3, 5, 7 y 9. Tiempo del prototipo (diferencias completas, 281 dimensiones): 3,8 ms por clip.
- **Validación cruzada por persona en `train`:** 10 personas (p01 con p02, p09 con p13, p14 con p17 como una sola persona cada una; p06, p07, p08, p10, p11, p12 y p15), 231 comandos `k0` de notebook y 78 `"otro"` (72 válidos y 6 no válidos de V2). 36 combinaciones.
- **Las cinco mejores por el criterio de 4.3** (`n_tau5 + n_c60`, después `n_4cl`):

  | suma | `n_4cl` | `n_tau5` | `n_c60` | esquema | PCA | K | `tau_cv` |
  |---:|---:|---:|---:|---|---|---:|---:|
  | 411 | 224 | 190 | 221 | `z_columna` | 20 | 5 | 0,6900 |
  | 411 | 221 | 195 | 216 | `sin_dur_zcr` | 30 | 5 | 0,7672 |
  | 408 | 221 | 187 | 221 | `sin_dur_zcr` | 30 | 3 | 0,7354 |
  | 408 | 221 | 187 | 221 | `z_columna` | 20 | 3 | 0,6519 |
  | 407 | 225 | 192 | 215 | `z_columna` | 20 | 7 | 0,7173 |

- **Cuál gana y por qué:** `z_columna`, 20 componentes y `K = 5` empatan en 411 con `sin_dur_zcr`, 30 y `K = 5` (exactamente, con enteros) y gana por `n_4cl` (224 contra 221). **Es un empate muy justo:** si el criterio fuera solo `n_tau5`, ganaría `sin_dur_zcr`, 30 y `K = 5` (195), seguida de `z_columna`, 20 y `K = 7` (192) y de `z_columna`, 20 y `K = 5` (190). La suma se eligió porque `n_tau5` depende de una distancia puntual (el cuarto `"otro"` más cercano) y es más ruidoso; con el criterio de la suma, la regla es determinista y sale solo de `train`. En `val`, las otras dos dan (con su propio τ\*): `sin_dur_zcr`, 30 y `K = 5`, τ\* 0,8339, acierto 77,61 % (52 de 67) y `"otro"` aceptado 2,94 %; `z_columna`, 20 y `K = 7`, τ\* 0,7852, acierto 80,60 % (54 de 67) y `"otro"` aceptado 2,94 %.
- **Sin PCA:** en la esfera L2 de 281 dimensiones `n_tau5` cae a 68,8–75,8 % (`z_columna`) y 65,4–75,3 % (`sin_dur_zcr`), contra 80–84 % con PCA: la validación cruzada la descarta sin mirar `val`. Con el esquema `"contrato"`, `n_tau5` queda en 23–45 %.
- **Modelo elegido** (`z_columna`, 20, `K = 5`), τ\* = **0,7738**:
  - `val` (101 filas = 67 comandos y 34 `"otro"`, de los que 14 no válidos): sin umbral de distancia (solo con confianza ≥ 0,60), acierto 86,57 % (58 de 67) y `"otro"` aceptado 11,76 % (4 de 34); con τ\*, **acierto 85,07 % (57 de 67), `frr` 14,93 % y `"otro"` aceptado 2,94 % (1 de 34)**; identificación sin rechazo 92,54 % (62 de 67); matriz `[[20,0,0,3],[0,19,1,2],[0,0,18,4],[0,0,1,33]]`.
  - `test` (105 filas = 71 comandos y 34 `"otro"`, de los que 6 no válidos): sin umbral de distancia, acierto 92,96 % (66 de 71) y `"otro"` aceptado 26,47 % (9 de 34); con τ\*, **acierto 77,46 % (55 de 71), `frr` 22,54 % y `"otro"` aceptado 14,71 % (5 de 34)**; identificación sin rechazo 98,59 % (70 de 71); matriz `[[18,0,0,6],[0,17,0,7],[0,0,20,3],[1,2,2,29]]`: ninguna confusión entre comandos (lo que el sistema acepta como comando, y es comando, es el correcto), pero 16 comandos se rechazan como `"otro"` (6 + 7 + 3) y 5 `"otro"` se aceptan como comando.
- **Dónde se pierde el acierto (`val`):** los `"otro"` que se aceptan sin umbral de distancia son `salida` (de p04 y p20), `identidad` (de p20) y una palabra libre (de p21), que quedan muy cerca de `salir` e `identificar`. `nb_p21_otro_02_salida` y `nb_p21_otro_03_identidad` duran 2,46 y 2,43 s (casi el tope de 2,5 s de V2) y se rechazan solas por confianza (menor que 0,60) con `K = 5`.
- **Qué se hizo con `test`, para que quede escrito:** la verificación del 24/10 imprimió `val` y `test` de los cuatro primeros de la validación cruzada antes de escribir este contrato. El esquema, las componentes y `K` salen del criterio de 4.3 (solo `train`), τ\* sale solo de `val`, y el modelo no se ajustó después de ver `test`. Los números de `test` de arriba **no se usaron para elegir nada** y no se pueden usar para ajustar el modelo en esta versión.
- **Lo que no se resolvió (decisión de Santiago, no se cambia en esta versión):** las metas `metas.voz_acierto_min` (0,90) y `metas.voz_otro_aceptado_max` (0,05) no se cumplen en `test` con rechazo (77,46 % y 14,71 %), y en `val` el acierto queda en 85,07 %. El `"otro"` aceptado de `val` (2,94 %) sale por construcción, porque τ\* se calibra en `val`; la estimación honesta es la de `test`, que se miró una sola vez con esta versión y esos números van al informe tal cual. Hipótesis sobre la causa: en `train` hay 434 reconocer, 437 identificar, 418 salir y 433 otro (con copias), por lo que `"otro"` no está en minoría global; pero de los 433 `"otro"`, 214 son de TTS y 15 de ruido, y de personas reales hay 204 (unas 68 tomas originales con sus copias) repartidas en 6 tipos, así que de cada palabra parecida a un comando ("salida", "identidad") hay muy pocas tomas reales frente a las más de 400 de cada comando. Agregar tomas reales de `"otro"` parecidas a los comandos sería un cambio para una versión V4.1 que se mediría otra vez con `train` y `val` (el modelo no se toca ahora: se cierra V4 versión 1 y se sigue con la Fase 5).
- **Ejemplos de la calibración** (`cfg` real: `confianza_min` 0.60 y `otro_aceptado_max` 0.05; cada fila es `(verdad, prediccion, confianza, distancia)`):
  1. **Ejemplo A, separable** (`n_otro_no_validos = 0`; `n_otro = 4`, `m = 0`): `(reconocer, reconocer, 0.9, 0.50)`, `(identificar, identificar, 0.8, 0.60)`, `(salir, salir, 1.0, 0.70)`, `(reconocer, reconocer, 0.7, 0.55)`, `(otro, reconocer, 0.9, 1.00)`, `(otro, salir, 0.8, 1.30)`, `(otro, identificar, 0.7, 1.50)`, `(otro, otro, 0.9, 0.20)`. `B = [1.00, 1.30, 1.50]`, `b = 1.00`, `C = [0.50, 0.55, 0.60, 0.70]`, `a = 0.70`: **τ\* = 0.85**. Con τ = 0.85 las decisiones son: `reconocer`, `identificar`, `salir`, `reconocer`, `otro`, `otro`, `otro`, `otro`.
  2. **Ejemplo B, con solape:** igual, con las distancias `[0.50, 0.90, 1.30, 0.70, 1.10, 1.60, 2.00, 0.20]`. Con `otro_aceptado_max` 0.05 (`m = 0`): `b = 1.10`, `a = 0.90`, **τ\* = 1.00**. Con `otro_aceptado_max` 0.30 (`m = 1`): `b = 1.60`, `a = 1.30`, **τ\* = 1.45**.
  3. **Confianza baja:** el ejemplo A más la fila `(salir, salir, 0.5, 0.95)`: no cuenta para `C` (confianza menor que 0.60) y **τ\* sigue en 0.85**.
  4. **Con no válidos:** el ejemplo A con `n_otro_no_validos = 16` (`n_otro = 20`, `m = 1`): `b = B[1] = 1.30`, `a = 0.70`, **τ\* = 1.00**.
  5. **A lo sumo `m` aceptables sin umbral:** el ejemplo A sin las filas `(otro, salir, 0.8, 1.30)` y `(otro, identificar, 0.7, 1.50)`, con `n_otro_no_validos = 18` (`n_otro = 20`, `m = 1`, `B = [1.00]`, entra por el punto 4): `a = máx(C) = 0.70`; las distancias de todos los `"otro"` son `[0.20, 1.00]`, la menor mayor que `a` es `1.00`: **τ\* = 0.85**. (Con `n_otro_no_validos = 16` da `m = 0` y entra por el punto 3 dando también `0.85`).
  6. **Caso B vacío y b = 1.2 · a:** las 4 filas de comando del ejemplo A más `(otro, otro, 0.9, 0.20)`, con `n_otro_no_validos = 0` (`n_otro = 1`, `m = 0`): `B` queda vacío (`len(B) = 0 ≤ m`), `a = máx(C) = 0.70`; entre las filas de `es_otro` ninguna supera `a` (`0.20 ≤ 0.70`), por lo que `b = 1.2 · a = 0.84`, y **τ\* = (0.70 + 0.84) / 2 = 0.77**.
- **Ejemplo de `evaluar`** (filas `(verdad, prediccion, decision)`): `(reconocer, reconocer, reconocer)`, `(identificar, identificar, otro)`, `(salir, identificar, identificar)`, `(salir, salir, salir)`, `(otro, salir, salir)`, `(otro, otro, otro)`, `(otro, identificar, otro)`, `(otro, otro, otro)`. `identificacion = 3/4 = 0.75`, `acierto = 2/4 = 0.5`, `frr = 0.5`, `otro_aceptado = 1/4 = 0.25` y `matriz = [[1,0,0,0],[0,0,0,1],[0,1,1,0],[0,0,1,3]]` (orden `reconocer, identificar, salir, otro`).
- **Tipos:** como en H4, "float" es `isinstance(x, float)` (un `np.float64` vale) y "int" es un `int` de Python que no sea `bool`; los arrays se controlan por su dtype. Las comparaciones son absolutas salvo donde dice "relativa"; "igual" sin tolerancia se compara a 1e-12 e "idéntico" o `np.array_equal`, exacto.
- **Cómo se arma el dataset chico:** las clases de comandos son separables en los clips sintéticos, así que `calibrar_umbral` puede encontrar `C` y los tests de 6.8 no le piden metas al modelo. Con 4 clips por persona y clase (16 filas por persona y 32 de entrenamiento por pliegue en `train`), con 32 filas de entrenamiento por pliegue, `30 ≤ 31` y no se descarta (`n_componentes = 30 ≤ n_entrenamiento − 1 = 31`); el test no depende de cuál combinación gane, solo de que el archivo y las funciones coincidan con el oráculo.
