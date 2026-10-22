# Contrato · H1 · Generador de huellas y dataset

**Versión 1.2 · 22/10/2026** (la 1 es del 16/10/2026 y la 1.1, del 22/10/2026). **Cambio en la 1.2:** el corte de las variantes es como mucho del 10 % del área, porque lo pidió el profesor el 22/10: `corte_periferico` pasa de `[0.15, 0.20]` a `[0.05, 0.10]` (contrato de la Fase 0, versión 1.1), y el criterio de 5.5 sobre las fracciones del test de corte nombra los extremos de `corte_periferico` en lugar de 0,15 y 0,20. Está anotado en el Documento 2. **Cambios en la 1.1:** se suman las aclaraciones de cómo leen el contrato los tests (sección 7) y tres criterios en 5.4, 5.5 y 5.7, marcados con *(1.1)*. Los pidió el chat que escribió los tests y el cambio está anotado en el Documento 2.

Es lo que verifica `tests/test_h1.py`. Los tests se escriben en otro chat solo a partir de este archivo y se congelan (commit) antes de implementar. Se apoya en el contrato de la Fase 0 (`config.yaml` y `src/config.cargar_config`). Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

## 0. Convenciones

- **Imágenes:** `np.ndarray` de `uint8` y 2 dimensiones `(alto, ancho)`, con crestas oscuras sobre fondo claro.
- **Coordenadas:** `(x, y)` en píxeles, como `float`. `x` es la columna e `y` la fila; el eje y apunta hacia abajo y el píxel `imagen[i, j]` está en `(x, y) = (j, i)`.
- **Orientación:** ángulo de la cresta en radianes en `[0, π)`, medido desde el eje x hacia el eje y (como en OpenCV).
- **Índice de Poincaré** de un campo `θ` en `(x0, y0)`: se toman `N = 72` puntos `p_k = (x0 + r·cos φ_k, y0 + r·sen φ_k)`, con `φ_k = 2πk/N`, `k = 0…N` (el último repite el primero) y `r = 8 px`. En cada punto se interpolan bilinealmente `cos 2θ` y `sen 2θ` (por ejemplo con `scipy.ndimage.map_coordinates(..., order=1)`, que recibe `(fila, columna) = (y, x)`) y se vuelve a `θ = ½·atan2(sen, cos)`. Cada `Δθ_k = θ_{k+1} − θ_k` se lleva al intervalo `(−π/2, π/2]` y el índice es `Σ Δθ_k / (2π)`. Con esta definición un **núcleo** tiene índice **+½** y un **delta**, **−½**.
- **Transformación afín** de una variante, con `(cx, cy) = ((ancho − 1)/2, (alto − 1)/2)`, `α = rot_grados` pasado a radianes y `s = escala`:
  - `x' = cx + s·[cos α·(x − cx) + sen α·(y − cy)] + tx_px`
  - `y' = cy + s·[−sen α·(x − cx) + cos α·(y − cy)] + ty_px`

  Es la matriz de `cv2.getRotationMatrix2D((cx, cy), rot_grados, escala)` más la traslación: un ángulo positivo gira en sentido antihorario en la pantalla.
- **Período medido** de las crestas de una imagen: se toma el recorte central de 192 × 192 px (filas desde `(alto − 192)//2`, columnas desde `(ancho − 192)//2`), se le resta su media, se multiplica por la ventana `np.outer(np.hanning(192), np.hanning(192))` y se calcula la potencia `|FFT2|²`. Con `f` la frecuencia radial en ciclos/px (a partir de `np.fft.fftfreq(192)`), `f̄` es el promedio de `f` ponderado por la potencia en la banda `1/20 < f < 1/5`. El período medido es `1/f̄`.
- **Correlación** entre dos imágenes: coeficiente de Pearson de los píxeles donde las dos máscaras valen `True`.

## 1. Claves nuevas de config.yaml

Se suman a las de la Fase 0. `cargar_config` también las exige: si falta alguna, lanza `ValueError` con su camino completo.

```yaml
huellas:
  generador:
    tamano_px: [400, 320]            # fijo: (alto, ancho)
    periodo_base_px: [8.7, 9.3]      # fijo: rango del período de las crestas de cada base
    iteraciones: <entero ≥ 5>        # pasadas del filtrado de Gabor
    orientaciones: <entero ≥ 8>      # filtros del banco
  variantes:                         # además de las claves de la Fase 0
    prob_corte: <float en [0, 1]>
    elastica_max_px: <float en [0, 5]>
    presion_max: <float en (0, 1]>
    ruido_sigma_max: <float en [0, 30]>
    sal_pimienta_max: <float en [0, 0.05]>
  intrusos:
    tipos: [presilla, verticilo, arco_tienda, arco]   # fijo, en este orden
```

## 2. src/huellas/generador.py

```python
TIPOS: tuple[str, ...] = ("presilla", "verticilo", "arco_tienda", "arco")

@dataclass(frozen=True)
class Singularidades:
    nucleos: tuple[tuple[float, float], ...]   # (x, y)
    deltas: tuple[tuple[float, float], ...]

@dataclass(frozen=True)
class HuellaBase:
    tipo: str
    semilla: int
    imagen: np.ndarray            # uint8 (alto, ancho)
    mascara: np.ndarray           # bool (alto, ancho): True dentro de la huella
    orientacion: np.ndarray       # float64 (alto, ancho), en [0, π)
    singularidades: Singularidades
    periodo_px: float             # período usado para sintetizar las crestas

def campo_orientacion(singularidades: Singularidades, tamano: tuple[int, int],
                      theta_inf: float = 0.0) -> np.ndarray:
    # Modelo de Sherlock y Monro:
    #   θ(x, y) = θ∞ + ½·[Σ arg(z − c_k) − Σ arg(z − d_k)]  (mod π),  z = x + i·y,
    # con c_k los núcleos y d_k los deltas. tamano = (alto, ancho).
    # Devuelve float64 (alto, ancho) en [0, π). Sin singularidades: θ∞ mod π en toda la imagen.

def generar_base(tipo: str, semilla: int, cfg: dict) -> HuellaBase:
    # Patrón base de una identidad: singularidades según el tipo, campo de orientación,
    # máscara elíptica suave y crestas sintetizadas con filtrado de Gabor iterativo
    # (como SFinGe) desde ruido fijado por la semilla. ValueError si tipo no está en TIPOS.
```

**Reglas:**

- El tamaño es `cfg["huellas"]["generador"]["tamano_px"]`.
- Singularidades por tipo (posiciones sorteadas con la semilla):

| Tipo | Núcleos | Deltas | Geometría |
|---|---|---|---|
| `presilla` | 1 | 1 | El delta está al menos 40 px más abajo que el núcleo y al menos 40 px hacia un costado. |
| `verticilo` | 2 | 2 | Núcleos a 18–40 px entre sí. Un delta a la izquierda y otro a la derecha de los dos núcleos (a ≥ 40 px en x), ambos al menos 40 px más abajo que el punto medio de los núcleos. |
| `arco_tienda` | 1 | 1 | El delta está debajo del núcleo: entre 15 y 40 px más abajo y a no más de 10 px en x. |
| `arco` | 0 | 0 | — |

- Toda singularidad queda a ≥ 20 px de los bordes de la imagen y a ≥ 18 px de cualquier otra. Los núcleos quedan dentro de la máscara.
- `orientacion`: para `presilla`, `verticilo` y `arco_tienda` es `campo_orientacion(singularidades, tamano_px, 0.0)`. Para `arco` es un campo propio, sin singularidades (crestas en forma de lomo).
- `periodo_px` está en `periodo_base_px` (intervalo cerrado).
- Determinista: con los mismos `tipo`, `semilla` y `cfg` todos los arrays y valores son iguales.

## 3. src/huellas/variantes.py

```python
@dataclass(frozen=True)
class ParametrosVariante:
    rot_grados: float
    escala: float
    tx_px: float
    ty_px: float
    elastica_px: float            # amplitud máxima del desplazamiento elástico
    presion: float                # > 0 crestas más gruesas; < 0 más finas
    corte_fraccion: float         # fracción del área de la huella que se quita; 0 = sin corte
    corte_angulo_grados: float    # hacia dónde está la zona que se quita
    ruido_sigma: float            # desvío del ruido gaussiano, en niveles de gris
    sal_pimienta: float           # fracción de píxeles llevados a 0 o 255

SIN_CAMBIOS = ParametrosVariante(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

def sortear_parametros(rng: np.random.Generator, cfg: dict) -> ParametrosVariante: ...

@dataclass(frozen=True)
class Variante:
    imagen: np.ndarray                     # uint8 (alto, ancho)
    mascara: np.ndarray                    # bool (alto, ancho)
    singularidades: Singularidades         # posiciones reales en la variante
    punto_referencia: tuple[float, float] | None
    parametros: ParametrosVariante         # los aplicados (con el ángulo de corte usado)
    mapa_x: np.ndarray                     # float32 (alto, ancho): mapeo inverso de cv2.remap
    mapa_y: np.ndarray                     # float32 (alto, ancho)

def aplicar_variante(base: HuellaBase, params: ParametrosVariante,
                     rng: np.random.Generator) -> Variante: ...
```

**`sortear_parametros`** (distribuciones uniformes, con las claves de `cfg["huellas"]["variantes"]`):

- `rot_grados` en `[−rotacion_max_grados, rotacion_max_grados]`; `escala` en `[1 − escala_max, 1 + escala_max]`; `tx_px` y `ty_px` en `[−traslacion_max_px, traslacion_max_px]`.
- `elastica_px` en `[0, elastica_max_px]`; `presion` en `[−presion_max, presion_max]`; `ruido_sigma` en `[0, ruido_sigma_max]`; `sal_pimienta` en `[0, sal_pimienta_max]`.
- Con probabilidad `prob_corte`: `corte_fraccion` en `corte_periferico` (cerrado) y `corte_angulo_grados` en `[0, 360)`. Si no, los dos valen 0.

**`aplicar_variante`**, en este orden:

1. **Presión:** con `presion > 0` la imagen se mezcla con su erosión en escala de grises (núcleo 3 × 3) con peso `|presion|`, y las crestas engrosan; con `presion < 0`, lo mismo con la dilatación, y las crestas se afinan. Con `presion = 0` no cambia nada.
2. **Geometría:** un único `cv2.remap` con `(mapa_x, mapa_y)` = la inversa de la transformación afín de la sección 0, más un desplazamiento elástico suave sorteado con `rng`, de amplitud máxima `elastica_px` (si es 0, no hay desplazamiento). Lo que queda afuera de la imagen original se completa con fondo claro.
3. **Posiciones reales:** cada núcleo y cada delta se ubica pasando por el mismo `remap` un mapa auxiliar con un punto gaussiano en su posición de la base y tomando el máximo con precisión subpíxel. `punto_referencia`: con 1 núcleo, ese núcleo; con 2, el punto medio; sin núcleos, `None`.
4. **Corte** (si `corte_fraccion > 0`): se quita de la máscara el semiplano que se lleva esa fracción del área de la huella, del lado que indica `corte_angulo_grados` (dirección `(cos a, sen a)` en coordenadas de imagen: 0° quita la derecha y 90° la parte de abajo), y ahí la imagen pasa a fondo. Ningún núcleo puede quedar a menos de 30 px de la zona quitada: si con ese ángulo pasaría, se gira de a 30° hasta encontrar uno que sirva y `parametros` guarda el ángulo usado. Si ninguno sirve, no se corta y `corte_fraccion` queda en 0.
5. **Ruido:** gaussiano de desvío `ruido_sigma` y sal y pimienta en una fracción `sal_pimienta` de los píxeles (mitad a 0 y mitad a 255).

No modifica `base`. Con la misma `base`, los mismos `params` y un `rng` con la misma semilla, la variante es idéntica.

## 4. src/huellas/dataset.py

```python
COLUMNAS: tuple[str, ...] = (
    "archivo", "particion", "identidad", "intruso", "tipo", "semilla", "variante", "reservada_demo",
    "rot_grados", "escala", "tx_px", "ty_px", "elastica_px", "presion",
    "corte_fraccion", "corte_angulo_grados", "ruido_sigma", "sal_pimienta",
    "n_nucleos", "nucleos", "n_deltas", "deltas", "ref_x", "ref_y",
)

def plan_dataset(cfg: dict) -> list[dict]:
    # Una entrada por imagen, sin generar nada, con las claves archivo, particion,
    # identidad, intruso, tipo, semilla, variante y reservada_demo.

def generar_dataset(cfg: dict, destino: str | os.PathLike | None = None,
                    sobrescribir: bool = False) -> list[dict]:
    # Genera los PNG y metadatos.csv en destino (por defecto RAIZ / rutas.huellas).
    # Devuelve las filas de metadatos.csv como dicts con las claves de COLUMNAS.

def generar_lamina(cfg: dict, salida: str | os.PathLike | None = None) -> Path:
    # Lámina para la revisión visual. Por defecto RAIZ / rutas.figuras / "grilla_h1.png".
    # Devuelve la ruta escrita.
```

**Reglas:**

- **Autorizadas:** para cada identidad `k` de `huellas.identidades` (con su tipo y su semilla), variantes `v = 1…variantes_por_identidad`. Las primeras `particion.train` van a train, las `particion.val` siguientes a val y las `particion.test` últimas a test. `reservada_demo = 1` en las últimas `reservadas_demo`, que quedan dentro de test.
- **Intrusos:** `i = 1…intrusos.cantidad`, con `semilla = semilla_base + i` y `tipo = tipos[(i − 1) % len(tipos)]`, y variantes `v = 1…variantes_por_intruso`. Los primeros `intrusos.val` intrusos van a val y el resto a test.
- **Cada imagen** sale de `base = generar_base(tipo, semilla, cfg)`, `rng = np.random.default_rng([semilla, v])`, `params = sortear_parametros(rng, cfg)` y `variante = aplicar_variante(base, params, rng)`, con el mismo `rng` y en ese orden.
- **Archivos** (relativos a `destino`): `{particion}/id{k}_v{v:03d}.png` y `{particion}/intruso{i:02d}_v{v}.png`.
- **metadatos.csv** en `destino`: UTF-8, separado por comas, con encabezado igual a `COLUMNAS` y una fila por imagen, en el orden de `plan_dataset`.
  - `archivo` con `/`; `identidad`: 1, 2 o 3 (0 en los intrusos); `intruso`: el número de intruso (0 en las autorizadas); `reservada_demo`: 0 o 1.
  - Los parámetros son los de `variante.parametros`.
  - `nucleos` y `deltas`: JSON con la lista de `[x, y]` en la imagen de la variante (ej. `[[160.25, 181.5]]`); `n_nucleos` y `n_deltas`, su cantidad.
  - `ref_x` y `ref_y`: el punto de referencia, vacíos si no hay núcleo.
  - Redondeo: los `float` van con 4 decimales y las coordenadas de `nucleos` y `deltas`, con 2.
  - En lo que devuelve `generar_dataset` cada valor tiene su tipo: `archivo`, `particion`, `tipo`, `nucleos` y `deltas` son `str` (los dos últimos, el mismo JSON del CSV); las columnas enteras son `int`; los parámetros son `float`, con el mismo redondeo; `ref_x` y `ref_y` son `float`, o `None` cuando en el CSV están vacíos.
- Si `destino` ya tiene algún PNG en `train/`, `val/` o `test/`, o un `metadatos.csv`, y `sobrescribir` es `False`: `FileExistsError` sin tocar nada. Con `sobrescribir=True` borra solo esos PNG y ese `metadatos.csv` y vuelve a generar.
- La **lámina** tiene una fila por identidad autorizada (la base y 6 variantes, con los núcleos marcados) y una fila con bases de intrusos.
- **Línea de comandos:** `venv\Scripts\python -m src.huellas.dataset [--sobrescribir]` genera el dataset y la lámina con `config.yaml`.

## 5. Criterios que verifican los tests

### 5.1 Configuración

- Las claves de la sección 1 existen, con su tipo y rango; las `fijo`, con su valor exacto.
- `cargar_config` lanza `ValueError` que nombra el camino completo si falta cualquiera de ellas.

### 5.2 Campo de orientación (`campo_orientacion`)

- Devuelve `float64` de forma `(alto, ancho)` con valores en `[0, π)`.
- Con singularidades puestas a mano (por ejemplo, en una imagen de 400 × 320, un núcleo en (150, 150) y un delta en (230, 270)): índice de Poincaré +0,5 ± 0,05 en el núcleo, −0,5 ± 0,05 en el delta y 0 ± 0,05 en puntos a 40 px o más de los dos.
- Sin singularidades devuelve `θ∞ mod π` en toda la imagen.

### 5.3 Huella base (`generar_base`, con el `cfg` real, cada tipo y al menos 3 semillas)

- `imagen` es `uint8` de `tamano_px`; `mascara` (`bool`) y `orientacion` (`float64`, en `[0, π)`) tienen el mismo tamaño.
- Cantidad y geometría de las singularidades según la tabla de la sección 2, a ≥ 20 px de los bordes y a ≥ 18 px entre sí, con los núcleos dentro de la máscara.
- Índice de Poincaré sobre `orientacion`: +0,5 ± 0,05 en cada núcleo, −0,5 ± 0,05 en cada delta y 0 ± 0,05 en 5 puntos de la máscara que estén a ≥ 40 px de toda singularidad y a ≥ 20 px de los bordes. En el arco, 0 ± 0,05 en 5 puntos de la máscara a ≥ 40 px de los bordes.
- Para `presilla`, `verticilo` y `arco_tienda`, `orientacion` coincide con `campo_orientacion(singularidades, tamano_px, 0.0)` (comparando `cos 2θ` y `sen 2θ`, tolerancia 1e-9).
- Crestas oscuras: el gris promedio dentro de la máscara es al menos 50 niveles menor que afuera, y afuera (a más de 10 px de la máscara) el promedio es ≥ 230.
- `periodo_px` está en `periodo_base_px` y el período medido (sección 0) está entre 8 y 10 px.
- Determinismo: el mismo `(tipo, semilla)` da `imagen`, `mascara` y `orientacion` idénticas (`np.array_equal`) y las mismas singularidades.
- Semillas distintas, del mismo tipo o de tipos distintos: |correlación| < 0,35.
- `ValueError` con un tipo que no está en `TIPOS`.

### 5.4 Parámetros (`sortear_parametros`, 1000 sorteos con un `rng` de semilla fija)

- Todos dentro de sus rangos. `corte_fraccion` vale 0 o está en `corte_periferico`; cuando vale 0, `corte_angulo_grados` también.
- La proporción de sorteos con corte está a ±0,05 de `prob_corte`.
- El mismo `rng` (misma semilla) da los mismos parámetros.
- *(1.1)* Respeta los rangos de un `cfg` con rangos más chicos: no los tiene fijos en el código.

### 5.5 Variantes (`aplicar_variante`)

- `imagen` (`uint8`) y `mascara` (`bool`) tienen el tamaño de la base; `mapa_x` y `mapa_y` son `float32` del mismo tamaño. La base no se modifica.
- Con `SIN_CAMBIOS`: la imagen difiere de la base en a lo sumo 1 nivel de gris, las singularidades quedan a ≤ 0,5 px de las de la base y la máscara coincide en al menos el 99,5 % de los píxeles.
- Solo rotación, traslación y escala (`elastica_px`, `presion`, `corte_fraccion`, `ruido_sigma` y `sal_pimienta` en 0): cada núcleo, y cada delta que quede a ≥ 10 px de los bordes, está a ≤ 1 px de la fórmula afín de la sección 0. Se prueba con al menos 20 juegos de parámetros sorteados en los rangos de `cfg` y con los tres tipos autorizados.
- Con parámetros completos, elástica incluida: evaluando `mapa_x` y `mapa_y` (interpolación bilineal) en cada posición real de la variante que quede a ≥ 10 px de los bordes, se vuelve a su posición en la base con un error ≤ 0,5 px. Es lo que detecta un desplazamiento elástico aplicado al revés.
- `punto_referencia`: el núcleo si hay uno, el punto medio si hay dos y `None` en el arco.
- Corte: con los mismos parámetros y la misma semilla de `rng`, la fracción de la máscara que se pierde al pasar de `corte_fraccion = 0` a `corte_fraccion = f` está en `f ± 0,02`, y el disco de 20 px de radio alrededor de cada núcleo queda entero dentro de la máscara con corte.
- *(1.1)* Lado del corte: 0° quita la derecha y 90° la parte de abajo. Lo quitado es un semiplano del lado del ángulo guardado, y ese ángulo es el pedido más un múltiplo de 30°. El test de corte usa también `f = 0,40` (además de los extremos de `corte_periferico`), porque con esas fracciones la regla de los 30 px nunca entra en juego.
- Presión (el resto como en `SIN_CAMBIOS`): la fracción de píxeles de la máscara por debajo de 128 crece con la presión. Con `+presion_max` es al menos 0,05 mayor que con 0, y con `−presion_max`, al menos 0,05 menor.
- Ruido (el resto como en `SIN_CAMBIOS`): con `ruido_sigma = 10`, el desvío de la diferencia con la base dentro de la máscara está entre 7 y 13. Con `sal_pimienta = 0,02`, la fracción de píxeles en 0 o 255 que no lo estaban en la base está entre 0,01 y 0,03.
- Determinismo: la misma base, los mismos parámetros y un `rng` con la misma semilla dan una `imagen` idéntica.

### 5.6 Plan del dataset (`plan_dataset`)

- Con el `cfg` real hay 300 entradas: 72 en train, 114 en val (24 + 90) y 114 en test (24 + 90).
- Por identidad autorizada: variantes 1–24 en train, 25–32 en val y 33–40 en test, con `reservada_demo = 1` solo en la 39 y la 40.
- Intrusos 1–30 solo en val y 31–60 solo en test, con 3 variantes cada uno; tipos en el ciclo `presilla, verticilo, arco_tienda, arco`.
- Semillas: las de `config.yaml` para las identidades y `semilla_base + i` para los intrusos.
- Nombres de archivo con el formato de la sección 4, la carpeta igual a la partición y ningún archivo repetido.
- Con una configuración reducida (por ejemplo 5 variantes por identidad repartidas 3/1/1 con 1 reservada, y 4 intrusos con 2 variantes, 2 en val y 2 en test) los conteos siguen las mismas reglas.

### 5.7 Generación (`generar_dataset` con la configuración reducida, en `tmp_path`)

- Crea exactamente los PNG del plan y `metadatos.csv`. Cada PNG se lee con `cv2.imread(ruta, cv2.IMREAD_UNCHANGED)` como `uint8` 2D de `tamano_px`.
- `metadatos.csv` tiene el encabezado igual a `COLUMNAS` y una fila por PNG, en el orden del plan y con sus mismos valores en las columnas del plan. Las filas coinciden con lo que devuelve la función.
- `n_nucleos` según el tipo (1, 2, 1 y 0); `ref_x` y `ref_y` vacíos solo si `n_nucleos = 0`; en el verticilo, el punto de referencia es el punto medio de `nucleos` (±0,01 px); el punto de referencia cae dentro de la imagen.
- Parámetros dentro de los rangos de `cfg`.
- Reproducible: rehacer una fila a partir de su `tipo`, `semilla` y `variante` con la receta de la sección 4 da exactamente la misma imagen que el PNG. Generar dos veces en carpetas distintas da PNG idénticos byte a byte y el mismo `metadatos.csv`.
- Si ya hay datos: `FileExistsError` sin `sobrescribir`, sin cambiar ningún archivo; con `sobrescribir=True` vuelve a generar.
- *(1.1)* Con `sobrescribir=True` no se borra nada que no sea un PNG de `train/`, `val/` o `test/` o `metadatos.csv`.
- `generar_lamina(cfg, salida)` escribe en `salida` un PNG que se puede leer y devuelve esa ruta.

### 5.8 Aceptación (`@pytest.mark.aceptacion`)

- Dataset completo con el `cfg` real en `tmp_path`: 72 / 114 / 114 imágenes y `metadatos.csv` completo, en menos de 5 minutos en la notebook.

## 6. Verificación manual (no va en pytest; se anota en el Documento 2)

- En `docs/figuras/grilla_h1.png` se reconocen una presilla, un verticilo y un arco en tienda; las variantes muestran rotación, traslación, cortes, presión y ruido, y las marcas de núcleo caen sobre el núcleo.
- Tiempo de `venv\Scripts\python -m src.huellas.dataset` en la notebook.

## 7. Aclaraciones de la versión 1.1 (cómo leen el contrato los tests)

- **Sección 1:** los `<float …>` exigen `float` de Python (`3` no sirve, `3.0` sí). Las claves `fijo` se comparan por tipo y valor. Si falta una sección entera (`huellas.generador`, `huellas.variantes`, `huellas.intrusos` o `huellas`), el `ValueError` nombra todas sus hojas nuevas.
- **Distancia al borde**, a los centros de píxel: `min(x, y, ancho − 1 − x, alto − 1 − y)`. "Dentro de la imagen": `0 ≤ x ≤ ancho − 1` y `0 ≤ y ≤ alto − 1`.
- **Verticilo:** el delta izquierdo queda a ≥ 40 px a la izquierda del núcleo de más a la izquierda, y el derecho a ≥ 40 px a la derecha del de más a la derecha.
- **Crestas oscuras:** "afuera" son los píxeles a más de 10 px de la máscara, en las dos comparaciones.
- **Variantes:** las singularidades se comparan por cercanía, sin exigir orden. Tienen que estar todos los núcleos; un delta a menos de 10 px del borde, o afuera, puede faltar.
- **Núcleos lejos del borde:** el test afín no exceptúa núcleos cerca del borde, y el disco de 20 px tiene que quedar dentro de la máscara en cualquier variante. En la práctica, el generador tiene que poner los núcleos bien adentro.
- **Corte:** si ningún ángulo sirve, `corte_fraccion` y `corte_angulo_grados` quedan en 0 y la máscara no cambia. El ángulo usado se guarda en `[0, 360)`. El disco de 20 px se toma literal: entero dentro de la máscara con corte.
- **Sal y pimienta:** la fracción se mide sobre los píxeles que en la base no estaban en 0 ni en 255.
- **metadatos.csv:** UTF-8 sin BOM; los `float` pueden escribirse con hasta 4 decimales (`0.15` o `0.1500`). "Crea exactamente": en `destino` no queda ningún otro archivo. "`FileExistsError` sin tocar nada" incluye no crear carpetas.
- **Lámina:** `generar_lamina` se prueba con el `cfg` real.
- **Tiempo:** el archivo de tests llama unas 45 veces a `generar_base`; si una base tarda más de ~2,5 s, pytest pasa de los 2 minutos.
