
# Contrato · H2 · Preprocesamiento de huellas

**Versión 1 · 22/10/2026.** Es lo que verifica `tests/test_h2.py`. Los tests se escriben en otro chat solo a partir de este archivo y se congelan (commit) antes de implementar. Se apoya en los contratos de la Fase 0 (`config.yaml` y `src/config.cargar_config`) y de H1 (convenciones de la sección 0, `generar_base`, `sortear_parametros`, `aplicar_variante` y la receta de la sección 4). Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

**Cambios respecto del plan v0.3.2** (medidos con un prototipo sobre las 300 imágenes del dataset de `v0.1.2-huellas-datos`; quedan en el Documento 2):

- **Cortes de hasta el 10 % del área** (decisión 43, pedido del profesor el 22/10): `corte_periferico` es `[0.05, 0.10]` (contrato de la Fase 0, versión 1.1). Reemplaza el 15–20 % del plan; la prueba de estabilidad usa el máximo de `corte_periferico` en lugar del 15 %.
- **Ángulo de enderezado (decisión 44, reemplaza el estimador de la 18):** θ∞ del modelo de Sherlock y Monro ajustado al campo observado, en lugar de la orientación media en un anillo. Con los cortes de hasta el 10 %, el anillo da ≤ 1,6° sin corte, pero un corte se lleva parte del anillo y sesga el ángulo hasta 5,2° en la presilla y 10,6° en el verticilo, donde el criterio de ≤ 5° pasa justo (95,3 %); con los cortes de 15–20 % llegaba a 16°. El ajuste da ≤ 1,1° con corte o sin él (2,0° en el arco). Sigue siendo un único estimador para todas las huellas: en un anillo muy grande, el promedio tiende a θ∞, y en el arco el ajuste sin singularidades es el promedio circular del campo.
- **Período de la imagen realzada:** se compara con el período real de cada variante (`periodo_px · escala`) y no con el rango de 8 a 10 px, porque con escala 1,08 una base de 9,3 px queda en 10,04 px.
- **Estabilidad:** el umbral de la media de cos 2Δθ pasa de 0,9 a 0,95 (el prototipo da ≥ 0,998).

## 0. Convenciones

- Las de la sección 0 del contrato de H1: imágenes `uint8` de 2 dimensiones con crestas oscuras sobre fondo claro, coordenadas `(x, y)` con `imagen[i, j]` en `(x, y) = (j, i)`, orientación de la cresta en `[0, π)` medida desde el eje x hacia el eje y, transformación afín de una variante y período medido.
- **Ventana:** `(alto, ancho) = huellas.ventana_px = (250, 250)`, con centro `c_w = ((ancho − 1)/2, (alto − 1)/2) = (124.5, 124.5)`.
- **Matriz de la ventana:** con `a = angulo` y `(x_r, y_r) = punto_referencia`,

  `M = [[cos a, sen a, t_x], [−sen a, cos a, t_y]]`, con `(t_x, t_y)` tal que `M·(x_r, y_r, 1) = c_w`.

  Es `cv2.getRotationMatrix2D((x_r, y_r), grados(a), 1.0)` con otra traslación: rotación y traslación, sin escala. La ventana es `cv2.warpAffine(entrada, M, (ancho, alto))`: el píxel `(x_w, y_w)` de la ventana viene del punto `M⁻¹·(x_w, y_w)` de la entrada. Una orientación θ de la entrada queda como θ − a en la ventana, así que la orientación `angulo` queda horizontal.
- **θ∞ y el ángulo real:** con la convención de H1, girar la imagen α en sentido antihorario (`rot_grados = α`) le resta α a toda orientación. Las bases de presilla, verticilo y arco en tienda tienen θ∞ = 0, y el campo del arco es simétrico, con orientación media ≈ 0. Por eso **el ángulo real de una variante es −`rot_grados`**.
- **Error de ángulo** de una variante, en grados: `e = ((grados(angulo) + rot_grados + 90) mod 180) − 90`, en `[−90, 90)`.
- **Punto de referencia real:** `Variante.punto_referencia` (es el `ref_x, ref_y` de `metadatos.csv`): el núcleo, el punto medio de los dos núcleos o `None` en el arco.
- **Campo y crestas verdaderas en la ventana** de una variante: para cada píxel `(x_w, y_w)` de la ventana se toman `p = M⁻¹·(x_w, y_w)` (con `cv2.invertAffineTransform(matriz)`) y el punto de la base `q = (mapa_x(p), mapa_y(p))`, interpolando los mapas de la variante bilinealmente (por ejemplo con `scipy.ndimage.map_coordinates(..., order=1)`, que recibe `(fila, columna) = (y, x)`).
  - Campo verdadero: `θ_v = θ_b(q) − rad(rot_grados) − angulo`, donde `θ_b(q) = ½·atan2(s, c)` con `c` y `s` los valores de `cos 2θ` y `sen 2θ` de `base.orientacion` interpolados en `q`. La parte elástica gira el campo menos de 2° y no se descuenta.
  - Crestas verdaderas: `base.imagen` interpolada en `q` es menor que 128.
- **Máscara real en la ventana:** `cv2.warpAffine(variante.mascara (uint8), matriz, (ancho, alto), flags=cv2.INTER_NEAREST)` > 0. **Dentro de la entrada:** lo mismo con una imagen de unos del tamaño de la entrada.
- **Zona de comparación:** `mascara ∩ máscara real en la ventana`, erosionada con un cuadrado de 9 × 9 (4 px).
- **IoU** de dos máscaras: `|A ∩ B| / |A ∪ B|`.
- **Bloque de 2 × 2** de un esqueleto `E`: un `(i, j)` con `E[i, j]`, `E[i + 1, j]`, `E[i, j + 1]` y `E[i + 1, j + 1]` en `True`.

## 1. Claves nuevas de config.yaml

Se suman a las de la Fase 0 y H1. `cargar_config` también las exige: si falta alguna, lanza `ValueError` con su camino completo. `huellas.ventana_px: [250, 250]` y `huellas.periodo_crestas_px: [8, 10]` ya existen desde la Fase 0 y se usan tal cual (las dos son `fijo`).

```yaml
huellas:
  preprocesamiento:                     # H2 (contrato h2_preprocesamiento.md)
    mediana_px: <entero impar en [3, 7]>          # lado del filtro de mediana (sal y pimienta)
    umbral_segmentacion: <float en (0, 1)>        # fracción del percentil 95 de la energía local
    sigma_tensor_px: <float en [1, 10]>           # suavizado del tensor de estructura para buscar singularidades
    sigma_ajuste_px: <float en [1, 10]>           # suavizado del campo del ajuste y de la ventana
    radio_poincare_px: <float en [3, 15]>         # radio del círculo del índice de Poincaré
    margen_singularidades_px: <float en [0, 50]>  # se descartan singularidades más cerca del borde de la máscara
    holgura_nucleo_px: <float en [0, 15]>         # cuánto puede mover el ajuste cada núcleo detectado
    clahe_limite: <float en [1, 5]>
    clahe_grilla: <entero en [2, 16]>             # celdas por lado de CLAHE
```

## 2. src/huellas/preprocesamiento.py

```python
METODOS: tuple[str, ...] = ("nucleo", "nucleos", "curvatura")

@dataclass(frozen=True)
class HuellaPreprocesada:
    realzada: np.ndarray        # uint8 (250, 250): crestas oscuras; 255 fuera de la máscara
    binaria: np.ndarray         # bool (250, 250): True en las crestas
    esqueleto: np.ndarray       # bool (250, 250): crestas de 1 px
    mascara: np.ndarray         # bool (250, 250): True dentro de la huella
    orientacion: np.ndarray     # float64 (250, 250), en [0, π), en los ejes de la ventana
    punto_referencia: tuple[float, float]       # (x, y) en la imagen de entrada
    metodo: str                                 # uno de METODOS
    nucleos: tuple[tuple[float, float], ...]    # núcleos usados, (x, y) en la imagen de entrada
    angulo: float               # θ∞ estimado, en radianes, en [0, π), en la imagen de entrada
    matriz: np.ndarray          # float64 (2, 3): de la entrada a la ventana (sección 0)

def preprocesar(imagen: np.ndarray, cfg: dict) -> HuellaPreprocesada:
    # Lleva una huella a la ventana de 250 × 250 alineada por su punto de referencia y
    # enderezada por θ∞, con la imagen realzada, las crestas binarizadas y el esqueleto.

def generar_lamina(cfg: dict, salida: str | os.PathLike | None = None) -> Path:
    # Lámina para la revisión visual. Por defecto RAIZ / rutas.figuras / "lamina_h2.png".
    # Devuelve la ruta escrita.
```

## 3. Reglas de preprocesar

1. **Entrada:** `uint8` de forma `(alto, ancho)`, `(alto, ancho, 3)` (BGR) o `(alto, ancho, 4)` (BGRA), con alto y ancho ≥ 64. Cualquier otra cosa: `ValueError`. El color pasa a gris (por ejemplo con `cv2.cvtColor`). Se asume la resolución del dataset (crestas de 8 a 10 px de período): el lienzo puede tener cualquier tamaño, pero la huella no se reescala. No modifica `imagen`.
2. **Limpieza:** filtro de mediana de `mediana_px` contra la sal y pimienta.
3. **Segmentación:** energía local de la imagen pasa-altos (sin el escalón entre la huella y el fondo), umbral relativo `umbral_segmentacion` × percentil 95, apertura morfológica, la componente conexa más grande y los agujeros rellenos. Si no queda huella (por ejemplo, una imagen uniforme): `ValueError`.
4. **Campo de orientación:** tensor de estructura de los gradientes, suavizado con un filtro gaussiano (`sigma_tensor_px` para buscar singularidades; `sigma_ajuste_px` para el ajuste y la ventana), y su coherencia.
5. **Singularidades:** índice de Poincaré con radio `radio_poincare_px` sobre el campo suavizado, solo en puntos a ≥ `margen_singularidades_px` del borde de la máscara (y de la imagen). Los núcleos salen de las zonas de índice +½; una zona que vale +1 cuenta como dos núcleos juntos. Si hay más de dos, se usan los dos más fuertes.
6. **Ajuste de Sherlock y Monro** (Sherlock y Monro, 1993; con el signo de H1): `2θ(z) = 2θ∞ + Σ arg(z − c_k) − Σ arg(z − d_k)`, con tantos deltas como núcleos (en una huella hay tantos núcleos como deltas). Se ajustan por mínimos cuadrados los residuos de `(cos 2θ, sen 2θ)` contra el campo observado, ponderados por la coherencia, en la máscara y fuera de un disco chico alrededor de cada núcleo. Los parámetros son θ∞, los núcleos (a no más de `holgura_nucleo_px` de lo detectado en cada coordenada) y los deltas (libres). Los deltas arrancan en los detectados o, si faltan, en posiciones típicas; se prueban varios arranques y queda el de menor residuo.
7. **Punto de referencia, núcleos y ángulo:**
   - 1 núcleo → `metodo = "nucleo"`, `punto_referencia` = el núcleo ajustado.
   - 2 núcleos → `metodo = "nucleos"`, `punto_referencia` = el punto medio de los dos ajustados.
   - Ninguno → `metodo = "curvatura"`, `punto_referencia` = el centro de la zona de máxima curvatura de las crestas (punto de referencia de FingerCode, Jain et al., 2000), lejos del borde de la máscara.
   - `nucleos` son los ajustados (vacío con `"curvatura"`). `angulo` = θ∞ mod π; sin núcleos, el ajuste se reduce al promedio circular (ángulo doble) del campo en la máscara, ponderado por la coherencia.
   - Todo se calcula sobre la imagen completa, antes de recortar.
8. **Ventana:** `cv2.warpAffine` con `matriz`, sin reescalar; lo que cae fuera de la entrada es fondo y queda fuera de la máscara. La máscara pasa con vecino más cercano. `orientacion` es el campo de la entrada llevado a la ventana (θ − `angulo`).
9. **Realce:** CLAHE (`clahe_limite`, `clahe_grilla`) y filtrado de Gabor sintonizado píxel a píxel con `orientacion` y con la frecuencia de las crestas (Hong, Wan y Jain, 1998), calculado con margen para que el borde de la ventana no deje artefactos. `realzada` va de 0 a 255 con las crestas oscuras y vale 255 fuera de `mascara`.
10. **Binarización y esqueleto:** `binaria` son las crestas dentro de `mascara`; `esqueleto`, el adelgazamiento de `binaria` hasta 1 px, sin bloques de 2 × 2.
11. Determinista: la misma entrada da exactamente las mismas salidas.

**Lámina** (`generar_lamina`): una fila por identidad autorizada y una con un intruso de tipo arco. Cada fila muestra 3 variantes (al menos una con corte, si la identidad tiene alguna en el dataset): la imagen de entrada con el punto de referencia y una línea en la dirección de `angulo`, la ventana realzada y el esqueleto. Las variantes salen de la receta de la sección 4 de H1, sin leer `data/`. **Línea de comandos:** `venv\Scripts\python -m src.huellas.preprocesamiento` escribe la lámina con `config.yaml`.

## 4. Muestra de prueba

Los criterios 5.4 a 5.6 se miden sobre la **muestra**: las entradas de `plan_dataset(cfg)` (con el `cfg` real) de las identidades 1 a 3 (40 variantes cada una) y de los intrusos 1 a 12 (3 variantes cada uno), generadas con la receta de la sección 4 de H1: `base = generar_base(tipo, semilla, cfg)`, `rng = np.random.default_rng([semilla, v])`, `params = sortear_parametros(rng, cfg)` y `variante = aplicar_variante(base, params, rng)`. Por tipo quedan 49 variantes de presilla, de verticilo y de arco en tienda (40 de la identidad y 9 de intrusos) y 9 de arco (intrusos 4, 8 y 12): 156 variantes, de las cuales 38 tienen corte.

"En al menos el 95 %" se cuenta por tipo (o sobre las variantes con corte, cuando se dice). En la muestra son 49 variantes por tipo: pueden fallar a lo sumo 2; de las 38 con corte, a lo sumo 1.

## 5. Criterios que verifican los tests

### 5.1 Configuración

- Las claves de la sección 1 existen, con su tipo y rango (`mediana_px` impar), y `ventana_px` y `periodo_crestas_px` tienen sus valores fijos.
- `cargar_config` lanza `ValueError` que nombra el camino completo si falta cualquiera de las claves de la sección 1.

### 5.2 Formato, color, tamaño y errores (con las bases de las 3 identidades y de un arco)

- `preprocesar` devuelve una `HuellaPreprocesada`: `realzada` `uint8`; `binaria`, `esqueleto` y `mascara` `bool`; `orientacion` `float64` en `[0, π)`; todos de 250 × 250. `metodo` está en `METODOS`, `angulo` en `[0, π)` y `matriz` es `float64` de 2 × 3. `esqueleto` y `binaria` no están vacíos.
- No modifica la imagen de entrada. Determinismo: dos llamadas con la misma imagen dan arrays idénticos (`np.array_equal`) y los mismos valores.
- **Color:** con la imagen en BGR (`cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)`) y en BGRA, el resultado es el de la gris: punto de referencia a ≤ 0,5 px, ángulo a ≤ 0,5° y `mascara` igual en al menos el 99 % de los píxeles. Una huella teñida (canales distintos, por ejemplo `B = gris`, `G = 0,9·gris`, `R = 255 − 0,8·(255 − gris)`, redondeados) da salidas con el formato correcto y no vacías.
- **Lienzo más grande:** la base con un borde de 50 px de fondo (245) de cada lado (500 × 420) da el punto de referencia corrido (50, 50) ± 1 px y el mismo ángulo ± 1°.
- **Recorte:** un recorte de 300 × 260 (alto × ancho) centrado en el punto de referencia de la base da el punto de referencia corrido por el origen del recorte ± 1,5 px y el mismo ángulo ± 1°.
- **Lienzo más chico que la ventana:** un recorte de 200 × 180 centrado en el punto de referencia de la base da salidas de 250 × 250 no vacías, con `mascara` en `False` fuera de lo que viene de la entrada, y el punto de referencia corrido por el origen del recorte ± 2 px.
- `ValueError` con: una imagen uniforme (0, 128 y 245), un `float64`, un array de 1 o 4 dimensiones, una imagen de 2 canales y una de 50 × 300.

### 5.3 Ventana y matriz (en toda la muestra)

- `matriz` es una rotación más una traslación: su parte 2 × 2 es `[[cos a, sen a], [−sen a, cos a]]` con `a = angulo` (tolerancia 1e-9), con determinante 1, y `matriz · (x_r, y_r, 1) = (124.5, 124.5)` (tolerancia 1e-6).
- `metodo = "nucleo"` ⇒ `nucleos` tiene 1 punto y es `punto_referencia`; `"nucleos"` ⇒ 2 puntos, cuyo punto medio es `punto_referencia`; `"curvatura"` ⇒ `nucleos` vacío (tolerancia 1e-6).
- El centro de la ventana está dentro de la huella: `mascara[124:126, 124:126]` vale `True` en los 4 píxeles.
- `mascara` coincide con la máscara real en la ventana: IoU ≥ 0,90 dentro de la entrada, en cada variante.

### 5.4 Punto de referencia y núcleos (muestra)

- Por tipo, en al menos el 95 % de las variantes: `metodo` es `"nucleo"` en la presilla y el arco en tienda, `"nucleos"` en el verticilo y `"curvatura"` en el arco.
- Presilla, verticilo y arco en tienda, por separado: la distancia entre `punto_referencia` y el punto de referencia real es ≤ 8 px en al menos el 95 % de las variantes y ≤ 16 px en todas.
- Cada núcleo real tiene un punto de `nucleos` a ≤ 8 px en al menos el 95 % de las variantes de cada uno de esos tres tipos.
- Arcos (las 9 variantes y las 3 bases): `metodo = "curvatura"`, `nucleos` vacío y `preprocesar` no falla.

### 5.5 Ángulo (muestra)

- Para cada tipo, incluido el arco: |error de ángulo| ≤ 5° en al menos el 95 % de las variantes y ≤ 10° en todas.
- Las variantes con corte (`corte_fraccion > 0`), juntando los cuatro tipos: |error| ≤ 5° en al menos el 95 %.
- En las bases de las 3 identidades (sin variante): |`angulo`| ≤ 2° (llevado a `[−90, 90)`).

### 5.6 Máscara, campo, crestas, esqueleto y realce (muestra)

- **Campo:** en cada variante, la media de cos 2(`orientacion` − θ_v) en la zona de comparación es ≥ 0,95.
- **Crestas:** en cada variante, `binaria` coincide con las crestas verdaderas en al menos el 90 % de la zona de comparación (es lo que muestra que se limpió el ruido).
- **Esqueleto:** está contenido en `binaria` (y por lo tanto en `mascara`), no tiene ningún bloque de 2 × 2, y `esqueleto.sum() / mascara.sum()` está entre 0,06 y 0,16.
- **Realce:** `realzada` vale 255 fuera de `mascara`; dentro, el promedio sobre `~binaria` supera al promedio sobre `binaria` en al menos 60 niveles; y el período medido de `realzada` (sección 0 de H1, recorte central de 192 × 192 de la ventana) está a ≤ 1 px del período real `base.periodo_px · params.escala`.

### 5.7 Estabilidad (las bases de las 3 identidades)

- Para cada identidad y cada `corte_angulo_grados` `a` en {0, 90, 180, 270}: la variante `ParametrosVariante(12.0, 1.0, 10.0, −8.0, 0.0, 0.0, f, a, 0.0, 0.0)`, con `f` el máximo de `corte_periferico` (0,10), aplicada con `np.random.default_rng(0)`, comparada con la base, da una media de cos 2(`orientacion_variante` − `orientacion_base`) ≥ 0,95 en el disco de 80 px de radio alrededor del centro de la ventana, intersecado con las dos máscaras. Su error de ángulo es ≤ 3° y su punto de referencia queda a ≤ 5 px del real.

### 5.8 Lámina

- `generar_lamina(cfg, salida)` escribe en `salida` un PNG que se puede leer con `cv2.imread` y devuelve esa ruta.

### 5.9 Aceptación (`@pytest.mark.aceptacion`)

- Con las 300 imágenes del dataset (la receta de la sección 4 de H1 para todas las entradas de `plan_dataset`, en memoria): los criterios de 5.4 y 5.5 por tipo, con todos los intrusos, y además el arco: el punto de referencia de cada variante queda a ≤ 30 px del de su base llevado con la fórmula afín de H1 en al menos el 80 % de las variantes de arco.
- `preprocesar` de las 300 imágenes tarda menos de 2 minutos en total en la notebook (sin contar la generación).

## 6. Verificación manual (no va en pytest; se anota en el Documento 2)

- En `docs/figuras/lamina_h2.png` las ventanas de una misma identidad se ven en la misma pose (núcleo al centro y crestas lejanas horizontales), también la que tiene corte; los esqueletos siguen las crestas sin manchas del ruido.
- Tiempo de `preprocesar` por imagen en la notebook.

## 7. Aclaraciones para los tests

- **Tiempo:** `preprocesar` tarda alrededor de 0,15 s por imagen. La muestra tiene 15 bases y 156 variantes: generarlas y preprocesarlas una sola vez (fixtures con `scope="module"`) lleva menos de un minuto.
- Lo que se compara con `Variante` (`punto_referencia`, `singularidades`, `mascara`, `mapa_x`, `mapa_y`, `parametros`) viene de H1, que ya está probado; es el mismo valor que `metadatos.csv`.
- Las singularidades se comparan por cercanía, sin exigir orden.
- "Fondo" es 245, el gris de fondo de H1.
- **Recortes de 5.2:** el origen es `(round(x_r) − ancho_recorte // 2, round(y_r) − alto_recorte // 2)`, corrido lo necesario para que el recorte entre en la imagen de 400 × 320 (con las bases del dataset entra sin correrlo). El punto de referencia esperado es el de la base menos ese origen.
