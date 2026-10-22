# Contrato · Fase 0 · Setup del proyecto

**Versión 1.1 · 22/10/2026** (la 1 es del 16/10/2026). **Cambio en la 1.1:** `huellas.variantes.corte_periferico` pasa de `[0.15, 0.20]` a `[0.05, 0.10]`, porque el profesor pidió el 22/10 que a cada huella se le saque como mucho el 10 % de su área. En `tests/test_setup.py` cambia solo ese valor; el cambio está anotado en el Documento 2.

Es lo que verifica `tests/test_setup.py`. Los tests se escriben en otro chat solo a partir de este archivo y se congelan (commit) antes de implementar. Si algo de acá cambia, se cambia de forma explícita y se anota en el Documento 2.

**Raíz del repo:** la carpeta que contiene `config.yaml`. En los tests: `RAIZ = Path(__file__).resolve().parents[1]`.

## 1. Entorno

- Python 3.12 (`sys.version_info[:2] == (3, 12)`), corriendo dentro de un entorno virtual (`sys.prefix != sys.base_prefix`).
- Se importan sin error: `numpy`, `scipy`, `cv2`, `skimage`, `librosa`, `sklearn`, `soundfile`, `sounddevice`, `customtkinter`, `matplotlib`, `joblib`, `yaml`, `edge_tts`.
- `soundfile` maneja los formatos de los audios que van a llegar:
  - `"OPUS" in soundfile.available_subtypes("OGG")` (notas de voz de WhatsApp).
  - `"MPEG_LAYER_III" in soundfile.available_subtypes("MP3")` (voces sintéticas de edge-tts).
  - Ida y vuelta: 1 s de un seno de 440 Hz a 48 kHz, mono, guardado como OGG/OPUS en una carpeta temporal y leído de nuevo, da 48000 Hz, 1 canal y una duración entre 0,95 y 1,05 s.

## 2. Carpetas y archivos obligatorios

- **Carpetas:** `src/`, `src/core_ml/`, `src/huellas/`, `src/voz/`, `src/agente/`, `src/app/`, `tests/`, `notebooks/`, `data/`, `models/`, `docs/`, `docs/contratos/`.
- **Archivos:** `README.md`, `requirements.txt`, `config.yaml`, `.gitignore`, `pyproject.toml`, `src/__init__.py`, `src/config.py`, un `__init__.py` en cada una de las 5 subcarpetas de `src/`, `data/.gitkeep`, `models/.gitkeep`, `notebooks/.gitkeep` y `docs/contratos/fase0_setup.md` (este archivo).
- **Paquetes importables** (con la raíz en `sys.path`): `src`, `src.core_ml`, `src.huellas`, `src.voz`, `src.agente`, `src.app` y `src.config`.

## 3. requirements.txt

- Cada línea no vacía que no empieza con `#` tiene la forma `paquete==versión` (se admite `paquete[extra]==versión`). Nada de `>=`, `<=`, `~=`, `<`, `>`, `!=`, URLs, `-e` ni `-r`.
- Sin paquetes repetidos, comparando los nombres normalizados (minúsculas y `_` o `.` como `-`).
- Incluye al menos: `numpy`, `scipy`, `opencv-python`, `scikit-image`, `scikit-learn`, `librosa`, `soundfile`, `sounddevice`, `customtkinter`, `matplotlib`, `joblib`, `pyyaml`, `pytest`, `edge-tts`, `ipykernel`.
- Cada paquete fijado está instalado en el entorno que corre los tests con exactamente esa versión (`importlib.metadata.version`).

## 4. .gitignore

Se verifica con `git check-ignore --no-index -q <ruta>` corrido en la raíz: código 0 = ignorada, 1 = no ignorada. Las rutas no necesitan existir. Si `git` no está instalado o la raíz no es un repo de git, estos tests se saltean con `pytest.skip`.

- **Ignoradas:** `venv/pyvenv.cfg`, `venv/Lib/site-packages/numpy/__init__.py`, `src/__pycache__/config.cpython-312.pyc`, `tests/__pycache__/test_setup.cpython-312-pytest-9.1.1.pyc`, `data/huellas/train/id1_v001.png`, `data/huellas/metadatos.csv`, `data/voz/raw_opus/p01_reconocer.opus`, `models/kmeans_huellas.joblib`, `.pytest_cache/README.md`, `notebooks/.ipynb_checkpoints/exploracion-checkpoint.ipynb`.
- **No ignoradas:** `data/.gitkeep`, `models/.gitkeep`, `notebooks/.gitkeep`, `notebooks/exploracion.ipynb`, `config.yaml`, `requirements.txt`, `pyproject.toml`, `README.md`, `src/config.py`, `tests/test_setup.py`, `docs/contratos/fase0_setup.md`.

## 5. pyproject.toml

En la sección `[tool.pytest.ini_options]` (se lee con `tomllib`):

- `pythonpath` es una lista que contiene `"."`.
- `testpaths == ["tests"]`.
- `markers` tiene una entrada que empieza con `"aceptacion"` (tests lentos sobre el dataset completo).
- `addopts` contiene `-m` y `not aceptacion`: `pytest` a secas no corre los tests de aceptación; se corren con `pytest -m aceptacion`.

## 6. README.md

Contiene los títulos `## Instalación`, `## Tests`, `## Estructura del proyecto` y `## Licencias y créditos`, y los textos `-m venv venv`, `pip install -r requirements.txt` y `pytest`.

## 7. config.yaml

- Está en UTF-8 y `yaml.safe_load` devuelve un mapeo.
- **Claves obligatorias:** todas las hojas del esqueleto de abajo, con el camino separado por puntos (ej. `huellas.particion.train`). Una lista cuenta como hoja: `huellas.identidades` es una hoja. Puede haber claves de más.
- Los valores marcados `fijo` se comparan exactos; el resto, por tipo y rango. "Entero" quiere decir `int` que no sea `bool`.

```yaml
semilla_global: <entero>

rutas:
  datos: data                        # fijo
  huellas: data/huellas              # fijo
  voz: data/voz                      # fijo
  modelos: models                    # fijo
  figuras: docs/figuras              # fijo

huellas:
  identidades:                       # fijo salvo las semillas: 3 elementos en este orden
    - {id: 1, nombre: "Identidad Sintética #1", tipo: presilla, semilla: <entero>}
    - {id: 2, nombre: "Identidad Sintética #2", tipo: verticilo, semilla: <entero>}
    - {id: 3, nombre: "Identidad Sintética #3", tipo: arco_tienda, semilla: <entero>}
  variantes_por_identidad: 40        # fijo
  particion: {train: 24, val: 8, test: 8}   # fijo
  reservadas_demo: 2                 # fijo (salen de las 8 de test)
  intrusos:
    cantidad: 60                     # fijo
    variantes_por_intruso: 3         # fijo
    semilla_base: <entero>           # el intruso i (1 a 60) usa semilla_base + i
    val: 30                          # fijo
    test: 30                         # fijo
  variantes:
    rotacion_max_grados: 15          # fijo
    traslacion_max_px: 15            # fijo
    escala_max: 0.08                 # fijo
    corte_periferico: [0.05, 0.10]   # fijo (1.1; antes [0.15, 0.20])
  ventana_px: [250, 250]             # fijo
  periodo_crestas_px: [8, 10]        # fijo
  kmeans:
    k: 3                             # fijo
    n_init: 15                       # fijo
    max_iter: <entero mayor que 0>
    tol: <float mayor que 0>
  umbral:
    criterio: <"eer" o "far_max">
    far_max_val: <float en (0, 1)>   # se usa si criterio = far_max

voz:
  frecuencia_muestreo: 16000         # fijo
  comandos: [reconocer, identificar, salir]   # fijo
  clase_rechazo: otro                # fijo
  duracion_palabra_s: [0.3, 2.5]     # fijo
  push_to_talk_max_s: 3.0            # fijo
  pasabanda_hz: [80, 4000]           # fijo
  preenfasis: 0.97                   # fijo
  mfcc:
    n_mfcc: 13                       # fijo
    n_mels: <entero entre 26 y 40>
    fmax: 4000                       # fijo
    ventana_ms: 25                   # fijo
    avance_ms: 10                    # fijo
  tramos: 5                          # fijo
  knn:
    k_candidatos: [3, 5, 7, 9]       # fijo
    eps: <float en (0, 0.001]>
    confianza_min: 0.60              # fijo
    pca_componentes_candidatas: <lista no vacía de enteros mayores que 1>

metas:
  identificacion_min: 0.95           # fijo
  far_max: 0.05                      # fijo
  frr_max: 0.10                      # fijo
  silhouette_min: 0.50               # fijo
  voz_acierto_min: 0.90              # fijo
  voz_otro_aceptado_max: 0.05        # fijo
```

**Reglas de consistencia:**

- `particion.train + particion.val + particion.test == variantes_por_identidad` y `reservadas_demo <= particion.test`.
- `intrusos.val + intrusos.test == intrusos.cantidad`.
- Las 3 semillas de las identidades y las 60 de los intrusos (`semilla_base + i`, con i de 1 a 60) son todas distintas.
- `voz.mfcc.fmax <= voz.pasabanda_hz[1]` y `huellas.periodo_crestas_px[0] < huellas.periodo_crestas_px[1]`.
- **Rutas:** relativas (sin letra de unidad ni `/` inicial), con `/` como separador (sin `\`) y sin `..`. `rutas.huellas` y `rutas.voz` empiezan con `rutas.datos + "/"`.

## 8. src/config.py

```python
RAIZ: pathlib.Path
    # Raíz del repo (la carpeta de config.yaml), calculada desde la ubicación de
    # config.py: no depende del directorio de trabajo.

def cargar_config(ruta: str | os.PathLike | None = None) -> dict:
    # Lee el YAML (en UTF-8) y devuelve su contenido como dict.
    # - ruta None → RAIZ / "config.yaml".
    # - FileNotFoundError si el archivo no existe.
    # - ValueError si el contenido no es un mapeo (archivo vacío, lista, número).
    # - ValueError si falta alguna clave obligatoria de la sección 7; el mensaje incluye
    #   el camino completo de cada hoja que falta, ej. "huellas.particion.train".
    #   Si falta una sección entera, faltan todas sus hojas.
    # - Las claves de más no son error.
    # - No modifica el archivo y cada llamada devuelve un dict nuevo.
```

## 9. Verificación manual (no va en pytest; se anota en el Documento 2)

- Clonar el repo en otra carpeta y seguir solo el README: `pytest` pasa.
- VS Code toma el venv, el panel Testing encuentra los tests y un notebook corre con el kernel del venv.
- Se abre y se cierra una ventana de prueba de CustomTkinter con un gráfico de Matplotlib.
- `sounddevice` lista el micrófono de la notebook.
