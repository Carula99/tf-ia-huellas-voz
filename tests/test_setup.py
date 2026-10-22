"""Tests de la Fase 0 (setup del proyecto).

Verifican las secciones 1 a 8 de docs/contratos/fase0_setup.md (versión 1,
16/10/2026). La sección 9 es manual y se anota en el Documento 2.

Interpretaciones acordadas donde el contrato admitía más de una lectura:
- §3: cada línea útil es exactamente `paquete==versión` (o `paquete[extra]==versión`),
  sin espacios, comentarios al final ni markers. La versión instalada se compara
  como texto exacto.
- §4: se aísla el gitignore global del usuario (core.excludesFile) para que solo
  cuenten los ignores del propio repo.
- §5: en addopts, `-m` va seguido de `not aceptacion` (addopts puede ser texto o lista).
- §6: cada título es una línea completa del README.
- §7: los valores `fijo` se comparan por tipo y valor (3 != 3.0, 40.0 != 40) y los
  `<float ...>` exigen un float de Python (un int no cuenta). "Entero" es un int
  que no es bool. "Entre 26 y 40" incluye los extremos. `..` se busca como
  componente de la ruta. far_max_val se valida aunque el criterio sea "eer".
- §7: cada elemento de huellas.identidades puede tener claves de más; se exigen
  id, nombre y tipo exactos y semilla entera.
- §8: "un dict nuevo" quiere decir independiente: dos llamadas no comparten ningún
  dict ni lista, y modificar lo devuelto no afecta a las llamadas siguientes.

Las librerías del entorno y src.config se importan dentro de cada test: si algo
todavía no existe, fallan esos tests y no la colección entera.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import unicodedata
from importlib import metadata
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def importar(nombre: str):
    """Importa un módulo; si no se puede, el test falla diciendo por qué."""
    try:
        return importlib.import_module(nombre)
    except Exception as error:  # ImportError, OSError (p. ej. falta PortAudio), etc.
        motivo = f"{type(error).__name__}: {error}"
    pytest.fail(f"No se pudo importar {nombre!r}: {motivo}", pytrace=False)


def leer_texto(relativa: str) -> str:
    """Lee un archivo de texto del repo en UTF-8."""
    ruta = RAIZ / relativa
    if not ruta.is_file():
        pytest.fail(f"No existe {relativa} en la raíz del repo ({RAIZ}).", pytrace=False)
    try:
        return ruta.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        motivo = str(error)
    pista = ""
    if ruta.read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff"):
        pista = (
            " Parece UTF-16: es lo que genera `>` en Windows PowerShell 5"
            " (p. ej. `pip freeze > requirements.txt`)."
        )
    pytest.fail(f"{relativa} no está en UTF-8 ({motivo}).{pista}", pytrace=False)


def tipo_de(valor) -> str:
    if isinstance(valor, list):
        return "list[" + ", ".join(tipo_de(v) for v in valor) + "]"
    return type(valor).__name__


def es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def es_float(valor) -> bool:
    return isinstance(valor, float)


def es_numero(valor) -> bool:
    return es_entero(valor) or es_float(valor)


def iguales_exactos(a, b) -> bool:
    """Igualdad de valor y de tipo, recursiva (3 != 3.0, True != 1)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(iguales_exactos(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(iguales_exactos(a[k], b[k]) for k in a)
    return a == b


def pista_numero_como_texto(valor) -> str:
    patron = r"\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*"
    if isinstance(valor, str) and re.fullmatch(patron, valor):
        return (
            " PyYAML lo leyó como texto: ¿está entre comillas, o es notación"
            " científica sin punto decimal? (1e-4 es texto; 1.0e-4 es float)"
        )
    return ""


def obtener(cfg: dict, camino: str):
    """Valor de una clave con camino separado por puntos; falla si no está."""
    actual = cfg
    for parte in camino.split("."):
        if not isinstance(actual, dict) or parte not in actual:
            pytest.fail(f"config.yaml no tiene la clave obligatoria {camino!r}.", pytrace=False)
        actual = actual[parte]
    return actual


def numero(cfg: dict, camino: str):
    valor = obtener(cfg, camino)
    if not es_numero(valor):
        pytest.fail(
            f"{camino} tiene que ser un número; hay {valor!r} ({tipo_de(valor)}).",
            pytrace=False,
        )
    return valor


def par_de_numeros(cfg: dict, camino: str) -> list:
    valor = obtener(cfg, camino)
    if not (isinstance(valor, list) and len(valor) == 2 and all(es_numero(v) for v in valor)):
        pytest.fail(
            f"{camino} tiene que ser una lista de 2 números; hay {valor!r} ({tipo_de(valor)}).",
            pytrace=False,
        )
    return valor


def cargar_config_real() -> dict:
    """config.yaml leído con yaml.safe_load, sin pasar por src.config."""
    yaml = importar("yaml")
    datos = yaml.safe_load(leer_texto("config.yaml"))
    if not isinstance(datos, dict):
        pytest.fail(
            f"yaml.safe_load(config.yaml) tiene que devolver un mapeo y devolvió {tipo_de(datos)}.",
            pytrace=False,
        )
    return datos


def escribir_yaml(ruta: Path, datos) -> Path:
    yaml = importar("yaml")
    with open(ruta, "w", encoding="utf-8") as archivo:
        yaml.safe_dump(datos, archivo, allow_unicode=True, sort_keys=False)
    return ruta


def sin_clave(cfg: dict, camino: str) -> dict:
    """Copia profunda de cfg sin la clave `camino` (hoja o sección)."""
    copia = copy.deepcopy(cfg)
    *padres, ultima = camino.split(".")
    actual = copia
    for parte in padres:
        actual = actual.get(parte) if isinstance(actual, dict) else None
    if not isinstance(actual, dict) or ultima not in actual:
        pytest.fail(
            f"El config.yaml real no tiene {camino!r}: no se puede armar la copia sin esa clave.",
            pytrace=False,
        )
    del actual[ultima]
    return copia


def nombra(mensaje: str, camino: str) -> bool:
    """True si `camino` aparece completo en el mensaje (no como parte de otro camino)."""
    return re.search(r"(?<![\w.])" + re.escape(camino) + r"(?!\w)", mensaje) is not None


def lanza(tipo: type[BaseException], funcion, *args, que: str) -> BaseException:
    """Llama a funcion(*args) y devuelve la excepción `tipo`; falla si no se lanza.

    Una excepción de otro tipo se deja pasar para que se vea su traceback.
    """
    try:
        funcion(*args)
    except tipo as error:
        return error
    pytest.fail(f"{que}: se esperaba {tipo.__name__} y no se lanzó ninguna excepción.", pytrace=False)


def ids_contenedores(objeto, acumulado: set[int] | None = None) -> set[int]:
    """ids de todos los dicts y listas dentro de `objeto` (incluido él mismo)."""
    acumulado = set() if acumulado is None else acumulado
    if isinstance(objeto, (dict, list)):
        acumulado.add(id(objeto))
        for hijo in objeto.values() if isinstance(objeto, dict) else objeto:
            ids_contenedores(hijo, acumulado)
    return acumulado


def ensuciar(objeto) -> None:
    """Modifica en el lugar cada dict y lista dentro de `objeto`."""
    if isinstance(objeto, dict):
        for hijo in list(objeto.values()):
            ensuciar(hijo)
        objeto["__modificado_por_el_test__"] = True
    elif isinstance(objeto, list):
        for hijo in list(objeto):
            ensuciar(hijo)
        objeto.append("__modificado_por_el_test__")


class RutaPropia:
    """Un os.PathLike que no es pathlib.Path."""

    def __init__(self, ruta) -> None:
        self._ruta = os.fspath(ruta)

    def __fspath__(self) -> str:
        return self._ruta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_real() -> dict:
    return cargar_config_real()


@pytest.fixture
def modulo_config(monkeypatch):
    """src.config importado con la raíz del repo en sys.path."""
    monkeypatch.syspath_prepend(str(RAIZ))
    try:
        return importlib.import_module("src.config")
    except ModuleNotFoundError as error:
        motivo = str(error)
    pytest.fail(
        f"No se pudo importar src.config: {motivo}. ¿Existen src/__init__.py y src/config.py?",
        pytrace=False,
    )


@pytest.fixture
def cargar_config(modulo_config):
    funcion = getattr(modulo_config, "cargar_config", None)
    if not callable(funcion):
        pytest.fail("src/config.py no define la función cargar_config.", pytrace=False)
    return funcion


@pytest.fixture
def copia_config(tmp_path) -> Path:
    """Copia byte a byte del config.yaml real, en tmp_path."""
    origen = RAIZ / "config.yaml"
    if not origen.is_file():
        pytest.fail(f"No existe config.yaml en la raíz del repo ({RAIZ}).", pytrace=False)
    destino = tmp_path / "config.yaml"
    shutil.copyfile(origen, destino)
    return destino


# ---------------------------------------------------------------------------
# 1. Entorno
# ---------------------------------------------------------------------------

LIBRERIAS = [
    "numpy",
    "scipy",
    "cv2",
    "skimage",
    "librosa",
    "sklearn",
    "soundfile",
    "sounddevice",
    "customtkinter",
    "matplotlib",
    "joblib",
    "yaml",
    "edge_tts",
]


def test_python_es_version_3_12():
    assert sys.version_info[:2] == (3, 12), (
        f"Se esperaba Python 3.12 y corre {sys.version.split()[0]} ({sys.executable})."
    )


def test_corre_dentro_de_un_entorno_virtual():
    assert sys.prefix != sys.base_prefix, (
        f"pytest no corre dentro de un entorno virtual (sys.prefix == sys.base_prefix == "
        f"{sys.prefix}). Activá el venv o elegí su intérprete en VS Code."
    )


@pytest.mark.parametrize("modulo", LIBRERIAS)
def test_libreria_del_entorno_se_importa(modulo):
    importar(modulo)


def test_soundfile_soporta_ogg_opus():
    sf = importar("soundfile")
    subtipos = sf.available_subtypes("OGG")
    assert "OPUS" in subtipos, (
        f"soundfile no maneja OGG/OPUS (notas de voz de WhatsApp). Subtipos de OGG: "
        f"{sorted(subtipos)}; libsndfile {sf.__libsndfile_version__}."
    )


def test_soundfile_soporta_mp3():
    sf = importar("soundfile")
    subtipos = sf.available_subtypes("MP3")
    assert "MPEG_LAYER_III" in subtipos, (
        f"soundfile no maneja MP3 (voces sintéticas de edge-tts). Subtipos de MP3: "
        f"{sorted(subtipos)}; libsndfile {sf.__libsndfile_version__}."
    )


def test_soundfile_ida_y_vuelta_ogg_opus_conserva_formato_y_duracion(tmp_path):
    sf = importar("soundfile")
    np = importar("numpy")
    fs = 48_000
    t = np.arange(fs) / fs
    seno = 0.5 * np.sin(2 * np.pi * 440 * t)
    ruta = tmp_path / "seno_440hz.ogg"

    sf.write(str(ruta), seno, fs, format="OGG", subtype="OPUS")
    datos, fs_leida = sf.read(str(ruta), always_2d=True)

    canales = datos.shape[1]
    duracion = datos.shape[0] / fs_leida if fs_leida else float("nan")
    problemas = []
    if fs_leida != 48_000:
        problemas.append(f"frecuencia de muestreo {fs_leida} Hz (se esperaba 48000)")
    if canales != 1:
        problemas.append(f"{canales} canales (se esperaba 1)")
    if not 0.95 <= duracion <= 1.05:
        problemas.append(f"duración {duracion:.4f} s (se esperaba entre 0.95 y 1.05)")
    assert not problemas, "Ida y vuelta OGG/OPUS de 1 s a 48 kHz: " + "; ".join(problemas) + "."


# ---------------------------------------------------------------------------
# 2. Carpetas y archivos obligatorios
# ---------------------------------------------------------------------------

SUBPAQUETES = ["core_ml", "huellas", "voz", "agente", "app"]

CARPETAS = [
    "src",
    *[f"src/{p}" for p in SUBPAQUETES],
    "tests",
    "notebooks",
    "data",
    "models",
    "docs",
    "docs/contratos",
]

ARCHIVOS = [
    "README.md",
    "requirements.txt",
    "config.yaml",
    ".gitignore",
    "pyproject.toml",
    "src/__init__.py",
    "src/config.py",
    *[f"src/{p}/__init__.py" for p in SUBPAQUETES],
    "data/.gitkeep",
    "models/.gitkeep",
    "notebooks/.gitkeep",
    "docs/contratos/fase0_setup.md",
]

PAQUETES = ["src", *[f"src.{p}" for p in SUBPAQUETES], "src.config"]


@pytest.mark.parametrize("carpeta", CARPETAS)
def test_existe_carpeta_obligatoria(carpeta):
    assert (RAIZ / carpeta).is_dir(), f"Falta la carpeta {carpeta}/ en {RAIZ}."


@pytest.mark.parametrize("archivo", ARCHIVOS)
def test_existe_archivo_obligatorio(archivo):
    assert (RAIZ / archivo).is_file(), f"Falta el archivo {archivo} en {RAIZ}."


@pytest.mark.parametrize("nombre", PAQUETES)
def test_paquete_de_src_es_importable(monkeypatch, nombre):
    monkeypatch.syspath_prepend(str(RAIZ))
    modulo, motivo = None, ""
    try:
        modulo = importlib.import_module(nombre)
    except ModuleNotFoundError as error:
        motivo = str(error)
    if modulo is None:
        pytest.fail(f"No se pudo importar {nombre} con la raíz en sys.path: {motivo}.", pytrace=False)
    archivo = getattr(modulo, "__file__", None)
    assert archivo is not None, (
        f"{nombre} se importó como paquete de espacio de nombres: le falta el __init__.py."
    )
    assert Path(archivo).resolve().is_relative_to(RAIZ / "src"), (
        f"{nombre} se importó desde {archivo}, fuera de {RAIZ / 'src'}."
    )


# ---------------------------------------------------------------------------
# 3. requirements.txt
# ---------------------------------------------------------------------------

REQUISITOS_MINIMOS = [
    "numpy",
    "scipy",
    "opencv-python",
    "scikit-image",
    "scikit-learn",
    "librosa",
    "soundfile",
    "sounddevice",
    "customtkinter",
    "matplotlib",
    "joblib",
    "pyyaml",
    "pytest",
    "edge-tts",
    "ipykernel",
]

_NOMBRE = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
PATRON_REQUISITO = re.compile(
    rf"(?P<nombre>{_NOMBRE})"
    rf"(?:\[{_NOMBRE}(?:,{_NOMBRE})*\])?"
    r"==(?P<version>[A-Za-z0-9][A-Za-z0-9.+!_-]*)"
)


def normalizar(nombre: str) -> str:
    return re.sub(r"[-_.]+", "-", nombre).lower()


def lineas_requirements() -> list[tuple[int, str]]:
    """(número, texto) de cada línea de requirements.txt que no es vacía ni comentario."""
    lineas = []
    for numero_linea, linea in enumerate(leer_texto("requirements.txt").splitlines(), start=1):
        limpia = linea.strip()
        if limpia and not limpia.startswith("#"):
            lineas.append((numero_linea, limpia))
    return lineas


def nombre_de_linea(linea: str) -> str | None:
    """Nombre del paquete al comienzo de la línea, aunque la línea esté mal formada."""
    encontrado = re.match(_NOMBRE, linea.lstrip("﻿"))
    return encontrado.group(0) if encontrado else None


def test_requirements_cada_linea_fija_una_version_exacta():
    malas = []
    for numero_linea, linea in lineas_requirements():
        if not PATRON_REQUISITO.fullmatch(linea):
            pista = " (empieza con BOM: guardalo como UTF-8 sin BOM)" if linea.startswith("﻿") else ""
            malas.append(f"  línea {numero_linea}: {linea!r}{pista}")
    assert not malas, (
        "requirements.txt tiene líneas que no son `paquete==versión`:\n" + "\n".join(malas)
    )


def test_requirements_no_repite_paquetes():
    lineas_por_nombre: dict[str, list[int]] = {}
    for numero_linea, linea in lineas_requirements():
        nombre = nombre_de_linea(linea)
        if nombre:
            lineas_por_nombre.setdefault(normalizar(nombre), []).append(numero_linea)
    repetidos = {n: ls for n, ls in lineas_por_nombre.items() if len(ls) > 1}
    assert not repetidos, (
        "requirements.txt repite paquetes: "
        + ", ".join(f"{n} (líneas {ls})" for n, ls in repetidos.items())
    )


@pytest.mark.parametrize("paquete", REQUISITOS_MINIMOS)
def test_requirements_incluye_paquete(paquete):
    nombres = {normalizar(n) for _, linea in lineas_requirements() if (n := nombre_de_linea(linea))}
    assert normalizar(paquete) in nombres, f"requirements.txt no incluye {paquete}."


def test_requirements_versiones_fijadas_estan_instaladas():
    problemas = []
    for _, linea in lineas_requirements():
        requisito = PATRON_REQUISITO.fullmatch(linea)
        if not requisito:
            continue  # lo reporta test_requirements_cada_linea_fija_una_version_exacta
        nombre, version = requisito["nombre"], requisito["version"]
        try:
            instalada = metadata.version(nombre)
        except metadata.PackageNotFoundError:
            problemas.append(f"  {nombre}=={version}: no está instalado")
            continue
        if instalada != version:
            problemas.append(f"  {nombre}=={version}: está instalada la versión {instalada}")
    assert not problemas, (
        f"El entorno que corre los tests ({sys.prefix}) no coincide con requirements.txt:\n"
        + "\n".join(problemas)
    )


# ---------------------------------------------------------------------------
# 4. .gitignore
# ---------------------------------------------------------------------------

RUTAS_IGNORADAS = [
    "venv/pyvenv.cfg",
    "venv/Lib/site-packages/numpy/__init__.py",
    "src/__pycache__/config.cpython-312.pyc",
    "tests/__pycache__/test_setup.cpython-312-pytest-9.1.1.pyc",
    "data/huellas/train/id1_v001.png",
    "data/huellas/metadatos.csv",
    "data/voz/raw_opus/p01_reconocer.opus",
    "models/kmeans_huellas.joblib",
    ".pytest_cache/README.md",
    "notebooks/.ipynb_checkpoints/exploracion-checkpoint.ipynb",
]

RUTAS_NO_IGNORADAS = [
    "data/.gitkeep",
    "models/.gitkeep",
    "notebooks/.gitkeep",
    "notebooks/exploracion.ipynb",
    "config.yaml",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "src/config.py",
    "tests/test_setup.py",
    "docs/contratos/fase0_setup.md",
]


@pytest.fixture(scope="module")
def git_ignora(tmp_path_factory):
    """Función ruta -> bool según `git check-ignore --no-index -q`, corrido en la raíz.

    Se reemplaza core.excludesFile por un archivo vacío para que el gitignore
    global del usuario no influya: solo cuentan los ignores del repo.
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("git no está instalado: se saltean los tests del .gitignore.")
    if not (RAIZ / ".git").exists():
        pytest.skip(f"{RAIZ} no es un repo de git: se saltean los tests del .gitignore.")
    excludes_vacio = tmp_path_factory.mktemp("git") / "excludes_vacio"
    excludes_vacio.write_text("", encoding="utf-8")

    def ignora(ruta: str) -> bool:
        resultado = subprocess.run(
            [
                git,
                "-c",
                f"core.excludesFile={excludes_vacio.as_posix()}",
                "check-ignore",
                "--no-index",
                "-q",
                ruta,
            ],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if resultado.returncode not in (0, 1):
            pytest.fail(
                f"`git check-ignore --no-index -q {ruta}` terminó con código "
                f"{resultado.returncode}: {resultado.stderr.strip()}",
                pytrace=False,
            )
        return resultado.returncode == 0

    return ignora


@pytest.mark.parametrize("ruta", RUTAS_IGNORADAS)
def test_gitignore_ignora_ruta(git_ignora, ruta):
    assert git_ignora(ruta), f".gitignore no ignora {ruta} y debería."


@pytest.mark.parametrize("ruta", RUTAS_NO_IGNORADAS)
def test_gitignore_no_ignora_ruta(git_ignora, ruta):
    assert not git_ignora(ruta), f".gitignore ignora {ruta} y no debería."


# ---------------------------------------------------------------------------
# 5. pyproject.toml
# ---------------------------------------------------------------------------


def opciones_pytest() -> dict:
    tomllib = importar("tomllib")
    try:
        datos = tomllib.loads(leer_texto("pyproject.toml"))
    except tomllib.TOMLDecodeError as error:
        pytest.fail(f"pyproject.toml no es TOML válido: {error}.", pytrace=False)
    opciones = datos.get("tool", {}).get("pytest", {}).get("ini_options")
    if not isinstance(opciones, dict):
        pytest.fail("pyproject.toml no tiene la sección [tool.pytest.ini_options].", pytrace=False)
    return opciones


def test_pyproject_pythonpath_es_lista_con_punto():
    valor = opciones_pytest().get("pythonpath")
    assert isinstance(valor, list) and "." in valor, (
        f'pythonpath tiene que ser una lista que contenga "."; es {valor!r}.'
    )


def test_pyproject_testpaths_es_solo_tests():
    valor = opciones_pytest().get("testpaths")
    assert valor == ["tests"], f'testpaths tiene que ser ["tests"]; es {valor!r}.'


def test_pyproject_declara_marcador_aceptacion():
    marcadores = opciones_pytest().get("markers")
    assert isinstance(marcadores, list), f"markers tiene que ser una lista; es {marcadores!r}."
    nombres = [re.match(r"\s*([^:(\s]*)", m).group(1) for m in marcadores if isinstance(m, str)]
    assert "aceptacion" in nombres, (
        f'markers no tiene una entrada "aceptacion: ..."; tiene {marcadores!r}.'
    )


def test_pyproject_addopts_excluye_aceptacion_por_defecto():
    addopts = opciones_pytest().get("addopts")
    if isinstance(addopts, str):
        try:
            argumentos = shlex.split(addopts)
        except ValueError as error:
            pytest.fail(f"addopts no se puede separar en argumentos ({error}): {addopts!r}.", pytrace=False)
    elif isinstance(addopts, list) and all(isinstance(a, str) for a in addopts):
        argumentos = addopts
    else:
        pytest.fail(f"addopts tiene que ser texto o lista de textos; es {addopts!r}.", pytrace=False)
    tiene = any(
        argumentos[i] == "-m" and argumentos[i + 1] == "not aceptacion"
        for i in range(len(argumentos) - 1)
    )
    assert tiene, f'addopts tiene que incluir -m "not aceptacion"; tiene {argumentos!r}.'


# ---------------------------------------------------------------------------
# 6. README.md
# ---------------------------------------------------------------------------

TITULOS_README = {
    "instalacion": "## Instalación",
    "tests": "## Tests",
    "estructura": "## Estructura del proyecto",
    "licencias": "## Licencias y créditos",
}

TEXTOS_README = ["-m venv venv", "pip install -r requirements.txt", "pytest"]


def leer_readme() -> str:
    return unicodedata.normalize("NFC", leer_texto("README.md"))


@pytest.mark.parametrize("titulo", list(TITULOS_README.values()), ids=list(TITULOS_README))
def test_readme_tiene_titulo(titulo):
    lineas = [linea.rstrip() for linea in leer_readme().splitlines()]
    assert unicodedata.normalize("NFC", titulo) in lineas, (
        f"README.md no tiene la línea de título {titulo!r}."
    )


@pytest.mark.parametrize("texto", TEXTOS_README)
def test_readme_contiene_texto(texto):
    assert texto in leer_readme(), f"README.md no contiene {texto!r}."


# ---------------------------------------------------------------------------
# 7. config.yaml
# ---------------------------------------------------------------------------

# Hojas del esqueleto de la sección 7 (una lista cuenta como hoja).
CLAVES_OBLIGATORIAS = [
    "semilla_global",
    "rutas.datos",
    "rutas.huellas",
    "rutas.voz",
    "rutas.modelos",
    "rutas.figuras",
    "huellas.identidades",
    "huellas.variantes_por_identidad",
    "huellas.particion.train",
    "huellas.particion.val",
    "huellas.particion.test",
    "huellas.reservadas_demo",
    "huellas.intrusos.cantidad",
    "huellas.intrusos.variantes_por_intruso",
    "huellas.intrusos.semilla_base",
    "huellas.intrusos.val",
    "huellas.intrusos.test",
    "huellas.variantes.rotacion_max_grados",
    "huellas.variantes.traslacion_max_px",
    "huellas.variantes.escala_max",
    "huellas.variantes.corte_periferico",
    "huellas.ventana_px",
    "huellas.periodo_crestas_px",
    "huellas.kmeans.k",
    "huellas.kmeans.n_init",
    "huellas.kmeans.max_iter",
    "huellas.kmeans.tol",
    "huellas.umbral.criterio",
    "huellas.umbral.far_max_val",
    "voz.frecuencia_muestreo",
    "voz.comandos",
    "voz.clase_rechazo",
    "voz.duracion_palabra_s",
    "voz.push_to_talk_max_s",
    "voz.pasabanda_hz",
    "voz.preenfasis",
    "voz.mfcc.n_mfcc",
    "voz.mfcc.n_mels",
    "voz.mfcc.fmax",
    "voz.mfcc.ventana_ms",
    "voz.mfcc.avance_ms",
    "voz.tramos",
    "voz.knn.k_candidatos",
    "voz.knn.eps",
    "voz.knn.confianza_min",
    "voz.knn.pca_componentes_candidatas",
    "metas.identificacion_min",
    "metas.far_max",
    "metas.frr_max",
    "metas.silhouette_min",
    "metas.voz_acierto_min",
    "metas.voz_otro_aceptado_max",
]

# Secciones intermedias del esqueleto: "huellas", "huellas.particion", "voz.mfcc", etc.
SECCIONES = sorted(
    {".".join(c.split(".")[:i]) for c in CLAVES_OBLIGATORIAS for i in range(1, c.count(".") + 1)}
)

# Valores `fijo` (se comparan por tipo y valor). Las identidades van aparte.
VALORES_FIJOS = {
    "rutas.datos": "data",
    "rutas.huellas": "data/huellas",
    "rutas.voz": "data/voz",
    "rutas.modelos": "models",
    "rutas.figuras": "docs/figuras",
    "huellas.variantes_por_identidad": 40,
    "huellas.particion.train": 24,
    "huellas.particion.val": 8,
    "huellas.particion.test": 8,
    "huellas.reservadas_demo": 2,
    "huellas.intrusos.cantidad": 60,
    "huellas.intrusos.variantes_por_intruso": 3,
    "huellas.intrusos.val": 30,
    "huellas.intrusos.test": 30,
    "huellas.variantes.rotacion_max_grados": 15,
    "huellas.variantes.traslacion_max_px": 15,
    "huellas.variantes.escala_max": 0.08,
    "huellas.variantes.corte_periferico": [0.05, 0.10],  # contrato 1.1 (antes [0.15, 0.20])
    "huellas.ventana_px": [250, 250],
    "huellas.periodo_crestas_px": [8, 10],
    "huellas.kmeans.k": 3,
    "huellas.kmeans.n_init": 15,
    "voz.frecuencia_muestreo": 16000,
    "voz.comandos": ["reconocer", "identificar", "salir"],
    "voz.clase_rechazo": "otro",
    "voz.duracion_palabra_s": [0.3, 2.5],
    "voz.push_to_talk_max_s": 3.0,
    "voz.pasabanda_hz": [80, 4000],
    "voz.preenfasis": 0.97,
    "voz.mfcc.n_mfcc": 13,
    "voz.mfcc.fmax": 4000,
    "voz.mfcc.ventana_ms": 25,
    "voz.mfcc.avance_ms": 10,
    "voz.tramos": 5,
    "voz.knn.k_candidatos": [3, 5, 7, 9],
    "voz.knn.confianza_min": 0.60,
    "metas.identificacion_min": 0.95,
    "metas.far_max": 0.05,
    "metas.frr_max": 0.10,
    "metas.silhouette_min": 0.50,
    "metas.voz_acierto_min": 0.90,
    "metas.voz_otro_aceptado_max": 0.05,
}

IDENTIDADES_FIJAS = [
    {"id": 1, "nombre": "Identidad Sintética #1", "tipo": "presilla"},
    {"id": 2, "nombre": "Identidad Sintética #2", "tipo": "verticilo"},
    {"id": 3, "nombre": "Identidad Sintética #3", "tipo": "arco_tienda"},
]

# Valores que no son fijos: (camino, qué tiene que ser, verificación).
RESTRICCIONES = [
    ("semilla_global", "un entero", es_entero),
    ("huellas.intrusos.semilla_base", "un entero", es_entero),
    ("huellas.kmeans.max_iter", "un entero mayor que 0", lambda v: es_entero(v) and v > 0),
    ("huellas.kmeans.tol", "un float mayor que 0", lambda v: es_float(v) and v > 0),
    (
        "huellas.umbral.criterio",
        '"eer" o "far_max"',
        lambda v: isinstance(v, str) and v in ("eer", "far_max"),
    ),
    ("huellas.umbral.far_max_val", "un float en (0, 1)", lambda v: es_float(v) and 0 < v < 1),
    ("voz.mfcc.n_mels", "un entero entre 26 y 40", lambda v: es_entero(v) and 26 <= v <= 40),
    ("voz.knn.eps", "un float en (0, 0.001]", lambda v: es_float(v) and 0 < v <= 0.001),
    (
        "voz.knn.pca_componentes_candidatas",
        "una lista no vacía de enteros mayores que 1",
        lambda v: isinstance(v, list) and len(v) > 0 and all(es_entero(x) and x > 1 for x in v),
    ),
]

CLAVES_RUTAS = ["datos", "huellas", "voz", "modelos", "figuras"]


def test_config_yaml_esta_en_utf8_y_es_un_mapeo():
    yaml = importar("yaml")
    datos = yaml.safe_load(leer_texto("config.yaml"))
    assert isinstance(datos, dict), (
        f"yaml.safe_load(config.yaml) tiene que devolver un mapeo y devolvió {tipo_de(datos)}."
    )


@pytest.mark.parametrize("camino", CLAVES_OBLIGATORIAS)
def test_config_yaml_tiene_clave_obligatoria(config_real, camino):
    obtener(config_real, camino)


@pytest.mark.parametrize(("camino", "esperado"), list(VALORES_FIJOS.items()), ids=list(VALORES_FIJOS))
def test_config_valor_fijo(config_real, camino, esperado):
    valor = obtener(config_real, camino)
    assert iguales_exactos(valor, esperado), (
        f"{camino}: se esperaba {esperado!r} ({tipo_de(esperado)}) y hay {valor!r} ({tipo_de(valor)})."
    )


def test_config_identidades_fijas_en_orden(config_real):
    identidades = obtener(config_real, "huellas.identidades")
    assert isinstance(identidades, list) and len(identidades) == 3, (
        f"huellas.identidades tiene que ser una lista de 3 elementos; es {identidades!r}."
    )
    problemas = []
    for i, (real, esperada) in enumerate(zip(identidades, IDENTIDADES_FIJAS), start=1):
        if not isinstance(real, dict):
            problemas.append(f"  elemento {i}: no es un mapeo ({tipo_de(real)})")
            continue
        for clave, valor in esperada.items():
            if clave not in real:
                problemas.append(f"  elemento {i}: falta {clave!r}")
            elif not iguales_exactos(real[clave], valor):
                problemas.append(
                    f"  elemento {i}: {clave} = {real[clave]!r} ({tipo_de(real[clave])}), "
                    f"se esperaba {valor!r} ({tipo_de(valor)})"
                )
    assert not problemas, "huellas.identidades no coincide con el contrato:\n" + "\n".join(problemas)


def test_config_semillas_de_identidades_son_enteras(config_real):
    identidades = obtener(config_real, "huellas.identidades")
    assert isinstance(identidades, list), (
        f"huellas.identidades tiene que ser una lista; es {tipo_de(identidades)}."
    )
    malas = []
    for i, identidad in enumerate(identidades, start=1):
        semilla = identidad.get("semilla", "<falta>") if isinstance(identidad, dict) else "<no es un mapeo>"
        if not es_entero(semilla):
            malas.append(f"  elemento {i}: semilla = {semilla!r}{pista_numero_como_texto(semilla)}")
    assert not malas, "Las semillas de huellas.identidades tienen que ser enteros:\n" + "\n".join(malas)


@pytest.mark.parametrize(
    ("camino", "descripcion", "es_valido"), RESTRICCIONES, ids=[r[0] for r in RESTRICCIONES]
)
def test_config_valor_con_tipo_y_rango(config_real, camino, descripcion, es_valido):
    valor = obtener(config_real, camino)
    assert es_valido(valor), (
        f"{camino} tiene que ser {descripcion}; hay {valor!r} ({tipo_de(valor)})."
        + pista_numero_como_texto(valor)
    )


def test_config_particion_suma_variantes_por_identidad(config_real):
    train = numero(config_real, "huellas.particion.train")
    val = numero(config_real, "huellas.particion.val")
    test = numero(config_real, "huellas.particion.test")
    total = numero(config_real, "huellas.variantes_por_identidad")
    assert train + val + test == total, (
        f"particion.train + val + test = {train} + {val} + {test} = {train + val + test}, "
        f"distinto de variantes_por_identidad = {total}."
    )


def test_config_reservadas_demo_no_superan_test(config_real):
    reservadas = numero(config_real, "huellas.reservadas_demo")
    test = numero(config_real, "huellas.particion.test")
    assert reservadas <= test, f"reservadas_demo = {reservadas} supera particion.test = {test}."


def test_config_intrusos_val_mas_test_es_cantidad(config_real):
    val = numero(config_real, "huellas.intrusos.val")
    test = numero(config_real, "huellas.intrusos.test")
    cantidad = numero(config_real, "huellas.intrusos.cantidad")
    assert val + test == cantidad, (
        f"intrusos.val + intrusos.test = {val} + {test} = {val + test}, distinto de "
        f"intrusos.cantidad = {cantidad}."
    )


def test_config_semillas_de_identidades_e_intrusos_son_distintas(config_real):
    identidades = obtener(config_real, "huellas.identidades")
    base = obtener(config_real, "huellas.intrusos.semilla_base")
    semillas_id = (
        [e.get("semilla") if isinstance(e, dict) else None for e in identidades]
        if isinstance(identidades, list)
        else []
    )
    if not (identidades and all(es_entero(s) for s in semillas_id) and es_entero(base)):
        pytest.fail(
            "Para comparar semillas, las de huellas.identidades y huellas.intrusos.semilla_base "
            f"tienen que ser enteros; hay {semillas_id!r} y {base!r}.",
            pytrace=False,
        )
    usos: dict[int, list[str]] = {}
    for i, semilla in enumerate(semillas_id, start=1):
        usos.setdefault(semilla, []).append(f"identidad {i}")
    for i in range(1, 61):
        usos.setdefault(base + i, []).append(f"intruso {i}")
    choques = {s: quienes for s, quienes in usos.items() if len(quienes) > 1}
    assert not choques, "Hay semillas repetidas: " + "; ".join(
        f"{s} la usan {', '.join(quienes)}" for s, quienes in sorted(choques.items())
    )


def test_config_fmax_no_supera_pasabanda(config_real):
    fmax = numero(config_real, "voz.mfcc.fmax")
    pasabanda = par_de_numeros(config_real, "voz.pasabanda_hz")
    assert fmax <= pasabanda[1], (
        f"voz.mfcc.fmax = {fmax} supera el extremo superior de voz.pasabanda_hz = {pasabanda[1]}."
    )


def test_config_periodo_crestas_es_creciente(config_real):
    periodo = par_de_numeros(config_real, "huellas.periodo_crestas_px")
    assert periodo[0] < periodo[1], (
        f"huellas.periodo_crestas_px = {periodo}: el primero tiene que ser menor que el segundo."
    )


@pytest.mark.parametrize("clave", CLAVES_RUTAS)
def test_config_ruta_es_relativa_con_barras_y_sin_subir(config_real, clave):
    camino = f"rutas.{clave}"
    ruta = obtener(config_real, camino)
    assert isinstance(ruta, str), f"{camino} tiene que ser texto; es {ruta!r} ({tipo_de(ruta)})."
    problemas = []
    if re.match(r"[A-Za-z]:", ruta):
        problemas.append("tiene letra de unidad")
    if ruta.startswith("/"):
        problemas.append("empieza con '/'")
    if "\\" in ruta:
        problemas.append("usa '\\' como separador")
    if ".." in ruta.split("/"):
        problemas.append("tiene '..'")
    assert not problemas, f"{camino} = {ruta!r}: " + ", ".join(problemas) + "."


@pytest.mark.parametrize("clave", ["huellas", "voz"])
def test_config_ruta_esta_dentro_de_datos(config_real, clave):
    datos = obtener(config_real, "rutas.datos")
    ruta = obtener(config_real, f"rutas.{clave}")
    assert isinstance(datos, str) and isinstance(ruta, str) and ruta.startswith(datos + "/"), (
        f"rutas.{clave} = {ruta!r} tiene que empezar con rutas.datos + '/' = {datos!r} + '/'."
    )


# ---------------------------------------------------------------------------
# 8. src/config.py
# ---------------------------------------------------------------------------


def test_config_py_raiz_es_path_a_la_raiz_del_repo(modulo_config):
    raiz = getattr(modulo_config, "RAIZ", None)
    assert isinstance(raiz, Path), (
        f"src.config.RAIZ tiene que ser un pathlib.Path; es {raiz!r} ({type(raiz).__name__})."
    )
    assert raiz.resolve() == RAIZ, (
        f"src.config.RAIZ apunta a {raiz.resolve()} y la raíz del repo es {RAIZ}."
    )


def test_config_py_raiz_no_depende_del_directorio_de_trabajo(tmp_path):
    codigo = "import json, src.config as c\nprint(json.dumps({'raiz': str(c.RAIZ)}))\n"
    entorno = {
        **os.environ,
        "PYTHONPATH": str(RAIZ),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    resultado = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=tmp_path,
        env=entorno,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert resultado.returncode == 0, (
        f"Importar src.config con otra carpeta como directorio de trabajo falló:\n{resultado.stderr}"
    )
    raiz = Path(json.loads(resultado.stdout.strip().splitlines()[-1])["raiz"])
    assert raiz.is_absolute() and raiz.resolve() == RAIZ, (
        f"Con el directorio de trabajo en {tmp_path}, src.config.RAIZ vale {raiz} "
        f"y tendría que ser {RAIZ}."
    )


def test_cargar_config_sin_ruta_lee_config_yaml_de_la_raiz(cargar_config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # que no lo encuentre por estar parado en la raíz
    cfg = cargar_config()
    assert isinstance(cfg, dict), f"cargar_config() tiene que devolver un dict; devolvió {tipo_de(cfg)}."
    assert cfg == cargar_config_real(), (
        "cargar_config() no devolvió el contenido de RAIZ/config.yaml."
    )


@pytest.mark.parametrize(
    "convertir", [str, Path, RutaPropia], ids=["str", "pathlib.Path", "os.PathLike"]
)
def test_cargar_config_acepta_str_y_pathlike(cargar_config, copia_config, convertir):
    cfg = cargar_config(convertir(copia_config))
    assert isinstance(cfg, dict), f"cargar_config tiene que devolver un dict; devolvió {tipo_de(cfg)}."
    assert cfg == cargar_config_real(), (
        f"cargar_config({convertir.__name__}(ruta)) no devolvió el contenido del archivo."
    )


def test_cargar_config_lee_en_utf8(cargar_config, tmp_path):
    datos = cargar_config_real()
    datos["prueba_utf8"] = "ñandú – acción ü €"
    ruta = escribir_yaml(tmp_path / "config.yaml", datos)
    cfg = cargar_config(ruta)
    assert cfg.get("prueba_utf8") == "ñandú – acción ü €", (
        f"cargar_config no leyó el archivo en UTF-8: prueba_utf8 = {cfg.get('prueba_utf8')!r}."
    )
    nombre = cfg["huellas"]["identidades"][0]["nombre"]
    assert nombre == datos["huellas"]["identidades"][0]["nombre"], (
        f"cargar_config no leyó el archivo en UTF-8: nombre de la identidad 1 = {nombre!r}."
    )


def test_cargar_config_archivo_inexistente_lanza_file_not_found_error(cargar_config, tmp_path):
    lanza(FileNotFoundError, cargar_config, tmp_path / "no_existe.yaml", que="Archivo inexistente")


@pytest.mark.parametrize(
    "contenido", ["", "- reconocer\n- identificar\n", "42\n"], ids=["vacio", "lista", "numero"]
)
def test_cargar_config_contenido_que_no_es_mapeo_lanza_value_error(cargar_config, tmp_path, contenido):
    ruta = tmp_path / "config.yaml"
    ruta.write_text(contenido, encoding="utf-8")
    lanza(ValueError, cargar_config, ruta, que=f"Contenido {contenido!r}")


@pytest.mark.parametrize("camino", CLAVES_OBLIGATORIAS)
def test_cargar_config_sin_una_hoja_lanza_value_error_que_la_nombra(cargar_config, tmp_path, camino):
    ruta = escribir_yaml(tmp_path / "config.yaml", sin_clave(cargar_config_real(), camino))
    error = lanza(ValueError, cargar_config, ruta, que=f"Config sin {camino}")
    assert nombra(str(error), camino), f"El ValueError no nombra {camino!r}. Mensaje: {error}"


@pytest.mark.parametrize("seccion", SECCIONES)
def test_cargar_config_sin_una_seccion_nombra_todas_sus_hojas(cargar_config, tmp_path, seccion):
    ruta = escribir_yaml(tmp_path / "config.yaml", sin_clave(cargar_config_real(), seccion))
    error = lanza(ValueError, cargar_config, ruta, que=f"Config sin la sección {seccion}")
    hojas = [c for c in CLAVES_OBLIGATORIAS if c.startswith(seccion + ".")]
    no_nombradas = [h for h in hojas if not nombra(str(error), h)]
    assert not no_nombradas, (
        f"Sin la sección {seccion!r}, el ValueError no nombra estas hojas: {no_nombradas}. "
        f"Mensaje: {error}"
    )


def test_cargar_config_acepta_claves_de_mas(cargar_config, tmp_path):
    datos = cargar_config_real()
    datos["clave_de_mas"] = "sobra"
    obtener(datos, "huellas")["clave_de_mas"] = {"anidada": 1}
    obtener(datos, "voz.mfcc")["clave_de_mas"] = [1, 2]
    ruta = escribir_yaml(tmp_path / "config.yaml", datos)
    cfg = cargar_config(ruta)
    assert cfg == datos, "Con claves de más, cargar_config no devolvió el contenido completo del archivo."


def test_cargar_config_no_modifica_el_archivo(cargar_config, copia_config):
    archivos = [copia_config, RAIZ / "config.yaml"]
    antes = {a: (a.read_bytes(), a.stat().st_mtime_ns) for a in archivos}
    cargar_config(copia_config)
    cargar_config()
    cambiados = [str(a) for a in archivos if (a.read_bytes(), a.stat().st_mtime_ns) != antes[a]]
    assert not cambiados, f"cargar_config modificó: {cambiados}."


@pytest.mark.parametrize("con_ruta", [False, True], ids=["sin_ruta", "con_ruta"])
def test_cargar_config_devuelve_un_dict_nuevo_e_independiente(cargar_config, copia_config, con_ruta):
    def llamar():
        return cargar_config(copia_config) if con_ruta else cargar_config()

    esperado = cargar_config_real()
    primero, segundo = llamar(), llamar()
    assert primero is not segundo, "Dos llamadas devolvieron el mismo objeto dict."
    compartidos = ids_contenedores(primero) & ids_contenedores(segundo)
    assert not compartidos, (
        f"Dos llamadas devolvieron dicts que comparten {len(compartidos)} dicts/listas internos."
    )
    ensuciar(primero)
    assert segundo == esperado and llamar() == esperado, (
        "Modificar el dict devuelto por una llamada cambió lo que devuelven las otras."
    )
