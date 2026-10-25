"""Tests de la Fase 5 · Integración (contrato f5_integracion.md, versión 1 del 25/10/2026).

Salen solo del contrato: cada test verifica un punto de su sección 6 (la sección 7 es manual).
El oráculo de la tabla de reglas (4.1), las cuentas del push-to-talk (4.3), la receta de H1
(0.3 y 4.4) y los dobles de prueba (sección 5) están escritos acá.

Supuestos que el contrato no fija y se acordaron con Santiago antes de escribir el archivo:
- huellas.identidades de config.yaml es una lista de mapeos con las claves id, nombre, tipo y
  semilla (la receta de H1 toma de ahí el tipo y la semilla de cada identidad).
- En el metadatos.csv de huellas, intruso es el número de intruso (0 en las filas de identidades)
  e identidad, variante y reservada_demo son enteros (reservada_demo vale 0 o 1).
- Pantalla.identificacion es "" en cualquier estado que no sea EVALUADA_AUTORIZADA, también en
  FIN después de "identificar" y "salir" (manda 6.9 sobre 4.6).

Correr:           venv\\Scripts\\python -m pytest tests/test_f5.py
Sin aceptación:   venv\\Scripts\\python -m pytest tests/test_f5.py -m "not aceptacion"
"""

from __future__ import annotations

import collections
import copy
import csv
import dataclasses
import datetime
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import soundfile as sf
import yaml

RAIZ = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Valores del contrato (se comparan carácter por carácter)
# ---------------------------------------------------------------------------

ESTADOS = ("ESPERANDO_MUESTRA", "HUELLA_CARGADA", "EVALUADA_AUTORIZADA", "EVALUADA_RECHAZADA", "FIN")
NO_FIN = ESTADOS[:4]
EVALUADAS = ("EVALUADA_AUTORIZADA", "EVALUADA_RECHAZADA")
PERCEPCIONES = ("H", "HI", "R", "I", "S", "O")
COMANDOS = {"R": "reconocer", "I": "identificar", "S": "salir", "O": "otro"}

TIPOS_ACCION_C = ("mostrar_huella", "resultado", "identidad", "aviso", "cerrar", "ninguna")
AUTORIZADO_C = "AUTORIZADO"
NO_AUTORIZADO_C = "NO AUTORIZADO"
REGLAS_AVISO_C = frozenset({2, 4, 6, 8, 10})
TEXTOS_C = {
    1: "Huella cargada: decí Reconocer",
    2: "Primero elegí una huella",
    4: "Primero decí Reconocer",
    6: "Huella no autorizada: no se muestra ninguna identidad",
    8: "No entendí, repetí el comando",
    9: "Cerrando la aplicación",
    10: "No se pudo leer la huella: elegí otra",
}
COLORES_C = ("neutro", "verde", "rojo")
AVISOS_C = {
    "inicio": "Elegí una huella para empezar",
    "audio_ilegible": "No se pudo leer el audio: elegí otro archivo",
    "sin_microfono": "No hay micrófono disponible: usá «Abrir audio…»",
    "error": "Error inesperado: {tipo}: {mensaje}",
}
ORIGENES_C = ("reservada", "intruso_test", "variante_nueva", "intruso_nuevo", "archivo")
NOMBRES = {1: "Identidad Sintética #1", 2: "Identidad Sintética #2", 3: "Identidad Sintética #3"}
ETIQUETAS_VOZ = ("reconocer", "identificar", "salir", "otro")
CAMPOS_PANTALLA = (
    "estado", "huella", "imagen", "esqueleto", "cartel", "color", "identificacion", "distancia", "tau",
    "punto_pca", "mensaje", "aviso", "regla", "comando", "detalle_voz", "microfono", "grabando",
    "nivel_db", "cerrar",
)
ENCABEZADO_REGISTRO = ["momento", "percepcion", "detalle", "regla", "estado_antes", "estado_despues", "mensaje"]

# Claves nuevas de config.yaml (sección 1) con sus valores
CLAVES_NUEVAS = (
    ("app", "pre_roll_s", 0.25),
    ("app", "post_roll_s", 0.15),
    ("app", "bufer_s", 5.0),
    ("app", "medidor_ms", 50),
    ("app", "variante_nueva_desde", 1000),
    ("app", "intruso_nuevo_desde", 1000),
    ("metas", "respuesta_max_s", 1.0),
)

# Datos reales (6.11)
MODELO_H = "models/modelo_huellas.npz"
MODELO_V = "models/modelo_voz.npz"
META_H = "data/huellas/metadatos.csv"
CARAC_H = "data/huellas/caracteristicas.npz"
META_V = "data/voz/dataset/metadatos.csv"
# Sección 8: distancia normalizada de las reservadas contra τ* = 1,5613
DIST_RESERVADAS = {
    "id1_v039": 1.2867, "id1_v040": 0.6012, "id2_v039": 1.0351,
    "id2_v040": 1.5177, "id3_v039": 0.9906, "id3_v040": 1.4872,
}

BASURA = b"esto no es un PNG ni un WAV\x00\x01\x02" * 20


# ---------------------------------------------------------------------------
# Módulos de src/ (se importan en fixtures: si falta uno, fallan solo sus tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    from src.config import cargar_config
    return cargar_config()


@pytest.fixture(scope="module")
def ag():
    import src.agente.agente as modulo
    return modulo


@pytest.fixture(scope="module")
def per():
    import src.agente.percepcion as modulo
    return modulo


@pytest.fixture(scope="module")
def cap():
    import src.agente.captura as modulo
    return modulo


@pytest.fixture(scope="module")
def mue():
    import src.app.muestras as modulo
    return modulo


@pytest.fixture(scope="module")
def vis():
    import src.app.visor as modulo
    return modulo


@pytest.fixture(scope="module")
def ctl():
    import src.app.controlador as modulo
    return modulo


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def es_int(x) -> bool:
    """El "int" de 0.2: int de Python o np.integer, que no sea bool."""
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def db(bloque, piso: float = 1e-10) -> float:
    """Oráculo de nivel_db (4.3)."""
    x = np.asarray(bloque, dtype=np.float64)
    rms = math.sqrt(float(np.mean(x * x))) if x.size else 0.0
    return 20.0 * math.log10(max(rms, piso))


def escribir_png(ruta: Path, imagen: np.ndarray) -> None:
    ok, datos = cv2.imencode(".png", imagen)
    assert ok, f"no se pudo codificar el PNG de prueba {ruta}"
    datos.tofile(str(ruta))  # así funciona también con tildes en la ruta (Windows)


def leer_png(ruta: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def identidad_cfg(cfg: dict, k: int) -> dict:
    for ident in cfg["huellas"]["identidades"]:
        if int(ident["id"]) == k:
            return ident
    raise KeyError(f"la identidad {k} no está en huellas.identidades")


def importar_en_subproceso(modulo: str, nombres, extra: tuple[str, ...] = ()) -> dict:
    """Importa `modulo` en un Python aparte (cwd=RAIZ, sin pantalla) y devuelve
    {nombre: nombre in sys.modules}, más lo que las líneas `extra` agreguen al diccionario `r`."""
    codigo = "\n".join([
        "import json, sys",
        f"import {modulo} as m",
        f"r = {{n: (n in sys.modules) for n in {list(nombres)!r}}}",
        *extra,
        'print("RESULTADO=" + json.dumps(r))',
    ])
    entorno = dict(os.environ)
    entorno.pop("DISPLAY", None)
    entorno.pop("WAYLAND_DISPLAY", None)
    entorno["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True, text=True, timeout=180, env=entorno
    )
    assert proc.returncode == 0, (
        f"importar {modulo} en un subproceso falló (código {proc.returncode}):\n{proc.stderr[-3000:]}"
    )
    lineas = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULTADO=")]
    assert lineas, f"el subproceso no imprimió el resultado; salida:\n{proc.stdout[-2000:]}"
    return json.loads(lineas[-1][len("RESULTADO="):])


def espiar(monkeypatch, nombre: str, modulos) -> list:
    """Envuelve la función `nombre` en cada módulo que la tenga y anota los argumentos de cada llamada."""
    llamadas = []
    for modulo in modulos:
        if hasattr(modulo, nombre):
            def espia(*args, _original=getattr(modulo, nombre), **kwargs):
                llamadas.append(args)
                return _original(*args, **kwargs)
            monkeypatch.setattr(modulo, nombre, espia)
    return llamadas


def requerir(*relativos: str) -> None:
    faltan = [r for r in relativos if not (RAIZ / r).exists()]
    if faltan:
        pytest.skip("faltan datos reales: " + ", ".join(faltan))


# ---------------------------------------------------------------------------
# Dobles de prueba (sección 5)
# ---------------------------------------------------------------------------

AUTORIZADAS = {11: 1, 12: 2, 13: 3}   # clave de la imagen falsa -> identidad que decide el evaluar falso
RECHAZADAS = {20: 2, 21: 1}           # clave -> mas_cercana (el agente nunca la usa)
ILEGIBLE = 99                         # con esta clave, el preparar falso lanza ValueError
D_FALSO = 5
TAU_FALSO = 1.5


def imagen_falsa(clave: int) -> np.ndarray:
    """Imagen uint8 chica; el píxel (0, 0) es la clave que leen los dobles."""
    imagen = (np.arange(24 * 20).reshape(24, 20) % 7 + 30).astype(np.uint8)
    imagen[0, 0] = clave
    return imagen


def esqueleto_falso(clave: int) -> np.ndarray:
    esqueleto = np.zeros((250, 250), dtype=bool)
    esqueleto[clave % 250, :] = True
    esqueleto[:, (7 * clave) % 250] = True
    return esqueleto


def z_falso(clave: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, D_FALSO) * (clave / 10.0) + 0.25 * np.arange(D_FALSO)


def nombre_falso(clave: int) -> str:
    return {11: "id1_v039", 12: "id2_v039", 13: "id3_v040", 20: "intruso31_v1", 21: "intruso32_v1",
            ILEGIBLE: "no_es_huella.png"}.get(clave, f"huella{clave}")


@dataclasses.dataclass(frozen=True, eq=False)
class PreparadaFalsa:
    clave: int
    esqueleto: np.ndarray


@dataclasses.dataclass(frozen=True, eq=False)
class ResultadoFalso:
    identidad: int
    mas_cercana: int
    distancia: float
    tau: float
    z: np.ndarray


def resultado_falso(clave: int, identidad: int | None = None) -> ResultadoFalso:
    if identidad is None:
        identidad = AUTORIZADAS.get(clave, 0)
    mas_cercana = AUTORIZADAS.get(clave) or RECHAZADAS.get(clave, 1)
    distancia = 0.375 + 0.125 * identidad + clave / 1024 if identidad else 2.0 + clave / 64
    return ResultadoFalso(int(identidad), int(mas_cercana), np.float64(distancia), TAU_FALSO, z_falso(clave))


class PrepararFalso:
    """`preparar` falso: devuelve una PreparadaFalsa con su esqueleto, o lanza ValueError si la imagen
    no es uint8 de 2 dimensiones o tiene la clave ILEGIBLE. Con `lanzar`, lanza esa excepción."""

    def __init__(self):
        self.llamadas: list = []
        self.devueltas: list = []
        self.lanzar: BaseException | None = None

    def __call__(self, imagen):
        self.llamadas.append(imagen)
        if self.lanzar is not None:
            raise self.lanzar
        if not isinstance(imagen, np.ndarray) or imagen.dtype != np.uint8 or imagen.ndim != 2:
            raise ValueError("preparar falso: no es una imagen uint8 de 2 dimensiones")
        clave = int(imagen[0, 0])
        if clave == ILEGIBLE:
            raise ValueError("preparar falso: no se encontró una huella")
        preparada = PreparadaFalsa(clave, esqueleto_falso(clave))
        self.devueltas.append(preparada)
        return preparada


class EvaluarFalso:
    """`evaluar` falso: decide según la clave de la huella (AUTORIZADAS / RECHAZADAS), salvo que
    `decisiones` tenga identidades que pisan la decisión de las próximas llamadas. Cuenta las llamadas."""

    def __init__(self):
        self.llamadas: list = []
        self.ultimo = None
        self.lanzar: BaseException | None = None
        self.decisiones: list[int] = []
        self.sin_z = False

    def __call__(self, preparada):
        self.llamadas.append(preparada)
        if self.lanzar is not None:
            raise self.lanzar
        identidad = self.decisiones.pop(0) if self.decisiones else None
        resultado = resultado_falso(preparada.clave, identidad)
        if self.sin_z:
            resultado = SimpleNamespace(identidad=resultado.identidad, mas_cercana=resultado.mas_cercana,
                                        distancia=resultado.distancia, tau=resultado.tau)
        self.ultimo = resultado
        return resultado


@dataclasses.dataclass(frozen=True)
class RecFalso:
    """Con los campos del Reconocimiento de V4."""
    comando: str
    prediccion: str | None
    confianza: float
    distancia: float
    tau: float
    valido: bool
    motivo: str


def rec_falso(comando: str, valido: bool = True, motivo: str | None = None) -> RecFalso:
    if valido:
        return RecFalso(comando, comando, 0.875, 0.3125, 0.75, True, motivo or "ok")
    return RecFalso(comando, None, 0.0, 0.0, 0.75, False, motivo or "sin_voz")


class VozFalsa:
    """`reconocer_voz` falso: devuelve los Reconocimientos de `cola` en orden (sin cola, un "otro" no
    válido con motivo "sin_voz") y anota cada (y, fs) que recibe."""

    def __init__(self):
        self.cola: list[RecFalso] = []
        self.llamadas: list = []

    def __call__(self, y, fs):
        self.llamadas.append((y, fs))
        return self.cola.pop(0) if self.cola else rec_falso("otro", valido=False, motivo="sin_voz")


class GeneradorFalso:
    """Imita a GeneradorDemo con imágenes falsas (las variantes autorizan, los intrusos no)."""

    def __init__(self, mue):
        self.mue = mue
        self.llamadas: list = []
        self.n = collections.Counter()

    def variante_nueva(self, identidad):
        self.llamadas.append(("variante_nueva", identidad))
        if identidad not in (1, 2, 3):
            raise ValueError(f"identidad {identidad} desconocida")
        v = 1000 + self.n[identidad]
        self.n[identidad] += 1
        entrada = self.mue.EntradaHuella(nombre=f"id{identidad}_v{v} (nueva)", origen="variante_nueva",
                                         verdad=identidad, ruta=None, variante=v)
        return entrada, imagen_falsa(10 + identidad)

    def intruso_nuevo(self):
        self.llamadas.append(("intruso_nuevo",))
        i = 1000 + self.n["intruso"]
        self.n["intruso"] += 1
        entrada = self.mue.EntradaHuella(nombre=f"intruso{i}_v1 (nuevo)", origen="intruso_nuevo", verdad=0,
                                         ruta=None, variante=1)
        return entrada, imagen_falsa(21)


class StreamFalso:
    def __init__(self, kwargs: dict, falla_al_arrancar: bool = False):
        self.kwargs = kwargs
        self.callback = kwargs.get("callback")
        self.falla_al_arrancar = falla_al_arrancar
        self.llamadas = collections.Counter()

    def start(self):
        self.llamadas["start"] += 1
        if self.falla_al_arrancar:
            raise OSError("stream falso: no se pudo arrancar")

    def stop(self):
        self.llamadas["stop"] += 1

    def close(self):
        self.llamadas["close"] += 1


class FabricaFalsa:
    """Imita sounddevice.InputStream: guarda el callback para pasarle bloques conocidos y anota
    start, stop y close. Puede lanzar OSError al crearse o al arrancar."""

    def __init__(self, falla_al_crear: bool = False, falla_al_arrancar: bool = False):
        self.falla_al_crear = falla_al_crear
        self.falla_al_arrancar = falla_al_arrancar
        self.llamadas: list = []
        self.streams: list[StreamFalso] = []

    def __call__(self, *args, **kwargs):
        self.llamadas.append((args, kwargs))
        if self.falla_al_crear:
            raise OSError("fábrica falsa: no hay dispositivo de entrada")
        stream = StreamFalso(kwargs, self.falla_al_arrancar)
        self.streams.append(stream)
        return stream


def alimentar(mic, fab: FabricaFalsa, senal: np.ndarray, hasta: int, bloque: int = 100) -> None:
    """Pasa al callback del stream falso la señal desde bufer.escritas hasta `hasta`, en bloques (n, 1)."""
    desde = mic.bufer.escritas
    assert desde <= hasta, f"el búfer ya tiene {desde} muestras (se pidió alimentar hasta {hasta})"
    callback = fab.streams[-1].callback
    for i in range(desde, hasta, bloque):
        j = min(i + bloque, hasta)
        callback(senal[i:j].reshape(-1, 1).astype(np.float32, copy=True), j - i, None, None)


# ---------------------------------------------------------------------------
# Oráculo de la tabla de reglas (4.1) y chequeo de una Accion
# ---------------------------------------------------------------------------

def oraculo(estado: str, percepcion: str, autoriza: bool) -> tuple[int, str]:
    """(regla, estado siguiente) de la tabla de 4.1. `autoriza`: si evaluar autoriza la huella cargada."""
    if estado == "FIN":
        return 0, "FIN"
    if percepcion == "H":
        return 1, "HUELLA_CARGADA"
    if percepcion == "HI":
        return 10, "ESPERANDO_MUESTRA"
    if percepcion == "S":
        return 9, "FIN"
    if percepcion == "O":
        return 8, estado
    evaluada = "EVALUADA_AUTORIZADA" if autoriza else "EVALUADA_RECHAZADA"
    if percepcion == "R":
        return {"ESPERANDO_MUESTRA": (2, estado), "HUELLA_CARGADA": (3, evaluada),
                "EVALUADA_AUTORIZADA": (7, evaluada), "EVALUADA_RECHAZADA": (7, evaluada)}[estado]
    if percepcion == "I":
        return {"ESPERANDO_MUESTRA": (2, estado), "HUELLA_CARGADA": (4, estado),
                "EVALUADA_AUTORIZADA": (5, estado), "EVALUADA_RECHAZADA": (6, estado)}[estado]
    raise AssertionError(f"percepción desconocida {percepcion!r}")


def verificar_accion(ag, accion, regla: int, estado: str, resultado=None, contexto: str = "") -> None:
    """La Accion tiene la regla, el tipo, el texto, el estado y los campos opcionales de 4.1.
    `resultado` es lo que devolvió evaluar (se usa en las reglas 3, 5 y 7)."""
    pre = f"[{contexto}] " if contexto else ""
    assert accion.regla == regla, f"{pre}Accion.regla = {accion.regla!r}, esperada {regla}"
    assert es_int(accion.regla), f"{pre}Accion.regla no es int: {type(accion.regla)}"
    assert isinstance(accion.estado, ag.Estado), f"{pre}Accion.estado no es un Estado: {accion.estado!r}"
    assert accion.estado == ag.Estado[estado], f"{pre}Accion.estado = {accion.estado!r}, esperado {estado}"
    tipo = {0: "ninguna", 1: "mostrar_huella", 3: "resultado", 5: "identidad", 7: "resultado",
            9: "cerrar"}.get(regla, "aviso")
    assert accion.tipo == tipo, f"{pre}Accion.tipo = {accion.tipo!r}, esperado {tipo!r} (regla {regla})"
    if regla == 0:
        texto = ""
    elif regla in (3, 7):
        texto = AUTORIZADO_C if resultado.identidad != 0 else NO_AUTORIZADO_C
    elif regla == 5:
        texto = NOMBRES[resultado.identidad]
    else:
        texto = TEXTOS_C[regla]
    assert accion.texto == texto, f"{pre}Accion.texto = {accion.texto!r}, esperado {texto!r} (regla {regla})"

    if regla in (3, 7):
        esperado = {"autorizado": resultado.identidad != 0, "identidad": None, "nombre": None,
                    "distancia": float(resultado.distancia), "tau": float(resultado.tau)}
    elif regla == 5:
        esperado = {"autorizado": None, "identidad": resultado.identidad, "nombre": NOMBRES[resultado.identidad],
                    "distancia": float(resultado.distancia), "tau": float(resultado.tau)}
    else:
        esperado = dict.fromkeys(("autorizado", "identidad", "nombre", "distancia", "tau"))
    for campo, valor in esperado.items():
        obtenido = getattr(accion, campo)
        if valor is None:
            assert obtenido is None, f"{pre}Accion.{campo} = {obtenido!r}, esperado None (regla {regla})"
        else:
            assert obtenido == valor, f"{pre}Accion.{campo} = {obtenido!r}, esperado {valor!r} (regla {regla})"
    if regla in (3, 5, 7):
        assert isinstance(accion.distancia, float) and isinstance(accion.tau, float), (
            f"{pre}distancia y tau tienen que ser float: {type(accion.distancia)}, {type(accion.tau)}")
    if regla in (3, 7):
        assert isinstance(accion.autorizado, bool), f"{pre}Accion.autorizado no es bool: {accion.autorizado!r}"
    if regla == 5:
        assert es_int(accion.identidad), f"{pre}Accion.identidad no es int: {accion.identidad!r}"


def nuevo_agente(ag, cfg):
    prep, eva = PrepararFalso(), EvaluarFalso()
    return ag.Agente(cfg, prep, eva), prep, eva


def percibir(agente, percepcion: str, clave: int = 11):
    """H: huella legible con esa clave; HI: imagen None; HIV: imagen que preparar rechaza con ValueError;
    R, I, S, O: los comandos."""
    if percepcion == "H":
        return agente.percibir_huella(imagen_falsa(clave), nombre_falso(clave))
    if percepcion == "HI":
        return agente.percibir_huella(None, "rota.png")
    if percepcion == "HIV":
        return agente.percibir_huella(imagen_falsa(ILEGIBLE), nombre_falso(ILEGIBLE))
    return agente.percibir_comando(COMANDOS[percepcion])


def llevar_a(ag, agente, estado: str, autoriza: bool = True, identificar: bool = False) -> None:
    """Lleva al agente a `estado` con percepciones de la tabla."""
    if estado == "HUELLA_CARGADA":
        percibir(agente, "H", 11 if autoriza else 20)
    elif estado == "EVALUADA_AUTORIZADA":
        percibir(agente, "H", 12)
        percibir(agente, "R")
        if identificar:
            percibir(agente, "I")
    elif estado == "EVALUADA_RECHAZADA":
        percibir(agente, "H", 20)
        percibir(agente, "R")
    elif estado == "FIN":
        percibir(agente, "S")
    assert agente.estado == ag.Estado[estado], f"no se pudo llevar al agente a {estado}: quedó en {agente.estado!r}"


def foto(agente) -> dict:
    imagen = agente.imagen
    return {"estado": agente.estado, "huella": agente.huella, "imagen": imagen,
            "imagen_copia": None if imagen is None else imagen.copy(), "preparada": agente.preparada,
            "resultado": agente.resultado, "identificada": agente.identificada}


def comparar_foto(agente, antes: dict, contexto: str, salvo_estado: bool = False) -> None:
    if not salvo_estado:
        assert agente.estado == antes["estado"], f"[{contexto}] estado {agente.estado!r}, antes {antes['estado']!r}"
    assert agente.huella == antes["huella"], f"[{contexto}] huella {agente.huella!r}, antes {antes['huella']!r}"
    if antes["imagen"] is None:
        assert agente.imagen is None, f"[{contexto}] imagen cambió: era None"
    else:
        assert agente.imagen is not None and np.array_equal(agente.imagen, antes["imagen_copia"]), (
            f"[{contexto}] la imagen cambió")
    assert agente.preparada is antes["preparada"], f"[{contexto}] preparada cambió"
    assert agente.resultado is antes["resultado"], f"[{contexto}] resultado cambió"
    assert agente.identificada == antes["identificada"], (
        f"[{contexto}] identificada {agente.identificada!r}, antes {antes['identificada']!r}")


def verificar_atributos(agente, accion, antes: dict, clave: int | None, prep, eva, contexto: str) -> None:
    """Los atributos del agente después de la regla (4.1). `antes` es la foto previa; `clave`, la de la
    huella percibida (para la regla 1)."""
    r = accion.regla
    if r == 1:
        assert agente.huella == nombre_falso(clave), f"[{contexto}] regla 1: huella = {agente.huella!r}"
        assert agente.imagen is not None and np.array_equal(agente.imagen, imagen_falsa(clave)), f"[{contexto}] regla 1: imagen"
        assert agente.preparada is prep.devueltas[-1], f"[{contexto}] regla 1: preparada no es lo que devolvió preparar"
        assert agente.resultado is None and agente.identificada is False, (
            f"[{contexto}] regla 1: resultado = {agente.resultado!r}, identificada = {agente.identificada!r}; "
            "esperados None y False")
    elif r == 10:
        obtenido = (agente.huella, agente.imagen, agente.preparada, agente.resultado, agente.identificada)
        assert obtenido == ("", None, None, None, False), f"[{contexto}] regla 10 no descartó la huella: {obtenido!r}"
    elif r in (3, 7):
        assert agente.resultado is eva.ultimo, f"[{contexto}] regla {r}: resultado no es lo que devolvió evaluar"
        assert agente.identificada is False, f"[{contexto}] regla {r}: identificada = {agente.identificada!r}"
        comparar_foto(agente, {**antes, "resultado": eva.ultimo, "identificada": False}, contexto, salvo_estado=True)
    elif r == 5:
        assert agente.identificada is True, f"[{contexto}] regla 5: identificada = {agente.identificada!r}"
        comparar_foto(agente, {**antes, "identificada": True}, contexto)
    else:  # 0, 2, 4, 6, 8 y 9 no cambian nada (la 9 solo el estado)
        comparar_foto(agente, antes, f"{contexto}, regla {r}", salvo_estado=r == 9)


# ===========================================================================
# 6.1 Configuración
# ===========================================================================

def test_config_claves_nuevas_con_sus_valores(cfg):
    crudo = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    for seccion, clave, valor in CLAVES_NUEVAS:
        assert clave in (crudo.get(seccion) or {}), f"config.yaml no tiene {seccion}.{clave}"
        assert crudo[seccion][clave] == valor, f"config.yaml: {seccion}.{clave} = {crudo[seccion][clave]!r}, esperado {valor!r}"
        assert cfg[seccion][clave] == valor, f"cargar_config(): {seccion}.{clave} = {cfg[seccion][clave]!r}, esperado {valor!r}"


@pytest.mark.parametrize("seccion,clave,valor", CLAVES_NUEVAS, ids=[f"{s}.{c}" for s, c, _ in CLAVES_NUEVAS])
def test_config_sin_clave_nueva_lanza_value_error(tmp_path, seccion, clave, valor):
    from src.config import cargar_config

    datos = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    entera = tmp_path / "config_entera.yaml"
    entera.write_text(yaml.safe_dump(datos, allow_unicode=True, sort_keys=False), encoding="utf-8")
    cargar_config(entera)  # la copia sin cambios se carga bien

    del datos[seccion][clave]
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        cargar_config(ruta)
    assert clave in str(error.value), f"el ValueError no nombra {seccion}.{clave}: {error.value}"


def test_config_relaciones_entre_claves(cfg):
    app = cfg["app"]
    minimo = app["pre_roll_s"] + cfg["voz"]["push_to_talk_max_s"] + app["post_roll_s"] + 1.0
    assert app["bufer_s"] >= minimo, f"bufer_s = {app['bufer_s']} < {minimo} (pre + push_to_talk_max + post + 1)"
    assert app["variante_nueva_desde"] > 140, f"variante_nueva_desde = {app['variante_nueva_desde']}, tiene que ser > 140"
    assert app["intruso_nuevo_desde"] > 160, f"intruso_nuevo_desde = {app['intruso_nuevo_desde']}, tiene que ser > 160"


def test_nombres_identidades_y_etiquetas_voz(ag, cfg):
    assert ag.nombres_identidades(cfg) == NOMBRES, f"nombres_identidades = {ag.nombres_identidades(cfg)!r}"
    etiquetas = ag.etiquetas_voz(cfg)
    assert isinstance(etiquetas, tuple) and etiquetas == ETIQUETAS_VOZ, f"etiquetas_voz = {etiquetas!r}, esperado {ETIQUETAS_VOZ!r}"


# ===========================================================================
# 6.2 Agente (dobles de prueba)
# ===========================================================================

def test_textos_exactos_del_contrato(ag, ctl, mue):
    assert [e.name for e in ag.Estado] == list(ESTADOS), f"Estado = {[e.name for e in ag.Estado]}"
    assert all(e.value == e.name for e in ag.Estado), "cada Estado vale su propio nombre"
    assert issubclass(ag.Estado, str), "Estado tiene que ser (str, Enum)"
    assert tuple(ag.TIPOS_ACCION) == TIPOS_ACCION_C, f"TIPOS_ACCION = {ag.TIPOS_ACCION!r}"
    assert ag.AUTORIZADO == AUTORIZADO_C, f"AUTORIZADO = {ag.AUTORIZADO!r}"
    assert ag.NO_AUTORIZADO == NO_AUTORIZADO_C, f"NO_AUTORIZADO = {ag.NO_AUTORIZADO!r}"
    assert ag.REGLAS_AVISO == REGLAS_AVISO_C, f"REGLAS_AVISO = {ag.REGLAS_AVISO!r}"
    assert ag.TEXTOS == TEXTOS_C, f"TEXTOS = {ag.TEXTOS!r}"
    assert tuple(ctl.COLORES) == COLORES_C, f"COLORES = {ctl.COLORES!r}"
    assert ctl.AVISOS == AVISOS_C, f"AVISOS = {ctl.AVISOS!r}"
    assert tuple(mue.ORIGENES) == ORIGENES_C, f"ORIGENES = {mue.ORIGENES!r}"
    campos = tuple(f.name for f in dataclasses.fields(ctl.Pantalla))
    assert campos == CAMPOS_PANTALLA, f"campos de Pantalla = {campos}, esperados {CAMPOS_PANTALLA}"


@pytest.mark.parametrize("percepcion", PERCEPCIONES)
@pytest.mark.parametrize("estado", ESTADOS)
def test_tabla_reglas_completa(ag, cfg, estado, percepcion):
    """Las 30 celdas de 4.1. EVALUADA_* se prueba con huellas que autorizan y que rechazan; HI, con
    imagen None y con una imagen que preparar rechaza con ValueError."""
    if estado == "EVALUADA_AUTORIZADA" or estado in ("ESPERANDO_MUESTRA", "FIN"):
        variantes = (True,)
    elif estado == "EVALUADA_RECHAZADA":
        variantes = (False,)
    else:
        variantes = (True, False)
    formas = ("HI", "HIV") if percepcion == "HI" else (percepcion,)
    for autoriza in variantes:
        for forma in formas:
            agente, prep, eva = nuevo_agente(ag, cfg)
            llevar_a(ag, agente, estado, autoriza)
            antes = foto(agente)
            clave = 13 if autoriza else 21
            accion = percibir(agente, forma, clave)
            regla, siguiente = oraculo(estado, percepcion, autoriza)
            contexto = f"{estado} + {forma}, {'autoriza' if autoriza else 'rechaza'}"
            verificar_accion(ag, accion, regla, siguiente, eva.ultimo, contexto)
            verificar_atributos(agente, accion, antes, clave, prep, eva, contexto)
            assert agente.estado == ag.Estado[siguiente], (
                f"[{contexto}] agente.estado = {agente.estado!r}, esperado {siguiente}")


def test_campos_de_la_accion_por_regla(ag, cfg):
    agente, prep, eva = nuevo_agente(ag, cfg)
    verificar_accion(ag, percibir(agente, "R"), 2, "ESPERANDO_MUESTRA", contexto="regla 2")
    verificar_accion(ag, percibir(agente, "H", 12), 1, "HUELLA_CARGADA", contexto="regla 1")
    verificar_accion(ag, percibir(agente, "I"), 4, "HUELLA_CARGADA", contexto="regla 4")

    accion = percibir(agente, "R")
    verificar_accion(ag, accion, 3, "EVALUADA_AUTORIZADA", eva.ultimo, "regla 3 autorizada")
    assert accion.autorizado is True and accion.identidad is None and accion.nombre is None, (
        f"la regla 3 autorizada no lleva identidad ni nombre: {accion!r}")

    accion = percibir(agente, "I")
    verificar_accion(ag, accion, 5, "EVALUADA_AUTORIZADA", eva.ultimo, "regla 5")
    assert (accion.identidad, accion.nombre, accion.texto) == (2, NOMBRES[2], NOMBRES[2]), f"regla 5: {accion!r}"

    accion = percibir(agente, "R")
    verificar_accion(ag, accion, 7, "EVALUADA_AUTORIZADA", eva.ultimo, "regla 7 autorizada")
    assert accion.identidad is None and accion.nombre is None, f"la regla 7 no lleva identidad ni nombre: {accion!r}"

    verificar_accion(ag, percibir(agente, "O"), 8, "EVALUADA_AUTORIZADA", contexto="regla 8")
    verificar_accion(ag, percibir(agente, "H", 20), 1, "HUELLA_CARGADA", contexto="regla 1 (intrusa)")
    accion = percibir(agente, "R")
    verificar_accion(ag, accion, 3, "EVALUADA_RECHAZADA", eva.ultimo, "regla 3 rechazada")
    assert accion.autorizado is False, f"regla 3 rechazada: autorizado = {accion.autorizado!r}"
    verificar_accion(ag, percibir(agente, "R"), 7, "EVALUADA_RECHAZADA", eva.ultimo, "regla 7 rechazada")
    verificar_accion(ag, percibir(agente, "I"), 6, "EVALUADA_RECHAZADA", contexto="regla 6")
    verificar_accion(ag, percibir(agente, "HI"), 10, "ESPERANDO_MUESTRA", contexto="regla 10")
    verificar_accion(ag, percibir(agente, "S"), 9, "FIN", contexto="regla 9")
    verificar_accion(ag, percibir(agente, "I"), 0, "FIN", contexto="regla 0")


def test_nunca_se_revela_una_identidad_en_secuencias_al_azar(ag, cfg):
    rng = np.random.default_rng(2000)
    opciones = ["H11", "H12", "H13", "H20", "H21", "HI", "HIV", "R", "I", "O", "S"]
    pesos = [0.06] * 5 + [0.05, 0.05, 0.20, 0.20, 0.17, 0.03]
    nombres = tuple(NOMBRES.values())
    vistas = collections.Counter()
    agente = None
    estado, cargada = "ESPERANDO_MUESTRA", None
    for paso in range(2000):
        if agente is None or (estado == "FIN" and rng.random() < 0.25):
            agente, prep, eva = nuevo_agente(ag, cfg)
            estado, cargada = "ESPERANDO_MUESTRA", None
        op = str(rng.choice(opciones, p=pesos))
        antes = foto(agente)
        if op in ("HI", "HIV"):
            tabla, clave = "HI", None
            accion = percibir(agente, op)
        elif op.startswith("H"):
            tabla, clave = "H", int(op[1:])
            accion = percibir(agente, "H", clave)
        else:
            tabla, clave = op, None
            accion = percibir(agente, op)
        regla, siguiente = oraculo(estado, tabla, cargada in AUTORIZADAS)
        contexto = f"paso {paso}: {estado} + {op}"
        verificar_accion(ag, accion, regla, siguiente, eva.ultimo, contexto)
        verificar_atributos(agente, accion, antes, clave, prep, eva, contexto)
        assert agente.estado == ag.Estado[siguiente], f"[{contexto}] agente.estado = {agente.estado!r}"
        if accion.identidad is not None or accion.nombre is not None:
            assert accion.regla == 5 and accion.estado == ag.Estado.EVALUADA_AUTORIZADA, (
                f"[{contexto}] se reveló una identidad fuera de la regla 5: {accion!r}")
            assert accion.identidad == AUTORIZADAS[cargada], f"[{contexto}] identidad de otra huella: {accion!r}"
        if accion.regla != 5:
            assert not any(n in accion.texto for n in nombres), f"[{contexto}] el texto nombra una identidad: {accion.texto!r}"
        vistas[regla] += 1
        if regla == 1:
            cargada = clave
        elif regla == 10:
            cargada = None
        estado = siguiente
    assert set(vistas) == set(range(11)), f"la secuencia no pasó por todas las reglas: {dict(vistas)}"


def test_llamadas_a_preparar_y_evaluar(ag, cfg):
    agente, prep, eva = nuevo_agente(ag, cfg)

    def paso(percepcion, clave=11, n_prep=0, n_eva=0):
        antes_p, antes_e = len(prep.llamadas), len(eva.llamadas)
        accion = percibir(agente, percepcion, clave)
        dp, de = len(prep.llamadas) - antes_p, len(eva.llamadas) - antes_e
        assert (dp, de) == (n_prep, n_eva), (
            f"regla {accion.regla} ({percepcion}): llamó {dp} veces a preparar y {de} a evaluar; "
            f"esperado {n_prep} y {n_eva}")
        return accion

    assert paso("R").regla == 2 and paso("I").regla == 2 and paso("O").regla == 8
    imagen = imagen_falsa(12)
    accion = agente.percibir_huella(imagen, "id2_v039")
    assert accion.regla == 1 and len(prep.llamadas) == 1, "la regla 1 llama a preparar una vez"
    assert np.array_equal(prep.llamadas[0], imagen), "preparar no recibió la imagen"
    assert paso("I").regla == 4
    assert paso("R", n_eva=1).regla == 3
    assert eva.llamadas[-1] is prep.devueltas[-1], "evaluar no recibió lo que devolvió preparar (regla 3)"
    assert paso("I").regla == 5 and paso("O").regla == 8
    assert paso("R", n_eva=1).regla == 7
    assert eva.llamadas[-1] is prep.devueltas[-1], "evaluar no recibió lo que devolvió preparar (regla 7)"
    assert paso("HI").regla == 10                      # imagen None: no llama a preparar
    assert paso("HIV", n_prep=1).regla == 10           # preparar lanza ValueError
    assert paso("H", 20, n_prep=1).regla == 1
    assert paso("R", n_eva=1).regla == 3 and paso("I").regla == 6
    assert paso("S").regla == 9
    for percepcion in ("H", "HI", "HIV", "R", "I", "S", "O"):
        assert paso(percepcion).regla == 0, f"en FIN, {percepcion} tiene que dar la regla 0"


def test_regla_7_sigue_al_ultimo_resultado(ag, cfg):
    agente, prep, eva = nuevo_agente(ag, cfg)
    percibir(agente, "H", 11)
    eva.decisiones = [1, 0, 1]
    pasos = [("R", 3, "EVALUADA_AUTORIZADA"), ("I", 5, "EVALUADA_AUTORIZADA"), ("R", 7, "EVALUADA_RECHAZADA"),
             ("I", 6, "EVALUADA_RECHAZADA"), ("R", 7, "EVALUADA_AUTORIZADA"), ("I", 5, "EVALUADA_AUTORIZADA")]
    for i, (percepcion, regla, estado) in enumerate(pasos):
        accion = percibir(agente, percepcion)
        verificar_accion(ag, accion, regla, estado, eva.ultimo, f"paso {i}: {percepcion}")
        assert agente.estado == ag.Estado[estado], f"paso {i}: agente.estado = {agente.estado!r}, esperado {estado}"
        if percepcion == "R":
            assert agente.identificada is False, f"paso {i}: identificada tiene que volver a False con Reconocer"
            assert agente.resultado is eva.ultimo, f"paso {i}: el agente no guardó el último resultado"


CASOS_EXCEPCION = [
    ("preparar RuntimeError sin huella", "ESPERANDO_MUESTRA", "preparar", RuntimeError("falla de preparar"), "H"),
    ("preparar RuntimeError con huella", "HUELLA_CARGADA", "preparar", RuntimeError("falla de preparar"), "H"),
    ("preparar KeyError identificada", "EVALUADA_AUTORIZADA", "preparar", KeyError("falla"), "H"),
    ("evaluar RuntimeError en Reconocer", "HUELLA_CARGADA", "evaluar", RuntimeError("falla de evaluar"), "R"),
    ("evaluar ValueError identificada", "EVALUADA_AUTORIZADA", "evaluar", ValueError("falla de evaluar"), "R"),
    ("evaluar RuntimeError rechazada", "EVALUADA_RECHAZADA", "evaluar", RuntimeError("falla de evaluar"), "R"),
]


@pytest.mark.parametrize("caso,estado,quien,excepcion,percepcion", CASOS_EXCEPCION, ids=[c[0] for c in CASOS_EXCEPCION])
def test_excepciones_de_preparar_o_evaluar_no_cambian_el_estado(ag, cfg, caso, estado, quien, excepcion, percepcion):
    agente, prep, eva = nuevo_agente(ag, cfg)
    llevar_a(ag, agente, estado, autoriza=True, identificar=True)
    antes = foto(agente)
    (prep if quien == "preparar" else eva).lanzar = excepcion
    with pytest.raises(type(excepcion)):
        percibir(agente, percepcion, 13)
    comparar_foto(agente, antes, caso)
    (prep if quien == "preparar" else eva).lanzar = None
    regla_i = {"ESPERANDO_MUESTRA": 2, "HUELLA_CARGADA": 4, "EVALUADA_AUTORIZADA": 5, "EVALUADA_RECHAZADA": 6}[estado]
    assert percibir(agente, "I").regla == regla_i, f"[{caso}] el agente no sigue funcionando después de la excepción"


def test_preparar_value_error_aplica_regla_10_y_descarta_la_huella(ag, cfg):
    agente, prep, eva = nuevo_agente(ag, cfg)
    llevar_a(ag, agente, "EVALUADA_AUTORIZADA", identificar=True)
    accion = percibir(agente, "HIV")
    verificar_accion(ag, accion, 10, "ESPERANDO_MUESTRA", contexto="ValueError de preparar")
    assert (agente.huella, agente.imagen, agente.preparada, agente.resultado, agente.identificada) == ("", None, None, None, False), (
        f"la regla 10 no descartó la huella anterior: huella={agente.huella!r}, identificada={agente.identificada!r}")
    assert percibir(agente, "I").regla == 2, "después de la regla 10, Identificar tiene que dar la regla 2"


@pytest.mark.parametrize("comando", ["Reconocer", "hola", ""])
def test_comando_desconocido_lanza_value_error_sin_cambiar_nada(ag, cfg, comando):
    for estado in NO_FIN:
        agente, prep, eva = nuevo_agente(ag, cfg)
        llevar_a(ag, agente, estado, autoriza=True, identificar=True)
        antes = foto(agente)
        n_prep, n_eva = len(prep.llamadas), len(eva.llamadas)
        with pytest.raises(ValueError):
            agente.percibir_comando(comando)
        comparar_foto(agente, antes, f"comando {comando!r} en {estado}")
        assert (len(prep.llamadas), len(eva.llamadas)) == (n_prep, n_eva), f"comando {comando!r}: llamó a preparar o evaluar"


@pytest.mark.parametrize("estado,identidad", [("HUELLA_CARGADA", 4), ("HUELLA_CARGADA", -1), ("EVALUADA_AUTORIZADA", 7)])
def test_identidad_fuera_de_rango_lanza_value_error(ag, cfg, estado, identidad):
    agente, prep, eva = nuevo_agente(ag, cfg)
    llevar_a(ag, agente, estado, autoriza=True, identificar=True)
    antes = foto(agente)
    eva.decisiones = [identidad]
    with pytest.raises(ValueError):
        percibir(agente, "R")
    comparar_foto(agente, antes, f"identidad {identidad} en {estado}")


def test_el_agente_no_modifica_la_imagen(ag, cfg):
    agente, prep, eva = nuevo_agente(ag, cfg)
    imagen = imagen_falsa(12)
    copia = imagen.copy()
    agente.percibir_huella(imagen, "id2_v039")
    for comando in ("reconocer", "identificar", "reconocer", "otro"):
        agente.percibir_comando(comando)
    agente.percibir_huella(imagen, "id2_v039")
    assert np.array_equal(imagen, copia), "el agente modificó la imagen que recibió"
    ilegible = imagen_falsa(ILEGIBLE)
    copia_ilegible = ilegible.copy()
    agente.percibir_huella(ilegible, "no_es_huella.png")
    assert np.array_equal(ilegible, copia_ilegible), "el agente modificó la imagen ilegible que recibió"


# ===========================================================================
# 6.3 y 6.9 Controlador con dobles de prueba
# ===========================================================================

def visor_falso(vis):
    media = np.array([0.1, -0.2, 0.3, 0.05, -0.4])
    componentes = np.array([[0.6, 0.0, 0.8, 0.0, 0.0], [0.0, 0.8, 0.0, -0.6, 0.0]])
    return vis.VisorPCA(media=media, componentes=componentes, proporcion=np.array([0.5, 0.3]),
                        puntos=np.zeros((4, 2)), identidades=np.array([1, 2, 3, 1], dtype=np.int64),
                        centroides=np.zeros((3, 2)))


@pytest.fixture
def entorno(tmp_path, cfg, mue, vis, ctl):
    """PNG falsos, dobles y una fábrica de controladores con ellos."""
    carpeta = tmp_path / "huellas"
    carpeta.mkdir()

    def png(stem, clave):
        ruta = carpeta / f"{stem}.png"
        escribir_png(ruta, imagen_falsa(clave))
        return ruta

    E = mue.EntradaHuella
    e = {
        "id1": E(nombre="id1_v040 (reservada)", origen="reservada", verdad=1, ruta=png("id1_v040", 11), variante=40),
        "id2": E(nombre="id2_v039 (reservada)", origen="reservada", verdad=2, ruta=png("id2_v039", 12), variante=39),
        "id3": E(nombre="id3_v039 (reservada)", origen="reservada", verdad=3, ruta=png("id3_v039", 13), variante=39),
        "intruso": E(nombre="intruso31_v1", origen="intruso_test", verdad=0, ruta=png("intruso31_v1", 20), variante=1),
        "intruso2": E(nombre="intruso32_v1", origen="intruso_test", verdad=0, ruta=png("intruso32_v1", 21), variante=1),
        "no_huella": E(nombre="no_es_huella.png", origen="archivo", verdad=None, ruta=png("no_es_huella", ILEGIBLE), variante=None),
    }
    rota = carpeta / "rota.png"
    rota.write_bytes(BASURA)
    e["rota"] = E(nombre="rota.png", origen="archivo", verdad=None, ruta=rota, variante=None)
    x = SimpleNamespace(
        cfg=cfg, mue=mue, ctl=ctl, tmp=tmp_path, e=e, visor=visor_falso(vis),
        clave={"id1": 11, "id2": 12, "id3": 13, "intruso": 20, "intruso2": 21},
        prep=PrepararFalso(), eva=EvaluarFalso(), voz=VozFalsa(), gen=GeneradorFalso(mue),
    )

    def crear(**kw):
        kw.setdefault("visor", x.visor)
        kw.setdefault("generador", x.gen)
        kw.setdefault("entradas", list(e.values()))
        return ctl.Controlador(cfg, x.prep, x.eva, x.voz, **kw)

    x.crear = crear
    return x


def hu(x, clave_entrada: str) -> tuple[str, int]:
    """(nombre, clave) de una entrada del entorno."""
    return x.e[clave_entrada].nombre, x.clave[clave_entrada]


def decir(c, x, comando: str, valido: bool = True, motivo: str | None = None):
    x.voz.cola.append(rec_falso(comando, valido, motivo))
    return c.comando_audio((np.arange(1600) / 1600).astype(np.float32), 16000)


def ver(p, x, *, estado, regla, mensaje, aviso, huella=None, identificacion="", con_visor=True,
        cerrar=None, microfono=None, grabando=None, comando=None, contexto=""):
    """Compara la Pantalla con 4.6. `huella` es (nombre, clave) de la huella cargada o None."""
    pre = f"[{contexto}] " if contexto else ""

    def igual(campo, esperado):
        obtenido = getattr(p, campo)
        assert obtenido == esperado, f"{pre}Pantalla.{campo} = {obtenido!r}, esperado {esperado!r}"

    assert isinstance(p.estado, str), f"{pre}Pantalla.estado no es str: {p.estado!r}"
    igual("estado", estado)
    igual("regla", regla)
    igual("mensaje", mensaje)
    igual("aviso", aviso)
    igual("identificacion", identificacion)
    cartel, color = {"EVALUADA_AUTORIZADA": (AUTORIZADO_C, "verde"),
                     "EVALUADA_RECHAZADA": (NO_AUTORIZADO_C, "rojo")}.get(estado, ("", "neutro"))
    igual("cartel", cartel)
    igual("color", color)
    if huella is None:
        igual("huella", "")
        assert p.imagen is None and p.esqueleto is None, f"{pre}sin huella, imagen y esqueleto tienen que ser None"
    else:
        nombre, clave = huella
        igual("huella", nombre)
        assert p.imagen is not None and np.array_equal(p.imagen, imagen_falsa(clave)), f"{pre}Pantalla.imagen no es la huella cargada"
        assert p.esqueleto is not None and p.esqueleto.dtype == bool and np.array_equal(p.esqueleto, esqueleto_falso(clave)), (
            f"{pre}Pantalla.esqueleto no es el del preparar falso")
    if estado in EVALUADAS:
        resultado = x.eva.ultimo
        assert p.distancia == pytest.approx(float(resultado.distancia), abs=1e-12), (
            f"{pre}Pantalla.distancia = {p.distancia!r}, esperado {float(resultado.distancia)!r}")
        assert p.tau == pytest.approx(float(resultado.tau), abs=1e-12), f"{pre}Pantalla.tau = {p.tau!r}"
        if con_visor and hasattr(resultado, "z"):
            punto = (np.asarray(resultado.z) - x.visor.media) @ x.visor.componentes.T
            assert isinstance(p.punto_pca, tuple) and len(p.punto_pca) == 2, f"{pre}punto_pca = {p.punto_pca!r}"
            assert all(isinstance(c, float) for c in p.punto_pca), f"{pre}punto_pca no son float: {p.punto_pca!r}"
            assert np.allclose(p.punto_pca, punto, rtol=0, atol=1e-12), f"{pre}punto_pca = {p.punto_pca!r}, esperado {tuple(punto)!r}"
        else:
            assert p.punto_pca is None, f"{pre}punto_pca = {p.punto_pca!r}, esperado None"
    else:
        assert p.distancia is None and p.tau is None and p.punto_pca is None, (
            f"{pre}fuera de EVALUADA_*, distancia, tau y punto_pca son None: {p.distancia!r}, {p.tau!r}, {p.punto_pca!r}")
    igual("cerrar", estado == "FIN" if cerrar is None else cerrar)
    if microfono is not None:
        igual("microfono", microfono)
        if not microfono:
            assert p.nivel_db is None, f"{pre}sin micrófono, nivel_db = {p.nivel_db!r}, esperado None"
    if grabando is not None:
        igual("grabando", grabando)
    if comando is not None:
        igual("comando", comando)


def sin_nombres_de_identidad(p, contexto: str) -> None:
    for campo in dataclasses.fields(p):
        valor = getattr(p, campo.name)
        if isinstance(valor, str):
            for nombre in NOMBRES.values():
                assert nombre not in valor, f"[{contexto}] Pantalla.{campo.name} = {valor!r} nombra {nombre!r}"


def llevar_controlador(c, x, estado: str, identificar: bool = False):
    """Lleva al controlador a `estado`; devuelve (huella, identificacion) de la pantalla."""
    if estado == "ESPERANDO_MUESTRA":
        return None, ""
    if estado == "HUELLA_CARGADA":
        c.elegir(x.e["id1"])
        return hu(x, "id1"), ""
    if estado == "EVALUADA_AUTORIZADA":
        c.elegir(x.e["id3"])
        p = decir(c, x, "reconocer")
        if identificar:
            p = decir(c, x, "identificar")
        assert p.estado == estado, f"no se pudo llevar al controlador a {estado}: {p.estado!r}"
        return hu(x, "id3"), NOMBRES[3] if identificar else ""
    c.elegir(x.e["intruso"])
    p = decir(c, x, "reconocer")
    assert p.estado == estado, f"no se pudo llevar al controlador a {estado}: {p.estado!r}"
    return hu(x, "intruso"), ""


def test_controlador_construccion_y_pantalla_inicial(entorno, ag):
    from src.agente.captura import Microfono

    x = entorno
    fab = FabricaFalsa()
    mic = Microfono(x.cfg, fs=1000, fabrica=fab)
    entradas = list(x.e.values())
    c = x.crear(entradas=entradas, microfono=mic)
    assert isinstance(c.agente, ag.Agente), f"Controlador.agente = {c.agente!r}"
    assert c.entradas == entradas, "Controlador.entradas no es la lista que se pasó"
    assert fab.llamadas == [], "el controlador abrió el micrófono al crearse"
    p = c.pantalla()
    ver(p, x, estado="ESPERANDO_MUESTRA", regla=0, mensaje=AVISOS_C["inicio"], aviso=False, microfono=False,
        grabando=False, comando="", contexto="pantalla inicial")
    assert p.detalle_voz == "", f"detalle_voz inicial = {p.detalle_voz!r}"
    sin_entradas = x.ctl.Controlador(x.cfg, x.prep, x.eva, x.voz)
    assert sin_entradas.entradas == [], f"sin entradas, Controlador.entradas = {sin_entradas.entradas!r}"


def test_escenario_1_huella_autorizada(entorno):
    x = entorno
    c = x.crear()
    p = c.elegir(x.e["id2"])
    ver(p, x, estado="HUELLA_CARGADA", regla=1, mensaje=TEXTOS_C[1], aviso=False, huella=hu(x, "id2"), contexto="elegir")
    p = decir(c, x, "reconocer")
    ver(p, x, estado="EVALUADA_AUTORIZADA", regla=3, mensaje=AUTORIZADO_C, aviso=False, huella=hu(x, "id2"),
        comando="reconocer", contexto="reconocer")
    p = decir(c, x, "identificar")
    ver(p, x, estado="EVALUADA_AUTORIZADA", regla=5, mensaje=NOMBRES[2], aviso=False, huella=hu(x, "id2"),
        identificacion=NOMBRES[2], comando="identificar", contexto="identificar")


def test_escenario_2_huella_intrusa(entorno):
    x = entorno
    c = x.crear()
    p = c.elegir(x.e["intruso"])
    ver(p, x, estado="HUELLA_CARGADA", regla=1, mensaje=TEXTOS_C[1], aviso=False, huella=hu(x, "intruso"), contexto="elegir")
    p = decir(c, x, "reconocer")
    ver(p, x, estado="EVALUADA_RECHAZADA", regla=3, mensaje=NO_AUTORIZADO_C, aviso=False, huella=hu(x, "intruso"), contexto="reconocer")
    sin_nombres_de_identidad(p, "reconocer intrusa")
    p = decir(c, x, "identificar")
    ver(p, x, estado="EVALUADA_RECHAZADA", regla=6, mensaje=TEXTOS_C[6], aviso=True, huella=hu(x, "intruso"), contexto="identificar")
    sin_nombres_de_identidad(p, "identificar intrusa")


def test_escenario_3_cambio_de_huella(entorno):
    x = entorno
    c = x.crear()
    c.elegir(x.e["id1"])
    decir(c, x, "reconocer")
    p = decir(c, x, "identificar")
    ver(p, x, estado="EVALUADA_AUTORIZADA", regla=5, mensaje=NOMBRES[1], aviso=False, huella=hu(x, "id1"),
        identificacion=NOMBRES[1], contexto="identificar autorizada")
    p = c.elegir(x.e["intruso"])
    ver(p, x, estado="HUELLA_CARGADA", regla=1, mensaje=TEXTOS_C[1], aviso=False, huella=hu(x, "intruso"), contexto="elegir intrusa")
    sin_nombres_de_identidad(p, "después de cambiar de huella")
    p = decir(c, x, "identificar")
    ver(p, x, estado="HUELLA_CARGADA", regla=4, mensaje=TEXTOS_C[4], aviso=True, huella=hu(x, "intruso"), contexto="identificar sin reconocer")
    sin_nombres_de_identidad(p, "identificar sin reconocer")
    p = decir(c, x, "reconocer")
    ver(p, x, estado="EVALUADA_RECHAZADA", regla=3, mensaje=NO_AUTORIZADO_C, aviso=False, huella=hu(x, "intruso"), contexto="reconocer intrusa")
    p = decir(c, x, "identificar")
    ver(p, x, estado="EVALUADA_RECHAZADA", regla=6, mensaje=TEXTOS_C[6], aviso=True, huella=hu(x, "intruso"), contexto="identificar intrusa")
    sin_nombres_de_identidad(p, "identificar intrusa")


def test_escenario_4_comandos_sin_huella(entorno):
    x = entorno
    c = x.crear()
    for comando in ("reconocer", "identificar"):
        p = decir(c, x, comando)
        ver(p, x, estado="ESPERANDO_MUESTRA", regla=2, mensaje=TEXTOS_C[2], aviso=True, comando=comando,
            contexto=f"{comando} sin huella")
    assert x.eva.llamadas == [] and x.prep.llamadas == [], "sin huella no se llama a preparar ni a evaluar"


@pytest.mark.parametrize("estado", NO_FIN)
def test_escenario_4_ruido_o_comando_otro(entorno, estado):
    x = entorno
    c = x.crear()
    huella, identificacion = llevar_controlador(c, x, estado, identificar=True)
    p = decir(c, x, "otro")
    ver(p, x, estado=estado, regla=8, mensaje=TEXTOS_C[8], aviso=True, huella=huella, identificacion=identificacion,
        comando="otro", contexto=f"otro válido en {estado}")
    p = decir(c, x, "otro", valido=False, motivo="sin_voz")
    ver(p, x, estado=estado, regla=8, mensaje=TEXTOS_C[8], aviso=True, huella=huella, identificacion=identificacion,
        comando="otro", contexto=f"audio no válido en {estado}")
    assert "sin_voz" in p.detalle_voz, f"detalle_voz no nombra el motivo: {p.detalle_voz!r}"


@pytest.mark.parametrize("estado", NO_FIN)
def test_escenario_5_salir_cierra_y_despues_no_llama_a_nada(entorno, monkeypatch, estado):
    from src.agente.captura import Microfono
    import src.agente.percepcion as percepcion
    import src.voz.audio as audio

    x = entorno
    fab = FabricaFalsa()
    c = x.crear(microfono=Microfono(x.cfg, fs=1000, fabrica=fab))
    p = c.abrir_microfono()
    assert p.microfono is True, f"abrir_microfono con la fábrica falsa: microfono = {p.microfono!r}"
    huella, _ = llevar_controlador(c, x, estado, identificar=True)
    p = decir(c, x, "salir")
    # identificacion es "" fuera de EVALUADA_AUTORIZADA, también después de identificar (6.9)
    ver(p, x, estado="FIN", regla=9, mensaje=TEXTOS_C[9], aviso=False, huella=huella, identificacion="",
        cerrar=True, microfono=False, comando="salir", contexto=f"salir desde {estado}")
    stream = fab.streams[0]
    assert (stream.llamadas["stop"], stream.llamadas["close"]) == (1, 1), f"stop/close del micrófono: {dict(stream.llamadas)}"

    leidas_im = espiar(monkeypatch, "leer_imagen", (percepcion, x.ctl))
    leidos_au = espiar(monkeypatch, "leer_audio", (audio, x.ctl))
    wav = x.tmp / "salir.wav"
    sf.write(wav, np.zeros(1600, dtype=np.float32), 16000)
    n = (len(x.prep.llamadas), len(x.eva.llamadas), len(x.voz.llamadas), len(x.gen.llamadas), len(fab.llamadas))
    eventos = [("elegir", lambda: c.elegir(x.e["id2"])), ("abrir_archivo", lambda: c.abrir_archivo(x.e["id2"].ruta)),
               ("variante_nueva", lambda: c.variante_nueva(1)), ("intruso_nuevo", lambda: c.intruso_nuevo()),
               ("comando_audio", lambda: c.comando_audio(np.zeros(1600, np.float32), 16000)),
               ("comando_wav", lambda: c.comando_wav(wav)), ("abrir_microfono", lambda: c.abrir_microfono()),
               ("apretar", lambda: c.apretar()), ("soltar", lambda: c.soltar()), ("actualizar", lambda: c.actualizar()),
               ("cerrar", lambda: c.cerrar()), ("pantalla", lambda: c.pantalla())]
    for nombre, evento in eventos:
        p = evento()
        ver(p, x, estado="FIN", regla=9, mensaje=TEXTOS_C[9], aviso=False, huella=huella, cerrar=True,
            microfono=False, contexto=f"{nombre} después de salir")
    ahora = (len(x.prep.llamadas), len(x.eva.llamadas), len(x.voz.llamadas), len(x.gen.llamadas), len(fab.llamadas))
    assert ahora == n, f"después de salir se llamó a algo (preparar, evaluar, voz, generador, fábrica): {n} -> {ahora}"
    assert leidas_im == [] and leidos_au == [], f"después de salir se leyeron archivos: {leidas_im} {leidos_au}"
    assert (stream.llamadas["stop"], stream.llamadas["close"]) == (1, 1), f"stop/close se repitieron: {dict(stream.llamadas)}"


def test_pantalla_punto_pca_sin_visor_o_sin_z(entorno):
    x = entorno
    c = x.crear(visor=None)
    c.elegir(x.e["id1"])
    p = decir(c, x, "reconocer")
    ver(p, x, estado="EVALUADA_AUTORIZADA", regla=3, mensaje=AUTORIZADO_C, aviso=False, huella=hu(x, "id1"),
        con_visor=False, contexto="sin visor")
    x.eva.sin_z = True
    c = x.crear()
    c.elegir(x.e["intruso"])
    p = decir(c, x, "reconocer")
    ver(p, x, estado="EVALUADA_RECHAZADA", regla=3, mensaje=NO_AUTORIZADO_C, aviso=False, huella=hu(x, "intruso"),
        con_visor=False, contexto="resultado sin z")


def test_controlador_variante_e_intruso_nuevos(entorno):
    x = entorno
    c = x.crear()
    p = c.variante_nueva(2)
    ver(p, x, estado="HUELLA_CARGADA", regla=1, mensaje=TEXTOS_C[1], aviso=False, huella=("id2_v1000 (nueva)", 12), contexto="variante_nueva(2)")
    p = decir(c, x, "reconocer")
    ver(p, x, estado="EVALUADA_AUTORIZADA", regla=3, mensaje=AUTORIZADO_C, aviso=False, huella=("id2_v1000 (nueva)", 12), contexto="reconocer variante")
    p = c.intruso_nuevo()
    ver(p, x, estado="HUELLA_CARGADA", regla=1, mensaje=TEXTOS_C[1], aviso=False, huella=("intruso1000_v1 (nuevo)", 21), contexto="intruso_nuevo")
    assert x.gen.llamadas == [("variante_nueva", 2), ("intruso_nuevo",)], f"llamadas al generador: {x.gen.llamadas}"


def test_controlador_errores_sin_excepciones(entorno):
    x = entorno
    c = x.crear()
    c.elegir(x.e["id1"])
    x.eva.lanzar = RuntimeError("falla")
    p = decir(c, x, "reconocer")
    ver(p, x, estado="HUELLA_CARGADA", regla=0, mensaje="Error inesperado: RuntimeError: falla", aviso=True,
        huella=hu(x, "id1"), contexto="evaluar lanza")
    p = decir(c, x, "identificar")
    ver(p, x, estado="HUELLA_CARGADA", regla=4, mensaje=TEXTOS_C[4], aviso=True, huella=hu(x, "id1"), contexto="evento siguiente")
    x.eva.lanzar = None
    p = decir(c, x, "reconocer")
    ver(p, x, estado="EVALUADA_AUTORIZADA", regla=3, mensaje=AUTORIZADO_C, aviso=False, huella=hu(x, "id1"), contexto="reconocer de nuevo")

    x.voz.cola.append(None)  # reconocer_voz devuelve algo sin los campos de V4: el error se atrapa
    p = c.comando_audio(np.zeros(100, np.float32), 16000)
    assert p.regla == 0 and p.aviso is True and p.mensaje.startswith("Error inesperado: "), f"error de voz: {p.mensaje!r}"
    assert p.estado == "EVALUADA_AUTORIZADA", f"el error de voz cambió el estado: {p.estado!r}"


def test_controlador_audio_ilegible_no_llama_a_reconocer_voz(entorno):
    x = entorno
    c = x.crear()
    basura = x.tmp / "basura.wav"
    basura.write_bytes(BASURA)
    for ruta in (x.tmp / "no_existe.wav", basura):
        p = c.comando_wav(ruta)
        ver(p, x, estado="ESPERANDO_MUESTRA", regla=0, mensaje=AVISOS_C["audio_ilegible"], aviso=True, contexto=f"comando_wav({ruta.name})")
    assert x.voz.llamadas == [], "con un audio ilegible no se llama a reconocer_voz"


def test_controlador_png_roto_aplica_regla_10(entorno):
    x = entorno
    c = x.crear()
    c.elegir(x.e["id1"])
    p = c.abrir_archivo(x.e["rota"].ruta)
    ver(p, x, estado="ESPERANDO_MUESTRA", regla=10, mensaje=TEXTOS_C[10], aviso=True, contexto="abrir_archivo(rota.png)")
    p = c.abrir_archivo(x.tmp / "no_existe.png")
    ver(p, x, estado="ESPERANDO_MUESTRA", regla=10, mensaje=TEXTOS_C[10], aviso=True, contexto="abrir_archivo(no_existe.png)")
    p = c.elegir(x.e["no_huella"])
    ver(p, x, estado="ESPERANDO_MUESTRA", regla=10, mensaje=TEXTOS_C[10], aviso=True, contexto="PNG que preparar rechaza")
    p = c.abrir_archivo(x.e["id2"].ruta)
    ver(p, x, estado="HUELLA_CARGADA", regla=1, mensaje=TEXTOS_C[1], aviso=False, huella=("id2_v039.png", 12),
        contexto="abrir_archivo de un PNG bueno")


def test_controlador_variante_nueva_sin_generador_avisa_error(entorno):
    x = entorno
    c = x.crear(generador=None)
    for evento in (lambda: c.variante_nueva(1), lambda: c.intruso_nuevo()):
        p = evento()
        assert p.mensaje.startswith("Error inesperado: "), f"sin generador, mensaje = {p.mensaje!r}"
        assert p.aviso is True and p.regla == 0 and p.estado == "ESPERANDO_MUESTRA", f"sin generador: {p!r}"
    x.gen.llamadas.clear()
    c = x.crear()
    p = c.variante_nueva(4)
    assert p.mensaje.startswith("Error inesperado: ValueError: "), f"variante_nueva(4): mensaje = {p.mensaje!r}"
    p = c.variante_nueva(1)
    assert p.regla == 1, "después de un error, variante_nueva(1) tiene que funcionar"


def test_controlador_comando_wav_pasa_lo_de_leer_audio(entorno):
    from src.voz.audio import leer_audio

    x = entorno
    c = x.crear()
    rng = np.random.default_rng(48)
    y = (0.25 * rng.standard_normal(4800)).clip(-1, 1).astype(np.float32)
    ruta = x.tmp / "comando ñ.wav"
    sf.write(ruta, y, 48000)
    y_ref, fs_ref = leer_audio(ruta)
    x.voz.cola.append(rec_falso("identificar"))
    p = c.comando_wav(ruta)
    assert len(x.voz.llamadas) == 1, f"reconocer_voz se llamó {len(x.voz.llamadas)} veces"
    y_rec, fs_rec = x.voz.llamadas[0]
    assert fs_rec == fs_ref == 48000, f"fs pasado = {fs_rec!r}, leer_audio dio {fs_ref!r}"
    assert y_rec.dtype == y_ref.dtype and np.array_equal(y_rec, y_ref), "el audio pasado no es el de leer_audio"
    ver(p, x, estado="ESPERANDO_MUESTRA", regla=2, mensaje=TEXTOS_C[2], aviso=True, comando="identificar", contexto="comando_wav")
    assert "identificar" in p.detalle_voz, f"detalle_voz no nombra la predicción: {p.detalle_voz!r}"


@pytest.mark.parametrize("caso", ["sin_microfono", "falla_al_crear", "falla_al_arrancar"])
def test_controlador_sin_microfono_avisa(entorno, caso):
    from src.agente.captura import Microfono

    x = entorno
    mic = None
    if caso != "sin_microfono":
        fab = FabricaFalsa(falla_al_crear=caso == "falla_al_crear", falla_al_arrancar=caso == "falla_al_arrancar")
        mic = Microfono(x.cfg, fs=1000, fabrica=fab)
    c = x.crear(microfono=mic)
    for nombre, evento in (("abrir_microfono", c.abrir_microfono), ("apretar", c.apretar), ("soltar", c.soltar)):
        p = evento()
        ver(p, x, estado="ESPERANDO_MUESTRA", regla=0, mensaje=AVISOS_C["sin_microfono"], aviso=True,
            microfono=False, grabando=False, contexto=f"{caso}: {nombre}")
    p = c.actualizar()
    assert p.microfono is False and p.nivel_db is None, f"{caso}: actualizar sin micrófono: {p!r}"
    assert x.voz.llamadas == [], f"{caso}: se llamó a reconocer_voz sin micrófono"


@pytest.fixture
def controlador_con_microfono(entorno):
    from src.agente.captura import Microfono

    x = entorno
    fab = FabricaFalsa()
    mic = Microfono(x.cfg, fs=1000, fabrica=fab)
    c = x.crear(microfono=mic)
    p = c.abrir_microfono()
    assert p.microfono is True and p.aviso is False, f"abrir_microfono con la fábrica falsa: {p!r}"
    assert p.nivel_db == pytest.approx(db(np.zeros(0)), abs=1e-9), f"nivel con el búfer vacío: {p.nivel_db!r}"
    return x, c, mic, fab, np.arange(20000, dtype=np.float32)


def test_controlador_push_to_talk_con_fabrica_falsa(controlador_con_microfono):
    x, c, mic, fab, rampa = controlador_con_microfono
    x.voz.cola.append(rec_falso("reconocer"))
    alimentar(mic, fab, rampa, 5000)
    p = c.actualizar()
    assert p.grabando is False and x.voz.llamadas == [], "sin apretar no se graba ni se reconoce"
    assert p.nivel_db == pytest.approx(db(rampa[4950:5000]), abs=1e-9), f"nivel_db = {p.nivel_db!r}"
    p = c.apretar()
    assert p.grabando is True and p.microfono is True, f"después de apretar: grabando = {p.grabando!r}"
    alimentar(mic, fab, rampa, 6200)
    p = c.actualizar()
    assert p.grabando is True and x.voz.llamadas == [], "mientras se graba, actualizar no reconoce"
    p = c.soltar()
    assert p.grabando is False, "después de soltar, grabando tiene que ser False"
    for hasta in (6300, 6349):
        alimentar(mic, fab, rampa, hasta)
        c.actualizar()
        assert x.voz.llamadas == [], f"actualizar con {hasta} muestras (antes del post-roll) llamó a reconocer_voz"
    alimentar(mic, fab, rampa, 6350)
    p = c.actualizar()
    assert len(x.voz.llamadas) == 1, f"después del post-roll, reconocer_voz se llamó {len(x.voz.llamadas)} veces"
    y, fs = x.voz.llamadas[0]
    assert fs == 1000, f"fs pasado = {fs!r}"
    assert np.array_equal(y, rampa[4750:6350]), (
        f"el clip no es la ventana [4750, 6350): largo {len(y)}, primeras {y[:2]}, últimas {y[-2:]}")
    assert p.comando == "reconocer" and p.regla == 2, f"comando = {p.comando!r}, regla = {p.regla!r}"
    c.actualizar()
    assert len(x.voz.llamadas) == 1, "el mismo clip se reconoció dos veces"


def test_controlador_push_to_talk_corta_en_el_tope(controlador_con_microfono):
    x, c, mic, fab, rampa = controlador_con_microfono
    alimentar(mic, fab, rampa, 5000)
    c.apretar()
    alimentar(mic, fab, rampa, 7999)
    p = c.actualizar()
    assert p.grabando is True, "antes de los 3 s no se corta"
    alimentar(mic, fab, rampa, 8000)
    p = c.actualizar()
    assert p.grabando is False, "a los 3 s (3000 muestras) actualizar tiene que cortar"
    assert x.voz.llamadas == [], "el clip no está completo hasta el post-roll"
    alimentar(mic, fab, rampa, 8150)
    c.actualizar()
    assert len(x.voz.llamadas) == 1, f"reconocer_voz se llamó {len(x.voz.llamadas)} veces"
    y, fs = x.voz.llamadas[0]
    assert fs == 1000 and np.array_equal(y, rampa[4750:8150]), f"el clip con tope no es [4750, 8150): largo {len(y)}"


def test_controlador_cerrar_se_puede_repetir(entorno):
    from src.agente.captura import Microfono

    x = entorno
    fab = FabricaFalsa()
    c = x.crear(microfono=Microfono(x.cfg, fs=1000, fabrica=fab))
    c.abrir_microfono()
    c.elegir(x.e["id1"])
    for i in range(3):
        p = c.cerrar()
        assert p.cerrar is True, f"cerrar() #{i + 1}: Pantalla.cerrar = {p.cerrar!r}"
        assert p.microfono is False, f"cerrar() #{i + 1}: el micrófono sigue abierto"
    stream = fab.streams[0]
    assert (stream.llamadas["stop"], stream.llamadas["close"]) == (1, 1), f"stop/close: {dict(stream.llamadas)}"
    n = len(x.prep.llamadas)
    p = c.elegir(x.e["id2"])
    assert p.cerrar is True and len(x.prep.llamadas) == n, "después de cerrar(), elegir no hace nada"


def test_controlador_registro_csv(entorno):
    x = entorno
    ruta = x.tmp / "app" / "registro.csv"
    c = x.crear(registro=ruta)
    assert ruta.parent.is_dir() and ruta.exists(), "el controlador no creó la carpeta y el registro al construirse"
    eventos = (
        ("huella", x.e["id1"].nombre, lambda: c.elegir(x.e["id1"])),
        ("comando", "reconocer", lambda: decir(c, x, "reconocer")),
        ("comando", "identificar", lambda: decir(c, x, "identificar")),
        ("aviso", AVISOS_C["audio_ilegible"], lambda: c.comando_wav(x.tmp / "no_existe.wav")),
        ("comando", "otro", lambda: decir(c, x, "otro", valido=False, motivo="sin_voz")),
        ("huella", "rota.png", lambda: c.abrir_archivo(x.e["rota"].ruta)),
        ("aviso", AVISOS_C["sin_microfono"], lambda: c.apretar()),
        ("comando", "salir", lambda: decir(c, x, "salir")),
    )
    pantallas = [c.pantalla()]
    for _, _, evento in eventos:
        pantallas.append(evento())
    c.cerrar()
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.reader(f))
    assert filas and filas[0] == ENCABEZADO_REGISTRO, f"encabezado del registro: {filas[:1]}"
    datos = [dict(zip(filas[0], fila)) for fila in filas[1:]]
    # La pantalla inicial no viene de un evento; si el registro la anota, se ignora esa fila
    if datos and datos[0]["mensaje"] == AVISOS_C["inicio"]:
        datos = datos[1:]
    assert len(datos) == len(eventos), f"el registro tiene {len(datos)} filas, se esperaban {len(eventos)}: {datos}"
    for i, (fila, (percepcion, detalle, _)) in enumerate(zip(datos, eventos)):
        antes, despues = pantallas[i], pantallas[i + 1]
        contexto = f"fila {i + 1} ({percepcion} {detalle!r})"
        datetime.datetime.fromisoformat(fila["momento"])
        assert fila["percepcion"] == percepcion, f"{contexto}: percepcion = {fila['percepcion']!r}"
        assert fila["regla"] == str(despues.regla), f"{contexto}: regla = {fila['regla']!r}, pantalla {despues.regla}"
        assert fila["estado_antes"] == antes.estado, f"{contexto}: estado_antes = {fila['estado_antes']!r}, pantalla {antes.estado!r}"
        assert fila["estado_despues"] == despues.estado, f"{contexto}: estado_despues = {fila['estado_despues']!r}, pantalla {despues.estado!r}"
        assert fila["mensaje"] == despues.mensaje, f"{contexto}: mensaje = {fila['mensaje']!r}, pantalla {despues.mensaje!r}"
        if percepcion == "comando":
            assert detalle in fila["detalle"], f"{contexto}: detalle {fila['detalle']!r} no nombra el comando"
        else:
            assert fila["detalle"] == detalle, f"{contexto}: detalle = {fila['detalle']!r}, esperado {detalle!r}"


def test_desde_archivos_sin_modelos_lanza_file_not_found(ctl, cfg, tmp_path):
    modelos = tmp_path / "modelos"
    modelos.mkdir()
    with pytest.raises(FileNotFoundError) as error:
        ctl.Controlador.desde_archivos(cfg, modelos=modelos, huellas=tmp_path / "huellas", microfono=False)
    mensaje = str(error.value)
    assert "modelo_huellas.npz" in mensaje or "modelo_voz.npz" in mensaje, f"el error no nombra el modelo: {mensaje!r}"
    assert "--sobrescribir" in mensaje, f"el error no dice cómo regenerarlo: {mensaje!r}"


# ===========================================================================
# 6.4 Percepción de la huella
# ===========================================================================

@pytest.fixture(scope="module")
def receta(cfg):
    """La receta de H1 (0.3 y 4.4), con las bases guardadas por (tipo, semilla)."""
    from src.huellas.generador import generar_base
    from src.huellas.variantes import aplicar_variante, sortear_parametros

    bases = {}

    def base(tipo, semilla):
        if (tipo, semilla) not in bases:
            bases[(tipo, semilla)] = generar_base(tipo, semilla, cfg)
        return bases[(tipo, semilla)]

    def variante(tipo, semilla, v):
        rng = np.random.default_rng([semilla, v])
        return aplicar_variante(base(tipo, semilla), sortear_parametros(rng, cfg), rng).imagen

    def de_identidad(k, v):
        ident = identidad_cfg(cfg, k)
        return variante(ident["tipo"], int(ident["semilla"]), v)

    def tipo_semilla_intruso(i):
        tipos = cfg["huellas"]["intrusos"]["tipos"]
        return tipos[(i - 1) % len(tipos)], int(cfg["huellas"]["intrusos"]["semilla_base"]) + i

    def de_intruso(i, v=1):
        return variante(*tipo_semilla_intruso(i), v)

    return SimpleNamespace(base=base, identidad=de_identidad, intruso=de_intruso, tipo_semilla_intruso=tipo_semilla_intruso)


@pytest.fixture(scope="module")
def huellas_chicas(cfg, receta):
    """2 variantes de cada identidad y un intruso, con un ModeloHuellas hecho a mano (sección 5)."""
    from src.huellas.caracteristicas import BLOQUES, ajustar_escalador, caracteristicas
    from src.huellas.clasificacion import ModeloHuellas
    from src.huellas.preprocesamiento import preprocesar

    imagenes, verdades = [], []
    for k in (1, 2, 3):
        for v in (1, 2):
            imagenes.append(receta.identidad(k, v))
            verdades.append(k)
    imagenes.append(receta.intruso(1000))
    verdades.append(0)
    preparadas = [preprocesar(im, cfg) for im in imagenes]
    bloques = [caracteristicas(h, cfg) for h in preparadas]
    apilados = {b: np.stack([bl[b] for bl in bloques]) for b in BLOQUES}
    escalador = ajustar_escalador(apilados)
    Z = np.asarray(escalador.transformar(apilados), dtype=np.float64)
    verdades = np.array(verdades)
    ids = np.array([1, 2, 3], dtype=np.int64)
    centroides = np.stack([Z[verdades == k].mean(axis=0) for k in ids])

    def modelo(tau, escalador=escalador):
        return ModeloHuellas(identidades=ids, centroides=centroides, dispersion=np.ones(3), tau=float(tau),
                             pureza=1.0, inercia=0.0, escalador=escalador)

    distancias = np.array([float(modelo(1.0).puntajes(z)[0]) for z in Z])
    unicas = np.unique(np.round(distancias, 9))
    medio = len(unicas) // 2
    tau = (unicas[medio - 1] + unicas[medio]) / 2  # acepta unas y rechaza otras
    return SimpleNamespace(imagenes=imagenes, verdades=verdades, escalador=escalador, modelo=modelo(tau),
                           distancias=distancias, tau=tau)


def test_evaluar_huella_igual_a_reconocer_de_h4(cfg, per, huellas_chicas):
    from src.huellas.caracteristicas import caracteristicas
    from src.huellas.clasificacion import reconocer

    hc = huellas_chicas
    decisiones = []
    for i, imagen in enumerate(hc.imagenes):
        r = reconocer(imagen, hc.modelo, cfg)
        preparada = per.preparar_huella(imagen, cfg)
        e = per.evaluar_huella(preparada, hc.modelo, cfg)
        for campo in ("identidad", "mas_cercana", "distancia", "tau"):
            assert getattr(e, campo) == getattr(r, campo), (
                f"huella {i}: evaluar_huella.{campo} = {getattr(e, campo)!r}, reconocer de H4 dio {getattr(r, campo)!r}")
        z = np.asarray(hc.escalador.transformar(caracteristicas(preparada, cfg)))
        assert isinstance(e.z, np.ndarray) and e.z.dtype == np.float64 and e.z.shape == z.shape and e.z.ndim == 1, (
            f"huella {i}: z tiene que ser float64 de (d,): {getattr(e.z, 'dtype', None)} {getattr(e.z, 'shape', None)}")
        assert np.array_equal(e.z, z), f"huella {i}: z no es escalador.transformar(caracteristicas(...))"
        decisiones.append(int(e.identidad))
    assert any(d != 0 for d in decisiones) and any(d == 0 for d in decisiones), (
        f"con τ = {hc.tau:.4f} el test tenía que aceptar unas y rechazar otras: {decisiones}")


def test_evaluar_huella_sin_escalador_lanza(cfg, per, huellas_chicas):
    hc = huellas_chicas
    preparada = per.preparar_huella(hc.imagenes[0], cfg)
    with pytest.raises(ValueError):
        per.evaluar_huella(preparada, dataclasses.replace(hc.modelo, escalador=None), cfg)


def test_leer_imagen_png_y_archivos_que_no_se_pueden_leer(per, tmp_path):
    rng = np.random.default_rng(64)
    carpeta = tmp_path / "huellas ñandú"
    carpeta.mkdir()
    gris = rng.integers(0, 256, size=(400, 320), dtype=np.uint8)
    color = rng.integers(0, 256, size=(60, 50, 3), dtype=np.uint8)
    for nombre, imagen in (("id1_v039.png", gris), ("color ñ.png", color)):
        ruta = carpeta / nombre
        escribir_png(ruta, imagen)
        for forma in (ruta, str(ruta)):
            leida = per.leer_imagen(forma)
            assert isinstance(leida, np.ndarray) and leida.dtype == np.uint8 and np.array_equal(leida, imagen), (
                f"leer_imagen({nombre!r}) no dio la imagen idéntica: {getattr(leida, 'shape', None)}")
    vacio = carpeta / "vacio.png"
    vacio.write_bytes(b"")
    basura = carpeta / "basura.png"
    basura.write_bytes(BASURA)
    for ruta in (carpeta / "no_existe.png", vacio, basura):
        assert per.leer_imagen(ruta) is None, f"leer_imagen({ruta.name!r}) tiene que dar None"


def test_preparar_huella_rechaza_lo_que_no_es_huella(cfg, ag, per, huellas_chicas, tmp_path):
    huella = huellas_chicas.imagenes[0]
    ruta16 = tmp_path / "16bits.png"
    escribir_png(ruta16, huella.astype(np.uint16) * 257)
    imagen16 = per.leer_imagen(ruta16)
    assert imagen16 is not None and imagen16.dtype == np.uint16, f"un PNG de 16 bits vuelve como uint16: {getattr(imagen16, 'dtype', None)}"
    casos = {
        "fondo liso (245)": np.full((400, 320), 245, dtype=np.uint8),
        "50 × 50": np.ascontiguousarray(huella[175:225, 135:185]),
        "float64": huella.astype(np.float64),
        "uint16 (PNG de 16 bits)": imagen16,
    }
    for nombre, imagen in casos.items():
        with pytest.raises(ValueError):
            per.preparar_huella(imagen, cfg)
        agente = ag.Agente(cfg, lambda im: per.preparar_huella(im, cfg), EvaluarFalso())
        accion = agente.percibir_huella(imagen, nombre)
        assert accion.regla == 10 and agente.estado == ag.Estado.ESPERANDO_MUESTRA, (
            f"{nombre}: el agente con preparar_huella dio la regla {accion.regla}, esperada 10")


# ===========================================================================
# 6.5 Búfer circular y push-to-talk (rampas)
# ===========================================================================

def escribir_hasta(bufer, rampa, hasta, bloque=128):
    while bufer.escritas < hasta:
        i = bufer.escritas
        bufer.escribir(rampa[i:min(i + bloque, hasta)])


def test_bufer_circular_rampa(cap):
    capacidad = 1000
    bufer = cap.BuferCircular(capacidad)
    assert bufer.capacidad == capacidad and bufer.escritas == 0, f"recién creado: {bufer.capacidad}, {bufer.escritas}"
    vacio = bufer.ultimas(10)
    assert vacio.dtype == np.float32 and vacio.size == 0, "ultimas sin nada escrito tiene que estar vacío"
    rampa = np.arange(20000, dtype=np.float32)
    pos = 0
    for largo in (0, 1, 7, 300, 0, 999, 1000, 1001, 2500, 3, 0, 250, 64, 1200, 17):
        bloque = rampa[pos:pos + largo].copy()
        bufer.escribir(bloque)
        bloque[:] = -1  # el búfer guarda una copia
        pos += largo
        assert bufer.escritas == pos, f"escritas = {bufer.escritas}, suma de largos = {pos}"
        lo = max(0, pos - capacidad)
        medio = (lo + pos) // 2
        for desde, hasta in ((lo, pos), (lo, lo), (pos, pos), (medio, pos), (lo, medio)):
            leido = bufer.leer(desde, hasta)
            assert leido.dtype == np.float32 and np.array_equal(leido, rampa[desde:hasta]), (
                f"leer({desde}, {hasta}) con {pos} escritas: largo {len(leido)}, primeras {leido[:3]}")
        for n in (0, 1, 10, capacidad - 1, capacidad, capacidad + 5, 10**6):
            esperado = rampa[pos - min(n, pos, capacidad):pos]
            obtenido = bufer.ultimas(n)
            assert obtenido.dtype == np.float32 and np.array_equal(obtenido, esperado), (
                f"ultimas({n}) con {pos} escritas: largo {len(obtenido)}, esperado {len(esperado)}")
    copia = bufer.leer(pos - 100, pos)
    copia[:] = -5
    assert np.array_equal(bufer.leer(pos - 100, pos), rampa[pos - 100:pos]), "leer no devolvió una copia"
    assert cap.BuferCircular(np.int64(10)).capacidad == 10, "np.integer vale como capacidad"


def test_bufer_circular_errores(cap):
    bufer = cap.BuferCircular(100)
    bufer.escribir(np.arange(250, dtype=np.float32))  # válidas: [150, 250)
    for desde, hasta, motivo in ((200, 199, "desde > hasta"), (240, 251, "hasta > escritas"),
                                 (149, 200, "desde < escritas - capacidad")):
        with pytest.raises(ValueError):
            bufer.leer(desde, hasta)
        assert bufer.escritas == 250, f"{motivo}: el error cambió escritas"
    corto = cap.BuferCircular(100)
    corto.escribir(np.arange(50, dtype=np.float32))
    with pytest.raises(ValueError):
        corto.leer(-1, 10)
    with pytest.raises(ValueError):
        bufer.escribir(np.zeros((10, 1), dtype=np.float32))
    assert bufer.escritas == 250, "un bloque de 2 dimensiones no se escribe"
    for capacidad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            cap.BuferCircular(capacidad)


def test_bufer_circular_hilos(cap):
    capacidad, bloque, bloques = 4096, 64, 200
    bufer = cap.BuferCircular(capacidad)
    rampa = np.arange(bloque * bloques, dtype=np.float32)
    errores = []

    def escritor():
        try:
            for i in range(bloques):
                bufer.escribir(rampa[i * bloque:(i + 1) * bloque])
                time.sleep(0.0005)
        except Exception as e:  # noqa: BLE001
            errores.append(e)

    hilo = threading.Thread(target=escritor)
    hilo.start()
    lecturas = 0
    try:
        while hilo.is_alive() or lecturas == 0:
            fin = bufer.escritas
            n = min(fin, 256)
            leido = bufer.leer(fin - n, fin)
            assert np.array_equal(leido, rampa[fin - n:fin]), f"lectura [{fin - n}, {fin}) fuera de la rampa"
            ultimas = bufer.ultimas(256)
            if ultimas.size:
                assert np.array_equal(ultimas, np.arange(ultimas[0], ultimas[0] + ultimas.size, dtype=np.float32)), (
                    "ultimas no es un tramo seguido de la rampa")
                assert ultimas[-1] < rampa.size, "ultimas devolvió muestras fuera de la rampa"
            lecturas += 1
    finally:
        hilo.join()
    assert not errores, f"el hilo escritor falló: {errores!r}"
    assert bufer.escritas == rampa.size and np.array_equal(bufer.ultimas(capacidad), rampa[-capacidad:])


@pytest.fixture
def ptt(cap, cfg):
    bufer = cap.BuferCircular(5000)
    return cap.PushToTalk(bufer, 1000, cfg), bufer, np.arange(20000, dtype=np.float32)


def test_push_to_talk_cuentas_a_1000_hz(ptt):
    p, bufer, _ = ptt
    assert (p.pre, p.post, p.maximo) == (250, 150, 3000), f"pre, post, maximo = {(p.pre, p.post, p.maximo)}"
    assert all(es_int(v) for v in (p.pre, p.post, p.maximo)), "pre, post y maximo tienen que ser int"
    assert p.estado == "reposo" and p.ventana() is None and p.clip() is None and p.tope() is False


def test_push_to_talk_ventana_y_clip(ptt):
    p, bufer, rampa = ptt
    escribir_hasta(bufer, rampa, 5000)
    assert p.apretar() is True and p.estado == "grabando", f"apretar: estado = {p.estado!r}"
    assert p.ventana() is None and p.clip() is None, "mientras se graba no hay ventana ni clip"
    escribir_hasta(bufer, rampa, 6200)
    assert p.tope() is False
    assert p.soltar() is True and p.estado == "esperando", f"soltar: estado = {p.estado!r}"
    assert p.ventana() == (4750, 6350), f"ventana = {p.ventana()!r}, esperada (4750, 6350)"
    for hasta in (6200, 6300, 6349):
        escribir_hasta(bufer, rampa, hasta)
        assert p.clip() is None, f"con {hasta} escritas (< 6350) clip tiene que ser None"
        assert p.estado == "esperando" and p.ventana() == (4750, 6350)
    escribir_hasta(bufer, rampa, 6350)
    clip = p.clip()
    assert clip is not None and clip.dtype == np.float32 and np.array_equal(clip, rampa[4750:6350]), (
        f"clip no es la rampa en [4750, 6350): largo {None if clip is None else len(clip)}")
    assert p.estado == "reposo" and p.ventana() is None and p.clip() is None, "después del clip vuelve a reposo"


def test_push_to_talk_pre_roll_recortado_al_principio(ptt):
    p, bufer, rampa = ptt
    escribir_hasta(bufer, rampa, 100)
    p.apretar()
    escribir_hasta(bufer, rampa, 1300)
    p.soltar()
    assert p.ventana() == (0, 1450), f"apretado en 100: ventana = {p.ventana()!r}, esperada (0, 1450)"
    escribir_hasta(bufer, rampa, 1450)
    assert np.array_equal(p.clip(), rampa[0:1450])


def test_push_to_talk_ventana_con_tope(ptt):
    p, bufer, rampa = ptt
    escribir_hasta(bufer, rampa, 5000)
    p.apretar()
    escribir_hasta(bufer, rampa, 7999)
    assert p.tope() is False, "con 2999 muestras grabadas todavía no es el tope"
    escribir_hasta(bufer, rampa, 8000)
    assert p.tope() is True, "con 3000 muestras grabadas es el tope"
    escribir_hasta(bufer, rampa, 9000)
    assert p.tope() is True
    assert p.soltar() is True
    assert p.ventana() == (4750, 8150), f"soltado en 9000 con inicio 5000: ventana = {p.ventana()!r}, esperada (4750, 8150)"
    assert p.tope() is False, "fuera de grabando, tope es False"
    assert np.array_equal(p.clip(), rampa[4750:8150]), "el clip con tope no es [4750, 8150)"


def test_push_to_talk_clip_tardio_recorta_lo_pisado(ptt):
    p, bufer, rampa = ptt
    escribir_hasta(bufer, rampa, 5000)
    p.apretar()
    escribir_hasta(bufer, rampa, 6200)
    p.soltar()
    escribir_hasta(bufer, rampa, 10000)  # el búfer (5000) ya pisó [4750, 5000)
    assert np.array_equal(p.clip(), rampa[5000:6350]), "un clip tardío es leer(max(a, escritas - capacidad), b)"


def test_push_to_talk_repeticiones_y_cancelar(ptt):
    p, bufer, rampa = ptt
    assert p.soltar() is False and p.estado == "reposo", "soltar sin apretar no hace nada"
    escribir_hasta(bufer, rampa, 5000)
    assert p.apretar() is True
    escribir_hasta(bufer, rampa, 5100)
    assert p.apretar() is False, "la segunda pulsación (repetición de teclas) devuelve False"
    escribir_hasta(bufer, rampa, 6200)
    assert p.soltar() is True and p.soltar() is False
    assert p.ventana() == (4750, 6350), f"apretar dos veces movió inicio_voz: ventana = {p.ventana()!r}"
    assert p.apretar() is False and p.estado == "esperando", "apretar en esperando no hace nada"
    p.cancelar()
    assert p.estado == "reposo" and p.ventana() is None and p.clip() is None, "cancelar desde esperando"
    p.apretar()
    p.cancelar()
    assert p.estado == "reposo" and p.ventana() is None, "cancelar desde grabando"
    p.cancelar()
    assert p.estado == "reposo", "cancelar desde reposo"


def test_push_to_talk_bufer_chico_o_fs_invalido(cap, cfg):
    cap.PushToTalk(cap.BuferCircular(4400), 1000, cfg)  # justo pre + maximo + post + fs
    with pytest.raises(ValueError):
        cap.PushToTalk(cap.BuferCircular(4399), 1000, cfg)
    for fs in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            cap.PushToTalk(cap.BuferCircular(10**6), fs, cfg)


def test_nivel_db(cap):
    t = np.arange(4800) / 48000
    seno = np.sin(2 * np.pi * 1000 * t)  # 100 períodos exactos
    obtenido = cap.nivel_db(seno)
    assert isinstance(obtenido, float), f"nivel_db no devuelve float: {type(obtenido)}"
    assert obtenido == pytest.approx(20 * math.log10(1 / math.sqrt(2)), abs=1e-9), f"seno de amplitud 1: {obtenido!r}"
    assert cap.nivel_db(np.zeros(480)) == pytest.approx(-200.0, abs=1e-9), "ceros: 20·log10(1e-10)"
    assert cap.nivel_db(np.zeros(0)) == pytest.approx(-200.0, abs=1e-9), "bloque vacío: 20·log10(1e-10)"
    assert cap.nivel_db(np.zeros(480), piso=1e-5) == pytest.approx(-100.0, abs=1e-9), "ceros con piso 1e-5"
    assert cap.nivel_db(np.zeros(0, dtype=np.float32), 1e-3) == pytest.approx(-60.0, abs=1e-9)
    for malo, kw in ((np.array([0.1, np.nan]), {}), (np.array([0.1, np.inf]), {}), (np.zeros((10, 2)), {}),
                     (np.zeros(10), {"piso": 0}), (np.zeros(10), {"piso": -1e-10})):
        with pytest.raises(ValueError):
            cap.nivel_db(malo, **kw)


# ===========================================================================
# 6.6 Micrófono (con la fábrica falsa)
# ===========================================================================

def test_microfono_abre_con_la_fabrica_falsa(cap, cfg):
    fab = FabricaFalsa()
    mic = cap.Microfono(cfg, fs=48000, fabrica=fab)
    assert mic.abierto is False and mic.nivel_db() is None, "sin abrir: abierto False y nivel_db None"
    mic.abrir()
    assert len(fab.llamadas) == 1, f"la fábrica se llamó {len(fab.llamadas)} veces"
    args, kw = fab.llamadas[0]
    assert args == (), f"la fábrica se llama solo con argumentos por nombre: {args!r}"
    esperado = {"samplerate": 48000, "channels": 1, "dtype": "float32"}
    assert {k: kw.get(k) for k in esperado} == esperado, f"argumentos de la fábrica: {kw!r}"
    assert kw.get("device", None) is None and callable(kw.get("callback")), f"device/callback: {kw!r}"
    stream = fab.streams[0]
    assert stream.llamadas["start"] == 1 and mic.abierto is True, f"start: {dict(stream.llamadas)}, abierto {mic.abierto}"
    assert mic.fs == 48000 and mic.bufer.capacidad == round(5.0 * 48000), f"capacidad = {mic.bufer.capacidad}"
    rampa = np.arange(10000, dtype=np.float32)
    pos = 0
    for largo in (480, 480, 1000, 37, 3, 4000, 4000):
        indata = rampa[pos:pos + largo].reshape(-1, 1).copy()
        stream.callback(indata, largo, None, None)
        indata[:] = -1  # sounddevice reusa indata: el búfer tiene que haber copiado
        pos += largo
    assert mic.bufer.escritas == 10000 and np.array_equal(mic.bufer.leer(0, 10000), rampa), "el callback no dejó la rampa en orden"
    assert mic.nivel_db() == pytest.approx(db(rampa[-2400:]), abs=1e-9), f"nivel_db = {mic.nivel_db()!r} (últimas 2400 muestras)"


def test_microfono_pasa_el_dispositivo(cap, cfg):
    fab = FabricaFalsa()
    cap.Microfono(cfg, dispositivo=3, fs=16000, fabrica=fab).abrir()
    assert fab.llamadas[0][1].get("device") == 3 and fab.llamadas[0][1].get("samplerate") == 16000, f"{fab.llamadas[0][1]!r}"


def test_microfono_cerrar_y_abrir_dos_veces(cap, cfg):
    cap.Microfono(cfg, fs=48000, fabrica=FabricaFalsa()).cerrar()  # cerrar sin abrir no falla
    fab = FabricaFalsa()
    mic = cap.Microfono(cfg, fs=48000, fabrica=fab)
    mic.abrir()
    mic.abrir()
    assert len(fab.llamadas) == 1 and fab.streams[0].llamadas["start"] == 1, "abrir dos veces crea un solo stream"
    mic.cerrar()
    mic.cerrar()
    stream = fab.streams[0]
    assert (stream.llamadas["stop"], stream.llamadas["close"]) == (1, 1), f"stop/close: {dict(stream.llamadas)}"
    assert mic.abierto is False and mic.nivel_db() is None, "después de cerrar: abierto False y nivel_db None"


@pytest.mark.parametrize("falla", ["al_crear", "al_arrancar"])
def test_microfono_fabrica_que_falla_da_sin_microfono(cap, cfg, falla):
    assert issubclass(cap.SinMicrofono, RuntimeError)
    fab = FabricaFalsa(falla_al_crear=falla == "al_crear", falla_al_arrancar=falla == "al_arrancar")
    mic = cap.Microfono(cfg, fs=48000, fabrica=fab)
    with pytest.raises(cap.SinMicrofono) as error:
        mic.abrir()
    original = "no hay dispositivo de entrada" if falla == "al_crear" else "no se pudo arrancar"
    assert original in str(error.value), f"SinMicrofono no lleva el mensaje original: {error.value!r}"
    assert mic.abierto is False, "con la fábrica que falla, abierto tiene que ser False"
    if falla == "al_arrancar":
        assert fab.streams[0].llamadas["close"] >= 1, "el stream que no arrancó se tiene que cerrar"
    mic.cerrar()


def test_captura_se_importa_sin_sounddevice():
    r = importar_en_subproceso("src.agente.captura", ["sounddevice"])
    assert r["sounddevice"] is False, "importar src.agente.captura importó sounddevice"


# ===========================================================================
# 6.7 Huellas de la app
# ===========================================================================

FILAS_CATALOGO = [
    # archivo, particion, identidad, intruso, tipo, semilla, variante, reservada_demo
    ("test/intruso33_v1.png", "test", 0, 33, "arco", 9033, 1, 0),
    ("train/id1_v001.png", "train", 1, 0, "presilla", 101, 1, 0),
    ("test/id2_v039.png", "test", 2, 0, "verticilo", 202, 39, 1),
    ("val/id1_v031.png", "val", 1, 0, "presilla", 101, 31, 0),
    ("test/id1_v036.png", "test", 1, 0, "presilla", 101, 36, 0),
    ("val/intruso21_v1.png", "val", 0, 21, "presilla", 9021, 1, 0),
    ("test/id1_v039.png", "test", 1, 0, "presilla", 101, 39, 1),
    ("test/intruso31_v1.png", "test", 0, 31, "verticilo", 9031, 1, 0),
    ("test/intruso31_v2.png", "test", 0, 31, "verticilo", 9031, 2, 0),
    ("test/intruso31_v3.png", "test", 0, 31, "verticilo", 9031, 3, 0),
    ("train/intruso1_v1.png", "train", 0, 1, "presilla", 9001, 1, 0),
    ("test/id3_v040.png", "test", 3, 0, "arco", 303, 40, 1),
    ("test/intruso32_v1.png", "test", 0, 32, "arco_tienda", 9032, 1, 0),
    ("val/intruso22_v2.png", "val", 0, 22, "verticilo", 9022, 2, 0),
]


def test_catalogo_reservadas_e_intrusos_de_test(cfg, mue, tmp_path):
    carpeta = tmp_path / "huellas"
    carpeta.mkdir()
    with open(carpeta / "metadatos.csv", "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["archivo", "particion", "identidad", "intruso", "tipo", "semilla", "variante", "reservada_demo"])
        escritor.writerows(FILAS_CATALOGO)
    E = mue.EntradaHuella
    esperado = [
        E(nombre="id2_v039 (reservada)", origen="reservada", verdad=2, ruta=carpeta / "test/id2_v039.png", variante=39),
        E(nombre="id1_v039 (reservada)", origen="reservada", verdad=1, ruta=carpeta / "test/id1_v039.png", variante=39),
        E(nombre="id3_v040 (reservada)", origen="reservada", verdad=3, ruta=carpeta / "test/id3_v040.png", variante=40),
        E(nombre="intruso33_v1", origen="intruso_test", verdad=0, ruta=carpeta / "test/intruso33_v1.png", variante=1),
        E(nombre="intruso31_v1", origen="intruso_test", verdad=0, ruta=carpeta / "test/intruso31_v1.png", variante=1),
        E(nombre="intruso32_v1", origen="intruso_test", verdad=0, ruta=carpeta / "test/intruso32_v1.png", variante=1),
    ]
    for forma in (carpeta, str(carpeta)):  # sin los PNG: catalogo no lee las imágenes
        obtenido = mue.catalogo(cfg, forma)
        assert obtenido == esperado, "catalogo dio:\n" + "\n".join(map(repr, obtenido))
        for entrada in obtenido:
            assert isinstance(entrada.ruta, Path) and es_int(entrada.verdad) and es_int(entrada.variante), f"tipos de {entrada!r}"
    with pytest.raises(FileNotFoundError):
        mue.catalogo(cfg, tmp_path / "no_existe")


def test_entrada_archivo(mue, tmp_path):
    ruta = tmp_path / "mis huellas" / "dedo ñ.png"
    for forma in (ruta, str(ruta)):
        assert mue.entrada_archivo(forma) == mue.EntradaHuella(nombre="dedo ñ.png", origen="archivo", verdad=None, ruta=ruta, variante=None), (
            f"entrada_archivo({forma!r}) = {mue.entrada_archivo(forma)!r}")


@pytest.fixture
def bases_contadas(monkeypatch, mue, receta):
    """Cambia generar_base de muestras por la de la receta (con bases guardadas, para no generarlas
    de nuevo en cada test) y cuenta cuántas veces la pide el generador por (tipo, semilla)."""
    contador = collections.Counter()

    def generar_base_contada(tipo, semilla, cfg):
        contador[(tipo, int(semilla))] += 1
        return receta.base(tipo, int(semilla))

    monkeypatch.setattr(mue, "generar_base", generar_base_contada)
    return contador


def test_generador_variante_nueva(cfg, mue, receta, bases_contadas):
    generador = mue.GeneradorDemo(cfg)
    for k, v in ((1, 1000), (2, 1000), (1, 1001)):
        entrada, imagen = generador.variante_nueva(k)
        assert entrada == mue.EntradaHuella(nombre=f"id{k}_v{v} (nueva)", origen="variante_nueva", verdad=k, ruta=None, variante=v), (
            f"variante_nueva({k}) dio {entrada!r}, esperado id{k}_v{v}")
        assert isinstance(imagen, np.ndarray) and imagen.dtype == np.uint8 and imagen.shape == (400, 320), (
            f"imagen {getattr(imagen, 'dtype', None)} {getattr(imagen, 'shape', None)}")
        assert np.array_equal(imagen, receta.identidad(k, v)), f"variante_nueva({k}) no es la receta de H1 con v = {v}"


def test_generador_variante_nueva_desde_39_es_la_del_dataset(cfg, mue, receta, bases_contadas):
    cfg39 = copy.deepcopy(cfg)
    cfg39["app"]["variante_nueva_desde"] = 39
    entrada, imagen = mue.GeneradorDemo(cfg39).variante_nueva(1)
    assert entrada.variante == 39 and entrada.nombre == "id1_v39 (nueva)", f"{entrada!r}"
    assert np.array_equal(imagen, receta.identidad(1, 39)), "con variante_nueva_desde = 39 no es la receta con v = 39"


@pytest.mark.parametrize("identidad", [4, 0])
def test_generador_variante_nueva_identidad_invalida(cfg, mue, bases_contadas, identidad):
    with pytest.raises(ValueError):
        mue.GeneradorDemo(cfg).variante_nueva(identidad)


def test_generador_intruso_nuevo(cfg, mue, receta, bases_contadas):
    generador = mue.GeneradorDemo(cfg)
    tipos = cfg["huellas"]["intrusos"]["tipos"]
    for i in (1000, 1001):
        entrada, imagen = generador.intruso_nuevo()
        assert entrada == mue.EntradaHuella(nombre=f"intruso{i}_v1 (nuevo)", origen="intruso_nuevo", verdad=0, ruta=None, variante=1), (
            f"intruso_nuevo #{i - 999} dio {entrada!r}")
        assert imagen.dtype == np.uint8 and imagen.shape == (400, 320)
        tipo, semilla = receta.tipo_semilla_intruso(i)
        assert tipo == tipos[(i - 1) % 4] and semilla == cfg["huellas"]["intrusos"]["semilla_base"] + i
        assert np.array_equal(imagen, receta.intruso(i)), f"intruso_nuevo #{i - 999} no es la receta con i = {i}"


def test_generador_dos_sesiones_dan_la_misma_secuencia(cfg, mue, bases_contadas):
    def sesion():
        g = mue.GeneradorDemo(cfg)
        return [g.variante_nueva(1), g.variante_nueva(3), g.intruso_nuevo(), g.variante_nueva(1), g.intruso_nuevo()]

    a, b = sesion(), sesion()
    for i, ((ea, ia), (eb, ib)) in enumerate(zip(a, b)):
        assert ea == eb and np.array_equal(ia, ib), f"llamada {i}: las dos sesiones difieren ({ea.nombre} / {eb.nombre})"


def test_generador_base_una_vez_por_tipo_y_semilla(cfg, mue, receta, bases_contadas):
    generador = mue.GeneradorDemo(cfg)
    for _ in range(3):
        generador.variante_nueva(1)
    for _ in range(2):
        generador.variante_nueva(2)
        generador.intruso_nuevo()
    usadas = {(identidad_cfg(cfg, k)["tipo"], int(identidad_cfg(cfg, k)["semilla"])) for k in (1, 2)}
    usadas |= {receta.tipo_semilla_intruso(i) for i in (1000, 1001)}
    assert usadas <= set(bases_contadas), f"no se generaron las bases {usadas - set(bases_contadas)}"
    repetidas = {k: n for k, n in bases_contadas.items() if n > 1}
    assert not repetidas, f"bases generadas más de una vez: {repetidas}"


# ===========================================================================
# 6.8 Visor PCA
# ===========================================================================

def escribir_caracteristicas(carpeta: Path, n_train: int, rng) -> dict:
    d = 12
    particion = np.array((["train"] * n_train + ["val"] * 6 + ["test"] * 6))
    particion = particion[rng.permutation(particion.size)]
    n = particion.size
    escalas = np.array([4.0, 2.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    X = rng.normal(size=(n, d)) * escalas + rng.normal(size=d)
    identidad = np.arange(n, dtype=np.int64) % 4
    archivo = np.array([f"{p}/x{i:03d}.png" for i, p in enumerate(particion)])
    carpeta.mkdir(parents=True, exist_ok=True)
    np.savez(carpeta / "caracteristicas.npz", X=X, particion=particion, identidad=identidad, archivo=archivo)
    return {"X": X, "particion": particion, "identidad": identidad}


def modelo_para_visor(T: np.ndarray, identidades_T: np.ndarray, d_extra: int = 0):
    from src.huellas.clasificacion import ModeloHuellas

    centroides = np.stack([T[identidades_T == k].mean(axis=0) for k in (1, 2, 3)])
    if d_extra:
        centroides = np.hstack([centroides, np.zeros((3, d_extra))])
    return ModeloHuellas(identidades=np.array([1, 2, 3], dtype=np.int64), centroides=centroides, dispersion=np.ones(3),
                         tau=1.0, pureza=1.0, inercia=0.0, escalador=None)


@pytest.fixture(scope="module")
def datos_visor(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("visor") / "huellas"
    datos = escribir_caracteristicas(carpeta, 24, np.random.default_rng(58))
    datos["carpeta"] = carpeta
    datos["T"] = datos["X"][datos["particion"] == "train"]
    datos["ids_T"] = datos["identidad"][datos["particion"] == "train"]
    return datos


def test_visor_pca_igual_al_pca_de_h3(cfg, vis, datos_visor):
    from sklearn.decomposition import PCA as PCASklearn
    from src.core_ml.pca import PCA

    T, ids_T = datos_visor["T"], datos_visor["ids_T"]
    modelo = modelo_para_visor(T, ids_T)
    visor = vis.armar_visor(cfg, modelo, datos_visor["carpeta"])
    pca = PCA(2).ajustar(T)
    for campo, esperado in (("media", pca.media), ("componentes", pca.componentes), ("proporcion", pca.proporcion_varianza),
                            ("puntos", pca.transformar(T)), ("centroides", pca.transformar(modelo.centroides))):
        obtenido = getattr(visor, campo)
        assert obtenido.dtype == np.float64 and obtenido.shape == np.shape(esperado), f"{campo}: {obtenido.dtype} {obtenido.shape}"
        assert np.allclose(obtenido, esperado, rtol=0, atol=1e-12), f"{campo} no coincide con el PCA de H3: {obtenido} vs {esperado}"
    referencia = PCASklearn(2).fit(T)
    assert np.allclose(np.abs(visor.componentes), np.abs(referencia.components_), rtol=0, atol=1e-9), (
        "|componentes| no coincide con sklearn.decomposition.PCA(2)")
    assert visor.identidades.dtype == np.int64 and np.array_equal(visor.identidades, ids_T), "identidades no son las de train"
    for i in (0, 5, len(T) - 1):
        fila = visor.proyectar(T[i])
        assert fila.shape == (2,) and np.allclose(fila, visor.puntos[i], rtol=0, atol=1e-12), f"proyectar(T[{i}]) = {fila}"
    todas = visor.proyectar(T)
    assert todas.shape == (len(T), 2) and np.allclose(todas, visor.puntos, rtol=0, atol=1e-12), "proyectar(T) no da puntos"


def test_visor_pca_errores(cfg, vis, datos_visor, tmp_path):
    T, ids_T = datos_visor["T"], datos_visor["ids_T"]
    modelo = modelo_para_visor(T, ids_T)
    with pytest.raises(FileNotFoundError):
        vis.armar_visor(cfg, modelo, tmp_path / "sin_npz")
    escribir_caracteristicas(tmp_path / "pocas", 2, np.random.default_rng(3))  # 2 filas de train
    with pytest.raises(ValueError):
        vis.armar_visor(cfg, modelo, tmp_path / "pocas")
    with pytest.raises(ValueError):
        vis.armar_visor(cfg, modelo_para_visor(T, ids_T, d_extra=1), datos_visor["carpeta"])
    visor = vis.armar_visor(cfg, modelo, datos_visor["carpeta"])
    d = T.shape[1]
    malos = {"vector de otro d": np.zeros(d + 1), "matriz de otro d": np.zeros((3, d + 1)),
             "vector con NaN": np.where(np.arange(d) == 2, np.nan, 0.0), "matriz con Inf": np.where(np.eye(3, d) > 0, np.inf, 0.0)}
    for nombre, z in malos.items():
        with pytest.raises(ValueError):
            visor.proyectar(z)


# ===========================================================================
# 6.10 La app se puede importar
# ===========================================================================

@pytest.mark.parametrize("modulo,funcion", [("src.app.ventana", "ejecutar"), ("src.app.__main__", "main")])
def test_app_se_importa_sin_crear_ventanas(modulo, funcion):
    extra = (f"r['llamable'] = callable(getattr(m, {funcion!r}, None))",
             "tk = sys.modules.get('tkinter')",
             "r['ventana'] = bool(tk is not None and getattr(tk, '_default_root', None) is not None)")
    r = importar_en_subproceso(modulo, [], extra)
    assert r["llamable"] is True, f"{modulo}.{funcion} no existe o no es llamable"
    assert r["ventana"] is False, f"importar {modulo} creó una ventana de Tk"


def test_agente_y_app_no_usan_sklearn():
    archivos = sorted((RAIZ / "src" / "agente").rglob("*.py")) + sorted((RAIZ / "src" / "app").rglob("*.py"))
    nombres = {p.relative_to(RAIZ).as_posix() for p in archivos}
    for esperado in ("src/agente/agente.py", "src/agente/percepcion.py", "src/agente/captura.py", "src/app/muestras.py",
                     "src/app/visor.py", "src/app/controlador.py", "src/app/ventana.py", "src/app/__main__.py"):
        assert esperado in nombres, f"falta {esperado}"
    con_sklearn = [p.relative_to(RAIZ).as_posix() for p in archivos if "sklearn" in p.read_text(encoding="utf-8")]
    assert not con_sklearn, f"estos archivos mencionan sklearn: {con_sklearn}"


@pytest.mark.parametrize("modulo,prohibidos", [
    ("src.agente.agente", ["tkinter", "customtkinter", "sounddevice", "matplotlib"]),
    ("src.app.controlador", ["tkinter", "customtkinter"]),
    ("src.app.muestras", ["tkinter", "customtkinter"]),
    ("src.app.visor", ["tkinter", "customtkinter"]),
])
def test_modulos_sin_tk_ni_hardware(modulo, prohibidos):
    r = importar_en_subproceso(modulo, prohibidos)
    importados = [n for n in prohibidos if r[n]]
    assert not importados, f"importar {modulo} importó {importados}"


# ===========================================================================
# 6.11 Aceptación (modelos y datos reales)
# ===========================================================================

def a_48k(y: np.ndarray, fs: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if fs == 48000:
        return np.asarray(y, dtype=np.float32)
    g = math.gcd(48000, int(fs))
    return resample_poly(np.asarray(y, dtype=np.float64), 48000 // g, int(fs) // g).astype(np.float32)


def ruido(n: int, semilla: int = 2026) -> np.ndarray:
    return (1e-4 * np.random.default_rng(semilla).standard_normal(n)).astype(np.float32)


@pytest.fixture(scope="module")
def huellas_reales(cfg):
    requerir(MODELO_H, META_H)
    from src.agente.percepcion import evaluar_huella, preparar_huella
    from src.app.controlador import Controlador
    from src.app.muestras import GeneradorDemo, catalogo
    from src.huellas.clasificacion import cargar_modelo

    modelo = cargar_modelo(RAIZ / MODELO_H)

    def crear():
        voz = VozFalsa()
        c = Controlador(cfg, lambda im: preparar_huella(im, cfg), lambda h: evaluar_huella(h, modelo, cfg), voz,
                        generador=GeneradorDemo(cfg), entradas=catalogo(cfg))
        return c, SimpleNamespace(voz=voz)

    return SimpleNamespace(modelo=modelo, crear=crear)


def entrada_por_nombre(c, nombre):
    for entrada in c.entradas:
        if entrada.nombre == nombre:
            return entrada
    raise AssertionError(f"no está {nombre!r} en las entradas del controlador")


@pytest.mark.aceptacion
def test_aceptacion_catalogo_real(cfg):
    requerir(META_H)
    from src.app.muestras import catalogo

    entradas = catalogo(cfg)
    origenes = collections.Counter(e.origen for e in entradas)
    assert len(entradas) == 36 and origenes == {"reservada": 6, "intruso_test": 30}, f"catalogo real: {len(entradas)} entradas, {dict(origenes)}"
    assert [e.origen for e in entradas] == ["reservada"] * 6 + ["intruso_test"] * 30, "primero las reservadas"


@pytest.mark.aceptacion
def test_aceptacion_reservadas_autorizadas_con_su_identidad(cfg, huellas_reales):
    from src.huellas.clasificacion import reconocer

    c, x = huellas_reales.crear()
    reservadas = [e for e in c.entradas if e.origen == "reservada"]
    assert len(reservadas) == 6
    for entrada in reservadas:
        stem = entrada.ruta.stem
        c.elegir(entrada)
        p = decir(c, x, "reconocer")
        assert p.cartel == AUTORIZADO_C, f"{stem}: cartel {p.cartel!r} (distancia {p.distancia!r}, τ {p.tau!r})"
        r = reconocer(leer_png(entrada.ruta), huellas_reales.modelo, cfg)
        assert p.distancia == pytest.approx(float(r.distancia), abs=1e-12), f"{stem}: distancia {p.distancia!r} vs H4 {r.distancia!r}"
        assert p.distancia == pytest.approx(DIST_RESERVADAS[stem], abs=1e-3), f"{stem}: distancia {p.distancia:.4f}, sección 8 {DIST_RESERVADAS[stem]}"
        p = decir(c, x, "identificar")
        assert p.identificacion == NOMBRES[entrada.verdad], f"{stem}: identificacion {p.identificacion!r}"


@pytest.mark.aceptacion
def test_aceptacion_intrusos_de_test_rechazados(cfg, huellas_reales):
    c, x = huellas_reales.crear()
    intrusos = [e for e in c.entradas if e.origen == "intruso_test"]
    assert len(intrusos) == 30
    for entrada in intrusos:
        c.elegir(entrada)
        p = decir(c, x, "reconocer")
        assert p.cartel == NO_AUTORIZADO_C, f"{entrada.nombre}: cartel {p.cartel!r} (distancia {p.distancia!r})"
        p = decir(c, x, "identificar")
        assert p.regla == 6 and p.identificacion == "", f"{entrada.nombre}: regla {p.regla}, identificacion {p.identificacion!r}"


@pytest.mark.aceptacion
def test_aceptacion_escenario_3_real(cfg, huellas_reales):
    c, x = huellas_reales.crear()
    c.elegir(entrada_por_nombre(c, "id1_v040 (reservada)"))
    assert decir(c, x, "reconocer").cartel == AUTORIZADO_C
    assert decir(c, x, "identificar").identificacion == NOMBRES[1]
    p = c.elegir(entrada_por_nombre(c, "intruso31_v1"))
    assert p.identificacion == "" and p.cartel == "", f"después de cambiar de huella: {p.identificacion!r} {p.cartel!r}"
    p = decir(c, x, "identificar")
    assert p.regla == 4 and p.identificacion == "", f"identificar sin reconocer: regla {p.regla}"
    assert decir(c, x, "reconocer").cartel == NO_AUTORIZADO_C
    assert decir(c, x, "identificar").regla == 6


@pytest.mark.aceptacion
def test_aceptacion_variantes_e_intrusos_nuevos(cfg, huellas_reales):
    c, x = huellas_reales.crear()
    for k in (1, 2, 3):
        for n in range(10):
            p = c.variante_nueva(k)
            assert p.huella == f"id{k}_v{1000 + n} (nueva)", f"huella {p.huella!r}"
            p = decir(c, x, "reconocer")
            assert p.cartel == AUTORIZADO_C, f"id{k}_v{1000 + n}: {p.cartel!r} (distancia {p.distancia!r})"
            assert decir(c, x, "identificar").identificacion == NOMBRES[k], f"id{k}_v{1000 + n}: otra identidad"
    for n in range(10):
        c.intruso_nuevo()
        p = decir(c, x, "reconocer")
        assert p.cartel == NO_AUTORIZADO_C, f"intruso{1000 + n}_v1: {p.cartel!r} (distancia {p.distancia!r})"


@pytest.mark.aceptacion
def test_aceptacion_variante_39_es_el_png_del_dataset(cfg):
    requerir("data/huellas/test/id1_v039.png")
    from src.app.muestras import GeneradorDemo

    cfg39 = copy.deepcopy(cfg)
    cfg39["app"]["variante_nueva_desde"] = 39
    _, imagen = GeneradorDemo(cfg39).variante_nueva(1)
    assert np.array_equal(imagen, leer_png(RAIZ / "data/huellas/test/id1_v039.png")), "no es data/huellas/test/id1_v039.png"


@pytest.mark.aceptacion
def test_aceptacion_visor_real(cfg):
    requerir(MODELO_H, CARAC_H)
    from src.app.visor import armar_visor
    from src.huellas.clasificacion import cargar_modelo

    visor = armar_visor(cfg, cargar_modelo(RAIZ / MODELO_H))
    assert np.allclose(visor.proporcion, [0.4764, 0.3727], rtol=0, atol=1e-3), f"proporcion = {visor.proporcion}"
    esperado = [[0.252, -1.423], [1.271, 0.903], [-1.523, 0.520]]
    assert np.allclose(visor.centroides, esperado, rtol=0, atol=1e-3), f"centroides = {visor.centroides}"


@pytest.fixture(scope="module")
def clips_voz(cfg):
    """Para cada clase, el primer clip de test de la notebook (por archivo) que V4 decide bien."""
    requerir(MODELO_V, META_V)
    from src.agente.agente import etiquetas_voz
    from src.voz.audio import leer_audio
    from src.voz.clasificacion import cargar_modelo, reconocer

    modelo = cargar_modelo(RAIZ / MODELO_V)
    carpeta = RAIZ / "data" / "voz" / "dataset"
    with open(carpeta / "metadatos.csv", encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    clips, faltan = {}, []
    for clase in etiquetas_voz(cfg):
        candidatas = sorted((f for f in filas if f["particion"] == "test" and f["origen"] == "notebook" and f["clase"] == clase),
                            key=lambda f: f["archivo"])
        for fila in candidatas:
            y, fs = leer_audio(carpeta / fila["archivo"])
            if reconocer(y, fs, modelo, cfg).comando == clase:
                clips[clase] = (y, fs, fila["archivo"])
                break
        else:
            faltan.append(clase)
    return SimpleNamespace(clips=clips, faltan=faltan, reconocer=lambda y, fs: reconocer(y, fs, modelo, cfg))


@pytest.mark.aceptacion
def test_aceptacion_voz_real_recorrido(cfg, ctl, clips_voz):
    requerir(MODELO_H, META_H)
    assert not clips_voz.faltan, f"no hay clip de test de la notebook que V4 reconozca para {clips_voz.faltan}"
    c = ctl.Controlador.desde_archivos(cfg, microfono=False)
    clip = clips_voz.clips
    c.elegir(entrada_por_nombre(c, "id2_v039 (reservada)"))
    p = c.comando_audio(*clip["reconocer"][:2])
    assert p.comando == "reconocer" and p.cartel == AUTORIZADO_C, f"reconocer ({clip['reconocer'][2]}): {p.comando!r} {p.cartel!r}"
    p = c.comando_audio(*clip["identificar"][:2])
    assert p.comando == "identificar" and p.identificacion == NOMBRES[2], f"identificar: {p.comando!r} {p.identificacion!r}"
    p = c.comando_audio(*clip["otro"][:2])
    assert p.comando == "otro" and p.regla == 8, f"otro ({clip['otro'][2]}): {p.comando!r}, regla {p.regla}"
    p = c.comando_audio(*clip["salir"][:2])
    assert p.comando == "salir" and p.cerrar is True, f"salir ({clip['salir'][2]}): {p.comando!r}, cerrar {p.cerrar!r}"


@pytest.mark.aceptacion
def test_aceptacion_push_to_talk_con_audio_real(cfg, ctl, clips_voz):
    from src.agente.captura import Microfono

    assert not clips_voz.faltan, f"no hay clip de test de la notebook que V4 reconozca para {clips_voz.faltan}"
    y, fs, archivo = clips_voz.clips["reconocer"]
    voz = clips_voz.reconocer
    recibidos = []

    def espia(y_, fs_):
        recibidos.append((np.array(y_, copy=True), fs_))
        return voz(y_, fs_)

    fab = FabricaFalsa()
    mic = Microfono(cfg, fs=48000, fabrica=fab)
    c = ctl.Controlador(cfg, PrepararFalso(), EvaluarFalso(), espia, microfono=mic)
    assert c.abrir_microfono().microfono is True
    clip = a_48k(y, fs)
    senal = np.concatenate([ruido(48000, 1), clip, ruido(48000, 2)])
    alimentar(mic, fab, senal, 48000, bloque=480)
    c.apretar()
    alimentar(mic, fab, senal, 48000 + clip.size, bloque=480)
    c.soltar()
    alimentar(mic, fab, senal, senal.size, bloque=480)
    p = c.actualizar()
    fin_voz = min(48000 + clip.size, 48000 + 144000)
    a, b = 48000 - 12000, fin_voz + 7200
    assert len(recibidos) == 1, f"reconocer_voz se llamó {len(recibidos)} veces"
    y_rec, fs_rec = recibidos[0]
    assert fs_rec == 48000 and np.array_equal(y_rec, senal[a:b]), f"la ventana no es [{a}, {b}) ({archivo})"
    assert p.comando == voz(senal[a:b], 48000).comando, f"Pantalla.comando = {p.comando!r} difiere de V4 directo"


@pytest.mark.aceptacion
def test_aceptacion_tiempos(cfg, ctl, clips_voz):
    """Mediana de 5 corridas para elegir (H2) y para comando_audio en HUELLA_CARGADA."""
    requerir(MODELO_H, META_H)
    assert not clips_voz.faltan, f"no hay clip de test de la notebook que V4 reconozca para {clips_voz.faltan}"
    c = ctl.Controlador.desde_archivos(cfg, microfono=False)
    entrada = entrada_por_nombre(c, "id2_v039 (reservada)")
    tiempos = []
    for _ in range(5):
        t0 = time.perf_counter()
        p = c.elegir(entrada)
        tiempos.append(time.perf_counter() - t0)
        assert p.regla == 1, f"elegir: regla {p.regla} ({p.mensaje!r})"
    meta = cfg["metas"]["respuesta_max_s"]
    assert statistics.median(tiempos) < meta, f"elegir tarda {statistics.median(tiempos):.3f} s (meta {meta} s)"

    y, fs, _ = clips_voz.clips["reconocer"]
    clip = a_48k(y, fs)
    total = round(3.4 * 48000)
    pre = round(cfg["app"]["pre_roll_s"] * 48000)
    clip = clip[:total - pre]
    audio = np.concatenate([ruido(pre, 3), clip, ruido(total - pre - clip.size, 4)])
    tiempos = []
    for _ in range(5):
        c.elegir(entrada)
        t0 = time.perf_counter()
        p = c.comando_audio(audio, 48000)
        tiempos.append(time.perf_counter() - t0)
        assert not p.mensaje.startswith("Error inesperado"), f"comando_audio falló: {p.mensaje!r}"
    limite = meta - cfg["app"]["post_roll_s"]
    assert statistics.median(tiempos) < limite, f"comando_audio tarda {statistics.median(tiempos):.3f} s (límite {limite} s)"
