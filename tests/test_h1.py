"""Tests de H1 · generador de huellas sintéticas y dataset.

Verifican la sección 5 de docs/contratos/h1_generador.md (versión 1, 16/10/2026).
La sección 6 es manual y se anota en el Documento 2.

Interpretaciones acordadas donde el contrato admitía más de una lectura:
- §1: los `<float ...>` exigen un float de Python (un int no cuenta) y los `fijo` se
  comparan por tipo y valor (400 != 400.0), como en la Fase 0. Si falta una sección
  entera (huellas.generador, huellas.variantes, huellas.intrusos o huellas), el
  ValueError nombra todas sus hojas nuevas (regla de la Fase 0).
- Distancia al borde, a los centros de píxel: min(x, y, ancho - 1 - x, alto - 1 - y).
  "Dentro de la imagen": 0 <= x <= ancho - 1 y 0 <= y <= alto - 1.
- §2, verticilo: el delta izquierdo queda a >= 40 px a la izquierda del núcleo de más a
  la izquierda, y el derecho a >= 40 px a la derecha del de más a la derecha.
- 5.3: "afuera" son los píxeles a más de 10 px de la máscara, en las dos comparaciones de
  las crestas oscuras. Los 5 puntos de control de Poincaré se toman repartidos entre los
  puntos de una grilla de 4 px que cumplen las distancias pedidas.
- 5.5: las singularidades de la variante se emparejan por cercanía (núcleo con núcleo y
  delta con delta), sin exigir el orden. Tienen que estar todos los núcleos; un delta que
  queda a menos de 10 px del borde, o afuera, no se verifica y puede faltar.
- 5.5, corte: si ningún ángulo sirve, corte_fraccion y corte_angulo_grados quedan en 0 y
  la máscara no cambia. Eso se acepta, pero cada tipo tiene que cortar en alguno de los
  casos del test. El ángulo usado se guarda en [0, 360). El disco de 20 px se toma
  literal: entero dentro de la máscara con corte. Se prueba f en los extremos de
  corte_periferico y también f = 0,40 (5.5 no limita f), porque con 0,15 a 0,20 el corte
  no se acerca a núcleos centrales y la regla de los 30 px no se pondría a prueba.
- 5.5, sal y pimienta: la fracción se mide sobre los píxeles que en la base no estaban en
  0 ni en 255.
- 5.7: metadatos.csv en UTF-8 sin BOM; un float puede escribirse con hasta 4 decimales
  (0.15 o 0.1500). "Crea exactamente" quiere decir que en destino no queda ningún otro
  archivo. Los PNG se leen con cv2.imdecode(np.fromfile(ruta), cv2.IMREAD_UNCHANGED), que
  decodifica igual que cv2.imread(ruta, cv2.IMREAD_UNCHANGED) y no falla con tildes en la
  ruta (Windows). generar_lamina se prueba con el cfg real: con el reducido hay 5
  variantes por identidad y la lámina muestra 6.

Criterios sumados a la sección 5 al escribir estos tests (se anotan en el Documento 2):
- Corte: 0° quita la derecha y 90° la parte de abajo; lo quitado es un semiplano del lado
  del ángulo guardado, y ese ángulo es el pedido más un múltiplo de 30°.
- sobrescribir=True no borra nada que no sea un PNG de train/, val/ o test/ o metadatos.csv.
- sortear_parametros respeta los rangos de un cfg con rangos más chicos (no los tiene
  fijos en el código).

src.config y los módulos de src.huellas se importan en fixtures: si algo todavía no
existe, fallan (o dan error) esos tests y no la colección entera. Lo lento es generar
bases y datasets: las bases de 5.3 y 5.5 se generan una vez y se reutilizan, y cada
dataset reducido se genera una sola vez.
"""

from __future__ import annotations

import copy
import csv
import importlib
import io
import itertools
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from scipy import ndimage

RAIZ = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Valores del contrato
# ---------------------------------------------------------------------------

TIPOS_CONTRATO = ("presilla", "verticilo", "arco_tienda", "arco")
TIPOS_AUTORIZADOS = ("presilla", "verticilo", "arco_tienda")
NUCLEOS_POR_TIPO = {"presilla": 1, "verticilo": 2, "arco_tienda": 1, "arco": 0}
DELTAS_POR_TIPO = {"presilla": 1, "verticilo": 2, "arco_tienda": 1, "arco": 0}

COLUMNAS_CONTRATO = (
    "archivo", "particion", "identidad", "intruso", "tipo", "semilla", "variante", "reservada_demo",
    "rot_grados", "escala", "tx_px", "ty_px", "elastica_px", "presion",
    "corte_fraccion", "corte_angulo_grados", "ruido_sigma", "sal_pimienta",
    "n_nucleos", "nucleos", "n_deltas", "deltas", "ref_x", "ref_y",
)
COLUMNAS_PLAN = (
    "archivo", "particion", "identidad", "intruso", "tipo", "semilla", "variante", "reservada_demo",
)
COLUMNAS_TEXTO = ("archivo", "particion", "tipo", "nucleos", "deltas")
COLUMNAS_ENTERAS = (
    "identidad", "intruso", "semilla", "variante", "reservada_demo", "n_nucleos", "n_deltas",
)
CAMPOS_PARAMETROS = (
    "rot_grados", "escala", "tx_px", "ty_px", "elastica_px", "presion",
    "corte_fraccion", "corte_angulo_grados", "ruido_sigma", "sal_pimienta",
)
COLUMNAS_REFERENCIA = ("ref_x", "ref_y")

SIN_CAMBIOS_CONTRATO = {
    "rot_grados": 0.0, "escala": 1.0, "tx_px": 0.0, "ty_px": 0.0, "elastica_px": 0.0,
    "presion": 0.0, "corte_fraccion": 0.0, "corte_angulo_grados": 0.0,
    "ruido_sigma": 0.0, "sal_pimienta": 0.0,
}

# 5.3 · 3 semillas por tipo: la de la identidad de config.yaml y las de los primeros
# intrusos de ese tipo (semilla_base + i, con semilla_base = 10000). Son las bases reales
# del dataset; si config.yaml cambia, siguen siendo semillas válidas para generar_base.
SEMILLAS_BASE = {
    "presilla": (101, 10001, 10005),
    "verticilo": (202, 10002, 10006),
    "arco_tienda": (303, 10003, 10007),
    "arco": (10004, 10008, 10012),
}
CASOS_BASE = [(tipo, semilla) for tipo in TIPOS_CONTRATO for semilla in SEMILLAS_BASE[tipo]]
IDS_BASE = [f"{tipo}-{semilla}" for tipo, semilla in CASOS_BASE]
CASOS_CON_SINGULARIDADES = [caso for caso in CASOS_BASE if caso[0] != "arco"]
IDS_CON_SINGULARIDADES = [f"{tipo}-{semilla}" for tipo, semilla in CASOS_CON_SINGULARIDADES]

# 5.6 · (autorizadas, intrusos) por partición, con el cfg real y con el reducido.
CONTEOS_PLAN = {
    "real": {"train": (72, 0), "val": (24, 90), "test": (24, 90)},
    "reducida": {"train": (9, 0), "val": (3, 4), "test": (3, 4)},
}
CONFIGS_PLAN = list(CONTEOS_PLAN)

# 5.1 · claves nuevas de config.yaml
CLAVES_H1 = [
    "huellas.generador.tamano_px",
    "huellas.generador.periodo_base_px",
    "huellas.generador.iteraciones",
    "huellas.generador.orientaciones",
    "huellas.variantes.prob_corte",
    "huellas.variantes.elastica_max_px",
    "huellas.variantes.presion_max",
    "huellas.variantes.ruido_sigma_max",
    "huellas.variantes.sal_pimienta_max",
    "huellas.intrusos.tipos",
]
SECCIONES_CON_CLAVES_H1 = ["huellas.generador", "huellas.variantes", "huellas.intrusos", "huellas"]


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------


def importar(nombre: str):
    """Importa un módulo de src con la raíz del repo en sys.path; si falla, el test falla."""
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))
    try:
        return importlib.import_module(nombre)
    except Exception as error:  # ModuleNotFoundError, SyntaxError, etc.
        motivo = f"{type(error).__name__}: {error}"
    pytest.fail(f"No se pudo importar {nombre}: {motivo}", pytrace=False)


def atributo(modulo, nombre: str):
    if not hasattr(modulo, nombre):
        pytest.fail(f"{modulo.__name__} no define {nombre}.", pytrace=False)
    return getattr(modulo, nombre)


def tipo_de(valor) -> str:
    if isinstance(valor, list):
        return "list[" + ", ".join(tipo_de(v) for v in valor) + "]"
    return type(valor).__name__


def es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def es_float(valor) -> bool:
    return isinstance(valor, float)


def iguales_exactos(a, b) -> bool:
    """Igualdad de valor y de tipo, recursiva (400 != 400.0, True != 1)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(iguales_exactos(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(iguales_exactos(a[k], b[k]) for k in a)
    return a == b


def obtener(cfg: dict, camino: str):
    """Valor de una clave con camino separado por puntos; falla si no está."""
    actual = cfg
    for parte in camino.split("."):
        if not isinstance(actual, dict) or parte not in actual:
            pytest.fail(f"config.yaml no tiene la clave obligatoria {camino!r}.", pytrace=False)
        actual = actual[parte]
    return actual


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


def escribir_yaml(ruta: Path, datos) -> Path:
    with open(ruta, "w", encoding="utf-8") as archivo:
        yaml.safe_dump(datos, archivo, allow_unicode=True, sort_keys=False)
    return ruta


def nombra(mensaje: str, camino: str) -> bool:
    """True si `camino` aparece completo en el mensaje (no como parte de otro camino)."""
    return re.search(r"(?<![\w.])" + re.escape(camino) + r"(?!\w)", mensaje) is not None


def lanza(tipo: type[BaseException], funcion, *args, que: str, **kwargs) -> BaseException:
    """Llama a funcion(*args, **kwargs) y devuelve la excepción `tipo`; falla si no se lanza.

    Una excepción de otro tipo se deja pasar para que se vea su traceback.
    """
    try:
        funcion(*args, **kwargs)
    except tipo as error:
        return error
    pytest.fail(f"{que}: se esperaba {tipo.__name__} y no se lanzó ninguna excepción.", pytrace=False)


def resumen(problemas: list[str], maximo: int = 15) -> str:
    """Las primeras líneas de una lista de problemas, para los mensajes de error."""
    lineas = [f"  - {p}" for p in problemas[:maximo]]
    if len(problemas) > maximo:
        lineas.append(f"  ... y {len(problemas) - maximo} más")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Geometría y mediciones de la sección 0
# ---------------------------------------------------------------------------


def tamano(cfg: dict) -> tuple[int, int]:
    """(alto, ancho) de huellas.generador.tamano_px."""
    alto, ancho = obtener(cfg, "huellas.generador.tamano_px")
    return int(alto), int(ancho)


def como_puntos(valor, que: str) -> list[tuple[float, float]]:
    """Lista de (x, y) a partir de una secuencia de pares; falla si no tiene esa forma."""
    puntos, valido = [], True
    try:
        for par in valor:
            if len(par) != 2:
                valido = False
                break
            x, y = float(par[0]), float(par[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                valido = False
                break
            puntos.append((x, y))
    except (TypeError, ValueError):
        valido = False
    if not valido:
        pytest.fail(f"{que} tiene que ser una secuencia de pares (x, y) finitos; es {valor!r}.", pytrace=False)
    return puntos


def fmt(punto) -> str:
    return f"({punto[0]:.2f}, {punto[1]:.2f})"


def distancia_al_borde(punto, alto: int, ancho: int) -> float:
    x, y = punto
    return min(x, y, ancho - 1 - x, alto - 1 - y)


def dentro_de_la_imagen(punto, alto: int, ancho: int) -> bool:
    x, y = punto
    return 0 <= x <= ancho - 1 and 0 <= y <= alto - 1


def distancia_al_mas_cercano(punto, candidatos) -> float:
    return min((math.dist(punto, c) for c in candidatos), default=math.inf)


def indices_poincare(theta: np.ndarray, puntos, r: float = 8.0, n: int = 72) -> list[float]:
    """Índice de Poincaré de la sección 0 en cada punto (x, y): n + 1 puntos sobre un círculo de radio r."""
    theta = np.asarray(theta, dtype=np.float64)
    cos2, sen2 = np.cos(2 * theta), np.sin(2 * theta)
    phi = 2 * np.pi * np.arange(n + 1) / n
    indices = []
    for x0, y0 in puntos:
        xs, ys = x0 + r * np.cos(phi), y0 + r * np.sin(phi)
        c = ndimage.map_coordinates(cos2, [ys, xs], order=1)
        s = ndimage.map_coordinates(sen2, [ys, xs], order=1)
        saltos = np.diff(0.5 * np.arctan2(s, c))
        saltos = np.pi / 2 - np.mod(np.pi / 2 - saltos, np.pi)  # al intervalo (-π/2, π/2]
        indices.append(float(saltos.sum() / (2 * np.pi)))
    return indices


def periodo_medido(imagen: np.ndarray) -> float:
    """Período de las crestas según la sección 0 (recorte central de 192 × 192 px)."""
    alto, ancho = imagen.shape
    fila, columna = (alto - 192) // 2, (ancho - 192) // 2
    recorte = imagen[fila:fila + 192, columna:columna + 192].astype(np.float64)
    recorte = (recorte - recorte.mean()) * np.outer(np.hanning(192), np.hanning(192))
    potencia = np.abs(np.fft.fft2(recorte)) ** 2
    frecuencias = np.fft.fftfreq(192)
    f = np.hypot(frecuencias[:, None], frecuencias[None, :])
    banda = (f > 1 / 20) & (f < 1 / 5)
    if potencia[banda].sum() == 0:
        return float("nan")
    return float(1 / ((f[banda] * potencia[banda]).sum() / potencia[banda].sum()))


def correlacion(imagen_a, mascara_a, imagen_b, mascara_b) -> float:
    """Pearson de los píxeles donde las dos máscaras valen True (sección 0)."""
    comun = np.asarray(mascara_a, dtype=bool) & np.asarray(mascara_b, dtype=bool)
    a = np.asarray(imagen_a, dtype=np.float64)[comun]
    b = np.asarray(imagen_b, dtype=np.float64)[comun]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def transformacion_afin(punto, datos: dict, alto: int, ancho: int) -> tuple[float, float]:
    """Fórmula afín de la sección 0 (rotación y escala alrededor del centro, más traslación)."""
    x, y = punto
    cx, cy = (ancho - 1) / 2, (alto - 1) / 2
    a, s = math.radians(datos["rot_grados"]), datos["escala"]
    dx, dy = x - cx, y - cy
    return (
        cx + s * (math.cos(a) * dx + math.sin(a) * dy) + datos["tx_px"],
        cy + s * (-math.sin(a) * dx + math.cos(a) * dy) + datos["ty_px"],
    )


def evaluar_bilineal(mapa, punto) -> float:
    x, y = punto
    return float(ndimage.map_coordinates(np.asarray(mapa, dtype=np.float64), [[y], [x]], order=1)[0])


def disco(x: float, y: float, radio: float, alto: int, ancho: int) -> np.ndarray:
    filas, columnas = np.ogrid[0:alto, 0:ancho]
    return (columnas - x) ** 2 + (filas - y) ** 2 <= radio ** 2


def grilla_lejos_de(singularidades, alto, ancho, *, distancia, margen, paso) -> list[tuple[float, float]]:
    """Puntos de una grilla a >= margen px de los bordes y a >= distancia de cada singularidad."""
    puntos = []
    for y in range(margen, alto - margen, paso):
        for x in range(margen, ancho - margen, paso):
            if all(math.dist((x, y), s) >= distancia for s in singularidades):
                puntos.append((float(x), float(y)))
    return puntos


def puntos_de_control(mascara, singularidades, *, distancia, margen, cantidad=5, paso=4):
    """`cantidad` puntos de la máscara que cumplen las distancias, repartidos de arriba abajo."""
    alto, ancho = mascara.shape
    candidatos = [
        p
        for p in grilla_lejos_de(singularidades, alto, ancho, distancia=distancia, margen=margen, paso=paso)
        if mascara[int(p[1]), int(p[0])]
    ]
    if len(candidatos) <= cantidad:
        return candidatos
    indices = np.linspace(0, len(candidatos) - 1, cantidad + 2)[1:-1]
    return [candidatos[int(round(i))] for i in indices]


# ---------------------------------------------------------------------------
# Parámetros de variantes
# ---------------------------------------------------------------------------


def parametros(var, **cambios):
    """ParametrosVariante con los valores de SIN_CAMBIOS del contrato, salvo `cambios`."""
    datos = {**SIN_CAMBIOS_CONTRATO, **{k: float(v) for k, v in cambios.items()}}
    return atributo(var, "ParametrosVariante")(**datos)


def datos_al_azar(cfg: dict, rng: np.random.Generator, **cambios) -> dict:
    """Parámetros sorteados por el test en los rangos de cfg (sin corte, salvo `cambios`)."""
    v = cfg["huellas"]["variantes"]
    datos = {
        "rot_grados": rng.uniform(-v["rotacion_max_grados"], v["rotacion_max_grados"]),
        "escala": rng.uniform(1 - v["escala_max"], 1 + v["escala_max"]),
        "tx_px": rng.uniform(-v["traslacion_max_px"], v["traslacion_max_px"]),
        "ty_px": rng.uniform(-v["traslacion_max_px"], v["traslacion_max_px"]),
        "elastica_px": rng.uniform(0, v["elastica_max_px"]),
        "presion": rng.uniform(-v["presion_max"], v["presion_max"]),
        "corte_fraccion": 0.0,
        "corte_angulo_grados": 0.0,
        "ruido_sigma": rng.uniform(0, v["ruido_sigma_max"]),
        "sal_pimienta": rng.uniform(0, v["sal_pimienta_max"]),
    }
    datos.update(cambios)
    return {clave: float(valor) for clave, valor in datos.items()}


def aplicar(var, base, params, semilla):
    """aplicar_variante con un rng nuevo de esa semilla."""
    return atributo(var, "aplicar_variante")(base, params, np.random.default_rng(semilla))


def campos_de(params, que: str) -> dict:
    """Los 10 campos de un ParametrosVariante como floats."""
    try:
        return {campo: float(getattr(params, campo)) for campo in CAMPOS_PARAMETROS}
    except (AttributeError, TypeError, ValueError) as error:
        motivo = f"{type(error).__name__}: {error}"
    pytest.fail(f"{que}: {params!r} no tiene los campos numéricos de ParametrosVariante ({motivo}).", pytrace=False)


def problemas_de_rango(p: dict, cfg: dict, *, que: str, angulo_redondeado: bool = False) -> list[str]:
    """Problemas de un juego de parámetros con los rangos de cfg (secciones 3 y 5.4).

    Con angulo_redondeado=True se admite 360.0, que puede salir de redondear a 4 decimales
    un ángulo como 359.99996.
    """
    v = cfg["huellas"]["variantes"]
    rot, tras, esc = v["rotacion_max_grados"], v["traslacion_max_px"], v["escala_max"]
    tolerancia = 1e-9
    reglas = [
        ("rot_grados", -rot, rot),
        ("escala", 1 - esc, 1 + esc),
        ("tx_px", -tras, tras),
        ("ty_px", -tras, tras),
        ("elastica_px", 0, v["elastica_max_px"]),
        ("presion", -v["presion_max"], v["presion_max"]),
        ("ruido_sigma", 0, v["ruido_sigma_max"]),
        ("sal_pimienta", 0, v["sal_pimienta_max"]),
    ]
    problemas = []
    for campo, minimo, maximo in reglas:
        valor = p[campo]
        if not (minimo - tolerancia <= valor <= maximo + tolerancia):
            problemas.append(f"{que}: {campo} = {valor} fuera de [{minimo}, {maximo}]")
    fraccion, angulo = p["corte_fraccion"], p["corte_angulo_grados"]
    corte_min, corte_max = v["corte_periferico"]
    if fraccion == 0:
        if angulo != 0:
            problemas.append(f"{que}: corte_fraccion = 0 y corte_angulo_grados = {angulo} (tiene que valer 0)")
    else:
        if not (corte_min - tolerancia <= fraccion <= corte_max + tolerancia):
            problemas.append(
                f"{que}: corte_fraccion = {fraccion} no es 0 ni está en [{corte_min}, {corte_max}]"
            )
        tope = angulo <= 360 if angulo_redondeado else angulo < 360
        if not (angulo >= 0 and tope):
            problemas.append(f"{que}: corte_angulo_grados = {angulo} fuera de [0, 360)")
    return problemas


# ---------------------------------------------------------------------------
# Archivos y metadatos.csv
# ---------------------------------------------------------------------------

PATRON_ENTERO = re.compile(r"-?\d+")
PATRON_DECIMAL = re.compile(r"-?\d+(?:\.\d{1,4})?")


def leer_png(ruta: Path):
    """Imagen decodificada como cv2.imread(ruta, cv2.IMREAD_UNCHANGED); None si no se puede."""
    try:
        datos = np.fromfile(str(ruta), dtype=np.uint8)
    except OSError:
        return None
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_UNCHANGED)


def png_chico() -> bytes:
    """Un PNG válido de 8 × 8 px, para simular datos viejos."""
    ok, codificado = cv2.imencode(".png", np.full((8, 8), 200, dtype=np.uint8))
    assert ok
    return codificado.tobytes()


def foto_de_carpeta(carpeta: Path) -> dict:
    """{ruta relativa: (tipo, contenido, mtime_ns)} de todo lo que hay dentro de `carpeta`."""
    foto = {}
    for ruta in sorted(carpeta.rglob("*")):
        relativa = ruta.relative_to(carpeta).as_posix()
        if ruta.is_dir():
            foto[relativa] = ("carpeta", None, None)
        else:
            foto[relativa] = ("archivo", ruta.read_bytes(), ruta.stat().st_mtime_ns)
    return foto


def leer_metadatos(destino) -> tuple[list[str], list[dict[str, str]]]:
    """(encabezado, filas como dicts de texto) de destino/metadatos.csv."""
    ruta = Path(destino) / "metadatos.csv"
    if not ruta.is_file():
        pytest.fail(f"No existe {ruta}.", pytrace=False)
    crudo = ruta.read_bytes()
    if crudo.startswith(b"\xef\xbb\xbf"):
        pytest.fail("metadatos.csv empieza con BOM: tiene que estar en UTF-8 sin BOM.", pytrace=False)
    try:
        texto = crudo.decode("utf-8")
    except UnicodeDecodeError as error:
        motivo = str(error)
    else:
        motivo = None
    if motivo is not None:
        pytest.fail(f"metadatos.csv no está en UTF-8 ({motivo}).", pytrace=False)
    filas = list(csv.reader(io.StringIO(texto, newline="")))
    if not filas:
        pytest.fail("metadatos.csv está vacío.", pytrace=False)
    encabezado, cuerpo = filas[0], filas[1:]
    malas = [n for n, fila in enumerate(cuerpo, start=2) if len(fila) != len(encabezado)]
    if malas:
        pytest.fail(
            f"metadatos.csv tiene filas con una cantidad de campos distinta del encabezado "
            f"({len(encabezado)}): líneas {malas[:10]}.",
            pytrace=False,
        )
    return encabezado, [dict(zip(encabezado, fila)) for fila in cuerpo]


def convertir_fila(fila: dict[str, str], linea: int) -> tuple[dict, list[str]]:
    """Fila del CSV con los tipos de la sección 4; devuelve (fila convertida, problemas)."""
    convertida, problemas = {}, []
    for columna in COLUMNAS_TEXTO:
        convertida[columna] = fila.get(columna, "")
    for columna in COLUMNAS_ENTERAS:
        texto = fila.get(columna, "")
        if PATRON_ENTERO.fullmatch(texto):
            convertida[columna] = int(texto)
        else:
            convertida[columna] = None
            problemas.append(f"línea {linea}, {columna} = {texto!r}: no es un entero")
    for columna in CAMPOS_PARAMETROS + COLUMNAS_REFERENCIA:
        texto = fila.get(columna, "")
        if columna in COLUMNAS_REFERENCIA and texto == "":
            convertida[columna] = None
        elif PATRON_DECIMAL.fullmatch(texto):
            convertida[columna] = float(texto)
        else:
            convertida[columna] = None
            problemas.append(f"línea {linea}, {columna} = {texto!r}: no es un número con hasta 4 decimales")
    return convertida, problemas


def filas_convertidas(destino) -> list[dict]:
    """Filas de metadatos.csv con sus tipos; falla si alguna no se puede convertir."""
    _, filas = leer_metadatos(destino)
    convertidas, problemas = [], []
    for linea, fila in enumerate(filas, start=2):
        convertida, problemas_fila = convertir_fila(fila, linea)
        convertidas.append(convertida)
        problemas.extend(problemas_fila)
    if problemas:
        pytest.fail("metadatos.csv tiene valores mal formados:\n" + resumen(problemas), pytrace=False)
    return convertidas


def puntos_json(texto: str):
    """Lista de (x, y) de un JSON como [[160.25, 181.5]]; None si no tiene esa forma."""
    try:
        valor = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(valor, list):
        return None
    puntos = []
    for par in valor:
        numeros = isinstance(par, list) and len(par) == 2 and all(
            isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in par
        )
        if not numeros:
            return None
        puntos.append((float(par[0]), float(par[1])))
    return puntos


def entradas(plan, que: str) -> list[dict]:
    """El plan como lista de dicts con las claves del plan; falla si no tiene esa forma."""
    if not isinstance(plan, list) or not all(isinstance(e, dict) for e in plan):
        pytest.fail(f"{que} tiene que devolver una lista de dicts; devolvió {tipo_de(plan)}.", pytrace=False)
    for numero, entrada in enumerate(plan, start=1):
        faltan = [c for c in COLUMNAS_PLAN if c not in entrada]
        if faltan:
            pytest.fail(f"{que}: a la entrada {numero} le faltan las claves {faltan}.", pytrace=False)
    return plan


def configuracion_reducida(cfg: dict) -> dict:
    """Copia profunda del cfg real con los conteos del ejemplo de 5.6 (solo cambian conteos)."""
    reducida = copy.deepcopy(cfg)
    huellas = reducida["huellas"]
    huellas["variantes_por_identidad"] = 5
    huellas["particion"]["train"] = 3
    huellas["particion"]["val"] = 1
    huellas["particion"]["test"] = 1
    huellas["reservadas_demo"] = 1
    huellas["intrusos"]["cantidad"] = 4
    huellas["intrusos"]["variantes_por_intruso"] = 2
    huellas["intrusos"]["val"] = 2
    huellas["intrusos"]["test"] = 2
    return reducida


def configuracion_con_rangos_chicos(cfg: dict) -> dict:
    """Copia profunda del cfg real con rangos de variantes más chicos (criterio sumado)."""
    chica = copy.deepcopy(cfg)
    chica["huellas"]["variantes"].update(
        {
            "rotacion_max_grados": 3,
            "traslacion_max_px": 2,
            "escala_max": 0.01,
            "corte_periferico": [0.16, 0.17],
            "prob_corte": 0.5,
            "elastica_max_px": 1.0,
            "presion_max": 0.2,
            "ruido_sigma_max": 2.0,
            "sal_pimienta_max": 0.005,
        }
    )
    return chica


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cargar_config():
    funcion = getattr(importar("src.config"), "cargar_config", None)
    if not callable(funcion):
        pytest.fail("src/config.py no define la función cargar_config.", pytrace=False)
    return funcion


@pytest.fixture(scope="module")
def cfg(cargar_config) -> dict:
    """config.yaml real, leído una vez con cargar_config(). Los tests no lo modifican."""
    datos = cargar_config()
    if not isinstance(datos, dict):
        pytest.fail(f"cargar_config() tiene que devolver un dict; devolvió {tipo_de(datos)}.", pytrace=False)
    return datos


@pytest.fixture(scope="module")
def cfg_reducido(cfg) -> dict:
    return configuracion_reducida(cfg)


@pytest.fixture(scope="module")
def gen():
    return importar("src.huellas.generador")


@pytest.fixture(scope="module")
def var():
    return importar("src.huellas.variantes")


@pytest.fixture(scope="module")
def ds():
    return importar("src.huellas.dataset")


@pytest.fixture(scope="module")
def bases(gen, cfg):
    """Función (tipo, semilla) -> copia de generar_base(tipo, semilla, cfg real).

    Cada base se genera una sola vez en el módulo y cada test recibe una copia profunda,
    así un test que la modifique no afecta a los demás.
    """
    generar = atributo(gen, "generar_base")
    cache: dict = {}

    def obtener(tipo: str, semilla: int):
        clave = (tipo, semilla)
        if clave not in cache:
            try:
                cache[clave] = generar(tipo, semilla, copy.deepcopy(cfg))
            except Exception as error:
                cache[clave] = error
                raise
        guardada = cache[clave]
        if isinstance(guardada, Exception):
            pytest.fail(
                f"generar_base({tipo!r}, {semilla}, cfg) ya falló antes en este módulo: "
                f"{type(guardada).__name__}: {guardada}",
                pytrace=False,
            )
        return copy.deepcopy(guardada)

    return obtener


# ---------------------------------------------------------------------------
# 5.1 Configuración
# ---------------------------------------------------------------------------

VALORES_FIJOS_H1 = {
    "huellas.generador.tamano_px": [400, 320],
    "huellas.generador.periodo_base_px": [8.7, 9.3],
    "huellas.intrusos.tipos": ["presilla", "verticilo", "arco_tienda", "arco"],
}

RESTRICCIONES_H1 = [
    ("huellas.generador.iteraciones", "un entero >= 5", lambda v: es_entero(v) and v >= 5),
    ("huellas.generador.orientaciones", "un entero >= 8", lambda v: es_entero(v) and v >= 8),
    ("huellas.variantes.prob_corte", "un float en [0, 1]", lambda v: es_float(v) and 0 <= v <= 1),
    ("huellas.variantes.elastica_max_px", "un float en [0, 5]", lambda v: es_float(v) and 0 <= v <= 5),
    ("huellas.variantes.presion_max", "un float en (0, 1]", lambda v: es_float(v) and 0 < v <= 1),
    ("huellas.variantes.ruido_sigma_max", "un float en [0, 30]", lambda v: es_float(v) and 0 <= v <= 30),
    (
        "huellas.variantes.sal_pimienta_max",
        "un float en [0, 0.05]",
        lambda v: es_float(v) and 0 <= v <= 0.05,
    ),
]


@pytest.mark.parametrize("camino", CLAVES_H1)
def test_config_tiene_clave_nueva_de_h1(cfg, camino):
    obtener(cfg, camino)


@pytest.mark.parametrize(
    ("camino", "esperado"), list(VALORES_FIJOS_H1.items()), ids=list(VALORES_FIJOS_H1)
)
def test_config_valor_fijo_de_h1(cfg, camino, esperado):
    valor = obtener(cfg, camino)
    assert iguales_exactos(valor, esperado), (
        f"{camino}: se esperaba {esperado!r} ({tipo_de(esperado)}) y hay {valor!r} ({tipo_de(valor)})."
    )


@pytest.mark.parametrize(
    ("camino", "descripcion", "es_valido"), RESTRICCIONES_H1, ids=[r[0] for r in RESTRICCIONES_H1]
)
def test_config_valor_de_h1_con_tipo_y_rango(cfg, camino, descripcion, es_valido):
    valor = obtener(cfg, camino)
    pista = ""
    if es_entero(valor) and "float" in descripcion:
        pista = f" (escribilo como {valor}.0)"
    assert es_valido(valor), f"{camino} tiene que ser {descripcion}; hay {valor!r} ({tipo_de(valor)}).{pista}"


@pytest.mark.parametrize("camino", CLAVES_H1)
def test_cargar_config_sin_clave_de_h1_lanza_value_error_que_la_nombra(cargar_config, cfg, tmp_path, camino):
    ruta = escribir_yaml(tmp_path / "config.yaml", sin_clave(cfg, camino))
    error = lanza(ValueError, cargar_config, ruta, que=f"Config sin {camino}")
    assert nombra(str(error), camino), f"El ValueError no nombra {camino!r}. Mensaje: {error}"


@pytest.mark.parametrize("seccion", SECCIONES_CON_CLAVES_H1)
def test_cargar_config_sin_seccion_nombra_sus_hojas_de_h1(cargar_config, cfg, tmp_path, seccion):
    ruta = escribir_yaml(tmp_path / "config.yaml", sin_clave(cfg, seccion))
    error = lanza(ValueError, cargar_config, ruta, que=f"Config sin la sección {seccion}")
    hojas = [c for c in CLAVES_H1 if c.startswith(seccion + ".")]
    no_nombradas = [h for h in hojas if not nombra(str(error), h)]
    assert not no_nombradas, (
        f"Sin la sección {seccion!r}, el ValueError no nombra estas hojas de H1: {no_nombradas}. "
        f"Mensaje: {error}"
    )


# ---------------------------------------------------------------------------
# 5.2 Campo de orientación
# ---------------------------------------------------------------------------

TAMANO_CAMPO = (400, 320)  # (alto, ancho)

# (núcleos, deltas) en (x, y). El primero es el ejemplo del contrato.
CAMPOS_A_MANO = {
    "ejemplo_del_contrato": (((150.0, 150.0),), ((230.0, 270.0),)),
    "dos_nucleos_y_dos_deltas": (((150.0, 180.0), (176.0, 192.0)), ((100.0, 250.0), (230.0, 245.0))),
    "nucleo_y_delta_a_20_px": (((160.0, 170.0),), ((160.0, 190.0),)),
}
CAMPOS_PARA_FORMA = {"sin_singularidades": ((), ()), **CAMPOS_A_MANO}


def singularidades(gen, nucleos, deltas):
    return atributo(gen, "Singularidades")(nucleos=tuple(nucleos), deltas=tuple(deltas))


@pytest.mark.parametrize("theta_inf", [0.0, 1.2, -2.0, 7.0])
@pytest.mark.parametrize("caso", list(CAMPOS_PARA_FORMA))
def test_campo_orientacion_es_float64_alto_ancho_en_0_pi(gen, caso, theta_inf):
    nucleos, deltas = CAMPOS_PARA_FORMA[caso]
    campo = atributo(gen, "campo_orientacion")(singularidades(gen, nucleos, deltas), TAMANO_CAMPO, theta_inf)
    assert isinstance(campo, np.ndarray), f"Devolvió {tipo_de(campo)}, no un np.ndarray."
    assert campo.dtype == np.float64, f"dtype {campo.dtype}; se esperaba float64."
    assert campo.shape == TAMANO_CAMPO, f"Forma {campo.shape}; se esperaba (alto, ancho) = {TAMANO_CAMPO}."
    assert np.all(np.isfinite(campo)), "El campo tiene valores no finitos."
    minimo, maximo = float(campo.min()), float(campo.max())
    assert minimo >= 0 and maximo < np.pi, f"Valores en [{minimo}, {maximo}]; tienen que estar en [0, π)."


@pytest.mark.parametrize("caso", list(CAMPOS_A_MANO))
def test_campo_orientacion_indice_de_poincare_en_nucleos_deltas_y_lejos(gen, caso):
    nucleos, deltas = CAMPOS_A_MANO[caso]
    campo = atributo(gen, "campo_orientacion")(singularidades(gen, nucleos, deltas), TAMANO_CAMPO)
    lejanos = grilla_lejos_de(nucleos + deltas, *TAMANO_CAMPO, distancia=40, margen=20, paso=20)
    assert len(lejanos) >= 20, f"Solo hay {len(lejanos)} puntos lejanos para controlar."
    casos = (
        [("núcleo", p, 0.5) for p in nucleos]
        + [("delta", p, -0.5) for p in deltas]
        + [("punto a >= 40 px de todo", p, 0.0) for p in lejanos]
    )
    indices = indices_poincare(campo, [p for _, p, _ in casos])
    problemas = [
        f"{clase} {fmt(punto)}: índice {indice:+.3f} (esperado {esperado:+.1f} ± 0,05)"
        for (clase, punto, esperado), indice in zip(casos, indices)
        if abs(indice - esperado) > 0.05
    ]
    assert not problemas, f"Caso {caso}:\n" + resumen(problemas)


@pytest.mark.parametrize("theta_inf", [0.0, 0.4, 2.0, -0.7, 3.5, 10.0])
def test_campo_orientacion_sin_singularidades_es_theta_inf_mod_pi(gen, theta_inf):
    campo = atributo(gen, "campo_orientacion")(singularidades(gen, (), ()), TAMANO_CAMPO, theta_inf)
    campo = np.asarray(campo, dtype=np.float64)
    esperado = theta_inf % math.pi
    assert campo.shape == TAMANO_CAMPO, f"Forma {campo.shape}; se esperaba {TAMANO_CAMPO}."
    error = float(np.abs(campo - esperado).max())
    assert error <= 1e-12, (
        f"Con θ∞ = {theta_inf} se esperaba {esperado:.12f} en toda la imagen; "
        f"hay valores entre {campo.min():.12f} y {campo.max():.12f}."
    )


# ---------------------------------------------------------------------------
# 5.3 Huella base
# ---------------------------------------------------------------------------


def test_tipos_es_la_tupla_del_contrato(gen):
    tipos = atributo(gen, "TIPOS")
    assert tipos == TIPOS_CONTRATO and isinstance(tipos, tuple), (
        f"TIPOS tiene que ser la tupla {TIPOS_CONTRATO}; es {tipos!r}."
    )


@pytest.mark.parametrize(("tipo", "semilla"), CASOS_BASE, ids=IDS_BASE)
def test_base_imagen_mascara_y_orientacion_con_tipo_y_tamano(cfg, bases, tipo, semilla):
    huella = bases(tipo, semilla)
    forma = tamano(cfg)
    problemas = []
    for campo, dtype in (("imagen", np.uint8), ("mascara", np.bool_), ("orientacion", np.float64)):
        valor = getattr(huella, campo, None)
        if not isinstance(valor, np.ndarray):
            problemas.append(f"{campo} no es un np.ndarray ({tipo_de(valor)})")
            continue
        if valor.dtype != dtype:
            problemas.append(f"{campo} es {valor.dtype}; se esperaba {np.dtype(dtype)}")
        if valor.shape != forma:
            problemas.append(f"{campo} tiene forma {valor.shape}; se esperaba tamano_px = {forma}")
    orientacion = getattr(huella, "orientacion", None)
    if isinstance(orientacion, np.ndarray) and orientacion.size:
        minimo, maximo = float(np.min(orientacion)), float(np.max(orientacion))
        if not (np.all(np.isfinite(orientacion)) and minimo >= 0 and maximo < np.pi):
            problemas.append(f"orientacion tiene valores en [{minimo}, {maximo}]; tienen que estar en [0, π)")
    if getattr(huella, "tipo", None) != tipo or getattr(huella, "semilla", None) != semilla:
        problemas.append(
            f"tipo/semilla = {getattr(huella, 'tipo', None)!r}/{getattr(huella, 'semilla', None)!r}; "
            f"se esperaba {tipo!r}/{semilla}"
        )
    assert not problemas, f"generar_base({tipo!r}, {semilla}):\n" + resumen(problemas)


@pytest.mark.parametrize(("tipo", "semilla"), CASOS_BASE, ids=IDS_BASE)
def test_base_singularidades_segun_la_tabla_del_tipo(bases, tipo, semilla):
    huella = bases(tipo, semilla)
    nucleos = como_puntos(huella.singularidades.nucleos, "singularidades.nucleos")
    deltas = como_puntos(huella.singularidades.deltas, "singularidades.deltas")
    que = f"generar_base({tipo!r}, {semilla}), núcleos {[fmt(p) for p in nucleos]}, deltas {[fmt(p) for p in deltas]}"
    esperado = (NUCLEOS_POR_TIPO[tipo], DELTAS_POR_TIPO[tipo])
    assert (len(nucleos), len(deltas)) == esperado, (
        f"{que}: tiene {len(nucleos)} núcleos y {len(deltas)} deltas; el {tipo} lleva {esperado[0]} y {esperado[1]}."
    )
    problemas = []
    if tipo == "presilla":
        (nx, ny), (dx, dy) = nucleos[0], deltas[0]
        if dy - ny < 40:
            problemas.append(f"el delta está {dy - ny:.2f} px más abajo que el núcleo (mínimo 40)")
        if abs(dx - nx) < 40:
            problemas.append(f"el delta está {abs(dx - nx):.2f} px hacia un costado (mínimo 40)")
    elif tipo == "verticilo":
        separacion = math.dist(nucleos[0], nucleos[1])
        if not 18 <= separacion <= 40:
            problemas.append(f"los núcleos están a {separacion:.2f} px entre sí (entre 18 y 40)")
        izquierdo, derecho = sorted(deltas)
        x_min = min(x for x, _ in nucleos)
        x_max = max(x for x, _ in nucleos)
        y_medio = (nucleos[0][1] + nucleos[1][1]) / 2
        if x_min - izquierdo[0] < 40:
            problemas.append(
                f"el delta izquierdo está {x_min - izquierdo[0]:.2f} px a la izquierda del núcleo "
                f"de más a la izquierda (mínimo 40)"
            )
        if derecho[0] - x_max < 40:
            problemas.append(
                f"el delta derecho está {derecho[0] - x_max:.2f} px a la derecha del núcleo "
                f"de más a la derecha (mínimo 40)"
            )
        for nombre, delta in (("izquierdo", izquierdo), ("derecho", derecho)):
            if delta[1] - y_medio < 40:
                problemas.append(
                    f"el delta {nombre} está {delta[1] - y_medio:.2f} px más abajo que el punto medio "
                    f"de los núcleos (mínimo 40)"
                )
    elif tipo == "arco_tienda":
        (nx, ny), (dx, dy) = nucleos[0], deltas[0]
        if not 15 <= dy - ny <= 40:
            problemas.append(f"el delta está {dy - ny:.2f} px más abajo que el núcleo (entre 15 y 40)")
        if abs(dx - nx) > 10:
            problemas.append(f"el delta está a {abs(dx - nx):.2f} px en x del núcleo (máximo 10)")
    assert not problemas, f"{que}:\n" + resumen(problemas)


@pytest.mark.parametrize(("tipo", "semilla"), CASOS_BASE, ids=IDS_BASE)
def test_base_singularidades_lejos_de_bordes_separadas_y_nucleos_en_la_mascara(bases, tipo, semilla):
    huella = bases(tipo, semilla)
    mascara = np.asarray(huella.mascara, dtype=bool)
    alto, ancho = mascara.shape
    nucleos = como_puntos(huella.singularidades.nucleos, "singularidades.nucleos")
    deltas = como_puntos(huella.singularidades.deltas, "singularidades.deltas")
    todas = [("núcleo", p) for p in nucleos] + [("delta", p) for p in deltas]
    problemas = []
    for clase, punto in todas:
        borde = distancia_al_borde(punto, alto, ancho)
        if borde < 20:
            problemas.append(f"{clase} {fmt(punto)} a {borde:.2f} px del borde (mínimo 20)")
    for (clase_a, a), (clase_b, b) in itertools.combinations(todas, 2):
        if math.dist(a, b) < 18:
            problemas.append(f"{clase_a} {fmt(a)} y {clase_b} {fmt(b)} a {math.dist(a, b):.2f} px (mínimo 18)")
    for punto in nucleos:
        fila, columna = int(round(punto[1])), int(round(punto[0]))
        if not (0 <= fila < alto and 0 <= columna < ancho and mascara[fila, columna]):
            problemas.append(f"núcleo {fmt(punto)} fuera de la máscara")
    assert not problemas, f"generar_base({tipo!r}, {semilla}):\n" + resumen(problemas)


@pytest.mark.parametrize(("tipo", "semilla"), CASOS_BASE, ids=IDS_BASE)
def test_base_indice_de_poincare_en_singularidades_y_lejos_de_ellas(bases, tipo, semilla):
    huella = bases(tipo, semilla)
    mascara = np.asarray(huella.mascara, dtype=bool)
    nucleos = como_puntos(huella.singularidades.nucleos, "singularidades.nucleos")
    deltas = como_puntos(huella.singularidades.deltas, "singularidades.deltas")
    if tipo == "arco":
        control = puntos_de_control(mascara, [], distancia=0, margen=40)
        descripcion = "de la máscara a >= 40 px de los bordes"
    else:
        control = puntos_de_control(mascara, nucleos + deltas, distancia=40, margen=20)
        descripcion = "de la máscara a >= 40 px de toda singularidad y a >= 20 px de los bordes"
    problemas = [] if len(control) >= 5 else [f"solo hay {len(control)} puntos {descripcion}; se necesitan 5"]
    casos = (
        [("núcleo", p, 0.5) for p in nucleos]
        + [("delta", p, -0.5) for p in deltas]
        + [("punto de control", p, 0.0) for p in control]
    )
    indices = indices_poincare(huella.orientacion, [p for _, p, _ in casos])
    problemas += [
        f"{clase} {fmt(punto)}: índice {indice:+.3f} (esperado {esperado:+.1f} ± 0,05)"
        for (clase, punto, esperado), indice in zip(casos, indices)
        if abs(indice - esperado) > 0.05
    ]
    assert not problemas, f"generar_base({tipo!r}, {semilla}):\n" + resumen(problemas)


@pytest.mark.parametrize(("tipo", "semilla"), CASOS_CON_SINGULARIDADES, ids=IDS_CON_SINGULARIDADES)
def test_base_orientacion_es_campo_orientacion_de_sus_singularidades(gen, cfg, bases, tipo, semilla):
    huella = bases(tipo, semilla)
    campo = atributo(gen, "campo_orientacion")(huella.singularidades, tamano(cfg), 0.0)
    theta = np.asarray(huella.orientacion, dtype=np.float64)
    campo = np.asarray(campo, dtype=np.float64)
    assert campo.shape == theta.shape, f"campo_orientacion da forma {campo.shape} y orientacion {theta.shape}."
    error_cos = float(np.abs(np.cos(2 * theta) - np.cos(2 * campo)).max())
    error_sen = float(np.abs(np.sin(2 * theta) - np.sin(2 * campo)).max())
    assert max(error_cos, error_sen) <= 1e-9, (
        f"generar_base({tipo!r}, {semilla}).orientacion no coincide con campo_orientacion(singularidades, "
        f"tamano_px, 0.0): diferencia máxima {error_cos:.3g} en cos 2θ y {error_sen:.3g} en sen 2θ (tolerancia 1e-9)."
    )


@pytest.mark.parametrize(("tipo", "semilla"), CASOS_BASE, ids=IDS_BASE)
def test_base_crestas_oscuras_sobre_fondo_claro(bases, tipo, semilla):
    huella = bases(tipo, semilla)
    mascara = np.asarray(huella.mascara, dtype=bool)
    imagen = np.asarray(huella.imagen, dtype=np.float64)
    assert mascara.any(), f"generar_base({tipo!r}, {semilla}): la máscara está vacía."
    afuera = ndimage.distance_transform_edt(~mascara) > 10
    assert afuera.any(), f"generar_base({tipo!r}, {semilla}): no hay píxeles a más de 10 px de la máscara."
    gris_dentro, gris_afuera = float(imagen[mascara].mean()), float(imagen[afuera].mean())
    problemas = []
    if gris_afuera - gris_dentro < 50:
        problemas.append(
            f"gris promedio dentro {gris_dentro:.1f} y afuera {gris_afuera:.1f}: diferencia "
            f"{gris_afuera - gris_dentro:.1f} (mínimo 50)"
        )
    if gris_afuera < 230:
        problemas.append(f"gris promedio afuera {gris_afuera:.1f} (mínimo 230)")
    assert not problemas, f"generar_base({tipo!r}, {semilla}):\n" + resumen(problemas)


@pytest.mark.parametrize(("tipo", "semilla"), CASOS_BASE, ids=IDS_BASE)
def test_base_periodo_en_su_rango_y_periodo_medido_entre_8_y_10(cfg, bases, tipo, semilla):
    huella = bases(tipo, semilla)
    minimo, maximo = obtener(cfg, "huellas.generador.periodo_base_px")
    medido = periodo_medido(np.asarray(huella.imagen))
    problemas = []
    if not minimo <= huella.periodo_px <= maximo:
        problemas.append(f"periodo_px = {huella.periodo_px} fuera de periodo_base_px = [{minimo}, {maximo}]")
    if not 8 <= medido <= 10:
        problemas.append(f"período medido {medido:.3f} px (entre 8 y 10)")
    assert not problemas, f"generar_base({tipo!r}, {semilla}):\n" + resumen(problemas)


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_base_es_determinista(gen, cfg, bases, tipo):
    semilla = SEMILLAS_BASE[tipo][0]
    primera = bases(tipo, semilla)
    segunda = atributo(gen, "generar_base")(tipo, semilla, copy.deepcopy(cfg))
    distintos = [c for c in ("imagen", "mascara", "orientacion") if not np.array_equal(getattr(primera, c), getattr(segunda, c))]
    if primera.singularidades != segunda.singularidades:
        distintos.append(f"singularidades ({primera.singularidades} y {segunda.singularidades})")
    if primera.periodo_px != segunda.periodo_px:
        distintos.append(f"periodo_px ({primera.periodo_px} y {segunda.periodo_px})")
    assert not distintos, f"Dos llamadas a generar_base({tipo!r}, {semilla}, cfg) difieren en: {', '.join(distintos)}."


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_bases_del_mismo_tipo_con_semillas_distintas_no_se_parecen(bases, tipo):
    problemas = []
    for semilla_a, semilla_b in itertools.combinations(SEMILLAS_BASE[tipo], 2):
        a, b = bases(tipo, semilla_a), bases(tipo, semilla_b)
        r = correlacion(a.imagen, a.mascara, b.imagen, b.mascara)
        if not abs(r) < 0.35:
            problemas.append(f"semillas {semilla_a} y {semilla_b}: correlación {r:+.3f} (|r| < 0,35)")
    assert not problemas, f"Bases de tipo {tipo}:\n" + resumen(problemas)


@pytest.mark.parametrize(("tipo_a", "tipo_b"), list(itertools.combinations(TIPOS_CONTRATO, 2)))
def test_bases_de_tipos_distintos_no_se_parecen(bases, tipo_a, tipo_b):
    problemas = []
    for semilla_a in SEMILLAS_BASE[tipo_a]:
        for semilla_b in SEMILLAS_BASE[tipo_b]:
            a, b = bases(tipo_a, semilla_a), bases(tipo_b, semilla_b)
            r = correlacion(a.imagen, a.mascara, b.imagen, b.mascara)
            if not abs(r) < 0.35:
                problemas.append(
                    f"{tipo_a} {semilla_a} y {tipo_b} {semilla_b}: correlación {r:+.3f} (|r| < 0,35)"
                )
    assert not problemas, "\n" + resumen(problemas)


@pytest.mark.parametrize("tipo", ["lazo", "Presilla", "arco tienda", ""])
def test_generar_base_con_tipo_invalido_lanza_value_error(gen, cfg, tipo):
    lanza(ValueError, atributo(gen, "generar_base"), tipo, 101, copy.deepcopy(cfg), que=f"generar_base({tipo!r}, ...)")


# ---------------------------------------------------------------------------
# 5.4 Parámetros
# ---------------------------------------------------------------------------

N_SORTEOS = 1000
SEMILLAS_SORTEO = [7, 2026]


@pytest.fixture(scope="module")
def configs_sorteo(cfg) -> dict:
    return {"real": cfg, "rangos_chicos": configuracion_con_rangos_chicos(cfg)}


def sorteos(var, configuracion: dict, semilla: int) -> list:
    sortear = atributo(var, "sortear_parametros")
    rng = np.random.default_rng(semilla)
    copia = copy.deepcopy(configuracion)
    return [sortear(rng, copia) for _ in range(N_SORTEOS)]


@pytest.mark.parametrize("semilla", SEMILLAS_SORTEO)
@pytest.mark.parametrize("nombre_cfg", ["real", "rangos_chicos"])
def test_sortear_parametros_quedan_dentro_de_sus_rangos(var, configs_sorteo, nombre_cfg, semilla):
    configuracion = configs_sorteo[nombre_cfg]
    clase = atributo(var, "ParametrosVariante")
    problemas = []
    for n, params in enumerate(sorteos(var, configuracion, semilla), start=1):
        if not isinstance(params, clase):
            problemas.append(f"sorteo {n}: devolvió {tipo_de(params)}, no un ParametrosVariante")
            continue
        problemas.extend(problemas_de_rango(campos_de(params, f"sorteo {n}"), configuracion, que=f"sorteo {n}"))
    assert not problemas, f"{N_SORTEOS} sorteos con cfg {nombre_cfg} y semilla {semilla}:\n" + resumen(problemas)


@pytest.mark.parametrize("semilla", SEMILLAS_SORTEO)
@pytest.mark.parametrize("nombre_cfg", ["real", "rangos_chicos"])
def test_sortear_parametros_corta_con_la_probabilidad_del_cfg(var, configs_sorteo, nombre_cfg, semilla):
    configuracion = configs_sorteo[nombre_cfg]
    prob_corte = configuracion["huellas"]["variantes"]["prob_corte"]
    con_corte = sum(campos_de(p, "sorteo")["corte_fraccion"] > 0 for p in sorteos(var, configuracion, semilla))
    proporcion = con_corte / N_SORTEOS
    assert abs(proporcion - prob_corte) <= 0.05, (
        f"Con cfg {nombre_cfg} y semilla {semilla}: {con_corte} de {N_SORTEOS} sorteos con corte "
        f"(proporción {proporcion:.3f}); se esperaba prob_corte = {prob_corte} ± 0,05."
    )


@pytest.mark.parametrize("semilla", SEMILLAS_SORTEO)
def test_sortear_parametros_misma_semilla_mismos_parametros(var, cfg, semilla):
    primeros, segundos = sorteos(var, cfg, semilla), sorteos(var, cfg, semilla)
    distintos = [n for n, (a, b) in enumerate(zip(primeros, segundos), start=1) if a != b]
    assert not distintos, (
        f"Dos rngs con semilla {semilla} dieron parámetros distintos en {len(distintos)} de {N_SORTEOS} "
        f"sorteos (el primero, el {distintos[0]}: {primeros[distintos[0] - 1]} y {segundos[distintos[0] - 1]})."
    )


# ---------------------------------------------------------------------------
# 5.5 Variantes
# ---------------------------------------------------------------------------


def base_del_tipo(bases, tipo: str):
    """La base de la primera semilla del tipo (identidad de config.yaml, o el primer arco)."""
    return bases(tipo, SEMILLAS_BASE[tipo][0])


def test_sin_cambios_es_el_del_contrato(var):
    sin_cambios = atributo(var, "SIN_CAMBIOS")
    esperado = parametros(var)
    assert sin_cambios == esperado, f"SIN_CAMBIOS es {sin_cambios!r}; el contrato dice {esperado!r}."


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_variante_tiene_tipos_y_tamanos_y_no_modifica_la_base(var, cfg, bases, tipo):
    base = base_del_tipo(bases, tipo)
    antes = copy.deepcopy(base)
    corte_min = float(cfg["huellas"]["variantes"]["corte_periferico"][0])
    datos = datos_al_azar(cfg, np.random.default_rng(3), corte_fraccion=corte_min, corte_angulo_grados=200.0)
    variante = aplicar(var, base, parametros(var, **datos), semilla=4)
    forma = antes.imagen.shape
    problemas = []
    for campo, dtype in (("imagen", np.uint8), ("mascara", np.bool_), ("mapa_x", np.float32), ("mapa_y", np.float32)):
        valor = getattr(variante, campo, None)
        if not isinstance(valor, np.ndarray):
            problemas.append(f"{campo} no es un np.ndarray ({tipo_de(valor)})")
            continue
        if valor.dtype != dtype:
            problemas.append(f"{campo} es {valor.dtype}; se esperaba {np.dtype(dtype)}")
        if valor.shape != forma:
            problemas.append(f"{campo} tiene forma {valor.shape}; la base, {forma}")
    if not isinstance(getattr(variante, "parametros", None), atributo(var, "ParametrosVariante")):
        problemas.append(f"parametros es {tipo_de(getattr(variante, 'parametros', None))}, no un ParametrosVariante")
    for campo in ("imagen", "mascara", "orientacion"):
        if not np.array_equal(getattr(base, campo), getattr(antes, campo)):
            problemas.append(f"aplicar_variante modificó base.{campo}")
    for campo in ("singularidades", "periodo_px", "tipo", "semilla"):
        if getattr(base, campo) != getattr(antes, campo):
            problemas.append(f"aplicar_variante modificó base.{campo}")
    assert not problemas, f"Variante de la base {tipo}:\n" + resumen(problemas)


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_variante_con_sin_cambios_reproduce_la_base(var, bases, tipo):
    base = base_del_tipo(bases, tipo)
    variante = aplicar(var, base, atributo(var, "SIN_CAMBIOS"), semilla=5)
    problemas = []
    diferencia = int(np.abs(variante.imagen.astype(np.int16) - base.imagen.astype(np.int16)).max())
    if diferencia > 1:
        problemas.append(f"la imagen difiere de la base hasta en {diferencia} niveles de gris (máximo 1)")
    for clase, de_base, de_variante in (
        ("núcleo", base.singularidades.nucleos, variante.singularidades.nucleos),
        ("delta", base.singularidades.deltas, variante.singularidades.deltas),
    ):
        esperados = como_puntos(de_base, f"base: {clase}s")
        obtenidos = como_puntos(de_variante, f"variante: {clase}s")
        if len(obtenidos) != len(esperados):
            problemas.append(f"la variante tiene {len(obtenidos)} {clase}s y la base {len(esperados)}")
        for punto in esperados:
            distancia = distancia_al_mas_cercano(punto, obtenidos)
            if distancia > 0.5:
                problemas.append(f"{clase} de la base {fmt(punto)}: el más cercano de la variante está a {distancia:.3f} px (máximo 0,5)")
    coincidencia = float(np.mean(np.asarray(variante.mascara, dtype=bool) == np.asarray(base.mascara, dtype=bool)))
    if coincidencia < 0.995:
        problemas.append(f"la máscara coincide en el {100 * coincidencia:.2f} % de los píxeles (mínimo 99,5 %)")
    assert not problemas, f"SIN_CAMBIOS sobre la base {tipo}:\n" + resumen(problemas)


@pytest.mark.parametrize("tipo", TIPOS_AUTORIZADOS)
def test_variante_afin_ubica_las_singularidades_segun_la_formula(var, cfg, bases, tipo):
    base = base_del_tipo(bases, tipo)
    alto, ancho = base.imagen.shape
    v = cfg["huellas"]["variantes"]
    rot, esc, tras = float(v["rotacion_max_grados"]), float(v["escala_max"]), float(v["traslacion_max_px"])
    juegos = [  # las cuatro esquinas del rango y 20 sorteados
        (rot, 1 + esc, tras, tras),
        (-rot, 1 - esc, -tras, -tras),
        (rot, 1 - esc, -tras, tras),
        (-rot, 1 + esc, tras, -tras),
    ]
    rng = np.random.default_rng(2026)
    juegos += [
        (rng.uniform(-rot, rot), rng.uniform(1 - esc, 1 + esc), rng.uniform(-tras, tras), rng.uniform(-tras, tras))
        for _ in range(20)
    ]
    nucleos_base = como_puntos(base.singularidades.nucleos, "base: núcleos")
    deltas_base = como_puntos(base.singularidades.deltas, "base: deltas")
    problemas, deltas_verificados = [], 0
    for n, (r, s, tx, ty) in enumerate(juegos, start=1):
        datos = {**SIN_CAMBIOS_CONTRATO, "rot_grados": r, "escala": s, "tx_px": tx, "ty_px": ty}
        variante = aplicar(var, base, parametros(var, **datos), semilla=n)
        juego = f"juego {n} (rot {r:+.2f}°, escala {s:.4f}, tx {tx:+.2f}, ty {ty:+.2f})"
        nucleos = como_puntos(variante.singularidades.nucleos, "variante: núcleos")
        deltas = como_puntos(variante.singularidades.deltas, "variante: deltas")
        if len(nucleos) != len(nucleos_base):
            problemas.append(f"{juego}: la variante tiene {len(nucleos)} núcleos y la base {len(nucleos_base)}")
        for punto in nucleos_base:
            esperado = transformacion_afin(punto, datos, alto, ancho)
            distancia = distancia_al_mas_cercano(esperado, nucleos)
            if distancia > 1:
                problemas.append(f"{juego}: núcleo esperado en {fmt(esperado)}, el más cercano está a {distancia:.3f} px (máximo 1)")
        for punto in deltas_base:
            esperado = transformacion_afin(punto, datos, alto, ancho)
            if distancia_al_borde(esperado, alto, ancho) < 10:
                continue
            deltas_verificados += 1
            distancia = distancia_al_mas_cercano(esperado, deltas)
            if distancia > 1:
                problemas.append(f"{juego}: delta esperado en {fmt(esperado)}, el más cercano está a {distancia:.3f} px (máximo 1)")
    if deltas_base and deltas_verificados == 0:
        problemas.append("ningún delta quedó a >= 10 px de los bordes: no se verificó ninguno")
    assert not problemas, f"Solo rotación, escala y traslación sobre la base {tipo}:\n" + resumen(problemas)


@pytest.mark.parametrize("tipo", TIPOS_AUTORIZADOS)
def test_variante_mapas_devuelven_cada_singularidad_a_su_posicion_en_la_base(var, cfg, bases, tipo):
    base = base_del_tipo(bases, tipo)
    alto, ancho = base.imagen.shape
    v = cfg["huellas"]["variantes"]
    originales = {
        "núcleo": como_puntos(base.singularidades.nucleos, "base: núcleos"),
        "delta": como_puntos(base.singularidades.deltas, "base: deltas"),
    }
    problemas, verificadas = [], 0
    for n in range(8):
        rng = np.random.default_rng(100 + n)
        datos = datos_al_azar(
            cfg,
            rng,
            elastica_px=v["elastica_max_px"],
            corte_fraccion=v["corte_periferico"][0],
            corte_angulo_grados=rng.uniform(0, 360),
        )
        variante = aplicar(var, base, parametros(var, **datos), semilla=200 + n)
        obtenidas = {
            "núcleo": como_puntos(variante.singularidades.nucleos, "variante: núcleos"),
            "delta": como_puntos(variante.singularidades.deltas, "variante: deltas"),
        }
        for clase, puntos in obtenidas.items():
            for punto in puntos:
                if distancia_al_borde(punto, alto, ancho) < 10:
                    continue
                verificadas += 1
                vuelta = (evaluar_bilineal(variante.mapa_x, punto), evaluar_bilineal(variante.mapa_y, punto))
                error = distancia_al_mas_cercano(vuelta, originales[clase])
                if error > 0.5:
                    problemas.append(
                        f"variante {n + 1} (elástica {datos['elastica_px']:.2f} px): {clase} en {fmt(punto)} vuelve a "
                        f"{fmt(vuelta)}, a {error:.3f} px de su posición en la base (máximo 0,5)"
                    )
    assert verificadas > 0, "Ninguna singularidad quedó a >= 10 px de los bordes: no se verificó nada."
    assert not problemas, f"Mapas de las variantes de la base {tipo}:\n" + resumen(problemas)


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_variante_punto_de_referencia(var, cfg, bases, tipo):
    base = base_del_tipo(bases, tipo)
    datos = datos_al_azar(cfg, np.random.default_rng(7), elastica_px=cfg["huellas"]["variantes"]["elastica_max_px"])
    variante = aplicar(var, base, parametros(var, **datos), semilla=8)
    nucleos = como_puntos(variante.singularidades.nucleos, "variante: núcleos")
    referencia = variante.punto_referencia
    if tipo == "arco":
        assert referencia is None, f"En el arco (sin núcleos) punto_referencia tiene que ser None; es {referencia!r}."
        return
    assert len(nucleos) == NUCLEOS_POR_TIPO[tipo], (
        f"La variante del {tipo} tiene {len(nucleos)} núcleos; se esperaban {NUCLEOS_POR_TIPO[tipo]}."
    )
    esperado = nucleos[0] if len(nucleos) == 1 else ((nucleos[0][0] + nucleos[1][0]) / 2, (nucleos[0][1] + nucleos[1][1]) / 2)
    que = "el núcleo" if len(nucleos) == 1 else "el punto medio de los núcleos"
    assert referencia is not None, f"punto_referencia es None; se esperaba {que} {fmt(esperado)}."
    punto = como_puntos([referencia], "punto_referencia")[0]
    assert math.dist(punto, esperado) <= 1e-6, (
        f"punto_referencia = {fmt(punto)}; se esperaba {que} {fmt(esperado)}."
    )


ANGULOS_DE_CORTE = (0.0, 45.0, 90.0, 180.0, 270.0)
# Además de los extremos de corte_periferico: con 0,15 a 0,20 el borde del corte queda
# lejos de núcleos centrales; con 0,40 pasa cerca y obliga a girar de a 30°.
FRACCION_DE_CORTE_GRANDE = 0.40


@dataclass
class CasoDeCorte:
    fraccion: float
    angulo: float
    mascara: np.ndarray
    parametros: object
    nucleos: list


@pytest.fixture(scope="module")
def cortes(var, cfg, bases):
    """Función tipo -> (máscara sin corte, casos con corte) con los mismos parámetros y semilla."""
    cache: dict = {}

    def obtener(tipo: str):
        if tipo not in cache:
            base = base_del_tipo(bases, tipo)
            corte_min, corte_max = (float(f) for f in cfg["huellas"]["variantes"]["corte_periferico"])
            datos = datos_al_azar(cfg, np.random.default_rng(41))
            sin_corte = aplicar(var, base, parametros(var, **datos), semilla=77)
            casos = []
            for fraccion in (corte_min, corte_max, FRACCION_DE_CORTE_GRANDE):
                for angulo in ANGULOS_DE_CORTE:
                    con_corte = aplicar(
                        var,
                        base,
                        parametros(var, **{**datos, "corte_fraccion": fraccion, "corte_angulo_grados": angulo}),
                        semilla=77,
                    )
                    casos.append(
                        CasoDeCorte(
                            fraccion,
                            angulo,
                            np.asarray(con_corte.mascara, dtype=bool),
                            con_corte.parametros,
                            como_puntos(con_corte.singularidades.nucleos, "variante: núcleos"),
                        )
                    )
            cache[tipo] = (np.asarray(sin_corte.mascara, dtype=bool), casos)
        return cache[tipo]

    return obtener


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_variante_corte_quita_la_fraccion_pedida_sin_tocar_los_nucleos(cortes, tipo):
    mascara_sin_corte, casos = cortes(tipo)
    alto, ancho = mascara_sin_corte.shape
    area = int(mascara_sin_corte.sum())
    assert area > 0, "La máscara de la variante sin corte está vacía."
    problemas, cortados = [], 0
    for caso in casos:
        etiqueta = f"f = {caso.fraccion}, ángulo pedido {caso.angulo:g}°"
        usados = campos_de(caso.parametros, etiqueta)
        if usados["corte_fraccion"] == 0:
            if NUCLEOS_POR_TIPO[tipo] == 0:
                problemas.append(f"{etiqueta}: no cortó, y sin núcleos cualquier ángulo sirve")
            if not np.array_equal(caso.mascara, mascara_sin_corte):
                problemas.append(f"{etiqueta}: parametros dice que no cortó (corte_fraccion = 0) pero la máscara cambió")
            if usados["corte_angulo_grados"] != 0:
                problemas.append(f"{etiqueta}: no cortó y corte_angulo_grados = {usados['corte_angulo_grados']} (tiene que valer 0)")
            continue
        cortados += 1
        if abs(usados["corte_fraccion"] - caso.fraccion) > 1e-12:
            problemas.append(f"{etiqueta}: parametros.corte_fraccion = {usados['corte_fraccion']} y se pidió {caso.fraccion}")
        agregados = int((caso.mascara & ~mascara_sin_corte).sum())
        if agregados:
            problemas.append(f"{etiqueta}: la máscara con corte tiene {agregados} píxeles que no estaban sin corte")
        perdida = 1 - int((caso.mascara & mascara_sin_corte).sum()) / area
        if abs(perdida - caso.fraccion) > 0.02:
            problemas.append(f"{etiqueta}: se perdió {perdida:.4f} de la máscara (esperado {caso.fraccion} ± 0,02)")
        for x, y in caso.nucleos:
            circulo = disco(x, y, 20, alto, ancho)
            sale_de_la_imagen = x - 20 < 0 or y - 20 < 0 or x + 20 > ancho - 1 or y + 20 > alto - 1
            faltan = int((circulo & ~caso.mascara).sum())
            if faltan or sale_de_la_imagen:
                ya_faltaban = int((circulo & ~mascara_sin_corte).sum())
                detalle = "; el disco sale de la imagen" if sale_de_la_imagen else ""
                problemas.append(
                    f"{etiqueta}: el disco de 20 px del núcleo {fmt((x, y))} no queda entero en la máscara con "
                    f"corte: faltan {faltan} píxeles ({ya_faltaban} ya faltaban sin corte){detalle}"
                )
    if NUCLEOS_POR_TIPO[tipo] and cortados == 0:
        problemas.append(f"no cortó en ninguno de los {len(casos)} casos: siempre quedó corte_fraccion = 0")
    assert not problemas, f"Corte sobre la base {tipo}:\n" + resumen(problemas)


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_variante_corte_quita_un_semiplano_del_lado_del_angulo_guardado(cortes, tipo):
    """Criterio sumado: el lado del corte y el giro de a 30°."""
    mascara_sin_corte, casos = cortes(tipo)
    alto, ancho = mascara_sin_corte.shape
    filas, columnas = np.mgrid[0:alto, 0:ancho]
    problemas = []
    for caso in casos:
        etiqueta = f"f = {caso.fraccion}, ángulo pedido {caso.angulo:g}°"
        usados = campos_de(caso.parametros, etiqueta)
        if usados["corte_fraccion"] == 0:
            continue  # lo verifica el test de la fracción
        angulo = usados["corte_angulo_grados"]
        if not 0 <= angulo < 360:
            problemas.append(f"{etiqueta}: el ángulo guardado {angulo} está fuera de [0, 360)")
        resto = (angulo - caso.angulo) % 30
        if min(resto, 30 - resto) > 1e-6:
            problemas.append(f"{etiqueta}: el ángulo guardado {angulo} no es el pedido más un múltiplo de 30°")
        quitados = mascara_sin_corte & ~caso.mascara
        if not quitados.any() or not caso.mascara.any():
            continue  # lo verifica el test de la fracción
        a = math.radians(angulo)
        proyeccion = columnas * math.cos(a) + filas * math.sin(a)
        minimo_quitado, maximo_quedado = float(proyeccion[quitados].min()), float(proyeccion[caso.mascara].max())
        if minimo_quitado < maximo_quedado - 1:
            problemas.append(
                f"{etiqueta}: lo quitado no es un semiplano del lado (cos a, sen a) del ángulo guardado "
                f"{angulo:g}°: hay píxeles quitados a {maximo_quedado - minimo_quitado:.1f} px detrás del borde del corte"
            )
    assert not problemas, f"Lado del corte sobre la base {tipo}:\n" + resumen(problemas)


@pytest.mark.parametrize(
    ("angulo", "lado"),
    [(0.0, "derecha"), (90.0, "abajo"), (180.0, "izquierda"), (270.0, "arriba")],
    ids=["0-derecha", "90-abajo", "180-izquierda", "270-arriba"],
)
def test_variante_corte_quita_el_lado_que_indica_el_angulo(var, cfg, bases, angulo, lado):
    """Criterio sumado: 0° quita la derecha y 90° la parte de abajo (y apunta hacia abajo)."""
    base = base_del_tipo(bases, "arco")  # sin núcleos: el ángulo pedido siempre sirve
    fraccion = float(cfg["huellas"]["variantes"]["corte_periferico"][1])
    sin_corte = aplicar(var, base, parametros(var), semilla=13)
    con_corte = aplicar(var, base, parametros(var, corte_fraccion=fraccion, corte_angulo_grados=angulo), semilla=13)
    usado = campos_de(con_corte.parametros, "variante")["corte_angulo_grados"]
    assert usado == angulo, f"Sin núcleos no hay que girar: se pidió {angulo:g}° y se guardó {usado}°."
    mascara_con = np.asarray(con_corte.mascara, dtype=bool)
    quitados = np.asarray(sin_corte.mascara, dtype=bool) & ~mascara_con
    assert quitados.any(), f"Con corte_fraccion = {fraccion} y ángulo {angulo:g}° no se quitó nada de la máscara."
    filas_q, columnas_q = np.nonzero(quitados)
    filas_k, columnas_k = np.nonzero(mascara_con)
    coordenada = {
        "derecha": (columnas_q, columnas_k),
        "izquierda": (-columnas_q, -columnas_k),
        "abajo": (filas_q, filas_k),
        "arriba": (-filas_q, -filas_k),
    }[lado]
    quitado, quedado = coordenada
    assert quitado.min() >= quedado.max() - 1, (
        f"Con ángulo {angulo:g}° se tenía que quitar la parte de {lado}: lo quitado va de x {columnas_q.min()} a "
        f"{columnas_q.max()} e y {filas_q.min()} a {filas_q.max()}; lo que queda, de x {columnas_k.min()} a "
        f"{columnas_k.max()} e y {filas_k.min()} a {filas_k.max()}."
    )


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_variante_presion_cambia_la_fraccion_de_pixeles_oscuros(var, cfg, bases, tipo):
    base = base_del_tipo(bases, tipo)
    mascara = np.asarray(base.mascara, dtype=bool)
    presion_max = float(cfg["huellas"]["variantes"]["presion_max"])
    niveles = [-presion_max, -presion_max / 2, 0.0, presion_max / 2, presion_max]
    fracciones = [
        float(np.mean(aplicar(var, base, parametros(var, presion=p), semilla=3).imagen[mascara] < 128))
        for p in niveles
    ]
    detalle = ", ".join(f"presión {p:+.3f}: {f:.4f}" for p, f in zip(niveles, fracciones))
    problemas = []
    if not all(a < b for a, b in zip(fracciones, fracciones[1:])):
        problemas.append("la fracción de píxeles de la máscara por debajo de 128 no crece con la presión")
    if fracciones[4] - fracciones[2] < 0.05:
        problemas.append(f"con +presion_max sube {fracciones[4] - fracciones[2]:.4f} respecto de 0 (mínimo 0,05)")
    if fracciones[2] - fracciones[0] < 0.05:
        problemas.append(f"con -presion_max baja {fracciones[2] - fracciones[0]:.4f} respecto de 0 (mínimo 0,05)")
    assert not problemas, f"Presión sobre la base {tipo} ({detalle}):\n" + resumen(problemas)


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_variante_ruido_gaussiano_de_sigma_10(var, bases, tipo):
    base = base_del_tipo(bases, tipo)
    mascara = np.asarray(base.mascara, dtype=bool)
    variante = aplicar(var, base, parametros(var, ruido_sigma=10.0), semilla=17)
    diferencia = variante.imagen.astype(np.float64)[mascara] - base.imagen.astype(np.float64)[mascara]
    desvio = float(diferencia.std())
    assert 7 <= desvio <= 13, (
        f"Con ruido_sigma = 10, el desvío de la diferencia con la base {tipo} dentro de la máscara es "
        f"{desvio:.2f} (entre 7 y 13)."
    )


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_variante_sal_y_pimienta_de_0_02(var, bases, tipo):
    base = base_del_tipo(bases, tipo)
    variante = aplicar(var, base, parametros(var, sal_pimienta=0.02), semilla=23)
    no_saturados = (base.imagen != 0) & (base.imagen != 255)
    assert no_saturados.any(), f"La base {tipo} tiene todos los píxeles en 0 o 255."
    nuevos = no_saturados & ((variante.imagen == 0) | (variante.imagen == 255))
    fraccion = float(nuevos.sum() / no_saturados.sum())
    assert 0.01 <= fraccion <= 0.03, (
        f"Con sal_pimienta = 0,02, pasó a 0 o 255 el {fraccion:.4f} de los píxeles que en la base {tipo} "
        f"no estaban saturados ({int(nuevos.sum())} de {int(no_saturados.sum())}); se esperaba entre 0,01 y 0,03."
    )


@pytest.mark.parametrize("tipo", TIPOS_CONTRATO)
def test_variante_es_determinista(var, cfg, bases, tipo):
    base = base_del_tipo(bases, tipo)
    v = cfg["huellas"]["variantes"]
    datos = datos_al_azar(
        cfg,
        np.random.default_rng(11),
        elastica_px=v["elastica_max_px"],
        corte_fraccion=v["corte_periferico"][1],
        corte_angulo_grados=123.0,
    )
    params = parametros(var, **datos)
    primera, segunda = aplicar(var, base, params, semilla=99), aplicar(var, base, params, semilla=99)
    distintos = [c for c in ("imagen", "mascara", "mapa_x", "mapa_y") if not np.array_equal(getattr(primera, c), getattr(segunda, c))]
    for campo in ("singularidades", "punto_referencia", "parametros"):
        if getattr(primera, campo) != getattr(segunda, campo):
            distintos.append(campo)
    assert not distintos, (
        f"La misma base {tipo}, los mismos parámetros y un rng con la misma semilla dieron variantes que "
        f"difieren en: {', '.join(distintos)}."
    )


# ---------------------------------------------------------------------------
# 5.6 Plan del dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def configs_plan(cfg, cfg_reducido) -> dict:
    return {"real": cfg, "reducida": cfg_reducido}


@pytest.fixture(scope="module")
def plan_de(ds, configs_plan):
    """Función nombre -> plan_dataset(cfg) para el cfg "real" o el "reducida"."""
    plan_dataset = atributo(ds, "plan_dataset")
    cache: dict = {}

    def obtener(nombre: str) -> list[dict]:
        if nombre not in cache:
            cache[nombre] = entradas(plan_dataset(copy.deepcopy(configs_plan[nombre])), f"plan_dataset (cfg {nombre})")
        return cache[nombre]

    return obtener


@pytest.mark.parametrize("nombre", CONFIGS_PLAN)
def test_plan_entradas_tienen_las_claves_del_plan_con_sus_tipos(plan_de, nombre):
    problemas = []
    for numero, entrada in enumerate(plan_de(nombre), start=1):
        for columna in COLUMNAS_PLAN:
            valor = entrada[columna]
            esperado_texto = columna in ("archivo", "particion", "tipo")
            if esperado_texto and not isinstance(valor, str):
                problemas.append(f"entrada {numero}: {columna} = {valor!r} tendría que ser str")
            if not esperado_texto and not es_entero(valor):
                problemas.append(f"entrada {numero}: {columna} = {valor!r} tendría que ser int")
    assert not problemas, f"plan_dataset con cfg {nombre}:\n" + resumen(problemas)


@pytest.mark.parametrize("nombre", CONFIGS_PLAN)
def test_plan_cantidad_de_entradas_por_particion(plan_de, nombre):
    plan = plan_de(nombre)
    medido = {
        particion: (
            sum(1 for e in plan if e["particion"] == particion and e["intruso"] == 0),
            sum(1 for e in plan if e["particion"] == particion and e["intruso"] != 0),
        )
        for particion in ("train", "val", "test")
    }
    otras = sorted({e["particion"] for e in plan} - {"train", "val", "test"})
    esperado = CONTEOS_PLAN[nombre]
    total_esperado = sum(a + i for a, i in esperado.values())
    assert not otras, f"Hay entradas en particiones desconocidas: {otras}."
    assert len(plan) == total_esperado and medido == esperado, (
        f"Con cfg {nombre} se esperaban {total_esperado} entradas, (autorizadas, intrusos) = {esperado}; "
        f"hay {len(plan)} y {medido}."
    )


@pytest.mark.parametrize("nombre", CONFIGS_PLAN)
def test_plan_autorizadas_por_variante_particion_y_reservadas(plan_de, configs_plan, nombre):
    plan, huellas = plan_de(nombre), configs_plan[nombre]["huellas"]
    total = huellas["variantes_por_identidad"]
    train, val = huellas["particion"]["train"], huellas["particion"]["val"]
    reservadas = huellas["reservadas_demo"]
    problemas = []
    for identidad in huellas["identidades"]:
        k = identidad["id"]
        propias = [e for e in plan if e["identidad"] == k]
        variantes = sorted(e["variante"] for e in propias)
        if variantes != list(range(1, total + 1)):
            problemas.append(f"identidad {k}: variantes {variantes}; se esperaban 1 a {total}, una vez cada una")
        for e in propias:
            v = e["variante"]
            particion = "train" if v <= train else ("val" if v <= train + val else "test")
            reservada = 1 if v > total - reservadas else 0
            if e["particion"] != particion:
                problemas.append(f"identidad {k}, variante {v}: partición {e['particion']!r} (esperada {particion!r})")
            if e["reservada_demo"] != reservada:
                problemas.append(f"identidad {k}, variante {v}: reservada_demo = {e['reservada_demo']} (esperado {reservada})")
            if e["intruso"] != 0:
                problemas.append(f"identidad {k}, variante {v}: intruso = {e['intruso']} (esperado 0)")
            if e["tipo"] != identidad["tipo"]:
                problemas.append(f"identidad {k}, variante {v}: tipo {e['tipo']!r} (esperado {identidad['tipo']!r})")
    ids = {identidad["id"] for identidad in huellas["identidades"]}
    for e in plan:
        if e["identidad"] not in ids and e["reservada_demo"] != 0:
            problemas.append(f"{e['archivo']}: reservada_demo = {e['reservada_demo']} y no es de una identidad autorizada")
        if e["identidad"] not in ids | {0}:
            problemas.append(f"{e['archivo']}: identidad = {e['identidad']}, que no está en config.yaml")
    assert not problemas, f"plan_dataset con cfg {nombre}:\n" + resumen(problemas)


@pytest.mark.parametrize("nombre", CONFIGS_PLAN)
def test_plan_intrusos_por_particion_variantes_y_tipos(plan_de, configs_plan, nombre):
    plan, intrusos = plan_de(nombre), configs_plan[nombre]["huellas"]["intrusos"]
    cantidad, por_intruso, en_val = intrusos["cantidad"], intrusos["variantes_por_intruso"], intrusos["val"]
    problemas = []
    for i in range(1, cantidad + 1):
        propias = [e for e in plan if e["intruso"] == i]
        variantes = sorted(e["variante"] for e in propias)
        if variantes != list(range(1, por_intruso + 1)):
            problemas.append(f"intruso {i}: variantes {variantes}; se esperaban 1 a {por_intruso}, una vez cada una")
        particion = "val" if i <= en_val else "test"
        tipo = TIPOS_CONTRATO[(i - 1) % len(TIPOS_CONTRATO)]
        for e in propias:
            if e["particion"] != particion:
                problemas.append(f"intruso {i}, variante {e['variante']}: partición {e['particion']!r} (esperada {particion!r})")
            if e["tipo"] != tipo:
                problemas.append(f"intruso {i}, variante {e['variante']}: tipo {e['tipo']!r} (esperado {tipo!r})")
            if e["identidad"] != 0:
                problemas.append(f"intruso {i}, variante {e['variante']}: identidad = {e['identidad']} (esperado 0)")
    sobrantes = sorted({e["intruso"] for e in plan} - set(range(cantidad + 1)))
    if sobrantes:
        problemas.append(f"hay intrusos que no existen en el cfg: {sobrantes}")
    assert not problemas, f"plan_dataset con cfg {nombre}:\n" + resumen(problemas)


@pytest.mark.parametrize("nombre", CONFIGS_PLAN)
def test_plan_semillas_de_config_y_de_intrusos(plan_de, configs_plan, nombre):
    plan, huellas = plan_de(nombre), configs_plan[nombre]["huellas"]
    semilla_de = {identidad["id"]: identidad["semilla"] for identidad in huellas["identidades"]}
    semilla_base = huellas["intrusos"]["semilla_base"]
    problemas = []
    for e in plan:
        esperada = semilla_de.get(e["identidad"]) if e["intruso"] == 0 else semilla_base + e["intruso"]
        if e["semilla"] != esperada:
            problemas.append(f"{e['archivo']}: semilla {e['semilla']} (esperada {esperada})")
    assert not problemas, f"plan_dataset con cfg {nombre}:\n" + resumen(problemas)


@pytest.mark.parametrize("nombre", CONFIGS_PLAN)
def test_plan_nombres_de_archivo_carpeta_y_sin_repetidos(plan_de, nombre):
    plan = plan_de(nombre)
    problemas = []
    for e in plan:
        if e["intruso"] == 0:
            esperado = f"{e['particion']}/id{e['identidad']}_v{e['variante']:03d}.png"
        else:
            esperado = f"{e['particion']}/intruso{e['intruso']:02d}_v{e['variante']}.png"
        if e["archivo"] != esperado:
            problemas.append(f"archivo {e['archivo']!r} (esperado {esperado!r})")
        if e["archivo"].split("/")[0] != e["particion"]:
            problemas.append(f"{e['archivo']!r} no está en la carpeta de su partición {e['particion']!r}")
    repetidos = sorted(a for a, n in Counter(e["archivo"] for e in plan).items() if n > 1)
    if repetidos:
        problemas.append(f"archivos repetidos: {repetidos[:10]}")
    assert not problemas, f"plan_dataset con cfg {nombre}:\n" + resumen(problemas)


# ---------------------------------------------------------------------------
# 5.7 Generación (con la configuración reducida, en carpetas temporales)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plan_reducido(plan_de) -> list[dict]:
    return plan_de("reducida")


@pytest.fixture(scope="module")
def dataset_a(ds, cfg_reducido, tmp_path_factory):
    """Dataset reducido generado en una carpeta vacía: (destino, filas devueltas)."""
    destino = tmp_path_factory.mktemp("dataset_a")
    filas = atributo(ds, "generar_dataset")(copy.deepcopy(cfg_reducido), destino)
    return destino, filas


@pytest.fixture(scope="module")
def dataset_b(ds, cfg_reducido, plan_reducido, tmp_path_factory):
    """Segunda generación, en otra carpeta que ya tenía datos viejos, con sobrescribir=True.

    Datos viejos: un PNG que no está en el plan, un PNG del plan con otro contenido y un
    metadatos.csv viejo. Archivos ajenos (no se tienen que borrar): notas.txt y train/leeme.txt.
    """
    destino = tmp_path_factory.mktemp("dataset_b")
    pisado = plan_reducido[0]["archivo"]
    viejos = {"train/sobrante.png": png_chico(), pisado: png_chico(), "metadatos.csv": b"datos viejos\n"}
    ajenos = {"notas.txt": "no borrar\n".encode("utf-8"), "train/leeme.txt": "tampoco\n".encode("utf-8")}
    for relativa, contenido in {**viejos, **ajenos}.items():
        (destino / relativa).parent.mkdir(parents=True, exist_ok=True)
        (destino / relativa).write_bytes(contenido)
    filas = atributo(ds, "generar_dataset")(copy.deepcopy(cfg_reducido), str(destino), sobrescribir=True)
    return destino, filas, pisado, ajenos


def test_columnas_es_la_tupla_del_contrato(ds):
    columnas = atributo(ds, "COLUMNAS")
    assert columnas == COLUMNAS_CONTRATO and isinstance(columnas, tuple), (
        f"COLUMNAS tiene que ser la tupla del contrato {COLUMNAS_CONTRATO}; es {columnas!r}."
    )


def test_generar_dataset_crea_exactamente_los_png_del_plan_y_metadatos(dataset_a, plan_reducido):
    destino, _ = dataset_a
    esperados = {e["archivo"] for e in plan_reducido} | {"metadatos.csv"}
    creados = {p.relative_to(destino).as_posix() for p in destino.rglob("*") if p.is_file()}
    faltan, sobran = sorted(esperados - creados), sorted(creados - esperados)
    assert not faltan and not sobran, (
        f"En {destino} faltan {len(faltan)} archivos {faltan[:10]} y sobran {len(sobran)} {sobran[:10]}."
    )


def test_generar_dataset_png_son_uint8_2d_de_tamano_px(cfg_reducido, dataset_a, plan_reducido):
    destino, _ = dataset_a
    forma = tamano(cfg_reducido)
    problemas = []
    for e in plan_reducido:
        imagen = leer_png(destino / e["archivo"])
        if imagen is None:
            problemas.append(f"{e['archivo']}: no existe o no se puede leer como PNG")
        elif imagen.dtype != np.uint8 or imagen.shape != forma:
            problemas.append(f"{e['archivo']}: {imagen.dtype} de forma {imagen.shape} (se esperaba uint8 {forma})")
    assert not problemas, "\n" + resumen(problemas)


def test_metadatos_tiene_encabezado_y_una_fila_por_png_en_el_orden_del_plan(dataset_a, plan_reducido):
    destino, _ = dataset_a
    encabezado, filas = leer_metadatos(destino)
    assert encabezado == list(COLUMNAS_CONTRATO), (
        f"El encabezado de metadatos.csv tiene que ser COLUMNAS:\n  esperado {list(COLUMNAS_CONTRATO)}\n  hay      {encabezado}"
    )
    assert len(filas) == len(plan_reducido), (
        f"metadatos.csv tiene {len(filas)} filas y el plan {len(plan_reducido)} entradas."
    )
    problemas = []
    for linea, (fila, entrada) in enumerate(zip(filas, plan_reducido), start=2):
        convertida, _ = convertir_fila(fila, linea)
        for columna in COLUMNAS_PLAN:
            if convertida[columna] != entrada[columna]:
                problemas.append(f"línea {linea}: {columna} = {fila[columna]!r}; en el plan, {entrada[columna]!r}")
    assert not problemas, "metadatos.csv no sigue el orden ni los valores del plan:\n" + resumen(problemas)


def test_generar_dataset_devuelve_las_filas_de_metadatos_con_sus_tipos(dataset_a):
    destino, devueltas = dataset_a
    _, filas = leer_metadatos(destino)
    assert isinstance(devueltas, list), f"generar_dataset tiene que devolver una lista; devolvió {tipo_de(devueltas)}."
    assert len(devueltas) == len(filas), f"Devolvió {len(devueltas)} filas y metadatos.csv tiene {len(filas)}."
    problemas = []
    for linea, (devuelta, fila) in enumerate(zip(devueltas, filas), start=2):
        if not isinstance(devuelta, dict) or set(devuelta) != set(COLUMNAS_CONTRATO):
            claves = sorted(devuelta) if isinstance(devuelta, dict) else tipo_de(devuelta)
            problemas.append(f"fila de la línea {linea}: tiene que ser un dict con las claves de COLUMNAS; es {claves}")
            continue
        convertida, problemas_fila = convertir_fila(fila, linea)
        problemas.extend(problemas_fila)
        for columna in COLUMNAS_CONTRATO:
            valor, esperado = devuelta[columna], convertida[columna]
            if columna in COLUMNAS_TEXTO:
                bien = isinstance(valor, str) and valor == esperado
                tipo = "str"
            elif columna in COLUMNAS_ENTERAS:
                bien = es_entero(valor) and valor == esperado
                tipo = "int"
            elif columna in COLUMNAS_REFERENCIA and esperado is None:
                bien, tipo = valor is None, "None"
            else:
                bien = es_float(valor) and valor == esperado and round(valor, 4) == valor
                tipo = "float con 4 decimales"
            if not bien:
                problemas.append(
                    f"línea {linea}, {columna}: devolvió {valor!r} ({tipo_de(valor)}); en el CSV {fila[columna]!r} "
                    f"(se espera {tipo})"
                )
    assert not problemas, "Lo que devuelve generar_dataset no coincide con metadatos.csv:\n" + resumen(problemas)


def test_metadatos_nucleos_segun_el_tipo_y_punto_de_referencia(cfg_reducido, dataset_a):
    destino, _ = dataset_a
    alto, ancho = tamano(cfg_reducido)
    problemas = []
    for fila in filas_convertidas(destino):
        archivo = fila["archivo"]
        nucleos, deltas = puntos_json(fila["nucleos"]), puntos_json(fila["deltas"])
        if nucleos is None or deltas is None:
            problemas.append(f"{archivo}: nucleos = {fila['nucleos']!r} y deltas = {fila['deltas']!r} tienen que ser listas JSON de [x, y]")
            continue
        esperados = NUCLEOS_POR_TIPO.get(fila["tipo"])
        if fila["n_nucleos"] != esperados or len(nucleos) != fila["n_nucleos"]:
            problemas.append(f"{archivo} ({fila['tipo']}): n_nucleos = {fila['n_nucleos']} y {len(nucleos)} en nucleos (esperados {esperados})")
        if len(deltas) != fila["n_deltas"]:
            problemas.append(f"{archivo}: n_deltas = {fila['n_deltas']} y hay {len(deltas)} en deltas")
        sin_referencia = fila["ref_x"] is None or fila["ref_y"] is None
        if sin_referencia != (fila["n_nucleos"] == 0) or (fila["ref_x"] is None) != (fila["ref_y"] is None):
            problemas.append(f"{archivo}: ref_x = {fila['ref_x']}, ref_y = {fila['ref_y']} con n_nucleos = {fila['n_nucleos']} (vacíos solo si no hay núcleos)")
            continue
        if sin_referencia or not nucleos:
            continue
        referencia = (fila["ref_x"], fila["ref_y"])
        if len(nucleos) == 2:
            esperado = ((nucleos[0][0] + nucleos[1][0]) / 2, (nucleos[0][1] + nucleos[1][1]) / 2)
            que = "el punto medio de nucleos"
        else:
            esperado, que = nucleos[0], "el núcleo"
        if abs(referencia[0] - esperado[0]) > 0.01 or abs(referencia[1] - esperado[1]) > 0.01:
            problemas.append(f"{archivo}: punto de referencia {fmt(referencia)}; {que} es {fmt(esperado)} (± 0,01)")
        if not dentro_de_la_imagen(referencia, alto, ancho):
            problemas.append(f"{archivo}: el punto de referencia {fmt(referencia)} cae fuera de la imagen")
    assert not problemas, "\n" + resumen(problemas)


def test_metadatos_parametros_dentro_de_los_rangos_del_cfg(cfg_reducido, dataset_a):
    destino, _ = dataset_a
    problemas = []
    for fila in filas_convertidas(destino):
        problemas.extend(problemas_de_rango(fila, cfg_reducido, que=fila["archivo"], angulo_redondeado=True))
    assert not problemas, "\n" + resumen(problemas)


@dataclass
class FilaRehecha:
    fila: dict
    texto: dict
    imagen: np.ndarray
    parametros: object
    nucleos: list
    deltas: list
    referencia: object


@pytest.fixture(scope="module")
def filas_rehechas(gen, var, cfg_reducido, dataset_a):
    """Cada fila de metadatos.csv rehecha con la receta de la sección 4."""
    destino, _ = dataset_a
    _, textos = leer_metadatos(destino)
    filas = filas_convertidas(destino)
    generar_base = atributo(gen, "generar_base")
    sortear = atributo(var, "sortear_parametros")
    aplicar_variante = atributo(var, "aplicar_variante")
    bases_reducidas: dict = {}
    rehechas = []
    for fila, texto in zip(filas, textos):
        clave = (fila["tipo"], fila["semilla"])
        if clave not in bases_reducidas:
            bases_reducidas[clave] = generar_base(fila["tipo"], fila["semilla"], copy.deepcopy(cfg_reducido))
        rng = np.random.default_rng([fila["semilla"], fila["variante"]])
        params = sortear(rng, copy.deepcopy(cfg_reducido))
        variante = aplicar_variante(copy.deepcopy(bases_reducidas[clave]), params, rng)
        rehechas.append(
            FilaRehecha(
                fila,
                texto,
                np.asarray(variante.imagen),
                variante.parametros,
                como_puntos(variante.singularidades.nucleos, "variante: núcleos"),
                como_puntos(variante.singularidades.deltas, "variante: deltas"),
                variante.punto_referencia,
            )
        )
    return destino, rehechas


def test_rehacer_cada_fila_con_la_receta_da_la_misma_imagen_que_el_png(filas_rehechas):
    destino, rehechas = filas_rehechas
    problemas = []
    for r in rehechas:
        png = leer_png(destino / r.fila["archivo"])
        if png is None:
            problemas.append(f"{r.fila['archivo']}: no se puede leer")
        elif png.shape != r.imagen.shape or png.dtype != r.imagen.dtype:
            problemas.append(f"{r.fila['archivo']}: el PNG es {png.dtype} {png.shape} y la variante rehecha {r.imagen.dtype} {r.imagen.shape}")
        elif not np.array_equal(png, r.imagen):
            distintos = int((png != r.imagen).sum())
            maximo = int(np.abs(png.astype(np.int16) - r.imagen.astype(np.int16)).max())
            problemas.append(f"{r.fila['archivo']}: {distintos} píxeles distintos (hasta {maximo} niveles)")
    assert not problemas, "Rehacer la fila con tipo, semilla y variante no da el PNG guardado:\n" + resumen(problemas)


def test_rehacer_cada_fila_da_sus_parametros_singularidades_y_referencia(filas_rehechas):
    _, rehechas = filas_rehechas
    problemas = []
    for r in rehechas:
        archivo = r.fila["archivo"]
        usados = campos_de(r.parametros, archivo)
        for campo in CAMPOS_PARAMETROS:
            if r.fila[campo] != round(usados[campo], 4):
                problemas.append(f"{archivo}: {campo} = {r.texto[campo]!r}; la variante rehecha usó {usados[campo]!r}")
        for columna, puntos in (("nucleos", r.nucleos), ("deltas", r.deltas)):
            esperado = [[round(x, 2), round(y, 2)] for x, y in puntos]
            try:
                leido = json.loads(r.texto[columna])
            except json.JSONDecodeError:
                leido = None
            if leido != esperado:
                problemas.append(f"{archivo}: {columna} = {r.texto[columna]!r}; se esperaba {json.dumps(esperado)}")
            if r.fila["n_" + columna] != len(puntos):
                problemas.append(f"{archivo}: n_{columna} = {r.fila['n_' + columna]}; la variante tiene {len(puntos)}")
        if r.referencia is None:
            if r.fila["ref_x"] is not None or r.fila["ref_y"] is not None:
                problemas.append(f"{archivo}: la variante no tiene punto de referencia y ref_x/ref_y no están vacíos")
        else:
            x, y = como_puntos([r.referencia], "punto_referencia")[0]
            if r.fila["ref_x"] != round(x, 4) or r.fila["ref_y"] != round(y, 4):
                problemas.append(f"{archivo}: ref = ({r.texto['ref_x']!r}, {r.texto['ref_y']!r}); la variante tiene {fmt((x, y))}")
    assert not problemas, "Las filas de metadatos.csv no coinciden con sus variantes rehechas:\n" + resumen(problemas)


def test_generar_dos_veces_en_carpetas_distintas_da_los_mismos_archivos(dataset_a, dataset_b, plan_reducido):
    destino_a, destino_b = dataset_a[0], dataset_b[0]
    distintos = [
        e["archivo"]
        for e in plan_reducido
        if not (destino_b / e["archivo"]).is_file()
        or (destino_a / e["archivo"]).read_bytes() != (destino_b / e["archivo"]).read_bytes()
    ]
    csv_igual = (destino_a / "metadatos.csv").read_bytes() == (destino_b / "metadatos.csv").read_bytes()
    assert not distintos and csv_igual, (
        f"Dos generaciones en carpetas distintas no dan lo mismo: {len(distintos)} PNG distintos {distintos[:10]}"
        f"{'' if csv_igual else ' y metadatos.csv distinto'}."
    )


@pytest.mark.parametrize("dato_viejo", ["train/viejo.png", "val/viejo.png", "test/viejo.png", "metadatos.csv"])
def test_generar_dataset_con_datos_previos_lanza_file_exists_error_sin_tocar_nada(ds, cfg_reducido, tmp_path, dato_viejo):
    destino = tmp_path / "destino"
    for relativa, contenido in ((dato_viejo, png_chico()), ("notas.txt", b"ajeno\n")):
        (destino / relativa).parent.mkdir(parents=True, exist_ok=True)
        (destino / relativa).write_bytes(contenido)
    antes = foto_de_carpeta(destino)
    lanza(
        FileExistsError,
        atributo(ds, "generar_dataset"),
        copy.deepcopy(cfg_reducido),
        destino,
        que=f"generar_dataset en una carpeta con {dato_viejo}",
    )
    despues = foto_de_carpeta(destino)
    cambios = sorted(set(antes) ^ set(despues)) + sorted(k for k in set(antes) & set(despues) if antes[k] != despues[k])
    assert not cambios, f"Con {dato_viejo} ya presente, generar_dataset cambió estos elementos de destino: {cambios[:10]}."


def test_generar_dataset_con_sobrescribir_vuelve_a_generar(dataset_a, dataset_b, plan_reducido):
    destino_a = dataset_a[0]
    destino_b, _, pisado, _ = dataset_b
    problemas = []
    png_en_b = {
        p.relative_to(destino_b).as_posix()
        for carpeta in ("train", "val", "test")
        for p in (destino_b / carpeta).glob("*.png")
    }
    esperados = {e["archivo"] for e in plan_reducido}
    if png_en_b != esperados:
        problemas.append(f"PNG sobrantes {sorted(png_en_b - esperados)} y faltantes {sorted(esperados - png_en_b)[:10]}")
    if (destino_b / pisado).read_bytes() != (destino_a / pisado).read_bytes():
        problemas.append(f"{pisado} conserva el contenido viejo")
    if (destino_b / "metadatos.csv").read_bytes() != (destino_a / "metadatos.csv").read_bytes():
        problemas.append("metadatos.csv no es el regenerado")
    assert not problemas, "generar_dataset(..., sobrescribir=True) sobre datos viejos:\n" + resumen(problemas)


def test_generar_dataset_con_sobrescribir_no_borra_archivos_ajenos(dataset_b):
    """Criterio sumado: solo se borran los PNG de train/, val/ y test/ y metadatos.csv."""
    destino, _, _, ajenos = dataset_b
    problemas = []
    for relativa, contenido in ajenos.items():
        ruta = destino / relativa
        if not ruta.is_file():
            problemas.append(f"borró {relativa}")
        elif ruta.read_bytes() != contenido:
            problemas.append(f"cambió el contenido de {relativa}")
    assert not problemas, "generar_dataset(..., sobrescribir=True) tocó archivos ajenos: " + "; ".join(problemas)


def test_generar_lamina_escribe_un_png_legible_y_devuelve_esa_ruta(ds, cfg, tmp_path):
    salida = tmp_path / "grilla_h1.png"
    devuelta = atributo(ds, "generar_lamina")(copy.deepcopy(cfg), salida)
    assert isinstance(devuelta, Path), f"generar_lamina tiene que devolver un Path; devolvió {tipo_de(devuelta)}."
    assert devuelta.resolve() == salida.resolve(), f"Devolvió {devuelta} y la salida pedida es {salida}."
    imagen = leer_png(salida)
    assert imagen is not None and imagen.ndim in (2, 3) and imagen.size > 0, (
        f"{salida} no existe o no se puede leer como PNG."
    )


# ---------------------------------------------------------------------------
# 5.8 Aceptación (pytest -m aceptacion)
# ---------------------------------------------------------------------------


@pytest.mark.aceptacion
def test_aceptacion_dataset_completo_con_cfg_real_en_menos_de_5_minutos(ds, cfg, tmp_path):
    inicio = time.perf_counter()
    atributo(ds, "generar_dataset")(copy.deepcopy(cfg), tmp_path)
    duracion = time.perf_counter() - inicio
    conteos = {p: len(list((tmp_path / p).glob("*.png"))) for p in ("train", "val", "test")}
    encabezado, filas = leer_metadatos(tmp_path)
    archivos_csv = {f["archivo"] for f in filas}
    archivos_png = {p.relative_to(tmp_path).as_posix() for p in tmp_path.glob("*/*.png")}
    incompletas = [
        f["archivo"] for f in filas
        if any(f[c] == "" for c in COLUMNAS_CONTRATO if c not in COLUMNAS_REFERENCIA)
    ]
    problemas = []
    if conteos != {"train": 72, "val": 114, "test": 114}:
        problemas.append(f"imágenes por partición {conteos} (esperado 72 / 114 / 114)")
    if encabezado != list(COLUMNAS_CONTRATO) or len(filas) != 300:
        problemas.append(f"metadatos.csv con {len(filas)} filas (esperado 300) o encabezado distinto de COLUMNAS")
    if archivos_csv != archivos_png:
        problemas.append("los archivos de metadatos.csv no son los PNG generados")
    if incompletas:
        problemas.append(f"filas con columnas vacías: {incompletas[:10]}")
    if duracion >= 300:
        problemas.append(f"tardó {duracion:.1f} s (máximo 300 s)")
    assert not problemas, f"Dataset completo ({duracion:.1f} s):\n" + resumen(problemas)
