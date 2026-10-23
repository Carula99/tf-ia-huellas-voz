"""Tests de H2 · preprocesamiento de huellas.

Verifican la sección 5 de docs/contratos/h2_preprocesamiento.md (versión 1.2, 22/10/2026).
La sección 6 es manual y se anota en el Documento 2.

Las huellas salen de H1, que ya está probado, con la receta de la sección 4 de su contrato.
De H1 se usan solo generar_base, ParametrosVariante, sortear_parametros, aplicar_variante,
plan_dataset y los campos de HuellaBase (imagen, mascara, orientacion, singularidades,
periodo_px) y de Variante (imagen, mascara, singularidades, punto_referencia, parametros,
mapa_x, mapa_y).

Lecturas del contrato (sección 7, "Sumadas en la versión 1.1"):
- Punto de referencia de una base: el que da preprocesar(base.imagen), con su angulo, en
  las 4 bases de 5.2 (también el arco) y en el arco de 5.9.
- Base de arco de 5.2: la del intruso 4 (arco, semilla 10004).
- Zona de comparación (5.6): binary_erosion con un 9 × 9 y border_value=0, así que queda
  a >= 4 px del borde de la ventana.
- "En al menos el 95 %" es la proporción >= 0,95 (en los 9 arcos de la muestra no puede
  fallar ninguno). Una variante en la que preprocesar lanza una excepción cuenta como
  falla en todos los criterios.
- 5.6: el esqueleto y el realce se exigen en cada variante; el período se mide sobre
  realzada tal cual.
- 5.9: todos los criterios de 5.4 y 5.5 sobre las 300 imágenes, más el arco a <= 30 px en
  el 80 % y el tiempo; son varios tests con @pytest.mark.aceptacion que comparten un fixture.
- Sección 1, como en H1 1.1: los <float> exigen float de Python, los enteros int que no
  sea bool, las claves fijas se comparan por tipo y valor y, si falta la sección entera,
  el ValueError nombra sus 9 claves.
- Formato, entradas con ValueError, diferencias de ángulo, 5.3, 5.7 y 5.8: como lo dice
  la sección 7.

Cambio de la versión 1.2 (sección 7, "Sumadas en la versión 1.2"): la matriz gira por
a_v = ((angulo + π/2) mod π) − π/2, en [−π/2, π/2), el menor de los dos giros que dejan
θ∞ horizontal, para que la ventana no quede cabeza abajo. angulo sigue en [0, π): 5.5 y
θ_v no cambian (θ − a_v es θ − angulo mod π). En 5.3 la parte 2 × 2 se compara con a_v y
se suma un criterio: cada delta real de las variantes con núcleo, llevado con matriz,
queda debajo del centro de la ventana (y_w > 124.5).

src.config y los módulos de src.huellas se importan en fixtures: si algo todavía no existe,
fallan (o dan error) esos tests y no la colección entera. La muestra de la sección 4 (15
bases y 156 variantes) se genera y se preprocesa una sola vez (fixtures de módulo); de cada
variante se guarda solo lo que miden los tests, para no tener 156 variantes en memoria.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from scipy import ndimage

RAIZ = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Valores del contrato
# ---------------------------------------------------------------------------

TIPOS = ("presilla", "verticilo", "arco_tienda", "arco")
TIPOS_CON_NUCLEO = ("presilla", "verticilo", "arco_tienda")
METODOS_CONTRATO = ("nucleo", "nucleos", "curvatura")
METODO_POR_TIPO = {"presilla": "nucleo", "verticilo": "nucleos", "arco_tienda": "nucleo", "arco": "curvatura"}
CAMPOS_HUELLA = (
    "realzada", "binaria", "esqueleto", "mascara", "orientacion",
    "punto_referencia", "metodo", "nucleos", "angulo", "matriz",
)

# Sección 0: ventana de (alto, ancho) = (250, 250) con centro (124.5, 124.5).
ALTO_W, ANCHO_W = 250, 250
VENTANA = (ALTO_W, ANCHO_W)
CENTRO_W = ((ANCHO_W - 1) / 2, (ALTO_W - 1) / 2)
FONDO = 245  # gris de fondo de H1 (sección 7)

# Sección 1
SECCION_H2 = "huellas.preprocesamiento"
# camino: (tipo, mínimo, máximo, incluye el mínimo, incluye el máximo)
CLAVES_H2 = {
    "huellas.preprocesamiento.mediana_px": ("entero impar", 3, 7, True, True),
    "huellas.preprocesamiento.umbral_segmentacion": ("float", 0.0, 1.0, False, False),
    "huellas.preprocesamiento.sigma_tensor_px": ("float", 1.0, 10.0, True, True),
    "huellas.preprocesamiento.sigma_ajuste_px": ("float", 1.0, 10.0, True, True),
    "huellas.preprocesamiento.radio_poincare_px": ("float", 3.0, 15.0, True, True),
    "huellas.preprocesamiento.margen_singularidades_px": ("float", 0.0, 50.0, True, True),
    "huellas.preprocesamiento.holgura_nucleo_px": ("float", 0.0, 15.0, True, True),
    "huellas.preprocesamiento.clahe_limite": ("float", 1.0, 5.0, True, True),
    "huellas.preprocesamiento.clahe_grilla": ("entero", 2, 16, True, True),
}
CLAVES_FIJAS = {"huellas.ventana_px": [250, 250], "huellas.periodo_crestas_px": [8, 10]}

# Sección 4: la muestra
IDENTIDADES_MUESTRA = (1, 2, 3)
INTRUSOS_MUESTRA = tuple(range(1, 13))
VARIANTES_POR_TIPO_MUESTRA = {"presilla": 49, "verticilo": 49, "arco_tienda": 49, "arco": 9}
CON_CORTE_MUESTRA = 38
PROPORCION_MIN = 0.95

# 5.2: bases de las 3 identidades y del intruso 4 (arco)
BASES_52 = {
    "presilla": ("identidad", 1),
    "verticilo": ("identidad", 2),
    "arco_tienda": ("identidad", 3),
    "arco": ("intruso", 4),
}
NOMBRES_52 = tuple(BASES_52)
BORDE_LIENZO = 50
RECORTE_GRANDE = (300, 260)  # (alto, ancho)
RECORTE_CHICO = (200, 180)
TOL_COLOR_PX, TOL_COLOR_GRADOS, COINCIDENCIA_MASCARA_COLOR = 0.5, 0.5, 0.99
TOL_LIENZO_PX, TOL_RECORTE_PX, TOL_CHICO_PX, TOL_LIENZO_GRADOS = 1.0, 1.5, 2.0, 1.0

# 5.3
TOL_ROTACION, TOL_CENTRO, TOL_NUCLEOS = 1e-9, 1e-6, 1e-6
IOU_MIN = 0.90

# 5.4 y 5.5
REF_95_PX, REF_MAX_PX, NUCLEO_95_PX = 8.0, 16.0, 8.0
ANGULO_95_GRADOS, ANGULO_MAX_GRADOS, ANGULO_BASES_GRADOS = 5.0, 10.0, 2.0

# 5.6
EROSION_ZONA = 9
CAMPO_MIN, CRESTAS_MIN = 0.95, 0.90
DENSIDAD_ESQUELETO = (0.06, 0.16)
CONTRASTE_MIN, TOL_PERIODO_PX = 60.0, 1.0

# 5.7
ANGULOS_CORTE_ESTABILIDAD = (0, 90, 180, 270)
RADIO_ESTABILIDAD, CAMPO_ESTABILIDAD_MIN = 80.0, 0.95
ANGULO_ESTABILIDAD_GRADOS, REF_ESTABILIDAD_PX = 3.0, 5.0

# 5.9
ARCO_ACEPTACION_PX, ARCO_ACEPTACION_PROPORCION = 30.0, 0.80
TIEMPO_ACEPTACION_S = 120.0


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
    """Igualdad de valor y de tipo, recursiva (250 != 250.0, True != 1)."""
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


def leer_config_real() -> dict:
    with open(RAIZ / "config.yaml", encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def sin_clave(datos: dict, camino: str) -> dict:
    """Copia profunda de datos sin la clave `camino` (hoja o sección)."""
    copia = copy.deepcopy(datos)
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


def resumen(problemas: list[str], maximo: int = 20) -> str:
    """Las primeras líneas de una lista de problemas, para los mensajes de error."""
    lineas = [f"  - {p}" for p in problemas[:maximo]]
    if len(problemas) > maximo:
        lineas.append(f"  ... y {len(problemas) - maximo} más")
    return "\n".join(lineas)


def fmt(punto) -> str:
    if punto is None:
        return "None"
    return f"({float(punto[0]):.2f}, {float(punto[1]):.2f})"


def como_punto(valor) -> tuple[float, float]:
    x, y = valor
    return float(x), float(y)


def distancia(a, b) -> float:
    return math.dist(como_punto(a), como_punto(b))


def descripcion_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# Geometría y mediciones (sección 0 de H2 y de H1)
# ---------------------------------------------------------------------------


def a_intervalo(grados: float) -> float:
    """Lleva un ángulo en grados a [−90, 90)."""
    return (grados + 90.0) % 180.0 - 90.0


def error_angulo(angulo: float, rot_grados: float) -> float:
    """Error de ángulo de la sección 0: ((grados(angulo) + rot_grados + 90) mod 180) − 90."""
    return a_intervalo(math.degrees(float(angulo)) + float(rot_grados))


def giro_de_la_ventana(angulo: float) -> float:
    """a_v = ((angulo + π/2) mod π) − π/2, en [−π/2, π/2): el giro de la matriz (sección 0, 1.2)."""
    return (float(angulo) + math.pi / 2) % math.pi - math.pi / 2


def diferencia_angulos(a: float, b: float) -> float:
    """Diferencia entre dos ángulos de preprocesar, en grados, en [−90, 90)."""
    return a_intervalo(math.degrees(float(a)) - math.degrees(float(b)))


def como_matriz(matriz) -> np.ndarray:
    return np.asarray(matriz, dtype=np.float64).reshape(2, 3)


def aplicar_matriz(matriz, punto) -> tuple[float, float]:
    m = como_matriz(matriz)
    x, y = como_punto(punto)
    return float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2])


def llevar_a_la_ventana(mascara: np.ndarray, matriz) -> np.ndarray:
    """warpAffine con vecino más cercano de una máscara de la entrada a la ventana (bool)."""
    return cv2.warpAffine(
        np.asarray(mascara).astype(np.uint8), como_matriz(matriz), (ANCHO_W, ALTO_W), flags=cv2.INTER_NEAREST
    ) > 0


def dentro_de_la_entrada(forma, matriz) -> np.ndarray:
    """Píxeles de la ventana que vienen de dentro de la entrada (sección 0)."""
    return llevar_a_la_ventana(np.ones(tuple(forma)[:2], dtype=np.uint8), matriz)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else float("nan")


def periodo_medido(imagen: np.ndarray) -> float:
    """Período de las crestas según la sección 0 de H1 (recorte central de 192 × 192 px)."""
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


def transformacion_afin(punto, rot_grados, escala, tx_px, ty_px, alto: int, ancho: int) -> tuple[float, float]:
    """Fórmula afín de la sección 0 de H1 (rotación y escala alrededor del centro, más traslación)."""
    x, y = como_punto(punto)
    cx, cy = (ancho - 1) / 2, (alto - 1) / 2
    a, s = math.radians(rot_grados), escala
    dx, dy = x - cx, y - cy
    return (
        cx + s * (math.cos(a) * dx + math.sin(a) * dy) + tx_px,
        cy + s * (-math.sin(a) * dx + math.cos(a) * dy) + ty_px,
    )


def bloques_2x2(esqueleto: np.ndarray) -> np.ndarray:
    """(i, j) de cada bloque de 2 × 2 en True (sección 0)."""
    e = np.asarray(esqueleto, dtype=bool)
    bloque = e[:-1, :-1] & e[1:, :-1] & e[:-1, 1:] & e[1:, 1:]
    return np.argwhere(bloque)


def disco_en_la_ventana(radio: float) -> np.ndarray:
    filas, columnas = np.mgrid[0:ALTO_W, 0:ANCHO_W]
    return (columnas - CENTRO_W[0]) ** 2 + (filas - CENTRO_W[1]) ** 2 <= radio ** 2


def origen_recorte(punto, alto_recorte: int, ancho_recorte: int, alto: int, ancho: int) -> tuple[int, int]:
    """Origen (x0, y0) de un recorte centrado en `punto` (sección 7), corrido para que entre."""
    x0 = round(float(punto[0])) - ancho_recorte // 2
    y0 = round(float(punto[1])) - alto_recorte // 2
    x0 = min(max(x0, 0), ancho - ancho_recorte)
    y0 = min(max(y0, 0), alto - alto_recorte)
    return int(x0), int(y0)


def recortar(imagen: np.ndarray, origen, alto_recorte: int, ancho_recorte: int) -> np.ndarray:
    x0, y0 = origen
    return np.ascontiguousarray(imagen[y0:y0 + alto_recorte, x0:x0 + ancho_recorte])


@dataclass
class Verdad:
    """Campo, crestas y máscaras verdaderas de una variante en su ventana (sección 0)."""

    campo: np.ndarray            # θ_v, float32 (250, 250)
    crestas: np.ndarray          # bool (250, 250)
    mascara_real: np.ndarray     # bool (250, 250)
    dentro: np.ndarray           # bool (250, 250): viene de dentro de la entrada


def verdad_en_la_ventana(base, variante, resultado) -> Verdad:
    """Lleva el campo y las crestas de la base a la ventana de `resultado` (sección 0)."""
    matriz = como_matriz(resultado.matriz)
    inversa = cv2.invertAffineTransform(matriz)
    filas, columnas = np.mgrid[0:ALTO_W, 0:ANCHO_W].astype(np.float64)
    px = inversa[0, 0] * columnas + inversa[0, 1] * filas + inversa[0, 2]
    py = inversa[1, 0] * columnas + inversa[1, 1] * filas + inversa[1, 2]
    mapa_x = np.asarray(variante.mapa_x, dtype=np.float64)
    mapa_y = np.asarray(variante.mapa_y, dtype=np.float64)
    qx = ndimage.map_coordinates(mapa_x, [py, px], order=1, mode="nearest")
    qy = ndimage.map_coordinates(mapa_y, [py, px], order=1, mode="nearest")
    theta = np.asarray(base.orientacion, dtype=np.float64)
    c = ndimage.map_coordinates(np.cos(2 * theta), [qy, qx], order=1, mode="nearest")
    s = ndimage.map_coordinates(np.sin(2 * theta), [qy, qx], order=1, mode="nearest")
    campo = 0.5 * np.arctan2(s, c) - math.radians(float(variante.parametros.rot_grados)) - float(resultado.angulo)
    gris = ndimage.map_coordinates(np.asarray(base.imagen, dtype=np.float64), [qy, qx], order=1, mode="nearest")
    return Verdad(
        campo=campo.astype(np.float32),
        crestas=gris < 128,
        mascara_real=llevar_a_la_ventana(variante.mascara, matriz),
        dentro=dentro_de_la_entrada(np.shape(variante.imagen), matriz),
    )


def zona_de_comparacion(mascara, verdad: Verdad) -> np.ndarray:
    """mascara ∩ máscara real en la ventana, erosionada con un 9 × 9 (fuera de la ventana = False)."""
    union = np.asarray(mascara, dtype=bool) & verdad.mascara_real
    estructura = np.ones((EROSION_ZONA, EROSION_ZONA), dtype=bool)
    return ndimage.binary_erosion(union, structure=estructura, border_value=0)


# ---------------------------------------------------------------------------
# Casos (variantes y bases preprocesadas) y criterios por proporción
# ---------------------------------------------------------------------------


@dataclass
class Caso:
    """Lo que los tests necesitan de una imagen preprocesada (variante o base)."""

    archivo: str                          # archivo del plan, o "base <tipo> semilla <s>"
    tipo: str
    clave_base: tuple[str, int]
    forma: tuple[int, int]                # (alto, ancho) de la entrada
    rot_grados: float = 0.0
    escala: float = 1.0
    tx_px: float = 0.0
    ty_px: float = 0.0
    corte_fraccion: float = 0.0
    ref_real: tuple[float, float] | None = None
    nucleos_reales: tuple[tuple[float, float], ...] = ()
    deltas_reales: tuple[tuple[float, float], ...] = ()
    periodo_real: float = float("nan")
    resultado: object | None = None
    error: str | None = None
    tiempo_s: float = 0.0
    verdad: Verdad | None = None
    error_verdad: str | None = None


def resumir(resultado) -> SimpleNamespace:
    """Solo los valores de una HuellaPreprocesada que miden 5.4, 5.5 y 5.9 (sin arrays)."""
    return SimpleNamespace(
        metodo=resultado.metodo,
        punto_referencia=resultado.punto_referencia,
        nucleos=resultado.nucleos,
        angulo=resultado.angulo,
    )


def preprocesar_caso(caso: Caso, preprocesar, imagen: np.ndarray, cfg: dict) -> Caso:
    """Corre preprocesar sobre una copia de `imagen` y guarda el resultado o la excepción."""
    entrada = np.array(imagen, copy=True)
    inicio = time.perf_counter()
    try:
        caso.resultado = preprocesar(entrada, cfg)
    except Exception as error:  # cuenta como falla en cada criterio
        caso.error = f"preprocesar lanzó {descripcion_error(error)}"
    caso.tiempo_s = time.perf_counter() - inicio
    return caso


def caso_de_variante(entrada: dict, base, variante) -> Caso:
    params = variante.parametros
    ref = variante.punto_referencia
    return Caso(
        archivo=entrada["archivo"],
        tipo=entrada["tipo"],
        clave_base=(entrada["tipo"], entrada["semilla"]),
        forma=tuple(np.shape(variante.imagen)[:2]),
        rot_grados=float(params.rot_grados),
        escala=float(params.escala),
        tx_px=float(params.tx_px),
        ty_px=float(params.ty_px),
        corte_fraccion=float(params.corte_fraccion),
        ref_real=None if ref is None else como_punto(ref),
        nucleos_reales=tuple(como_punto(n) for n in variante.singularidades.nucleos),
        deltas_reales=tuple(como_punto(d) for d in variante.singularidades.deltas),
        periodo_real=float(base.periodo_px) * float(params.escala),
    )


def caso_de_base(clave: tuple[str, int], base) -> Caso:
    return Caso(
        archivo=f"base {clave[0]} semilla {clave[1]}",
        tipo=clave[0],
        clave_base=clave,
        forma=tuple(np.shape(base.imagen)[:2]),
        nucleos_reales=tuple(como_punto(n) for n in base.singularidades.nucleos),
        periodo_real=float(base.periodo_px),
    )


def receta_h1(h1, cfg: dict, entrada: dict, base):
    """Variante de una entrada del plan con la receta de la sección 4 de H1."""
    rng = np.random.default_rng([entrada["semilla"], entrada["variante"]])
    params = h1.sortear_parametros(rng, cfg)
    return h1.aplicar_variante(base, params, rng)


def verdad_de(caso: Caso) -> Verdad:
    if caso.verdad is None:
        raise RuntimeError(f"no se pudo calcular la verdad en la ventana ({caso.error_verdad})")
    return caso.verdad


def revisar(casos: list[Caso], chequeo) -> list[tuple[Caso, bool, str]]:
    """Aplica chequeo(caso) -> (ok, detalle) a cada caso; una excepción es una falla."""
    filas = []
    for caso in casos:
        if caso.error is not None:
            filas.append((caso, False, caso.error))
            continue
        try:
            ok, detalle = chequeo(caso)
        except Exception as error:
            ok, detalle = False, descripcion_error(error)
        filas.append((caso, bool(ok), detalle))
    return filas


def _fallas(filas) -> list[str]:
    return [f"{caso.archivo}: {detalle}" for caso, ok, detalle in filas if not ok]


def exigir_todas(filas, que: str) -> None:
    if not filas:
        pytest.fail(f"{que}: no hay casos para medir.", pytrace=False)
    fallas = _fallas(filas)
    if fallas:
        pytest.fail(
            f"{que}: fallan {len(fallas)} de {len(filas)} (se exigen todas).\n{resumen(fallas)}",
            pytrace=False,
        )


def exigir_proporcion(filas, que: str, minimo: float = PROPORCION_MIN) -> None:
    if not filas:
        pytest.fail(f"{que}: no hay casos para medir.", pytrace=False)
    fallas = _fallas(filas)
    pasan = len(filas) - len(fallas)
    proporcion = pasan / len(filas)
    if proporcion < minimo:
        pytest.fail(
            f"{que}: pasan {pasan} de {len(filas)} ({proporcion:.1%}); se exige al menos el "
            f"{minimo:.0%}. Fallan {len(fallas)}:\n{resumen(fallas)}",
            pytrace=False,
        )


def de_tipo(casos: list[Caso], tipo: str) -> list[Caso]:
    elegidos = [c for c in casos if c.tipo == tipo]
    if not elegidos:
        pytest.fail(f"No hay casos de tipo {tipo!r} para medir.", pytrace=False)
    return elegidos


# Chequeos de 5.4 y 5.5 (los usan la muestra y la aceptación)


def chequeo_metodo(tipo: str):
    esperado = METODO_POR_TIPO[tipo]

    def chequeo(caso):
        metodo = caso.resultado.metodo
        return metodo == esperado, f"metodo {metodo!r}; se espera {esperado!r}"

    return chequeo


def chequeo_referencia(limite: float):
    def chequeo(caso):
        if caso.ref_real is None:
            return False, "la variante no tiene punto de referencia real"
        d = distancia(caso.resultado.punto_referencia, caso.ref_real)
        return d <= limite, (
            f"distancia {d:.2f} px (límite {limite:g}); punto {fmt(caso.resultado.punto_referencia)}, "
            f"real {fmt(caso.ref_real)}"
        )

    return chequeo


def chequeo_nucleos(limite: float):
    def chequeo(caso):
        detectados = [como_punto(n) for n in caso.resultado.nucleos]
        lejos = []
        for real in caso.nucleos_reales:
            d = min((math.dist(real, n) for n in detectados), default=math.inf)
            if d > limite:
                lejos.append(f"núcleo real {fmt(real)} a {d:.2f} px")
        detalle = "; ".join(lejos) + f" (límite {limite:g} px; nucleos = {[fmt(n) for n in detectados]})"
        return not lejos, detalle

    return chequeo


def chequeo_curvatura(caso):
    r = caso.resultado
    ok = r.metodo == "curvatura" and len(r.nucleos) == 0
    return ok, f"metodo {r.metodo!r} y {len(r.nucleos)} núcleos; se espera 'curvatura' sin núcleos"


def chequeo_angulo(limite: float):
    def chequeo(caso):
        e = error_angulo(caso.resultado.angulo, caso.rot_grados)
        return abs(e) <= limite, (
            f"error de ángulo {e:+.2f}° (límite {limite:g}°; angulo = "
            f"{math.degrees(float(caso.resultado.angulo)):.2f}°, rot_grados = {caso.rot_grados:+.2f})"
        )

    return chequeo


def chequeo_arco_sigue_a_su_base(referencias_base: dict):
    """5.9: punto de la variante a <= 30 px del de su base llevado con la fórmula afín de H1."""

    def chequeo(caso):
        ref_base = referencias_base.get(caso.clave_base)
        if isinstance(ref_base, str):
            return False, f"la base no tiene punto de referencia: {ref_base}"
        alto, ancho = caso.forma
        esperado = transformacion_afin(
            ref_base, caso.rot_grados, caso.escala, caso.tx_px, caso.ty_px, alto, ancho
        )
        d = distancia(caso.resultado.punto_referencia, esperado)
        return d <= ARCO_ACEPTACION_PX, (
            f"distancia {d:.2f} px (límite {ARCO_ACEPTACION_PX:g}); punto "
            f"{fmt(caso.resultado.punto_referencia)}, el de la base llevado {fmt(esperado)}"
        )

    return chequeo


# Formato de una HuellaPreprocesada (5.2)


def problemas_de_formato(r, clase) -> list[str]:
    """Lo que no cumple el formato de 5.2 (lista vacía si está todo bien)."""
    if not isinstance(r, clase):
        return [f"preprocesar devuelve {type(r).__name__}, no HuellaPreprocesada"]
    problemas = []
    for nombre, tipo in (
        ("realzada", np.uint8), ("binaria", np.bool_), ("esqueleto", np.bool_),
        ("mascara", np.bool_), ("orientacion", np.float64),
    ):
        valor = getattr(r, nombre, None)
        if not isinstance(valor, np.ndarray):
            problemas.append(f"{nombre} no es un np.ndarray (es {type(valor).__name__})")
            continue
        if valor.dtype != tipo:
            problemas.append(f"{nombre} es {valor.dtype}; se espera {np.dtype(tipo)}")
        if valor.shape != VENTANA:
            problemas.append(f"{nombre} tiene forma {valor.shape}; se espera {VENTANA}")
    orientacion = getattr(r, "orientacion", None)
    if isinstance(orientacion, np.ndarray) and orientacion.dtype.kind == "f":
        no_finitos = int(np.count_nonzero(~np.isfinite(orientacion)))
        fuera = int(np.count_nonzero((orientacion < 0) | (orientacion >= np.pi)))
        if no_finitos or fuera:
            problemas.append(
                f"orientacion tiene {no_finitos} valores no finitos y {fuera} fuera de [0, π) "
                f"(mín {np.nanmin(orientacion):.4f}, máx {np.nanmax(orientacion):.4f})"
            )
    ref = getattr(r, "punto_referencia", None)
    if not (
        isinstance(ref, tuple) and len(ref) == 2
        and all(isinstance(v, float) and math.isfinite(v) for v in ref)
    ):
        problemas.append(f"punto_referencia tiene que ser una tupla de 2 float finitos; es {ref!r}")
    metodo = getattr(r, "metodo", None)
    if metodo not in METODOS_CONTRATO:
        problemas.append(f"metodo = {metodo!r} no está en {METODOS_CONTRATO}")
    nucleos = getattr(r, "nucleos", None)
    if not (
        isinstance(nucleos, tuple)
        and all(
            isinstance(n, tuple) and len(n) == 2 and all(isinstance(v, float) and math.isfinite(v) for v in n)
            for n in nucleos
        )
    ):
        problemas.append(f"nucleos tiene que ser una tupla de pares (x, y) de float; es {nucleos!r}")
    angulo = getattr(r, "angulo", None)
    if not (isinstance(angulo, float) and math.isfinite(angulo) and 0.0 <= angulo < math.pi):
        problemas.append(f"angulo tiene que ser un float en [0, π); es {angulo!r}")
    matriz = getattr(r, "matriz", None)
    if not isinstance(matriz, np.ndarray):
        problemas.append(f"matriz no es un np.ndarray (es {type(matriz).__name__})")
    else:
        if matriz.dtype != np.float64 or matriz.shape != (2, 3):
            problemas.append(f"matriz es {matriz.dtype} de forma {matriz.shape}; se espera float64 de (2, 3)")
        elif not np.all(np.isfinite(matriz)):
            problemas.append("matriz tiene valores no finitos")
    for nombre in ("binaria", "esqueleto"):
        valor = getattr(r, nombre, None)
        if isinstance(valor, np.ndarray) and not np.any(valor):
            problemas.append(f"{nombre} está vacío (ningún True)")
    return problemas


def problemas_de_salida_no_vacia(r, clase) -> list[str]:
    """Formato de 5.2 y, además, mascara con algún True ("no vacías")."""
    problemas = problemas_de_formato(r, clase)
    mascara = getattr(r, "mascara", None)
    if isinstance(mascara, np.ndarray) and not np.any(mascara):
        problemas.append("mascara está vacía (ningún True)")
    return problemas


def iguales(a, b) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> dict:
    config = importar("src.config")
    return atributo(config, "cargar_config")()


@pytest.fixture(scope="module")
def h1():
    generador = importar("src.huellas.generador")
    variantes = importar("src.huellas.variantes")
    dataset = importar("src.huellas.dataset")
    return SimpleNamespace(
        generar_base=atributo(generador, "generar_base"),
        ParametrosVariante=atributo(variantes, "ParametrosVariante"),
        sortear_parametros=atributo(variantes, "sortear_parametros"),
        aplicar_variante=atributo(variantes, "aplicar_variante"),
        plan_dataset=atributo(dataset, "plan_dataset"),
    )


@pytest.fixture(scope="module")
def pre():
    return importar("src.huellas.preprocesamiento")


@pytest.fixture(scope="module")
def preprocesar(pre):
    return atributo(pre, "preprocesar")


@pytest.fixture(scope="module")
def clase_huella(pre):
    return atributo(pre, "HuellaPreprocesada")


@pytest.fixture(scope="module")
def plan(cfg, h1) -> list[dict]:
    return h1.plan_dataset(cfg)


@pytest.fixture(scope="module")
def entradas_muestra(plan) -> list[dict]:
    """Entradas del plan de las identidades 1 a 3 y de los intrusos 1 a 12 (sección 4)."""
    return [e for e in plan if e["identidad"] in IDENTIDADES_MUESTRA or e["intruso"] in INTRUSOS_MUESTRA]


@pytest.fixture(scope="module")
def bases_muestra(cfg, h1, entradas_muestra) -> dict:
    """Las 15 bases de la muestra, por (tipo, semilla)."""
    bases = {}
    for entrada in entradas_muestra:
        clave = (entrada["tipo"], entrada["semilla"])
        if clave not in bases:
            bases[clave] = h1.generar_base(entrada["tipo"], entrada["semilla"], cfg)
    return bases


@pytest.fixture(scope="module")
def claves_52(plan) -> dict:
    """(tipo, semilla) de las bases de 5.2: identidades 1 a 3 e intruso 4."""
    claves = {}
    for nombre, (campo, numero) in BASES_52.items():
        entrada = next((e for e in plan if e[campo] == numero), None)
        if entrada is None:
            pytest.fail(f"plan_dataset no tiene entradas con {campo} = {numero}.", pytrace=False)
        if entrada["tipo"] != nombre:
            pytest.fail(
                f"En el plan, {campo} {numero} es de tipo {entrada['tipo']!r}; el contrato espera {nombre!r}.",
                pytrace=False,
            )
        claves[nombre] = (entrada["tipo"], entrada["semilla"])
    return claves


@pytest.fixture(scope="module")
def bases_preprocesadas(cfg, bases_muestra, preprocesar) -> dict:
    """preprocesar(base.imagen) de las 15 bases de la muestra, por (tipo, semilla)."""
    return {
        clave: preprocesar_caso(caso_de_base(clave, base), preprocesar, base.imagen, cfg)
        for clave, base in bases_muestra.items()
    }


@pytest.fixture(scope="module")
def muestra(cfg, h1, entradas_muestra, bases_muestra, preprocesar) -> list[Caso]:
    """Las 156 variantes de la sección 4, preprocesadas, con su verdad en la ventana."""
    casos = []
    for entrada in entradas_muestra:
        base = bases_muestra[(entrada["tipo"], entrada["semilla"])]
        variante = receta_h1(h1, cfg, entrada, base)
        caso = preprocesar_caso(caso_de_variante(entrada, base, variante), preprocesar, variante.imagen, cfg)
        if caso.error is None:
            try:
                caso.verdad = verdad_en_la_ventana(base, variante, caso.resultado)
            except Exception as error:
                caso.error_verdad = descripcion_error(error)
        casos.append(caso)
    return casos


def base_52(bases_muestra, claves_52, nombre):
    return bases_muestra[claves_52[nombre]]


def resultado_de_base(bases_preprocesadas, clave):
    caso = bases_preprocesadas[clave]
    if caso.error is not None:
        pytest.fail(f"{caso.archivo}: {caso.error}", pytrace=False)
    return caso.resultado


# ---------------------------------------------------------------------------
# Sección 4 · la muestra (control de la receta de H1, no de H2)
# ---------------------------------------------------------------------------


def test_la_muestra_tiene_la_composicion_de_la_seccion_4(muestra, bases_muestra):
    por_tipo = Counter(c.tipo for c in muestra)
    con_corte = sum(c.corte_fraccion > 0 for c in muestra)
    esperado = (dict(VARIANTES_POR_TIPO_MUESTRA), CON_CORTE_MUESTRA, 15)
    medido = (dict(por_tipo), con_corte, len(bases_muestra))
    assert medido == esperado, (
        f"La muestra no es la de la sección 4: variantes por tipo {dict(por_tipo)}, {con_corte} con corte y "
        f"{len(bases_muestra)} bases; se esperan {dict(VARIANTES_POR_TIPO_MUESTRA)}, {CON_CORTE_MUESTRA} y 15."
    )


# ---------------------------------------------------------------------------
# 5.1 · Configuración
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clave", list(CLAVES_H2))
def test_clave_de_preprocesamiento_con_su_tipo_y_rango(cfg, clave):
    tipo, minimo, maximo, con_minimo, con_maximo = CLAVES_H2[clave]
    valor = obtener(cfg, clave)
    rango = f"{'[' if con_minimo else '('}{minimo}, {maximo}{']' if con_maximo else ')'}"
    if tipo == "float":
        assert es_float(valor), (
            f"{clave} tiene que ser un float de Python (por ejemplo 8.0, no 8); es {valor!r} ({tipo_de(valor)})."
        )
    else:
        assert es_entero(valor), f"{clave} tiene que ser un entero (int, no bool); es {valor!r} ({tipo_de(valor)})."
    por_abajo = minimo <= valor if con_minimo else minimo < valor
    por_arriba = valor <= maximo if con_maximo else valor < maximo
    assert por_abajo and por_arriba, f"{clave} = {valor!r} está fuera del rango {rango}."
    if tipo == "entero impar":
        assert valor % 2 == 1, f"{clave} = {valor!r} tiene que ser impar."


@pytest.mark.parametrize("clave", list(CLAVES_FIJAS))
def test_ventana_y_periodo_de_crestas_con_su_valor_fijo(cfg, clave):
    valor, esperado = obtener(cfg, clave), CLAVES_FIJAS[clave]
    assert iguales_exactos(valor, esperado), (
        f"{clave} = {valor!r} ({tipo_de(valor)}); se espera {esperado!r} ({tipo_de(esperado)})."
    )


@pytest.mark.parametrize("clave", list(CLAVES_H2))
def test_cargar_config_exige_cada_clave_nueva(tmp_path, clave):
    cargar_config = atributo(importar("src.config"), "cargar_config")
    ruta = escribir_yaml(tmp_path / "config.yaml", sin_clave(leer_config_real(), clave))
    error = lanza(ValueError, cargar_config, ruta, que=f"cargar_config con un config.yaml sin {clave}")
    assert nombra(str(error), clave), (
        f"El ValueError de cargar_config sin {clave} no nombra ese camino completo. Mensaje: {error}"
    )


def test_cargar_config_sin_la_seccion_nombra_sus_nueve_claves(tmp_path):
    cargar_config = atributo(importar("src.config"), "cargar_config")
    ruta = escribir_yaml(tmp_path / "config.yaml", sin_clave(leer_config_real(), SECCION_H2))
    error = lanza(ValueError, cargar_config, ruta, que=f"cargar_config con un config.yaml sin {SECCION_H2}")
    faltan = [clave for clave in CLAVES_H2 if not nombra(str(error), clave)]
    assert not faltan, (
        f"Sin la sección {SECCION_H2}, el ValueError no nombra {len(faltan)} de sus 9 claves: "
        f"{', '.join(faltan)}. Mensaje: {error}"
    )


# ---------------------------------------------------------------------------
# 5.2 · Formato, color, tamaño y errores (bases de las 3 identidades y del intruso 4)
# ---------------------------------------------------------------------------


def test_metodos_del_contrato(pre):
    metodos = atributo(pre, "METODOS")
    assert metodos == METODOS_CONTRATO and isinstance(metodos, tuple), (
        f"METODOS = {metodos!r}; se espera la tupla {METODOS_CONTRATO!r}."
    )


def test_huella_preprocesada_es_dataclass_congelada_con_los_campos_del_contrato(clase_huella):
    assert dataclasses.is_dataclass(clase_huella), "HuellaPreprocesada tiene que ser una dataclass."
    campos = {f.name for f in dataclasses.fields(clase_huella)}
    assert campos == set(CAMPOS_HUELLA), (
        f"Campos de HuellaPreprocesada: sobran {sorted(campos - set(CAMPOS_HUELLA))} y faltan "
        f"{sorted(set(CAMPOS_HUELLA) - campos)}."
    )
    assert clase_huella.__dataclass_params__.frozen, "HuellaPreprocesada tiene que ser @dataclass(frozen=True)."


@pytest.mark.parametrize("nombre", NOMBRES_52)
def test_preprocesar_devuelve_el_formato_del_contrato(bases_preprocesadas, claves_52, clase_huella, nombre):
    resultado = resultado_de_base(bases_preprocesadas, claves_52[nombre])
    problemas = problemas_de_formato(resultado, clase_huella)
    assert not problemas, f"Base {nombre}: la salida no tiene el formato de 5.2:\n{resumen(problemas)}"


@pytest.mark.parametrize("nombre", NOMBRES_52)
def test_preprocesar_no_modifica_la_imagen_de_entrada(cfg, preprocesar, bases_muestra, claves_52, nombre):
    entrada = np.array(base_52(bases_muestra, claves_52, nombre).imagen, copy=True)
    original = entrada.copy()
    preprocesar(entrada, cfg)
    cambiados = int(np.count_nonzero(entrada != original))
    assert cambiados == 0, f"Base {nombre}: preprocesar modificó {cambiados} píxeles de la imagen de entrada."


@pytest.mark.parametrize("nombre", NOMBRES_52)
def test_preprocesar_es_determinista(cfg, preprocesar, bases_muestra, bases_preprocesadas, claves_52, nombre):
    primera = resultado_de_base(bases_preprocesadas, claves_52[nombre])
    segunda = preprocesar(np.array(base_52(bases_muestra, claves_52, nombre).imagen, copy=True), cfg)
    distintos = [
        campo for campo in ("realzada", "binaria", "esqueleto", "mascara", "orientacion", "matriz")
        if not np.array_equal(getattr(primera, campo), getattr(segunda, campo))
    ]
    distintos += [
        f"{campo} ({getattr(primera, campo)!r} y {getattr(segunda, campo)!r})"
        for campo in ("punto_referencia", "nucleos", "angulo", "metodo")
        if not iguales(getattr(primera, campo), getattr(segunda, campo))
    ]
    assert not distintos, f"Base {nombre}: dos llamadas con la misma imagen difieren en: {', '.join(distintos)}."


@pytest.mark.parametrize("conversion", ["BGR", "BGRA"])
@pytest.mark.parametrize("nombre", NOMBRES_52)
def test_imagen_en_color_da_lo_mismo_que_la_gris(
    cfg, preprocesar, bases_muestra, bases_preprocesadas, claves_52, nombre, conversion
):
    gris = base_52(bases_muestra, claves_52, nombre).imagen
    codigo = cv2.COLOR_GRAY2BGR if conversion == "BGR" else cv2.COLOR_GRAY2BGRA
    color = preprocesar(cv2.cvtColor(gris, codigo), cfg)
    referencia = resultado_de_base(bases_preprocesadas, claves_52[nombre])
    d = distancia(color.punto_referencia, referencia.punto_referencia)
    da = diferencia_angulos(color.angulo, referencia.angulo)
    coincidencia = float(np.mean(np.asarray(color.mascara) == np.asarray(referencia.mascara)))
    problemas = []
    if d > TOL_COLOR_PX:
        problemas.append(
            f"punto de referencia a {d:.3f} px del de la gris (límite {TOL_COLOR_PX}): "
            f"{fmt(color.punto_referencia)} y {fmt(referencia.punto_referencia)}"
        )
    if abs(da) > TOL_COLOR_GRADOS:
        problemas.append(f"ángulo a {da:+.3f}° del de la gris (límite {TOL_COLOR_GRADOS}°)")
    if coincidencia < COINCIDENCIA_MASCARA_COLOR:
        problemas.append(f"mascara igual en el {coincidencia:.2%} de los píxeles (mínimo 99 %)")
    assert not problemas, f"Base {nombre} en {conversion}:\n{resumen(problemas)}"


@pytest.mark.parametrize("nombre", NOMBRES_52)
def test_huella_tenida_da_salidas_con_formato_y_no_vacias(
    cfg, preprocesar, bases_muestra, clase_huella, claves_52, nombre
):
    gris = base_52(bases_muestra, claves_52, nombre).imagen.astype(np.float64)
    azul = np.round(gris)
    verde = np.round(0.9 * gris)
    rojo = np.round(255 - 0.8 * (255 - gris))
    tenida = np.dstack([azul, verde, rojo]).clip(0, 255).astype(np.uint8)
    problemas = problemas_de_salida_no_vacia(preprocesar(tenida, cfg), clase_huella)
    assert not problemas, f"Base {nombre} teñida (BGR con canales distintos):\n{resumen(problemas)}"


@pytest.mark.parametrize("nombre", NOMBRES_52)
def test_lienzo_mas_grande_corre_el_punto_de_referencia(
    cfg, preprocesar, bases_muestra, bases_preprocesadas, claves_52, nombre
):
    gris = base_52(bases_muestra, claves_52, nombre).imagen
    lienzo = np.pad(gris, BORDE_LIENZO, mode="constant", constant_values=FONDO)
    referencia = resultado_de_base(bases_preprocesadas, claves_52[nombre])
    resultado = preprocesar(lienzo, cfg)
    esperado = (referencia.punto_referencia[0] + BORDE_LIENZO, referencia.punto_referencia[1] + BORDE_LIENZO)
    d = distancia(resultado.punto_referencia, esperado)
    da = diferencia_angulos(resultado.angulo, referencia.angulo)
    problemas = []
    if d > TOL_LIENZO_PX:
        problemas.append(
            f"punto de referencia {fmt(resultado.punto_referencia)}; se espera {fmt(esperado)} ± {TOL_LIENZO_PX} px "
            f"(está a {d:.2f} px)"
        )
    if abs(da) > TOL_LIENZO_GRADOS:
        problemas.append(f"ángulo a {da:+.2f}° del de la base (límite {TOL_LIENZO_GRADOS}°)")
    assert not problemas, f"Base {nombre} con un borde de {BORDE_LIENZO} px ({lienzo.shape}):\n{resumen(problemas)}"


@pytest.mark.parametrize("nombre", NOMBRES_52)
def test_recorte_corre_el_punto_de_referencia(cfg, preprocesar, bases_muestra, bases_preprocesadas, claves_52, nombre):
    gris = base_52(bases_muestra, claves_52, nombre).imagen
    referencia = resultado_de_base(bases_preprocesadas, claves_52[nombre])
    alto_r, ancho_r = RECORTE_GRANDE
    origen = origen_recorte(referencia.punto_referencia, alto_r, ancho_r, *gris.shape)
    resultado = preprocesar(recortar(gris, origen, alto_r, ancho_r), cfg)
    esperado = (referencia.punto_referencia[0] - origen[0], referencia.punto_referencia[1] - origen[1])
    d = distancia(resultado.punto_referencia, esperado)
    da = diferencia_angulos(resultado.angulo, referencia.angulo)
    problemas = []
    if d > TOL_RECORTE_PX:
        problemas.append(
            f"punto de referencia {fmt(resultado.punto_referencia)}; se espera {fmt(esperado)} ± {TOL_RECORTE_PX} px "
            f"(está a {d:.2f} px)"
        )
    if abs(da) > TOL_LIENZO_GRADOS:
        problemas.append(f"ángulo a {da:+.2f}° del de la base (límite {TOL_LIENZO_GRADOS}°)")
    assert not problemas, (
        f"Base {nombre}, recorte de {alto_r} × {ancho_r} con origen {origen}:\n{resumen(problemas)}"
    )


@pytest.mark.parametrize("nombre", NOMBRES_52)
def test_lienzo_mas_chico_que_la_ventana(
    cfg, preprocesar, bases_muestra, bases_preprocesadas, clase_huella, claves_52, nombre
):
    gris = base_52(bases_muestra, claves_52, nombre).imagen
    referencia = resultado_de_base(bases_preprocesadas, claves_52[nombre])
    alto_r, ancho_r = RECORTE_CHICO
    origen = origen_recorte(referencia.punto_referencia, alto_r, ancho_r, *gris.shape)
    recorte = recortar(gris, origen, alto_r, ancho_r)
    resultado = preprocesar(recorte, cfg)
    problemas = problemas_de_salida_no_vacia(resultado, clase_huella)
    if not problemas:
        afuera = int(np.count_nonzero(resultado.mascara & ~dentro_de_la_entrada(recorte.shape, resultado.matriz)))
        if afuera:
            problemas.append(f"mascara vale True en {afuera} píxeles que no vienen de la entrada")
        esperado = (referencia.punto_referencia[0] - origen[0], referencia.punto_referencia[1] - origen[1])
        d = distancia(resultado.punto_referencia, esperado)
        if d > TOL_CHICO_PX:
            problemas.append(
                f"punto de referencia {fmt(resultado.punto_referencia)}; se espera {fmt(esperado)} ± {TOL_CHICO_PX} px "
                f"(está a {d:.2f} px)"
            )
    assert not problemas, (
        f"Base {nombre}, recorte de {alto_r} × {ancho_r} con origen {origen}:\n{resumen(problemas)}"
    )


def _entradas_invalidas(gris: np.ndarray) -> dict:
    alto, ancho = gris.shape
    fila = alto // 2 - 25
    columna = (ancho - 300) // 2
    return {
        "uniforme 0": np.zeros((alto, ancho), dtype=np.uint8),
        "uniforme 128": np.full((alto, ancho), 128, dtype=np.uint8),
        "uniforme 245": np.full((alto, ancho), 245, dtype=np.uint8),
        "float64": gris.astype(np.float64),
        "1 dimensión": gris.ravel().copy(),
        "4 dimensiones": gris.reshape(1, alto, ancho, 1).copy(),
        "2 canales": np.dstack([gris, gris]),
        "50 × 300": np.ascontiguousarray(gris[fila:fila + 50, columna:columna + 300]),
    }


ENTRADAS_INVALIDAS = (
    "uniforme 0", "uniforme 128", "uniforme 245", "float64",
    "1 dimensión", "4 dimensiones", "2 canales", "50 × 300",
)


@pytest.mark.parametrize("caso", ENTRADAS_INVALIDAS)
def test_entradas_invalidas_lanzan_value_error(cfg, preprocesar, bases_muestra, claves_52, caso):
    entrada = _entradas_invalidas(base_52(bases_muestra, claves_52, "presilla").imagen)[caso]
    lanza(
        ValueError, preprocesar, entrada, cfg,
        que=f"preprocesar con una entrada {caso} (dtype {entrada.dtype}, forma {entrada.shape})",
    )


# ---------------------------------------------------------------------------
# 5.3 · Ventana y matriz (las 156 variantes de la muestra)
# ---------------------------------------------------------------------------


def chequeo_matriz(caso):
    r = caso.resultado
    m = como_matriz(r.matriz)
    a = giro_de_la_ventana(r.angulo)
    rotacion = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    dif_rotacion = float(np.max(np.abs(m[:, :2] - rotacion)))
    determinante = float(np.linalg.det(m[:, :2]))
    centro = aplicar_matriz(m, r.punto_referencia)
    dif_centro = max(abs(centro[0] - CENTRO_W[0]), abs(centro[1] - CENTRO_W[1]))
    ok = dif_rotacion <= TOL_ROTACION and abs(determinante - 1) <= TOL_ROTACION and dif_centro <= TOL_CENTRO
    return ok, (
        f"parte 2 × 2 a {dif_rotacion:.2e} de la rotación de a_v = {math.degrees(a):+.2f}° "
        f"(angulo = {math.degrees(float(r.angulo)):.2f}°; tolerancia {TOL_ROTACION:g}), "
        f"determinante {determinante:.12f}, matriz·punto_referencia = {fmt(centro)} "
        f"(a {dif_centro:.2e} de {CENTRO_W}, tolerancia {TOL_CENTRO:g})"
    )


def chequeo_deltas_abajo(caso):
    """(1.2) Cada delta real, llevado con matriz, queda debajo del centro de la ventana."""
    r = caso.resultado
    llevados = [aplicar_matriz(r.matriz, d) for d in caso.deltas_reales]
    arriba = [
        f"delta real {fmt(d)} → {fmt(w)} en la ventana"
        for d, w in zip(caso.deltas_reales, llevados)
        if not w[1] > CENTRO_W[1]
    ]
    return not arriba, (
        "; ".join(arriba) + f" (se espera y_w > {CENTRO_W[1]}; angulo = {math.degrees(float(r.angulo)):.2f}°, "
        f"a_v = {math.degrees(giro_de_la_ventana(r.angulo)):+.2f}°, rot_grados = {caso.rot_grados:+.2f})"
    )


def chequeo_metodo_y_nucleos(caso):
    r = caso.resultado
    nucleos = [como_punto(n) for n in r.nucleos]
    ref = como_punto(r.punto_referencia)
    if r.metodo == "nucleo":
        ok = len(nucleos) == 1 and math.dist(nucleos[0], ref) <= TOL_NUCLEOS
    elif r.metodo == "nucleos":
        ok = len(nucleos) == 2 and math.dist(
            ((nucleos[0][0] + nucleos[1][0]) / 2, (nucleos[0][1] + nucleos[1][1]) / 2), ref
        ) <= TOL_NUCLEOS
    elif r.metodo == "curvatura":
        ok = len(nucleos) == 0
    else:
        return False, f"metodo {r.metodo!r} no está en {METODOS_CONTRATO}"
    return ok, f"metodo {r.metodo!r}, nucleos {[fmt(n) for n in nucleos]}, punto_referencia {fmt(ref)}"


def chequeo_centro_en_la_huella(caso):
    centro = np.asarray(caso.resultado.mascara)[124:126, 124:126]
    return bool(np.all(centro)), f"mascara[124:126, 124:126] = {centro.tolist()}"


def chequeo_iou(caso):
    v = verdad_de(caso)
    mascara = np.asarray(caso.resultado.mascara, dtype=bool)
    valor = iou(mascara & v.dentro, v.mascara_real & v.dentro)
    return valor >= IOU_MIN, f"IoU {valor:.3f} (mínimo {IOU_MIN})"


@pytest.mark.parametrize("tipo", TIPOS)
def test_matriz_es_rotacion_por_el_giro_chico_y_lleva_el_punto_al_centro(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_matriz), f"5.3 · matriz de la ventana ({tipo})")


@pytest.mark.parametrize("tipo", TIPOS_CON_NUCLEO)
def test_ventana_derecha_con_los_deltas_debajo_del_centro(muestra, tipo):
    exigir_todas(
        revisar(de_tipo(muestra, tipo), chequeo_deltas_abajo),
        f"5.3 (1.2) · deltas reales debajo del centro de la ventana ({tipo})",
    )


@pytest.mark.parametrize("tipo", TIPOS)
def test_metodo_coherente_con_nucleos_y_punto_de_referencia(muestra, tipo):
    exigir_todas(
        revisar(de_tipo(muestra, tipo), chequeo_metodo_y_nucleos), f"5.3 · metodo, nucleos y punto_referencia ({tipo})"
    )


@pytest.mark.parametrize("tipo", TIPOS)
def test_centro_de_la_ventana_dentro_de_la_huella(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_centro_en_la_huella), f"5.3 · centro de la ventana ({tipo})")


@pytest.mark.parametrize("tipo", TIPOS)
def test_mascara_coincide_con_la_real_en_la_ventana(muestra, tipo):
    exigir_todas(
        revisar(de_tipo(muestra, tipo), chequeo_iou), f"5.3 · IoU de mascara con la máscara real, dentro de la entrada ({tipo})"
    )


# ---------------------------------------------------------------------------
# 5.4 · Punto de referencia y núcleos (muestra)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tipo", TIPOS)
def test_metodo_segun_el_tipo(muestra, tipo):
    exigir_proporcion(
        revisar(de_tipo(muestra, tipo), chequeo_metodo(tipo)),
        f"5.4 · metodo {METODO_POR_TIPO[tipo]!r} en las variantes de {tipo}",
    )


@pytest.mark.parametrize("tipo", TIPOS_CON_NUCLEO)
def test_punto_de_referencia_a_8_px_del_real_en_el_95(muestra, tipo):
    exigir_proporcion(
        revisar(de_tipo(muestra, tipo), chequeo_referencia(REF_95_PX)),
        f"5.4 · punto de referencia a <= {REF_95_PX:g} px del real ({tipo})",
    )


@pytest.mark.parametrize("tipo", TIPOS_CON_NUCLEO)
def test_punto_de_referencia_a_16_px_del_real_en_todas(muestra, tipo):
    exigir_todas(
        revisar(de_tipo(muestra, tipo), chequeo_referencia(REF_MAX_PX)),
        f"5.4 · punto de referencia a <= {REF_MAX_PX:g} px del real ({tipo})",
    )


@pytest.mark.parametrize("tipo", TIPOS_CON_NUCLEO)
def test_cada_nucleo_real_tiene_un_nucleo_detectado_cerca(muestra, tipo):
    exigir_proporcion(
        revisar(de_tipo(muestra, tipo), chequeo_nucleos(NUCLEO_95_PX)),
        f"5.4 · cada núcleo real con un punto de nucleos a <= {NUCLEO_95_PX:g} px ({tipo})",
    )


def test_arcos_y_sus_bases_dan_curvatura_sin_nucleos(muestra, bases_preprocesadas):
    casos = de_tipo(muestra, "arco") + [c for clave, c in bases_preprocesadas.items() if clave[0] == "arco"]
    exigir_todas(revisar(casos, chequeo_curvatura), "5.4 · arcos (9 variantes y 3 bases) con metodo 'curvatura'")


# ---------------------------------------------------------------------------
# 5.5 · Ángulo (muestra)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tipo", TIPOS)
def test_error_de_angulo_a_5_grados_en_el_95(muestra, tipo):
    exigir_proporcion(
        revisar(de_tipo(muestra, tipo), chequeo_angulo(ANGULO_95_GRADOS)),
        f"5.5 · |error de ángulo| <= {ANGULO_95_GRADOS:g}° ({tipo})",
    )


@pytest.mark.parametrize("tipo", TIPOS)
def test_error_de_angulo_a_10_grados_en_todas(muestra, tipo):
    exigir_todas(
        revisar(de_tipo(muestra, tipo), chequeo_angulo(ANGULO_MAX_GRADOS)),
        f"5.5 · |error de ángulo| <= {ANGULO_MAX_GRADOS:g}° ({tipo})",
    )


def test_error_de_angulo_con_corte_a_5_grados_en_el_95(muestra):
    con_corte = [c for c in muestra if c.corte_fraccion > 0]
    exigir_proporcion(
        revisar(con_corte, chequeo_angulo(ANGULO_95_GRADOS)),
        f"5.5 · |error de ángulo| <= {ANGULO_95_GRADOS:g}° en las {len(con_corte)} variantes con corte",
    )


@pytest.mark.parametrize("tipo", TIPOS_CON_NUCLEO)
def test_angulo_de_la_base_de_cada_identidad(bases_preprocesadas, claves_52, tipo):
    exigir_todas(
        revisar([bases_preprocesadas[claves_52[tipo]]], chequeo_angulo(ANGULO_BASES_GRADOS)),
        f"5.5 · |angulo| <= {ANGULO_BASES_GRADOS:g}° en la base de la identidad ({tipo})",
    )


# ---------------------------------------------------------------------------
# 5.6 · Campo, crestas, esqueleto y realce (muestra, en cada variante)
# ---------------------------------------------------------------------------


def chequeo_campo(caso):
    v = verdad_de(caso)
    zona = zona_de_comparacion(caso.resultado.mascara, v)
    if not np.any(zona):
        return False, "la zona de comparación está vacía"
    orientacion = np.asarray(caso.resultado.orientacion, dtype=np.float64)
    media = float(np.mean(np.cos(2 * (orientacion[zona] - v.campo[zona].astype(np.float64)))))
    return media >= CAMPO_MIN, f"media de cos 2Δθ {media:.4f} (mínimo {CAMPO_MIN}) en {int(zona.sum())} píxeles"


def chequeo_crestas(caso):
    v = verdad_de(caso)
    zona = zona_de_comparacion(caso.resultado.mascara, v)
    if not np.any(zona):
        return False, "la zona de comparación está vacía"
    acuerdo = float(np.mean(np.asarray(caso.resultado.binaria, dtype=bool)[zona] == v.crestas[zona]))
    return acuerdo >= CRESTAS_MIN, f"binaria coincide en el {acuerdo:.2%} de la zona (mínimo {CRESTAS_MIN:.0%})"


def chequeo_esqueleto_contenido(caso):
    esqueleto = np.asarray(caso.resultado.esqueleto, dtype=bool)
    fuera_binaria = int(np.count_nonzero(esqueleto & ~np.asarray(caso.resultado.binaria, dtype=bool)))
    fuera_mascara = int(np.count_nonzero(esqueleto & ~np.asarray(caso.resultado.mascara, dtype=bool)))
    return fuera_binaria == 0 and fuera_mascara == 0, (
        f"{fuera_binaria} píxeles del esqueleto fuera de binaria y {fuera_mascara} fuera de mascara"
    )


def chequeo_esqueleto_sin_bloques(caso):
    bloques = bloques_2x2(caso.resultado.esqueleto)
    primeros = ", ".join(f"(i, j) = ({i}, {j})" for i, j in bloques[:3])
    return len(bloques) == 0, f"{len(bloques)} bloques de 2 × 2 en el esqueleto ({primeros})"


def chequeo_densidad_esqueleto(caso):
    total = int(np.count_nonzero(caso.resultado.mascara))
    if total == 0:
        return False, "mascara vacía"
    densidad = int(np.count_nonzero(caso.resultado.esqueleto)) / total
    minimo, maximo = DENSIDAD_ESQUELETO
    return minimo <= densidad <= maximo, f"esqueleto.sum() / mascara.sum() = {densidad:.4f} (rango [{minimo}, {maximo}])"


def chequeo_realce_fondo(caso):
    realzada = np.asarray(caso.resultado.realzada)
    fuera = realzada[~np.asarray(caso.resultado.mascara, dtype=bool)]
    distintos = int(np.count_nonzero(fuera != 255))
    minimo = int(fuera.min()) if fuera.size else 255
    return distintos == 0, f"{distintos} píxeles fuera de mascara distintos de 255 (mínimo {minimo})"


def chequeo_realce_contraste(caso):
    realzada = np.asarray(caso.resultado.realzada, dtype=np.float64)
    mascara = np.asarray(caso.resultado.mascara, dtype=bool)
    binaria = np.asarray(caso.resultado.binaria, dtype=bool)
    valles, crestas = realzada[mascara & ~binaria], realzada[mascara & binaria]
    if valles.size == 0 or crestas.size == 0:
        return False, f"{valles.size} píxeles de valle y {crestas.size} de cresta dentro de mascara"
    diferencia = float(valles.mean() - crestas.mean())
    return diferencia >= CONTRASTE_MIN, (
        f"promedio en ~binaria {valles.mean():.1f} y en binaria {crestas.mean():.1f}: diferencia "
        f"{diferencia:.1f} (mínimo {CONTRASTE_MIN:g})"
    )


def chequeo_realce_periodo(caso):
    periodo = periodo_medido(np.asarray(caso.resultado.realzada))
    dif = abs(periodo - caso.periodo_real)
    return dif <= TOL_PERIODO_PX, (
        f"período medido {periodo:.2f} px; real {caso.periodo_real:.2f} px (periodo_px · escala), "
        f"diferencia {dif:.2f} (límite {TOL_PERIODO_PX:g})"
    )


@pytest.mark.parametrize("tipo", TIPOS)
def test_campo_de_orientacion_coincide_con_el_verdadero(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_campo), f"5.6 · campo en la zona de comparación ({tipo})")


@pytest.mark.parametrize("tipo", TIPOS)
def test_crestas_binarizadas_coinciden_con_las_verdaderas(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_crestas), f"5.6 · crestas en la zona de comparación ({tipo})")


@pytest.mark.parametrize("tipo", TIPOS)
def test_esqueleto_contenido_en_binaria_y_en_mascara(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_esqueleto_contenido), f"5.6 · esqueleto ⊆ binaria ({tipo})")


@pytest.mark.parametrize("tipo", TIPOS)
def test_esqueleto_sin_bloques_de_2x2(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_esqueleto_sin_bloques), f"5.6 · esqueleto de 1 px ({tipo})")


@pytest.mark.parametrize("tipo", TIPOS)
def test_densidad_del_esqueleto(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_densidad_esqueleto), f"5.6 · densidad del esqueleto ({tipo})")


@pytest.mark.parametrize("tipo", TIPOS)
def test_realzada_vale_255_fuera_de_la_mascara(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_realce_fondo), f"5.6 · realzada fuera de mascara ({tipo})")


@pytest.mark.parametrize("tipo", TIPOS)
def test_realzada_con_contraste_entre_crestas_y_valles(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_realce_contraste), f"5.6 · contraste de realzada ({tipo})")


@pytest.mark.parametrize("tipo", TIPOS)
def test_periodo_de_la_realzada_igual_al_real(muestra, tipo):
    exigir_todas(revisar(de_tipo(muestra, tipo), chequeo_realce_periodo), f"5.6 · período de realzada ({tipo})")


# ---------------------------------------------------------------------------
# 5.7 · Estabilidad con corte (bases de las 3 identidades)
# ---------------------------------------------------------------------------

CASOS_ESTABILIDAD = [(tipo, a) for tipo in TIPOS_CON_NUCLEO for a in ANGULOS_CORTE_ESTABILIDAD]
IDS_ESTABILIDAD = [f"{tipo}-corte{a}" for tipo, a in CASOS_ESTABILIDAD]


@pytest.fixture(scope="module")
def estabilidad(cfg, h1, preprocesar, bases_muestra, claves_52) -> dict:
    """Variante (12°, escala 1, (10, −8), corte máximo hacia a) de cada identidad, preprocesada."""
    fraccion = float(max(obtener(cfg, "huellas.variantes.corte_periferico")))
    casos = {}
    for tipo, a in CASOS_ESTABILIDAD:
        clave = claves_52[tipo]
        base = bases_muestra[clave]
        params = h1.ParametrosVariante(12.0, 1.0, 10.0, -8.0, 0.0, 0.0, fraccion, float(a), 0.0, 0.0)
        variante = h1.aplicar_variante(base, params, np.random.default_rng(0))
        entrada = {"archivo": f"{tipo} con corte {fraccion:g} hacia {a}°", "tipo": tipo, "semilla": clave[1]}
        caso = preprocesar_caso(caso_de_variante(entrada, base, variante), preprocesar, variante.imagen, cfg)
        caso.archivo += (
            f" (corte aplicado {variante.parametros.corte_fraccion:g} hacia "
            f"{variante.parametros.corte_angulo_grados:g}°)"
        )
        casos[(tipo, a)] = caso
    return casos


def _caso_estabilidad(estabilidad, tipo, a) -> Caso:
    caso = estabilidad[(tipo, a)]
    if caso.error is not None:
        pytest.fail(f"5.7 · {caso.archivo}: {caso.error}", pytrace=False)
    return caso


@pytest.mark.parametrize(("tipo", "a"), CASOS_ESTABILIDAD, ids=IDS_ESTABILIDAD)
def test_estabilidad_del_campo_con_corte(estabilidad, bases_preprocesadas, claves_52, tipo, a):
    caso = _caso_estabilidad(estabilidad, tipo, a)
    base = resultado_de_base(bases_preprocesadas, claves_52[tipo])
    zona = (
        disco_en_la_ventana(RADIO_ESTABILIDAD)
        & np.asarray(caso.resultado.mascara, dtype=bool)
        & np.asarray(base.mascara, dtype=bool)
    )
    assert np.any(zona), f"5.7 · {caso.archivo}: el disco de {RADIO_ESTABILIDAD:g} px ∩ las dos máscaras está vacío."
    dif = np.asarray(caso.resultado.orientacion, dtype=np.float64) - np.asarray(base.orientacion, dtype=np.float64)
    media = float(np.mean(np.cos(2 * dif[zona])))
    assert media >= CAMPO_ESTABILIDAD_MIN, (
        f"5.7 · {caso.archivo}: media de cos 2(orientacion_variante − orientacion_base) = {media:.4f} en el disco "
        f"de {RADIO_ESTABILIDAD:g} px ({int(zona.sum())} píxeles); mínimo {CAMPO_ESTABILIDAD_MIN}."
    )


@pytest.mark.parametrize(("tipo", "a"), CASOS_ESTABILIDAD, ids=IDS_ESTABILIDAD)
def test_estabilidad_del_angulo_con_corte(estabilidad, tipo, a):
    caso = _caso_estabilidad(estabilidad, tipo, a)
    ok, detalle = chequeo_angulo(ANGULO_ESTABILIDAD_GRADOS)(caso)
    assert ok, f"5.7 · {caso.archivo}: {detalle}"


@pytest.mark.parametrize(("tipo", "a"), CASOS_ESTABILIDAD, ids=IDS_ESTABILIDAD)
def test_estabilidad_del_punto_de_referencia_con_corte(estabilidad, tipo, a):
    caso = _caso_estabilidad(estabilidad, tipo, a)
    ok, detalle = chequeo_referencia(REF_ESTABILIDAD_PX)(caso)
    assert ok, f"5.7 · {caso.archivo}: {detalle}"


# ---------------------------------------------------------------------------
# 5.8 · Lámina
# ---------------------------------------------------------------------------


def test_generar_lamina_escribe_un_png_legible_en_salida(cfg, pre, tmp_path):
    generar_lamina = atributo(pre, "generar_lamina")
    salida = tmp_path / "lamina_h2.png"
    resultado = generar_lamina(cfg, salida)
    assert Path(resultado).resolve() == salida.resolve(), (
        f"generar_lamina devolvió {resultado!r}; se espera la ruta de salida {salida}."
    )
    assert salida.is_file(), f"generar_lamina no escribió {salida}."
    contenido = salida.read_bytes()
    assert contenido[:8] == b"\x89PNG\r\n\x1a\n", f"{salida} no es un PNG (empieza con {contenido[:8]!r})."
    # Decodifica igual que cv2.imread(ruta) y no falla con tildes en la ruta (Windows).
    imagen = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert imagen is not None and imagen.size > 0, f"OpenCV no pudo leer {salida}."


# ---------------------------------------------------------------------------
# 5.9 · Aceptación (las 300 imágenes del dataset, en memoria)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_completo(cfg, h1, plan, preprocesar):
    """Las 300 imágenes con la receta de H1, preprocesadas (solo se guardan los valores que se miden)."""
    bases, casos, bases_resumidas = {}, [], {}
    for entrada in plan:
        clave = (entrada["tipo"], entrada["semilla"])
        if clave not in bases:
            bases[clave] = h1.generar_base(entrada["tipo"], entrada["semilla"], cfg)
        base = bases[clave]
        variante = receta_h1(h1, cfg, entrada, base)
        caso = preprocesar_caso(caso_de_variante(entrada, base, variante), preprocesar, variante.imagen, cfg)
        if caso.error is None:
            try:
                caso.resultado = resumir(caso.resultado)
            except Exception as error:
                caso.error = f"la salida de preprocesar no tiene los campos del contrato: {descripcion_error(error)}"
        casos.append(caso)
    tiempo = sum(c.tiempo_s for c in casos)
    identidades = {(e["tipo"], e["semilla"]) for e in plan if e["identidad"] != 0}
    for clave, base in bases.items():  # sin contar el tiempo: bases de arco y de identidades
        if clave[0] == "arco" or clave in identidades:
            caso = preprocesar_caso(caso_de_base(clave, base), preprocesar, base.imagen, cfg)
            if caso.error is None:
                caso.resultado = resumir(caso.resultado)
            bases_resumidas[clave] = caso
    return SimpleNamespace(casos=casos, bases=bases_resumidas, identidades=identidades, tiempo_s=tiempo)


@pytest.mark.aceptacion
def test_aceptacion_son_las_300_imagenes_del_plan(dataset_completo, plan):
    assert len(dataset_completo.casos) == len(plan) == 300, (
        f"El dataset en memoria tiene {len(dataset_completo.casos)} imágenes y el plan {len(plan)}; se esperan 300."
    )


@pytest.mark.aceptacion
@pytest.mark.parametrize("tipo", TIPOS)
def test_aceptacion_metodo_segun_el_tipo(dataset_completo, tipo):
    exigir_proporcion(
        revisar(de_tipo(dataset_completo.casos, tipo), chequeo_metodo(tipo)),
        f"5.9 · metodo {METODO_POR_TIPO[tipo]!r} en las variantes de {tipo}",
    )


@pytest.mark.aceptacion
@pytest.mark.parametrize("tipo", TIPOS_CON_NUCLEO)
def test_aceptacion_punto_de_referencia_a_8_px_en_el_95(dataset_completo, tipo):
    exigir_proporcion(
        revisar(de_tipo(dataset_completo.casos, tipo), chequeo_referencia(REF_95_PX)),
        f"5.9 · punto de referencia a <= {REF_95_PX:g} px del real ({tipo})",
    )


@pytest.mark.aceptacion
@pytest.mark.parametrize("tipo", TIPOS_CON_NUCLEO)
def test_aceptacion_punto_de_referencia_a_16_px_en_todas(dataset_completo, tipo):
    exigir_todas(
        revisar(de_tipo(dataset_completo.casos, tipo), chequeo_referencia(REF_MAX_PX)),
        f"5.9 · punto de referencia a <= {REF_MAX_PX:g} px del real ({tipo})",
    )


@pytest.mark.aceptacion
@pytest.mark.parametrize("tipo", TIPOS_CON_NUCLEO)
def test_aceptacion_cada_nucleo_real_tiene_un_nucleo_cerca(dataset_completo, tipo):
    exigir_proporcion(
        revisar(de_tipo(dataset_completo.casos, tipo), chequeo_nucleos(NUCLEO_95_PX)),
        f"5.9 · cada núcleo real con un punto de nucleos a <= {NUCLEO_95_PX:g} px ({tipo})",
    )


@pytest.mark.aceptacion
def test_aceptacion_arcos_y_sus_bases_dan_curvatura(dataset_completo):
    casos = de_tipo(dataset_completo.casos, "arco") + [
        c for clave, c in dataset_completo.bases.items() if clave[0] == "arco"
    ]
    exigir_todas(revisar(casos, chequeo_curvatura), "5.9 · arcos (45 variantes y 15 bases) con metodo 'curvatura'")


@pytest.mark.aceptacion
@pytest.mark.parametrize("tipo", TIPOS)
def test_aceptacion_error_de_angulo_a_5_grados_en_el_95(dataset_completo, tipo):
    exigir_proporcion(
        revisar(de_tipo(dataset_completo.casos, tipo), chequeo_angulo(ANGULO_95_GRADOS)),
        f"5.9 · |error de ángulo| <= {ANGULO_95_GRADOS:g}° ({tipo})",
    )


@pytest.mark.aceptacion
@pytest.mark.parametrize("tipo", TIPOS)
def test_aceptacion_error_de_angulo_a_10_grados_en_todas(dataset_completo, tipo):
    exigir_todas(
        revisar(de_tipo(dataset_completo.casos, tipo), chequeo_angulo(ANGULO_MAX_GRADOS)),
        f"5.9 · |error de ángulo| <= {ANGULO_MAX_GRADOS:g}° ({tipo})",
    )


@pytest.mark.aceptacion
def test_aceptacion_error_de_angulo_con_corte(dataset_completo):
    con_corte = [c for c in dataset_completo.casos if c.corte_fraccion > 0]
    exigir_proporcion(
        revisar(con_corte, chequeo_angulo(ANGULO_95_GRADOS)),
        f"5.9 · |error de ángulo| <= {ANGULO_95_GRADOS:g}° en las {len(con_corte)} variantes con corte",
    )


@pytest.mark.aceptacion
@pytest.mark.parametrize("tipo", TIPOS_CON_NUCLEO)
def test_aceptacion_angulo_de_la_base_de_cada_identidad(dataset_completo, tipo):
    casos = [c for clave, c in dataset_completo.bases.items() if clave in dataset_completo.identidades and clave[0] == tipo]
    exigir_todas(
        revisar(casos, chequeo_angulo(ANGULO_BASES_GRADOS)),
        f"5.9 · |angulo| <= {ANGULO_BASES_GRADOS:g}° en la base de la identidad ({tipo})",
    )


@pytest.mark.aceptacion
def test_aceptacion_punto_de_referencia_del_arco_sigue_a_su_base(dataset_completo):
    referencias = {}
    for clave, caso in dataset_completo.bases.items():
        if clave[0] == "arco":
            referencias[clave] = caso.error if caso.error is not None else caso.resultado.punto_referencia
    exigir_proporcion(
        revisar(de_tipo(dataset_completo.casos, "arco"), chequeo_arco_sigue_a_su_base(referencias)),
        f"5.9 · punto de referencia del arco a <= {ARCO_ACEPTACION_PX:g} px del de su base llevado con la fórmula afín",
        minimo=ARCO_ACEPTACION_PROPORCION,
    )


@pytest.mark.aceptacion
def test_aceptacion_tiempo_de_preprocesar_las_300(dataset_completo):
    n = len(dataset_completo.casos)
    total = dataset_completo.tiempo_s
    assert total < TIEMPO_ACEPTACION_S, (
        f"preprocesar de las {n} imágenes tardó {total:.1f} s ({total / max(n, 1):.3f} s por imagen); "
        f"se exige menos de {TIEMPO_ACEPTACION_S:g} s."
    )
