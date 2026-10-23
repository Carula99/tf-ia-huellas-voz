"""Tests de H3 · características de huellas por bloques, escalador, PCA propio y silhouette propia.

Verifican la sección 6 de docs/contratos/h3_caracteristicas.md (versión 1, 22/10/2026).
La sección 7 es manual y se anota en el Documento 2.

Las huellas salen de H1 y H2, que ya están probados: la receta de la sección 4 de H1 y
preprocesar de H2. De H1 se usan solo generar_base, ParametrosVariante, sortear_parametros,
aplicar_variante, plan_dataset, generar_dataset y los campos imagen y parametros de Variante;
de H2, preprocesar y HuellaPreprocesada.

Los oráculos de la sección 4 (bloques de orientación, Gabor y minucias, y extraer_minucias)
están escritos acá, solo a partir del contrato, sin importar nada de
src/huellas/caracteristicas.py. Sobre la muestra de la sección 5 dan los valores del
prototipo de la sección 8 (silhouette 0,672 en train y 0,700 en val; 0,865, 0,832 y 0,512
por bloque en val; contribuciones 1,851, 1,990 y 1,852; densidades 1,027, 1,059 y 1,033;
distancias 1,250, 0,980 y 1,085; corte directo con la peor a 0,831; d = 374).

Lecturas del contrato:
- 6.4: el oráculo de Gabor se compara en las 120 variantes de la muestra (el contrato pide al
  menos 6; con todas tarda unos segundos más).
- 6.6 y 6.9: "las mismas proyecciones que scikit-learn" se prueba con n_componentes=None
  (k = min(n − 1, d) = 71) y con n_componentes=3 (el de la figura), contra
  sklearn.decomposition.PCA(n_components=k, svd_solver="full").
- 6.7: la densidad y la distancia al centroide se promedian, por identidad, sobre sus 40
  variantes (train, val y test). Con esa lectura salen los valores de la sección 8.
- Los umbrales de silhouette (6.6 y 6.9) se miden con sklearn.metrics.silhouette_score; la
  silhouette propia se compara aparte con la de scikit-learn.
- Las tolerancias del contrato son absolutas salvo donde dice "relativa". "Es igual"
  sin tolerancia en el contrato (6.2: ajustar_transformar, transformar de un vector y el
  determinismo; 6.5: transformar de un vector y fila por fila) se compara a 1e-12;
  "np.array_equal" (6.4 y 6.8) se compara exacto.
- 6.8: los PNG se escriben con cv2.imencode + tofile (los mismos bytes que cv2.imwrite, sin
  problemas con tildes en la ruta) y se leen con cv2.imdecode(np.fromfile(...)), como dice
  4.5. Las 2 de val de cada identidad son las dos primeras de val del plan (25 y 26); la
  carpeta sin fuga tiene, con esos nombres, las variantes 27 y 28, y los mismos intrusos.
  "Con sobrescribir=True lo vuelve a escribir": se reemplaza el npz por bytes que no son un
  npz y tiene que volver a ser el mismo npz.
- 6.2 · "todas las filas iguales": 8 filas iguales con valores exactos en binario, así la media es
  exactamente la fila y traza(Σ) = 0 sin depender del redondeo (con 10 copias de una fila
  cualquiera, la media puede diferir en 1 ulp y la traza queda en ~1e-30, no en 0).
- No se prueban los FileNotFoundError de 4.5 porque no están en la sección 6.

src.config y los módulos de src se importan en fixtures: si algo todavía no existe, fallan
(o dan error) esos tests y no la colección entera. La muestra (3 bases y 120 variantes), sus
salidas de H2, los oráculos, los bloques y el escalador se calculan una sola vez (fixtures de
módulo).
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import functools
import importlib
import math
import re
import shutil
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from scipy import ndimage
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA as PCA_SKLEARN
from sklearn.metrics import silhouette_score

RAIZ = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Valores del contrato
# ---------------------------------------------------------------------------

BLOQUES_CONTRATO = ("orientacion", "gabor", "minucias")

# Sección 0: ventana de 250 × 250 con centro c_w = (124.5, 124.5)
ALTO_W, ANCHO_W = 250, 250
VENTANA = (ALTO_W, ANCHO_W)
C_W = 124.5

# Sección 1. camino: ("entero", mín, máx) | ("float", mín, máx) |
# ("lista", largo mín, largo máx, mín, máx) | ("fijo", valor); todos los rangos son cerrados.
SECCION_H3 = "huellas.caracteristicas"
CLAVES_H3 = {
    "huellas.caracteristicas.grilla_orientacion": ("entero", 3, 8),
    "huellas.caracteristicas.cobertura_min": ("fijo", 0.5),
    "huellas.caracteristicas.gabor_periodos_px": ("lista", 1, 3, 6.0, 12.0),
    "huellas.caracteristicas.gabor_orientaciones": ("entero", 4, 12),
    "huellas.caracteristicas.anillos_gabor_px": ("lista", 2, 6, 0.0, 124.5),
    "huellas.caracteristicas.sectores_gabor": ("entero", 4, 16),
    "huellas.caracteristicas.erosion_gabor_px": ("float", 0.0, 20.0),
    "huellas.caracteristicas.borde_minucia_px": ("fijo", 14.0),
    "huellas.caracteristicas.componente_min_px": ("entero", 0, 50),
    "huellas.caracteristicas.espuela_px": ("entero", 0, 20),
    "huellas.caracteristicas.par_terminaciones_px": ("float", 0.0, 20.0),
    "huellas.caracteristicas.par_bifurcaciones_px": ("float", 0.0, 20.0),
    "huellas.caracteristicas.anillos_minucias_px": ("lista", 2, 6, 0.0, 124.5),
    "huellas.caracteristicas.grilla_minucias": ("entero", 2, 6),
}

# Sección 6.4: largos con el cfg real
LARGOS_REALES = {"orientacion": 50, "gabor": 300, "minucias": 25}

# Sección 5: la muestra
IDENTIDADES = (1, 2, 3)
POR_PARTICION = {"train": 24, "val": 8, "test": 8}
CORTES_POR_IDENTIDAD = {1: 11, 2: 9, 3: 10}

# Tolerancias
TOL_ORACULO = 1e-9          # 6.4: orientación y minucias
TOL_GABOR_REL = 1e-6        # 6.4: Gabor (relativa)
TOL_SUMA = 1e-9             # 6.4: los F globales suman 1
TOL_PCA = 1e-6              # 6.2, 6.6 y 6.9
TOL_PCA_PROPIEDADES = 1e-9  # 6.2: ortonormalidad, media y varianza de las proyecciones
TOL_IGUAL = 1e-12           # "es igual" sin tolerancia en el contrato
TOL_ESCALADOR = 1e-9        # 6.5
TOL_X = 1e-12               # 6.8
TOL_SILHOUETTE = 1e-9       # 6.3 y 6.6
DESVIO_MIN = 1e-9           # 4.5: columnas con desvío <= 1e-9 se descartan

# Umbrales
SILHOUETTE_MIN = 0.50
SILHOUETTE_BLOQUE_MIN = 0.30
CONTRIBUCION_RANGO = (1.6, 2.4)
DENSIDAD_RANGO = (0.85, 1.15)
DISTANCIA_CORTE_MAX = 1.5
ANGULOS_CORTE = (0, 90, 180, 270)
TIEMPO_ACEPTACION_S = 180.0

# 6.2: datos del PCA
FORMAS_PCA = ((40, 5), (20, 60), (100, 30))

# 6.8: dataset chico
COLUMNAS_CHICO = ("archivo", "particion", "identidad", "intruso", "tipo")
CLAVES_NPZ = ("archivo", "particion", "tipo", "identidad", "intruso", "orientacion", "gabor", "minucias", "X")
TRAIN_CHICO, VAL_CHICO, INTRUSOS_CHICO = 4, 2, (1, 2)


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


def iguales_exactos(a, b) -> bool:
    """Igualdad de valor y de tipo, recursiva (14 != 14.0, True != 1)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(iguales_exactos(x, y) for x, y in zip(a, b))
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


def descripcion_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def exigir_sin_problemas(problemas: list[str], que: str, total: int | None = None) -> None:
    if problemas:
        de = f" de {total}" if total is not None else ""
        pytest.fail(f"{que}: fallan {len(problemas)}{de}.\n{resumen(problemas)}", pytrace=False)


def max_dif(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return math.inf
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def iguales_con_nan(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def comparar_con_nan(medido, esperado, *, atol: float = 0.0, rtol: float = 0.0, nombre_indice=None) -> str | None:
    """Compara dos vectores con NaN (mismos NaN; valores a atol + rtol·|esperado|). None si coinciden."""
    nombre_indice = nombre_indice or (lambda i: f"índice {i}")
    try:
        m = np.asarray(medido, dtype=np.float64)
    except (TypeError, ValueError) as error:
        return f"no es un array de números ({descripcion_error(error)})"
    e = np.asarray(esperado, dtype=np.float64)
    if m.shape != e.shape:
        return f"forma {m.shape}; se espera {e.shape}"
    problemas = []
    infinitos = np.flatnonzero(np.isinf(m))
    if infinitos.size:
        problemas.append(f"{infinitos.size} valores ±Inf (el primero en {nombre_indice(int(infinitos[0]))})")
    nan_m, nan_e = np.isnan(m), np.isnan(e)
    sobran = np.flatnonzero(nan_m & ~nan_e)
    faltan = np.flatnonzero(~nan_m & nan_e)
    if sobran.size:
        problemas.append(
            f"NaN donde no va en {sobran.size} lugares (el primero en {nombre_indice(int(sobran[0]))}, "
            f"se espera {float(e[sobran[0]])!r})"
        )
    if faltan.size:
        problemas.append(
            f"falta NaN en {faltan.size} lugares (el primero en {nombre_indice(int(faltan[0]))}, "
            f"vale {float(m[faltan[0]])!r})"
        )
    ambos = ~nan_m & ~nan_e & ~np.isinf(m)
    dif = np.where(ambos, np.abs(m - e), 0.0)
    limite = atol + rtol * np.abs(e)
    malos = np.flatnonzero(ambos & ~(dif <= limite))
    if malos.size:
        peor = int(malos[np.argmax(dif[malos])])
        relativa = dif[peor] / abs(e[peor]) if e[peor] != 0 else math.inf
        problemas.append(
            f"{malos.size} valores fuera de tolerancia; el peor en {nombre_indice(peor)}: medido {float(m[peor])!r}, "
            f"esperado {float(e[peor])!r} (|dif| {dif[peor]:.3e}, relativa {relativa:.3e})"
        )
    return "; ".join(problemas) or None


def silhouette_sk(X, etiquetas) -> float:
    return float(silhouette_score(np.asarray(X, dtype=np.float64), np.asarray(etiquetas)))


def alinear_signos(referencia_filas, otras_filas):
    """Signo (+1 o −1) por fila de `otras_filas` que la alinea con la misma fila de `referencia_filas`."""
    productos = np.sum(np.asarray(referencia_filas) * np.asarray(otras_filas), axis=1)
    return np.where(productos < 0, -1.0, 1.0)


def proyecciones_sklearn(X_ajuste, X_todas, k):
    sk = PCA_SKLEARN(n_components=k, svd_solver="full").fit(np.asarray(X_ajuste, dtype=np.float64))
    return sk, sk.transform(np.asarray(X_todas, dtype=np.float64))


def comparar_proyecciones(P_propio, P_sklearn) -> tuple[float, np.ndarray]:
    """máx |dif| de dos proyecciones (n, k) salvo el signo de cada componente."""
    P_propio = np.asarray(P_propio, dtype=np.float64)
    if P_propio.shape != P_sklearn.shape:
        return math.inf, np.ones(P_sklearn.shape[1])
    signos = alinear_signos(P_propio.T, P_sklearn.T)
    return max_dif(P_propio, P_sklearn * signos), signos


def contribucion(Zb: np.ndarray) -> float:
    """Media de ‖z_i − z_j‖² sobre los pares i ≠ j (igual sobre ordenados que sobre no ordenados)."""
    return float(pdist(np.asarray(Zb, dtype=np.float64), "sqeuclidean").mean())


def columnas_usadas_por_definicion(X_train: np.ndarray) -> np.ndarray:
    """4.5: al menos 2 valores no NaN y desvío (ddof=1) > 1e-9."""
    X = np.asarray(X_train, dtype=np.float64)
    cuenta = np.count_nonzero(~np.isnan(X), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        desvio = np.nanstd(X, axis=0, ddof=1)
    return (cuenta >= 2) & (desvio > DESVIO_MIN)


def estadisticas_de_train(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X_train, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(X, axis=0), np.nanstd(X, axis=0, ddof=1)


def tramos_por_bloque(conteos: dict) -> dict:
    """Columnas de cada bloque en el vector escalado, en el orden de BLOQUES (sección 8)."""
    tramos, inicio = {}, 0
    for b in BLOQUES_CONTRATO:
        tramos[b] = slice(inicio, inicio + int(conteos[b]))
        inicio += int(conteos[b])
    return tramos


# ---------------------------------------------------------------------------
# Geometría de la sección 0
# ---------------------------------------------------------------------------

_FILAS, _COLUMNAS = np.mgrid[0:ALTO_W, 0:ANCHO_W]
R_PIX = np.hypot(_COLUMNAS - C_W, _FILAS - C_W)
FI_PIX = np.mod(np.arctan2(_FILAS - C_W, _COLUMNAS - C_W), 2 * np.pi)
# Vecinos en orden circular P_1 … P_8: arriba, arriba-derecha, derecha, abajo-derecha, abajo,
# abajo-izquierda, izquierda, arriba-izquierda, como (Δfila, Δcolumna).
CIRCULO = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def anillos(radios) -> list[np.ndarray]:
    return [(R_PIX >= radios[a]) & (R_PIX < radios[a + 1]) for a in range(len(radios) - 1)]


def sectores(S: int) -> list[np.ndarray]:
    indice = np.minimum(np.floor(FI_PIX * S / (2 * np.pi)).astype(int), S - 1)
    return [indice == s for s in range(S)]


def bordes_grilla(g: int) -> np.ndarray:
    return np.round(np.linspace(0, 250, g + 1)).astype(int)


def celdas(g: int) -> list[np.ndarray]:
    """Celdas de la grilla de g × g, por filas (i afuera, j adentro)."""
    b = bordes_grilla(g)
    resultado = []
    for i in range(g):
        for j in range(g):
            celda = np.zeros(VENTANA, dtype=bool)
            celda[b[i]:b[i + 1], b[j]:b[j + 1]] = True
            resultado.append(celda)
    return resultado


def distancia_al_borde(mascara) -> np.ndarray:
    return ndimage.distance_transform_edt(np.pad(np.asarray(mascara, dtype=bool), 1))[1:-1, 1:-1]


def poca_cobertura(zona: np.ndarray, W: np.ndarray, cobertura_min: float) -> bool:
    """|A ∩ W| < cobertura_min · |A| (con 0,5 justo, la zona se usa)."""
    return np.count_nonzero(zona & W) < cobertura_min * np.count_nonzero(zona)


def crossing_number(E) -> np.ndarray:
    """CN de cada píxel de la ventana (fuera de la ventana cuenta como 0)."""
    P = np.pad(np.asarray(E, dtype=bool).astype(np.int16), 1)
    vecinos = [P[1 + di:1 + di + ALTO_W, 1 + dj:1 + dj + ANCHO_W] for di, dj in CIRCULO]
    return sum(np.abs(vecinos[k] - vecinos[(k + 1) % 8]) for k in range(8)) // 2


@functools.lru_cache(maxsize=4)
def _geometria(grilla_o, anillos_g, sectores_g, anillos_m, grilla_m) -> SimpleNamespace:
    zonas_gabor = [a & s for a in anillos(anillos_g) for s in sectores(sectores_g)]  # anillo afuera
    return SimpleNamespace(
        celdas_orientacion=celdas(grilla_o),
        zonas_gabor=zonas_gabor,
        anillos_minucias=anillos(anillos_m),
        celdas_minucias=celdas(grilla_m),
        bordes_minucias=bordes_grilla(grilla_m),
    )


def geometria(c: dict) -> SimpleNamespace:
    return _geometria(
        c["grilla_orientacion"], tuple(c["anillos_gabor_px"]), c["sectores_gabor"],
        tuple(c["anillos_minucias_px"]), c["grilla_minucias"],
    )


@functools.lru_cache(maxsize=4)
def _filtros_gabor(periodos, K) -> tuple[np.ndarray, ...]:
    filtros = []
    for P in periodos:
        sigma = P / 2
        L = 2 * int(round(3 * sigma)) + 1
        for k in range(K):
            g = cv2.getGaborKernel((L, L), sigma, math.pi * k / K + math.pi / 2, P, 1.0, 0.0, ktype=cv2.CV_64F)
            g = g - g.mean()
            filtros.append(g / np.abs(g).sum())
    return tuple(filtros)


def filtros_gabor(c: dict) -> tuple[np.ndarray, ...]:
    return _filtros_gabor(tuple(c["gabor_periodos_px"]), c["gabor_orientaciones"])


def largos_esperados(c: dict) -> dict:
    F = len(c["gabor_periodos_px"]) * c["gabor_orientaciones"]
    R = len(c["anillos_gabor_px"]) - 1
    R_m = len(c["anillos_minucias_px"]) - 1
    return {
        "orientacion": 2 * c["grilla_orientacion"] ** 2,
        "gabor": F * R * c["sectores_gabor"] + F,
        "minucias": 3 + 2 * R_m + c["grilla_minucias"] ** 2,
    }


def nombre_de_indice(bloque: str, c: dict):
    """Función que describe la posición i de un bloque crudo (para los mensajes)."""
    if bloque == "orientacion":
        g = c["grilla_orientacion"]
        return lambda i: f"celda ({i // 2 // g}, {i // 2 % g}) {'cos' if i % 2 == 0 else 'sen'} 2θ (índice {i})"
    if bloque == "gabor":
        F = len(c["gabor_periodos_px"]) * c["gabor_orientaciones"]
        R, S = len(c["anillos_gabor_px"]) - 1, c["sectores_gabor"]

        def gabor(i):
            if i < F * R * S:
                return f"filtro {i // (R * S)}, anillo {i % (R * S) // S}, sector {i % S} (índice {i})"
            return f"global del filtro {i - F * R * S} (índice {i})"

        return gabor
    R_m, g_m = len(c["anillos_minucias_px"]) - 1, c["grilla_minucias"]

    def minucias(i):
        if i < 3:
            return ("densidad de terminaciones", "densidad de bifurcaciones", "proporción de bifurcaciones")[i] + f" (índice {i})"
        if i < 3 + R_m:
            return f"terminaciones del anillo {i - 3} (índice {i})"
        if i < 3 + 2 * R_m:
            return f"bifurcaciones del anillo {i - 3 - R_m} (índice {i})"
        k = i - 3 - 2 * R_m
        return f"celda ({k // g_m}, {k % g_m}) de minucias (índice {i})"

    return minucias


# ---------------------------------------------------------------------------
# Oráculos de la sección 4
# ---------------------------------------------------------------------------


def esqueleto_limpio(esqueleto, componente_min: int) -> np.ndarray:
    """4.3.1: sin las componentes conexas (vecinos-8) de menos de componente_min píxeles."""
    E = np.asarray(esqueleto, dtype=bool)
    etiquetas, n = ndimage.label(E, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return E.copy()
    conservar = np.bincount(etiquetas.ravel()) >= componente_min
    conservar[0] = False
    return conservar[etiquetas]


def uniones_alcanzadas(t: tuple[int, int], E: np.ndarray, union: np.ndarray, pasos: int) -> set:
    """4.3.3: búsqueda en anchura desde t por E (vecinos-8), hasta `pasos` pasos.

    Alcanza las uniones pero no sigue desde ellas; las otras terminaciones se atraviesan.
    """
    distancia = {t: 0}
    cola = deque([t])
    alcanzadas = set()
    while cola:
        p = cola.popleft()
        if distancia[p] >= pasos:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                q = (p[0] + di, p[1] + dj)
                if (di == 0 and dj == 0) or q in distancia:
                    continue
                if not (0 <= q[0] < ALTO_W and 0 <= q[1] < ANCHO_W) or not E[q]:
                    continue
                distancia[q] = distancia[p] + 1
                if union[q]:
                    alcanzadas.add(q)
                else:
                    cola.append(q)
    return alcanzadas


def sin_pares(puntos: list, umbral: float) -> list:
    """4.3.4: saca los puntos que tienen otro a distancia euclídea < umbral (los dos del par)."""
    if len(puntos) < 2:
        return list(puntos)
    P = np.array(puntos, dtype=np.float64)
    d = cdist(P, P)
    np.fill_diagonal(d, np.inf)
    cerca = (d < umbral).any(axis=1)
    return [p for p, c in zip(puntos, cerca) if not c]


def como_xy(puntos: list) -> np.ndarray:
    """(fila, columna) → (x, y) = (j, i) en float64, ordenados por y y, a igual y, por x."""
    if not puntos:
        return np.zeros((0, 2), dtype=np.float64)
    xy = np.array([(j, i) for i, j in puntos], dtype=np.float64)
    return xy[np.lexsort((xy[:, 0], xy[:, 1]))]


def minucias_oraculo(esqueleto, V: np.ndarray, c: dict):
    """extraer_minucias según 4.3. Devuelve (terminaciones, bifurcaciones, E limpio, CN de E)."""
    E = esqueleto_limpio(esqueleto, c["componente_min_px"])
    CN = crossing_number(E)
    T0 = [tuple(int(v) for v in p) for p in np.argwhere(E & (CN == 1) & V)]
    B0 = [tuple(int(v) for v in p) for p in np.argwhere(E & (CN == 3) & V)]
    union = E & (CN >= 3)
    espuelas, uniones = set(), set()
    for t in T0:
        alcanzadas = uniones_alcanzadas(t, E, union, c["espuela_px"])
        if alcanzadas:
            espuelas.add(t)
            uniones |= alcanzadas
    T1 = [t for t in T0 if t not in espuelas]
    B1 = [b for b in B0 if not any(max(abs(b[0] - u[0]), abs(b[1] - u[1])) <= 2 for u in uniones)]
    T2 = sin_pares(T1, c["par_terminaciones_px"])
    B2 = sin_pares(B1, c["par_bifurcaciones_px"])
    return como_xy(T2), como_xy(B2), E, CN


def bloque_orientacion_oraculo(mascara, orientacion, c: dict) -> np.ndarray:
    """4.1."""
    M = np.asarray(mascara, dtype=bool)
    theta = np.asarray(orientacion, dtype=np.float64)
    cos2, sen2 = np.cos(2 * theta), np.sin(2 * theta)
    valores = []
    for celda in geometria(c).celdas_orientacion:
        if poca_cobertura(celda, M, c["cobertura_min"]):
            valores += [np.nan, np.nan]
        else:
            zona = celda & M
            valores += [cos2[zona].mean(), sen2[zona].mean()]
    return np.array(valores, dtype=np.float64)


def bloque_gabor_oraculo(mascara, G: np.ndarray, realzada, c: dict) -> np.ndarray:
    """4.2."""
    filtros = filtros_gabor(c)
    zonas = geometria(c).zonas_gabor
    F = len(filtros)
    M = np.asarray(mascara, dtype=bool)
    if not M.any() or not G.any():
        return np.full(F * len(zonas) + F, np.nan)
    f = 255.0 - np.asarray(realzada).astype(np.float64)
    mu = f[M].mean()
    f = np.where(M, f, mu) - mu
    sin_dato = [poca_cobertura(z, G, c["cobertura_min"]) for z in zonas]
    valores, medias = [], []
    for g in filtros:
        respuesta = np.abs(cv2.filter2D(f, -1, g, borderType=cv2.BORDER_REFLECT))
        for zona, nd in zip(zonas, sin_dato):
            valores.append(np.nan if nd else respuesta[zona & G].mean())
        medias.append(respuesta[G].mean())
    medias = np.array(medias, dtype=np.float64)
    total = medias.sum()
    globales = medias / total if total != 0 else np.full(F, np.nan)
    return np.concatenate([np.array(valores, dtype=np.float64), globales])


def bloque_minucias_oraculo(V: np.ndarray, T: np.ndarray, B: np.ndarray, c: dict) -> np.ndarray:
    """4.4."""
    geo = geometria(c)
    radios = c["anillos_minucias_px"]
    cmin = c["cobertura_min"]
    n_v = np.count_nonzero(V)
    n_t, n_b = len(T), len(B)
    if n_v == 0:
        valores = [np.nan, np.nan, np.nan]
    else:
        valores = [1000 * n_t / n_v, 1000 * n_b / n_v, n_b / (n_t + n_b) if n_t + n_b else 0.0]
    for puntos in (T, B):
        r = np.hypot(puntos[:, 0] - C_W, puntos[:, 1] - C_W)
        for a, anillo in enumerate(geo.anillos_minucias):
            if poca_cobertura(anillo, V, cmin):
                valores.append(np.nan)
            else:
                cuenta = np.count_nonzero((r >= radios[a]) & (r < radios[a + 1]))
                valores.append(1000 * cuenta / np.count_nonzero(anillo & V))
    todas = np.vstack([T, B])
    x, y = todas[:, 0], todas[:, 1]
    b = geo.bordes_minucias
    g = len(b) - 1
    for k, celda in enumerate(geo.celdas_minucias):
        if poca_cobertura(celda, V, cmin):
            valores.append(np.nan)
        else:
            i, j = divmod(k, g)
            cuenta = np.count_nonzero((x >= b[j]) & (x < b[j + 1]) & (y >= b[i]) & (y < b[i + 1]))
            valores.append(1000 * cuenta / np.count_nonzero(celda & V))
    return np.array(valores, dtype=np.float64)


def oraculo(h, c: dict) -> SimpleNamespace:
    """Todo lo de la sección 4 para una HuellaPreprocesada."""
    M = np.asarray(h.mascara, dtype=bool)
    D = distancia_al_borde(M)
    G = D >= c["erosion_gabor_px"]
    V = D >= c["borde_minucia_px"]
    T, B, E, CN = minucias_oraculo(h.esqueleto, V, c)
    return SimpleNamespace(
        T=T, B=B, E=E, CN=CN, G=G, V=V,
        bloques={
            "orientacion": bloque_orientacion_oraculo(M, h.orientacion, c),
            "gabor": bloque_gabor_oraculo(M, G, h.realzada, c),
            "minucias": bloque_minucias_oraculo(V, T, B, c),
        },
    )


def nan_esperados(mascara, G, V, c: dict) -> dict:
    """6.4 · Celdas sin huella: True donde el bloque tiene que valer NaN (los globales, False)."""
    geo = geometria(c)
    cmin = c["cobertura_min"]
    M = np.asarray(mascara, dtype=bool)
    orientacion = np.repeat([poca_cobertura(z, M, cmin) for z in geo.celdas_orientacion], 2)
    F = len(filtros_gabor(c))
    zonas = [poca_cobertura(z, G, cmin) for z in geo.zonas_gabor]
    gabor = np.array(zonas * F + [False] * F)
    anillos_v = [poca_cobertura(a, V, cmin) for a in geo.anillos_minucias]
    celdas_v = [poca_cobertura(z, V, cmin) for z in geo.celdas_minucias]
    minucias = np.array([False] * 3 + anillos_v * 2 + celdas_v)
    return {"orientacion": orientacion, "gabor": gabor, "minucias": minucias}


# ---------------------------------------------------------------------------
# Datos de prueba del PCA y la silhouette
# ---------------------------------------------------------------------------


def datos_pca(n: int, d: int, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """X (n, d) de rango completo, con columnas correlacionadas y media distinta de 0, y 7 filas nuevas."""
    rng = np.random.default_rng(semilla)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    mezcla = (q * np.geomspace(5.0, 0.5, d)) @ q.T
    media = rng.uniform(-10.0, 10.0, size=d)
    X = rng.normal(size=(n, d)) @ mezcla + media
    nuevas = rng.normal(size=(7, d)) @ mezcla + media
    return X, nuevas


def k_para(n: int, d: int, caso: str) -> int:
    return {"1": 1, "2": 2, "max": min(n - 1, d)}[caso]


def datos_silhouette(grupos: int, escala: float, semilla: int, unitario: bool = False):
    """Datos aleatorios de `grupos` grupos (el primero de una sola fila si unitario) y sus etiquetas enteras."""
    rng = np.random.default_rng(semilla)
    centros = rng.normal(scale=3.0, size=(grupos, 4))
    tamanos = rng.integers(4, 12, size=grupos)
    if unitario:
        tamanos[0] = 1
    X = np.vstack([centros[g] + rng.normal(size=(t, 4)) for g, t in enumerate(tamanos)]) * escala
    etiquetas = np.repeat(np.arange(grupos), tamanos)
    orden = rng.permutation(len(etiquetas))
    return X[orden], etiquetas[orden]


def como_texto(etiquetas) -> list[str]:
    nombres = ("zeta", "alfa", "mu", "beta", "omega")
    return [nombres[int(e)] for e in etiquetas]


# ---------------------------------------------------------------------------
# HuellaPreprocesada armada a mano (casos sintéticos de 6.4)
# ---------------------------------------------------------------------------


def huella_a_mano(clase, mascara, esqueleto, realzada=None, orientacion=None):
    realzada = np.full(VENTANA, 255, dtype=np.uint8) if realzada is None else realzada
    orientacion = np.zeros(VENTANA, dtype=np.float64) if orientacion is None else orientacion
    try:
        return clase(
            realzada=realzada, binaria=np.array(esqueleto, dtype=bool, copy=True), esqueleto=esqueleto,
            mascara=mascara, orientacion=orientacion, punto_referencia=(124.5, 124.5), metodo="curvatura",
            nucleos=(), angulo=0.0, matriz=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
    except Exception as error:
        pytest.fail(f"No se pudo armar una HuellaPreprocesada a mano: {descripcion_error(error)}", pytrace=False)


def caso_sintetico_de_corte(clase):
    """6.4: máscara en x < 180 y esqueleto en las filas y = 24, 36, …, 228 entre x = 20 y x = 179."""
    mascara = np.zeros(VENTANA, dtype=bool)
    mascara[:, :180] = True
    esqueleto = np.zeros(VENTANA, dtype=bool)
    for y in range(24, 229, 12):
        esqueleto[y, 20:180] = True
    return huella_a_mano(clase, mascara, esqueleto)


# ---------------------------------------------------------------------------
# Archivos del dataset chico (6.8)
# ---------------------------------------------------------------------------


def escribir_png(ruta: Path, imagen: np.ndarray) -> None:
    """Los mismos bytes que cv2.imwrite, sin problemas con tildes en la ruta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", imagen)
    assert ok, f"cv2.imencode no pudo codificar {ruta.name}"
    buffer.tofile(str(ruta))


def leer_png(ruta: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def escribir_metadatos(carpeta: Path, entradas: list[dict]) -> None:
    with open(carpeta / "metadatos.csv", "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(COLUMNAS_CHICO)
        for e in entradas:
            escritor.writerow([e[col] for col in COLUMNAS_CHICO])


def listado(carpeta: Path) -> list[str]:
    return sorted(p.relative_to(carpeta).as_posix() for p in carpeta.rglob("*"))


def leer_npz(ruta: Path) -> dict:
    try:
        with np.load(ruta, allow_pickle=False) as datos:
            return {clave: datos[clave] for clave in datos.files}
    except Exception as error:
        pytest.fail(f"No se pudo leer {ruta} con np.load(allow_pickle=False): {descripcion_error(error)}", pytrace=False)


def npz_iguales(a: dict, b: dict) -> list[str]:
    distintos = sorted(set(a) ^ set(b))
    for clave in sorted(set(a) & set(b)):
        x, y = a[clave], b[clave]
        if x.dtype.kind == "f" and y.dtype.kind == "f":
            ok = iguales_con_nan(x, y)
        else:
            ok = x.shape == y.shape and bool(np.array_equal(x, y))
        if not ok:
            distintos.append(clave)
    return distintos


def receta_h1(h1, cfg: dict, entrada: dict, base):
    """Variante de una entrada del plan con la receta de la sección 4 de H1."""
    rng = np.random.default_rng([entrada["semilla"], entrada["variante"]])
    params = h1.sortear_parametros(rng, cfg)
    return h1.aplicar_variante(base, params, rng)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class Caso:
    """Una variante de la muestra con su salida de H2."""

    archivo: str
    identidad: int
    particion: str
    variante: int
    corte: float
    h: object


@dataclass
class Calculado:
    """Lo que devuelve src/huellas/caracteristicas.py para una variante (o el error que lanzó)."""

    bloques: dict | None = None
    error_bloques: str | None = None
    minucias: object | None = None
    error_minucias: str | None = None
    problemas_formato: list[str] = field(default_factory=list)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return atributo(importar("src.config"), "cargar_config")()


@pytest.fixture(scope="module")
def c(cfg) -> dict:
    return obtener(cfg, SECCION_H3)


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
        generar_dataset=atributo(dataset, "generar_dataset"),
    )


@pytest.fixture(scope="module")
def h2():
    pre = importar("src.huellas.preprocesamiento")
    return SimpleNamespace(
        preprocesar=atributo(pre, "preprocesar"), HuellaPreprocesada=atributo(pre, "HuellaPreprocesada")
    )


@pytest.fixture(scope="module")
def car():
    return importar("src.huellas.caracteristicas")


@pytest.fixture(scope="module")
def mod_pca():
    return importar("src.core_ml.pca")


@pytest.fixture(scope="module")
def mod_metricas():
    return importar("src.core_ml.metricas")


@pytest.fixture(scope="module")
def plan(cfg, h1) -> list[dict]:
    return h1.plan_dataset(cfg)


@pytest.fixture(scope="module")
def bases(cfg, h1):
    """generar_base por (tipo, semilla), generadas a medida que se piden."""
    cache = {}

    def base_de(entrada: dict):
        clave = (entrada["tipo"], entrada["semilla"])
        if clave not in cache:
            cache[clave] = h1.generar_base(entrada["tipo"], entrada["semilla"], cfg)
        return cache[clave]

    return base_de


@pytest.fixture(scope="module")
def entradas_muestra(plan) -> list[dict]:
    return [e for e in plan if e["identidad"] in IDENTIDADES]


@pytest.fixture(scope="module")
def muestra(cfg, h1, h2, entradas_muestra, bases) -> list[Caso]:
    """Las 120 variantes de la sección 5 con su salida de H2."""
    casos = []
    for e in entradas_muestra:
        variante = receta_h1(h1, cfg, e, bases(e))
        casos.append(Caso(
            archivo=e["archivo"], identidad=int(e["identidad"]), particion=e["particion"],
            variante=int(e["variante"]), corte=float(variante.parametros.corte_fraccion),
            h=h2.preprocesar(variante.imagen, cfg),
        ))
    return casos


@pytest.fixture(scope="module")
def oraculos(muestra, c) -> list[SimpleNamespace]:
    return [oraculo(caso.h, c) for caso in muestra]


def problemas_de_bloques(bloques, largos: dict) -> list[str]:
    """Formato de 6.4: exactamente las claves de BLOQUES, float64 de 1 dimensión, el largo, sin ±Inf."""
    if not isinstance(bloques, dict):
        return [f"caracteristicas devuelve {type(bloques).__name__}, no un dict"]
    problemas = []
    if set(bloques) != set(BLOQUES_CONTRATO):
        problemas.append(f"claves {sorted(bloques)}; se esperan {list(BLOQUES_CONTRATO)}")
    for b in BLOQUES_CONTRATO:
        if b not in bloques:
            continue
        v = bloques[b]
        if not isinstance(v, np.ndarray):
            problemas.append(f"{b} no es un np.ndarray (es {type(v).__name__})")
            continue
        if v.dtype != np.float64 or v.ndim != 1 or v.shape[0] != largos[b]:
            problemas.append(f"{b} es {v.dtype} de forma {v.shape}; se espera float64 de forma ({largos[b]},)")
        elif np.isinf(v).any():
            problemas.append(f"{b} tiene {int(np.isinf(v).sum())} valores ±Inf")
    return problemas


@pytest.fixture(scope="module")
def calculados(muestra, car, cfg, c) -> list[Calculado]:
    """caracteristicas y extraer_minucias de src para cada variante de la muestra."""
    caracteristicas = atributo(car, "caracteristicas")
    extraer_minucias = atributo(car, "extraer_minucias")
    largos = largos_esperados(c)
    resultados = []
    for caso in muestra:
        r = Calculado()
        try:
            r.bloques = caracteristicas(caso.h, cfg)
            r.problemas_formato = problemas_de_bloques(r.bloques, largos)
        except Exception as error:
            r.error_bloques = f"caracteristicas lanzó {descripcion_error(error)}"
        try:
            r.minucias = extraer_minucias(caso.h, cfg)
        except Exception as error:
            r.error_minucias = f"extraer_minucias lanzó {descripcion_error(error)}"
        resultados.append(r)
    return resultados


@pytest.fixture(scope="module")
def apilados(muestra, calculados) -> dict:
    """Los bloques crudos de las 120 variantes, (120, d_b) por bloque."""
    problemas = []
    for caso, r in zip(muestra, calculados):
        if r.error_bloques:
            problemas.append(f"{caso.archivo}: {r.error_bloques}")
        elif r.problemas_formato:
            problemas.append(f"{caso.archivo}: {'; '.join(r.problemas_formato)}")
    if problemas:
        pytest.fail(
            f"No se pueden apilar los bloques de la muestra: {len(problemas)} variantes fallan.\n{resumen(problemas)}",
            pytrace=False,
        )
    return {b: np.vstack([r.bloques[b] for r in calculados]) for b in BLOQUES_CONTRATO}


@pytest.fixture(scope="module")
def particion(muestra) -> np.ndarray:
    return np.array([caso.particion for caso in muestra])


@pytest.fixture(scope="module")
def identidad(muestra) -> np.ndarray:
    return np.array([caso.identidad for caso in muestra])


@pytest.fixture(scope="module")
def con_corte(muestra) -> np.ndarray:
    return np.array([caso.corte > 0 for caso in muestra])


@pytest.fixture(scope="module")
def escalador(car, apilados, particion):
    ajustar_escalador = atributo(car, "ajustar_escalador")
    train = particion == "train"
    return ajustar_escalador({b: apilados[b][train].copy() for b in BLOQUES_CONTRATO})


@pytest.fixture(scope="module")
def Z(escalador, apilados) -> np.ndarray:
    """Los 120 vectores escalados de la muestra."""
    resultado = np.asarray(escalador.transformar({b: apilados[b].copy() for b in BLOQUES_CONTRATO}))
    if resultado.ndim != 2 or resultado.shape[0] != 120:
        pytest.fail(f"transformar de los 120 bloques devuelve forma {resultado.shape}; se espera (120, d).", pytrace=False)
    return resultado.astype(np.float64)


@pytest.fixture(scope="module")
def tramos(escalador) -> dict:
    """Columnas de cada bloque en Z."""
    return tramos_por_bloque({b: int(np.asarray(escalador.columnas[b]).sum()) for b in BLOQUES_CONTRATO})


def por_variante(muestra, *listas):
    return zip(muestra, *listas)


# ---------------------------------------------------------------------------
# Sección 5 · la muestra (control de la receta de H1, no de H3)
# ---------------------------------------------------------------------------


def test_la_muestra_es_la_de_la_seccion_5(muestra):
    por_particion = {
        k: {p: sum(1 for x in muestra if x.identidad == k and x.particion == p) for p in POR_PARTICION}
        for k in IDENTIDADES
    }
    cortes = {k: sum(1 for x in muestra if x.identidad == k and x.corte > 0) for k in IDENTIDADES}
    esperado = ({k: dict(POR_PARTICION) for k in IDENTIDADES}, CORTES_POR_IDENTIDAD)
    assert len(muestra) == 120 and (por_particion, cortes) == esperado, (
        f"La muestra no es la de la sección 5: {len(muestra)} variantes, por identidad y partición {por_particion} "
        f"y con corte {cortes}; se esperan 120, {POR_PARTICION} por identidad y {CORTES_POR_IDENTIDAD}."
    )


# ---------------------------------------------------------------------------
# 6.1 · Configuración
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clave", list(CLAVES_H3))
def test_clave_de_caracteristicas_con_su_tipo_y_rango(cfg, clave):
    regla = CLAVES_H3[clave]
    valor = obtener(cfg, clave)
    if regla[0] == "fijo":
        esperado = regla[1]
        assert iguales_exactos(valor, esperado), (
            f"{clave} = {valor!r} ({tipo_de(valor)}); se espera el valor fijo {esperado!r} ({tipo_de(esperado)})."
        )
    elif regla[0] == "entero":
        _, minimo, maximo = regla
        assert es_entero(valor), f"{clave} tiene que ser un entero (int, no bool); es {valor!r} ({tipo_de(valor)})."
        assert minimo <= valor <= maximo, f"{clave} = {valor!r} está fuera del rango [{minimo}, {maximo}]."
    elif regla[0] == "float":
        _, minimo, maximo = regla
        assert isinstance(valor, float), (
            f"{clave} tiene que ser un float de Python (por ejemplo 10.0, no 10); es {valor!r} ({tipo_de(valor)})."
        )
        assert minimo <= valor <= maximo, f"{clave} = {valor!r} está fuera del rango [{minimo}, {maximo}]."
    else:
        _, largo_min, largo_max, minimo, maximo = regla
        assert isinstance(valor, list), f"{clave} tiene que ser una lista; es {valor!r} ({tipo_de(valor)})."
        assert largo_min <= len(valor) <= largo_max, (
            f"{clave} tiene {len(valor)} elementos; se esperan entre {largo_min} y {largo_max}: {valor!r}."
        )
        assert all(isinstance(v, float) for v in valor), (
            f"{clave} tiene que ser una lista de float de Python (8.0, no 8); es {valor!r} ({tipo_de(valor)})."
        )
        fuera = [v for v in valor if not minimo <= v <= maximo]
        assert not fuera, f"{clave} = {valor!r} tiene valores fuera de [{minimo}, {maximo}]: {fuera}."
        assert all(a < b for a, b in zip(valor, valor[1:])), f"{clave} = {valor!r} no es estrictamente creciente."


@pytest.mark.parametrize("clave", list(CLAVES_H3))
def test_cargar_config_exige_cada_clave_nueva(tmp_path, clave):
    cargar_config = atributo(importar("src.config"), "cargar_config")
    ruta = escribir_yaml(tmp_path / "config.yaml", sin_clave(leer_config_real(), clave))
    error = lanza(ValueError, cargar_config, ruta, que=f"cargar_config con un config.yaml sin {clave}")
    assert nombra(str(error), clave), (
        f"El ValueError de cargar_config sin {clave} no nombra ese camino completo. Mensaje: {error}"
    )


def test_cargar_config_sin_la_seccion_nombra_sus_14_claves(tmp_path):
    cargar_config = atributo(importar("src.config"), "cargar_config")
    ruta = escribir_yaml(tmp_path / "config.yaml", sin_clave(leer_config_real(), SECCION_H3))
    error = lanza(ValueError, cargar_config, ruta, que=f"cargar_config con un config.yaml sin {SECCION_H3}")
    faltan = [clave for clave in CLAVES_H3 if not nombra(str(error), clave)]
    assert not faltan, (
        f"Sin la sección {SECCION_H3}, el ValueError no nombra {len(faltan)} de sus 14 claves: "
        f"{', '.join(faltan)}. Mensaje: {error}"
    )


# ---------------------------------------------------------------------------
# 6.2 · PCA propio
# ---------------------------------------------------------------------------

CASOS_ORACULO_PCA = [(n, d, k) for n, d in FORMAS_PCA for k in ("1", "2", "max")]


def ajustar_pca(mod_pca, X, n_componentes):
    PCA = atributo(mod_pca, "PCA")
    return PCA(n_componentes).ajustar(np.array(X, copy=True))


@pytest.mark.parametrize(("n", "d", "caso_k"), CASOS_ORACULO_PCA, ids=[f"{n}x{d}-k{k}" for n, d, k in CASOS_ORACULO_PCA])
def test_pca_igual_a_sklearn_salvo_el_signo(mod_pca, n, d, caso_k):
    X, nuevas = datos_pca(n, d, semilla=1000 + n + d)
    k = k_para(n, d, caso_k)
    pca = ajustar_pca(mod_pca, X, k)
    sk = PCA_SKLEARN(n_components=k, svd_solver="full").fit(X)
    problemas = []
    formas = {"media": (d,), "componentes": (k, d), "varianza": (k,), "proporcion_varianza": (k,)}
    for nombre, forma in formas.items():
        valor = getattr(pca, nombre, None)
        if not isinstance(valor, np.ndarray) or valor.dtype != np.float64 or valor.shape != forma:
            descripcion = f"{valor.dtype} de forma {valor.shape}" if isinstance(valor, np.ndarray) else repr(type(valor))
            problemas.append(f"{nombre} es {descripcion}; se espera float64 de forma {forma}")
    if problemas:
        pytest.fail(f"PCA({k}) sobre X de {n} × {d}:\n{resumen(problemas)}", pytrace=False)
    signos = alinear_signos(sk.components_, pca.componentes)
    dif_media = max_dif(pca.media, sk.mean_)
    dif_var = float(np.max(np.abs(pca.varianza - sk.explained_variance_) / np.abs(sk.explained_variance_)))
    dif_prop = max_dif(pca.proporcion_varianza, sk.explained_variance_ratio_)
    dif_comp = max_dif(pca.componentes * signos[:, None], sk.components_)
    dif_X = max_dif(np.asarray(pca.transformar(X)) * signos, sk.transform(X))
    dif_nuevas = max_dif(np.asarray(pca.transformar(nuevas)) * signos, sk.transform(nuevas))
    for nombre, dif, tipo in (
        ("media", dif_media, "abs"), ("varianza", dif_var, "relativa"), ("proporcion_varianza", dif_prop, "abs"),
        ("componentes (salvo el signo)", dif_comp, "abs"), ("transformar(X)", dif_X, "abs"),
        ("transformar(filas nuevas)", dif_nuevas, "abs"),
    ):
        if not dif <= TOL_PCA:
            problemas.append(f"{nombre}: máx |dif| {'relativa ' if tipo == 'relativa' else ''}{dif:.3e} (tolerancia {TOL_PCA:g})")
    assert not problemas, f"PCA({k}) sobre X de {n} × {d} contra sklearn (svd_solver='full'):\n{resumen(problemas)}"


@pytest.mark.parametrize(("n", "d"), FORMAS_PCA, ids=[f"{n}x{d}" for n, d in FORMAS_PCA])
def test_pca_componentes_ortonormales_varianza_ordenada_y_signo(mod_pca, n, d):
    X, _ = datos_pca(n, d, semilla=2000 + n + d)
    pca = ajustar_pca(mod_pca, X, None)
    comp = np.asarray(pca.componentes, dtype=np.float64)
    var = np.asarray(pca.varianza, dtype=np.float64)
    problemas = []
    dif_I = max_dif(comp @ comp.T, np.eye(comp.shape[0]))
    if not dif_I <= TOL_PCA_PROPIEDADES:
        problemas.append(f"componentes · componentesᵀ difiere de I en {dif_I:.3e} (tolerancia {TOL_PCA_PROPIEDADES:g})")
    if np.any(np.diff(var) > 0):
        problemas.append(f"varianza no es no creciente: {var.tolist()}")
    if np.any(var < 0):
        problemas.append(f"varianza tiene valores negativos: {var[var < 0].tolist()}")
    negativas = [i for i, fila in enumerate(comp) if not fila[int(np.argmax(np.abs(fila)))] > 0]
    if negativas:
        problemas.append(f"en las componentes {negativas} la entrada de mayor valor absoluto no es positiva")
    assert not problemas, f"PCA() sobre X de {n} × {d}:\n{resumen(problemas)}"


@pytest.mark.parametrize(("n", "d"), FORMAS_PCA, ids=[f"{n}x{d}" for n, d in FORMAS_PCA])
def test_pca_proyecciones_de_ajuste_con_media_0_y_varianza_igual(mod_pca, n, d):
    X, _ = datos_pca(n, d, semilla=3000 + n + d)
    pca = ajustar_pca(mod_pca, X, None)
    P = np.asarray(pca.transformar(X), dtype=np.float64)
    var = np.asarray(pca.varianza, dtype=np.float64)
    media = float(np.max(np.abs(P.mean(axis=0))))
    relativa = float(np.max(np.abs(P.var(axis=0, ddof=1) - var) / var))
    assert media <= TOL_PCA_PROPIEDADES and relativa <= TOL_PCA_PROPIEDADES, (
        f"PCA() sobre X de {n} × {d}: las proyecciones de las filas de ajuste tienen media de hasta {media:.3e} "
        f"y varianza (ddof=1) a {relativa:.3e} relativa de varianza (tolerancia {TOL_PCA_PROPIEDADES:g} en las dos)."
    )


def test_pca_api_ajustar_transformar_vector_y_determinismo(mod_pca):
    PCA = atributo(mod_pca, "PCA")
    problemas = []
    X, _ = datos_pca(40, 5, semilla=4000)
    original = X.copy()
    pca = PCA(2)
    devuelto = pca.ajustar(X)
    if devuelto is not pca:
        problemas.append(f"ajustar devuelve {type(devuelto).__name__} y no el mismo objeto")
    if not np.array_equal(X, original):
        problemas.append("ajustar modificó X")
    lote = np.asarray(pca.transformar(X))
    directo = np.asarray(PCA(2).ajustar_transformar(X))
    if not max_dif(directo, lote) <= TOL_IGUAL:
        problemas.append(f"ajustar_transformar(X) difiere de ajustar(X).transformar(X) en {max_dif(directo, lote):.3e}")
    vector = np.asarray(pca.transformar(X[3]))
    if vector.shape != (2,):
        problemas.append(f"transformar de un vector de forma {X[3].shape} da forma {vector.shape}; se espera (2,)")
    elif not max_dif(vector, lote[3]) <= TOL_IGUAL:
        problemas.append(f"transformar(X[3]) difiere de la fila 3 de transformar(X) en {max_dif(vector, lote[3]):.3e}")
    if not np.array_equal(X, original):
        problemas.append("transformar o ajustar_transformar modificaron X")
    otra = PCA(2).ajustar(np.array(X, copy=True))
    for nombre in ("media", "componentes", "varianza", "proporcion_varianza"):
        if not max_dif(getattr(otra, nombre), getattr(pca, nombre)) <= TOL_IGUAL:
            problemas.append(f"dos ajustes con el mismo X dan {nombre} distintos")
    for n, d in FORMAS_PCA:
        Xf, _ = datos_pca(n, d, semilla=4100 + n)
        k = np.asarray(PCA(None).ajustar(Xf).componentes).shape[0]
        if k != min(n - 1, d):
            problemas.append(f"con n_componentes=None y X de {n} × {d} hay {k} componentes; se esperan {min(n - 1, d)}")
    assert not problemas, f"API del PCA:\n{resumen(problemas)}"


@pytest.mark.parametrize("n_componentes", [0, -1, 2.5, True], ids=["0", "-1", "2.5", "True"])
def test_pca_constructor_rechaza_n_componentes_invalido(mod_pca, n_componentes):
    PCA = atributo(mod_pca, "PCA")
    lanza(ValueError, PCA, n_componentes, que=f"PCA(n_componentes={n_componentes!r})")


def _x_invalidas() -> dict:
    X, _ = datos_pca(40, 5, semilla=5000)
    X20, _ = datos_pca(20, 60, semilla=5001)
    con_nan, con_inf = X.copy(), X.copy()
    con_nan[3, 2] = np.nan
    con_inf[5, 1] = np.inf
    return {
        "1 dimensión": (2, X[:, 0].copy()),
        "3 dimensiones": (2, X.reshape(40, 5, 1).copy()),
        "una sola fila": (1, X[:1].copy()),
        "con NaN": (2, con_nan),
        "con Inf": (2, con_inf),
        # Valores exactos en binario: la media de las filas es exactamente la fila y traza(Σ) = 0.
        "todas las filas iguales": (2, np.tile([3.0, -1.0, 2.0, 0.5, 10.0], (8, 1))),
        "k > min(n − 1, d) en 40 × 5": (6, X),
        "k > min(n − 1, d) en 20 × 60": (20, X20),
    }


@pytest.mark.parametrize("caso", list(_x_invalidas()))
def test_pca_ajustar_rechaza_x_invalido(mod_pca, caso):
    PCA = atributo(mod_pca, "PCA")
    k, X = _x_invalidas()[caso]
    lanza(ValueError, PCA(k).ajustar, X, que=f"PCA({k}).ajustar con X {caso} (forma {X.shape})")


def test_pca_transformar_rechaza_otro_d_y_exige_ajustar(mod_pca):
    PCA = atributo(mod_pca, "PCA")
    X, _ = datos_pca(40, 5, semilla=6000)
    lanza(RuntimeError, PCA(2).transformar, X, que="transformar antes de ajustar")
    pca = PCA(2).ajustar(X)
    lanza(ValueError, pca.transformar, X[:, :4], que="transformar de filas con d = 4 (ajustado con d = 5)")
    lanza(ValueError, pca.transformar, X[0, :4], que="transformar de un vector con d = 4 (ajustado con d = 5)")


# ---------------------------------------------------------------------------
# 6.3 · Silhouette propia
# ---------------------------------------------------------------------------

CASOS_SILHOUETTE = [(g, e, t) for g in (2, 3, 4, 5) for e in (1e-3, 1.0, 1e3) for t in ("enteras", "texto")]


@pytest.mark.parametrize(
    ("grupos", "escala", "etiquetas_tipo"), CASOS_SILHOUETTE, ids=[f"{g}grupos-{e:g}-{t}" for g, e, t in CASOS_SILHOUETTE]
)
def test_silhouette_igual_a_sklearn(mod_metricas, grupos, escala, etiquetas_tipo):
    silhouette = atributo(mod_metricas, "silhouette")
    X, etiquetas = datos_silhouette(grupos, escala, semilla=grupos * 10 + int(math.log10(escala)) + 7)
    etiquetas = como_texto(etiquetas) if etiquetas_tipo == "texto" else etiquetas
    propia = silhouette(X, etiquetas)
    esperada = silhouette_score(X, etiquetas)
    assert isinstance(propia, float), f"silhouette devuelve {type(propia).__name__}; se espera float."
    assert abs(propia - esperada) <= TOL_SILHOUETTE, (
        f"{grupos} grupos, escala {escala:g}, etiquetas {etiquetas_tipo}: silhouette {propia!r}; "
        f"sklearn {esperada!r} (|dif| {abs(propia - esperada):.3e}, tolerancia {TOL_SILHOUETTE:g})."
    )


@pytest.mark.parametrize("etiquetas_tipo", ["enteras", "texto"])
def test_silhouette_con_un_grupo_de_una_sola_fila(mod_metricas, etiquetas_tipo):
    silhouette = atributo(mod_metricas, "silhouette")
    X, etiquetas = datos_silhouette(3, 1.0, semilla=77, unitario=True)
    etiquetas = como_texto(etiquetas) if etiquetas_tipo == "texto" else etiquetas
    propia, esperada = silhouette(X, etiquetas), silhouette_score(X, etiquetas)
    assert abs(propia - esperada) <= TOL_SILHOUETTE, (
        f"Con un grupo de una sola fila (etiquetas {etiquetas_tipo}): silhouette {propia!r}; sklearn {esperada!r}."
    )


def _silhouette_invalidas() -> dict:
    rng = np.random.default_rng(88)
    X = rng.normal(size=(10, 3))
    con_nan = X.copy()
    con_nan[2, 1] = np.nan
    etiquetas = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    return {
        "una sola etiqueta distinta": (X, np.zeros(10, dtype=int)),
        "tantas etiquetas distintas como filas": (X, np.arange(10)),
        "largos distintos": (X, etiquetas[:9]),
        "X con NaN": (con_nan, etiquetas),
    }


@pytest.mark.parametrize("caso", list(_silhouette_invalidas()))
def test_silhouette_rechaza_entradas_invalidas(mod_metricas, caso):
    silhouette = atributo(mod_metricas, "silhouette")
    X, etiquetas = _silhouette_invalidas()[caso]
    lanza(ValueError, silhouette, X, etiquetas, que=f"silhouette con {caso}")


@pytest.mark.parametrize("ruta", ["src/core_ml/pca.py", "src/core_ml/metricas.py", "src/huellas/caracteristicas.py"])
def test_archivo_no_usa_scikit_learn(ruta):
    archivo = RAIZ / ruta
    assert archivo.is_file(), f"No existe {ruta}."
    texto = archivo.read_text(encoding="utf-8")
    lineas = [n for n, linea in enumerate(texto.splitlines(), 1) if "sklearn" in linea]
    assert not lineas, f"{ruta} contiene el texto 'sklearn' en las líneas {lineas}: no puede usar scikit-learn."


# ---------------------------------------------------------------------------
# 6.4 · Formato y definiciones exactas (muestra)
# ---------------------------------------------------------------------------


def test_bloques_del_contrato(car):
    bloques = atributo(car, "BLOQUES")
    assert isinstance(bloques, tuple) and bloques == BLOQUES_CONTRATO, (
        f"BLOQUES = {bloques!r}; se espera la tupla {BLOQUES_CONTRATO!r}."
    )


def test_largos_de_bloque_con_el_cfg_real(car, cfg):
    largos = atributo(car, "largos_de_bloque")(cfg)
    ok = isinstance(largos, dict) and largos == LARGOS_REALES and all(es_entero(v) for v in largos.values())
    assert ok, f"largos_de_bloque(cfg) = {largos!r}; se espera {LARGOS_REALES} (con enteros)."


def test_caracteristicas_devuelve_los_tres_bloques_con_su_formato(muestra, calculados):
    problemas = []
    for caso, r in por_variante(muestra, calculados):
        if r.error_bloques:
            problemas.append(f"{caso.archivo}: {r.error_bloques}")
        elif r.problemas_formato:
            problemas.append(f"{caso.archivo}: {'; '.join(r.problemas_formato)}")
    exigir_sin_problemas(problemas, "6.4 · formato de los bloques", len(muestra))


def variantes_elegidas(muestra) -> list[int]:
    """Índices de 6 variantes: por identidad, la primera con corte y la primera sin corte del plan."""
    elegidas = []
    for k in IDENTIDADES:
        de_k = [i for i, caso in enumerate(muestra) if caso.identidad == k]
        elegidas.append(next(i for i in de_k if muestra[i].corte > 0))
        elegidas.append(next(i for i in de_k if muestra[i].corte == 0))
    return elegidas


def test_caracteristicas_no_modifica_la_huella_y_es_determinista(muestra, calculados, car, cfg):
    caracteristicas = atributo(car, "caracteristicas")
    campos = ("realzada", "binaria", "esqueleto", "mascara", "orientacion", "matriz")
    problemas = []
    for i in variantes_elegidas(muestra):
        caso = muestra[i]
        antes = {nombre: np.array(getattr(caso.h, nombre), copy=True) for nombre in campos}
        otros_antes = (caso.h.punto_referencia, caso.h.metodo, caso.h.nucleos, caso.h.angulo)
        try:
            primera = caracteristicas(caso.h, cfg)
            segunda = caracteristicas(caso.h, cfg)
        except Exception as error:
            problemas.append(f"{caso.archivo}: caracteristicas lanzó {descripcion_error(error)}")
            continue
        cambiados = [n for n in campos if not np.array_equal(getattr(caso.h, n), antes[n])]
        if (caso.h.punto_referencia, caso.h.metodo, caso.h.nucleos, caso.h.angulo) != otros_antes:
            cambiados.append("punto_referencia, metodo, nucleos o angulo")
        if cambiados:
            problemas.append(f"{caso.archivo}: caracteristicas modificó {', '.join(cambiados)}")
        referencia = calculados[i].bloques
        for b in BLOQUES_CONTRATO:
            try:
                iguales = iguales_con_nan(primera[b], segunda[b]) and iguales_con_nan(primera[b], referencia[b])
            except Exception as error:
                iguales = False
                problemas.append(f"{caso.archivo}: no se pudo comparar el bloque {b} ({descripcion_error(error)})")
                continue
            if not iguales:
                problemas.append(f"{caso.archivo}: tres llamadas dan bloques {b} distintos (np.array_equal con NaN)")
    exigir_sin_problemas(problemas, "6.4 · no modifica h y es determinista")


def problemas_de_oraculo(muestra, calculados, oraculos, bloque: str, c: dict, **tolerancia) -> list[str]:
    problemas = []
    nombre = nombre_de_indice(bloque, c)
    for caso, r, o in por_variante(muestra, calculados, oraculos):
        if r.error_bloques:
            problemas.append(f"{caso.archivo}: {r.error_bloques}")
            continue
        if not isinstance(r.bloques, dict) or bloque not in r.bloques:
            problemas.append(f"{caso.archivo}: caracteristicas no devuelve el bloque {bloque}")
            continue
        detalle = comparar_con_nan(r.bloques[bloque], o.bloques[bloque], nombre_indice=nombre, **tolerancia)
        if detalle:
            problemas.append(f"{caso.archivo}: {detalle}")
    return problemas


def test_oraculo_bloque_de_orientacion(muestra, calculados, oraculos, c):
    problemas = problemas_de_oraculo(muestra, calculados, oraculos, "orientacion", c, atol=TOL_ORACULO)
    exigir_sin_problemas(problemas, f"6.4 · bloque de orientación contra la sección 4.1 (tolerancia {TOL_ORACULO:g})", len(muestra))


def test_oraculo_bloque_de_gabor(muestra, calculados, oraculos, c):
    problemas = problemas_de_oraculo(muestra, calculados, oraculos, "gabor", c, rtol=TOL_GABOR_REL)
    exigir_sin_problemas(
        problemas, f"6.4 · bloque de Gabor contra la sección 4.2 (tolerancia relativa {TOL_GABOR_REL:g})", len(muestra)
    )


def test_oraculo_bloque_de_minucias(muestra, calculados, oraculos, c):
    problemas = problemas_de_oraculo(muestra, calculados, oraculos, "minucias", c, atol=TOL_ORACULO)
    exigir_sin_problemas(problemas, f"6.4 · bloque de minucias contra la sección 4.4 (tolerancia {TOL_ORACULO:g})", len(muestra))


def fmt_puntos(puntos: np.ndarray, maximo: int = 6) -> str:
    texto = ", ".join(f"({x:g}, {y:g})" for x, y in np.asarray(puntos)[:maximo])
    return f"[{texto}{', …' if len(puntos) > maximo else ''}]"


def diferencia_de_minucias(medido, esperado, nombre: str) -> str | None:
    if not isinstance(medido, np.ndarray):
        return f"{nombre} no es un np.ndarray (es {type(medido).__name__})"
    if medido.dtype != np.float64 or medido.ndim != 2 or medido.shape[1:] != (2,):
        return f"{nombre} es {medido.dtype} de forma {medido.shape}; se espera float64 de forma (k, 2)"
    if medido.shape == esperado.shape and np.array_equal(medido, esperado):
        return None
    m = {tuple(p) for p in medido.tolist()}
    e = {tuple(p) for p in esperado.tolist()}
    if m == e and len(medido) == len(esperado):
        return f"{nombre}: las mismas {len(e)}, pero en otro orden (se ordenan por y y, a igual y, por x)"
    sobran = np.array(sorted(m - e, key=lambda p: (p[1], p[0]))).reshape(-1, 2)
    faltan = np.array(sorted(e - m, key=lambda p: (p[1], p[0]))).reshape(-1, 2)
    return (
        f"{nombre}: {len(medido)} medidas y {len(esperado)} esperadas; sobran {len(sobran)} {fmt_puntos(sobran)} "
        f"y faltan {len(faltan)} {fmt_puntos(faltan)}"
    )


def test_oraculo_extraer_minucias(muestra, calculados, oraculos, car):
    Minucias = atributo(car, "Minucias")
    problemas = []
    for caso, r, o in por_variante(muestra, calculados, oraculos):
        if r.error_minucias:
            problemas.append(f"{caso.archivo}: {r.error_minucias}")
            continue
        if not isinstance(r.minucias, Minucias):
            problemas.append(f"{caso.archivo}: extraer_minucias devuelve {type(r.minucias).__name__}, no Minucias")
            continue
        detalles = [
            diferencia_de_minucias(r.minucias.terminaciones, o.T, "terminaciones"),
            diferencia_de_minucias(r.minucias.bifurcaciones, o.B, "bifurcaciones"),
        ]
        detalles = [d for d in detalles if d]
        if detalles:
            problemas.append(f"{caso.archivo}: {'; '.join(detalles)}")
    exigir_sin_problemas(problemas, "6.4 · extraer_minucias contra la sección 4.3", len(muestra))


def test_minucias_en_el_esqueleto_lejos_del_borde_y_sin_pares(muestra, calculados, oraculos, c):
    problemas = []
    for caso, r, o in por_variante(muestra, calculados, oraculos):
        if r.error_minucias:
            problemas.append(f"{caso.archivo}: {r.error_minucias}")
            continue
        try:
            listas = (
                ("terminación", np.asarray(r.minucias.terminaciones, dtype=np.float64).reshape(-1, 2), 1,
                 c["par_terminaciones_px"]),
                ("bifurcación", np.asarray(r.minucias.bifurcaciones, dtype=np.float64).reshape(-1, 2), 3,
                 c["par_bifurcaciones_px"]),
            )
        except Exception as error:
            problemas.append(f"{caso.archivo}: no se pueden leer las minucias ({descripcion_error(error)})")
            continue
        D = distancia_al_borde(caso.h.mascara)
        esqueleto = np.asarray(caso.h.esqueleto, dtype=bool)
        detalles = []
        for nombre, puntos, cn, par in listas:
            for x, y in puntos:
                if x != int(x) or y != int(y) or not (0 <= x < ANCHO_W and 0 <= y < ALTO_W):
                    detalles.append(f"{nombre} ({x:g}, {y:g}) no es un píxel de la ventana")
                    continue
                i, j = int(y), int(x)
                if not esqueleto[i, j]:
                    detalles.append(f"{nombre} ({x:g}, {y:g}) no está en esqueleto")
                elif o.CN[i, j] != cn:
                    detalles.append(f"{nombre} ({x:g}, {y:g}) tiene CN = {o.CN[i, j]} en el esqueleto limpio (se espera {cn})")
                if D[i, j] < c["borde_minucia_px"]:
                    detalles.append(
                        f"{nombre} ({x:g}, {y:g}) a {D[i, j]:.2f} px del borde de la máscara (mínimo {c['borde_minucia_px']:g})"
                    )
            if len(puntos) >= 2:
                d = cdist(puntos, puntos)
                np.fill_diagonal(d, np.inf)
                i, j = np.unravel_index(int(np.argmin(d)), d.shape)
                if d[i, j] < par:
                    detalles.append(
                        f"dos {nombre}s a {d[i, j]:.2f} px: {fmt_puntos(puntos[[i, j]])} (mínimo {par:g})"
                    )
        if detalles:
            problemas.append(f"{caso.archivo}: {'; '.join(detalles[:4])}")
    exigir_sin_problemas(problemas, "6.4 · minucias (CN, borde y pares)", len(muestra))


def test_nan_si_y_solo_si_falta_huella_y_globales_completos(muestra, calculados, oraculos, c):
    F = len(filtros_gabor(c))
    problemas = []
    for caso, r, o in por_variante(muestra, calculados, oraculos):
        if r.error_bloques:
            problemas.append(f"{caso.archivo}: {r.error_bloques}")
            continue
        esperados = nan_esperados(caso.h.mascara, o.G, o.V, c)
        detalles = []
        for b, mascara_nan in esperados.items():
            try:
                medido = np.isnan(np.asarray(r.bloques[b], dtype=np.float64))
            except Exception as error:
                detalles.append(f"no se puede leer el bloque {b} ({descripcion_error(error)})")
                continue
            if medido.shape != mascara_nan.shape:
                detalles.append(f"el bloque {b} tiene forma {medido.shape}; se espera {mascara_nan.shape}")
                continue
            nombre = nombre_de_indice(b, c)
            sobran = np.flatnonzero(medido & ~mascara_nan)
            faltan = np.flatnonzero(~medido & mascara_nan)
            if sobran.size:
                detalles.append(f"{b}: NaN con cobertura suficiente en {sobran.size} lugares ({nombre(int(sobran[0]))}, …)")
            if faltan.size:
                detalles.append(f"{b}: sin NaN con menos de cobertura_min en {faltan.size} lugares ({nombre(int(faltan[0]))}, …)")
        try:
            globales_gabor = np.asarray(r.bloques["gabor"], dtype=np.float64)[-F:]
            globales_minucias = np.asarray(r.bloques["minucias"], dtype=np.float64)[:3]
            if np.isnan(globales_minucias).any():
                detalles.append(f"globales de minucias con NaN: {globales_minucias.tolist()}")
            if np.isnan(globales_gabor).any():
                detalles.append("globales de Gabor con NaN")
            elif not abs(globales_gabor.sum() - 1) <= TOL_SUMA:
                detalles.append(f"los {F} globales de Gabor suman {globales_gabor.sum()!r} (se espera 1 ± {TOL_SUMA:g})")
        except Exception as error:
            detalles.append(f"no se pueden leer los globales ({descripcion_error(error)})")
        if detalles:
            problemas.append(f"{caso.archivo}: {'; '.join(detalles)}")
    exigir_sin_problemas(problemas, "6.4 · celdas sin huella y globales", len(muestra))


def test_caso_sintetico_de_corte_sin_minucias_falsas_en_el_borde_recto(car, cfg, h2):
    extraer_minucias = atributo(car, "extraer_minucias")
    resultado = extraer_minucias(caso_sintetico_de_corte(h2.HuellaPreprocesada), cfg)
    esperadas = np.array([(20.0, float(y)) for y in range(24, 229, 12)], dtype=np.float64)
    detalles = [
        diferencia_de_minucias(resultado.terminaciones, esperadas, "terminaciones"),
        diferencia_de_minucias(resultado.bifurcaciones, np.zeros((0, 2)), "bifurcaciones"),
    ]
    detalles = [d for d in detalles if d]
    assert not detalles, (
        "Caso sintético de corte (máscara en x < 180, 18 crestas horizontales de x = 20 a x = 179): se esperan "
        f"exactamente las 18 terminaciones (20, y) y ninguna en x = 179 ni ninguna bifurcación.\n{resumen(detalles)}"
    )


def test_caso_sintetico_de_cobertura_justa(car, cfg, h2):
    caracteristicas = atributo(car, "caracteristicas")
    mascara = np.zeros(VENTANA, dtype=bool)
    mascara[:, :175] = True
    h = huella_a_mano(h2.HuellaPreprocesada, mascara, np.zeros(VENTANA, dtype=bool))
    bloque = np.asarray(caracteristicas(h, cfg)["orientacion"], dtype=np.float64)
    assert bloque.shape == (50,), f"El bloque de orientación tiene forma {bloque.shape}; se espera (50,)."
    nan_por_celda = np.isnan(bloque).reshape(5, 5, 2)
    esperado = np.zeros((5, 5), dtype=bool)
    esperado[:, 4] = True
    problemas = []
    for i in range(5):
        for j in range(5):
            if not (nan_por_celda[i, j, 0] == nan_por_celda[i, j, 1] == esperado[i, j]):
                problemas.append(
                    f"celda ({i}, {j}): NaN = {nan_por_celda[i, j].tolist()}; se espera "
                    f"{'NaN' if esperado[i, j] else 'valores (cobertura ' + ('0,5 justa' if j == 3 else '1') + ')'}"
                )
    validos = bloque[~np.isnan(bloque)].reshape(-1, 2)
    if validos.size and not max_dif(validos, np.tile([1.0, 0.0], (len(validos), 1))) <= TOL_IGUAL:
        problemas.append(f"con orientacion = 0 las celdas válidas tienen que valer [1, 0]; hay {validos[:3].tolist()}")
    assert not problemas, (
        f"Caso sintético de cobertura (máscara en x < 175): NaN exactamente en la columna j = 4.\n{resumen(problemas)}"
    )


def test_sin_huella_los_tres_bloques_son_todo_nan(car, cfg, h2, c):
    caracteristicas = atributo(car, "caracteristicas")
    vacia = np.zeros(VENTANA, dtype=bool)
    bloques = caracteristicas(huella_a_mano(h2.HuellaPreprocesada, vacia, vacia.copy()), cfg)
    largos = largos_esperados(c)
    problemas = []
    for b in BLOQUES_CONTRATO:
        v = np.asarray(bloques[b], dtype=np.float64)
        if v.shape != (largos[b],):
            problemas.append(f"{b} tiene forma {v.shape}; se espera ({largos[b]},)")
        elif not np.isnan(v).all():
            problemas.append(f"{b} tiene {int((~np.isnan(v)).sum())} valores que no son NaN")
    assert not problemas, f"Con mascara y esqueleto vacíos:\n{resumen(problemas)}"


CASOS_FORMA = [(campo, forma) for campo in ("realzada", "mascara", "esqueleto", "orientacion") for forma in ((250, 249), (125, 500))]


@pytest.mark.parametrize(("campo", "forma"), CASOS_FORMA, ids=[f"{c}-{f[0]}x{f[1]}" for c, f in CASOS_FORMA])
def test_caracteristicas_rechaza_campos_que_no_son_de_250x250(car, cfg, h2, campo, forma):
    caracteristicas = atributo(car, "caracteristicas")
    h = caso_sintetico_de_corte(h2.HuellaPreprocesada)
    tipos = {"realzada": np.uint8, "mascara": bool, "esqueleto": bool, "orientacion": np.float64}
    relleno = {"realzada": 255, "mascara": True, "esqueleto": False, "orientacion": 0.0}
    otra = dataclasses.replace(h, **{campo: np.full(forma, relleno[campo], dtype=tipos[campo])})
    lanza(ValueError, caracteristicas, otra, cfg, que=f"caracteristicas con {campo} de forma {forma}")


# ---------------------------------------------------------------------------
# 6.5 · Escalador y vector (muestra)
# ---------------------------------------------------------------------------


def test_vector_escalado_finito_con_las_estadisticas_de_train(escalador, apilados, particion, Z):
    train = particion == "train"
    problemas = []
    usadas = {}
    for nombre in ("medias", "desvios", "columnas", "pesos"):
        valor = getattr(escalador, nombre, None)
        if not isinstance(valor, dict) or set(valor) != set(BLOQUES_CONTRATO):
            problemas.append(f"Escalador.{nombre} tiene que ser un dict con las claves {list(BLOQUES_CONTRATO)}")
    if problemas:
        pytest.fail(f"6.5 · formato del escalador:\n{resumen(problemas)}", pytrace=False)
    for b in BLOQUES_CONTRATO:
        d_b = apilados[b].shape[1]
        columnas = np.asarray(escalador.columnas[b])
        if columnas.dtype != bool or columnas.shape != (d_b,):
            problemas.append(f"columnas[{b}] es {columnas.dtype} de forma {columnas.shape}; se espera bool de ({d_b},)")
            continue
        usadas[b] = columnas
        medias, desvios = estadisticas_de_train(apilados[b][train])
        for nombre, esperado in (("medias", medias), ("desvios", desvios)):
            valor = np.asarray(getattr(escalador, nombre)[b], dtype=np.float64)
            if valor.shape != (d_b,):
                problemas.append(f"{nombre}[{b}] tiene forma {valor.shape}; se espera ({d_b},)")
                continue
            dif = max_dif(valor[columnas], esperado[columnas])
            if not dif <= TOL_ESCALADOR:
                peor = int(np.flatnonzero(columnas)[np.argmax(np.abs(valor[columnas] - esperado[columnas]))])
                problemas.append(
                    f"{nombre}[{b}] difiere de la {'media' if nombre == 'medias' else 'desvío (ddof=1)'} de train sin "
                    f"NaN en {dif:.3e} (columna {peor}: {float(valor[peor])!r} y {float(esperado[peor])!r}; tolerancia {TOL_ESCALADOR:g})"
                )
    d = sum(int(u.sum()) for u in usadas.values())
    if Z.shape != (120, d):
        problemas.append(f"Z tiene forma {Z.shape}; se espera (120, {d}) (d = columnas usadas)")
    if not np.isfinite(Z).all():
        problemas.append(f"Z tiene {int((~np.isfinite(Z)).sum())} valores NaN o ±Inf")
    medias_train = np.abs(Z[train].mean(axis=0))
    if medias_train.size and not medias_train.max() <= TOL_ESCALADOR:
        problemas.append(f"la media de Z en train llega a {medias_train.max():.3e} en la columna {int(medias_train.argmax())}")
    assert not problemas, f"6.5 · escalador y vector escalado de la muestra:\n{resumen(problemas)}"


def test_imputacion_en_0_columnas_descartadas_y_pesos(escalador, apilados, particion, Z, tramos):
    train = particion == "train"
    problemas = []
    for b in BLOQUES_CONTRATO:
        esperadas = columnas_usadas_por_definicion(apilados[b][train])
        columnas = np.asarray(escalador.columnas[b], dtype=bool)
        if columnas.shape != esperadas.shape or not np.array_equal(columnas, esperadas):
            de_mas = np.flatnonzero(columnas & ~esperadas).tolist() if columnas.shape == esperadas.shape else "?"
            de_menos = np.flatnonzero(~columnas & esperadas).tolist() if columnas.shape == esperadas.shape else "?"
            problemas.append(
                f"{b}: columnas usadas de más {de_mas} y de menos {de_menos} (se descartan las que tienen menos de 2 "
                f"valores o desvío <= {DESVIO_MIN:g} en train; descartadas por definición: "
                f"{np.flatnonzero(~esperadas).tolist()})"
            )
            continue
        peso = float(escalador.pesos[b])
        if not abs(peso - 1 / math.sqrt(columnas.sum())) <= TOL_IGUAL:
            problemas.append(f"pesos[{b}] = {peso!r}; se espera 1/√{int(columnas.sum())} = {1 / math.sqrt(columnas.sum())!r}")
        Zb = Z[:, tramos[b]]
        X = apilados[b][:, columnas]
        filas, cols = np.nonzero(np.isnan(X))
        no_cero = [(int(f), int(k)) for f, k in zip(filas, cols) if Zb[f, k] != 0.0]
        if no_cero:
            f, k = no_cero[0]
            problemas.append(
                f"{b}: {len(no_cero)} entradas NaN de columnas usadas no valen 0 en Z (fila {f}, columna usada {k}: {float(Zb[f, k])!r})"
            )
    assert not problemas, f"6.5 · imputación, columnas y pesos:\n{resumen(problemas)}"


def test_transformar_de_un_vector_es_la_fila_del_lote(escalador, apilados, Z):
    problemas = []
    for i in range(Z.shape[0]):
        vector = np.asarray(escalador.transformar({b: apilados[b][i].copy() for b in BLOQUES_CONTRATO}))
        if vector.shape != (Z.shape[1],):
            problemas.append(f"fila {i}: transformar de un vector da forma {vector.shape}; se espera ({Z.shape[1]},)")
        elif not max_dif(vector, Z[i]) <= TOL_IGUAL:
            problemas.append(f"fila {i}: difiere de la fila del lote en {max_dif(vector, Z[i]):.3e}")
    exigir_sin_problemas(problemas, "6.5 · transformar de un vector", Z.shape[0])


def _bloques_invalidos(apilados) -> dict:
    vector = {b: apilados[b][0].copy() for b in BLOQUES_CONTRATO}
    sin_gabor = {b: v for b, v in vector.items() if b != "gabor"}
    con_hu = dict(vector, hu=np.zeros(7))
    corto = dict(vector, minucias=vector["minucias"][:-1].copy())
    con_inf = {b: v.copy() for b, v in vector.items()}
    con_inf["orientacion"][0] = np.inf
    return {"falta la clave gabor": sin_gabor, "sobra la clave hu": con_hu,
            "minucias con un valor de menos": corto, "orientacion con un Inf": con_inf}


@pytest.mark.parametrize("caso", ["falta la clave gabor", "sobra la clave hu", "minucias con un valor de menos", "orientacion con un Inf"])
def test_transformar_rechaza_bloques_invalidos(escalador, apilados, caso):
    bloques = _bloques_invalidos(apilados)[caso]
    lanza(ValueError, escalador.transformar, bloques, que=f"transformar con {caso}")


def test_ajustar_escalador_rechaza_una_sola_fila(car, apilados):
    ajustar_escalador = atributo(car, "ajustar_escalador")
    lanza(
        ValueError, ajustar_escalador, {b: apilados[b][:1].copy() for b in BLOQUES_CONTRATO},
        que="ajustar_escalador con una sola fila",
    )


def test_transformar_trabaja_fila_por_fila(escalador, apilados, Z):
    problemas = []
    orden = np.arange(Z.shape[0])[::-1]
    invertido = np.asarray(escalador.transformar({b: apilados[b][orden].copy() for b in BLOQUES_CONTRATO}))
    if not max_dif(invertido, Z[orden]) <= TOL_IGUAL:
        problemas.append(f"con las filas en orden inverso, las filas cambian hasta {max_dif(invertido, Z[orden]):.3e}")
    for nombre, filas in (("las 10 primeras", np.arange(10)), ("solo las de test", np.arange(32, 40))):
        parte = np.asarray(escalador.transformar({b: apilados[b][filas].copy() for b in BLOQUES_CONTRATO}))
        if not max_dif(parte, Z[filas]) <= TOL_IGUAL:
            problemas.append(f"con {nombre} en el lote, las filas cambian hasta {max_dif(parte, Z[filas]):.3e}")
    assert not problemas, f"6.5 · la fila de una variante depende de las otras filas del lote:\n{resumen(problemas)}"


@pytest.mark.parametrize("bloque", BLOQUES_CONTRATO)
def test_contribucion_de_cada_bloque_equilibrada_en_train(Z, tramos, particion, bloque):
    train = particion == "train"
    valores = {b: contribucion(Z[train][:, tramos[b]]) for b in BLOQUES_CONTRATO}
    minimo, maximo = CONTRIBUCION_RANGO
    assert minimo <= valores[bloque] <= maximo, (
        f"6.5 · contribución del bloque {bloque} en las 72 de train: {valores[bloque]:.4f}; se exige [{minimo}, {maximo}]. "
        f"Todas: {', '.join(f'{b} {v:.4f}' for b, v in valores.items())}."
    )


# ---------------------------------------------------------------------------
# 6.6 · Separación (muestra)
# ---------------------------------------------------------------------------


def test_silhouette_en_train_y_la_propia_igual_a_sklearn(Z, particion, identidad, mod_metricas):
    silhouette = atributo(mod_metricas, "silhouette")
    train = particion == "train"
    esperada = silhouette_sk(Z[train], identidad[train])
    propia = silhouette(Z[train], identidad[train])
    assert esperada >= SILHOUETTE_MIN, (
        f"6.6 · silhouette de Z en las 72 de train: {esperada:.4f}; se exige >= {SILHOUETTE_MIN}."
    )
    assert abs(propia - esperada) <= TOL_SILHOUETTE, (
        f"6.6 · silhouette propia en train {propia!r}; sklearn {esperada!r} (tolerancia {TOL_SILHOUETTE:g})."
    )


def test_silhouette_en_val(Z, particion, identidad):
    val = particion == "val"
    valor = silhouette_sk(Z[val], identidad[val])
    assert valor >= SILHOUETTE_MIN, f"6.6 · silhouette de Z en las 24 de val: {valor:.4f}; se exige >= {SILHOUETTE_MIN}."


@pytest.mark.parametrize("bloque", BLOQUES_CONTRATO)
def test_silhouette_de_cada_bloque_en_val(Z, tramos, particion, identidad, bloque):
    val = particion == "val"
    valores = {b: silhouette_sk(Z[val][:, tramos[b]], identidad[val]) for b in BLOQUES_CONTRATO}
    assert valores[bloque] >= SILHOUETTE_BLOQUE_MIN, (
        f"6.6 · silhouette del bloque {bloque} en las 24 de val: {valores[bloque]:.4f}; se exige >= {SILHOUETTE_BLOQUE_MIN}. "
        f"Todas: {', '.join(f'{b} {v:.4f}' for b, v in valores.items())}."
    )


@pytest.mark.parametrize("n_componentes", [None, 3], ids=["k=None", "k=3"])
def test_pca_de_la_muestra_igual_a_sklearn(mod_pca, Z, particion, n_componentes):
    train = particion == "train"
    pca = ajustar_pca(mod_pca, Z[train], n_componentes)
    k = min(int(train.sum()) - 1, Z.shape[1]) if n_componentes is None else n_componentes
    _, P_sk = proyecciones_sklearn(Z[train], Z, k)
    dif, _ = comparar_proyecciones(pca.transformar(Z), P_sk)
    assert dif <= TOL_PCA, (
        f"6.6 · PCA({n_componentes}) ajustado con las 72 de train: las proyecciones de las 120 difieren de las de "
        f"sklearn (k = {k}) en {dif:.3e} salvo el signo (tolerancia {TOL_PCA:g})."
    )


# ---------------------------------------------------------------------------
# 6.7 · Cortes parciales (muestra y bases)
# ---------------------------------------------------------------------------


def test_densidad_de_minucias_con_corte_igual_que_sin_corte(apilados, identidad, con_corte):
    densidad = apilados["minucias"][:, 0] + apilados["minucias"][:, 1]
    minimo, maximo = DENSIDAD_RANGO
    lineas, fallan = [], []
    for k in IDENTIDADES:
        con = densidad[(identidad == k) & con_corte].mean()
        sin = densidad[(identidad == k) & ~con_corte].mean()
        cociente = con / sin
        lineas.append(f"identidad {k}: con corte {con:.4f}, sin corte {sin:.4f}, cociente {cociente:.4f}")
        if not minimo <= cociente <= maximo:
            fallan.append(k)
    assert not fallan, (
        f"6.7 · densidad de minucias (dens_t + dens_b) con corte / sin corte fuera de [{minimo}, {maximo}] en las "
        f"identidades {fallan}:\n{resumen(lineas)}"
    )


def test_distancia_al_centroide_con_corte(Z, identidad, particion, con_corte):
    train = particion == "train"
    lineas, fallan = [], []
    for k in IDENTIDADES:
        centroide = Z[(identidad == k) & train].mean(axis=0)
        distancia = np.linalg.norm(Z - centroide, axis=1)
        con = distancia[(identidad == k) & con_corte].mean()
        sin = distancia[(identidad == k) & ~con_corte].mean()
        lineas.append(f"identidad {k}: con corte {con:.4f}, sin corte {sin:.4f}, cociente {con / sin:.4f}")
        if not con <= DISTANCIA_CORTE_MAX * sin:
            fallan.append(k)
    assert not fallan, (
        f"6.7 · distancia media al centroide propio con corte > {DISTANCIA_CORTE_MAX} × la de sin corte en las "
        f"identidades {fallan}:\n{resumen(lineas)}"
    )


@pytest.fixture(scope="module")
def corte_directo(cfg, h1, h2, car, entradas_muestra, bases, escalador) -> list[SimpleNamespace]:
    """Por identidad y ángulo, la variante (12°, 1, (10, −8), corte máximo hacia a) escalada."""
    caracteristicas = atributo(car, "caracteristicas")
    fraccion = float(max(obtener(cfg, "huellas.variantes.corte_periferico")))
    casos = []
    for k in IDENTIDADES:
        entrada = next(e for e in entradas_muestra if e["identidad"] == k)
        for a in ANGULOS_CORTE:
            params = h1.ParametrosVariante(12.0, 1.0, 10.0, -8.0, 0.0, 0.0, fraccion, float(a), 0.0, 0.0)
            variante = h1.aplicar_variante(bases(entrada), params, np.random.default_rng(0))
            h = h2.preprocesar(variante.imagen, cfg)
            z = np.asarray(escalador.transformar(caracteristicas(h, cfg)), dtype=np.float64)
            casos.append(SimpleNamespace(
                identidad=k, angulo=a, z=z,
                aplicado=(float(variante.parametros.corte_fraccion), float(variante.parametros.corte_angulo_grados)),
            ))
    return casos


def test_corte_directo_queda_en_su_identidad(corte_directo, Z, identidad, particion):
    train = particion == "train"
    centroides = {k: Z[(identidad == k) & train].mean(axis=0) for k in IDENTIDADES}
    maximos = {k: float(np.linalg.norm(Z[(identidad == k) & train] - centroides[k], axis=1).max()) for k in IDENTIDADES}
    lineas, fallan = [], []
    for caso in corte_directo:
        d = {j: float(np.linalg.norm(caso.z - centroides[j])) for j in IDENTIDADES}
        propia = d[caso.identidad]
        mas_cerca = all(propia < d[j] for j in IDENTIDADES if j != caso.identidad)
        ok = mas_cerca and propia <= maximos[caso.identidad]
        lineas.append(
            f"identidad {caso.identidad}, corte {caso.aplicado[0]:g} hacia {caso.aplicado[1]:g}° (pedido {caso.angulo}°): "
            f"distancias {', '.join(f'{j}: {v:.3f}' for j, v in d.items())}; máximo de train {maximos[caso.identidad]:.3f}"
            f"{'' if ok else '  ← falla'}"
        )
        if not ok:
            fallan.append(f"{caso.identidad}/{caso.angulo}°")
    assert not fallan, (
        f"6.7 · corte directo: {len(fallan)} de {len(corte_directo)} no quedan más cerca de su centroide que de los "
        f"otros dos a una distancia no mayor que la variante de train más lejana ({', '.join(fallan)}):\n"
        f"{resumen(lineas, maximo=12)}"
    )


# ---------------------------------------------------------------------------
# 6.8 · Archivo de características y figura (dataset chico en tmp_path)
# ---------------------------------------------------------------------------


def elegir_dataset_chico(plan) -> tuple[list[dict], dict]:
    """Entradas del dataset chico (en el orden del plan) y, por nombre de val, la entrada de su reemplazo sin fuga."""
    elegidas, reemplazos = [], {}
    for k in IDENTIDADES:
        train = [e for e in plan if e["identidad"] == k and e["particion"] == "train"][:TRAIN_CHICO]
        val = [e for e in plan if e["identidad"] == k and e["particion"] == "val"][:2 * VAL_CHICO]
        elegidas += train + val[:VAL_CHICO]
        reemplazos.update({e["archivo"]: otra for e, otra in zip(val[:VAL_CHICO], val[VAL_CHICO:])})
    intrusos = [e for e in plan if e["intruso"] in INTRUSOS_CHICO]
    elegidas += intrusos
    orden = {e["archivo"]: i for i, e in enumerate(plan)}
    return sorted(elegidas, key=lambda e: orden[e["archivo"]]), reemplazos


@pytest.fixture(scope="module")
def dataset_chico(tmp_path_factory, cfg, h1, plan, bases) -> SimpleNamespace:
    """Los PNG y metadatos.csv del dataset chico de 6.8, más las imágenes de la carpeta sin fuga."""
    entradas, reemplazos = elegir_dataset_chico(plan)
    intrusos_fuera_de_val = [e["archivo"] for e in entradas if e["intruso"] and e["particion"] != "val"]
    if intrusos_fuera_de_val:
        pytest.fail(f"En el plan, los intrusos 1 y 2 no están en val: {intrusos_fuera_de_val}.", pytrace=False)
    carpeta = tmp_path_factory.mktemp("h3_chico")
    imagenes = {}
    for e in entradas:
        imagenes[e["archivo"]] = receta_h1(h1, cfg, e, bases(e)).imagen
        escribir_png(carpeta / e["archivo"], imagenes[e["archivo"]])
    escribir_metadatos(carpeta, entradas)
    otras_val = {nombre: receta_h1(h1, cfg, otra, bases(otra)).imagen for nombre, otra in reemplazos.items()}
    return SimpleNamespace(carpeta=carpeta, entradas=entradas, imagenes=imagenes, otras_val=otras_val)


@pytest.fixture(scope="module")
def generado_chico(car, cfg, dataset_chico) -> SimpleNamespace:
    generar_caracteristicas = atributo(car, "generar_caracteristicas")
    carpeta = dataset_chico.carpeta
    antes = listado(carpeta)
    try:
        ruta = generar_caracteristicas(cfg, carpeta)
    except Exception as error:
        pytest.fail(f"generar_caracteristicas(cfg, carpeta) lanzó {descripcion_error(error)}", pytrace=False)
    despues = listado(carpeta)
    npz = carpeta / "caracteristicas.npz"
    return SimpleNamespace(ruta=ruta, antes=antes, despues=despues, datos=leer_npz(npz) if npz.is_file() else None)


def test_generar_caracteristicas_devuelve_la_ruta_y_no_crea_otros_archivos(dataset_chico, generado_chico):
    esperada = dataset_chico.carpeta / "caracteristicas.npz"
    problemas = []
    if generado_chico.ruta is None or Path(generado_chico.ruta).resolve() != esperada.resolve():
        problemas.append(f"devuelve {generado_chico.ruta!r}; se espera {esperada}")
    nuevos = sorted(set(generado_chico.despues) - set(generado_chico.antes))
    borrados = sorted(set(generado_chico.antes) - set(generado_chico.despues))
    if nuevos != ["caracteristicas.npz"] or borrados:
        problemas.append(f"archivos nuevos {nuevos} y borrados {borrados}; se espera solo caracteristicas.npz nuevo")
    assert not problemas, f"6.8 · generar_caracteristicas:\n{resumen(problemas)}"


def test_npz_con_las_claves_y_las_filas_de_metadatos(dataset_chico, generado_chico):
    datos = generado_chico.datos
    assert datos is not None, "6.8 · no se escribió caracteristicas.npz."
    entradas = dataset_chico.entradas
    n = len(entradas)
    problemas = []
    if set(datos) != set(CLAVES_NPZ):
        problemas.append(f"claves {sorted(datos)}; se esperan exactamente {sorted(CLAVES_NPZ)}")
    for clave in ("archivo", "particion", "tipo"):
        if clave in datos:
            v = datos[clave]
            if v.dtype.kind != "U" or v.shape != (n,):
                problemas.append(f"{clave} es {v.dtype} de forma {v.shape}; se espera un array de texto de forma ({n},)")
            elif v.tolist() != [str(e[clave]) for e in entradas]:
                malas = [i for i, e in enumerate(entradas) if v[i] != str(e[clave])]
                problemas.append(f"{clave} no coincide con metadatos.csv en las filas {malas[:5]} (fila {malas[0]}: {v[malas[0]]!r})")
    for clave in ("identidad", "intruso"):
        if clave in datos:
            v = datos[clave]
            if v.dtype.kind not in "iu" or v.shape != (n,):
                problemas.append(f"{clave} es {v.dtype} de forma {v.shape}; se espera un array de enteros de forma ({n},)")
            elif v.tolist() != [int(e[clave]) for e in entradas]:
                problemas.append(f"{clave} = {v.tolist()}; metadatos.csv tiene {[int(e[clave]) for e in entradas]}")
    for clave in ("orientacion", "gabor", "minucias", "X"):
        if clave in datos:
            v = datos[clave]
            if v.dtype != np.float64 or v.ndim != 2 or v.shape[0] != n:
                problemas.append(f"{clave} es {v.dtype} de forma {v.shape}; se espera float64 de forma ({n}, d)")
    assert not problemas, f"6.8 · contenido de caracteristicas.npz:\n{resumen(problemas)}"


def test_npz_bloques_crudos_iguales_a_caracteristicas_del_png(car, cfg, h2, dataset_chico, generado_chico):
    caracteristicas = atributo(car, "caracteristicas")
    datos = generado_chico.datos
    assert datos is not None, "6.8 · no se escribió caracteristicas.npz."
    problemas = []
    for i, e in enumerate(dataset_chico.entradas):
        imagen = leer_png(dataset_chico.carpeta / e["archivo"])
        bloques = caracteristicas(h2.preprocesar(imagen, cfg), cfg)
        distintos = [b for b in BLOQUES_CONTRATO if b not in datos or not iguales_con_nan(datos[b][i], bloques[b])]
        if distintos:
            problemas.append(f"fila {i} ({e['archivo']}): bloques {distintos} distintos de caracteristicas(preprocesar(png))")
    exigir_sin_problemas(problemas, "6.8 · bloques crudos del npz", len(dataset_chico.entradas))


def test_npz_x_es_el_escalador_de_train_aplicado_a_todas_las_filas(car, generado_chico):
    ajustar_escalador = atributo(car, "ajustar_escalador")
    datos = generado_chico.datos
    assert datos is not None, "6.8 · no se escribió caracteristicas.npz."
    train = datos["particion"] == "train"
    escalador = ajustar_escalador({b: datos[b][train].copy() for b in BLOQUES_CONTRATO})
    esperado = np.asarray(escalador.transformar({b: datos[b].copy() for b in BLOQUES_CONTRATO}), dtype=np.float64)
    dif = max_dif(datos["X"], esperado)
    assert dif <= TOL_X, (
        f"6.8 · X del npz (forma {datos['X'].shape}) difiere de ajustar_escalador(filas de train).transformar(todas) "
        f"(forma {esperado.shape}) en {dif:.3e} (tolerancia {TOL_X:g})."
    )


def test_sin_fuga_las_filas_de_train_no_dependen_de_val(car, cfg, dataset_chico, generado_chico, tmp_path):
    generar_caracteristicas = atributo(car, "generar_caracteristicas")
    otra = tmp_path / "otra"
    for e in dataset_chico.entradas:
        imagen = dataset_chico.otras_val.get(e["archivo"], dataset_chico.imagenes[e["archivo"]])
        escribir_png(otra / e["archivo"], imagen)
    escribir_metadatos(otra, dataset_chico.entradas)
    generar_caracteristicas(cfg, otra)
    datos_otra = leer_npz(otra / "caracteristicas.npz")
    datos = generado_chico.datos
    assert datos is not None, "6.8 · no se escribió caracteristicas.npz en la carpeta original."
    train = datos["particion"] == "train"
    val_distintos = [b for b in BLOQUES_CONTRATO if not iguales_con_nan(datos[b][~train], datos_otra[b][~train])]
    assert val_distintos, "Control del test: las filas de val de las dos carpetas tendrían que ser distintas."
    assert np.array_equal(datos["X"][train], datos_otra["X"][train]), (
        f"6.8 · sin fuga: con otras imágenes de val, las filas de train de X cambian hasta "
        f"{max_dif(datos['X'][train], datos_otra['X'][train]):.3e} (se exige np.array_equal): el escalador usa filas de val."
    )


def test_npz_existente_sin_sobrescribir_da_file_exists_error(car, cfg, dataset_chico, generado_chico, tmp_path):
    generar_caracteristicas = atributo(car, "generar_caracteristicas")
    copia = tmp_path / "copia"
    shutil.copytree(dataset_chico.carpeta, copia)
    npz = copia / "caracteristicas.npz"
    assert npz.is_file(), "6.8 · la carpeta original no tiene caracteristicas.npz."
    contenido, fecha = npz.read_bytes(), npz.stat().st_mtime_ns
    lanza(FileExistsError, generar_caracteristicas, cfg, copia, que="generar_caracteristicas con el npz ya escrito")
    assert npz.read_bytes() == contenido and npz.stat().st_mtime_ns == fecha, (
        "6.8 · después del FileExistsError, caracteristicas.npz cambió (contenido o fecha de modificación)."
    )


def test_npz_existente_con_sobrescribir_lo_vuelve_a_escribir(car, cfg, dataset_chico, generado_chico, tmp_path):
    generar_caracteristicas = atributo(car, "generar_caracteristicas")
    copia = tmp_path / "copia"
    shutil.copytree(dataset_chico.carpeta, copia)
    npz = copia / "caracteristicas.npz"
    npz.write_bytes(b"esto no es un npz")
    ruta = generar_caracteristicas(cfg, copia, sobrescribir=True)
    assert Path(ruta).resolve() == npz.resolve(), f"6.8 · con sobrescribir=True devuelve {ruta!r}; se espera {npz}."
    distintos = npz_iguales(leer_npz(npz), generado_chico.datos)
    assert not distintos, f"6.8 · con sobrescribir=True el npz reescrito difiere del original en {distintos}."


def test_sin_filas_de_train_value_error_sin_escribir(car, cfg, dataset_chico, tmp_path):
    generar_caracteristicas = atributo(car, "generar_caracteristicas")
    carpeta = tmp_path / "sin_train"
    entradas = [e for e in dataset_chico.entradas if e["particion"] != "train"]
    for e in entradas:
        escribir_png(carpeta / e["archivo"], dataset_chico.imagenes[e["archivo"]])
    escribir_metadatos(carpeta, entradas)
    lanza(ValueError, generar_caracteristicas, cfg, carpeta, que="generar_caracteristicas sin filas de train")
    assert not (carpeta / "caracteristicas.npz").exists(), "6.8 · sin filas de train se escribió caracteristicas.npz."


def test_generar_figura_escribe_un_png_legible(car, cfg, dataset_chico, generado_chico, tmp_path):
    generar_figura = atributo(car, "generar_figura")
    salida = tmp_path / "pca_h3.png"
    resultado = generar_figura(cfg, dataset_chico.carpeta, salida)
    assert Path(resultado).resolve() == salida.resolve(), f"generar_figura devolvió {resultado!r}; se espera {salida}."
    assert salida.is_file(), f"generar_figura no escribió {salida}."
    contenido = salida.read_bytes()
    assert contenido[:8] == b"\x89PNG\r\n\x1a\n", f"{salida} no es un PNG (empieza con {contenido[:8]!r})."
    # Decodifica igual que cv2.imread(ruta) y no falla con tildes en la ruta (Windows).
    imagen = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert imagen is not None and imagen.size > 0, f"OpenCV no pudo leer {salida}."


# ---------------------------------------------------------------------------
# 6.9 · Aceptación (dataset completo de H1 en tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_completo(tmp_path_factory, cfg, h1, car) -> SimpleNamespace:
    generar_caracteristicas = atributo(car, "generar_caracteristicas")
    carpeta = tmp_path_factory.mktemp("h3_completo")
    h1.generar_dataset(cfg, carpeta)
    inicio = time.perf_counter()
    ruta = generar_caracteristicas(cfg, carpeta)
    tiempo = time.perf_counter() - inicio
    return SimpleNamespace(carpeta=carpeta, ruta=ruta, tiempo_s=tiempo, datos=leer_npz(carpeta / "caracteristicas.npz"))


def _train_y_val_autorizadas(datos) -> tuple[np.ndarray, np.ndarray]:
    autorizadas = (datos["identidad"] > 0) & (datos["intruso"] == 0)
    return datos["particion"] == "train", (datos["particion"] == "val") & autorizadas


@pytest.mark.aceptacion
def test_aceptacion_generar_caracteristicas_tarda_menos_de_3_minutos(dataset_completo):
    assert dataset_completo.tiempo_s < TIEMPO_ACEPTACION_S, (
        f"6.9 · generar_caracteristicas del dataset completo tardó {dataset_completo.tiempo_s:.1f} s; "
        f"se exige menos de {TIEMPO_ACEPTACION_S:g} s."
    )


@pytest.mark.aceptacion
def test_aceptacion_npz_con_300_filas_y_x_finito(dataset_completo):
    datos = dataset_completo.datos
    X = datos["X"]
    filas = {clave: datos[clave].shape[0] for clave in datos}
    assert all(v == 300 for v in filas.values()) and np.isfinite(X).all(), (
        f"6.9 · filas por clave {filas} (se esperan 300) y {int((~np.isfinite(X)).sum())} valores no finitos en X."
    )


@pytest.mark.aceptacion
def test_aceptacion_silhouette_en_train_y_en_val(dataset_completo):
    datos = dataset_completo.datos
    train, val = _train_y_val_autorizadas(datos)
    s_train = silhouette_sk(datos["X"][train], datos["identidad"][train])
    s_val = silhouette_sk(datos["X"][val], datos["identidad"][val])
    assert int(train.sum()) == 72 and int(val.sum()) == 24, (
        f"6.9 · hay {int(train.sum())} filas de train y {int(val.sum())} autorizadas de val; se esperan 72 y 24."
    )
    assert s_train >= SILHOUETTE_MIN and s_val >= SILHOUETTE_MIN, (
        f"6.9 · silhouette de X: train {s_train:.4f} y val {s_val:.4f}; se exige >= {SILHOUETTE_MIN} en las dos."
    )


@pytest.mark.aceptacion
@pytest.mark.parametrize("bloque", BLOQUES_CONTRATO)
def test_aceptacion_contribucion_de_cada_bloque(dataset_completo, bloque):
    datos = dataset_completo.datos
    train, _ = _train_y_val_autorizadas(datos)
    conteos = {b: int(columnas_usadas_por_definicion(datos[b][train]).sum()) for b in BLOQUES_CONTRATO}
    tramos = tramos_por_bloque(conteos)
    valores = {b: contribucion(datos["X"][train][:, tramos[b]]) for b in BLOQUES_CONTRATO}
    minimo, maximo = CONTRIBUCION_RANGO
    assert minimo <= valores[bloque] <= maximo, (
        f"6.9 · contribución del bloque {bloque} en train: {valores[bloque]:.4f}; se exige [{minimo}, {maximo}]. "
        f"Todas: {', '.join(f'{b} {v:.4f}' for b, v in valores.items())} (columnas usadas {conteos})."
    )


@pytest.mark.aceptacion
@pytest.mark.parametrize("n_componentes", [None, 3], ids=["k=None", "k=3"])
def test_aceptacion_pca_igual_a_sklearn(dataset_completo, mod_pca, n_componentes):
    datos = dataset_completo.datos
    train, _ = _train_y_val_autorizadas(datos)
    X = datos["X"]
    pca = ajustar_pca(mod_pca, X[train], n_componentes)
    k = min(int(train.sum()) - 1, X.shape[1]) if n_componentes is None else n_componentes
    _, P_sk = proyecciones_sklearn(X[train], X, k)
    dif, _ = comparar_proyecciones(pca.transformar(X), P_sk)
    assert dif <= TOL_PCA, (
        f"6.9 · PCA({n_componentes}) ajustado con train: las proyecciones de las 300 filas difieren de las de sklearn "
        f"(k = {k}) en {dif:.3e} salvo el signo (tolerancia {TOL_PCA:g})."
    )


@pytest.mark.aceptacion
def test_aceptacion_generar_figura(dataset_completo, car, cfg):
    generar_figura = atributo(car, "generar_figura")
    salida = dataset_completo.carpeta / "pca_h3.png"
    resultado = generar_figura(cfg, dataset_completo.carpeta, salida)
    assert Path(resultado).resolve() == salida.resolve(), f"generar_figura devolvió {resultado!r}; se espera {salida}."
    contenido = salida.read_bytes() if salida.is_file() else b""
    imagen = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_UNCHANGED) if contenido else None
    assert imagen is not None and imagen.size > 0, f"6.9 · generar_figura no escribió un PNG legible en {salida}."
