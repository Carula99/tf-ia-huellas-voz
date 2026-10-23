"""H3 · Características de huellas por bloques: orientación, Gabor y minucias.

Contrato: docs/contratos/h3_caracteristicas.md (versión 1.1). A partir de la ventana de H2
(HuellaPreprocesada), caracteristicas() arma tres bloques crudos, con NaN donde falta huella:

1. orientación: (cos 2θ, sen 2θ) medios en una grilla de 5 × 5 celdas (50 valores);
2. Gabor: 12 filtros (6 orientaciones × períodos de 8 y 10 px); la media del valor absoluto de
   la respuesta en 3 anillos × 8 sectores alrededor del punto de referencia, lejos del borde de
   la máscara, y la energía global de cada filtro normalizada (300 valores);
3. minucias por crossing number, sin las que están cerca del borde de la máscara (el borde
   recto de los cortes y el de la ventana), las espuelas, los pares cercanos ni las componentes
   chicas del esqueleto: densidades globales, por anillo y en una grilla de 4 × 4 (25 valores).

Toda celda, sector o anillo con menos del 50 % de huella vale NaN. El escalador, ajustado solo
con train, estandariza cada columna, imputa los NaN con la media de train (0 estandarizado),
descarta las columnas sin datos y pesa cada bloque con 1/√columnas, así los tres aportan lo
mismo a la distancia. Sin los momentos de Hu (decisión 46).

Línea de comandos, desde la raíz del repo (escribe data/huellas/caracteristicas.npz y
docs/figuras/pca_h3.png):
    venv\\Scripts\\python -m src.huellas.caracteristicas [--sobrescribir]
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass
from functools import lru_cache
import io
import math
import os
from pathlib import Path
import sys
import time
import warnings

import cv2
import numpy as np
from scipy import ndimage

from ..config import RAIZ, cargar_config
from ..core_ml.metricas import silhouette
from ..core_ml.pca import PCA
from .preprocesamiento import HuellaPreprocesada, preprocesar

BLOQUES: tuple[str, ...] = ("orientacion", "gabor", "minucias")

DESVIO_MIN = 1e-9
"""Las columnas con desvío de train menor o igual se descartan (4.5)."""

COLUMNAS_METADATOS: tuple[str, ...] = ("archivo", "particion", "identidad", "intruso", "tipo")
NOMBRE_NPZ = "caracteristicas.npz"
NOMBRE_FIGURA = "pca_h3.png"

# Vecinos en orden circular P_1 … P_8 (arriba, arriba-derecha, derecha, abajo-derecha, abajo,
# abajo-izquierda, izquierda, arriba-izquierda), como (Δfila, Δcolumna).
_CIRCULO = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


@dataclass(frozen=True)
class Minucias:
    """Salida de extraer_minucias: (x, y) en la ventana, ordenadas por y y, a igual y, por x."""

    terminaciones: np.ndarray    # float64 (k_t, 2)
    bifurcaciones: np.ndarray    # float64 (k_b, 2)


# --------------------------------------------------------------------------- geometría (sección 0)
@dataclass(frozen=True)
class _Geometria:
    ventana: tuple[int, int]                 # (alto, ancho)
    centro: tuple[float, float]              # (x, y) = (124.5, 124.5)
    celdas_orientacion: tuple                # (filas, columnas) como slices, por filas
    zonas_gabor: tuple                       # bool (alto, ancho) por anillo × sector, el anillo afuera
    anillos_minucias: tuple                  # bool (alto, ancho) por anillo
    celdas_minucias: tuple                   # (filas, columnas) como slices, por filas


def _bordes(lado: int, g: int) -> np.ndarray:
    return np.round(np.linspace(0, lado, g + 1)).astype(int)


def _celdas(alto: int, ancho: int, g: int) -> tuple:
    filas, columnas = _bordes(alto, g), _bordes(ancho, g)
    return tuple((slice(int(filas[i]), int(filas[i + 1])), slice(int(columnas[j]), int(columnas[j + 1])))
                 for i in range(g) for j in range(g))


@lru_cache(maxsize=8)
def _geometria(ventana: tuple[int, int], grilla_o: int, anillos_g: tuple, sectores_g: int,
               anillos_m: tuple, grilla_m: int) -> _Geometria:
    alto, ancho = ventana
    cx, cy = (ancho - 1) / 2, (alto - 1) / 2
    yy, xx = np.mgrid[0:alto, 0:ancho]
    r = np.hypot(xx - cx, yy - cy)
    fi = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
    sector = np.minimum(np.floor(fi * sectores_g / (2 * np.pi)).astype(int), sectores_g - 1)

    def anillos(radios: tuple) -> tuple:
        return tuple((r >= radios[a]) & (r < radios[a + 1]) for a in range(len(radios) - 1))

    zonas = tuple(anillo & (sector == s) for anillo in anillos(anillos_g) for s in range(sectores_g))
    return _Geometria(
        ventana=(alto, ancho), centro=(cx, cy),
        celdas_orientacion=_celdas(alto, ancho, grilla_o),
        zonas_gabor=zonas,
        anillos_minucias=anillos(anillos_m),
        celdas_minucias=_celdas(alto, ancho, grilla_m),
    )


def _parametros(cfg: dict) -> dict:
    return cfg["huellas"]["caracteristicas"]


def _ventana(cfg: dict) -> tuple[int, int]:
    alto, ancho = cfg["huellas"]["ventana_px"]
    return int(alto), int(ancho)


def _geometria_de(cfg: dict) -> _Geometria:
    c = _parametros(cfg)
    return _geometria(
        _ventana(cfg), int(c["grilla_orientacion"]), tuple(float(v) for v in c["anillos_gabor_px"]),
        int(c["sectores_gabor"]), tuple(float(v) for v in c["anillos_minucias_px"]), int(c["grilla_minucias"]),
    )


@lru_cache(maxsize=8)
def _filtros_gabor(periodos: tuple, orientaciones: int) -> tuple:
    """4.2: para cada período y cada k, el filtro de media 0 y Σ|g| = 1 que responde a π·k/K."""
    filtros = []
    for periodo in periodos:
        sigma = periodo / 2
        lado = 2 * int(round(3 * sigma)) + 1
        for k in range(orientaciones):
            g = cv2.getGaborKernel((lado, lado), sigma, math.pi * k / orientaciones + math.pi / 2, periodo,
                                   1.0, 0.0, ktype=cv2.CV_64F)
            g = g - g.mean()
            filtros.append(g / np.abs(g).sum())
    return tuple(filtros)


def _filtros_de(cfg: dict) -> tuple:
    c = _parametros(cfg)
    return _filtros_gabor(tuple(float(v) for v in c["gabor_periodos_px"]), int(c["gabor_orientaciones"]))


def _distancia_al_borde(mascara: np.ndarray) -> np.ndarray:
    """Distancia euclídea al píxel más cercano fuera de la máscara; fuera de la ventana cuenta como fuera."""
    return ndimage.distance_transform_edt(np.pad(mascara, 1))[1:-1, 1:-1]


def largos_de_bloque(cfg: dict) -> dict[str, int]:
    """Largo de cada bloque crudo: 2·g², F·R·S + F y 3 + 2·R_m + g_m²."""
    c = _parametros(cfg)
    F = len(c["gabor_periodos_px"]) * int(c["gabor_orientaciones"])
    R = len(c["anillos_gabor_px"]) - 1
    R_m = len(c["anillos_minucias_px"]) - 1
    return {
        "orientacion": 2 * int(c["grilla_orientacion"]) ** 2,
        "gabor": F * R * int(c["sectores_gabor"]) + F,
        "minucias": 3 + 2 * R_m + int(c["grilla_minucias"]) ** 2,
    }


# --------------------------------------------------------------------------- minucias (4.3)
def _esqueleto_limpio(esqueleto: np.ndarray, componente_min: int) -> np.ndarray:
    """El esqueleto sin sus componentes conexas (vecinos-8) de menos de componente_min píxeles."""
    etiquetas, n = ndimage.label(esqueleto, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return esqueleto.copy()
    conservar = np.bincount(etiquetas.ravel()) >= componente_min
    conservar[0] = False
    return conservar[etiquetas]


def _crossing_number(E: np.ndarray) -> np.ndarray:
    """CN = ½ Σ |P_k − P_{k+1}| de cada píxel; los vecinos fuera de la ventana valen 0."""
    alto, ancho = E.shape
    P = np.pad(E.astype(np.int16), 1)
    vecinos = [P[1 + di:1 + di + alto, 1 + dj:1 + dj + ancho] for di, dj in _CIRCULO]
    return sum(np.abs(vecinos[k] - vecinos[(k + 1) % 8]) for k in range(8)) // 2


def _uniones_alcanzadas(t: tuple[int, int], E: np.ndarray, union: np.ndarray, pasos: int) -> set:
    """Búsqueda en anchura desde t por E (vecinos-8), hasta `pasos` pasos: alcanza las uniones
    pero no sigue desde ellas; las otras terminaciones se atraviesan (sección 8)."""
    alto, ancho = E.shape
    distancia = {t: 0}
    cola = deque([t])
    alcanzadas = set()
    while cola:
        p = cola.popleft()
        if distancia[p] >= pasos:
            continue
        for di, dj in _CIRCULO:
            q = (p[0] + di, p[1] + dj)
            if q in distancia or not (0 <= q[0] < alto and 0 <= q[1] < ancho) or not E[q]:
                continue
            distancia[q] = distancia[p] + 1
            if union[q]:
                alcanzadas.add(q)
            else:
                cola.append(q)
    return alcanzadas


def _sin_pares(puntos: list, umbral: float) -> list:
    """Saca los puntos que tienen otro a distancia euclídea < umbral (los dos del par)."""
    if len(puntos) < 2:
        return list(puntos)
    P = np.array(puntos, dtype=np.float64)
    dif = P[:, None, :] - P[None, :, :]
    d = np.sqrt((dif ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    cerca = (d < umbral).any(axis=1)
    return [p for p, c in zip(puntos, cerca) if not c]


def _como_xy(puntos: list) -> np.ndarray:
    """(fila, columna) → (x, y) = (j, i) en float64, ordenados por y y, a igual y, por x."""
    if not puntos:
        return np.zeros((0, 2), dtype=np.float64)
    xy = np.array([(j, i) for i, j in puntos], dtype=np.float64)
    return xy[np.lexsort((xy[:, 0], xy[:, 1]))]


def _minucias(esqueleto: np.ndarray, V: np.ndarray, c: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pasos 1 a 5 de 4.3 con la zona válida V = D(mascara) ≥ borde_minucia_px."""
    E = _esqueleto_limpio(esqueleto, int(c["componente_min_px"]))
    CN = _crossing_number(E)
    T0 = [(int(i), int(j)) for i, j in np.argwhere(E & (CN == 1) & V)]
    B0 = [(int(i), int(j)) for i, j in np.argwhere(E & (CN == 3) & V)]
    union = E & (CN >= 3)

    espuelas, uniones = set(), set()
    for t in T0:
        alcanzadas = _uniones_alcanzadas(t, E, union, int(c["espuela_px"]))
        if alcanzadas:
            espuelas.add(t)
            uniones |= alcanzadas
    T1 = [t for t in T0 if t not in espuelas]
    if uniones:
        U = np.array(sorted(uniones))
        B1 = [b for b in B0 if not np.any((np.abs(U[:, 0] - b[0]) <= 2) & (np.abs(U[:, 1] - b[1]) <= 2))]
    else:
        B1 = B0

    T2 = _sin_pares(T1, float(c["par_terminaciones_px"]))
    B2 = _sin_pares(B1, float(c["par_bifurcaciones_px"]))
    return _como_xy(T2), _como_xy(B2)


def _comprobar_ventana(h: HuellaPreprocesada, campos: tuple[str, ...], ventana: tuple[int, int]) -> None:
    for campo in campos:
        forma = np.shape(getattr(h, campo))
        if forma != ventana:
            raise ValueError(f"{campo} tiene forma {forma}; se espera {ventana} (huellas.ventana_px).")


def extraer_minucias(huella: HuellaPreprocesada, cfg: dict) -> Minucias:
    """Terminaciones y bifurcaciones del esqueleto según 4.3, lejos del borde de la máscara."""
    _comprobar_ventana(huella, ("mascara", "esqueleto"), _ventana(cfg))
    c = _parametros(cfg)
    V = _distancia_al_borde(np.asarray(huella.mascara, dtype=bool)) >= c["borde_minucia_px"]
    T, B = _minucias(np.asarray(huella.esqueleto, dtype=bool), V, c)
    return Minucias(terminaciones=T, bifurcaciones=B)


# --------------------------------------------------------------------------- bloques (4.1, 4.2 y 4.4)
def _bloque_orientacion(mascara: np.ndarray, orientacion: np.ndarray, geo: _Geometria, cmin: float) -> np.ndarray:
    """4.1: [media de cos 2θ, media de sen 2θ] dentro de la máscara en cada celda, o NaN."""
    cos2, sen2 = np.cos(2 * orientacion), np.sin(2 * orientacion)
    valores = np.full(2 * len(geo.celdas_orientacion), np.nan)
    for n, (filas, columnas) in enumerate(geo.celdas_orientacion):
        dentro = mascara[filas, columnas]
        if np.count_nonzero(dentro) < cmin * dentro.size or not dentro.any():
            continue
        valores[2 * n] = cos2[filas, columnas][dentro].mean()
        valores[2 * n + 1] = sen2[filas, columnas][dentro].mean()
    return valores


def _bloque_gabor(realzada: np.ndarray, mascara: np.ndarray, G: np.ndarray, geo: _Geometria,
                  filtros: tuple, cmin: float) -> np.ndarray:
    """4.2: media de |respuesta| en cada anillo × sector dentro de G, para cada filtro, y la
    energía global de cada filtro dividida por la suma (los F globales suman 1)."""
    F, zonas = len(filtros), geo.zonas_gabor
    if not mascara.any() or not G.any():
        return np.full(F * len(zonas) + F, np.nan)
    f = 255.0 - np.asarray(realzada).astype(np.float64)
    mu = f[mascara].mean()
    f = np.where(mascara, f, mu) - mu
    indices = []
    for zona in zonas:
        dentro = zona & G
        cuenta = np.count_nonzero(dentro)
        indices.append(None if cuenta < cmin * np.count_nonzero(zona) or cuenta == 0 else np.flatnonzero(dentro))
    en_G = np.flatnonzero(G)
    valores = np.full(F * len(zonas), np.nan)
    medias = np.empty(F)
    for n, g in enumerate(filtros):
        respuesta = np.abs(cv2.filter2D(f, -1, g, borderType=cv2.BORDER_REFLECT)).ravel()
        for z, idx in enumerate(indices):
            if idx is not None:
                valores[n * len(zonas) + z] = respuesta[idx].mean()
        medias[n] = respuesta[en_G].mean()
    total = medias.sum()
    globales = medias / total if total != 0 else np.full(F, np.nan)
    return np.concatenate([valores, globales])


def _bloque_minucias(V: np.ndarray, T: np.ndarray, B: np.ndarray, geo: _Geometria, c: dict) -> np.ndarray:
    """4.4: densidades (minucias cada 1000 px² de zona válida) globales, por anillo y en la grilla."""
    cmin = float(c["cobertura_min"])
    radios = [float(v) for v in c["anillos_minucias_px"]]
    n_v = np.count_nonzero(V)
    n_t, n_b = len(T), len(B)
    if n_v == 0:
        valores = [np.nan, np.nan, np.nan]
    else:
        valores = [1000 * n_t / n_v, 1000 * n_b / n_v, n_b / (n_t + n_b) if n_t + n_b else 0.0]
    cx, cy = geo.centro
    for puntos in (T, B):
        r = np.hypot(puntos[:, 0] - cx, puntos[:, 1] - cy)
        for a, anillo in enumerate(geo.anillos_minucias):
            dentro = np.count_nonzero(anillo & V)
            if dentro < cmin * np.count_nonzero(anillo) or dentro == 0:
                valores.append(np.nan)
            else:
                valores.append(1000 * np.count_nonzero((r >= radios[a]) & (r < radios[a + 1])) / dentro)
    todas = np.vstack([T, B])
    x, y = todas[:, 0], todas[:, 1]
    for filas, columnas in geo.celdas_minucias:
        zona = V[filas, columnas]
        dentro = np.count_nonzero(zona)
        if dentro < cmin * zona.size or dentro == 0:
            valores.append(np.nan)
        else:
            en_celda = (x >= columnas.start) & (x < columnas.stop) & (y >= filas.start) & (y < filas.stop)
            valores.append(1000 * np.count_nonzero(en_celda) / dentro)
    return np.array(valores, dtype=np.float64)


def caracteristicas(huella: HuellaPreprocesada, cfg: dict) -> dict[str, np.ndarray]:
    """Los 3 bloques crudos (claves BLOQUES), float64 de 1 dimensión, con NaN donde falta huella.

    Raises:
        ValueError: si realzada, mascara, esqueleto u orientacion no son de la ventana (250 × 250).
    """
    ventana = _ventana(cfg)
    _comprobar_ventana(huella, ("realzada", "mascara", "esqueleto", "orientacion"), ventana)
    c = _parametros(cfg)
    geo = _geometria_de(cfg)
    cmin = float(c["cobertura_min"])
    mascara = np.asarray(huella.mascara, dtype=bool)
    D = _distancia_al_borde(mascara)
    G = D >= c["erosion_gabor_px"]
    V = D >= c["borde_minucia_px"]
    T, B = _minucias(np.asarray(huella.esqueleto, dtype=bool), V, c)
    orientacion = np.asarray(huella.orientacion, dtype=np.float64)
    return {
        "orientacion": _bloque_orientacion(mascara, orientacion, geo, cmin),
        "gabor": _bloque_gabor(huella.realzada, mascara, G, geo, _filtros_de(cfg), cmin),
        "minucias": _bloque_minucias(V, T, B, geo, c),
    }


# --------------------------------------------------------------------------- escalador (4.5)
def _comprobar_claves(bloques, que: str) -> None:
    if not isinstance(bloques, dict) or set(bloques) != set(BLOQUES):
        claves = sorted(bloques) if isinstance(bloques, dict) else type(bloques).__name__
        raise ValueError(f"{que}: se espera un dict con exactamente las claves {list(BLOQUES)}; tiene {claves}.")


@dataclass(frozen=True)
class Escalador:
    """Estandarización por columna con las estadísticas de train, imputación y peso por bloque."""

    medias: dict[str, np.ndarray]     # (d_b,) por bloque
    desvios: dict[str, np.ndarray]    # (d_b,) por bloque
    columnas: dict[str, np.ndarray]   # bool (d_b,): columnas que se usan
    pesos: dict[str, float]           # 1 / √(columnas usadas del bloque)

    def transformar(self, bloques: dict[str, np.ndarray]) -> np.ndarray:
        """Un vector (d_b,) o un array (n, d_b) por bloque → (d,) o (n, d), siempre finito.

        z = (x − media) / desvio en las columnas usadas, los NaN pasan a 0 (la media de train) y
        se multiplica por el peso del bloque; los bloques van uno al lado del otro, en el orden
        de BLOQUES. Cada fila depende solo de sí misma.
        """
        _comprobar_claves(bloques, "transformar")
        arrays = {b: np.asarray(bloques[b], dtype=np.float64) for b in BLOQUES}
        dimensiones = {a.ndim for a in arrays.values()}
        if len(dimensiones) != 1 or not dimensiones <= {1, 2}:
            raise ValueError("Los bloques tienen que ser todos vectores (d_b,) o todos arrays (n, d_b).")
        uno = arrays[BLOQUES[0]].ndim == 1
        if not uno and len({a.shape[0] for a in arrays.values()}) != 1:
            raise ValueError("Los bloques tienen distinta cantidad de filas.")
        for b, a in arrays.items():
            d_b = self.columnas[b].shape[0]
            if a.shape[-1] != d_b:
                raise ValueError(f"El bloque {b} tiene {a.shape[-1]} valores por fila; se esperan {d_b}.")
            if np.isinf(a).any():
                raise ValueError(f"El bloque {b} tiene valores ±Inf.")
        partes = []
        for b in BLOQUES:
            usadas = self.columnas[b]
            z = (np.atleast_2d(arrays[b])[:, usadas] - self.medias[b][usadas]) / self.desvios[b][usadas]
            z[np.isnan(z)] = 0.0
            partes.append(z * self.pesos[b])
        X = np.hstack(partes)
        return X[0] if uno else X


def ajustar_escalador(bloques_train: dict[str, np.ndarray]) -> Escalador:
    """Media y desvío (ddof=1) de cada columna con sus valores no NaN de train.

    Una columna se usa si tiene al menos 2 valores y desvío > 1e-9; el peso del bloque es
    1/√(columnas usadas).

    Raises:
        ValueError: si faltan o sobran claves, si las formas no coinciden, si hay menos de 2
            filas, si hay ±Inf o si un bloque se queda sin columnas.
    """
    _comprobar_claves(bloques_train, "ajustar_escalador")
    arrays = {b: np.asarray(bloques_train[b], dtype=np.float64) for b in BLOQUES}
    for b, a in arrays.items():
        if a.ndim != 2:
            raise ValueError(f"El bloque {b} tiene que ser un array (n, d_b); tiene forma {a.shape}.")
    filas = {a.shape[0] for a in arrays.values()}
    if len(filas) != 1:
        raise ValueError(f"Los bloques tienen distinta cantidad de filas: {sorted(filas)}.")
    n = filas.pop()
    if n < 2:
        raise ValueError(f"Hacen falta al menos 2 filas de train para el escalador; hay {n}.")
    medias, desvios, columnas, pesos = {}, {}, {}, {}
    for b in BLOQUES:
        a = arrays[b]
        if np.isinf(a).any():
            raise ValueError(f"El bloque {b} tiene valores ±Inf.")
        cuenta = np.count_nonzero(~np.isnan(a), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # columnas sin datos: NaN
            media = np.nanmean(a, axis=0)
            desvio = np.nanstd(a, axis=0, ddof=1)
        usadas = (cuenta >= 2) & (desvio > DESVIO_MIN)
        if not usadas.any():
            raise ValueError(f"El bloque {b} no tiene ninguna columna con datos en train.")
        medias[b], desvios[b], columnas[b] = media, desvio, usadas
        pesos[b] = 1.0 / math.sqrt(int(usadas.sum()))
    return Escalador(medias=medias, desvios=desvios, columnas=columnas, pesos=pesos)


# --------------------------------------------------------------------------- archivo de características
def _leer_metadatos(ruta: Path) -> list[dict[str, str]]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta} (el metadatos.csv del dataset de H1).")
    with open(ruta, encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        faltan = [col for col in COLUMNAS_METADATOS if col not in (lector.fieldnames or [])]
        if faltan:
            raise ValueError(f"A {ruta} le faltan las columnas {faltan}.")
        return [{col: fila[col] for col in COLUMNAS_METADATOS} for fila in lector]


def _leer_png(ruta: Path) -> np.ndarray:
    """cv2.imread no acepta rutas con tildes en Windows: se lee el archivo y se decodifica."""
    imagen = cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_UNCHANGED)
    if imagen is None:
        raise ValueError(f"No se pudo leer {ruta} como imagen.")
    return imagen


def generar_caracteristicas(cfg: dict, carpeta: str | os.PathLike | None = None,
                            sobrescribir: bool = False) -> Path:
    """Calcula los bloques de cada huella del dataset de H1 y escribe carpeta / caracteristicas.npz.

    Lee metadatos.csv (archivo, particion, identidad, intruso y tipo) y cada PNG, aplica
    preprocesar y caracteristicas, ajusta el escalador solo con las filas de train y guarda los
    bloques crudos (con sus NaN) y X, los vectores escalados, con una fila por fila de
    metadatos.csv. Por defecto, carpeta = RAIZ / rutas.huellas. Devuelve la ruta escrita.

    Raises:
        FileExistsError: si el npz ya existe y sobrescribir es False (no toca nada).
        FileNotFoundError: si falta metadatos.csv o un PNG.
        ValueError: si no hay filas de train (no escribe nada).
    """
    carpeta = RAIZ / cfg["rutas"]["huellas"] if carpeta is None else Path(os.fspath(carpeta))
    salida = carpeta / NOMBRE_NPZ
    if salida.exists() and not sobrescribir:
        raise FileExistsError(f"{salida} ya existe; usá --sobrescribir (sobrescribir=True) para volver a escribirlo.")
    filas = _leer_metadatos(carpeta / "metadatos.csv")
    particion = np.array([f["particion"] for f in filas], dtype=str)
    train = particion == "train"
    if not train.any():
        raise ValueError(f"{carpeta / 'metadatos.csv'} no tiene filas de train: no se puede ajustar el escalador.")
    rutas = [carpeta / f["archivo"] for f in filas]
    faltan = [str(r) for r in rutas if not r.is_file()]
    if faltan:
        raise FileNotFoundError(f"Faltan {len(faltan)} imágenes del dataset; la primera es {faltan[0]}.")

    crudos: dict[str, list[np.ndarray]] = {b: [] for b in BLOQUES}
    for ruta in rutas:
        bloques = caracteristicas(preprocesar(_leer_png(ruta), cfg), cfg)
        for b in BLOQUES:
            crudos[b].append(bloques[b])
    apilados = {b: np.vstack(v) for b, v in crudos.items()}
    escalador = ajustar_escalador({b: apilados[b][train] for b in BLOQUES})
    X = escalador.transformar(apilados)

    np.savez(
        salida,
        archivo=np.array([f["archivo"] for f in filas], dtype=str),
        particion=particion,
        tipo=np.array([f["tipo"] for f in filas], dtype=str),
        identidad=np.array([int(f["identidad"]) for f in filas], dtype=np.int64),
        intruso=np.array([int(f["intruso"]) for f in filas], dtype=np.int64),
        **apilados,
        X=X,
    )
    return salida


# --------------------------------------------------------------------------- figura
_COLORES = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf")


def _coma(valor: float, decimales: int = 3) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def _silhouette_si_se_puede(X: np.ndarray, etiquetas: np.ndarray) -> float | None:
    """La silhouette propia, si hay al menos 2 grupos y más filas que grupos."""
    grupos = len(np.unique(etiquetas))
    if grupos < 2 or len(etiquetas) <= grupos:
        return None
    return silhouette(X, etiquetas)


def _con_corte(carpeta: Path, archivos: np.ndarray) -> np.ndarray | None:
    """Las filas con corte, si metadatos.csv tiene la columna corte_fraccion (la del dataset de H1)."""
    try:
        with open(carpeta / "metadatos.csv", encoding="utf-8-sig", newline="") as archivo:
            lector = csv.DictReader(archivo)
            if "corte_fraccion" not in (lector.fieldnames or []):
                return None
            fraccion = {fila["archivo"]: float(fila["corte_fraccion"] or 0.0) for fila in lector}
    except (OSError, ValueError):
        return None
    return np.array([fraccion.get(str(a), 0.0) > 0 for a in archivos], dtype=bool)


def generar_figura(cfg: dict, carpeta: str | os.PathLike | None = None,
                   salida: str | os.PathLike | None = None) -> Path:
    """PCA propio de carpeta / caracteristicas.npz, ajustado con las filas de train de X.

    Proyección 2D (PC1–PC2) y 3D (PC1–PC2–PC3) de todas las filas: las autorizadas con un color
    por identidad (train relleno, val y test sin relleno) y los intrusos en gris, con la varianza
    explicada de cada eje y la silhouette de train y de las autorizadas de val. Si metadatos.csv
    tiene corte_fraccion, las variantes con corte llevan un punto negro. Por defecto se escribe
    en RAIZ / rutas.figuras / "pca_h3.png"; devuelve la ruta escrita.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    carpeta = RAIZ / cfg["rutas"]["huellas"] if carpeta is None else Path(os.fspath(carpeta))
    salida = RAIZ / cfg["rutas"]["figuras"] / NOMBRE_FIGURA if salida is None else Path(os.fspath(salida))
    with np.load(carpeta / NOMBRE_NPZ, allow_pickle=False) as datos:
        X, particion, archivo = datos["X"], datos["particion"], datos["archivo"]
        identidad, intruso = datos["identidad"], datos["intruso"]
    train = particion == "train"
    autorizada = (identidad > 0) & (intruso == 0)
    k = min(3, int(train.sum()) - 1, X.shape[1])
    pca = PCA(k).ajustar(X[train])
    P = np.hstack([pca.transformar(X), np.zeros((len(X), 3 - k))])
    proporcion = list(pca.proporcion_varianza) + [0.0] * (3 - k)
    val = (particion == "val") & autorizada
    s_train = _silhouette_si_se_puede(X[train & autorizada], identidad[train & autorizada])
    s_val = _silhouette_si_se_puede(X[val], identidad[val])
    corte = _con_corte(carpeta, archivo)

    figura = Figure(figsize=(15, 6.6))
    FigureCanvasAgg(figura)
    ax2 = figura.add_subplot(1, 2, 1)
    ax3 = figura.add_subplot(1, 2, 2, projection="3d")
    estilos = (("train", "o", True), ("val", "o", False), ("test", "^", False))
    for n, ident in enumerate(sorted(set(identidad[autorizada].tolist()))):
        color = _COLORES[n % len(_COLORES)]
        for part, marcador, relleno in estilos:
            m = autorizada & (identidad == ident) & (particion == part)
            if not m.any():
                continue
            estilo = dict(marker=marcador, s=34, linewidths=1.2, label=f"Identidad {ident} · {part}")
            if relleno:
                estilo.update(c=color)
            else:
                estilo.update(facecolors="none", edgecolors=color)
            ax2.scatter(P[m, 0], P[m, 1], **estilo)
            ax3.scatter(P[m, 0], P[m, 1], P[m, 2], depthshade=False, **estilo)
    if (~autorizada).any():
        estilo = dict(marker="x", c="0.5", s=24, linewidths=0.9, label="Intrusos (val y test)")
        ax2.scatter(P[~autorizada, 0], P[~autorizada, 1], **estilo)
        ax3.scatter(P[~autorizada, 0], P[~autorizada, 1], P[~autorizada, 2], depthshade=False, **estilo)
    if corte is not None and corte.any():
        estilo = dict(marker=".", c="black", s=10, label="con corte (punto negro)")
        ax2.scatter(P[corte, 0], P[corte, 1], **estilo)
        ax3.scatter(P[corte, 0], P[corte, 1], P[corte, 2], depthshade=False, **estilo)

    ejes = [f"PC{i + 1} ({_coma(100 * proporcion[i], 1)} % de la varianza)" for i in range(3)]
    ax2.set_xlabel(ejes[0])
    ax2.set_ylabel(ejes[1])
    ax2.set_title("PC1–PC2")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(alpha=0.3)
    ax3.set_xlabel(ejes[0], fontsize=8)
    ax3.set_ylabel(ejes[1], fontsize=8)
    ax3.set_zlabel(ejes[2], fontsize=8)
    ax3.set_title("PC1–PC2–PC3")
    textos = [f"{nombre} {_coma(s)}" for nombre, s in (("train", s_train), ("val (autorizadas)", s_val)) if s is not None]
    figura.suptitle(
        "H3 · PCA propio de las características (orientación + Gabor + minucias), ajustado con train"
        + (f"\nsilhouette: {' · '.join(textos)}" if textos else ""),
        fontsize=11,
    )
    figura.subplots_adjust(left=0.06, right=0.97, bottom=0.1, top=0.86, wspace=0.12)
    buffer = io.BytesIO()
    figura.savefig(buffer, format="png", dpi=110)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(buffer.getvalue())
    return salida


# --------------------------------------------------------------------------- línea de comandos
def _contribucion(Zb: np.ndarray) -> float:
    """Media de ‖z_i − z_j‖² sobre los pares ordenados i ≠ j = 2 · Σ‖z_i − z̄‖² / (n − 1)."""
    centrado = Zb - Zb.mean(axis=0)
    return 2.0 * float(np.sum(centrado ** 2)) / (Zb.shape[0] - 1)


def _resumen(datos: dict) -> list[str]:
    """Silhouette de train y de val (total y por bloque) y contribución de cada bloque en train."""
    particion, identidad, X = datos["particion"], datos["identidad"], datos["X"]
    autorizada = (identidad > 0) & (datos["intruso"] == 0)
    train = particion == "train"
    val = (particion == "val") & autorizada
    escalador = ajustar_escalador({b: datos[b][train] for b in BLOQUES})
    conteos = {b: int(escalador.columnas[b].sum()) for b in BLOQUES}
    tramos, inicio = {}, 0
    for b in BLOQUES:
        tramos[b] = slice(inicio, inicio + conteos[b])
        inicio += conteos[b]
    lineas = [f"Vector escalado: d = {X.shape[1]} columnas ("
              + ", ".join(f"{b} {conteos[b]} de {datos[b].shape[1]}" for b in BLOQUES) + ")"]

    def fila(nombre: str, columnas) -> str:
        s_tr = _silhouette_si_se_puede(X[train][:, columnas], identidad[train])
        s_va = _silhouette_si_se_puede(X[val][:, columnas], identidad[val])
        tr = _coma(s_tr) if s_tr is not None else "—"
        va = _coma(s_va) if s_va is not None else "—"
        return f"  {nombre:<12} train {tr}   val {va}"

    lineas.append(f"Silhouette (train: {int(train.sum())} huellas; val: {int(val.sum())} autorizadas)")
    lineas.append(fila("total", slice(None)))
    lineas += [fila(b, tramos[b]) for b in BLOQUES]
    lineas.append("Contribución a la distancia en train (2 con columnas completas): "
                  + ", ".join(f"{b} {_coma(_contribucion(X[train][:, tramos[b]]))}" for b in BLOQUES))
    return lineas


def _ms_por_huella(cfg: dict, carpeta: Path, archivos, cantidad: int = 20) -> float:
    """Tiempo medio de caracteristicas() (sin la lectura ni preprocesar) en las primeras huellas."""
    total = 0.0
    elegidos = list(archivos)[:cantidad]
    for archivo in elegidos:
        huella = preprocesar(_leer_png(carpeta / str(archivo)), cfg)
        inicio = time.perf_counter()
        caracteristicas(huella, cfg)
        total += time.perf_counter() - inicio
    return 1000 * total / len(elegidos)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera las características de H3 (caracteristicas.npz) y la figura del PCA (pca_h3.png).")
    parser.add_argument("--sobrescribir", action="store_true", help="vuelve a escribir caracteristicas.npz si ya existe")
    args = parser.parse_args(argv)
    cfg = cargar_config()
    inicio = time.perf_counter()
    try:
        ruta = generar_caracteristicas(cfg, sobrescribir=args.sobrescribir)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1
    medio = time.perf_counter()
    figura = generar_figura(cfg)
    fin = time.perf_counter()
    with np.load(ruta, allow_pickle=False) as d:
        datos = {clave: d[clave] for clave in d.files}
    n = len(datos["archivo"])
    print(f"Características: {ruta} ({n} huellas en {_coma(medio - inicio, 1)} s, "
          f"{_coma(1000 * (medio - inicio) / n, 0)} ms por huella con la lectura y preprocesar)")
    print(f"Figura: {figura} ({_coma(fin - medio, 1)} s)")
    print(f"caracteristicas(): {_coma(_ms_por_huella(cfg, ruta.parent, datos['archivo']), 0)} ms por huella "
          f"(sin la lectura ni preprocesar; 20 huellas)")
    for linea in _resumen(datos):
        print(linea)
    return 0


if __name__ == "__main__":
    sys.exit(main())
