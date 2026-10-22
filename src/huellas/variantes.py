"""H1 · Variantes de una huella base: las "tomas" de un mismo dedo.

Contrato: docs/contratos/h1_generador.md, secciones 0 y 3. Una variante aplica, en este orden:
presión (erosión o dilatación en escala de grises), geometría (afín más una deformación
elástica suave, en un único cv2.remap), posiciones reales de las singularidades, corte
periférico y ruido. Con la misma base, los mismos parámetros y un rng con la misma semilla,
la variante es idéntica.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

from .generador import FONDO, HuellaBase, Singularidades

SIGMA_PUNTO_PX = 2.5
"""Desvío del punto gaussiano del mapa auxiliar que ubica cada singularidad."""

MARGEN_CORTE_PX = 30.0
"""Distancia mínima entre un núcleo y la zona que quita el corte."""

PASO_CORTE_GRADOS = 30.0
GRILLA_ELASTICA = (5, 4)
"""Puntos de control (filas, columnas) del desplazamiento elástico, interpolados con bicúbica."""


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


@dataclass(frozen=True)
class Variante:
    imagen: np.ndarray                     # uint8 (alto, ancho)
    mascara: np.ndarray                    # bool (alto, ancho)
    singularidades: Singularidades         # posiciones reales en la variante
    punto_referencia: tuple[float, float] | None
    parametros: ParametrosVariante         # los aplicados (con el ángulo de corte usado)
    mapa_x: np.ndarray                     # float32 (alto, ancho): mapeo inverso de cv2.remap
    mapa_y: np.ndarray                     # float32 (alto, ancho)


# --------------------------------------------------------------------------- parámetros
def sortear_parametros(rng: np.random.Generator, cfg: dict) -> ParametrosVariante:
    """Parámetros uniformes en los rangos de cfg["huellas"]["variantes"] (sección 3)."""
    v = cfg["huellas"]["variantes"]
    rotacion, traslacion, escala = (float(v[k]) for k in ("rotacion_max_grados", "traslacion_max_px", "escala_max"))
    presion_max = float(v["presion_max"])
    rot_grados = float(rng.uniform(-rotacion, rotacion))
    factor = float(rng.uniform(1.0 - escala, 1.0 + escala))
    tx_px = float(rng.uniform(-traslacion, traslacion))
    ty_px = float(rng.uniform(-traslacion, traslacion))
    elastica_px = float(rng.uniform(0.0, float(v["elastica_max_px"])))
    presion = float(rng.uniform(-presion_max, presion_max))
    ruido_sigma = float(rng.uniform(0.0, float(v["ruido_sigma_max"])))
    sal_pimienta = float(rng.uniform(0.0, float(v["sal_pimienta_max"])))
    if rng.random() < float(v["prob_corte"]):
        corte_min, corte_max = (float(f) for f in v["corte_periferico"])
        corte_fraccion = float(rng.uniform(corte_min, corte_max))
        corte_angulo_grados = float(rng.uniform(0.0, 360.0))
    else:
        corte_fraccion, corte_angulo_grados = 0.0, 0.0
    return ParametrosVariante(rot_grados, factor, tx_px, ty_px, elastica_px, presion,
                              corte_fraccion, corte_angulo_grados, ruido_sigma, sal_pimienta)


# --------------------------------------------------------------------------- geometría
def _desplazamiento_elastico(alto: int, ancho: int, amplitud: float,
                             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Desplazamiento suave: vectores al azar en una grilla gruesa, interpolados y escalados
    para que el más largo mida exactamente `amplitud`."""
    filas, columnas = GRILLA_ELASTICA
    control = rng.standard_normal((2, filas, columnas)).astype(np.float32)
    dx = cv2.resize(control[0], (ancho, alto), interpolation=cv2.INTER_CUBIC).astype(np.float64)
    dy = cv2.resize(control[1], (ancho, alto), interpolation=cv2.INTER_CUBIC).astype(np.float64)
    norma = float(np.sqrt(dx ** 2 + dy ** 2).max())
    factor = amplitud / norma if norma > 0 else 0.0
    return dx * factor, dy * factor


def _mapas(alto: int, ancho: int, params: ParametrosVariante,
           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Mapeo inverso de cv2.remap: la inversa de la afín de la sección 0, más la elástica.

    Afín directa: p' = c + s·R(α)·(p − c) + t, con R(α) = [[cos α, sen α], [−sen α, cos α]].
    Inversa: p = c + R(α)ᵀ·(p' − c − t) / s.
    """
    cx, cy = (ancho - 1) / 2, (alto - 1) / 2
    alfa = math.radians(params.rot_grados)
    coseno, seno = math.cos(alfa), math.sin(alfa)
    u = np.arange(ancho, dtype=np.float64)[None, :] - cx - params.tx_px
    w = np.arange(alto, dtype=np.float64)[:, None] - cy - params.ty_px
    mapa_x = cx + (coseno * u - seno * w) / params.escala
    mapa_y = cy + (seno * u + coseno * w) / params.escala
    if params.elastica_px > 0:
        dx, dy = _desplazamiento_elastico(alto, ancho, float(params.elastica_px), rng)
        mapa_x = mapa_x + dx
        mapa_y = mapa_y + dy
    return mapa_x.astype(np.float32), mapa_y.astype(np.float32)


def _ubicar(punto: tuple[float, float], mapa_x: np.ndarray, mapa_y: np.ndarray) -> tuple[float, float] | None:
    """Posición en la variante de un punto de la base: se pasa por el mismo remap un punto
    gaussiano centrado en él y se toma el máximo con precisión subpíxel (ajuste cuadrático
    del logaritmo en una ventana de 5 × 5, exacto para una gaussiana). None si el punto quedó
    afuera de la imagen."""
    alto, ancho = mapa_x.shape
    x0, y0 = float(punto[0]), float(punto[1])
    y = np.arange(alto, dtype=np.float32)[:, None]
    x = np.arange(ancho, dtype=np.float32)[None, :]
    auxiliar = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * SIGMA_PUNTO_PX ** 2)).astype(np.float32)
    salida = cv2.remap(auxiliar, mapa_x, mapa_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    fila, columna = np.unravel_index(int(np.argmax(salida)), salida.shape)
    if salida[fila, columna] < 0.5:
        return None
    f0, f1 = max(0, fila - 2), min(alto, fila + 3)
    c0, c1 = max(0, columna - 2), min(ancho, columna + 3)
    ventana = salida[f0:f1, c0:c1].astype(np.float64)
    ff, cc = np.mgrid[f0:f1, c0:c1]
    validos = ventana > 1e-4
    dx, dy = (cc - columna)[validos].astype(np.float64), (ff - fila)[validos].astype(np.float64)
    xs, ys = float(columna), float(fila)
    if validos.sum() >= 6:
        a = np.column_stack([np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy])
        coef, *_ = np.linalg.lstsq(a, np.log(ventana[validos]), rcond=None)
        hessiana = np.array([[2 * coef[3], coef[4]], [coef[4], 2 * coef[5]]])
        if np.all(np.linalg.eigvalsh(hessiana) < 0):
            desplazamiento = np.linalg.solve(hessiana, -coef[1:3])
            if np.all(np.abs(desplazamiento) <= 1.5):
                xs, ys = columna + float(desplazamiento[0]), fila + float(desplazamiento[1])
    if not (0 <= xs <= ancho - 1 and 0 <= ys <= alto - 1):
        return None
    return xs, ys


# --------------------------------------------------------------------------- corte
def _cortar(mascara: np.ndarray, nucleos, fraccion: float, angulo_pedido: float):
    """Semiplano que se lleva `fraccion` del área de la máscara, del lado (cos a, sen a).

    Prueba el ángulo pedido y, si algún núcleo quedaría a menos de MARGEN_CORTE_PX de la
    zona quitada, gira de a PASO_CORTE_GRADOS. Devuelve (quitados, ángulo usado, proyección,
    umbral) o None si ningún ángulo sirve.
    """
    alto, ancho = mascara.shape
    area = int(mascara.sum())
    a_quitar = int(round(fraccion * area))
    if area == 0 or a_quitar <= 0:
        return None
    a_quitar = min(a_quitar, area)
    filas = np.arange(alto, dtype=np.float64)[:, None]
    columnas = np.arange(ancho, dtype=np.float64)[None, :]
    for k in range(int(round(360 / PASO_CORTE_GRADOS))):
        angulo = (float(angulo_pedido) + PASO_CORTE_GRADOS * k) % 360.0
        a = math.radians(angulo)
        coseno, seno = math.cos(a), math.sin(a)
        proyeccion = columnas * coseno + filas * seno
        umbral = float(np.partition(proyeccion[mascara], area - a_quitar)[area - a_quitar])
        if all(umbral - (x * coseno + y * seno) >= MARGEN_CORTE_PX for x, y in nucleos):
            return mascara & (proyeccion >= umbral), angulo, proyeccion, umbral
    return None


# --------------------------------------------------------------------------- variante
def aplicar_variante(base: HuellaBase, params: ParametrosVariante,
                     rng: np.random.Generator) -> Variante:
    """Variante de la base con los parámetros dados (sección 3). No modifica la base."""
    alto, ancho = base.imagen.shape

    # 1. Presión: mezcla con la erosión (crestas más gruesas) o con la dilatación (más finas).
    imagen = base.imagen.astype(np.float32)
    presion = float(params.presion)
    if presion != 0.0:
        nucleo = np.ones((3, 3), np.uint8)
        morfologica = cv2.erode(base.imagen, nucleo) if presion > 0 else cv2.dilate(base.imagen, nucleo)
        imagen = (1.0 - abs(presion)) * imagen + abs(presion) * morfologica.astype(np.float32)

    # 2. Geometría: un único remap (afín + elástica); lo de afuera se completa con fondo.
    mapa_x, mapa_y = _mapas(alto, ancho, params, rng)
    imagen = cv2.remap(imagen, mapa_x, mapa_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=float(FONDO))
    mascara = cv2.remap(base.mascara.astype(np.float32), mapa_x, mapa_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0) > 0.5

    # 3. Posiciones reales de núcleos y deltas, y punto de referencia.
    nucleos = tuple(p for p in (_ubicar(q, mapa_x, mapa_y) for q in base.singularidades.nucleos) if p is not None)
    deltas = tuple(p for p in (_ubicar(q, mapa_x, mapa_y) for q in base.singularidades.deltas) if p is not None)
    if len(nucleos) == 1:
        referencia = nucleos[0]
    elif len(nucleos) == 2:
        referencia = ((nucleos[0][0] + nucleos[1][0]) / 2, (nucleos[0][1] + nucleos[1][1]) / 2)
    else:
        referencia = None

    # 4. Corte periférico, sin acercarse a los núcleos.
    usados = replace(params, corte_fraccion=0.0, corte_angulo_grados=0.0)
    if params.corte_fraccion > 0:
        corte = _cortar(mascara, nucleos, float(params.corte_fraccion), float(params.corte_angulo_grados))
        if corte is not None:
            quitados, angulo, proyeccion, umbral = corte
            mascara = mascara & ~quitados
            fondo = np.clip((proyeccion - umbral) / 3.0 + 0.5, 0.0, 1.0).astype(np.float32)  # borde de 3 px
            imagen = (1.0 - fondo) * imagen + fondo * float(FONDO)
            usados = replace(params, corte_angulo_grados=angulo)

    # 5. Ruido gaussiano y sal y pimienta.
    if params.ruido_sigma > 0:
        imagen = imagen + rng.normal(0.0, float(params.ruido_sigma), imagen.shape)
    imagen = np.clip(np.rint(imagen), 0, 255).astype(np.uint8)
    if params.sal_pimienta > 0:
        sorteo = rng.random(imagen.shape)
        mitad = float(params.sal_pimienta) / 2
        imagen[sorteo < mitad] = 0
        imagen[(sorteo >= mitad) & (sorteo < 2 * mitad)] = 255

    return Variante(
        imagen=imagen,
        mascara=mascara,
        singularidades=Singularidades(nucleos=nucleos, deltas=deltas),
        punto_referencia=referencia,
        parametros=usados,
        mapa_x=mapa_x,
        mapa_y=mapa_y,
    )
