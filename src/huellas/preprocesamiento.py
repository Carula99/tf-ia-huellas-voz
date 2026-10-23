"""H2 · Preprocesamiento de huellas: de una imagen cualquiera a la ventana alineada.

Contrato: docs/contratos/h2_preprocesamiento.md (versión 1.2). preprocesar() hace, en orden:

1. gris y filtro de mediana (la sal y pimienta de H1);
2. segmentación por la energía local de la imagen pasa-altos;
3. campo de orientación de las crestas con el tensor de estructura, y su coherencia;
4. singularidades con el índice de Poincaré, lejos del borde de la máscara;
5. ajuste del modelo de Sherlock y Monro al campo observado: θ∞, los núcleos y tantos
   deltas libres como núcleos (decisión 44). θ∞ es el ángulo para enderezar; sin núcleos,
   el ajuste se reduce al promedio circular del campo;
6. punto de referencia: el núcleo, el punto medio de los dos núcleos o, sin núcleos, el
   centro de la zona de máxima curvatura de las crestas;
7. ventana de 250 × 250 px sin reescalar, con el punto de referencia en el centro y girada
   por el menor de los dos giros que dejan θ∞ horizontal (contrato 1.2);
8. realce con CLAHE y un filtro de Gabor orientado píxel a píxel, binarización y esqueleto.

Línea de comandos, desde la raíz del repo (escribe la lámina de revisión):
    venv\\Scripts\\python -m src.huellas.preprocesamiento
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from skimage.morphology import skeletonize

from ..config import RAIZ, cargar_config
from .dataset import plan_dataset
from .generador import FONDO, generar_base
from .variantes import aplicar_variante, sortear_parametros

METODOS: tuple[str, ...] = ("nucleo", "nucleos", "curvatura")

LADO_MIN_PX = 64
"""Alto y ancho mínimos de la entrada."""

# Segmentación
SIGMA_SUAVE_PX = 1.0           # suavizado antes del pasa-altos
SIGMA_PASA_ALTOS_PX = 3.0      # lo que se resta: la media local, sin el escalón huella–fondo
SIGMA_ENERGIA_PX = 6.0         # ventana de la energía local
APERTURA_PX = 7                # apertura morfológica de la máscara
ENERGIA_MIN = 5.0              # por debajo (en el percentil 95) no hay huella

# Campo y singularidades
SIGMA_GRADIENTE_PX = 1.0       # suavizado antes de los gradientes de Sobel
SIGMA_POINCARE_PX = 2.0        # suavizado extra del campo doble antes del índice de Poincaré
PASO_POINCARE_PX = 2           # grilla del mapa de índices
PUNTOS_POINCARE = 36           # puntos del círculo de Poincaré
UMBRAL_INDICE = 0.25           # zona de un núcleo: índice > +0,25 (delta: < −0,25)
CARGA_MIN = 0.5                # carga mínima (en núcleos) de una zona para aceptarla
SEPARACION_FUSION_PX = 8.0     # una zona de carga ~2 se separa en dos núcleos a ±8 px

# Ajuste de Sherlock y Monro
PASO_AJUSTE_PX = 6             # grilla de puntos del ajuste
EROSION_AJUSTE_PX = 6          # el ajuste no mira el borde de la máscara
EXCLUSION_NUCLEO_PX = 8.0      # ni un disco chico alrededor de cada núcleo
ANILLO_INICIAL_PX = (40.0, 160.0)   # anillo del ángulo con el que arranca el ajuste
EVALUACIONES_MAX = 300

# Respaldo de curvatura (arcos)
SIGMA_CURVATURA_PX = 10.0      # suavizado del campo doble antes de medir la curvatura
SIGMA_MAPA_CURVATURA_PX = 6.0  # suavizado del mapa de curvatura
MARGEN_CURVATURA_PX = 10.0     # además del margen de las singularidades

# Ventana y realce
MARGEN_VENTANA_PX = 24         # se filtra una ventana más grande y se recorta, sin artefactos de borde
ORIENTACIONES_GABOR = 16
SIGMA_MEDIA_GABOR_PX = 6.0     # se resta la media local antes del Gabor


@dataclass(frozen=True)
class HuellaPreprocesada:
    """Salida de preprocesar (sección 2 del contrato)."""

    realzada: np.ndarray        # uint8 (250, 250): crestas oscuras; 255 fuera de la máscara
    binaria: np.ndarray         # bool (250, 250): True en las crestas
    esqueleto: np.ndarray       # bool (250, 250): crestas de 1 px
    mascara: np.ndarray         # bool (250, 250): True dentro de la huella
    orientacion: np.ndarray     # float64 (250, 250), en [0, π), en los ejes de la ventana
    punto_referencia: tuple[float, float]       # (x, y) en la imagen de entrada
    metodo: str                                 # uno de METODOS
    nucleos: tuple[tuple[float, float], ...]    # núcleos usados, (x, y) en la imagen de entrada
    angulo: float               # θ∞ estimado, en radianes, en [0, π), en la imagen de entrada
    matriz: np.ndarray          # float64 (2, 3): de la entrada a la ventana


# --------------------------------------------------------------------------- ángulos
def _a_cero_pi(angulo: float) -> float:
    """Lleva un ángulo a [0, π); el módulo puede devolver π por redondeo y eso pasa a 0."""
    angulo = float(angulo) % math.pi
    return 0.0 if angulo >= math.pi else angulo


def _campo_a_cero_pi(theta: np.ndarray) -> np.ndarray:
    theta = np.mod(np.asarray(theta, dtype=np.float64), np.pi)
    theta[theta >= np.pi] = 0.0
    return theta


def giro_ventana(angulo: float) -> float:
    """a_v = ((angulo + π/2) mod π) − π/2, en [−π/2, π/2): de los dos giros que dejan horizontal
    la orientación `angulo` (difieren en 180°), el más chico. Así la huella no queda cabeza abajo."""
    return (float(angulo) + math.pi / 2) % math.pi - math.pi / 2


def matriz_ventana(punto: tuple[float, float], angulo: float, ventana: tuple[int, int]) -> np.ndarray:
    """Matriz 2 × 3 de la entrada a la ventana (sección 0): giro a_v y punto al centro."""
    alto, ancho = ventana
    a = giro_ventana(angulo)
    coseno, seno = math.cos(a), math.sin(a)
    rotacion = np.array([[coseno, seno], [-seno, coseno]], dtype=np.float64)
    centro = np.array([(ancho - 1) / 2, (alto - 1) / 2], dtype=np.float64)
    traslacion = centro - rotacion @ np.array(punto, dtype=np.float64)
    return np.hstack([rotacion, traslacion[:, None]])


# --------------------------------------------------------------------------- entrada y máscara
def _a_gris(imagen) -> np.ndarray:
    """uint8 de (alto, ancho), BGR o BGRA a gris; cualquier otra cosa, ValueError (regla 1)."""
    imagen = np.asarray(imagen)
    if imagen.dtype != np.uint8:
        raise ValueError(f"La imagen tiene que ser uint8; es {imagen.dtype}.")
    if imagen.ndim == 3 and imagen.shape[2] == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    elif imagen.ndim == 3 and imagen.shape[2] == 4:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
    elif imagen.ndim != 2:
        raise ValueError(
            f"La imagen tiene que ser (alto, ancho), (alto, ancho, 3) o (alto, ancho, 4); es {imagen.shape}."
        )
    if min(imagen.shape) < LADO_MIN_PX:
        raise ValueError(f"La imagen tiene que medir al menos {LADO_MIN_PX} × {LADO_MIN_PX} px; es {imagen.shape}.")
    return np.ascontiguousarray(imagen)


def _segmentar(gris: np.ndarray, umbral: float) -> np.ndarray:
    """Máscara de la huella: energía local de la imagen pasa-altos, que no ve el escalón entre la
    huella y el fondo, por encima de `umbral` × su percentil 95; apertura, la componente más
    grande y los agujeros rellenos (regla 3)."""
    f = cv2.GaussianBlur(gris.astype(np.float32), (0, 0), SIGMA_SUAVE_PX)
    pasa_altos = f - cv2.GaussianBlur(f, (0, 0), SIGMA_PASA_ALTOS_PX)
    energia = np.sqrt(cv2.GaussianBlur(pasa_altos * pasa_altos, (0, 0), SIGMA_ENERGIA_PX))
    referencia = float(np.percentile(energia, 95))
    if referencia < ENERGIA_MIN:
        raise ValueError("No se encontró una huella en la imagen (no hay crestas).")
    mascara = (energia > umbral * referencia).astype(np.uint8)
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (APERTURA_PX, APERTURA_PX))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemento)
    n, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara, 8)
    if n <= 1:
        raise ValueError("No se encontró una huella en la imagen.")
    mayor = 1 + int(np.argmax(estadisticas[1:, cv2.CC_STAT_AREA]))
    return ndimage.binary_fill_holes(etiquetas == mayor)


def _distancia_al_borde(mascara: np.ndarray) -> np.ndarray:
    """Distancia de cada píxel al borde de la máscara; el borde de la imagen cuenta como borde."""
    con_marco = np.pad(mascara, 1).astype(np.uint8)
    return cv2.distanceTransform(con_marco, cv2.DIST_L2, 5)[1:-1, 1:-1]


# --------------------------------------------------------------------------- campo de orientación
@dataclass
class _Campo:
    theta: np.ndarray       # orientación de las crestas, float64 en [0, π)
    coherencia: np.ndarray  # float64 en [0, 1]
    u: np.ndarray           # cos 2θ
    v: np.ndarray           # sen 2θ


def _campo(gris: np.ndarray, mascara: np.ndarray, sigma: float) -> _Campo:
    """Tensor de estructura de los gradientes, suavizado con un gaussiano de `sigma` solo con los
    píxeles de la máscara. La cresta es perpendicular al gradiente: su ángulo doble es el del
    gradiente más π (regla 4)."""
    f = cv2.GaussianBlur(gris.astype(np.float32), (0, 0), SIGMA_GRADIENTE_PX)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    peso = mascara.astype(np.float32)
    a = cv2.GaussianBlur((gx * gx - gy * gy) * peso, (0, 0), sigma).astype(np.float64)
    b = cv2.GaussianBlur((2 * gx * gy) * peso, (0, 0), sigma).astype(np.float64)
    energia = cv2.GaussianBlur((gx * gx + gy * gy) * peso, (0, 0), sigma).astype(np.float64)
    u, v = -a, -b
    modulo = np.hypot(u, v)
    coherencia = np.clip(modulo / (energia + 1e-6), 0.0, 1.0)
    theta = _campo_a_cero_pi(0.5 * np.arctan2(v, u))
    return _Campo(theta, coherencia, u / (modulo + 1e-12), v / (modulo + 1e-12))


# --------------------------------------------------------------------------- singularidades
def _indices_poincare(u: np.ndarray, v: np.ndarray, xs: np.ndarray, ys: np.ndarray, radio: float) -> np.ndarray:
    """Índice de Poincaré del campo doble (u, v) en los puntos (xs, ys): vueltas del ángulo doble
    en un círculo de `radio`, divididas por 2 (un núcleo da +½ y un delta −½)."""
    phi = 2 * np.pi * np.arange(PUNTOS_POINCARE + 1) / PUNTOS_POINCARE
    px = xs[:, None] + radio * np.cos(phi)[None, :]
    py = ys[:, None] + radio * np.sin(phi)[None, :]
    cu = ndimage.map_coordinates(u, [py.ravel(), px.ravel()], order=1, mode="nearest").reshape(px.shape)
    cv = ndimage.map_coordinates(v, [py.ravel(), px.ravel()], order=1, mode="nearest").reshape(px.shape)
    doble = np.arctan2(cv, cu)
    saltos = (np.diff(doble, axis=1) + np.pi) % (2 * np.pi) - np.pi
    return saltos.sum(axis=1) / (4 * np.pi)


def _singularidades(campo: _Campo, mascara: np.ndarray, radio: float, margen: float):
    """Núcleos y deltas como (carga, x, y): cada zona con índice > +0,25 (o < −0,25) es una
    singularidad en su centro ponderado; la carga es su área sobre la de un núcleo aislado, así
    que dos núcleos que se juntan dan una sola zona de carga ~2 (regla 5)."""
    u = cv2.GaussianBlur(campo.u.astype(np.float32), (0, 0), SIGMA_POINCARE_PX).astype(np.float64)
    v = cv2.GaussianBlur(campo.v.astype(np.float32), (0, 0), SIGMA_POINCARE_PX).astype(np.float64)
    alto, ancho = mascara.shape
    distancia = _distancia_al_borde(mascara)
    ys, xs = np.mgrid[0:alto:PASO_POINCARE_PX, 0:ancho:PASO_POINCARE_PX]
    validos = distancia[ys, xs] >= margen
    indice = np.zeros(ys.shape, dtype=np.float64)
    if validos.any():
        indice[validos] = _indices_poincare(u, v, xs[validos].astype(np.float64), ys[validos].astype(np.float64), radio)
    area_nucleo = 0.5 * math.pi * radio ** 2 / PASO_POINCARE_PX ** 2
    resultado = {}
    for signo, nombre in ((1.0, "nucleos"), (-1.0, "deltas")):
        etiquetas, n = ndimage.label(signo * indice > UMBRAL_INDICE, structure=np.ones((3, 3)))
        zonas = []
        for k in range(1, n + 1):
            sel = etiquetas == k
            peso = signo * indice[sel]
            carga = float(peso.sum() / area_nucleo)
            if carga >= CARGA_MIN:
                zonas.append((carga, float((xs[sel] * peso).sum() / peso.sum()),
                              float((ys[sel] * peso).sum() / peso.sum())))
        resultado[nombre] = sorted(zonas, key=lambda z: -z[0])
    return resultado["nucleos"], resultado["deltas"]


def _nucleos_detectados(zonas) -> list[tuple[float, float]]:
    """Regla del punto de referencia: los dos núcleos más fuertes; una sola zona de carga ~2 se
    separa en dos núcleos a ±SEPARACION_FUSION_PX en vertical (el ajuste los acomoda)."""
    if len(zonas) >= 2:
        return [(zonas[0][1], zonas[0][2]), (zonas[1][1], zonas[1][2])]
    if len(zonas) == 1 and zonas[0][0] >= 1.5:
        _, x, y = zonas[0]
        return [(x, y - SEPARACION_FUSION_PX), (x, y + SEPARACION_FUSION_PX)]
    if len(zonas) == 1:
        return [(zonas[0][1], zonas[0][2])]
    return []


# --------------------------------------------------------------------------- respaldo de curvatura
def _centro_de_curvatura(campo: _Campo, mascara: np.ndarray, margen: float) -> tuple[float, float]:
    """Centro de la zona de máxima curvatura de las crestas (para huellas sin núcleo, como el arco).

    La curvatura es cuánto gira θ a lo largo de la cresta: ½·(u·∂v/∂s − v·∂u/∂s), con s en la
    dirección de la cresta. Se toma la zona donde supera la mitad de su máximo, lejos del borde, y
    su centro ponderado: en el arco el máximo solo es poco estable a lo largo del eje.
    """
    u = cv2.GaussianBlur(campo.u.astype(np.float32), (0, 0), SIGMA_CURVATURA_PX).astype(np.float64)
    v = cv2.GaussianBlur(campo.v.astype(np.float32), (0, 0), SIGMA_CURVATURA_PX).astype(np.float64)
    modulo = np.hypot(u, v) + 1e-12
    u, v = u / modulo, v / modulo
    theta = 0.5 * np.arctan2(v, u)
    tx, ty = np.cos(theta), np.sin(theta)
    duy, dux = np.gradient(u)
    dvy, dvx = np.gradient(v)
    curvatura = np.abs(0.5 * (u * (tx * dvx + ty * dvy) - v * (tx * dux + ty * duy)))
    curvatura[_distancia_al_borde(mascara) < margen + MARGEN_CURVATURA_PX] = 0.0
    curvatura = cv2.GaussianBlur(curvatura.astype(np.float32), (0, 0), SIGMA_MAPA_CURVATURA_PX).astype(np.float64)
    if not np.any(curvatura > 0):   # huella muy chica: el centro de la máscara
        ys, xs = np.nonzero(mascara)
        return float(xs.mean()), float(ys.mean())
    fila, columna = np.unravel_index(int(np.argmax(curvatura)), curvatura.shape)
    etiquetas, _ = ndimage.label(curvatura >= 0.5 * curvatura[fila, columna])
    ys, xs = np.nonzero(etiquetas == etiquetas[fila, columna])
    peso = curvatura[ys, xs]
    return float((xs * peso).sum() / peso.sum()), float((ys * peso).sum() / peso.sum())


# --------------------------------------------------------------------------- ajuste de Sherlock y Monro
def _angulo_inicial(campo: _Campo, mascara: np.ndarray, punto: tuple[float, float]) -> float:
    """Promedio circular del campo en un anillo alrededor del punto: solo el arranque del ajuste."""
    alto, ancho = mascara.shape
    ys, xs = np.mgrid[0:alto, 0:ancho]
    r = np.hypot(xs - punto[0], ys - punto[1])
    sel = mascara & (r >= ANILLO_INICIAL_PX[0]) & (r <= ANILLO_INICIAL_PX[1])
    if not sel.any():
        sel = mascara
    w = campo.coherencia[sel]
    return 0.5 * math.atan2(float((w * campo.v[sel]).sum()), float((w * campo.u[sel]).sum()))


def _muestras_del_ajuste(campo: _Campo, mascara: np.ndarray, nucleos):
    """Puntos de una grilla dentro de la máscara (sin su borde ni un disco alrededor de cada
    núcleo), con cos 2θ, sen 2θ y el peso √coherencia."""
    lado = 2 * EROSION_AJUSTE_PX + 1
    interior = cv2.erode(mascara.astype(np.uint8), np.ones((lado, lado), np.uint8)).astype(bool)
    alto, ancho = mascara.shape
    ys, xs = np.mgrid[PASO_AJUSTE_PX // 2:alto:PASO_AJUSTE_PX, PASO_AJUSTE_PX // 2:ancho:PASO_AJUSTE_PX]
    validos = interior[ys, xs]
    for cx, cy in nucleos:
        validos &= np.hypot(xs - cx, ys - cy) > EXCLUSION_NUCLEO_PX
    ys, xs = ys[validos], xs[validos]
    peso = np.sqrt(campo.coherencia[ys, xs])
    return xs.astype(np.float64), ys.astype(np.float64), campo.u[ys, xs], campo.v[ys, xs], peso


def _doble_angulo_modelo(parametros: np.ndarray, x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """2θ(z) = 2θ∞ + Σ arg(z − c_k) − Σ arg(z − d_k), con n núcleos y n deltas (regla 6)."""
    doble = np.full_like(x, 2.0 * parametros[0])
    for k in range(n):
        doble += np.arctan2(y - parametros[2 + 2 * k], x - parametros[1 + 2 * k])
    for k in range(n):
        doble -= np.arctan2(y - parametros[2 + 2 * (n + k)], x - parametros[1 + 2 * (n + k)])
    return doble


def _ajustar_sherlock_monro(campo: _Campo, mascara: np.ndarray, nucleos, deltas, angulo_0: float,
                            holgura: float):
    """θ∞ y los núcleos ajustados. Sin núcleos, θ∞ es el promedio circular ponderado del campo.

    Los núcleos se mueven a lo sumo `holgura` px por coordenada; los deltas son libres y arrancan
    en los detectados más cercanos a los núcleos o, si faltan, en posiciones típicas (abajo y a
    los costados, o justo abajo como en el arco en tienda), giradas por el ángulo inicial.
    Queda el arranque de menor residuo.
    """
    x, y, c2, s2, w = _muestras_del_ajuste(campo, mascara, nucleos)
    n = len(nucleos)
    if n == 0 or x.size < 4 * n + 1:
        return _a_cero_pi(0.5 * math.atan2(float((w * s2).sum()), float((w * c2).sum()))), list(nucleos)
    centro = np.mean(np.array(nucleos, dtype=np.float64), axis=0)
    coseno, seno = math.cos(angulo_0), math.sin(angulo_0)

    def girado(dx: float, dy: float) -> tuple[float, float]:
        return centro[0] + dx * coseno - dy * seno, centro[1] + dx * seno + dy * coseno

    arranques = [[girado(-80, 90)], [girado(80, 90)], [girado(0, 30)]] if n == 1 else [[girado(-80, 90), girado(80, 90)]]
    if len(deltas) >= n:
        cercanos = sorted(deltas, key=lambda d: min(math.dist(d, c) for c in nucleos))[:n]
        arranques.insert(0, cercanos)
    abajo = [-np.inf] + [c - holgura for nucleo in nucleos for c in nucleo] + [-np.inf] * (2 * n)
    arriba = [np.inf] + [c + holgura for nucleo in nucleos for c in nucleo] + [np.inf] * (2 * n)

    def residuos(p: np.ndarray) -> np.ndarray:
        doble = _doble_angulo_modelo(p, x, y, n)
        return np.concatenate([w * (c2 - np.cos(doble)), w * (s2 - np.sin(doble))])

    mejor = None
    for delta_0 in arranques:
        p0 = [angulo_0] + [c for nucleo in nucleos for c in nucleo] + [c for d in delta_0 for c in d]
        ajuste = least_squares(residuos, p0, bounds=(abajo, arriba), method="trf", x_scale="jac",
                               max_nfev=EVALUACIONES_MAX)
        if mejor is None or ajuste.cost < mejor.cost:
            mejor = ajuste
    ajustados = [(float(mejor.x[1 + 2 * k]), float(mejor.x[2 + 2 * k])) for k in range(n)]
    return _a_cero_pi(mejor.x[0]), ajustados


# --------------------------------------------------------------------------- realce
def _gabor_orientado(imagen: np.ndarray, theta: np.ndarray, periodo: float) -> np.ndarray:
    """Respuesta de un banco de Gabor de 16 orientaciones, interpolando en cada píxel entre las
    dos más cercanas a su orientación (Hong, Wan y Jain, 1998). Positiva en los valles y negativa
    sobre las crestas oscuras."""
    sigma = 0.5 * periodo
    lado = int(2 * round(3 * sigma) + 1)
    f = imagen.astype(np.float32)
    f = f - cv2.GaussianBlur(f, (0, 0), SIGMA_MEDIA_GABOR_PX)
    respuestas = []
    for k in range(ORIENTACIONES_GABOR):
        cresta = np.pi * k / ORIENTACIONES_GABOR
        # En cv2.getGaborKernel, theta es la normal a las franjas: la cresta + π/2.
        nucleo = cv2.getGaborKernel((lado, lado), sigma, cresta + np.pi / 2, periodo, 1.0, 0, ktype=cv2.CV_32F)
        nucleo -= nucleo.mean()
        respuestas.append(cv2.filter2D(f, -1, nucleo, borderType=cv2.BORDER_REFLECT))
    respuestas = np.stack(respuestas)
    posicion = theta / np.pi * ORIENTACIONES_GABOR
    k0 = np.floor(posicion).astype(np.intp) % ORIENTACIONES_GABOR
    k1 = (k0 + 1) % ORIENTACIONES_GABOR
    peso = (posicion - np.floor(posicion)).astype(np.float32)
    r0 = np.take_along_axis(respuestas, k0[None], 0)[0]
    r1 = np.take_along_axis(respuestas, k1[None], 0)[0]
    return (1.0 - peso) * r0 + peso * r1


# --------------------------------------------------------------------------- preprocesar
def preprocesar(imagen: np.ndarray, cfg: dict) -> HuellaPreprocesada:
    """Lleva una huella a la ventana de 250 × 250 alineada por su punto de referencia y
    enderezada por θ∞, con la imagen realzada, las crestas binarizadas y el esqueleto.

    Raises:
        ValueError: si la imagen no es uint8 de (alto, ancho), (alto, ancho, 3) o (alto, ancho, 4),
            si mide menos de 64 px de lado o si no tiene una huella.
    """
    p = cfg["huellas"]["preprocesamiento"]
    alto_w, ancho_w = (int(n) for n in cfg["huellas"]["ventana_px"])
    periodo = float(np.mean([float(n) for n in cfg["huellas"]["periodo_crestas_px"]]))
    margen = float(p["margen_singularidades_px"])

    # 1-3. Gris sin sal y pimienta, máscara y campo de orientación.
    gris = cv2.medianBlur(_a_gris(imagen), int(p["mediana_px"]))
    mascara = _segmentar(gris, float(p["umbral_segmentacion"]))
    campo_fino = _campo(gris, mascara, float(p["sigma_tensor_px"]))
    campo = _campo(gris, mascara, float(p["sigma_ajuste_px"]))

    # 4-6. Singularidades, ajuste de Sherlock y Monro y punto de referencia.
    zonas_nucleo, zonas_delta = _singularidades(campo_fino, mascara, float(p["radio_poincare_px"]), margen)
    nucleos = _nucleos_detectados(zonas_nucleo)
    deltas = [(x, y) for _, x, y in zonas_delta]
    if nucleos:
        punto_0 = tuple(np.mean(np.array(nucleos), axis=0))
    else:
        punto_0 = _centro_de_curvatura(campo, mascara, margen)
    angulo, nucleos = _ajustar_sherlock_monro(campo, mascara, nucleos, deltas,
                                              _angulo_inicial(campo, mascara, punto_0),
                                              float(p["holgura_nucleo_px"]))
    metodo = ("curvatura", "nucleo", "nucleos")[len(nucleos)]
    if nucleos:
        punto = (float(np.mean([n[0] for n in nucleos])), float(np.mean([n[1] for n in nucleos])))
    else:
        punto = (float(punto_0[0]), float(punto_0[1]))

    # 7. Ventana, filtrada con un margen alrededor para que el borde no deje artefactos.
    matriz = matriz_ventana(punto, angulo, (alto_w, ancho_w))
    con_margen = matriz.copy()
    con_margen[:, 2] += MARGEN_VENTANA_PX
    tamano = (ancho_w + 2 * MARGEN_VENTANA_PX, alto_w + 2 * MARGEN_VENTANA_PX)
    fondo = float(np.median(gris[~mascara])) if (~mascara).any() else float(FONDO)
    gris_w = cv2.warpAffine(gris, con_margen, tamano, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=fondo)
    mascara_w = cv2.warpAffine(mascara.astype(np.uint8), con_margen, tamano, flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    # El campo doble pasa a la ventana y se gira 2·a_v: la orientación queda θ − a_v.
    u_w = cv2.warpAffine(campo.u.astype(np.float32), con_margen, tamano, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=1.0).astype(np.float64)
    v_w = cv2.warpAffine(campo.v.astype(np.float32), con_margen, tamano, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0).astype(np.float64)
    a = giro_ventana(angulo)
    c2, s2 = math.cos(2 * a), math.sin(2 * a)
    theta_w = _campo_a_cero_pi(0.5 * np.arctan2(v_w * c2 - u_w * s2, u_w * c2 + v_w * s2))

    # 8. Realce, binarización y esqueleto.
    clahe = cv2.createCLAHE(clipLimit=float(p["clahe_limite"]),
                            tileGridSize=(int(p["clahe_grilla"]), int(p["clahe_grilla"])))
    respuesta = _gabor_orientado(clahe.apply(gris_w), theta_w, periodo)
    escala = float(np.percentile(np.abs(respuesta[mascara_w]), 95)) if mascara_w.any() else 1.0
    realzada = np.clip(128.0 + 127.0 * respuesta / (escala + 1e-9), 0, 255)
    realzada[~mascara_w] = 255
    recorte = (slice(MARGEN_VENTANA_PX, MARGEN_VENTANA_PX + alto_w),
               slice(MARGEN_VENTANA_PX, MARGEN_VENTANA_PX + ancho_w))
    mascara_v = np.ascontiguousarray(mascara_w[recorte])
    binaria = np.ascontiguousarray((respuesta[recorte] < 0) & mascara_v)
    esqueleto = skeletonize(binaria, method="lee").astype(bool)

    return HuellaPreprocesada(
        realzada=np.ascontiguousarray(np.rint(realzada[recorte]).astype(np.uint8)),
        binaria=binaria,
        esqueleto=esqueleto,
        mascara=mascara_v,
        orientacion=np.ascontiguousarray(theta_w[recorte]),
        punto_referencia=punto,
        metodo=metodo,
        nucleos=tuple((float(x), float(y)) for x, y in nucleos),
        angulo=float(angulo),
        matriz=matriz,
    )


# --------------------------------------------------------------------------- lámina
def _elegir(entradas: list[dict], cfg: dict, bases: dict) -> list[tuple[dict, object]]:
    """Tres variantes de una identidad: la de corte con más rotación (si hay) y las dos sin corte
    con más rotación hacia cada lado."""
    variantes = []
    for entrada in entradas:
        base = bases[(entrada["tipo"], entrada["semilla"])]
        rng = np.random.default_rng([entrada["semilla"], entrada["variante"]])
        variantes.append((entrada, aplicar_variante(base, sortear_parametros(rng, cfg), rng)))
    con_corte = sorted((x for x in variantes if x[1].parametros.corte_fraccion > 0),
                       key=lambda x: -abs(x[1].parametros.rot_grados))
    sin_corte = [x for x in variantes if x[1].parametros.corte_fraccion == 0]
    positivas = sorted((x for x in sin_corte if x[1].parametros.rot_grados >= 0), key=lambda x: -x[1].parametros.rot_grados)
    negativas = sorted((x for x in sin_corte if x[1].parametros.rot_grados < 0), key=lambda x: x[1].parametros.rot_grados)
    elegidas = con_corte[:1] + positivas[:1] + negativas[:1]
    resto = [x for x in con_corte[1:] + positivas[1:] + negativas[1:] if x not in elegidas]
    return (elegidas + resto)[:3]


def _rotulo(imagen: np.ndarray, texto: str) -> np.ndarray:
    banda = np.full((26, imagen.shape[1], 3), 255, np.uint8)
    cv2.putText(banda, texto, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([banda, imagen])


def _paneles(variante, resultado: HuellaPreprocesada, nombre: str, alto: int) -> list[np.ndarray]:
    """La entrada con el punto de referencia, la dirección de θ∞ y el borde de la ventana; la
    ventana realzada; el esqueleto."""
    entrada = cv2.cvtColor(variante.imagen, cv2.COLOR_GRAY2BGR)
    x, y = resultado.punto_referencia
    a = resultado.angulo
    for s in (-1, 1):
        cv2.line(entrada, (int(round(x)), int(round(y))),
                 (int(round(x + s * 90 * math.cos(a))), int(round(y + s * 90 * math.sin(a)))),
                 (0, 160, 255), 2, cv2.LINE_AA)
    inversa = cv2.invertAffineTransform(resultado.matriz)
    lado = resultado.realzada.shape[0] - 1
    esquinas = np.array([[0, 0], [lado, 0], [lado, lado], [0, lado]], dtype=np.float64)
    borde = esquinas @ inversa[:, :2].T + inversa[:, 2]
    cv2.polylines(entrada, [np.round(borde).astype(np.int32)], True, (200, 120, 0), 1, cv2.LINE_AA)
    cv2.circle(entrada, (int(round(x)), int(round(y))), 7, (0, 0, 220), 2, cv2.LINE_AA)
    escala = alto / entrada.shape[0]
    entrada = cv2.resize(entrada, (int(round(entrada.shape[1] * escala)), alto), interpolation=cv2.INTER_AREA)
    p = variante.parametros
    corte = f", corte {100 * p.corte_fraccion:.0f}%" if p.corte_fraccion > 0 else ""
    esqueleto = np.full(resultado.esqueleto.shape + (3,), 255, np.uint8)
    esqueleto[~resultado.mascara] = (235, 235, 235)
    esqueleto[resultado.esqueleto] = (0, 0, 0)
    return [
        _rotulo(entrada, f"{nombre}: rot {p.rot_grados:+.0f} gr{corte}"),
        _rotulo(cv2.cvtColor(resultado.realzada, cv2.COLOR_GRAY2BGR), f"ventana ({resultado.metodo})"),
        _rotulo(esqueleto, "esqueleto"),
    ]


def _escribir_png(ruta: Path, imagen: np.ndarray) -> None:
    """cv2.imwrite no acepta rutas con tildes en Windows: se codifica y se escribe aparte."""
    ok, codificado = cv2.imencode(".png", imagen, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"No se pudo codificar {ruta} como PNG.")
    codificado.tofile(str(ruta))


def generar_lamina(cfg: dict, salida: str | os.PathLike | None = None) -> Path:
    """Lámina para la revisión visual (sección 3 del contrato): una fila por identidad autorizada y
    una con un intruso de tipo arco; en cada una, 3 variantes del dataset (al menos una con corte,
    si la identidad tiene) con la entrada, la ventana realzada y el esqueleto. Las variantes salen
    de la receta de H1, sin leer data/. Por defecto se escribe en RAIZ / rutas.figuras /
    "lamina_h2.png"; devuelve la ruta escrita."""
    salida = RAIZ / cfg["rutas"]["figuras"] / "lamina_h2.png" if salida is None else Path(os.fspath(salida))
    plan = plan_dataset(cfg)
    grupos = [[e for e in plan if e["identidad"] == int(i["id"])] for i in cfg["huellas"]["identidades"]]
    arco = next(e["intruso"] for e in plan if e["intruso"] > 0 and e["tipo"] == "arco")
    grupos.append([e for e in plan if e["intruso"] == arco])
    bases = {}
    for grupo in grupos:
        for entrada in grupo:
            clave = (entrada["tipo"], entrada["semilla"])
            if clave not in bases:
                bases[clave] = generar_base(entrada["tipo"], entrada["semilla"], cfg)
    alto_w = int(cfg["huellas"]["ventana_px"][0])
    renglones = []
    for grupo in grupos:
        primero = grupo[0]
        titulo = (f"Identidad {primero['identidad']}: {primero['tipo']}" if primero["identidad"]
                  else f"Intruso {primero['intruso']}: {primero['tipo']}")
        paneles = []
        for entrada, variante in _elegir(grupo, cfg, bases):
            nombre = Path(entrada["archivo"]).stem
            paneles += _paneles(variante, preprocesar(variante.imagen, cfg), nombre, alto_w)
        separador = np.full((paneles[0].shape[0], 8, 3), 255, np.uint8)
        fila = np.hstack([x for panel in paneles for x in (panel, separador)][:-1])
        cabecera = np.full((30, fila.shape[1], 3), 255, np.uint8)
        cv2.putText(cabecera, titulo, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        renglones.append(np.vstack([cabecera, fila, np.full((12, fila.shape[1], 3), 255, np.uint8)]))
    ancho = max(r.shape[1] for r in renglones)
    renglones = [cv2.copyMakeBorder(r, 0, 0, 0, ancho - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                 for r in renglones]
    lamina = cv2.copyMakeBorder(np.vstack(renglones), 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    salida.parent.mkdir(parents=True, exist_ok=True)
    _escribir_png(salida, lamina)
    return salida


# --------------------------------------------------------------------------- línea de comandos
def main(argv: list[str] | None = None) -> int:
    cfg = cargar_config()
    inicio = time.perf_counter()
    ruta = generar_lamina(cfg)
    print(f"Lámina: {ruta} ({time.perf_counter() - inicio:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
