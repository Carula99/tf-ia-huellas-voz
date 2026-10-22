"""H1 · Huellas base sintéticas.

Contrato: docs/contratos/h1_generador.md, secciones 0 y 2. Cada identidad tiene una huella
base, que sale entera de su semilla:

1. singularidades (núcleos y deltas) según el tipo, con la geometría de la tabla del contrato;
2. campo de orientación de Sherlock y Monro (el arco tiene un campo propio, sin singularidades);
3. máscara elíptica suave;
4. crestas sintetizadas con filtrado de Gabor iterativo desde ruido, como en SFinGe: cada
   pasada filtra con el banco de Gabor y en cada píxel se queda con la respuesta del filtro
   de su orientación, así el ruido se va ordenando en crestas paralelas al campo;
5. realce: se empareja el contraste local (donde dos crestas se juntan, los filtros se
   cancelan en parte y quedaban manchas grises) y una curva tanh define el borde de cada
   cresta, para que la imagen se parezca más a una captura que a una onda suave.

La misma semilla da exactamente la misma base (todos los sorteos salen de un único
np.random.Generator con esa semilla).
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import cv2
import numpy as np

TIPOS: tuple[str, ...] = ("presilla", "verticilo", "arco_tienda", "arco")

FONDO: int = 245
"""Gris del fondo, afuera de la huella."""

GRIS_MEDIO, CONTRASTE = 128.0, 100.0
"""Las crestas se dibujan como GRIS_MEDIO − CONTRASTE · c, con c en [−1, 1]: cresta oscura (c = 1)."""

BORDE_SUAVE_PX = 3.0
"""Ancho (escala de la logística) del borde suave de la máscara, en píxeles."""

NITIDEZ = 2.0
"""Pendiente K de la curva tanh(K·c)/tanh(K) del realce: con 2 el borde de la cresta ocupa
alrededor de 1,5 px; con más, los bordes diagonales se ven escalonados."""


@dataclass(frozen=True)
class Singularidades:
    """Posiciones (x, y) en píxeles de los núcleos y los deltas."""

    nucleos: tuple[tuple[float, float], ...]
    deltas: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class HuellaBase:
    """Patrón base de una identidad (sección 2 del contrato)."""

    tipo: str
    semilla: int
    imagen: np.ndarray            # uint8 (alto, ancho)
    mascara: np.ndarray           # bool (alto, ancho): True dentro de la huella
    orientacion: np.ndarray       # float64 (alto, ancho), en [0, π)
    singularidades: Singularidades
    periodo_px: float             # período usado para sintetizar las crestas


# --------------------------------------------------------------------------- orientación
def _a_cero_pi(theta: np.ndarray) -> np.ndarray:
    """Lleva un ángulo a [0, π); np.mod puede devolver π por redondeo y eso pasa a 0."""
    theta = np.mod(theta, np.pi)
    theta[theta >= np.pi] = 0.0
    return theta


def campo_orientacion(singularidades: Singularidades, tamano: tuple[int, int],
                      theta_inf: float = 0.0) -> np.ndarray:
    """Campo de orientación de Sherlock y Monro, en [0, π).

    θ(x, y) = θ∞ + ½·[Σ arg(z − c_k) − Σ arg(z − d_k)]  (mod π),  con z = x + i·y,
    c_k los núcleos y d_k los deltas. Con esta convención un núcleo tiene índice de
    Poincaré +½ y un delta −½ (sección 0). tamano = (alto, ancho).
    """
    alto, ancho = int(tamano[0]), int(tamano[1])
    y = np.arange(alto, dtype=np.float64)[:, None]
    x = np.arange(ancho, dtype=np.float64)[None, :]
    suma = np.zeros((alto, ancho), dtype=np.float64)
    for cx, cy in singularidades.nucleos:
        suma += np.arctan2(y - float(cy), x - float(cx))
    for dx, dy in singularidades.deltas:
        suma -= np.arctan2(y - float(dy), x - float(dx))
    return _a_cero_pi(float(theta_inf) + 0.5 * suma)


def _campo_arco(rng: np.random.Generator, alto: int, ancho: int) -> np.ndarray:
    """Campo propio del arco: crestas en forma de lomo, sin singularidades.

    Las crestas siguen la familia de curvas y = y0 − h(y0)·exp(−(x − cx)² / (2·a²)), con la
    altura del lomo h decayendo lejos del ápice; θ es el ángulo de su pendiente. Como
    |θ| < π/2 en todo el plano, el doble ángulo es continuo y no hay singularidades.
    """
    cx = ancho / 2 + rng.uniform(-15, 15)
    y_apice = alto * rng.uniform(0.38, 0.50)
    altura = rng.uniform(30, 55)
    a = rng.uniform(55, 85)
    caida = rng.uniform(100, 140)
    y = np.arange(alto, dtype=np.float64)[:, None]
    x = np.arange(ancho, dtype=np.float64)[None, :]
    h = altura * np.exp(-((y - y_apice) / caida) ** 2)
    pendiente = h * (x - cx) / a ** 2 * np.exp(-((x - cx) ** 2) / (2 * a ** 2))
    return _a_cero_pi(np.arctan(pendiente))


# --------------------------------------------------------------------------- singularidades
def _sortear_puntos(tipo: str, rng: np.random.Generator, alto: int, ancho: int):
    """Núcleos y deltas de un tipo (sección 2), cerca del centro y con márgenes de sobra."""
    cx = ancho / 2 + rng.uniform(-12, 12)
    cy = alto * 0.45 + rng.uniform(-12, 12)
    if tipo == "presilla":
        lado = 1.0 if rng.random() < 0.5 else -1.0
        return [(cx, cy)], [(cx + lado * rng.uniform(60, 85), cy + rng.uniform(75, 100))]
    if tipo == "verticilo":
        separacion = rng.uniform(20, 34)
        eje = np.pi / 2 + rng.uniform(-0.6, 0.6)          # los dos núcleos, casi uno sobre otro
        ux, uy = 0.5 * separacion * math.cos(eje), 0.5 * separacion * math.sin(eje)
        bajada = rng.uniform(75, 95)
        deltas = [
            (cx - rng.uniform(70, 90), cy + bajada + rng.uniform(-10, 10)),
            (cx + rng.uniform(70, 90), cy + bajada + rng.uniform(-10, 10)),
        ]
        return [(cx - ux, cy - uy), (cx + ux, cy + uy)], deltas
    if tipo == "arco_tienda":
        return [(cx, cy)], [(cx + rng.uniform(-6, 6), cy + rng.uniform(22, 34))]
    return [], []


def _geometria_valida(tipo: str, nucleos, deltas, alto: int, ancho: int) -> bool:
    """Reglas de la sección 2 y de 5.3, para no depender solo de los rangos de los sorteos."""
    todos = list(nucleos) + list(deltas)
    if any(min(x, y, ancho - 1 - x, alto - 1 - y) < 25 for x, y in todos):
        return False
    if any(math.dist(a, b) < 20 for a, b in itertools.combinations(todos, 2)):
        return False
    if tipo == "presilla":
        (nx, ny), (dx, dy) = nucleos[0], deltas[0]
        return dy - ny >= 45 and abs(dx - nx) >= 45
    if tipo == "verticilo":
        if not 19 <= math.dist(nucleos[0], nucleos[1]) <= 38:
            return False
        izquierdo, derecho = sorted(deltas)
        x_min, x_max = min(x for x, _ in nucleos), max(x for x, _ in nucleos)
        y_medio = (nucleos[0][1] + nucleos[1][1]) / 2
        return (x_min - izquierdo[0] >= 45 and derecho[0] - x_max >= 45
                and izquierdo[1] - y_medio >= 45 and derecho[1] - y_medio >= 45)
    if tipo == "arco_tienda":
        (nx, ny), (dx, dy) = nucleos[0], deltas[0]
        return 17 <= dy - ny <= 38 and abs(dx - nx) <= 8
    return True


def _singularidades(tipo: str, rng: np.random.Generator, alto: int, ancho: int) -> Singularidades:
    for _ in range(100):
        nucleos, deltas = _sortear_puntos(tipo, rng, alto, ancho)
        if _geometria_valida(tipo, nucleos, deltas, alto, ancho):
            return Singularidades(
                nucleos=tuple((float(x), float(y)) for x, y in nucleos),
                deltas=tuple((float(x), float(y)) for x, y in deltas),
            )
    raise RuntimeError(f"No se pudieron sortear singularidades válidas para el tipo {tipo!r}.")


# --------------------------------------------------------------------------- máscara y crestas
def _mascara_suave(rng: np.random.Generator, alto: int, ancho: int) -> np.ndarray:
    """Elipse (más larga hacia abajo, como la yema del dedo) con borde suave, en [0, 1]."""
    cx = ancho / 2 + rng.uniform(-6, 6)
    cy = alto / 2 + rng.uniform(-6, 6)
    a = ancho * rng.uniform(0.40, 0.44)
    b_arriba = alto * rng.uniform(0.40, 0.44)
    b_abajo = alto * rng.uniform(0.44, 0.48)
    y = np.arange(alto, dtype=np.float64)[:, None]
    x = np.arange(ancho, dtype=np.float64)[None, :]
    b = np.where(y < cy, b_arriba, b_abajo)
    rho = np.sqrt(((x - cx) / a) ** 2 + ((y - cy) / b) ** 2)
    distancia = (rho - 1.0) * min(a, b_arriba)                    # ≈ px afuera del borde (> 0)
    return 1.0 / (1.0 + np.exp(np.clip(distancia / BORDE_SUAVE_PX, -50, 50)))


def _banco_gabor(periodo: float, n_orientaciones: int) -> list[np.ndarray]:
    """Filtros de Gabor de media 0 para crestas de orientación πk/n (k = 0…n−1)."""
    sigma = 0.5 * periodo
    lado = int(2 * round(3 * sigma) + 1)
    banco = []
    for k in range(n_orientaciones):
        cresta = np.pi * k / n_orientaciones
        # En cv2.getGaborKernel, theta es la normal a las franjas: la cresta + π/2.
        g = cv2.getGaborKernel((lado, lado), sigma, cresta + np.pi / 2, periodo, 1.0, 0, ktype=cv2.CV_32F)
        g -= g.mean()
        g /= np.abs(g).sum()
        banco.append(g)
    return banco


def _sintetizar_crestas(orientacion: np.ndarray, dentro: np.ndarray, periodo: float,
                        rng: np.random.Generator, iteraciones: int, n_orientaciones: int) -> np.ndarray:
    """Crestas en [−1, 1] por filtrado de Gabor iterativo desde ruido uniforme."""
    banco = _banco_gabor(periodo, n_orientaciones)
    posicion = orientacion / np.pi * n_orientaciones                 # en [0, n)
    k0 = np.floor(posicion).astype(np.intp) % n_orientaciones
    k1 = (k0 + 1) % n_orientaciones
    peso = (posicion - np.floor(posicion)).astype(np.float32)
    crestas = rng.uniform(-1.0, 1.0, orientacion.shape).astype(np.float32)
    for _ in range(iteraciones):
        respuestas = np.stack([cv2.filter2D(crestas, -1, g, borderType=cv2.BORDER_REFLECT) for g in banco])
        r0 = np.take_along_axis(respuestas, k0[None], 0)[0]
        r1 = np.take_along_axis(respuestas, k1[None], 0)[0]
        crestas = (1.0 - peso) * r0 + peso * r1
        escala = float(np.percentile(np.abs(crestas[dentro]), 90)) + 1e-9
        crestas = np.clip(crestas / escala, -1.0, 1.0)
    return crestas


def _realzar(crestas: np.ndarray, dentro: np.ndarray, periodo: float) -> np.ndarray:
    """Contraste local parejo y bordes de cresta definidos, en [−1, 1].

    Se divide por la amplitud local (√2 · RMS en un entorno de unos 0,75 períodos, con un piso
    para no amplificar zonas casi vacías) y se aplica tanh(K·c)/tanh(K), con K = NITIDEZ.
    """
    crestas = crestas.astype(np.float64)
    amplitud = np.sqrt(cv2.GaussianBlur(crestas * crestas, (0, 0), 0.75 * periodo))
    piso = 0.25 * float(np.median(amplitud[dentro])) + 1e-9
    normalizadas = crestas / (np.sqrt(2.0) * np.maximum(amplitud, piso))
    return np.tanh(NITIDEZ * normalizadas) / np.tanh(NITIDEZ)


# --------------------------------------------------------------------------- base
def generar_base(tipo: str, semilla: int, cfg: dict) -> HuellaBase:
    """Huella base de una identidad (sección 2). ValueError si tipo no está en TIPOS."""
    if tipo not in TIPOS:
        raise ValueError(f"Tipo de huella desconocido: {tipo!r}. Tiene que ser uno de {TIPOS}.")
    generador = cfg["huellas"]["generador"]
    alto, ancho = (int(v) for v in generador["tamano_px"])
    periodo_min, periodo_max = (float(v) for v in generador["periodo_base_px"])
    rng = np.random.default_rng(int(semilla))

    singularidades = _singularidades(tipo, rng, alto, ancho)
    if tipo == "arco":
        orientacion = _campo_arco(rng, alto, ancho)
    else:
        orientacion = campo_orientacion(singularidades, (alto, ancho), 0.0)
    suave = _mascara_suave(rng, alto, ancho)
    mascara = suave > 0.5
    periodo = float(rng.uniform(periodo_min, periodo_max))
    crestas = _sintetizar_crestas(orientacion, mascara, periodo, rng,
                                  int(generador["iteraciones"]), int(generador["orientaciones"]))
    crestas = _realzar(crestas, mascara, periodo)
    piel = GRIS_MEDIO - CONTRASTE * crestas
    imagen = np.clip(np.rint(suave * piel + (1.0 - suave) * FONDO), 0, 255).astype(np.uint8)
    return HuellaBase(
        tipo=tipo,
        semilla=int(semilla),
        imagen=imagen,
        mascara=mascara,
        orientacion=orientacion,
        singularidades=singularidades,
        periodo_px=periodo,
    )
