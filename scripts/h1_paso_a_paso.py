"""H1 paso a paso: cómo se fabrica la huella de UNA identidad, con una figura por cada paso.

No es parte del pipeline ni tiene tests: es una herramienta para entender (y explicar) el
generador de H1. Rehace el código de src/huellas/generador.py y variantes.py en el mismo
orden y con los mismos sorteos, pero partido en pasos chicos, comentado, con lo que pasa en
cada uno impreso y dibujado. Con --verificar compara el resultado contra el generador del
repo, píxel por píxel.

Qué hace, en orden (el número es el de la figura que guarda):
  Etapa A · el molde de la identidad
    01 sortear núcleos y deltas          06 una pasada de Gabor, por dentro
    02 campo de orientación θ(x, y)      07 las 15 pasadas (ruido → crestas)
    03 máscara elíptica y período        08 realce (contraste local + tanh)
    04 banco de 16 filtros de Gabor      09 máscara + fondo = huella base
    05 ruido inicial (la semilla)
  Etapa B · una toma (variante) de ese dedo
    10 parámetros sorteados              13 dónde quedaron el núcleo y el delta
    11 presión (erosión / dilatación)    14 corte periférico
    12 geometría: afín + elástica        15 ruido gaussiano + sal y pimienta
  y al final: resumen.png (todo junto) y pasadas.gif (el ruido volviéndose crestas).

Uso, desde la raíz del repo:
    venv\\Scripts\\python scripts\\h1_paso_a_paso.py                    (identidad 1, toma 11)
    venv\\Scripts\\python scripts\\h1_paso_a_paso.py --identidad 2
    venv\\Scripts\\python scripts\\h1_paso_a_paso.py --mostrar          (abre cada figura y espera que la cierres)
    venv\\Scripts\\python scripts\\h1_paso_a_paso.py --verificar        (compara con src/huellas)
    venv\\Scripts\\python scripts\\h1_paso_a_paso.py --tipo presilla --semilla 999   (otra "persona")
Las figuras quedan en data/h1_paso_a_paso/id1/ (data/ no va a Git). Necesita NumPy, OpenCV,
SciPy y Matplotlib, que ya están en el venv.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------- constantes
# Copia de config.yaml (sección huellas) y de las constantes de generador.py. Si cambiás algo
# en config.yaml, --verificar te avisa que ya no coincide.
ALTO, ANCHO = 400, 320                 # tamano_px
PERIODO_BASE_PX = (8.7, 9.3)           # periodo_base_px
ITERACIONES = 15                       # pasadas de Gabor
ORIENTACIONES = 16                     # filtros del banco
VARIANTES = dict(rotacion_max_grados=15.0, traslacion_max_px=15.0, escala_max=0.08, presion_max=0.7,
                 elastica_max_px=3.0, ruido_sigma_max=10.0, sal_pimienta_max=0.01, prob_corte=0.3,
                 corte_periferico=(0.05, 0.10))
IDENTIDADES = {1: ("presilla", 101), 2: ("verticilo", 202), 3: ("arco_tienda", 303)}

FONDO = 245                            # gris del fondo, afuera de la huella
GRIS_MEDIO, CONTRASTE = 128.0, 100.0   # píxel = 128 − 100·c   (c = 1: cresta oscura)
BORDE_SUAVE_PX = 3.0                   # borde suave de la máscara
NITIDEZ = 2.0                          # K de tanh(K·c)/tanh(K)
SIGMA_PUNTO_PX = 2.5                   # punto gaussiano que sigue al núcleo
MARGEN_CORTE_PX = 30.0                 # el corte no se acerca a menos de esto de un núcleo
PASO_CORTE_GRADOS = 30.0
GRILLA_ELASTICA = (5, 4)               # (filas, columnas) de vectores al azar

ROJO, AZUL, VERDE = "#c8262a", "#1b5fc4", "#14873f"

# se completan en main(): la salida y matplotlib (se elige el backend según --mostrar)
SALIDA: Path = Path(".")
MOSTRAR = False
plt = None


# ============================================================================= utilidades
def hablar(texto: str = "") -> None:
    """Imprime un párrafo cortado a 100 columnas (respeta las líneas en blanco)."""
    for parrafo in texto.split("\n"):
        print(textwrap.fill(parrafo, 100, subsequent_indent="   ") if parrafo.strip() else "")


def encabezado(n: int, titulo: str) -> None:
    print("\n" + "=" * 100)
    print(f"PASO {n} · {titulo}")
    print("=" * 100)


def a_gris(c: np.ndarray) -> np.ndarray:
    """Crestas c en [−1, 1] → gris uint8 con la misma convención que la huella (cresta = oscuro)."""
    return np.clip(np.rint(GRIS_MEDIO - CONTRASTE * c), 0, 255).astype(np.uint8)


def guardar_png(ruta: Path, imagen: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", imagen)
    assert ok
    buf.tofile(str(ruta))               # tofile: anda aunque la ruta tenga acentos o espacios


def mostrar_gris(ax, imagen, titulo: str = "", **kw) -> None:
    ax.imshow(imagen, cmap="gray", vmin=0, vmax=255, interpolation="nearest", **kw)
    ax.set_title(titulo, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])


def marcar(ax, nucleos, deltas, r: float = 9.0) -> None:
    """Círculo rojo = núcleo, triángulo azul = delta."""
    from matplotlib.patches import Circle, Polygon
    for x, y in nucleos:
        ax.add_patch(Circle((x, y), r, fill=False, ec=ROJO, lw=2))
        ax.plot([x], [y], ".", color=ROJO, ms=4)
    for x, y in deltas:
        ax.add_patch(Polygon([(x, y - r), (x + r * 0.9, y + r * 0.65), (x - r * 0.9, y + r * 0.65)],
                             closed=True, fill=False, ec=AZUL, lw=2))


def figura(filas: int, columnas: int, ancho_panel: float = 3.3, alto_panel: float = 4.3):
    fig, ejes = plt.subplots(filas, columnas, figsize=(ancho_panel * columnas, alto_panel * filas + 0.9))
    return fig, np.atleast_1d(ejes).ravel()


def cerrar(fig, n: int, nombre: str, titulo: str, pie: str = "") -> None:
    """Pone el título del paso, guarda la figura (y la muestra si pediste --mostrar)."""
    fig.suptitle(f"Paso {n} · {titulo}", fontsize=14, fontweight="bold", x=0.012, ha="left")
    alto_pulgadas = fig.get_figheight()
    inferior = 0.0
    if pie:
        envuelto = "\n".join(textwrap.wrap(pie, max(40, int(fig.get_figwidth() * 12.5))))
        fig.text(0.012, 0.15 / alto_pulgadas, envuelto, fontsize=9.5, ha="left", va="bottom", color="#333333")
        inferior = (0.3 + 0.17 * (envuelto.count("\n") + 1)) / alto_pulgadas
    fig.tight_layout(rect=[0, inferior, 1, 0.94])
    fig.canvas.draw()                                   # con las proporciones fijas, el 2º ajuste ya ve los títulos reales
    fig.tight_layout(rect=[0, inferior, 1, 0.94], h_pad=1.6, w_pad=1.2)
    ruta = SALIDA / f"{n:02d}_{nombre}.png"
    fig.savefig(ruta, dpi=110)
    print(f"   [figura guardada: {ruta}]")
    if MOSTRAR:
        plt.show()
    plt.close(fig)


# ============================================================================= ETAPA A
# --- paso 1: singularidades ---------------------------------------------------------------
def _a_cero_pi(theta: np.ndarray) -> np.ndarray:
    """Lleva un ángulo a [0, π): una raya no tiene sentido, 10° y 190° son lo mismo."""
    theta = np.mod(theta, np.pi)
    theta[theta >= np.pi] = 0.0
    return theta


def _sortear_puntos(tipo: str, rng: np.random.Generator):
    """Sortea dónde van los núcleos y los deltas de un tipo, cerca del centro."""
    cx = ANCHO / 2 + rng.uniform(-12, 12)
    cy = ALTO * 0.45 + rng.uniform(-12, 12)
    if tipo == "presilla":                       # 1 núcleo y 1 delta, a un costado y más abajo
        lado = 1.0 if rng.random() < 0.5 else -1.0
        return [(cx, cy)], [(cx + lado * rng.uniform(60, 85), cy + rng.uniform(75, 100))], (cx, cy)
    if tipo == "verticilo":                      # 2 núcleos casi uno sobre otro y 2 deltas, uno a cada lado
        separacion = rng.uniform(20, 34)
        eje = np.pi / 2 + rng.uniform(-0.6, 0.6)
        ux, uy = 0.5 * separacion * math.cos(eje), 0.5 * separacion * math.sin(eje)
        bajada = rng.uniform(75, 95)
        deltas = [(cx - rng.uniform(70, 90), cy + bajada + rng.uniform(-10, 10)),
                  (cx + rng.uniform(70, 90), cy + bajada + rng.uniform(-10, 10))]
        return [(cx - ux, cy - uy), (cx + ux, cy + uy)], deltas, (cx, cy)
    if tipo == "arco_tienda":                    # 1 núcleo y 1 delta casi uno sobre otro
        return [(cx, cy)], [(cx + rng.uniform(-6, 6), cy + rng.uniform(22, 34))], (cx, cy)
    return [], [], (cx, cy)                      # arco: sin singularidades


def _geometria_valida(tipo: str, nucleos, deltas) -> bool:
    """Reglas del contrato: a más de 25 px del borde, separados, y la forma propia de cada tipo."""
    todos = list(nucleos) + list(deltas)
    if any(min(x, y, ANCHO - 1 - x, ALTO - 1 - y) < 25 for x, y in todos):
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


def paso_1_singularidades(E: dict) -> None:
    encabezado(1, "Sortear dónde van los núcleos y los deltas")
    tipo, rng = E["tipo"], E["rng"]
    for intento in range(1, 101):
        nucleos, deltas, centro = _sortear_puntos(tipo, rng)
        if _geometria_valida(tipo, nucleos, deltas):
            break
    else:
        raise RuntimeError("no salió una geometría válida")
    E.update(nucleos=tuple((float(x), float(y)) for x, y in nucleos),
             deltas=tuple((float(x), float(y)) for x, y in deltas))
    hablar(f"Tipo: {tipo}, semilla {E['semilla']}. Todos los sorteos del generador salen de UN solo "
           f"np.random.default_rng({E['semilla']}): por eso la misma semilla da siempre la misma huella.")
    hablar(f"Salió válido al intento {intento}. Núcleos: "
           f"{[(round(x, 1), round(y, 1)) for x, y in E['nucleos']]}. Deltas: "
           f"{[(round(x, 1), round(y, 1)) for x, y in E['deltas']]}.")
    hablar("Un núcleo es donde las crestas dan una U (índice de Poincaré +1/2) y un delta es donde se "
           "abren en tres ramas (−1/2). Hay tantos núcleos como deltas: así, lejos de todo, las crestas "
           "vuelven a ser horizontales.")

    from matplotlib.patches import Rectangle
    fig, ejes = figura(1, 2, 5.2, 5.6)
    ax = ejes[0]
    ax.set_xlim(0, ANCHO); ax.set_ylim(ALTO, 0); ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), ANCHO, ALTO, fill=False, ec="#888888"))
    ax.add_patch(Rectangle((25, 25), ANCHO - 50, ALTO - 50, fill=False, ec="#bbbbbb", ls=":"))
    cx, cy = centro
    if tipo == "presilla":
        for signo in (-1, 1):
            ax.add_patch(Rectangle((cx + signo * 85 if signo > 0 else cx - 85, cy + 75), 25, 25,
                                   fill=True, fc="#1b5fc41f", ec=AZUL, ls="--", lw=1))
    elif tipo == "verticilo":
        for x0 in (cx - 90, cx + 70):
            ax.add_patch(Rectangle((x0, cy + 65), 20, 40, fill=True, fc="#1b5fc41f", ec=AZUL, ls="--", lw=1))
    elif tipo == "arco_tienda":
        ax.add_patch(Rectangle((cx - 6, cy + 22), 12, 12, fill=True, fc="#1b5fc41f", ec=AZUL, ls="--", lw=1))
    ax.add_patch(Rectangle((ANCHO / 2 - 12, ALTO * 0.45 - 12), 24, 24, fill=False, ec=ROJO, ls="--", lw=1))
    marcar(ax, E["nucleos"], E["deltas"])
    ax.set_title("la hoja de 400 × 320 px y dónde cayeron", fontsize=9.5)
    ax.set_xlabel("rojo punteado: zona del centro · azul punteado: zona posible del delta\ngris punteado: margen de 25 px del borde", fontsize=8)
    ax = ejes[1]; ax.axis("off")
    filas = [f"tipo: {tipo}", f"semilla: {E['semilla']}", f"intentos hasta una geometría válida: {intento}", ""]
    filas += [f"núcleo {i + 1}: x = {x:6.1f}   y = {y:6.1f}" for i, (x, y) in enumerate(E["nucleos"])]
    filas += [f"delta  {i + 1}: x = {x:6.1f}   y = {y:6.1f}" for i, (x, y) in enumerate(E["deltas"])]
    filas += ["", "reglas que tiene que cumplir:", "  · a ≥ 25 px del borde", "  · separados ≥ 20 px entre sí", "  · la forma propia de cada tipo",
              "    (ej. presilla: delta ≥ 45 px más abajo", "     y ≥ 45 px al costado del núcleo)"]
    ax.text(0.0, 0.95, "\n".join(filas), family="monospace", fontsize=10, va="top")
    cerrar(fig, 1, "singularidades", "Dónde van los núcleos y los deltas",
           "El generador sortea posiciones cerca del centro y las acepta si cumplen la geometría de su tipo. Estas coordenadas son la "
           "'verdad terreno' de la identidad: después se las sigue en cada toma.")


# --- paso 2: campo de orientación ---------------------------------------------------------
def campo_orientacion(nucleos, deltas) -> np.ndarray:
    """θ(x, y) = θ∞ + ½·[Σ arg(z − c) − Σ arg(z − d)]  (mod π),  con θ∞ = 0 y z = x + i·y."""
    y = np.arange(ALTO, dtype=np.float64)[:, None]
    x = np.arange(ANCHO, dtype=np.float64)[None, :]
    suma = np.zeros((ALTO, ANCHO), dtype=np.float64)
    for cx, cy in nucleos:
        suma += np.arctan2(y - float(cy), x - float(cx))
    for dx, dy in deltas:
        suma -= np.arctan2(y - float(dy), x - float(dx))
    return _a_cero_pi(0.0 + 0.5 * suma)


def campo_arco(rng: np.random.Generator) -> np.ndarray:
    """El arco no tiene singularidades: las crestas siguen un lomo y = y0 − h·exp(−(x−cx)²/2a²)."""
    cx = ANCHO / 2 + rng.uniform(-15, 15)
    y_apice = ALTO * rng.uniform(0.38, 0.50)
    altura = rng.uniform(30, 55)
    a = rng.uniform(55, 85)
    caida = rng.uniform(100, 140)
    y = np.arange(ALTO, dtype=np.float64)[:, None]
    x = np.arange(ANCHO, dtype=np.float64)[None, :]
    h = altura * np.exp(-((y - y_apice) / caida) ** 2)
    pendiente = h * (x - cx) / a ** 2 * np.exp(-((x - cx) ** 2) / (2 * a ** 2))
    return _a_cero_pi(np.arctan(pendiente))


def paso_2_campo(E: dict) -> None:
    encabezado(2, "Campo de orientación: hacia dónde apuntan las crestas en cada píxel")
    tipo = E["tipo"]
    if tipo == "arco":
        E["orient"] = campo_arco(E["rng"])
        hablar("El arco no tiene núcleos ni deltas: usa un campo propio en forma de lomo.")
    else:
        E["orient"] = campo_orientacion(E["nucleos"], E["deltas"])
        hablar("Modelo de Sherlock y Monro:  θ(z) = θ∞ + ½ · [ Σ arg(z − c) − Σ arg(z − d) ]  (mod π),  con z = x + i·y.")
        hablar("arg(z − c) es el ángulo bajo el que ves el píxel z desde el núcleo c. Si das una vuelta alrededor "
               "de c, ese ángulo gira 360°; θ gira la mitad (180°): las crestas hacen una U. Con un delta es al revés.")
    th = E["orient"]
    hablar(f"Resultado: una matriz de {th.shape[0]} × {th.shape[1]} ángulos en [0, π). Mínimo {th.min():.3f} rad, máximo {th.max():.3f} rad.")

    y = np.arange(ALTO, dtype=np.float64)[:, None]
    x = np.arange(ANCHO, dtype=np.float64)[None, :]
    suma_c = sum((np.arctan2(y - cy, x - cx) for cx, cy in E["nucleos"]), np.zeros((ALTO, ANCHO)))
    suma_d = sum((np.arctan2(y - cy, x - cx) for cx, cy in E["deltas"]), np.zeros((ALTO, ANCHO)))
    hsv = plt.get_cmap("hsv")
    fig, ejes = figura(1, 4, 3.3, 4.6)
    if tipo != "arco":
        ejes[0].imshow(hsv(np.mod(suma_c, 2 * np.pi) / (2 * np.pi)))
        ejes[0].set_title("Σ arg(z − c)\n(el color es el ángulo, 0–360°)", fontsize=9.5)
        ejes[1].imshow(hsv(np.mod(suma_d, 2 * np.pi) / (2 * np.pi)))
        ejes[1].set_title("Σ arg(z − d)", fontsize=9.5)
    else:
        for a in ejes[:2]:
            a.axis("off")
        ejes[0].text(0.5, 0.5, "el arco usa un campo\npropio (sin núcleos\nni deltas)", ha="center", va="center", transform=ejes[0].transAxes)
    ejes[2].imshow(hsv(th / np.pi))
    ejes[2].set_title("θ = ½·[Σc − Σd]  mod π\n(el color es el ángulo, 0–180°)", fontsize=9.5)
    paso = 16
    ys, xs = np.mgrid[paso // 2:ALTO:paso, paso // 2:ANCHO:paso]
    t = th[ys, xs]
    ejes[3].set_facecolor("#f5f5f5")
    ejes[3].quiver(xs.ravel(), ys.ravel(), np.cos(t).ravel(), -np.sin(t).ravel(), angles="xy", scale_units="xy",
                   scale=0.115, pivot="mid", headwidth=0, headlength=0, headaxislength=0, width=0.006,
                   color=hsv((t / np.pi).ravel()))
    ejes[3].set_xlim(0, ANCHO); ejes[3].set_ylim(ALTO, 0); ejes[3].set_aspect("equal")
    ejes[3].set_title("las 'brújulas' (sin punta: es una raya)", fontsize=9.5)
    for a in ejes:
        a.set_xticks([]); a.set_yticks([])
        marcar(a, E["nucleos"], E["deltas"])
        if not a.images and a is not ejes[3]:
            pass
    cerrar(fig, 2, "campo_orientacion", "Campo de orientación θ(x, y)",
           "Cada píxel termina con un ángulo. Todavía no hay ninguna cresta dibujada: es solo el mapa de hacia dónde van a apuntar. "
           "Sherlock y Monro lo calcula con una fórmula y no se dibuja a mano.")


# --- paso 3: máscara y período ------------------------------------------------------------
def mascara_suave(rng: np.random.Generator):
    """Elipse más larga hacia abajo (la yema del dedo), con borde suave. Devuelve (suave, parámetros)."""
    cx = ANCHO / 2 + rng.uniform(-6, 6)
    cy = ALTO / 2 + rng.uniform(-6, 6)
    a = ANCHO * rng.uniform(0.40, 0.44)
    b_arriba = ALTO * rng.uniform(0.40, 0.44)
    b_abajo = ALTO * rng.uniform(0.44, 0.48)
    y = np.arange(ALTO, dtype=np.float64)[:, None]
    x = np.arange(ANCHO, dtype=np.float64)[None, :]
    b = np.where(y < cy, b_arriba, b_abajo)
    rho = np.sqrt(((x - cx) / a) ** 2 + ((y - cy) / b) ** 2)
    distancia = (rho - 1.0) * min(a, b_arriba)                    # ≈ píxeles afuera del borde
    suave = 1.0 / (1.0 + np.exp(np.clip(distancia / BORDE_SUAVE_PX, -50, 50)))
    return suave, dict(cx=cx, cy=cy, a=a, b_arriba=b_arriba, b_abajo=b_abajo)


def paso_3_mascara(E: dict) -> None:
    encabezado(3, "Máscara elíptica y período de las crestas")
    rng = E["rng"]
    suave, p = mascara_suave(rng)
    E["suave"], E["mascara"] = suave, suave > 0.5
    E["periodo"] = float(rng.uniform(*PERIODO_BASE_PX))
    hablar(f"Elipse de semiejes a = {p['a']:.1f} px (horizontal), b = {p['b_arriba']:.1f} px (arriba) y "
           f"{p['b_abajo']:.1f} px (abajo), centrada en ({p['cx']:.1f}, {p['cy']:.1f}). Vale 1 adentro y 0 afuera, y en el borde pasa de uno a otro "
           f"con una curva logística de escala {BORDE_SUAVE_PX:.0f} px (el tramo de 10 % a 90 % ocupa unos {4.4 * BORDE_SUAVE_PX:.0f} px): así el borde no es un escalón.")
    hablar(f"Período de las crestas de esta identidad: {E['periodo']:.3f} px (sorteo entre {PERIODO_BASE_PX[0]} y {PERIODO_BASE_PX[1]}). "
           f"Es la distancia de cresta a cresta, y decide el tamaño de los filtros de Gabor.")
    hablar(f"La máscara binaria (suave > 0,5) cubre {int(E['mascara'].sum())} px = {100 * E['mascara'].mean():.1f} % de la hoja.")
    fig, ejes = figura(1, 3, 3.3, 4.3)
    mostrar_gris(ejes[0], suave * 255, "máscara suave (blanco = 1)")
    mostrar_gris(ejes[1], E["mascara"] * 255, "máscara binaria (suave > 0,5)")
    hsv = plt.get_cmap("hsv")
    ejes[2].imshow(hsv(E["orient"] / np.pi) * (0.35 + 0.65 * E["mascara"][..., None]))
    ejes[2].set_title("campo θ solo dentro de la huella", fontsize=9.5)
    ejes[2].set_xticks([]); ejes[2].set_yticks([])
    cerrar(fig, 3, "mascara", "Máscara elíptica (la forma de la yema)",
           f"La máscara solo recorta: las crestas se sintetizan en toda la hoja y después se muestran adentro. "
           f"Período elegido: {E['periodo']:.2f} px.")


# --- paso 4: banco de Gabor ---------------------------------------------------------------
def banco_gabor(periodo: float, n: int):
    """n filtros de Gabor (media 0), uno por ángulo de cresta πk/n."""
    sigma = 0.5 * periodo
    lado = int(2 * round(3 * sigma) + 1)
    banco = []
    for k in range(n):
        cresta = np.pi * k / n
        # En cv2.getGaborKernel, theta es la normal a las franjas: la cresta + π/2.
        g = cv2.getGaborKernel((lado, lado), sigma, cresta + np.pi / 2, periodo, 1.0, 0, ktype=cv2.CV_32F)
        g -= g.mean()
        g /= np.abs(g).sum()
        banco.append(g)
    return banco, sigma, lado


def paso_4_banco(E: dict) -> None:
    encabezado(4, "Banco de filtros de Gabor")
    banco, sigma, lado = banco_gabor(E["periodo"], ORIENTACIONES)
    E["banco"] = banco
    hablar(f"Un filtro de Gabor es una onda de período λ = {E['periodo']:.2f} px recortada por una campana gaussiana de "
           f"σ = λ/2 = {sigma:.2f} px:  g = exp(−(x′² + y′²)/2σ²) · cos(2π·x′/λ),  con x′ perpendicular a la cresta.")
    hablar(f"Cada kernel mide {lado} × {lado} px, tiene media 0 y suma de |g| igual a 1. Hay {ORIENTACIONES} filtros, uno cada "
           f"{180 / ORIENTACIONES:.2f}° (de 0° a {180 - 180 / ORIENTACIONES:.2f}°). Un filtro 'prende' donde hay rayas de SU ángulo y "
           f"período, y se apaga con las demás.")
    tiles = []
    for g in banco:
        t = np.clip(128 + 127 * g / np.abs(g).max(), 0, 255).astype(np.uint8)
        tiles.append(np.pad(cv2.resize(t, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST), 2, constant_values=255))
    montaje = np.vstack([np.hstack(tiles[:8]), np.hstack(tiles[8:])])
    fig, ejes = plt.subplots(2, 1, figsize=(10.5, 7.6), gridspec_kw=dict(height_ratios=[1.25, 1]))
    mostrar_gris(ejes[0], montaje, f"los {ORIENTACIONES} filtros: ángulos 0°, 11,25°, 22,5° … 168,75° (gris = 0)")
    perfil = banco[0][:, lado // 2]
    x = np.arange(lado) - lado // 2
    ejes[1].plot(x, perfil, color=AZUL, lw=2)
    ejes[1].axhline(0, color="#999999", lw=0.8)
    envolvente = perfil.max() * np.exp(-x ** 2 / (2 * sigma ** 2))
    ejes[1].plot(x, envolvente, "--", color="#999999", lw=1.2, label="campana gaussiana (σ = λ/2)")
    ejes[1].plot(x, -envolvente, "--", color="#999999", lw=1.2)
    ejes[1].legend(fontsize=8.5, loc="upper right")
    ejes[1].set_xlabel("posición perpendicular a la cresta (px)")
    ejes[1].set_ylabel("peso del kernel")
    ejes[1].set_title(f"perfil del filtro de 0°: una onda de {E['periodo']:.1f} px dentro de una campana", fontsize=9.5)
    cerrar(fig, 4, "banco_gabor", "Banco de Gabor: 16 'peines' con distinta inclinación",
           "Más adelante, en cada píxel se usa el peine cuya inclinación coincide con θ(x, y). Esa es la forma de obligar al ruido a ordenarse "
           "en rayas que siguen el campo.")


# --- paso 5: ruido -------------------------------------------------------------------------
def paso_5_ruido(E: dict) -> None:
    encabezado(5, "Ruido inicial (acá entra la semilla)")
    ruido = E["rng"].uniform(-1.0, 1.0, E["orient"].shape).astype(np.float32)
    E["ruido"] = ruido
    hablar(f"Una matriz de {ruido.shape[0]} × {ruido.shape[1]} valores uniformes en [−1, 1] del mismo rng (con la semilla "
           f"{E['semilla']}): ruido continuo, no solo grises y negros. Media {ruido.mean():+.3f}, desvío {ruido.std():.3f} "
           f"(uniforme en [−1, 1]: 0,577). No hay ninguna estructura: todo lo que veas después sale de filtrar esto.")
    fig, ejes = figura(1, 2, 4.2, 4.6)
    mostrar_gris(ejes[0], a_gris(ruido), "ruido (pasada 0)")
    ejes[1].hist(ruido.ravel(), bins=40, color=AZUL)
    ejes[1].set_title("histograma: plano de −1 a 1", fontsize=9.5)
    ejes[1].set_xlabel("valor"); ejes[1].set_ylabel("píxeles")
    cerrar(fig, 5, "ruido", "Ruido inicial",
           "Si cambiás la semilla, el ruido cae en otros píxeles y sale otra 'persona'. Con la misma semilla, siempre el mismo.")


# --- paso 6: una pasada, por dentro --------------------------------------------------------
def _seleccion(E: dict):
    posicion = E["orient"] / np.pi * ORIENTACIONES
    k0 = np.floor(posicion).astype(np.intp) % ORIENTACIONES
    k1 = (k0 + 1) % ORIENTACIONES
    peso = (posicion - np.floor(posicion)).astype(np.float32)
    return posicion, k0, k1, peso


def una_pasada(crestas, banco, k0, k1, peso, dentro):
    """Una pasada: filtra con los 16 filtros, elige en cada píxel el de su ángulo y normaliza."""
    respuestas = np.stack([cv2.filter2D(crestas, -1, g, borderType=cv2.BORDER_REFLECT) for g in banco])
    r0 = np.take_along_axis(respuestas, k0[None], 0)[0]
    r1 = np.take_along_axis(respuestas, k1[None], 0)[0]
    elegida = (1.0 - peso) * r0 + peso * r1
    escala = float(np.percentile(np.abs(elegida[dentro]), 90)) + 1e-9
    return respuestas, elegida, np.clip(elegida / escala, -1.0, 1.0), escala


def paso_6_una_pasada(E: dict) -> None:
    encabezado(6, "Una pasada de Gabor, por dentro")
    posicion, k0, k1, peso = _seleccion(E)
    E["k0"], E["k1"], E["peso"] = k0, k1, peso
    respuestas, elegida, norm, escala = una_pasada(E["ruido"], E["banco"], k0, k1, peso, E["mascara"])
    hablar("1) Se pasa el ruido por los 16 filtros: salen 16 imágenes, cada una realza las rayas de su ángulo.")
    hablar("2) En cada píxel se mira θ(x, y) del campo y se toma la respuesta del filtro de ESE ángulo. Como θ cae entre dos "
           "filtros vecinos, se mezclan los dos: (1 − p)·r[k] + p·r[k+1], con k = piso(θ·16/π) y p la parte fraccionaria.")
    hablar(f"3) Se divide por el percentil 90 de |c| dentro de la máscara (acá {escala:.4f}) y se recorta a [−1, 1] "
           f"para que la amplitud no se dispare en las pasadas siguientes.")
    hablar("Resultado: rayas cortas que ya van en la dirección del campo. Con las pasadas siguientes se ordenan más.")
    zy, zx = int(E["nucleos"][0][1]) if E["nucleos"] else ALTO // 2, int(E["nucleos"][0][0]) if E["nucleos"] else ANCHO // 2
    y0, x0 = int(np.clip(zy - 40, 0, ALTO - 80)), int(np.clip(zx - 30, 0, ANCHO - 60))
    fig, ejes = figura(2, 4, 3.0, 3.9)
    mostrar_gris(ejes[0], a_gris(E["ruido"]), "ruido de entrada")
    for ax, k in zip(ejes[1:4], (0, 4, 8)):
        mostrar_gris(ax, a_gris(np.clip(respuestas[k] / (np.abs(respuestas[k]).max() + 1e-9), -1, 1)),
                     f"respuesta del filtro k = {k}\n(rayas a {180 * k / ORIENTACIONES:.1f}°)")
    ejes[4].imshow(plt.get_cmap("hsv")(posicion / ORIENTACIONES))
    ejes[4].set_title("θ(x, y) del campo: dice qué\nfiltro usar en cada píxel", fontsize=9.5)
    ejes[4].set_xticks([]); ejes[4].set_yticks([])
    mostrar_gris(ejes[5], a_gris(np.clip(elegida / np.abs(elegida).max(), -1, 1)), "respuesta elegida por píxel\n(mezcla de los 2 filtros vecinos)")
    mostrar_gris(ejes[6], a_gris(norm), "÷ percentil 90 y recorte\n= pasada 1")
    mostrar_gris(ejes[7], a_gris(norm[y0:y0 + 80, x0:x0 + 60]), "zoom de la pasada 1\n(80 × 60 px cerca del núcleo)")
    ejes[7].images[0].set_interpolation("nearest")
    cerrar(fig, 6, "una_pasada", "Una pasada de Gabor, por dentro",
           "Arriba: el ruido y 3 de las 16 respuestas (cada filtro ve rayas de su ángulo). Abajo: el campo elige, píxel a píxel, cuál respuesta "
           "usar; eso es lo que ordena el ruido.")


# --- paso 7: las 15 pasadas -------------------------------------------------------------------
def coherencia(c: np.ndarray, th_teorico: np.ndarray, mascara: np.ndarray) -> float:
    """Qué tanto las crestas siguen el campo teórico: media de cos 2Δθ (1 = perfecto, 0 = azar)."""
    from scipy import ndimage as ndi
    gx = cv2.Sobel(c.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(c.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    s = 8.0
    gxx = ndi.gaussian_filter(gx * gx, s)
    gyy = ndi.gaussian_filter(gy * gy, s)
    gxy = ndi.gaussian_filter(gx * gy, s)
    th = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + np.pi / 2     # orientación local de las crestas
    m = ndi.binary_erosion(mascara, iterations=15)
    return float(np.mean(np.cos(2 * (th - th_teorico))[m]))


def paso_7_pasadas(E: dict) -> None:
    encabezado(7, f"Repetir {ITERACIONES} veces: el ruido se vuelve crestas")
    crestas = E["ruido"].copy()
    pasadas = [crestas.copy()]
    for _ in range(ITERACIONES):
        _, _, crestas, _ = una_pasada(crestas, E["banco"], E["k0"], E["k1"], E["peso"], E["mascara"])
        pasadas.append(crestas.copy())
    E["pasadas"], E["crestas"] = pasadas, crestas
    try:
        coh = [coherencia(c, E["orient"], E["mascara"]) for c in pasadas]
        hablar(f"Qué tanto siguen las crestas al campo teórico (cos 2Δθ medio; 1 = perfecto, 0 = azar): ruido {coh[0]:.2f}, "
               f"pasada 1 {coh[1]:.3f}, pasada 5 {coh[5]:.3f}, pasada 15 {coh[15]:.3f}.")
        hablar("O sea: con UNA pasada la dirección ya sale casi perfecta. Las otras 14 unen tramos de cresta, borran defectos y dejan el "
               "período parejo (15 pasadas y 16 filtros son parámetros de config.yaml, no constantes de la teoría). Las minucias "
               "(terminaciones y bifurcaciones) aparecen solas donde el filtrado deja una cresta cortada o partida: nadie las dibuja.")
    except ImportError:
        coh = None
        hablar("(Sin SciPy no se calcula la coherencia con el campo; el resto anda igual.)")
    fig = plt.figure(figsize=(15.5, 8.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.5])
    elegidas = [0, 1, 2, 3, 5, 8, 12, 15]
    for i, k in enumerate(elegidas):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        mostrar_gris(ax, a_gris(pasadas[k]), "ruido (pasada 0)" if k == 0 else f"pasada {k}")
        if k == 15:
            marcar(ax, E["nucleos"], E["deltas"])
    ax = fig.add_subplot(gs[:, 4])
    if coh is not None:
        ax.plot(range(len(coh)), coh, "o-", color=AZUL, ms=4)
        ax.set_ylim(0, 1.02); ax.set_xlabel("pasada de Gabor"); ax.set_ylabel("cos 2Δθ medio")
        ax.set_title("cuánto siguen las crestas\nal campo teórico", fontsize=9.5)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")
    cerrar(fig, 7, "pasadas", "Las 15 pasadas: del ruido a las crestas",
           "El orden aparece casi de golpe en la pasada 1; el resto limpia. Todavía las crestas son 'blandas': la onda pasa por grises "
           "entre cresta y valle. Eso se arregla en el próximo paso.")


# --- paso 8: realce ---------------------------------------------------------------------------
def realzar(crestas: np.ndarray, dentro: np.ndarray, periodo: float):
    """Contraste local parejo + bordes definidos. Devuelve (amplitud, piso, normalizadas, resultado)."""
    crestas = crestas.astype(np.float64)
    amplitud = np.sqrt(cv2.GaussianBlur(crestas * crestas, (0, 0), 0.75 * periodo))
    piso = 0.25 * float(np.median(amplitud[dentro])) + 1e-9
    normalizadas = crestas / (np.sqrt(2.0) * np.maximum(amplitud, piso))
    return amplitud, piso, normalizadas, np.tanh(NITIDEZ * normalizadas) / np.tanh(NITIDEZ)


def paso_8_realce(E: dict) -> None:
    encabezado(8, "Realce: contraste parejo y bordes de cresta definidos")
    amplitud, piso, norm, real = realzar(E["crestas"], E["mascara"], E["periodo"])
    E["real"] = real
    hablar("Después de 15 pasadas la onda es suave (grises entre cresta y valle) y su amplitud varía: donde dos crestas se juntan, los "
           "filtros se cancelan en parte y quedan manchas grises. Se arregla en dos movimientos:")
    hablar(f"a) Se mide la amplitud local = raíz de la media de c² en un entorno gaussiano de 0,75 períodos ({0.75 * E['periodo']:.1f} px) y "
           f"se divide por √2·amplitud (con un piso del 25 % de la mediana, {piso:.3f}, para no amplificar zonas casi vacías). Una onda "
           f"senoidal de amplitud A tiene RMS A/√2, así que queda con amplitud ≈ 1.")
    hablar(f"b) Se aplica tanh(K·c)/tanh(K) con K = {NITIDEZ:g}: achata los picos y empina los bordes, así que cada cresta pasa de una onda "
           f"suave a una franja con bordes marcados. (Se agregó el 22/10, decisión 42, porque en la lámina se veían borrosas; se compararon "
           f"K = 1,5, 2 y 3 a tamaño real.)")
    y0 = 300
    x0 = ANCHO // 2
    fig, ejes = figura(2, 3, 3.6, 4.2)
    mostrar_gris(ejes[0], a_gris(E["crestas"]), "pasada 15 (sin realce)")
    ejes[1].imshow(amplitud, cmap="viridis"); ejes[1].set_title("amplitud local", fontsize=9.5)
    ejes[1].set_xticks([]); ejes[1].set_yticks([])
    mostrar_gris(ejes[2], a_gris(np.clip(norm, -1, 1)), "÷ (√2 · amplitud local)")
    mostrar_gris(ejes[3], a_gris(real), "tanh(2c)/tanh(2) = crestas realzadas")
    c = np.linspace(-1, 1, 201)
    ejes[4].plot(c, c, color="#bbbbbb", lw=1.5, label="sin realce")
    for K, est in ((1.5, "--"), (3.0, ":")):
        ejes[4].plot(c, np.tanh(K * c) / np.tanh(K), est, color="#666666", lw=1.4, label=f"K = {K:g}")
    ejes[4].plot(c, np.tanh(NITIDEZ * c) / np.tanh(NITIDEZ), color=AZUL, lw=2.4, label=f"K = {NITIDEZ:g} (usado)")
    ejes[4].legend(fontsize=8.5, loc="upper left"); ejes[4].grid(alpha=0.3)
    ejes[4].set_title("la curva tanh", fontsize=9.5); ejes[4].set_xlabel("c normalizada"); ejes[4].set_ylabel("salida")
    filas = np.arange(y0, y0 + 60)
    ejes[5].plot(filas, a_gris(E["crestas"])[y0:y0 + 60, x0], color="#d96a2b", lw=1.8, label="sin realce")
    ejes[5].plot(filas, a_gris(real)[y0:y0 + 60, x0], color=AZUL, lw=1.8, label="con realce")
    ejes[5].set_ylim(0, 255); ejes[5].grid(alpha=0.3); ejes[5].legend(fontsize=8.5)
    ejes[5].set_title(f"corte vertical (columna {x0}, filas {y0}–{y0 + 60})", fontsize=9.5)
    ejes[5].set_xlabel("fila"); ejes[5].set_ylabel("nivel de gris")
    for a in (ejes[0], ejes[2], ejes[3]):
        marcar(a, E["nucleos"], E["deltas"])
    cerrar(fig, 8, "realce", "Realce: de ondas suaves a crestas nítidas",
           "Abajo a la derecha se ve el efecto en una sola línea de píxeles: la onda suave pasa a mesetas con bordes empinados.")


# --- paso 9: composición ----------------------------------------------------------------------
def paso_9_base(E: dict) -> None:
    encabezado(9, "Máscara + fondo = huella base de la identidad")
    piel = GRIS_MEDIO - CONTRASTE * E["real"]
    imagen = np.clip(np.rint(E["suave"] * piel + (1.0 - E["suave"]) * FONDO), 0, 255).astype(np.uint8)
    E["base"] = imagen
    hablar(f"Cada valor c de las crestas se pasa a gris: piel ={GRIS_MEDIO:.0f} − {CONTRASTE:.0f}·c (cresta c = 1 → gris 28, valle c = −1 → gris 228). "
           f"Después se mezcla con el fondo (gris {FONDO}) según la máscara suave: imagen = suave·piel + (1 − suave)·{FONDO}.")
    hablar(f"Listo: la huella base de la identidad (uint8, {imagen.shape[0]} × {imagen.shape[1]}). Núcleos y deltas están en las coordenadas del paso 1. "
           "Esta base se guarda y de ella salen las 40 tomas de la identidad.")
    fig, ejes = figura(1, 3, 3.6, 4.5)
    mostrar_gris(ejes[0], a_gris(E["real"]), "crestas realzadas (toda la hoja)")
    mostrar_gris(ejes[1], E["suave"] * 255, "máscara suave")
    mostrar_gris(ejes[2], imagen, "HUELLA BASE")
    marcar(ejes[2], E["nucleos"], E["deltas"])
    cerrar(fig, 9, "base", "Huella base de la identidad",
           "Acá termina la etapa A. Todo salió de UNA semilla: tipo, posiciones, elipse, período y ruido. Con la misma semilla siempre sale esta imagen.")


# ============================================================================= ETAPA B
@dataclass(frozen=True)
class ParametrosVariante:
    rot_grados: float
    escala: float
    tx_px: float
    ty_px: float
    elastica_px: float
    presion: float
    corte_fraccion: float
    corte_angulo_grados: float
    ruido_sigma: float
    sal_pimienta: float


def sortear_parametros(rng: np.random.Generator) -> ParametrosVariante:
    v = VARIANTES
    rot = float(rng.uniform(-v["rotacion_max_grados"], v["rotacion_max_grados"]))
    factor = float(rng.uniform(1.0 - v["escala_max"], 1.0 + v["escala_max"]))
    tx = float(rng.uniform(-v["traslacion_max_px"], v["traslacion_max_px"]))
    ty = float(rng.uniform(-v["traslacion_max_px"], v["traslacion_max_px"]))
    elastica = float(rng.uniform(0.0, v["elastica_max_px"]))
    presion = float(rng.uniform(-v["presion_max"], v["presion_max"]))
    sigma = float(rng.uniform(0.0, v["ruido_sigma_max"]))
    sal = float(rng.uniform(0.0, v["sal_pimienta_max"]))
    if rng.random() < v["prob_corte"]:
        corte = float(rng.uniform(*v["corte_periferico"]))
        angulo = float(rng.uniform(0.0, 360.0))
    else:
        corte, angulo = 0.0, 0.0
    return ParametrosVariante(rot, factor, tx, ty, elastica, presion, corte, angulo, sigma, sal)


def paso_10_parametros(E: dict) -> None:
    encabezado(10, f"Parámetros de la toma {E['v']} de esta identidad")
    rng = np.random.default_rng([E["semilla"], E["v"]])
    E["rng_v"] = rng
    p = sortear_parametros(rng)
    E["params"] = p
    hablar(f"Cada toma tiene su propia semilla: np.random.default_rng([{E['semilla']}, {E['v']}]) (identidad, número de toma). Con eso se "
           f"sortean, uniformes dentro de los rangos de config.yaml, los parámetros de esta toma (la misma toma siempre sale igual):")
    v = VARIANTES
    filas = [("rotación (°)", p.rot_grados, -v["rotacion_max_grados"], v["rotacion_max_grados"]),
             ("escala", p.escala, 1 - v["escala_max"], 1 + v["escala_max"]),
             ("traslación x (px)", p.tx_px, -v["traslacion_max_px"], v["traslacion_max_px"]),
             ("traslación y (px)", p.ty_px, -v["traslacion_max_px"], v["traslacion_max_px"]),
             ("elástica máx. (px)", p.elastica_px, 0, v["elastica_max_px"]),
             ("presión (+ = más gruesas)", p.presion, -v["presion_max"], v["presion_max"]),
             ("corte (fracción del área)", p.corte_fraccion, 0, v["corte_periferico"][1]),
             ("ángulo del corte (°)", p.corte_angulo_grados, 0, 360),
             ("ruido gaussiano σ", p.ruido_sigma, 0, v["ruido_sigma_max"]),
             ("sal y pimienta (fracción)", p.sal_pimienta, 0, v["sal_pimienta_max"])]
    for nombre, valor, lo, hi in filas:
        print(f"   {nombre:28s} {valor:+9.4f}   (rango {lo:g} a {hi:g})")
    hablar("El corte aparece solo en ~30 % de las tomas (prob_corte = 0,3); si no hay, fracción y ángulo valen 0.")
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for i, (nombre, valor, lo, hi) in enumerate(filas):
        y = len(filas) - i
        ax.plot([0, 1], [y, y], color="#cccccc", lw=4, solid_capstyle="round")
        ax.plot([(valor - lo) / (hi - lo)], [y], "o", color=AZUL, ms=10)
        ax.text(-0.17, y, nombre, ha="right", va="center", fontsize=10)
        ax.text(-0.03, y, f"{lo:g}", ha="right", va="center", fontsize=8.5, color="#777777")
        ax.text(1.03, y, f"{hi:g}", ha="left", va="center", fontsize=8.5, color="#777777")
        ax.text(1.17, y, f"{valor:+.3f}" if nombre != "sal y pimienta (fracción)" else f"{valor:.4f}", ha="left", va="center",
                fontsize=10.5, family="monospace", fontweight="bold")
    ax.set_xlim(-0.62, 1.4); ax.set_ylim(0.4, len(filas) + 0.6); ax.axis("off")
    cerrar(fig, 10, "parametros", f"Parámetros sorteados para la toma {E['v']}",
           "Cada punto azul es el valor sorteado dentro de su rango. Una toma = la base + estas diez perillas.")


# --- paso 11: presión -----------------------------------------------------------------------
def paso_11_presion(E: dict) -> None:
    encabezado(11, "Presión: engrosar o afinar las crestas")
    base, p = E["base"], E["params"]
    imagen = base.astype(np.float32)
    presion = float(p.presion)
    nucleo = np.ones((3, 3), np.uint8)
    if presion != 0.0:
        morf = cv2.erode(base, nucleo) if presion > 0 else cv2.dilate(base, nucleo)
        imagen = (1.0 - abs(presion)) * imagen + abs(presion) * morf.astype(np.float32)
    else:
        morf = base
    E["imagen"] = imagen
    nombre = "erosión (mínimo del vecindario 3×3)" if presion > 0 else "dilatación (máximo del vecindario 3×3)"
    antes = float((base[E["mascara"]] < 128).mean())
    despues = float((imagen[E["mascara"]] < 128).mean())
    hablar(f"Presión = {presion:+.3f}. {'Positiva' if presion > 0 else 'Negativa'}: se usa {nombre} y se mezcla con la base: "
           f"I' = (1 − |p|)·I + |p|·morf(I).")
    hablar("Las crestas son OSCURAS sobre fondo claro. La erosión en escala de grises toma el MÍNIMO del vecindario: el negro avanza y la "
           "cresta ENGORDA (dedo apretado). La dilatación toma el MÁXIMO: gana el claro y la cresta ADELGAZA (dedo seco).")
    hablar(f"Píxeles oscuros (< 128) dentro de la máscara: {100 * antes:.1f} % antes → {100 * despues:.1f} % después.")
    zy = int(E["nucleos"][0][1]) if E["nucleos"] else ALTO // 2
    zx = int(E["nucleos"][0][0]) if E["nucleos"] else ANCHO // 2
    y0, x0 = int(np.clip(zy - 20, 0, ALTO - 40)), int(np.clip(zx - 20, 0, ANCHO - 40))
    fig, ejes = figura(2, 3, 3.4, 3.9)
    mostrar_gris(ejes[0], base, "huella base")
    mostrar_gris(ejes[1], morf, nombre.split(" (")[0] + " completa")
    mostrar_gris(ejes[2], np.clip(np.rint(imagen), 0, 255).astype(np.uint8), f"mezcla con |p| = {abs(presion):.2f}")
    for a, img, t in zip(ejes[3:], (base, morf, imagen), ("zoom base (40 × 40 px)", "zoom morfológica", "zoom mezcla")):
        mostrar_gris(a, np.clip(np.rint(img[y0:y0 + 40, x0:x0 + 40]), 0, 255).astype(np.uint8), t)
    cerrar(fig, 11, "presion", "Presión: erosión o dilatación",
           f"Píxeles oscuros: {100 * antes:.0f} % → {100 * despues:.0f} %. Presión +: erosión, las crestas oscuras engordan. "
           "Presión −: dilatación, las crestas oscuras adelgazan.")


# --- paso 12: geometría -----------------------------------------------------------------------
def desplazamiento_elastico(amplitud: float, rng: np.random.Generator):
    """Vectores al azar en una grilla de 5×4, interpolados (bicúbica) y escalados: el mayor mide `amplitud`."""
    filas, columnas = GRILLA_ELASTICA
    control = rng.standard_normal((2, filas, columnas)).astype(np.float32)
    dx = cv2.resize(control[0], (ANCHO, ALTO), interpolation=cv2.INTER_CUBIC).astype(np.float64)
    dy = cv2.resize(control[1], (ANCHO, ALTO), interpolation=cv2.INTER_CUBIC).astype(np.float64)
    norma = float(np.sqrt(dx ** 2 + dy ** 2).max())
    factor = amplitud / norma if norma > 0 else 0.0
    return dx * factor, dy * factor


def mapas(p: ParametrosVariante, rng: np.random.Generator):
    """Mapeo INVERSO para cv2.remap: para cada píxel de la salida, de dónde sacar el valor de la entrada."""
    cx, cy = (ANCHO - 1) / 2, (ALTO - 1) / 2
    alfa = math.radians(p.rot_grados)
    coseno, seno = math.cos(alfa), math.sin(alfa)
    u = np.arange(ANCHO, dtype=np.float64)[None, :] - cx - p.tx_px
    w = np.arange(ALTO, dtype=np.float64)[:, None] - cy - p.ty_px
    mapa_x = cx + (coseno * u - seno * w) / p.escala
    mapa_y = cy + (seno * u + coseno * w) / p.escala
    if p.elastica_px > 0:
        dx, dy = desplazamiento_elastico(float(p.elastica_px), rng)
        mapa_x = mapa_x + dx
        mapa_y = mapa_y + dy
    return mapa_x.astype(np.float32), mapa_y.astype(np.float32)


def _rejilla() -> np.ndarray:
    r = np.full((ALTO, ANCHO), 255, np.uint8)
    for k in range(0, ALTO, 20):
        r[k:k + 2, :] = 40
    for k in range(0, ANCHO, 20):
        r[:, k:k + 2] = 40
    return r


def paso_12_geometria(E: dict) -> None:
    encabezado(12, "Geometría: rotación, escala, traslación y deformación elástica, en UN solo remap")
    p = E["params"]
    mx_a, my_a = mapas(replace(p, elastica_px=0.0), None)          # solo la afín (no gasta sorteos)
    mx, my = mapas(p, E["rng_v"])                                  # afín + elástica (esta sí usa el rng)
    E["mapa_x"], E["mapa_y"] = mx, my
    img = cv2.remap(E["imagen"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(FONDO))
    mascara = cv2.remap(E["mascara"].astype(np.float32), mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0) > 0.5
    E["imagen"], E["mascara_v"] = img, mascara
    hablar(f"Afín: p' = c + s·R(α)·(p − c) + t, con α = {p.rot_grados:+.2f}°, s = {p.escala:.3f}, t = ({p.tx_px:+.1f}, {p.ty_px:+.1f}) px y c el "
           f"centro de la imagen. cv2.remap trabaja al revés (para cada píxel de la salida, de dónde tomar el de la entrada), así que el código "
           f"usa la inversa de esa fórmula.")
    hablar(f"Elástica: se sortean {GRILLA_ELASTICA[0]}×{GRILLA_ELASTICA[1]} vectores al azar (normales), se interpolan suave a toda la imagen y se "
           f"escalan para que el mayor mida exactamente {p.elastica_px:.2f} px. Se SUMA a los mapas de la afín: un solo remap, una sola interpolación "
           f"(así la imagen no se ensucia con dos remuestreos).")
    hablar("Lo que queda fuera de la imagen original se completa con fondo (gris 245).")
    fig, ejes = figura(1, 4, 3.3, 4.4)
    ejes[0].imshow(E["base"], cmap="gray", vmin=0, vmax=255, alpha=0.45)
    sx, sy = np.meshgrid(np.arange(10, ANCHO, 20), np.arange(10, ALTO, 20))
    ddx, ddy = (mx - mx_a)[sy, sx], (my - my_a)[sy, sx]
    ejes[0].quiver(sx, sy, ddx, ddy, angles="xy", scale_units="xy", scale=1 / 8, color="#14181f", width=0.004)
    ejes[0].set_title("campo elástico que se suma al mapa\n(flechas ×8)", fontsize=9.5)
    ejes[0].set_xticks([]); ejes[0].set_yticks([])
    ra = cv2.remap(_rejilla(), mx_a, my_a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    # la elástica real mide ≤ 3 px y en una rejilla de 20 px no se vería: acá se la exagera ×8 SOLO para dibujar
    rb = cv2.remap(_rejilla(), mx_a + 8 * (mx - mx_a), my_a + 8 * (my - my_a), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    mostrar_gris(ejes[1], ra, "rejilla con la afín sola\n(rota, achica y corre todo igual)")
    mostrar_gris(ejes[2], rb, "afín + elástica exagerada ×8\n(solo para verla; la real es ≤ 3 px)")
    mostrar_gris(ejes[3], np.clip(np.rint(img), 0, 255).astype(np.uint8), "la huella tras el remap")
    cerrar(fig, 12, "geometria", "Geometría: afín + elástica",
           "La rejilla muestra cómo se mueve la 'hoja de goma'. La afín mueve todo igual (dedo corrido o girado); la elástica corre unas "
           "zonas más que otras (acá exagerada ×8 solo para dibujarla).")


# --- paso 13: posiciones reales ---------------------------------------------------------------
def ubicar(punto, mapa_x, mapa_y):
    """Dónde quedó un punto de la base: se pasa un punto gaussiano por el MISMO remap y se busca su máximo (subpíxel)."""
    x0, y0 = float(punto[0]), float(punto[1])
    y = np.arange(ALTO, dtype=np.float32)[:, None]
    x = np.arange(ANCHO, dtype=np.float32)[None, :]
    auxiliar = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * SIGMA_PUNTO_PX ** 2)).astype(np.float32)
    salida = cv2.remap(auxiliar, mapa_x, mapa_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    fila, columna = np.unravel_index(int(np.argmax(salida)), salida.shape)
    if salida[fila, columna] < 0.5:
        return None, auxiliar, salida
    f0, f1 = max(0, fila - 2), min(ALTO, fila + 3)
    c0, c1 = max(0, columna - 2), min(ANCHO, columna + 3)
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
    if not (0 <= xs <= ANCHO - 1 and 0 <= ys <= ALTO - 1):
        return None, auxiliar, salida
    return (xs, ys), auxiliar, salida


def _afin_directa(punto, p: ParametrosVariante):
    cx, cy = (ANCHO - 1) / 2, (ALTO - 1) / 2
    a = math.radians(p.rot_grados)
    dx, dy = punto[0] - cx, punto[1] - cy
    return (cx + p.escala * (math.cos(a) * dx + math.sin(a) * dy) + p.tx_px,
            cy + p.escala * (-math.sin(a) * dx + math.cos(a) * dy) + p.ty_px)


def paso_13_posiciones(E: dict) -> None:
    encabezado(13, "Dónde quedaron el núcleo y el delta en esta toma")
    p, mx, my = E["params"], E["mapa_x"], E["mapa_y"]
    nuc, dels = [], []
    ref_aux = None
    for q in E["nucleos"]:
        r, aux, sal = ubicar(q, mx, my)
        if r is not None:
            nuc.append(r)
        ref_aux = ref_aux or (q, aux, sal, r)
    for q in E["deltas"]:
        r, _, _ = ubicar(q, mx, my)
        if r is not None:
            dels.append(r)
    E["nucleos_v"], E["deltas_v"] = tuple(nuc), tuple(dels)
    hablar("El generador guarda en metadatos.csv dónde quedó cada singularidad en cada toma (es la verdad terreno contra la que se pueden medir las etapas siguientes). "
           "Para saberlo no alcanza con la fórmula afín, porque la elástica también mueve el núcleo.")
    hablar("Truco: se dibuja un punto gaussiano (σ = 2,5 px) sobre el núcleo de la base, se lo pasa por el MISMO remap que la huella y se "
           "busca su máximo con precisión de subpíxel (ajuste cuadrático del logaritmo). Así la verdad terreno acompaña a la imagen.")
    for i, (a, b) in enumerate(zip(E["nucleos"], nuc)):
        f = _afin_directa(a, p)
        print(f"   núcleo {i + 1}: base ({a[0]:.1f}, {a[1]:.1f}) → toma ({b[0]:.2f}, {b[1]:.2f});  solo con la fórmula afín caería en "
              f"({f[0]:.2f}, {f[1]:.2f}), a {math.dist(b, f):.2f} px")
    for i, (a, b) in enumerate(zip(E["deltas"], dels)):
        print(f"   delta  {i + 1}: base ({a[0]:.1f}, {a[1]:.1f}) → toma ({b[0]:.2f}, {b[1]:.2f})")
    fig, ejes = figura(1, 3, 3.7, 4.5)
    if ref_aux is not None and ref_aux[3] is not None:
        q, aux, sal, r = ref_aux
        f = _afin_directa(q, p)

        def recorte(imagen, centro, lado=60):
            x0 = int(np.clip(round(centro[0]) - lado // 2, 0, ANCHO - lado))
            y0 = int(np.clip(round(centro[1]) - lado // 2, 0, ALTO - lado))
            return imagen[y0:y0 + lado, x0:x0 + lado], x0, y0

        cb, x0, y0 = recorte(E["base"], q)
        ejes[0].imshow(cb, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ejes[0].imshow(np.ma.masked_less(aux[y0:y0 + 60, x0:x0 + 60], 0.05), cmap="autumn", alpha=0.75, vmin=0, vmax=1, interpolation="nearest")
        ejes[0].set_title("base: punto gaussiano (σ = 2,5 px)\nsobre el núcleo · zoom 60 × 60", fontsize=9.5)
        cs, x1, y1 = recorte(sal, r)
        ejes[1].imshow(cs, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
        ejes[1].plot([r[0] - x1], [r[1] - y1], "c+", ms=16, mew=2, label="máximo (subpíxel)")
        ejes[1].plot([f[0] - x1], [f[1] - y1], "wx", ms=10, mew=2, label="solo fórmula afín")
        ejes[1].legend(fontsize=8, loc="lower right", framealpha=0.85)
        ejes[1].set_title("el mismo punto tras el MISMO remap\n(la elástica lo corrió un poco más)", fontsize=9.5)
    else:
        for a in ejes[:2]:
            a.axis("off")
        ejes[0].text(0.5, 0.5, "esta identidad no tiene\nnúcleos (arco)", ha="center", va="center", transform=ejes[0].transAxes)
    ejes[2].imshow(np.clip(np.rint(E["imagen"]), 0, 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
    marcar(ejes[2], nuc, dels)
    ejes[2].set_title("posiciones reales en la toma", fontsize=9.5)
    for a in ejes:
        a.set_xticks([]); a.set_yticks([])
    cerrar(fig, 13, "posiciones", "Dónde quedaron el núcleo y el delta",
           "Esas coordenadas van a metadatos.csv junto con los parámetros de la toma. Son la verdad terreno: sirven para medir contra algo "
           "conocido a las etapas que vienen después.")


# --- paso 14: corte ---------------------------------------------------------------------------
def cortar(mascara, nucleos, fraccion: float, angulo_pedido: float):
    """Semiplano que se lleva `fraccion` del área de la máscara; gira de a 30° si se acerca a un núcleo."""
    area = int(mascara.sum())
    a_quitar = int(round(fraccion * area))
    if area == 0 or a_quitar <= 0:
        return None
    a_quitar = min(a_quitar, area)
    filas = np.arange(ALTO, dtype=np.float64)[:, None]
    columnas = np.arange(ANCHO, dtype=np.float64)[None, :]
    for k in range(int(round(360 / PASO_CORTE_GRADOS))):
        angulo = (float(angulo_pedido) + PASO_CORTE_GRADOS * k) % 360.0
        a = math.radians(angulo)
        coseno, seno = math.cos(a), math.sin(a)
        proyeccion = columnas * coseno + filas * seno
        umbral = float(np.partition(proyeccion[mascara], area - a_quitar)[area - a_quitar])
        if all(umbral - (x * coseno + y * seno) >= MARGEN_CORTE_PX for x, y in nucleos):
            return mascara & (proyeccion >= umbral), angulo, proyeccion, umbral, k
    return None


def paso_14_corte(E: dict) -> None:
    encabezado(14, "Corte periférico (a veces el dedo no apoya entero)")
    p = E["params"]
    imagen, mascara = E["imagen"], E["mascara_v"]
    E["params_usados"] = replace(p, corte_fraccion=0.0, corte_angulo_grados=0.0)
    if p.corte_fraccion <= 0:
        hablar("Esta toma salió SIN corte (solo el 30 % de las tomas lo tienen). Probá con otra: --variante N (por ejemplo la 11 en la identidad 1).")
        return
    corte = cortar(mascara, E["nucleos_v"], float(p.corte_fraccion), float(p.corte_angulo_grados))
    if corte is None:
        hablar("Ningún ángulo permite cortar sin acercarse al núcleo: no se corta.")
        return
    quitados, angulo, proyeccion, umbral, giros = corte
    area = int(mascara.sum())
    nueva = mascara & ~quitados
    fondo = np.clip((proyeccion - umbral) / 3.0 + 0.5, 0.0, 1.0).astype(np.float32)        # borde de 3 px
    despues = (1.0 - fondo) * imagen + fondo * float(FONDO)
    E["imagen"], E["mascara_v"] = despues, nueva
    E["params_usados"] = replace(p, corte_angulo_grados=angulo)
    a = math.radians(angulo)
    dist = min(umbral - (x * math.cos(a) + y * math.sin(a)) for x, y in E["nucleos_v"]) if E["nucleos_v"] else float("nan")
    hablar(f"Se pidió quitar el {100 * p.corte_fraccion:.2f} % del área de la huella ({area} px) hacia los {p.corte_angulo_grados:.1f}°. "
           f"Se quitó el {100 * quitados.sum() / area:.2f} % "
           f"({'ángulo pedido' if giros == 0 else f'giró {giros} × 30° → {angulo:.0f}°'}). El corte es un semiplano: se proyecta cada píxel "
           f"sobre la dirección (cos a, sen a) y se quitan los de mayor proyección hasta llegar al área pedida (el umbral es un percentil).")
    hablar(f"Regla del contrato: el corte nunca se acerca a menos de {MARGEN_CORTE_PX:.0f} px de un núcleo. Acá el núcleo queda a {dist:.1f} px.")
    fig, ejes = figura(1, 3, 3.6, 4.5)
    mostrar_gris(ejes[0], mascara * 255, "máscara antes del corte")
    vista = np.dstack([mascara * 255.0] * 3).astype(np.uint8)
    vista[quitados] = (200, 38, 42)
    ejes[1].imshow(vista); ejes[1].set_title("zona que se quita (rojo)", fontsize=9.5)
    ejes[1].set_xticks([]); ejes[1].set_yticks([])
    ejes[1].annotate("", xy=(ANCHO / 2 + 90 * math.cos(a), ALTO / 2 + 90 * math.sin(a)), xytext=(ANCHO / 2, ALTO / 2),
                     arrowprops=dict(arrowstyle="->", color=AZUL, lw=2))
    mostrar_gris(ejes[2], np.clip(np.rint(despues), 0, 255).astype(np.uint8), "huella tras el corte")
    for ax in (ejes[0], ejes[1], ejes[2]):
        marcar(ax, E["nucleos_v"], E["deltas_v"])
    cerrar(fig, 14, "corte", "Corte periférico",
           "Simula apoyar el dedo mordiendo el borde del lector. Siempre se saca un pedazo del borde y se respeta el núcleo.")


# --- paso 15: ruido -----------------------------------------------------------------------------
def paso_15_ruido(E: dict) -> None:
    encabezado(15, "Ruido gaussiano y sal y pimienta (el polvo en el vidrio)")
    p, rng = E["params"], E["rng_v"]
    imagen = E["imagen"]
    sin_ruido = np.clip(np.rint(imagen), 0, 255).astype(np.uint8)
    if p.ruido_sigma > 0:
        imagen = imagen + rng.normal(0.0, float(p.ruido_sigma), imagen.shape)
    gauss = np.clip(np.rint(imagen), 0, 255).astype(np.uint8)
    final = gauss.copy()
    if p.sal_pimienta > 0:
        sorteo = rng.random(final.shape)
        mitad = float(p.sal_pimienta) / 2
        final[sorteo < mitad] = 0
        final[(sorteo >= mitad) & (sorteo < 2 * mitad)] = 255
    E["final"] = final
    pep = float(((final == 0) & (gauss != 0)).mean())
    sal = float(((final == 255) & (gauss != 255)).mean())
    hablar(f"Ruido gaussiano: a cada píxel se le suma un valor de N(0, σ²) con σ = {p.ruido_sigma:.2f} niveles de gris (de 0 a 255).")
    hablar(f"Sal y pimienta: una fracción {p.sal_pimienta:.4f} de los píxeles, elegidos al azar, se llevan a 0 (pimienta, la mitad) o a 255 (sal, la otra mitad). "
           f"Acá: {100 * pep:.2f} % de pimienta y {100 * sal:.2f} % de sal.")
    hablar("Esto es lo último: la imagen final es la toma. Se guarda como PNG de 8 bits.")
    zy = int(E["nucleos_v"][0][1]) if E["nucleos_v"] else ALTO // 2
    zx = int(E["nucleos_v"][0][0]) if E["nucleos_v"] else ANCHO // 2
    y0, x0 = int(np.clip(zy - 30, 0, ALTO - 60)), int(np.clip(zx - 30, 0, ANCHO - 60))
    fig, ejes = figura(2, 3, 3.5, 4.2)
    mostrar_gris(ejes[0], sin_ruido, "antes del ruido")
    mostrar_gris(ejes[1], gauss, f"+ gaussiano (σ = {p.ruido_sigma:.1f})")
    mostrar_gris(ejes[2], final, "+ sal y pimienta = TOMA FINAL")
    for a, img, t in zip(ejes[3:], (sin_ruido, gauss, final), ("zoom (60 × 60 px)", "zoom", "zoom")):
        mostrar_gris(a, img[y0:y0 + 60, x0:x0 + 60], t)
    marcar(ejes[2], E["nucleos_v"], E["deltas_v"])
    cerrar(fig, 15, "ruido_final", "Ruido y toma final",
           "Con esto termina la toma. Las 40 tomas de una identidad son este mismo proceso con 40 semillas [semilla, v] distintas.")


# ============================================================================= cierre
def resumen(E: dict) -> None:
    fig, ejes = plt.subplots(2, 5, figsize=(16, 9.6))
    ejes = ejes.ravel()
    hsv = plt.get_cmap("hsv")
    ejes[0].imshow(hsv(E["orient"] / np.pi)); ejes[0].set_title("A1-A2 · singularidades\ny campo θ(x, y)", fontsize=10)
    marcar(ejes[0], E["nucleos"], E["deltas"]); ejes[0].set_xticks([]); ejes[0].set_yticks([])
    mostrar_gris(ejes[1], a_gris(E["pasadas"][0]), "A5 · ruido (semilla)")
    mostrar_gris(ejes[2], a_gris(E["pasadas"][1]), "A6 · 1ª pasada de Gabor")
    mostrar_gris(ejes[3], a_gris(E["pasadas"][ITERACIONES]), f"A7 · pasada {ITERACIONES}")
    mostrar_gris(ejes[4], E["base"], "A8-A9 · realce + máscara\n= HUELLA BASE")
    marcar(ejes[4], E["nucleos"], E["deltas"])
    img = E["imagen_etapas"]
    for a, (clave, t) in zip(ejes[5:], (("presion", "B11 · presión"), ("geometria", "B12 · afín + elástica"), ("corte", "B14 · corte"),
                                        ("final", "B15 · ruido = TOMA"), (None, ""))):
        if clave is None:
            a.axis("off")
            continue
        mostrar_gris(a, img[clave], t)
    marcar(ejes[8], E["nucleos_v"], E["deltas_v"])
    p = E["params"]
    ejes[9].axis("off")
    ejes[9].text(0.0, 0.95, f"Identidad: {E['tipo']}, semilla {E['semilla']}\nToma {E['v']}: rot {p.rot_grados:+.1f}°, esc {p.escala:.3f},\n"
                 f"t ({p.tx_px:+.1f}, {p.ty_px:+.1f}) px, elástica {p.elastica_px:.2f} px,\npresión {p.presion:+.2f}, "
                 f"corte {100 * p.corte_fraccion:.1f} %,\nσ {p.ruido_sigma:.1f}, sal y pimienta {100 * p.sal_pimienta:.2f} %",
                 va="top", fontsize=10, family="monospace")
    fig.suptitle("H1 en una imagen: de una semilla a una toma", fontsize=14, fontweight="bold", x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ruta = SALIDA / "resumen.png"
    fig.savefig(ruta, dpi=110)
    print(f"   [figura guardada: {ruta}]")
    if MOSTRAR:
        plt.show()
    plt.close(fig)


def gif_pasadas(E: dict) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("   (sin Pillow no se arma el GIF)")
        return
    cuadros = [a_gris(c) for c in E["pasadas"]] + [a_gris(E["real"]), E["base"]]
    imgs = [Image.fromarray(c) for c in cuadros]
    duraciones = [500] + [250] * ITERACIONES + [600, 1800]
    ruta = SALIDA / "pasadas.gif"
    imgs[0].save(ruta, save_all=True, append_images=imgs[1:], duration=duraciones, loop=0)
    print(f"   [GIF guardado: {ruta}  (ruido → 15 pasadas → realce → huella base)]")


def verificar(E: dict, raiz: Path, tipo: str, semilla: int, v: int, identidad: int) -> bool:
    """Compara contra el generador del repo: tiene que dar exactamente los mismos píxeles."""
    encabezado(0, "Verificación contra src/huellas (generar_base y aplicar_variante)")
    sys.path.insert(0, str(raiz))
    from src.config import cargar_config
    from src.huellas import generador as G
    from src.huellas import variantes as V
    cfg = cargar_config()
    base = G.generar_base(tipo, semilla, cfg)
    ok_base = bool(np.array_equal(base.imagen, E["base"]))
    ok_mask = bool(np.array_equal(base.mascara, E["mascara"]))
    ok_theta = bool(np.allclose(base.orientacion, E["orient"]))
    print(f"   huella base:      {'IDÉNTICA' if ok_base else 'DISTINTA'} al generador del repo (máscara {'ok' if ok_mask else 'DISTINTA'}, θ {'ok' if ok_theta else 'DISTINTO'})")
    rng = np.random.default_rng([semilla, v])
    params = V.sortear_parametros(rng, cfg)
    var = V.aplicar_variante(base, params, rng)
    ok_var = bool(np.array_equal(var.imagen, E["final"]))
    ok_pos = bool(np.allclose(np.array(var.singularidades.nucleos + var.singularidades.deltas),
                              np.array(E["nucleos_v"] + E["deltas_v"])))
    print(f"   toma {v:>3}:         {'IDÉNTICA' if ok_var else 'DISTINTA'} (posiciones de núcleos y deltas {'ok' if ok_pos else 'DISTINTAS'})")
    ok_png = True
    if identidad:
        for ruta in sorted((raiz / "data" / "huellas").glob(f"*/id{identidad}_v{v:03d}.png")):
            png = cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_GRAYSCALE)
            ok_png = bool(np.array_equal(png, E["final"]))
            print(f"   {ruta.relative_to(raiz)}: {'IDÉNTICO' if ok_png else 'DISTINTO'} a la toma de este script")
    return ok_base and ok_mask and ok_theta and ok_var and ok_pos and ok_png


# ============================================================================= main
def main() -> int:
    global SALIDA, MOSTRAR, plt
    ap = argparse.ArgumentParser(description="H1 paso a paso: de una semilla a una toma, con una figura por paso.")
    ap.add_argument("--identidad", type=int, choices=sorted(IDENTIDADES), default=1, help="1 presilla, 2 verticilo, 3 arco en tienda")
    ap.add_argument("--tipo", choices=["presilla", "verticilo", "arco_tienda", "arco"], help="para experimentar: otro tipo")
    ap.add_argument("--semilla", type=int, help="para experimentar: otra semilla (sale otra 'persona')")
    ap.add_argument("--variante", type=int, default=11, help="qué toma armar (la 11 de la identidad 1 tiene corte); default 11")
    ap.add_argument("--mostrar", action="store_true", help="abre cada figura y espera que la cierres antes de seguir")
    ap.add_argument("--verificar", action="store_true", help="compara con src/huellas (generar_base y aplicar_variante)")
    ap.add_argument("--sin-gif", action="store_true", help="no arma pasadas.gif")
    ap.add_argument("--salida", type=Path, help="carpeta de salida (default data/h1_paso_a_paso/idN)")
    ap.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parents[1], help="raíz del repo (default: la de este script)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")        # θ, π, → en la consola de Windows
    except Exception:
        pass
    import matplotlib
    if not args.mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    plt = _plt
    MOSTRAR = args.mostrar

    tipo, semilla = IDENTIDADES[args.identidad]
    custom = args.tipo is not None or args.semilla is not None
    tipo = args.tipo or tipo
    semilla = args.semilla if args.semilla is not None else semilla
    nombre = f"id{args.identidad}" if not custom else f"{tipo}_s{semilla}"
    SALIDA = args.salida or (args.raiz / "data" / "h1_paso_a_paso" / nombre)
    SALIDA.mkdir(parents=True, exist_ok=True)
    print(f"H1 paso a paso · {tipo} · semilla {semilla} · toma {args.variante}\nFiguras en: {SALIDA}")

    E: dict = dict(tipo=tipo, semilla=semilla, v=args.variante, rng=np.random.default_rng(int(semilla)))
    paso_1_singularidades(E)
    paso_2_campo(E)
    paso_3_mascara(E)
    paso_4_banco(E)
    paso_5_ruido(E)
    paso_6_una_pasada(E)
    paso_7_pasadas(E)
    paso_8_realce(E)
    paso_9_base(E)
    guardar_png(SALIDA / "base.png", E["base"])

    # Etapa B: se guardan las imágenes intermedias para el resumen
    etapas: dict = {}
    paso_10_parametros(E)
    paso_11_presion(E)
    etapas["presion"] = np.clip(np.rint(E["imagen"]), 0, 255).astype(np.uint8)
    paso_12_geometria(E)
    etapas["geometria"] = np.clip(np.rint(E["imagen"]), 0, 255).astype(np.uint8)
    paso_13_posiciones(E)
    paso_14_corte(E)
    etapas["corte"] = np.clip(np.rint(E["imagen"]), 0, 255).astype(np.uint8)
    paso_15_ruido(E)
    etapas["final"] = E["final"]
    E["imagen_etapas"] = etapas
    guardar_png(SALIDA / "toma_final.png", E["final"])

    encabezado(16, "Resumen")
    resumen(E)
    if not args.sin_gif:
        gif_pasadas(E)
    hablar("\nListo. Mirá las figuras 01 a 15 en orden: cada una es un paso, y el título de abajo dice qué pasa. Probá con --identidad 2, "
           "--variante 5 o --semilla 999 para ver cómo cambia.")
    ok = True
    if args.verificar:
        ok = verificar(E, args.raiz, tipo, semilla, args.variante, 0 if custom else args.identidad)
        print("\n" + ("✔ Este script reproduce al generador del repo píxel por píxel." if ok else "✘ HAY DIFERENCIAS con el generador del repo."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
