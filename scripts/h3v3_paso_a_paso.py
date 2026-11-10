"""H3 + V3 paso a paso: de la huella y de la palabra a un vector de números, con una figura por paso.

No es parte del pipeline ni tiene tests: es una herramienta para entender (y explicar) el hito
"Extracción de características" de huellas (H3) y de audios (V3), y como anexo el criterio de
desempate del K-NN (V4). Usa las funciones de src/ para lo pesado y rehace a mano lo que conviene
mirar por dentro (las celdas de orientación, las zonas de Gabor, el crossing number, los MFCC
paso a paso, los tramos y el voto del K-NN). Con --verificar compara lo rehecho contra src/ y
contra los vectores guardados en data/huellas/caracteristicas.npz y data/voz/prepro/caracteristicas.npz.

Qué hace, en orden (el número es el de la figura que guarda):
  Parte A · huellas (H3), con train/id1_v011 (la toma 11 de la identidad 1, con corte)
    01 de la imagen a la ficha            06 minucias: limpieza
    02 orientación por celdas (2θ)        07 minucias: densidades
    03 filtros de Gabor                   08 escalador y pesos por bloque
    04 Gabor por zonas (FingerCode)       09 PCA y separación de las identidades
    05 minucias: crossing number
  Parte B · audios (V3), con p01_reconocer_01 (k0)
    10 tramas de 25 ms                    14 5 tramos: un vector de largo fijo
    11 MFCC de una trama, paso a paso     15 escalador y pesos por bloque
    12 MFCC en el tiempo y CMN            16 PCA y separación de los comandos
    13 Δ, energía y cruces por cero
  Anexo · criterio de desempate (V4)
    17 desempate en el voto del K-NN      18 desempate al elegir el modelo
  y GUION.md: por figura, una explicación con manzanitas, qué decir y el detalle con los
  números de esta corrida, más preguntas probables.

Uso, desde la raíz del repo:
    venv\\Scripts\\python scripts\\h3v3_paso_a_paso.py
    venv\\Scripts\\python scripts\\h3v3_paso_a_paso.py --mostrar       (abre cada figura y espera que la cierres)
    venv\\Scripts\\python scripts\\h3v3_paso_a_paso.py --verificar     (compara con src/ y los .npz)
    venv\\Scripts\\python scripts\\h3v3_paso_a_paso.py --huella train/id2_v001.png --toma p01_salir_02
Las figuras quedan en data/h3v3_paso_a_paso/ (data/ no va a Git).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

RAIZ: Path = Path(__file__).resolve().parents[1]
SALIDA: Path = Path(".")
MOSTRAR = False
plt = None
cfg: dict = {}
GUION: list[dict] = []       # un dict por paso: titulo, figura, pie, manzanita, decir, detalle

ROJO, AZUL, VERDE, NARANJA, GRIS, VIOLETA = "#c8262a", "#1b5fc4", "#14873f", "#e69f00", "#888888", "#7a5195"
ID_COLOR = {1: AZUL, 2: VERDE, 3: NARANJA}
CLASE_COLOR = {"reconocer": AZUL, "identificar": VERDE, "salir": ROJO, "otro": GRIS}
BLOQUE_COLOR = {"orientacion": AZUL, "gabor": NARANJA, "minucias": VERDE,
                "mfcc": AZUL, "delta": VIOLETA, "energia": ROJO, "zcr": VERDE, "duracion": NARANJA}


# ============================================================================= utilidades
def es(texto: str) -> str:
    """Coma decimal (1.33 → 1,33); no toca nombres de archivo (ahí al punto lo sigue una letra)."""
    return re.sub(r"(?<=\d)\.(?=\d)", ",", texto)


def encabezado(n: int, titulo: str) -> None:
    GUION.append(dict(n=n, titulo=es(titulo), figura="", pie="", manzanita="", decir="", detalle=[]))
    print("\n" + "=" * 100 + f"\nPASO {n} · {es(titulo)}\n" + "=" * 100)


def _imprimir(etiqueta: str, texto: str) -> None:
    for i, parrafo in enumerate(texto.split("\n")):
        print(textwrap.fill((etiqueta if i == 0 else "") + parrafo, 100, subsequent_indent="   "))


def hablar(texto: str) -> None:
    """Detalle técnico, con los números de esta corrida."""
    texto = es(texto)
    GUION[-1]["detalle"].append(texto)
    _imprimir("", texto)


def manzanita(texto: str) -> None:
    """La explicación simple, con una analogía."""
    GUION[-1]["manzanita"] = es(texto)
    _imprimir("[con manzanitas] ", es(texto))


def decir(texto: str) -> None:
    """Lo que se puede decir en voz alta mientras se muestra la figura (primera persona)."""
    GUION[-1]["decir"] = es(texto)
    _imprimir("[qué decir] ", es(texto))


def figura(filas: int, columnas: int, ancho: float, alto: float, **kw):
    fig, ejes = plt.subplots(filas, columnas, figsize=(ancho, alto), squeeze=False, **kw)
    return fig, ejes


def cerrar(fig, n: int, nombre: str, titulo: str, pie: str = "") -> None:
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if t.get_text():
            t.set_text(es(t.get_text()))
    fig.suptitle(es(f"Paso {n} · {titulo}"), fontsize=14, fontweight="bold", x=0.012, ha="left")
    alto = fig.get_figheight()
    inferior = 0.0
    if pie:
        envuelto = "\n".join(textwrap.wrap(es(pie), max(40, int(fig.get_figwidth() * 12.5))))
        fig.text(0.012, 0.15 / alto, envuelto, fontsize=9.5, ha="left", va="bottom", color="#333333")
        inferior = (0.3 + 0.17 * (envuelto.count("\n") + 1)) / alto
    superior = 1 - 0.55 / alto
    fig.tight_layout(rect=[0, inferior, 1, superior])
    fig.canvas.draw()
    fig.tight_layout(rect=[0, inferior, 1, superior], h_pad=1.4, w_pad=1.2)
    ruta = SALIDA / f"{n:02d}_{nombre}.png"
    fig.savefig(ruta, dpi=110)
    GUION[-1].update(figura=ruta.name, pie=es(pie))
    print(f"   [figura guardada: {ruta}]")
    if MOSTRAR:
        plt.show()
    plt.close(fig)


def sin_ejes(ax, titulo: str = "") -> None:
    ax.set_xticks([]); ax.set_yticks([])
    if titulo:
        ax.set_title(titulo, fontsize=9.5)


def contribucion(Zb: np.ndarray) -> float:
    """Media de ‖z_i − z_j‖² sobre los pares i ≠ j (= 2·Σ var con ddof=1)."""
    n = len(Zb)
    return float(2 * n / (n - 1) * np.sum(np.var(Zb, axis=0)))


def tira_vector(ax, x: np.ndarray, largos: dict, titulo: str) -> None:
    """El vector de características como barras, coloreado por bloque."""
    o = 0
    for b, d in largos.items():
        ax.bar(np.arange(o, o + d), x[o:o + d], width=1.0, color=BLOQUE_COLOR[b], label=f"{b} ({d})")
        o += d
    ax.set_xlim(0, o); ax.axhline(0, color="#999999", lw=0.6)
    ax.set_title(titulo, fontsize=10, loc="left"); ax.legend(fontsize=8.5, ncol=len(largos), loc="upper right")
    ax.set_xlabel("columna del vector")


# ============================================================================= PARTE A · HUELLAS
def cargar_huella(E: dict) -> None:
    import cv2
    from src.huellas.caracteristicas import caracteristicas, extraer_minucias
    from src.huellas.preprocesamiento import preprocesar
    ruta = RAIZ / cfg["rutas"]["huellas"] / E["huella"]
    E["img"] = cv2.imdecode(np.fromfile(str(ruta), np.uint8), cv2.IMREAD_UNCHANGED)
    E["h"] = preprocesar(E["img"], cfg)
    E["bloques_h"] = caracteristicas(E["h"], cfg)
    E["min"] = extraer_minucias(E["h"], cfg)
    z = np.load(RAIZ / cfg["rutas"]["huellas"] / "caracteristicas.npz", allow_pickle=False)
    E["npz_h"] = {k: z[k] for k in z.files}
    archivos = list(E["npz_h"]["archivo"])
    E["fila_h"] = archivos.index(E["huella"]) if E["huella"] in archivos else None
    c = cfg["huellas"]["caracteristicas"]
    E["ch"] = c
    yy, xx = np.mgrid[0:250, 0:250]
    E["polar"] = (np.hypot(xx - 124.5, yy - 124.5), np.mod(np.arctan2(yy - 124.5, xx - 124.5), 2 * np.pi))


def _bordes(g: int) -> np.ndarray:
    return np.round(np.linspace(0, 250, g + 1)).astype(int)


def _dist_borde(mascara: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return ndimage.distance_transform_edt(np.pad(mascara, 1))[1:-1, 1:-1]


def paso_1_ficha(E: dict) -> None:
    encabezado(1, "De la imagen a la ficha de números")
    h, img = E["h"], E["img"]
    i = E["fila_h"]
    X = E["npz_h"]["X"][i]
    largos = {"orientacion": 50, "gabor": 300, "minucias": int(X.size - 350)}
    manzanita("Para saber si dos huellas son del mismo dedo no comparo foto contra foto: con que una esté corrida dos "
              "píxeles, todos los píxeles cambian. Hago como con una manzana: en vez de mandar la foto, lleno una ficha "
              "con lo que la describe (color, tamaño, si tiene manchas). Acá la ficha tiene tres secciones: hacia dónde "
              "van las crestas, qué textura tienen y dónde hay minucias. Dos tomas del mismo dedo dan fichas parecidas "
              "aunque las fotos sean distintas.")
    decir(f"En H2 la huella ya quedó alineada: una ventana de 250 × 250 píxeles centrada en el núcleo y enderezada, con "
          f"la imagen realzada, la máscara, el esqueleto y el campo de orientación. H3 convierte eso en un vector de "
          f"{X.size} números, en tres bloques: orientación ({largos['orientacion']}), textura con filtros de Gabor "
          f"({largos['gabor']}) y minucias ({largos['minucias']}). Lo voy a mostrar con {E['huella']}, que es una toma con "
          f"un corte, para que se vea qué pasa donde falta huella.")
    hablar(f"Entrada: {E['huella']} ({img.shape[1]} × {img.shape[0]} px). H2 la lleva a 250 × 250 con el método "
           f"«{h.metodo}» (punto de referencia en ({h.punto_referencia[0]:.1f}, {h.punto_referencia[1]:.1f}) y ángulo "
           f"θ∞ = {math.degrees(h.angulo):.1f}°). Cada bloque crudo tiene NaN donde falta huella; el escalador los "
           f"reemplaza por la media de train y el vector final tiene {X.size} columnas (50 + 300 + 24: una celda de la "
           f"grilla de minucias no tiene datos en train y se descarta).")
    fig = plt.figure(figsize=(15.5, 7.8))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img if img.ndim == 2 else img[..., 0], cmap="gray", vmin=0, vmax=255)
    A, b = h.matriz[:, :2], h.matriz[:, 2]
    esquinas = np.array([[0, 0], [250, 0], [250, 250], [0, 250], [0, 0]], dtype=float)
    orig = (np.linalg.inv(A) @ (esquinas - b).T).T
    ax.plot(orig[:, 0], orig[:, 1], color=ROJO, lw=1.5)
    ax.plot([h.punto_referencia[0]], [h.punto_referencia[1]], "o", color=ROJO, ms=5)
    sin_ejes(ax, "entrada (H1) y la ventana de H2")
    paneles = [(h.realzada, "gray", "realzada"), (h.mascara, "gray", "máscara (dónde hay huella)"),
               (~h.esqueleto, "gray", "esqueleto (crestas de 1 px)")]
    for j, (im, cmap, tit) in enumerate(paneles, start=1):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(im, cmap=cmap); sin_ejes(ax, tit)
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(np.where(h.mascara, h.orientacion, np.nan), cmap="hsv", vmin=0, vmax=np.pi)
    sin_ejes(ax, "orientación θ (color = ángulo)")
    ax = fig.add_subplot(gs[1, :])
    tira_vector(ax, X, largos, f"la ficha: el vector de {X.size} números de {E['huella']} (ya escalado)")
    cerrar(fig, 1, "ficha", "De la imagen a la ficha de números",
           f"H2 deja la huella alineada en 250 × 250; H3 la resume en {X.size} números: orientación, Gabor y minucias. "
           "Dos tomas del mismo dedo dan vectores parecidos aunque las imágenes no coincidan píxel a píxel.")


def paso_2_orientacion(E: dict) -> None:
    encabezado(2, "Bloque 1 · orientación de las crestas por celdas")
    h, c = E["h"], E["ch"]
    g = int(c["grilla_orientacion"])
    b = _bordes(g)
    th = np.asarray(h.orientacion, np.float64)
    cos2, sen2 = np.cos(2 * th), np.sin(2 * th)
    valores, medias, coher, vacias = [], [], [], []
    for i in range(g):
        for j in range(g):
            m = h.mascara[b[i]:b[i + 1], b[j]:b[j + 1]]
            if np.count_nonzero(m) < c["cobertura_min"] * m.size or not m.any():
                valores += [np.nan, np.nan]; medias.append(np.nan); coher.append(np.nan); vacias.append((i, j))
                continue
            cm = cos2[b[i]:b[i + 1], b[j]:b[j + 1]][m].mean()
            sm = sen2[b[i]:b[i + 1], b[j]:b[j + 1]][m].mean()
            valores += [cm, sm]
            medias.append(0.5 * math.atan2(sm, cm) % math.pi); coher.append(math.hypot(cm, sm))
    E["orient_propio"] = np.array(valores)
    ingenuo = (5 + 175) / 2
    doble = round(0.5 * math.degrees(math.atan2(math.sin(math.radians(10)) + math.sin(math.radians(350)),
                                                math.cos(math.radians(10)) + math.cos(math.radians(350)))), 6) % 180
    manzanita("Es como mirar un campo arado desde un avión: en cada parcela anoto hacia dónde van los surcos. Pero una "
              "raya no tiene punta: 0° y 180° son la misma raya. Si promedio 5° y 175° (dos rayas casi horizontales) "
              f"me da {ingenuo:.0f}°, ¡una raya vertical! Por eso uso el ángulo doble: 2·5° = 10° y 2·175° = 350°, que en "
              f"la circunferencia quedan pegados, y el promedio da {doble:.0f}°, horizontal, como tiene que ser.")
    decir(f"El primer bloque es la orientación. Divido la ventana en una grilla de {g} × {g} y en cada celda promedio "
          f"cos 2θ y sen 2θ de los píxeles con huella. Uso el ángulo doble porque una cresta a 5° y otra a 175° son casi "
          f"la misma raya, y con el ángulo común el promedio daría 90°. Son {g * g} celdas con 2 números cada una: "
          f"{2 * g * g} valores. Si una celda tiene menos de la mitad de huella, como las {len(vacias)} de las esquinas que "
          f"marqué, o las que tapa un corte, queda vacía y después se completa con el promedio de train.")
    hablar(f"Grilla de {g} × {g} con bordes {[int(v) for v in b]}. Por celda: [media de cos 2θ, media de sen 2θ] dentro de la "
           f"máscara; la dirección media es θ̄ = ½·atan2(s̄, c̄) y la coherencia √(c̄² + s̄²) (1 = todas las crestas "
           f"paralelas, 0 = desordenadas). En esta huella: {len(vacias)} celda(s) sin huella suficiente "
           f"(cobertura < {c['cobertura_min']:.0%}) → NaN; coherencia media {np.nanmean(coher):.2f}.")
    fig, ejes = figura(1, 4, 16, 5.4, gridspec_kw=dict(width_ratios=[1, 1, 0.62, 0.62]))
    ax = ejes[0, 0]
    ax.imshow(h.realzada, cmap="gray")
    paso = 10
    for y in range(5, 250, paso):
        for x in range(5, 250, paso):
            if h.mascara[y, x]:
                t = th[y, x]
                ax.plot([x - 4 * math.cos(t), x + 4 * math.cos(t)], [y - 4 * math.sin(t), y + 4 * math.sin(t)],
                        color=ROJO, lw=1)
    sin_ejes(ax, "θ cada 10 px sobre la imagen realzada")
    ax = ejes[0, 1]
    ax.imshow(h.realzada, cmap="gray", alpha=0.35)
    for k, (i, j) in enumerate((i, j) for i in range(g) for j in range(g)):
        cx, cy = (b[j] + b[j + 1]) / 2, (b[i] + b[i + 1]) / 2
        if (i, j) in vacias:
            ax.add_patch(plt.Rectangle((b[j], b[i]), b[j + 1] - b[j], b[i + 1] - b[i], fill=False, hatch="xx",
                                       ec=ROJO, lw=0))
            ax.text(cx, cy, "NaN", ha="center", va="center", color=ROJO, fontsize=9, fontweight="bold")
            continue
        largo = 20 * coher[k]
        t = medias[k]
        ax.plot([cx - largo * math.cos(t), cx + largo * math.cos(t)], [cy - largo * math.sin(t), cy + largo * math.sin(t)],
                color=AZUL, lw=3, solid_capstyle="round")
    for v in b:
        ax.axhline(v, color="#999999", lw=0.6); ax.axvline(v, color="#999999", lw=0.6)
    ax.set_xlim(0, 250); ax.set_ylim(250, 0)
    sin_ejes(ax, f"dirección media por celda (largo = coherencia); {len(vacias)} sin huella")
    circ = np.linspace(0, 2 * np.pi, 200)
    for ax, flechas, media, col_m, tit in [
            (ejes[0, 2], [(5, "5°"), (175, "175°")], ingenuo, ROJO, f"ángulo común: promedio {ingenuo:.0f}° ✘"),
            (ejes[0, 3], [(10, "2·5° = 10°"), (350, "2·175° = 350°")], 2 * doble, VERDE,
             f"ángulo doble: promedio {doble:.0f}° ✔")]:
        ax.set_aspect("equal"); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axis("off")
        ax.plot(np.cos(circ), np.sin(circ), color="#cccccc")
        for ang, txt in flechas:
            ax.annotate("", (math.cos(math.radians(ang)), math.sin(math.radians(ang))), (0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=AZUL, lw=1.6))
            ax.text(1.25 * math.cos(math.radians(ang)), 1.25 * math.sin(math.radians(ang)) + 0.12 * math.copysign(1, math.sin(math.radians(ang)) or 1),
                    txt, color=AZUL, fontsize=8.5, ha="center", va="center")
        ax.annotate("", (0.8 * math.cos(math.radians(media)), 0.8 * math.sin(math.radians(media))), (0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col_m, lw=2, ls="--"))
        ax.set_title(tit, fontsize=9.5, color=col_m)
    cerrar(fig, 2, "orientacion", "Bloque 1 · orientación de las crestas por celdas",
           f"Grilla de {g} × {g}: en cada celda, la media de cos 2θ y sen 2θ ({2 * g * g} valores). El ángulo doble hace que "
           "5° y 175° (casi la misma raya) se promedien bien. Las celdas con menos de la mitad de huella quedan NaN.")


def _imagen_gabor(h) -> np.ndarray:
    f = 255.0 - np.asarray(h.realzada).astype(np.float64)
    mu = f[h.mascara].mean()
    return np.where(h.mascara, f, mu) - mu


def paso_3_gabor_filtros(E: dict) -> None:
    import cv2
    from src.huellas.caracteristicas import _filtros_de
    encabezado(3, "Bloque 2 · filtros de Gabor: plantillas de rayas")
    c = E["ch"]
    filtros = _filtros_de(cfg)
    periodos, K = [float(p) for p in c["gabor_periodos_px"]], int(c["gabor_orientaciones"])
    f = _imagen_gabor(E["h"])
    E["filtros"], E["f_gabor"] = filtros, f
    respuestas = [np.abs(cv2.filter2D(f, -1, g, borderType=cv2.BORDER_REFLECT)) for g in filtros]
    E["respuestas"] = respuestas
    manzanita("Un filtro de Gabor es una plantilla de rayas: rayas de cierto ancho y en cierta dirección. La apoyo en cada "
              "lugar de la huella y me fijo cuánto se parecen. Es como pasar un peine: entra suave donde los dientes van "
              "en la dirección del pelo y se traba donde van cruzados. Donde las crestas coinciden con la plantilla, la "
              "respuesta brilla; donde van cruzadas, queda oscura.")
    decir(f"El segundo bloque mide la textura con {len(filtros)} filtros de Gabor: {len(periodos)} anchos de raya "
          f"({' y '.join(f'{p:.0f}' for p in periodos)} px, cerca de la separación real entre crestas) por {K} direcciones "
          f"(cada {180 // K}°). Cada filtro es una plantilla de rayas. Acá abajo se ve que el filtro de rayas horizontales "
          f"brilla donde las crestas van horizontales y el vertical brilla en otro lado: cada filtro marca dónde está su "
          f"dirección.")
    hablar(f"Filtros: para cada período P en {periodos} y k = 0 … {K - 1}: σ = P/2, tamaño L = 2·round(3σ) + 1 "
           f"({', '.join(str(2 * round(3 * p / 2) + 1) for p in periodos)} px), orientación de cresta π·k/{K}; a cada uno se "
           f"le resta su media (no responde a zonas parejas) y se divide por Σ|g|. Imagen: 255 − realzada (cresta = "
           f"positivo) centrada en la media de la huella. Respuesta = |f ∗ g|.")
    fig = plt.figure(figsize=(15.5, 7.6))
    gs = fig.add_gridspec(2, 2 * K, height_ratios=[0.75, 1.4])
    for n, g in enumerate(filtros):
        ax = fig.add_subplot(gs[0, n])
        lim = np.abs(g).max()
        ax.imshow(g, cmap="RdBu_r", vmin=-lim, vmax=lim)
        sin_ejes(ax, f"P {periodos[n // K]:.0f} · {180 * (n % K) / K:.0f}°")
        ax.title.set_fontsize(8)
    elegidos = [0, K // 2, 1]
    ancho = 2 * K // (len(elegidos) + 1)
    ax = fig.add_subplot(gs[1, 0:ancho])
    ax.imshow(E["h"].realzada, cmap="gray"); sin_ejes(ax, "realzada")
    vmax = max(respuestas[n].max() for n in elegidos)
    for m, n in enumerate(elegidos, start=1):
        ax = fig.add_subplot(gs[1, m * ancho:(m + 1) * ancho])
        ax.imshow(np.where(E["h"].mascara, respuestas[n], 0), cmap="magma", vmin=0, vmax=vmax)
        sin_ejes(ax, f"|respuesta| del filtro de {180 * (n % K) / K:.0f}° (P = {periodos[n // K]:.0f})")
    cerrar(fig, 3, "gabor_filtros", "Bloque 2 · filtros de Gabor: plantillas de rayas",
           f"Arriba, los {len(filtros)} filtros ({len(periodos)} anchos × {K} direcciones). Abajo, cuánto «brilla» cada uno: "
           "cada filtro se enciende donde las crestas van en su dirección.")


def paso_4_gabor_zonas(E: dict) -> None:
    encabezado(4, "Bloque 2 · Gabor por zonas (anillos × sectores)")
    h, c = E["h"], E["ch"]
    r, fi = E["polar"]
    radios = [float(v) for v in c["anillos_gabor_px"]]
    S = int(c["sectores_gabor"])
    G = _dist_borde(h.mascara) >= c["erosion_gabor_px"]
    sector = np.minimum(np.floor(fi * S / (2 * np.pi)).astype(int), S - 1)
    zonas = [(r >= radios[a]) & (r < radios[a + 1]) & (sector == s) for a in range(len(radios) - 1) for s in range(S)]
    R = len(radios) - 1
    F = len(E["filtros"])
    valores = np.full(F * len(zonas), np.nan)
    medias = np.empty(F)
    for n, resp in enumerate(E["respuestas"]):
        for z, zona in enumerate(zonas):
            dentro = zona & G
            if np.count_nonzero(dentro) >= c["cobertura_min"] * np.count_nonzero(zona) and dentro.any():
                valores[n * len(zonas) + z] = resp[dentro].mean()
        medias[n] = resp[G].mean()
    E["gabor_propio"] = np.concatenate([valores, medias / medias.sum()])
    n_nan = int(np.isnan(valores[:len(zonas)]).sum())
    manzanita(f"Saber cuánto brilla cada filtro en toda la huella no alcanza: importa DÓNDE brilla. Corto la huella como "
              f"una pizza: {R} anillos y {S} porciones, {R * S} pedazos. En cada pedazo anoto cuánto brilló cada uno de los "
              f"{F} filtros. Es la idea del FingerCode (Jain y otros, 2000): la huella queda descripta como «en este "
              f"pedazo, rayas así».")
    decir(f"Para no perder dónde está cada textura, divido la ventana como una pizza alrededor del núcleo: {R} anillos "
          f"(de {radios[0]:.0f} a {radios[-1]:.0f} px) por {S} sectores, {R * S} zonas. En cada zona guardo la respuesta "
          f"media de cada filtro: {F} × {R * S} = {F * R * S} valores, más {F} con la energía total de cada filtro, "
          f"que suman 1. En total, {F * R * S + F}. Solo miro a más de {c['erosion_gabor_px']:.0f} px del borde de la "
          f"huella, porque en el borde el filtro ve el corte y no las crestas.")
    hablar(f"Zona útil G: distancia al borde de la máscara ≥ {c['erosion_gabor_px']:.0f} px. Una zona con menos de la "
           f"mitad dentro de G vale NaN: acá {n_nan} de {len(zonas)} zonas. Las {F} energías globales: "
           + ", ".join(f"{v:.3f}" for v in medias / medias.sum()) + ".")
    fig, ejes = figura(1, 4, 16, 4.8, gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    ax = ejes[0, 0]
    ax.imshow(h.realzada, cmap="gray")
    ax.imshow(np.where(G, np.nan, 1.0), cmap="Reds", alpha=0.35, vmin=0, vmax=1.5)
    t = np.linspace(0, 2 * np.pi, 300)
    for rad in radios:
        ax.plot(124.5 + rad * np.cos(t), 124.5 + rad * np.sin(t), color=AZUL, lw=1)
    for s in range(S):
        a = 2 * np.pi * s / S
        ax.plot([124.5 + radios[0] * math.cos(a), 124.5 + radios[-1] * math.cos(a)],
                [124.5 + radios[0] * math.sin(a), 124.5 + radios[-1] * math.sin(a)], color=AZUL, lw=1)
    ax.set_xlim(0, 250); ax.set_ylim(250, 0)
    sin_ejes(ax, f"{R} anillos × {S} sectores (rojo: fuera de G)")
    K = int(c["gabor_orientaciones"])
    for ax, n in [(ejes[0, 1], 0), (ejes[0, 2], K // 2)]:
        mapa = np.full((250, 250), np.nan)
        for z, zona in enumerate(zonas):
            v = valores[n * len(zonas) + z]
            if not np.isnan(v):
                mapa[zona] = v
        ax.imshow(mapa, cmap="magma", vmin=0, vmax=np.nanmax(valores))
        ax.set_xlim(0, 250); ax.set_ylim(250, 0)
        sin_ejes(ax, f"media por zona, filtro de {180 * (n % K) / K:.0f}°")
    ax = ejes[0, 3]
    etiquetas = [f"{180 * (n % K) / K:.0f}°\nP{float(c['gabor_periodos_px'][n // K]):.0f}" for n in range(F)]
    ax.bar(range(F), medias / medias.sum(), color=NARANJA)
    ax.set_xticks(range(F), etiquetas, fontsize=7)
    ax.set_title(f"energía global de cada filtro (suman 1)", fontsize=9.5)
    cerrar(fig, 4, "gabor_zonas", "Bloque 2 · Gabor por zonas (anillos × sectores)",
           f"{F} filtros × {R * S} zonas + {F} energías globales = {F * R * S + F} valores. Cada filtro tiene su propio mapa: "
           "dice en qué parte de la huella las crestas van en su dirección.")


def _crossing_number_vecinos(v: np.ndarray) -> int:
    orden = [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
    P = [int(v[i, j]) for i, j in orden]
    return sum(abs(P[k] - P[(k + 1) % 8]) for k in range(8)) // 2


def paso_5_crossing(E: dict) -> None:
    from src.huellas.caracteristicas import _crossing_number, _esqueleto_limpio
    encabezado(5, "Bloque 3 · minucias: el crossing number")
    h, c = E["h"], E["ch"]
    Eesq = _esqueleto_limpio(np.asarray(h.esqueleto, bool), int(c["componente_min_px"]))
    CN = _crossing_number(Eesq)
    E["esq"], E["CN"] = Eesq, CN
    ejemplos = [("terminación", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])),
                ("cresta que sigue", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
                ("bifurcación", np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))]
    manzanita("Recorro los 8 vecinos de cada píxel del esqueleto dando una vuelta completa, como caminar alrededor de una "
              "manzana contando cuántas calles cruzo. Si cruzo 1, la cresta se termina ahí (terminación); si cruzo 2, la "
              "cresta entra y sigue; si cruzo 3, se abre en dos (bifurcación). Las terminaciones y las bifurcaciones son "
              "las minucias: los «detalles» que hacen única a una huella.")
    decir("El tercer bloque son las minucias, los puntos donde una cresta termina o se divide. Las busco en el esqueleto "
          "con el crossing number: doy la vuelta a los 8 vecinos de cada píxel y cuento cuántas veces paso de vacío a "
          "cresta. Uno es una terminación, dos es una cresta normal y tres es una bifurcación.")
    cuenta = {k: int(np.count_nonzero(Eesq & (CN == k))) for k in (1, 2, 3)}
    hablar(f"CN(p) = ½·Σ |P_k − P_k+1| sobre los 8 vecinos en ronda. Antes de contar se sacan las componentes del "
           f"esqueleto de menos de {c['componente_min_px']} píxeles. En esta huella: {cuenta[1]} píxeles con CN = 1, "
           f"{cuenta[2]} con CN = 2 y {cuenta[3]} con CN = 3 (todavía sin limpiar: ver el paso siguiente).")
    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.7, 0.7, 0.7, 1.6, 1.6])
    for j, (nombre, v) in enumerate(ejemplos):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(1 - v, cmap="gray", vmin=0, vmax=1)
        for (i, k), val in np.ndenumerate(v):
            ax.add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, ec="#999999"))
        ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, ec=ROJO, lw=2))
        sin_ejes(ax, f"{nombre}\nCN = {_crossing_number_vecinos(v)}")
    ys, xs = np.nonzero(Eesq)
    cy, cx = 124, 124
    lado = 60
    for j, (y0, x0, tit) in enumerate([(cy - lado // 2, cx - lado // 2, "zoom al centro"),
                                       (None, None, "todo el esqueleto")], start=3):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(~Eesq, cmap="gray")
        for k, col in [(1, ROJO), (3, AZUL)]:
            yy, xx = np.nonzero(Eesq & (CN == k))
            ax.plot(xx, yy, "o", ms=4 if y0 is not None else 2.5, color=col, mfc="none",
                    label="CN = 1 (terminación)" if k == 1 else "CN = 3 (bifurcación)")
        if y0 is not None:
            ax.set_xlim(x0, x0 + lado); ax.set_ylim(y0 + lado, y0)
        sin_ejes(ax, tit)
    ax.legend(fontsize=8.5, loc="lower right")
    cerrar(fig, 5, "crossing", "Bloque 3 · minucias: el crossing number",
           "CN = mitad de los cambios vacío↔cresta al dar la vuelta a los 8 vecinos: 1 terminación, 2 cresta, 3 bifurcación. "
           "En rojo y azul, todos los candidatos antes de limpiar.")


def paso_6_limpieza(E: dict) -> None:
    from src.huellas.caracteristicas import _sin_pares, _uniones_alcanzadas
    encabezado(6, "Bloque 3 · minucias: limpiar las falsas")
    h, c = E["h"], E["ch"]
    Eesq, CN = E["esq"], E["CN"]
    D = _dist_borde(h.mascara)
    V = D >= c["borde_minucia_px"]
    T_todas = [(int(i), int(j)) for i, j in np.argwhere(Eesq & (CN == 1))]
    B_todas = [(int(i), int(j)) for i, j in np.argwhere(Eesq & (CN == 3))]
    T0 = [p for p in T_todas if V[p]]
    B0 = [p for p in B_todas if V[p]]
    union = Eesq & (CN >= 3)
    espuelas, uniones = set(), set()
    for t in T0:
        alc = _uniones_alcanzadas(t, Eesq, union, int(c["espuela_px"]))
        if alc:
            espuelas.add(t); uniones |= alc
    T1 = [t for t in T0 if t not in espuelas]
    if uniones:
        U = np.array(sorted(uniones))
        B1 = [p for p in B0 if not np.any((np.abs(U[:, 0] - p[0]) <= 2) & (np.abs(U[:, 1] - p[1]) <= 2))]
    else:
        B1 = B0
    T2 = _sin_pares(T1, float(c["par_terminaciones_px"]))
    B2 = _sin_pares(B1, float(c["par_bifurcaciones_px"]))
    E["T_propio"], E["B_propio"] = T2, B2
    E["V"] = V
    manzanita("El esqueleto tiene «pelitos» y cortes que no son minucias de verdad, son defectos del dibujo. Es como contar "
              "las ramas de un árbol: un brote de dos centímetros no es una rama, y donde la foto corta el árbol tampoco "
              "termina una rama. Entonces saco lo que está pegado al borde, los pelitos (espuelas) y los pares de puntas "
              "enfrentadas, que suelen ser una cresta cortada.")
    decir(f"No todo lo que da el crossing number es una minucia. Saco tres cosas. Lo que está a menos de "
          f"{c['borde_minucia_px']:.0f} px del borde, porque ahí las crestas se cortan por el recorte y no porque "
          f"terminen: esto es clave en las huellas con corte como esta. Las espuelas, que son pelitos de hasta "
          f"{c['espuela_px']} px que salen de una cresta. Y los pares: dos terminaciones a menos de "
          f"{c['par_terminaciones_px']:.0f} px suelen ser una cresta cortada. De {len(T_todas) + len(B_todas)} candidatos "
          f"quedan {len(T2) + len(B2)} minucias: {len(T2)} terminaciones y {len(B2)} bifurcaciones.")
    hablar(f"Terminaciones: {len(T_todas)} candidatas → {len(T0)} lejos del borde → {len(T1)} sin espuelas → {len(T2)} sin "
           f"pares. Bifurcaciones: {len(B_todas)} → {len(B0)} → {len(B1)} (sin las pegadas a una espuela) → {len(B2)}. "
           f"El prototipo midió que, con este filtro, las huellas con corte tienen la misma densidad de minucias que las "
           f"sin corte (cociente 1,03, 1,06 y 1,03 por identidad).")
    fig, ejes = figura(1, 2, 14, 7)
    for ax, tit, fase in [(ejes[0, 0], f"candidatos: {len(T_todas)} terminaciones y {len(B_todas)} bifurcaciones", 0),
                          (ejes[0, 1], f"después de limpiar: {len(T2)} terminaciones y {len(B2)} bifurcaciones", 1)]:
        ax.imshow(~Eesq, cmap="gray")
        ax.contour(V.astype(float), levels=[0.5], colors=[VERDE], linewidths=1.2)
        if fase == 0:
            for pts, col, mk in [(T_todas, ROJO, "o"), (B_todas, AZUL, "s")]:
                if pts:
                    P = np.array(pts); ax.plot(P[:, 1], P[:, 0], mk, color=col, ms=6, mfc="none")
        else:
            fuera = [p for p in T_todas + B_todas if not V[p]]
            sacadas = [p for p in T0 if p not in T2] + [p for p in B0 if p not in B2]
            for pts, col, mk, lab in [(fuera, GRIS, "x", "cerca del borde"), (sacadas, NARANJA, "x", "espuela o par"),
                                      (T2, ROJO, "o", "terminación"), (B2, AZUL, "s", "bifurcación")]:
                if pts:
                    P = np.array(pts); ax.plot(P[:, 1], P[:, 0], mk, color=col, ms=7, mfc="none", mew=1.8, label=lab)
            ax.legend(fontsize=9, loc="lower right")
        sin_ejes(ax, tit)
    cerrar(fig, 6, "limpieza", "Bloque 3 · minucias: limpiar las falsas",
           f"Verde: la zona válida (a ≥ {c['borde_minucia_px']:.0f} px del borde de la huella). Se descartan las minucias del "
           "borde (el corte no es una terminación), las espuelas y los pares muy juntos.")


def paso_7_densidades(E: dict) -> None:
    encabezado(7, "Bloque 3 · minucias: cuántas hay y dónde")
    h, c = E["h"], E["ch"]
    V = E["V"]
    T = np.array([(j, i) for i, j in E["T_propio"]], float).reshape(-1, 2)
    B = np.array([(j, i) for i, j in E["B_propio"]], float).reshape(-1, 2)
    r, _ = E["polar"]
    radios = [float(v) for v in c["anillos_minucias_px"]]
    nv = np.count_nonzero(V)
    valores = [1000 * len(T) / nv, 1000 * len(B) / nv, len(B) / (len(T) + len(B)) if len(T) + len(B) else 0.0]
    por_anillo = {}
    for nombre, P in (("T", T), ("B", B)):
        rp = np.hypot(P[:, 0] - 124.5, P[:, 1] - 124.5)
        for a in range(len(radios) - 1):
            anillo = (r >= radios[a]) & (r < radios[a + 1])
            dentro = np.count_nonzero(anillo & V)
            v = np.nan if dentro < c["cobertura_min"] * np.count_nonzero(anillo) or dentro == 0 else \
                1000 * np.count_nonzero((rp >= radios[a]) & (rp < radios[a + 1])) / dentro
            valores.append(v); por_anillo[(nombre, a)] = v
    gm = int(c["grilla_minucias"])
    b = _bordes(gm)
    todas = np.vstack([T, B])
    grilla = np.full((gm, gm), np.nan)
    for i in range(gm):
        for j in range(gm):
            zona = V[b[i]:b[i + 1], b[j]:b[j + 1]]
            dentro = np.count_nonzero(zona)
            if dentro >= c["cobertura_min"] * zona.size and dentro > 0:
                en = (todas[:, 0] >= b[j]) & (todas[:, 0] < b[j + 1]) & (todas[:, 1] >= b[i]) & (todas[:, 1] < b[i + 1])
                grilla[i, j] = 1000 * np.count_nonzero(en) / dentro
            valores.append(grilla[i, j])
    E["minucias_propio"] = np.array(valores)
    manzanita("La lista exacta de puntos cambia con cada toma (un poco de presión y aparece o desaparece una). Entonces no "
              "guardo la lista: guardo cuántas hay por zona, como contar cuántas manzanas con gusano hay en cada cajón. Eso "
              "se parece mucho entre dos tomas del mismo dedo aunque los puntos no coincidan uno a uno.")
    decir(f"Con las minucias limpias armo {len(valores)} números: la densidad de terminaciones y de bifurcaciones (cuántas "
          f"cada 1000 píxeles de huella válida) y qué proporción son bifurcaciones; la densidad en {len(radios) - 1} anillos "
          f"alrededor del núcleo, para cada tipo; y la densidad en una grilla de {gm} × {gm}. Uso densidades y no la cantidad "
          f"porque así una huella cortada, que tiene menos superficie, no parece tener menos minucias.")
    hablar(f"Globales: {valores[0]:.2f} terminaciones y {valores[1]:.2f} bifurcaciones cada 1000 px² (zona válida de {nv} "
           f"px²), proporción de bifurcaciones {valores[2]:.2f}. Anillos {radios}: "
           + "; ".join(f"{'T' if k[0] == 'T' else 'B'} anillo {k[1] + 1} = {v:.2f}" for k, v in por_anillo.items())
           + f". Grilla de {gm} × {gm}: {int(np.isnan(grilla).sum())} celda(s) sin zona válida suficiente (NaN).")
    fig, ejes = figura(1, 3, 15.5, 5.2, gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
    ax = ejes[0, 0]
    ax.imshow(~E["esq"], cmap="gray", alpha=0.5)
    t = np.linspace(0, 2 * np.pi, 300)
    for rad in radios[1:]:
        ax.plot(124.5 + rad * np.cos(t), 124.5 + rad * np.sin(t), color=VIOLETA, lw=1)
    if len(T):
        ax.plot(T[:, 0], T[:, 1], "o", color=ROJO, mfc="none", ms=7, mew=1.6, label="terminación")
    if len(B):
        ax.plot(B[:, 0], B[:, 1], "s", color=AZUL, mfc="none", ms=7, mew=1.6, label="bifurcación")
    ax.set_xlim(0, 250); ax.set_ylim(250, 0); ax.legend(fontsize=8.5, loc="lower right")
    sin_ejes(ax, "minucias finales y anillos")
    ax = ejes[0, 1]
    x = np.arange(len(radios) - 1)
    ax.bar(x - 0.2, [por_anillo[("T", a)] for a in x], 0.4, color=ROJO, label="terminaciones")
    ax.bar(x + 0.2, [por_anillo[("B", a)] for a in x], 0.4, color=AZUL, label="bifurcaciones")
    ax.set_xticks(x, [f"{radios[a]:.0f}–{radios[a + 1]:.0f} px" for a in x])
    ax.set_ylabel("minucias cada 1000 px²"); ax.legend(fontsize=9); ax.set_title("densidad por anillo", fontsize=9.5)
    ax = ejes[0, 2]
    im = ax.imshow(grilla, cmap="Greens", vmin=0)
    for i in range(gm):
        for j in range(gm):
            claro = not np.isnan(grilla[i, j]) and grilla[i, j] > 0.6 * np.nanmax(grilla)
            ax.text(j, i, "NaN" if np.isnan(grilla[i, j]) else f"{grilla[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if claro else "black")
    sin_ejes(ax, f"densidad en la grilla de {gm} × {gm}")
    cerrar(fig, 7, "densidades", "Bloque 3 · minucias: cuántas hay y dónde",
           f"{len(valores)} valores: 3 globales, {2 * (len(radios) - 1)} por anillo y {gm * gm} en la grilla. Densidades (por 1000 px² de "
           "huella válida) para que una huella cortada no parezca tener menos minucias.")


def paso_8_escalador_h(E: dict) -> None:
    from src.core_ml.metricas import silhouette
    from src.huellas.caracteristicas import BLOQUES, ajustar_escalador
    encabezado(8, "Escalador y pesos: que los tres bloques valgan lo mismo")
    z = E["npz_h"]
    tr = z["particion"] == "train"
    va = (z["particion"] == "val") & (z["identidad"] > 0)
    esc = ajustar_escalador({b: z[b][tr] for b in BLOQUES})
    E["esc_h"] = esc
    X = z["X"]
    o, filas = 0, []
    for b in BLOQUES:
        d = int(esc.columnas[b].sum())
        Zb = X[:, o:o + d]; o += d
        filas.append((b, d, esc.pesos[b], contribucion(Zb[tr] / esc.pesos[b]), contribucion(Zb[tr]),
                      silhouette(Zb[va], z["identidad"][va])))
    sil_tr, sil_va = silhouette(X[tr], z["identidad"][tr]), silhouette(X[va], z["identidad"][va])
    E["sil_h"] = (sil_tr, sil_va)
    manzanita("Primero paso todo a la misma unidad: «cuántos desvíos se aleja del promedio» (como comparar notas de "
              "distintas materias con la nota estandarizada). Después está el problema del jurado: si un jurado tiene 300 "
              "votos y otro 24, el de 300 decide solo. Por eso divido cada bloque por la raíz de su cantidad de columnas, y "
              "así los tres bloques pesan lo mismo en la distancia.")
    decir(f"Los bloques tienen escalas y tamaños muy distintos. Primero estandarizo cada columna con la media y el desvío "
          f"de train, y lo que quedó vacío pasa a cero, que es el promedio de train. Después peso cada bloque "
          f"por uno sobre la raíz de sus columnas. Sin ese peso, Gabor aportaría {filas[1][3]:.0f} a la distancia contra "
          f"{filas[0][3]:.0f} de orientación y {filas[2][3]:.0f} de minucias; con el peso, cada uno aporta cerca de 2. Y los "
          f"tres bloques separan las identidades por su cuenta: la silhouette en validación da "
          f"{filas[0][5]:.2f}, {filas[1][5]:.2f} y {filas[2][5]:.2f}, y todo junto {sil_va:.2f}.")
    hablar("z = (x − media_train) / desvío_train, NaN → 0, y cada bloque × 1/√(columnas). " +
           "; ".join(f"{b}: {d} columnas, peso {p:.3f}, contribución sin peso {sp:.1f} → con peso {cp:.2f}, silhouette val "
                     f"{s:.3f}" for b, d, p, sp, cp, s in filas) +
           f". Silhouette de las 374 columnas: {sil_tr:.3f} en train y {sil_va:.3f} en val (va de −1 a 1: cuánto más cerca "
           f"está cada huella de su grupo que del grupo vecino). Los momentos de Hu se probaron y se sacaron: separaban "
           f"0,13 y bajaban el total de 0,67 a 0,51.")
    fig, ejes = figura(1, 2, 14, 4.8)
    ax = ejes[0, 0]
    x = np.arange(len(filas))
    ax.bar(x - 0.2, [f[3] for f in filas], 0.4, color=GRIS, label="sin peso")
    ax.bar(x + 0.2, [f[4] for f in filas], 0.4, color=[BLOQUE_COLOR[f[0]] for f in filas], label="con peso 1/√columnas")
    for i, f in enumerate(filas):
        ax.text(i - 0.2, f[3] * 1.15, f"{f[3]:.0f}", ha="center", fontsize=9)
        ax.text(i + 0.2, f[4] * 1.15, f"{f[4]:.2f}", ha="center", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(0.7, 3000); ax.set_xticks(x, [f"{f[0]}\n({f[1]} columnas)" for f in filas])
    ax.set_ylabel("aporte a la distancia² (escala log)"); ax.legend(fontsize=9)
    ax.set_title("cuánto pesa cada bloque en la distancia (train)", fontsize=9.5)
    ax = ejes[0, 1]
    ax.bar(x, [f[5] for f in filas], color=[BLOQUE_COLOR[f[0]] for f in filas])
    ax.axhline(sil_va, color="black", ls="--", lw=1)
    ax.text(len(filas) - 0.5, sil_va + 0.02, f"todo junto: {sil_va:.2f}", ha="right", fontsize=9)
    for i, f in enumerate(filas):
        ax.text(i, f[5] + 0.02, f"{f[5]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [f[0] for f in filas]); ax.set_ylim(0, 1); ax.set_ylabel("silhouette en val")
    ax.set_title("cuánto separa cada bloque las 3 identidades (val)", fontsize=9.5)
    cerrar(fig, 8, "escalador_huellas", "Escalador y pesos: que los tres bloques valgan lo mismo",
           "z por columna con media y desvío de train (lo vacío → 0) y cada bloque × 1/√columnas: sin el peso, Gabor decidiría solo. "
           "Los tres bloques separan por su cuenta, y juntos separan más.")


def paso_9_pca_h(E: dict) -> None:
    from src.core_ml.pca import PCA
    encabezado(9, "PCA: las 300 huellas en 2 dimensiones")
    z = E["npz_h"]
    X, part, ide = z["X"], z["particion"], z["identidad"]
    tr = part == "train"
    p = PCA(2).ajustar(X[tr])
    P = p.transformar(X)
    v1, v2 = p.proporcion_varianza
    from src.huellas.clasificacion import cargar_modelo
    mh = cargar_modelo(RAIZ / cfg["rutas"]["modelos"] / "modelo_huellas.npz")
    puntajes, pred = mh.puntajes(X)
    i = E["fila_h"]
    misma = tr & (ide == ide[i])
    rango = int((puntajes[misma] > puntajes[i]).sum()) + 1
    estrella = (f"La estrella es la toma que seguí: con el corte, la presión máxima y la escala reducida queda en el borde de "
                f"su grupo (es la {rango}.ª más lejana de las {int(misma.sum())} de train de su identidad), pero sigue "
                f"estando más cerca de la identidad {int(pred[i])} que de las otras.")
    manzanita("374 dimensiones no se pueden dibujar. El PCA busca las dos direcciones en las que los puntos están más "
              "desparramados, como sacarle una foto a una nube de puntos desde el ángulo en que mejor se ve. La foto "
              "aplasta un poco (se pierde lo que no entra en el plano), pero alcanza para ver si los grupos se separan.")
    decir(f"Para ver si la ficha sirve, proyecto las 300 huellas con PCA ajustado solo con train. Las dos primeras "
          f"componentes explican el {100 * (v1 + v2):.0f} % de la varianza, y se ven tres grupos compactos y lejos entre sí, "
          f"uno por identidad. Validación y test, sin relleno, caen en el grupo de su identidad, incluidas las que tienen "
          f"corte. Los intrusos, en gris, quedan afuera o entre los grupos. La silhouette es {E['sil_h'][0]:.2f} en train y "
          f"{E['sil_h'][1]:.2f} en validación. {estrella} Ojo: la decisión no se toma en este dibujo sino en las 374 "
          f"dimensiones.")
    hablar(f"PCA propio (src/core_ml/pca.py, comparado con scikit-learn en los tests hasta 1e-11): PC1 {100 * v1:.1f} % y "
           f"PC2 {100 * v2:.1f} %. {E['huella']}: distancia normalizada a su centroide {puntajes[i]:.2f} (el umbral τ* de H4 es "
           f"{mh.tau:.2f}).")
    fig, ejes = figura(1, 1, 9.5, 7.2)
    ax = ejes[0, 0]
    intr = ide == 0
    ax.scatter(P[intr, 0], P[intr, 1], s=22, c=GRIS, alpha=0.6, label="intrusos (val y test)")
    for k in (1, 2, 3):
        m = ide == k
        ax.scatter(P[m & tr, 0], P[m & tr, 1], s=30, c=ID_COLOR[k], label=f"identidad {k} · train")
        ax.scatter(P[m & ~tr, 0], P[m & ~tr, 1], s=40, facecolors="none", edgecolors=ID_COLOR[k], lw=1.5,
                   label=f"identidad {k} · val y test")
    ax.plot(P[i, 0], P[i, 1], "*", ms=22, color="black", mfc="yellow")
    ax.set_xlabel(f"PC1 ({100 * v1:.1f} %)"); ax.set_ylabel(f"PC2 ({100 * v2:.1f} %)")
    ax.legend(fontsize=8.5, loc="best"); ax.set_title(f"silhouette: train {E['sil_h'][0]:.2f} · val {E['sil_h'][1]:.2f}",
                                                      fontsize=10)
    cerrar(fig, 9, "pca_huellas", "PCA: las 300 huellas en 2 dimensiones",
           f"PCA ajustado solo con train ({100 * (v1 + v2):.0f} % de la varianza en 2D). Tres grupos separados; val y test caen en "
           "el suyo. La estrella es la huella que seguimos. La decisión se toma en las 374 dimensiones, no en el dibujo.")


# ============================================================================= PARTE B · AUDIOS
def cargar_voz(E: dict) -> None:
    import soundfile as sf
    from src.voz.caracteristicas import caracteristicas
    z = np.load(RAIZ / cfg["rutas"]["voz"] / "prepro" / "caracteristicas.npz", allow_pickle=False)
    E["npz_v"] = {k: z[k] for k in z.files}
    archivos = list(E["npz_v"]["archivo"])
    candidatos = [a for a in archivos if a.endswith(f"/nb_{E['toma']}_k0.wav")]
    if not candidatos:
        raise SystemExit(f"No encuentro nb_{E['toma']}_k0.wav entre los clips válidos de V2.")
    E["archivo_v"] = candidatos[0]
    E["fila_v"] = archivos.index(candidatos[0])
    y, fs = sf.read(str(RAIZ / cfg["rutas"]["voz"] / "prepro" / candidatos[0]), dtype="float32")
    E["y"], E["fs"] = y.reshape(-1), int(fs)
    E["bloques_v"] = caracteristicas(E["y"], cfg)
    v = cfg["voz"]
    E["W"] = int(round(v["mfcc"]["ventana_ms"] * fs / 1000))
    E["H"] = int(round(v["mfcc"]["avance_ms"] * fs / 1000))
    E["M"] = int(round(v["preprocesamiento"]["margen_ms"] * fs / 1000))
    E["NF"] = int(v["mfcc"]["n_fft"])
    E["m"] = E["M"] // E["H"]


def mfcc_a_mano(y: np.ndarray, E: dict) -> dict:
    """Los MFCC como los calcula librosa, pero paso a paso (para dibujar cada etapa)."""
    import librosa
    import scipy.fft
    v = cfg["voz"]
    fs, W, H, NF = E["fs"], E["W"], E["H"], E["NF"]
    tramas = librosa.util.frame(np.asarray(y, np.float32), frame_length=NF, hop_length=H)
    ventana = librosa.util.pad_center(librosa.filters.get_window("hann", W, fftbins=True), size=NF)
    espectro = np.abs(np.fft.rfft(tramas * ventana[:, None], axis=0)) ** 2           # |X|², (NF/2 + 1, N)
    banco = librosa.filters.mel(sr=fs, n_fft=NF, n_mels=int(v["mfcc"]["n_mels"]), fmin=float(v["pasabanda_hz"][0]),
                                fmax=float(v["mfcc"]["fmax"]))
    mel = banco @ espectro
    log_mel = 10 * np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 80.0)                             # top_db = 80, como librosa
    mfcc = scipy.fft.dct(log_mel, axis=0, type=2, norm="ortho")[: int(v["mfcc"]["n_mfcc"])]
    return dict(tramas=tramas, ventana=ventana, espectro=espectro, banco=banco, mel=mel, log_mel=log_mel, mfcc=mfcc)


def paso_10_tramas(E: dict) -> None:
    encabezado(10, "Audio: la palabra cortada en tramas de 25 ms")
    y, fs, W, H, M, NF, m = E["y"], E["fs"], E["W"], E["H"], E["M"], E["NF"], E["m"]
    n_stft = 1 + (len(y) - NF) // H
    E["MM"] = mfcc_a_mano(y, E)
    N = n_stft - 2 * m
    manzanita("La voz cambia todo el tiempo, pero en 25 milésimas casi no cambia. Es como una película: la corto en "
              "fotogramas de 25 ms, uno cada 10 ms (se pisan un poco, para no perder nada entre uno y otro), y a cada "
              "fotograma le hago una ficha. Después junto las fichas de toda la palabra.")
    decir(f"Del lado de la voz, arranco del clip de V2: la palabra recortada, filtrada y a −26 dBFS. Para "
          f"{E['toma']} son {len(y)} muestras, {len(y) / fs:.2f} s. La corto en tramas de {W} muestras, 25 ms, que avanzan "
          f"de a {H}, 10 ms. Las {m} primeras y las {m} últimas son el margen de silencio que dejó V2, así que las saco: "
          f"quedan {N} tramas de palabra.")
    hablar(f"W = {W} muestras (25 ms, ventana de Hann), H = {H} (10 ms), FFT de {NF} puntos, sin centrar: "
           f"1 + ({len(y)} − {NF}) // {H} = {n_stft} tramas. Margen M = {M} muestras = {m} tramas por lado → N = {N}.")
    fig, ejes = figura(2, 1, 14, 6.8, gridspec_kw=dict(height_ratios=[1, 1]))
    ax = ejes[0, 0]
    t = np.arange(len(y)) / fs
    ax.plot(t, y, color="#333333", lw=0.5)
    ax.axvspan(0, M / fs, color=NARANJA, alpha=0.25); ax.axvspan((len(y) - M) / fs, len(y) / fs, color=NARANJA, alpha=0.25)
    ax.axvspan(M / fs, (len(y) - M) / fs, color=VERDE, alpha=0.1)
    ax.set_xlim(0, len(y) / fs); ax.set_xlabel("tiempo (s)")
    ax.set_title(f"{E['archivo_v']}: naranja el margen ({m} tramas por lado), verde la palabra ({N} tramas)", fontsize=10,
                 loc="left")
    ax = ejes[1, 0]
    k0 = n_stft // 2
    t0, t1 = (k0 * H - H) / fs, ((k0 + 4) * H + NF) / fs
    seg = (t >= t0) & (t <= t1)
    ax.plot(t[seg] * 1000, y[seg], color="#333333", lw=0.8)
    lim = np.abs(y[seg]).max()
    for j, k in enumerate(range(k0, k0 + 4)):
        a = (k * H + (NF - W) // 2) / fs * 1000
        col = [AZUL, VERDE, ROJO, VIOLETA][j]
        ax.add_patch(plt.Rectangle((a, -lim * (1.05 + 0.12 * j)), W / fs * 1000, 0.08 * lim, color=col, alpha=0.8))
        ax.text(a + W / fs * 1000 + 0.6, -lim * (1.05 + 0.12 * j) + 0.04 * lim, f"trama {k}", fontsize=8.5, ha="left",
                va="center", color=col)
    ax.set_ylim(-lim * 1.55, lim * 1.1); ax.set_xlabel("tiempo (ms)")
    ax.set_title("zoom: cada trama dura 25 ms y la siguiente arranca 10 ms después (se pisan)", fontsize=10, loc="left")
    cerrar(fig, 10, "tramas", "Audio: la palabra cortada en tramas de 25 ms",
           f"Tramas de 25 ms cada 10 ms. Se sacan las {m} tramas de margen de cada lado: quedan {N} tramas de palabra. "
           "A cada trama se le calcula una ficha (MFCC, energía y cruces por cero).")


def paso_11_mfcc_trama(E: dict) -> None:
    encabezado(11, "MFCC de una trama, paso a paso")
    MM, fs, NF = E["MM"], E["fs"], E["NF"]
    k = MM["tramas"].shape[1] // 2
    f = np.fft.rfftfreq(NF, 1 / fs)
    esp_db = 10 * np.log10(MM["espectro"][:, k] + 1e-12)
    v = cfg["voz"]["mfcc"]
    manzanita("El oído no escucha las frecuencias parejas: distingue bien 200 de 300 Hz, pero 3200 de 3300 le suenan casi "
              "igual. Los 40 filtros mel imitan eso: angostos en los graves y anchos en los agudos. Después tomo el "
              "logaritmo, porque el oído escucha en decibeles. Y por último la transformada coseno resume la «silueta» del "
              "espectro, que depende de la forma de la boca, en 13 números: el primero es el volumen general, el segundo "
              "dice si hay más graves o más agudos, y los siguientes son detalles cada vez más finos.")
    decir(f"Así se calculan los MFCC de una trama. Multiplico la trama por una ventana de Hann y saco el espectro con la "
          f"FFT. Lo paso por {v['n_mels']} filtros triangulares en escala mel, de 80 a {v['fmax']} Hz, que imitan cómo oye "
          f"el oído: finos en los graves y anchos en los agudos. Tomo el logaritmo de la energía de cada filtro y le aplico "
          f"la transformada coseno: me quedo con los {v['n_mfcc']} primeros coeficientes, que describen la forma del "
          f"espectro, o sea, la posición de la boca y la lengua en ese momento.")
    hablar(f"Trama {k}: |X(f)|² con FFT de {NF} puntos (resolución {fs / NF:.2f} Hz), {v['n_mels']} filtros mel entre "
           f"{cfg['voz']['pasabanda_hz'][0]} y {v['fmax']} Hz, log10 (con piso de 80 dB bajo el máximo, como librosa) y "
           f"DCT tipo II ortonormal: c_n = Σ_m log E_m · cos[πn/{v['n_mels']}·(m − ½)], n = 0 … {v['n_mfcc'] - 1}.")
    fig, ejes = figura(2, 3, 15.5, 8.2)
    ax = ejes[0, 0]
    tr = MM["tramas"][:, k]
    ax.plot(tr, color=GRIS, lw=0.8, label="trama")
    ax.plot(tr * MM["ventana"], color=AZUL, lw=0.9, label="× ventana de Hann")
    ax.set_xlabel("muestra"); ax.legend(fontsize=8.5); ax.set_title("1 · la trama y la ventana", fontsize=9.5)
    ax = ejes[0, 1]
    ax.plot(f, esp_db, color=AZUL, lw=0.8)
    ax.set_xlim(0, fs / 2); ax.set_xlabel("frecuencia (Hz)"); ax.set_ylabel("dB")
    ax.set_title("2 · espectro de potencia |X(f)|²", fontsize=9.5)
    ax = ejes[0, 2]
    ax.plot(f, esp_db - esp_db.max(), color=GRIS, lw=0.6)
    for m_i, fila in enumerate(MM["banco"]):
        ax.plot(f, 30 * fila / fila.max() - 30, color=[AZUL, VERDE, ROJO, VIOLETA][m_i % 4], lw=0.8)
    ax.set_xlim(0, 4500); ax.set_ylim(-70, 2); ax.set_xlabel("frecuencia (Hz)")
    ax.set_title(f"3 · {v['n_mels']} filtros mel (finos abajo, anchos arriba)", fontsize=9.5)
    ax = ejes[1, 0]
    ax.bar(range(len(MM["log_mel"][:, k])), MM["log_mel"][:, k], color=NARANJA)
    ax.set_xlabel("filtro mel"); ax.set_ylabel("dB"); ax.set_title("4 · log de la energía de cada filtro", fontsize=9.5)
    ax = ejes[1, 1]
    c = MM["mfcc"][:, k]
    ax.bar(range(len(c)), c, color=[ROJO if i == 0 else AZUL for i in range(len(c))])
    ax.set_xlabel("coeficiente"); ax.set_xticks(range(len(c)))
    ax.set_title(f"5 · transformada coseno → {len(c)} MFCC (c0 en rojo: el volumen)", fontsize=9.5)
    ax = ejes[1, 2]
    rec = scipy_idct(c, len(MM["log_mel"][:, k]))
    ax.plot(MM["log_mel"][:, k], color=NARANJA, lw=1.5, label="log mel (40 valores)")
    ax.plot(rec, color=AZUL, lw=1.5, ls="--", label=f"lo que guardan los {len(c)} MFCC")
    ax.set_xlabel("filtro mel"); ax.legend(fontsize=8.5)
    ax.set_title("6 · los 13 MFCC guardan la silueta, no el detalle", fontsize=9.5)
    cerrar(fig, 11, "mfcc_trama", "MFCC de una trama, paso a paso",
           "Ventana → espectro → 40 filtros mel → logaritmo → transformada coseno → 13 MFCC. Los 13 números guardan la forma "
           "general del espectro (la «silueta» que da la boca), que es lo que distingue un sonido de otro.")


def scipy_idct(c: np.ndarray, n: int) -> np.ndarray:
    import scipy.fft
    full = np.zeros(n)
    full[:len(c)] = c
    return scipy.fft.idct(full, type=2, norm="ortho")


def paso_12_cmn(E: dict) -> None:
    encabezado(12, "MFCC en el tiempo y la normalización cepstral (CMN)")
    MM, m, H, fs = E["MM"], E["m"], E["H"], E["fs"]
    w = MM["mfcc"][:, m:MM["mfcc"].shape[1] - m]
    media = w.mean(axis=1, keepdims=True)
    wc = w - media
    E["w_cmn"] = wc
    manzanita("Un micrófono que realza los graves es como mirar con anteojos de sol: tiñe todo igual. En el logaritmo, un "
              "filtro fijo (el micrófono, la sala) es una suma constante en todas las tramas. Si a cada coeficiente le "
              "resto su promedio en la palabra, me saco los anteojos: queda cómo se mueve la palabra, no qué micrófono se "
              "usó.")
    decir(f"Hago eso en cada trama y me queda una matriz: {w.shape[0]} coeficientes por {w.shape[1]} tramas. Arriba se ve "
          f"el espectro mel en el tiempo, y abajo los MFCC antes y después de la normalización cepstral. La normalización "
          f"le resta a cada coeficiente su promedio en la palabra: así se cancela el efecto del micrófono y de la sala, "
          f"que en el dominio logarítmico es una constante. Esto importa porque la demo es con otra gente y otro lugar.")
    hablar(f"Matriz MFCC de la palabra: {w.shape[0]} × {w.shape[1]}. Medias que se restan (CMN), de c0 a c12: "
           + ", ".join(f"{v:.1f}" for v in media[:, 0]) + ". c0 tiene la media más grande porque es el nivel general.")
    fig, ejes = figura(3, 1, 14, 9.4, sharex=True)
    t = (np.arange(w.shape[1]) + m) * H / fs
    ext = [t[0], t[-1] + H / fs]
    ax = ejes[0, 0]
    ax.imshow(MM["log_mel"][:, m:MM["log_mel"].shape[1] - m], origin="lower", aspect="auto", cmap="magma", extent=ext + [0, 40])
    ax.set_ylabel("filtro mel"); ax.set_title("espectro mel (log), 40 filtros × tramas", fontsize=9.5, loc="left")
    for ax, mat, tit in [(ejes[1, 0], w, "MFCC antes de CMN (c0 domina la escala)"),
                         (ejes[2, 0], wc, "MFCC después de CMN: a cada fila se le restó su promedio")]:
        lim = np.abs(mat[1:]).max()
        ax.imshow(mat, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext + [-0.5, 12.5])
        ax.set_ylabel("coeficiente"); ax.set_title(tit, fontsize=9.5, loc="left")
    ejes[2, 0].set_xlabel("tiempo (s)")
    cerrar(fig, 12, "mfcc_cmn", "MFCC en el tiempo y la normalización cepstral (CMN)",
           "Una ficha de 13 MFCC por trama. CMN le resta a cada coeficiente su media en la palabra: un micrófono o una sala distintos "
           "suman una constante en el log, y así se cancelan.")


def paso_13_delta(E: dict) -> None:
    import librosa
    encabezado(13, "Δ, energía y cruces por cero")
    y, W, H, m, fs = E["y"], E["W"], E["H"], E["m"], E["fs"]
    wc = E["w_cmn"]
    d = librosa.feature.delta(wc, width=int(cfg["voz"]["mfcc"]["delta_ventana"]), mode="nearest")
    fr = librosa.util.frame(y, frame_length=W, hop_length=H).astype(np.float64)
    fr = fr[:, m:fr.shape[1] - m]
    rms = 20 * np.log10(np.maximum(np.sqrt(np.mean(fr ** 2, axis=0)), float(cfg["voz"]["mfcc"]["piso_rms"])))
    zcr = (np.abs(np.diff(np.sign(fr), axis=0)) > 0).mean(axis=0)
    N = min(wc.shape[1], len(rms))
    E.update(delta=d[:, :N], rms=rms[:N], zcr=zcr[:N], N=N)
    kz = int(np.argmax(zcr[:N]))
    manzanita("Los MFCC dicen dónde está la boca; Δ dice hacia dónde se mueve, como saber que un auto está en la esquina Y que "
              "va frenando. La energía es el volumen de cada trama. Y los cruces por cero cuentan cuántas veces la onda "
              "cruza el cero: un soplido como la «s» es ruido y cruza muchísimo; una vocal, que es una onda lenta, cruza poco.")
    decir(f"A los MFCC les sumo tres cosas por trama. Δ, la derivada en el tiempo, con un filtro de Savitzky-Golay de "
          f"{cfg['voz']['mfcc']['delta_ventana']} tramas: dice cómo cambia la boca de una trama a la otra. La energía en dB, "
          f"que marca las sílabas. Y la tasa de cruces por cero, que es alta en las consonantes de soplido: en "
          f"«reconocer» el pico está en la «c» de «-cer», que en Argentina suena como una «s».")
    hablar(f"Δ: librosa.feature.delta con ancho {cfg['voz']['mfcc']['delta_ventana']} (Savitzky y Golay, 1964). Energía: "
           f"20·log10(rms) por trama de {W} muestras. ZCR: fracción de muestras consecutivas que cambian de signo. "
           f"Máximo de ZCR en la trama {kz} ({zcr[kz]:.2f}, a los {(kz + m) * H / fs:.2f} s del clip).")
    fig, ejes = figura(3, 1, 14, 8.6, sharex=True)
    t = (np.arange(N) + m) * H / fs
    ax = ejes[0, 0]
    lim = np.abs(d[:, :N]).max()
    ax.imshow(d[:, :N], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=[t[0], t[-1] + H / fs, -0.5, 12.5])
    ax.set_yticks(range(0, 13, 2))
    ax.set_ylabel("coeficiente"); ax.set_title("Δ MFCC: cuánto cambia cada coeficiente de una trama a la siguiente", fontsize=9.5, loc="left")
    ax = ejes[1, 0]
    ax.plot(t, rms[:N], color=ROJO, lw=1.5); ax.set_ylabel("dBFS")
    ax.set_title("energía por trama: las sílabas «re-co-no-cer» son los montes", fontsize=9.5, loc="left")
    ax = ejes[2, 0]
    ax.plot(t, zcr[:N], color=VERDE, lw=1.5); ax.set_ylabel("fracción")
    ax.axvspan(t[max(0, kz - 3)], t[min(N - 1, kz + 3)], color=VERDE, alpha=0.15)
    ax.set_title("cruces por cero: alto en las consonantes de soplido", fontsize=9.5, loc="left")
    ax.set_xlabel("tiempo (s)")
    cerrar(fig, 13, "delta_energia_zcr", "Δ, energía y cruces por cero",
           "Δ = hacia dónde se mueve la boca. Energía = volumen de cada trama. Cruces por cero = cuánto «ruido» hay (alto en la «s»). "
           "Por trama quedan 13 MFCC + 13 Δ + energía + ZCR.")


def paso_14_tramos(E: dict) -> None:
    import soundfile as sf
    from src.voz.caracteristicas import estadisticas_tramos
    encabezado(14, "5 tramos: un vector del mismo largo para cualquier palabra")
    K = int(cfg["voz"]["tramos"])
    N = E["N"]
    wc = E["w_cmn"][:, :N]
    bordes = [(int(np.floor(i * N / K)), max(int(np.floor(i * N / K)) + 1, int(np.floor((i + 1) * N / K)))) for i in range(K)]
    propio = np.concatenate([np.concatenate([wc[:, a:b].mean(1), wc[:, a:b].std(1)]) for a, b in bordes])
    E["tramos_ok"] = bool(np.allclose(propio, estadisticas_tramos(wc, K)))
    z = E["npz_v"]
    otra = np.where((z["clase"] == z["clase"][E["fila_v"]]) & (z["copia"] == 0) & (z["origen"] == "notebook") &
                    (z["particion"] == "train"), z["duracion"][:, 0], -1)
    j = int(np.argmax(otra))
    y2, _ = sf.read(str(RAIZ / cfg["rutas"]["voz"] / "prepro" / str(z["archivo"][j])), dtype="float32")
    w2 = mfcc_a_mano(y2.reshape(-1), E)["mfcc"]
    w2 = w2[:, E["m"]:w2.shape[1] - E["m"]]
    w2 = w2 - w2.mean(1, keepdims=True)
    N2 = w2.shape[1]
    largos = {"mfcc": 130, "delta": 130, "energia": 10, "zcr": 10, "duracion": 1}
    X = z["X"][E["fila_v"]]
    manzanita("«Reconocer» dicho lento dura el doble que dicho rápido, y el K-NN necesita fichas del mismo largo para "
              "compararlas. Entonces corto cada palabra en 5 gajos iguales, sea la manzana grande o chica, y de cada gajo "
              "anoto el promedio y cuánto varía. Así «re-co-no-cer» rápido o lento da 5 gajos comparables y no pierdo el "
              "orden: el principio y el final de la palabra no se mezclan.")
    decir(f"Las palabras duran distinto: esta tiene {N} tramas y la más larga de train, {N2}. Para que todas den un vector "
          f"del mismo largo, divido las tramas en {K} tramos iguales y en cada uno guardo la media y el desvío de cada "
          f"valor. Con 13 MFCC son 2 × {K} × 13 = 130; lo mismo para Δ; 10 de energía; 10 de cruces por cero; y la "
          f"duración de la palabra. En total, 281 números. Los tramos conservan el orden temporal, que importa: el "
          f"principio de «reconocer» no se parece al final.")
    hablar(f"Tramo i = tramas [⌊i·N/{K}⌋, ⌊(i+1)·N/{K}⌋): con N = {N}: " + ", ".join(f"[{a}, {b})" for a, b in bordes) +
           f". Comparado con: {z['archivo'][j]} ({z['duracion'][j, 0]:.2f} s, {N2} tramas). Vector: 130 + 130 + 10 + 10 + 1 = 281.")
    fig = plt.figure(figsize=(15.5, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8])
    for col, (mat, nombre, n) in enumerate([(wc, E["archivo_v"].split("/")[-1], N), (w2, str(z["archivo"][j]).split("/")[-1], N2)]):
        ax = fig.add_subplot(gs[0, col])
        lim = np.abs(mat[1:]).max()
        ax.imshow(mat, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(1, K):
            ax.axvline(np.floor(i * n / K) - 0.5, color="black", lw=2)
        for i in range(K):
            ax.text((np.floor(i * n / K) + np.floor((i + 1) * n / K)) / 2 - 0.5, 11.8, f"tramo {i + 1}", ha="center",
                    fontsize=8.5, bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1.5))
        ax.set_xlabel("trama"); ax.set_ylabel("MFCC")
        ax.set_title(f"{nombre}: {n} tramas → {K} tramos", fontsize=9.5, loc="left")
    ax = fig.add_subplot(gs[1, :])
    tira_vector(ax, X, largos, f"el vector de 281 números de {E['archivo_v'].split('/')[-1]} (ya escalado)")
    cerrar(fig, 14, "tramos", "5 tramos: un vector del mismo largo para cualquier palabra",
           f"Una palabra corta y una larga se parten igual en {K} tramos; de cada tramo, media y desvío. Siempre salen 281 números "
           "y se conserva el orden de la palabra.")


def paso_15_escalador_v(E: dict) -> None:
    from src.core_ml.metricas import silhouette
    from src.voz.caracteristicas import BLOQUES, ajustar_escalador, largos_de_bloque
    encabezado(15, "Escalador y pesos de los bloques de voz")
    z = E["npz_v"]
    tr, va = z["particion"] == "train", z["particion"] == "val"
    com = np.isin(z["clase"], cfg["voz"]["comandos"])
    esc = ajustar_escalador({b: z[b][tr] for b in BLOQUES})
    E["esc_v"] = esc
    largos = largos_de_bloque(cfg)
    o, filas = 0, []
    for b in BLOQUES:
        d = largos[b]
        Zb = z["X"][:, o:o + d]; o += d
        filas.append((b, d, esc.pesos[b], contribucion(Zb[tr] / esc.pesos[b]), contribucion(Zb[tr]), contribucion(Zb[va]),
                      silhouette(Zb[va & com], z["clase"][va & com])))
    manzanita("Igual que con las huellas: todo a la misma unidad y que ningún bloque grite más que los otros. 130 columnas "
              "de MFCC contra 1 de duración: sin el peso, la duración no se escucharía.")
    decir(f"Igual que en huellas, estandarizo con train y peso cada bloque por uno sobre la raíz de sus columnas: así los "
          f"cinco bloques aportan exactamente 2 a la distancia en train. Después, en V4, con validación cruzada por persona "
          f"se probaron tres esquemas de pesos y ganó estandarizar cada columna sin peso por bloque, que le da más peso a "
          f"los MFCC y a Δ. O sea: V3 deja el vector armado y parejo, y V4 elige cómo pesarlo midiendo.")
    hablar("; ".join(f"{b}: {d} col., peso {p:.3f}, aporte sin peso {sp:.0f} → con peso {cp:.2f} en train ({cv:.2f} en val), "
                     f"silhouette de los comandos en val {s:.3f}" for b, d, p, sp, cp, cv, s in filas) +
           ". La silhouette por bloque es baja (los comandos se pisan bastante en cualquier bloque solo); juntos y con PCA "
           "separan más (paso 16).")
    fig, ejes = figura(1, 2, 14, 4.8)
    ax = ejes[0, 0]
    x = np.arange(len(filas))
    ax.bar(x - 0.2, [f[3] for f in filas], 0.4, color=GRIS, label="sin peso")
    ax.bar(x + 0.2, [f[4] for f in filas], 0.4, color=[BLOQUE_COLOR[f[0]] for f in filas], label="con peso 1/√columnas")
    for i, f in enumerate(filas):
        ax.text(i - 0.2, f[3] * 1.15, f"{f[3]:.0f}", ha="center", fontsize=9)
        ax.text(i + 0.2, f[4] * 1.15, f"{f[4]:.1f}", ha="center", fontsize=9)
    ax.set_yscale("log"); ax.set_xticks(x, [f"{f[0]}\n({f[1]})" for f in filas])
    ax.set_ylabel("aporte a la distancia² (log)"); ax.legend(fontsize=9)
    ax.set_title("cuánto pesa cada bloque (train)", fontsize=9.5)
    ax = ejes[0, 1]
    ax.bar(x, [f[6] for f in filas], color=[BLOQUE_COLOR[f[0]] for f in filas])
    for i, f in enumerate(filas):
        ax.text(i, f[6] + (0.005 if f[6] >= 0 else -0.015), f"{f[6]:.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="#999999", lw=0.6)
    vals = [f[6] for f in filas]
    ax.set_ylim(min(0, min(vals)) - 0.03, max(vals) + 0.03)
    ax.set_xticks(x, [f[0] for f in filas]); ax.set_ylabel("silhouette de los 3 comandos (val)")
    ax.set_title("cuánto separa cada bloque solo", fontsize=9.5)
    cerrar(fig, 15, "escalador_voz", "Escalador y pesos de los bloques de voz",
           "Cada bloque aporta 2 a la distancia en train. V4 después eligió, por validación cruzada, estandarizar cada columna "
           "sin peso por bloque (z_columna).")


def paso_16_pca_v(E: dict) -> None:
    from src.core_ml.metricas import silhouette
    from src.core_ml.pca import PCA
    encabezado(16, "PCA: los comandos en 2 dimensiones")
    z = E["npz_v"]
    X, part, clase, copia = z["X"], z["particion"], z["clase"], z["copia"]
    tr = part == "train"
    com = np.isin(clase, cfg["voz"]["comandos"])
    p2 = PCA(2).ajustar(X[tr])
    p20 = PCA(20).ajustar(X[tr])
    P = p2.transformar(X)
    k0tr = tr & (copia == 0) & com
    va = (part == "val") & com
    s = {nom: (silhouette(X[m], clase[m]), silhouette(p20.transformar(X[m]), clase[m])) for nom, m in
         [("train k0", k0tr), ("val", va)]}
    v1, v2 = p2.proporcion_varianza
    manzanita("Acá las nubes se pisan más que con las huellas, y está bien: la foto en 2D muestra solo 2 de 281 "
              "direcciones. Además el K-NN no necesita grupos redondos y lejanos, como las huellas: le alcanza con que los "
              "vecinos más cercanos de cada palabra sean de la misma clase, como reconocer una fruta mirando las tres "
              "frutas más parecidas que ya conocés.")
    decir(f"Con PCA ajustado con train, en 2D se ven los tres comandos en zonas distintas pero que se pisan: las dos "
          f"primeras componentes explican solo el {100 * (v1 + v2):.0f} % de la varianza. La silhouette es baja, "
          f"{s['train k0'][0]:.2f} en train con las 281 columnas, y sube a {s['train k0'][1]:.2f} con 20 componentes. Igual "
          f"alcanza: el K-NN mira vecinos cercanos, no grupos enteros, y en la validación cruzada con personas que no vio "
          f"acertó más del 90 % de los comandos.")
    hablar(f"PCA propio con train: PC1 {100 * v1:.1f} %, PC2 {100 * v2:.1f} %; con 20 componentes, "
           f"{100 * p20.proporcion_varianza.sum():.1f} %. Silhouette de los 3 comandos (281 → PCA 20): train k0 "
           f"{s['train k0'][0]:.3f} → {s['train k0'][1]:.3f}; val {s['val'][0]:.3f} → {s['val'][1]:.3f}.")
    fig, ejes = figura(1, 2, 15, 6.4)
    for ax, mascara, tit in [(ejes[0, 0], tr & (copia == 0), "train (originales k0)"), (ejes[0, 1], part == "val", "val")]:
        for cl in ["otro"] + cfg["voz"]["comandos"]:
            m = mascara & (clase == cl)
            ax.scatter(P[m, 0], P[m, 1], s=14 if cl == "otro" else 18, c=CLASE_COLOR[cl], alpha=0.35 if cl == "otro" else 0.75,
                       label=cl)
        ax.set_xlabel(f"PC1 ({100 * v1:.1f} %)"); ax.set_ylabel(f"PC2 ({100 * v2:.1f} %)"); ax.legend(fontsize=8.5)
        ax.set_title(tit, fontsize=10)
    i = E["fila_v"]
    ejes[0, 0].plot(P[i, 0], P[i, 1], "*", ms=20, color="black", mfc="yellow")
    cerrar(fig, 16, "pca_voz", "PCA: los comandos en 2 dimensiones",
           f"En 2D ({100 * (v1 + v2):.0f} % de la varianza) los comandos ocupan zonas distintas pero se pisan. Al K-NN le alcanza: decide por "
           "los vecinos más cercanos, en 20 dimensiones. La estrella es la palabra que seguimos.")


# ============================================================================= ANEXO · DESEMPATE
def paso_17_desempate_knn(E: dict) -> None:
    from src.core_ml.knn import KNN
    from src.huellas.clasificacion import cargar_modelo as cargar_h
    from src.voz.clasificacion import cargar_modelo
    encabezado(17, "Anexo · criterio de desempate en el voto del K-NN")
    modelo = cargar_modelo(RAIZ / cfg["rutas"]["modelos"] / "modelo_voz.npz")
    z = E["npz_v"]
    q = z["particion"] != "train"
    Zq = modelo.transformar(z["X"][q], cfg)
    knn = KNN(modelo.k, modelo.eps).ajustar(modelo.entrenamiento, modelo.etiquetas)
    d, idx = knn.vecinos(Zq)
    w = 1.0 / (d + modelo.eps)
    et = modelo.etiquetas[idx]
    margen, empates, propias = [], 0, []
    for i in range(len(Zq)):
        votos = {c: w[i][et[i] == c].sum() for c in np.unique(et[i])}
        dmin = {c: d[i][et[i] == c].min() for c in votos}
        orden = sorted(votos, key=lambda c: -votos[c])
        vmax = votos[orden[0]]
        cand = [c for c in sorted(votos) if abs(votos[c] - vmax) <= 1e-12]
        dm = min(dmin[c] for c in cand)
        propias.append([c for c in cand if dmin[c] == dm][0])
        if len(orden) > 1:
            margen.append((votos[orden[0]] - votos[orden[1]]) / w[i].sum())
            empates += votos[orden[0]] - votos[orden[1]] <= 1e-12
        else:
            margen.append(1.0)
    margen = np.array(margen)
    E["desempate_ok"] = bool(np.array_equal(np.array(propias), knn.predecir(Zq)))
    mh = cargar_h(RAIZ / cfg["rutas"]["modelos"] / "modelo_huellas.npz")
    Xh = E["npz_h"]["X"]
    Dh = np.sqrt(((Xh[:, None, :] - mh.centroides[None]) ** 2).sum(2)) / mh.dispersion[None]
    sh = np.sort(Dh, axis=1)
    gap_h = float((sh[:, 1] - sh[:, 0]).min())
    manzanita("Cada vecino vota, y vota más fuerte cuanto más cerca está (su voto vale 1/distancia). Si dos clases empatan "
              "en votos, gana la que tiene al vecino más cercano: el testigo que estaba más pegado a lo que pasó es el más "
              "confiable. Y si todavía empatan, gana la primera en orden alfabético, para que con los mismos datos la "
              "respuesta sea siempre la misma.")
    decir(f"El criterio de desempate está en el K-NN de V4. Cada uno de los {modelo.k} vecinos vota por su clase con peso "
          f"uno sobre su distancia. Si dos clases empatan en votos, gana la que tiene el vecino más cercano; si siguen "
          f"empatadas, la primera en orden alfabético; y si dos vecinos están a la misma distancia, entra el de índice menor. "
          f"En el ejemplo, la clase A tiene vecinos a 1 y a 3, y la B dos a 1,5: los dos votos dan 4/3, empatan, y gana A "
          f"porque su vecino más cercano está a 1. Con los datos reales, en {len(Zq)} consultas de validación y test, el "
          f"empate exacto no pasó nunca: el margen más chico fue {100 * margen.min():.1f} % del voto. La regla está para que "
          f"el resultado sea siempre el mismo.")
    hablar(f"Regla del contrato de V4: voto(c) = Σ 1/(d_j + ε) de sus vecinos, ε = {modelo.eps:g}; empate si los votos "
           f"difieren en ≤ 1e-12 → menor distancia mínima → primera clase en orden alfabético; distancias iguales → índice "
           f"de train menor (argsort estable). Datos reales ({len(Zq)} clips válidos de val y test, K = {modelo.k}): "
           f"{empates} empates; margen mínimo {100 * margen.min():.2f} %, {int((margen < 0.05).sum())} consultas con margen "
           f"< 5 %. En huellas no hay votación: gana el centroide de menor distancia normalizada (argmin; un empate exacto se "
           f"resolvería por la identidad menor). Ahí la diferencia más chica entre el primero y el segundo fue {gap_h:.2f}.")
    fig, ejes = figura(1, 2, 15, 5.2, gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = ejes[0, 0]
    ax.axhline(0, color="#cccccc", lw=1)
    ax.plot([0], [0], "*", ms=22, color="black", mfc="yellow")
    ax.text(0, -0.3, "consulta", ha="center", va="top", fontsize=9)
    for x, cl, col in [(-1.0, "A", AZUL), (3.0, "A", AZUL), (1.5, "B", ROJO), (-1.5, "B", ROJO)]:
        ax.plot([x], [0], "o", ms=16, color=col)
        ax.text(x, 0, cl, ha="center", va="center", color="white", fontweight="bold")
        arriba = cl == "B"
        ax.text(x, 0.3 if arriba else -0.3, f"d = {abs(x):.1f}\nvoto 1/{abs(x):.1f}", ha="center",
                va="bottom" if arriba else "top", fontsize=8.5, color=col)
    ax.text(-3.3, 1.0, "voto(A) = 1/1 + 1/3 = 4/3\nvoto(B) = 1/1,5 + 1/1,5 = 4/3  → empate\n"
                        "desempate: vecino más cercano de A a 1,0 < de B a 1,5 → gana A", fontsize=9.5, va="top")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-1.0, 1.15); ax.axis("off")
    ax.set_title("ejemplo con K = 4 (el del contrato de V4)", fontsize=10)
    ax = ejes[0, 1]
    ax.hist(100 * margen, bins=np.linspace(0, 100, 41), color=AZUL)
    ax.axvline(0, color=ROJO, lw=2)
    ax.set_xlabel("margen entre la 1.ª y la 2.ª clase (% del voto total)"); ax.set_ylabel("consultas")
    ax.set_title(f"datos reales: {len(Zq)} consultas de val y test, {empates} empates (margen mínimo {100 * margen.min():.1f} %)",
                 fontsize=9.5)
    cerrar(fig, 17, "desempate_knn", "Anexo · criterio de desempate en el voto del K-NN",
           "Voto = Σ 1/distancia. Empate → gana la clase con el vecino más cercano → si persiste, la primera alfabética. "
           "Con los datos reales nunca hubo empate exacto: la regla garantiza que la respuesta sea determinista.")


def paso_18_desempate_eleccion(E: dict) -> None:
    encabezado(18, "Anexo · el empate al elegir el modelo de voz")
    filas = [(411, 224, 190, 221, "z_columna", "20", 5), (411, 221, 195, 216, "sin_dur_zcr", "30", 5),
             (408, 221, 187, 221, "sin_dur_zcr", "30", 3), (408, 221, 187, 221, "z_columna", "20", 3),
             (407, 225, 192, 215, "z_columna", "20", 7)]
    manzanita("Como en un campeonato: si dos equipos terminan con los mismos puntos, no se tira una moneda, se mira la "
              "diferencia de goles, que está en el reglamento desde antes de empezar. Acá pasó de verdad: dos "
              "configuraciones sacaron 411 puntos y desempató el segundo criterio, fijado en el contrato antes de medir.")
    decir("El otro desempate que hubo fue al elegir los hiperparámetros del K-NN. Con validación cruzada por persona "
          "puntué 36 combinaciones con la suma de dos conteos: comandos que pasan el umbral de rechazo y comandos que pasan "
          "la confianza mínima. Las dos mejores empataron exacto en 411. El contrato ya fijaba qué mirar después: los "
          "comandos acertados sin rechazo, 224 contra 221. Ganó z_columna con 20 componentes y K = 5. Si después siguieran "
          "empatadas, deciden el esquema, la cantidad de componentes y el K menor, en ese orden.")
    hablar("Valores de la validación cruzada (contrato V4, sección 8; 10 personas, 231 comandos y 78 «otro»). Orden del "
           "criterio (elegir_mejor): (1) mayor n_tau5 + n_c60; (2) mayor n_4cl; (3) esquema en el orden de la lista; "
           "(4) componentes en el orden [20, 30, sin PCA]; (5) menor K. Si se usara solo n_tau5 habría ganado sin_dur_zcr "
           "con 30 y K = 5 (195): el empate fue muy justo, y la suma se eligió antes porque n_tau5 solo depende de una "
           "distancia puntual y es más ruidoso.")
    fig, ejes = figura(1, 1, 13, 4.2)
    ax = ejes[0, 0]
    ax.axis("off")
    tabla = ax.table(cellText=[[str(f[0]), str(f[1]), str(f[2]), str(f[3]), f[4], f[5], str(f[6])] for f in filas],
                     colLabels=["suma (1)", "n_4cl (2)", "n_tau5", "n_c60", "esquema", "PCA", "K"], loc="center",
                     cellLoc="center")
    tabla.auto_set_font_size(False); tabla.set_fontsize(11); tabla.scale(1, 2.2)
    for (r, c), celda in tabla.get_celld().items():
        if r == 1:
            celda.set_facecolor("#d8f0dc")
        if r == 2:
            celda.set_facecolor("#fbe3d6")
        if r in (1, 2) and c in (0, 1):
            celda.set_text_props(fontweight="bold")
    cerrar(fig, 18, "desempate_eleccion", "Anexo · el empate al elegir el modelo de voz",
           "Las dos mejores combinaciones empataron en 411 (criterio 1) y desempató n_4cl, los comandos acertados sin rechazo (224 "
           "contra 221). El orden de los criterios estaba fijado en el contrato antes de medir.")


# ============================================================================= guion y verificación
PREGUNTAS = [
    ("¿Por qué no comparan las imágenes directamente?",
     "Porque dos tomas del mismo dedo nunca coinciden píxel a píxel (rotación, presión, ruido, cortes). Las características "
     "describen lo que se mantiene entre tomas: hacia dónde van las crestas, su textura y dónde están las minucias."),
    ("¿Por qué el ángulo doble en la orientación?",
     "Porque una cresta no tiene sentido: 5° y 175° son casi la misma raya. Con cos 2θ y sen 2θ quedan juntas y el promedio "
     "da bien; con el ángulo común daría 90°."),
    ("¿Qué es el FingerCode?",
     "Describir la huella por la respuesta de un banco de filtros de Gabor en zonas alrededor del núcleo (Jain y otros, "
     "2000). Lo uso simplificado: media del valor absoluto de la respuesta por zona."),
    ("¿Por qué descartan minucias cerca del borde?",
     "Porque ahí la cresta se corta por el borde de la imagen o por el corte, no porque termine. Si no, una huella cortada "
     "tendría muchas terminaciones falsas. Con el filtro, la densidad con corte y sin corte da casi igual (1,03 a 1,06)."),
    ("¿Por qué densidades y no cantidad de minucias?",
     "Para que una huella con menos superficie (cortada) no parezca tener menos minucias."),
    ("¿Por qué dividir cada bloque por √columnas?",
     "Porque la distancia euclídea suma columnas: el bloque de 300 columnas de Gabor decidiría solo. Con el peso, cada bloque "
     "aporta lo mismo (2 en train)."),
    ("¿Por qué sacaron los momentos de Hu?",
     "Se midió la separación de cada bloque: Hu separaba 0,13 y bajaba el total de 0,67 a 0,51."),
    ("¿Qué son los MFCC?",
     "El espectro de cada trama de 25 ms pasado por 40 filtros mel (como oye el oído), el logaritmo de esas energías y la "
     "transformada coseno; me quedo con los 13 primeros, que describen la forma general del espectro (la boca)."),
    ("¿Para qué la CMN?",
     "Un micrófono o una sala distintos son un filtro fijo, que en el logaritmo es una suma constante. Restar la media de "
     "cada coeficiente en la palabra la cancela."),
    ("¿Por qué 5 tramos con media y desvío?",
     "Para que cualquier palabra, corta o larga, dé un vector de 281 números, y conservar el orden temporal (principio, medio "
     "y final de la palabra)."),
    ("¿Por qué en voz la silhouette es tan baja?",
     "Porque mide grupos enteros y los comandos se pisan en el espacio global. El K-NN decide por los vecinos más cercanos: "
     "con validación cruzada por persona acertó más del 90 % de los comandos. Con PCA 20 la separación además sube."),
    ("¿Hay fuga de datos en el escalador?",
     "Respecto de val y test, no: la media y el desvío de cada columna salen solo de train y se aplican igual a val y test. "
     "Dentro de la validación cruzada de V4 hay una fuga chica, aceptada y anotada en el contrato: el escalador se ajustó "
     "con todo train, incluida la persona que sale del pliegue."),
    ("¿Y si hay empate en el K-NN?",
     "Gana la clase con el vecino más cercano y, si sigue el empate, la primera alfabética. Con los datos reales no pasó "
     "nunca, pero así el resultado es siempre el mismo."),
]


def escribir_guion(E: dict) -> None:
    partes = [f"# H3 + V3 paso a paso · {E['huella']} y {E['toma']}\n",
              "Generado por `scripts/h3v3_paso_a_paso.py`, con los números de esta corrida. Por figura: la explicación "
              "con manzanitas, qué decir y el detalle.\n", "## Versión corta: una frase por figura\n"]
    partes += [f"{p['n']}. **{p['titulo']}** ({p['figura']}). {p['pie']}" for p in GUION if p["figura"]]
    partes.append("")
    for p in GUION:
        partes.append(f"## Paso {p['n']} · {p['titulo']}\n")
        if p["figura"]:
            partes.append(f"![]({p['figura']})\n")
        if p["manzanita"]:
            partes.append(f"**Con manzanitas.** {p['manzanita']}\n")
        if p["decir"]:
            partes.append(f"**Qué decir.** «{p['decir']}»\n")
        if p["detalle"]:
            partes.append("**Detalle.** " + "\n\n".join(p["detalle"]) + "\n")
    partes.append("## Preguntas probables\n")
    partes += [f"**{q}** {es(r)}\n" for q, r in PREGUNTAS]
    (SALIDA / "GUION.md").write_text("\n".join(partes), encoding="utf-8")
    print(f"   [guion guardado: {SALIDA / 'GUION.md'}]")


def verificar(E: dict) -> bool:
    from src.huellas.caracteristicas import BLOQUES as BH, extraer_minucias
    from src.voz.caracteristicas import BLOQUES as BV
    print("\n" + "=" * 100 + "\nVerificación contra src/ y los .npz\n" + "=" * 100)
    oks = []

    def informe(nombre: str, ok: bool, detalle: str = "") -> None:
        oks.append(bool(ok))
        print(f"   {'✔' if ok else '✘'} {nombre}{(': ' + detalle) if detalle else ''}")

    b = E["bloques_h"]
    informe("orientación rehecha = src", np.allclose(E["orient_propio"], b["orientacion"], equal_nan=True, atol=1e-12))
    informe("Gabor rehecho = src", np.allclose(E["gabor_propio"], b["gabor"], equal_nan=True, rtol=1e-9, atol=1e-12))
    m = extraer_minucias(E["h"], cfg)
    informe("minucias rehechas = extraer_minucias",
            sorted((float(j), float(i)) for i, j in E["T_propio"]) == sorted(map(tuple, m.terminaciones.tolist())) and
            sorted((float(j), float(i)) for i, j in E["B_propio"]) == sorted(map(tuple, m.bifurcaciones.tolist())))
    informe("bloque de minucias rehecho = src", np.allclose(E["minucias_propio"], b["minucias"], equal_nan=True, atol=1e-12))
    i = E["fila_h"]
    for k in BH:
        dif = float(np.nanmax(np.abs(b[k] - E["npz_h"][k][i])))
        informe(f"huella, bloque {k} = data/huellas/caracteristicas.npz", dif <= 1e-9 * max(1.0, float(np.nanmax(np.abs(b[k])))),
                f"máx |dif| = {dif:.1e}")
    informe("huella, vector escalado = X del .npz",
            np.allclose(E["esc_h"].transformar({k: E["npz_h"][k][i] for k in BH}), E["npz_h"]["X"][i], atol=1e-9))
    from src.voz.caracteristicas import caracteristicas
    import librosa
    S = librosa.feature.mfcc(y=E["y"], sr=E["fs"], n_mfcc=13, n_fft=E["NF"], win_length=E["W"], hop_length=E["H"],
                             n_mels=int(cfg["voz"]["mfcc"]["n_mels"]), fmin=float(cfg["voz"]["pasabanda_hz"][0]),
                             fmax=float(cfg["voz"]["mfcc"]["fmax"]), center=False)
    dif = float(np.abs(S - E["MM"]["mfcc"]).max())
    informe("MFCC a mano = librosa.feature.mfcc", dif < 1e-3, f"máx |dif| = {dif:.1e} (librosa calcula en float32)")
    informe("tramos rehechos = estadisticas_tramos", E["tramos_ok"])
    bv = caracteristicas(E["y"], cfg)
    j = E["fila_v"]
    for k in BV:
        dif = float(np.abs(bv[k] - E["npz_v"][k][j]).max())
        informe(f"voz, bloque {k} = data/voz/prepro/caracteristicas.npz", dif <= 1e-4, f"máx |dif| = {dif:.1e}")
    informe("voz, vector escalado = X del .npz", np.allclose(E["esc_v"].transformar({k: E["npz_v"][k][j] for k in BV}),
                                                              E["npz_v"]["X"][j], atol=1e-12))
    informe("voto y desempate rehechos = KNN.predecir (val y test)", E["desempate_ok"])
    return all(oks)


# ============================================================================= main
def main() -> int:
    global RAIZ, SALIDA, MOSTRAR, plt, cfg
    ap = argparse.ArgumentParser(description="H3 + V3 paso a paso: de la huella y de la palabra a un vector de números.")
    ap.add_argument("--huella", default="train/id1_v011.png", help="huella del dataset de H1 (default train/id1_v011.png)")
    ap.add_argument("--toma", default="p01_reconocer_01", help="toma de voz, sin extensión (default p01_reconocer_01)")
    ap.add_argument("--mostrar", action="store_true", help="abre cada figura y espera que la cierres")
    ap.add_argument("--verificar", action="store_true", help="compara con src/ y con los .npz")
    ap.add_argument("--salida", type=Path, help="carpeta de salida (default data/h3v3_paso_a_paso)")
    ap.add_argument("--raiz", type=Path, default=RAIZ, help="raíz del repo")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import matplotlib
    if not args.mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    plt, MOSTRAR, RAIZ = _plt, args.mostrar, args.raiz
    sys.path.insert(0, str(RAIZ))
    from src.config import cargar_config
    cfg = cargar_config()
    SALIDA = args.salida or (RAIZ / "data" / "h3v3_paso_a_paso")
    SALIDA.mkdir(parents=True, exist_ok=True)
    E: dict = dict(huella=args.huella, toma=args.toma)
    print(f"H3 + V3 paso a paso · {args.huella} y {args.toma}\nFiguras en: {SALIDA}")
    cargar_huella(E)
    if E["fila_h"] is None:
        print(f"{args.huella} no está en data/huellas/caracteristicas.npz.")
        return 2
    for paso in (paso_1_ficha, paso_2_orientacion, paso_3_gabor_filtros, paso_4_gabor_zonas, paso_5_crossing,
                 paso_6_limpieza, paso_7_densidades, paso_8_escalador_h, paso_9_pca_h):
        paso(E)
    cargar_voz(E)
    for paso in (paso_10_tramas, paso_11_mfcc_trama, paso_12_cmn, paso_13_delta, paso_14_tramos, paso_15_escalador_v,
                 paso_16_pca_v, paso_17_desempate_knn, paso_18_desempate_eleccion):
        paso(E)
    escribir_guion(E)
    print("\nListo. Mirá las figuras 01 a 18 en orden, o GUION.md, que las tiene con la explicación de cada una.")
    ok = True
    if args.verificar:
        ok = verificar(E)
        print("\n" + ("✔ Lo rehecho coincide con src/ y con los vectores guardados." if ok else "✘ HAY DIFERENCIAS."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
