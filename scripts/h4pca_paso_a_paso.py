"""H4 + PCA paso a paso: clusterización de las huellas con K-means y los gráficos PCA, con una figura por paso.

No es parte del pipeline ni tiene tests: es una herramienta para entender (y explicar) el hito
"Clusterización y gráficos PCA". Rehace a mano el K-means propio (k-means++ y Lloyd, con los mismos
sorteos), la correspondencia cluster → identidad, la distancia normalizada, el umbral τ* y el PCA, y
con --verificar lo compara contra src/core_ml, src/huellas/clasificacion.py, models/modelo_huellas.npz
y (si está instalado) scikit-learn.

Qué hace, en orden (el número es el de la figura que guarda):
  Parte A · clusterización de las huellas (H4)
    01 el problema: 72 huellas sin etiqueta       05 cada cluster recibe un nombre (pureza)
    02 k-means++: elegir los 3 centros iniciales  06 distancia normalizada por la dispersión
    03 Lloyd: asignar y mover hasta que no cambie 07 el umbral τ* con validación
    04 15 arranques y por qué k = 3               08 el examen: test
  Parte B · gráficos PCA
    09 qué hace el PCA (ejemplo en 2D)            12 de qué está hecha cada componente
    10 cuánta varianza explica cada componente    13 PCA de la voz en 2D y 3D
    11 PCA de las huellas en 2D y 3D              14 voz: el PCA con los pesos de V3 y con los de V4
  y GUION.md: por figura, la explicación con manzanitas, qué decir y el detalle con los números de
  esta corrida, más preguntas probables.

Uso, desde la raíz del repo:
    venv\\Scripts\\python scripts\\h4pca_paso_a_paso.py
    venv\\Scripts\\python scripts\\h4pca_paso_a_paso.py --mostrar      (abre cada figura y espera que la cierres)
    venv\\Scripts\\python scripts\\h4pca_paso_a_paso.py --verificar    (compara con src/, el modelo y scikit-learn)
Las figuras quedan en data/h4pca_paso_a_paso/ (data/ no va a Git). Lee data/huellas/caracteristicas.npz,
data/huellas/metadatos.csv, data/voz/prepro/caracteristicas.npz y models/modelo_huellas.npz.
"""

from __future__ import annotations

import argparse
import csv
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
GUION: list[dict] = []

ROJO, AZUL, VERDE, NARANJA, GRIS, VIOLETA = "#c8262a", "#1b5fc4", "#14873f", "#e69f00", "#888888", "#7a5195"
ID_COLOR = {1: AZUL, 2: VERDE, 3: NARANJA}
CLASE_COLOR = {"reconocer": AZUL, "identificar": VERDE, "salir": ROJO, "otro": GRIS}
BLOQUE_COLOR = {"orientacion": AZUL, "gabor": NARANJA, "minucias": VERDE,
                "mfcc": AZUL, "delta": VIOLETA, "energia": ROJO, "zcr": VERDE, "duracion": NARANJA}
BLOQUE_NOMBRE = {"orientacion": "orientación", "gabor": "Gabor", "minucias": "minucias", "mfcc": "MFCC",
                 "delta": "Δ MFCC", "energia": "energía", "zcr": "cruces por cero", "duracion": "duración"}


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
    GUION[-1]["detalle"].append(es(texto))
    _imprimir("", es(texto))


def manzanita(texto: str) -> None:
    GUION[-1]["manzanita"] = es(texto)
    _imprimir("[con manzanitas] ", es(texto))


def decir(texto: str) -> None:
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
    coma_en_ejes(fig)
    ruta = SALIDA / f"{n:02d}_{nombre}.png"
    fig.savefig(ruta, dpi=110)
    GUION[-1].update(figura=ruta.name, pie=es(pie))
    print(f"   [figura guardada: {ruta}]")
    if MOSTRAR:
        plt.show()
    plt.close(fig)


def coma_en_ejes(fig) -> None:
    """Coma decimal también en los números de los ejes: fija los rótulos ya dibujados, sin mover los límites."""
    fig.canvas.draw()
    for ax in fig.axes:
        for nombre in ("x", "y", "z"):
            eje = getattr(ax, f"{nombre}axis", None)
            if eje is None:
                continue
            rotulos = [t.get_text() for t in eje.get_majorticklabels()]
            nuevos = [es(t) for t in rotulos]
            if nuevos == rotulos:
                continue
            lim = getattr(ax, f"get_{nombre}lim")()
            eje.set_ticks(eje.get_majorticklocs(), nuevos)
            getattr(ax, f"set_{nombre}lim")(lim)


def dist2(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Distancias² exactas (n, k), como src/core_ml/kmeans.py."""
    D2 = np.empty((X.shape[0], C.shape[0]))
    for j in range(C.shape[0]):
        dif = X - C[j]
        D2[:, j] = np.einsum("ij,ij->i", dif, dif)
    return D2


def pca_propio(X: np.ndarray, k: int | None = None) -> dict:
    """PCA a mano: Σ = Xcᵀ·Xc/(n − 1), autovectores con eigh, signo con la entrada mayor positiva."""
    media = X.mean(axis=0)
    Xc = X - media
    S = Xc.T @ Xc / (X.shape[0] - 1)
    val, vec = np.linalg.eigh(S)
    orden = np.argsort(val)[::-1]
    val, vec = np.maximum(val[orden], 0), vec[:, orden]
    k = k or min(X.shape[0] - 1, X.shape[1])
    V = vec[:, :k].T
    signo = np.sign(V[np.arange(k), np.argmax(np.abs(V), axis=1)])
    V = V * signo[:, None]
    return dict(media=media, componentes=V, varianza=val[:k], proporcion=val[:k] / np.trace(S), traza=float(np.trace(S)))


def proyectar(P: dict, X: np.ndarray) -> np.ndarray:
    return (X - P["media"]) @ P["componentes"].T


# ============================================================================= datos
def cargar(E: dict) -> None:
    z = np.load(RAIZ / cfg["rutas"]["huellas"] / "caracteristicas.npz", allow_pickle=False)
    E["h"] = {k: z[k] for k in z.files}
    with open(RAIZ / cfg["rutas"]["huellas"] / "metadatos.csv", encoding="utf-8") as f:
        meta = {r["archivo"]: r for r in csv.DictReader(f)}
    E["corte"] = np.array([float(meta[a]["corte_fraccion"]) > 0 for a in E["h"]["archivo"]])
    v = np.load(RAIZ / cfg["rutas"]["voz"] / "prepro" / "caracteristicas.npz", allow_pickle=False)
    E["v"] = {k: v[k] for k in v.files}
    h = E["h"]
    E["tr"] = h["particion"] == "train"
    E["Xtr"], E["ytr"] = h["X"][E["tr"]], h["identidad"][E["tr"]]
    E["vis"] = pca_propio(E["Xtr"], 3)                 # para dibujar las huellas en 2D
    E["P_tr"] = proyectar(E["vis"], E["Xtr"])


def puntos_2d(ax, P2, colores, **kw) -> None:
    ax.scatter(P2[:, 0], P2[:, 1], c=colores, s=kw.pop("s", 28), **kw)


def ejes_pca(ax, P: dict) -> None:
    ax.set_xlabel(f"PC1 ({100 * P['proporcion'][0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({100 * P['proporcion'][1]:.1f} %)")


# ============================================================================= PARTE A · K-MEANS
def paso_1_problema(E: dict) -> None:
    encabezado(1, "El problema: agrupar huellas sin saber de quién son")
    P2, y = E["P_tr"], E["ytr"]
    n, d = E["Xtr"].shape
    manzanita(f"Te dan un cajón con {n} frutas mezcladas, sin etiqueta, y te piden armar 3 pilas de frutas parecidas. No "
              "sabés que son manzanas, peras y naranjas: solo las juntás por parecido. Recién al final, mirando qué quedó "
              "en cada pila, les ponés nombre. Eso es el K-means: aprendizaje no supervisado.")
    decir(f"La consigna pide agrupar con K-means, con k igual a 3. Tengo las {n} huellas de train, cada una como un vector "
          f"de {d} números de H3. El K-means no ve las etiquetas: a la izquierda está lo que ve él, puntos sin nombre, y a "
          f"la derecha lo que sabemos nosotros. Las identidades se usan recién al final, para ponerle nombre a cada grupo. "
          f"El dibujo es una proyección en 2D; el K-means trabaja en las {d} dimensiones.")
    hablar(f"{n} vectores de train ({int((y == 1).sum())}, {int((y == 2).sum())} y {int((y == 3).sum())} por identidad), "
           f"d = {d}. Para dibujar se usa el PCA de train (paso 9 en adelante).")
    fig, ejes = figura(1, 2, 13, 5.6)
    puntos_2d(ejes[0, 0], P2, GRIS)
    ejes[0, 0].set_title("lo que ve el K-means: puntos sin etiqueta", fontsize=10)
    for k in (1, 2, 3):
        m = y == k
        ejes[0, 1].scatter(P2[m, 0], P2[m, 1], c=ID_COLOR[k], s=28, label=f"identidad {k}")
    ejes[0, 1].legend(fontsize=9); ejes[0, 1].set_title("lo que sabemos nosotros (solo para nombrar al final)", fontsize=10)
    for ax in ejes[0]:
        ejes_pca(ax, E["vis"])
    cerrar(fig, 1, "problema", "El problema: agrupar huellas sin saber de quién son",
           f"K-means es no supervisado: agrupa los {n} vectores por parecido, sin mirar la identidad. El dibujo es una "
           f"proyección PCA; el algoritmo trabaja en {d} dimensiones.")


def kmeanspp(X: np.ndarray, k: int, rng: np.random.Generator, registro: list | None = None) -> np.ndarray:
    """k-means++ "greedy" como src/core_ml/kmeans.py (mismos sorteos con el mismo generador)."""
    n = X.shape[0]
    ensayos = 2 + int(math.log(k))
    C = np.empty((k, X.shape[1]))
    i0 = int(rng.integers(n))
    C[0] = X[i0]
    d2 = dist2(X, C[:1])[:, 0]
    if registro is not None:
        registro.append(dict(elegido=i0, candidatas=[i0], d2=None))
    for c in range(1, k):
        acumulada = np.cumsum(d2)
        sorteo = rng.random(ensayos) * acumulada[-1]
        candidatas = np.minimum(np.searchsorted(acumulada, sorteo, side="right"), n - 1)
        d2_cand = np.minimum(d2[None, :], dist2(X, X[candidatas]).T)
        mejor = int(np.argmin(d2_cand.sum(axis=1)))
        if registro is not None:
            registro.append(dict(elegido=int(candidatas[mejor]), candidatas=[int(i) for i in candidatas], d2=d2.copy(),
                                 sumas=d2_cand.sum(axis=1)))
        C[c] = X[candidatas[mejor]]
        d2 = d2_cand[mejor]
    return C


def lloyd(X: np.ndarray, C: np.ndarray, tol_abs: float, max_iter: int, registro: list | None = None):
    k = C.shape[0]
    antes = None
    it = 0
    for it in range(1, max_iter + 1):
        et = np.argmin(dist2(X, C), axis=1)
        nuevos = C.copy()
        for j in range(k):
            if (et == j).any():
                nuevos[j] = X[et == j].mean(axis=0)
        mov = float(np.sum((nuevos - C) ** 2))
        if registro is not None:
            registro.append(dict(C=C.copy(), etiquetas=et.copy(), J=float(dist2(X, C)[np.arange(len(X)), et].sum()),
                                 nuevos=nuevos.copy(), mov=mov))
        C = nuevos
        if (antes is not None and np.array_equal(et, antes)) or mov <= tol_abs:
            break
        antes = et
    D2 = dist2(X, C)
    et = np.argmin(D2, axis=1)
    return C, et, float(D2[np.arange(len(X)), et].sum()), it


def paso_2_kmeanspp(E: dict) -> None:
    encabezado(2, "k-means++: elegir los 3 centros de arranque")
    X, P2 = E["Xtr"], E["P_tr"]
    kc = cfg["huellas"]["kmeans"]
    rng = np.random.default_rng(int(cfg["semilla_global"]))
    reg: list = []
    C0 = kmeanspp(X, int(kc["k"]), rng, reg)
    E["C0"], E["reg_pp"] = C0, reg
    manzanita("Para arrancar necesito 3 representantes. El primero lo elijo al azar. El segundo también es un sorteo, pero "
              "le doy más chances a las frutas que están lejos del primero (la chance es proporcional a la distancia al "
              "cuadrado): así es casi seguro que caiga en otra pila. Sorteo 3 candidatos y me quedo con el que deja a todas "
              "las frutas más cerca de algún representante. Lo mismo para el tercero.")
    decir(f"El K-means arranca con 3 centros. Los elijo con k-means++: el primero es una huella al azar, y cada uno de los "
          f"siguientes se sortea con probabilidad proporcional a la distancia al cuadrado al centro más cercano ya elegido. "
          f"Así los centros arrancan lejos entre sí. En cada paso sorteo {2 + int(math.log(3))} candidatos y me quedo con el "
          f"que más baja la suma de distancias. En este arranque los tres centros cayeron uno en cada identidad.")
    hablar(f"Semilla {cfg['semilla_global']}, un solo generador para los {kc['n_init']} arranques (como src/core_ml/kmeans.py). "
           f"Primer arranque: centro 1 = fila {reg[0]['elegido']} (identidad {E['ytr'][reg[0]['elegido']]}); "
           + "; ".join(f"centro {i + 1}: candidatas {r['candidatas']} → fila {r['elegido']} (identidad {E['ytr'][r['elegido']]})"
                       for i, r in enumerate(reg) if i > 0) + ".")
    y = E["ytr"]
    chances = []
    fig, ejes = figura(1, 3, 15.5, 5.2)
    for paso, ax in enumerate(ejes[0]):
        prob = np.full(len(X), 1 / len(X)) if paso == 0 else reg[paso]["d2"] / reg[paso]["d2"].sum()
        por_grupo = {k: float(prob[y == k].sum()) for k in (1, 2, 3)}
        chances.append(por_grupo)
        if paso == 0:
            puntos_2d(ax, P2, GRIS, alpha=0.6)
            ax.set_title("centro 1: al azar, todas con la misma chance", fontsize=10)
        else:
            ax.scatter(P2[:, 0], P2[:, 1], c=prob, cmap="viridis", s=26, vmin=0)
            for c in reg[paso]["candidatas"]:
                ax.plot(P2[c, 0], P2[c, 1], "o", ms=15, mfc="none", mec=ROJO, mew=1.6)
            ax.set_title(f"centro {paso + 1}: chance ∝ distancia² (○ rojo: las {len(reg[paso]['candidatas'])} candidatas)",
                         fontsize=10)
        for j in range(paso):
            i = reg[j]["elegido"]
            ax.plot(P2[i, 0], P2[i, 1], "*", ms=20, color="black", mfc="yellow")
        i = reg[paso]["elegido"]
        ax.plot(P2[i, 0], P2[i, 1], "*", ms=24, color="black", mfc=ROJO)
        for k in (1, 2, 3):
            cx, cy = P2[y == k, :2].mean(axis=0)
            xy_ = (cx + 0.22, cy) if cy < 0 else (cx, cy - 0.32)
            ax.text(*xy_, f"chance {100 * por_grupo[k]:.1f} %", ha="left" if cy < 0 else "center",
                    va="center" if cy < 0 else "top", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbbbbb"))
        ejes_pca(ax, E["vis"])
    hablar("Chance total de cada identidad en cada sorteo: " + "; ".join(
        f"centro {p + 1}: " + ", ".join(f"id {k} {100 * c[k]:.1f} %" for k in (1, 2, 3)) for p, c in enumerate(chances)) + ".")
    cerrar(fig, 2, "kmeanspp", "k-means++: elegir los 3 centros de arranque",
           "★ rojo: el centro elegido en cada sorteo; ★ amarillo: los ya elegidos. Color: la chance de cada huella (más clara, más "
           "chance), proporcional a la distancia² al centro más cercano: el grupo que ya tiene centro casi no tiene chance.")


def paso_3_lloyd(E: dict) -> None:
    encabezado(3, "Lloyd: asignar al más cercano y mover a la media")
    X, P2, vis = E["Xtr"], E["P_tr"], E["vis"]
    kc = cfg["huellas"]["kmeans"]
    tol_abs = float(kc["tol"]) * float(np.mean(np.var(X, axis=0)))
    reg: list = []
    C, et, J, it = lloyd(X, E["C0"], tol_abs, int(kc["max_iter"]), reg)
    E["lloyd1"] = (C, et, J, it)
    J0 = reg[0]["J"]
    manzanita("Dos pasos que se repiten. Uno: cada fruta va a la pila del representante que más se le parece. Dos: cada "
              "representante se muda al «promedio» de su pila. Repito hasta que ninguna fruta cambie de pila. Cada vuelta "
              "baja la suma de distancias, así que siempre termina.")
    decir(f"Después vienen las iteraciones de Lloyd. Cada huella va al centro más cercano, y cada centro se mueve a la media "
          f"de sus huellas. Se repite hasta que ninguna huella cambia de grupo. Lo que se minimiza es la inercia J, la suma "
          f"de las distancias al cuadrado de cada huella a su centro: arranca en {J0:.1f} y termina en {J:.2f}. Como los "
          f"grupos están muy separados, alcanzó con {it} iteraciones.")
    hablar(f"J = Σ_j Σ_(x∈C_j) ‖x − c_j‖². Tolerancia: tol · media(var(X)) = {tol_abs:.2e}. Iteraciones: "
           + "; ".join(f"{i + 1}: J = {r['J']:.3f}, movimiento² de los centros = {r['mov']:.3f}" for i, r in enumerate(reg))
           + f". Al final J = {J:.4f}; se para porque las etiquetas no cambian.")
    serie, nombres = [], []
    for i, r in enumerate(reg):
        serie.append(r["J"]); nombres.append(f"asignar {i + 1}")
        nc = 0 if i == 0 else int((r["etiquetas"] != reg[i - 1]["etiquetas"]).sum())
        cambios = "" if i == 0 else ("\n(nadie cambia)" if nc == 0 else f"\n({nc} cambian)")
        nombres[-1] += cambios
        serie.append(float(((X - r["nuevos"][r["etiquetas"]]) ** 2).sum())); nombres.append(f"mover {i + 1}")
    fig, ejes = figura(1, 3, 15.5, 5.2, gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
    pal = [ROJO, VIOLETA, "#17becf"]
    r = reg[0]
    for p, ax in enumerate(ejes[0, :2]):
        ax.scatter(P2[:, 0], P2[:, 1], c=[pal[e] for e in r["etiquetas"]], s=26, alpha=0.7)
        Cv, Nv = proyectar(vis, r["C"]), proyectar(vis, r["nuevos"])
        if p == 0:
            ax.scatter(Cv[:, 0], Cv[:, 1], marker="X", s=260, c=pal, edgecolors="black", lw=1.5)
            ax.set_title(f"asignar: cada huella al centro más cercano (J = {r['J']:.2f})", fontsize=10)
        else:
            ax.scatter(Cv[:, 0], Cv[:, 1], marker="X", s=200, c=pal, alpha=0.3)
            for a, b in zip(Cv, Nv):
                ax.annotate("", b[:2], a[:2], arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
            ax.scatter(Nv[:, 0], Nv[:, 1], marker="X", s=260, c=pal, edgecolors="black", lw=1.5)
            ax.set_title(f"mover: cada centro a la media de su grupo (J = {serie[1]:.2f})", fontsize=10)
        ejes_pca(ax, vis)
    ax = ejes[0, 2]
    ax.plot(range(len(serie)), serie, "o-", color=AZUL)
    for x_, v_ in enumerate(serie):
        ax.text(x_, v_ + 2.5, f"{v_:.2f}", ha="center", fontsize=9)
    ax.set_xlim(-0.4, len(serie) - 0.6)
    ax.set_xticks(range(len(serie)), nombres, fontsize=8.5)
    ax.set_ylim(0, max(serie) * 1.15); ax.set_ylabel("inercia J")
    ax.set_title("J baja en cada paso; si nadie cambia, termina", fontsize=10)
    cerrar(fig, 3, "lloyd", "Lloyd: asignar al más cercano y mover a la media",
           "Color: el grupo de cada huella. X: los centros; la flecha, a dónde se mueven (la media de su grupo). En la 2.ª vuelta "
           f"nadie cambia de grupo y se para: {it} iteraciones.")


def paso_4_arranques(E: dict) -> None:
    from src.core_ml.kmeans import KMeans
    from src.core_ml.metricas import silhouette
    encabezado(4, "15 arranques, y por qué k = 3")
    X = E["Xtr"]
    kc = cfg["huellas"]["kmeans"]
    tol_abs = float(kc["tol"]) * float(np.mean(np.var(X, axis=0)))
    rng = np.random.default_rng(int(cfg["semilla_global"]))
    arr = []
    for _ in range(int(kc["n_init"])):
        arr.append(lloyd(X, kmeanspp(X, int(kc["k"]), rng), tol_abs, int(kc["max_iter"])))
    mejor = min(range(len(arr)), key=lambda i: (arr[i][2], i))
    E["mejor"] = arr[mejor]
    Js = np.array([a[2] for a in arr])
    ks = list(range(1, 7))
    inercias, sils = [], []
    for k in ks:
        km = KMeans(k, int(kc["n_init"]), int(kc["max_iter"]), float(kc["tol"]), int(cfg["semilla_global"])).ajustar(X)
        inercias.append(km.inercia)
        sils.append(silhouette(X, km.etiquetas) if k >= 2 else np.nan)
    # solo para comparar (no está en src/): los mismos 15 arranques con los 3 centros elegidos al azar, sin k-means++
    rng_azar = np.random.default_rng(int(cfg["semilla_global"]))
    Js_azar = np.array([lloyd(X, X[rng_azar.choice(len(X), int(kc["k"]), replace=False)], tol_abs, int(kc["max_iter"]))[2]
                        for _ in range(int(kc["n_init"]))])
    trabados = int((Js_azar > Js.min() * 1.01).sum())
    manzanita("Si elijo los 3 representantes completamente al azar, muchas veces caen dos en la misma pila: entonces una pila "
              "se parte en dos y otras dos quedan juntas, y Lloyd no lo arregla (queda trabado en una solución mala). Con "
              "k-means++ eso casi no pasa, y además lo hago 15 veces y me quedo con la mejor. Y para ver si 3 pilas es lo "
              "natural, pruebo con 1 a 6: con 3 la suma cae de golpe y después casi no baja (el «codo»), y la separación es "
              "máxima.")
    decir(f"Hago {kc['n_init']} arranques y me quedo con el de menor inercia. Con k-means++ los {len(Js)} llegan a la misma "
          f"solución, J = {Js.min():.2f}. Para mostrar para qué sirve, a la izquierda comparo con elegir los 3 centros al azar: "
          f"{trabados} de {len(Js_azar)} arranques quedan trabados en una solución mala, con J de hasta {Js_azar.max():.0f}, "
          f"porque dos centros arrancan en la misma identidad. El k lo fija la consigna en 3, pero lo verifiqué: la inercia "
          f"cae mucho hasta k = 3 y después casi nada, y la silhouette es máxima en k = 3 ({sils[2]:.2f}).")
    hablar("J de cada arranque con k-means++: " + ", ".join(f"{j:.3f}" for j in Js) + f". Elegido: el {mejor + 1} (el primero "
           "de menor J). Con centros al azar (solo para comparar, misma semilla): " + ", ".join(f"{j:.1f}" for j in Js_azar)
           + ". Inercia según k: " + ", ".join(f"k={k}: {j:.1f}" for k, j in zip(ks, inercias)) + ". Silhouette: "
           + ", ".join(f"k={k}: {s:.3f}" for k, s in zip(ks[1:], sils[1:])) + ".")
    fig, ejes = figura(1, 3, 15.5, 4.8)
    ax = ejes[0, 0]
    pos = np.arange(1, len(Js) + 1)
    ax.bar(pos - 0.2, Js_azar, width=0.4, color=GRIS, label="centros al azar")
    ax.bar(pos + 0.2, Js, width=0.4, color=AZUL, label="k-means++")
    ax.bar([pos[mejor] + 0.2], [Js[mejor]], width=0.4, color=ROJO, label="el elegido")
    ax.set_xticks(pos); ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("arranque"); ax.set_ylabel("inercia J al terminar"); ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, max(Js_azar.max(), Js.max()) * 1.22)
    ax.set_title(f"{kc['n_init']} arranques: al azar {trabados} quedan trabados; con k-means++, ninguno", fontsize=10)
    ax = ejes[0, 1]
    ax.plot(ks, inercias, "o-", color=AZUL); ax.plot([3], [inercias[2]], "o", ms=14, mfc="none", mec=ROJO, mew=2)
    ax.set_xticks(ks)
    ax.set_xlabel("k"); ax.set_ylabel("inercia J"); ax.set_title("el codo: después de k = 3 casi no baja", fontsize=10)
    ax = ejes[0, 2]
    ax.plot(ks[1:], sils[1:], "o-", color=VERDE); ax.plot([3], [sils[2]], "o", ms=14, mfc="none", mec=ROJO, mew=2)
    ax.set_xticks(ks[1:])
    ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.set_title("separación de los grupos según k (máxima en 3)", fontsize=10)
    cerrar(fig, 4, "arranques", "15 arranques, y por qué k = 3",
           "Se queda el arranque de menor inercia. Con centros al azar (gris, solo para comparar) varios arranques quedan "
           "trabados; con k-means++ todos llegan a la mejor solución. k = 3 lo pide la consigna, y los datos lo confirman: "
           "codo en la inercia y silhouette máxima.")


def paso_5_correspondencia(E: dict) -> None:
    encabezado(5, "Cada cluster recibe el nombre de una identidad")
    C, et, J, it = E["mejor"]
    y = E["ytr"]
    tabla = np.array([[int(((et == j) & (y == k)).sum()) for k in (1, 2, 3)] for j in range(3)])
    mapa = {j: int(np.argmax(tabla[j]) + 1) for j in range(3)}
    pureza = tabla.max(axis=1).sum() / len(y)
    E["mapa"], E["pureza"] = mapa, pureza
    medias = np.array([E["Xtr"][y == k].mean(axis=0) for k in (1, 2, 3)])
    dif = max(float(np.abs(C[j] - medias[mapa[j] - 1]).max()) for j in range(3))
    manzanita("Recién ahora miro las etiquetas, para ponerle nombre a cada pila: la pila que tiene 24 frutas de la identidad "
              "2 se llama «identidad 2». Si una pila estuviera mezclada, gana la identidad que más tiene (y si empatan, la de "
              "número menor). Acá ninguna está mezclada.")
    decir(f"Recién acá uso las identidades, y solo para nombrar los grupos: cada cluster recibe la identidad que más huellas "
          f"tiene adentro. La tabla muestra que cada cluster tiene las 24 huellas de una sola identidad: la pureza es "
          f"{100 * pureza:.0f} %. O sea que el K-means encontró solo, sin que se lo dijera nadie, las tres identidades. Los "
          f"centroides quedan en la media de cada identidad.")
    hablar(f"Tabla cluster × identidad: {tabla.tolist()}. Mapa: " + ", ".join(f"cluster {j} → identidad {k}" for j, k in mapa.items())
           + f". Pureza = Σ mayoritarias / n = {pureza:.3f}. Diferencia máxima entre cada centroide y la media de su identidad: "
           f"{dif:.1e}. Empate en la mayoría → la identidad menor (np.unique ordena).")
    fig, ejes = figura(1, 2, 12.5, 4.6, gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax = ejes[0, 0]
    ax.imshow(tabla, cmap="Blues", vmin=0)
    for (j, k), v in np.ndenumerate(tabla):
        ax.text(k, j, str(v), ha="center", va="center", fontsize=14, color="white" if v > 12 else "black")
    ax.set_xticks(range(3), [f"id {k}" for k in (1, 2, 3)]); ax.set_yticks(range(3), [f"cluster {j}" for j in range(3)])
    ax.set_xlabel("identidad (recién ahora se mira)")
    ax.set_title(f"huellas de cada identidad en cada cluster · pureza {100 * pureza:.0f} %", fontsize=10)
    ax = ejes[0, 1]
    P2 = E["P_tr"]
    for j in range(3):
        m = et == j
        ax.scatter(P2[m, 0], P2[m, 1], c=ID_COLOR[mapa[j]], s=28, label=f"cluster {j} → identidad {mapa[j]}")
    Cv = proyectar(E["vis"], C)
    ax.scatter(Cv[:, 0], Cv[:, 1], marker="X", s=260, c=[ID_COLOR[mapa[j]] for j in range(3)], edgecolors="black", lw=1.5)
    ax.legend(fontsize=9); ejes_pca(ax, E["vis"]); ax.set_title("los clusters con su nombre", fontsize=10)
    cerrar(fig, 5, "correspondencia", "Cada cluster recibe el nombre de una identidad",
           f"Cada cluster toma la identidad mayoritaria de sus huellas. Pureza {100 * pureza:.0f} %: los 3 grupos que encontró el "
           "K-means sin mirar las etiquetas son exactamente las 3 identidades.")


def paso_6_normalizada(E: dict) -> None:
    encabezado(6, "Distancia normalizada: medir en «dispersiones» de cada grupo")
    C, et, J, it = E["mejor"]
    X = E["Xtr"]
    mapa = E["mapa"]
    orden = [next(j for j in range(3) if mapa[j] == k) for k in (1, 2, 3)]
    Cid = C[orden]
    et_id = np.array([mapa[j] for j in et])
    I = np.array([float(((X[et_id == k] - Cid[k - 1]) ** 2).sum()) for k in (1, 2, 3)])
    nk = np.array([int((et_id == k).sum()) for k in (1, 2, 3)])
    rbar = math.sqrt(I.sum() / len(X))
    piso = float(cfg["huellas"]["umbral"]["piso_dispersion"])
    r = np.maximum(np.sqrt(I / nk), piso * rbar)
    E["Cid"], E["r"] = Cid, r
    eu = [np.linalg.norm(X[et_id == k] - Cid[k - 1], axis=1) for k in (1, 2, 3)]
    manzanita("Una pila de naranjas puede ser muy pareja y una de peras, más variada. Estar a 3 cm del centro es mucho para "
              "las naranjas y poco para las peras. Entonces mido la distancia en «cuántas veces la variación típica de esa "
              "pila», como medir en desvíos: así una fruta a 1 está en lo normal de su pila, sea cual sea.")
    decir(f"Para decidir uso la distancia de la huella a cada centroide, pero dividida por la dispersión de ese grupo en "
          f"train, la raíz de su inercia sobre su cantidad de huellas. Los grupos no son igual de anchos: las dispersiones dan "
          f"{r[0]:.3f}, {r[1]:.3f} y {r[2]:.3f}. Así, estar a una distancia 1 quiere decir lo mismo en los tres grupos. El "
          f"puntaje de una huella es la menor de esas distancias normalizadas, y la identidad candidata es la de ese "
          f"centroide. Hay un piso, la mitad de la dispersión general, para no dividir por cero si un grupo quedara con una "
          f"sola huella; acá no actúa.")
    hablar(f"r_k = max(√(I_k / n_k), {piso}·r̄): I = {[round(v, 3) for v in I]}, n_k = {nk.tolist()}, r̄ = {rbar:.4f}, piso "
           f"{piso * rbar:.4f} → r = {[round(v, 4) for v in r]}. Puntaje s(x) = min_k ‖x − c_k‖ / r_k. En el prototipo, sobre 600 "
           "huellas nuevas, la normalizada tuvo error de igual tasa 0,3 % contra 0,7 % de la euclídea.")
    fig, ejes = figura(1, 2, 13, 4.6)
    for ax, norm, tit in [(ejes[0, 0], False, "distancia euclídea a su centroide"),
                          (ejes[0, 1], True, "distancia normalizada (÷ r_k)")]:
        for k in (1, 2, 3):
            v = eu[k - 1] / (r[k - 1] if norm else 1)
            ax.scatter(v, np.full(len(v), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), c=ID_COLOR[k], s=24)
            ax.plot([np.median(v)] * 2, [k - 0.3, k + 0.3], color="black", lw=2)
        ax.set_yticks([1, 2, 3], [f"identidad {k}\nr = {r[k - 1]:.2f}" for k in (1, 2, 3)])
        ax.set_title(tit + " (train; raya: mediana)", fontsize=10)
    ejes[0, 0].set_xlabel("‖x − c_k‖"); ejes[0, 1].set_xlabel("‖x − c_k‖ / r_k")
    cerrar(fig, 6, "normalizada", "Distancia normalizada: medir en «dispersiones» de cada grupo",
           "Cada grupo tiene su ancho. Dividiendo por su dispersión r_k, las tres identidades quedan en la misma escala: el "
           "umbral significa lo mismo para todas.")


def paso_7_umbral(E: dict) -> None:
    encabezado(7, "El umbral τ*: con validación, en la mitad del margen")
    h = E["h"]
    va = h["particion"] == "val"
    Xv, yv = h["X"][va], h["identidad"][va]
    D = np.sqrt(dist2(Xv, E["Cid"])) / E["r"][None, :]
    j = np.argmin(D, axis=1)
    s, pred = D[np.arange(len(Xv)), j], j + 1
    aut = yv > 0
    bien = aut & (pred == yv)
    u = cfg["huellas"]["umbral"]
    m = math.floor(float(u["far_max_val"]) * int((~aut).sum()) + 1e-9)
    b = float(np.sort(s[~aut])[m])
    a = float(s[bien & (s < b)].max())
    tau = (a + b) / 2
    E["tau"] = tau
    grilla = np.linspace(0, max(s.max(), 4), 600)
    far = [(s[~aut] <= t).mean() for t in grilla]
    frr = [1 - (bien & (s <= t))[aut].mean() for t in grilla]
    manzanita("Para decidir si una huella es de la casa o de un intruso pongo una vara: si está más cerca que la vara de "
              "alguna pila, entra; si no, afuera. Para ubicar la vara uso validación: la autorizada más lejana está en "
              f"{a:.2f} y el intruso más cercano en {b:.2f}. Pongo la vara justo en el medio, para dejar el mismo margen de "
              "los dos lados: si la pongo pegada a uno, cualquier huella un poco distinta en test cae del lado equivocado.")
    decir(f"Falta el umbral para rechazar intrusos. Lo calibro con validación: {int(aut.sum())} autorizadas y "
          f"{int((~aut).sum())} intrusos que el modelo no vio. Para cada umbral calculo la FAR, intrusos aceptados, y la FRR, "
          f"autorizadas rechazadas o mal identificadas. El criterio pide FAR de a lo sumo 1 %, o sea ningún intruso. Eso deja "
          f"un intervalo sin errores: de {a:.3f}, la autorizada más lejana, a {b:.3f}, el intruso más cercano. Pongo τ* en "
          f"la mitad: {tau:.3f}. Probado con 600 huellas nuevas, el umbral en la mitad cumple las metas el 99,6 % de las "
          f"veces, contra 85,8 % con la distancia euclídea y el umbral en el borde.")
    hablar(f"FAR(τ) = intrusos con s ≤ τ; FRR(τ) = autorizadas que no quedan aceptadas con su identidad. far_max_val = "
           f"{u['far_max_val']} con {int((~aut).sum())} intrusos → se pueden aceptar m = {m}. b = {b:.4f} (intruso más cercano), "
           f"a = {a:.4f} (autorizada bien identificada más lejana debajo de b), τ* = (a + b)/2 = {tau:.4f}. En val: "
           f"{int(bien.sum())} de {int(aut.sum())} autorizadas bien identificadas.")
    fig, ejes = figura(2, 1, 13, 7.6, sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax = ejes[0, 0]
    ax.plot(grilla, 100 * np.array(far), color=ROJO, lw=2, label="FAR: intrusos aceptados")
    ax.plot(grilla, 100 * np.array(frr), color=AZUL, lw=2, label="FRR: autorizadas rechazadas")
    ax.axvspan(a, b, color=VERDE, alpha=0.15, label="sin errores en val")
    ax.axvline(tau, color="black", ls="--", lw=1.5, label=f"τ* = {tau:.3f}")
    ax.set_ylabel("%"); ax.legend(fontsize=9, loc="center right"); ax.set_title("FAR y FRR en validación según el umbral", fontsize=10)
    ax = ejes[1, 0]
    tope = 4.5
    yj = franjas(ax, s, yv, tope)
    ia, ib = int(np.argmin(np.abs(s - a))), int(np.argmin(np.abs(s - b)))
    ax.axvspan(a, b, color=VERDE, alpha=0.15); ax.axvline(tau, color="black", ls="--", lw=1.5)
    ax.annotate(f"a = {a:.3f}\nla autorizada más lejana", (a, yj[ia]), xytext=(a - 0.75, 0.45), ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", shrinkB=5))
    ax.annotate(f"b = {b:.3f}\nel intruso más cercano", (b, yj[ib]), xytext=(b + 0.75, 0.62), ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", shrinkB=5))
    ax.set_xlabel("puntaje s(x) = distancia normalizada al centroide más cercano")
    ax.set_title("cada huella de validación", fontsize=10)
    ax.set_xlim(0, tope)
    cerrar(fig, 7, "umbral", "El umbral τ*: con validación, en la mitad del margen",
           f"Se acepta si s(x) ≤ τ*. En validación hay un intervalo sin errores [{a:.2f}, {b:.2f}); τ* = {tau:.3f} va en la mitad para "
           "dejar margen hacia los dos lados.")


def franjas(ax, s: np.ndarray, y: np.ndarray, tope: float, extra: list | None = None) -> np.ndarray:
    """Autorizadas arriba (color por identidad) e intrusos abajo, con un poco de ruido vertical. Devuelve la altura de cada una."""
    yj = (y > 0).astype(float) + np.random.default_rng(0).uniform(-0.2, 0.2, len(s))
    for k in (1, 2, 3):
        m = y == k
        ax.scatter(s[m], yj[m], c=ID_COLOR[k], s=30, label=f"identidad {k}", zorder=3)
    m = y == 0
    ax.scatter(s[m], yj[m], c=GRIS, s=18, label="intrusos", zorder=3)
    fuera = int((s > tope).sum())
    if fuera:
        ax.text(tope - 0.05, -0.38, f"{fuera} intrusos más lejos, fuera del gráfico →", ha="right", fontsize=8.5, color="#555555")
    h, l = ax.get_legend_handles_labels()
    for e in extra or []:
        h.append(e); l.append(e.get_label())
    ax.legend(h, l, fontsize=8.5, ncol=2, loc="upper right")
    ax.set_yticks([0, 1], ["intrusos", "autorizadas"]); ax.set_ylim(-0.45, 1.5)
    return yj


def paso_8_test(E: dict) -> None:
    encabezado(8, "El examen: test, una sola vez y con todo fijo")
    h = E["h"]
    te = h["particion"] == "test"
    Xt, yt, arch = h["X"][te], h["identidad"][te], h["archivo"][te]
    D = np.sqrt(dist2(Xt, E["Cid"])) / E["r"][None, :]
    j = np.argmin(D, axis=1)
    s, pred = D[np.arange(len(Xt)), j], j + 1
    tau = E["tau"]
    dec = np.where(s <= tau, pred, 0)
    aut = yt > 0
    ident = float((pred[aut] == yt[aut]).mean())
    far = float((dec[~aut] != 0).mean())
    frr = float((dec[aut] != yt[aut]).mean())
    M = np.array([[int(((yt == v) & (dec == d)).sum()) for d in (1, 2, 3, 0)] for v in (1, 2, 3, 0)])
    E["test"] = dict(identificacion=ident, far=far, frr=frr, matriz=M)
    rech = [(str(a), float(v)) for a, v, ok in zip(arch, s, aut & (dec != yt)) if ok]
    cerca = int(np.argmin(np.where(~aut, s, np.inf)))
    manzanita("Test es el examen final: se mira una sola vez, con la vara ya clavada. Si después de ver el examen moviera la "
              "vara, estaría haciendo trampa: el resultado ya no diría cómo le va con huellas nuevas.")
    decir(f"Con todo fijo, evalué test una sola vez: {int(aut.sum())} autorizadas y {int((~aut).sum())} huellas de intrusos, "
          f"personas que no están autorizadas. La identificación da {100 * ident:.0f} %: todas las autorizadas quedan más cerca de su propio "
          f"centroide. Ningún intruso entra, FAR {100 * far:.0f} %. Y una autorizada queda afuera: FRR {100 * frr:.1f} %, "
          f"que es {Path(rech[0][0]).stem if rech else '-'}, con un corte y la escala reducida, a {rech[0][1] if rech else 0:.2f} contra "
          f"τ* {tau:.2f}. Y no conviene subir τ* para dejarla pasar: el intruso más cercano de test está en {s[cerca]:.3f}, "
          f"prácticamente en el mismo lugar, así que entraría también; además sería ajustar con test. Las metas eran "
          f"identificación de al menos 95 %, FAR de a lo sumo 5 % y FRR de a lo sumo 10 %: se cumplen.")
    hablar(f"Test: identificación {ident:.3f}, FAR {far:.3f} ({int((dec[~aut] != 0).sum())} de {int((~aut).sum())}), FRR {frr:.3f} "
           f"({int((dec[aut] != yt[aut]).sum())} de {int(aut.sum())}). Rechazadas: {[(a_, round(v, 3)) for a_, v in rech]}. "
           f"Intruso más cercano: {arch[cerca]} en {s[cerca]:.3f}. Matriz (filas: real 1, 2, 3, intruso; columnas: decisión): "
           f"{M.tolist()}.")
    fig, ejes = figura(1, 2, 14.5, 5, gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = ejes[0, 0]
    tope = 4.5
    yj = franjas(ax, s, yt, tope, extra=[plt.Line2D([], [], color="black", ls="--", label=f"τ* = {tau:.3f}")])
    ax.axvline(tau, color="black", ls="--", lw=1.5)
    for nombre_, v_ in rech:
        i = int(np.flatnonzero(arch == nombre_)[0])
        ax.annotate(f"{Path(nombre_).stem}: {v_:.2f}\n(corte y escala reducida)", (v_, yj[i]), xytext=(v_ + 0.9, 0.55),
                    ha="center", fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlim(0, tope)
    ax.set_xlabel("puntaje s(x)"); ax.set_title("cada huella de test", fontsize=10)
    ax = ejes[0, 1]
    ax.imshow(np.log1p(M), cmap="Blues")
    for (i, jj), v in np.ndenumerate(M):
        ax.text(jj, i, str(v), ha="center", va="center", fontsize=13, color="white" if v > 20 else "black")
    et = ["id 1", "id 2", "id 3", "intruso"]
    ax.set_xticks(range(4), [e.replace("intruso", "rechazada") for e in et]); ax.set_yticks(range(4), et)
    ax.set_xlabel("decisión"); ax.set_ylabel("real")
    ax.set_title(f"identificación {100 * ident:.0f} % · FAR {100 * far:.0f} % · FRR {100 * frr:.1f} %", fontsize=10)
    cerrar(fig, 8, "test", "El examen: test, una sola vez y con todo fijo",
           "Ningún intruso aceptado y ninguna autorizada confundida con otra; una rechazada (un corte grande y escala reducida). "
           "Las metas (≥ 95 %, ≤ 5 %, ≤ 10 %) se cumplen.")


# ============================================================================= PARTE B · PCA
def paso_9_que_es(E: dict) -> None:
    encabezado(9, "Qué hace el PCA (ejemplo en 2D)")
    rng = np.random.default_rng(int(cfg["semilla_global"]))
    X = rng.multivariate_normal([0, 0], [[3, 1.6], [1.6, 1.2]], 150)
    P = pca_propio(X, 2)
    Z = proyectar(P, X)
    v1, v2 = P["componentes"]
    l1, l2 = P["varianza"]
    manzanita("Imaginá una nube de puntos y una linterna que proyecta su sombra en la pared. Según desde dónde ilumines, la "
              "sombra sale más ancha o más angosta. El PCA busca el ángulo en que la sombra es más ancha (donde los puntos "
              "más se diferencian): esa es la primera componente. La segunda es la mejor dirección perpendicular a la primera, "
              "y así. Cada componente es una mezcla de las columnas originales.")
    decir(f"El PCA busca nuevas direcciones, combinaciones de las columnas originales, ordenadas por cuánto se desparraman "
          f"los datos en cada una. En este ejemplo en 2D, la primera componente va a lo largo de la nube y se queda con el "
          f"{100 * P['proporcion'][0]:.0f} % de la varianza; la segunda, perpendicular, con el resto. Matemáticamente: armo la "
          f"matriz de covarianza de los datos centrados, saco sus autovectores, que son las direcciones, y sus autovalores, "
          f"que son la varianza en cada dirección. Proyectar es restar la media y multiplicar por los autovectores. Lo ajusto "
          f"solo con train y después proyecto validación y test con eso, sin volver a ajustar.")
    hablar("Σ = Xcᵀ·Xc / (n − 1); Σ·v = λ·v (np.linalg.eigh), de mayor a menor λ; proporción de varianza = λ / traza(Σ); "
           f"z = (x − media)·Vᵀ. En el ejemplo: λ = {l1:.2f} y {l2:.2f}, proporciones {100 * P['proporcion'][0]:.1f} % y "
           f"{100 * P['proporcion'][1]:.1f} %. El PCA del proyecto (src/core_ml/pca.py) es propio y coincide con scikit-learn "
           "hasta 1e-11.")
    fig, ejes = figura(1, 2, 13, 5.4, gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = ejes[0, 0]
    ax.scatter(X[:, 0], X[:, 1], s=14, c=GRIS, zorder=2)
    m = P["media"]
    largo = np.abs(Z[:, 0]).max() * 1.05
    ax.plot(*np.array([m - largo * v1, m + largo * v1]).T, color=ROJO, lw=1, ls="--", alpha=0.6)
    for x in X[::5]:
        pr = m + ((x - m) @ v1) * v1
        ax.plot([x[0], pr[0]], [x[1], pr[1]], color=ROJO, lw=0.6, alpha=0.6)
        ax.plot(pr[0], pr[1], "o", ms=3, color=ROJO)
    for v, l, col, nom in [(v1, l1, ROJO, "PC1"), (v2, l2, AZUL, "PC2")]:
        ax.annotate("", m + 2 * math.sqrt(l) * v, m, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.8), zorder=4)
        ax.text(*(m + 2 * math.sqrt(l) * v + 0.35 * v), nom, color=col, fontsize=12, fontweight="bold", ha="center", va="center")
    ax.set_aspect("equal"); ax.set_xlabel("columna 1"); ax.set_ylabel("columna 2")
    ax.set_title("los datos y sus dos componentes (rojo fino: cada punto «cae» sobre PC1)", fontsize=10)
    ax = ejes[0, 1]
    rj = np.random.default_rng(1).uniform(-0.12, 0.12, len(Z))
    ax.scatter(Z[:, 0], 1 + rj, s=12, c=ROJO)
    ax.scatter(Z[:, 1], rj, s=12, c=AZUL)
    for fila, c_, col in [(1, 0, ROJO), (0, 1, AZUL)]:
        d_ = 2 * Z[:, c_].std()
        ax.plot([-d_, d_], [fila - 0.24] * 2, color=col, lw=3)
        ax.text(d_ + 0.2, fila - 0.24, f"{100 * P['proporcion'][c_]:.0f} % de la varianza", color=col, va="center", fontsize=9.5)
    ax.set_yticks([0, 1], ["sombra sobre PC2", "sombra sobre PC1"]); ax.set_ylim(-0.45, 1.3)
    ax.set_xlabel("coordenada sobre la componente")
    ax.set_title("la «sombra» sobre cada componente: la de PC1 es la más ancha", fontsize=10)
    cerrar(fig, 9, "que_es_pca", "Qué hace el PCA (ejemplo en 2D)",
           "PC1 es la dirección de mayor varianza; PC2, la mayor perpendicular a PC1. Son los autovectores de la covarianza, y "
           "sus autovalores son la varianza que explica cada una.")


def paso_10_varianza(E: dict) -> None:
    encabezado(10, "Cuánta varianza explica cada componente")
    v = E["v"]
    Ph = pca_propio(E["Xtr"])
    Pv = pca_propio(v["X"][v["particion"] == "train"])
    E["Ph_full"], E["Pv_full"] = Ph, Pv
    ch, cv = np.cumsum(Ph["proporcion"]), np.cumsum(Pv["proporcion"])
    manzanita("Es como resumir un libro: en las huellas, dos oraciones cuentan casi toda la historia; en la voz, la historia "
              "está repartida en muchas páginas y dos oraciones no alcanzan.")
    decir(f"Este gráfico dice cuánto se pierde al dibujar en pocas dimensiones. En huellas, las dos primeras componentes ya "
          f"explican el {100 * ch[1]:.0f} % de la varianza: el dibujo en 2D es fiel. En voz, las dos primeras explican solo el "
          f"{100 * cv[1]:.0f} %, y hacen falta 20 para llegar al {100 * cv[19]:.0f} %. Por eso el dibujo de la voz se ve "
          f"mezclado y por eso el K-NN de la voz trabaja con 20 componentes y no con 2.")
    hablar(f"Huellas (train, {E['Xtr'].shape[0]} × {E['Xtr'].shape[1]}): PC1 {100 * Ph['proporcion'][0]:.1f} %, PC2 "
           f"{100 * Ph['proporcion'][1]:.1f} %, PC3 {100 * Ph['proporcion'][2]:.1f} %; acumulada con 3: {100 * ch[2]:.1f} %. "
           f"Voz (train, {int((v['particion'] == 'train').sum())} × {v['X'].shape[1]}): PC1 {100 * Pv['proporcion'][0]:.1f} %, PC2 "
           f"{100 * Pv['proporcion'][1]:.1f} %; con 10: {100 * cv[9]:.1f} %; con 20: {100 * cv[19]:.1f} %; con 30: {100 * cv[29]:.1f} %.")
    fig, ejes = figura(1, 2, 14, 4.8)
    for ax, P, c, nom, n in [(ejes[0, 0], Ph, ch, "huellas", 15), (ejes[0, 1], Pv, cv, "voz", 40)]:
        ax.bar(range(1, n + 1), 100 * P["proporcion"][:n], color=AZUL if nom == "huellas" else VIOLETA, label="cada componente")
        ax.plot(range(1, n + 1), 100 * c[:n], "o-", color=ROJO, ms=3, label="acumulada")
        for kk in ([2] if nom == "huellas" else [2, 20]):
            ax.annotate(f"{100 * c[kk - 1]:.0f} % con {kk}", (kk, 100 * c[kk - 1]), xytext=(8, -14), textcoords="offset points",
                        fontsize=9, color=ROJO)
        if nom == "voz":
            ax.axvline(20, color="black", ls=":", lw=1.2)
            ax.text(20.5, 8, "el K-NN de la voz\nusa 20", fontsize=9)
        ax.set_ylim(0, 105); ax.set_xlabel("componente"); ax.set_ylabel("% de la varianza"); ax.legend(fontsize=9, loc="center right")
        ax.set_title(f"{nom} ({'72 huellas de train' if nom == 'huellas' else 'clips de train'})", fontsize=10)
    cerrar(fig, 10, "varianza", "Cuánta varianza explica cada componente",
           f"Huellas: {100 * ch[1]:.0f} % en 2 componentes (el dibujo es fiel). Voz: {100 * cv[1]:.0f} % en 2 y {100 * cv[19]:.0f} % en 20 (el dibujo "
           "en 2D muestra poco; el K-NN usa 20).")


def paso_11_pca_huellas(E: dict) -> None:
    from src.core_ml.metricas import silhouette
    encabezado(11, "PCA de las huellas en 2D y 3D")
    h = E["h"]
    X, part, ide, corte = h["X"], h["particion"], h["identidad"], E["corte"]
    P = E["vis"]
    Z = proyectar(P, X)
    tr, va = part == "train", (part == "val") & (ide > 0)
    s_tr, s_va = silhouette(X[tr], ide[tr]), silhouette(X[va], ide[va])
    Cv = proyectar(P, E["Cid"])
    tipo = h["tipo"]
    tipo_id = {k: str(tipo[ide == k][0]) for k in (1, 2, 3)}
    nt = lambda t: t.replace("_", " en ")
    # a qué identidad queda más cerca cada intruso en el dibujo (o si queda lejos de todas, "en el medio")
    intr = ide == 0
    d2d = np.sqrt(dist2(Z[intr, :2], Cv[:, :2]))
    cerca, medio = np.argmin(d2d, axis=1) + 1, d2d.min(axis=1) > 1.0
    tipos_i = tipo[intr]
    vecinos = {}
    for t in sorted(set(map(str, tipos_i))):
        mt = (tipos_i == t) & ~medio
        if mt.any():
            vecinos[t] = int(np.bincount(cerca[mt]).argmax())
    n_medio = int(medio.sum())
    tipos_medio = sorted(set(map(str, tipos_i[medio])))
    plural = lambda t: {"presilla": "presillas", "verticilo": "verticilos", "arco": "arcos"}.get(t, nt(t).replace("arco ", "arcos "))
    por_id = {k: [plural(t) for t, kk in vecinos.items() if kk == k] for k in (1, 2, 3)}
    junto = ", ".join(f"{' y '.join(ts)} junto a la {k}" for k, ts in por_id.items() if ts)
    texto_medio = (f", y {n_medio} huellas de {' y '.join(map(nt, tipos_medio))} quedan en el medio" if n_medio else "")
    manzanita("Es la foto de la nube de huellas desde el mejor ángulo. Se ven tres montones bien separados, uno por persona. "
              "Los intrusos con el mismo tipo de dibujo que una persona caen cerca de ella en la foto, pero en las otras "
              "371 direcciones que la foto no muestra quedan lejos: la decisión mira todas.")
    decir(f"Este es el gráfico PCA de las huellas, ajustado solo con train y con las 300 huellas proyectadas. Las tres "
          f"identidades forman grupos compactos y lejos entre sí; validación, en círculos, y test, en triángulos, caen sobre "
          f"su grupo, y las que tienen corte, con un punto negro, también. Las cruces son los centroides del K-means. Los "
          f"intrusos, en gris, caen cerca de la identidad con el dibujo más parecido ({junto}){texto_medio}. Que un intruso "
          f"se vea cerca en este dibujo no importa: la decisión se toma con la distancia en las 374 dimensiones, y ahí ningún intruso de "
          f"validación ni de test entra. La silhouette da {s_tr:.2f} en train y {s_va:.2f} en validación.")
    hablar(f"PC1 {100 * P['proporcion'][0]:.1f} %, PC2 {100 * P['proporcion'][1]:.1f} %, PC3 {100 * P['proporcion'][2]:.1f} %. "
           f"Silhouette (va de −1 a 1): {s_tr:.3f} en train y {s_va:.3f} en las autorizadas de val. Con corte: "
           f"{int((corte & (ide > 0)).sum())} autorizadas. Tipos: " + ", ".join(f"id {k} {nt(t)}" for k, t in tipo_id.items())
           + f". Intrusos en el dibujo: {junto}{texto_medio} (a más de 1 de todo centroide en 2D).")
    fig = plt.figure(figsize=(15.5, 6.6))
    ax = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    marcas = {"train": "o", "val": "o", "test": "^"}
    for a in (ax, ax3):
        intr = ide == 0
        args = (Z[intr, 0], Z[intr, 1]) if a is ax else (Z[intr, 0], Z[intr, 1], Z[intr, 2])
        a.scatter(*args, c=GRIS, s=14, alpha=0.5)
        for k in (1, 2, 3):
            for p in ("train", "val", "test"):
                m = (ide == k) & (part == p)
                args = (Z[m, 0], Z[m, 1]) if a is ax else (Z[m, 0], Z[m, 1], Z[m, 2])
                if p == "train":
                    a.scatter(*args, c=ID_COLOR[k], s=26, marker="o")
                else:
                    a.scatter(*args, facecolors="none", edgecolors=ID_COLOR[k], s=40, marker=marcas[p], lw=1.4)
        m = corte & (ide > 0)
        args = (Z[m, 0], Z[m, 1]) if a is ax else (Z[m, 0], Z[m, 1], Z[m, 2])
        a.scatter(*args, c="black", s=6)
        args = (Cv[:, 0], Cv[:, 1]) if a is ax else (Cv[:, 0], Cv[:, 1], Cv[:, 2])
        a.scatter(*args, marker="X", s=200, c=[ID_COLOR[k] for k in (1, 2, 3)], edgecolors="black", lw=1.3)
    if n_medio:
        cx, cy = Z[intr, :2][medio].mean(axis=0)
        ax.annotate(f"{n_medio} huellas intrusas de tipo {' y '.join(map(nt, tipos_medio))}\n(lejos de todas en el dibujo)", (cx, cy),
                    xytext=(cx, cy - 0.6), ha="center", fontsize=8.5, color="#444444",
                    arrowprops=dict(arrowstyle="->", color="#777777"))
    L = plt.Line2D
    manijas = [L([], [], marker="o", ls="", color=ID_COLOR[k], label=f"id {k} ({nt(tipo_id[k])})") for k in (1, 2, 3)]
    manijas += [L([], [], marker="o", ls="", color=GRIS, alpha=0.6, label="intrusos"),
                L([], [], marker="o", ls="", color="black", mfc="#555555", label="train"),
                L([], [], marker="o", ls="", color="black", mfc="none", label="val"),
                L([], [], marker="^", ls="", color="black", mfc="none", label="test"),
                L([], [], marker="o", ls="", color="black", ms=3, label="con corte"),
                L([], [], marker="X", ls="", color="black", mfc="white", ms=10, label="centroide")]
    ejes_pca(ax, P); ax.legend(handles=manijas, fontsize=8, ncol=2, loc="lower left")
    ax.set_title(f"2D · silhouette train {s_tr:.2f}, val {s_va:.2f}", fontsize=10)
    ax3.set_xlabel(f"PC1 ({100 * P['proporcion'][0]:.0f} %)"); ax3.set_ylabel(f"PC2 ({100 * P['proporcion'][1]:.0f} %)")
    ax3.set_zlabel(f"PC3 ({100 * P['proporcion'][2]:.0f} %)"); ax3.set_title("3D", fontsize=10); ax3.view_init(22, -60)
    cerrar(fig, 11, "pca_huellas", "PCA de las huellas en 2D y 3D",
           "PCA ajustado solo con train. Tres grupos compactos y separados; val (○) y test (△) caen en el suyo, también las "
           "que tienen corte (•). X: centroides del K-means. Los intrusos caen cerca de la identidad con el dibujo más parecido; la "
           "decisión se toma en las 374 dimensiones, donde ninguno entra.")


def paso_12_cargas(E: dict) -> None:
    encabezado(12, "De qué está hecha cada componente")
    Ph, Pv = E["Ph_full"], E["Pv_full"]
    largos_h = {"orientacion": 50, "gabor": 300, "minucias": E["Xtr"].shape[1] - 350}
    largos_v = {"mfcc": 130, "delta": 130, "energia": 10, "zcr": 10, "duracion": 1}

    def reparto(P, largos, n=3):
        filas = []
        for c in range(n):
            w = P["componentes"][c] ** 2
            o, f = 0, {}
            for b, d in largos.items():
                f[b] = float(w[o:o + d].sum()); o += d
            filas.append(f)
        return filas

    rh, rv = reparto(Ph, largos_h), reparto(Pv, largos_v)
    v = E["v"]
    trv = v["particion"] == "train"
    z1 = proyectar(Pv, v["X"][trv])[:, 0]
    corr = float(np.corrcoef(z1, v["duracion"][trv, 0])[0, 1])
    E["corr_dur"] = corr
    manzanita("Cada componente es una receta que mezcla ingredientes, las columnas originales. Acá miro cuánto aporta cada "
              "bloque a la receta. En la voz, la primera componente es casi solo un ingrediente: la duración de la palabra.")
    decir(f"Cada componente es una mezcla de las columnas; las barras muestran qué parte de cada una viene de cada bloque. En "
          f"huellas, las primeras componentes mezclan los tres bloques. En voz, la primera componente es casi pura duración: "
          f"su correlación con la duración de la palabra es {abs(corr):.2f}. Pasa porque en V3 cada bloque pesa lo mismo, y "
          f"entonces la única columna de duración pesa tanto como los 130 MFCC. Eso se ve en el gráfico de la voz y llevó a "
          f"probar otro peso en V4.")
    hablar("Fracción de Σ v² de cada componente en cada bloque. Huellas: "
           + "; ".join(f"PC{c + 1}: " + ", ".join(f"{BLOQUE_NOMBRE[b]} {100 * x:.0f} %" for b, x in f.items()) for c, f in enumerate(rh))
           + ". Voz: " + "; ".join(f"PC{c + 1}: " + ", ".join(f"{BLOQUE_NOMBRE[b]} {100 * x:.0f} %" for b, x in f.items()) for c, f in enumerate(rv))
           + f". Correlación de PC1 de la voz con la duración: {corr:.3f}.")
    fig, ejes = figura(1, 3, 15.5, 4.8, gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, filas, largos, tit in [(ejes[0, 0], rh, largos_h, "huellas"), (ejes[0, 1], rv, largos_v, "voz")]:
        abajo = np.zeros(len(filas))
        for b in largos:
            vals = np.array([f[b] for f in filas])
            ax.bar([f"PC{c + 1}" for c in range(len(filas))], 100 * vals, bottom=100 * abajo, color=BLOQUE_COLOR[b], label=BLOQUE_NOMBRE[b])
            abajo += vals
        ax.set_ylabel("% de la componente"); ax.legend(fontsize=8.5, loc="upper center", ncol=3); ax.set_ylim(0, 128)
        ax.set_yticks(range(0, 101, 20))
        ax.set_title(f"{tit}: de qué bloques está hecha cada componente", fontsize=10)
    ax = ejes[0, 2]
    ax.scatter(v["duracion"][trv, 0], z1, s=6, c=GRIS, alpha=0.5)
    ax.set_xlabel("duración de la palabra (s)"); ax.set_ylabel("PC1 de la voz")
    ax.set_title(f"voz: PC1 es la duración (correlación {corr:.2f})", fontsize=10)
    cerrar(fig, 12, "cargas", "De qué está hecha cada componente",
           f"Cada componente mezcla columnas de varios bloques. En voz, PC1 sigue a la duración (correlación {abs(corr):.2f}): con el "
           "peso por bloque de V3, la única columna de duración pesa tanto como los 130 MFCC.")


def _pca_voz(ax, ax3, Z, v, P, titulo):
    clase, part = v["clase"], v["particion"]
    marcas = {"train": "o", "val": "s", "test": "^"}
    for cl in ["otro", "reconocer", "identificar", "salir"]:
        for p in ("train", "val", "test"):
            m = (clase == cl) & (part == p) & ((v["copia"] == 0) | (p != "train"))
            if not m.any():
                continue
            kw = dict(c=CLASE_COLOR[cl], s=12, alpha=0.5 if cl == "otro" else 0.7) if p == "train" else \
                dict(facecolors="none", edgecolors=CLASE_COLOR[cl], s=34, lw=1.2)
            ax.scatter(Z[m, 0], Z[m, 1], marker=marcas[p], **kw)
            if ax3 is not None:
                ax3.scatter(Z[m, 0], Z[m, 1], Z[m, 2], marker=marcas[p], **kw)
    ax.set_xlabel(f"PC1 ({100 * P['proporcion'][0]:.1f} %)"); ax.set_ylabel(f"PC2 ({100 * P['proporcion'][1]:.1f} %)")
    ax.set_title(titulo, fontsize=10)


def leyenda_voz(ax, loc="best") -> None:
    L = plt.Line2D
    manijas = [L([], [], marker="o", ls="", color=CLASE_COLOR[c], label=f"«{c}»") for c in ("reconocer", "identificar", "salir", "otro")]
    manijas += [L([], [], marker="o", ls="", color="black", mfc="#555555", ms=4, label="train (sin copias)"),
                L([], [], marker="s", ls="", color="black", mfc="none", label="val"),
                L([], [], marker="^", ls="", color="black", mfc="none", label="test")]
    ax.legend(handles=manijas, fontsize=8, ncol=2, loc=loc)


def paso_13_pca_voz(E: dict) -> None:
    from src.core_ml.metricas import silhouette
    encabezado(13, "PCA de la voz en 2D y 3D")
    v = E["v"]
    tr = v["particion"] == "train"
    P = pca_propio(v["X"][tr], 3)
    Z = proyectar(P, v["X"])
    com = np.isin(v["clase"], cfg["voz"]["comandos"])
    k0 = tr & (v["copia"] == 0) & com
    va = (v["particion"] == "val") & com
    s_tr, s_va = silhouette(v["X"][k0], v["clase"][k0]), silhouette(v["X"][va], v["clase"][va])
    manzanita("La misma foto con las palabras. Acá los montones se pisan: «salir», que es corta y tiene la «s», queda aparte, "
              "pero «reconocer» e «identificar» se mezclan en esta foto. Se separan mirando más direcciones, y eso hace el "
              "K-NN con 20.")
    decir(f"Este es el gráfico PCA de la voz, también ajustado con train; muestro los originales de train, validación en "
          f"cuadrados y test en triángulos. La separación es mucho menor que en huellas: «salir», que es corta y tiene la «s», "
          f"queda a la izquierda, pero «reconocer» e «identificar» se pisan en estas componentes, y «otro» se mezcla con todo, "
          f"porque tiene palabras parecidas a propósito. La silhouette de los comandos da {s_tr:.2f} en train y {s_va:.2f} en "
          f"validación. La información que los distingue está repartida en muchas componentes: por eso el K-NN usa 20.")
    hablar(f"PC1 {100 * P['proporcion'][0]:.1f} %, PC2 {100 * P['proporcion'][1]:.1f} %, PC3 {100 * P['proporcion'][2]:.1f} %. "
           f"Silhouette de los 3 comandos: {s_tr:.3f} en train (originales k0) y {s_va:.3f} en val. Train se dibuja sin las copias "
           "aumentadas, para no tapar.")
    fig = plt.figure(figsize=(15.5, 6.6))
    ax = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    _pca_voz(ax, ax3, Z, v, P, f"2D · silhouette de los comandos: train {s_tr:.2f}, val {s_va:.2f}")
    leyenda_voz(ax, "lower right")
    ax3.set_xlabel(f"PC1 ({100 * P['proporcion'][0]:.0f} %)"); ax3.set_ylabel(f"PC2 ({100 * P['proporcion'][1]:.0f} %)")
    ax3.set_zlabel(f"PC3 ({100 * P['proporcion'][2]:.0f} %)"); ax3.set_title("3D", fontsize=10); ax3.view_init(22, -60)
    cerrar(fig, 13, "pca_voz", "PCA de la voz en 2D y 3D",
           "PCA ajustado con train. «salir» se separa; «reconocer» e «identificar» se pisan en 2D y en parte se separan en la 3.ª "
           "componente; «otro» se mezcla con todo. La información está repartida en muchas componentes.")


def paso_14_pesos_voz(E: dict) -> None:
    from src.core_ml.metricas import silhouette
    from src.voz.clasificacion import factores_esquema, reponderar_bloques
    encabezado(14, "Voz: el PCA con los pesos de V3 y con los de V4")
    v = E["v"]
    tr = v["particion"] == "train"
    Xz = reponderar_bloques(v["X"], factores_esquema("z_columna", cfg), cfg)
    resultados = []
    for nombre, Xb in [("pesos de V3 (cada bloque igual)", v["X"]), ("z_columna, elegido en V4 (cada columna igual)", Xz)]:
        P = pca_propio(Xb[tr], 20)
        Z = proyectar(P, Xb)
        corr = float(np.corrcoef(Z[tr, 0], v["duracion"][tr, 0])[0, 1])
        com = np.isin(v["clase"], cfg["voz"]["comandos"])
        k0 = tr & (v["copia"] == 0) & com
        s20 = silhouette(Z[k0], v["clase"][k0])
        resultados.append((nombre, P, Z, corr, s20))
    manzanita("Si en una receta un ingrediente tapa el gusto de todo lo demás, solo sentís ese ingrediente. Bajándole la "
              "cantidad a la duración, empiezan a notarse los otros: cómo suena cada palabra.")
    decir(f"La lectura del PCA sirvió para tomar una decisión. Con los pesos de V3, la primera componente es la duración "
          f"(correlación {abs(resultados[0][3]):.2f}). En V4 se probó dar el mismo peso a cada columna, z_columna: la duración "
          f"pasa a ser una columna entre 281 y deja de dominar (correlación {abs(resultados[1][3]):.2f}), y la validación cruzada "
          f"por persona eligió ese esquema. En el dibujo, «salir» queda bien aparte. «reconocer» e «identificar» todavía "
          f"se pisan en 2D: con 20 componentes, la silhouette de los comandos en train mejora poco, de "
          f"{resultados[0][4]:.2f} a {resultados[1][4]:.2f}. La elección no salió de este dibujo sino de la validación "
          f"cruzada.")
    hablar("; ".join(f"{n}: PC1 {100 * P['proporcion'][0]:.1f} %, correlación de PC1 con la duración {c:.3f}, silhouette de los "
                     f"comandos (train k0, PCA 20) {s:.3f}" for n, P, Z, c, s in resultados) + ".")
    fig, ejes = figura(1, 2, 15, 6)
    for ax, (nombre, P, Z, corr, s20) in zip(ejes[0], resultados):
        _pca_voz(ax, None, Z, v, P, f"{nombre}\nPC1 ~ duración: correlación {abs(corr):.2f}")
    leyenda_voz(ejes[0, 0], "lower right")
    cerrar(fig, 14, "pesos_voz", "Voz: el PCA con los pesos de V3 y con los de V4",
           "Con un peso por bloque, la duración domina PC1. Con el mismo peso por columna (lo que eligió V4 por validación cruzada), "
           "deja de dominar y «salir» queda bien aparte; «reconocer» e «identificar» siguen pisándose en 2D.")


# ============================================================================= guion y verificación
PREGUNTAS = [
    ("¿Por qué K-means si ya saben las identidades?",
     "Lo pide la consigna, y sirve para comprobar que las características separan solas: el K-means no ve las etiquetas y "
     "encontró exactamente las 3 identidades (pureza 100 %). Las identidades solo nombran los grupos al final."),
    ("¿Qué minimiza el K-means?",
     "La inercia: la suma de las distancias al cuadrado de cada huella al centro de su grupo. Lloyd la baja en cada vuelta "
     "(asignar al más cercano, mover a la media), así que siempre converge, aunque puede ser a un óptimo local: por eso los "
     "15 arranques."),
    ("¿Para qué k-means++?",
     "Para que los centros arranquen lejos entre sí (probabilidad ∝ distancia²). Arranca mejor y converge más rápido que "
     "elegir los 3 al azar."),
    ("¿Por qué k = 3?",
     "Lo fija la consigna (3 identidades autorizadas). Además, la inercia tiene el codo en 3 y la silhouette es máxima en 3."),
    ("¿Por qué la distancia normalizada?",
     "Porque cada grupo tiene su propia dispersión: la misma distancia euclídea es mucho para un grupo compacto y poco para "
     "uno disperso. Dividiendo por r_k, el umbral significa lo mismo para las tres identidades. En 600 huellas nuevas bajó "
     "el error de igual tasa de 0,7 % a 0,3 %."),
    ("¿Cómo eligieron τ*?",
     "Con validación: el criterio pide FAR ≤ 1 % (ningún intruso de 90); eso deja un intervalo sin errores [1,421; 1,702) y "
     "τ* va en la mitad, 1,561, para dejar margen hacia los dos lados."),
    ("¿Usar test para el umbral no habría sido mejor?",
     "Sería fuga de datos: test es el examen y se mira una sola vez, con todo fijo. τ* sale solo de validación."),
    ("¿Por qué falla id1_v036?",
     "Tiene un corte del 9 % y la escala reducida un 7 %: queda en 1,72, apenas arriba de τ*. Es 1 de 24 (FRR 4,2 %), dentro "
     "de la meta de 10 %."),
    ("¿Por qué no suben τ* para aceptar id1_v036?",
     "Porque sería ajustar con test, y porque el intruso más cercano de test (intruso49_v2) está en 1,720, casi en el mismo "
     "lugar que id1_v036 (1,721): para dejar pasar a una, entra el otro."),
    ("¿Qué es una componente principal?",
     "Una dirección nueva, combinación de las columnas originales, en la que los datos varían mucho. Son los autovectores de "
     "la covarianza; sus autovalores son la varianza que explica cada una."),
    ("¿Por qué ajustar el PCA solo con train?",
     "Para no usar información de validación ni de test. Se ajusta con train y después se proyecta todo con las mismas "
     "componentes."),
    ("¿Por qué en el PCA los intrusos caen al lado de una identidad y se rechazan igual?",
     "Las 2 primeras componentes separan sobre todo el tipo de huella, y los intrusos son de los mismos tipos. Se separan en "
     "las otras componentes; la decisión se toma en las 374 dimensiones, no en el dibujo."),
    ("¿Por qué la voz se ve tan mezclada en el PCA?",
     "Porque 2 componentes explican solo el 32 % de la varianza: la información que distingue las palabras está repartida. "
     "Por eso el K-NN usa 20 componentes. Aun así es la parte más difícil: con 4 personas nuevas acepta el 77,5 % de los "
     "comandos en test (la meta era 90 %), y en vivo, con otras 3, el 91,2 %."),
    ("¿El PCA es propio?",
     "Sí, src/core_ml/pca.py con NumPy (covarianza y eigh). Los tests lo comparan con scikit-learn: coincide hasta 1e-11."),
]


def escribir_guion() -> None:
    partes = ["# Clusterización y gráficos PCA, paso a paso\n",
              "Generado por `scripts/h4pca_paso_a_paso.py`, con los números de esta corrida. Por figura: la explicación con "
              "manzanitas, qué decir y el detalle.\n", "## Versión corta: una frase por figura\n"]
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
    from src.core_ml.kmeans import KMeans
    from src.core_ml.pca import PCA
    from src.huellas.clasificacion import cargar_modelo, correspondencia, evaluar, decidir as decidir_h
    print("\n" + "=" * 100 + "\nVerificación contra src/, el modelo guardado y scikit-learn\n" + "=" * 100)
    oks = []

    def informe(nombre: str, ok: bool, detalle: str = "") -> None:
        oks.append(bool(ok))
        print(f"   {'✔' if ok else '✘'} {nombre}{(': ' + detalle) if detalle else ''}")

    kc = cfg["huellas"]["kmeans"]
    km = KMeans(int(kc["k"]), int(kc["n_init"]), int(kc["max_iter"]), float(kc["tol"]), int(cfg["semilla_global"])).ajustar(E["Xtr"])
    C, et, J, it = E["mejor"]
    informe("K-means rehecho = src/core_ml/kmeans.py", np.allclose(C, km.centroides, atol=1e-12) and np.array_equal(et, km.etiquetas)
            and abs(J - km.inercia) < 1e-9, f"inercia {J:.4f}, {it} iteraciones")
    mapa, pureza = correspondencia(km.etiquetas, E["ytr"], int(kc["k"]))
    informe("correspondencia y pureza = src", mapa == E["mapa"] and abs(pureza - E["pureza"]) < 1e-12, f"pureza {pureza:.3f}")
    mh = cargar_modelo(RAIZ / cfg["rutas"]["modelos"] / "modelo_huellas.npz")
    informe("centroides = models/modelo_huellas.npz", np.allclose(E["Cid"], mh.centroides, atol=1e-9))
    informe("dispersiones r_k = modelo", np.allclose(E["r"], mh.dispersion, atol=1e-12), f"{np.round(mh.dispersion, 4).tolist()}")
    informe("τ* rehecho = modelo", abs(E["tau"] - mh.tau) < 1e-12, f"τ* = {mh.tau:.4f}")
    h = E["h"]
    te = h["particion"] == "test"
    s, p = mh.puntajes(h["X"][te])
    ev = evaluar(decidir_h(s, p, mh.tau), p, h["identidad"][te], int(kc["k"]))
    t = E["test"]
    informe("métricas de test = src (evaluar)", abs(ev["identificacion"] - t["identificacion"]) < 1e-12 and abs(ev["far"] - t["far"]) < 1e-12
            and abs(ev["frr"] - t["frr"]) < 1e-12 and np.array_equal(ev["matriz"], t["matriz"]))
    for nombre, Xtr, Pm in [("huellas", E["Xtr"], E["Ph_full"]), ("voz", E["v"]["X"][E["v"]["particion"] == "train"], E["Pv_full"])]:
        ps = PCA(20).ajustar(Xtr)
        dif = float(np.abs(ps.transformar(Xtr) - proyectar(Pm, Xtr)[:, :20]).max())
        informe(f"PCA a mano = src/core_ml/pca.py ({nombre}, 20 componentes)", dif < 1e-8, f"máx |dif| = {dif:.1e}")
        try:
            from sklearn.decomposition import PCA as PCAsk
            sk = PCAsk(n_components=20, svd_solver="full").fit(Xtr)
            Zs = sk.transform(Xtr)
            Zp = proyectar(Pm, Xtr)[:, :20]
            signos = np.sign(np.sum(Zs * Zp, axis=0))
            dif = float(np.abs(Zs * signos - Zp).max())
            informe(f"PCA a mano = scikit-learn ({nombre}, salvo el signo)", dif < 1e-8, f"máx |dif| = {dif:.1e}")
        except ImportError:
            print("   (scikit-learn no está instalado: se saltea la comparación)")
    return all(oks)


# ============================================================================= main
def main() -> int:
    global RAIZ, SALIDA, MOSTRAR, plt, cfg
    ap = argparse.ArgumentParser(description="H4 + PCA paso a paso: clusterización de las huellas y gráficos PCA.")
    ap.add_argument("--mostrar", action="store_true", help="abre cada figura y espera que la cierres")
    ap.add_argument("--verificar", action="store_true", help="compara con src/, el modelo guardado y scikit-learn")
    ap.add_argument("--salida", type=Path, help="carpeta de salida (default data/h4pca_paso_a_paso)")
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
    SALIDA = args.salida or (RAIZ / "data" / "h4pca_paso_a_paso")
    SALIDA.mkdir(parents=True, exist_ok=True)
    print(f"Clusterización y gráficos PCA, paso a paso\nFiguras en: {SALIDA}")
    E: dict = {}
    cargar(E)
    for paso in (paso_1_problema, paso_2_kmeanspp, paso_3_lloyd, paso_4_arranques, paso_5_correspondencia,
                 paso_6_normalizada, paso_7_umbral, paso_8_test, paso_9_que_es, paso_10_varianza, paso_11_pca_huellas,
                 paso_12_cargas, paso_13_pca_voz, paso_14_pesos_voz):
        paso(E)
    escribir_guion()
    print("\nListo. Mirá las figuras 01 a 14 en orden, o GUION.md, que las tiene con la explicación de cada una.")
    ok = True
    if args.verificar:
        ok = verificar(E)
        print("\n" + ("✔ Lo rehecho coincide con src/, el modelo guardado y scikit-learn." if ok else "✘ HAY DIFERENCIAS."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
