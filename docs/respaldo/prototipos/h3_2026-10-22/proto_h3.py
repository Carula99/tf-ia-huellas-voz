"""Prototipo de H3 (no es el código final): bloques de características sobre la ventana de H2."""
from __future__ import annotations
import math
import cv2
import numpy as np
from scipy import ndimage

LADO = 250
CENTRO = (LADO - 1) / 2

P = dict(
    grilla=5, cobertura_celda=0.5,
    gabor_periodos=(8.0, 10.0), gabor_orient=6,
    anillos=(20.0, 50.0, 80.0, 110.0), sectores=8, erosion_gabor=10, cobertura_sector=0.5,
    borde_minucia=14.0, espuela=12, par_terminaciones=10.0, par_bifurcaciones=8.0, componente_min=20,
    anillos_minucias=(0.0, 40.0, 75.0, 110.0), grilla_minucias=4,
)


def distancia_al_borde(mascara):
    """Distancia al borde de la máscara; el borde de la ventana también cuenta."""
    return ndimage.distance_transform_edt(np.pad(mascara, 1))[1:-1, 1:-1]


# ------------------------------------------------------------------ bloque 1: orientación en grilla
def bloque_orientacion(orientacion, mascara, p=P):
    n = p["grilla"]
    bordes = np.linspace(0, LADO, n + 1).round().astype(int)
    c2, s2 = np.cos(2 * orientacion), np.sin(2 * orientacion)
    out = []
    for i in range(n):
        for j in range(n):
            sl = (slice(bordes[i], bordes[i + 1]), slice(bordes[j], bordes[j + 1]))
            m = mascara[sl]
            if m.mean() < p["cobertura_celda"]:
                out += [np.nan, np.nan]
            else:
                out += [float(c2[sl][m].mean()), float(s2[sl][m].mean())]
    return np.array(out)


# ------------------------------------------------------------------ bloque 2: Gabor por sectores
def banco_gabor(p=P):
    banco = []
    for periodo in p["gabor_periodos"]:
        sigma = 0.5 * periodo
        lado = int(2 * round(3 * sigma) + 1)
        for k in range(p["gabor_orient"]):
            th = np.pi * k / p["gabor_orient"]
            g = cv2.getGaborKernel((lado, lado), sigma, th + np.pi / 2, periodo, 1.0, 0, ktype=cv2.CV_32F)
            g -= g.mean()
            g /= np.abs(g).sum()
            banco.append(g)
    return banco


_BANCO = None
yy, xx = np.mgrid[0:LADO, 0:LADO]
RADIO = np.hypot(xx - CENTRO, yy - CENTRO)
ANG = np.mod(np.arctan2(yy - CENTRO, xx - CENTRO), 2 * np.pi)


def sectores(p=P):
    an = p["anillos"]; ns = p["sectores"]
    out = []
    for a in range(len(an) - 1):
        for s in range(ns):
            out.append((RADIO >= an[a]) & (RADIO < an[a + 1]) & (ANG >= 2 * np.pi * s / ns) & (ANG < 2 * np.pi * (s + 1) / ns))
    return out


_SECT = None


def bloque_gabor(realzada, mascara, p=P):
    global _BANCO, _SECT
    if _BANCO is None:
        _BANCO = banco_gabor(p); _SECT = sectores(p)
    f = (255.0 - realzada.astype(np.float32))          # crestas positivas
    f[~mascara] = float(f[mascara].mean()) if mascara.any() else 0.0
    f -= f[mascara].mean()
    valida = distancia_al_borde(mascara) >= p["erosion_gabor"]
    resp = [np.abs(cv2.filter2D(f, -1, g, borderType=cv2.BORDER_REFLECT)) for g in _BANCO]
    aad, glob = [], []
    for r in resp:
        for sec in _SECT:
            z = sec & valida
            if z.sum() < p["cobertura_celda"] * sec.sum():
                aad.append(np.nan)
            else:
                aad.append(float(r[z].mean()))
    total = np.array([r[valida].mean() for r in resp])
    glob = list(total / total.sum())
    return np.array(aad), np.array(glob)


# ------------------------------------------------------------------ bloque 3: minucias
VEC = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # (dy, dx) en orden circular


def crossing_number(esq):
    e = np.pad(esq.astype(np.uint8), 1)
    vec = [e[1 + dy:e.shape[0] - 1 + dy, 1 + dx:e.shape[1] - 1 + dx] for dy, dx in VEC]
    cn = np.zeros(esq.shape, np.int32)
    for k in range(8):
        cn += np.abs(vec[k].astype(np.int32) - vec[(k + 1) % 8].astype(np.int32))
    cn = cn // 2
    vecinos = sum(v.astype(np.int32) for v in vec)
    return cn, vecinos


def recorrer(esq, y, x, largo):
    """Sigue la rama desde una terminación hasta `largo` px; devuelve el punto donde llega a una
    bifurcación (o None)."""
    previo = None; actual = (y, x)
    for _ in range(largo):
        sig = []
        for dy, dx in VEC:
            q = (actual[0] + dy, actual[1] + dx)
            if 0 <= q[0] < LADO and 0 <= q[1] < LADO and esq[q] and q != previo and q != actual:
                sig.append(q)
        if len(sig) == 0:
            return None
        if len(sig) >= 2:
            return actual   # llegó a una unión
        previo, actual = actual, sig[0]
    return None


def minucias(esqueleto, mascara, p=P, devolver_todo=False):
    esq = esqueleto.copy()
    # componentes muy chicas: afuera
    lab, n = ndimage.label(esq, structure=np.ones((3, 3)))
    if n:
        tam = ndimage.sum(esq, lab, index=np.arange(1, n + 1))
        chicas = np.isin(lab, 1 + np.nonzero(tam < p["componente_min"])[0])
        esq &= ~chicas
    cn, vecinos = crossing_number(esq)
    ter = np.argwhere(esq & (cn == 1))
    bif = np.argwhere(esq & (cn == 3))
    crudas = (len(ter), len(bif))
    dist = distancia_al_borde(mascara)
    # 1. borde de la máscara (incluye el borde recto de los cortes y el de la ventana)
    ter = [tuple(t) for t in ter if dist[tuple(t)] >= p["borde_minucia"]]
    bif = [tuple(b) for b in bif if dist[tuple(b)] >= p["borde_minucia"]]
    # 2. espuelas: terminación unida por una rama corta a una bifurcación
    bif_set = set(bif); quitar_t, quitar_b = set(), set()
    for t in ter:
        u = recorrer(esq, t[0], t[1], p["espuela"])
        if u is not None:
            quitar_t.add(t)
            cerca = [b for b in bif_set if abs(b[0] - u[0]) <= 2 and abs(b[1] - u[1]) <= 2]
            quitar_b.update(cerca)
    ter = [t for t in ter if t not in quitar_t]; bif = [b for b in bif if b not in quitar_b]
    # 3. pares de terminaciones cercanas (cresta cortada) y de bifurcaciones cercanas (puentes)
    def sin_pares(puntos, d):
        pts = np.array(puntos, dtype=float).reshape(-1, 2); fuera = set()
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if np.hypot(*(pts[i] - pts[j])) < d:
                    fuera.update((i, j))
        return [puntos[i] for i in range(len(puntos)) if i not in fuera]
    ter = sin_pares(ter, p["par_terminaciones"]); bif = sin_pares(bif, p["par_bifurcaciones"])
    if devolver_todo:
        return ter, bif, crudas
    return ter, bif


def bloque_minucias(esqueleto, mascara, p=P):
    ter, bif = minucias(esqueleto, mascara, p)
    valida = distancia_al_borde(mascara) >= p["borde_minucia"]
    area = max(valida.sum(), 1)
    out = [1000 * len(ter) / area, 1000 * len(bif) / area, len(bif) / max(len(ter) + len(bif), 1)]
    an = p["anillos_minucias"]
    for lista in (ter, bif):
        r = np.array([math.hypot(y - CENTRO, x - CENTRO) for y, x in lista]) if lista else np.zeros(0)
        for a in range(len(an) - 1):
            anillo = (RADIO >= an[a]) & (RADIO < an[a + 1])
            zona = valida & anillo
            if zona.sum() < p["cobertura_celda"] * anillo.sum():
                out.append(np.nan)
            else:
                out.append(1000 * np.sum((r >= an[a]) & (r < an[a + 1])) / zona.sum())
    # densidad en grilla (todas las minucias)
    n = p["grilla_minucias"]; b = np.linspace(0, LADO, n + 1).round().astype(int)
    todos = np.array(ter + bif, dtype=float).reshape(-1, 2)
    for i in range(n):
        for j in range(n):
            zona = valida[b[i]:b[i + 1], b[j]:b[j + 1]]
            if zona.mean() < 0.5:
                out.append(np.nan)
            else:
                dentro = np.sum((todos[:, 0] >= b[i]) & (todos[:, 0] < b[i + 1]) & (todos[:, 1] >= b[j]) & (todos[:, 1] < b[j + 1]))
                out.append(1000 * dentro / zona.sum())
    return np.array(out)


# ------------------------------------------------------------------ bloque 4: momentos de Hu
def bloque_hu(realzada, mascara, p=P):
    f = (255.0 - realzada.astype(np.float64)) / 255.0
    energia = cv2.GaussianBlur((f - 0.5) ** 2, (0, 0), 4.0)
    energia[~mascara] = 0
    hu = cv2.HuMoments(cv2.moments(energia.astype(np.float32))).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def vector(fila, p=P):
    o = bloque_orientacion(fila["orientacion"].astype(np.float64), fila["mascara"], p)
    g_aad, g_glob = bloque_gabor(fila["realzada"], fila["mascara"], p)
    m = bloque_minucias(fila["esqueleto"], fila["mascara"], p)
    h = bloque_hu(fila["realzada"], fila["mascara"], p)
    return {"orientacion": o, "gabor": np.concatenate([g_aad, g_glob]), "minucias": m, "hu": h}
