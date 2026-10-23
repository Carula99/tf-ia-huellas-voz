"""Prototipo 2 de H3, con las definiciones exactas del contrato (no es el código final)."""
from __future__ import annotations
import math
from dataclasses import dataclass
import cv2
import numpy as np
from scipy import ndimage

LADO = 250
CENTRO = (LADO - 1) / 2
BLOQUES = ("orientacion", "gabor", "minucias")
PAR_TERMINACIONES_PX = 10.0
PAR_BIFURCACIONES_PX = 8.0
COMPONENTE_MIN_PX = 20

C = dict(grilla_orientacion=5, cobertura_min=0.5, gabor_periodos_px=[8.0, 10.0], gabor_orientaciones=6,
         anillos_gabor_px=[20.0, 50.0, 80.0, 110.0], sectores_gabor=8, erosion_gabor_px=10.0,
         borde_minucia_px=14.0, espuela_px=12, anillos_minucias_px=[0.0, 40.0, 75.0, 110.0], grilla_minucias=4)

_yy, _xx = np.mgrid[0:LADO, 0:LADO]
RADIO = np.hypot(_xx - CENTRO, _yy - CENTRO)
FI = np.mod(np.arctan2(_yy - CENTRO, _xx - CENTRO), 2 * np.pi)


def distancia_al_borde(mascara):
    return ndimage.distance_transform_edt(np.pad(mascara, 1))[1:-1, 1:-1]


def bordes_grilla(g):
    return np.round(np.linspace(0, LADO, g + 1)).astype(int)


def largos(c=C):
    g = c["grilla_orientacion"]; F = len(c["gabor_periodos_px"]) * c["gabor_orientaciones"]
    R = len(c["anillos_gabor_px"]) - 1; S = c["sectores_gabor"]
    Rm = len(c["anillos_minucias_px"]) - 1; gm = c["grilla_minucias"]
    return {"orientacion": 2 * g * g, "gabor": F * R * S + F, "minucias": 3 + 2 * Rm + gm * gm}


def bloque_orientacion(h, c=C):
    b = bordes_grilla(c["grilla_orientacion"]); out = []
    c2, s2 = np.cos(2 * h.orientacion), np.sin(2 * h.orientacion)
    for i in range(len(b) - 1):
        for j in range(len(b) - 1):
            sl = (slice(b[i], b[i + 1]), slice(b[j], b[j + 1])); m = h.mascara[sl]
            if m.mean() < c["cobertura_min"]:
                out += [np.nan, np.nan]
            else:
                out += [c2[sl][m].mean(), s2[sl][m].mean()]
    return np.array(out, dtype=np.float64)


def nucleos_gabor(c=C):
    out = []
    for P in c["gabor_periodos_px"]:
        sigma = P / 2; L = 2 * int(round(3 * sigma)) + 1
        for k in range(c["gabor_orientaciones"]):
            g = cv2.getGaborKernel((L, L), sigma, math.pi * k / c["gabor_orientaciones"] + math.pi / 2, P, 1.0, 0.0, ktype=cv2.CV_64F)
            g = g - g.mean(); g = g / np.abs(g).sum(); out.append(g)
    return out


def sectores(c=C):
    an, S = c["anillos_gabor_px"], c["sectores_gabor"]
    return [((RADIO >= an[a]) & (RADIO < an[a + 1]) & (FI >= 2 * math.pi * s / S) & (FI < 2 * math.pi * (s + 1) / S))
            for a in range(len(an) - 1) for s in range(S)]


def bloque_gabor(h, c=C):
    f = 255.0 - h.realzada.astype(np.float64)
    media = f[h.mascara].mean(); f[~h.mascara] = media; f = f - media
    G = distancia_al_borde(h.mascara) >= c["erosion_gabor_px"]
    secs = sectores(c); aad, tot = [], []
    for g in nucleos_gabor(c):
        r = np.abs(cv2.filter2D(f, -1, g, borderType=cv2.BORDER_REFLECT))
        for sec in secs:
            z = sec & G
            aad.append(np.nan if z.sum() < c["cobertura_min"] * sec.sum() else r[z].mean())
        tot.append(r[G].mean())
    tot = np.array(tot)
    return np.concatenate([np.array(aad), tot / tot.sum()])


VEC = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def crossing_number(esq):
    e = np.pad(esq.astype(np.int32), 1)
    v = [e[1 + dy:e.shape[0] - 1 + dy, 1 + dx:e.shape[1] - 1 + dx] for dy, dx in VEC]
    return sum(np.abs(v[k] - v[(k + 1) % 8]) for k in range(8)) // 2


def _recorrer(esq, y, x, largo):
    previo, actual = None, (y, x)
    for _ in range(largo):
        sig = [(actual[0] + dy, actual[1] + dx) for dy, dx in VEC
               if 0 <= actual[0] + dy < LADO and 0 <= actual[1] + dx < LADO and esq[actual[0] + dy, actual[1] + dx]
               and (actual[0] + dy, actual[1] + dx) != previo]
        if not sig:
            return None
        if len(sig) >= 2:
            return actual
        previo, actual = actual, sig[0]
    return None


@dataclass(frozen=True)
class Minucias:
    terminaciones: np.ndarray
    bifurcaciones: np.ndarray


def extraer_minucias(h, c=C):
    esq = h.esqueleto.copy()
    lab, n = ndimage.label(esq, structure=np.ones((3, 3)))
    if n:
        tam = ndimage.sum(esq, lab, index=np.arange(1, n + 1))
        esq &= ~np.isin(lab, 1 + np.nonzero(tam < COMPONENTE_MIN_PX)[0])
    cn = crossing_number(esq)
    D = distancia_al_borde(h.mascara)
    ter = [tuple(p) for p in np.argwhere(esq & (cn == 1)) if D[tuple(p)] >= c["borde_minucia_px"]]
    bif = [tuple(p) for p in np.argwhere(esq & (cn == 3)) if D[tuple(p)] >= c["borde_minucia_px"]]
    fuera_t, fuera_b = set(), set()
    for t in ter:
        u = _recorrer(esq, t[0], t[1], c["espuela_px"])
        if u is not None:
            fuera_t.add(t); fuera_b.update(b for b in bif if abs(b[0] - u[0]) <= 2 and abs(b[1] - u[1]) <= 2)
    ter = [t for t in ter if t not in fuera_t]; bif = [b for b in bif if b not in fuera_b]

    def sin_pares(pts, d):
        a = np.array(pts, dtype=float).reshape(-1, 2)
        if len(a) < 2:
            return pts
        dd = np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1]); np.fill_diagonal(dd, np.inf)
        return [pts[i] for i in range(len(pts)) if dd[i].min() >= d]
    ter = sin_pares(ter, PAR_TERMINACIONES_PX); bif = sin_pares(bif, PAR_BIFURCACIONES_PX)
    a_xy = lambda pts: np.array([(float(x), float(y)) for y, x in pts], dtype=np.float64).reshape(-1, 2)
    return Minucias(a_xy(ter), a_xy(bif))


def bloque_minucias(h, m: Minucias, c=C):
    V = distancia_al_borde(h.mascara) >= c["borde_minucia_px"]
    nt, nb = len(m.terminaciones), len(m.bifurcaciones)
    area = V.sum()
    out = [1000 * nt / area, 1000 * nb / area, nb / (nt + nb) if nt + nb else 0.0] if area else [np.nan] * 3
    an = c["anillos_minucias_px"]
    for pts in (m.terminaciones, m.bifurcaciones):
        r = np.hypot(pts[:, 0] - CENTRO, pts[:, 1] - CENTRO)
        for a in range(len(an) - 1):
            anillo = (RADIO >= an[a]) & (RADIO < an[a + 1]); z = (anillo & V).sum()
            out.append(np.nan if z < c["cobertura_min"] * anillo.sum() else 1000 * np.sum((r >= an[a]) & (r < an[a + 1])) / z)
    b = bordes_grilla(c["grilla_minucias"])
    todos = np.vstack([m.terminaciones, m.bifurcaciones])
    for i in range(len(b) - 1):
        for j in range(len(b) - 1):
            z = V[b[i]:b[i + 1], b[j]:b[j + 1]].sum(); celda = (b[i + 1] - b[i]) * (b[j + 1] - b[j])
            if z < c["cobertura_min"] * celda:
                out.append(np.nan)
            else:
                dentro = np.sum((todos[:, 1] >= b[i]) & (todos[:, 1] < b[i + 1]) & (todos[:, 0] >= b[j]) & (todos[:, 0] < b[j + 1]))
                out.append(1000 * dentro / z)
    return np.array(out, dtype=np.float64)


def caracteristicas(h, c=C):
    m = extraer_minucias(h, c)
    return {"orientacion": bloque_orientacion(h, c), "gabor": bloque_gabor(h, c), "minucias": bloque_minucias(h, m, c)}


@dataclass(frozen=True)
class Escalador:
    bloques: tuple
    medias: dict
    desvios: dict
    columnas: dict
    pesos: dict

    def transformar(self, bloques):
        uno = np.ndim(bloques[self.bloques[0]]) == 1
        partes = []
        for b in self.bloques:
            A = np.atleast_2d(np.asarray(bloques[b], dtype=np.float64))
            Z = (A[:, self.columnas[b]] - self.medias[b][self.columnas[b]]) / self.desvios[b][self.columnas[b]]
            Z[np.isnan(Z)] = 0.0
            partes.append(Z * self.pesos[b])
        out = np.hstack(partes)
        return out[0] if uno else out


def ajustar_escalador(bloques_train, bloques=BLOQUES):
    medias, desvios, columnas, pesos = {}, {}, {}, {}
    for b in bloques:
        A = np.asarray(bloques_train[b], dtype=np.float64)
        finitos = np.isfinite(A).sum(0)
        with np.errstate(all="ignore"):
            mu = np.nanmean(A, 0); sd = np.nanstd(A, 0, ddof=1)
        ok = (finitos >= 2) & np.isfinite(sd) & (sd > 1e-9)
        medias[b], desvios[b], columnas[b] = mu, sd, ok
        pesos[b] = 1 / math.sqrt(ok.sum())
    return Escalador(tuple(bloques), medias, desvios, columnas, pesos)


class PCA:
    def __init__(self, n_componentes=None):
        self.n_componentes = n_componentes

    def ajustar(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        k = self.n_componentes or min(n - 1, d)
        self.media = X.mean(0); Xc = X - self.media
        S = Xc.T @ Xc / (n - 1)
        val, vec = np.linalg.eigh(S)
        orden = np.argsort(val)[::-1][:k]
        comp = vec[:, orden].T
        signo = np.sign(comp[np.arange(k), np.argmax(np.abs(comp), axis=1)])
        self.componentes = comp * signo[:, None]
        self.varianza = val[orden]
        self.proporcion_varianza = self.varianza / np.trace(S)
        return self

    def transformar(self, X):
        return (np.asarray(X, dtype=np.float64) - self.media) @ self.componentes.T


def silhouette(X, y):
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    etiquetas = np.unique(y); s = np.zeros(len(X))
    for i in range(len(X)):
        propio = y == y[i]
        if propio.sum() == 1:
            continue
        a = D[i, propio].sum() / (propio.sum() - 1)
        b = min(D[i, y == e].mean() for e in etiquetas if e != y[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())
