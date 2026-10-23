"""Prototipo de H4 con las definiciones exactas del contrato (h4_clasificacion.md, versión 1).

No es el código final. Sirve para medir los valores de la sección 8 del contrato.
"""
from __future__ import annotations

import math
import numpy as np

NO_AUTORIZADO = 0


def _es_entero(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


# ------------------------------------------------------------------ core_ml/kmeans.py
class KMeans:
    def __init__(self, k, n_init=15, max_iter=300, tol=1e-4, semilla=None):
        if not (_es_entero(k) and k >= 1):
            raise ValueError("k")
        if not (_es_entero(n_init) and n_init >= 1):
            raise ValueError("n_init")
        if not (_es_entero(max_iter) and max_iter >= 1):
            raise ValueError("max_iter")
        if isinstance(tol, bool) or not isinstance(tol, (int, float, np.floating)) or not (tol >= 0) or not math.isfinite(tol):
            raise ValueError("tol")
        self.k, self.n_init, self.max_iter, self.tol, self.semilla = int(k), int(n_init), int(max_iter), float(tol), semilla
        self.centroides = None

    @staticmethod
    def _d2(X, C):
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)

    def _kmeanspp(self, X, rng):
        n, k = len(X), self.k
        m = 2 + int(math.log(k))
        C = np.empty((k, X.shape[1]))
        C[0] = X[rng.integers(n)]
        d2 = ((X - C[0]) ** 2).sum(1)
        for c in range(1, k):
            acum = np.cumsum(d2)
            u = rng.random(m) * acum[-1]
            cand = np.minimum(np.searchsorted(acum, u, side="right"), n - 1)
            d2c = np.minimum(d2[None, :], self._d2(X, X[cand]).T)       # (m, n)
            mejor = int(np.argmin(d2c.sum(1)))
            C[c] = X[cand[mejor]]
            d2 = d2c[mejor]
        return C

    def _lloyd(self, X, C, tol_abs):
        etiq_ant = None
        for it in range(1, self.max_iter + 1):
            etiq = self._d2(X, C).argmin(1)
            Cn = C.copy()
            for j in range(self.k):
                m = etiq == j
                if m.any():
                    Cn[j] = X[m].mean(0)
            mov = float(((Cn - C) ** 2).sum())
            C = Cn
            if (etiq_ant is not None and np.array_equal(etiq, etiq_ant)) or mov <= tol_abs:
                break
            etiq_ant = etiq
        d2 = self._d2(X, C)
        etiq = d2.argmin(1)
        return C, etiq, float(d2[np.arange(len(X)), etiq].sum()), it

    def ajustar(self, X, init=None):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < self.k or X.shape[1] < 1:
            raise ValueError("X")
        X = X.astype(np.float64)
        if not np.isfinite(X).all():
            raise ValueError("X no finito")
        if len(np.unique(X, axis=0)) < self.k:
            raise ValueError("menos filas distintas que k")
        tol_abs = self.tol * float(X.var(0).mean())
        if init is not None:
            init = np.asarray(init, dtype=np.float64)
            if init.shape != (self.k, X.shape[1]) or not np.isfinite(init).all():
                raise ValueError("init")
            arranques = [init.copy()]
        else:
            rng = np.random.default_rng(self.semilla)
            arranques = None
        mejor = None
        for r in range(1 if init is not None else self.n_init):
            C0 = arranques[0] if init is not None else self._kmeanspp(X, rng)
            res = self._lloyd(X, C0, tol_abs)
            if mejor is None or res[2] < mejor[2]:
                mejor = res
        self.centroides, self.etiquetas, self.inercia, self.n_iter = mejor
        self.inercia_cluster = np.array([float(((X[self.etiquetas == j] - self.centroides[j]) ** 2).sum()) for j in range(self.k)])
        return self

    def distancias(self, X):
        if self.centroides is None:
            raise RuntimeError("sin ajustar")
        X = np.asarray(X, dtype=np.float64)
        v = X.ndim == 1
        X2 = X[None] if v else X
        if X2.ndim != 2 or X2.shape[1] != self.centroides.shape[1] or not np.isfinite(X2).all():
            raise ValueError("X")
        D = np.sqrt(self._d2(X2, self.centroides))
        return D[0] if v else D

    def predecir(self, X):
        D = self.distancias(X)
        return D.argmin(-1)


# ------------------------------------------------------------------ core_ml/metricas.py (se suma)
def matriz_confusion(verdad, prediccion, etiquetas):
    verdad, prediccion, etiquetas = list(verdad), list(prediccion), list(etiquetas)
    if len(verdad) != len(prediccion) or len(set(etiquetas)) != len(etiquetas) or not etiquetas:
        raise ValueError("largos o etiquetas")
    pos = {e: i for i, e in enumerate(etiquetas)}
    M = np.zeros((len(etiquetas), len(etiquetas)), dtype=np.int64)
    for v, p in zip(verdad, prediccion):
        if v not in pos or p not in pos:
            raise ValueError("valor fuera de etiquetas")
        M[pos[v], pos[p]] += 1
    return M


# ------------------------------------------------------------------ huellas/clasificacion.py
def correspondencia(etiquetas, identidades, k):
    mapa, aciertos = {}, 0
    for j in range(k):
        ids = identidades[etiquetas == j]
        if ids.size == 0:
            raise ValueError(f"cluster {j} vacío")
        u, c = np.unique(ids, return_counts=True)
        mapa[j] = int(u[np.argmax(c)])            # empate: la identidad menor (np.unique ordena)
        aciertos += int(c.max())
    if len(set(mapa.values())) != k:
        raise ValueError("no es uno a uno")
    return mapa, aciertos / len(etiquetas)


def dispersiones(X, etiquetas, centroides, piso):
    k = len(centroides)
    I_k = np.array([((X[etiquetas == j] - centroides[j]) ** 2).sum() for j in range(k)])
    n_k = np.array([(etiquetas == j).sum() for j in range(k)])
    r_bar = math.sqrt(I_k.sum() / len(X))
    if r_bar == 0:
        raise ValueError("sin dispersión")
    return np.maximum(np.sqrt(I_k / n_k), piso * r_bar)


def puntajes(X, centroides, dispersion, identidades):
    X = np.asarray(X, dtype=np.float64)
    v = X.ndim == 1
    X2 = X[None] if v else X
    Dn = np.sqrt(((X2[:, None, :] - centroides[None]) ** 2).sum(2)) / dispersion
    j = Dn.argmin(1)
    s, p = Dn[np.arange(len(X2)), j], identidades[j]
    return (float(s[0]), int(p[0])) if v else (s, p)


def far_frr(puntaje, prediccion, verdad, tau):
    aut = verdad != NO_AUTORIZADO
    acepta = puntaje <= tau
    far = float(np.mean(acepta[~aut]))
    frr = float(np.mean(~(acepta[aut] & (prediccion[aut] == verdad[aut]))))
    return far, frr


def calibrar_umbral(puntaje, prediccion, verdad, criterio, far_max_val):
    s, p, y = (np.asarray(a) for a in (puntaje, prediccion, verdad))
    aut = y != NO_AUTORIZADO
    if not aut.any() or aut.all():
        raise ValueError("faltan autorizadas o intrusos")
    ok = aut & (p == y)                                   # autorizadas bien identificadas
    puntos = np.unique(s)                                 # quiebres de FAR y FRR
    if criterio == "far_max":
        si = np.sort(s[~aut])
        m = int(math.floor(far_max_val * len(si) + 1e-9))
        b = si[m] if m < len(si) else np.inf
        abajo = s[ok & (s < b)]
        if abajo.size == 0:
            raise ValueError("ninguna autorizada bien identificada por debajo del primer intruso que no se puede aceptar")
        a = abajo.max()
    elif criterio == "eer":
        cand = np.concatenate([[-np.inf], puntos])        # FAR y FRR son constantes en [cand_i, cand_i+1)
        E = np.array([max(*far_frr(s, p, y, t)) for t in cand])
        idx = np.nonzero(E == E.min())[0]
        a = cand[idx[0]]
        b = cand[idx[-1] + 1] if idx[-1] + 1 < len(cand) else np.inf
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError("el error mínimo queda en un extremo")
    else:
        raise ValueError("criterio")
    if not np.isfinite(b):
        raise ValueError("sin intrusos por encima")
    return float((a + b) / 2)


def decidir(puntaje, prediccion, tau):
    return np.where(np.asarray(puntaje) <= tau, prediccion, NO_AUTORIZADO)


def evaluar(decision, prediccion, verdad, k=3):
    aut = verdad != NO_AUTORIZADO
    etiquetas = list(range(1, k + 1)) + [NO_AUTORIZADO]
    return dict(identificacion=float(np.mean(prediccion[aut] == verdad[aut])),
                far=float(np.mean(decision[~aut] != NO_AUTORIZADO)),
                frr=float(np.mean(decision[aut] != verdad[aut])),
                matriz=matriz_confusion(verdad, decision, etiquetas))
