"""Prototipo de H4 (23/10/2026): K-means propio, correspondencia cluster → identidad y umbral.

No es el código final: sirve para medir sobre data/huellas/caracteristicas.npz (H3, tag
v0.3-huellas-features) antes de escribir docs/contratos/h4_*.md.
"""
from __future__ import annotations

import numpy as np

NO_AUTORIZADO = 0


# ---------------------------------------------------------------- K-means propio
def _kmeanspp(X, k, rng, n_ensayos=None):
    """k-means++ "greedy" (como scikit-learn): en cada paso se sortean 2 + ⌊ln k⌋ candidatos con
    probabilidad ∝ D² y se queda el que más baja la inercia."""
    n = len(X)
    if n_ensayos is None:
        n_ensayos = 2 + int(np.log(k))
    C = np.empty((k, X.shape[1]))
    i0 = rng.integers(n)
    C[0] = X[i0]
    d2 = ((X - C[0]) ** 2).sum(1)
    pot = d2.sum()
    for c in range(1, k):
        u = rng.random(n_ensayos) * pot
        cand = np.searchsorted(np.cumsum(d2), u)
        cand = np.minimum(cand, n - 1)
        d2c = ((X[None, :, :] - X[cand][:, None, :]) ** 2).sum(2)     # (n_ensayos, n)
        d2c = np.minimum(d2, d2c)
        pots = d2c.sum(1)
        mejor = int(np.argmin(pots))
        C[c] = X[cand[mejor]]
        d2 = d2c[mejor]
        pot = pots[mejor]
    return C


def _lloyd(X, C, max_iter, tol_abs):
    for it in range(1, max_iter + 1):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        etiq = d2.argmin(1)
        Cn = C.copy()
        for j in range(len(C)):
            m = etiq == j
            if m.any():
                Cn[j] = X[m].mean(0)
        mov = ((Cn - C) ** 2).sum()
        C = Cn
        if mov <= tol_abs:
            break
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
    etiq = d2.argmin(1)
    inercia = float(d2[np.arange(len(X)), etiq].sum())
    return C, etiq, inercia, it


def kmeans(X, k=3, n_init=15, max_iter=300, tol=1e-4, semilla=0, init=None):
    """Devuelve (centroides, etiquetas, inercia, iteraciones) del mejor arranque.
    tol como en scikit-learn: movimiento² total ≤ tol · media de las varianzas de las columnas."""
    X = np.asarray(X, float)
    tol_abs = tol * X.var(0).mean()
    rng = np.random.default_rng(semilla)
    mejor = None
    arranques = [np.asarray(init, float)] if init is not None else [None] * n_init
    for C0 in arranques:
        C = _kmeanspp(X, k, rng) if C0 is None else C0.copy()
        res = _lloyd(X, C, max_iter, tol_abs)
        if mejor is None or res[2] < mejor[2]:
            mejor = res
    return mejor


def correspondencia(etiquetas, identidades, k=3):
    """cluster → identidad mayoritaria del train; exige que sea uno a uno."""
    mapa, pureza = {}, []
    for j in range(k):
        ids, cnt = np.unique(identidades[etiquetas == j], return_counts=True)
        mapa[j] = int(ids[cnt.argmax()])
        pureza.append(cnt.max())
    uno_a_uno = len(set(mapa.values())) == k
    return mapa, sum(pureza) / len(etiquetas), uno_a_uno


# ---------------------------------------------------------------- decisión y métricas
def decidir(dist, pred, tau):
    """dist (n,) al centroide más cercano, pred (n,) su identidad; tau escalar o array por fila."""
    return np.where(dist <= tau, pred, NO_AUTORIZADO)


def metricas(decision, verdad):
    aut = verdad > 0
    frr = float(np.mean(decision[aut] != verdad[aut]))          # rechazada o mal identificada
    rech = float(np.mean(decision[aut] == 0))
    far = float(np.mean(decision[~aut] != 0)) if (~aut).any() else float("nan")
    return dict(far=far, frr=frr, rechazo_aut=rech)


def matriz_confusion(decision, verdad):
    M = np.zeros((4, 4), int)                    # filas: verdad (1, 2, 3, no aut.); columnas: decisión
    idx = lambda v: np.where(v == 0, 3, v - 1)
    np.add.at(M, (idx(verdad), idx(decision)), 1)
    return M


def barrido(score, pred, verdad):
    """FAR(τ) y FRR(τ) en los τ candidatos (cada score observado, más −∞)."""
    t = np.concatenate([[-np.inf], np.unique(score)])
    aut = verdad > 0
    far = np.array([np.mean(score[~aut] <= x) for x in t])
    frr = np.array([np.mean(~((score[aut] <= x) & (pred[aut] == verdad[aut]))) for x in t])
    return t, far, frr
