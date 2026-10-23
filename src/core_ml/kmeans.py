"""K-means propio con NumPy (k-means++ "greedy" y Lloyd).

Contrato: docs/contratos/h4_clasificacion.md (versión 1.1), secciones 2 y 4.1. Lo usa H4 para
agrupar las huellas de train en k = 3 clusters; está en core_ml porque no depende de las huellas.

- Tolerancia como en scikit-learn: tol_abs = tol · media(np.var(X, axis=0)).
- Sin init: n_init arranques en orden con un único generador np.random.default_rng(semilla);
  cada uno empieza con k-means++ "greedy" (Arthur y Vassilvitskii, 2007): 2 + ⌊ln k⌋ filas
  candidatas sorteadas con probabilidad ∝ D² y queda la que más baja Σ min D².
- Lloyd: cada fila al centroide más cercano y cada centroide a la media de sus filas (un cluster
  sin filas conserva su centroide). Para si las etiquetas no cambian respecto de la iteración
  anterior o si el movimiento² total de los centroides es ≤ tol_abs.
- Al final, las etiquetas se recalculan con los centroides finales y queda el arranque de menor
  inercia (el primero, si empatan).
"""

from __future__ import annotations

import math

import numpy as np


def _es_entero(valor) -> bool:
    return isinstance(valor, (int, np.integer)) and not isinstance(valor, (bool, np.bool_))


def _distancias2(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Distancias² exactas (n, k), sin el truco |x|² + |c|² − 2x·c, que pierde precisión."""
    D2 = np.empty((X.shape[0], C.shape[0]))
    for j in range(C.shape[0]):
        dif = X - C[j]
        D2[:, j] = np.einsum("ij,ij->i", dif, dif)
    return D2


class KMeans:
    """K-means con inicialización k-means++ "greedy" y n_init arranques.

    Atributos después de ajustar:
        centroides: (k, d) float64
        etiquetas: (n,) int64           cluster de cada fila de X (0 a k − 1)
        inercia: float                  Σ de las distancias² de cada fila a su centroide
        inercia_cluster: (k,) float64   la misma suma, por cluster
        n_iter: int                     iteraciones de Lloyd del arranque elegido
    """

    def __init__(self, k: int, n_init: int = 15, max_iter: int = 300, tol: float = 1e-4,
                 semilla: int | None = None) -> None:
        for nombre, valor in (("k", k), ("n_init", n_init), ("max_iter", max_iter)):
            if not (_es_entero(valor) and valor >= 1):
                raise ValueError(f"{nombre} tiene que ser un entero >= 1; es {valor!r}.")
        if (isinstance(tol, (bool, np.bool_)) or not isinstance(tol, (int, float, np.integer, np.floating))
                or not math.isfinite(tol) or tol < 0):
            raise ValueError(f"tol tiene que ser un número real finito >= 0; es {tol!r}.")
        if semilla is not None and not _es_entero(semilla):
            raise ValueError(f"semilla tiene que ser None o un entero; es {semilla!r}.")
        self.k = int(k)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.semilla = None if semilla is None else int(semilla)
        self.centroides: np.ndarray | None = None
        self.etiquetas: np.ndarray | None = None
        self.inercia: float | None = None
        self.inercia_cluster: np.ndarray | None = None
        self.n_iter: int | None = None

    # ------------------------------------------------------------------ ajuste
    def _kmeanspp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = X.shape[0]
        ensayos = 2 + int(math.log(self.k))
        C = np.empty((self.k, X.shape[1]))
        C[0] = X[rng.integers(n)]
        d2 = _distancias2(X, C[:1])[:, 0]
        for c in range(1, self.k):
            acumulada = np.cumsum(d2)
            sorteo = rng.random(ensayos) * acumulada[-1]
            # side="right": una fila con D² = 0 (ya elegida o repetida) nunca sale sorteada
            candidatas = np.minimum(np.searchsorted(acumulada, sorteo, side="right"), n - 1)
            d2_candidatas = np.minimum(d2[None, :], _distancias2(X, X[candidatas]).T)   # (ensayos, n)
            mejor = int(np.argmin(d2_candidatas.sum(axis=1)))
            C[c] = X[candidatas[mejor]]
            d2 = d2_candidatas[mejor]
        return C

    def _lloyd(self, X: np.ndarray, C: np.ndarray, tol_abs: float):
        etiquetas_antes = None
        iteracion = 0
        for iteracion in range(1, self.max_iter + 1):
            etiquetas = np.argmin(_distancias2(X, C), axis=1)
            nuevos = C.copy()
            for j in range(self.k):
                filas = etiquetas == j
                if filas.any():
                    nuevos[j] = X[filas].mean(axis=0)
            movimiento = float(np.sum((nuevos - C) ** 2))
            C = nuevos
            if (etiquetas_antes is not None and np.array_equal(etiquetas, etiquetas_antes)) or movimiento <= tol_abs:
                break
            etiquetas_antes = etiquetas
        D2 = _distancias2(X, C)
        etiquetas = np.argmin(D2, axis=1).astype(np.int64)
        d2 = D2[np.arange(X.shape[0]), etiquetas]
        inercia_cluster = np.array([float(d2[etiquetas == j].sum()) for j in range(self.k)], dtype=np.float64)
        return C, etiquetas, float(d2.sum()), inercia_cluster, int(iteracion)

    def ajustar(self, X: np.ndarray, init: np.ndarray | None = None) -> "KMeans":
        """Agrupa las filas de X (n, d) en k clusters (sección 4.1). Devuelve self."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X tiene que tener 2 dimensiones (n, d); tiene forma {X.shape}.")
        n, d = X.shape
        if n < self.k or d < 1:
            raise ValueError(f"X tiene que tener al menos k = {self.k} filas y 1 columna; tiene forma {X.shape}.")
        X = X.astype(np.float64)          # copia: X no se modifica
        if not np.isfinite(X).all():
            raise ValueError("X tiene valores NaN o infinitos.")
        if np.unique(X, axis=0).shape[0] < self.k:
            raise ValueError(f"X tiene menos de k = {self.k} filas distintas.")
        if init is not None:
            init = np.array(init, dtype=np.float64)
            if init.shape != (self.k, d) or not np.isfinite(init).all():
                raise ValueError(f"init tiene que ser ({self.k}, {d}) y finito; tiene forma {init.shape}.")
        tol_abs = self.tol * float(np.mean(np.var(X, axis=0)))

        if init is not None:
            mejor = self._lloyd(X, init, tol_abs)
        else:
            rng = np.random.default_rng(self.semilla)
            mejor = None
            for _ in range(self.n_init):
                resultado = self._lloyd(X, self._kmeanspp(X, rng), tol_abs)
                if mejor is None or resultado[2] < mejor[2]:
                    mejor = resultado
        self.centroides, self.etiquetas, self.inercia, self.inercia_cluster, self.n_iter = mejor
        return self

    # ------------------------------------------------------------------ uso
    def distancias(self, X: np.ndarray) -> np.ndarray:
        """Distancias euclídeas a cada centroide: (n, d) → (n, k), o (d,) → (k,)."""
        if self.centroides is None:
            raise RuntimeError("El K-means no está ajustado: llamá antes a ajustar(X).")
        X = np.asarray(X, dtype=np.float64)
        vector = X.ndim == 1
        X2 = X[None, :] if vector else X
        if X2.ndim != 2 or X2.shape[1] != self.centroides.shape[1]:
            raise ValueError(f"X tiene forma {X.shape}; se esperan {self.centroides.shape[1]} columnas.")
        if not np.isfinite(X2).all():
            raise ValueError("X tiene valores NaN o infinitos.")
        D = np.sqrt(_distancias2(X2, self.centroides))
        return D[0] if vector else D

    def predecir(self, X: np.ndarray):
        """Índice del centroide más cercano: (n, d) → (n,) int64, o (d,) → int."""
        D = self.distancias(X)
        if D.ndim == 1:
            return int(np.argmin(D))
        return np.argmin(D, axis=1).astype(np.int64)
