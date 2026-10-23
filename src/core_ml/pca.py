"""PCA propio con NumPy (compartido por los carriles de huellas y de voz).

Contrato: docs/contratos/h3_caracteristicas.md (versión 1.1), sección 2. Se ajusta con la
matriz de covarianza Σ = Xcᵀ·Xc / (n − 1) y sus autovectores (np.linalg.eigh), de mayor a menor
autovalor. El signo de cada componente queda fijo: su entrada de mayor valor absoluto es
positiva, así dos ajustes con los mismos datos dan lo mismo. Admite cualquier cantidad de
componentes y vectores sueltos, porque el carril de voz lo usa en V3.
"""

from __future__ import annotations

import numpy as np


def _es_entero(valor) -> bool:
    return isinstance(valor, (int, np.integer)) and not isinstance(valor, (bool, np.bool_))


class PCA:
    """Análisis de componentes principales.

    Atributos después de ajustar (float64):
        media: (d,)               media de cada columna de X
        componentes: (k, d)       una componente por fila, ortonormales
        varianza: (k,)            autovalores de Σ, de mayor a menor
        proporcion_varianza: (k,) varianza / traza(Σ)
    """

    def __init__(self, n_componentes: int | None = None) -> None:
        if n_componentes is not None and not (_es_entero(n_componentes) and n_componentes >= 1):
            raise ValueError(f"n_componentes tiene que ser None o un entero >= 1; es {n_componentes!r}.")
        self.n_componentes = None if n_componentes is None else int(n_componentes)
        self.media: np.ndarray | None = None
        self.componentes: np.ndarray | None = None
        self.varianza: np.ndarray | None = None
        self.proporcion_varianza: np.ndarray | None = None

    def ajustar(self, X: np.ndarray) -> "PCA":
        """Calcula la media, las componentes y la varianza de X (n, d). Devuelve self."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X tiene que tener 2 dimensiones (n, d); tiene forma {X.shape}.")
        n, d = X.shape
        if n < 2 or d < 1:
            raise ValueError(f"X tiene que tener al menos 2 filas y 1 columna; tiene forma {X.shape}.")
        X = X.astype(np.float64)          # copia: X no se modifica
        if not np.isfinite(X).all():
            raise ValueError("X tiene valores NaN o infinitos.")
        if np.all(X == X[0]):
            raise ValueError("Todas las filas de X son iguales: no hay varianza para descomponer.")
        k_max = min(n - 1, d)
        k = k_max if self.n_componentes is None else self.n_componentes
        if k > k_max:
            raise ValueError(f"n_componentes = {k} es mayor que min(n − 1, d) = {k_max} (X de {n} × {d}).")

        media = X.mean(axis=0)
        Xc = X - media
        sigma = Xc.T @ Xc / (n - 1)
        traza = float(np.trace(sigma))
        if traza == 0.0:
            raise ValueError("traza(Σ) = 0: todas las filas de X son iguales.")

        autovalores, autovectores = np.linalg.eigh(sigma)     # de menor a mayor
        varianza = autovalores[::-1][:k]
        componentes = autovectores[:, ::-1][:, :k].T.copy()
        mayor = np.argmax(np.abs(componentes), axis=1)        # la primera, si hay empate
        signos = np.where(componentes[np.arange(k), mayor] < 0, -1.0, 1.0)
        componentes *= signos[:, None]

        self.media = media
        self.componentes = componentes
        self.varianza = np.maximum(varianza, 0.0)             # los negativos por redondeo valen 0
        self.proporcion_varianza = self.varianza / traza
        return self

    def transformar(self, X: np.ndarray) -> np.ndarray:
        """(X − media) · componentesᵀ: (n, d) → (n, k), o un vector (d,) → (k,)."""
        if self.componentes is None:
            raise RuntimeError("Hay que llamar a ajustar antes de transformar.")
        X = np.asarray(X)
        d = self.componentes.shape[1]
        if X.ndim not in (1, 2) or X.shape[-1] != d:
            raise ValueError(f"X tiene que ser (n, {d}) o ({d},); tiene forma {X.shape}.")
        X = X.astype(np.float64)
        if not np.isfinite(X).all():
            raise ValueError("X tiene valores NaN o infinitos.")
        return (X - self.media) @ self.componentes.T

    def ajustar_transformar(self, X: np.ndarray) -> np.ndarray:
        """ajustar(X).transformar(X)."""
        return self.ajustar(X).transformar(X)
