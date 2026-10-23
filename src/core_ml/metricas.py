"""Métricas propias con NumPy (compartidas por los carriles de huellas y de voz).

Contrato: docs/contratos/h3_caracteristicas.md (versión 1.1), sección 2.
"""

from __future__ import annotations

import numpy as np


def silhouette(X: np.ndarray, etiquetas) -> float:
    """Coeficiente de silhouette medio con distancia euclídea (Rousseeuw, 1987).

    Para cada fila i, a(i) es la distancia media a las otras filas de su grupo y b(i) el mínimo,
    entre los otros grupos, de la distancia media a sus filas; s(i) = (b − a) / max(a, b), y
    s(i) = 0 si su grupo tiene una sola fila o si max(a, b) = 0. Devuelve el promedio de s(i).

    Raises:
        ValueError: si X no es de 2 dimensiones o tiene valores no finitos, si los largos no
            coinciden o si la cantidad de etiquetas distintas no está entre 2 y n − 1.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X tiene que tener 2 dimensiones (n, d); tiene forma {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("X tiene valores NaN o infinitos.")
    etiquetas = np.asarray(etiquetas)
    n = X.shape[0]
    if etiquetas.ndim != 1 or etiquetas.shape[0] != n:
        raise ValueError(f"Hay {etiquetas.size} etiquetas para {n} filas de X.")
    unicas, codigos = np.unique(etiquetas, return_inverse=True)
    grupos = len(unicas)
    if not 2 <= grupos <= n - 1:
        raise ValueError(f"Hay {grupos} etiquetas distintas en {n} filas; tienen que ser entre 2 y {n - 1}.")

    # Distancias exactas fila por fila (sin el truco |x|² + |y|² − 2x·y, que pierde precisión).
    distancias = np.empty((n, n))
    for i in range(n):
        dif = X - X[i]
        distancias[i] = np.sqrt(np.einsum("ij,ij->i", dif, dif))
    pertenencia = np.zeros((n, grupos))
    pertenencia[np.arange(n), codigos] = 1.0
    sumas = distancias @ pertenencia                  # (n, grupos): suma de distancias a cada grupo
    tamanos = pertenencia.sum(axis=0)
    propio = tamanos[codigos]

    filas = np.arange(n)
    a = np.zeros(n)
    varias = propio > 1
    a[varias] = sumas[filas[varias], codigos[varias]] / (propio[varias] - 1)
    medias_otros = sumas / tamanos
    medias_otros[filas, codigos] = np.inf
    b = medias_otros.min(axis=1)

    mayor = np.maximum(a, b)
    s = np.zeros(n)
    usar = varias & (mayor > 0)
    s[usar] = (b[usar] - a[usar]) / mayor[usar]
    return float(s.mean())
