"""Fase 5 · El visor PCA 2D de entrenamiento.

Contrato: docs/contratos/f5_integracion.md (versión 1.1), secciones 3.2 y 4.5. PCA propio (H3) de 2
componentes ajustado con las filas de train de caracteristicas.npz: los mismos ejes PC1 y PC2 de la
figura de H3 (pca_h3.png). La app marca ahí la huella evaluada. No se dibujan círculos de umbral:
τ* se aplica en el espacio completo con la distancia normalizada de H4, y la proyección achica las
distancias; la comparación con τ* va en una barra aparte.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from ..config import RAIZ
from ..core_ml.pca import PCA
from ..huellas.clasificacion import ModeloHuellas

NOMBRE_NPZ = "caracteristicas.npz"


@dataclass(frozen=True, eq=False)   # eq=False: los arrays no se comparan con ==
class VisorPCA:
    media: np.ndarray          # (d,) float64
    componentes: np.ndarray    # (2, d) float64
    proporcion: np.ndarray     # (2,) float64: varianza explicada de PC1 y PC2
    puntos: np.ndarray         # (n, 2) float64: las filas de train proyectadas
    identidades: np.ndarray    # (n,) int64: la identidad de cada fila de train
    centroides: np.ndarray     # (k, 2) float64: los centroides del modelo proyectados

    def proyectar(self, z: np.ndarray) -> np.ndarray:
        """(z − media) @ componentes.T: un vector (d,) da (2,) y una matriz (n, d), (n, 2)."""
        z = np.asarray(z, dtype=np.float64)
        d = self.media.shape[0]
        if z.ndim not in (1, 2) or z.shape[-1] != d:
            raise ValueError(f"z tiene forma {z.shape}; se esperan {d} columnas.")
        if not np.isfinite(z).all():
            raise ValueError("z tiene valores NaN o infinitos.")
        return (z - self.media) @ self.componentes.T


def armar_visor(cfg: dict, modelo: ModeloHuellas, carpeta: str | os.PathLike | None = None) -> VisorPCA:
    """PCA(2) propio de las filas de train de carpeta / caracteristicas.npz, con los centroides del modelo."""
    carpeta = RAIZ / cfg["rutas"]["huellas"] if carpeta is None else Path(os.fspath(carpeta))
    ruta = carpeta / NOMBRE_NPZ
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta} (el archivo de características de H3).")
    with np.load(ruta, allow_pickle=False) as datos:
        X = np.asarray(datos["X"], dtype=np.float64)
        train = datos["particion"] == "train"
        identidades = np.asarray(datos["identidad"][train], dtype=np.int64)
    T = X[train]
    if T.shape[0] < 3:
        raise ValueError(f"{ruta} tiene {T.shape[0]} filas de train; hacen falta al menos 3.")
    centroides = np.asarray(modelo.centroides, dtype=np.float64)
    if centroides.shape[1] != T.shape[1]:
        raise ValueError(f"X tiene {T.shape[1]} columnas y los centroides del modelo, {centroides.shape[1]}.")
    pca = PCA(2).ajustar(T)
    return VisorPCA(media=np.asarray(pca.media, dtype=np.float64),
                    componentes=np.asarray(pca.componentes, dtype=np.float64),
                    proporcion=np.asarray(pca.proporcion_varianza, dtype=np.float64),
                    puntos=np.asarray(pca.transformar(T), dtype=np.float64), identidades=identidades,
                    centroides=np.asarray(pca.transformar(centroides), dtype=np.float64))
