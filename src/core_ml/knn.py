"""Clasificador K-NN propio con NumPy y voto ponderado por el inverso de la distancia.

Contrato: docs/contratos/v4_clasificacion.md (versión 1.1), secciones 2 y 4.1.
Solo NumPy: no importa scikit-learn.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class KNN:
    """Clasificador K-NN con voto ponderado w_j = 1 / (d_j + eps)."""

    def __init__(self, k: int = 5, eps: float = 1e-9) -> None:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k debe ser un entero >= 1 (no bool), recibido {k!r}.")
        if (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float, np.floating))
            or not math.isfinite(float(eps))
            or float(eps) <= 0.0
            or float(eps) > 1e-3
        ):
            raise ValueError(f"eps debe ser un número real finito en (0, 1e-3], recibido {eps!r}.")

        self.k: int = int(k)
        self.eps: float = float(eps)
        self.X_entrenamiento: np.ndarray | None = None
        self.etiquetas: np.ndarray | None = None
        self.clases: np.ndarray | None = None

    def ajustar(self, X: np.ndarray, etiquetas: Any) -> "KNN":
        """Ajusta el clasificador guardando copia de los datos de entrenamiento."""
        arr_x = np.asarray(X)
        if arr_x.ndim != 2:
            raise ValueError(f"X debe tener 2 dimensiones (n, d), recibido ndim={arr_x.ndim}.")
        n, d = arr_x.shape
        if d < 1:
            raise ValueError(f"d debe ser >= 1, recibido d={d}.")
        if n < self.k:
            raise ValueError(f"n debe ser >= k ({self.k}), recibido n={n}.")
        if not np.isfinite(arr_x).all():
            raise ValueError("X contiene valores no finitos (NaN o Inf).")

        arr_et = np.asarray(etiquetas)
        if arr_et.ndim != 1 or len(arr_et) != n:
            raise ValueError(f"etiquetas debe ser 1D con {n} elementos, recibido shape={arr_et.shape}.")

        self.X_entrenamiento = np.asarray(arr_x, dtype=np.float64).copy()
        self.etiquetas = arr_et.copy()
        self.clases = np.unique(self.etiquetas)
        return self

    def _validar_consulta(self, X: np.ndarray) -> tuple[np.ndarray, bool]:
        """Valida X y devuelve (X2, es_vector)."""
        if self.X_entrenamiento is None or self.etiquetas is None or self.clases is None:
            raise RuntimeError("El K-NN no está ajustado: llamá a ajustar(X, etiquetas) primero.")

        arr = np.asarray(X)
        if arr.ndim == 1:
            es_vector = True
            X2 = arr[np.newaxis, :]
        elif arr.ndim == 2:
            es_vector = False
            X2 = arr
        else:
            raise ValueError(f"X debe tener 1 o 2 dimensiones, recibido ndim={arr.ndim}.")

        if X2.shape[1] != self.X_entrenamiento.shape[1]:
            raise ValueError(
                f"Dimensión incorrecta: se esperaban {self.X_entrenamiento.shape[1]} columnas, "
                f"recibidas {X2.shape[1]}."
            )
        if not np.isfinite(X2).all():
            raise ValueError("La consulta contiene valores no finitos (NaN o Inf).")

        return np.asarray(X2, dtype=np.float64), es_vector

    def vecinos(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (distancias, indices) de los k vecinos más cercanos ordenados de menor a mayor."""
        X2, es_vector = self._validar_consulta(X)
        assert self.X_entrenamiento is not None

        n_q = X2.shape[0]
        n_tr = self.X_entrenamiento.shape[0]

        dist = np.empty((n_q, self.k), dtype=np.float64)
        idx = np.empty((n_q, self.k), dtype=np.int64)

        for i in range(n_q):
            diff = self.X_entrenamiento - X2[i]
            d_i = np.sqrt(np.sum(diff * diff, axis=1))
            # argsort con kind='stable' desempata distancias iguales por índice menor
            idx_ord = np.argsort(d_i, kind="stable")[: self.k]
            idx[i] = idx_ord
            dist[i] = d_i[idx_ord]

        if es_vector:
            return dist[0], idx[0]
        return dist, idx

    def votar(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[Any, float, float]:
        """Voto ponderado de los k vecinos más cercanos."""
        X2, es_vector = self._validar_consulta(X)
        assert self.X_entrenamiento is not None
        assert self.etiquetas is not None
        assert self.clases is not None

        dist, idx = self.vecinos(X2)
        n_q = X2.shape[0]

        prediccion = np.empty(n_q, dtype=self.etiquetas.dtype)
        confianza = np.empty(n_q, dtype=np.float64)
        distancia = np.empty(n_q, dtype=np.float64)

        for i in range(n_q):
            d_i = dist[i]
            idx_i = idx[i]
            w_i = 1.0 / (d_i + self.eps)
            w_total = float(np.sum(w_i))

            et_i = self.etiquetas[idx_i]

            votos: dict[Any, float] = {}
            d_min: dict[Any, float] = {}

            for c in self.clases:
                mask = et_i == c
                if np.any(mask):
                    votos[c] = float(np.sum(w_i[mask]))
                    d_min[c] = float(np.min(d_i[mask]))

            v_max = max(votos.values())
            # Candidatos con voto máximo (tolerancia 1e-12)
            candidatos = [c for c in self.clases if c in votos and abs(votos[c] - v_max) <= 1e-12]

            # Desempate por menor distancia mínima
            min_dist_cand = min(d_min[c] for c in candidatos)
            candidatos = [c for c in candidatos if d_min[c] == min_dist_cand]

            # Si persisten empatados, primer orden en self.clases (alfabético de np.unique)
            ganador = candidatos[0]

            prediccion[i] = ganador
            confianza[i] = votos[ganador] / w_total
            distancia[i] = float(np.mean(d_i[et_i == ganador]))

        if es_vector:
            # Para consulta escalar, desenvuelve a tipos nativos
            etiqueta_val = prediccion[0]
            if hasattr(etiqueta_val, "item"):
                etiqueta_val = etiqueta_val.item()
            return etiqueta_val, float(confianza[0]), float(distancia[0])

        return prediccion, confianza, distancia

    def predecir(self, X: np.ndarray) -> np.ndarray | Any:
        """Devuelve únicamente la predicción de votar."""
        res = self.votar(X)
        if isinstance(res, tuple) and len(res) == 3:
            return res[0]
        return res

