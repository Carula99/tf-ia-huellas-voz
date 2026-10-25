"""Fase 5 · Percepción de la huella: leer el archivo, prepararla (H2) y evaluarla (H3 + H4).

Contrato: docs/contratos/f5_integracion.md (versión 1.1), secciones 2.2 y 4.2. Es reconocer() de H4
partido en dos: H2 corre una sola vez al cargar la huella (la app muestra el esqueleto) y "Reconocer"
corre solo H3 y H4. evaluar_huella(preparar_huella(im)) da lo mismo que reconocer(im) de H4, y además
el vector z en el espacio de X, para marcar la huella en el visor PCA.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np

from ..huellas.caracteristicas import caracteristicas
from ..huellas.clasificacion import NO_AUTORIZADO, ModeloHuellas
from ..huellas.preprocesamiento import HuellaPreprocesada, preprocesar


@dataclass(frozen=True, eq=False)   # eq=False: z es un array
class EvaluacionHuella:
    identidad: int        # la decisión: 1 a k, o 0 si se rechaza
    mas_cercana: int      # identidad del centroide más cercano (no se muestra nunca)
    distancia: float      # distancia normalizada a ese centroide (el puntaje de H4)
    tau: float            # τ* del modelo
    z: np.ndarray         # (d,) float64: la huella en el espacio de X (para el visor PCA)


def leer_imagen(ruta: str | os.PathLike) -> np.ndarray | None:
    """La imagen del archivo tal como está (gris, color o 16 bits), o None si no se puede leer.

    Se lee con np.fromfile y cv2.imdecode porque cv2.imread no abre rutas con tildes en Windows.
    """
    try:
        datos = np.fromfile(os.fspath(ruta), dtype=np.uint8)
        if datos.size == 0:
            return None
        return cv2.imdecode(datos, cv2.IMREAD_UNCHANGED)   # None si no la puede decodificar
    except (OSError, TypeError, ValueError, cv2.error):
        return None


def preparar_huella(imagen: np.ndarray, cfg: dict) -> HuellaPreprocesada:
    """H2 (preprocesar): ValueError si la imagen no es uint8, mide menos de 64 px o no tiene una huella."""
    return preprocesar(imagen, cfg)


def evaluar_huella(huella: HuellaPreprocesada, modelo: ModeloHuellas, cfg: dict) -> EvaluacionHuella:
    """H3 y H4 sobre una huella ya preparada: la segunda mitad de reconocer() de H4."""
    if modelo.escalador is None:
        raise ValueError("El modelo no tiene escalador: no puede llevar una huella al espacio de X.")
    z = np.asarray(modelo.escalador.transformar(caracteristicas(huella, cfg)), dtype=np.float64)
    distancia, mas_cercana = modelo.puntajes(z)
    identidad = mas_cercana if distancia <= modelo.tau else NO_AUTORIZADO
    return EvaluacionHuella(identidad=int(identidad), mas_cercana=int(mas_cercana),
                            distancia=float(distancia), tau=float(modelo.tau), z=z)
