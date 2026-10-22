"""V1 · Lectura, remuestreo y escritura de audio.

Contrato: docs/contratos/v1_dataset.md, secciones 0 y 3. Todo el carril de voz lee y escribe audio
con estas tres funciones, así un mismo archivo da siempre las mismas muestras.
"""

from __future__ import annotations

import os
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _a_mono(y: np.ndarray) -> np.ndarray:
    """Promedio de los canales, en float32 de 1 dimensión."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1, dtype=np.float64).astype(np.float32)
    return np.ascontiguousarray(y.reshape(-1))


def _leer_con_pyav(ruta: Path) -> tuple[np.ndarray, int]:
    """Decodifica con PyAV todos los cuadros del primer flujo de audio, en orden."""
    import av  # PyAV: solo hace falta para lo que soundfile no abre (.m4a y algunas notas de voz)

    with av.open(str(ruta)) as contenedor:
        flujo = contenedor.streams.audio[0]
        partes = []
        for cuadro in contenedor.decode(flujo):
            x = cuadro.to_ndarray()
            if cuadro.format.is_planar:          # (canales, muestras)
                x = x.T
            else:                                # (1, muestras · canales) intercalado
                x = x.reshape(-1, len(cuadro.layout.channels))
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float64) / float(-np.iinfo(x.dtype).min)
            partes.append(x.astype(np.float32))
        fs = int(flujo.rate)
    if not partes:
        return np.zeros(0, dtype=np.float32), fs
    return _a_mono(np.concatenate(partes, axis=0)), fs


def leer_audio(ruta: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Lee WAV, OGG/Opus (.ogg u .opus), MP3 y M4A.

    Primero con soundfile; si no puede abrir el archivo (los .m4a y algunas notas de voz de
    WhatsApp que libsndfile rechaza), con PyAV. Devuelve float32 mono en [−1, 1] y la frecuencia
    del archivo.
    """
    ruta = Path(os.fspath(ruta))
    try:
        y, fs = sf.read(str(ruta), dtype="float32", always_2d=True)
        y = _a_mono(y)
    except (sf.LibsndfileError, RuntimeError, TypeError):
        y, fs = _leer_con_pyav(ruta)
    # Los códecs con pérdida (Opus, AAC, MP3) pueden pasar apenas de ±1 al decodificar, sobre todo
    # al final de un archivo cortado de golpe: se recorta para devolver siempre [−1, 1].
    return np.clip(y, -1.0, 1.0), int(fs)


def remuestrear(y: np.ndarray, fs_origen: int, fs_destino: int) -> np.ndarray:
    """resample_poly en float64 con los factores reducidos por el máximo común divisor; float32."""
    y = np.asarray(y)
    if int(fs_origen) == int(fs_destino):
        return y.astype(np.float32, copy=True)
    g = gcd(int(fs_origen), int(fs_destino))
    z = resample_poly(y.astype(np.float64), int(fs_destino) // g, int(fs_origen) // g)
    return z.astype(np.float32)


def guardar_wav(ruta: str | os.PathLike, y: np.ndarray, fs: int) -> None:
    """WAV mono PCM de 16 bits. Recorta a [−1, 1] antes de escribir y crea las carpetas."""
    ruta = Path(os.fspath(ruta))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.asarray(y, dtype=np.float64).reshape(-1), -1.0, 1.0)
    sf.write(str(ruta), y, int(fs), subtype="PCM_16", format="WAV")
