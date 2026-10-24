"""V2 · Aumento de datos de voz para entrenamiento.

Contrato: docs/contratos/v2_preprocesamiento.md, secciones 0 y 4. Aplica cambio de velocidad como
una cinta (un único remuestreo polifásico que modifica duración y tono a la vez) y suma ruido real
de las grabaciones de train antes de buscar y recortar la palabra.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import scipy.signal


def semilla_copia(cfg: dict, archivo: str, copia: int) -> np.random.Generator:
    """Generador aleatorio determinista para `(archivo, copia)`."""
    semilla_global = int(cfg["semilla_global"])
    crc = zlib.crc32(archivo.encode("utf-8"))
    return np.random.default_rng([semilla_global, crc, int(copia)])


def cambiar_velocidad(y: np.ndarray, factor: float) -> np.ndarray:
    """Cambia la velocidad y el tono con un solo remuestreo (`float32`).

    Con `factor > 1` dura `factor` veces menos y suena `factor` veces más agudo.
    `factor` se redondea a 2 decimales; con `factor == 1` devuelve una copia.
    """
    y_arr = np.asarray(y).reshape(-1)
    f = round(float(factor), 2)
    if f <= 0.0:
        raise ValueError(f"El factor de velocidad tiene que ser > 0; se recibió {factor!r}.")
    d = int(round(100 * f))
    if d == 100:
        return np.array(y_arr, dtype=np.float32, copy=True)
    g = math.gcd(100, d)
    res = scipy.signal.resample_poly(y_arr.astype(np.float64), 100 // g, d // g)
    return res.astype(np.float32)


def sumar_ruido(
    y: np.ndarray,
    ruido: np.ndarray,
    snr_db: float,
    palabra: tuple[float, float],
    fs: int,
) -> np.ndarray:
    """Suma `g · ruido[:len(y)]` a `y` con relación señal-ruido `snr_db` medida sobre `palabra`."""
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    r64 = np.asarray(ruido, dtype=np.float64).reshape(-1)
    if len(r64) < len(y64):
        raise ValueError(
            f"El ruido ({len(r64)} muestras) es más corto que el audio ({len(y64)} muestras)."
        )
    r_tramo = r64[: len(y64)]
    m0 = int(round(float(palabra[0]) * fs))
    m1 = int(round(float(palabra[1]) * fs))
    m0 = max(0, min(len(y64), m0))
    m1 = max(m0, min(len(y64), m1))
    seg_palabra = y64[m0:m1]
    ps = float(np.mean(seg_palabra * seg_palabra)) if seg_palabra.size > 0 else 0.0
    pn = float(np.mean(r_tramo * r_tramo)) if r_tramo.size > 0 else 0.0
    if ps <= 0.0:
        raise ValueError("La potencia de la palabra (Ps) es 0: no se puede calcular la SNR.")
    if pn <= 0.0:
        raise ValueError("La potencia del ruido (Pn) es 0: no se puede calcular la SNR.")
    g = math.sqrt(ps / (pn * (10.0 ** (float(snr_db) / 10.0))))
    return (y64 + g * r_tramo).astype(np.float32)


def sortear_aumento(cfg: dict, rng: np.random.Generator, n_ruidos: int) -> dict:
    """Sortea los parámetros de una copia aumentada en el orden del contrato."""
    if n_ruidos <= 0:
        raise ValueError(f"n_ruidos tiene que ser >= 1; se recibió {n_ruidos}.")
    aum = cfg["voz"]["preprocesamiento"]["aumento"]
    vel_min, vel_max = aum["velocidad"]
    snr_min, snr_max = aum["snr_db"]
    factor = round(float(rng.uniform(float(vel_min), float(vel_max))), 2)
    snr_db = float(rng.uniform(float(snr_min), float(snr_max)))
    ruido = int(rng.integers(int(n_ruidos)))
    desplazamiento = float(rng.random())
    return {
        "factor": factor,
        "snr_db": snr_db,
        "ruido": ruido,
        "desplazamiento": desplazamiento,
    }


def aumentar(
    y: np.ndarray,
    fs: int,
    palabra: tuple[float, float],
    ruidos: list[np.ndarray],
    params: dict,
    cfg: dict,
) -> np.ndarray:
    """Construye la copia aumentada de un clip de V1 antes de preprocesar (`float32`)."""
    factor = round(float(params["factor"]), 2)
    z = cambiar_velocidad(y, factor)
    ini_s = float(palabra[0]) / factor
    fin_s = float(palabra[1]) / factor

    clip_max_s = float(cfg["voz"]["dataset"]["clip_max_s"])
    c_max = int(round(clip_max_s * fs))
    if len(z) > c_max:
        c = int(round((ini_s + fin_s) / 2.0 * fs))
        i0 = min(max(0, c - c_max // 2), len(z) - c_max)
        z = z[i0 : i0 + c_max]
        ini_s -= i0 / fs
        fin_s -= i0 / fs

    idx_ruido = int(params["ruido"])
    if not (0 <= idx_ruido < len(ruidos)):
        raise ValueError(f"Índice de ruido fuera de rango: {idx_ruido} (hay {len(ruidos)} ruidos).")
    r = np.asarray(ruidos[idx_ruido]).reshape(-1)
    if len(r) < len(z):
        raise ValueError(
            f"El ruido elegido ({len(r)} muestras) es más corto que la copia ({len(z)} muestras)."
        )
    despl = float(params["desplazamiento"])
    off = int(math.floor(despl * (len(r) - len(z) + 1)))
    return sumar_ruido(z, r[off:], float(params["snr_db"]), (ini_s, fin_s), fs)

