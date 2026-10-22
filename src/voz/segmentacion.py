"""V1 · Envolvente, detector de palabras y cortes de audios con varias palabras.

Contrato: docs/contratos/v1_dataset.md, secciones 0 y 4. Los umbrales son relativos al tramo más
fuerte del audio, así que el detector no depende del volumen de cada micrófono.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _tramas(fs: int, trama_ms: float, avance_ms: float) -> tuple[int, int]:
    return int(round(trama_ms * fs / 1000)), int(round(avance_ms * fs / 1000))


def envolvente_db(y: np.ndarray, fs: int, trama_ms: float, avance_ms: float) -> np.ndarray:
    """Nivel RMS de cada trama en dBFS (float64), con piso de −120 dB.

    La trama k cubre y[k·H : k·H + N]; hay K = 1 + (len(y) − N) // H tramas (0 si len(y) < N).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, h = _tramas(fs, trama_ms, avance_ms)
    if n <= 0 or h <= 0 or len(y) < n:
        return np.zeros(0, dtype=np.float64)
    k = 1 + (len(y) - n) // h
    # Vista de las tramas sin copiar el audio (la suma acumulada perdía precisión en los silencios).
    tramas = np.lib.stride_tricks.sliding_window_view(y * y, n)[::h][:k]
    rms = np.sqrt(tramas.mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-6))


def _corridas(activa: np.ndarray) -> list[list[int]]:
    """[k_ini, k_fin] de cada corrida de tramas activas."""
    if not activa.any():
        return []
    a = activa.astype(np.int8)
    cambios = np.diff(np.concatenate(([0], a, [0])))
    inicios = np.flatnonzero(cambios == 1)
    fines = np.flatnonzero(cambios == -1) - 1
    return [[int(i), int(f)] for i, f in zip(inicios, fines)]


def detectar_palabras(y: np.ndarray, fs: int, cfg: dict) -> list[tuple[float, float]]:
    """Segmentos (inicio_s, fin_s) de las palabras, en orden y sin solaparse (sección 0)."""
    d = cfg["voz"]["dataset"]["deteccion"]
    y = np.asarray(y).reshape(-1)
    niveles = envolvente_db(y, fs, d["trama_ms"], d["avance_ms"])
    if niveles.size == 0:
        return []
    tope = float(niveles.max())
    if tope < d["nivel_min_db"]:
        return []
    n, h = _tramas(fs, d["trama_ms"], d["avance_ms"])
    piso = float(np.percentile(niveles, 10))
    umbral = max(tope - d["rango_db"], piso + d["margen_piso_db"])

    unidas: list[list[int]] = []
    for k_ini, k_fin in _corridas(niveles > umbral):
        if unidas and (k_ini * h - (unidas[-1][1] * h + n)) / fs < d["pausa_min_s"]:
            unidas[-1][1] = k_fin
        else:
            unidas.append([k_ini, k_fin])

    segmentos = []
    for k_ini, k_fin in unidas:
        inicio = k_ini * h / fs
        fin = min(k_fin * h + n, len(y)) / fs
        if fin - inicio < d["segmento_min_s"]:
            continue
        if niveles[k_ini:k_fin + 1].max() < tope - d["dif_palabra_db"]:
            continue  # ruido de fondo, voz lejana o golpe suave: no es una palabra
        segmentos.append((inicio, fin))
    return segmentos


@dataclass(frozen=True)
class Corte:
    inicio_s: float          # la palabra, en el audio entero
    fin_s: float
    muestra_inicio: int      # el clip que se guarda, en muestras del audio entero
    muestra_fin: int


def cortar_repeticiones(y: np.ndarray, fs: int, cfg: dict) -> list[Corte]:
    """Un corte por segmento de detectar_palabras(y, fs, cfg), en orden (sección 4).

    El clip deja `whatsapp.margen_s` de cada lado, sin pasar de los bordes del audio ni del punto
    medio del silencio con la palabra vecina, y dura a lo sumo `clip_max_s`.
    """
    dset = cfg["voz"]["dataset"]
    margen = float(dset["whatsapp"]["margen_s"])
    clip_max = float(dset["clip_max_s"])
    y = np.asarray(y).reshape(-1)
    duracion = len(y) / fs
    segmentos = detectar_palabras(y, fs, cfg)
    cortes = []
    for i, (inicio, fin) in enumerate(segmentos):
        tope_izq = 0.0 if i == 0 else (segmentos[i - 1][1] + inicio) / 2
        tope_der = duracion if i == len(segmentos) - 1 else (fin + segmentos[i + 1][0]) / 2
        t0 = max(tope_izq, inicio - margen)
        t1 = min(tope_der, fin + margen)
        if t1 - t0 > clip_max:
            lado = max(0.0, (clip_max - (fin - inicio)) / 2)
            t0, t1 = inicio - lado, fin + lado
        m0 = max(0, min(len(y), int(round(t0 * fs))))
        m1 = max(m0, min(len(y), int(round(t1 * fs))))
        cortes.append(Corte(float(inicio), float(fin), m0, m1))
    return cortes
