"""V2 · Preprocesamiento de voz y generación del dataset preprocesado.

Contrato: docs/contratos/v2_preprocesamiento.md, secciones 0, 3 y 5. De cada audio busca la palabra
principal sobre la señal sin filtrar, filtra el audio en fase cero (Butterworth pasabanda 80–4000 Hz
con sosfiltfilt y pre-énfasis), recorta la palabra con margen y normaliza su RMS a un nivel fijo.

Uso, desde la raíz del repo:
    venv\\Scripts\\python -m src.voz.preprocesamiento [--sobrescribir]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal
import soundfile as sf

from src.config import RAIZ, cargar_config
from src.voz.audio import remuestrear
from src.voz.aumento import aumentar, semilla_copia, sortear_aumento
from src.voz.segmentacion import detectar_palabras, envolvente_db

MOTIVOS_V2: tuple[str, ...] = ("sin_voz", "duracion")

COLUMNAS_V2: tuple[str, ...] = (
    "archivo",
    "particion",
    "origen",
    "codigo",
    "hablante",
    "clase",
    "detalle",
    "fuente",
    "copia",
    "valido",
    "motivo",
    "n_segmentos",
    "palabra_inicio_s",
    "palabra_fin_s",
    "duracion_palabra_s",
    "muestras",
    "factor",
    "snr_db",
    "ruido",
)

PARTICIONES_V2: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class Preprocesado:
    y: np.ndarray            # palabra con margen, acondicionada y normalizada (float32); 0 muestras si no es válido
    valido: bool
    motivo: str              # "" si es válido; si no, uno de MOTIVOS_V2
    inicio_s: float | None   # la palabra principal en el audio de entrada (a fs); None si no hay segmentos
    fin_s: float | None
    n_segmentos: int         # cuántos segmentos dio detectar_palabras


def palabra_principal(
    y: np.ndarray, fs: int, cfg: dict
) -> tuple[tuple[float, float] | None, int]:
    """Devuelve `(inicio_s, fin_s)` del segmento con la trama más fuerte (o `None`) y `n_segmentos`."""
    y_arr = np.asarray(y).reshape(-1)
    segs = detectar_palabras(y_arr, fs, cfg)
    if not segs:
        return None, 0
    d = cfg["voz"]["dataset"]["deteccion"]
    niveles = envolvente_db(y_arr, fs, d["trama_ms"], d["avance_ms"])
    h = int(round(float(d["avance_ms"]) * fs / 1000.0))
    tiempos = np.arange(len(niveles), dtype=np.float64) * h / fs

    mejor: tuple[float, float] | None = None
    mejor_nivel = -np.inf
    for ini_s, fin_s in segs:
        mascara = (tiempos >= ini_s) & (tiempos < fin_s)
        nivel_seg = float(niveles[mascara].max()) if np.any(mascara) else -np.inf
        if nivel_seg > mejor_nivel:
            mejor_nivel = nivel_seg
            mejor = (float(ini_s), float(fin_s))
    return mejor, len(segs)


def pasabanda(y: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    """Filtro Butterworth pasabanda en fase cero (`sosfiltfilt`), en `float64`."""
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    orden = int(cfg["voz"]["preprocesamiento"]["orden_filtro"])
    banda = cfg["voz"]["pasabanda_hz"]
    sos = scipy.signal.butter(orden, banda, btype="bandpass", fs=fs, output="sos")
    return scipy.signal.sosfiltfilt(sos, y64).astype(np.float64)


def preenfasis(y: np.ndarray, a: float) -> np.ndarray:
    """Filtro de pre-énfasis `z[0] = x[0]`, `z[n] = x[n] − a·x[n−1]`, en `float64`."""
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.copy()
    z = np.empty_like(x)
    z[0] = x[0]
    z[1:] = x[1:] - float(a) * x[:-1]
    return z


def recortar(
    x: np.ndarray, fs: int, inicio_s: float, fin_s: float, cfg: dict
) -> np.ndarray:
    """Recorta `[m0 − M : m1 + M]` completando con ceros fuera del audio (`float64`)."""
    x64 = np.asarray(x, dtype=np.float64).reshape(-1)
    m0 = int(round(float(inicio_s) * fs))
    m1 = int(round(float(fin_s) * fs))
    margen_ms = float(cfg["voz"]["preprocesamiento"]["margen_ms"])
    m_margen = int(round(margen_ms * fs / 1000.0))

    largo = (m1 - m0) + 2 * m_margen
    clip = np.zeros(largo, dtype=np.float64)
    ini_req = m0 - m_margen
    fin_req = m1 + m_margen
    s0 = max(0, ini_req)
    s1 = min(len(x64), fin_req)
    if s1 > s0:
        clip[s0 - ini_req : s1 - ini_req] = x64[s0:s1]
    return clip


def normalizar_rms(clip: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    """Normaliza el clip para que el RMS de la palabra (`clip[M : len − M]`) valga `nivel_rms_db` (`float32`)."""
    c64 = np.asarray(clip, dtype=np.float64).reshape(-1)
    margen_ms = float(cfg["voz"]["preprocesamiento"]["margen_ms"])
    m_margen = int(round(margen_ms * fs / 1000.0))
    palabra = c64[m_margen : len(c64) - m_margen]
    if palabra.size == 0:
        raise ValueError("El recorte no contiene muestras de palabra.")
    rms = float(np.sqrt(np.mean(palabra * palabra)))
    if rms <= 0.0:
        raise ValueError("La palabra es toda ceros: no se puede normalizar el RMS.")
    nivel_rms_db = float(cfg["voz"]["preprocesamiento"]["nivel_rms_db"])
    escala = (10.0 ** (nivel_rms_db / 20.0)) / rms
    return (c64 * escala).astype(np.float32)


def preprocesar(y: np.ndarray, fs: int, cfg: dict) -> Preprocesado:
    """Acondiciona, recorta y normaliza la palabra principal de `y`."""
    y_arr = np.asarray(y).reshape(-1)
    fs_trabajo = int(cfg["voz"]["frecuencia_muestreo"])
    if int(fs) != fs_trabajo:
        if y_arr.size == 0:
            y_arr = np.zeros(0, dtype=np.float32)
        else:
            y_arr = remuestrear(y_arr, int(fs), fs_trabajo)

    pal, n_seg = palabra_principal(y_arr, fs_trabajo, cfg)
    if pal is None:
        return Preprocesado(
            y=np.zeros(0, dtype=np.float32),
            valido=False,
            motivo="sin_voz",
            inicio_s=None,
            fin_s=None,
            n_segmentos=0,
        )

    ini_s, fin_s = float(pal[0]), float(pal[1])
    dur_min, dur_max = cfg["voz"]["duracion_palabra_s"]
    duracion = fin_s - ini_s
    if not (float(dur_min) <= duracion <= float(dur_max)):
        return Preprocesado(
            y=np.zeros(0, dtype=np.float32),
            valido=False,
            motivo="duracion",
            inicio_s=ini_s,
            fin_s=fin_s,
            n_segmentos=int(n_seg),
        )

    coef_pre = float(cfg["voz"]["preenfasis"])
    acond = preenfasis(pasabanda(y_arr, fs_trabajo, cfg), coef_pre)
    rec = recortar(acond, fs_trabajo, ini_s, fin_s, cfg)
    norm = normalizar_rms(rec, fs_trabajo, cfg)
    return Preprocesado(
        y=norm,
        valido=True,
        motivo="",
        inicio_s=ini_s,
        fin_s=fin_s,
        n_segmentos=int(n_seg),
    )


def _vacio_a_none(valor: str | None) -> str | None:
    if valor is None:
        return None
    limpio = valor.strip()
    return limpio if limpio != "" else None


def _armar_fila(
    base_v1: dict,
    copia: int,
    res: Preprocesado,
    archivo_rel: str | None,
    factor: float | None = None,
    snr_db: float | None = None,
    ruido_rel: str | None = None,
) -> tuple[dict, dict[str, str]]:
    """Construye el dict tipado de retorno y el dict de strings para `metadatos.csv`."""
    tiene_palabra = res.inicio_s is not None and res.fin_s is not None
    ini_r = round(float(res.inicio_s), 3) if tiene_palabra else None
    fin_r = round(float(res.fin_s), 3) if tiene_palabra else None
    dur_r = (
        round(float(res.fin_s) - float(res.inicio_s), 3) if tiene_palabra else None
    )
    n_seg_val = int(res.n_segmentos) if tiene_palabra else None
    muestras = int(len(res.y)) if res.valido else 0
    motivo_val = None if res.valido else res.motivo
    factor_r = round(float(factor), 2) if factor is not None else None
    snr_r = round(float(snr_db), 2) if snr_db is not None else None

    fila_retorno = {
        "archivo": archivo_rel if res.valido else None,
        "particion": _vacio_a_none(base_v1.get("particion")),
        "origen": _vacio_a_none(base_v1.get("origen")),
        "codigo": _vacio_a_none(base_v1.get("codigo")),
        "hablante": _vacio_a_none(base_v1.get("hablante")),
        "clase": _vacio_a_none(base_v1.get("clase")),
        "detalle": _vacio_a_none(base_v1.get("detalle")),
        "fuente": base_v1["archivo"],
        "copia": int(copia),
        "valido": bool(res.valido),
        "motivo": motivo_val,
        "n_segmentos": n_seg_val,
        "palabra_inicio_s": ini_r,
        "palabra_fin_s": fin_r,
        "duracion_palabra_s": dur_r,
        "muestras": muestras,
        "factor": factor_r,
        "snr_db": snr_r,
        "ruido": ruido_rel,
    }

    fila_csv = {
        "archivo": archivo_rel if (res.valido and archivo_rel is not None) else "",
        "particion": fila_retorno["particion"] or "",
        "origen": fila_retorno["origen"] or "",
        "codigo": fila_retorno["codigo"] or "",
        "hablante": fila_retorno["hablante"] or "",
        "clase": fila_retorno["clase"] or "",
        "detalle": fila_retorno["detalle"] or "",
        "fuente": fila_retorno["fuente"],
        "copia": str(fila_retorno["copia"]),
        "valido": "1" if res.valido else "0",
        "motivo": "" if res.valido else res.motivo,
        "n_segmentos": "" if n_seg_val is None else str(n_seg_val),
        "palabra_inicio_s": "" if ini_r is None else f"{ini_r:.3f}",
        "palabra_fin_s": "" if fin_r is None else f"{fin_r:.3f}",
        "duracion_palabra_s": "" if dur_r is None else f"{dur_r:.3f}",
        "muestras": str(muestras),
        "factor": "" if factor_r is None else f"{factor_r:.2f}",
        "snr_db": "" if snr_r is None else f"{snr_r:.2f}",
        "ruido": ruido_rel or "",
    }
    return fila_retorno, fila_csv


def generar_prepro(
    cfg: dict,
    origen: str | os.PathLike | None = None,
    destino: str | os.PathLike | None = None,
    sobrescribir: bool = False,
) -> list[dict]:
    """Genera los WAV `float32` y `metadatos.csv` del dataset preprocesado en `destino`."""
    carpeta_origen = (
        RAIZ / cfg["rutas"]["voz"] / "dataset"
        if origen is None
        else Path(os.fspath(origen))
    )
    carpeta_destino = (
        RAIZ / cfg["rutas"]["voz"] / "prepro"
        if destino is None
        else Path(os.fspath(destino))
    )

    # 1. Proteger destino antes de tocar carpetas o archivos
    existentes: list[Path] = []
    csv_destino = carpeta_destino / "metadatos.csv"
    if csv_destino.exists():
        existentes.append(csv_destino)
    for part in PARTICIONES_V2:
        dir_part = carpeta_destino / part
        if dir_part.is_dir():
            existentes.extend(dir_part.glob("*.wav"))
    if existentes and not sobrescribir:
        raise FileExistsError(
            f"En {carpeta_destino} ya existen archivos de V2; usá sobrescribir=True."
        )

    # 2. Leer y validar la entrada de V1
    csv_origen = carpeta_origen / "metadatos.csv"
    if not csv_origen.is_file():
        raise ValueError(f"No existe el archivo de metadatos de V1: {csv_origen}")

    with csv_origen.open("r", encoding="utf-8", newline="") as f:
        filas_v1 = list(csv.DictReader(f))

    fs_esperada = int(cfg["voz"]["frecuencia_muestreo"])
    copias_cfg = int(cfg["voz"]["preprocesamiento"]["aumento"]["copias"])

    for fila in filas_v1:
        rel = fila["archivo"]
        ruta_wav = carpeta_origen / rel
        if not ruta_wav.is_file():
            raise ValueError(f"Falta el archivo WAV de V1 nombrado en metadatos.csv: {ruta_wav}")
        info = sf.info(str(ruta_wav))
        if info.samplerate != fs_esperada:
            raise ValueError(
                f"El archivo {ruta_wav} tiene frecuencia {info.samplerate} Hz; se esperaba {fs_esperada} Hz."
            )

    archivos_ruido_train = sorted(
        fila["archivo"]
        for fila in filas_v1
        if fila.get("particion") == "train" and fila.get("origen") == "ruido"
    )
    ruidos_train: list[np.ndarray] = [
        sf.read(str(carpeta_origen / rel), dtype="float32")[0].reshape(-1)
        for rel in archivos_ruido_train
    ]

    # 3. Limpiar archivos previos de V2 solo cuando la validación de entrada ya pasó
    if existentes and sobrescribir:
        for p in existentes:
            if p.is_file():
                p.unlink()

    filas_retorno: list[dict] = []
    filas_csv: list[dict[str, str]] = []

    for fila in filas_v1:
        fuente_rel = fila["archivo"]
        particion = fila["particion"]
        stem = Path(fuente_rel).stem
        y_v1, _ = sf.read(str(carpeta_origen / fuente_rel), dtype="float32")
        y_v1 = y_v1.reshape(-1)

        res0 = preprocesar(y_v1, fs_esperada, cfg)
        rel_k0 = f"{particion}/{stem}_k0.wav" if res0.valido else None
        if res0.valido and rel_k0 is not None:
            ruta_k0 = carpeta_destino / rel_k0
            ruta_k0.parent.mkdir(parents=True, exist_ok=True)
            scipy.io.wavfile.write(str(ruta_k0), fs_esperada, res0.y.astype(np.float32))

        ret0, csv0 = _armar_fila(fila, 0, res0, rel_k0)
        filas_retorno.append(ret0)
        filas_csv.append(csv0)

        lleva_copias = (
            particion == "train"
            and fila.get("origen") != "ruido"
            and res0.valido
            and copias_cfg > 0
        )
        if lleva_copias:
            if not ruidos_train:
                raise ValueError(
                    "Hay clips de train que llevan copias aumentadas pero no hay clips de ruido en train."
                )
            palabra_base = (float(res0.inicio_s), float(res0.fin_s))
            for k in range(1, copias_cfg + 1):
                rng = semilla_copia(cfg, fuente_rel, k)
                params = sortear_aumento(cfg, rng, len(ruidos_train))
                y_aum = aumentar(y_v1, fs_esperada, palabra_base, ruidos_train, params, cfg)
                res_k = preprocesar(y_aum, fs_esperada, cfg)
                rel_kk = f"{particion}/{stem}_k{k}.wav" if res_k.valido else None
                if res_k.valido and rel_kk is not None:
                    ruta_kk = carpeta_destino / rel_kk
                    ruta_kk.parent.mkdir(parents=True, exist_ok=True)
                    scipy.io.wavfile.write(str(ruta_kk), fs_esperada, res_k.y.astype(np.float32))

                ruido_usado = archivos_ruido_train[int(params["ruido"])]
                ret_k, csv_k = _armar_fila(
                    fila,
                    k,
                    res_k,
                    rel_kk,
                    factor=params["factor"],
                    snr_db=params["snr_db"],
                    ruido_rel=ruido_usado,
                )
                filas_retorno.append(ret_k)
                filas_csv.append(csv_k)

    carpeta_destino.mkdir(parents=True, exist_ok=True)
    with csv_destino.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNAS_V2), lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas_csv)

    return filas_retorno


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V2 · Genera el dataset de voz preprocesado (data/voz/prepro)."
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="Reemplaza los WAV y metadatos.csv existentes en data/voz/prepro.",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    cfg = cargar_config()
    filas = generar_prepro(cfg, sobrescribir=args.sobrescribir)
    dt = time.perf_counter() - t0

    print(f"V2 · Dataset preprocesado en {RAIZ / cfg['rutas']['voz'] / 'prepro'} ({dt:.1f} s)")
    print(f"Total de filas en metadatos.csv: {len(filas)}")

    conteo = Counter(
        (
            r["particion"],
            r["clase"],
            "k0" if r["copia"] == 0 else "aumentada",
            "valido" if r["valido"] else "no_valido",
        )
        for r in filas
    )
    print("\nClips por partición, clase y copia:")
    for clave in sorted(conteo):
        part, clase, tipo_copia, estado = clave
        print(f"  {part:5s} | {clase:11s} | {tipo_copia:9s} | {estado:9s}: {conteo[clave]}")

    motivos = Counter(
        (r["particion"], "k0" if r["copia"] == 0 else "aumentada", r["motivo"])
        for r in filas
        if not r["valido"]
    )
    print("\nNo válidos por motivo:")
    for clave in sorted(motivos):
        part, tipo_copia, mot = clave
        print(f"  {part:5s} | {tipo_copia:9s} | {mot:10s}: {motivos[clave]}")

    validos_train = sum(1 for r in filas if r["particion"] == "train" and r["valido"])
    print(f"\nVectores válidos en train: {validos_train}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

