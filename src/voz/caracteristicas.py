"""V3 · Características acústicas de voz por bloques, escalador ponderado y PCA propio.

Contrato: docs/contratos/v3_caracteristicas.md (versión 1). A partir de cada clip válido de V2
(palabra recortada con margen, filtrada entre 80 y 4000 Hz con pre-énfasis y normalizada a
−26 dBFS), caracteristicas() descarta las tramas del margen, aplica normalización cepstral de
media (CMN) sobre la palabra y construye cinco bloques crudos en 5 tramos iguales (d = 281):

1. mfcc: media y desvío por tramo de los 13 MFCC con CMN (130 valores);
2. delta: media y desvío por tramo de las 13 derivadas temporales Δ (130 valores);
3. energia: media y desvío por tramo de la energía RMS en dB (10 valores);
4. zcr: media y desvío por tramo de la tasa de cruces por cero (10 valores);
5. duracion: duración de la palabra en segundos, (len(clip) − 2·M) / fs (1 valor).

El escalador, ajustado solo con las filas válidas de train, estandariza cada columna (ddof=1),
imputa eventuales NaN con 0.0 (la media de train), descarta columnas constantes o sin datos y
pondera cada bloque por 1/√(columnas usadas), de modo que los cinco bloques aporten lo mismo (2,0)
a la distancia euclídea cuadrática media en train.

Línea de comandos, desde la raíz del repo (escribe data/voz/prepro/caracteristicas.npz y
docs/figuras/pca_v3.png):
    venv\\Scripts\\python -m src.voz.caracteristicas [--sobrescribir]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import sys
import time
import warnings

import librosa
import numpy as np
import soundfile as sf

from ..config import RAIZ, cargar_config
from ..core_ml.metricas import silhouette
from ..core_ml.pca import PCA

BLOQUES: tuple[str, ...] = ("mfcc", "delta", "energia", "zcr", "duracion")

DESVIO_MIN = 1e-9
"""Las columnas con desvío de train menor o igual se descartan (sección 4.3)."""

COLUMNAS_REQUERIDAS: tuple[str, ...] = (
    "archivo",
    "particion",
    "origen",
    "hablante",
    "clase",
    "copia",
    "valido",
)

VALORES_VALIDO = frozenset({"1", "True", "true", 1, True})
NOMBRE_NPZ = "caracteristicas.npz"
NOMBRE_FIGURA = "pca_v3.png"


def largos_de_bloque(cfg: dict) -> dict[str, int]:
    """Devuelve la cantidad de columnas de cada bloque según `cfg`."""
    k = int(cfg["voz"]["tramos"])
    n_mfcc = int(cfg["voz"]["mfcc"]["n_mfcc"])
    return {
        "mfcc": 2 * k * n_mfcc,
        "delta": 2 * k * n_mfcc,
        "energia": 2 * k,
        "zcr": 2 * k,
        "duracion": 1,
    }


def estadisticas_tramos(matriz: np.ndarray, k: int) -> np.ndarray:
    """Divide `matriz` (filas × N tramas) en `k` tramos y concatena `[media, desvío (ddof=0)]`."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError(f"k tiene que ser un entero >= 1 (no bool); se recibió {k!r}.")
    k_int = int(k)

    arr = np.asarray(matriz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(
            f"matriz tiene que ser un array 2D (F >= 1, N >= 1); tiene forma {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("matriz contiene valores no finitos (NaN o ±Inf).")

    n_tramas = arr.shape[1]
    partes: list[np.ndarray] = []
    for i in range(k_int):
        a = int(np.floor(i * n_tramas / k_int))
        b = max(a + 1, int(np.floor((i + 1) * n_tramas / k_int)))
        s = arr[:, a:b]
        partes.append(np.concatenate([s.mean(axis=1), s.std(axis=1, ddof=0)]))
    return np.concatenate(partes).astype(np.float64)


def caracteristicas(clip: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Calcula los 5 bloques crudos de un clip válido de V2 (sección 4.2)."""
    clip_arr = np.asarray(clip)
    if clip_arr.ndim != 1:
        raise ValueError(f"clip tiene que ser de 1 dimensión; tiene forma {clip_arr.shape}.")
    if not np.all(np.isfinite(clip_arr)):
        raise ValueError("clip contiene valores no finitos (NaN o ±Inf).")

    voz = cfg["voz"]
    mfcc_cfg = voz["mfcc"]
    fs = int(voz["frecuencia_muestreo"])
    m_muestras = int(round(float(voz["preprocesamiento"]["margen_ms"]) * fs / 1000.0))
    w_len = int(round(float(mfcc_cfg["ventana_ms"]) * fs / 1000.0))
    hop = int(round(float(mfcc_cfg["avance_ms"]) * fs / 1000.0))
    n_fft = int(mfcc_cfg["n_fft"])
    n_mfcc = int(mfcc_cfg["n_mfcc"])
    n_mels = int(mfcc_cfg["n_mels"])
    fmin = float(voz["pasabanda_hz"][0])
    fmax = float(mfcc_cfg["fmax"])
    ancho_delta = int(mfcc_cfg["delta_ventana"])
    piso_rms = float(mfcc_cfg["piso_rms"])
    k_tramos = int(voz["tramos"])
    m_tramas = m_muestras // hop

    min_muestras = 2 * m_muestras + n_fft
    if len(clip_arr) < min_muestras:
        raise ValueError(
            f"clip tiene {len(clip_arr)} muestras; se requieren al menos 2*M + n_fft = {min_muestras}."
        )

    y = np.asarray(clip_arr, dtype=np.float32)

    s_mfcc = librosa.feature.mfcc(
        y=y,
        sr=fs,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=w_len,
        hop_length=hop,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        center=False,
    ).astype(np.float64)

    if s_mfcc.shape[1] > 2 * m_tramas + k_tramos:
        w = s_mfcc[:, m_tramas : s_mfcc.shape[1] - m_tramas]
    else:
        w = s_mfcc
    w = w - w.mean(axis=1, keepdims=True)

    if w.shape[1] >= ancho_delta:
        d = librosa.feature.delta(w, width=ancho_delta, mode="nearest").astype(np.float64)
    else:
        d = np.zeros_like(w)

    fr = librosa.util.frame(y, frame_length=w_len, hop_length=hop).astype(np.float64)
    if fr.shape[1] > 2 * m_tramas + k_tramos:
        fr = fr[:, m_tramas : fr.shape[1] - m_tramas]
    rms = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(fr ** 2, axis=0)), piso_rms))[None, :]
    zcr = (np.abs(np.diff(np.sign(fr), axis=0)) > 0).mean(axis=0)[None, :]

    n_cols = min(w.shape[1], rms.shape[1])
    return {
        "mfcc": estadisticas_tramos(w[:, :n_cols], k_tramos),
        "delta": estadisticas_tramos(d[:, :n_cols], k_tramos),
        "energia": estadisticas_tramos(rms[:, :n_cols], k_tramos),
        "zcr": estadisticas_tramos(zcr[:, :n_cols], k_tramos),
        "duracion": np.array([(len(clip_arr) - 2 * m_muestras) / float(fs)], dtype=np.float64),
    }


# --------------------------------------------------------------------------- escalador (4.3)
def _comprobar_claves(bloques: dict, que: str) -> None:
    if not isinstance(bloques, dict) or set(bloques) != set(BLOQUES):
        claves = sorted(bloques) if isinstance(bloques, dict) else type(bloques).__name__
        raise ValueError(
            f"{que}: se espera un dict con exactamente las claves {list(BLOQUES)}; tiene {claves}."
        )


@dataclass(frozen=True)
class Escalador:
    """Estandarización por columna con las estadísticas de train, imputación y peso por bloque."""

    medias: dict[str, np.ndarray]     # (d_b,) por bloque
    desvios: dict[str, np.ndarray]    # (d_b,) por bloque
    columnas: dict[str, np.ndarray]   # bool (d_b,): columnas que se usan
    pesos: dict[str, float]           # 1 / √(columnas usadas del bloque)

    def transformar(self, bloques: dict[str, np.ndarray]) -> np.ndarray:
        """Transforma un vector `(d_b,)` o lote `(n, d_b)` por bloque en `(d,)` o `(n, d)`."""
        _comprobar_claves(bloques, "transformar")
        arrays = {b: np.asarray(bloques[b], dtype=np.float64) for b in BLOQUES}
        dimensiones = {a.ndim for a in arrays.values()}
        if len(dimensiones) != 1 or not dimensiones <= {1, 2}:
            raise ValueError(
                "Los bloques tienen que ser todos vectores (d_b,) o todos arrays (n, d_b)."
            )
        uno = arrays[BLOQUES[0]].ndim == 1
        if not uno and len({a.shape[0] for a in arrays.values()}) != 1:
            raise ValueError("Los bloques tienen distinta cantidad de filas.")
        for b, a in arrays.items():
            d_b = self.columnas[b].shape[0]
            if a.shape[-1] != d_b:
                raise ValueError(
                    f"El bloque {b} tiene {a.shape[-1]} valores por fila; se esperan {d_b}."
                )
            if np.isinf(a).any():
                raise ValueError(f"El bloque {b} tiene valores ±Inf.")
        partes = []
        for b in BLOQUES:
            usadas = self.columnas[b]
            z = (
                np.atleast_2d(arrays[b])[:, usadas] - self.medias[b][usadas]
            ) / self.desvios[b][usadas]
            z[np.isnan(z)] = 0.0
            partes.append(z * self.pesos[b])
        X = np.hstack(partes)
        return X[0] if uno else X


def ajustar_escalador(bloques_train: dict[str, np.ndarray]) -> Escalador:
    """Ajusta media y desvío (`ddof=1`) por columna sobre `train` y pondera cada bloque."""
    _comprobar_claves(bloques_train, "ajustar_escalador")
    arrays = {b: np.asarray(bloques_train[b], dtype=np.float64) for b in BLOQUES}
    for b, a in arrays.items():
        if a.ndim != 2:
            raise ValueError(
                f"El bloque {b} tiene que ser un array 2D (n, d_b); tiene forma {a.shape}."
            )
    filas = {a.shape[0] for a in arrays.values()}
    if len(filas) != 1:
        raise ValueError(f"Los bloques tienen distinta cantidad de filas: {sorted(filas)}.")
    n = filas.pop()
    if n < 2:
        raise ValueError(f"Hacen falta al menos 2 filas de train para el escalador; hay {n}.")

    medias, desvios, columnas, pesos = {}, {}, {}, {}
    for b in BLOQUES:
        a = arrays[b]
        if np.isinf(a).any():
            raise ValueError(f"El bloque {b} tiene valores ±Inf.")
        cuenta = np.count_nonzero(~np.isnan(a), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            media = np.nanmean(a, axis=0)
            desvio = np.nanstd(a, axis=0, ddof=1)
        usadas = (cuenta >= 2) & (desvio > DESVIO_MIN)
        if not usadas.any():
            raise ValueError(f"El bloque {b} no tiene ninguna columna con varianza en train.")
        medias[b], desvios[b], columnas[b] = media, desvio, usadas
        pesos[b] = 1.0 / math.sqrt(int(usadas.sum()))
    return Escalador(medias=medias, desvios=desvios, columnas=columnas, pesos=pesos)


# --------------------------------------------------------------------------- archivo de características
def _leer_metadatos_validos(ruta: Path) -> list[dict[str, str]]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta} (el metadatos.csv del dataset de V2).")
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        encabezado = lector.fieldnames or []
        faltan = [col for col in COLUMNAS_REQUERIDAS if col not in encabezado]
        if faltan:
            raise ValueError(f"A {ruta} le faltan las columnas requeridas {faltan}.")
        validas: list[dict[str, str]] = []
        for fila in lector:
            if fila.get("valido") in VALORES_VALIDO:
                validas.append({col: fila[col] for col in COLUMNAS_REQUERIDAS})
        return validas


def generar_caracteristicas(
    cfg: dict,
    carpeta: str | os.PathLike | None = None,
    salida: str | os.PathLike | None = None,
    sobrescribir: bool = False,
) -> Path:
    """Extrae las características de los clips válidos de V2 y escribe `caracteristicas.npz`."""
    dir_prepro = (
        RAIZ / cfg["rutas"]["voz"] / "prepro"
        if carpeta is None
        else Path(os.fspath(carpeta))
    )
    ruta_salida = (
        dir_prepro / NOMBRE_NPZ if salida is None else Path(os.fspath(salida))
    )
    if ruta_salida.exists() and not sobrescribir:
        raise FileExistsError(
            f"{ruta_salida} ya existe; usá --sobrescribir (sobrescribir=True) para regenerarlo."
        )

    filas = _leer_metadatos_validos(dir_prepro / "metadatos.csv")
    particion = np.array([f["particion"] for f in filas], dtype=str)
    train = particion == "train"
    if int(train.sum()) < 2:
        raise ValueError(
            f"{dir_prepro / 'metadatos.csv'} tiene {int(train.sum())} filas válidas de train; se requieren al menos 2."
        )

    fs_esperada = int(cfg["voz"]["frecuencia_muestreo"])
    rutas_wav: list[Path] = []
    for f in filas:
        rel = f["archivo"]
        if not rel:
            raise FileNotFoundError("Una fila marcada como válida tiene 'archivo' vacío.")
        ruta_w = dir_prepro / rel
        if not ruta_w.is_file():
            raise FileNotFoundError(f"Falta el archivo WAV de una fila válida: {ruta_w}.")
        info = sf.info(str(ruta_w))
        if info.samplerate != fs_esperada:
            raise ValueError(
                f"El archivo {ruta_w} tiene frecuencia {info.samplerate} Hz; se esperaba {fs_esperada} Hz."
            )
        rutas_wav.append(ruta_w)

    crudos: dict[str, list[np.ndarray]] = {b: [] for b in BLOQUES}
    for ruta_w in rutas_wav:
        wav, _ = sf.read(str(ruta_w), dtype="float32")
        bl = caracteristicas(wav.reshape(-1), cfg)
        for b in BLOQUES:
            crudos[b].append(bl[b])

    apilados = {b: np.vstack(crudos[b]).astype(np.float64) for b in BLOQUES}
    escalador = ajustar_escalador({b: apilados[b][train] for b in BLOQUES})
    X = escalador.transformar(apilados)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ruta_salida,
        archivo=np.array([f["archivo"] for f in filas], dtype=str),
        particion=particion,
        origen=np.array([f["origen"] for f in filas], dtype=str),
        hablante=np.array([f["hablante"] for f in filas], dtype=str),
        clase=np.array([f["clase"] for f in filas], dtype=str),
        copia=np.array([int(f["copia"]) for f in filas], dtype=np.int64),
        **apilados,
        X=X,
    )
    return ruta_salida


# --------------------------------------------------------------------------- figura
_COLORES_CLASE = {
    "reconocer": "#1f77b4",
    "identificar": "#ff7f0e",
    "salir": "#2ca02c",
    "otro": "0.55",
}


def _coma(valor: float, decimales: int = 3) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def _silhouette_segura(X: np.ndarray, etiquetas: np.ndarray) -> float | None:
    grupos = len(np.unique(etiquetas))
    if grupos < 2 or len(etiquetas) <= grupos:
        return None
    return silhouette(X, etiquetas)


def generar_figura(
    cfg: dict,
    carpeta: str | os.PathLike | None = None,
    salida: str | os.PathLike | None = None,
) -> Path:
    """Dibuja la proyección PCA 2D y 3D de `caracteristicas.npz` y la guarda en `salida`."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if carpeta is None:
        ruta_npz = RAIZ / cfg["rutas"]["voz"] / "prepro" / NOMBRE_NPZ
    else:
        p_carp = Path(os.fspath(carpeta))
        ruta_npz = p_carp / NOMBRE_NPZ if p_carp.is_dir() else p_carp

    ruta_salida = (
        RAIZ / cfg["rutas"]["figuras"] / NOMBRE_FIGURA
        if salida is None
        else Path(os.fspath(salida))
    )

    with np.load(ruta_npz, allow_pickle=False) as datos:
        X = datos["X"]
        particion = datos["particion"]
        clase = datos["clase"]
        copia = datos["copia"]

    train = particion == "train"
    k = min(3, int(train.sum()) - 1, X.shape[1])
    pca = PCA(k).ajustar(X[train])
    P = np.hstack([pca.transformar(X), np.zeros((len(X), 3 - k))])
    proporcion = list(pca.proporcion_varianza) + [0.0] * (3 - k)

    comandos_cfg = list(cfg["voz"]["comandos"])
    es_cmd = np.isin(clase, comandos_cfg)
    tr_k0_cmd = train & (copia == 0) & es_cmd
    va_cmd = (particion == "val") & es_cmd
    s_tr = _silhouette_segura(X[tr_k0_cmd], clase[tr_k0_cmd])
    s_va = _silhouette_segura(X[va_cmd], clase[va_cmd])

    figura = Figure(figsize=(15, 6.6))
    FigureCanvasAgg(figura)
    ax2 = figura.add_subplot(1, 2, 1)
    ax3 = figura.add_subplot(1, 2, 2, projection="3d")

    estilos = (("train", "o", True, 26, 0.70), ("val", "s", False, 42, 0.95), ("test", "^", False, 42, 0.95))
    clases_orden = comandos_cfg + [cfg["voz"]["clase_rechazo"]]
    for cl in clases_orden:
        color = _COLORES_CLASE.get(str(cl), "#d62728")
        for part, marcador, relleno, tam, alfa in estilos:
            # En train graficamos k0 para que la nube de 1722 puntos sea legible, o todos si no hay k0
            m_base = (clase == cl) & (particion == part)
            m = m_base & (copia == 0) if (part == "train" and np.any(m_base & (copia == 0))) else m_base
            if not m.any():
                continue
            etiqueta = f"{cl} · {part}" + (" (k0)" if part == "train" else "")
            kw = dict(marker=marcador, s=tam, linewidths=1.1, alpha=alfa, label=etiqueta)
            if relleno:
                kw.update(c=color)
            else:
                kw.update(facecolors="none", edgecolors=color)
            ax2.scatter(P[m, 0], P[m, 1], **kw)
            ax3.scatter(P[m, 0], P[m, 1], P[m, 2], depthshade=False, **kw)

    ejes = [f"PC{i + 1} ({_coma(100 * proporcion[i], 1)} % de la varianza)" for i in range(3)]
    ax2.set_xlabel(ejes[0])
    ax2.set_ylabel(ejes[1])
    ax2.set_title("PC1–PC2")
    ax2.legend(fontsize=7, loc="best", ncol=2)
    ax2.grid(alpha=0.3)

    ax3.set_xlabel(ejes[0], fontsize=8)
    ax3.set_ylabel(ejes[1], fontsize=8)
    ax3.set_zlabel(ejes[2], fontsize=8)
    ax3.set_title("PC1–PC2–PC3")

    textos = [
        f"{nom} {_coma(val)}"
        for nom, val in (("train k0 (3 cmd)", s_tr), ("val (3 cmd)", s_va))
        if val is not None
    ]
    figura.suptitle(
        "V3 · PCA propio de las características de voz (MFCC + Δ + energía + ZCR + duración), ajustado con train"
        + (f"\nsilhouette (281d): {' · '.join(textos)}" if textos else ""),
        fontsize=11,
    )
    figura.subplots_adjust(left=0.06, right=0.97, bottom=0.1, top=0.86, wspace=0.12)

    buffer = io.BytesIO()
    figura.savefig(buffer, format="png", dpi=110)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_bytes(buffer.getvalue())
    return ruta_salida


# --------------------------------------------------------------------------- línea de comandos
def _contribucion(Zb: np.ndarray) -> float:
    """Media de ‖z_i − z_j‖² sobre los pares ordenados i ≠ j = 2 · Σ‖z_i − z̄‖² / (n − 1)."""
    centrado = Zb - Zb.mean(axis=0)
    return 2.0 * float(np.sum(centrado ** 2)) / (Zb.shape[0] - 1)


def _resumen(datos: dict, cfg: dict) -> list[str]:
    particion = datos["particion"]
    clase = datos["clase"]
    copia = datos["copia"]
    X = datos["X"]
    train = particion == "train"
    val = particion == "val"

    escalador = ajustar_escalador({b: datos[b][train] for b in BLOQUES})
    conteos = {b: int(escalador.columnas[b].sum()) for b in BLOQUES}
    tramos_col, inicio = {}, 0
    for b in BLOQUES:
        tramos_col[b] = slice(inicio, inicio + conteos[b])
        inicio += conteos[b]

    pca_full = PCA(n_componentes=30).ajustar(X[train])
    pca20 = PCA(n_componentes=20).ajustar(X[train])
    X_p30 = pca_full.transformar(X)
    X_p20 = pca20.transformar(X)

    lineas = [
        f"Vector escalado: d = {X.shape[1]} columnas ("
        + ", ".join(f"{b} {conteos[b]} de {datos[b].shape[1]}" for b in BLOQUES)
        + ")",
        "Varianza explicada PCA en train: "
        + f"PC1 {_coma(100 * pca_full.proporcion_varianza[0], 2)} %, "
        + f"k=2 {_coma(100 * float(np.sum(pca_full.proporcion_varianza[:2])), 2)} %, "
        + f"k=3 {_coma(100 * float(np.sum(pca_full.proporcion_varianza[:3])), 2)} %, "
        + f"k=20 {_coma(100 * float(np.sum(pca20.proporcion_varianza)), 2)} %, "
        + f"k=30 {_coma(100 * float(np.sum(pca_full.proporcion_varianza)), 2)} %",
    ]

    comandos_cfg = list(cfg["voz"]["comandos"])
    es_cmd = np.isin(clase, comandos_cfg)
    tr_cmd_all = train & es_cmd
    tr_cmd_k0 = train & (copia == 0) & es_cmd
    va_cmd = val & es_cmd

    lineas.append(
        f"Silhouette 3 comandos (train todos={int(tr_cmd_all.sum())}, train k0={int(tr_cmd_k0.sum())}, val={int(va_cmd.sum())}):"
    )
    for nom, mat in (("total 281d", X), ("PCA 30", X_p30), ("PCA 20", X_p20)):
        s_all = _silhouette_segura(mat[tr_cmd_all], clase[tr_cmd_all])
        s_k0 = _silhouette_segura(mat[tr_cmd_k0], clase[tr_cmd_k0])
        s_va = _silhouette_segura(mat[va_cmd], clase[va_cmd])
        lineas.append(
            f"  {nom:<12} train {_coma(s_all) if s_all is not None else '—'}   "
            f"train k0 {_coma(s_k0) if s_k0 is not None else '—'}   "
            f"val {_coma(s_va) if s_va is not None else '—'}"
        )
    for b in BLOQUES:
        s_all = _silhouette_segura(X[tr_cmd_all][:, tramos_col[b]], clase[tr_cmd_all])
        s_k0 = _silhouette_segura(X[tr_cmd_k0][:, tramos_col[b]], clase[tr_cmd_k0])
        s_va = _silhouette_segura(X[va_cmd][:, tramos_col[b]], clase[va_cmd])
        lineas.append(
            f"  {b:<12} train {_coma(s_all) if s_all is not None else '—'}   "
            f"train k0 {_coma(s_k0) if s_k0 is not None else '—'}   "
            f"val {_coma(s_va) if s_va is not None else '—'}"
        )

    lineas.append(
        "Contribución a la distancia en train: "
        + ", ".join(f"{b} {_coma(_contribucion(X[train][:, tramos_col[b]]))}" for b in BLOQUES)
    )
    lineas.append(
        "Contribución a la distancia en val:   "
        + ", ".join(f"{b} {_coma(_contribucion(X[val][:, tramos_col[b]]))}" for b in BLOQUES)
    )
    return lineas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera las características de V3 (caracteristicas.npz) y la figura del PCA (pca_v3.png)."
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="vuelve a escribir caracteristicas.npz si ya existe",
    )
    args = parser.parse_args(argv)
    cfg = cargar_config()
    inicio = time.perf_counter()
    try:
        ruta = generar_caracteristicas(cfg, sobrescribir=args.sobrescribir)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1
    medio = time.perf_counter()
    figura = generar_figura(cfg)
    fin = time.perf_counter()
    with np.load(ruta, allow_pickle=False) as d:
        datos = {clave: d[clave] for clave in d.files}
    n = len(datos["archivo"])
    print(
        f"Características: {ruta} ({n} clips válidos en {_coma(medio - inicio, 2)} s, "
        f"{_coma(1000 * (medio - inicio) / n, 2)} ms por clip)"
    )
    print(f"Figura: {figura} ({_coma(fin - medio, 2)} s)")
    for linea in _resumen(datos, cfg):
        print(linea)
    return 0


if __name__ == "__main__":
    sys.exit(main())

