"""Tests de la etapa V3: características acústicas por bloques, escalador ponderado y PCA propio.

Verifica cada criterio de la sección 6 del contrato ``v3_caracteristicas.md``.
"""

from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf
import yaml
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import silhouette_score

RAIZ = Path(__file__).resolve().parents[1]
BLOQUES_ESPERADOS: tuple[str, ...] = ("mfcc", "delta", "energia", "zcr", "duracion")


# =============================================================================
# Funciones auxiliares y oráculo del contrato (secciones 4 y 5)
# =============================================================================


def clip_sintetico(
    f0_hz: float,
    mod_hz: float,
    dur_palabra_s: float,
    amp: float,
    semilla: int,
    fs: int = 16000,
    margen_muestras: int = 640,
) -> np.ndarray:
    """Generador determinista de clips sintéticos según la sección 5 del contrato."""
    l_palabra = int(round(dur_palabra_s * fs))
    n_tot = l_palabra + 2 * margen_muestras
    t = np.arange(n_tot, dtype=np.float64) / fs
    envolvente = 1.0 + 0.4 * np.sin(2.0 * np.pi * mod_hz * t)
    portadora = (
        0.6 * np.sin(2.0 * np.pi * f0_hz * t)
        + 0.3 * np.sin(2.0 * np.pi * 2.0 * f0_hz * t)
        + 0.1 * np.sin(2.0 * np.pi * 3.0 * f0_hz * t)
    )
    ruido = np.random.default_rng(semilla).normal(0.0, 0.02, size=n_tot)
    senal = (envolvente * portadora + ruido) * amp
    return senal.astype(np.float32)


def _oraculo_estadisticas_tramos(matriz: np.ndarray, k: int) -> np.ndarray:
    """Oráculo exacto de estadísticas por tramo según la sección 4.1 del contrato."""
    arr = np.asarray(matriz, dtype=np.float64)
    _, n_tramas = arr.shape
    partes: list[np.ndarray] = []
    for i in range(k):
        a = int(np.floor(i * n_tramas / k))
        b = max(a + 1, int(np.floor((i + 1) * n_tramas / k)))
        s = arr[:, a:b]
        partes.append(np.concatenate([s.mean(axis=1), s.std(axis=1, ddof=0)]))
    return np.concatenate(partes).astype(np.float64)


def _oraculo_caracteristicas(clip: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Oráculo exacto de los 5 bloques según la sección 4.2 del contrato."""
    voz = cfg["voz"]
    mfcc_cfg = voz["mfcc"]
    fs = int(voz["frecuencia_muestreo"])
    m_muestras = int(round(voz["preprocesamiento"]["margen_ms"] * fs / 1000))
    w_len = int(round(mfcc_cfg["ventana_ms"] * fs / 1000))
    hop = int(round(mfcc_cfg["avance_ms"] * fs / 1000))
    n_fft = int(mfcc_cfg["n_fft"])
    n_mfcc = int(mfcc_cfg["n_mfcc"])
    n_mels = int(mfcc_cfg["n_mels"])
    fmin = float(voz["pasabanda_hz"][0])
    fmax = float(mfcc_cfg["fmax"])
    ancho_delta = int(mfcc_cfg["delta_ventana"])
    piso_rms = float(mfcc_cfg["piso_rms"])
    k_tramos = int(voz["tramos"])
    m_tramas = m_muestras // hop

    y = np.asarray(clip, dtype=np.float32)

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
    rms = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(fr**2, axis=0)), piso_rms))[None, :]
    zcr = (np.abs(np.diff(np.sign(fr), axis=0)) > 0).mean(axis=0)[None, :]

    n_cols = min(w.shape[1], rms.shape[1])
    return {
        "mfcc": _oraculo_estadisticas_tramos(w[:, :n_cols], k_tramos),
        "delta": _oraculo_estadisticas_tramos(d[:, :n_cols], k_tramos),
        "energia": _oraculo_estadisticas_tramos(rms[:, :n_cols], k_tramos),
        "zcr": _oraculo_estadisticas_tramos(zcr[:, :n_cols], k_tramos),
        "duracion": np.array([(len(clip) - 2 * m_muestras) / float(fs)], dtype=np.float64),
    }


def _contribucion_bloque_pares(z_bloque: np.ndarray) -> float:
    """Media sobre pares ordenados i != j de ||z_{i,b} - z_{j,b}||^2 (sección 4.3)."""
    n = z_bloque.shape[0]
    dif = z_bloque[:, None, :] - z_bloque[None, :, :]
    dist2 = np.sum(dif**2, axis=-1)
    return float(np.sum(dist2) / (n * (n - 1)))


# =============================================================================
# Fixtures de alcance de módulo (muestra sintética y dataset en tmp_path)
# =============================================================================


@pytest.fixture(scope="module")
def cfg_real() -> dict:
    """Carga la configuración real del repositorio mediante src.config.cargar_config()."""
    from src.config import cargar_config

    return cargar_config()


@pytest.fixture(scope="module")
def muestra_sintetica(cfg_real: dict) -> dict:
    """Construye una sola vez los 18 clips de train y 9 de val de la sección 5 y sus bloques."""
    from src.voz.caracteristicas import BLOQUES, caracteristicas

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    m_muestras = int(round(cfg_real["voz"]["preprocesamiento"]["margen_ms"] * fs / 1000))

    especificaciones = [
        # (clase, mod_hz, f0_centro, f0_delta, dur_centro, dur_delta)
        ("reconocer", 3.0, 220.0, 15.0, 0.75, 0.05),
        ("identificar", 6.0, 520.0, 25.0, 1.15, 0.06),
        ("salir", 1.5, 1200.0, 40.0, 0.42, 0.03),
    ]

    clips_train: list[np.ndarray] = []
    clases_train: list[str] = []
    clips_val: list[np.ndarray] = []
    clases_val: list[str] = []

    semilla = 100
    for clase, mod_hz, f0_c, f0_d, dur_c, dur_d in especificaciones:
        despl_train = np.linspace(-1.0, 1.0, 6)
        amps_train = np.linspace(0.18, 0.32, 6)
        for idx in range(6):
            f0 = f0_c + despl_train[idx] * f0_d
            dur = dur_c + despl_train[5 - idx] * dur_d
            clip = clip_sintetico(
                f0_hz=f0,
                mod_hz=mod_hz,
                dur_palabra_s=dur,
                amp=float(amps_train[idx]),
                semilla=semilla,
                fs=fs,
                margen_muestras=m_muestras,
            )
            clips_train.append(clip)
            clases_train.append(clase)
            semilla += 1

        despl_val = np.linspace(-0.8, 0.8, 3)
        amps_val = np.linspace(0.20, 0.30, 3)
        for idx in range(3):
            f0 = f0_c + despl_val[idx] * f0_d
            dur = dur_c + despl_val[2 - idx] * dur_d
            clip = clip_sintetico(
                f0_hz=f0,
                mod_hz=mod_hz,
                dur_palabra_s=dur,
                amp=float(amps_val[idx]),
                semilla=semilla,
                fs=fs,
                margen_muestras=m_muestras,
            )
            clips_val.append(clip)
            clases_val.append(clase)
            semilla += 1

    dicts_train = [caracteristicas(c, cfg_real) for c in clips_train]
    dicts_val = [caracteristicas(c, cfg_real) for c in clips_val]

    bloques_train = {
        b: np.vstack([d[b] for d in dicts_train]).astype(np.float64) for b in BLOQUES
    }
    bloques_val = {
        b: np.vstack([d[b] for d in dicts_val]).astype(np.float64) for b in BLOQUES
    }

    return {
        "clips_train": clips_train,
        "clases_train": np.array(clases_train, dtype=str),
        "bloques_train": bloques_train,
        "clips_val": clips_val,
        "clases_val": np.array(clases_val, dtype=str),
        "bloques_val": bloques_val,
    }


def _armar_dataset_v2(
    carpeta: Path,
    cfg: dict,
    semilla_val_test: int = 500,
) -> list[dict]:
    """Crea en carpeta un dataset preprocesado V2 sintético con WAV y metadatos.csv."""
    carpeta.mkdir(parents=True, exist_ok=True)
    fs = int(cfg["voz"]["frecuencia_muestreo"])
    m_muestras = int(round(cfg["voz"]["preprocesamiento"]["margen_ms"] * fs / 1000))

    try:
        from src.voz.preprocesamiento import COLUMNAS_V2

        columnas_csv = list(COLUMNAS_V2)
    except Exception:
        columnas_csv = [
            "id",
            "archivo",
            "hablante",
            "origen",
            "particion",
            "clase",
            "copia",
            "valido",
            "motivo",
            "duracion_crudo_s",
            "duracion_palabra_s",
            "rms_dbfs",
            "pico_dbfs",
            "snr_db",
        ]

    for col_req in ("archivo", "particion", "origen", "hablante", "clase", "copia", "valido"):
        if col_req not in columnas_csv:
            columnas_csv.append(col_req)

    params_clase = {
        "reconocer": (220.0, 3.0, 0.75),
        "identificar": (520.0, 6.0, 1.15),
        "salir": (1200.0, 1.5, 0.42),
        "otro": (800.0, 4.5, 0.90),
    }
    clases_4 = ["reconocer", "identificar", "salir", "otro"]

    # Especificación de filas: 16 válidas train (12 k0 + 4 k1), 8 válidas val, 4 válidas test,
    # más 3 filas no válidas (valido=0, archivo="") intercaladas sin WAV en disco.
    plan_filas: list[tuple[str, str, int, int, str]] = []
    # Fila inválida inicial
    plan_filas.append(("train", "otro", 0, 0, "hab_01"))
    # 12 filas válidas train con copia=0 (3 por cada una de las 4 clases)
    for rep in range(3):
        for c in clases_4:
            plan_filas.append(("train", c, 0, 1, f"hab_0{(rep % 3) + 1}"))
    # 4 filas válidas train con copia=1 (1 por clase)
    for c in clases_4:
        plan_filas.append(("train", c, 1, 1, "hab_01"))
    # Fila inválida intermedia
    plan_filas.append(("val", "otro", 0, 0, "hab_04"))
    # 8 filas válidas val (2 por cada una de las 4 clases)
    for rep in range(2):
        for c in clases_4:
            plan_filas.append(("val", c, 0, 1, f"hab_0{4 + rep}"))
    # 4 filas válidas test (1 por cada una de las 4 clases)
    for c in clases_4:
        plan_filas.append(("test", c, 0, 1, "hab_06"))
    # Fila inválida final
    plan_filas.append(("test", "otro", 0, 0, "hab_06"))

    filas_guardadas: list[dict] = []
    with (carpeta / "metadatos.csv").open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=columnas_csv)
        writer.writeheader()

        for idx, (particion, clase, copia, valido, hablante) in enumerate(plan_filas):
            f0_base, mod_hz, dur_base = params_clase[clase]
            if valido == 1:
                nombre_wav = f"clip_{idx:03d}_{particion}_{clase}_k{copia}.wav"
                semilla = (1000 + idx) if particion == "train" else (semilla_val_test + idx * 17)
                variacion = ((idx % 5) - 2) * 0.015
                dur_s = dur_base + variacion
                f0_s = f0_base * (1.0 + variacion)
                amp = 0.22 + (idx % 4) * 0.02
                wav = clip_sintetico(
                    f0_hz=f0_s,
                    mod_hz=mod_hz,
                    dur_palabra_s=dur_s,
                    amp=amp,
                    semilla=semilla,
                    fs=fs,
                    margen_muestras=m_muestras,
                )
                sf.write(carpeta / nombre_wav, wav, fs, subtype="FLOAT")
                dur_palabra_real = (len(wav) - 2 * m_muestras) / float(fs)
            else:
                nombre_wav = ""
                dur_palabra_real = 0.0

            valores_base = {
                "id": f"id_{idx:03d}",
                "archivo": nombre_wav,
                "hablante": hablante,
                "origen": "notebook",
                "particion": particion,
                "clase": clase,
                "copia": copia,
                "valido": valido,
                "motivo": "ok" if valido == 1 else "corto",
                "duracion_crudo_s": f"{dur_palabra_real + 0.2:.4f}",
                "duracion_palabra_s": f"{dur_palabra_real:.4f}",
                "rms_dbfs": "-26.0",
                "pico_dbfs": "-10.0",
                "snr_db": "25.0",
            }
            fila_csv = {col: valores_base.get(col, "0") for col in columnas_csv}
            writer.writerow(fila_csv)
            filas_guardadas.append(valores_base)

    return filas_guardadas


# =============================================================================
# 6.1 Configuración
# =============================================================================


def test_config_claves_v3_tipos_rangos_y_valores_reales(cfg_real: dict) -> None:
    """Verifica que las claves de la sección 1 existen en config.yaml con su tipo, rango y valor real."""
    assert "voz" in cfg_real, "Falta la sección 'voz' en config.yaml."
    voz = cfg_real["voz"]
    assert "mfcc" in voz, "Falta la subsección 'voz.mfcc' en config.yaml."
    mfcc = voz["mfcc"]

    for clave in ("n_mfcc", "n_mels", "fmax", "ventana_ms", "avance_ms", "n_fft", "delta_ventana", "piso_rms"):
        assert clave in mfcc, f"Falta la clave 'voz.mfcc.{clave}' en config.yaml."

    n_mfcc = mfcc["n_mfcc"]
    assert isinstance(n_mfcc, int) and not isinstance(n_mfcc, bool) and n_mfcc == 13, (
        f"voz.mfcc.n_mfcc esperado 13 (int), medido {n_mfcc!r} ({type(n_mfcc).__name__})."
    )

    n_mels = mfcc["n_mels"]
    assert isinstance(n_mels, int) and not isinstance(n_mels, bool), (
        f"voz.mfcc.n_mels debe ser int, medido {type(n_mels).__name__}."
    )
    assert 26 <= n_mels <= 40, f"voz.mfcc.n_mels fuera de rango [26, 40]: medido {n_mels}."
    assert n_mels == 40, f"En config.yaml real voz.mfcc.n_mels debe ser 40, medido {n_mels}."

    assert mfcc["fmax"] == 4000, f"voz.mfcc.fmax esperado 4000, medido {mfcc['fmax']}."
    assert mfcc["ventana_ms"] == 25, f"voz.mfcc.ventana_ms esperado 25, medido {mfcc['ventana_ms']}."
    assert mfcc["avance_ms"] == 10, f"voz.mfcc.avance_ms esperado 10, medido {mfcc['avance_ms']}."

    n_fft = mfcc["n_fft"]
    assert isinstance(n_fft, int) and not isinstance(n_fft, bool), (
        f"voz.mfcc.n_fft debe ser int, medido {type(n_fft).__name__}."
    )
    assert n_fft == 512, f"voz.mfcc.n_fft esperado 512, medido {n_fft}."

    delta_ventana = mfcc["delta_ventana"]
    assert isinstance(delta_ventana, int) and not isinstance(delta_ventana, bool), (
        f"voz.mfcc.delta_ventana debe ser int, medido {type(delta_ventana).__name__}."
    )
    assert 3 <= delta_ventana <= 9 and delta_ventana % 2 == 1, (
        f"voz.mfcc.delta_ventana debe ser impar en [3, 9], medido {delta_ventana}."
    )
    assert delta_ventana == 5, (
        f"En config.yaml real voz.mfcc.delta_ventana debe ser 5, medido {delta_ventana}."
    )

    piso_rms = mfcc["piso_rms"]
    assert isinstance(piso_rms, (int, float)) and not isinstance(piso_rms, bool), (
        f"voz.mfcc.piso_rms debe ser numérico, medido {type(piso_rms).__name__}."
    )
    assert 0.0 < float(piso_rms) <= 1e-3, (
        f"voz.mfcc.piso_rms debe estar en (0, 1e-3], medido {piso_rms}."
    )
    assert float(piso_rms) == pytest.approx(1.0e-6, rel=1e-12), (
        f"En config.yaml real voz.mfcc.piso_rms debe ser 1.0e-6, medido {piso_rms}."
    )

    fmin = voz["pasabanda_hz"][0]
    assert fmin == 80, f"voz.pasabanda_hz[0] (fmin) esperado 80 Hz, medido {fmin}."


@pytest.mark.parametrize("clave_faltante", ["n_fft", "delta_ventana", "piso_rms"])
def test_cargar_config_exige_claves_nuevas_v3(tmp_path: Path, clave_faltante: str) -> None:
    """Verifica que cargar_config lanza ValueError con el camino completo si falta una clave nueva de V3."""
    from src.config import cargar_config

    datos = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    datos["voz"]["mfcc"].pop(clave_faltante, None)

    ruta_copia = tmp_path / f"config_sin_{clave_faltante}.yaml"
    ruta_copia.write_text(yaml.safe_dump(datos, sort_keys=False), encoding="utf-8")

    camino_esperado = f"voz.mfcc.{clave_faltante}"
    with pytest.raises(ValueError) as exc_info:
        cargar_config(ruta_copia)

    mensaje = str(exc_info.value)
    assert camino_esperado in mensaje, (
        f"El ValueError debe mencionar el camino completo '{camino_esperado}', pero dijo: {mensaje!r}."
    )


# =============================================================================
# 6.2 PCA propio y silhouette propia (src/core_ml/)
# =============================================================================


@pytest.mark.parametrize("k", [2, 3, 20, 30, None])
def test_pca_propio_coincide_con_sklearn(k: int | None) -> None:
    """Verifica que PCA propio coincide con sklearn.decomposition.PCA en matrices (60, 281)."""
    from src.core_ml.pca import PCA

    rng = np.random.default_rng(20261024)
    # Matriz (60, 281) con valores singulares bien separados para evitar degeneración
    u_base = rng.standard_normal((60, 60))
    v_base = rng.standard_normal((60, 281))
    escalas = np.linspace(15.0, 0.5, 60)
    x_datos = (u_base * escalas) @ v_base + rng.normal(0.0, 0.05, size=(60, 281))

    pca = PCA(n_componentes=k)
    retorno = pca.ajustar(x_datos)
    assert retorno is pca, "PCA.ajustar(X) debe devolver self."

    k_sk = k if k is not None else min(x_datos.shape[0] - 1, x_datos.shape[1])
    sk_pca = SklearnPCA(n_components=k_sk, svd_solver="full").fit(x_datos)

    for nombre_attr in ("media", "componentes", "varianza", "proporcion_varianza"):
        valor_attr = getattr(pca, nombre_attr)
        assert isinstance(valor_attr, np.ndarray) and valor_attr.dtype == np.float64, (
            f"PCA.{nombre_attr} debe ser np.ndarray float64, medido {type(valor_attr)} con dtype {getattr(valor_attr, 'dtype', None)}."
        )

    assert np.allclose(pca.media, sk_pca.mean_, rtol=1e-6, atol=1e-6), (
        f"Diferencia máxima en media para k={k}: {np.max(np.abs(pca.media - sk_pca.mean_)):.3e} (esperado <= 1e-6)."
    )
    assert np.allclose(pca.varianza, sk_pca.explained_variance_, rtol=1e-6, atol=1e-6), (
        f"Diferencia máxima en varianza para k={k}: "
        f"{np.max(np.abs(pca.varianza - sk_pca.explained_variance_)):.3e} (esperado <= 1e-6)."
    )
    assert np.allclose(
        pca.proporcion_varianza, sk_pca.explained_variance_ratio_, rtol=1e-6, atol=1e-6
    ), (
        f"Diferencia máxima en proporcion_varianza para k={k}: "
        f"{np.max(np.abs(pca.proporcion_varianza - sk_pca.explained_variance_ratio_)):.3e} (esperado <= 1e-6)."
    )

    # Verificar formas, ortonormalidad y que en cada componente la entrada de mayor |valor| es positiva
    assert pca.componentes.shape == sk_pca.components_.shape, (
        f"Forma de componentes para k={k} esperada {sk_pca.components_.shape}, medida {pca.componentes.shape}."
    )
    gram = pca.componentes @ pca.componentes.T
    assert np.allclose(gram, np.eye(pca.componentes.shape[0]), rtol=1e-6, atol=1e-6), (
        f"Las filas de PCA.componentes no son ortonormales para k={k}."
    )
    idx_max = np.argmax(np.abs(pca.componentes), axis=1)
    entradas_max = pca.componentes[np.arange(pca.componentes.shape[0]), idx_max]
    assert np.all(entradas_max > 0), (
        f"Todas las componentes deben tener su entrada de mayor |valor| positiva; mínimo medido: {np.min(entradas_max):.3e}."
    )

    # Comparar componentes salvo el signo de cada fila (para autovalores no nulos; cuando k=None y n=60,
    # la componente 60 corresponde al autovalor 0 de centrado, pero con SVD también coincide salvo signo)
    n_efectivas = pca.componentes.shape[0] if k is not None else 59
    signos = np.where(
        np.sum(pca.componentes[:n_efectivas] * sk_pca.components_[:n_efectivas], axis=1) >= 0.0,
        1.0,
        -1.0,
    )
    comp_sk_alineadas = sk_pca.components_[:n_efectivas] * signos[:, None]
    assert np.allclose(pca.componentes[:n_efectivas], comp_sk_alineadas, rtol=1e-6, atol=1e-6), (
        f"Diferencia máxima en componentes para k={k}: "
        f"{np.max(np.abs(pca.componentes[:n_efectivas] - comp_sk_alineadas)):.3e} (esperado <= 1e-6)."
    )

    proy_propia = pca.transformar(x_datos)
    proy_ajustar_transf = PCA(n_componentes=k).ajustar_transformar(x_datos)
    proy_sk = sk_pca.transform(x_datos)

    signos_proy = np.where(
        np.sum(pca.componentes * sk_pca.components_, axis=1) >= 0.0,
        1.0,
        -1.0,
    )
    proy_sk_alineada = proy_sk * signos_proy[None, :]

    assert np.allclose(proy_propia, proy_sk_alineada, rtol=1e-6, atol=1e-6), (
        f"Diferencia máxima en transformar(X) para k={k}: "
        f"{np.max(np.abs(proy_propia - proy_sk_alineada)):.3e} (esperado <= 1e-6)."
    )
    assert np.allclose(proy_ajustar_transf, proy_propia, rtol=1e-12, atol=1e-12), (
        "ajustar_transformar(X) debe coincidir con ajustar(X).transformar(X)."
    )


def test_silhouette_propia_coincide_con_sklearn() -> None:
    """Verifica que silhouette de src.core_ml.metricas coincide con sklearn.metrics.silhouette_score."""
    from src.core_ml.metricas import silhouette

    rng = np.random.default_rng(42)
    centros = {
        "reconocer": np.full(20, -3.0),
        "identificar": np.full(20, 0.0),
        "salir": np.full(20, 3.0),
        "otro": np.full(20, 6.0),
    }
    filas: list[np.ndarray] = []
    etiquetas: list[str] = []
    for etiqueta, centro in centros.items():
        filas.append(centro + rng.normal(0.0, 0.8, size=(12, 20)))
        etiquetas.extend([etiqueta] * 12)

    x_mat = np.vstack(filas)
    etiquetas_arr = np.array(etiquetas, dtype=str)

    sil_propia = silhouette(x_mat, etiquetas_arr)
    sil_sk = float(silhouette_score(x_mat, etiquetas_arr))

    assert isinstance(sil_propia, float), (
        f"silhouette debe devolver float, medido {type(sil_propia).__name__}."
    )
    assert abs(sil_propia - sil_sk) <= 1e-9, (
        f"silhouette propia ({sil_propia:.12f}) difiere de sklearn ({sil_sk:.12f}) en "
        f"{abs(sil_propia - sil_sk):.3e} > 1e-9."
    )


@pytest.mark.parametrize(
    "ruta_relativa",
    [
        "src/core_ml/pca.py",
        "src/core_ml/metricas.py",
        "src/voz/caracteristicas.py",
    ],
)
def test_modulos_no_importan_sklearn(ruta_relativa: str) -> None:
    """Verifica que pca.py, metricas.py y caracteristicas.py no contienen 'sklearn'."""
    ruta = RAIZ / ruta_relativa
    assert ruta.is_file(), f"No existe el archivo requerido: {ruta}."
    contenido = ruta.read_text(encoding="utf-8")
    assert "sklearn" not in contenido, (
        f"El archivo {ruta_relativa} no debe importar ni mencionar 'sklearn'."
    )


# =============================================================================
# 6.3 Estadísticas por tramo y formato de bloques
# =============================================================================


def test_largos_de_bloque_real_y_dinamico(cfg_real: dict) -> None:
    """Verifica largos_de_bloque con el cfg real (suma 281) y ante cambios en tramos o n_mfcc."""
    from src.voz.caracteristicas import BLOQUES, largos_de_bloque

    assert tuple(BLOQUES) == BLOQUES_ESPERADOS, (
        f"BLOQUES esperado {BLOQUES_ESPERADOS}, medido {BLOQUES!r}."
    )

    esperado_real = {"mfcc": 130, "delta": 130, "energia": 10, "zcr": 10, "duracion": 1}
    medido_real = largos_de_bloque(cfg_real)
    assert medido_real == esperado_real, (
        f"largos_de_bloque(cfg_real) esperado {esperado_real}, medido {medido_real}."
    )
    assert sum(medido_real.values()) == 281, (
        f"La suma de largos_de_bloque con cfg_real debe ser 281, medido {sum(medido_real.values())}."
    )

    cfg_copia = copy.deepcopy(cfg_real)
    cfg_copia["voz"]["tramos"] = 7
    cfg_copia["voz"]["mfcc"]["n_mfcc"] = 10
    esperado_mod = {"mfcc": 140, "delta": 140, "energia": 14, "zcr": 14, "duracion": 1}
    medido_mod = largos_de_bloque(cfg_copia)
    assert medido_mod == esperado_mod, (
        f"largos_de_bloque dinámico con K=7, n_mfcc=10 esperado {esperado_mod}, medido {medido_mod}."
    )


@pytest.mark.parametrize(
    ("filas", "n_tramas", "k_tramos"),
    [
        (13, 72, 5),
        (1, 27, 5),
        (4, 5, 5),
        (3, 3, 5),  # N < k: tramos de 1 columna con desvío 0
        (2, 1, 5),  # N = 1 < k: todos los tramos toman la única columna con desvío 0
    ],
)
def test_estadisticas_tramos_exactitud_y_n_menor_que_k(
    filas: int, n_tramas: int, k_tramos: int
) -> None:
    """Verifica estadisticas_tramos en cada tramo incluyendo el caso N < k."""
    from src.voz.caracteristicas import estadisticas_tramos

    rng = np.random.default_rng(123 + filas * 10 + n_tramas)
    matriz = rng.standard_normal((filas, n_tramas))

    medido = estadisticas_tramos(matriz, k_tramos)
    esperado = _oraculo_estadisticas_tramos(matriz, k_tramos)

    assert isinstance(medido, np.ndarray) and medido.ndim == 1 and medido.dtype == np.float64, (
        f"estadisticas_tramos debe devolver np.ndarray 1D float64, medido shape={medido.shape}, dtype={medido.dtype}."
    )
    assert len(medido) == 2 * k_tramos * filas, (
        f"Largo esperado {2 * k_tramos * filas}, medido {len(medido)}."
    )
    assert np.allclose(medido, esperado, rtol=1e-12, atol=1e-12), (
        f"Diferencia máxima en estadisticas_tramos (F={filas}, N={n_tramas}, k={k_tramos}): "
        f"{np.max(np.abs(medido - esperado)):.3e}."
    )
    if n_tramas == 1:
        for i in range(k_tramos):
            desvios_tramo = medido[(2 * i + 1) * filas : (2 * i + 2) * filas]
            assert np.all(desvios_tramo == 0.0), (
                f"Con N=1 los desvíos del tramo {i} deben ser 0.0, medido {desvios_tramo}."
            )


@pytest.mark.parametrize(
    ("matriz_invalida", "k_invalido", "motivo"),
    [
        (np.ones(10), 5, "matriz 1D"),
        (np.ones((2, 5, 3)), 5, "matriz 3D"),
        (np.empty((3, 0)), 5, "N == 0 columnas"),
        (np.empty((0, 5)), 5, "F == 0 filas"),
        (np.array([[1.0, np.nan, 2.0]]), 2, "matriz con NaN"),
        (np.array([[1.0, np.inf, 2.0]]), 2, "matriz con +Inf"),
        (np.array([[1.0, -np.inf, 2.0]]), 2, "matriz con -Inf"),
        (np.ones((2, 10)), 0, "k == 0"),
        (np.ones((2, 10)), -3, "k < 0"),
        (np.ones((2, 10)), True, "k es bool True"),
        (np.ones((2, 10)), False, "k es bool False"),
    ],
)
def test_estadisticas_tramos_errores(
    matriz_invalida: np.ndarray, k_invalido: int, motivo: str
) -> None:
    """Verifica que estadisticas_tramos lanza ValueError ante entradas inválidas."""
    from src.voz.caracteristicas import estadisticas_tramos

    with pytest.raises(ValueError, match=".*"):
        estadisticas_tramos(matriz_invalida, k_invalido)


def test_caracteristicas_oraculo_exacto_e_inmutabilidad(cfg_real: dict) -> None:
    """Verifica que caracteristicas coincide con el oráculo de 4.2 en 6 duraciones, no muta clip y es determinista."""
    from src.voz.caracteristicas import BLOQUES, caracteristicas, largos_de_bloque

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    m_muestras = int(round(cfg_real["voz"]["preprocesamiento"]["margen_ms"] * fs / 1000))
    largos = largos_de_bloque(cfg_real)

    casos = [
        (200.0, 2.5, 0.30, 0.25, 11),
        (310.0, 3.5, 0.48, 0.20, 22),
        (480.0, 5.0, 0.75, 0.30, 33),
        (650.0, 4.0, 1.10, 0.18, 44),
        (920.0, 2.0, 1.55, 0.28, 55),
        (1150.0, 1.5, 2.00, 0.22, 66),
    ]

    for f0, mod, dur, amp, sem in casos:
        clip = clip_sintetico(f0, mod, dur, amp, sem, fs=fs, margen_muestras=m_muestras)
        copia_clip = clip.copy()

        res1 = caracteristicas(clip, cfg_real)
        res2 = caracteristicas(clip, cfg_real)
        oraculo = _oraculo_caracteristicas(clip, cfg_real)

        assert np.array_equal(clip, copia_clip), (
            f"caracteristicas modificó el array de entrada para dur={dur}s."
        )
        assert set(res1.keys()) == set(BLOQUES), (
            f"Claves devueltas {set(res1.keys())} no coinciden con BLOQUES {set(BLOQUES)}."
        )

        for b in BLOQUES:
            arr = res1[b]
            assert isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.dtype == np.float64, (
                f"Bloque '{b}' debe ser np.ndarray 1D float64, medido shape={arr.shape}, dtype={arr.dtype}."
            )
            assert len(arr) == largos[b], (
                f"Largo del bloque '{b}' esperado {largos[b]}, medido {len(arr)}."
            )
            assert np.all(np.isfinite(arr)), f"El bloque '{b}' contiene NaN o ±Inf para dur={dur}s."
            assert np.array_equal(arr, res2[b]), (
                f"Dos llamadas seguidas a caracteristicas dieron resultados distintos en '{b}'."
            )
            max_diff = float(np.max(np.abs(arr - oraculo[b])))
            assert np.allclose(arr, oraculo[b], rtol=1e-9, atol=1e-9), (
                f"Bloque '{b}' para dur={dur}s difiere del oráculo 4.2 en {max_diff:.3e} (tolerancia 1e-9)."
            )


@pytest.mark.parametrize("factor_escala", [0.25, 4.0])
def test_caracteristicas_invarianza_escala_y_duracion(
    cfg_real: dict, factor_escala: float
) -> None:
    """Verifica invarianza de mfcc, delta, zcr, duracion y desvíos de energia, y salto de 20·log10(c) en medias de energia."""
    from src.voz.caracteristicas import caracteristicas

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    m_muestras = int(round(cfg_real["voz"]["preprocesamiento"]["margen_ms"] * fs / 1000))

    clip_base = clip_sintetico(
        f0_hz=350.0,
        mod_hz=3.5,
        dur_palabra_s=0.85,
        amp=0.20,
        semilla=77,
        fs=fs,
        margen_muestras=m_muestras,
    )
    clip_escalado = (clip_base * np.float32(factor_escala)).astype(np.float32)

    base = caracteristicas(clip_base, cfg_real)
    esc = caracteristicas(clip_escalado, cfg_real)

    for bloque_inv in ("mfcc", "delta", "zcr", "duracion"):
        diff = float(np.max(np.abs(esc[bloque_inv] - base[bloque_inv])))
        assert np.allclose(esc[bloque_inv], base[bloque_inv], rtol=1e-5, atol=1e-5), (
            f"El bloque '{bloque_inv}' no fue invariante al escalar por c={factor_escala}: "
            f"diferencia máxima {diff:.3e} > 1e-5."
        )

    # En energia (F=1), cada tramo i aporta [media_i, desvio_i]: pares = medias, impares = desvíos
    medias_base = base["energia"][0::2]
    desvios_base = base["energia"][1::2]
    medias_esc = esc["energia"][0::2]
    desvios_esc = esc["energia"][1::2]

    diff_desvios = float(np.max(np.abs(desvios_esc - desvios_base)))
    assert np.allclose(desvios_esc, desvios_base, rtol=1e-5, atol=1e-5), (
        f"Los K desvíos de energia deben ser invariantes ante c={factor_escala}: "
        f"diferencia máxima {diff_desvios:.3e} > 1e-5."
    )

    salto_esperado_db = 20.0 * np.log10(factor_escala)
    salto_medido_db = medias_esc - medias_base
    diff_medias = float(np.max(np.abs(salto_medido_db - salto_esperado_db)))
    assert np.allclose(salto_medido_db, salto_esperado_db, rtol=1e-4, atol=1e-4), (
        f"Las K medias de energia debían aumentar en {salto_esperado_db:.6f} dB para c={factor_escala}, "
        f"pero se midió {salto_medido_db} (dif máx {diff_medias:.3e} > 1e-4)."
    )

    dur_esperada = (len(clip_base) - 2 * m_muestras) / float(fs)
    dur_medida = float(base["duracion"][0])
    assert abs(dur_medida - dur_esperada) <= 1e-12, (
        f"Duración esperada {dur_esperada:.15f} s, medida {dur_medida:.15f} s."
    )


def test_caracteristicas_errores(cfg_real: dict) -> None:
    """Verifica que caracteristicas lanza ValueError si clip no es 1D, es corto (< 2*M + n_fft) o tiene NaN/±Inf."""
    from src.voz.caracteristicas import caracteristicas

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    m_muestras = int(round(cfg_real["voz"]["preprocesamiento"]["margen_ms"] * fs / 1000))
    n_fft = int(cfg_real["voz"]["mfcc"]["n_fft"])
    min_muestras = 2 * m_muestras + n_fft

    # Clip 2D
    with pytest.raises(ValueError):
        caracteristicas(np.zeros((1, min_muestras + 100), dtype=np.float32), cfg_real)

    # Clip demasiado corto (exactamente 2*M + n_fft - 1 muestras = 1791 con cfg real)
    with pytest.raises(ValueError):
        caracteristicas(np.zeros(min_muestras - 1, dtype=np.float32), cfg_real)

    # Clip con NaN, +Inf y -Inf
    for valor_no_finito in (np.nan, np.inf, -np.inf):
        clip_malo = np.zeros(min_muestras + 200, dtype=np.float32)
        clip_malo[50] = valor_no_finito
        with pytest.raises(ValueError):
            caracteristicas(clip_malo, cfg_real)


# =============================================================================
# 6.4 Escalador, contribución por bloque y separación en la muestra sintética
# =============================================================================


def test_escalador_medias_desvios_y_centrado_train(muestra_sintetica: dict) -> None:
    """Verifica que ajustar_escalador calcula media y desvío ddof=1 de train y centra Z_train en 0."""
    from src.voz.caracteristicas import BLOQUES, ajustar_escalador

    bloques_train = muestra_sintetica["bloques_train"]
    esc = ajustar_escalador(bloques_train)

    for b in BLOQUES:
        mat = bloques_train[b]
        usadas = esc.columnas[b]
        assert usadas.dtype == bool and usadas.shape == (mat.shape[1],), (
            f"columnas['{b}'] debe ser bool de forma ({mat.shape[1]},), medido {usadas.shape} ({usadas.dtype})."
        )
        medias_esp = np.nanmean(mat[:, usadas], axis=0)
        desvios_esp = np.nanstd(mat[:, usadas], axis=0, ddof=1)
        assert np.allclose(esc.medias[b][usadas], medias_esp, rtol=1e-9, atol=1e-9), (
            f"Medias del bloque '{b}' no coinciden con nanmean de train."
        )
        assert np.allclose(esc.desvios[b][usadas], desvios_esp, rtol=1e-9, atol=1e-9), (
            f"Desvíos del bloque '{b}' no coinciden con nanstd(ddof=1) de train."
        )

    z_train = esc.transformar(bloques_train)
    medias_z = np.mean(z_train, axis=0)
    max_media = float(np.max(np.abs(medias_z)))
    assert np.allclose(medias_z, 0.0, atol=1e-9), (
        f"Cada columna usada de Z_train debe tener media 0 (tol 1e-9), máximo medido: {max_media:.3e}."
    )


def test_escalador_nan_y_descarte_columnas_constantes(muestra_sintetica: dict) -> None:
    """Verifica imputación de NaN a 0.0 y descarte de columnas con <2 datos no NaN o desvío <= 1e-9."""
    from src.voz.caracteristicas import BLOQUES, ajustar_escalador

    bloques_mod = {b: muestra_sintetica["bloques_train"][b].copy() for b in BLOQUES}
    # Columna 0 de mfcc: constante (desvío 0 <= 1e-9) -> se descarta
    bloques_mod["mfcc"][:, 0] = 5.0
    # Columna 1 de mfcc: sólo 1 dato no NaN (< 2) -> se descarta
    bloques_mod["mfcc"][1:, 1] = np.nan
    # Columna 2 de mfcc: todos NaN -> se descarta
    bloques_mod["mfcc"][:, 2] = np.nan
    # Columna 3 de mfcc: válida pero con un NaN en la fila 0 -> se usa y el NaN se imputa a 0.0
    bloques_mod["mfcc"][0, 3] = np.nan

    esc = ajustar_escalador(bloques_mod)
    assert not esc.columnas["mfcc"][0], "La columna constante debía quedar con columnas['mfcc'][0] == False."
    assert not esc.columnas["mfcc"][1], "La columna con 1 solo no-NaN debía quedar con columnas['mfcc'][1] == False."
    assert not esc.columnas["mfcc"][2], "La columna toda NaN debía quedar con columnas['mfcc'][2] == False."
    assert esc.columnas["mfcc"][3], "La columna 3 con suficientes datos válidos debía quedar en True."

    media_col3 = float(np.nanmean(bloques_mod["mfcc"][:, 3]))
    desvio_col3 = float(np.nanstd(bloques_mod["mfcc"][:, 3], ddof=1))
    assert abs(float(esc.medias["mfcc"][3]) - media_col3) <= 1e-9, (
        f"Media de columna con NaN esperada {media_col3}, medida {esc.medias['mfcc'][3]}."
    )
    assert abs(float(esc.desvios["mfcc"][3]) - desvio_col3) <= 1e-9, (
        f"Desvío de columna con NaN esperado {desvio_col3}, medido {esc.desvios['mfcc'][3]}."
    )

    z_mod = esc.transformar(bloques_mod)
    d_esperado = sum(int(esc.columnas[b].sum()) for b in BLOQUES)
    assert z_mod.shape == (bloques_mod["mfcc"].shape[0], d_esperado), (
        f"Forma de Z esperada {(bloques_mod['mfcc'].shape[0], d_esperado)}, medida {z_mod.shape}."
    )
    # Como las columnas 0, 1 y 2 de mfcc fueron descartadas, la columna 3 de mfcc es la columna 0 de Z
    assert z_mod[0, 0] == 0.0, (
        f"La entrada NaN en una columna usada debe imputarse a 0.0 exacto en Z, medido {z_mod[0, 0]}."
    )
    assert np.all(np.isfinite(z_mod)), "El resultado de Escalador.transformar debe ser siempre finito."


def test_escalador_pesos_y_contribucion_por_bloque(muestra_sintetica: dict) -> None:
    """Verifica que pesos[b] == 1/sqrt(columnas_usadas) y que cada bloque aporta 2.0 en train."""
    from src.voz.caracteristicas import BLOQUES, ajustar_escalador

    bloques_train = muestra_sintetica["bloques_train"]
    esc = ajustar_escalador(bloques_train)
    z_train = esc.transformar(bloques_train)

    offset = 0
    for b in BLOQUES:
        k_b = int(esc.columnas[b].sum())
        peso_esperado = 1.0 / np.sqrt(k_b)
        assert abs(esc.pesos[b] - peso_esperado) <= 1e-12, (
            f"Peso de '{b}' esperado {peso_esperado:.12f}, medido {esc.pesos[b]:.12f}."
        )

        z_b = z_train[:, offset : offset + k_b]
        offset += k_b

        contrib_pares = _contribucion_bloque_pares(z_b)
        contrib_var = float(2.0 * np.sum(np.var(z_b, axis=0, ddof=1)))
        assert abs(contrib_pares - 2.0) <= 1e-9, (
            f"Contribución por pares del bloque '{b}' en train esperada 2.0, medida {contrib_pares:.12f}."
        )
        assert abs(contrib_var - 2.0) <= 1e-9, (
            f"Contribución 2·Σvar del bloque '{b}' en train esperada 2.0, medida {contrib_var:.12f}."
        )


def test_escalador_transformar_individual_vs_lote_e_independencia(
    muestra_sintetica: dict,
) -> None:
    """Verifica que transformar de un clip 1D coincide con su fila en lote 2D y no depende de otras filas."""
    from src.voz.caracteristicas import BLOQUES, ajustar_escalador

    esc = ajustar_escalador(muestra_sintetica["bloques_train"])
    bloques_val = muestra_sintetica["bloques_val"]
    z_lote = esc.transformar(bloques_val)

    for idx in range(z_lote.shape[0]):
        individual = {b: bloques_val[b][idx] for b in BLOQUES}
        z_ind = esc.transformar(individual)
        assert z_ind.ndim == 1 and z_ind.dtype == np.float64, (
            f"transformar individual debe devolver 1D float64, medido shape={z_ind.shape}, dtype={z_ind.dtype}."
        )
        assert np.allclose(z_ind, z_lote[idx], rtol=1e-12, atol=1e-12), (
            f"Fila {idx}: transformar individual difiere del lote en "
            f"{np.max(np.abs(z_ind - z_lote[idx])):.3e} > 1e-12."
        )

    bloques_val_alterados = {b: bloques_val[b].copy() for b in BLOQUES}
    for b in BLOQUES:
        bloques_val_alterados[b][1:] = bloques_val_alterados[b][1:] * 7.5 + 3.0
    z_lote_alterado = esc.transformar(bloques_val_alterados)
    assert np.array_equal(z_lote[0], z_lote_alterado[0]), (
        "La transformación de la fila 0 no debe depender de las demás filas del lote."
    )


def test_muestra_sintetica_silhouette_train_y_val(muestra_sintetica: dict) -> None:
    """Verifica que en la muestra sintética de 3 clases la silhouette sobre Z es >= 0.25 en train y val."""
    from src.core_ml.metricas import silhouette
    from src.voz.caracteristicas import ajustar_escalador

    esc = ajustar_escalador(muestra_sintetica["bloques_train"])
    z_train = esc.transformar(muestra_sintetica["bloques_train"])
    z_val = esc.transformar(muestra_sintetica["bloques_val"])

    sil_train = silhouette(z_train, muestra_sintetica["clases_train"])
    sil_val = silhouette(z_val, muestra_sintetica["clases_val"])

    assert sil_train >= 0.25, (
        f"Silhouette en train de la muestra sintética esperada >= 0.25, medida {sil_train:.4f}."
    )
    assert sil_val >= 0.25, (
        f"Silhouette en val de la muestra sintética esperada >= 0.25, medida {sil_val:.4f}."
    )


def test_escalador_errores(muestra_sintetica: dict) -> None:
    """Verifica que ajustar_escalador y transformar lanzan ValueError ante claves, formas o valores inválidos."""
    from src.voz.caracteristicas import BLOQUES, ajustar_escalador

    base = muestra_sintetica["bloques_train"]
    esc = ajustar_escalador(base)

    # Clave faltante y clave sobrante
    sin_zcr = {b: base[b] for b in BLOQUES if b != "zcr"}
    con_extra = {**base, "extra": np.ones((base["mfcc"].shape[0], 2))}
    for d_invalido in (sin_zcr, con_extra):
        with pytest.raises(ValueError):
            ajustar_escalador(d_invalido)
        with pytest.raises(ValueError):
            esc.transformar(d_invalido)

    # Bloque 1D en ajustar_escalador
    mal_1d = {**base, "duracion": base["duracion"][:, 0]}
    with pytest.raises(ValueError):
        ajustar_escalador(mal_1d)

    # Cantidad de filas distinta entre bloques
    mal_filas = {**base, "energia": base["energia"][:-1]}
    with pytest.raises(ValueError):
        ajustar_escalador(mal_filas)
    with pytest.raises(ValueError):
        esc.transformar(mal_filas)

    # Menos de 2 filas en ajustar_escalador
    una_fila = {b: base[b][:1] for b in BLOQUES}
    with pytest.raises(ValueError):
        ajustar_escalador(una_fila)

    # Un bloque completo con todas sus columnas constantes en train
    bloque_constante = {**base, "duracion": np.ones_like(base["duracion"])}
    with pytest.raises(ValueError):
        ajustar_escalador(bloque_constante)

    # Presencia de ±Inf en ajustar_escalador y en transformar
    con_inf = {b: base[b].copy() for b in BLOQUES}
    con_inf["energia"][0, 0] = np.inf
    with pytest.raises(ValueError):
        ajustar_escalador(con_inf)
    with pytest.raises(ValueError):
        esc.transformar(con_inf)

    # Largo d_b incorrecto en transformar
    mal_cols = {**base, "energia": base["energia"][:, :-1]}
    with pytest.raises(ValueError):
        esc.transformar(mal_cols)


# =============================================================================
# 6.5 Archivo de características y figura (en tmp_path)
# =============================================================================


def test_generar_caracteristicas_crea_npz_unico_y_formato(
    tmp_path: Path, cfg_real: dict
) -> None:
    """Verifica que generar_caracteristicas devuelve carpeta/'caracteristicas.npz', no crea otros archivos y guarda solo válidos."""
    from src.voz.caracteristicas import (
        BLOQUES,
        ajustar_escalador,
        caracteristicas,
        generar_caracteristicas,
    )

    carpeta = tmp_path / "prepro_v2"
    filas_csv = _armar_dataset_v2(carpeta, cfg_real)
    archivos_antes = set(carpeta.iterdir())

    ruta_npz = generar_caracteristicas(cfg_real, carpeta=carpeta)
    assert ruta_npz == carpeta / "caracteristicas.npz", (
        f"Ruta devuelta {ruta_npz} no coincide con {carpeta / 'caracteristicas.npz'}."
    )
    assert ruta_npz.is_file(), "El archivo caracteristicas.npz no fue creado."

    archivos_nuevos = set(carpeta.iterdir()) - archivos_antes
    assert archivos_nuevos == {ruta_npz}, (
        f"generar_caracteristicas no debe crear otros archivos; nuevos encontrados: {archivos_nuevos}."
    )

    claves_esperadas = {
        "archivo",
        "particion",
        "origen",
        "hablante",
        "clase",
        "copia",
        "mfcc",
        "delta",
        "energia",
        "zcr",
        "duracion",
        "X",
    }
    filas_validas = [f for f in filas_csv if f["valido"] == 1]

    with np.load(ruta_npz, allow_pickle=False) as datos:
        assert set(datos.files) == claves_esperadas, (
            f"Claves del .npz {set(datos.files)} no coinciden con {claves_esperadas}."
        )
        archivos_esperados = [f["archivo"] for f in filas_validas]
        assert list(datos["archivo"].tolist()) == archivos_esperados, (
            "Las filas del .npz deben ser únicamente las de valido==1 y en el mismo orden de metadatos.csv."
        )
        assert datos["copia"].dtype == np.int64, (
            f"El array 'copia' debe ser int64, medido {datos['copia'].dtype}."
        )

        # Verificar coincidencia de cada fila de los 5 bloques crudos y de X
        bloques_calculados: list[dict[str, np.ndarray]] = []
        for idx, fila in enumerate(filas_validas):
            wav, sr = sf.read(carpeta / fila["archivo"], dtype="float32")
            assert sr == int(cfg_real["voz"]["frecuencia_muestreo"])
            feat = caracteristicas(wav, cfg_real)
            bloques_calculados.append(feat)
            for b in BLOQUES:
                assert datos[b].dtype == np.float64, f"El bloque '{b}' debe ser float64."
                assert np.allclose(datos[b][idx], feat[b], rtol=1e-9, atol=1e-9), (
                    f"Fila {idx} del bloque '{b}' no coincide con caracteristicas(wav, cfg)."
                )

        mask_train = datos["particion"] == "train"
        bloques_train = {
            b: np.vstack([bloques_calculados[i][b] for i in range(len(filas_validas)) if mask_train[i]])
            for b in BLOQUES
        }
        bloques_todos = {
            b: np.vstack([bloques_calculados[i][b] for i in range(len(filas_validas))])
            for b in BLOQUES
        }
        esc = ajustar_escalador(bloques_train)
        x_esperado = esc.transformar(bloques_todos)
        assert datos["X"].dtype == np.float64, f"X debe ser float64, medido {datos['X'].dtype}."
        assert np.allclose(datos["X"], x_esperado, rtol=1e-12, atol=1e-12), (
            f"X en caracteristicas.npz difiere del escalador ajustado en train: "
            f"dif máx {np.max(np.abs(datos['X'] - x_esperado)):.3e}."
        )


def test_generar_caracteristicas_sin_fuga_entre_particiones(
    tmp_path: Path, cfg_real: dict
) -> None:
    """Verifica que cambiar los WAV de val y test no modifica las filas de train en X."""
    from src.voz.caracteristicas import generar_caracteristicas

    carpeta_a = tmp_path / "prepro_a"
    carpeta_b = tmp_path / "prepro_b"
    _armar_dataset_v2(carpeta_a, cfg_real, semilla_val_test=500)
    _armar_dataset_v2(carpeta_b, cfg_real, semilla_val_test=900)

    npz_a = generar_caracteristicas(cfg_real, carpeta=carpeta_a)
    npz_b = generar_caracteristicas(cfg_real, carpeta=carpeta_b)

    with np.load(npz_a, allow_pickle=False) as da, np.load(npz_b, allow_pickle=False) as db:
        mask_train_a = da["particion"] == "train"
        mask_train_b = db["particion"] == "train"
        assert np.array_equal(da["X"][mask_train_a], db["X"][mask_train_b]), (
            "Fuga entre particiones detectada: las filas de train en X cambiaron al alterar val y test."
        )


def test_generar_caracteristicas_sobrescribir_y_errores(
    tmp_path: Path, cfg_real: dict
) -> None:
    """Verifica FileExistsError, sobrescribir=True, FileNotFoundError y ValueError sin dejar .npz escrito."""
    from src.voz.caracteristicas import generar_caracteristicas

    carpeta = tmp_path / "prepro_ok"
    _armar_dataset_v2(carpeta, cfg_real)
    ruta_npz = generar_caracteristicas(cfg_real, carpeta=carpeta)
    bytes_originales = ruta_npz.read_bytes()

    # Si ya existe y sobrescribir=False -> FileExistsError sin modificar el archivo
    ruta_npz.write_bytes(b"centinela_no_tocar")
    with pytest.raises(FileExistsError):
        generar_caracteristicas(cfg_real, carpeta=carpeta, sobrescribir=False)
    assert ruta_npz.read_bytes() == b"centinela_no_tocar", (
        "Con sobrescribir=False no se debe modificar el archivo existente."
    )

    # Con sobrescribir=True lo sobrescribe correctamente
    generar_caracteristicas(cfg_real, carpeta=carpeta, sobrescribir=True)
    assert ruta_npz.read_bytes() == bytes_originales, (
        "Con sobrescribir=True debía regenerar el archivo caracteristicas.npz."
    )

    # Si falta metadatos.csv -> FileNotFoundError
    carpeta_sin_csv = tmp_path / "sin_csv"
    carpeta_sin_csv.mkdir()
    with pytest.raises(FileNotFoundError):
        generar_caracteristicas(cfg_real, carpeta=carpeta_sin_csv)

    # Si falta el WAV de una fila válida -> FileNotFoundError
    carpeta_sin_wav = tmp_path / "sin_wav"
    filas = _armar_dataset_v2(carpeta_sin_wav, cfg_real)
    primer_wav_valido = next(f["archivo"] for f in filas if f["valido"] == 1)
    (carpeta_sin_wav / primer_wav_valido).unlink()
    with pytest.raises(FileNotFoundError):
        generar_caracteristicas(cfg_real, carpeta=carpeta_sin_wav)
    assert not (carpeta_sin_wav / "caracteristicas.npz").exists(), (
        "No debe quedar caracteristicas.npz escrito cuando falta un WAV válido."
    )

    # Si un WAV tiene otra frecuencia de muestreo (ej. 8000 Hz) -> ValueError sin dejar .npz
    carpeta_sr_mal = tmp_path / "sr_mal"
    filas_sr = _armar_dataset_v2(carpeta_sr_mal, cfg_real)
    wav_valido = next(f["archivo"] for f in filas_sr if f["valido"] == 1)
    sf.write(carpeta_sr_mal / wav_valido, np.zeros(8000, dtype=np.float32), 8000, subtype="FLOAT")
    with pytest.raises(ValueError):
        generar_caracteristicas(cfg_real, carpeta=carpeta_sr_mal)
    assert not (carpeta_sr_mal / "caracteristicas.npz").exists(), (
        "No debe quedar caracteristicas.npz escrito cuando un WAV tiene frecuencia inválida."
    )

    # Si hay menos de 2 filas válidas de train -> ValueError sin dejar .npz
    carpeta_poco_train = tmp_path / "poco_train"
    _armar_dataset_v2(carpeta_poco_train, cfg_real)
    ruta_csv = carpeta_poco_train / "metadatos.csv"
    with ruta_csv.open("r", encoding="utf-8", newline="") as f_in:
        lector = list(csv.DictReader(f_in))
        campos = list(lector[0].keys())
    vistas_train = 0
    for fila in lector:
        if fila["particion"] == "train" and str(fila["valido"]) in {"1", "True", "true"}:
            vistas_train += 1
            if vistas_train > 1:
                fila["valido"] = "0"
    with ruta_csv.open("w", encoding="utf-8", newline="") as f_out:
        escritor = csv.DictWriter(f_out, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(lector)

    with pytest.raises(ValueError):
        generar_caracteristicas(cfg_real, carpeta=carpeta_poco_train)
    assert not (carpeta_poco_train / "caracteristicas.npz").exists(), (
        "No debe quedar caracteristicas.npz escrito si hay menos de 2 filas válidas de train."
    )

    # Si falta una columna requerida en metadatos.csv -> ValueError sin dejar .npz
    carpeta_col_faltante = tmp_path / "col_faltante"
    _armar_dataset_v2(carpeta_col_faltante, cfg_real)
    ruta_csv_col = carpeta_col_faltante / "metadatos.csv"
    with ruta_csv_col.open("r", encoding="utf-8", newline="") as f_in:
        lector_col = list(csv.DictReader(f_in))
        campos_sin_clase = [c for c in lector_col[0].keys() if c != "clase"]
    with ruta_csv_col.open("w", encoding="utf-8", newline="") as f_out:
        escritor_col = csv.DictWriter(f_out, fieldnames=campos_sin_clase)
        escritor_col.writeheader()
        for fila in lector_col:
            fila.pop("clase", None)
            escritor_col.writerow(fila)

    with pytest.raises(ValueError):
        generar_caracteristicas(cfg_real, carpeta=carpeta_col_faltante)
    assert not (carpeta_col_faltante / "caracteristicas.npz").exists(), (
        "No debe quedar caracteristicas.npz escrito si faltan columnas requeridas en metadatos.csv."
    )


def test_generar_figura_escribe_png_valido(tmp_path: Path, cfg_real: dict) -> None:
    """Verifica que generar_figura escribe un PNG no vacío legible como imagen y devuelve su Path."""
    from src.voz.caracteristicas import generar_caracteristicas, generar_figura

    carpeta = tmp_path / "prepro_fig"
    _armar_dataset_v2(carpeta, cfg_real)
    ruta_npz = generar_caracteristicas(cfg_real, carpeta=carpeta)

    salida_png = tmp_path / "pca_v3.png"
    ruta_devuelta = generar_figura(cfg_real, carpeta, salida_png)

    assert ruta_devuelta == salida_png, (
        f"generar_figura debía devolver {salida_png}, devolvió {ruta_devuelta}."
    )
    assert salida_png.is_file() and salida_png.stat().st_size > 0, (
        "El archivo PNG generado no existe o está vacío."
    )
    cabecera = salida_png.read_bytes()[:8]
    assert cabecera == b"\x89PNG\r\n\x1a\n", (
        f"El archivo generado no tiene cabecera PNG válida: {cabecera!r}."
    )

    # También debe aceptar directamente la ruta al archivo .npz
    salida_png_npz = tmp_path / "pca_v3_desde_npz.png"
    ruta_devuelta_npz = generar_figura(cfg_real, ruta_npz, salida_png_npz)
    assert ruta_devuelta_npz == salida_png_npz and salida_png_npz.is_file() and salida_png_npz.stat().st_size > 0, (
        "generar_figura debía aceptar directamente la ruta al archivo .npz."
    )


# =============================================================================
# 6.6 Aceptación (con los datos reales de data/voz/prepro/)
# =============================================================================


@pytest.mark.aceptacion
def test_aceptacion_v3_datos_reales(tmp_path: Path, cfg_real: dict) -> None:
    """Verifica todos los criterios de aceptación de la sección 6.6 sobre data/voz/prepro/."""
    carpeta_real = RAIZ / "data" / "voz" / "prepro"
    ruta_csv = carpeta_real / "metadatos.csv"
    if not ruta_csv.is_file():
        pytest.skip(f"No existe {ruta_csv}; se saltea el test de aceptación 6.6.")

    from src.core_ml.metricas import silhouette
    from src.core_ml.pca import PCA
    from src.voz.caracteristicas import (
        BLOQUES,
        generar_caracteristicas,
        generar_figura,
        largos_de_bloque,
    )

    salida_npz = tmp_path / "caracteristicas.npz"
    t0 = time.perf_counter()
    ruta_escrita = generar_caracteristicas(cfg_real, carpeta=carpeta_real, salida=salida_npz)
    tiempo_s = time.perf_counter() - t0

    assert ruta_escrita == salida_npz and salida_npz.is_file(), (
        f"generar_caracteristicas no escribió {salida_npz}."
    )
    assert tiempo_s < 60.0, (
        f"generar_caracteristicas tardó {tiempo_s:.2f} s (máximo permitido: 60.0 s)."
    )

    with ruta_csv.open("r", encoding="utf-8", newline="") as f_csv:
        filas_validas = [
            fila
            for fila in csv.DictReader(f_csv)
            if fila["valido"] in {"1", "True", "true"}
        ]

    assert len(filas_validas) == 1908, (
        f"Se esperaban 1908 filas válidas en metadatos.csv, se encontraron {len(filas_validas)}."
    )
    archivos_csv = [f["archivo"] for f in filas_validas]
    duraciones_csv = np.array([float(f["duracion_palabra_s"]) for f in filas_validas], dtype=np.float64)

    with np.load(salida_npz, allow_pickle=False) as datos:
        X = datos["X"]
        particion = datos["particion"]
        clase = datos["clase"]
        copia = datos["copia"]

        assert list(datos["archivo"].tolist()) == archivos_csv, (
            "Las filas del .npz no coinciden exactamente con las filas válidas de metadatos.csv."
        )
        assert X.shape == (1908, 281), f"X.shape esperado (1908, 281), medido {X.shape}."
        assert np.all(np.isfinite(X)), "X contiene NaN o ±Inf."

        for b in BLOQUES:
            assert np.all(np.isfinite(datos[b])), f"El bloque '{b}' contiene NaN o ±Inf."

        n_train = int(np.sum(particion == "train"))
        n_val = int(np.sum(particion == "val"))
        n_test = int(np.sum(particion == "test"))
        assert (n_train, n_val, n_test) == (1722, 87, 99), (
            f"Conteo por partición esperado (1722, 87, 99), medido ({n_train}, {n_val}, {n_test})."
        )

        # Duración coincide con duracion_palabra_s de metadatos.csv (tolerancia 1e-3 s)
        dur_npz = datos["duracion"].reshape(-1)
        diff_dur = float(np.max(np.abs(dur_npz - duraciones_csv)))
        assert diff_dur <= 1e-3, (
            f"El bloque duracion difiere de duracion_palabra_s en {diff_dur:.5f} s (máximo 1e-3 s)."
        )

        # Contribución por bloque en train (2.0 ± 1e-6) y en val ([1.3, 2.7])
        mask_train = particion == "train"
        mask_val = particion == "val"
        largos = largos_de_bloque(cfg_real)
        offset = 0
        for b in BLOQUES:
            ancho = largos[b]
            z_tr_b = X[mask_train, offset : offset + ancho]
            z_va_b = X[mask_val, offset : offset + ancho]
            offset += ancho

            c_train = _contribucion_bloque_pares(z_tr_b)
            c_val = _contribucion_bloque_pares(z_va_b)
            assert abs(c_train - 2.0) <= 1e-6, (
                f"Contribución del bloque '{b}' en train esperada 2.0 ± 1e-6, medida {c_train:.9f}."
            )
            assert 1.3 <= c_val <= 2.7, (
                f"Contribución del bloque '{b}' en val esperada en [1.3, 2.7], medida {c_val:.4f}."
            )

        # PCA propio vs sklearn en k=30 y k=20, y varianza explicada acumulada en k=30 >= 0.70
        x_train = X[mask_train]
        proyecciones_pca: dict[int, np.ndarray] = {}
        for k in (30, 20):
            pca_propio = PCA(n_componentes=k).ajustar(x_train)
            pca_sk = SklearnPCA(n_components=k, svd_solver="full").fit(x_train)

            z_pca = pca_propio.transformar(X)
            z_sk = pca_sk.transform(X)
            signos = np.where(
                np.sum(pca_propio.componentes * pca_sk.components_, axis=1) >= 0.0,
                1.0,
                -1.0,
            )
            z_sk_alineado = z_sk * signos[None, :]
            diff_proy = float(np.max(np.abs(z_pca - z_sk_alineado)))
            assert np.allclose(z_pca, z_sk_alineado, rtol=1e-6, atol=1e-6), (
                f"Proyección de PCA({k}) difiere de sklearn en {diff_proy:.3e} > 1e-6."
            )
            if k == 30:
                var_acum_30 = float(np.sum(pca_propio.proporcion_varianza))
                assert var_acum_30 >= 0.70, (
                    f"Las primeras 30 componentes deben explicar >= 70% de la varianza de train, "
                    f"medido {var_acum_30 * 100:.2f}%."
                )
            proyecciones_pca[k] = z_pca

        # Silhouette de los 3 comandos en train originales (copia == 0) y en val
        es_comando = np.isin(clase, ["reconocer", "identificar", "salir"])
        mask_train_k0_cmd = mask_train & (copia == 0) & es_comando
        mask_val_cmd = mask_val & es_comando

        for nombre_sub, mascara in (("train_k0", mask_train_k0_cmd), ("val", mask_val_cmd)):
            etiquetas_sub = clase[mascara]
            sil_281 = silhouette(X[mascara], etiquetas_sub)
            sil_30 = silhouette(proyecciones_pca[30][mascara], etiquetas_sub)
            sil_20 = silhouette(proyecciones_pca[20][mascara], etiquetas_sub)

            assert sil_281 > 0.0 and sil_30 > 0.0 and sil_20 > 0.0, (
                f"Las silhouettes en {nombre_sub} deben ser positivas: "
                f"281d={sil_281:.4f}, PCA30={sil_30:.4f}, PCA20={sil_20:.4f}."
            )
            assert sil_20 > sil_281, (
                f"En {nombre_sub}, silhouette con PCA(20) ({sil_20:.4f}) debe superar a 281d ({sil_281:.4f})."
            )

    salida_png = tmp_path / "pca_v3.png"
    ruta_png = generar_figura(cfg_real, carpeta=tmp_path, salida=salida_png)
    assert ruta_png == salida_png and salida_png.is_file() and salida_png.stat().st_size > 0, (
        f"generar_figura no escribió un PNG válido en {salida_png}."
    )
