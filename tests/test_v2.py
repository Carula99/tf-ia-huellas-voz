"""Tests de la etapa V2: preprocesamiento y aumento de voz.

Verifica todos los criterios de la sección 7 del contrato v2_preprocesamiento.md
(Versión 1 · 23/10/2026).
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import inspect
import math
import shutil
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
import scipy.signal
import soundfile as sf
import yaml

RAIZ = Path(__file__).resolve().parents[1]


# =============================================================================
# Funciones auxiliares para síntesis de señales de prueba (16 kHz por defecto)
# =============================================================================


def _llamar_detectar_palabras(
    y: np.ndarray, fs: int, cfg: dict
) -> list[tuple[float, float]]:
    """Invoca detectar_palabras de V1 con los parámetros de voz.dataset.deteccion."""
    from src.voz.segmentacion import detectar_palabras

    det_cfg = cfg["voz"]["dataset"]["deteccion"]
    sig = inspect.signature(detectar_palabras)
    params = list(sig.parameters.keys())
    if len(params) == 3 and params[2] in ("cfg", "config", "deteccion"):
        try:
            segs = detectar_palabras(y, fs, cfg)
        except KeyError:
            segs = detectar_palabras(y, fs, det_cfg)
    else:
        segs = detectar_palabras(y, fs, **det_cfg)

    resultado: list[tuple[float, float]] = []
    for s in segs:
        if hasattr(s, "inicio_s") and hasattr(s, "fin_s"):
            resultado.append((float(s.inicio_s), float(s.fin_s)))
        else:
            resultado.append((float(s[0]), float(s[1])))
    return resultado


def _tono_armonico(
    duracion_s: float,
    fs: int = 16000,
    f0: float = 180.0,
    pico: float = 0.5,
    rampa_ms: float = 20.0,
) -> np.ndarray:
    """Genera un tono con tres armónicos y rampas suaves de ataque y caída."""
    n = round(duracion_s * fs)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / fs
    senal = (
        np.sin(2.0 * np.pi * f0 * t)
        + 0.55 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 3.0 * f0 * t)
    )
    n_rampa = round(rampa_ms * fs / 1000.0)
    if n_rampa > 0 and 2 * n_rampa <= n:
        ventana_rampa = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_rampa) / n_rampa))
        senal[:n_rampa] *= ventana_rampa
        senal[-n_rampa:] *= ventana_rampa[::-1]
    max_abs = float(np.max(np.abs(senal)))
    if max_abs > 0.0:
        senal = senal * (pico / max_abs)
    return senal.astype(np.float64)


def _audio_con_palabra(
    duracion_palabra_s: float = 0.8,
    inicio_s: float = 0.8,
    duracion_total_s: float = 3.0,
    fs: int = 16000,
    f0: float = 180.0,
    pico: float = 0.5,
    rampa_ms: float = 20.0,
) -> np.ndarray:
    """Inserta una palabra sintética dentro de un audio de silencio de duracion_total_s."""
    n_total = round(duracion_total_s * fs)
    audio = np.zeros(n_total, dtype=np.float64)
    palabra = _tono_armonico(
        duracion_palabra_s, fs=fs, f0=f0, pico=pico, rampa_ms=rampa_ms
    )
    m0 = round(inicio_s * fs)
    m1 = min(n_total, m0 + len(palabra))
    audio[m0:m1] = palabra[: m1 - m0]
    return audio


def _audio_palabra_28s_con_cierres(
    fs: int = 16000, duracion_total_s: float = 3.0, pico: float = 0.5
) -> np.ndarray:
    """Genera una palabra de 2,8 s (de 0,1 s a 2,9 s) con cierres internos de 0,1 s."""
    n_total = round(duracion_total_s * fs)
    audio = np.zeros(n_total, dtype=np.float64)
    # 8 bloques de 0,2625 s separados por 7 cierres de 0,1 s:
    # 8 * 0,2625 + 7 * 0,1 = 2,1 + 0,7 = 2,8 s exactos.
    n_bloques = 8
    cierre_s = 0.1
    dur_bloque_s = (2.8 - (n_bloques - 1) * cierre_s) / n_bloques
    cursor_s = 0.1
    for i in range(n_bloques):
        bloque = _tono_armonico(
            dur_bloque_s, fs=fs, f0=170.0 + 10.0 * (i % 3), pico=pico, rampa_ms=10.0
        )
        m0 = round(cursor_s * fs)
        m1 = min(n_total, m0 + len(bloque))
        audio[m0:m1] = bloque[: m1 - m0]
        cursor_s += dur_bloque_s + cierre_s
    return audio


def _ruido_estacionario(
    duracion_s: float = 3.5,
    fs: int = 16000,
    amplitud: float = 0.03,
    semilla: int = 101,
) -> np.ndarray:
    """Genera ruido estacionario filtrado en banda de voz con envolvente uniforme."""
    n = round(duracion_s * fs)
    rng = np.random.default_rng(semilla)
    t = np.arange(n, dtype=np.float64) / fs
    # Suma densa de tonos con fases aleatorias: estacionario y sin picos impulsivos
    freqs = np.linspace(120.0, 3200.0, 160)
    fases = rng.uniform(0.0, 2.0 * np.pi, size=len(freqs))
    ruido = np.zeros(n, dtype=np.float64)
    for f, phi in zip(freqs, fases):
        ruido += np.sin(2.0 * np.pi * f * t + phi)
    max_abs = float(np.max(np.abs(ruido)))
    if max_abs > 0.0:
        ruido = ruido * (amplitud / max_abs)
    return ruido.astype(np.float64)


# =============================================================================
# Fixtures de módulo
# =============================================================================


@pytest.fixture(scope="module")
def cfg_real() -> dict:
    """Carga la configuración real del repositorio."""
    from src.config import cargar_config

    return cargar_config()


@pytest.fixture(scope="module")
def dataset_v1_sintetico(tmp_path_factory: pytest.TempPathFactory, cfg_real: dict) -> dict:
    """Arma un dataset de V1 de prueba en tmp_path y ejecuta generar_prepro."""
    from src.voz.preprocesamiento import generar_prepro

    base_dir = tmp_path_factory.mktemp("v2_dataset_sintetico")
    origen = base_dir / "dataset_v1"
    destino = base_dir / "prepro_v2"
    for part in ("train", "val", "test"):
        (origen / part).mkdir(parents=True, exist_ok=True)

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    cfg = copy.deepcopy(cfg_real)
    cfg["voz"]["preprocesamiento"]["aumento"]["copias"] = 2

    # Palabra modulada de 1,0 s con núcleo bien energético para que incluso a 10 dB de SNR
    # supere cómodamente el umbral del detector en todas las copias aumentadas.
    def _palabra_robusta(f0: float, duracion_s: float = 1.0, inicio_s: float = 0.9) -> np.ndarray:
        audio = _audio_con_palabra(
            duracion_palabra_s=duracion_s,
            inicio_s=inicio_s,
            duracion_total_s=3.0,
            fs=fs,
            f0=f0,
            pico=0.65,
            rampa_ms=15.0,
        )
        m0 = round(inicio_s * fs)
        m1 = round((inicio_s + duracion_s) * fs)
        env = 0.55 + 0.45 * np.sin(np.pi * np.linspace(0.0, 1.0, m1 - m0)) ** 2
        audio[m0:m1] *= env
        return audio

    filas_v1 = [
        {
            "archivo": "train/nb_p01_identificar_01.wav",
            "particion": "train",
            "origen": "notebook",
            "codigo": "p01",
            "hablante": "nb_p01",
            "clase": "identificar",
            "detalle": "identificar",
            "audio": _palabra_robusta(180.0, duracion_s=1.0, inicio_s=0.8),
        },
        {
            "archivo": "train/nb_p01_verificar_01.wav",
            "particion": "train",
            "origen": "notebook",
            "codigo": "p01",
            "hablante": "nb_p01",
            "clase": "verificar",
            "detalle": "verificar",
            "audio": _palabra_robusta(210.0, duracion_s=1.1, inicio_s=0.7),
        },
        {
            "archivo": "train/nb_p01_otro_01.wav",
            "particion": "train",
            "origen": "notebook",
            "codigo": "p01",
            "hablante": "nb_p01",
            "clase": "otro",
            "detalle": "hola",
            "audio": _palabra_robusta(240.0, duracion_s=0.95, inicio_s=0.9),
        },
        {
            "archivo": "train/nb_p01_otro_silencio.wav",
            "particion": "train",
            "origen": "notebook",
            "codigo": "p01",
            "hablante": "nb_p01",
            "clase": "otro",
            "detalle": "silencio",
            "audio": np.zeros(3 * fs, dtype=np.float64),
        },
        {
            "archivo": "train/nb_p01_otro_largo.wav",
            "particion": "train",
            "origen": "notebook",
            "codigo": "p01",
            "hablante": "nb_p01",
            "clase": "otro",
            "detalle": "largo",
            "audio": _audio_palabra_28s_con_cierres(fs=fs, duracion_total_s=3.0, pico=0.5),
        },
        # Ruidos de train desordenados a propósito en el CSV para verificar que V2
        # los ordena por "archivo"
        {
            "archivo": "train/ruido_02.wav",
            "particion": "train",
            "origen": "ruido",
            "codigo": "ruido",
            "hablante": "ninguno",
            "clase": "otro",
            "detalle": "pasillo",
            "audio": _ruido_estacionario(duracion_s=3.2, fs=fs, amplitud=0.03, semilla=202),
        },
        {
            "archivo": "train/ruido_01.wav",
            "particion": "train",
            "origen": "ruido",
            "codigo": "ruido",
            "hablante": "ninguno",
            "clase": "otro",
            "detalle": "aula",
            "audio": _ruido_estacionario(duracion_s=3.2, fs=fs, amplitud=0.025, semilla=101),
        },
        {
            "archivo": "train/ruido_03.wav",
            "particion": "train",
            "origen": "ruido",
            "codigo": "ruido",
            "hablante": "ninguno",
            "clase": "otro",
            "detalle": "ventilador",
            "audio": _ruido_estacionario(duracion_s=3.2, fs=fs, amplitud=0.035, semilla=303),
        },
        {
            "archivo": "val/nb_p02_identificar_01.wav",
            "particion": "val",
            "origen": "notebook",
            "codigo": "p02",
            "hablante": "nb_p02",
            "clase": "identificar",
            "detalle": "identificar",
            "audio": _palabra_robusta(195.0, duracion_s=0.9, inicio_s=0.85),
        },
        {
            "archivo": "val/nb_p02_otro_01.wav",
            "particion": "val",
            "origen": "notebook",
            "codigo": "p02",
            "hablante": "nb_p02",
            "clase": "otro",
            "detalle": "buenas",
            "audio": _palabra_robusta(230.0, duracion_s=0.85, inicio_s=0.9),
        },
        {
            "archivo": "val/ruido_val_01.wav",
            "particion": "val",
            "origen": "ruido",
            "codigo": "ruido",
            "hablante": "ninguno",
            "clase": "otro",
            "detalle": "aula_val",
            "audio": _ruido_estacionario(duracion_s=3.2, fs=fs, amplitud=0.04, semilla=404),
        },
        {
            "archivo": "test/nb_p03_salir_01.wav",
            "particion": "test",
            "origen": "notebook",
            "codigo": "p03",
            "hablante": "nb_p03",
            "clase": "salir",
            "detalle": "salir",
            "audio": _palabra_robusta(200.0, duracion_s=0.85, inicio_s=0.9),
        },
        {
            "archivo": "test/nb_p03_otro_01.wav",
            "particion": "test",
            "origen": "notebook",
            "codigo": "p03",
            "hablante": "nb_p03",
            "clase": "otro",
            "detalle": "chau",
            "audio": _palabra_robusta(250.0, duracion_s=0.8, inicio_s=1.0),
        },
    ]

    columnas_v1 = [
        "archivo",
        "particion",
        "origen",
        "codigo",
        "hablante",
        "clase",
        "detalle",
    ]
    csv_v1 = origen / "metadatos.csv"
    with csv_v1.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=columnas_v1, lineterminator="\n")
        writer.writeheader()
        for item in filas_v1:
            sf.write(origen / item["archivo"], item["audio"], fs, subtype="PCM_16")
            writer.writerow({k: item[k] for k in columnas_v1})

    filas_retorno = generar_prepro(cfg, origen=origen, destino=destino, sobrescribir=False)
    return {
        "cfg": cfg,
        "fs": fs,
        "origen": origen,
        "destino": destino,
        "filas_v1": filas_v1,
        "filas_retorno": filas_retorno,
    }


# =============================================================================
# 7.1 Configuración
# =============================================================================


def test_config_claves_tipos_rangos_y_valores_reales(cfg_real: dict) -> None:
    """Verifica existencia, tipos, rangos y valores reales de voz.preprocesamiento."""
    assert "voz" in cfg_real and isinstance(cfg_real["voz"], dict), (
        f"Se esperaba el bloque 'voz' como dict en config.yaml; se obtuvo {type(cfg_real.get('voz'))}"
    )
    voz = cfg_real["voz"]
    assert "preprocesamiento" in voz and isinstance(voz["preprocesamiento"], dict), (
        "Falta la clave 'voz.preprocesamiento' o no es un dict"
    )
    pre = voz["preprocesamiento"]

    # margen_ms: float en [20, 100], valor real 40
    margen_ms = pre.get("margen_ms")
    assert isinstance(margen_ms, (int, float)) and not isinstance(margen_ms, bool), (
        f"voz.preprocesamiento.margen_ms debe ser numérico; se obtuvo {type(margen_ms)}"
    )
    assert 20.0 <= float(margen_ms) <= 100.0, (
        f"voz.preprocesamiento.margen_ms fuera de [20, 100]: {margen_ms}"
    )
    assert float(margen_ms) == pytest.approx(40.0), (
        f"En config.yaml real margen_ms debe ser 40; se midió {margen_ms}"
    )

    # orden_filtro: entero fijo en 4
    orden = pre.get("orden_filtro")
    assert isinstance(orden, int) and not isinstance(orden, bool), (
        f"voz.preprocesamiento.orden_filtro debe ser int; se obtuvo {type(orden)}"
    )
    assert orden == 4, (
        f"voz.preprocesamiento.orden_filtro debe valer 4; se midió {orden}"
    )

    # nivel_rms_db: float en [-35, -15], valor real -26.0
    nivel_rms_db = pre.get("nivel_rms_db")
    assert isinstance(nivel_rms_db, (int, float)) and not isinstance(nivel_rms_db, bool), (
        f"voz.preprocesamiento.nivel_rms_db debe ser numérico; se obtuvo {type(nivel_rms_db)}"
    )
    assert -35.0 <= float(nivel_rms_db) <= -15.0, (
        f"voz.preprocesamiento.nivel_rms_db fuera de [-35, -15]: {nivel_rms_db}"
    )
    assert float(nivel_rms_db) == pytest.approx(-26.0), (
        f"En config.yaml real nivel_rms_db debe ser -26.0; se midió {nivel_rms_db}"
    )

    # Bloque aumento
    assert "aumento" in pre and isinstance(pre["aumento"], dict), (
        "Falta la clave 'voz.preprocesamiento.aumento' o no es un dict"
    )
    aum = pre["aumento"]

    # copias: entero en [0, 5], valor real 2
    copias = aum.get("copias")
    assert isinstance(copias, int) and not isinstance(copias, bool), (
        f"voz.preprocesamiento.aumento.copias debe ser int; se obtuvo {type(copias)}"
    )
    assert 0 <= copias <= 5, (
        f"voz.preprocesamiento.aumento.copias fuera de [0, 5]: {copias}"
    )
    assert copias == 2, (
        f"En config.yaml real copias debe ser 2; se midió {copias}"
    )

    # velocidad: [min, max] con 0.8 <= min <= 1 <= max <= 1.2, valor real [0.92, 1.08]
    velocidad = aum.get("velocidad")
    assert isinstance(velocidad, list) and len(velocidad) == 2, (
        f"voz.preprocesamiento.aumento.velocidad debe ser lista de 2 elementos; se obtuvo {velocidad}"
    )
    v_min, v_max = float(velocidad[0]), float(velocidad[1])
    assert 0.8 <= v_min <= 1.0 <= v_max <= 1.2, (
        f"velocidad fuera de 0.8 <= min <= 1 <= max <= 1.2: {velocidad}"
    )
    assert v_min == pytest.approx(0.92) and v_max == pytest.approx(1.08), (
        f"En config.yaml real velocidad debe ser [0.92, 1.08]; se midió {velocidad}"
    )

    # snr_db: [min, max] con 0 <= min <= max <= 40, valor real [10.0, 25.0]
    snr_db = aum.get("snr_db")
    assert isinstance(snr_db, list) and len(snr_db) == 2, (
        f"voz.preprocesamiento.aumento.snr_db debe ser lista de 2 elementos; se obtuvo {snr_db}"
    )
    s_min, s_max = float(snr_db[0]), float(snr_db[1])
    assert 0.0 <= s_min <= s_max <= 40.0, (
        f"snr_db fuera de 0 <= min <= max <= 40: {snr_db}"
    )
    assert s_min == pytest.approx(10.0) and s_max == pytest.approx(25.0), (
        f"En config.yaml real snr_db debe ser [10.0, 25.0]; se midió {snr_db}"
    )


@pytest.mark.parametrize(
    "camino",
    [
        "voz.preprocesamiento",
        "voz.preprocesamiento.margen_ms",
        "voz.preprocesamiento.orden_filtro",
        "voz.preprocesamiento.nivel_rms_db",
        "voz.preprocesamiento.aumento",
        "voz.preprocesamiento.aumento.copias",
        "voz.preprocesamiento.aumento.velocidad",
        "voz.preprocesamiento.aumento.snr_db",
    ],
)
def test_config_clave_faltante_lanza_value_error_con_camino(
    cfg_real: dict, tmp_path: Path, camino: str
) -> None:
    """Verifica que cargar_config lance ValueError nombrando el camino completo si falta alguna clave."""
    from src.config import cargar_config

    datos = copy.deepcopy(cfg_real)
    partes = camino.split(".")
    nodo = datos
    for p in partes[:-1]:
        nodo = nodo[p]
    del nodo[partes[-1]]

    archivo_cfg = tmp_path / "config_incompleto.yaml"
    archivo_cfg.write_text(yaml.safe_dump(datos, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        cargar_config(archivo_cfg)

    mensaje = str(exc_info.value)
    assert camino in mensaje, (
        f"El ValueError al faltar '{camino}' debe incluir el camino completo; se obtuvo: '{mensaje}'"
    )


# =============================================================================
# 7.2 Palabra principal y validez (palabra_principal, preprocesar)
# =============================================================================


def test_palabra_unica_08s_valida_igual_a_detectar_palabras(cfg_real: dict) -> None:
    """Una palabra de 0,8 s dentro de 3 s a 16 kHz es válida y coincide con detectar_palabras."""
    from src.voz.preprocesamiento import Preprocesado, palabra_principal, preprocesar

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    audio = _audio_con_palabra(duracion_palabra_s=0.8, inicio_s=0.9, duracion_total_s=3.0, fs=fs)

    segs_v1 = _llamar_detectar_palabras(audio, fs, cfg_real)
    assert len(segs_v1) == 1, (
        f"El audio sintético de prueba debía producir 1 segmento en V1; dio {len(segs_v1)}"
    )
    esp_inicio, esp_fin = segs_v1[0]

    pal, n_seg = palabra_principal(audio, fs, cfg_real)
    assert n_seg == 1, f"palabra_principal debía informar n_segmentos=1; devolvió {n_seg}"
    assert pal is not None, "palabra_principal devolvió None para una palabra válida de 0,8 s"
    assert pal[0] == pytest.approx(esp_inicio, abs=1e-6) and pal[1] == pytest.approx(esp_fin, abs=1e-6), (
        f"palabra_principal devolvió {pal}, pero detectar_palabras dio {(esp_inicio, esp_fin)}"
    )

    res = preprocesar(audio, fs, cfg_real)
    assert isinstance(res, Preprocesado) and dataclasses.is_dataclass(res), (
        f"preprocesar debe devolver una instancia de Preprocesado; devolvió {type(res)}"
    )
    assert res.valido is True, f"Se esperaba valido=True; se obtuvo valido={res.valido} (motivo='{res.motivo}')"
    assert res.motivo == "", f"Para un clip válido motivo debe ser ''; se obtuvo '{res.motivo}'"
    assert res.n_segmentos == 1, f"Se esperaba n_segmentos=1; se obtuvo {res.n_segmentos}"
    assert res.inicio_s == pytest.approx(esp_inicio, abs=1e-6), (
        f"inicio_s medido ({res.inicio_s}) != esperado de detectar_palabras ({esp_inicio})"
    )
    assert res.fin_s == pytest.approx(esp_fin, abs=1e-6), (
        f"fin_s medido ({res.fin_s}) != esperado de detectar_palabras ({esp_fin})"
    )


def test_dos_palabras_gana_mas_fuerte_y_en_empate_gana_primera(cfg_real: dict) -> None:
    """Con dos palabras separadas por 0,6 s gana la más fuerte (+3 dB), y si empatan gana la primera."""
    from src.voz.preprocesamiento import palabra_principal, preprocesar

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    avance_ms = float(cfg_real["voz"]["dataset"]["deteccion"]["avance_ms"])
    h_muestras = round(avance_ms * fs / 1000.0)

    dur_palabra_s = 0.6
    onda_base = _tono_armonico(dur_palabra_s, fs=fs, f0=190.0, pico=0.35, rampa_ms=20.0)
    n_pal = len(onda_base)

    # Ubicamos la primera palabra en un múltiplo exacto del avance H y la segunda corrida
    # otro múltiplo exacto de H tal que la separación entre el fin de la 1ª y el inicio de la 2ª sea ~0,6 s.
    m_ini_1 = 25 * h_muestras
    salto_tramas = round((n_pal + round(0.6 * fs)) / h_muestras)
    m_ini_2 = m_ini_1 + salto_tramas * h_muestras

    # Caso 1: la segunda palabra es 3 dB más fuerte
    audio_segunda_fuerte = np.zeros(3 * fs, dtype=np.float64)
    factor_3db = 10.0 ** (3.0 / 20.0)
    audio_segunda_fuerte[m_ini_1 : m_ini_1 + n_pal] = onda_base
    audio_segunda_fuerte[m_ini_2 : m_ini_2 + n_pal] = onda_base * factor_3db

    segs_1 = _llamar_detectar_palabras(audio_segunda_fuerte, fs, cfg_real)
    assert len(segs_1) == 2, f"Se esperaban 2 segmentos en el caso +3 dB; se detectaron {len(segs_1)}"

    pal_1, n_seg_1 = palabra_principal(audio_segunda_fuerte, fs, cfg_real)
    res_1 = preprocesar(audio_segunda_fuerte, fs, cfg_real)
    assert n_seg_1 == 2 and res_1.n_segmentos == 2, (
        f"n_segmentos debe ser 2; se obtuvo palabra_principal={n_seg_1}, preprocesar={res_1.n_segmentos}"
    )
    assert pal_1 == pytest.approx(segs_1[1], abs=1e-6), (
        f"Cuando la 2ª palabra es 3 dB más fuerte debe ganar {segs_1[1]}; ganó {pal_1}"
    )
    assert (res_1.inicio_s, res_1.fin_s) == pytest.approx(segs_1[1], abs=1e-6), (
        f"preprocesar debía elegir la 2ª palabra {segs_1[1]}; eligió {(res_1.inicio_s, res_1.fin_s)}"
    )

    # Caso 2: dos palabras exactamente iguales corridas un múltiplo exacto del avance H
    audio_empate = np.zeros(3 * fs, dtype=np.float64)
    audio_empate[m_ini_1 : m_ini_1 + n_pal] = onda_base
    audio_empate[m_ini_2 : m_ini_2 + n_pal] = onda_base

    segs_2 = _llamar_detectar_palabras(audio_empate, fs, cfg_real)
    assert len(segs_2) == 2, f"Se esperaban 2 segmentos en el caso de empate; se detectaron {len(segs_2)}"

    pal_2, n_seg_2 = palabra_principal(audio_empate, fs, cfg_real)
    res_2 = preprocesar(audio_empate, fs, cfg_real)
    assert n_seg_2 == 2 and res_2.n_segmentos == 2, (
        f"n_segmentos debe ser 2 en el empate; se obtuvo {n_seg_2} y {res_2.n_segmentos}"
    )
    assert pal_2 == pytest.approx(segs_2[0], abs=1e-6), (
        f"En caso de empate debe ganar la 1ª palabra {segs_2[0]}; ganó {pal_2}"
    )
    assert (res_2.inicio_s, res_2.fin_s) == pytest.approx(segs_2[0], abs=1e-6), (
        f"preprocesar debía elegir la 1ª palabra {segs_2[0]} en empate; eligió {(res_2.inicio_s, res_2.fin_s)}"
    )


@pytest.mark.parametrize(
    "nombre_caso,construir_audio",
    [
        ("silencio_3s", lambda fs: np.zeros(3 * fs, dtype=np.float64)),
        ("vacio_0_muestras", lambda fs: np.zeros(0, dtype=np.float64)),
        ("corto_100_muestras", lambda fs: 0.4 * np.sin(2.0 * np.pi * 300.0 * np.arange(100) / fs)),
    ],
)
def test_audios_sin_voz_no_validos(
    cfg_real: dict, nombre_caso: str, construir_audio
) -> None:
    """Silencio, audio vacío y audio de 100 muestras son no válidos con motivo 'sin_voz' sin errores."""
    from src.voz.preprocesamiento import MOTIVOS_V2, palabra_principal, preprocesar

    assert "sin_voz" in MOTIVOS_V2, f"'sin_voz' debe estar en MOTIVOS_V2 ({MOTIVOS_V2})"
    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    audio = construir_audio(fs)

    pal, n_seg = palabra_principal(audio, fs, cfg_real)
    assert pal is None and n_seg == 0, (
        f"[{nombre_caso}] palabra_principal debía devolver (None, 0); devolvió ({pal}, {n_seg})"
    )

    res = preprocesar(audio, fs, cfg_real)
    assert res.valido is False, f"[{nombre_caso}] Se esperaba valido=False; se obtuvo True"
    assert res.motivo == "sin_voz", (
        f"[{nombre_caso}] Se esperaba motivo='sin_voz'; se obtuvo '{res.motivo}'"
    )
    assert isinstance(res.y, np.ndarray) and res.y.dtype == np.float32 and len(res.y) == 0, (
        f"[{nombre_caso}] res.y debe ser ndarray float32 de 0 muestras; "
        f"se obtuvo shape={res.y.shape}, dtype={res.y.dtype}"
    )
    assert res.inicio_s is None and res.fin_s is None, (
        f"[{nombre_caso}] inicio_s y fin_s deben ser None; se obtuvo ({res.inicio_s}, {res.fin_s})"
    )
    assert res.n_segmentos == 0, (
        f"[{nombre_caso}] n_segmentos debe ser 0; se obtuvo {res.n_segmentos}"
    )


@pytest.mark.parametrize(
    "nombre_caso,construir_audio,dur_esperada_aprox",
    [
        (
            "corta_02s",
            lambda fs: _audio_con_palabra(
                duracion_palabra_s=0.2, inicio_s=1.0, duracion_total_s=3.0, fs=fs, pico=0.5, rampa_ms=10.0
            ),
            0.2,
        ),
        (
            "larga_28s_con_cierres_01s",
            lambda fs: _audio_palabra_28s_con_cierres(fs=fs, duracion_total_s=3.0, pico=0.5),
            2.8,
        ),
    ],
)
def test_palabra_fuera_de_rango_duracion_no_valida(
    cfg_real: dict, nombre_caso: str, construir_audio, dur_esperada_aprox: float
) -> None:
    """Palabras de 0,2 s y de 2,8 s con cierres internos de 0,1 s dan no válido con motivo 'duracion'."""
    from src.voz.preprocesamiento import MOTIVOS_V2, preprocesar

    assert "duracion" in MOTIVOS_V2, f"'duracion' debe estar en MOTIVOS_V2 ({MOTIVOS_V2})"
    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    audio = construir_audio(fs)

    res = preprocesar(audio, fs, cfg_real)
    assert res.valido is False, (
        f"[{nombre_caso}] Una palabra de ~{dur_esperada_aprox} s debe ser no válida; dio valido=True"
    )
    assert res.motivo == "duracion", (
        f"[{nombre_caso}] Se esperaba motivo='duracion'; se obtuvo '{res.motivo}'"
    )
    assert res.inicio_s is not None and res.fin_s is not None, (
        f"[{nombre_caso}] Con motivo='duracion', inicio_s y fin_s no deben ser None; "
        f"se obtuvo ({res.inicio_s}, {res.fin_s})"
    )
    dur_medida = res.fin_s - res.inicio_s
    assert dur_medida == pytest.approx(dur_esperada_aprox, abs=0.06), (
        f"[{nombre_caso}] Duración detectada ({dur_medida:.3f} s) lejos de {dur_esperada_aprox} s"
    )
    assert len(res.y) == 0 and res.y.dtype == np.float32, (
        f"[{nombre_caso}] Cuando no es válido, res.y debe tener 0 muestras float32; "
        f"se obtuvo len={len(res.y)}, dtype={res.y.dtype}"
    )


@pytest.mark.parametrize("borde", ["inicio", "fin"])
def test_palabra_pegada_al_borde_es_valida_y_completa_margen_con_ceros(
    cfg_real: dict, borde: str
) -> None:
    """Una palabra que empieza en la primera muestra o termina en la última es válida y rellena M ceros."""
    from src.voz.preprocesamiento import preprocesar

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    margen_ms = float(cfg_real["voz"]["preprocesamiento"]["margen_ms"])
    m_margen = round(margen_ms * fs / 1000.0)

    n_total = 3 * fs
    audio = np.zeros(n_total, dtype=np.float64)
    # Rampa nula en el borde pegado para que la primera/última trama tenga energía plena
    onda = _tono_armonico(0.8, fs=fs, f0=200.0, pico=0.5, rampa_ms=0.0)

    if borde == "inicio":
        audio[: len(onda)] = onda
    else:
        audio[n_total - len(onda) :] = onda

    res = preprocesar(audio, fs, cfg_real)
    assert res.valido is True, (
        f"Palabra pegada al {borde} debe ser válida; dio valido=False (motivo='{res.motivo}')"
    )
    if borde == "inicio":
        assert res.inicio_s == pytest.approx(0.0, abs=1e-6), (
            f"Palabra al inicio debía tener inicio_s=0.0; dio {res.inicio_s}"
        )
        afuera = res.y[:m_margen]
        assert np.all(afuera == 0.0), (
            f"Las primeras M={m_margen} muestras fuera del audio deben ser 0; "
            f"máximo medido={np.max(np.abs(afuera))}"
        )
    else:
        assert res.fin_s == pytest.approx(n_total / fs, abs=1e-6), (
            f"Palabra al final debía tener fin_s={n_total / fs}; dio {res.fin_s}"
        )
        afuera = res.y[-m_margen:]
        assert np.all(afuera == 0.0), (
            f"Las últimas M={m_margen} muestras fuera del audio deben ser 0; "
            f"máximo medido={np.max(np.abs(afuera))}"
        )


def test_sin_filtrar_antes_de_detectar_conserva_fricativa_s(cfg_real: dict) -> None:
    """Una 's' (>4,5 kHz, 150 ms) seguida de un tono de 300 ms tiene inicio_s a <= 30 ms del inicio de la 's'."""
    from src.voz.preprocesamiento import palabra_principal, preprocesar

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    audio = np.zeros(3 * fs, dtype=np.float64)

    inicio_s_esperado = 1.0
    n_s = round(0.150 * fs)
    n_tono = round(0.300 * fs)
    pico = 0.5

    # Fricativa "s": ruido blanco filtrado pasaaltos por encima de 4,5 kHz
    rng = np.random.default_rng(2026)
    ruido_crudo = rng.standard_normal(n_s + 2000)
    sos_hp = scipy.signal.butter(6, 4500.0, btype="highpass", fs=fs, output="sos")
    s_filtrada = scipy.signal.sosfiltfilt(sos_hp, ruido_crudo)[1000 : 1000 + n_s]
    s_filtrada = s_filtrada * (pico / float(np.max(np.abs(s_filtrada))))

    tono = _tono_armonico(0.300, fs=fs, f0=190.0, pico=pico, rampa_ms=10.0)
    assert len(tono) == n_tono

    m0 = round(inicio_s_esperado * fs)
    audio[m0 : m0 + n_s] = s_filtrada
    audio[m0 + n_s : m0 + n_s + n_tono] = tono

    pal, _ = palabra_principal(audio, fs, cfg_real)
    assert pal is not None, "No se detectó palabra en el estímulo 's' + tono"
    diferencia_s = abs(pal[0] - inicio_s_esperado)
    assert diferencia_s <= 0.030, (
        f"inicio_s ({pal[0]:.4f} s) se alejó {diferencia_s * 1000:.1f} ms del inicio de la 's' "
        f"({inicio_s_esperado:.4f} s); el límite es <= 30 ms (¿se filtró antes de detectar?)"
    )

    res = preprocesar(audio, fs, cfg_real)
    assert res.valido is True, (
        f"La palabra 's' (150 ms) + tono (300 ms) debe ser válida (450 ms >= 300 ms); "
        f"dio valido=False, motivo='{res.motivo}', inicio_s={res.inicio_s}, fin_s={res.fin_s}"
    )
    assert abs(res.inicio_s - inicio_s_esperado) <= 0.030, (
        f"En preprocesar, inicio_s ({res.inicio_s:.4f} s) debe estar a <= 30 ms de {inicio_s_esperado} s"
    )


@pytest.mark.parametrize("fs_origen", [44100, 48000])
def test_palabra_a_otra_frecuencia_coincide_con_remuestrear_antes(
    cfg_real: dict, fs_origen: int
) -> None:
    """Una palabra a 44,1 o 48 kHz en preprocesar da lo mismo que remuestreada antes con remuestrear."""
    from src.voz.audio import remuestrear
    from src.voz.preprocesamiento import preprocesar

    fs_destino = int(cfg_real["voz"]["frecuencia_muestreo"])
    audio_alta_fs = _audio_con_palabra(
        duracion_palabra_s=0.8, inicio_s=0.85, duracion_total_s=3.0, fs=fs_origen, f0=185.0, pico=0.5
    )

    res_directo = preprocesar(audio_alta_fs, fs_origen, cfg_real)
    audio_16k = remuestrear(audio_alta_fs, fs_origen, fs_destino)
    res_previo = preprocesar(audio_16k, fs_destino, cfg_real)

    assert res_directo.valido is True and res_previo.valido is True, (
        f"Ambos caminos a {fs_origen} Hz deben dar valido=True; "
        f"directo={res_directo.valido}, previo={res_previo.valido}"
    )
    assert res_directo.inicio_s == pytest.approx(res_previo.inicio_s, abs=1e-6), (
        f"inicio_s difiere a {fs_origen} Hz: directo={res_directo.inicio_s} vs previo={res_previo.inicio_s}"
    )
    assert res_directo.fin_s == pytest.approx(res_previo.fin_s, abs=1e-6), (
        f"fin_s difiere a {fs_origen} Hz: directo={res_directo.fin_s} vs previo={res_previo.fin_s}"
    )
    assert res_directo.y.shape == res_previo.y.shape, (
        f"El largo del clip difiere a {fs_origen} Hz: {res_directo.y.shape} vs {res_previo.y.shape}"
    )
    max_diff = float(np.max(np.abs(res_directo.y - res_previo.y)))
    assert max_diff <= 1e-6, (
        f"El clip preprocesado desde {fs_origen} Hz difiere del remuestreado antes en {max_diff:.2e}"
    )


# =============================================================================
# 7.3 Acondicionamiento, recorte y nivel
# =============================================================================


def test_pasabanda_igual_a_butter_sosfiltfilt(cfg_real: dict) -> None:
    """pasabanda coincide con scipy.signal.butter + sosfiltfilt con tolerancia 1e-9 y devuelve float64."""
    from src.voz.preprocesamiento import pasabanda

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    orden = int(cfg_real["voz"]["preprocesamiento"]["orden_filtro"])
    banda = cfg_real["voz"]["pasabanda_hz"]

    audio = _audio_con_palabra(duracion_palabra_s=0.9, inicio_s=0.7, duracion_total_s=3.0, fs=fs)
    audio += _ruido_estacionario(duracion_s=3.0, fs=fs, amplitud=0.05, semilla=77)

    filtrado = pasabanda(audio.astype(np.float32), fs, cfg_real)
    assert filtrado.dtype == np.float64, (
        f"pasabanda debe devolver float64; devolvió {filtrado.dtype}"
    )
    assert len(filtrado) == len(audio), (
        f"pasabanda debe conservar el largo ({len(audio)}); devolvió {len(filtrado)}"
    )

    sos = scipy.signal.butter(orden, banda, btype="bandpass", fs=fs, output="sos")
    esperado = scipy.signal.sosfiltfilt(sos, audio.astype(np.float32).astype(np.float64))
    max_diff = float(np.max(np.abs(filtrado - esperado)))
    assert max_diff <= 1e-9, (
        f"pasabanda difiere de sosfiltfilt(butter(...)) en {max_diff:.2e} (tolerancia 1e-9)"
    )


@pytest.mark.parametrize(
    "freq_hz,ganancia_min_db,ganancia_max_db",
    [
        (300.0, -0.3, 0.3),
        (1000.0, -0.3, 0.3),
        (2500.0, -0.3, 0.3),
        (80.0, -6.5, -5.5),
        (4000.0, -6.5, -5.5),
        (40.0, -300.0, -40.0),
        (6000.0, -300.0, -40.0),
    ],
)
def test_pasabanda_respuesta_en_frecuencia(
    cfg_real: dict, freq_hz: float, ganancia_min_db: float, ganancia_max_db: float
) -> None:
    """Verifica la ganancia de pasabanda con senos de 3 s medidos lejos de los bordes."""
    from src.voz.preprocesamiento import pasabanda

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    t = np.arange(3 * fs, dtype=np.float64) / fs
    seno = 0.5 * np.sin(2.0 * np.pi * freq_hz * t)

    salida = pasabanda(seno, fs, cfg_real)
    # Medimos en la ventana central [0,5 s, 2,5 s], lejos de los bordes transitorios
    m0, m1 = round(0.5 * fs), round(2.5 * fs)
    rms_in = float(np.sqrt(np.mean(seno[m0:m1] ** 2)))
    rms_out = float(np.sqrt(np.mean(salida[m0:m1] ** 2)))
    ganancia_db = 20.0 * math.log10(max(rms_out, 1e-20) / rms_in)

    assert ganancia_min_db <= ganancia_db <= ganancia_max_db, (
        f"A {freq_hz} Hz la ganancia medida fue {ganancia_db:.2f} dB; "
        f"se esperaba en [{ganancia_min_db}, {ganancia_max_db}] dB"
    )


def test_pasabanda_fase_cero_en_pulso_1khz(cfg_real: dict) -> None:
    """Un pulso de un tono de 1 kHz filtrado con pasabanda mantiene su máximo a 0 +- 1 muestra."""
    from src.voz.preprocesamiento import pasabanda

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    n_total = 3 * fs
    pulso = np.zeros(n_total, dtype=np.float64)

    n_pulso = round(0.060 * fs)
    m0 = n_total // 2 - n_pulso // 2
    t_pulso = np.arange(n_pulso, dtype=np.float64) / fs
    ventana = np.hanning(n_pulso)
    pulso[m0 : m0 + n_pulso] = ventana * np.sin(2.0 * np.pi * 1000.0 * t_pulso)

    filtrado = pasabanda(pulso, fs, cfg_real)
    corr = scipy.signal.correlate(filtrado, pulso, mode="full")
    lags = scipy.signal.correlation_lags(len(filtrado), len(pulso), mode="full")
    desplazamiento = int(lags[int(np.argmax(corr))])

    assert abs(desplazamiento) <= 1, (
        f"El filtro pasabanda debe ser de fase cero (desplazamiento 0 +- 1 muestra); "
        f"se midió desplazamiento={desplazamiento}"
    )


def test_preenfasis_formula_y_coeficiente_cero(cfg_real: dict) -> None:
    """Verifica z[0]=x[0], z[n]=x[n]-a*x[n-1], tipo float64 y que con a=0 devuelve lo mismo."""
    from src.voz.preprocesamiento import preenfasis

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    a = float(cfg_real["voz"]["preenfasis"])
    x = _tono_armonico(0.5, fs=fs, f0=220.0, pico=0.6).astype(np.float32)

    z = preenfasis(x, a)
    assert z.dtype == np.float64, f"preenfasis debe devolver float64; devolvió {z.dtype}"
    assert len(z) == len(x), f"preenfasis debe conservar el largo ({len(x)}); devolvió {len(z)}"

    x64 = x.astype(np.float64)
    esperado = np.empty_like(x64)
    esperado[0] = x64[0]
    esperado[1:] = x64[1:] - a * x64[:-1]
    max_diff = float(np.max(np.abs(z - esperado)))
    assert max_diff <= 1e-12, (
        f"preenfasis no coincide con la fórmula z[n] = x[n] - a*x[n-1]; diff={max_diff:.2e}"
    )

    z_cero = preenfasis(x, 0.0)
    assert np.allclose(z_cero, x64, atol=1e-12), (
        "Con a=0, preenfasis debe devolver la misma señal en float64"
    )


def test_recortar_largo_ubicacion_palabra_y_ceros_afuera(cfg_real: dict) -> None:
    """recortar devuelve m1-m0+2M muestras float64, la palabra en [M, M+m1-m0) y ceros fuera del audio."""
    from src.voz.preprocesamiento import recortar

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    margen_ms = float(cfg_real["voz"]["preprocesamiento"]["margen_ms"])
    m_margen = round(margen_ms * fs / 1000.0)

    n_total = fs  # 1 segundo
    x = np.arange(1, n_total + 1, dtype=np.float64) / n_total

    # Caso interior
    inicio_s, fin_s = 0.25, 0.75
    m0, m1 = round(inicio_s * fs), round(fin_s * fs)
    clip = recortar(x, fs, inicio_s, fin_s, cfg_real)
    assert clip.dtype == np.float64, f"recortar debe devolver float64; devolvió {clip.dtype}"
    assert len(clip) == (m1 - m0 + 2 * m_margen), (
        f"Largo esperado {m1 - m0 + 2 * m_margen}; se obtuvo {len(clip)}"
    )
    assert np.array_equal(clip[m_margen : m_margen + (m1 - m0)], x[m0:m1]), (
        "El tramo central [M, M + m1 - m0) no coincide con x[m0:m1]"
    )
    assert np.array_equal(clip[:m_margen], x[m0 - m_margen : m0]), (
        "El margen izquierdo interior no coincide con x[m0 - M : m0]"
    )
    assert np.array_equal(clip[m_margen + (m1 - m0) :], x[m1 : m1 + m_margen]), (
        "El margen derecho interior no coincide con x[m1 : m1 + M]"
    )

    # Caso desborde en ambos extremos (inicio cerca de 0 y fin cerca de len(x))
    ini_borde_s = 0.01  # m0 = 160 < 640
    fin_borde_s = 0.99  # m1 = 15840, m1 + 640 > 16000
    m0_b, m1_b = round(ini_borde_s * fs), round(fin_borde_s * fs)
    clip_b = recortar(x, fs, ini_borde_s, fin_borde_s, cfg_real)
    assert len(clip_b) == (m1_b - m0_b + 2 * m_margen), (
        f"Largo esperado con desborde {m1_b - m0_b + 2 * m_margen}; se obtuvo {len(clip_b)}"
    )
    n_ceros_izq = m_margen - m0_b
    n_ceros_der = (m1_b + m_margen) - n_total
    assert np.all(clip_b[:n_ceros_izq] == 0.0), (
        f"Las primeras {n_ceros_izq} muestras fuera del audio deben ser 0"
    )
    assert np.all(clip_b[-n_ceros_der:] == 0.0), (
        f"Las últimas {n_ceros_der} muestras fuera del audio deben ser 0"
    )
    assert np.array_equal(clip_b[m_margen : m_margen + (m1_b - m0_b)], x[m0_b:m1_b]), (
        "La palabra en [M, M + m1 - m0) no coincide cuando hay relleno con ceros"
    )


@pytest.mark.parametrize("amplitud_inicial", [0.003, 0.05, 0.4, 2.2])
def test_normalizar_rms_lleva_palabra_a_nivel_fijo_sin_depender_del_margen(
    cfg_real: dict, amplitud_inicial: float
) -> None:
    """normalizar_rms deja el RMS de la palabra en nivel_rms_db +- 0,01 dB sin importar los márgenes."""
    from src.voz.preprocesamiento import normalizar_rms

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    margen_ms = float(cfg_real["voz"]["preprocesamiento"]["margen_ms"])
    nivel_objetivo_db = float(cfg_real["voz"]["preprocesamiento"]["nivel_rms_db"])
    m_margen = round(margen_ms * fs / 1000.0)

    palabra = _tono_armonico(0.7, fs=fs, f0=200.0, pico=amplitud_inicial)
    clip_margen_cero = np.concatenate(
        [np.zeros(m_margen), palabra, np.zeros(m_margen)]
    )
    clip_margen_ruido = np.concatenate(
        [np.full(m_margen, 0.9), palabra, np.full(m_margen, -0.9)]
    )

    norm_cero = normalizar_rms(clip_margen_cero, fs, cfg_real)
    norm_ruido = normalizar_rms(clip_margen_ruido, fs, cfg_real)

    assert norm_cero.dtype == np.float32, (
        f"normalizar_rms debe devolver float32; devolvió {norm_cero.dtype}"
    )
    tramo_palabra = norm_cero[m_margen : len(norm_cero) - m_margen].astype(np.float64)
    rms_medido_db = 20.0 * math.log10(float(np.sqrt(np.mean(tramo_palabra ** 2))))
    assert abs(rms_medido_db - nivel_objetivo_db) <= 0.01, (
        f"Con amplitud {amplitud_inicial}, el RMS de la palabra quedó en {rms_medido_db:.4f} dBFS; "
        f"se esperaba {nivel_objetivo_db:.4f} +- 0,01 dBFS"
    )

    tramo_palabra_ruido = norm_ruido[m_margen : len(norm_ruido) - m_margen]
    assert np.allclose(tramo_palabra, tramo_palabra_ruido, atol=1e-7), (
        "El contenido de los márgenes no debe alterar el factor de normalización de la palabra"
    )


def test_normalizar_rms_palabra_ceros_lanza_value_error(cfg_real: dict) -> None:
    """normalizar_rms lanza ValueError si la región de la palabra es toda ceros (incluso con margen no nulo)."""
    from src.voz.preprocesamiento import normalizar_rms

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    margen_ms = float(cfg_real["voz"]["preprocesamiento"]["margen_ms"])
    m_margen = round(margen_ms * fs / 1000.0)

    clip_palabra_ceros = np.zeros(2 * m_margen + 8000, dtype=np.float64)
    clip_palabra_ceros[:m_margen] = 0.5
    clip_palabra_ceros[-m_margen:] = -0.5

    with pytest.raises(ValueError, match=".*"):
        normalizar_rms(clip_palabra_ceros, fs, cfg_real)


def test_preprocesar_equivale_a_cadena_pasabanda_preenfasis_recortar_normalizar(
    cfg_real: dict,
) -> None:
    """preprocesar de un audio válido coincide con normalizar_rms(recortar(preenfasis(pasabanda(y)))) (tol 1e-6)."""
    from src.voz.preprocesamiento import (
        normalizar_rms,
        pasabanda,
        preenfasis,
        preprocesar,
        recortar,
    )

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    audio = _audio_con_palabra(duracion_palabra_s=0.85, inicio_s=0.75, duracion_total_s=3.0, fs=fs)

    res = preprocesar(audio, fs, cfg_real)
    assert res.valido is True, "El audio de prueba debía ser válido"
    assert res.y.dtype == np.float32, f"preprocesar.y debe ser float32; es {res.y.dtype}"

    a = float(cfg_real["voz"]["preenfasis"])
    esperado = normalizar_rms(
        recortar(
            preenfasis(pasabanda(audio, fs, cfg_real), a),
            fs,
            res.inicio_s,
            res.fin_s,
            cfg_real,
        ),
        fs,
        cfg_real,
    )
    max_diff = float(np.max(np.abs(res.y - esperado)))
    assert max_diff <= 1e-6, (
        f"preprocesar difiere de la composición manual de las etapas en {max_diff:.2e} (tol 1e-6)"
    )


@pytest.mark.parametrize("factor_escala", [0.1, 3.0])
def test_preprocesar_invariante_al_escalar_entrada(
    cfg_real: dict, factor_escala: float
) -> None:
    """Multiplicar el audio de entrada por 0,1 o por 3 da el mismo clip (tolerancia 1e-5)."""
    from src.voz.preprocesamiento import preprocesar

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    audio_base = _audio_con_palabra(
        duracion_palabra_s=0.85, inicio_s=0.8, duracion_total_s=3.0, fs=fs, pico=0.3
    )

    res_base = preprocesar(audio_base, fs, cfg_real)
    res_escalado = preprocesar(audio_base * factor_escala, fs, cfg_real)

    assert res_base.valido is True and res_escalado.valido is True, (
        f"Tanto el audio base como el escalado por {factor_escala} deben ser válidos"
    )
    assert res_escalado.y.shape == res_base.y.shape, (
        f"El largo del clip cambió al escalar por {factor_escala}: "
        f"{res_escalado.y.shape} vs {res_base.y.shape}"
    )
    max_diff = float(np.max(np.abs(res_escalado.y - res_base.y)))
    assert max_diff <= 1e-5, (
        f"Escalar la entrada por {factor_escala} cambió el clip de salida en {max_diff:.2e} (tol 1e-5)"
    )


# =============================================================================
# 7.4 Aumento (aumento.py)
# =============================================================================


@pytest.mark.parametrize("factor", [0.92, 1.08])
def test_cambiar_velocidad_frecuencia_largo_y_resample_poly(factor: float) -> None:
    """Un seno de 440 Hz de 1 s con factor 0,92 y 1,08 queda en 440*factor +- 2 Hz y len(y)/factor +- 2 muestras."""
    from src.voz.aumento import cambiar_velocidad

    fs = 16000
    f0 = 440.0
    t = np.arange(fs, dtype=np.float64) / fs
    seno = (0.5 * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)

    salida = cambiar_velocidad(seno, factor)
    assert salida.dtype == np.float32, (
        f"cambiar_velocidad debe devolver float32; devolvió {salida.dtype}"
    )
    largo_esperado = len(seno) / factor
    assert abs(len(salida) - largo_esperado) <= 2, (
        f"Para factor={factor}, largo medido={len(salida)}, esperado={largo_esperado:.1f} +- 2 muestras"
    )

    # Pico de la FFT con zero-padding para resolución < 0,1 Hz
    n_fft = 1 << 18
    espectro = np.abs(np.fft.rfft(salida.astype(np.float64) * np.hanning(len(salida)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    freq_pico = float(freqs[int(np.argmax(espectro))])
    freq_esperada = f0 * factor
    assert abs(freq_pico - freq_esperada) <= 2.0, (
        f"Para factor={factor}, pico FFT en {freq_pico:.2f} Hz; se esperaba {freq_esperada:.2f} +- 2 Hz"
    )

    # Equivalencia con resample_poly según sección 4
    d = round(100 * round(factor, 2))
    g = math.gcd(100, d)
    esperado_poly = scipy.signal.resample_poly(
        seno.astype(np.float64), 100 // g, d // g
    ).astype(np.float32)
    max_diff = float(np.max(np.abs(salida - esperado_poly)))
    assert max_diff <= 1e-6, (
        f"cambiar_velocidad({factor}) difiere de resample_poly en {max_diff:.2e} (tol 1e-6)"
    )


def test_cambiar_velocidad_redondea_a_dos_decimales_y_copia_con_factor_uno() -> None:
    """1,083 se trata como 1,08; con factor 1 devuelve una copia float32 igual."""
    from src.voz.aumento import cambiar_velocidad

    fs = 16000
    t = np.arange(fs, dtype=np.float64) / fs
    y = (0.4 * np.sin(2.0 * np.pi * 350.0 * t)).astype(np.float32)

    z_1083 = cambiar_velocidad(y, 1.083)
    z_108 = cambiar_velocidad(y, 1.08)
    assert np.array_equal(z_1083, z_108), (
        "cambiar_velocidad debe redondear el factor a 2 decimales (1.083 debe dar igual que 1.08)"
    )

    z_uno = cambiar_velocidad(y, 1.0)
    assert z_uno.dtype == np.float32, f"Con factor=1 debe devolver float32; dio {z_uno.dtype}"
    assert np.allclose(z_uno, y, atol=1e-7), "Con factor=1 la señal devuelta debe ser igual a la entrada"
    assert z_uno is not y and not np.shares_memory(z_uno, y), (
        "Con factor=1 cambiar_velocidad debe devolver una copia independiente del arreglo original"
    )


@pytest.mark.parametrize("snr_db", [10.0, 17.5, 25.0])
def test_sumar_ruido_snr_exacto_y_muestras_fuera_de_palabra(snr_db: float) -> None:
    """La SNR sobre la palabra es snr_db +- 0,01 dB y en todo el clip el audio es y + g*ruido[:len(y)]."""
    from src.voz.aumento import sumar_ruido

    fs = 16000
    palabra = (0.8, 1.8)
    y = _audio_con_palabra(
        duracion_palabra_s=1.0, inicio_s=0.8, duracion_total_s=3.0, fs=fs, pico=0.5
    ).astype(np.float32)

    # Ruido periódico cada 0,2 s para que la potencia media en [0.8, 1.8) y en [0, 3.0) sea idéntica
    bloque_ruido = _ruido_estacionario(duracion_s=0.2, fs=fs, amplitud=0.08, semilla=55)
    ruido = np.tile(bloque_ruido, 20).astype(np.float32)  # 4,0 s >= 3,0 s

    mezcla = sumar_ruido(y, ruido, snr_db, palabra, fs)
    assert mezcla.dtype == np.float32, f"sumar_ruido debe devolver float32; devolvió {mezcla.dtype}"
    assert len(mezcla) == len(y), f"sumar_ruido cambió el largo: {len(mezcla)} vs {len(y)}"

    m0, m1 = round(palabra[0] * fs), round(palabra[1] * fs)
    y64 = y.astype(np.float64)
    r64 = ruido[: len(y)].astype(np.float64)
    ps = float(np.mean(y64[m0:m1] ** 2))
    pn = float(np.mean(r64 ** 2))
    g_esperado = math.sqrt(ps / (pn * (10.0 ** (snr_db / 10.0))))

    ruido_sumado_palabra = mezcla[m0:m1].astype(np.float64) - y64[m0:m1]
    p_ruido_medido = float(np.mean(ruido_sumado_palabra ** 2))
    snr_medido_db = 10.0 * math.log10(ps / p_ruido_medido)
    assert abs(snr_medido_db - snr_db) <= 0.01, (
        f"SNR medida sobre la palabra ({snr_medido_db:.4f} dB) != esperada ({snr_db:.4f} +- 0,01 dB)"
    )

    esperado = (y64 + g_esperado * r64).astype(np.float32)
    assert np.allclose(mezcla[:m0], esperado[:m0], atol=1e-6) and np.allclose(
        mezcla[m1:], esperado[m1:], atol=1e-6
    ), "Fuera de la palabra el audio debe ser exactamente y + g*ruido[:len(y)]"


@pytest.mark.parametrize(
    "caso_invalido",
    ["ruido_mas_corto", "ruido_ceros", "palabra_ceros"],
)
def test_sumar_ruido_casos_invalidos_lanzan_value_error(caso_invalido: str) -> None:
    """sumar_ruido lanza ValueError con ruido más corto que y, ruido de ceros o palabra de ceros."""
    from src.voz.aumento import sumar_ruido

    fs = 16000
    palabra = (0.5, 1.5)
    y = _audio_con_palabra(duracion_palabra_s=1.0, inicio_s=0.5, duracion_total_s=2.0, fs=fs)
    ruido = _ruido_estacionario(duracion_s=2.5, fs=fs, amplitud=0.05)

    if caso_invalido == "ruido_mas_corto":
        ruido_corto = ruido[: len(y) - 1]
        with pytest.raises(ValueError):
            sumar_ruido(y, ruido_corto, 15.0, palabra, fs)
    elif caso_invalido == "ruido_ceros":
        with pytest.raises(ValueError):
            sumar_ruido(y, np.zeros(len(y), dtype=np.float64), 15.0, palabra, fs)
    else:
        y_palabra_ceros = np.zeros_like(y)
        y_palabra_ceros[: round(0.4 * fs)] = 0.5  # Energía solo fuera de la palabra
        with pytest.raises(ValueError):
            sumar_ruido(y_palabra_ceros, ruido, 15.0, palabra, fs)


def test_sortear_aumento_determinismo_rangos_y_cobertura(cfg_real: dict) -> None:
    """Verifica determinismo de sortear_aumento y rangos/tipos en 1000 sorteos."""
    from src.voz.aumento import sortear_aumento

    n_ruidos = 4
    rng_a = np.random.default_rng(98765)
    rng_b = np.random.default_rng(98765)
    assert sortear_aumento(cfg_real, rng_a, n_ruidos) == sortear_aumento(
        cfg_real, rng_b, n_ruidos
    ), "sortear_aumento con el mismo generador debe devolver exactamente el mismo dict"

    vel_min, vel_max = cfg_real["voz"]["preprocesamiento"]["aumento"]["velocidad"]
    snr_min, snr_max = cfg_real["voz"]["preprocesamiento"]["aumento"]["snr_db"]

    rng = np.random.default_rng(123456)
    ruidos_vistos: set[int] = set()
    for i in range(1000):
        d = sortear_aumento(cfg_real, rng, n_ruidos)
        assert set(d.keys()) == {"factor", "snr_db", "ruido", "desplazamiento"}, (
            f"Claves inesperadas en sortear_aumento: {set(d.keys())}"
        )
        factor = d["factor"]
        assert round(factor, 2) == pytest.approx(factor, abs=1e-12), (
            f"Sorteo {i}: factor={factor} no está redondeado a 2 decimales"
        )
        assert float(vel_min) <= factor <= float(vel_max), (
            f"Sorteo {i}: factor={factor} fuera de [{vel_min}, {vel_max}]"
        )
        snr = d["snr_db"]
        assert float(snr_min) <= snr <= float(snr_max), (
            f"Sorteo {i}: snr_db={snr} fuera de [{snr_min}, {snr_max}]"
        )
        idx_ruido = d["ruido"]
        assert isinstance(idx_ruido, int) and not isinstance(idx_ruido, bool), (
            f"Sorteo {i}: 'ruido' debe ser int; se obtuvo {type(idx_ruido)}"
        )
        assert 0 <= idx_ruido < n_ruidos, (
            f"Sorteo {i}: índice de ruido {idx_ruido} fuera de [0, {n_ruidos - 1}]"
        )
        ruidos_vistos.add(idx_ruido)

        despl = d["desplazamiento"]
        assert 0.0 <= despl < 1.0, (
            f"Sorteo {i}: desplazamiento={despl} fuera de [0, 1)"
        )

    assert ruidos_vistos == set(range(n_ruidos)), (
        f"En 1000 sorteos debían aparecer todos los ruidos 0..{n_ruidos - 1}; aparecieron {ruidos_vistos}"
    )


def test_semilla_copia_reproducible_y_distinta_por_archivo_o_copia(cfg_real: dict) -> None:
    """semilla_copia da la misma secuencia para el mismo archivo y copia, y distinta si cambia alguno."""
    from src.voz.aumento import semilla_copia

    archivo_1 = "train/nb_p01_identificar_01.wav"
    archivo_2 = "train/nb_p01_identificar_02.wav"

    g1 = semilla_copia(cfg_real, archivo_1, 1)
    g2 = semilla_copia(cfg_real, archivo_1, 1)
    g_otra_copia = semilla_copia(cfg_real, archivo_1, 2)
    g_otro_archivo = semilla_copia(cfg_real, archivo_2, 1)

    esperado_rng = np.random.default_rng(
        [int(cfg_real["semilla_global"]), zlib.crc32(archivo_1.encode("utf-8")), 1]
    )

    muestras_1 = g1.random(8)
    muestras_2 = g2.random(8)
    muestras_esp = esperado_rng.random(8)
    assert np.array_equal(muestras_1, muestras_2), (
        "semilla_copia debe ser determinista para el mismo archivo y copia"
    )
    assert np.array_equal(muestras_1, muestras_esp), (
        "semilla_copia no coincide con default_rng([semilla_global, crc32(archivo), copia])"
    )
    assert not np.array_equal(muestras_1, g_otra_copia.random(8)), (
        "semilla_copia debe cambiar al cambiar el número de copia"
    )
    assert not np.array_equal(muestras_1, g_otro_archivo.random(8)), (
        "semilla_copia debe cambiar al cambiar el nombre del archivo"
    )


@pytest.mark.parametrize("factor", [0.92, 1.08])
def test_aumentar_composicion_recorte_a_clip_max_y_error_ruido_corto(
    cfg_real: dict, factor: float
) -> None:
    """aumentar coincide con los pasos manuales de la sección 4, respeta C muestras y valida largo del ruido."""
    from src.voz.aumento import aumentar, cambiar_velocidad, sumar_ruido
    from src.voz.preprocesamiento import preprocesar

    fs = int(cfg_real["voz"]["frecuencia_muestreo"])
    clip_max_s = float(cfg_real["voz"]["dataset"]["clip_max_s"])
    c_max = round(clip_max_s * fs)

    palabra = (0.9, 1.9)
    y = _audio_con_palabra(
        duracion_palabra_s=1.0, inicio_s=0.9, duracion_total_s=3.0, fs=fs, pico=0.6
    ).astype(np.float32)

    ruidos = [
        _ruido_estacionario(duracion_s=3.6, fs=fs, amplitud=0.02, semilla=11).astype(np.float32),
        _ruido_estacionario(duracion_s=4.0, fs=fs, amplitud=0.025, semilla=22).astype(np.float32),
    ]
    params = {
        "factor": factor,
        "snr_db": 18.0,
        "ruido": 1,
        "desplazamiento": 0.35,
    }

    copia_aum = aumentar(y, fs, palabra, ruidos, params, cfg_real)
    assert copia_aum.dtype == np.float32, (
        f"aumentar debe devolver float32; devolvió {copia_aum.dtype}"
    )

    # Composición manual de los pasos 1, 2 y 3 de la sección 4
    z = cambiar_velocidad(y, factor)
    ini_z, fin_z = palabra[0] / factor, palabra[1] / factor
    if len(z) > c_max:
        centro = round((ini_z + fin_z) / 2.0 * fs)
        i0 = min(max(0, centro - c_max // 2), len(z) - c_max)
        z = z[i0 : i0 + c_max]
        ini_z -= i0 / fs
        fin_z -= i0 / fs

    r = ruidos[params["ruido"]]
    off = math.floor(params["desplazamiento"] * (len(r) - len(z) + 1))
    esperado = sumar_ruido(z, r[off:], params["snr_db"], (ini_z, fin_z), fs)

    if factor < 1.0:
        assert len(copia_aum) == c_max, (
            f"Con factor={factor} < 1 sobre un clip de 3 s, la salida debe tener C={c_max} muestras; "
            f"tuvo {len(copia_aum)}"
        )
        res_pre = preprocesar(copia_aum, fs, cfg_real)
        assert res_pre.valido is True, (
            "La copia aumentada con factor < 1 debe seguir conteniendo la palabra válida"
        )
    else:
        largo_vel = len(cambiar_velocidad(y, factor))
        assert len(copia_aum) == largo_vel, (
            f"Con factor={factor} >= 1, la salida debe tener {largo_vel} muestras; tuvo {len(copia_aum)}"
        )

    max_diff = float(np.max(np.abs(copia_aum - esperado)))
    assert max_diff <= 1e-6, (
        f"aumentar difiere de la composición manual de la sección 4 en {max_diff:.2e}"
    )

    # ValueError si el ruido elegido es más corto que z
    ruidos_cortos = [ruidos[0], ruidos[1][: len(z) - 5]]
    with pytest.raises(ValueError):
        aumentar(y, fs, palabra, ruidos_cortos, params, cfg_real)


# =============================================================================
# 7.5 Dataset preprocesado (generar_prepro, con dataset de V1 en tmp_path)
# =============================================================================


def test_generar_prepro_columnas_orden_y_tipos_retorno(dataset_v1_sintetico: dict) -> None:
    """Verifica encabezado COLUMNAS_V2, formato CSV (UTF-8 sin BOM, \\n), orden y tipos devueltos."""
    from src.voz.preprocesamiento import COLUMNAS_V2

    columnas_esperadas = (
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
    assert tuple(COLUMNAS_V2) == columnas_esperadas, (
        f"COLUMNAS_V2 no coincide con el contrato: {COLUMNAS_V2} vs {columnas_esperadas}"
    )

    destino: Path = dataset_v1_sintetico["destino"]
    filas_retorno: list[dict] = dataset_v1_sintetico["filas_retorno"]
    filas_v1: list[dict] = dataset_v1_sintetico["filas_v1"]

    csv_path = destino / "metadatos.csv"
    assert csv_path.is_file(), "generar_prepro no creó metadatos.csv en destino"

    raw = csv_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "metadatos.csv no debe tener BOM UTF-8"
    assert b"\r\n" not in raw, "metadatos.csv debe usar fin de línea '\\n', no '\\r\\n'"

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == columnas_esperadas, (
            f"Encabezado de metadatos.csv ({reader.fieldnames}) != COLUMNAS_V2 ({columnas_esperadas})"
        )
        filas_csv = list(reader)

    assert len(filas_csv) == len(filas_retorno), (
        f"Cantidad de filas en CSV ({len(filas_csv)}) != devueltas por generar_prepro ({len(filas_retorno)})"
    )

    # Orden de fuentes igual al de V1, y dentro de cada fuente copia 0, 1, ...
    orden_fuentes_visto: list[str] = []
    copias_por_fuente: dict[str, list[int]] = {}
    for f_ret in filas_retorno:
        fuente = f_ret["fuente"]
        if not orden_fuentes_visto or orden_fuentes_visto[-1] != fuente:
            orden_fuentes_visto.append(fuente)
        copias_por_fuente.setdefault(fuente, []).append(f_ret["copia"])

    orden_v1 = [item["archivo"] for item in filas_v1]
    assert orden_fuentes_visto == orden_v1, (
        f"El orden de clips en metadatos.csv ({orden_fuentes_visto}) no respeta el de V1 ({orden_v1})"
    )
    for fuente, lista_copias in copias_por_fuente.items():
        assert lista_copias == list(range(len(lista_copias))), (
            f"Las copias de {fuente} no están en orden 0..k: {lista_copias}"
        )

    # Coincidencia fila a fila entre el CSV y los dicts devueltos (con sus tipos)
    cols_int = {"copia", "n_segmentos", "muestras"}
    cols_float_3 = {"palabra_inicio_s", "palabra_fin_s", "duracion_palabra_s"}
    cols_float_2 = {"factor", "snr_db"}

    for idx, (f_csv, f_ret) in enumerate(zip(filas_csv, filas_retorno)):
        assert isinstance(f_ret["valido"], bool), (
            f"Fila {idx}: 'valido' debe ser bool en el dict devuelto; es {type(f_ret['valido'])}"
        )
        assert f_csv["valido"] == ("1" if f_ret["valido"] else "0"), (
            f"Fila {idx}: 'valido' en CSV ('{f_csv['valido']}') no coincide con {f_ret['valido']}"
        )
        for col in cols_int:
            if f_csv[col] == "":
                assert f_ret[col] is None, f"Fila {idx}: {col} vacío en CSV debe ser None"
            else:
                assert isinstance(f_ret[col], int) and not isinstance(f_ret[col], bool), (
                    f"Fila {idx}: '{col}' debe ser int; se obtuvo {type(f_ret[col])}"
                )
                assert f_ret[col] == int(f_csv[col]), (
                    f"Fila {idx}: '{col}'={f_ret[col]} != CSV '{f_csv[col]}'"
                )
        for col in cols_float_3:
            if f_csv[col] == "":
                assert f_ret[col] is None, f"Fila {idx}: '{col}' vacío en CSV debe ser None en dict"
            else:
                assert isinstance(f_ret[col], float), (
                    f"Fila {idx}: '{col}' debe ser float; es {type(f_ret[col])}"
                )
                assert f_ret[col] == pytest.approx(float(f_csv[col]), abs=1e-9), (
                    f"Fila {idx}: '{col}'={f_ret[col]} != CSV '{f_csv[col]}'"
                )
                assert round(f_ret[col], 3) == pytest.approx(f_ret[col], abs=1e-9), (
                    f"Fila {idx}: '{col}'={f_ret[col]} debe estar redondeado a 3 decimales"
                )
        for col in cols_float_2:
            if f_csv[col] == "":
                assert f_ret[col] is None, f"Fila {idx}: '{col}' vacío en CSV debe ser None en dict"
            else:
                assert isinstance(f_ret[col], float), (
                    f"Fila {idx}: '{col}' debe ser float; es {type(f_ret[col])}"
                )
                assert f_ret[col] == pytest.approx(float(f_csv[col]), abs=1e-9), (
                    f"Fila {idx}: '{col}'={f_ret[col]} != CSV '{f_csv[col]}'"
                )
                assert round(f_ret[col], 2) == pytest.approx(f_ret[col], abs=1e-9), (
                    f"Fila {idx}: '{col}'={f_ret[col]} debe estar redondeado a 2 decimales"
                )


def test_generar_prepro_reglas_de_copias_y_ruidos_de_train(dataset_v1_sintetico: dict) -> None:
    """Verifica que solo los clips de train válidos que no son de ruido llevan copias y usan ruido de train."""
    filas: list[dict] = dataset_v1_sintetico["filas_retorno"]
    copias_cfg = int(dataset_v1_sintetico["cfg"]["voz"]["preprocesamiento"]["aumento"]["copias"])

    por_fuente: dict[str, list[dict]] = {}
    for f in filas:
        por_fuente.setdefault(f["fuente"], []).append(f)

    ruidos_train_esperados = {
        "train/ruido_01.wav",
        "train/ruido_02.wav",
        "train/ruido_03.wav",
    }

    for fuente, grupo in por_fuente.items():
        copia_0 = grupo[0]
        assert copia_0["copia"] == 0
        assert copia_0["factor"] is None and copia_0["snr_db"] is None and copia_0["ruido"] is None, (
            f"La copia 0 de {fuente} debe tener factor, snr_db y ruido vacíos (None)"
        )

        debe_tener_copias = (
            copia_0["particion"] == "train"
            and copia_0["origen"] != "ruido"
            and copia_0["valido"] is True
        )
        if debe_tener_copias:
            assert len(grupo) == 1 + copias_cfg, (
                f"El clip de train válido {fuente} debía tener {1 + copias_cfg} filas; tiene {len(grupo)}"
            )
            for c_row in grupo[1:]:
                assert c_row["ruido"] in ruidos_train_esperados, (
                    f"La copia {c_row['copia']} de {fuente} usó ruido='{c_row['ruido']}', "
                    f"que no es un ruido de train ({ruidos_train_esperados})"
                )
                assert c_row["ruido"] != "val/ruido_val_01.wav", (
                    "Nunca debe usarse un clip de ruido de val en el aumento"
                )
        else:
            assert len(grupo) == 1, (
                f"El clip {fuente} (particion={copia_0['particion']}, origen={copia_0['origen']}, "
                f"valido={copia_0['valido']}) no debe tener copias aumentadas; tiene {len(grupo)} filas"
            )


def test_generar_prepro_wav_escritos_formato_y_contenido_exacto(
    dataset_v1_sintetico: dict,
) -> None:
    """Se escribe exactamente un WAV FLOAT mono a 16 kHz por fila válida y coincide con preprocesar/aumentar."""
    from src.voz.aumento import aumentar, semilla_copia, sortear_aumento
    from src.voz.preprocesamiento import preprocesar

    cfg: dict = dataset_v1_sintetico["cfg"]
    fs: int = dataset_v1_sintetico["fs"]
    origen: Path = dataset_v1_sintetico["origen"]
    destino: Path = dataset_v1_sintetico["destino"]
    filas: list[dict] = dataset_v1_sintetico["filas_retorno"]

    archivos_en_disco = {
        p.relative_to(destino).as_posix() for p in destino.rglob("*") if p.is_file()
    }
    wav_esperados = {f["archivo"] for f in filas if f["valido"]}
    assert archivos_en_disco == (wav_esperados | {"metadatos.csv"}), (
        f"Archivos en destino ({archivos_en_disco}) != esperados ({wav_esperados | {'metadatos.csv'}})"
    )

    # Cargamos los ruidos de train ordenados por archivo tal como exige la sección 1
    archivos_ruido_train = sorted(
        item["archivo"]
        for item in dataset_v1_sintetico["filas_v1"]
        if item["particion"] == "train" and item["origen"] == "ruido"
    )
    ruidos_train = [
        sf.read(origen / rel, dtype="float64")[0] for rel in archivos_ruido_train
    ]

    # Cacheamos preprocesar de cada copia 0 para obtener la palabra original
    pre_copia_0: dict[str, tuple[float, float]] = {}

    for f in filas:
        fuente = f["fuente"]
        copia = f["copia"]
        stem_v1 = Path(fuente).stem
        nombre_esperado = f"{f['particion']}/{stem_v1}_k{copia}.wav"

        y_v1, _ = sf.read(origen / fuente, dtype="float64")
        if copia == 0:
            res_esp = preprocesar(y_v1, fs, cfg)
            if res_esp.valido:
                pre_copia_0[fuente] = ( float(res_esp.inicio_s), float(res_esp.fin_s) )
        else:
            rng = semilla_copia(cfg, fuente, copia)
            params = sortear_aumento(cfg, rng, len(ruidos_train))
            y_aum = aumentar(y_v1, fs, pre_copia_0[fuente], ruidos_train, params, cfg)
            res_esp = preprocesar(y_aum, fs, cfg)

        if f["valido"]:
            assert f["archivo"] == nombre_esperado, (
                f"Nombre de archivo '{f['archivo']}' != esperado '{nombre_esperado}'"
            )
            ruta_wav = destino / f["archivo"]
            info = sf.info(ruta_wav)
            assert info.samplerate == fs, f"{f['archivo']}: samplerate={info.samplerate} != {fs}"
            assert info.channels == 1, f"{f['archivo']}: channels={info.channels} != 1 (mono)"
            assert info.subtype == "FLOAT", (
                f"{f['archivo']}: subtype='{info.subtype}' != 'FLOAT' (float32)"
            )
            assert info.frames == f["muestras"], (
                f"{f['archivo']}: frames en WAV ({info.frames}) != columna muestras ({f['muestras']})"
            )

            audio_leido, _ = sf.read(ruta_wav, dtype="float32")
            max_diff = float(np.max(np.abs(audio_leido - res_esp.y)))
            assert max_diff <= 1e-7, (
                f"El WAV {f['archivo']} difiere de preprocesar esperado en {max_diff:.2e} (tol 1e-7)"
            )


def test_generar_prepro_filas_no_validas_y_duracion_palabra(dataset_v1_sintetico: dict) -> None:
    """Verifica motivos de filas no válidas (sin_voz, duracion) y que duracion_palabra_s = fin - inicio."""
    filas: list[dict] = dataset_v1_sintetico["filas_retorno"]
    por_fuente_k0 = {f["fuente"]: f for f in filas if f["copia"] == 0}

    fila_silencio = por_fuente_k0["train/nb_p01_otro_silencio.wav"]
    assert fila_silencio["valido"] is False, "El clip de silencio debe ser no válido"
    assert fila_silencio["motivo"] == "sin_voz", (
        f"El silencio debe tener motivo='sin_voz'; tuvo '{fila_silencio['motivo']}'"
    )
    assert fila_silencio["archivo"] in (None, ""), (
        f"Fila no válida debe tener archivo vacío; tuvo '{fila_silencio['archivo']}'"
    )
    assert fila_silencio["muestras"] == 0, (
        f"Fila no válida debe tener muestras=0; tuvo {fila_silencio['muestras']}"
    )
    assert (
        fila_silencio["palabra_inicio_s"] is None
        and fila_silencio["palabra_fin_s"] is None
        and fila_silencio["duracion_palabra_s"] is None
    ), "En sin_voz, palabra_inicio_s, palabra_fin_s y duracion_palabra_s deben ser None"

    fila_larga = por_fuente_k0["train/nb_p01_otro_largo.wav"]
    assert fila_larga["valido"] is False, "El clip con palabra de 2,8 s debe ser no válido"
    assert fila_larga["motivo"] == "duracion", (
        f"La palabra de 2,8 s debe tener motivo='duracion'; tuvo '{fila_larga['motivo']}'"
    )
    assert fila_larga["archivo"] in (None, "") and fila_larga["muestras"] == 0, (
        "Fila no válida por duración debe tener archivo vacío y muestras=0"
    )
    assert fila_larga["palabra_inicio_s"] is not None and fila_larga["palabra_fin_s"] is not None, (
        "Con motivo='duracion' deben informarse palabra_inicio_s y palabra_fin_s"
    )

    for f in filas:
        if f["palabra_inicio_s"] is not None:
            resta = f["palabra_fin_s"] - f["palabra_inicio_s"]
            assert abs(f["duracion_palabra_s"] - resta) <= 0.001, (
                f"En {f['fuente']} (k={f['copia']}), duracion_palabra_s ({f['duracion_palabra_s']}) "
                f"!= fin - inicio ({resta:.4f}) +- 0,001"
            )


def test_generar_prepro_reproducibilidad_e_independencia_del_orden(
    dataset_v1_sintetico: dict, tmp_path: Path
) -> None:
    """Generar dos veces da los mismos WAV byte a byte y el mismo CSV; reordenar V1 no cambia ningún WAV."""
    from src.voz.preprocesamiento import generar_prepro

    cfg: dict = dataset_v1_sintetico["cfg"]
    origen: Path = dataset_v1_sintetico["origen"]
    destino_1: Path = dataset_v1_sintetico["destino"]

    destino_2 = tmp_path / "repeticion_exacta"
    generar_prepro(cfg, origen=origen, destino=destino_2, sobrescribir=False)

    assert (destino_1 / "metadatos.csv").read_bytes() == (
        destino_2 / "metadatos.csv"
    ).read_bytes(), "Dos corridas con el mismo origen deben producir el mismo metadatos.csv byte a byte"

    wavs_1 = sorted(
        p.relative_to(destino_1).as_posix()
        for p in destino_1.rglob("*.wav")
    )
    for rel in wavs_1:
        assert (destino_1 / rel).read_bytes() == (destino_2 / rel).read_bytes(), (
            f"El WAV {rel} no es idéntico byte a byte entre dos corridas"
        )

    # Invertimos el orden de todas las filas de metadatos.csv en un nuevo origen
    origen_invertido = tmp_path / "origen_invertido"
    shutil.copytree(origen, origen_invertido)
    csv_orig = origen / "metadatos.csv"
    with csv_orig.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        filas_orig = list(reader)

    with (origen_invertido / "metadatos.csv").open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reversed(filas_orig):
            writer.writerow(row)

    destino_invertido = tmp_path / "destino_invertido"
    generar_prepro(cfg, origen=origen_invertido, destino=destino_invertido, sobrescribir=False)

    wavs_inv = sorted(
        p.relative_to(destino_invertido).as_posix()
        for p in destino_invertido.rglob("*.wav")
    )
    assert wavs_inv == wavs_1, (
        "Cambiar el orden de las filas del CSV de V1 cambió el conjunto de WAV generados"
    )
    for rel in wavs_1:
        assert (destino_1 / rel).read_bytes() == (destino_invertido / rel).read_bytes(), (
            f"El WAV {rel} cambió al reordenar las filas de metadatos.csv de V1"
        )


@pytest.mark.parametrize(
    "caso_error",
    ["sin_metadatos_csv", "wav_faltante", "wav_44100_hz", "sin_ruidos_train"],
)
def test_generar_prepro_errores_de_entrada_y_copias_cero_sin_ruido(
    dataset_v1_sintetico: dict, tmp_path: Path, caso_error: str
) -> None:
    """Verifica ValueError ante entradas inválidas y que con copias=0 no hacen falta ruidos de train."""
    from src.voz.preprocesamiento import generar_prepro

    cfg: dict = copy.deepcopy(dataset_v1_sintetico["cfg"])
    origen_base: Path = dataset_v1_sintetico["origen"]
    origen_mod = tmp_path / f"origen_{caso_error}"
    shutil.copytree(origen_base, origen_mod)
    destino_mod = tmp_path / f"destino_{caso_error}"

    if caso_error == "sin_metadatos_csv":
        (origen_mod / "metadatos.csv").unlink()
        with pytest.raises(ValueError):
            generar_prepro(cfg, origen=origen_mod, destino=destino_mod)

    elif caso_error == "wav_faltante":
        (origen_mod / "train" / "nb_p01_identificar_01.wav").unlink()
        with pytest.raises(ValueError):
            generar_prepro(cfg, origen=origen_mod, destino=destino_mod)

    elif caso_error == "wav_44100_hz":
        wav_44k = _tono_armonico(1.0, fs=44100, f0=200.0, pico=0.5)
        sf.write(
            origen_mod / "train" / "nb_p01_identificar_01.wav",
            wav_44k,
            44100,
            subtype="PCM_16",
        )
        with pytest.raises(ValueError):
            generar_prepro(cfg, origen=origen_mod, destino=destino_mod)

    elif caso_error == "sin_ruidos_train":
        # Eliminamos solo los ruidos de train del CSV (dejando el de val)
        csv_path = origen_mod / "metadatos.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            filas_filtradas = [
                r
                for r in reader
                if not (r["particion"] == "train" and r["origen"] == "ruido")
            ]
        with csv_path.open("w", encoding="utf-8", newline="\n") as f:
            writer = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            writer.writeheader()
            writer.writerows(filas_filtradas)

        with pytest.raises(ValueError):
            generar_prepro(cfg, origen=origen_mod, destino=destino_mod)

        # Con copias = 0 no hacen falta ruidos de train
        cfg_sin_copias = copy.deepcopy(cfg)
        cfg_sin_copias["voz"]["preprocesamiento"]["aumento"]["copias"] = 0
        destino_sin_copias = tmp_path / "destino_copias_0"
        filas_k0 = generar_prepro(
            cfg_sin_copias, origen=origen_mod, destino=destino_sin_copias
        )
        assert len(filas_k0) == len(filas_filtradas) and all(
            r["copia"] == 0 for r in filas_k0
        ), "Con copias=0 debe generar todas las copias 0 sin exigir ruidos de train"


def test_generar_prepro_sobrescribir_protege_y_limpia_selectivamente(
    dataset_v1_sintetico: dict, tmp_path: Path
) -> None:
    """FileExistsError sin sobrescribir sin tocar nada; con sobrescribir=True preserva archivos ajenos."""
    from src.voz.preprocesamiento import generar_prepro

    cfg: dict = dataset_v1_sintetico["cfg"]
    origen: Path = dataset_v1_sintetico["origen"]

    # Caso 1: ya existe metadatos.csv y sobrescribir=False -> FileExistsError sin crear carpetas
    dest_protegido = tmp_path / "protegido_csv"
    dest_protegido.mkdir(parents=True)
    csv_previo = dest_protegido / "metadatos.csv"
    contenido_previo = b"archivo,particion\n"
    csv_previo.write_bytes(contenido_previo)

    with pytest.raises(FileExistsError):
        generar_prepro(cfg, origen=origen, destino=dest_protegido, sobrescribir=False)

    assert csv_previo.read_bytes() == contenido_previo, (
        "Con sobrescribir=False no debe modificarse el metadatos.csv existente"
    )
    assert not (dest_protegido / "train").exists(), (
        "Con sobrescribir=False no deben crearse carpetas si se aborta por FileExistsError"
    )

    # Caso 2: ya existe un WAV en train/ y sobrescribir=False -> FileExistsError
    dest_con_wav = tmp_path / "protegido_wav"
    (dest_con_wav / "train").mkdir(parents=True)
    wav_viejo = dest_con_wav / "train" / "viejo_k0.wav"
    wav_viejo.write_bytes(b"RIFF_VIEJO")

    with pytest.raises(FileExistsError):
        generar_prepro(cfg, origen=origen, destino=dest_con_wav, sobrescribir=False)
    assert wav_viejo.read_bytes() == b"RIFF_VIEJO", (
        "Con sobrescribir=False no debe tocarse el WAV existente"
    )

    # Caso 3: con sobrescribir=True borra WAV viejos de train/val/test y metadatos.csv,
    # pero conserva cualquier otro archivo ajeno.
    testigo_raiz = dest_con_wav / "nota_ajena.txt"
    testigo_train = dest_con_wav / "train" / "leeme.txt"
    testigo_raiz.write_text("conservar_raiz", encoding="utf-8")
    testigo_train.write_text("conservar_train", encoding="utf-8")

    generar_prepro(cfg, origen=origen, destino=dest_con_wav, sobrescribir=True)

    assert not wav_viejo.exists(), (
        "Con sobrescribir=True deben borrarse los WAV previos en train/, val/ o test/"
    )
    assert testigo_raiz.is_file() and testigo_raiz.read_text(encoding="utf-8") == "conservar_raiz", (
        "sobrescribir=True no debe borrar archivos que no sean WAV de train/val/test ni metadatos.csv"
    )
    assert testigo_train.is_file() and testigo_train.read_text(encoding="utf-8") == "conservar_train", (
        "sobrescribir=True no debe borrar archivos no-WAV dentro de train/"
    )


# =============================================================================
# 7.6 Aceptación (@pytest.mark.aceptacion, con los datos reales)
# =============================================================================


@pytest.mark.aceptacion
def test_aceptacion_dataset_preprocesado_real(cfg_real: dict, tmp_path: Path) -> None:
    """Genera el dataset preprocesado completo con los datos reales de V1 y verifica las métricas de 7.6."""
    from src.voz.preprocesamiento import generar_prepro

    origen_real = RAIZ / cfg_real["rutas"]["voz"] / "dataset"
    csv_v1_path = origen_real / "metadatos.csv"
    if not csv_v1_path.is_file():
        pytest.skip(f"No están los datos reales de V1 en {csv_v1_path}")

    with csv_v1_path.open("r", encoding="utf-8", newline="") as f:
        filas_v1 = list(csv.DictReader(f))

    destino_real = tmp_path / "prepro_aceptacion"
    t0 = time.perf_counter()
    filas_v2 = generar_prepro(cfg_real, origen=origen_real, destino=destino_real, sobrescribir=False)
    duracion_s = time.perf_counter() - t0
    assert duracion_s < 300.0, (
        f"generar_prepro tardó {duracion_s:.1f} s en el dataset real; el límite es < 300 s (5 min)"
    )

    por_fuente: dict[str, list[dict]] = {}
    for row in filas_v2:
        por_fuente.setdefault(row["fuente"], []).append(row)

    # 1. Cada clip de V1 tiene su copia 0, y los de train válidos que no son de ruido tienen sus 2 copias
    assert len(por_fuente) == len(filas_v1), (
        f"Cantidad de fuentes en V2 ({len(por_fuente)}) != clips en V1 ({len(filas_v1)})"
    )
    for v1_row in filas_v1:
        archivo_v1 = v1_row["archivo"]
        assert archivo_v1 in por_fuente, f"Falta el clip de V1 {archivo_v1} en la salida de V2"
        grupo = por_fuente[archivo_v1]
        k0 = grupo[0]
        assert k0["copia"] == 0
        if k0["particion"] == "train" and k0["origen"] != "ruido" and k0["valido"]:
            assert [r["copia"] for r in grupo] == [0, 1, 2], (
                f"El clip de train válido {archivo_v1} debía tener copias [0, 1, 2]; "
                f"tuvo {[r['copia'] for r in grupo]}"
            )
        else:
            assert [r["copia"] for r in grupo] == [0], (
                f"El clip {archivo_v1} solo debía tener copia [0]; tuvo {[r['copia'] for r in grupo]}"
            )

    # 2. Todos los clips de comandos son válidos en la copia 0, y en >= 97 % inicio_s y fin_s
    #    están a <= 0,011 s de los de V1.
    v1_por_archivo = {r["archivo"]: r for r in filas_v1}
    comandos_k0 = [
        r for r in filas_v2 if r["copia"] == 0 and r["clase"] != "otro"
    ]
    assert len(comandos_k0) > 0, "No se encontraron clips de comandos en la copia 0"
    comandos_invalidos = [r["fuente"] for r in comandos_k0 if not r["valido"]]
    assert not comandos_invalidos, (
        f"Todos los comandos deben ser válidos en la copia 0; fallaron {len(comandos_invalidos)}: "
        f"{comandos_invalidos[:5]}"
    )

    coincidencias_tiempos = 0
    for r in comandos_k0:
        v1_r = v1_por_archivo[r["fuente"]]
        ini_v1 = float(v1_r["palabra_inicio_s"])
        fin_v1 = float(v1_r["palabra_fin_s"])
        if (
            abs(r["palabra_inicio_s"] - ini_v1) <= 0.011
            and abs(r["palabra_fin_s"] - fin_v1) <= 0.011
        ):
            coincidencias_tiempos += 1

    prop_coincidencia = coincidencias_tiempos / len(comandos_k0)
    assert prop_coincidencia >= 0.97, (
        f"Solo el {prop_coincidencia * 100:.2f} % de los comandos tiene inicio y fin a <= 0,011 s "
        f"de V1; se exige >= 97 %"
    )

    # 3. Los no válidos de la copia 0 son todos de 'otro'
    no_validos_k0_no_otro = [
        r["fuente"] for r in filas_v2 if r["copia"] == 0 and not r["valido"] and r["clase"] != "otro"
    ]
    assert not no_validos_k0_no_otro, (
        f"Los no válidos de la copia 0 deben ser todos de clase 'otro'; se encontraron: {no_validos_k0_no_otro}"
    )

    # 4. Al menos el 90 % de las copias aumentadas son válidas, y al menos el 85 % de las de cada clase
    copias_aum = [r for r in filas_v2 if r["copia"] > 0]
    assert len(copias_aum) > 0, "No se generaron copias aumentadas"
    prop_aum_validas = sum(1 for r in copias_aum if r["valido"]) / len(copias_aum)
    assert prop_aum_validas >= 0.90, (
        f"Solo el {prop_aum_validas * 100:.2f} % de las copias aumentadas es válido; se exige >= 90 %"
    )

    clases_aum = sorted({r["clase"] for r in copias_aum})
    for c in clases_aum:
        de_clase = [r for r in copias_aum if r["clase"] == c]
        prop_clase = sum(1 for r in de_clase if r["valido"]) / len(de_clase)
        assert prop_clase >= 0.85, (
            f"En la clase '{c}', solo el {prop_clase * 100:.2f} % de las copias aumentadas es válido; "
            f"se exige >= 85 %"
        )

    # 5. Ninguna copia en val ni en test; todos los ruido son clips de ruido de train
    copias_val_test = [
        r["fuente"] for r in copias_aum if r["particion"] in ("val", "test")
    ]
    assert not copias_val_test, (
        f"No debe haber copias aumentadas en val ni en test; hubo {len(copias_val_test)}"
    )
    ruidos_train_reales = {
        r["archivo"]
        for r in filas_v1
        if r["particion"] == "train" and r["origen"] == "ruido"
    }
    ruidos_usados = {r["ruido"] for r in copias_aum}
    assert ruidos_usados <= ruidos_train_reales, (
        f"Se usaron ruidos que no son de train: {ruidos_usados - ruidos_train_reales}"
    )

    # 6. train tiene al menos 1500 filas válidas
    validas_train = sum(
        1 for r in filas_v2 if r["particion"] == "train" and r["valido"]
    )
    assert validas_train >= 1500, (
        f"train tiene {validas_train} filas válidas; se exigen al menos 1500"
    )

