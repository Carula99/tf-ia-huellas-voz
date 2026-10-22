"""Tests de la etapa V1 · Dataset de voz.

Verifican la sección 7 de docs/contratos/v1_dataset.md (versión 1.1, 22/10/2026). Salen solo de ese
contrato (y de la firma cargar_config(ruta) de la Fase 0, en la que se apoya) y se congelan antes de
implementar. La sección 8 es manual y no está acá.

Aclaraciones acordadas al escribirlos (22/10/2026; están en la sección 11 del contrato):
- Los clips de ruido se llaman ruido_{nombre}_c{k:02d}.wav con {nombre} = nombre del archivo sin
  extensión: de ruido_20261020_093611.wav sale ruido_ruido_20261020_093611_c01.wav.
- En la aceptación, el mínimo por comando cuenta solo los clips de la notebook de cada hablante.
- Reproducir un clip (7.7) se compara contra el audio remuestreado recortado a [−1, 1], porque
  guardar_wav recorta antes de escribir.
- "Dura a lo sumo clip_max_s" (7.4) se prueba con palabras más cortas que clip_max_s; la regla 3 de
  la sección 4 (palabra más larga que clip_max_s → el clip es la palabra) se prueba aparte.
- En WhatsApp, los casos de cortada y doble están armados para que den lo mismo midiendo la palabra
  con el corte o con la detección sobre el clip. larga se mide con la palabra del corte
  (Corte.inicio_s y fin_s), la "palabra sola" de la regla 3 de la sección 4: si dura más que
  clip_max_s, el clip es la palabra entera y ahí el detector no encuentra nada (el percentil 10 de
  los niveles queda dentro de la palabra), así que con la detección sobre el clip ese audio entraría
  con un clip más largo que clip_max_s.
- Por lo mismo, la palabra de 2,8 s de la toma que provoca duracion tiene cierres internos de 0,1 s,
  como el habla: una palabra pareja que ocupa más del 90 % de la toma da sin_palabra.
- No se fijan: el código en TTS y en el ruido, la forma de "vacío" en lo que devuelve inventario
  ("" o None) ni el texto de nota en los motivos automáticos.
- Un código sin ningún audio puesto en val o en test no se prueba (el contrato admite códigos sin
  audios en las listas y a la vez prohíbe hablantes sin sesiones de la notebook en val y test).

Correr desde la raíz del repo:
    venv\\Scripts\\python -m pytest tests/test_v1.py                  (todo menos la aceptación)
    venv\\Scripts\\python -m pytest tests/test_v1.py -m aceptacion    (7.8, con los datos reales)
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import io
import math
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml
from scipy.signal import resample_poly

RAIZ = Path(__file__).resolve().parents[1]
FS = 16000  # voz.frecuencia_muestreo (fija en la Fase 0)
TOL_PCM = 1 / 32767

COLUMNAS_CONTRATO = (
    "archivo", "particion", "origen", "codigo", "hablante", "clase", "detalle",
    "estilo", "distancia_cm", "toma", "velocidad_tts", "fuente", "muestra_inicio", "muestra_fin",
    "fs_fuente", "duracion_s", "n_segmentos", "palabra_inicio_s", "palabra_fin_s", "pico", "rms_db",
)
COLUMNAS_EXCLUIDAS_CONTRATO = ("fuente", "toma", "origen", "codigo", "clase", "motivo", "nota")
ORIGENES_CONTRATO = ("notebook", "whatsapp", "tts", "ruido")
MOTIVOS_CONTRATO = ("whatsapp_val_test", "notas", "saturada", "sin_palabra",
                    "varias_palabras", "cortada", "duracion", "doble", "larga")
ENTERAS = ("distancia_cm", "toma", "muestra_inicio", "muestra_fin", "fs_fuente", "n_segmentos")
DECIMALES = {"duracion_s": 4, "pico": 4, "palabra_inicio_s": 3, "palabra_fin_s": 3, "rms_db": 2}


# --------------------------------------------------------------------------------------------
# Módulos del proyecto (se importan en fixtures: si algo no existe, fallan esos tests y no la
# colección entera)
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    from src.config import cargar_config
    return cargar_config()


@pytest.fixture(scope="module")
def audio():
    return importlib.import_module("src.voz.audio")


@pytest.fixture(scope="module")
def seg():
    return importlib.import_module("src.voz.segmentacion")


@pytest.fixture(scope="module")
def tts():
    return importlib.import_module("src.voz.tts")


@pytest.fixture(scope="module")
def ds():
    return importlib.import_module("src.voz.dataset")


# --------------------------------------------------------------------------------------------
# Ayudas: señales sintéticas y archivos
# --------------------------------------------------------------------------------------------

def _tono(fs, dur, pico=0.3, f0=150.0, rampa_s=0.02):
    """'Palabra' sintética: tono con armónicos (f0 … 4·f0), rampas lineales de 20 ms y pico dado."""
    n = int(round(dur * fs))
    t = np.arange(n) / fs
    y = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t + 0.4)
         + 0.3 * np.sin(2 * np.pi * 3 * f0 * t + 1.1) + 0.2 * np.sin(2 * np.pi * 4 * f0 * t + 2.0))
    nr = min(int(round(rampa_s * fs)), n // 2)
    if nr > 0:
        r = np.arange(nr) / nr
        y[:nr] *= r
        y[n - nr:] *= r[::-1]
    return pico * y / np.max(np.abs(y))


def _ruido(fs, dur, nivel_db, semilla):
    """Ruido gaussiano con RMS de nivel_db dBFS."""
    rng = np.random.default_rng(semilla)
    return rng.standard_normal(int(round(dur * fs))) * 10 ** (nivel_db / 20)


def _rms_db(y):
    y = np.asarray(y, dtype=np.float64)
    return 20 * math.log10(max(math.sqrt(float(np.mean(y * y))) if len(y) else 0.0, 1e-6))


def _nivel_tono(pico, f0=150.0):
    """Nivel (dBFS) de la parte estable de _tono con ese pico."""
    return _rms_db(_tono(FS, 1.0, pico, f0, rampa_s=0.0))


def _poner(base, senal, inicio_s, fs):
    i = int(round(inicio_s * fs))
    senal = senal[: max(0, len(base) - i)]
    base[i:i + len(senal)] += senal
    return base


def _audio(fs, total_s, palabras, ruido_db=None, semilla=0, f0=150.0):
    """palabras: lista de (inicio_s, dur_s, pico). Devuelve float32 en [−1, 1]."""
    y = np.zeros(int(round(total_s * fs)))
    if ruido_db is not None:
        y += _ruido(fs, total_s, ruido_db, semilla)
    for ini, dur, pico in palabras:
        _poner(y, _tono(fs, dur, pico, f0), ini, fs)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _escribir_m4a(ruta, y, fs):
    import av  # PyAV: se suma en V1 (sección 10 del contrato)

    datos = np.ascontiguousarray(np.asarray(y, dtype=np.float32).reshape(1, -1))
    with av.open(str(ruta), "w") as contenedor:
        flujo = contenedor.add_stream("aac", rate=fs)
        flujo.layout = "mono"
        cuadro = av.AudioFrame.from_ndarray(datos, format="fltp", layout="mono")
        cuadro.sample_rate = fs
        cuadro.pts = 0
        for paquete in flujo.encode(cuadro):
            contenedor.mux(paquete)
        for paquete in flujo.encode(None):
            contenedor.mux(paquete)


def _escribir(ruta, y, fs):
    """WAV PCM_16, OGG/Opus (.ogg u .opus) y MP3 con soundfile; M4A (AAC) con PyAV."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ext = ruta.suffix.lower()
    if ext == ".wav":
        sf.write(str(ruta), y, fs, subtype="PCM_16")
    elif ext in (".ogg", ".opus"):
        sf.write(str(ruta), y, fs, format="OGG", subtype="OPUS")
    elif ext == ".mp3":
        sf.write(str(ruta), y, fs, format="MP3", subtype="MPEG_LAYER_III")
    elif ext == ".m4a":
        _escribir_m4a(ruta, y, fs)
    else:
        raise AssertionError(f"extensión no prevista en los tests: {ext}")


def _frecuencia_dominante(y, fs):
    y = np.asarray(y, dtype=np.float64)
    espectro = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    k = int(np.argmax(espectro))
    p = 0.0
    if 0 < k < len(espectro) - 1:
        a, b, c = np.log(espectro[k - 1:k + 2] + 1e-30)
        p = 0.5 * (a - c) / (a - 2 * b + c)
    return (k + p) * fs / len(y)


def _huella(carpeta):
    """{ruta relativa: sha256} de archivos y {ruta relativa: None} de carpetas."""
    carpeta = Path(carpeta)
    out = {}
    for p in sorted(carpeta.rglob("*")):
        rel = p.relative_to(carpeta).as_posix()
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out


def _vacio(v):
    return v is None or v == ""


def _txt(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _clave(c):
    return (c["origen"], c["fuente"], int(c["toma"]))


def _hablante(codigo, grupos):
    for grupo in grupos:
        if codigo in grupo:
            return grupo[0]
    return codigo


def _particion_de(codigo, particiones):
    return [p for p in ("train", "val", "test") if codigo in particiones[p]]


def _leer_csv(ruta):
    """(encabezado, filas como dicts de texto). Exige UTF-8 sin BOM."""
    datos = Path(ruta).read_bytes()
    assert not datos.startswith(b"\xef\xbb\xbf"), f"{Path(ruta).name} empieza con BOM"
    texto = datos.decode("utf-8")
    filas = list(csv.reader(io.StringIO(texto, newline="")))
    assert filas, f"{Path(ruta).name} está vacío"
    encabezado = filas[0]
    cuerpo = []
    for i, f in enumerate(filas[1:], start=2):
        assert len(f) == len(encabezado), f"{Path(ruta).name}, línea {i}: {len(f)} campos, se esperaban {len(encabezado)}"
        cuerpo.append(dict(zip(encabezado, f)))
    return encabezado, cuerpo


def _tipar(fila):
    out = {}
    for k, v in fila.items():
        if v == "":
            out[k] = None
        elif k in ENTERAS:
            out[k] = int(v)
        elif k in DECIMALES:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _filas_iguales(a, b):
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if abs(x - y) > 1e-9:
                return False
        elif x != y:
            return False
    return True


def _leer_texto_tolerante(ruta):
    datos = Path(ruta).read_bytes()
    try:
        return datos.decode("utf-8-sig")
    except UnicodeDecodeError:
        return datos.decode("cp1252")


# --------------------------------------------------------------------------------------------
# 7.1 Configuración
# --------------------------------------------------------------------------------------------

HOJAS_V1 = [
    "voz.dataset.clip_max_s",
    "voz.dataset.saturacion",
    "voz.dataset.saturacion_frac",
    "voz.dataset.mismo_hablante",
    "voz.dataset.particiones.train",
    "voz.dataset.particiones.val",
    "voz.dataset.particiones.test",
    "voz.dataset.min_por_comando",
    "voz.dataset.deteccion.trama_ms",
    "voz.dataset.deteccion.avance_ms",
    "voz.dataset.deteccion.rango_db",
    "voz.dataset.deteccion.margen_piso_db",
    "voz.dataset.deteccion.dif_palabra_db",
    "voz.dataset.deteccion.nivel_min_db",
    "voz.dataset.deteccion.pausa_min_s",
    "voz.dataset.deteccion.segmento_min_s",
    "voz.dataset.deteccion.margen_borde_s",
    "voz.dataset.whatsapp.margen_s",
    "voz.dataset.whatsapp.factor_doble",
    "voz.dataset.ruido.clips_por_archivo",
    "voz.dataset.ruido.particiones.train",
    "voz.dataset.ruido.particiones.val",
    "voz.dataset.ruido.particiones.test",
    "voz.dataset.tts.voces",
    "voz.dataset.tts.velocidades",
    "voz.dataset.tts.otros",
]

RE_CODIGO = re.compile(r"^p(0[1-9]|[1-9][0-9])$")
RE_VELOCIDAD = re.compile(r"^[+-][0-9]+%$")


def _es_entero(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _es_real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _real_en(lo, hi, abierto_abajo=False):
    def chequeo(v):
        return _es_real(v) and (lo < v if abierto_abajo else lo <= v) and v <= hi
    return chequeo


def _lista_de_textos(v):
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _lista_de_codigos(v):
    return _lista_de_textos(v) and all(RE_CODIGO.match(x) for x in v)


REGLAS_V1 = {
    "voz.dataset.clip_max_s": ("igual a 3.0", lambda v: _es_real(v) and v == 3.0),
    "voz.dataset.saturacion": ("igual a 0.98", lambda v: _es_real(v) and v == 0.98),
    "voz.dataset.saturacion_frac": ("número en (0, 0.05]", _real_en(0, 0.05, abierto_abajo=True)),
    "voz.dataset.mismo_hablante": ("[[p01, p02], [p09, p13], [p14, p17]]",
                                   lambda v: v == [["p01", "p02"], ["p09", "p13"], ["p14", "p17"]]),
    "voz.dataset.particiones.train": ("lista de códigos p01 … p99", _lista_de_codigos),
    "voz.dataset.particiones.val": ("lista de códigos p01 … p99", _lista_de_codigos),
    "voz.dataset.particiones.test": ("lista de códigos p01 … p99", _lista_de_codigos),
    "voz.dataset.min_por_comando": ("entero ≥ 1", lambda v: _es_entero(v) and v >= 1),
    "voz.dataset.deteccion.trama_ms": ("igual a 20", lambda v: _es_real(v) and v == 20),
    "voz.dataset.deteccion.avance_ms": ("igual a 10", lambda v: _es_real(v) and v == 10),
    "voz.dataset.deteccion.rango_db": ("número en [20, 40]", _real_en(20, 40)),
    "voz.dataset.deteccion.margen_piso_db": ("número en [3, 15]", _real_en(3, 15)),
    "voz.dataset.deteccion.dif_palabra_db": ("número en [6, 20]", _real_en(6, 20)),
    "voz.dataset.deteccion.nivel_min_db": ("número en [-70, -40]", _real_en(-70, -40)),
    "voz.dataset.deteccion.pausa_min_s": ("número en [0.15, 0.5]", _real_en(0.15, 0.5)),
    "voz.dataset.deteccion.segmento_min_s": ("número en [0.05, 0.2]", _real_en(0.05, 0.2)),
    "voz.dataset.deteccion.margen_borde_s": ("número en [0, 0.1]", _real_en(0, 0.1)),
    "voz.dataset.whatsapp.margen_s": ("número en [0.1, 0.5]", _real_en(0.1, 0.5)),
    "voz.dataset.whatsapp.factor_doble": ("número en [1.3, 2.5]", _real_en(1.3, 2.5)),
    "voz.dataset.ruido.clips_por_archivo": ("entero ≥ 1", lambda v: _es_entero(v) and v >= 1),
    "voz.dataset.ruido.particiones.train": ("lista de nombres", _lista_de_textos),
    "voz.dataset.ruido.particiones.val": ("lista de nombres", _lista_de_textos),
    "voz.dataset.ruido.particiones.test": ("lista de nombres", _lista_de_textos),
    "voz.dataset.tts.voces": ("lista de textos sin repetir",
                              lambda v: _lista_de_textos(v) and len(set(v)) == len(v)),
    "voz.dataset.tts.velocidades": ('lista de textos como "-10%" o "+10%"',
                                    lambda v: _lista_de_textos(v) and all(RE_VELOCIDAD.match(x) for x in v)),
    "voz.dataset.tts.otros": ("lista de textos", _lista_de_textos),
}


def _valor(datos, camino):
    v = datos
    for parte in camino.split("."):
        v = v[parte]
    return v


def test_reglas_cubren_todas_las_hojas_nuevas():
    assert set(REGLAS_V1) == set(HOJAS_V1)


@pytest.mark.parametrize("camino", HOJAS_V1)
def test_config_clave_nueva_con_tipo_y_rango(cfg, camino):
    descripcion, chequeo = REGLAS_V1[camino]
    try:
        v = _valor(cfg, camino)
    except (KeyError, TypeError):
        pytest.fail(f"falta {camino} en config.yaml")
    assert chequeo(v), f"{camino} = {v!r}; se esperaba {descripcion}"


def test_config_clip_max_igual_a_push_to_talk(cfg):
    assert cfg["voz"]["dataset"]["clip_max_s"] == cfg["voz"]["push_to_talk_max_s"], (
        f"voz.dataset.clip_max_s = {cfg['voz']['dataset']['clip_max_s']}, "
        f"voz.push_to_talk_max_s = {cfg['voz']['push_to_talk_max_s']}")


def _config_yaml_real():
    return yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("camino", HOJAS_V1)
def test_cargar_config_sin_la_clave_lanza_valueerror_con_su_camino(tmp_path, camino):
    from src.config import cargar_config

    datos = _config_yaml_real()
    partes = camino.split(".")
    padre = _valor(datos, ".".join(partes[:-1]))
    assert partes[-1] in padre, f"{camino} no está en config.yaml"
    del padre[partes[-1]]
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        cargar_config(ruta)
    assert camino in str(error.value), f"el mensaje no nombra {camino}: {error.value}"


def test_cargar_config_sin_voz_dataset_nombra_todas_sus_hojas(tmp_path):
    from src.config import cargar_config

    datos = _config_yaml_real()
    del datos["voz"]["dataset"]
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        cargar_config(ruta)
    faltan = [c for c in HOJAS_V1 if c not in str(error.value)]
    assert not faltan, f"el mensaje no nombra {faltan}: {error.value}"


def test_config_real_particiones_sin_codigos_repetidos_ni_en_comun(cfg):
    p = cfg["voz"]["dataset"]["particiones"]
    for nombre in ("train", "val", "test"):
        repetidos = sorted(c for c, n in Counter(p[nombre]).items() if n > 1)
        assert not repetidos, f"códigos repetidos en particiones.{nombre}: {repetidos}"
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        comunes = sorted(set(p[a]) & set(p[b]))
        assert not comunes, f"códigos en {a} y en {b}: {comunes}"


def test_config_real_grupos_de_mismo_hablante_y_p08_p15_en_train(cfg):
    d = cfg["voz"]["dataset"]
    deben = [c for grupo in d["mismo_hablante"] for c in grupo] + ["p08", "p15"]
    fuera = [c for c in deben if c not in d["particiones"]["train"]]
    assert not fuera, f"deberían estar en train y no están: {fuera}"


def test_config_real_reparto_10_4_4_de_los_hablantes_de_la_notebook(cfg):
    d = cfg["voz"]["dataset"]
    notebook = [f"p{i:02d}" for i in range(1, 22)]
    sin_lista = [c for c in notebook if not any(c in d["particiones"][p] for p in ("train", "val", "test"))]
    assert not sin_lista, f"códigos de la notebook (p01 a p21) sin partición: {sin_lista}"
    conteo = {}
    for p in ("train", "val", "test"):
        conteo[p] = len({_hablante(c, d["mismo_hablante"]) for c in d["particiones"][p] if c in notebook})
    assert conteo == {"train": 10, "val": 4, "test": 4}, f"hablantes de la notebook por partición: {conteo}"


def test_config_real_codigos_desde_p22_solo_en_train(cfg):
    d = cfg["voz"]["dataset"]
    malos = [(p, c) for p in ("val", "test") for c in d["particiones"][p] if int(c[1:]) >= 22]
    assert not malos, f"códigos solo de WhatsApp fuera de train: {malos}"


def test_config_real_ruido_particiones_disjuntas_y_no_vacias(cfg):
    r = cfg["voz"]["dataset"]["ruido"]["particiones"]
    vacias = [p for p in ("train", "val", "test") if not r[p]]
    assert not vacias, f"ruido.particiones vacías: {vacias}"
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        comunes = sorted(set(r[a]) & set(r[b]))
        assert not comunes, f"grabaciones de ruido en {a} y en {b}: {comunes}"


def test_config_real_tts_otros_sin_comandos(cfg):
    otros = cfg["voz"]["dataset"]["tts"]["otros"]
    comandos = [w for w in otros if w in cfg["voz"]["comandos"]]
    assert not comandos, f"tts.otros tiene comandos: {comandos}"


def test_config_real_plan_tts_de_300_entradas_o_menos(cfg, tts):
    n = len(tts.plan_tts(copy.deepcopy(cfg)))
    assert n <= 300, f"el plan de TTS tiene {n} entradas; el máximo es 300"


# --------------------------------------------------------------------------------------------
# 7.2 Audio
# --------------------------------------------------------------------------------------------

FORMATOS_LECTURA = [
    ("wav", 44100), ("wav", 16000), ("ogg", 48000), ("ogg", 16000),
    ("opus", 48000), ("opus", 16000), ("mp3", 24000), ("m4a", 48000),
]


@pytest.mark.parametrize("ext,fs", FORMATOS_LECTURA, ids=[f"{e}_{f}" for e, f in FORMATOS_LECTURA])
def test_leer_audio_formato_da_float32_mono_y_la_frecuencia_del_archivo(audio, tmp_path, ext, fs):
    dur = 1.0
    t = np.arange(int(dur * fs)) / fs
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ruta = tmp_path / f"seno.{ext}"
    _escribir(ruta, y, fs)
    z, fz = audio.leer_audio(ruta)
    assert fz == fs, f"frecuencia leída {fz}, se esperaba {fs}"
    assert isinstance(z, np.ndarray) and z.dtype == np.float32, f"tipo {type(z).__name__} {getattr(z, 'dtype', '')}"
    assert z.ndim == 1, f"{z.ndim} dimensiones, se esperaba 1"
    assert np.max(np.abs(z)) <= 1.0, f"valores fuera de [−1, 1]: máx |y| = {np.max(np.abs(z))}"
    if ext == "wav":
        assert len(z) == len(y), f"{len(z)} muestras, se esperaban {len(y)}"
        assert np.max(np.abs(z - y)) <= 1e-4
    else:
        assert abs(len(z) / fz - dur) <= 0.05, f"duración {len(z) / fz:.3f} s, se esperaba {dur} ± 0,05 s"
    f_pico = _frecuencia_dominante(z, fz)
    assert abs(f_pico - 440) <= 2, f"frecuencia del tono leído {f_pico:.1f} Hz, se esperaba 440 Hz"


def test_leer_audio_wav_estereo_da_el_promedio_de_los_canales(audio, tmp_path):
    fs = 44100
    t = np.arange(fs) / fs
    izq = 0.4 * np.sin(2 * np.pi * 440 * t)
    der = 0.2 * np.sin(2 * np.pi * 660 * t + 0.3)
    ruta = tmp_path / "estereo.wav"
    sf.write(str(ruta), np.stack([izq, der], axis=1), fs, subtype="PCM_16")
    z, fz = audio.leer_audio(ruta)
    assert fz == fs and z.ndim == 1 and z.dtype == np.float32
    err = float(np.max(np.abs(z - (izq + der) / 2)))
    assert err <= 1e-4, f"diferencia con el promedio de los canales: {err}"


@pytest.mark.parametrize("fs", [44100, 48000])
def test_leer_audio_m4a_misma_frecuencia_y_duracion(audio, tmp_path, fs):
    dur = 1.5
    t = np.arange(int(dur * fs)) / fs
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ruta = tmp_path / "seno.m4a"
    _escribir_m4a(ruta, y, fs)
    z, fz = audio.leer_audio(ruta)
    assert fz == fs, f"frecuencia leída {fz}, se esperaba {fs}"
    assert z.dtype == np.float32 and z.ndim == 1
    assert abs(len(z) / fz - dur) <= 0.05, f"duración {len(z) / fz:.3f} s, se esperaba {dur} ± 0,05 s"


FACTORES = {44100: (160, 441), 48000: (1, 3), 24000: (2, 3)}


@pytest.mark.parametrize("fs_origen", [44100, 48000, 24000])
def test_remuestrear_3_s_dan_48000_muestras(audio, fs_origen):
    y = (0.3 * np.random.default_rng(1).standard_normal(3 * fs_origen)).clip(-1, 1).astype(np.float32)
    z = audio.remuestrear(y, fs_origen, FS)
    assert z.dtype == np.float32, f"tipo {z.dtype}, se esperaba float32"
    assert len(z) == 48000, f"{len(z)} muestras, se esperaban 48000"


@pytest.mark.parametrize("fs_origen", [44100, 48000, 24000])
def test_remuestrear_coincide_con_resample_poly(audio, fs_origen):
    g = math.gcd(fs_origen, FS)
    up, down = FACTORES[fs_origen]
    assert (FS // g, fs_origen // g) == (up, down)
    y = (0.3 * np.random.default_rng(fs_origen).standard_normal(fs_origen)).clip(-1, 1).astype(np.float32)
    esperado = resample_poly(y.astype(np.float64), up, down).astype(np.float32)
    z = audio.remuestrear(y, fs_origen, FS)
    assert z.shape == esperado.shape, f"forma {z.shape}, se esperaba {esperado.shape}"
    err = float(np.max(np.abs(z.astype(np.float64) - esperado)))
    assert err <= 1e-6, f"diferencia con resample_poly(y, {up}, {down}): {err}"


@pytest.mark.parametrize("fs_origen", [44100, 48000, 24000])
def test_remuestrear_conserva_frecuencia_y_amplitud_de_un_seno(audio, fs_origen):
    t = np.arange(3 * fs_origen) / fs_origen
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    z = audio.remuestrear(y, fs_origen, FS)
    f_pico = _frecuencia_dominante(z, FS)
    assert abs(f_pico - 440) <= 2, f"pico de la FFT en {f_pico:.2f} Hz, se esperaba 440 ± 2 Hz"
    medio = z[FS // 4: -FS // 4].astype(np.float64)
    amplitud = math.sqrt(2 * float(np.mean(medio * medio)))
    assert abs(amplitud - 0.5) <= 0.02 * 0.5, f"amplitud {amplitud:.4f}, se esperaba 0,5 ± 2 %"


def test_remuestrear_misma_frecuencia_devuelve_una_copia_igual(audio):
    y = (0.3 * np.random.default_rng(2).standard_normal(FS)).clip(-1, 1).astype(np.float32)
    original = y.copy()
    z = audio.remuestrear(y, FS, FS)
    assert z.dtype == np.float32
    assert np.array_equal(z, y), "con frecuencias iguales el audio debe quedar igual"
    assert not np.shares_memory(z, y), "con frecuencias iguales debe devolver una copia, no el mismo arreglo"
    z[:10] += 0.5
    assert np.array_equal(y, original), "modificar el resultado cambió el audio de entrada"


def test_guardar_wav_se_lee_como_16_khz_mono_pcm16_con_error_minimo(audio, tmp_path):
    y = (0.3 * np.random.default_rng(3).standard_normal(FS)).clip(-0.95, 0.95).astype(np.float32)
    ruta = tmp_path / "clip.wav"
    audio.guardar_wav(ruta, y, FS)
    info = sf.info(str(ruta))
    assert (info.samplerate, info.channels, info.subtype) == (FS, 1, "PCM_16"), (
        f"se leyó {info.samplerate} Hz, {info.channels} canales, {info.subtype}")
    z, _ = sf.read(str(ruta), dtype="float32")
    err = float(np.max(np.abs(z.astype(np.float64) - y)))
    assert err <= TOL_PCM, f"diferencia máxima {err:.2e}, se esperaba ≤ 1/32767"


def test_guardar_wav_recorta_lo_que_pasa_de_1(audio, tmp_path):
    y = np.array([0.0, 1.5, -1.5, 2.0, -3.0, 0.5, 1.0, -1.0] * 100, dtype=np.float32)
    ruta = tmp_path / "recorte.wav"
    audio.guardar_wav(ruta, y, FS)
    z, _ = sf.read(str(ruta), dtype="float32")
    err = float(np.max(np.abs(z.astype(np.float64) - np.clip(y, -1, 1))))
    assert err <= TOL_PCM, f"lo que pasa de ±1 no quedó recortado a ±1 (diferencia {err:.2e})"


def test_guardar_wav_crea_las_carpetas(audio, tmp_path):
    ruta = tmp_path / "a" / "b" / "c" / "clip.wav"
    audio.guardar_wav(ruta, np.zeros(FS // 10, dtype=np.float32), FS)
    assert ruta.is_file(), f"no se creó {ruta}"


# --------------------------------------------------------------------------------------------
# 7.3 Detector
# --------------------------------------------------------------------------------------------

def _det(cfg):
    return cfg["voz"]["dataset"]["deteccion"]


@pytest.mark.parametrize("n,fs,trama,avance,k", [
    (0, 16000, 20, 10, 0),
    (319, 16000, 20, 10, 0),
    (320, 16000, 20, 10, 1),
    (479, 16000, 20, 10, 1),
    (480, 16000, 20, 10, 2),
    (48000, 16000, 20, 10, 299),
    (8000, 8000, 25, 10, 98),
])
def test_envolvente_cantidad_de_tramas(seg, n, fs, trama, avance, k):
    y = (0.1 * np.random.default_rng(n).standard_normal(n)).astype(np.float32)
    niveles = seg.envolvente_db(y, fs, trama, avance)
    assert len(niveles) == k, f"{len(niveles)} tramas para {n} muestras a {fs} Hz; se esperaban {k}"


def test_envolvente_niveles_segun_la_formula(seg):
    rng = np.random.default_rng(4)
    y = (rng.standard_normal(4000) * np.linspace(0, 0.5, 4000)).astype(np.float32)
    y[1000:1700] = 0.0
    niveles = seg.envolvente_db(y, FS, 20, 10)
    assert np.asarray(niveles).dtype == np.float64, f"tipo {np.asarray(niveles).dtype}, se esperaba float64"
    N, H = 320, 160
    K = 1 + (len(y) - N) // H
    y64 = y.astype(np.float64)
    esperado = [20 * math.log10(max(math.sqrt(float(np.mean(y64[k * H:k * H + N] ** 2))), 1e-6)) for k in range(K)]
    err = float(np.max(np.abs(np.asarray(niveles) - esperado)))
    assert err <= 1e-6, f"diferencia con 20·log10(máx(rms, 1e-6)): {err} dB"


@pytest.mark.parametrize("A", [0.01, 0.1, 0.5, 0.9])
def test_envolvente_seno_da_20log10_A_sobre_raiz_de_2(seg, A):
    t = np.arange(FS) / FS
    y = (A * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    niveles = np.asarray(seg.envolvente_db(y, FS, 20, 10))
    esperado = 20 * math.log10(A / math.sqrt(2))
    err = float(np.max(np.abs(niveles - esperado)))
    assert err <= 0.1, f"niveles a {err:.3f} dB de {esperado:.2f} dB"


def test_envolvente_ceros_dan_menos_120(seg):
    niveles = np.asarray(seg.envolvente_db(np.zeros(FS, dtype=np.float32), FS, 20, 10))
    assert len(niveles) > 0 and np.allclose(niveles, -120.0), f"niveles {np.unique(niveles)}; se esperaba −120"


@pytest.mark.parametrize("caso", ["silencio", "ruido_-60_dBFS", "ruido_-35_dBFS"])
def test_detector_sin_palabras(seg, cfg, caso):
    if caso == "silencio":
        y = np.zeros(3 * FS, dtype=np.float32)
    else:
        nivel = float(caso.split("_")[1])
        y = _ruido(FS, 3.0, nivel, 5).astype(np.float32)
    segmentos = seg.detectar_palabras(y, FS, cfg)
    assert list(segmentos) == [], f"{caso}: se detectaron {segmentos}, no debería haber palabras"


def _chequear_bordes(segmentos, reales, tol=0.03):
    assert len(segmentos) == len(reales), f"{len(segmentos)} segmentos {segmentos}; se esperaban {len(reales)}: {reales}"
    for (i, f), (ri, rf) in zip(segmentos, reales):
        assert abs(i - ri) <= tol and abs(f - rf) <= tol, (
            f"segmento ({i:.3f}, {f:.3f}) s; el real es ({ri:.3f}, {rf:.3f}) s (tolerancia {tol * 1000:.0f} ms)")


@pytest.mark.parametrize("dur", [0.3, 0.8, 1.5])
@pytest.mark.parametrize("fondo", ["ceros", "ruido_25_dB_abajo", "ruido_40_dB_abajo"])
@pytest.mark.parametrize("pico", [0.02, 0.1, 0.5, 0.9])
def test_detector_una_palabra_da_un_segmento_con_bordes_a_30_ms(seg, cfg, pico, fondo, dur):
    ini = 0.9
    ruido_db = None if fondo == "ceros" else _nivel_tono(pico) - float(fondo.split("_")[1])
    y = _audio(FS, 3.0, [(ini, dur, pico)], ruido_db=ruido_db, semilla=11)
    segmentos = seg.detectar_palabras(y, FS, cfg)
    _chequear_bordes(segmentos, [(ini, ini + dur)])


def test_detector_devuelve_segmentos_en_orden_sin_solaparse(seg, cfg):
    y = _audio(FS, 6.0, [(0.5, 0.6, 0.3), (1.9, 0.5, 0.4), (3.2, 0.9, 0.3), (5.0, 0.5, 0.35)], ruido_db=-60, semilla=12)
    segmentos = [tuple(map(float, s)) for s in seg.detectar_palabras(y, FS, cfg)]
    assert len(segmentos) == 4, f"{len(segmentos)} segmentos: {segmentos}"
    for (i0, f0), (i1, f1) in zip(segmentos, segmentos[1:]):
        assert i0 < f0 <= i1 < f1, f"segmentos desordenados o solapados: {(i0, f0)} y {(i1, f1)}"


@pytest.mark.parametrize("delta,esperados", [(-0.08, 1), (0.08, 2)], ids=["pausa_min_menos_80ms", "pausa_min_mas_80ms"])
def test_detector_dos_palabras_segun_la_pausa(seg, cfg, delta, esperados):
    pausa = _det(cfg)["pausa_min_s"] + delta
    ini1, dur = 0.6, 0.5
    ini2 = ini1 + dur + pausa
    y = _audio(FS, 3.0, [(ini1, dur, 0.3), (ini2, dur, 0.3)])
    segmentos = seg.detectar_palabras(y, FS, cfg)
    if esperados == 1:
        _chequear_bordes(segmentos, [(ini1, ini2 + dur)])
    else:
        _chequear_bordes(segmentos, [(ini1, ini1 + dur), (ini2, ini2 + dur)])


def test_detector_cierre_interno_de_80_ms_no_parte_la_palabra(seg, cfg):
    y = _audio(FS, 3.0, [(0.8, 0.45, 0.3), (0.8 + 0.45 + 0.08, 0.45, 0.3)], ruido_db=-70, semilla=13)
    segmentos = seg.detectar_palabras(y, FS, cfg)
    _chequear_bordes(segmentos, [(0.8, 0.8 + 0.45 + 0.08 + 0.45)])


def _golpe(fs, pico=0.5, dur=0.05, semilla=14):
    g = np.random.default_rng(semilla).standard_normal(int(round(dur * fs)))
    nr = int(round(0.002 * fs))
    g[:nr] *= np.arange(nr) / nr
    g[-nr:] *= np.arange(nr)[::-1] / nr
    return pico * g / np.max(np.abs(g))


@pytest.mark.parametrize("con_palabra", [False, True], ids=["golpe_solo", "golpe_y_palabra"])
def test_detector_golpe_aislado_de_50_ms_no_da_segmento(seg, cfg, con_palabra):
    y = np.zeros(3 * FS)
    reales = []
    if con_palabra:
        _poner(y, _tono(FS, 0.8, 0.3), 0.4, FS)
        reales = [(0.4, 1.2)]
    _poner(y, _golpe(FS), 1.2 + _det(cfg)["pausa_min_s"] + 0.5, FS)
    segmentos = seg.detectar_palabras(y.astype(np.float32), FS, cfg)
    _chequear_bordes(segmentos, reales)


@pytest.mark.parametrize("baja_db,esperados", [(6, 2), (15, 1)])
def test_detector_dominancia_de_la_palabra_mas_fuerte(seg, cfg, baja_db, esperados):
    pico2 = 0.5 * 10 ** (-baja_db / 20)
    y = _audio(FS, 3.0, [(0.4, 0.8, 0.5), (1.8, 0.8, pico2)])
    segmentos = seg.detectar_palabras(y, FS, cfg)
    reales = [(0.4, 1.2), (1.8, 2.6)][:esperados]
    _chequear_bordes(segmentos, reales)


@pytest.mark.parametrize("factor", [0.1, 3.0])
def test_detector_escala_no_cambia_los_segmentos(seg, cfg, factor):
    base = np.zeros(3 * FS)
    _poner(base, _tono(FS, 0.9, 0.3), 0.7, FS)
    base += _ruido(FS, 3.0, _nivel_tono(0.3) - 30, 15)
    s1 = [tuple(map(float, s)) for s in seg.detectar_palabras(base.astype(np.float32), FS, cfg)]
    s2 = [tuple(map(float, s)) for s in seg.detectar_palabras((base * factor).astype(np.float32), FS, cfg)]
    assert len(s1) == 1, f"la palabra de referencia dio {s1}"
    assert len(s2) == len(s1) and all(abs(a - b) <= 1e-9 for x, z in zip(s1, s2) for a, b in zip(x, z)), (
        f"por {factor}: {s2}; sin escalar: {s1}")


@pytest.mark.parametrize("fs", [16000, 48000])
def test_detector_audio_tipo_whatsapp_en_opus(seg, cfg, tmp_path, fs):
    rng = np.random.default_rng(16)
    durs = rng.uniform(0.45, 1.0, 10)
    pausas = rng.uniform(0.6, 1.2, 9)
    bajas = rng.uniform(0, 6, 10)
    bajas[0], bajas[1] = 0.0, 6.0
    f0s = rng.uniform(110, 230, 10)
    total = 0.8 + durs.sum() + pausas.sum() + 0.8
    y = _ruido(fs, total, -45.0, 17)
    reales = []
    t = 0.8
    for i in range(10):
        _poner(y, _tono(fs, durs[i], 0.5 * 10 ** (-bajas[i] / 20), f0s[i]), t, fs)
        reales.append((t, t + durs[i]))
        t += durs[i] + (pausas[i] if i < 9 else 0)
    ruta = tmp_path / "nota.ogg"
    _escribir(ruta, np.clip(y, -1, 1).astype(np.float32), fs)
    z, fz = sf.read(str(ruta), dtype="float32")
    assert fz == fs
    if fz != FS:
        g = math.gcd(fz, FS)
        z = resample_poly(z.astype(np.float64), FS // g, fz // g).astype(np.float32)
    segmentos = seg.detectar_palabras(z, FS, cfg)
    _chequear_bordes(segmentos, reales)


# --------------------------------------------------------------------------------------------
# 7.4 Cortes
# --------------------------------------------------------------------------------------------

def _cfg_cortes(cfg):
    c = copy.deepcopy(cfg)
    c["voz"]["dataset"]["whatsapp"]["margen_s"] = 0.4
    return c


def test_corte_es_un_dataclass_inmutable_con_los_cuatro_campos(seg):
    assert is_dataclass(seg.Corte)
    nombres = [f.name for f in fields(seg.Corte)]
    assert nombres == ["inicio_s", "fin_s", "muestra_inicio", "muestra_fin"], f"campos {nombres}"
    corte = seg.Corte(0.5, 1.0, 1600, 25600)
    with pytest.raises(FrozenInstanceError):
        corte.inicio_s = 0.0


def test_cortes_uno_por_segmento_con_los_tiempos_del_detector(seg, cfg):
    c = _cfg_cortes(cfg)
    y = _audio(FS, 8.0, [(0.6, 0.5, 0.3), (2.0, 0.7, 0.4), (3.5, 0.6, 0.3), (5.2, 0.9, 0.35), (6.9, 0.5, 0.3)],
               ruido_db=-60, semilla=18)
    segmentos = seg.detectar_palabras(y, FS, c)
    cortes = seg.cortar_repeticiones(y, FS, c)
    assert len(segmentos) == 5, f"el audio de prueba dio {len(segmentos)} segmentos"
    assert len(cortes) == len(segmentos), f"{len(cortes)} cortes para {len(segmentos)} segmentos"
    for corte, (i, f) in zip(cortes, segmentos):
        assert (corte.inicio_s, corte.fin_s) == (i, f), f"corte ({corte.inicio_s}, {corte.fin_s}); segmento ({i}, {f})"


AUDIOS_CORTES = {
    "pausas_largas": (8.0, [(1.0, 0.5), (3.0, 0.7), (5.2, 0.6)]),
    "pausas_cortas": (5.0, [(0.8, 0.5), (1.9, 0.6), (3.0, 0.5), (4.0, 0.4)]),
    "pegadas_a_los_bordes": (3.0, [(0.06, 0.6), (1.4, 0.5), (2.34, 0.6)]),
}


@pytest.mark.parametrize("caso", list(AUDIOS_CORTES))
def test_cortes_contienen_su_palabra_dentro_del_audio_y_sin_solaparse(seg, cfg, caso):
    c = _cfg_cortes(cfg)
    total, palabras = AUDIOS_CORTES[caso]
    y = _audio(FS, total, [(i, d, 0.3) for i, d in palabras], ruido_db=-60, semilla=19)
    cortes = seg.cortar_repeticiones(y, FS, c)
    assert len(cortes) == len(palabras), f"{len(cortes)} cortes; se esperaban {len(palabras)}"
    c_max = c["voz"]["dataset"]["clip_max_s"]
    for k, corte in enumerate(cortes):
        mi, mf = corte.muestra_inicio, corte.muestra_fin
        assert isinstance(mi, (int, np.integer)) and isinstance(mf, (int, np.integer)), "muestras no enteras"
        assert 0 <= mi < mf <= len(y), f"corte {k + 1}: [{mi}, {mf}) fuera del audio de {len(y)} muestras"
        assert mi <= corte.inicio_s * FS + 0.5 and mf >= corte.fin_s * FS - 0.5, (
            f"corte {k + 1}: el clip [{mi}, {mf}) no contiene la palabra ({corte.inicio_s:.3f}, {corte.fin_s:.3f}) s")
        assert mf - mi <= round(c_max * FS) + 1, f"corte {k + 1}: dura {(mf - mi) / FS:.3f} s > {c_max} s"
    for a, b in zip(cortes, cortes[1:]):
        assert a.muestra_fin <= b.muestra_inicio, f"clips solapados: [{a.muestra_inicio}, {a.muestra_fin}) y [{b.muestra_inicio}, {b.muestra_fin})"


def test_cortes_con_pausas_largas_el_margen_es_margen_s(seg, cfg):
    c = _cfg_cortes(cfg)
    y = _audio(FS, 8.0, [(1.2, 0.6, 0.3), (3.3, 0.5, 0.3), (5.4, 0.8, 0.3)], ruido_db=-60, semilla=20)
    cortes = seg.cortar_repeticiones(y, FS, c)
    assert len(cortes) == 3
    for corte in cortes:
        assert abs(corte.muestra_inicio - round((corte.inicio_s - 0.4) * FS)) <= 1, (
            f"inicio del clip {corte.muestra_inicio}; se esperaba round((inicio_s − 0,4)·fs) = {round((corte.inicio_s - 0.4) * FS)}")
        assert abs(corte.muestra_fin - round((corte.fin_s + 0.4) * FS)) <= 1, (
            f"fin del clip {corte.muestra_fin}; se esperaba round((fin_s + 0,4)·fs) = {round((corte.fin_s + 0.4) * FS)}")


def test_cortes_con_pausa_corta_el_clip_termina_en_el_punto_medio(seg, cfg):
    c = _cfg_cortes(cfg)
    y = _audio(FS, 4.0, [(1.0, 0.6, 0.3), (2.2, 0.6, 0.3)], ruido_db=-60, semilla=21)
    cortes = seg.cortar_repeticiones(y, FS, c)
    assert len(cortes) == 2, f"{len(cortes)} cortes; la pausa de 0,6 s debería separar dos palabras"
    a, b = cortes
    assert b.inicio_s - a.fin_s < 0.8
    medio = round((a.fin_s + b.inicio_s) / 2 * FS)
    assert abs(a.muestra_fin - medio) <= 1, f"el primer clip termina en {a.muestra_fin}; el punto medio es {medio}"
    assert abs(b.muestra_inicio - medio) <= 1, f"el segundo clip empieza en {b.muestra_inicio}; el punto medio es {medio}"
    assert abs(a.muestra_inicio - round((a.inicio_s - 0.4) * FS)) <= 1
    assert abs(b.muestra_fin - round((b.fin_s + 0.4) * FS)) <= 1


def test_cortes_palabra_de_2_5_s_dura_clip_max_y_queda_centrada(seg, cfg):
    c = _cfg_cortes(cfg)
    y = _audio(FS, 5.0, [(1.25, 2.5, 0.3)], ruido_db=-60, semilla=22)
    cortes = seg.cortar_repeticiones(y, FS, c)
    assert len(cortes) == 1, f"{len(cortes)} cortes"
    corte = cortes[0]
    n = corte.muestra_fin - corte.muestra_inicio
    c_max = c["voz"]["dataset"]["clip_max_s"]
    assert abs(n - round(c_max * FS)) <= 1, f"el clip dura {n / FS:.4f} s; se esperaba {c_max} s"
    izq = corte.inicio_s * FS - corte.muestra_inicio
    der = corte.muestra_fin - corte.fin_s * FS
    assert abs(izq - der) <= 1.5, f"margen izquierdo {izq / FS:.4f} s, derecho {der / FS:.4f} s: la palabra no queda centrada"


def test_cortes_palabra_mas_larga_que_clip_max_el_clip_es_la_palabra(seg, cfg):
    c = _cfg_cortes(cfg)
    y = _audio(FS, 5.0, [(0.75, 3.5, 0.3)], ruido_db=-60, semilla=23)
    cortes = seg.cortar_repeticiones(y, FS, c)
    assert len(cortes) == 1
    corte = cortes[0]
    assert abs(corte.muestra_inicio - round(corte.inicio_s * FS)) <= 1 and abs(corte.muestra_fin - round(corte.fin_s * FS)) <= 1, (
        f"clip [{corte.muestra_inicio}, {corte.muestra_fin}); la palabra es ({corte.inicio_s:.3f}, {corte.fin_s:.3f}) s")


# --------------------------------------------------------------------------------------------
# 7.5 Voces sintéticas
# --------------------------------------------------------------------------------------------

class SintetizadorFalso:
    """Escribe un MP3 corto (24 kHz, una palabra) y anota cada llamada. Nunca usa internet."""

    def __init__(self):
        self.llamadas = []

    def __call__(self, texto, voz, velocidad, ruta):
        self.llamadas.append((texto, voz, velocidad, Path(ruta)))
        fs = 24000
        dur = 0.45 + 0.05 * len(texto)
        f0 = 120.0 + (sum(map(ord, voz)) % 100)
        y = np.zeros(int(round((0.3 + dur + 0.4) * fs)))
        _poner(y, _tono(fs, dur, 0.4, f0), 0.3, fs)
        sf.write(str(ruta), y.astype(np.float32), fs, format="MP3", subtype="MPEG_LAYER_III")


VOCES_PRUEBA = ["es-AR-ElenaNeural", "es-MX-JorgeNeural"]


def _plan_esperado(cfg):
    comandos = list(cfg["voz"]["comandos"])
    t = cfg["voz"]["dataset"]["tts"]
    plan = []
    for voz in t["voces"]:
        for palabra in comandos + list(t["otros"]):
            for k, vel in enumerate(t["velocidades"], start=1):
                plan.append({
                    "voz": voz, "clase": palabra if palabra in comandos else "otro", "palabra": palabra,
                    "texto": palabra[:1].upper() + palabra[1:], "velocidad": vel, "k": k,
                    "archivo": f"{voz}/{palabra}_r{k}.mp3",
                })
    return plan


def _cfg_tts(cfg):
    c = copy.deepcopy(cfg)
    c["voz"]["dataset"]["tts"]["voces"] = list(VOCES_PRUEBA)
    return c


def test_plan_tts_con_cfg_real_en_orden_y_sin_repetidos(cfg, tts):
    plan = tts.plan_tts(copy.deepcopy(cfg))
    esperado = _plan_esperado(cfg)
    t = cfg["voz"]["dataset"]["tts"]
    n = len(t["voces"]) * (len(cfg["voz"]["comandos"]) + len(t["otros"])) * len(t["velocidades"])
    assert len(plan) == n, f"{len(plan)} entradas; se esperaban voces × palabras × velocidades = {n}"
    archivos = [e["archivo"] for e in plan]
    assert len(set(archivos)) == len(archivos), "hay archivos repetidos en el plan"
    claves = ("voz", "clase", "palabra", "texto", "velocidad", "k", "archivo")
    for i, (e, x) in enumerate(zip(plan, esperado)):
        obtenido = {k: e.get(k) for k in claves}
        assert obtenido == x, f"entrada {i}: {obtenido}; se esperaba {x}"


@pytest.mark.parametrize("caso", ["otro_que_es_comando", "voces_repetidas", "palabras_repetidas", "velocidades_repetidas"])
def test_plan_tts_valueerror(cfg, tts, caso):
    c = copy.deepcopy(cfg)
    t = c["voz"]["dataset"]["tts"]
    if caso == "otro_que_es_comando":
        t["otros"] = list(t["otros"]) + [c["voz"]["comandos"][0]]
    elif caso == "voces_repetidas":
        t["voces"] = [VOCES_PRUEBA[0], VOCES_PRUEBA[1], VOCES_PRUEBA[0]]
    elif caso == "palabras_repetidas":
        t["otros"] = ["conocer", "salida", "conocer"]
    else:
        t["velocidades"] = ["-10%", "+10%", "-10%"]
    with pytest.raises(ValueError):
        tts.plan_tts(c)


def test_sintetizar_tts_una_llamada_por_entrada_y_devuelve_las_rutas(cfg, tts, tmp_path):
    c = _cfg_tts(cfg)
    destino = tmp_path / "tts"
    falso = SintetizadorFalso()
    escritas = tts.sintetizar_tts(c, destino=destino, sintetizador=falso)
    plan = _plan_esperado(c)
    esperadas = {(e["texto"], e["voz"], e["velocidad"], destino / e["archivo"]) for e in plan}
    assert len(falso.llamadas) == len(plan), f"{len(falso.llamadas)} llamadas; se esperaban {len(plan)}"
    assert set(falso.llamadas) == esperadas, "las llamadas (texto, voz, velocidad, ruta) no coinciden con el plan"
    assert {Path(p) for p in escritas} == {destino / e["archivo"] for e in plan}, "las rutas devueltas no son las del plan"
    faltan = [e["archivo"] for e in plan if not (destino / e["archivo"]).is_file()]
    assert not faltan, f"no se escribieron: {faltan[:5]}"


def test_sintetizar_tts_segunda_llamada_no_sintetiza_nada(cfg, tts, tmp_path):
    c = _cfg_tts(cfg)
    destino = tmp_path / "tts"
    tts.sintetizar_tts(c, destino=destino, sintetizador=SintetizadorFalso())
    falso = SintetizadorFalso()
    escritas = tts.sintetizar_tts(c, destino=destino, sintetizador=falso)
    assert falso.llamadas == [] and list(escritas) == [], f"se volvió a sintetizar: {falso.llamadas[:3]}"


def test_sintetizar_tts_rehace_solo_el_mp3_borrado(cfg, tts, tmp_path):
    c = _cfg_tts(cfg)
    destino = tmp_path / "tts"
    tts.sintetizar_tts(c, destino=destino, sintetizador=SintetizadorFalso())
    borrado = destino / _plan_esperado(c)[5]["archivo"]
    borrado.unlink()
    falso = SintetizadorFalso()
    escritas = tts.sintetizar_tts(c, destino=destino, sintetizador=falso)
    assert [ll[3] for ll in falso.llamadas] == [borrado], f"llamadas: {falso.llamadas}"
    assert [Path(p) for p in escritas] == [borrado] and borrado.is_file()


def test_sintetizar_tts_sobrescribir_rehace_todos(cfg, tts, tmp_path):
    c = _cfg_tts(cfg)
    destino = tmp_path / "tts"
    tts.sintetizar_tts(c, destino=destino, sintetizador=SintetizadorFalso())
    falso = SintetizadorFalso()
    escritas = tts.sintetizar_tts(c, destino=destino, sintetizador=falso, sobrescribir=True)
    n = len(_plan_esperado(c))
    assert len(falso.llamadas) == n and len(list(escritas)) == n, f"{len(falso.llamadas)} llamadas; se esperaban {n}"


# --------------------------------------------------------------------------------------------
# Fuente de prueba (7.6 y 7.7)
# --------------------------------------------------------------------------------------------

FS_NB = 44100
CLIPS_RUIDO = 3
PARTICIONES_PRUEBA = {"train": ["p01", "p02", "p08", "p22", "p99"], "val": ["p05"], "test": ["p06"]}
RUIDO_PARTICIONES_PRUEBA = {"train": ["ruido_20261017_120000"], "val": ["ruido_20261020_090000"], "test": []}
PERSONAS_NB = {  # código: (f0 de la voz, fecha de la sesión)
    "p01": (120.0, "2026-10-17"), "p02": (125.0, "2026-10-17"), "p08": (140.0, "2026-10-17"),
    "p05": (210.0, "2026-10-20"), "p06": (190.0, "2026-10-20"),
}
DUR_PALABRA = {"reconocer": 0.70, "identificar": 0.85, "salir": 0.50, "conocer": 0.60}
TOMAS_ESTANDAR = [  # (clase, toma, detalle, estilo, distancia_cm)
    ("reconocer", 1, "reconocer", "normal", 50), ("reconocer", 2, "reconocer", "lento", 50),
    ("identificar", 1, "identificar", "normal", 50), ("identificar", 2, "identificar", "rapido", 50),
    ("salir", 1, "salir", "normal", 50), ("salir", 2, "salir", "lejos", 80),
    ("otro", 1, "conocer", "normal", 50), ("otro", 2, "tos", "normal", 50),
]
TOMAS_ESPECIALES = [  # (código, clase, toma, tipo, motivo esperado)
    ("p01", "reconocer", 3, "pegada_al_final", "cortada"),
    ("p01", "identificar", 3, "palabra_de_2_8_s", "duracion"),
    ("p02", "reconocer", 3, "pico_1_en_el_0_1_por_ciento", None),
    ("p02", "salir", 3, "ruido_15_db_mas_bajo_despues", None),
    ("p06", "reconocer", 3, "2_por_ciento_en_mas_menos_1", "saturada"),
    ("p06", "identificar", 3, "vacia", "sin_palabra"),
    ("p06", "salir", 3, "dos_palabras_del_mismo_nivel", "varias_palabras"),
]
NOTA_NOTEBOOK = ("p05/p05_identificar_02.wav", "se oye una puerta")
NOTA_WHATSAPP = ("p22/p22_identificar_01.ogg", 2, "se cuela otra voz")
WHATSAPP_PRUEBA = [  # (ruta en whatsapp/, fs, duración total, [(inicio_s, dur_s)], f0)
    ("p08/p08_reconocer_01.ogg", 48000, 1.8, [(0.5, 0.7)], 140.0),
    ("p08/p08_identificar_01.opus", 16000, 1.9, [(0.45, 0.85)], 140.0),
    ("p08/p08_salir_01.m4a", 48000, 1.6, [(0.5, 0.5)], 140.0),
    ("p22/p22_identificar_01.ogg", 16000, 6.1, [(0.5, 0.6), (2.0, 0.6), (3.5, 0.6), (5.0, 0.6)], 230.0),
    # Cinco cortes: el tercero son dos repeticiones pegadas (0,6 s + 0,1 s de pausa + 0,6 s)
    ("p22/p22_reconocer_01.ogg", 48000, 7.3,
     [(0.5, 0.42), (1.75, 0.42), (3.0, 0.6), (3.7, 0.6), (5.1, 0.42), (6.35, 0.42)], 230.0),
    ("p22/p22_salir_01.ogg", 16000, 1.5, [(0.5, 0.5)], 230.0),
    ("p22/p22_salir_02.ogg", 16000, 1.2, [(0.5, 0.7)], 230.0),  # soltó el botón antes de terminar
    ("p22/p22_otro_01.ogg", 48000, 4.5, [(0.5, 3.5)], 180.0),
    ("p22/p22_otro_02.ogg", 16000, 1.4, [(0.5, 0.4)], 180.0),
    ("p05/p05_salir_01.ogg", 16000, 2.9, [(0.5, 0.5), (1.9, 0.5)], 210.0),
]
CORTES_WHATSAPP = {  # palabras (cortes) que tiene que encontrar en cada audio
    "p08/p08_reconocer_01.ogg": 1, "p08/p08_identificar_01.opus": 1, "p08/p08_salir_01.m4a": 1,
    "p22/p22_identificar_01.ogg": 4, "p22/p22_reconocer_01.ogg": 5, "p22/p22_salir_01.ogg": 1, "p22/p22_salir_02.ogg": 1,
    "p22/p22_otro_01.ogg": 1, "p22/p22_otro_02.ogg": 1, "p05/p05_salir_01.ogg": 2,
}
MOTIVOS_WHATSAPP = {
    ("p22/p22_identificar_01.ogg", 2): "notas", ("p22/p22_reconocer_01.ogg", 3): "doble",
    ("p22/p22_otro_01.ogg", 1): "larga", ("p22/p22_salir_02.ogg", 1): "cortada",
    ("p05/p05_salir_01.ogg", 1): "whatsapp_val_test", ("p05/p05_salir_01.ogg", 2): "whatsapp_val_test",
}


def _cfg_prueba(cfg):
    c = copy.deepcopy(cfg)
    d = c["voz"]["dataset"]
    d["particiones"] = copy.deepcopy(PARTICIONES_PRUEBA)
    d["tts"]["voces"] = list(VOCES_PRUEBA)
    d["ruido"]["clips_por_archivo"] = CLIPS_RUIDO
    d["ruido"]["particiones"] = copy.deepcopy(RUIDO_PARTICIONES_PRUEBA)
    return c


def _toma_notebook(tipo, f0, semilla, dur=0.7, ini=0.9, pico=0.3):
    n = 3 * FS_NB
    y = _ruido(FS_NB, 3.0, -75.0, semilla)
    if tipo == "palabra":
        _poner(y, _tono(FS_NB, dur, pico, f0), ini, FS_NB)
    elif tipo == "vacia":
        pass
    elif tipo == "pegada_al_final":
        w = _tono(FS_NB, 0.6, 0.3, f0)
        y[n - len(w):] += w
    elif tipo == "palabra_de_2_8_s":
        # Cuatro sílabas de 0,625 s con cierres de 0,1 s (2,8 s en total), como el habla: una
        # palabra pareja que ocupa más del 90 % de la toma no se detecta (el percentil 10 de los
        # niveles cae dentro de ella).
        for s in range(4):
            _poner(y, _tono(FS_NB, 0.625, 0.3, f0), 0.1 + s * 0.725, FS_NB)
    elif tipo == "dos_palabras_del_mismo_nivel":
        _poner(y, _tono(FS_NB, 0.5, 0.3, f0), 0.5, FS_NB)
        _poner(y, _tono(FS_NB, 0.5, 0.3, f0), 1.7, FS_NB)
    elif tipo == "ruido_15_db_mas_bajo_despues":
        _poner(y, _tono(FS_NB, 0.5, 0.3, f0), 0.6, FS_NB)
        # Ruido de banda hasta 8 kHz (hecho a 16 kHz y subido a 44,1 kHz): 15 dB bajo la palabra
        # también después de remuestrear.
        r = resample_poly(_ruido(FS, 0.6, _nivel_tono(0.3, f0) - 15, semilla + 1), 441, 160)
        nr = int(0.01 * FS_NB)
        r[:nr] *= np.arange(nr) / nr
        r[-nr:] *= np.arange(nr)[::-1] / nr
        _poner(y, r, 1.7, FS_NB)
    elif tipo == "pico_1_en_el_0_1_por_ciento":
        _poner(y, _tono(FS_NB, 0.7, 0.3, f0), 0.9, FS_NB)
        m = int(round(0.001 * n))
        i = int(round(1.2 * FS_NB))
        y[i:i + m] = 1.0
    elif tipo == "2_por_ciento_en_mas_menos_1":
        _poner(y, _tono(FS_NB, 0.9, 0.5, f0), 0.9, FS_NB)
        bloques = 12
        m = int(round(0.02 * n)) // bloques
        for b in range(bloques):
            i = int(round((1.0 + 0.06 * b) * FS_NB))
            y[i:i + m] = 1.0 if b % 2 == 0 else -1.0
    else:
        raise AssertionError(tipo)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _nombre_toma(codigo, clase, toma, detalle):
    if clase == "otro":
        return f"{codigo}/{codigo}_otro_{toma:02d}_{detalle}.wav"
    return f"{codigo}/{codigo}_{clase}_{toma:02d}.wav"


def _fila_sesion(archivo, persona, clase, detalle, estilo, distancia, toma, fecha, y):
    return {
        "archivo": archivo, "persona": persona, "clase": clase, "detalle": detalle, "estilo": estilo,
        "distancia_cm": "" if distancia is None else str(distancia), "toma": str(toma), "fs": str(FS_NB),
        "dispositivo": "Micrófono de prueba", "fecha": fecha,
        "pico": f"{float(np.max(np.abs(y))):.3f}", "rms": f"{math.sqrt(float(np.mean(np.square(y, dtype=np.float64)))):.4f}",
        "avisos": "",
    }


COLUMNAS_SESIONES = ["archivo", "persona", "clase", "detalle", "estilo", "distancia_cm", "toma", "fs",
                     "dispositivo", "fecha", "pico", "rms", "avisos"]


def _armar_fuente(raiz, cfg_p):
    """Escribe la fuente de prueba en raiz y devuelve la lista de candidatos esperados."""
    grupos = cfg_p["voz"]["dataset"]["mismo_hablante"]
    part = cfg_p["voz"]["dataset"]["particiones"]
    esperados = []
    filas = []
    nb = raiz / "notebook"
    semilla = 100

    def parti(codigo):
        return _particion_de(codigo, part)[0]

    # Tomas de la notebook
    tomas = []
    for codigo, (f0, fecha) in PERSONAS_NB.items():
        for clase, toma, detalle, estilo, distancia in TOMAS_ESTANDAR:
            tipo = "vacia" if detalle == "tos" else "palabra"
            tomas.append((codigo, clase, toma, detalle, estilo, distancia, tipo, None))
    for codigo, clase, toma, tipo, motivo in TOMAS_ESPECIALES:
        tomas.append((codigo, clase, toma, clase, "normal", 50, tipo, motivo))
    for i, (codigo, clase, toma, detalle, estilo, distancia, tipo, motivo) in enumerate(tomas):
        f0, fecha = PERSONAS_NB[codigo]
        archivo = _nombre_toma(codigo, clase, toma, detalle)
        semilla += 1
        y = _toma_notebook(tipo, f0, semilla, dur=DUR_PALABRA.get(detalle, 0.7) + 0.02 * (i % 3),
                           ini=0.85 + 0.05 * toma, pico=0.25 + 0.03 * toma)
        _escribir(nb / archivo, y, FS_NB)
        filas.append(_fila_sesion(archivo, codigo, clase, detalle, estilo, distancia, toma,
                                  f"{fecha}T10:{i // 60:02d}:{i % 60:02d}", y))
        if archivo == NOTA_NOTEBOOK[0]:
            motivo = "notas"
        p = parti(codigo)
        esperados.append({
            "origen": "notebook", "fuente": f"notebook/{archivo}", "toma": toma, "codigo": codigo,
            "hablante": _hablante(codigo, grupos), "clase": clase, "detalle": detalle, "estilo": estilo,
            "distancia_cm": distancia, "velocidad_tts": None, "fs_fuente": FS_NB, "tipo": tipo,
            "particion": None if motivo else p, "motivo": motivo,
            "nota": NOTA_NOTEBOOK[1] if motivo == "notas" else None,
            "archivo": None if motivo else f"{p}/nb_{Path(archivo).stem}.wav",
        })
    # Fila repetida: p08_salir_02 aparece primero con otros datos; vale la última
    repetida = next(f for f in filas if f["archivo"] == "p08/p08_salir_02.wav")
    vieja = dict(repetida, estilo="cerca", distancia_cm="30", fecha="2026-10-17T09:59:00")
    filas.insert(0, vieja)

    # Ruido ambiente
    for k, nombre in enumerate(("ruido_20261017_120000", "ruido_20261020_090000")):
        y = np.clip(_ruido(FS_NB, 10.0, -50.0, 900 + k), -1, 1).astype(np.float32)
        archivo = f"ruido/{nombre}.wav"
        _escribir(nb / archivo, y, FS_NB)
        fecha = f"2026-10-{17 if k == 0 else 20}T12:00:00"
        filas.append(_fila_sesion(archivo, "", "ruido", "ambiente", "", None, 1, fecha, y))
        p = _particion_de(nombre, RUIDO_PARTICIONES_PRUEBA)[0]
        for c in range(1, CLIPS_RUIDO + 1):
            esperados.append({
                "origen": "ruido", "fuente": f"notebook/{archivo}", "toma": c, "codigo": None,
                "hablante": f"ruido:{nombre}", "clase": "otro", "detalle": "ruido", "estilo": None,
                "distancia_cm": None, "velocidad_tts": None, "fs_fuente": FS_NB, "tipo": "ruido",
                "particion": p, "motivo": None, "nota": None,
                "archivo": f"{p}/ruido_{nombre}_c{c:02d}.wav",
            })

    with open(nb / "sesiones.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_SESIONES)
        w.writeheader()
        w.writerows(filas)
    (nb / "notas.txt").write_text(
        "Notas de las sesiones de prueba (las líneas que no empiezan con 'excluir:' son comentarios).\n"
        "- p05/p05_identificar_02.wav: alguien cerró una puerta durante la toma.\n"
        "\n"
        f"excluir: {NOTA_NOTEBOOK[0]} | {NOTA_NOTEBOOK[1]}\n", encoding="utf-8")

    # WhatsApp
    wa = raiz / "whatsapp"
    for j, (rel, fs, total, palabras, f0) in enumerate(WHATSAPP_PRUEBA):
        y = _audio(fs, total, [(i, d, 0.4) for i, d in palabras], ruido_db=-60, semilla=500 + j, f0=f0)
        _escribir(wa / rel, y, fs)
        codigo = rel.split("/")[0]
        clase = Path(rel).stem.split("_")[1]
        for k in range(1, CORTES_WHATSAPP[rel] + 1):
            motivo = MOTIVOS_WHATSAPP.get((rel, k))
            p = parti(codigo)
            esperados.append({
                "origen": "whatsapp", "fuente": f"whatsapp/{rel}", "toma": k, "codigo": codigo,
                "hablante": _hablante(codigo, grupos), "clase": clase, "detalle": "libre" if clase == "otro" else clase,
                "estilo": None, "distancia_cm": None, "velocidad_tts": None, "fs_fuente": fs, "tipo": "whatsapp",
                "particion": None if motivo else p, "motivo": motivo,
                "nota": NOTA_WHATSAPP[2] if motivo == "notas" else None,
                "archivo": None if motivo else f"{p}/wa_{Path(rel).stem}_s{k:02d}.wav",
            })
    (wa / "notas.txt").write_text(
        "# Exclusiones de los audios de WhatsApp\n"
        f"excluir: {NOTA_WHATSAPP[0]}#{NOTA_WHATSAPP[1]} | {NOTA_WHATSAPP[2]}\n", encoding="utf-8")
    (wa / "origen.csv").write_text(
        "archivo,comando,original\n"
        "p22/p22_identificar_01.ogg,identificar,PTT-20261020-WA0001.opus\n", encoding="utf-8")
    (wa / "p22" / "LEEME.txt").write_text("Este archivo no es un audio y se ignora.\n", encoding="utf-8")

    # Voces sintéticas (MP3 a 24 kHz del sintetizador falso)
    falso = SintetizadorFalso()
    for e in _plan_esperado(cfg_p):
        ruta = raiz / "tts" / e["archivo"]
        ruta.parent.mkdir(parents=True, exist_ok=True)
        falso(e["texto"], e["voz"], e["velocidad"], ruta)
        esperados.append({
            "origen": "tts", "fuente": f"tts/{e['archivo']}", "toma": e["k"], "codigo": None,
            "hablante": e["voz"], "clase": e["clase"], "detalle": e["palabra"], "estilo": None,
            "distancia_cm": None, "velocidad_tts": e["velocidad"], "fs_fuente": 24000, "tipo": "tts",
            "particion": "train", "motivo": None, "nota": None,
            "archivo": f"train/tts_{e['voz']}_{e['palabra']}_r{e['k']}.wav",
        })
    return esperados


@pytest.fixture(scope="module")
def fuente(cfg, tmp_path_factory):
    raiz = tmp_path_factory.mktemp("fuente_v1")
    cfg_p = _cfg_prueba(cfg)
    esperados = _armar_fuente(raiz, cfg_p)
    return SimpleNamespace(raiz=raiz, cfg=cfg_p, esperados=esperados,
                           por_clave={_clave(e): e for e in esperados}, huella=_huella(raiz))


@pytest.fixture(scope="module")
def candidatos(ds, fuente):
    return ds.inventario(copy.deepcopy(fuente.cfg), fuente.raiz)


@pytest.fixture(scope="module")
def cand_por_clave(candidatos):
    return {_clave(c): c for c in candidatos}


def _esperado(fuente, fuente_rel, toma):
    return next(e for e in fuente.esperados if e["fuente"] == fuente_rel and e["toma"] == toma)


# --------------------------------------------------------------------------------------------
# 7.6 Inventario y particiones
# --------------------------------------------------------------------------------------------

def test_constantes_del_modulo_dataset(ds):
    assert tuple(ds.COLUMNAS) == COLUMNAS_CONTRATO, f"COLUMNAS = {ds.COLUMNAS}"
    assert tuple(ds.COLUMNAS_EXCLUIDAS) == COLUMNAS_EXCLUIDAS_CONTRATO, f"COLUMNAS_EXCLUIDAS = {ds.COLUMNAS_EXCLUIDAS}"
    assert tuple(ds.ORIGENES) == ORIGENES_CONTRATO, f"ORIGENES = {ds.ORIGENES}"
    assert tuple(ds.MOTIVOS) == MOTIVOS_CONTRATO, f"MOTIVOS = {ds.MOTIVOS}"


def test_inventario_candidatos_tienen_las_columnas_mas_motivo_y_nota(candidatos):
    faltan = set()
    for c in candidatos:
        faltan |= set(COLUMNAS_CONTRATO + ("motivo", "nota")) - set(c)
    assert not faltan, f"claves que faltan en los candidatos: {sorted(faltan)}"


def test_inventario_un_candidato_por_toma_palabra_mp3_y_clip_de_ruido(fuente, candidatos):
    obtenido = Counter(c["fuente"] for c in candidatos)
    esperado = Counter(e["fuente"] for e in fuente.esperados)
    diferencias = {f: (obtenido.get(f, 0), esperado.get(f, 0)) for f in set(obtenido) | set(esperado)
                   if obtenido.get(f, 0) != esperado.get(f, 0)}
    assert not diferencias, f"candidatos por fuente (obtenidos, esperados): {dict(sorted(diferencias.items()))}"
    claves = [_clave(c) for c in candidatos]
    assert len(set(claves)) == len(claves), "hay candidatos repetidos (mismo origen, fuente y toma)"


@pytest.mark.parametrize("origen", ORIGENES_CONTRATO)
def test_inventario_etiquetas_de_sesiones_y_del_nombre(fuente, cand_por_clave, origen):
    errores = []
    for e in fuente.esperados:
        if e["origen"] != origen:
            continue
        c = cand_por_clave.get(_clave(e))
        if c is None:
            errores.append(f"falta {_clave(e)}")
            continue
        campos = ["origen", "clase", "detalle", "toma"]
        if origen in ("notebook", "whatsapp"):
            campos += ["codigo"]
        if origen == "notebook":
            campos += ["estilo", "distancia_cm"]
        if origen == "tts":
            campos += ["velocidad_tts"]
        for campo in campos:
            if _txt(c.get(campo)) != _txt(e[campo]):
                errores.append(f"{e['fuente']} #{e['toma']}: {campo} = {c.get(campo)!r}, se esperaba {e[campo]!r}")
    assert not errores, "\n".join(errores[:20])


def test_inventario_fila_repetida_en_sesiones_vale_la_ultima(cand_por_clave):
    c = cand_por_clave.get(("notebook", "notebook/p08/p08_salir_02.wav", 2))
    assert c is not None, "falta el candidato de p08_salir_02"
    assert (_txt(c["estilo"]), _txt(c["distancia_cm"])) == ("lejos", "80"), (
        f"estilo {c['estilo']!r} y distancia {c['distancia_cm']!r}; valen los de la última fila (lejos, 80)")


def test_inventario_hablante_segun_mismo_hablante_voz_o_ruido(fuente, cand_por_clave):
    errores = [f"{e['fuente']} #{e['toma']}: hablante {cand_por_clave[_clave(e)].get('hablante')!r}, se esperaba {e['hablante']!r}"
               for e in fuente.esperados
               if _clave(e) in cand_por_clave and cand_por_clave[_clave(e)].get("hablante") != e["hablante"]]
    assert not errores, "\n".join(errores[:20])


def test_inventario_particiones_segun_las_listas_tts_en_train_y_ruido_por_archivo(fuente, cand_por_clave):
    errores = []
    for e in fuente.esperados:
        c = cand_por_clave.get(_clave(e))
        if c is None:
            continue
        if e["particion"] is None:
            if not _vacio(c.get("particion")) or not _vacio(c.get("archivo")):
                errores.append(f"{e['fuente']} #{e['toma']} se excluye: particion y archivo deberían estar vacíos")
        elif c.get("particion") != e["particion"]:
            errores.append(f"{e['fuente']} #{e['toma']}: particion {c.get('particion')!r}, se esperaba {e['particion']!r}")
    assert not errores, "\n".join(errores[:20])


def test_inventario_clips_de_ruido_equiespaciados_y_sin_solaparse(fuente, cand_por_clave):
    C = round(fuente.cfg["voz"]["dataset"]["clip_max_s"] * FS)
    L = 10 * FS
    n = CLIPS_RUIDO
    for nombre in ("ruido_20261017_120000", "ruido_20261020_090000"):
        for k in range(n):
            c = cand_por_clave[("ruido", f"notebook/ruido/{nombre}.wav", k + 1)]
            ini = round(k * (L - C) / (n - 1))
            assert (int(c["muestra_inicio"]), int(c["muestra_fin"])) == (ini, ini + C), (
                f"{nombre} clip {k + 1}: [{c['muestra_inicio']}, {c['muestra_fin']}); se esperaba [{ini}, {ini + C})")


def test_inventario_persona_con_notebook_y_whatsapp_en_train_va_toda_a_train(fuente, cand_por_clave):
    p08 = [c for c in cand_por_clave.values() if c.get("codigo") == "p08"]
    assert {c["origen"] for c in p08} == {"notebook", "whatsapp"}
    malos = [(c["fuente"], c.get("particion"), c.get("motivo")) for c in p08
             if c["origen"] == "whatsapp" and (c.get("particion") != "train" or not _vacio(c.get("motivo")))]
    assert not malos, f"los WhatsApp de p08 (en train) deberían entrar en train: {malos}"


def test_inventario_persona_con_notebook_y_whatsapp_en_val_pierde_sus_whatsapp(ds, fuente):
    c = copy.deepcopy(fuente.cfg)
    c["voz"]["dataset"]["particiones"]["train"].remove("p08")
    c["voz"]["dataset"]["particiones"]["val"].append("p08")
    cands = [x for x in ds.inventario(c, fuente.raiz) if x.get("codigo") == "p08"]
    nb = [x for x in cands if x["origen"] == "notebook"]
    wa = [x for x in cands if x["origen"] == "whatsapp"]
    assert nb and wa
    malos_nb = [(x["fuente"], x.get("particion")) for x in nb if _vacio(x.get("motivo")) and x.get("particion") != "val"]
    assert not malos_nb, f"las tomas de la notebook de p08 deberían ir a val: {malos_nb}"
    malos_wa = [(x["fuente"], x.get("motivo")) for x in wa if x.get("motivo") != "whatsapp_val_test"]
    assert not malos_wa, f"los WhatsApp de p08 deberían quedar excluidos con whatsapp_val_test: {malos_wa}"
    assert all(_vacio(x.get("particion")) for x in wa), "un WhatsApp excluido no debe tener partición"


def test_inventario_admite_codigos_en_las_listas_sin_audios(fuente, candidatos):
    assert "p99" in fuente.cfg["voz"]["dataset"]["particiones"]["train"]
    assert not [c for c in candidatos if c.get("codigo") == "p99"]


def _mover(lista_de, lista_a, codigo):
    def cambio(d):
        d["particiones"][lista_de].remove(codigo)
        d["particiones"][lista_a].append(codigo)
    return cambio


CASOS_PARTICIONES = {
    "codigo_con_audios_en_ninguna_lista": (lambda d: d["particiones"].update(test=[]), ["p06"]),
    "codigo_en_dos_listas": (lambda d: d["particiones"]["train"].append("p05"), ["p05"]),
    "grupo_de_mismo_hablante_partido": (_mover("train", "val", "p02"), ["p01", "p02"]),
    "codigo_solo_whatsapp_en_val": (_mover("train", "val", "p22"), ["p22"]),
    "codigo_solo_whatsapp_en_test": (_mover("train", "test", "p22"), ["p22"]),
    "ruido_sin_particion": (lambda d: d["ruido"]["particiones"].update(val=[]), ["ruido_20261020_090000"]),
    "ruido_en_dos_listas": (lambda d: d["ruido"]["particiones"]["test"].append("ruido_20261020_090000"),
                            ["ruido_20261020_090000"]),
}


@pytest.mark.parametrize("caso", list(CASOS_PARTICIONES))
def test_inventario_valueerror_por_particiones(ds, fuente, caso):
    cambio, nombres = CASOS_PARTICIONES[caso]
    c = copy.deepcopy(fuente.cfg)
    cambio(c["voz"]["dataset"])
    with pytest.raises(ValueError) as error:
        ds.inventario(c, fuente.raiz)
    faltan = [n for n in nombres if n not in str(error.value)]
    assert not faltan, f"el mensaje no nombra {faltan}: {error.value}"


def _agregar_fila_sesion(raiz, archivo):
    with open(raiz / "notebook" / "sesiones.csv", "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([archivo, "p05", "reconocer", "reconocer", "normal", "50", "9", "44100",
                                "Micrófono de prueba", "2026-10-20T11:00:00", "0.300", "0.0500", ""])


def _copiar_audio(raiz, de, a):
    destino = raiz / a
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raiz / de, destino)


def _agregar_linea(ruta, linea):
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def _borrar_mp3(raiz, n):
    mp3 = sorted((raiz / "tts").rglob("*.mp3"))[:n]
    for p in mp3:
        p.unlink()


def _ruido_corto(raiz):
    y = np.clip(_ruido(FS_NB, 8.0, -50.0, 77), -1, 1).astype(np.float32)
    _escribir(raiz / "notebook" / "ruido" / "ruido_20261017_120000.wav", y, FS_NB)


CASOS_FUENTE = {
    "fila_de_sesiones_sin_archivo": (lambda r: _agregar_fila_sesion(r, "p05/p05_reconocer_09.wav"), ["p05_reconocer_09"]),
    "wav_de_persona_sin_fila": (lambda r: _copiar_audio(r, "notebook/p05/p05_reconocer_01.wav", "notebook/p05/p05_reconocer_08.wav"),
                                ["p05_reconocer_08"]),
    "wav_de_ruido_sin_fila": (lambda r: _copiar_audio(r, "notebook/ruido/ruido_20261017_120000.wav",
                                                      "notebook/ruido/ruido_20261021_100000.wav"), ["ruido_20261021_100000"]),
    "whatsapp_toma_de_un_digito": (lambda r: _copiar_audio(r, "whatsapp/p22/p22_salir_01.ogg", "whatsapp/p22/p22_salir_2.ogg"),
                                   ["p22_salir_2.ogg"]),
    "whatsapp_clase_que_no_existe": (lambda r: _copiar_audio(r, "whatsapp/p22/p22_salir_01.ogg", "whatsapp/p22/p22_hola_01.ogg"),
                                     ["p22_hola_01.ogg"]),
    "whatsapp_en_la_carpeta_de_otro_codigo": (lambda r: _copiar_audio(r, "whatsapp/p22/p22_salir_01.ogg", "whatsapp/p22/p08_salir_02.ogg"),
                                              ["p08_salir_02.ogg"]),
    "whatsapp_con_el_nombre_original": (lambda r: _copiar_audio(r, "whatsapp/p22/p22_salir_01.ogg", "whatsapp/p22/PTT-20261020-WA0007.opus"),
                                        ["PTT-20261020-WA0007.opus"]),
    "notas_notebook_con_archivo_inexistente": (lambda r: _agregar_linea(r / "notebook" / "notas.txt",
                                                                        "excluir: p05/p05_reconocer_07.wav | no existe"), ["p05_reconocer_07"]),
    "notas_whatsapp_con_archivo_inexistente": (lambda r: _agregar_linea(r / "whatsapp" / "notas.txt",
                                                                        "excluir: p22/p22_salir_05.ogg | no existe"), ["p22_salir_05"]),
    "ruido_mas_corto_que_sus_clips": (_ruido_corto, []),  # sección 6: ValueError si L < n·C
}


@pytest.mark.parametrize("caso", list(CASOS_FUENTE))
def test_inventario_valueerror_por_la_fuente(ds, fuente, tmp_path, caso):
    cambio, nombres = CASOS_FUENTE[caso]
    raiz = tmp_path / "fuente"
    shutil.copytree(fuente.raiz, raiz)
    cambio(raiz)
    with pytest.raises(ValueError) as error:
        ds.inventario(copy.deepcopy(fuente.cfg), raiz)
    faltan = [n for n in nombres if n not in str(error.value)]
    assert not faltan, f"el mensaje no nombra {faltan}: {error.value}"


def test_inventario_valueerror_si_faltan_mp3_del_plan(ds, fuente, tmp_path):
    raiz = tmp_path / "fuente"
    shutil.copytree(fuente.raiz, raiz)
    _borrar_mp3(raiz, 3)
    with pytest.raises(ValueError) as error:
        ds.inventario(copy.deepcopy(fuente.cfg), raiz)
    mensaje = str(error.value)
    assert re.search(r"(?<![0-9])3(?![0-9])", mensaje), f"el mensaje no dice que faltan 3 MP3: {mensaje}"
    assert "src.voz.tts" in mensaje, f"el mensaje no indica correr python -m src.voz.tts: {mensaje}"


def _con_motivo(fuente, motivo):
    return [e for e in fuente.esperados if e["motivo"] == motivo]


@pytest.mark.parametrize("motivo", MOTIVOS_CONTRATO)
def test_inventario_motivo_en_el_clip_armado_para_provocarlo(fuente, cand_por_clave, motivo):
    casos = _con_motivo(fuente, motivo)
    assert casos, f"la fuente de prueba no tiene un caso de {motivo}"
    errores = []
    for e in casos:
        c = cand_por_clave.get(_clave(e))
        if c is None:
            errores.append(f"falta el candidato {_clave(e)}")
        elif c.get("motivo") != motivo:
            errores.append(f"{e['fuente']} #{e['toma']}: motivo {c.get('motivo')!r}, se esperaba {motivo!r}")
    assert not errores, "\n".join(errores)


def test_inventario_los_clips_buenos_no_tienen_motivo(fuente, cand_por_clave):
    malos = [(e["fuente"], e["toma"], cand_por_clave[_clave(e)].get("motivo"), cand_por_clave[_clave(e)].get("nota"))
             for e in fuente.esperados
             if e["motivo"] is None and _clave(e) in cand_por_clave
             and not (_vacio(cand_por_clave[_clave(e)].get("motivo")) and _vacio(cand_por_clave[_clave(e)].get("nota")))]
    assert not malos, f"clips que deberían entrar y tienen motivo (fuente, toma, motivo, nota): {malos[:10]}"


def test_inventario_nota_con_el_texto_de_notas_txt(fuente, cand_por_clave):
    for e in _con_motivo(fuente, "notas"):
        c = cand_por_clave[_clave(e)]
        assert _txt(c.get("nota")).strip() == e["nota"], f"{e['fuente']}: nota {c.get('nota')!r}, se esperaba {e['nota']!r}"


def test_inventario_pico_1_en_el_0_1_por_ciento_no_satura(cand_por_clave):
    c = cand_por_clave[("notebook", "notebook/p02/p02_reconocer_03.wav", 3)]
    assert _vacio(c.get("motivo")), f"motivo {c.get('motivo')!r}; un pico de 1,0 en el 0,1 % de las muestras no satura"


def test_inventario_ruido_15_db_mas_bajo_no_es_una_segunda_palabra(cand_por_clave):
    c = cand_por_clave[("notebook", "notebook/p02/p02_salir_03.wav", 3)]
    assert _vacio(c.get("motivo")) and int(c["n_segmentos"]) == 1, (
        f"motivo {c.get('motivo')!r}, n_segmentos {c.get('n_segmentos')}; se esperaba que entre con 1 segmento")


def test_inventario_notas_con_numeral_excluye_solo_ese_corte(fuente, cand_por_clave):
    rel, k_excluido, _ = NOTA_WHATSAPP
    for k in range(1, CORTES_WHATSAPP[rel] + 1):
        c = cand_por_clave[("whatsapp", f"whatsapp/{rel}", k)]
        if k == k_excluido:
            assert c.get("motivo") == "notas", f"corte {k}: motivo {c.get('motivo')!r}, se esperaba 'notas'"
        else:
            assert _vacio(c.get("motivo")), f"corte {k}: motivo {c.get('motivo')!r}; solo se excluye el corte {k_excluido}"


def test_inventario_otro_sin_segmentos_entra(fuente, cand_por_clave):
    tos = [cand_por_clave[_clave(e)] for e in fuente.esperados if e["detalle"] == "tos"]
    assert tos
    malos = [(c["fuente"], c.get("n_segmentos"), c.get("motivo")) for c in tos
             if int(c["n_segmentos"]) != 0 or not _vacio(c.get("motivo"))]
    assert not malos, f"tomas de 'otro' sin palabra (fuente, n_segmentos, motivo): {malos}"


def test_inventario_no_escribe_nada_en_la_fuente(fuente, candidatos):
    assert _huella(fuente.raiz) == fuente.huella, "inventario cambió algo en la carpeta fuente"


# --------------------------------------------------------------------------------------------
# 7.7 Generación
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generado(ds, fuente, candidatos, tmp_path_factory):
    destino = tmp_path_factory.mktemp("dataset_a")
    meta, excl = ds.generar_dataset(copy.deepcopy(fuente.cfg), fuente.raiz, destino)
    return SimpleNamespace(destino=destino, meta=list(meta), excl=list(excl))


def _archivos(carpeta):
    return sorted(p.relative_to(carpeta).as_posix() for p in Path(carpeta).rglob("*") if p.is_file())


def test_generar_crea_exactamente_los_wav_que_entran_y_los_dos_csv(fuente, generado):
    esperados = sorted(e["archivo"] for e in fuente.esperados if e["archivo"])
    obtenidos = _archivos(generado.destino)
    sobran = sorted(set(obtenidos) - set(esperados) - {"metadatos.csv", "excluidas.csv"})
    faltan = sorted((set(esperados) | {"metadatos.csv", "excluidas.csv"}) - set(obtenidos))
    assert not sobran and not faltan, f"sobran {sobran[:10]}; faltan {faltan[:10]}"
    malos = [f["archivo"] for f in generado.meta if f["archivo"].split("/")[0] != f["particion"]]
    assert not malos, f"WAV fuera de la carpeta de su partición: {malos[:10]}"


def test_generar_wav_de_16_khz_mono_pcm16_con_la_duracion_de_la_fila(fuente, generado):
    c_max = fuente.cfg["voz"]["dataset"]["clip_max_s"]
    errores = []
    for f in generado.meta:
        info = sf.info(str(generado.destino / f["archivo"]))
        if (info.samplerate, info.channels, info.subtype) != (FS, 1, "PCM_16"):
            errores.append(f"{f['archivo']}: {info.samplerate} Hz, {info.channels} canales, {info.subtype}")
        if abs(info.frames / FS - f["duracion_s"]) > 0.5e-4 + 1e-9:
            errores.append(f"{f['archivo']}: dura {info.frames / FS:.5f} s y duracion_s = {f['duracion_s']}")
        if info.frames != f["muestra_fin"] - f["muestra_inicio"]:
            errores.append(f"{f['archivo']}: {info.frames} muestras y muestra_fin − muestra_inicio = {f['muestra_fin'] - f['muestra_inicio']}")
        if info.frames > round(c_max * FS):
            errores.append(f"{f['archivo']}: dura {info.frames / FS:.4f} s > clip_max_s")
    assert not errores, "\n".join(errores[:20])


@pytest.mark.parametrize("nombre,columnas,atributo", [
    ("metadatos.csv", COLUMNAS_CONTRATO, "meta"), ("excluidas.csv", COLUMNAS_EXCLUIDAS_CONTRATO, "excl")])
def test_generar_csv_con_el_encabezado_y_las_filas_que_devuelve(generado, nombre, columnas, atributo):
    encabezado, filas = _leer_csv(generado.destino / nombre)
    assert tuple(encabezado) == columnas, f"encabezado de {nombre}: {encabezado}"
    devueltas = getattr(generado, atributo)
    assert len(filas) == len(devueltas), f"{nombre} tiene {len(filas)} filas y la función devolvió {len(devueltas)}"
    for i, (txt, dev) in enumerate(zip(filas, devueltas)):
        assert _filas_iguales(_tipar(txt), dev), f"{nombre}, fila {i + 1}: en el CSV {_tipar(txt)}; devuelta {dev}"


@pytest.mark.parametrize("atributo", ["meta", "excl"])
def test_generar_filas_en_orden_de_origen_fuente_y_toma(generado, atributo):
    filas = getattr(generado, atributo)
    claves = [(ORIGENES_CONTRATO.index(f["origen"]), f["fuente"], f["toma"]) for f in filas]
    assert claves == sorted(claves), "las filas no están ordenadas por origen (como en ORIGENES), fuente y toma"


def test_generar_tipos_y_redondeo_de_lo_que_devuelve(generado):
    errores = []
    for tabla, filas in (("metadatos", generado.meta), ("excluidas", generado.excl)):
        for f in filas:
            for k, v in f.items():
                if v is None:
                    continue
                if k in ENTERAS:
                    ok = _es_entero(v)
                elif k in DECIMALES:
                    ok = isinstance(v, float) and round(v, DECIMALES[k]) == v
                else:
                    ok = isinstance(v, str) and v != ""
                if not ok:
                    errores.append(f"{tabla}, {f.get('archivo') or f.get('fuente')}: {k} = {v!r} ({type(v).__name__})")
    assert not errores, "\n".join(errores[:20])
    assert all(set(f) == set(COLUMNAS_CONTRATO) for f in generado.meta), "filas de metadatos con otras claves"
    assert all(set(f) == set(COLUMNAS_EXCLUIDAS_CONTRATO) for f in generado.excl), "filas de excluidas con otras claves"


def test_generar_cada_candidato_esta_una_sola_vez_entre_los_dos_csv(candidatos, generado):
    inventario_ = Counter(_clave(c) for c in candidatos)
    csvs = Counter(_clave(f) for f in generado.meta) + Counter(_clave(f) for f in generado.excl)
    assert inventario_ == csvs, (
        f"solo en el inventario: {sorted((inventario_ - csvs).elements())[:10]}; "
        f"solo en los CSV: {sorted((csvs - inventario_).elements())[:10]}")
    motivos_malos = [f for f in generado.excl if f["motivo"] not in MOTIVOS_CONTRATO]
    assert not motivos_malos, f"motivos fuera de MOTIVOS: {motivos_malos[:5]}"


def test_generar_etiquetas_y_particion_de_cada_fila(fuente, generado):
    errores = []
    for f in generado.meta:
        e = fuente.por_clave.get(_clave(f))
        if e is None:
            errores.append(f"fila inesperada {_clave(f)}")
            continue
        campos = ["archivo", "particion", "hablante", "clase", "detalle", "estilo", "distancia_cm", "velocidad_tts", "fs_fuente"]
        if f["origen"] in ("notebook", "whatsapp"):
            campos.append("codigo")
        for campo in campos:
            if f[campo] != e[campo]:
                errores.append(f"{f['fuente']} #{f['toma']}: {campo} = {f[campo]!r}, se esperaba {e[campo]!r}")
    for f in generado.excl:
        e = fuente.por_clave.get(_clave(f))
        if e is None or f["motivo"] != e["motivo"]:
            errores.append(f"excluida {_clave(f)} con motivo {f['motivo']!r}; se esperaba {e and e['motivo']!r}")
    assert not errores, "\n".join(errores[:20])


def test_generar_comandos_con_una_palabra_de_duracion_valida(fuente, generado):
    comandos = fuente.cfg["voz"]["comandos"]
    lo, hi = fuente.cfg["voz"]["duracion_palabra_s"]
    malos = []
    for f in generado.meta:
        if f["clase"] not in comandos:
            continue
        if f["n_segmentos"] != 1 or f["palabra_inicio_s"] is None or f["palabra_fin_s"] is None:
            malos.append(f"{f['archivo']}: n_segmentos {f['n_segmentos']}")
            continue
        d = f["palabra_fin_s"] - f["palabra_inicio_s"]
        if not (lo - 1e-3 - 1e-9 <= d <= hi + 1e-3 + 1e-9):
            malos.append(f"{f['archivo']}: la palabra dura {d:.3f} s, fuera de [{lo}, {hi}]")
    assert not malos, "\n".join(malos[:20])


def test_generar_medidas_del_clip_segun_el_contrato(generado):
    errores = []
    for f in generado.meta:
        y, _ = sf.read(str(generado.destino / f["archivo"]), dtype="float64")
        pico = float(np.max(np.abs(y))) if len(y) else 0.0
        rms_db = _rms_db(y)
        # El pico se mide antes de guardar y puede pasar de 1; el WAV queda recortado a ±1.
        if abs(pico - min(f["pico"], 1.0)) > 0.5e-4 + 2 * TOL_PCM:
            errores.append(f"{f['archivo']}: pico {f['pico']}, medido en el WAV {pico:.5f}")
        # Por debajo de −60 dBFS la cuantización de 16 bits mueve el nivel del WAV unas décimas.
        if f["rms_db"] > -60 and abs(rms_db - f["rms_db"]) > 0.005 + 0.01:
            errores.append(f"{f['archivo']}: rms_db {f['rms_db']}, medido en el WAV {rms_db:.3f}")
        if abs(f["duracion_s"] - (f["muestra_fin"] - f["muestra_inicio"]) / FS) > 0.5e-4 + 1e-9:
            errores.append(f"{f['archivo']}: duracion_s {f['duracion_s']} ≠ (muestra_fin − muestra_inicio)/fs")
        if f["n_segmentos"] == 0:
            if f["palabra_inicio_s"] is not None or f["palabra_fin_s"] is not None:
                errores.append(f"{f['archivo']}: sin segmentos pero con palabra_inicio_s/palabra_fin_s")
        elif not (0 <= f["palabra_inicio_s"] < f["palabra_fin_s"] <= f["duracion_s"] + 1e-3):
            errores.append(f"{f['archivo']}: palabra ({f['palabra_inicio_s']}, {f['palabra_fin_s']}) fuera del clip")
    assert not errores, "\n".join(errores[:20])


def test_generar_wav_reproducible_desde_la_fuente(audio, fuente, generado):
    cache = {}
    errores = []
    for f in generado.meta:
        if f["fuente"] not in cache:
            y, fs = audio.leer_audio(fuente.raiz / f["fuente"])
            cache[f["fuente"]] = (audio.remuestrear(y, fs, FS), fs)
        y16, fs = cache[f["fuente"]]
        if fs != f["fs_fuente"]:
            errores.append(f"{f['fuente']}: fs_fuente {f['fs_fuente']}, leer_audio da {fs}")
        esperado = np.clip(y16[f["muestra_inicio"]:f["muestra_fin"]].astype(np.float64), -1, 1)
        w, _ = sf.read(str(generado.destino / f["archivo"]), dtype="float64")
        if len(w) != len(esperado):
            errores.append(f"{f['archivo']}: {len(w)} muestras; el corte tiene {len(esperado)}")
            continue
        err = float(np.max(np.abs(w - esperado))) if len(w) else 0.0
        if err > TOL_PCM:
            errores.append(f"{f['archivo']}: diferencia {err:.2e} > 1/32767")
    assert not errores, "\n".join(errores[:20])


def test_generar_dos_veces_da_lo_mismo_byte_a_byte(ds, fuente, generado, tmp_path):
    destino = tmp_path / "dataset_b"
    ds.generar_dataset(copy.deepcopy(fuente.cfg), fuente.raiz, destino)
    a, b = _huella(generado.destino), _huella(destino)
    distintos = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert not distintos, f"archivos distintos entre las dos generaciones: {distintos[:10]}"


def test_generar_ningun_hablante_en_dos_particiones(generado):
    por_hablante = {}
    for f in generado.meta:
        por_hablante.setdefault(f["hablante"], set()).add(f["particion"])
    malos = {h: sorted(p) for h, p in por_hablante.items() if len(p) > 1}
    assert not malos, f"hablantes en más de una partición: {malos}"


@pytest.mark.parametrize("existente", ["train/viejo.wav", "val/viejo.wav", "test/viejo.wav", "metadatos.csv", "excluidas.csv"])
def test_generar_con_datos_previos_sin_sobrescribir_falla_sin_tocar_nada(ds, fuente, tmp_path, existente):
    destino = tmp_path / "dataset"
    ruta = destino / existente
    ruta.parent.mkdir(parents=True)
    ruta.write_bytes(b"datos previos")
    antes = _huella(destino)
    with pytest.raises(FileExistsError):
        ds.generar_dataset(copy.deepcopy(fuente.cfg), fuente.raiz, destino)
    assert _huella(destino) == antes, "generar_dataset cambió la carpeta destino al fallar (archivos o carpetas)"


def test_generar_con_sobrescribir_borra_solo_sus_archivos(ds, fuente, generado, tmp_path):
    destino = tmp_path / "dataset"
    ds.generar_dataset(copy.deepcopy(fuente.cfg), fuente.raiz, destino)
    ajenos = {"LEEME.txt": b"mio", "train/nota.txt": b"mia", "extra/viejo.wav": b"no es del dataset"}
    for rel, datos in ajenos.items():
        (destino / rel).parent.mkdir(parents=True, exist_ok=True)
        (destino / rel).write_bytes(datos)
    (destino / "val" / "nb_viejo.wav").write_bytes(b"wav viejo")
    ds.generar_dataset(copy.deepcopy(fuente.cfg), fuente.raiz, destino, sobrescribir=True)
    for rel, datos in ajenos.items():
        assert (destino / rel).is_file() and (destino / rel).read_bytes() == datos, f"se borró o cambió {rel}"
    assert not (destino / "val" / "nb_viejo.wav").exists(), "quedó un WAV viejo en val/"
    a = _huella(generado.destino)
    b = {k: v for k, v in _huella(destino).items() if k not in ajenos and k != "extra"}
    distintos = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert not distintos, f"después de sobrescribir, distinto de una generación nueva: {distintos[:10]}"


def test_generar_no_modifica_la_fuente(fuente, generado):
    assert _huella(fuente.raiz) == fuente.huella, "generar_dataset cambió algo en la carpeta fuente"


def _esperada_revision(filas, comandos, n, semilla):
    wa = [f["archivo"] for f in filas if f["clase"] in comandos and f["origen"] == "whatsapp"]
    archivos = wa if len(wa) >= n else [f["archivo"] for f in filas if f["clase"] in comandos]
    return random.Random(semilla).sample(sorted(archivos), min(n, len(archivos)))


@pytest.mark.parametrize("n", [3, 11, 12, 500])
def test_muestra_revision_formula_del_contrato(ds, fuente, generado, n):
    comandos = fuente.cfg["voz"]["comandos"]
    wa = [f for f in generado.meta if f["clase"] in comandos and f["origen"] == "whatsapp"]
    assert len(wa) == 11, f"la fuente de prueba debería dar 11 clips de comandos de WhatsApp; dio {len(wa)}"
    for semilla in (2026, 7):
        obtenida = list(ds.muestra_revision(copy.deepcopy(generado.meta), n, semilla))
        esperada = _esperada_revision(generado.meta, comandos, n, semilla)
        assert obtenida == esperada, f"n={n}, semilla={semilla}: {obtenida[:5]}…; se esperaba {esperada[:5]}…"


def test_muestra_revision_determinista_sin_repetidos_solo_comandos(ds, fuente, generado):
    comandos = fuente.cfg["voz"]["comandos"]
    por_archivo = {f["archivo"]: f for f in generado.meta}
    for n in (3, 30):
        a = list(ds.muestra_revision(generado.meta, n, 123))
        b = list(ds.muestra_revision(generado.meta, n, 123))
        assert a == b, "con la misma semilla dio muestras distintas"
        assert len(set(a)) == len(a), f"archivos repetidos: {a}"
        assert all(por_archivo[x]["clase"] in comandos for x in a), "hay clips que no son de comandos"
    assert all(por_archivo[x]["origen"] == "whatsapp" for x in ds.muestra_revision(generado.meta, 3, 123)), (
        "con suficientes clips de comandos de WhatsApp, la muestra tiene que salir de ellos")


# --------------------------------------------------------------------------------------------
# 7.8 Aceptación (datos reales; pytest -m aceptacion)
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real(cfg, tts, ds, tmp_path_factory):
    voz = RAIZ / cfg["rutas"]["voz"]
    sesiones = voz / "notebook" / "sesiones.csv"
    if not sesiones.is_file():
        pytest.skip(f"no están los datos reales: falta {sesiones}")
    faltan = [e["archivo"] for e in tts.plan_tts(copy.deepcopy(cfg)) if not (voz / "tts" / e["archivo"]).is_file()]
    if faltan:
        pytest.skip(f"faltan {len(faltan)} MP3 del plan de TTS (correr python -m src.voz.tts)")
    destino = tmp_path_factory.mktemp("dataset_real")
    t0 = time.perf_counter()
    meta, excl = ds.generar_dataset(copy.deepcopy(cfg), destino=destino)
    segundos = time.perf_counter() - t0
    filas = list(csv.DictReader(io.StringIO(_leer_texto_tolerante(sesiones), newline="")))
    return SimpleNamespace(voz=voz, meta=list(meta), excl=list(excl), segundos=segundos, sesiones=filas)


def _fecha(texto):
    m = re.search(r"\d{4}-\d{2}-\d{2}", texto or "")
    return m.group(0) if m else (texto or "").split("T")[0].split(" ")[0]


@pytest.mark.aceptacion
def test_aceptacion_genera_en_menos_de_2_minutos(real):
    assert real.segundos < 120, f"tardó {real.segundos:.1f} s"


@pytest.mark.aceptacion
def test_aceptacion_cada_toma_y_cada_audio_de_whatsapp_esta_en_un_csv(real):
    en_csv = {f["fuente"] for f in real.meta} | {f["fuente"] for f in real.excl}
    tomas = {f"notebook/{f['archivo']}" for f in real.sesiones}
    wa = real.voz / "whatsapp"
    audios = set()
    if wa.is_dir():
        audios = {f"whatsapp/{p.relative_to(wa).as_posix()}" for p in wa.glob("*/*")
                  if p.suffix.lower() in (".ogg", ".opus", ".m4a")}
    faltan = sorted((tomas | audios) - en_csv)
    assert not faltan, f"{len(faltan)} archivos sin fila en metadatos.csv ni en excluidas.csv: {faltan[:10]}"


@pytest.mark.aceptacion
def test_aceptacion_toma_vacia_de_p03_excluida_por_notas(real):
    filas = [f for f in real.excl if f["fuente"] == "notebook/p03/p03_identificar_02.wav"]
    assert filas and all(f["motivo"] == "notas" for f in filas), f"p03_identificar_02: {filas}"


@pytest.mark.aceptacion
@pytest.mark.parametrize("origen", ["notebook", "whatsapp"])
def test_aceptacion_exclusiones_automaticas_a_lo_sumo_5_por_ciento(cfg, real, origen):
    comandos = cfg["voz"]["comandos"]
    total = [f for f in real.meta + real.excl if f["origen"] == origen and f["clase"] in comandos]
    if not total:
        pytest.skip(f"no hay clips de comandos de {origen}")
    auto = [f for f in real.excl if f["origen"] == origen and f["clase"] in comandos and f["motivo"] != "notas"]
    frac = len(auto) / len(total)
    assert frac <= 0.05, (f"{origen}: {len(auto)} de {len(total)} clips de comandos excluidos automáticamente "
                          f"({100 * frac:.1f} %); por motivo: {dict(Counter(f['motivo'] for f in auto))}")


@pytest.mark.aceptacion
def test_aceptacion_al_menos_10_hablantes_reales(real):
    hablantes = {f["hablante"] for f in real.meta if f["origen"] in ("notebook", "whatsapp")}
    assert len(hablantes) >= 10, f"{len(hablantes)} hablantes reales: {sorted(hablantes)}"


@pytest.mark.aceptacion
def test_aceptacion_particion_de_cada_clip_segun_las_listas(cfg, real):
    part = cfg["voz"]["dataset"]["particiones"]
    malos = [(f["archivo"], f["codigo"], f["particion"]) for f in real.meta if f["origen"] in ("notebook", "whatsapp")
             and _particion_de(f["codigo"], part) != [f["particion"]]]
    assert not malos, f"clips en una partición que no es la de su código: {malos[:10]}"


@pytest.mark.aceptacion
def test_aceptacion_whatsapp_solo_en_train_y_ninguno_perdido(real):
    fuera = [f["archivo"] for f in real.meta if f["origen"] == "whatsapp" and f["particion"] != "train"]
    assert not fuera, f"clips de WhatsApp en val o test: {fuera[:10]}"
    perdidos = [f["fuente"] for f in real.excl if f["motivo"] == "whatsapp_val_test"]
    assert not perdidos, f"audios de WhatsApp excluidos por whatsapp_val_test: {perdidos[:10]}"


@pytest.mark.aceptacion
def test_aceptacion_min_por_comando_de_la_notebook_por_hablante(cfg, real):
    grupos = cfg["voz"]["dataset"]["mismo_hablante"]
    minimo = cfg["voz"]["dataset"]["min_por_comando"]
    hablantes = {_hablante(f["persona"], grupos) for f in real.sesiones if f["clase"] != "ruido" and f["persona"]}
    conteo = Counter((f["hablante"], f["clase"]) for f in real.meta if f["origen"] == "notebook")
    malos = [(h, c, conteo[(h, c)]) for h in sorted(hablantes) for c in cfg["voz"]["comandos"] if conteo[(h, c)] < minimo]
    assert not malos, f"hablantes con menos de {minimo} clips de la notebook de un comando (hablante, comando, clips): {malos}"


@pytest.mark.aceptacion
def test_aceptacion_hablantes_de_las_dos_fechas_en_cada_particion(cfg, real):
    fechas_de = {}
    for f in real.sesiones:
        if f["clase"] != "ruido" and f["persona"]:
            fechas_de.setdefault(f["persona"], set()).add(_fecha(f["fecha"]))
    todas = set().union(*fechas_de.values())
    assert len(todas) >= 2, f"fechas de grabación en sesiones.csv: {sorted(todas)}"
    for p in ("train", "val", "test"):
        codigos = {f["codigo"] for f in real.meta if f["origen"] == "notebook" and f["particion"] == p}
        fechas = set().union(*(fechas_de.get(c, set()) for c in codigos)) if codigos else set()
        assert fechas == todas, f"{p}: hablantes de las fechas {sorted(fechas)}; se esperaban {sorted(todas)}"


@pytest.mark.aceptacion
def test_aceptacion_cuatro_clases_en_cada_particion(cfg, real):
    clases = set(cfg["voz"]["comandos"]) | {cfg["voz"]["clase_rechazo"]}
    for p in ("train", "val", "test"):
        hay = {f["clase"] for f in real.meta if f["particion"] == p}
        assert clases <= hay, f"{p} no tiene clips de {sorted(clases - hay)}"


@pytest.mark.aceptacion
def test_aceptacion_tts_solo_en_train_y_menos_que_los_reales(real):
    tts_ = [f for f in real.meta if f["origen"] == "tts"]
    assert tts_ and all(f["particion"] == "train" for f in tts_), "hay clips de TTS fuera de train"
    reales = [f for f in real.meta if f["particion"] == "train" and f["origen"] in ("notebook", "whatsapp")]
    assert len(tts_) < len(reales), f"{len(tts_)} clips de TTS y {len(reales)} reales en train"
