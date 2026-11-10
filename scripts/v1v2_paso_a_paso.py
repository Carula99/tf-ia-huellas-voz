"""V1 + V2 paso a paso: de la grabación a la palabra lista para V3, con una figura por paso.

No es parte del pipeline ni tiene tests: es una herramienta para entender (y explicar) el hito
"Dataset de audio y preprocesamiento". Rehace lo que hacen src/voz/audio.py, segmentacion.py,
preprocesamiento.py y aumento.py en el mismo orden y con las mismas cuentas, pero partido en
pasos chicos, comentado, con lo que pasa en cada uno impreso y dibujado. Con --verificar compara
el resultado contra el código del repo y contra los WAV de data/voz/dataset y data/voz/prepro.

Qué hace, en orden (el número es el de la figura que guarda):
  Etapa A · V1, de las grabaciones al dataset
    01 las cuatro fuentes                 05 las reglas del detector (unir, descartar)
    02 una sesión: las 24 tomas           06 WhatsApp y control de calidad
    03 formato común: 16 kHz, mono        07 particiones por persona y conteos
    04 el detector de palabras
  Etapa B · V2, de un clip a la palabra lista para V3
    08 buscar la palabra ANTES de filtrar 12 nivel fijo: −26 dBFS
    09 pasabanda 80–4000 Hz, fase cero    13 aumento 1: velocidad (como una cinta)
    10 pre-énfasis                        14 aumento 2: ruido real, y otra vez preprocesar
    11 recorte con 40 ms de margen        15 el dataset preprocesado
  y al final: resumen.png (toda la cadena), audios/ (un WAV por etapa, para escuchar)
  y GUION.md (lo que imprime cada paso, con los números de esta corrida).

Uso, desde la raíz del repo:
    venv\\Scripts\\python scripts\\v1v2_paso_a_paso.py                       (toma p01_reconocer_01)
    venv\\Scripts\\python scripts\\v1v2_paso_a_paso.py --toma p01_salir_01
    venv\\Scripts\\python scripts\\v1v2_paso_a_paso.py --mostrar             (abre cada figura y espera que la cierres)
    venv\\Scripts\\python scripts\\v1v2_paso_a_paso.py --verificar           (compara con src/voz y los WAV del dataset)
Las figuras quedan en data/v1v2_paso_a_paso/<toma>/ (data/ no va a Git). Lee los audios de
data/voz/ y los valores de config.yaml. Necesita NumPy, SciPy, soundfile, PyAV, PyYAML y
Matplotlib, que ya están en el venv.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import sys
import textwrap
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

# se completan en main()
RAIZ: Path = Path(__file__).resolve().parents[1]
SALIDA: Path = Path(".")
MOSTRAR = False
plt = None
C: dict = {}                            # valores de config.yaml (ver cargar_valores)
GUION: list[tuple[str, list[str]]] = []  # (título del paso, párrafos) para GUION.md
PIES: list[tuple[int, str, str]] = []     # (paso, título, pie de la figura): la versión corta del guion

COLORES = {"reconocer": "#1b5fc4", "identificar": "#14873f", "salir": "#c8262a", "otro": "#7a5195"}
ORIGEN_COLOR = {"notebook": "#1b5fc4", "whatsapp": "#14873f", "tts": "#e69f00", "ruido": "#888888"}
ROJO, AZUL, VERDE, NARANJA, GRIS = "#c8262a", "#1b5fc4", "#14873f", "#e69f00", "#888888"
ESTILO = {"rapido": "rápido"}                    # sesiones.csv guarda los estilos sin tilde


# ============================================================================= configuración
def cargar_valores(raiz: Path) -> dict:
    """Los valores de config.yaml que usan V1 y V2 (así este script no se desincroniza)."""
    import yaml
    with open(raiz / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    v = cfg["voz"]
    d = v["dataset"]
    p = v["preprocesamiento"]
    return dict(
        cfg=cfg, fs=int(v["frecuencia_muestreo"]), comandos=list(v["comandos"]),
        dur_palabra=tuple(v["duracion_palabra_s"]), banda=tuple(v["pasabanda_hz"]),
        a=float(v["preenfasis"]), det=d["deteccion"], clip_max=float(d["clip_max_s"]),
        margen_wa=float(d["whatsapp"]["margen_s"]), saturacion=float(d["saturacion"]),
        saturacion_frac=float(d["saturacion_frac"]), particiones=d["particiones"],
        mismo_hablante=d["mismo_hablante"], tts_voces=list(d["tts"]["voces"]),
        orden=int(p["orden_filtro"]), margen_ms=float(p["margen_ms"]),
        nivel_db=float(p["nivel_rms_db"]), copias=int(p["aumento"]["copias"]),
        velocidad=tuple(p["aumento"]["velocidad"]), snr=tuple(p["aumento"]["snr_db"]),
        semilla=int(cfg["semilla_global"]), voz=raiz / cfg["rutas"]["voz"],
    )


# ============================================================================= utilidades
def es(texto: str) -> str:
    """Coma decimal (1.33 → 1,33). No toca nombres de archivo: ahí al punto lo sigue una letra."""
    return re.sub(r"(?<=\d)\.(?=\d)", ",", texto)


def hablar(texto: str = "") -> None:
    """Imprime un párrafo cortado a 100 columnas y lo guarda para GUION.md."""
    texto = es(texto)
    if GUION:
        GUION[-1][1].append(texto)
    for parrafo in texto.split("\n"):
        print(textwrap.fill(parrafo, 100, subsequent_indent="   ") if parrafo.strip() else "")


def encabezado(n: int, titulo: str) -> None:
    titulo = es(titulo)
    GUION.append((f"Paso {n} · {titulo}" if n else titulo, []))
    print("\n" + "=" * 100)
    print(f"PASO {n} · {titulo}" if n else titulo)
    print("=" * 100)


def figura(filas: int, columnas: int, ancho: float, alto: float, **kw):
    fig, ejes = plt.subplots(filas, columnas, figsize=(ancho, alto), squeeze=False, **kw)
    return fig, ejes


def cerrar(fig, n: int, nombre: str, titulo: str, pie: str = "") -> None:
    """Título del paso arriba, explicación corta abajo; guarda (y muestra con --mostrar)."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):                     # títulos, leyendas y anotaciones con coma decimal
        if t.get_text():
            t.set_text(es(t.get_text()))
    fig.suptitle(es(f"Paso {n} · {titulo}"), fontsize=14, fontweight="bold", x=0.012, ha="left")
    PIES.append((n, es(titulo), es(pie)))
    alto = fig.get_figheight()
    inferior = 0.0
    if pie:
        envuelto = "\n".join(textwrap.wrap(es(pie), max(40, int(fig.get_figwidth() * 12.5))))
        fig.text(0.012, 0.15 / alto, envuelto, fontsize=9.5, ha="left", va="bottom", color="#333333")
        inferior = (0.3 + 0.17 * (envuelto.count("\n") + 1)) / alto
    superior = 1 - 0.55 / alto
    fig.tight_layout(rect=[0, inferior, 1, superior])
    fig.canvas.draw()
    fig.tight_layout(rect=[0, inferior, 1, superior], h_pad=1.4, w_pad=1.2)
    ruta = SALIDA / f"{n:02d}_{nombre}.png"
    fig.savefig(ruta, dpi=110)
    GUION[-1][1].insert(0, f"![]({ruta.name})")
    print(f"   [figura guardada: {ruta}]")
    if MOSTRAR:
        plt.show()
    plt.close(fig)


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def rms(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean(y * y))) if y.size else 0.0


def onda(ax, y, fs, t0: float = 0.0, color="#333333", lw=0.6, **kw) -> None:
    t = t0 + np.arange(len(y)) / fs
    ax.plot(t, y, color=color, lw=lw, **kw)
    ax.set_xlim(t0, t0 + len(y) / fs)


def sombrear(ax, ini, fin, color=VERDE, alpha=0.18, **kw) -> None:
    ax.axvspan(ini, fin, color=color, alpha=alpha, lw=0, **kw)


def espectrograma(ax, y, fs, fmax=None, t0=0.0, rango_db=80.0, tope=None):
    """Espectrograma en dB (tramas de 25 ms cada 10 ms, ventana de Hann). tope: el 0 de la escala de colores."""
    n, h = int(0.025 * fs), int(0.010 * fs)
    f, t, Z = scipy.signal.stft(np.asarray(y, np.float64), fs, window="hann", nperseg=n,
                                noverlap=n - h, boundary=None, padded=False)
    P = 20 * np.log10(np.abs(Z) + 1e-10)
    tope = P.max() if tope is None else tope
    im = ax.imshow(P, origin="lower", aspect="auto", cmap="magma", vmin=tope - rango_db, vmax=tope,
                   extent=[t0 + t[0] - h / 2 / fs, t0 + t[-1] + h / 2 / fs, 0, fs / 2])
    ax.set_ylim(0, fmax or fs / 2)
    ax.set_ylabel("Hz")
    return im


def guardar_audio(nombre: str, y: np.ndarray, fs: int) -> Path:
    """WAV float32 (no recorta: lo que pasa de ±1 por el pre-énfasis queda igual)."""
    (SALIDA / "audios").mkdir(exist_ok=True)
    ruta = SALIDA / "audios" / nombre
    sf.write(str(ruta), np.asarray(y, np.float32), fs, subtype="FLOAT")
    return ruta


def leer_csv(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================================= V1: las funciones
def leer_audio(ruta: Path) -> tuple[np.ndarray, int]:
    """Como src/voz/audio.leer_audio: soundfile; si no puede (los .m4a), PyAV. Mono, [−1, 1]."""
    try:
        y, fs = sf.read(str(ruta), dtype="float32", always_2d=True)
        y = y.mean(axis=1, dtype=np.float64).astype(np.float32) if y.shape[1] > 1 else y[:, 0]
    except (sf.LibsndfileError, RuntimeError, TypeError):
        import av
        with av.open(str(ruta)) as cont:
            flujo = cont.streams.audio[0]
            partes = []
            for cuadro in cont.decode(flujo):
                x = cuadro.to_ndarray()
                x = x.T if cuadro.format.is_planar else x.reshape(-1, len(cuadro.layout.channels))
                if np.issubdtype(x.dtype, np.integer):
                    x = x.astype(np.float64) / float(-np.iinfo(x.dtype).min)
                partes.append(x.astype(np.float32))
            fs = int(flujo.rate)
        x = np.concatenate(partes, axis=0)
        y = x.mean(axis=1, dtype=np.float64).astype(np.float32) if x.shape[1] > 1 else x[:, 0]
    return np.clip(np.ascontiguousarray(y.reshape(-1)), -1.0, 1.0), int(fs)


def remuestrear(y: np.ndarray, fs_origen: int, fs_destino: int) -> np.ndarray:
    """resample_poly con up/down reducidos por el MCD (44100 → 16000: 160/441)."""
    if fs_origen == fs_destino:
        return np.asarray(y, np.float32).copy()
    g = math.gcd(fs_origen, fs_destino)
    return scipy.signal.resample_poly(np.asarray(y, np.float64), fs_destino // g, fs_origen // g).astype(np.float32)


def a_16_bits(y: np.ndarray, fs: int) -> np.ndarray:
    """Lo que hace guardar_wav de V1 (recortar a ±1 y PCM de 16 bits), leído de vuelta."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(y, np.float64), -1.0, 1.0), fs, subtype="PCM_16", format="WAV")
    buf.seek(0)
    return sf.read(buf, dtype="float32")[0].reshape(-1)


def tramas() -> tuple[int, int]:
    d, fs = C["det"], C["fs"]
    return int(round(d["trama_ms"] * fs / 1000)), int(round(d["avance_ms"] * fs / 1000))


def envolvente(y: np.ndarray) -> np.ndarray:
    """Nivel de cada trama en dBFS: 20·log10(máx(rms_k, 1e-6)). N = 320 muestras, H = 160."""
    y = np.asarray(y, np.float64).reshape(-1)
    n, h = tramas()
    if len(y) < n:
        return np.zeros(0)
    k = 1 + (len(y) - n) // h
    vent = np.lib.stride_tricks.sliding_window_view(y * y, n)[::h][:k]
    return 20.0 * np.log10(np.maximum(np.sqrt(vent.mean(axis=1)), 1e-6))


def detectar(y: np.ndarray) -> dict:
    """El detector de V1 (contrato v1_dataset.md, sección 0), devolviendo todo lo intermedio."""
    d, fs = C["det"], C["fs"]
    n, h = tramas()
    L = envolvente(y)
    R = dict(niveles=L, tope=None, piso=None, umbral=None, corridas=[], unidas=[], segmentos=[], descartes=[])
    if L.size == 0:
        return R
    tope = float(L.max())
    R["tope"] = tope
    if tope < d["nivel_min_db"]:
        return R
    piso = float(np.percentile(L, 10))
    umbral = max(tope - d["rango_db"], piso + d["margen_piso_db"])
    R.update(piso=piso, umbral=umbral)
    activa = (L > umbral).astype(np.int8)
    cambios = np.diff(np.concatenate(([0], activa, [0])))
    corridas = [[int(i), int(f)] for i, f in zip(np.flatnonzero(cambios == 1), np.flatnonzero(cambios == -1) - 1)]
    R["corridas"] = [tuple(c) for c in corridas]
    unidas: list[list[int]] = []
    for k0, k1 in corridas:                          # unir si el hueco dura menos que pausa_min_s
        if unidas and (k0 * h - (unidas[-1][1] * h + n)) / fs < d["pausa_min_s"]:
            unidas[-1][1] = k1
        else:
            unidas.append([k0, k1])
    R["unidas"] = [tuple(u) for u in unidas]
    for k0, k1 in unidas:
        ini, fin = k0 * h / fs, min(k1 * h + n, len(y)) / fs
        if fin - ini < d["segmento_min_s"]:
            R["descartes"].append((ini, fin, "corto"))
        elif L[k0:k1 + 1].max() < tope - d["dif_palabra_db"]:
            R["descartes"].append((ini, fin, "débil"))
        else:
            R["segmentos"].append((ini, fin))
    return R


def palabra_principal(y: np.ndarray) -> tuple[tuple[float, float] | None, int, dict]:
    """De los segmentos, el que tiene la trama más fuerte (V2, sección 0)."""
    R = detectar(y)
    if not R["segmentos"]:
        return None, 0, R
    _, h = tramas()
    t = np.arange(len(R["niveles"])) * h / C["fs"]
    mejor, nivel = None, -np.inf
    for ini, fin in R["segmentos"]:
        m = (t >= ini) & (t < fin)
        v = float(R["niveles"][m].max()) if m.any() else -np.inf
        if v > nivel:
            mejor, nivel = (float(ini), float(fin)), v
    return mejor, len(R["segmentos"]), R


# ============================================================================= V2: las funciones
def pasabanda(y: np.ndarray) -> np.ndarray:
    sos = scipy.signal.butter(C["orden"], C["banda"], btype="bandpass", fs=C["fs"], output="sos")
    return scipy.signal.sosfiltfilt(sos, np.asarray(y, np.float64))   # ida y vuelta: fase cero


def preenfasis(x: np.ndarray, a: float) -> np.ndarray:
    x = np.asarray(x, np.float64)
    z = x.copy()
    z[1:] = x[1:] - a * x[:-1]                     # z[n] = x[n] − a·x[n−1]
    return z


def margen_muestras() -> int:
    return int(round(C["margen_ms"] * C["fs"] / 1000))


def recortar(x: np.ndarray, ini: float, fin: float) -> np.ndarray:
    fs, M = C["fs"], margen_muestras()
    m0, m1 = int(round(ini * fs)), int(round(fin * fs))
    clip = np.zeros(m1 - m0 + 2 * M)
    a, b = max(0, m0 - M), min(len(x), m1 + M)
    if b > a:
        clip[a - (m0 - M):b - (m0 - M)] = x[a:b]      # ceros donde se sale del audio
    return clip


def normalizar(clip: np.ndarray) -> tuple[np.ndarray, float]:
    M = margen_muestras()
    r = rms(clip[M:len(clip) - M])                  # RMS solo de la palabra
    g = 10 ** (C["nivel_db"] / 20) / r
    return (np.asarray(clip, np.float64) * g).astype(np.float32), g


def preprocesar(y: np.ndarray) -> dict:
    """La cadena de V2 para un audio a 16 kHz. Lo mismo que se usa en vivo en la app."""
    pal, nseg, R = palabra_principal(y)
    P = dict(palabra=pal, n_segmentos=nseg, det=R, valido=False, motivo="sin_voz", y=np.zeros(0, np.float32))
    if pal is None:
        return P
    if not C["dur_palabra"][0] <= pal[1] - pal[0] <= C["dur_palabra"][1]:
        P["motivo"] = "duracion"
        return P
    filtrado = pasabanda(y)
    acond = preenfasis(filtrado, C["a"])
    clip = recortar(acond, *pal)
    norm, g = normalizar(clip)
    P.update(valido=True, motivo="", filtrado=filtrado, acond=acond, clip=clip, y=norm, ganancia=g)
    return P


def semilla_copia(archivo: str, copia: int) -> np.random.Generator:
    return np.random.default_rng([C["semilla"], zlib.crc32(archivo.encode("utf-8")), copia])


def sortear_aumento(rng: np.random.Generator, n_ruidos: int) -> dict:
    return dict(factor=round(float(rng.uniform(*C["velocidad"])), 2), snr_db=float(rng.uniform(*C["snr"])),
                ruido=int(rng.integers(n_ruidos)), desplazamiento=float(rng.random()))


def cambiar_velocidad(y: np.ndarray, factor: float) -> np.ndarray:
    d = int(round(100 * round(factor, 2)))
    if d == 100:
        return np.asarray(y, np.float32).copy()
    g = math.gcd(100, d)
    return scipy.signal.resample_poly(np.asarray(y, np.float64), 100 // g, d // g).astype(np.float32)


def sumar_ruido(y, ruido, snr_db, palabra) -> tuple[np.ndarray, float, float, float]:
    fs = C["fs"]
    y64, r = np.asarray(y, np.float64), np.asarray(ruido, np.float64)[:len(y)]
    m0 = max(0, min(len(y64), int(round(palabra[0] * fs))))
    m1 = max(m0, min(len(y64), int(round(palabra[1] * fs))))
    ps, pn = float(np.mean(y64[m0:m1] ** 2)), float(np.mean(r ** 2))
    g = math.sqrt(ps / (pn * 10 ** (snr_db / 10)))   # 10·log10(Ps / (g²·Pn)) = SNR
    return (y64 + g * r).astype(np.float32), g, ps, pn


def aumentar(y, palabra, ruidos, params) -> dict:
    """La copia aumentada, sin preprocesar, con todo lo intermedio (V2, sección 4)."""
    fs, f = C["fs"], params["factor"]
    z = cambiar_velocidad(y, f)
    ini, fin = palabra[0] / f, palabra[1] / f
    i0 = 0
    cmax = int(round(C["clip_max"] * fs))
    if len(z) > cmax:                                # más lento quedó > 3 s: 3 s centrados en la palabra
        c = int(round((ini + fin) / 2 * fs))
        i0 = min(max(0, c - cmax // 2), len(z) - cmax)
        z = z[i0:i0 + cmax]
        ini, fin = ini - i0 / fs, fin - i0 / fs
    r = ruidos[params["ruido"]]
    off = int(math.floor(params["desplazamiento"] * (len(r) - len(z) + 1)))
    con_ruido, g, ps, pn = sumar_ruido(z, r[off:], params["snr_db"], (ini, fin))
    return dict(veloz=z, palabra=(ini, fin), recorte_i0=i0, offset=off, g=g, ps=ps, pn=pn, y=con_ruido)


# ============================================================================= ETAPA A · V1
def paso_1_fuentes(E: dict) -> None:
    encabezado(1, "Las cuatro fuentes de audio")
    M = E["meta"]
    n_or = Counter(r["origen"] for r in M)
    n_ex = Counter(r["origen"] for r in E["excluidas"])
    fila_wa = next((r for r in M if r["origen"] == "whatsapp" and r["particion"] == "train" and r["clase"] == E["clase"]),
                   next(r for r in M if r["origen"] == "whatsapp"))
    fila_tts = next((r for r in M if r["origen"] == "tts" and r["clase"] == E["clase"]), next(r for r in M if r["origen"] == "tts"))
    fila_ru = next(r for r in M if r["origen"] == "ruido" and r["particion"] == "train")
    E.update(fila_wa=fila_wa)
    ejemplos = [("notebook", E["fuente_rel"]), ("whatsapp", fila_wa["fuente"]),
                ("tts", fila_tts["fuente"]), ("ruido", fila_ru["fuente"])]
    sesiones = {r["persona"] for r in E["sesiones"] if r["clase"] != "ruido"}
    personas = len(sesiones) - sum(len(g) - 1 for g in C["mismo_hablante"])
    wa_cod = {r["codigo"] for r in M + E["excluidas"] if r["origen"] == "whatsapp"}
    hablar(f"El carril de voz no es sintético: el K-NN tiene que entender a personas reales. V1 junta "
           f"audios de cuatro fuentes y los deja todos en el mismo formato:")
    hablar(f"· notebook: el micrófono de la demo. {len(sesiones)} sesiones de {personas} personas "
           f"(p01 y p02, p09 y p13, p14 y p17 son la misma persona), 24 tomas de 3 s cada una → "
           f"{n_or['notebook'] + n_ex['notebook']} tomas.\n"
           f"· whatsapp: notas de voz y grabaciones del celular (OGG/Opus y M4A) de {len(wa_cod)} personas, "
           f"{len(wa_cod - sesiones)} de ellas solo por WhatsApp → {n_or['whatsapp'] + n_ex['whatsapp']} audios.\n"
           f"· tts: {len(C['tts_voces'])} voces sintéticas (edge-tts; Argentina, Uruguay, Chile, México, España y "
           f"Colombia, mujer y varón) × 6 palabras × 2 velocidades = {n_or['tts'] + n_ex['tts']} clips, solo en train.\n"
           f"· ruido: 5 grabaciones de 90 s del ambiente, sin hablar → 10 clips de 3 s por archivo = {n_or['ruido']} clips "
           f"(clase «otro», y la materia prima del aumento de V2).")
    fig, ejes = figura(4, 1, 13, 9.6)
    for ax, (origen, rel) in zip(ejes[:, 0], ejemplos):
        ruta = C["voz"] / rel
        y, fs = leer_audio(ruta)
        try:
            info = sf.info(str(ruta))
            formato = f"{info.format} {info.subtype}".replace("MPEG_LAYER_III", "")
        except (sf.LibsndfileError, RuntimeError, TypeError):
            formato = {".m4a": "M4A (AAC)", ".ogg": "OGG Opus"}.get(ruta.suffix.lower(), ruta.suffix)
        onda(ax, y, fs, color=ORIGEN_COLOR[origen])
        extra = "   ·   naranja: los 10 clips de 3 s que salen de esta grabación" if origen == "ruido" else ""
        ax.set_title(f"{origen}: {rel}   ·   {formato}, {fs} Hz, {len(y) / fs:.2f} s{extra}", fontsize=10, loc="left")
        ax.set_ylabel("amplitud")
        if origen == "ruido":                        # dónde caen los 10 clips de 3 s
            for r in M:
                if r["fuente"] == rel:
                    sombrear(ax, int(r["muestra_inicio"]) / C["fs"], int(r["muestra_fin"]) / C["fs"], color=NARANJA, alpha=0.25)
        if origen == "notebook":
            E["crudo"], E["fs_crudo"] = y, fs
    ejes[-1, 0].set_xlabel("tiempo (s)")
    cerrar(fig, 1, "fuentes", "Las cuatro fuentes de audio",
           "Cada fuente llega distinta: WAV a 44,1 kHz (notebook), Opus a 48 o 16 kHz y AAC (WhatsApp), MP3 a 24 kHz (TTS) "
           "y ruido de 90 s. V1 las lee sin modificarlas y las lleva a un único formato: WAV de 16 kHz, mono, 16 bits, una palabra por archivo.")


def paso_2_sesion(E: dict) -> None:
    encabezado(2, f"Una sesión de la notebook: las 24 tomas de {E['codigo']}")
    filas = [r for r in E["sesiones"] if r["persona"] == E["codigo"] and r["clase"] != "ruido"]
    excl = {r["fuente"]: r["motivo"] for r in E["excluidas"]}
    orden = E["comandos"] + ["otro"]
    fig, ejes = figura(4, 6, 15, 8.2, sharey=True)
    for fila_i, clase in enumerate(orden):
        tomas = sorted((r for r in filas if r["clase"] == clase), key=lambda r: int(r["toma"]))
        for col, r in enumerate(tomas[:6]):
            ax = ejes[fila_i, col]
            y, fs = leer_audio(C["voz"] / "notebook" / r["archivo"])
            y16 = remuestrear(y, fs, C["fs"])
            onda(ax, y16, C["fs"], color=COLORES[clase], lw=0.4)
            for ini, fin in detectar(y16)["segmentos"]:
                sombrear(ax, ini, fin, color=COLORES[clase], alpha=0.15)
            etiqueta = (f"{ESTILO.get(r['estilo'], r['estilo'])} ({r['distancia_cm']} cm)" if clase != "otro"
                        else r["detalle"].replace("_", " "))
            rel = "notebook/" + r["archivo"]
            if rel in excl:
                etiqueta += f"  ✘ {excl[rel]}"
            ax.set_title(etiqueta, fontsize=8.5, color=ROJO if rel in excl else "#222222")
            ax.set_xticks([0, 1, 2, 3]); ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(clase, fontsize=10, color=COLORES[clase], fontweight="bold")
            if rel == E["fuente_rel"]:
                for s in ax.spines.values():
                    s.set_edgecolor("black"); s.set_linewidth(2.5)
    hablar("Cada persona grabó 24 tomas de 3 s con scripts/grabar_voces.py, todas con el micrófono de la notebook de la "
           "demo: «reconocer», «identificar» y «salir» en 6 estilos (normal dos veces a 50 cm, lento, rápido, lejos a "
           "80 cm y cerca a 30 cm) y 6 tomas de la clase «otro»: palabras parecidas a los comandos (conocer, salida, "
           "identidad), «hola», una palabra libre y una tos o un golpe.")
    hablar("Los estilos están para que el K-NN vea la misma palabra dicha de varias maneras, como la van a decir los "
           "profesores. Las palabras parecidas son las más importantes de «otro»: le enseñan al K-NN dónde termina un comando.")
    hablar(f"Lo sombreado es lo que encontró el detector de palabras (paso 4). El recuadro negro es la toma que sigue "
           f"este paso a paso: {E['fuente_rel']}.")
    cerrar(fig, 2, "sesion", f"Una sesión de la notebook: las 24 tomas de {E['codigo']}",
           "Cada persona grabó 24 tomas de 3 s con el micrófono de la demo: los 3 comandos en 6 estilos (columnas) y 6 de "
           "«otro» (palabras parecidas, «hola», una libre y una tos). Sombreado: la palabra que encontró el detector; en "
           "rojo, si la toma se excluyó y por qué.")


def paso_3_formato(E: dict) -> None:
    encabezado(3, "Formato común: 16 kHz, mono, 16 bits")
    x, fs0, fs = E["crudo"], E["fs_crudo"], C["fs"]
    g = math.gcd(fs0, fs)
    y = remuestrear(x, fs0, fs)
    y16 = a_16_bits(y, fs)
    E.update(v1=y16, v1_float=y)
    err = float(np.abs(np.clip(y, -1, 1) - y16).max())
    hablar(f"La toma se grabó a {fs0} Hz: {len(x)} muestras en {len(x) / fs0:.1f} s. Se pasa a {fs} Hz con "
           f"scipy.signal.resample_poly(y, up, down), con up/down = {fs}/{fs0} reducidos por el máximo común divisor "
           f"({g}): up = {fs // g}, down = {fs0 // g}. Por dentro intercala {fs // g - 1} ceros entre muestra y muestra, "
           f"filtra con un pasabajos FIR (el antialias) y se queda con 1 de cada {fs0 // g}. Quedan {len(y)} muestras.")
    f1, p1 = scipy.signal.welch(x.astype(np.float64), fs0, nperseg=2048)
    arriba = 10 * math.log10(p1[f1 > fs / 2].sum() / p1.sum() + 1e-30)
    hablar(f"¿Por qué 16 kHz? Por Nyquist, a fs = 16 kHz se conserva todo hasta 8 kHz, y la voz que importa para "
           f"distinguir palabras está por debajo (la telefonía de banda ancha usa lo mismo). Son {fs0 / fs:.2f} veces "
           f"menos muestras que procesar. Un estéreo se pasa a mono con el promedio de los canales.")
    hablar(f"En esta toma, la potencia por encima de 8 kHz es {arriba:.0f} dB respecto del total: el procesamiento de "
           f"voz de Windows ya entregaba la señal sin nada arriba de 8 kHz, así que pasar a 16 kHz no pierde nada."
           if arriba < -60 else
           f"En esta toma, la potencia por encima de 8 kHz es {arriba:.0f} dB respecto del total (una parte en "
           f"{10 ** (-arriba / 10):.0f}): lo que no entra a 16 kHz es muy poco.")
    hablar(f"Al guardar el WAV de 16 bits (PCM_16) cada muestra se redondea a un escalón de 1/32767: el WAV difiere del "
           f"audio remuestreado en a lo sumo un escalón (acá, {err * 32767:.2f}/32767). Ese WAV es lo que lee V2.")
    fig, ejes = figura(1, 2, 14, 4.8)
    ax = ejes[0, 0]
    ini, fin = E["v1_palabra_meta"] or (1.4, 1.6)      # sin palabra (una tos): el medio del audio
    t_c = (ini + fin) / 2
    ventana = 0.003
    a0, a1 = int((t_c) * fs0), int((t_c + ventana) * fs0)
    b0, b1 = int(t_c * fs), int((t_c + ventana) * fs)
    ax.plot(np.arange(a0, a1) / fs0 * 1000, x[a0:a1], "-o", ms=3, color=GRIS, lw=0.8, label=f"{fs0} Hz ({a1 - a0} muestras)")
    ax.plot(np.arange(b0, b1) / fs * 1000, y16[b0:b1], "o", ms=6, color=AZUL, label=f"{fs} Hz ({b1 - b0} muestras)")
    ax.set_xlabel("tiempo (ms)"); ax.set_ylabel("amplitud"); ax.legend(fontsize=9)
    ax.set_title(f"3 ms en el medio de la palabra: menos puntos, la misma forma", fontsize=10)
    ax = ejes[0, 1]
    f2, p2 = scipy.signal.welch(y16.astype(np.float64), fs, nperseg=744)
    ax.plot(f1 / 1000, 10 * np.log10(p1 + 1e-20), color=GRIS, lw=3, alpha=0.55, label=f"crudo, {fs0} Hz")
    ax.plot(f2 / 1000, 10 * np.log10(p2 + 1e-20), color=AZUL, lw=1, label=f"{fs} Hz")
    ax.axvspan(fs / 2000, fs0 / 2000, color=ROJO, alpha=0.07)
    ax.axvline(fs / 2000, color=ROJO, ls="--", lw=1)
    ax.text(fs / 2000 + 0.3, ax.get_ylim()[1] - 8, f"Nyquist: {fs // 2000} kHz\nlo de acá a la derecha\nno entra a 16 kHz "
            f"({arriba:.0f} dB\nde la potencia total)", color=ROJO, fontsize=9, va="top")
    ax.set_xlabel("frecuencia (kHz)"); ax.set_ylabel("densidad espectral (dB)"); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("espectro promedio de los 3 s (Welch)", fontsize=10)
    cerrar(fig, 3, "formato", "Formato común: 16 kHz, mono, 16 bits",
           f"resample_poly con up/down = {fs // g}/{fs0 // g}: pasabajos antialias y diezmado en un solo paso. "
           f"Hasta {fs // 2000} kHz el espectro es el mismo (las dos curvas se superponen); lo de arriba no entra a {fs // 1000} kHz.")


def _dibujar_detector(ax_onda, ax_niv, y, R, titulo, mostrar_corridas=True) -> None:
    fs = C["fs"]
    n, h = tramas()
    onda(ax_onda, y, fs, lw=0.5)
    for ini, fin in R["segmentos"]:
        sombrear(ax_onda, ini, fin)
    for ini, fin, motivo in R["descartes"]:
        sombrear(ax_onda, ini, fin, color=ROJO, alpha=0.15)
    ax_onda.set_title(titulo, fontsize=10, loc="left")
    ax_onda.set_ylabel("amplitud")
    L = R["niveles"]
    t = (np.arange(len(L)) * h + n / 2) / fs
    ax_niv.plot(t, L, color="#333333", lw=1)
    if R["umbral"] is not None:
        ax_niv.axhline(R["tope"], color=ROJO, lw=1, ls="--", label=f"tope {R['tope']:.1f} dB")
        ax_niv.axhline(R["piso"], color=GRIS, lw=1, ls=":", label=f"piso (percentil 10) {R['piso']:.1f} dB")
        ax_niv.axhline(R["umbral"], color=AZUL, lw=1.4, label=f"umbral {R['umbral']:.1f} dB")
        ax_niv.axhline(R["tope"] - C["det"]["dif_palabra_db"], color=NARANJA, lw=1, ls="-.",
                       label=f"tope − {C['det']['dif_palabra_db']:.0f} dB (dominancia)")
        if mostrar_corridas:
            for k0, k1 in R["corridas"]:
                ax_niv.plot([(k0 * h + n / 2) / fs, (k1 * h + n / 2) / fs], [R["umbral"]] * 2, color=AZUL, lw=5, alpha=0.5,
                            solid_capstyle="butt")
    for ini, fin in R["segmentos"]:
        sombrear(ax_niv, ini, fin)
    ax_niv.set_ylabel("nivel (dBFS)")
    ax_niv.set_ylim(max(-125, (R["piso"] or -120) - 8), (R["tope"] or 0) + 6)
    ax_niv.set_xlim(0, len(y) / fs)


def paso_4_detector(E: dict) -> None:
    encabezado(4, "El detector de palabras")
    d, fs = C["det"], C["fs"]
    n, h = tramas()
    y = E["v1_float"]
    R = detectar(y)
    E["det_v1"] = R
    K = len(R["niveles"])
    hablar(f"El detector mira la energía, no el espectro. Corta el audio en tramas de {d['trama_ms']} ms (N = {n} "
           f"muestras) que avanzan de a {d['avance_ms']} ms (H = {h}): K = 1 + (len − N) // H = 1 + ({len(y)} − {n}) // "
           f"{h} = {K} tramas. A cada una le calcula el nivel L_k = 20·log10(rms_k) en dBFS (con piso de −120 dB).")
    hablar(f"Los umbrales son RELATIVOS a la trama más fuerte (el tope), así no importa si la persona habló fuerte o "
           f"bajo ni qué tan sensible es el micrófono:\n"
           f"   tope = máx L_k = {R['tope']:.1f} dB · piso = percentil 10 de L_k = {R['piso']:.1f} dB\n"
           f"   umbral = máx(tope − {d['rango_db']:.0f}, piso + {d['margen_piso_db']:.0f}) = máx({R['tope'] - d['rango_db']:.1f}, "
           f"{R['piso'] + d['margen_piso_db']:.1f}) = {R['umbral']:.1f} dB")
    hablar(f"Una trama está activa si L_k > umbral. En esta toma hay {len(R['corridas'])} corrida(s) de tramas activas → "
           f"{len(R['segmentos'])} palabra(s): " + ", ".join(f"de {a:.2f} a {b:.2f} s ({b - a:.2f} s)" for a, b in R["segmentos"]) + ".")
    if R["piso"] < -90:
        hablar(f"El piso da {R['piso']:.0f} dB porque Windows tiene activa la supresión de ruido: entre palabra y palabra "
               f"el micrófono entrega casi ceros. Por eso acá manda tope − {d['rango_db']:.0f}; en los audios de WhatsApp, "
               f"con ruido de fondo, suele mandar piso + {d['margen_piso_db']:.0f}.")
    fig, ejes = figura(2, 1, 13, 7.4, sharex=True)
    _dibujar_detector(ejes[0, 0], ejes[1, 0], y, R, f"{E['nombre_v1']} (16 kHz)")
    ejes[1, 0].legend(fontsize=8.5, loc="upper right")
    ejes[1, 0].set_xlabel("tiempo (s)")
    ejes[1, 0].set_title("nivel de cada trama de 20 ms; barras azules: corridas de tramas activas", fontsize=10, loc="left")
    cerrar(fig, 4, "detector", "El detector de palabras",
           f"umbral = máx(tope − {d['rango_db']:.0f} dB, piso + {d['margen_piso_db']:.0f} dB). Verde: la palabra. "
           "Como todo se mide contra el tope, multiplicar el audio por una constante no cambia nada.")


def paso_5_reglas(E: dict) -> None:
    encabezado(5, "Las reglas del detector: unir, descartar, dominancia")
    d, fs = C["det"], C["fs"]
    n, h = tramas()
    # un caso con una pausa interna (se une), una tos y una toma excluida por dos palabras
    candidatos = sorted((r for r in E["sesiones"] if r["persona"] == E["codigo"] and r["clase"] in E["comandos"]),
                        key=lambda r: (r["estilo"] != "lento", r["archivo"]))
    union = None
    for r in candidatos:
        y, fs0 = leer_audio(C["voz"] / "notebook" / r["archivo"])
        R = detectar(remuestrear(y, fs0, fs))
        if len(R["corridas"]) > len(R["unidas"]):
            union = (r, remuestrear(y, fs0, fs), R)
            break
    tos = next((r for r in E["sesiones"] if r["persona"] == E["codigo"] and r["detalle"] == "no_verbal"), None)
    varias = next((r for r in E["excluidas"] if r["motivo"] == "varias_palabras" and (C["voz"] / r["fuente"]).exists()), None)
    casos = []
    if union:
        r, y, R = union
        k = next(i for i in range(1, len(R["corridas"]))
                 if R["corridas"][i][0] * h - (R["corridas"][i - 1][1] * h + n) < d["pausa_min_s"] * fs)
        hueco = (R["corridas"][k][0] * h - (R["corridas"][k - 1][1] * h + n)) / fs
        casos.append((y, R, f"{r['archivo']} ({r['estilo']}): {len(R['corridas'])} corridas, hueco de {hueco * 1000:.0f} ms "
                            f"< {d['pausa_min_s'] * 1000:.0f} ms → 1 palabra"))
        hablar(f"Unir: en {r['archivo']} (estilo {r['estilo']}) hay {len(R['corridas'])} corridas de tramas activas con un "
               f"hueco de {hueco * 1000:.0f} ms. Las consonantes como la «t» o la «c» son un cierre: un silencio cortito "
               f"DENTRO de la palabra. Si el hueco dura menos de pausa_min_s = {d['pausa_min_s']} s, es la misma palabra.")
    if tos:
        y, fs0 = leer_audio(C["voz"] / "notebook" / tos["archivo"])
        y = remuestrear(y, fs0, fs)
        R = detectar(y)
        casos.append((y, R, f"{tos['archivo']} (tos o golpe): {len(R['segmentos'])} segmento(s); en «otro» se acepta igual"))
        hablar(f"Descartar: un segmento que dura menos de segmento_min_s = {d['segmento_min_s']} s (un clic) o cuya trama "
               f"más fuerte no llega a tope − {d['dif_palabra_db']:.0f} dB (una voz de fondo, un golpe suave) no es una "
               f"palabra. En la tos de {E['codigo']} el detector da {len(R['segmentos'])} segmento(s): para la clase «otro» no "
               f"se exige palabra, entra igual.")
    if varias:
        y, fs0 = leer_audio(C["voz"] / varias["fuente"])
        y = remuestrear(y, fs0, fs)
        R = detectar(y)
        casos.append((y, R, f"{varias['fuente'].split('/')[-1]}: {len(R['segmentos'])} segmentos → se excluye (varias_palabras)"))
        t_tr = np.arange(len(R["niveles"])) * h / fs
        picos = [float(R["niveles"][(t_tr >= a) & (t_tr < b)].max()) for a, b in R["segmentos"]]
        hablar(f"Dominancia: un segmento cuenta como palabra solo si su trama más fuerte está a menos de "
               f"{d['dif_palabra_db']:.0f} dB del tope; así una voz de fondo o un golpe suave no se toman como una segunda "
               f"palabra. En {varias['fuente']} los dos segmentos (" +
               "; ".join(f"{a:.2f}–{b:.2f} s, con pico de {p:.0f} dB" for (a, b), p in zip(R["segmentos"], picos)) +
               f"; tope {R['tope']:.0f} dB) pasan la regla: son dos sonidos igual de fuertes, separados por más de "
               f"{d['pausa_min_s']} s. Un comando tiene una sola palabra, así que V1 excluye la toma con motivo «varias_palabras».")
    fig, ejes = figura(2, len(casos), 5.2 * len(casos), 6.6, sharex="col")
    for j, (y, R, titulo) in enumerate(casos):
        _dibujar_detector(ejes[0, j], ejes[1, j], y, R, "\n".join(textwrap.wrap(titulo, 50)))
        ejes[1, j].set_xlabel("tiempo (s)")
        ejes[0, j].title.set_fontsize(9)
    ejes[1, 0].legend(fontsize=7.5, loc="lower right")
    cerrar(fig, 5, "reglas", "Las reglas del detector: unir, descartar, dominancia",
           f"Huecos < {d['pausa_min_s']} s se unen (cierres de consonantes). Segmentos < {d['segmento_min_s']} s o más débiles "
           f"que tope − {d['dif_palabra_db']:.0f} dB se descartan (rojo). Calibrado el 22/10 con 108 combinaciones sobre los audios reales.")


def paso_6_calidad(E: dict) -> None:
    encabezado(6, "WhatsApp y el control de calidad")
    fs, m = C["fs"], C["margen_wa"]
    fila = E["fila_wa"]
    y, fs0 = leer_audio(C["voz"] / fila["fuente"])
    y16 = remuestrear(y, fs0, fs)
    R = detectar(y16)
    ini, fin = R["segmentos"][0]
    i0, i1 = int(fila["muestra_inicio"]), int(fila["muestra_fin"])
    hablar(f"En la notebook cada toma es una palabra en 3 s y se guarda entera. Los audios de WhatsApp duran lo "
           f"que la persona tuvo apretado el botón (0,9 a 5 s): V1 busca la palabra con el mismo detector y corta un clip "
           f"con {m} s de silencio a cada lado (sin pasar del borde del audio ni de la mitad del silencio con la palabra "
           f"vecina, y como mucho {C['clip_max']:.0f} s). En {fila['fuente']}: palabra de {ini:.2f} a {fin:.2f} s → clip de "
           f"{i0 / fs:.2f} a {i1 / fs:.2f} s.")
    ex = E["excluidas"]
    cortada = next((r for r in ex if r["motivo"] == "cortada" and (C["voz"] / r["fuente"]).exists()), None)
    motivos = Counter((r["motivo"], r["origen"]) for r in ex)
    total_cand = len(E["meta"]) + len(ex)
    hablar(f"Control de calidad: a cada clip se le corre el detector y se excluye, con el motivo anotado en "
           f"excluidas.csv, lo que está mal: saturado (más del {C['saturacion_frac'] * 100:.0f} % de las muestras en "
           f"±{C['saturacion']}), sin palabra, con dos palabras, con la palabra cortada en el borde o con una duración "
           f"fuera de {C['dur_palabra'][0]}–{C['dur_palabra'][1]} s. Se excluyeron {len(ex)} de {total_cand} "
           f"({100 * len(ex) / total_cand:.1f} %): " +
           ", ".join(f"{v} {k[0]} ({k[1]})" for k, v in sorted(motivos.items(), key=lambda kv: -kv[1])) + ".")
    hablar("La saturación se mide por fracción de muestras y no por el pico: los audios de WhatsApp vienen normalizados "
           "por el celular y muchos tocan 1,0 en unas pocas muestras sin estar recortados.")
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    onda(ax, y16, fs, color=VERDE)
    sombrear(ax, i0 / fs, i1 / fs, color=NARANJA, alpha=0.15)
    sombrear(ax, ini, fin, color=VERDE, alpha=0.25)
    ax.set_title(f"{fila['fuente']} → {fila['archivo']}: palabra (verde) + {m} s de cada lado (naranja)", fontsize=10, loc="left")
    ax.set_ylabel("amplitud")
    ax = fig.add_subplot(gs[1, :2])
    if cortada:
        yc, fsc = leer_audio(C["voz"] / cortada["fuente"])
        yc = remuestrear(yc, fsc, fs)
        Rc = detectar(yc)
        onda(ax, yc, fs, color=ROJO)
        for a, b in Rc["segmentos"]:
            sombrear(ax, a, b, color=ROJO, alpha=0.2)
        borde, dur = C["det"]["margen_borde_s"], len(yc) / fs
        lados = []
        if Rc["segmentos"] and Rc["segmentos"][0][0] < borde:
            lados.append("empezó a hablar antes de apretar el botón")
            ax.axvspan(0, borde, color="black", alpha=0.3)
        if Rc["segmentos"] and Rc["segmentos"][-1][1] > dur - borde:
            lados.append("soltó el botón antes de terminar")
            ax.axvspan(dur - borde, dur, color="black", alpha=0.3)
        porque = " y ".join(lados) or "la palabra toca el borde"
        ax.set_title(f"{cortada['fuente']}: excluida por «{cortada['motivo']}»: {porque}", fontsize=10, loc="left")
        hablar(f"Ejemplo de «cortada»: {cortada['fuente']} ({cortada['nota']}). La palabra queda a menos de "
               f"{borde * 1000:.0f} ms del borde del audio (gris oscuro): {porque}. Ese clip no tiene la palabra entera.")
    ax.set_xlabel("tiempo (s)"); ax.set_ylabel("amplitud")
    ax = fig.add_subplot(gs[:, 2])
    nombres = [k for k, _ in Counter(r["motivo"] for r in ex).most_common()]
    izq = np.zeros(len(nombres))
    for origen in ["notebook", "whatsapp", "tts"]:
        vals = np.array([motivos.get((mo, origen), 0) for mo in nombres])
        ax.barh(nombres, vals, left=izq, color=ORIGEN_COLOR[origen], label=origen)
        izq += vals
    for i, v in enumerate(izq):
        ax.text(v + 0.1, i, str(int(v)), va="center", fontsize=9)
    ax.set_xlim(0, izq.max() * 1.15 + 0.5)
    ax.invert_yaxis(); ax.legend(fontsize=9, loc="lower right"); ax.set_xlabel("clips excluidos")
    ax.set_title(f"excluidas.csv: {len(ex)} de {total_cand} ({100 * len(ex) / total_cand:.1f} %)", fontsize=10)
    cerrar(fig, 6, "calidad", "WhatsApp y el control de calidad",
           "Nada se descarta en silencio: cada exclusión queda en excluidas.csv con su motivo. «notas» es la única a mano "
           "(una toma vacía); las demás las decide el detector.")


def paso_7_particiones(E: dict) -> None:
    encabezado(7, "Particiones por persona y conteos")
    M = E["meta"]
    parts = ["train", "val", "test"]
    con_nb = {r["persona"] for r in E["sesiones"]}
    grupos = {c: g[0] for g in C["mismo_hablante"] for c in g}
    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 3); ax.axis("off")
    alto_max = 0
    for j, p in enumerate(parts):
        codigos = C["particiones"][p]
        hablantes = []
        for c in codigos:
            h = grupos.get(c, c)
            if h not in [x[0] for x in hablantes]:
                hablantes.append((h, [x for x in codigos if grupos.get(x, x) == h]))
        ax.text(j + 0.5, 0.15, f"{p}\n{len(hablantes)} personas", ha="center", va="center", fontsize=11, fontweight="bold")
        for i, (h, cods) in enumerate(hablantes):
            y0 = 1.45 + i * 0.95
            color = ORIGEN_COLOR["notebook"] if any(c in con_nb for c in cods) else ORIGEN_COLOR["whatsapp"]
            ax.add_patch(plt.Rectangle((j + 0.08, y0 - 0.38), 0.84, 0.76, color=color, alpha=0.22, lw=0))
            txt = " + ".join(cods) + ("  ★" if E["codigo"] in cods else "")
            ax.text(j + 0.5, y0, txt, ha="center", va="center", fontsize=9)
            alto_max = max(alto_max, y0)
    ax.set_ylim(alto_max + 1.6, -0.6)
    ax.text(0.5, alto_max + 0.85, f"+ {len(C['tts_voces'])} voces TTS", ha="center", fontsize=9, color=NARANJA)
    ax.text(1.5, alto_max + 1.45, "azul: grabó en la notebook · verde: solo WhatsApp · ★: la toma de este paso a paso",
            ha="center", fontsize=8.5, color="#444444")
    ax = fig.add_subplot(gs[0, 1])
    abajo = np.zeros(3)
    for origen in ["notebook", "whatsapp", "tts", "ruido"]:
        vals = np.array([sum(1 for r in M if r["particion"] == p and r["origen"] == origen) for p in parts])
        ax.bar(parts, vals, bottom=abajo, color=ORIGEN_COLOR[origen], label=origen)
        abajo += vals
    for i, v in enumerate(abajo):
        ax.text(i, v + 6, str(int(v)), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("clips"); ax.legend(fontsize=9); ax.set_title("clips por partición y fuente", fontsize=10)
    ax = fig.add_subplot(gs[0, 2])
    clases = E["comandos"] + ["otro"]
    ancho = 0.2
    for i, cl in enumerate(clases):
        vals = [sum(1 for r in M if r["particion"] == p and r["clase"] == cl) for p in parts]
        ax.bar(np.arange(3) + (i - 1.5) * ancho, vals, ancho, color=COLORES[cl], label=cl)
    ax.set_xticks(range(3), parts); ax.legend(fontsize=9); ax.set_title("clips por clase", fontsize=10)
    n_part = {p: sum(1 for r in M if r["particion"] == p) for p in parts}
    hablar("La partición se hace por PERSONA, no por clip: todas las tomas de alguien van a train, a val o a test. Si la "
           "misma voz estuviera en train y en test, el K-NN la reconocería por el timbre y el acierto en test mentiría "
           "(fuga de datos). Por lo mismo, los pares de códigos que son la misma persona van siempre juntos.")
    hablar(f"Val y test tienen solo personas que grabaron en la notebook (es el micrófono de la demo), con hombres y "
           f"mujeres y de las dos tandas (17/10 y 20/10) en cada una. Las 8 personas que solo mandaron WhatsApp y las "
           f"voces sintéticas van a train: suman variedad sin ensuciar la medición. El ruido se reparte por archivo "
           f"(3 grabaciones en train, 1 en val, 1 en test): el aumento de V2 solo usa el de train.")
    hablar(f"Resultado de V1: {len(M)} clips de una palabra a 16 kHz: train {n_part['train']}, val {n_part['val']}, "
           f"test {n_part['test']}. Todo queda en data/voz/dataset/metadatos.csv (una fila por clip, con de qué archivo "
           f"crudo y qué muestras sale) y es determinista: generarlo dos veces da los mismos WAV byte a byte.")
    cerrar(fig, 7, "particiones", "Particiones por persona y conteos",
           "Cada persona está en una sola partición (18 personas en train, 4 en val y 4 en test). Val y test: solo el "
           "micrófono de la demo. TTS y WhatsApp solo suman a train.")


# ============================================================================= ETAPA B · V2
def paso_8_detectar_antes(E: dict) -> None:
    encabezado(8, "La palabra se busca ANTES de filtrar")
    fs = C["fs"]
    P = preprocesar(E["v1"])
    E["P0"] = P
    pal = P["palabra"]
    encontrada = (f"{P['n_segmentos']} segmento(s), palabra de {pal[0]:.2f} a {pal[1]:.2f} s ({pal[1] - pal[0]:.2f} s)"
                  if pal else "ningún segmento")
    estado = "válida" if P["valido"] else f"NO válida («{P['motivo']}»)"
    hablar(f"V2 arranca del WAV de V1 ({E['nombre_v1']}, {len(E['v1']) / fs:.0f} s). Primero busca la palabra con el MISMO "
           f"detector de V1 y, si hay más de un segmento, se queda con el de la trama más fuerte (en vivo, un golpe o una "
           f"voz de fondo no se suman a la palabra). Acá: {encontrada} → {estado}. "
           f"Es válida si dura entre {C['dur_palabra'][0]} y {C['dur_palabra'][1]} s; si no hay "
           f"palabra es «sin_voz» y si dura fuera de ese rango, «duracion». Un audio no válido el agente lo responde "
           f"directamente con «no entendí», sin pasar por el K-NN.")
    # la toma de «salir» del mismo hablante donde más se nota
    mejor = None
    for r in E["sesiones"]:
        if r["persona"] != E["codigo"] or r["clase"] != "salir":
            continue
        y, fs0 = leer_audio(C["voz"] / "notebook" / r["archivo"])
        y = remuestrear(y, fs0, fs)
        a, b = palabra_principal(y)[0], palabra_principal(preenfasis(pasabanda(y), C["a"]))[0]
        if a and b and (mejor is None or b[0] - a[0] > mejor[3][0] - mejor[2][0]):
            mejor = (r, y, a, b)
    r, y, a, b = mejor
    perdida = b[0] - a[0]
    hablar(f"¿Por qué sobre el audio SIN filtrar? Porque la «s» de «salir» tiene casi toda su energía por encima de "
           f"4 kHz, justo lo que el pasabanda saca. En {r['archivo']}: sin filtrar la palabra arranca en {a[0]:.2f} s; "
           f"filtrando primero arrancaría en {b[0]:.2f} s: se perderían {perdida * 1000:.0f} ms, la «s» entera. En el "
           f"prototipo, con el filtro antes, 14 «salir» quedaban por debajo de 0,3 s. Sin filtrar, todos los comandos dan "
           f"válidos y el 99 % da la misma palabra que V1.")
    fig, ejes = figura(3, 1, 13, 9.2, sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
    espectrograma(ejes[0, 0], y, fs)
    ejes[0, 0].axhline(C["banda"][1], color="white", ls="--", lw=1)
    ejes[0, 0].annotate(f"{C['banda'][1]} Hz: corte del pasabanda", (0.01, C["banda"][1] + 150),
                        xycoords=("axes fraction", "data"), color="white", fontsize=9)
    for t, col in [(a[0], VERDE), (b[0], ROJO)]:
        ejes[0, 0].axvline(t, color=col, lw=2)
    ejes[0, 0].set_title(f"{r['archivo']}: la «s» es la mancha de arriba a la izquierda de la palabra", fontsize=10, loc="left")
    Ra, Rb = detectar(y), detectar(preenfasis(pasabanda(y), C["a"]))
    for ax, R, sig, txt, col in [(ejes[1, 0], Ra, y, "sin filtrar (lo que hace V2)", VERDE),
                                 (ejes[2, 0], Rb, preenfasis(pasabanda(y), C["a"]), "filtrado antes (descartado)", ROJO)]:
        n, h = tramas()
        L = R["niveles"]
        ax.plot((np.arange(len(L)) * h + n / 2) / fs, L, color="#333333", lw=1)
        ax.axhline(R["umbral"], color=AZUL, lw=1.2)
        for s0, s1 in R["segmentos"]:
            sombrear(ax, s0, s1, color=col, alpha=0.25)
        ax.set_ylabel("nivel (dBFS)")
        ax.set_ylim(R["tope"] - 45, R["tope"] + 5)
        ax.set_title(f"{txt}: palabra de {R['segmentos'][0][0]:.2f} a {R['segmentos'][-1][1]:.2f} s", fontsize=10, loc="left")
    ejes[2, 0].set_xlabel("tiempo (s)")
    ejes[2, 0].set_xlim(max(0, a[0] - 0.4), b[1] + 0.4)
    cerrar(fig, 8, "detectar_antes", "La palabra se busca ANTES de filtrar",
           f"Verde: inicio sin filtrar; rojo: inicio si se filtrara antes. La diferencia ({perdida * 1000:.0f} ms) es la «s». "
           "Por eso el orden es: detectar sobre el audio crudo, después filtrar el audio entero y recién ahí recortar.")


def paso_9_pasabanda(E: dict) -> None:
    encabezado(9, f"Pasabanda Butterworth {C['banda'][0]}–{C['banda'][1]} Hz, ida y vuelta")
    fs, (f1, f2) = C["fs"], C["banda"]
    P = E["P0"]
    sos = scipy.signal.butter(C["orden"], C["banda"], btype="bandpass", fs=fs, output="sos")
    f, H = scipy.signal.sosfreqz(sos, worN=8192, fs=fs)
    H2 = 20 * np.log10(np.abs(H) ** 2 + 1e-15)
    g_f1 = 20 * np.log10(abs(scipy.signal.sosfreqz(sos, worN=[f1], fs=fs)[1][0]) ** 2)
    g_f2 = 20 * np.log10(abs(scipy.signal.sosfreqz(sos, worN=[f2], fs=fs)[1][0]) ** 2)
    g_5k = 20 * np.log10(abs(scipy.signal.sosfreqz(sos, worN=[5000], fs=fs)[1][0]) ** 2)
    g_40 = 20 * np.log10(abs(scipy.signal.sosfreqz(sos, worN=[40], fs=fs)[1][0]) ** 2)
    hablar(f"El audio ENTERO (los 3 s) pasa por un Butterworth pasabanda de orden {C['orden']} entre {f1} y {f2} Hz: "
           f"scipy.signal.butter(..., output='sos') lo arma como secciones de segundo orden (más estable numéricamente) y "
           f"sosfiltfilt lo aplica dos veces, hacia adelante y hacia atrás. Hacia atrás deshace el desfase de ida: la fase "
           f"es cero y la palabra no se corre en el tiempo, así que los bordes que encontró el detector siguen valiendo.")
    hablar(f"Al aplicarlo dos veces la ganancia queda al cuadrado: |H|² en vez de |H|. En las frecuencias de corte el "
           f"Butterworth tiene −3 dB, que ida y vuelta da −6 dB (medido: {g_f1:.1f} dB en {f1} Hz y {g_f2:.1f} dB en "
           f"{f2} Hz). Entre 300 y 2500 Hz pasa sin tocar (±0,3 dB), y afuera cae rápido: {g_40:.0f} dB en 40 Hz y "
           f"{g_5k:.0f} dB en 5 kHz.")
    hablar(f"Abajo de {f1} Hz saca el zumbido de la red (50 Hz) y los golpes graves; arriba de {f2} Hz, ruido y "
           f"silbidos (y también la «s»: por eso el paso 8). Queda la banda donde están los formantes, que es lo que "
           f"distingue una vocal de otra y lo que van a mirar los MFCC de V3 (sus filtros mel van de 80 a 4000 Hz).")
    fig, ejes = figura(1, 3, 15.5, 4.9, gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    ax = ejes[0, 0]
    ax.semilogx(f[1:], H2[1:], color=AZUL, lw=1.6)
    ax.axhline(-6, color=ROJO, ls=":", lw=1)
    for fc, gfc, dx, ha in [(f1, g_f1, 8, "left"), (f2, g_f2, -8, "right")]:
        ax.axvline(fc, color=ROJO, ls="--", lw=1)
        ax.plot([fc], [gfc], "o", color=ROJO)
        ax.annotate(f"{fc} Hz: {gfc:.1f} dB", (fc, gfc), xytext=(dx, -18), textcoords="offset points", fontsize=9,
                    color=ROJO, ha=ha)
    ax.set_xlim(20, fs / 2); ax.set_ylim(-80, 5)
    ax.set_xlabel("frecuencia (Hz)"); ax.set_ylabel("|H|² (dB)")
    ax.set_title(f"respuesta ida y vuelta (sosfiltfilt): |H(f)|²", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    n_st = int(0.025 * fs)
    tope = 20 * np.log10(np.abs(scipy.signal.stft(E["v1"].astype(np.float64), fs, nperseg=n_st, noverlap=n_st - fs // 100,
                                                  boundary=None, padded=False)[2]).max())
    for ax, sig, tit in [(ejes[0, 1], E["v1"], "antes (WAV de V1)"), (ejes[0, 2], P["filtrado"], "después del pasabanda")]:
        espectrograma(ax, sig, fs, tope=tope)
        ax.axhline(f2, color="white", ls="--", lw=0.8); ax.axhline(f1, color="white", ls="--", lw=0.8)
        ax.set_title(tit, fontsize=10); ax.set_xlabel("tiempo (s)")
        ax.set_xlim(max(0, P["palabra"][0] - 0.3), min(len(sig) / fs, P["palabra"][1] + 0.3))
    cerrar(fig, 9, "pasabanda", f"Pasabanda Butterworth {f1}–{f2} Hz, ida y vuelta",
           f"Orden {C['orden']}, como secciones de segundo orden, con sosfiltfilt: fase cero (la palabra no se mueve) y "
           f"−6 dB en {f1} y {f2} Hz. Los dos espectrogramas tienen la misma escala de colores: arriba de la línea de "
           f"{f2} Hz casi todo se apaga ({g_5k:.0f} dB en 5 kHz).")


def paso_10_preenfasis(E: dict) -> None:
    a, fs = C["a"], C["fs"]
    encabezado(10, f"Pre-énfasis: z[n] = x[n] − {a}·x[n−1]")
    P = E["P0"]
    f = np.linspace(1, fs / 2, 4000)
    w = 2 * np.pi * f / fs
    Hp = 20 * np.log10(np.sqrt(1 + a * a - 2 * a * np.cos(w)))
    sos = scipy.signal.butter(C["orden"], C["banda"], btype="bandpass", fs=fs, output="sos")
    Hb = 20 * np.log10(np.abs(scipy.signal.sosfreqz(sos, worN=f, fs=fs)[1]) ** 2 + 1e-15)
    val = lambda fr: 20 * math.log10(math.sqrt(1 + a * a - 2 * a * math.cos(2 * math.pi * fr / fs)))
    hablar(f"Después del pasabanda va el pre-énfasis: z[n] = x[n] − {a}·x[n−1], un FIR de dos coeficientes, "
           f"H(z) = 1 − {a}·z⁻¹. Su ganancia es |H(f)| = √(1 + a² − 2a·cos(2πf/fs)): {val(100):.1f} dB en 100 Hz, "
           f"{val(1000):.1f} dB en 1 kHz y {val(4000):+.1f} dB en 4 kHz. Sube unos 6 dB por octava.")
    hablar("¿Para qué? La voz cae unos 6 dB por octava hacia los agudos (por cómo vibran las cuerdas vocales y cómo irradian "
           "los labios). El pre-énfasis compensa esa caída: los formantes altos y las consonantes quedan con un nivel "
           "parecido al de las vocales, y el espectro que ven los MFCC es más plano. Es un paso clásico antes de los MFCC.")
    M = margen_muestras()
    i0, i1 = int(round(P["palabra"][0] * fs)), int(round(P["palabra"][1] * fs))
    fa, pa = scipy.signal.welch(P["filtrado"][i0:i1], fs, nperseg=512)
    fb, pb = scipy.signal.welch(P["acond"][i0:i1], fs, nperseg=512)
    fig, ejes = figura(1, 2, 14, 4.9)
    ax = ejes[0, 0]
    ax.semilogx(f, Hp, color=NARANJA, lw=1.6, label="pre-énfasis")
    ax.semilogx(f, Hb, color=AZUL, lw=1, ls="--", label="pasabanda (ida y vuelta)")
    ax.semilogx(f, Hp + Hb, color="black", lw=1.6, label="los dos juntos")
    ax.set_xlim(20, fs / 2); ax.set_ylim(-60, 10); ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("frecuencia (Hz)"); ax.set_ylabel("ganancia (dB)"); ax.legend(fontsize=9)
    ax.set_title("respuesta en frecuencia", fontsize=10)
    ax = ejes[0, 1]
    ref = 10 * np.log10(pa.max())
    ax.plot(fa, 10 * np.log10(pa + 1e-20) - ref, color=AZUL, lw=1.2, label="solo pasabanda")
    ax.plot(fb, 10 * np.log10(pb + 1e-20) - 10 * np.log10(pb.max()), color="black", lw=1.2, label="+ pre-énfasis")
    ax.set_xlim(0, C["banda"][1] + 500); ax.set_ylim(-70, 5)
    ax.set_xlabel("frecuencia (Hz)"); ax.set_ylabel("dB (relativo al máximo)"); ax.legend(fontsize=9)
    ax.set_title("espectro promedio de la palabra: el pre-énfasis lo «endereza»", fontsize=10)
    cerrar(fig, 10, "preenfasis", f"Pre-énfasis: z[n] = x[n] − {a}·x[n−1]",
           f"Atenúa los graves ({val(100):.0f} dB en 100 Hz) y realza los agudos ({val(4000):+.1f} dB en 4 kHz): compensa la "
           "caída natural de la voz. Puede hacer que los picos pasen de ±1: por eso V2 guarda en float32 y no recorta.")


def paso_11_recorte(E: dict) -> None:
    fs, M = C["fs"], margen_muestras()
    encabezado(11, f"Recorte con {C['margen_ms']:.0f} ms de margen")
    P = E["P0"]
    ini, fin = P["palabra"]
    m0, m1 = int(round(ini * fs)), int(round(fin * fs))
    clip = P["clip"]
    hablar(f"Recién ahora se recorta, sobre el audio ya filtrado: m0 = round({ini:.2f}·{fs}) = {m0}, m1 = {m1} y "
           f"M = round({C['margen_ms']:.0f} ms · {fs}/1000) = {M} muestras de margen. El clip es x[m0 − M : m1 + M]: "
           f"{m1 - m0} muestras de palabra + 2·{M} = {len(clip)} muestras ({len(clip) / fs:.2f} s).")
    hablar("Se filtra antes de recortar a propósito: un filtro tarda unas muestras en «arrancar» (transitorio), y si se "
           "recortara primero ese transitorio caería sobre la palabra. Si la palabra está pegada al borde del audio, lo "
           "que falta del margen se completa con ceros (en vivo es común soltar el botón apenas termina la palabra).")
    fig, ejes = figura(2, 1, 13, 6.8)
    ax = ejes[0, 0]
    onda(ax, P["acond"], fs, lw=0.5)
    sombrear(ax, (m0 - M) / fs, (m1 + M) / fs, color=NARANJA, alpha=0.2)
    sombrear(ax, ini, fin, color=VERDE, alpha=0.25)
    ax.set_title("audio entero, filtrado y con pre-énfasis: verde la palabra, naranja lo que se recorta", fontsize=10, loc="left")
    ax.set_xlabel("tiempo (s)")
    ax = ejes[1, 0]
    n = np.arange(len(clip))
    ax.plot(n, clip, color="#333333", lw=0.5)
    ax.axvspan(0, M, color=NARANJA, alpha=0.2); ax.axvspan(len(clip) - M, len(clip), color=NARANJA, alpha=0.2)
    ax.axvspan(M, len(clip) - M, color=VERDE, alpha=0.12)
    for xpos, txt in [(M / 2, f"margen\n{M} muestras"), (len(clip) - M / 2, f"margen\n{M} muestras"),
                      (len(clip) / 2, f"palabra: {m1 - m0} muestras")]:
        ax.text(xpos, ax.get_ylim()[1] * 0.85, txt, ha="center", va="top", fontsize=9)
    ax.set_xlim(0, len(clip)); ax.set_xlabel("muestra del clip"); ax.set_title(f"el clip: {len(clip)} muestras", fontsize=10, loc="left")
    cerrar(fig, 11, "recorte", f"Recorte con {C['margen_ms']:.0f} ms de margen",
           f"Primero se filtra el audio entero, después se corta: los transitorios del filtro quedan afuera. El clip siempre "
           f"tiene la palabra en [M, len − M), con M = {M}: V3 sabe exactamente dónde empieza y dónde termina.")


def paso_12_nivel(E: dict) -> None:
    fs, nivel = C["fs"], C["nivel_db"]
    encabezado(12, f"Nivel fijo: la palabra a {nivel:.0f} dBFS")
    P = E["P0"]
    M = margen_muestras()
    filas = sorted((r for r in E["sesiones"] if r["persona"] == E["codigo"] and r["clase"] == E["clase"]), key=lambda r: int(r["toma"]))
    datos = []
    for r in filas:
        y, fs0 = leer_audio(C["voz"] / "notebook" / r["archivo"])
        Pr = preprocesar(a_16_bits(remuestrear(y, fs0, fs), fs))
        if not Pr["valido"]:
            continue
        antes = db(rms(Pr["clip"][M:len(Pr["clip"]) - M]))
        despues = db(rms(Pr["y"][M:len(Pr["y"]) - M]))
        etiqueta = (f"{ESTILO.get(r['estilo'], r['estilo'])}\n{r['distancia_cm']} cm" if E["clase"] != "otro" else r["detalle"])
        datos.append((etiqueta, antes, despues, Pr["y"], r["archivo"] == E["fuente_rel"].split("/", 1)[1]))
    rango = max(d[1] for d in datos) - min(d[1] for d in datos)
    hablar(f"Último paso de la cadena: el clip se multiplica por g = 10^({nivel:.0f}/20) / rms_palabra, con el RMS medido "
           f"SOLO sobre la palabra (sin los márgenes, así el nivel no depende de cuánto silencio quedó). En esta toma "
           f"rms_palabra = {db(rms(P['clip'][M:len(P['clip']) - M])):.1f} dBFS, así que g = {P['ganancia']:.1f} "
           f"({db(P['ganancia']):+.1f} dB).")
    hablar(f"Entre las {len(datos)} tomas de «{E['clase']}» de {E['codigo']} el nivel de la palabra variaba {rango:.1f} dB "
           f"(lejos, cerca, más fuerte, más suave). Después, todas quedan en {nivel:.0f} dBFS: para el K-NN, lo que cambia "
           f"es la forma de la palabra y no lo cerca que se habló del micrófono.")
    hablar(f"¿Por qué {nivel:.0f}? Es el nivel al que se lleva la voz en telefonía (UIT-T P.56) y deja lugar a los picos: "
           f"con −20 dBFS el 42 % de los clips pasaba de 0 dBFS por el pre-énfasis. Igual, V2 guarda en float32: nada se "
           f"recorta.")
    fig, ejes = figura(1, 2, 14.5, 5.2, gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = ejes[0, 0]
    x = np.arange(len(datos))
    for i, d in enumerate(datos):
        ax.annotate("", (i, d[2]), (i, d[1]), arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.2))
        ax.text(i + 0.08, (d[1] + d[2]) / 2, f"{d[2] - d[1]:+.1f} dB", fontsize=8.5, va="center")
    ax.plot(x, [d[1] for d in datos], "o", color=GRIS, ms=8, label="antes (palabra filtrada y recortada)")
    ax.plot(x, [d[2] for d in datos], "o", color=COLORES[E["clase"]], ms=8, label=f"después ({nivel:.0f} dBFS)")
    ax.set_xticks(x, [d[0] for d in datos], fontsize=8.5)
    ax.set_xlim(-0.5, len(datos) - 0.2)
    ax.set_ylim(min(d[1] for d in datos) - 5, nivel + 9); ax.set_ylabel("RMS de la palabra (dBFS)")
    ax.axhline(nivel, color=COLORES[E["clase"]], ls="--", lw=1)
    ax.legend(fontsize=8.5, loc="upper center", ncol=2)
    ax.set_title(f"las {len(datos)} tomas de «{E['clase']}» de {E['codigo']}", fontsize=10)
    ax = ejes[0, 1]
    for i, d in enumerate(datos):
        y = d[3]
        ax.plot(np.arange(len(y)) / fs, y - i * 0.5, color=COLORES[E["clase"]] if d[4] else "#444444", lw=0.5)
        ax.text(-0.02, -i * 0.5, d[0].replace("\n", " "), ha="right", va="center", fontsize=8.5)
    ax.set_yticks([]); ax.set_xlabel("tiempo del clip (s)")
    ax.set_title("después: mismo nivel, distinta duración y forma (en color, la toma de este paso a paso)", fontsize=10)
    cerrar(fig, 12, "nivel", f"Nivel fijo: la palabra a {nivel:.0f} dBFS",
           f"g = 10^({nivel:.0f}/20) / rms_palabra. Saca la diferencia de volumen entre tomas y entre personas: queda la forma de la palabra.")


def paso_13_velocidad(E: dict) -> None:
    fs = C["fs"]
    encabezado(13, "Aumento de datos 1: cambio de velocidad")
    P0 = E["P0"]
    if E["particion"] != "train" or not P0["valido"]:
        hablar(f"Esta toma es de {E['particion']}: val y test no se aumentan (se miden con audio real, tal cual). "
               f"Probá con una toma de train (por ejemplo --toma p01_reconocer_01) para ver los pasos 13 y 14.")
        return
    nombres = sorted(r["archivo"] for r in E["meta"] if r["particion"] == "train" and r["origen"] == "ruido")
    ruidos = [sf.read(str(C["voz"] / "dataset" / a), dtype="float32")[0].reshape(-1) for a in nombres]
    E["ruidos"], E["nombres_ruido"] = ruidos, nombres
    copias = []
    for k in range(1, C["copias"] + 1):
        params = sortear_aumento(semilla_copia(E["archivo_v1"], k), len(ruidos))
        A = aumentar(E["v1"], P0["palabra"], ruidos, params)
        copias.append((k, params, A))
    E["copias"] = copias
    hablar(f"Train tiene pocas tomas por persona. Para que el K-NN aprenda la palabra y no las condiciones exactas de la "
           f"grabación, cada clip de train da además {C['copias']} copias aumentadas. Cada copia tiene su propio generador "
           f"aleatorio, np.random.default_rng([semilla_global, crc32(nombre del archivo), copia]): el sorteo no depende del "
           f"orden en que se procesan los clips, y regenerar da exactamente lo mismo.")
    for k, p, A in copias:
        hablar(f"Copia {k}: factor = {p['factor']:.2f}, SNR = {p['snr_db']:.2f} dB, ruido = {nombres[p['ruido']].split('/')[-1]} "
               f"(n.º {p['ruido']} de {len(ruidos)}), desplazamiento = {p['desplazamiento']:.3f}.")
    k, p, A = copias[0]
    f = p["factor"]
    d = int(round(100 * f))
    g = math.gcd(100, d)
    semitonos = 12 * math.log2(f)
    hablar(f"Velocidad, como acelerar una cinta: un solo remuestreo, resample_poly(y, {100 // g}, {d // g}), y se reproduce "
           f"a la misma fs. Con factor {f:.2f} el audio dura {f:.2f} veces menos y todas las frecuencias se multiplican por "
           f"{f:.2f}: el tono sube 12·log2({f:.2f}) = {semitonos:+.2f} semitonos. El rango es {C['velocidad'][0]}–"
           f"{C['velocidad'][1]} (−1,4 a +1,3 semitonos): alguien que habla un poco más rápido y un poco más agudo, o al revés.")
    hablar("Se descartó el phase vocoder (cambiar la duración sin cambiar el tono) porque deja un sonido metálico.")
    i0, i1 = int(round(P0["palabra"][0] * fs)), int(round(P0["palabra"][1] * fs))
    j0, j1 = int(round(A["palabra"][0] * fs)), int(round(A["palabra"][1] * fs))
    fig, ejes = figura(1, 3, 15, 4.9, gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, sig, a0, a1, tit in [(ejes[0, 0], E["v1"], i0, i1, f"original: {(i1 - i0) / fs:.2f} s"),
                                 (ejes[0, 1], A["veloz"], j0, j1, f"factor {f:.2f}: {(j1 - j0) / fs:.2f} s")]:
        espectrograma(ax, sig[a0:a1], fs, fmax=2000, rango_db=60)
        ax.set_title(tit, fontsize=10); ax.set_xlabel("tiempo (s)")
    ax = ejes[0, 2]
    seg = min(4096, i1 - i0, j1 - j0)
    fa, pa = scipy.signal.welch(E["v1"][i0:i1].astype(np.float64), fs, nperseg=seg, nfft=8192)
    fb, pb = scipy.signal.welch(A["veloz"][j0:j1].astype(np.float64), fs, nperseg=seg, nfft=8192)
    ax.plot(fa, 10 * np.log10(pa + 1e-20), color=GRIS, lw=1.4, label="original")
    ax.plot(fb, 10 * np.log10(pb + 1e-20), color=COLORES[E["clase"]], lw=1.4, label=f"factor {f:.2f}")
    banda = (fa > 80) & (fa < 400)
    pico_a = float(fa[banda][np.argmax(pa[banda])])
    pico_b = float(fb[banda][np.argmax(pb[banda])])
    for pk, col in [(pico_a, GRIS), (pico_b, COLORES[E["clase"]])]:
        ax.axvline(pk, color=col, ls=":", lw=1.2)
    ax.set_xlim(0, 700); ax.set_xlabel("frecuencia (Hz)"); ax.set_ylabel("dB"); ax.legend(fontsize=9, loc="lower left")
    ax.set_title(f"armónico más fuerte: {pico_a:.0f} → {pico_b:.0f} Hz (×{pico_b / pico_a:.3f}; se esperaba ×{f:.2f})", fontsize=10)
    hablar(f"Medido en la palabra: el armónico más fuerte pasa de {pico_a:.0f} a {pico_b:.0f} Hz (×{pico_b / pico_a:.3f}).")
    cerrar(fig, 13, "velocidad", "Aumento de datos 1: cambio de velocidad",
           f"Un solo remuestreo: más rápido Y más agudo a la vez (factor {f:.2f}: dura {(i1 - i0) / fs:.2f} → {(j1 - j0) / fs:.2f} s). "
           "En el espectrograma (0–2 kHz) se ven los armónicos de la voz un poco más arriba y la palabra más corta.")


def paso_14_ruido(E: dict) -> None:
    fs = C["fs"]
    encabezado(14, "Aumento de datos 2: ruido real, y otra vez preprocesar")
    if "copias" not in E:
        hablar("Sin copias (la toma no es de train).")
        return
    resultados = []
    for k, p, A in E["copias"]:
        Pk = preprocesar(A["y"])
        resultados.append((k, p, A, Pk))
    E["Pk"] = resultados
    k, p, A, Pk = resultados[0]
    snr_medida = 10 * math.log10(A["ps"] / (A["g"] ** 2 * A["pn"]))
    hablar(f"A la copia, ya con la velocidad cambiada, se le suma un tramo de ruido REAL de la sala (uno de los {len(E['ruidos'])} clips de ruido "
           f"de train; nunca el de val o test). La relación señal-ruido se mide sobre la palabra: con Ps la potencia de la "
           f"palabra y Pn la del ruido, g = √(Ps / (Pn·10^(SNR/10))) y queda y + g·ruido. Copia {k}: Ps = {db(math.sqrt(A['ps'])):.1f} "
           f"dB, Pn = {db(math.sqrt(A['pn'])):.1f} dB, g = {A['g']:.3f} → SNR = {snr_medida:.2f} dB (sorteada entre "
           f"{C['snr'][0]:.0f} y {C['snr'][1]:.0f} dB).")
    if db(math.sqrt(A["pn"])) < -70:
        hablar(f"Ese ruido es casi silencio ({db(math.sqrt(A['pn'])):.0f} dB: la supresión de ruido de Windows ya lo había "
               f"limpiado), por eso g sale tan grande: lo que se suma son sobre todo los clics y golpes que quedaron.")
    hablar(f"Lo importante: el ruido se suma ANTES de buscar la palabra, y la copia pasa entera por preprocesar(), igual que "
           f"un audio en vivo. El detector la busca de nuevo y puede recortarla distinto: en la copia {k} la encontró de "
           f"{Pk['palabra'][0]:.2f} a {Pk['palabra'][1]:.2f} s (con la velocidad cambiada estaba en {A['palabra'][0]:.2f}–"
           f"{A['palabra'][1]:.2f} s). Así el K-NN aprende las palabras recortadas como las va a recortar el detector en la "
           f"demo. En el prototipo, con ruido de 25 dB en la evaluación: 89,6 % de acierto sumándolo antes, 84,8 % "
           f"sumándolo después del recorte y 81,8 % sin aumento.")
    for kk, pp, AA, PP in resultados[1:]:
        hablar(f"Copia {kk}: palabra de {PP['palabra'][0]:.2f} a {PP['palabra'][1]:.2f} s "
               f"({PP['palabra'][1] - PP['palabra'][0]:.2f} s), {'válida' if PP['valido'] else 'NO válida: ' + PP['motivo']}.")
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, len(resultados))
    for col, (kk, pp, AA, PP) in enumerate(resultados):
        ax = fig.add_subplot(gs[0, col])
        r = E["ruidos"][pp["ruido"]]
        onda(ax, r, fs, color=GRIS, lw=0.5)
        sombrear(ax, AA["offset"] / fs, (AA["offset"] + len(AA["y"])) / fs, color=NARANJA, alpha=0.25)
        ax.set_title(f"copia {kk} · ruido: {E['nombres_ruido'][pp['ruido']].split('/')[-1].replace('ruido_ruido_', '')}\n"
                     f"naranja: el tramo que se usa (desde {AA['offset'] / fs:.2f} s)", fontsize=9.5, loc="left")
        ax = fig.add_subplot(gs[1, col])
        onda(ax, AA["y"], fs, color="#333333", lw=0.5)
        sombrear(ax, *AA["palabra"], color=GRIS, alpha=0.25)
        if PP["palabra"]:
            sombrear(ax, *PP["palabra"], color=VERDE, alpha=0.25)
        ax.set_title(f"×{pp['factor']:.2f} + ruido a {pp['snr_db']:.1f} dB (g = {AA['g']:.2f})\n"
                     f"gris: dónde quedó la palabra · verde: lo que encontró el detector", fontsize=9.5, loc="left")
        ax.set_xlabel("tiempo (s)")
    ax = fig.add_subplot(gs[2, :])
    base = E["P0"]["y"]
    ax.plot(np.arange(len(base)) / fs, base, color=GRIS, lw=0.5, label=f"k0 (sin aumento): {len(base) / fs:.2f} s")
    for kk, pp, AA, PP in resultados:
        if PP["valido"]:
            ax.plot(np.arange(len(PP["y"])) / fs, PP["y"] - 0.6 * kk, lw=0.5, color=[COLORES[E["clase"]], NARANJA, VERDE][(kk - 1) % 3],
                    label=f"k{kk}: ×{pp['factor']:.2f}, {pp['snr_db']:.1f} dB, {len(PP['y']) / fs:.2f} s")
    ax.set_yticks([]); ax.legend(fontsize=9, loc="upper right"); ax.set_xlabel("tiempo del clip (s)")
    ax.set_title("los clips que van a V3: el original y sus copias, todos preprocesados igual", fontsize=10, loc="left")
    cerrar(fig, 14, "ruido", "Aumento de datos 2: ruido real, y otra vez preprocesar",
           "SNR medida sobre la palabra. El ruido se suma antes de detectar: la copia se preprocesa como un audio en vivo, "
           "así que a veces el recorte le pega algo de ruido a la palabra, como va a pasar en la demo.")


def paso_15_resultado(E: dict) -> None:
    encabezado(15, "El dataset preprocesado")
    P = E["prepro"]
    parts = ["train", "val", "test"]
    orig = {p: sum(1 for r in P if r["particion"] == p and r["copia"] == "0" and r["valido"] == "1") for p in parts}
    cop = {p: sum(1 for r in P if r["particion"] == p and r["copia"] != "0" and r["valido"] == "1") for p in parts}
    n_cop = sum(1 for r in P if r["copia"] != "0")
    no_val = [r for r in P if r["valido"] == "0"]
    no_val0 = [r for r in no_val if r["copia"] == "0"]
    clases_nv0 = Counter(r["clase"] for r in no_val0)
    hablar(f"Resultado de V2 (data/voz/prepro/metadatos.csv, una fila por clip y copia): train {orig['train']} originales + "
           f"{cop['train']} copias = {orig['train'] + cop['train']} clips válidos; val {orig['val']}; test {orig['test']}. "
           f"De las {n_cop} copias aumentadas son válidas {sum(cop.values())} ({100 * sum(cop.values()) / max(1, n_cop):.1f} %).")
    por_part = Counter(r["particion"] for r in no_val0)
    if set(clases_nv0) <= {"otro"}:
        hablar(f"Los originales no válidos son {len(no_val0)} ({', '.join(f'{por_part[p]} en {p}' for p in parts)}) y "
               f"todos son de la clase «otro»: toses, golpes y ruido, que no tienen una palabra de {C['dur_palabra'][0]} "
               f"a {C['dur_palabra'][1]} s. Todos los comandos de V1 dan válidos.")
    else:
        hablar(f"Los originales no válidos son {len(no_val0)}: " + ", ".join(f"{v} de {k}" for k, v in clases_nv0.items()) + ".")
    hablar("En las métricas, un no válido cuenta como «otro» («no entendí»): es acierto si era «otro» y error si era un comando.")
    hablar("Los WAV de V2 son float32 con la palabra en [M, len − M): eso es lo que lee V3 para calcular los MFCC.")
    fig, ejes = figura(1, 3, 15, 4.9, gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    ax = ejes[0, 0]
    ax.bar(parts, [orig[p] for p in parts], color=AZUL, label="originales (k0)")
    ax.bar(parts, [cop[p] for p in parts], bottom=[orig[p] for p in parts], color=NARANJA, label="copias aumentadas")
    for i, p in enumerate(parts):
        ax.text(i, orig[p] + cop[p] + 20, str(orig[p] + cop[p]), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("clips válidos"); ax.legend(fontsize=9); ax.set_title("clips válidos por partición", fontsize=10)
    ax = ejes[0, 1]
    motivos = Counter((r["motivo"], "copias" if r["copia"] != "0" else "originales") for r in no_val)
    etiquetas = sorted({m for m, _ in motivos})
    izq = np.zeros(len(etiquetas))
    for tipo, col in [("originales", AZUL), ("copias", NARANJA)]:
        vals = np.array([motivos.get((m, tipo), 0) for m in etiquetas])
        ax.barh(etiquetas, vals, left=izq, color=col, label=tipo)
        izq += vals
    for i, v in enumerate(izq):
        ax.text(v + 0.5, i, str(int(v)), va="center", fontsize=9)
    ax.legend(fontsize=9); ax.set_xlabel("clips no válidos"); ax.set_title("no válidos, por motivo", fontsize=10)
    ax = ejes[0, 2]
    bins = np.linspace(0, 3, 61)
    for cl in E["comandos"] + ["otro"]:
        dur = [float(r["duracion_palabra_s"]) for r in P if r["copia"] == "0" and r["clase"] == cl and r["duracion_palabra_s"]]
        ax.hist(dur, bins=bins, histtype="step", lw=1.5, color=COLORES[cl], label=f"{cl} ({len(dur)})")
    for lim in C["dur_palabra"]:
        ax.axvline(lim, color="black", ls="--", lw=1)
    ax.set_xlabel("duración de la palabra (s)"); ax.set_ylabel("clips"); ax.legend(fontsize=8.5)
    ax.set_title(f"duración de la palabra (originales): válida entre {C['dur_palabra'][0]} y {C['dur_palabra'][1]} s", fontsize=10)
    cerrar(fig, 15, "resultado", "El dataset preprocesado",
           f"Train pasa de {orig['train']} a {orig['train'] + cop['train']} clips con el aumento. Val y test no se aumentan: se miden "
           "con audio real tal como lo grabaron. Lo que no tiene una palabra de duración razonable es «otro».")


# ============================================================================= resumen y audios
def resumen(E: dict) -> None:
    fs = C["fs"]
    P0 = E["P0"]
    paneles = [("1 · crudo", E["crudo"], E["fs_crudo"], f"{E['fs_crudo']} Hz, {len(E['crudo']) / E['fs_crudo']:.1f} s"),
               ("2 · V1: 16 kHz, 16 bits", E["v1"], fs, f"palabra {P0['palabra'][0]:.2f}–{P0['palabra'][1]:.2f} s"),
               ("3 · pasabanda + pre-énfasis", P0["acond"], fs, "audio entero, fase cero"),
               ("4 · k0: recorte + −26 dBFS", P0["y"], fs, f"{len(P0['y']) / fs:.2f} s, listo para V3")]
    for k, p, A, Pk in E.get("Pk", []):
        if Pk["valido"]:
            paneles.append((f"{4 + k} · k{k}: ×{p['factor']:.2f} + ruido {p['snr_db']:.0f} dB", Pk["y"], fs, f"{len(Pk['y']) / fs:.2f} s"))
    cols = 3
    filas = math.ceil(len(paneles) / cols)
    fig, ejes = figura(filas, cols, 15, 3.1 * filas + 0.4)
    for ax, (tit, y, f, sub) in zip(ejes.ravel(), paneles):
        onda(ax, y, f, color=COLORES[E["clase"]] if tit.startswith(("4", "5", "6")) else "#333333", lw=0.4)
        if tit.startswith(("1", "2", "3")):
            sombrear(ax, *P0["palabra"])
        ax.set_title(es(f"{tit}\n{sub}"), fontsize=9.5, loc="left"); ax.tick_params(labelsize=7)
    for ax in ejes.ravel()[len(paneles):]:
        ax.axis("off")
    fig.suptitle(f"V1 + V2 · {E['fuente_rel']} → {E['nombre_v1']} → {Path(E['archivo_v1']).stem}_k0…k{C['copias']}",
                 fontsize=13, fontweight="bold", x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ruta = SALIDA / "resumen.png"
    fig.savefig(ruta, dpi=110)
    GUION.append(("Resumen", [f"![]({ruta.name})"]))
    print(f"   [figura guardada: {ruta}]")
    if MOSTRAR:
        plt.show()
    plt.close(fig)


def guardar_audios(E: dict) -> list[str]:
    fs = C["fs"]
    P0 = E["P0"]
    lista = [("01_crudo.wav", E["crudo"], E["fs_crudo"], f"la toma tal como se grabó ({E['fs_crudo']} Hz)"),
             ("02_v1_16k.wav", E["v1"], fs, "el WAV de V1 (16 kHz, 16 bits): suena igual"),
             ("03_pasabanda.wav", P0["filtrado"], fs, "después del pasabanda: sin graves ni agudos extremos"),
             ("04_preenfasis.wav", P0["acond"], fs, "con pre-énfasis: más «fina», más agudos"),
             ("05_k0.wav", P0["y"], fs, "recortada y a −26 dBFS: lo que lee V3")]
    for k, p, A, Pk in E.get("Pk", []):
        lista.append((f"{5 + 2 * k - 1:02d}_k{k}_antes_de_preprocesar.wav", A["y"], fs,
                      f"copia {k} antes de preprocesar: velocidad ×{p['factor']:.2f} + ruido a {p['snr_db']:.1f} dB"))
        if Pk["valido"]:
            lista.append((f"{5 + 2 * k:02d}_k{k}.wav", Pk["y"], fs, f"copia {k} preprocesada"))
    lineas = []
    for nombre, y, f, desc in lista:
        guardar_audio(nombre, y, f)
        lineas.append(f"- `audios/{nombre}`: {desc}")
    print(f"   [audios guardados en {SALIDA / 'audios'}: {len(lista)} WAV]")
    return lineas


def escribir_guion(E: dict, audios: list[str]) -> None:
    partes = [f"# V1 + V2 paso a paso · {E['fuente_rel']}\n",
              "Generado por `scripts/v1v2_paso_a_paso.py`: cada paso tiene su figura y lo que imprimió el script, con los "
              "números de esta corrida. Los audios (WAV float32) están en `audios/`.\n",
              "## Versión corta: una frase por figura\n"]
    partes += [f"{n}. **{t}** ({n:02d}). {p}" for n, t, p in PIES]
    partes.append("")
    for titulo, parrafos in GUION:
        partes.append(f"## {titulo}\n")
        for p in parrafos:
            partes.append(p.replace("\n", "  \n") + "\n")
    if audios:
        partes.append("## Audios para escuchar en orden\n")
        partes.append(es("\n".join(audios)) + "\n")
    partes.append("## Preguntas probables sobre este hito\n")
    partes += [f"**{q}** {r}\n" for q, r in PREGUNTAS]
    (SALIDA / "GUION.md").write_text("\n".join(partes), encoding="utf-8")
    print(f"   [guion guardado: {SALIDA / 'GUION.md'}]")


PREGUNTAS = [
    ("¿Por qué la voz no es sintética, como las huellas?",
     "Porque el agente tiene que entender a personas reales en la demo, con el micrófono de la notebook. Las voces de "
     "texto a voz solo suman variedad de timbre en train, y son menos que las tomas reales de train."),
    ("¿Por qué partir por persona y no por clip?",
     "Si una misma voz está en train y en test, el K-NN la reconoce por el timbre y el acierto en test sale inflado: es "
     "fuga de datos. Por eso cada persona (y los pares de códigos que son la misma persona) está en una sola partición."),
    ("¿Por qué val y test tienen solo audios de la notebook?",
     "Porque la demo es con ese micrófono: se mide en las mismas condiciones en que se va a usar. WhatsApp y TTS solo entrenan."),
    ("¿Por qué 16 kHz?",
     "Por Nyquist conserva hasta 8 kHz, que alcanza para la voz (es la banda ancha de la telefonía), y son 2,76 veces "
     "menos muestras que a 44,1 kHz. resample_poly hace el filtro antialias y el diezmado en un solo paso."),
    ("¿Por qué el detector usa umbrales relativos al tope?",
     "Para no depender del volumen: cada persona habla a distinta distancia y cada micrófono tiene otra ganancia. "
     "Multiplicar el audio por una constante no cambia los segmentos."),
    ("¿Por qué se busca la palabra antes de filtrar?",
     "Porque el pasabanda corta arriba de 4 kHz, que es donde está la «s» de «salir»: filtrando antes, la palabra perdía "
     "su comienzo y en el prototipo 14 «salir» quedaban por debajo de 0,3 s."),
    ("¿Para qué sosfiltfilt y no un filtro común?",
     "Porque aplica el filtro ida y vuelta: la fase queda en cero y la palabra no se corre en el tiempo, así los bordes "
     "que marcó el detector siguen valiendo. El costo es que la atenuación se duplica (−6 dB en los cortes)."),
    ("¿Para qué el pre-énfasis?",
     "La voz pierde unos 6 dB por octava hacia los agudos; el pre-énfasis (1 − 0,97·z⁻¹) lo compensa y deja el espectro "
     "más parejo antes de los MFCC."),
    ("¿Por qué normalizar a −26 dBFS y medir el RMS solo en la palabra?",
     "Para que el volumen no cuente para el K-NN. Medido sin los márgenes, el nivel no depende del silencio que quedó. "
     "−26 dBFS es el nivel de referencia de telefonía (UIT-T P.56) y deja lugar a los picos del pre-énfasis."),
    ("¿Por qué el ruido se suma antes de buscar la palabra?",
     "Para que las copias se recorten como se va a recortar un audio en vivo. En el prototipo, con ruido de 25 dB en la "
     "evaluación: 89,6 % sumándolo antes, 84,8 % después del recorte y 81,8 % sin aumento."),
    ("¿Por qué no se aumentan val y test?",
     "Porque se usan para medir: tienen que ser audio real, tal como se grabó. Y el ruido del aumento sale solo de las "
     "grabaciones de train."),
    ("¿Cómo sabés que está bien?",
     "Cada etapa tiene tests escritos en otro chat a partir del contrato y congelados antes del código (test_v1.py y "
     "test_v2.py). Además, este script rehace todo paso a paso y con --verificar da los mismos WAV que el repo, muestra "
     "por muestra."),
]


# ============================================================================= verificación
def verificar(E: dict) -> bool:
    encabezado(0, "Verificación contra src/voz y los WAV de data/voz")
    sys.path.insert(0, str(RAIZ))
    from src.config import cargar_config
    from src.voz import audio as A
    from src.voz import aumento as AU
    from src.voz import preprocesamiento as PR
    from src.voz import segmentacion as S
    cfg = cargar_config()
    fs = C["fs"]
    oks = []

    def informe(nombre: str, ok: bool, detalle: str = "") -> None:
        oks.append(ok)
        print(f"   {'✔' if ok else '✘'} {nombre}{(': ' + detalle) if detalle else ''}")

    x, fs0 = A.leer_audio(C["voz"] / E["fuente_rel"])
    informe("leer_audio", np.array_equal(x, E["crudo"]) and fs0 == E["fs_crudo"])
    informe("remuestrear", np.array_equal(A.remuestrear(x, fs0, fs), E["v1_float"]))
    ruta_v1 = C["voz"] / "dataset" / E["archivo_v1"]
    if ruta_v1.exists():
        y_v1 = sf.read(str(ruta_v1), dtype="float32")[0].reshape(-1)
        informe(f"WAV de V1 ({E['archivo_v1']})", np.array_equal(y_v1, E["v1"]), "idéntico muestra a muestra")
    else:
        y_v1 = E["v1"]
        informe(f"WAV de V1 ({E['archivo_v1']})", False, "no está en data/voz/dataset")
    seg_src = S.detectar_palabras(E["v1_float"], fs, cfg)
    informe("detectar_palabras", [tuple(s) for s in seg_src] == [tuple(s) for s in E["det_v1"]["segmentos"]],
            f"{[(round(a, 3), round(b, 3)) for a, b in seg_src]} (metadatos: {E['v1_palabra_meta']})")
    informe("envolvente_db", np.array_equal(S.envolvente_db(E["v1_float"], fs, 20, 10), E["det_v1"]["niveles"]))
    r0 = PR.preprocesar(y_v1, fs, cfg)
    informe("preprocesar (k0)", r0.valido and np.array_equal(r0.y, E["P0"]["y"]) and (r0.inicio_s, r0.fin_s) == E["P0"]["palabra"],
            f"máx |dif| = {float(np.abs(r0.y - E['P0']['y']).max()) if len(r0.y) == len(E['P0']['y']) else 'largos distintos'}")
    stem = Path(E["archivo_v1"]).stem
    ruta_k0 = C["voz"] / "prepro" / E["particion"] / f"{stem}_k0.wav"
    if ruta_k0.exists():
        k0 = sf.read(str(ruta_k0), dtype="float32")[0].reshape(-1)
        informe(f"data/voz/prepro/{E['particion']}/{stem}_k0.wav", len(k0) == len(E["P0"]["y"]) and np.abs(k0 - E["P0"]["y"]).max() <= 1e-7)
    for k, p, Aa, Pk in E.get("Pk", []):
        p_src = AU.sortear_aumento(cfg, AU.semilla_copia(cfg, E["archivo_v1"], k), len(E["ruidos"]))
        informe(f"sorteo de la copia {k}", p_src == p, f"factor {p['factor']}, SNR {p['snr_db']:.2f}, ruido {p['ruido']}")
        z_src = AU.aumentar(y_v1, fs, (r0.inicio_s, r0.fin_s), E["ruidos"], p_src, cfg)
        informe(f"aumentar (copia {k})", np.array_equal(z_src, Aa["y"]))
        rk = PR.preprocesar(z_src, fs, cfg)
        ruta_k = C["voz"] / "prepro" / E["particion"] / f"{stem}_k{k}.wav"
        if ruta_k.exists() and rk.valido:
            kk = sf.read(str(ruta_k), dtype="float32")[0].reshape(-1)
            informe(f"data/voz/prepro/{E['particion']}/{stem}_k{k}.wav",
                    len(kk) == len(Pk["y"]) and float(np.abs(kk - Pk["y"]).max()) <= 1e-6)
        fila = next((r for r in E["prepro"] if r["fuente"] == E["archivo_v1"] and r["copia"] == str(k)), None)
        if fila:
            informe(f"metadatos de V2, copia {k}", fila["factor"] == f"{p['factor']:.2f}" and fila["snr_db"] == f"{p['snr_db']:.2f}"
                    and fila["ruido"] == E["nombres_ruido"][p["ruido"]])
    return all(oks)


# ============================================================================= main
def main() -> int:
    global RAIZ, SALIDA, MOSTRAR, plt, C
    ap = argparse.ArgumentParser(description="V1 + V2 paso a paso: de la grabación a la palabra lista para V3.")
    ap.add_argument("--toma", default="p01_reconocer_01", help="toma de la notebook, sin extensión (default p01_reconocer_01)")
    ap.add_argument("--mostrar", action="store_true", help="abre cada figura y espera que la cierres antes de seguir")
    ap.add_argument("--verificar", action="store_true", help="compara con src/voz y con los WAV de data/voz/dataset y prepro")
    ap.add_argument("--salida", type=Path, help="carpeta de salida (default data/v1v2_paso_a_paso/<toma>)")
    ap.add_argument("--raiz", type=Path, default=RAIZ, help="raíz del repo (default: la de este script)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import matplotlib
    if not args.mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    plt = _plt
    MOSTRAR = args.mostrar
    RAIZ = args.raiz
    C = cargar_valores(RAIZ)

    codigo = args.toma.split("_")[0]
    fuente_rel = f"notebook/{codigo}/{args.toma}.wav"
    if not (C["voz"] / fuente_rel).exists():
        print(f"No existe {C['voz'] / fuente_rel}. Las tomas se llaman como pXX_<comando>_<toma>, por ejemplo p01_salir_03.")
        return 2
    meta = leer_csv(C["voz"] / "dataset" / "metadatos.csv")
    fila = next((r for r in meta if r["fuente"] == fuente_rel), None)
    if fila is None:
        print(f"{fuente_rel} no está en data/voz/dataset/metadatos.csv (se excluyó en V1: ver excluidas.csv).")
        return 2
    SALIDA = args.salida or (RAIZ / "data" / "v1v2_paso_a_paso" / args.toma)
    SALIDA.mkdir(parents=True, exist_ok=True)
    print(f"V1 + V2 paso a paso · {fuente_rel} ({fila['particion']})\nFiguras en: {SALIDA}")

    E: dict = dict(
        fuente_rel=fuente_rel, codigo=codigo, clase=fila["clase"], particion=fila["particion"],
        archivo_v1=fila["archivo"], nombre_v1=Path(fila["archivo"]).name, comandos=C["comandos"],
        v1_palabra_meta=((float(fila["palabra_inicio_s"]), float(fila["palabra_fin_s"])) if fila["palabra_inicio_s"] else None),
        meta=meta, excluidas=leer_csv(C["voz"] / "dataset" / "excluidas.csv"),
        sesiones=leer_csv(C["voz"] / "notebook" / "sesiones.csv"),
        prepro=leer_csv(C["voz"] / "prepro" / "metadatos.csv"),
    )
    paso_1_fuentes(E)
    paso_2_sesion(E)
    paso_3_formato(E)
    paso_4_detector(E)
    paso_5_reglas(E)
    paso_6_calidad(E)
    paso_7_particiones(E)
    paso_8_detectar_antes(E)
    valida = E["P0"]["valido"]
    if valida:
        paso_9_pasabanda(E)
        paso_10_preenfasis(E)
        paso_11_recorte(E)
        paso_12_nivel(E)
        paso_13_velocidad(E)
        paso_14_ruido(E)
    else:
        print(f"\n   {E['nombre_v1']} no es válida para V2 («{E['P0']['motivo']}»): el agente la contesta con «no entendí» "
              f"y no sigue la cadena. Los pasos 9 a 14 se saltean; probá con un comando, por ejemplo --toma {codigo}_reconocer_01.")
    paso_15_resultado(E)
    print("\n" + "=" * 100 + "\nResumen, audios y guion\n" + "=" * 100)
    audios = []
    if valida:
        resumen(E)
        audios = guardar_audios(E)
    escribir_guion(E, audios)
    print("\nListo. Mirá las figuras 01 a 15 en orden (o GUION.md, que las tiene con el texto de cada paso) y escuchá "
          "audios/ de 01 a 09.\nProbá con --toma p01_salir_02 o con la toma de otra persona de train para ver cómo cambia.")
    ok = True
    if args.verificar:
        ok = verificar(E)
        print("\n" + ("✔ Este script reproduce a V1 y V2 del repo, muestra por muestra." if ok else "✘ HAY DIFERENCIAS con el código del repo."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
