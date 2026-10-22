"""V1 · Dataset de voz: inventario de las cuatro fuentes, control de calidad y particiones.

Contrato: docs/contratos/v1_dataset.md (versión 1.1), secciones 1, 2 y 6. Junta las tomas de la
notebook, los audios de WhatsApp, las voces sintéticas y el ruido ambiente, los pasa a WAV de 16 kHz
con una palabra por clip, excluye solo lo que está mal (con el motivo) y reparte por hablante.

Uso, desde la raíz del repo:
    venv\\Scripts\\python -m src.voz.dataset [--sobrescribir]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from src.config import RAIZ, cargar_config
from src.voz.audio import guardar_wav, leer_audio, remuestrear
from src.voz.segmentacion import cortar_repeticiones, detectar_palabras
from src.voz.tts import plan_tts

COLUMNAS: tuple[str, ...] = (
    "archivo", "particion", "origen", "codigo", "hablante", "clase", "detalle",
    "estilo", "distancia_cm", "toma", "velocidad_tts", "fuente", "muestra_inicio", "muestra_fin",
    "fs_fuente", "duracion_s", "n_segmentos", "palabra_inicio_s", "palabra_fin_s", "pico", "rms_db",
)
COLUMNAS_EXCLUIDAS: tuple[str, ...] = ("fuente", "toma", "origen", "codigo", "clase", "motivo", "nota")
ORIGENES: tuple[str, ...] = ("notebook", "whatsapp", "tts", "ruido")
MOTIVOS: tuple[str, ...] = ("whatsapp_val_test", "notas", "saturada", "sin_palabra",
                            "varias_palabras", "cortada", "duracion", "doble", "larga")
PARTICIONES: tuple[str, ...] = ("train", "val", "test")

ENTERAS = ("distancia_cm", "toma", "muestra_inicio", "muestra_fin", "fs_fuente", "n_segmentos")
DECIMALES = {"duracion_s": 4, "pico": 4, "palabra_inicio_s": 3, "palabra_fin_s": 3, "rms_db": 2}
EXT_WHATSAPP = (".ogg", ".opus", ".m4a")
RE_CODIGO = re.compile(r"^p\d{2}$")


# --------------------------------------------------------------------------------------------
# Fuentes: qué archivos hay, con qué etiquetas (sin leer audio)
# --------------------------------------------------------------------------------------------

@dataclass
class _Fuente:
    origen: str
    ruta: Path                   # archivo crudo
    rel: str                     # relativo a la carpeta fuente, con "/"
    clase: str
    detalle: str | None = None
    codigo: str | None = None
    hablante: str | None = None
    particion: str | None = None
    estilo: str | None = None
    distancia_cm: int | None = None
    toma: int | None = None      # notebook: la de sesiones.csv; tts: k de la velocidad
    velocidad_tts: str | None = None
    excluir_todo: str | None = None          # nota de notas.txt para el archivo entero
    excluir_cortes: dict = field(default_factory=dict)   # WhatsApp: {k: nota}


def _texto(ruta: Path) -> str:
    datos = ruta.read_bytes()
    try:
        return datos.decode("utf-8-sig")
    except UnicodeDecodeError:
        return datos.decode("cp1252")


def _vacio_a_none(v):
    v = (v or "").strip()
    return v or None


def _leer_notas(ruta: Path) -> dict[tuple[str, int | None], str]:
    """{(archivo relativo, corte o None): motivo} de las líneas `excluir: <archivo>[#k] | <motivo>`."""
    if not ruta.is_file():
        return {}
    notas = {}
    for linea in _texto(ruta).splitlines():
        linea = linea.strip()
        if not linea.lower().startswith("excluir:"):
            continue
        cuerpo = linea[len("excluir:"):]
        archivo, _, motivo = cuerpo.partition("|")
        archivo = archivo.strip().replace("\\", "/")
        k = None
        if "#" in archivo:
            archivo, _, num = archivo.rpartition("#")
            k = int(num)
        notas[(archivo, k)] = motivo.strip()
    faltan = sorted({a for a, _ in notas if not (ruta.parent / a).is_file()})
    if faltan:
        raise ValueError(f"{ruta}: las exclusiones nombran archivos que no existen: {', '.join(faltan)}")
    return notas


def _hablante(codigo: str, grupos: list[list[str]]) -> str:
    for grupo in grupos:
        if codigo in grupo:
            return grupo[0]
    return codigo


def _largo_16k(ruta: Path, fs_destino: int) -> int:
    """Muestras que tendrá el archivo después de remuestrear (resample_poly redondea hacia arriba)."""
    info = sf.info(str(ruta))
    if info.samplerate == fs_destino:
        return info.frames
    from math import gcd

    g = gcd(info.samplerate, fs_destino)
    up, down = fs_destino // g, info.samplerate // g
    return -(-info.frames * up // down)


def _fuentes(cfg: dict, carpeta: Path) -> list[_Fuente]:
    """Todas las fuentes, ya validadas y con su partición, en el orden del contrato."""
    voz, d = cfg["voz"], cfg["voz"]["dataset"]
    comandos, otro = list(voz["comandos"]), voz["clase_rechazo"]
    grupos = [list(g) for g in d["mismo_hablante"]]
    fuentes: list[_Fuente] = []

    # Notebook: sesiones.csv es la fuente de las etiquetas; vale la última fila de cada archivo.
    nb = carpeta / "notebook"
    filas: dict[str, dict] = {}
    for fila in csv.DictReader(io.StringIO(_texto(nb / "sesiones.csv"), newline="")):
        filas[fila["archivo"].strip().replace("\\", "/")] = fila
    sin_archivo = sorted(a for a in filas if not (nb / a).is_file())
    en_disco = sorted(p.relative_to(nb).as_posix() for p in nb.glob("*/*.wav")
                      if p.parent.name == "ruido" or RE_CODIGO.match(p.parent.name))
    sin_fila = [a for a in en_disco if a not in filas]
    if sin_archivo or sin_fila:
        partes = []
        if sin_archivo:
            partes.append("filas de sesiones.csv sin su archivo: " + ", ".join(sin_archivo))
        if sin_fila:
            partes.append("archivos .wav sin su fila en sesiones.csv: " + ", ".join(sin_fila))
        raise ValueError(f"{nb}: " + "; ".join(partes))
    notas_nb = _leer_notas(nb / "notas.txt")
    for archivo, fila in filas.items():
        clase = fila["clase"].strip()
        f = _Fuente(origen="ruido" if clase == "ruido" else "notebook", ruta=nb / archivo,
                    rel=f"notebook/{archivo}", clase=otro if clase == "ruido" else clase)
        if f.origen == "ruido":
            f.detalle, f.hablante = "ruido", f"ruido:{Path(archivo).stem}"
        else:
            f.codigo = fila["persona"].strip()
            f.detalle = _vacio_a_none(fila.get("detalle"))
            f.estilo = _vacio_a_none(fila.get("estilo"))
            dist = _vacio_a_none(fila.get("distancia_cm"))
            f.distancia_cm = int(float(dist)) if dist is not None else None
            f.toma = int(float(fila["toma"]))
            f.excluir_todo = notas_nb.get((archivo, None))
        fuentes.append(f)

    # WhatsApp: pXX/pXX_<clase>_<nn>.<ext>, copiados por scripts/ordenar_whatsapp.py.
    wa = carpeta / "whatsapp"
    if wa.is_dir():
        malos = []
        patron = re.compile(rf"^(p\d{{2}})_({'|'.join(map(re.escape, comandos + [otro]))})_(\d{{2}})$")
        audios = sorted(p for p in wa.rglob("*") if p.is_file() and p.suffix.lower() in EXT_WHATSAPP)
        notas_wa = _leer_notas(wa / "notas.txt")
        for p in audios:
            rel = p.relative_to(wa).as_posix()
            m = patron.match(p.stem)
            if not m or p.parent.parent != wa or p.parent.name != m.group(1):
                malos.append(rel)
                continue
            clase = m.group(2)
            f = _Fuente(origen="whatsapp", ruta=p, rel=f"whatsapp/{rel}", clase=clase,
                        detalle=clase if clase in comandos else "libre", codigo=m.group(1),
                        excluir_todo=notas_wa.get((rel, None)),
                        excluir_cortes={k: v for (a, k), v in notas_wa.items() if a == rel and k is not None})
            fuentes.append(f)
        if malos:
            raise ValueError(f"{wa}: audios con un nombre que no sigue pXX_<clase>_<nn>.<ext> en la carpeta "
                             f"de su código: {', '.join(malos)}")

    # Voces sintéticas: los MP3 del plan.
    tts = carpeta / "tts"
    plan = plan_tts(cfg)
    faltan = [e["archivo"] for e in plan if not (tts / e["archivo"]).is_file()]
    if faltan:
        raise ValueError(f"faltan {len(faltan)} MP3 del plan de TTS en {tts} (por ejemplo {faltan[0]}): "
                         f"correr python -m src.voz.tts")
    for e in plan:
        fuentes.append(_Fuente(origen="tts", ruta=tts / e["archivo"], rel=f"tts/{e['archivo']}",
                               clase=e["clase"], detalle=e["palabra"], hablante=e["voz"], particion="train",
                               toma=e["k"], velocidad_tts=e["velocidad"]))

    _asignar_particiones(cfg, fuentes, grupos)
    _validar_ruido(cfg, fuentes)
    fuentes.sort(key=lambda f: (ORIGENES.index(f.origen), f.rel))
    return fuentes


def _asignar_particiones(cfg: dict, fuentes: list[_Fuente], grupos: list[list[str]]) -> None:
    d = cfg["voz"]["dataset"]
    listas = d["particiones"]
    donde: dict[str, list[str]] = {}
    for p in PARTICIONES:
        for c in listas[p]:
            donde.setdefault(c, []).append(p)
    errores = []
    en_dos = sorted(c for c, ps in donde.items() if len(ps) > 1)
    if en_dos:
        errores.append("códigos en más de una partición: " + ", ".join(en_dos))
    reales = [f for f in fuentes if f.origen in ("notebook", "whatsapp")]
    con_audios = sorted({f.codigo for f in reales})
    sin_lista = [c for c in con_audios if c not in donde]
    if sin_lista:
        errores.append("códigos con audios que no están en ninguna partición: " + ", ".join(sin_lista))
    for grupo in grupos:
        ps = {donde[c][0] for c in grupo if c in donde and len(donde[c]) == 1}
        if len(ps) > 1:
            errores.append(f"los códigos de la misma persona {', '.join(grupo)} están en particiones distintas")
    if not errores:
        con_notebook = {_hablante(f.codigo, grupos) for f in reales if f.origen == "notebook"}
        solo_wa = sorted({f.codigo for f in reales
                          if _hablante(f.codigo, grupos) not in con_notebook and donde[f.codigo][0] != "train"})
        if solo_wa:
            errores.append("hablantes sin sesiones en la notebook (solo WhatsApp) en val o test: " + ", ".join(solo_wa))
    if errores:
        raise ValueError("voz.dataset.particiones: " + "; ".join(errores))
    for f in reales:
        f.hablante = _hablante(f.codigo, grupos)
        f.particion = donde[f.codigo][0]


def _validar_ruido(cfg: dict, fuentes: list[_Fuente]) -> None:
    d = cfg["voz"]["dataset"]
    listas = d["ruido"]["particiones"]
    fs = int(cfg["voz"]["frecuencia_muestreo"])
    n = int(d["ruido"]["clips_por_archivo"])
    c = int(round(d["clip_max_s"] * fs))
    errores = []
    for f in fuentes:
        if f.origen != "ruido":
            continue
        nombre = f.ruta.stem
        ps = [p for p in PARTICIONES if nombre in listas[p]]
        if len(ps) != 1:
            errores.append(f"{nombre} está en {len(ps)} listas de voz.dataset.ruido.particiones (tiene que estar en una)")
            continue
        f.particion = ps[0]
        largo = _largo_16k(f.ruta, fs)
        if largo < n * c:
            errores.append(f"{nombre} tiene {largo / fs:.2f} s y no alcanza para {n} clips de "
                           f"{d['clip_max_s']} s sin solaparse")
    if errores:
        raise ValueError("ruido ambiente: " + "; ".join(errores))


# --------------------------------------------------------------------------------------------
# Clips y control de calidad
# --------------------------------------------------------------------------------------------

def _rms_db(y: np.ndarray) -> float:
    if len(y) == 0:
        return -120.0
    return 20.0 * float(np.log10(max(float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2))), 1e-6)))


def _coma(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def _clips(cfg: dict, f: _Fuente, y: np.ndarray, fs: int) -> list[tuple[int, int, int, tuple | None]]:
    """(toma, muestra_inicio, muestra_fin, corte) de cada clip de la fuente, en orden."""
    d = cfg["voz"]["dataset"]
    if f.origen in ("notebook", "tts"):
        return [(f.toma, 0, len(y), None)]
    if f.origen == "ruido":
        n = int(d["ruido"]["clips_por_archivo"])
        c = int(round(d["clip_max_s"] * fs))
        inicios = [0] if n == 1 else [round(k * (len(y) - c) / (n - 1)) for k in range(n)]
        return [(k + 1, i, i + c, None) for k, i in enumerate(inicios)]
    cortes = cortar_repeticiones(y, fs, cfg)
    return [(k, c.muestra_inicio, c.muestra_fin, (c.inicio_s, c.fin_s)) for k, c in enumerate(cortes, start=1)]


def _motivo(cfg: dict, f: _Fuente, toma: int, clip: np.ndarray, segmentos: list, palabra_corte,
            dur_fuente: float, mediana: float | None, n_cortes: int) -> tuple[str | None, str | None]:
    voz, d = cfg["voz"], cfg["voz"]["dataset"]
    comando = f.clase in voz["comandos"]
    if f.origen == "whatsapp" and f.particion in ("val", "test"):
        return "whatsapp_val_test", f"su hablante está en {f.particion}"
    nota = f.excluir_todo if f.excluir_todo is not None else f.excluir_cortes.get(toma)
    if nota is not None:
        return "notas", nota or None
    frac = float(np.mean(np.abs(clip) >= d["saturacion"])) if len(clip) else 0.0
    if frac > d["saturacion_frac"]:
        return "saturada", f"{100 * frac:.1f} % de las muestras en el tope".replace(".", ",")
    if comando:
        if not segmentos:
            return "sin_palabra", None
        if len(segmentos) > 1:
            return "varias_palabras", f"{len(segmentos)} palabras"
        borde = d["deteccion"]["margen_borde_s"]
        if f.origen == "notebook":
            a, b = segmentos[0]
        elif f.origen == "whatsapp":
            a, b = palabra_corte
        else:
            a, b = borde, dur_fuente - borde   # las voces sintéticas nunca están cortadas
        if f.origen in ("notebook", "whatsapp") and (a < borde or b > dur_fuente - borde):
            return "cortada", f"palabra de {_coma(a)} a {_coma(b)} s en un audio de {_coma(dur_fuente)} s"
        lo, hi = voz["duracion_palabra_s"]
        dur = segmentos[0][1] - segmentos[0][0]
        if not lo <= dur <= hi:
            return "duracion", f"duración {_coma(dur)} s"
        if f.origen == "whatsapp" and n_cortes >= 3:
            dw = palabra_corte[1] - palabra_corte[0]
            if dw > d["whatsapp"]["factor_doble"] * mediana:
                return "doble", f"dura {_coma(dw)} s y la mediana del audio es {_coma(mediana)} s"
    elif f.origen == "whatsapp":
        dw = palabra_corte[1] - palabra_corte[0]
        if dw > d["clip_max_s"]:
            return "larga", f"la palabra dura {_coma(dw)} s"
    return None, None


def _recorrer(cfg: dict, carpeta: Path) -> Iterator[tuple[dict, np.ndarray]]:
    """(candidato, clip a 16 kHz) de cada clip posible, en el orden de metadatos.csv."""
    fs = int(cfg["voz"]["frecuencia_muestreo"])
    for f in _fuentes(cfg, carpeta):
        y0, fs0 = leer_audio(f.ruta)
        y = remuestrear(y0, fs0, fs)
        clips = _clips(cfg, f, y, fs)
        cortes = [c for *_, c in clips if c is not None]
        mediana = statistics.median(b - a for a, b in cortes) if cortes else None
        for toma, m0, m1, corte in clips:
            clip = y[m0:m1]
            segmentos = detectar_palabras(clip, fs, cfg)
            motivo, nota = _motivo(cfg, f, toma, clip, segmentos, corte, len(y) / fs, mediana, len(cortes))
            prefijo = {"notebook": "nb", "whatsapp": "wa", "tts": "tts", "ruido": "ruido"}[f.origen]
            if f.origen == "notebook":
                nombre = f"nb_{f.ruta.stem}"
            elif f.origen == "whatsapp":
                nombre = f"wa_{f.ruta.stem}_s{toma:02d}"
            elif f.origen == "tts":
                nombre = f"tts_{f.hablante}_{f.detalle}_r{toma}"
            else:
                nombre = f"{prefijo}_{f.ruta.stem}_c{toma:02d}"
            entra = motivo is None
            cand = {
                "archivo": f"{f.particion}/{nombre}.wav" if entra else None,
                "particion": f.particion if entra else None,
                "origen": f.origen,
                "codigo": f.codigo,
                "hablante": f.hablante,
                "clase": f.clase,
                "detalle": f.detalle,
                "estilo": f.estilo,
                "distancia_cm": f.distancia_cm,
                "toma": int(toma),
                "velocidad_tts": f.velocidad_tts,
                "fuente": f.rel,
                "muestra_inicio": int(m0),
                "muestra_fin": int(m1),
                "fs_fuente": int(fs0),
                "duracion_s": round((m1 - m0) / fs, 4),
                "n_segmentos": len(segmentos),
                "palabra_inicio_s": round(segmentos[0][0], 3) if segmentos else None,
                "palabra_fin_s": round(segmentos[-1][1], 3) if segmentos else None,
                "pico": round(float(np.max(np.abs(clip))) if len(clip) else 0.0, 4),
                "rms_db": round(_rms_db(clip), 2),
                "motivo": motivo,
                "nota": nota,
            }
            yield cand, clip


def inventario(cfg: dict, fuente: str | os.PathLike | None = None) -> list[dict]:
    """Un candidato por clip posible de las cuatro fuentes, sin escribir nada."""
    carpeta = RAIZ / cfg["rutas"]["voz"] if fuente is None else Path(os.fspath(fuente))
    return [cand for cand, _ in _recorrer(cfg, carpeta)]


# --------------------------------------------------------------------------------------------
# Escritura
# --------------------------------------------------------------------------------------------

def _propios(destino: Path) -> list[Path]:
    """Lo que generar_dataset considera suyo en destino: WAV de train/val/test y los dos CSV."""
    propios = [p for part in PARTICIONES for p in sorted((destino / part).glob("*.wav")) if p.is_file()]
    propios += [destino / n for n in ("metadatos.csv", "excluidas.csv") if (destino / n).is_file()]
    return propios


def _celda(v) -> str:
    return "" if v is None else str(v)


def _escribir_csv(ruta: Path, columnas: tuple[str, ...], filas: list[dict]) -> None:
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columnas)
        for fila in filas:
            w.writerow([_celda(fila[c]) for c in columnas])


def generar_dataset(cfg: dict, fuente: str | os.PathLike | None = None,
                    destino: str | os.PathLike | None = None,
                    sobrescribir: bool = False) -> tuple[list[dict], list[dict]]:
    """Escribe los WAV, metadatos.csv y excluidas.csv en destino y devuelve sus filas."""
    carpeta = RAIZ / cfg["rutas"]["voz"] if fuente is None else Path(os.fspath(fuente))
    salida = RAIZ / cfg["rutas"]["voz"] / "dataset" if destino is None else Path(os.fspath(destino))
    fs = int(cfg["voz"]["frecuencia_muestreo"])
    _fuentes(cfg, carpeta)                      # valida todo antes de tocar el destino
    previos = _propios(salida)
    if previos and not sobrescribir:
        raise FileExistsError(f"{salida} ya tiene datos ({previos[0].name} y otros): usar sobrescribir=True")
    for p in previos:
        p.unlink()
    metadatos, excluidas = [], []
    for cand, clip in _recorrer(cfg, carpeta):
        if cand["motivo"] is None:
            guardar_wav(salida / cand["archivo"], clip, fs)
            metadatos.append({c: cand[c] for c in COLUMNAS})
        else:
            excluidas.append({c: cand[c] for c in COLUMNAS_EXCLUIDAS})
    salida.mkdir(parents=True, exist_ok=True)
    _escribir_csv(salida / "metadatos.csv", COLUMNAS, metadatos)
    _escribir_csv(salida / "excluidas.csv", COLUMNAS_EXCLUIDAS, excluidas)
    return metadatos, excluidas


def muestra_revision(filas: list[dict], n: int, semilla: int) -> list[str]:
    """Clips para escuchar: de comandos de WhatsApp si alcanzan; si no, de todos los comandos."""
    comandos = [f for f in filas if f["clase"] != "otro"]
    wa = [f["archivo"] for f in comandos if f["origen"] == "whatsapp"]
    archivos = wa if len(wa) >= n else [f["archivo"] for f in comandos]
    return random.Random(semilla).sample(sorted(archivos), min(n, len(archivos)))


# --------------------------------------------------------------------------------------------
# Línea de comandos
# --------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Genera el dataset de voz de V1 con config.yaml.")
    p.add_argument("--sobrescribir", action="store_true", help="borra el dataset anterior y lo vuelve a generar")
    a = p.parse_args(argv)
    cfg = cargar_config()
    meta, excl = generar_dataset(cfg, sobrescribir=a.sobrescribir)
    destino = RAIZ / cfg["rutas"]["voz"] / "dataset"
    print(f"Dataset en {destino}: {len(meta)} clips y {len(excl)} excluidos.\n")
    clases = list(cfg["voz"]["comandos"]) + [cfg["voz"]["clase_rechazo"]]
    conteo = Counter((f["particion"], f["clase"], f["origen"]) for f in meta)
    print(f"{'partición':<10}{'origen':<10}" + "".join(f"{c:>13}" for c in clases))
    for part in PARTICIONES:
        for origen in ORIGENES:
            fila = [conteo[(part, c, origen)] for c in clases]
            if any(fila):
                print(f"{part:<10}{origen:<10}" + "".join(f"{x:>13}" for x in fila))
    hablantes = {part: len({f["hablante"] for f in meta if f["particion"] == part
                            and f["origen"] in ("notebook", "whatsapp")}) for part in PARTICIONES}
    print(f"\nHablantes reales por partición: {hablantes}")
    print("Exclusiones por motivo:", dict(Counter((f["origen"], f["motivo"]) for f in excl)))
    print("\nPara escuchar (muestra de 30):")
    for archivo in muestra_revision(meta, 30, int(cfg["semilla_global"])):
        print("  ", archivo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
