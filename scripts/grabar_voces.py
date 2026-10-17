"""Grabación de voces con el micrófono de la notebook (herramienta de captura para V1).

No forma parte del pipeline con tests: solo graba y anota. Los audios quedan en
data/voz/notebook/ (fuera de Git) a la frecuencia nativa del micrófono; V2 los pasa
a 16 kHz igual que al resto. El reparto entrenamiento / test se decide después, en V1.

Uso, desde la raíz del repo:
    venv\\Scripts\\python scripts\\grabar_voces.py --listar        (ver micrófonos)
    venv\\Scripts\\python scripts\\grabar_voces.py --persona p07   (sesión de una persona)
    venv\\Scripts\\python scripts\\grabar_voces.py --ruido 60      (60 s de ruido ambiente)
Opcional: --dispositivo N para elegir el micrófono (por defecto, el del sistema).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import re
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "voz" / "notebook"
PLANILLA = DESTINO / "sesiones.csv"
DURACION_S = 3.0

COMANDOS = ["reconocer", "identificar", "salir"]
ESTILOS = [  # (estilo, distancia en cm, indicación) -> una toma de cada comando por estilo
    ("normal", 50, "normal, como hablás siempre, a unos 50 cm"),
    ("normal", 50, "normal otra vez, a unos 50 cm"),
    ("lento", 50, "lento: estirá la palabra hasta casi el doble, y arrancá apenas aparece GRABANDO"),
    ("rapido", 50, "un poco más rápido"),
    ("lejos", 80, "desde unos 80 cm, con voz firme"),
    ("cerca", 30, "más cerca, a unos 30 cm, con voz normal"),
]
OTROS = [  # clase "otro": palabras parecidas a los comandos y ruidos
    ("conocer", "Decí: CONOCER"),
    ("salida", "Decí: SALIDA"),
    ("identidad", "Decí: IDENTIDAD"),
    ("hola", "Decí: HOLA"),
    ("libre", "Decí cualquier palabra que NO sea un comando"),
    ("no_verbal", "Tosé, carraspeá o golpeá suave la mesa (sin hablar)"),
]
COLUMNAS = ["archivo", "persona", "clase", "detalle", "estilo", "distancia_cm", "toma",
            "fs", "dispositivo", "fecha", "pico", "rms", "avisos"]


# --------------------------------------------------------------------------- audio
def listar_microfonos() -> None:
    entrada_defecto = sd.default.device[0]
    print("Micrófonos disponibles (el marcado con * es el del sistema):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            marca = "*" if i == entrada_defecto else " "
            print(f" {marca} [{i:2d}] {d['name']} | {api} | {d['default_samplerate']:.0f} Hz")


def elegir_microfono(indice: int | None) -> tuple[int, int, str]:
    disp = sd.default.device[0] if indice is None else indice
    info = sd.query_devices(disp, "input")
    return int(disp), int(info["default_samplerate"]), str(info["name"])


def grabar(segundos: float, fs: int, disp: int) -> np.ndarray:
    audio = sd.rec(int(round(segundos * fs)), samplerate=fs, channels=1, dtype="float32", device=disp)
    sd.wait()
    return audio[:, 0].copy()


def revisar(y: np.ndarray, fs: int) -> list[str]:
    """Avisos de calidad: saturación, nivel muy bajo y palabra cortada en un borde."""
    avisos = []
    pico = float(np.max(np.abs(y))) if y.size else 0.0
    if pico >= 0.98:
        avisos.append("saturado: alejate un poco o hablá más bajo")
    if pico < 0.03:
        avisos.append("muy bajo: acercate o hablá más fuerte")
    trama = max(1, int(0.02 * fs))
    n = len(y) // trama
    if n >= 20:
        rms = np.sqrt(np.mean(y[: n * trama].reshape(n, trama) ** 2, axis=1))
        borde = max(1, int(0.15 * fs) // trama)
        tope = rms.max() + 1e-12
        if rms[:borde].max() > 0.35 * tope:
            avisos.append("empezó demasiado pronto: puede estar cortada al inicio")
        # El final es más estricto que el inicio: la palabra se apaga de a poco y con 0.35 no se detectaba
        # una toma real de p01 que terminó a 0,06 s del borde (0.18 del máximo). En silencio el filtro de
        # Windows deja ~0, así que 0.10 no da falsas alarmas; el inicio queda en 0.35 porque ahí el micrófono
        # deja pasar un ruidito al arrancar (hasta 0.08 del máximo en p01).
        if rms[-borde:].max() > 0.10 * tope:
            avisos.append("terminó demasiado tarde: puede estar cortada al final")
    return avisos


# --------------------------------------------------------------------------- planilla
def anotar(fila: dict) -> None:
    nueva = not PLANILLA.exists()
    with open(PLANILLA, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nueva:
            w.writeheader()
        w.writerow(fila)


def guardar(y: np.ndarray, fs: int, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(ruta), y, fs, subtype="PCM_16")


# --------------------------------------------------------------------------- sesiones
def tomas_de(persona: str) -> list[dict]:
    """Las 24 tomas de una persona (18 de comandos y 6 de 'otro'), en orden mezclado."""
    tomas = []
    for comando in COMANDOS:
        for k, (estilo, dist, indicacion) in enumerate(ESTILOS, start=1):
            tomas.append({"clase": comando, "detalle": comando, "estilo": estilo, "distancia_cm": dist,
                          "toma": k, "texto": f"Decí: {comando.upper()}  ({indicacion})"})
    for k, (palabra, texto) in enumerate(OTROS, start=1):
        tomas.append({"clase": "otro", "detalle": palabra, "estilo": "normal", "distancia_cm": 50,
                      "toma": k, "texto": f"{texto}  (a unos 50 cm)"})
    random.Random(persona).shuffle(tomas)   # mismo orden si se retoma la sesión
    return tomas


def sesion_persona(persona: str, disp: int, fs: int, nombre_disp: str) -> None:
    tomas = tomas_de(persona)
    carpeta = DESTINO / persona
    print(f"\nPersona {persona} | micrófono: {nombre_disp} ({fs} Hz) | {len(tomas)} tomas de {DURACION_S:.0f} s")
    print("Cuando aparezca GRABANDO, esperá medio segundo y decí la palabra una sola vez.")
    for n, t in enumerate(tomas, start=1):
        ruta = carpeta / f"{persona}_{t['clase']}_{t['toma']:02d}.wav"
        if t["clase"] == "otro":
            ruta = carpeta / f"{persona}_otro_{t['toma']:02d}_{t['detalle']}.wav"
        if ruta.exists():
            print(f"[{n:2d}/{len(tomas)}] ya grabada: {ruta.name}")
            continue
        while True:
            print(f"\n[{n:2d}/{len(tomas)}] {t['texto']}")
            r = input("   Enter = grabar · s = saltear · q = terminar: ").strip().lower()
            if r == "q":
                print("Sesión pausada. Para retomarla, volvé a correr el script con la misma persona.")
                return
            if r == "s":
                break
            print("   GRABANDO...", flush=True)
            y = grabar(DURACION_S, fs, disp)
            avisos = revisar(y, fs)
            pico, rms = float(np.max(np.abs(y))), float(np.sqrt(np.mean(y ** 2)))
            print(f"   listo · pico {pico:.2f} · rms {rms:.3f}")
            for a in avisos:
                print(f"   AVISO: {a}")
            r = input("   Enter = guardar · r = repetir: ").strip().lower()
            if r == "r":
                continue
            guardar(y, fs, ruta)
            anotar({"archivo": ruta.relative_to(DESTINO).as_posix(), "persona": persona, "clase": t["clase"],
                    "detalle": t["detalle"], "estilo": t["estilo"], "distancia_cm": t["distancia_cm"],
                    "toma": t["toma"], "fs": fs, "dispositivo": nombre_disp,
                    "fecha": dt.datetime.now().isoformat(timespec="seconds"),
                    "pico": round(pico, 3), "rms": round(rms, 4), "avisos": "; ".join(avisos)})
            break
    print(f"\nSesión de {persona} completa. Archivos en {carpeta}")


def sesion_ruido(segundos: int, disp: int, fs: int, nombre_disp: str) -> None:
    ruta = DESTINO / "ruido" / f"ruido_{dt.datetime.now():%Y%m%d_%H%M%S}.wav"
    print(f"\nRuido ambiente: {segundos} s con {nombre_disp} ({fs} Hz).")
    print("Notebook abierta en la mesa; nadie dice palabras (respirar, moverse y murmullo lejano está bien).")
    input("Enter para empezar...")
    print("GRABANDO...", flush=True)
    y = grabar(segundos, fs, disp)
    guardar(y, fs, ruta)
    pico, rms = float(np.max(np.abs(y))), float(np.sqrt(np.mean(y ** 2)))
    anotar({"archivo": ruta.relative_to(DESTINO).as_posix(), "persona": "", "clase": "ruido", "detalle": "ambiente",
            "estilo": "", "distancia_cm": "", "toma": 1, "fs": fs, "dispositivo": nombre_disp,
            "fecha": dt.datetime.now().isoformat(timespec="seconds"), "pico": round(pico, 3),
            "rms": round(rms, 4), "avisos": "saturado" if pico >= 0.98 else ""})
    print(f"Guardado: {ruta}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Grabación de comandos de voz con el micrófono de la notebook.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--listar", action="store_true", help="muestra los micrófonos disponibles")
    g.add_argument("--persona", help="código de la persona, p01 a p99 (nunca el nombre)")
    g.add_argument("--ruido", type=int, metavar="SEGUNDOS", help="graba ruido ambiente")
    p.add_argument("--dispositivo", type=int, help="índice del micrófono (ver --listar)")
    a = p.parse_args(argv)

    if a.listar:
        listar_microfonos()
        return 0
    disp, fs, nombre = elegir_microfono(a.dispositivo)
    DESTINO.mkdir(parents=True, exist_ok=True)
    if a.persona is not None:
        if not re.fullmatch(r"p\d{2}", a.persona):
            p.error("la persona va como código: p01, p02, … (el nombre real anotalo aparte, fuera del repo)")
        sesion_persona(a.persona, disp, fs, nombre)
    else:
        sesion_ruido(a.ruido, disp, fs, nombre)
    return 0


if __name__ == "__main__":
    sys.exit(main())
