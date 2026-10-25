"""Prueba en vivo del clasificador de voz (V4) con el micrófono de la notebook.

Herramienta de verificación manual (sección 7 del contrato).
Graba 3 segundos a la frecuencia nativa del micrófono, ejecuta la cadena completa
de preprocesamiento y clasificación con el modelo entrenado, y muestra el resultado
en un bucle interactivo hasta ingresar 'q'. Antes de cada toma se escribe qué se va
a decir; cada toma (WAV) y un resultados.csv se guardan en data/voz/vivo/<fecha_hora>/
(fuera de Git). Una corrida por persona.

Uso, desde la raíz del repo:
    venv\\Scripts\\python scripts/probar_voz.py
    venv\\Scripts\\python scripts/probar_voz.py --dispositivo N
    venv\\Scripts\\python scripts/probar_voz.py --listar
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.config import cargar_config
from src.voz.clasificacion import cargar_modelo, reconocer


DURACION_S = 3.0


def listar_microfonos() -> None:
    """Muestra los dispositivos de entrada de audio disponibles."""
    entrada_defecto = sd.default.device[0]
    print("Micrófonos disponibles (el marcado con * es el del sistema):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            marca = "*" if i == entrada_defecto else " "
            print(f" {marca} [{i:2d}] {d['name']} | {api} | {d['default_samplerate']:.0f} Hz")


def elegir_microfono(indice: int | None) -> tuple[int, int, str]:
    """Devuelve (id_dispositivo, fs_nativa, nombre)."""
    disp = sd.default.device[0] if indice is None else indice
    info = sd.query_devices(disp, "input")
    return int(disp), int(info["default_samplerate"]), str(info["name"])


def grabar(segundos: float, fs: int, disp: int) -> np.ndarray:
    """Graba la cantidad de segundos indicada y devuelve un array 1D float32."""
    audio = sd.rec(int(round(segundos * fs)), samplerate=fs, channels=1, dtype="float32", device=disp)
    sd.wait()
    return audio[:, 0].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba en vivo del reconocimiento de voz V4.")
    parser.add_argument("--dispositivo", type=int, default=None, help="Índice del micrófono a usar.")
    parser.add_argument("--modelo", type=str, default=None, help="Ruta al modelo .npz (por defecto models/modelo_voz.npz).")
    parser.add_argument("--listar", action="store_true", help="Lista los micrófonos disponibles y sale.")
    args = parser.parse_args()

    if args.listar:
        listar_microfonos()
        return

    cfg = cargar_config()
    ruta_modelo = Path(args.modelo) if args.modelo is not None else RAIZ / cfg["rutas"]["modelos"] / "modelo_voz.npz"

    if not ruta_modelo.is_file():
        print(f"Error: no existe el modelo entrenado en {ruta_modelo}")
        print("Ejecutá antes: venv\\Scripts\\python -m src.voz.clasificacion --sobrescribir")
        sys.exit(1)

    print(f"Cargando modelo desde: {ruta_modelo}")
    modelo = cargar_modelo(ruta_modelo)
    print(f"Modelo cargado: esquema={modelo.esquema}, PCA={modelo.n_componentes}, K={modelo.k}, tau*={modelo.tau:.4f}")

    disp, fs_nat, nombre_disp = elegir_microfono(args.dispositivo)
    print(f"Usando micrófono [{disp}]: {nombre_disp} ({fs_nat} Hz nativo)")
    print(f"Cada toma dura {DURACION_S:.1f} s. El preprocesador remuestrea a {cfg['voz']['frecuencia_muestreo']} Hz.")

    carpeta = RAIZ / "data" / "voz" / "vivo" / datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta / "resultados.csv"
    columnas = ["toma", "archivo", "esperado", "comando", "prediccion", "confianza",
                "distancia", "tau", "valido", "motivo", "pico"]
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columnas)
    print(f"Las tomas y los resultados se guardan en {carpeta} (fuera de Git).")
    toma = 0

    print("\n" + "=" * 65)
    print("MODO DE PRUEBA EN VIVO · COMANDOS DE VOZ")
    print("Comandos válidos: 'reconocer', 'identificar', 'salir'.")
    print("Cualquier otra palabra, tos o ruido debería responder 'otro'.")
    print("=" * 65)

    while True:
        try:
            linea = input("\nEscribí qué vas a decir (ej. reconocer, salida, tos, silencio) y Enter"
                          " para grabar 3 s ('q' para salir): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if linea.lower() == "q":
            print("Fin de la prueba.")
            break

        print(">> GRABANDO 3 segundos... ¡Hablá ahora!")
        y = grabar(DURACION_S, fs_nat, disp)
        pico = float(np.max(np.abs(y))) if y.size else 0.0
        print(f">> Grabación finalizada (pico máximo: {pico:.2f}). Analizando...")

        rec = reconocer(y, fs_nat, modelo, cfg)

        toma += 1
        nombre = f"toma_{toma:02d}.wav"
        sf.write(carpeta / nombre, y, fs_nat, subtype="FLOAT")
        dist_csv = f"{rec.distancia:.4f}" if math.isfinite(rec.distancia) else "inf"
        with open(ruta_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([toma, nombre, linea, rec.comando, rec.prediccion or "",
                                    f"{rec.confianza:.4f}", dist_csv, f"{rec.tau:.4f}",
                                    int(rec.valido), rec.motivo, f"{pico:.3f}"])

        print("-" * 50)
        print(f"  COMANDO DECIDIDO:  {rec.comando.upper()}")
        print(f"  Predicción K-NN:   {rec.prediccion if rec.prediccion is not None else '(ninguna)'}")
        print(f"  Confianza:         {rec.confianza:.4f}  (mínimo: {modelo.confianza_min:.2f})")
        dist_str = f"{rec.distancia:.4f}" if math.isfinite(rec.distancia) else "inf"
        print(f"  Distancia:         {dist_str}  (umbral tau*: {rec.tau:.4f})")
        print(f"  Audio válido (V2): {rec.valido}")
        if not rec.valido:
            print(f"  Motivo descarte:   {rec.motivo}")
        else:
            if rec.comando != "otro":
                print("  Diagnóstico:       Comando aceptado.")
            else:
                if rec.prediccion == "otro":
                    print("  Diagnóstico:       K-NN votó 'otro'.")
                elif rec.confianza < modelo.confianza_min:
                    print(f"  Diagnóstico:       Rechazado por confianza baja ({rec.confianza:.2f} < {modelo.confianza_min:.2f}).")
                elif rec.distancia > rec.tau:
                    print(f"  Diagnóstico:       Rechazado por distancia alta ({rec.distancia:.4f} > {rec.tau:.4f}).")
        print("-" * 50)


if __name__ == "__main__":
    main()
