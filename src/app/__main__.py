"""Fase 5 · La app de escritorio: control de acceso por huella, comandado por voz.

Contrato: docs/contratos/f5_integracion.md (versión 1.1), sección 3.4. Necesita los modelos
guardados de H4 y V4 (models/modelo_huellas.npz y models/modelo_voz.npz) y, para la lista de huellas
y el visor PCA, data/huellas/. Cada sesión deja un registro en data/app/ (fuera de Git, sin audio).

Desde la raíz del repo:
    venv\\Scripts\\python -m src.app                    (con el micrófono del sistema)
    venv\\Scripts\\python -m src.app --dispositivo N    (con otro micrófono)
    venv\\Scripts\\python -m src.app --listar           (lista los micrófonos y sale)
    venv\\Scripts\\python -m src.app --sin-microfono    (los comandos se dan con "Abrir audio…")
"""

from __future__ import annotations

import argparse
from datetime import datetime
import sys

from ..agente.captura import listar_microfonos
from ..config import RAIZ, cargar_config
from .controlador import Controlador


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.app",
                                     description="Control de acceso por huella, comandado por voz (TF IA I).")
    parser.add_argument("--dispositivo", type=int, default=None, help="índice del micrófono (ver --listar)")
    parser.add_argument("--listar", action="store_true", help="lista los micrófonos de entrada y sale")
    parser.add_argument("--sin-microfono", action="store_true", help="arranca sin abrir el micrófono")
    args = parser.parse_args(argv)

    if args.listar:
        try:
            microfonos = listar_microfonos()
        except Exception as e:
            print(f"No se pudieron listar los micrófonos: {e}")
            return 1
        print("Micrófonos de entrada (índice, nombre, frecuencia nativa):")
        for indice, nombre, fs in microfonos:
            print(f"  [{indice:2d}] {nombre} ({fs} Hz)")
        return 0

    cfg = cargar_config()
    registro = RAIZ / cfg["rutas"]["datos"] / "app" / f"registro_{datetime.now():%Y%m%d_%H%M%S}.csv"
    try:
        controlador = Controlador.desde_archivos(cfg, microfono=not args.sin_microfono,
                                                 dispositivo=args.dispositivo, registro=registro)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    from .ventana import ejecutar   # recién acá se importa Tk

    print(f"Registro de la sesión: {registro}")
    ejecutar(controlador)
    return 0


if __name__ == "__main__":
    sys.exit(main())
