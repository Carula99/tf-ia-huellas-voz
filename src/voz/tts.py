"""V1 · Voces sintéticas con edge-tts.

Contrato: docs/contratos/v1_dataset.md, sección 5. El plan dice qué MP3 hay que tener; la descarga
(con internet) está separada del armado del dataset, que trabaja siempre sin conexión.

Uso, desde la raíz del repo:
    venv\\Scripts\\python -m src.voz.tts [--sobrescribir]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Callable

from src.config import RAIZ, cargar_config

Sintetizador = Callable[[str, str, str, Path], None]


def _repetidos(valores) -> list:
    vistos, repetidos = set(), []
    for v in valores:
        if v in vistos and v not in repetidos:
            repetidos.append(v)
        vistos.add(v)
    return repetidos


def plan_tts(cfg: dict) -> list[dict]:
    """Un MP3 por voz, palabra y velocidad, en ese orden, sin sintetizar nada."""
    comandos = list(cfg["voz"]["comandos"])
    t = cfg["voz"]["dataset"]["tts"]
    voces, otros, velocidades = list(t["voces"]), list(t["otros"]), list(t["velocidades"])
    if any(w in comandos for w in otros):
        raise ValueError(f"tts.otros no puede tener comandos: {[w for w in otros if w in comandos]}")
    palabras = comandos + otros
    for nombre, lista in (("voces", voces), ("palabras", palabras), ("velocidades", velocidades)):
        if _repetidos(lista):
            raise ValueError(f"tts: {nombre} repetidas: {_repetidos(lista)}")
    plan = []
    for voz in voces:
        for palabra in palabras:
            for k, velocidad in enumerate(velocidades, start=1):
                plan.append({
                    "voz": voz,
                    "clase": palabra if palabra in comandos else cfg["voz"]["clase_rechazo"],
                    "palabra": palabra,
                    "texto": palabra[:1].upper() + palabra[1:],
                    "velocidad": velocidad,
                    "k": k,
                    "archivo": f"{voz}/{palabra}_r{k}.mp3",
                })
    return plan


def _sintetizar_edge(texto: str, voz: str, velocidad: str, ruta: Path) -> None:
    """Sintetizador por defecto: el servicio de texto a voz de Microsoft Edge (necesita internet)."""
    import edge_tts

    async def _guardar():
        await edge_tts.Communicate(texto, voz, rate=velocidad).save(str(ruta))

    asyncio.run(_guardar())


def sintetizar_tts(cfg: dict, destino: str | os.PathLike | None = None,
                   sintetizador: Sintetizador | None = None,
                   sobrescribir: bool = False) -> list[Path]:
    """Escribe los MP3 del plan que falten (todos con sobrescribir=True). Devuelve las rutas escritas."""
    carpeta = RAIZ / cfg["rutas"]["voz"] / "tts" if destino is None else Path(os.fspath(destino))
    sintetizar = _sintetizar_edge if sintetizador is None else sintetizador
    escritas = []
    for entrada in plan_tts(cfg):
        ruta = carpeta / entrada["archivo"]
        if ruta.exists() and not sobrescribir:
            continue
        ruta.parent.mkdir(parents=True, exist_ok=True)
        sintetizar(entrada["texto"], entrada["voz"], entrada["velocidad"], ruta)
        escritas.append(ruta)
    return escritas


def _voces_disponibles() -> set[str]:
    import edge_tts

    return {v["ShortName"] for v in asyncio.run(edge_tts.list_voices())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Baja las voces sintéticas del plan de V1 (necesita internet).")
    p.add_argument("--sobrescribir", action="store_true", help="vuelve a bajar también las que ya están")
    a = p.parse_args(argv)
    cfg = cargar_config()
    faltan = sorted(set(cfg["voz"]["dataset"]["tts"]["voces"]) - _voces_disponibles())
    if faltan:
        print(f"Estas voces de config.yaml no existen en edge-tts: {faltan}", file=sys.stderr)
        return 1
    plan = plan_tts(cfg)
    escritas = sintetizar_tts(cfg, sobrescribir=a.sobrescribir)
    print(f"Plan: {len(plan)} MP3. Escritos ahora: {len(escritas)}. "
          f"Carpeta: {RAIZ / cfg['rutas']['voz'] / 'tts'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
