"""Fase 5 · Las huellas que ofrece la app.

Contrato: docs/contratos/f5_integracion.md (versión 1.1), secciones 3.1 y 4.4:

- catalogo(): las 6 reservadas para la demo y un intruso de test por identidad intrusa (variante 1),
  leídos de metadatos.csv de H1.
- GeneradorDemo: variantes nuevas de cada identidad (desde app.variante_nueva_desde) e intrusos
  nuevos (desde app.intruso_nuevo_desde), con la receta de H1. Ninguna se usó para entrenar,
  calibrar ni elegir nada: el dataset usa las variantes 1 a 40 y los intrusos 1 a 60, y la muestra
  ampliada del prototipo de H4, las variantes 41 a 140 y los intrusos 61 a 160.
- entrada_archivo(): un PNG cualquiera ("Abrir archivo…").
"""

from __future__ import annotations

import collections
import csv
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from ..config import RAIZ
from ..huellas.generador import generar_base
from ..huellas.variantes import aplicar_variante, sortear_parametros

ORIGENES: tuple[str, ...] = ("reservada", "intruso_test", "variante_nueva", "intruso_nuevo", "archivo")
NOMBRE_METADATOS = "metadatos.csv"


@dataclass(frozen=True)
class EntradaHuella:
    nombre: str              # el rótulo de la lista
    origen: str              # uno de ORIGENES
    verdad: int | None       # identidad real (1 a k) o 0 si es intruso; None si es un archivo cualquiera
    ruta: Path | None        # el PNG (reservada, intruso_test, archivo); None si se genera
    variante: int | None     # variante de H1; None si es un archivo cualquiera


def catalogo(cfg: dict, carpeta: str | os.PathLike | None = None) -> list[EntradaHuella]:
    """Las reservadas para la demo y después un intruso de test por identidad intrusa, en el orden de
    metadatos.csv. No lee las imágenes. FileNotFoundError si no está el CSV."""
    carpeta = RAIZ / cfg["rutas"]["huellas"] if carpeta is None else Path(os.fspath(carpeta))
    ruta = carpeta / NOMBRE_METADATOS
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta} (el dataset de huellas de H1).")
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    reservadas, intrusos = [], []
    for fila in filas:
        archivo = fila["archivo"]
        stem = Path(archivo).stem
        identidad, variante = int(fila["identidad"]), int(fila["variante"])
        if int(fila["reservada_demo"]) == 1:
            reservadas.append(EntradaHuella(nombre=f"{stem} (reservada)", origen="reservada", verdad=identidad,
                                            ruta=carpeta / archivo, variante=variante))
        elif fila["particion"] == "test" and identidad == 0 and variante == 1:
            intrusos.append(EntradaHuella(nombre=stem, origen="intruso_test", verdad=0,
                                          ruta=carpeta / archivo, variante=1))
    return reservadas + intrusos


def entrada_archivo(ruta: str | os.PathLike) -> EntradaHuella:
    """Un PNG cualquiera, con su nombre de archivo como rótulo."""
    ruta = Path(os.fspath(ruta))
    return EntradaHuella(nombre=ruta.name, origen="archivo", verdad=None, ruta=ruta, variante=None)


class GeneradorDemo:
    """Genera en el momento huellas que no están en ningún lado, con la receta de H1. Guarda las bases
    que ya generó: la primera variante de una identidad tarda lo que su base (1–2 s); las demás, ~20 ms."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._bases: dict[tuple[str, int], object] = {}
        self._llamadas: collections.Counter = collections.Counter()   # por identidad, y "intruso"

    def _imagen(self, tipo: str, semilla: int, variante: int) -> np.ndarray:
        """La receta de la sección 4 de H1."""
        if (tipo, semilla) not in self._bases:
            self._bases[(tipo, semilla)] = generar_base(tipo, semilla, self.cfg)
        rng = np.random.default_rng([semilla, variante])
        return aplicar_variante(self._bases[(tipo, semilla)], sortear_parametros(rng, self.cfg), rng).imagen

    def variante_nueva(self, identidad: int) -> tuple[EntradaHuella, np.ndarray]:
        """La n-ésima variante nueva de esa identidad: v = app.variante_nueva_desde + n."""
        identidades = {int(ident["id"]): ident for ident in self.cfg["huellas"]["identidades"]}
        if isinstance(identidad, bool) or identidad not in identidades:
            raise ValueError(f"La identidad {identidad!r} no es una de {sorted(identidades)}.")
        ident = identidades[int(identidad)]
        v = int(self.cfg["app"]["variante_nueva_desde"]) + self._llamadas[int(identidad)]
        imagen = self._imagen(str(ident["tipo"]), int(ident["semilla"]), v)
        self._llamadas[int(identidad)] += 1
        entrada = EntradaHuella(nombre=f"id{int(identidad)}_v{v} (nueva)", origen="variante_nueva",
                                verdad=int(identidad), ruta=None, variante=v)
        return entrada, imagen

    def intruso_nuevo(self) -> tuple[EntradaHuella, np.ndarray]:
        """El n-ésimo intruso nuevo: i = app.intruso_nuevo_desde + n, con su variante 1."""
        intrusos = self.cfg["huellas"]["intrusos"]
        tipos = list(intrusos["tipos"])
        i = int(self.cfg["app"]["intruso_nuevo_desde"]) + self._llamadas["intruso"]
        imagen = self._imagen(str(tipos[(i - 1) % len(tipos)]), int(intrusos["semilla_base"]) + i, 1)
        self._llamadas["intruso"] += 1
        entrada = EntradaHuella(nombre=f"intruso{i}_v1 (nuevo)", origen="intruso_nuevo", verdad=0,
                                ruta=None, variante=1)
        return entrada, imagen
