"""Copia los audios de WhatsApp a data/voz/whatsapp/, uno por archivo y con el código de cada persona.

Herramienta sin tests, como scripts/grabar_voces.py. Contrato de V1 (docs/contratos/v1_dataset.md,
sección 1): los audios se copian sin modificarlos a whatsapp/pXX/pXX_<clase>_<nn>.<ext>.

La carpeta original tiene una subcarpeta por comando y, adentro, una por persona (con su nombre). El
nombre de la persona nunca se usa ni se escribe: el código sale de la hora del nombre de cada nota de
voz ("WhatsApp Ptt 2026-10-16 at HH.MM.SS.ogg") según la tabla TANDAS, y los .m4a de la grabadora son
todos de p15. En whatsapp/origen.csv queda el comando y el nombre del archivo original de cada copia.

Uso, desde la raíz del repo:
    venv\\Scripts\\python scripts\\ordenar_whatsapp.py --origen "C:\\...\\audios whatsapp" [--sobrescribir]
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.config import cargar_config  # noqa: E402

# Tandas de notas de voz del 16/10/2026: (desde, hasta, código). Confirmadas por Santiago el 22/10;
# J (23.01) y K (23.02) son la misma persona. Solo horas y códigos: los nombres quedan fuera del repo.
FECHA_TANDAS = "2026-10-16"
TANDAS = [
    ("18.48.00", "18.49.59", "p22"),   # A
    ("18.50.00", "18.51.59", "p09"),   # B
    ("18.54.00", "18.55.59", "p23"),   # C
    ("19.09.00", "19.09.45", "p24"),   # D (celular a 16 kHz)
    ("19.09.46", "19.10.59", "p08"),   # E (celular a 48 kHz)
    ("19.14.00", "19.15.59", "p25"),   # F
    ("21.23.00", "21.26.59", "p26"),   # G
    ("21.29.00", "21.30.59", "p27"),   # H
    ("22.51.00", "22.52.59", "p28"),   # I
    ("23.01.00", "23.02.59", "p29"),   # J y K
]
CODIGO_M4A = "p15"                      # M: los "Grabación….m4a"
RE_PTT = re.compile(r"WhatsApp Ptt (\d{4}-\d{2}-\d{2}) at (\d{2}\.\d{2}\.\d{2})", re.IGNORECASE)
RE_GRABACION = re.compile(r"^Grabaci.n(?: \((\d+)\))?$", re.IGNORECASE)
EXTENSIONES = (".ogg", ".opus", ".m4a")


def asignar(archivo: Path) -> tuple[str, tuple]:
    """(código, clave de orden) de un audio original; ValueError si no se puede asignar."""
    m = RE_PTT.search(archivo.name)
    if m:
        fecha, hora = m.groups()
        if fecha == FECHA_TANDAS:
            for desde, hasta, codigo in TANDAS:
                if desde <= hora <= hasta:
                    return codigo, (0, hora, archivo.name)
        raise ValueError(f"nota de voz fuera de las tandas conocidas: {archivo.name}")
    g = RE_GRABACION.match(archivo.stem)
    if g and archivo.suffix.lower() == ".m4a":
        return CODIGO_M4A, (1, int(g.group(1) or 1), archivo.name)
    raise ValueError(f"audio sin regla para asignarle un código: {archivo.name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Copia los audios de WhatsApp a data/voz/whatsapp/ con su código.")
    p.add_argument("--origen", required=True, help="carpeta con una subcarpeta por comando")
    p.add_argument("--sobrescribir", action="store_true", help="borra las copias anteriores y vuelve a copiar")
    a = p.parse_args(argv)

    cfg = cargar_config()
    clases = list(cfg["voz"]["comandos"]) + [cfg["voz"]["clase_rechazo"]]
    origen = Path(a.origen)
    destino = RAIZ / cfg["rutas"]["voz"] / "whatsapp"

    por_codigo_clase: dict[tuple[str, str], list[tuple[tuple, Path]]] = defaultdict(list)
    errores = []
    for f in sorted(origen.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in EXTENSIONES:
            continue
        clase = f.relative_to(origen).parts[0].lower()
        if clase not in clases:
            errores.append(f"carpeta de comando desconocida para {f.name}: {clase}")
            continue
        try:
            codigo, orden = asignar(f)
        except ValueError as e:
            errores.append(str(e))
            continue
        por_codigo_clase[(codigo, clase)].append((orden, f))
    if errores:
        print("No se copió nada:\n  " + "\n  ".join(errores), file=sys.stderr)
        return 1

    previas = [x for x in destino.glob("p*/*") if x.suffix.lower() in EXTENSIONES]
    if previas and not a.sobrescribir:
        print(f"{destino} ya tiene {len(previas)} audios: usar --sobrescribir.", file=sys.stderr)
        return 1
    for x in previas:
        x.unlink()

    filas = []
    for (codigo, clase), lista in sorted(por_codigo_clase.items()):
        for nn, (_, f) in enumerate(sorted(lista), start=1):
            nuevo = destino / codigo / f"{codigo}_{clase}_{nn:02d}{f.suffix.lower()}"
            nuevo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, nuevo)
            filas.append({"archivo": nuevo.relative_to(destino).as_posix(), "comando": clase, "original": f.name})
    with open(destino / "origen.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["archivo", "comando", "original"], lineterminator="\n")
        w.writeheader()
        w.writerows(filas)

    conteo = Counter((fila["archivo"].split("/")[0], fila["comando"]) for fila in filas)
    print(f"Copiados {len(filas)} audios a {destino}:")
    for codigo in sorted({c for c, _ in conteo}):
        print(f"  {codigo}: " + ", ".join(f"{k} {conteo[(codigo, k)]}" for k in clases if conteo[(codigo, k)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
