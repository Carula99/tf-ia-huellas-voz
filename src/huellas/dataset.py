"""H1 · Dataset de huellas: plan, generación de los PNG con metadatos.csv y lámina de revisión.

Contrato: docs/contratos/h1_generador.md, sección 4. Cada imagen sale de

    base = generar_base(tipo, semilla, cfg)
    rng = np.random.default_rng([semilla, v])
    params = sortear_parametros(rng, cfg)
    variante = aplicar_variante(base, params, rng)

así que cualquier fila de metadatos.csv se puede rehacer a partir de su tipo, su semilla y
su número de variante.

Línea de comandos, desde la raíz del repo:
    venv\\Scripts\\python -m src.huellas.dataset [--sobrescribir]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from ..config import RAIZ, cargar_config
from .generador import generar_base
from .variantes import aplicar_variante, sortear_parametros

COLUMNAS: tuple[str, ...] = (
    "archivo", "particion", "identidad", "intruso", "tipo", "semilla", "variante", "reservada_demo",
    "rot_grados", "escala", "tx_px", "ty_px", "elastica_px", "presion",
    "corte_fraccion", "corte_angulo_grados", "ruido_sigma", "sal_pimienta",
    "n_nucleos", "nucleos", "n_deltas", "deltas", "ref_x", "ref_y",
)
PARTICIONES: tuple[str, ...] = ("train", "val", "test")
PARAMETROS: tuple[str, ...] = (
    "rot_grados", "escala", "tx_px", "ty_px", "elastica_px", "presion",
    "corte_fraccion", "corte_angulo_grados", "ruido_sigma", "sal_pimienta",
)


# --------------------------------------------------------------------------- plan
def plan_dataset(cfg: dict) -> list[dict]:
    """Una entrada por imagen, sin generar nada (sección 4): primero las autorizadas, por
    identidad y variante, y después los intrusos, por intruso y variante."""
    huellas = cfg["huellas"]
    total = int(huellas["variantes_por_identidad"])
    en_train, en_val = int(huellas["particion"]["train"]), int(huellas["particion"]["val"])
    reservadas = int(huellas["reservadas_demo"])
    plan = []
    for identidad in huellas["identidades"]:
        k = int(identidad["id"])
        for v in range(1, total + 1):
            particion = "train" if v <= en_train else ("val" if v <= en_train + en_val else "test")
            plan.append({
                "archivo": f"{particion}/id{k}_v{v:03d}.png", "particion": particion,
                "identidad": k, "intruso": 0, "tipo": str(identidad["tipo"]),
                "semilla": int(identidad["semilla"]), "variante": v,
                "reservada_demo": 1 if v > total - reservadas else 0,
            })
    intrusos = huellas["intrusos"]
    tipos = list(intrusos["tipos"])
    for i in range(1, int(intrusos["cantidad"]) + 1):
        particion = "val" if i <= int(intrusos["val"]) else "test"
        for v in range(1, int(intrusos["variantes_por_intruso"]) + 1):
            plan.append({
                "archivo": f"{particion}/intruso{i:02d}_v{v}.png", "particion": particion,
                "identidad": 0, "intruso": i, "tipo": str(tipos[(i - 1) % len(tipos)]),
                "semilla": int(intrusos["semilla_base"]) + i, "variante": v, "reservada_demo": 0,
            })
    return plan


# --------------------------------------------------------------------------- generación
def _numero(valor: float) -> float:
    """Redondeo a 4 decimales, sin −0.0."""
    return round(float(valor), 4) + 0.0


def _texto(valor) -> str:
    if valor is None:
        return ""
    if isinstance(valor, float):
        return f"{valor:.4f}"
    return str(valor)


def _puntos_json(puntos) -> str:
    return json.dumps([[round(float(x), 2) + 0.0, round(float(y), 2) + 0.0] for x, y in puntos])


def _variante(entrada: dict, base, cfg: dict):
    """La receta de la sección 4 para una entrada del plan."""
    rng = np.random.default_rng([entrada["semilla"], entrada["variante"]])
    params = sortear_parametros(rng, cfg)
    return aplicar_variante(base, params, rng)


def _escribir_png(ruta: Path, imagen: np.ndarray) -> None:
    """cv2.imwrite no acepta rutas con tildes en Windows: se codifica y se escribe aparte."""
    ok, codificado = cv2.imencode(".png", imagen)
    if not ok:
        raise OSError(f"No se pudo codificar {ruta} como PNG.")
    codificado.tofile(str(ruta))


def _datos_existentes(destino: Path) -> list[Path]:
    """Lo que genera generar_dataset: PNG de train/, val/ o test/ y metadatos.csv."""
    existentes = [p for particion in PARTICIONES for p in sorted((destino / particion).glob("*.png")) if p.is_file()]
    if (destino / "metadatos.csv").is_file():
        existentes.append(destino / "metadatos.csv")
    return existentes


def generar_dataset(cfg: dict, destino: str | os.PathLike | None = None,
                    sobrescribir: bool = False) -> list[dict]:
    """Genera los PNG y metadatos.csv en destino (por defecto RAIZ / rutas.huellas).

    Si ya hay datos (PNG de train/, val/ o test/, o metadatos.csv): FileExistsError sin
    tocar nada, salvo con sobrescribir=True, que borra solo esos archivos y vuelve a generar.
    Devuelve las filas de metadatos.csv como dicts con las claves de COLUMNAS.
    """
    destino = RAIZ / cfg["rutas"]["huellas"] if destino is None else Path(os.fspath(destino))
    plan = plan_dataset(cfg)
    existentes = _datos_existentes(destino)
    if existentes and not sobrescribir:
        raise FileExistsError(
            f"{destino} ya tiene {len(existentes)} archivos del dataset (por ejemplo {existentes[0].name}). "
            "Usá sobrescribir=True (o --sobrescribir) para regenerarlo."
        )
    for ruta in existentes:
        ruta.unlink()
    for particion in sorted({e["particion"] for e in plan}):
        (destino / particion).mkdir(parents=True, exist_ok=True)

    filas, base, clave_base = [], None, None
    for entrada in plan:
        clave = (entrada["tipo"], entrada["semilla"])
        if clave != clave_base:                 # el plan agrupa las variantes de cada base
            base, clave_base = generar_base(entrada["tipo"], entrada["semilla"], cfg), clave
        variante = _variante(entrada, base, cfg)
        _escribir_png(destino / entrada["archivo"], variante.imagen)
        fila = {columna: entrada[columna] for columna in
                ("archivo", "particion", "identidad", "intruso", "tipo", "semilla", "variante", "reservada_demo")}
        fila.update({campo: _numero(getattr(variante.parametros, campo)) for campo in PARAMETROS})
        nucleos, deltas = variante.singularidades.nucleos, variante.singularidades.deltas
        referencia = variante.punto_referencia
        fila.update({
            "n_nucleos": len(nucleos), "nucleos": _puntos_json(nucleos),
            "n_deltas": len(deltas), "deltas": _puntos_json(deltas),
            "ref_x": None if referencia is None else _numero(referencia[0]),
            "ref_y": None if referencia is None else _numero(referencia[1]),
        })
        filas.append({columna: fila[columna] for columna in COLUMNAS})

    with open(destino / "metadatos.csv", "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(COLUMNAS)
        for fila in filas:
            escritor.writerow([_texto(fila[columna]) for columna in COLUMNAS])
    return filas


# --------------------------------------------------------------------------- lámina
def _elegir_variantes(entradas: list[dict], cfg: dict, cantidad: int = 6) -> list[dict]:
    """Variantes para la lámina: dos con corte (si las hay) y el resto en orden."""
    con_corte = []
    for entrada in entradas:
        params = sortear_parametros(np.random.default_rng([entrada["semilla"], entrada["variante"]]), cfg)
        if params.corte_fraccion > 0:
            con_corte.append(entrada)
    elegidas = con_corte[:2]
    elegidas += [e for e in entradas if e not in elegidas][: cantidad - len(elegidas)]
    return sorted(elegidas, key=lambda e: e["variante"])


def _cuadro(imagen: np.ndarray, nucleos, titulo: str, escala: float) -> np.ndarray:
    """Imagen en color, achicada, con los núcleos marcados y un título arriba."""
    alto, ancho = imagen.shape
    chica = cv2.resize(imagen, (int(round(ancho * escala)), int(round(alto * escala))), interpolation=cv2.INTER_AREA)
    color = cv2.cvtColor(chica, cv2.COLOR_GRAY2BGR)
    for x, y in nucleos:
        centro = (int(round(x * escala)), int(round(y * escala)))
        cv2.circle(color, centro, 9, (0, 0, 220), 2, cv2.LINE_AA)
        cv2.circle(color, centro, 2, (0, 0, 220), -1, cv2.LINE_AA)
    banda = np.full((24, color.shape[1], 3), 255, np.uint8)
    cv2.putText(banda, titulo, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([banda, color])


def generar_lamina(cfg: dict, salida: str | os.PathLike | None = None) -> Path:
    """Lámina para la revisión visual (sección 4): una fila por identidad autorizada (la base y
    6 variantes, con los núcleos marcados) y una fila con bases de intrusos, una por tipo."""
    salida = RAIZ / cfg["rutas"]["figuras"] / "grilla_h1.png" if salida is None else Path(os.fspath(salida))
    escala, columnas = 0.5, 7
    plan = plan_dataset(cfg)
    filas = []
    for identidad in cfg["huellas"]["identidades"]:
        base = generar_base(identidad["tipo"], int(identidad["semilla"]), cfg)
        cuadros = [_cuadro(base.imagen, base.singularidades.nucleos,
                           f"id{identidad['id']} {identidad['tipo']}: base", escala)]
        propias = [e for e in plan if e["identidad"] == int(identidad["id"])]
        for entrada in _elegir_variantes(propias, cfg, columnas - 1):
            variante = _variante(entrada, base, cfg)
            corte = " corte" if variante.parametros.corte_fraccion > 0 else ""
            cuadros.append(_cuadro(variante.imagen, variante.singularidades.nucleos,
                                   f"v{entrada['variante']} {variante.parametros.rot_grados:+.0f} grados{corte}", escala))
        filas.append(cuadros)
    intrusos = cfg["huellas"]["intrusos"]
    cuadros = []
    for i in range(1, min(len(intrusos["tipos"]), int(intrusos["cantidad"])) + 1):
        tipo = intrusos["tipos"][(i - 1) % len(intrusos["tipos"])]
        base = generar_base(tipo, int(intrusos["semilla_base"]) + i, cfg)
        cuadros.append(_cuadro(base.imagen, base.singularidades.nucleos, f"intruso {i}: {tipo}", escala))
    filas.append(cuadros)

    alto_cuadro, ancho_cuadro = filas[0][0].shape[:2]
    vacio = np.full((alto_cuadro, ancho_cuadro, 3), 255, np.uint8)
    separador = np.full((alto_cuadro, 6, 3), 255, np.uint8)
    renglones = []
    for cuadros in filas:
        cuadros = cuadros + [vacio] * (columnas - len(cuadros))
        partes = []
        for n, cuadro in enumerate(cuadros):
            partes += [cuadro] if n == 0 else [separador, cuadro]
        renglones += [np.hstack(partes), np.full((6, sum(p.shape[1] for p in partes), 3), 255, np.uint8)]
    lamina = np.vstack(renglones[:-1])
    salida.parent.mkdir(parents=True, exist_ok=True)
    _escribir_png(salida, lamina)
    return salida


# --------------------------------------------------------------------------- línea de comandos
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera el dataset de huellas de H1 y la lámina de revisión.")
    parser.add_argument("--sobrescribir", action="store_true", help="borra el dataset anterior y lo vuelve a generar")
    args = parser.parse_args(argv)
    cfg = cargar_config()
    inicio = time.perf_counter()
    try:
        filas = generar_dataset(cfg, sobrescribir=args.sobrescribir)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1
    medio = time.perf_counter()
    lamina = generar_lamina(cfg)
    fin = time.perf_counter()
    conteo = {p: sum(1 for f in filas if f["particion"] == p) for p in PARTICIONES}
    print(f"Dataset: {len(filas)} imágenes {conteo} en {RAIZ / cfg['rutas']['huellas']} ({medio - inicio:.1f} s)")
    print(f"Lámina: {lamina} ({fin - medio:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
