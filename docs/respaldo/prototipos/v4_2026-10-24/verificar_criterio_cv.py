"""Criterio de seleccion_hiperparametros en CV por hablante (solo train, sin val ni test).
Cuenta enteros (comandos bien decididos de 231) para que los empates no dependan del redondeo.
Reusa KNN, reponderar_bloques, normalizar_l2 y calibrar_umbral_voz de verificar_contrato_v4.py.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROTO_DIR = Path(r"C:\Users\Santiago\OneDrive\Escritorio\IA 1\Trabajo final\Prototipos\v4_2026-10-24")
sys.path.insert(0, str(PROTO_DIR))

from verificar_contrato_v4 import (  # noqa: E402
    ESQUEMAS_BLOQUES, KNN, REPO, calibrar_umbral_voz, normalizar_l2, reponderar_bloques,
)
from src.config import cargar_config  # noqa: E402
from src.core_ml.pca import PCA  # noqa: E402
from src.voz.caracteristicas import largos_de_bloque  # noqa: E402


def main():
    cfg = cargar_config()
    carpeta = REPO / cfg["rutas"]["voz"] / "prepro"
    npz = np.load(carpeta / "caracteristicas.npz", allow_pickle=False)
    tr = npz["particion"] == "train"           # solo train: val y test no se usan
    X, clase, hab = npz["X"][tr], npz["clase"][tr], npz["hablante"][tr]
    origen, copia = npz["origen"][tr], npz["copia"][tr]
    with open(carpeta / "metadatos.csv", encoding="utf-8", newline="") as f:
        n_inv = sum(1 for r in csv.DictReader(f) if r["particion"] == "train" and r["origen"] == "notebook"
                    and r["copia"] == "0" and r["clase"] == "otro" and r["valido"] in ("0", "False", "false"))
    largos = largos_de_bloque(cfg)
    comandos = list(cfg["voz"]["comandos"])
    eps, cmin = float(cfg["voz"]["knn"]["eps"]), float(cfg["voz"]["knn"]["confianza_min"])
    otro_max = float(cfg["metas"]["voz_otro_aceptado_max"])
    alias = {h: g[0] for g in cfg["voz"]["dataset"]["mismo_hablante"] for h in g}
    persona = np.array([alias.get(h, h) for h in hab])
    ev = (origen == "notebook") & (copia == 0)
    personas = sorted(set(persona[ev & np.isin(clase, comandos)]))

    filas = []
    for nom, fac in ESQUEMAS_BLOQUES.items():
        Xe = reponderar_bloques(X, fac, largos)
        for pca_k in (20, 30, None):
            plie = []
            for per in personas:
                a, q = persona != per, ev & (persona == per)
                if pca_k is None:
                    Xa, Xq = normalizar_l2(Xe[a]), normalizar_l2(Xe[q])
                else:
                    p = PCA(pca_k).ajustar(Xe[a])
                    Xa, Xq = normalizar_l2(p.transformar(Xe[a])), normalizar_l2(p.transformar(Xe[q]))
                plie.append((Xa, clase[a], Xq, clase[q]))
            for k in cfg["voz"]["knn"]["k_candidatos"]:
                P, C, D, V = [], [], [], []
                for Xa, ya, Xq, yq in plie:
                    pr, cf, dg = KNN(k=k, eps=eps).ajustar(Xa, ya).votar(Xq)
                    P.append(pr); C.append(cf); D.append(dg); V.append(yq)
                P, C, D, V = map(np.concatenate, (P, C, D, V))
                tau = calibrar_umbral_voz(D, C, P, V, comandos, "otro", cmin, otro_max, n_inv)
                cmd = np.isin(V, comandos)
                ok = P == V
                c60 = (P != "otro") & (C >= cmin)
                n_tau = int((cmd & ok & c60 & (D <= tau)).sum())
                n_c60 = int((cmd & ok & c60).sum())
                n_4cl = int((cmd & ok).sum())
                filas.append((n_tau + n_c60, n_4cl, n_tau, n_c60, nom, pca_k, k, int(cmd.sum())))

    # Orden: suma desc, n_4cl desc, luego esquema (orden de ESQUEMAS_BLOQUES), PCA (20, 30, None), K asc
    ordE, ordP = list(ESQUEMAS_BLOQUES), [20, 30, None]
    filas.sort(key=lambda r: (-r[0], -r[1], ordE.index(r[4]), ordP.index(r[5]), r[6]))
    print(f"CV por hablante en train: {len(personas)} personas, {filas[0][7]} comandos k0 evaluados")
    print("suma = n_tau5 + n_c60 | n_4cl | n_tau5 | n_c60 | esquema | PCA | K")
    for r in filas[:10]:
        print(f"{r[0]:4d} | {r[1]:3d} | {r[2]:3d} | {r[3]:3d} | {r[4]:12s} | {str(r[5]):4s} | {r[6]}")
    print("\nPor n_tau5 solo (primeros 3):")
    for r in sorted(filas, key=lambda r: (-r[2], -r[0]))[:3]:
        print(f"{r[2]:4d} | suma {r[0]:4d} | n_4cl {r[1]:3d} | {r[4]:12s} | {str(r[5]):4s} | {r[6]}")


if __name__ == "__main__":
    main()

