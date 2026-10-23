import pickle, sys, math, json, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, "/home/claude/h3"); sys.path.insert(0, "/home/claude/h2/repo2")
import proto_h3 as F
from medir_h3 import filas, escalar, silhouette, adelanto_h4, metadatos

part, ident, corte = metadatos(); tr = part == "train"; aut = ident > 0

def bloques_con(p):
    F._BANCO = None
    V = [F.vector(f, p) for f in filas]
    return {b: np.vstack([v[b] for v in V]) for b in ["orientacion", "gabor", "minucias", "hu"]}

def h4_far(Z, far_obj=0.05):
    tr_ = part == "train"
    cent = {k: Z[tr_ & (ident == k)].mean(0) for k in (1, 2, 3)}
    def dist(m):
        D = np.stack([np.linalg.norm(Z[m] - cent[k], axis=1) for k in (1, 2, 3)], 1)
        return D.min(1), np.array([1, 2, 3])[D.argmin(1)]
    va, te = part == "val", part == "test"
    dva, pva = dist(va); dte, pte = dist(te)
    av, at = ident[va] > 0, ident[te] > 0
    intr = np.sort(dva[~av])
    tau = intr[int(np.floor(far_obj * len(intr)))] - 1e-9  # acepta a lo sumo far_obj de intrusos de val
    frr = np.mean(~((dte[at] <= tau) & (pte[at] == ident[te][at]))); far = np.mean(dte[~at] <= tau)
    return far, frr

def evaluar(nombre, X, bloques):
    Z, info = escalar(X, tr, bloques)
    s_tr = silhouette(Z[tr], ident[tr]); s_vt = silhouette(Z[aut & ~tr], ident[aut & ~tr])
    r = adelanto_h4(Z, part, ident); far5, frr5 = h4_far(Z)
    print(f"{nombre:34s} d={sum(info[b]['d'] for b in bloques):4d} sil tr {s_tr:.3f} vt {s_vt:.3f} | EER: FAR {r['far_test']:.3f} FRR {r['frr_test']:.3f} id {r['identificacion_test']:.2f} | FAR≤5%val: FAR {far5:.3f} FRR {frr5:.3f}")

base = bloques_con(F.P)
for combo in [["orientacion", "gabor", "minucias", "hu"], ["orientacion", "gabor", "minucias"], ["orientacion", "gabor"], ["orientacion"], ["gabor"], ["orientacion", "minucias"]]:
    evaluar("+".join(c[:3] for c in combo), base, combo)
for g in (4, 6, 8):
    p = dict(F.P); p["grilla"] = g
    X = bloques_con(p)
    evaluar(f"grilla {g}x{g} O+G+M", X, ["orientacion", "gabor", "minucias"])
    evaluar(f"grilla {g}x{g} O+G", X, ["orientacion", "gabor"])
p = dict(F.P); p["anillos"] = (15.0, 35.0, 55.0, 75.0, 95.0, 115.0); p["sectores"] = 8
X = bloques_con(p); evaluar("gabor 5 anillos x 8 O+G+M", X, ["orientacion", "gabor", "minucias"])
p["sectores"] = 16; X = bloques_con(p); evaluar("gabor 5 anillos x 16 O+G+M", X, ["orientacion", "gabor", "minucias"])
