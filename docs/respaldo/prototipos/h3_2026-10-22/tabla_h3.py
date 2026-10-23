import pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, "/home/claude/h3"); sys.path.insert(0, "/home/claude/h2/repo2")
import proto_h3 as F
from medir_h3 import filas, escalar, silhouette, adelanto_h4, metadatos, contribuciones, pca_propio
from barrido_h3 import h4_far, bloques_con
part, ident, corte = metadatos(); tr = part == "train"; aut = ident > 0
X = bloques_con(F.P)
pickle.dump({"X": X, "part": part, "ident": ident, "corte": corte}, open("X_edt.pkl", "wb"))
filas_tabla = [("Orientación", ["orientacion"]), ("Gabor", ["gabor"]), ("Minucias", ["minucias"]), ("Hu", ["hu"]),
               ("Orientación + Gabor", ["orientacion", "gabor"]), ("Con Hu (4 bloques)", ["orientacion", "gabor", "minucias", "hu"]),
               ("Sin Hu (3 bloques)", ["orientacion", "gabor", "minucias"])]
for nombre, bl in filas_tabla:
    Z, info = escalar(X, tr, bl)
    r = adelanto_h4(Z, part, ident)
    print(f"{nombre:22s} col {sum(info[b]['d'] for b in bl):4d} | sil tr {silhouette(Z[tr], ident[tr]):.3f} vt {silhouette(Z[aut & ~tr], ident[aut & ~tr]):.3f} | FAR {100*r['far_test']:.1f} % FRR {100*r['frr_test']:.1f} % id {100*r['identificacion_test']:.0f} %")
print("contribuciones sin Hu:", {k: round(v, 3) for k, v in contribuciones(X, tr, ["orientacion", "gabor", "minucias"]).items()})
Z, info = escalar(X, tr, ["orientacion", "gabor", "minucias"]); print(info)
