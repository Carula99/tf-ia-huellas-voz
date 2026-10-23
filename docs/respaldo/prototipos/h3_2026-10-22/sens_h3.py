import pickle, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, "/home/claude/h3"); sys.path.insert(0, "/home/claude/h2/repo2")
import proto_h3 as F
from medir_h3 import filas, escalar, silhouette, adelanto_h4, metadatos
from barrido_h3 import h4_far, bloques_con
part, ident, corte = metadatos(); tr = part == "train"; aut = ident > 0

def margen(Z):
    cent = {k: Z[tr & (ident == k)].mean(0) for k in (1, 2, 3)}
    nt = ~tr
    D = np.stack([np.linalg.norm(Z[nt] - cent[k], axis=1) for k in (1, 2, 3)], 1).min(1)
    a, i = D[ident[nt] > 0], D[ident[nt] == 0]
    return np.percentile(i, 5) / np.percentile(a, 95), i.min() / a.max()

# cuánto del corte cae en la ventana: fracción de la ventana sin máscara, con y sin corte
fr = np.array([1 - f["mascara"].mean() for f in filas])
print(f"Ventana sin huella: sin corte {100*fr[~corte].mean():.1f} %, con corte {100*fr[corte].mean():.1f} % (máx {100*fr[corte].max():.1f} %)")

for nombre, cambios in [("base (cobertura 0,5; borde 14)", {}), ("cobertura 0,3", {"cobertura_celda": 0.3}), ("cobertura 0,7", {"cobertura_celda": 0.7}),
                        ("borde minucia 10", {"borde_minucia": 10.0}), ("borde minucia 18", {"borde_minucia": 18.0}),
                        ("sin filtro de borde", {"borde_minucia": 0.0})]:
    p = dict(F.P); p.update(cambios)
    X = bloques_con(p)
    Z, _ = escalar(X, tr, ["orientacion", "gabor", "minucias"])
    r = adelanto_h4(Z, part, ident); far5, frr5 = h4_far(Z); m95, mmin = margen(Z)
    print(f"{nombre:32s} sil tr {silhouette(Z[tr], ident[tr]):.3f} | EER test FAR {r['far_test']:.3f} FRR {r['frr_test']:.3f} | FAR≤5%: FAR {far5:.3f} FRR {frr5:.3f} | margen p5/p95 {m95:.2f} min/max {mmin:.2f}")
