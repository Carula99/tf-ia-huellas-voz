import pickle, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, "/home/claude/h3"); sys.path.insert(0, "/home/claude/h2/repo2")
import proto_h3 as F
from medir_h3 import filas, escalar, metadatos

part, ident, corte = metadatos()
aut = ident > 0

# 1. Minucias: crudas vs filtradas, con y sin corte, por identidad
print("Minucias por cada 1000 px² de zona válida (media por grupo):")
for k in (1, 2, 3):
    for c in (False, True):
        sel = [i for i in range(len(filas)) if ident[i] == k and corte[i] == c]
        cr, fi, cerca = [], [], []
        for i in sel:
            f = filas[i]
            ter, bif, crudas = F.minucias(f["esqueleto"], f["mascara"], devolver_todo=True)
            valida = F.distancia_al_borde(f["mascara"]) >= F.P["borde_minucia"]
            area = f["mascara"].sum(); areav = valida.sum()
            cr.append(1000 * sum(crudas) / area); fi.append(1000 * (len(ter) + len(bif)) / areav)
            # crudas cerca del borde
            cn, _ = F.crossing_number(f["esqueleto"])
            d = F.distancia_al_borde(f["mascara"])
            pts = np.argwhere(f["esqueleto"] & ((cn == 1) | (cn == 3)))
            cerca.append(np.mean([d[tuple(p)] < F.P["borde_minucia"] for p in pts]))
        print(f"  identidad {k} {'con corte' if c else 'sin corte'} (n={len(sel):2d}): crudas {np.mean(cr):5.2f}  filtradas {np.mean(fi):5.2f}  | de las crudas, {100*np.mean(cerca):4.1f} % estaban a < {F.P['borde_minucia']:g} px del borde")

# 2. Grilla: celdas faltantes con y sin corte, y efecto de imputar
X = pickle.load(open("X.pkl", "rb"))["X"]
O = X["orientacion"]
print("\nCeldas de la grilla 5x5 sin huella (cobertura < 50 %):")
for c in (False, True):
    m = aut & (corte == c)
    print(f"  autorizadas {'con' if c else 'sin'} corte: {np.isnan(O[m]).mean()*25*1/2*2:.2f} celdas por huella (de 25)")
tr = part == "train"
def dist_propio(Z):
    cent = {k: Z[tr & (ident == k)].mean(0) for k in (1, 2, 3)}
    d = np.array([np.linalg.norm(Z[i] - cent[ident[i]]) if ident[i] else np.nan for i in range(len(Z))])
    return d
Z, _ = escalar(X, tr, ["orientacion", "gabor", "minucias"])
d = dist_propio(Z)
print(f"\nDistancia a su centroide (O+G+M, imputando con la media de train): sin corte {np.nanmean(d[aut & ~corte]):.3f}, con corte {np.nanmean(d[aut & corte]):.3f}")
# Alternativa: celdas faltantes calculadas igual con los pocos píxeles que haya (sin umbral de cobertura)
p = dict(F.P); p["cobertura_celda"] = 0.0
O2 = np.vstack([F.bloque_orientacion(f["orientacion"].astype(np.float64), f["mascara"], p) for f in filas])
O2[np.isnan(O2)] = 0.0   # celdas sin ningún píxel: (0, 0)
X2 = dict(X); X2["orientacion"] = O2
Z2, _ = escalar(X2, tr, ["orientacion", "gabor", "minucias"])
d2 = dist_propio(Z2)
print(f"Sin imputar (celda con los píxeles que haya, vacía = (0, 0)): sin corte {np.nanmean(d2[aut & ~corte]):.3f}, con corte {np.nanmean(d2[aut & corte]):.3f}")
# Solo bloque de orientación
Zo, _ = escalar(X, tr, ["orientacion"]); Zo2, _ = escalar(X2, tr, ["orientacion"])
do, do2 = dist_propio(Zo), dist_propio(Zo2)
print(f"Solo orientación: imputando sin/con corte {np.nanmean(do[aut & ~corte]):.3f}/{np.nanmean(do[aut & corte]):.3f}; sin imputar {np.nanmean(do2[aut & ~corte]):.3f}/{np.nanmean(do2[aut & corte]):.3f}")
