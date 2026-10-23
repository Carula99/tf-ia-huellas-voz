import pickle, sys, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "/home/claude/h3"); sys.path.insert(0, "/home/claude/h2/repo2")
import proto_h3 as F
from medir_h3 import filas, escalar, metadatos, pca_propio
part, ident, corte = metadatos(); tr = part == "train"

# variante con corte que más ventana pierde, de la identidad 1
cands = [i for i in range(len(filas)) if ident[i] == 1 and corte[i]]
i = max(cands, key=lambda i: 1 - filas[i]["mascara"].mean())
f = filas[i]
ter, bif, _ = F.minucias(f["esqueleto"], f["mascara"], devolver_todo=True)
cn, _ = F.crossing_number(f["esqueleto"])
crudas = np.argwhere(f["esqueleto"] & ((cn == 1) | (cn == 3)))
kept = set(map(tuple, ter)) | set(map(tuple, bif))

fig, ax = plt.subplots(1, 3, figsize=(17, 6.2))
a = ax[0]
fondo = np.where(f["mascara"], 1.0, 0.82)
a.imshow(fondo, cmap="gray", vmin=0, vmax=1)
ys, xs = np.nonzero(f["esqueleto"]); a.scatter(xs, ys, s=0.15, c="0.35")
d = F.distancia_al_borde(f["mascara"]); a.contour(d >= F.P["borde_minucia"], levels=[0.5], colors="tab:orange", linewidths=1)
desc = np.array([p for p in map(tuple, crudas) if p not in kept])
if len(desc): a.scatter(desc[:, 1], desc[:, 0], s=14, marker="x", c="tab:red", lw=1, label=f"descartadas ({len(desc)})")
t = np.array(ter); b = np.array(bif)
if len(t): a.scatter(t[:, 1], t[:, 0], s=28, facecolors="none", edgecolors="tab:blue", lw=1.3, label=f"terminaciones ({len(t)})")
if len(b): a.scatter(b[:, 1], b[:, 0], s=28, marker="s", facecolors="none", edgecolors="tab:green", lw=1.3, label=f"bifurcaciones ({len(b)})")
a.set_title(f"Minucias de {f['e']['archivo'].split('/')[1][:-4]} (corte {100*f['p'].corte_fraccion:.0f} %)\nnaranja: a {F.P['borde_minucia']:g} px del borde de la máscara", fontsize=10)
a.legend(loc="lower left", fontsize=8); a.set_xticks([]); a.set_yticks([])

a = ax[1]
a.imshow(f["realzada"], cmap="gray")
n = F.P["grilla"]; bb = np.linspace(0, 250, n + 1).round().astype(int)
for r_ in range(n):
    for c_ in range(n):
        m = f["mascara"][bb[r_]:bb[r_+1], bb[c_]:bb[c_+1]]
        falta = m.mean() < F.P["cobertura_celda"]
        a.add_patch(Rectangle((bb[c_] - 0.5, bb[r_] - 0.5), bb[c_+1] - bb[c_], bb[r_+1] - bb[r_], fill=falta, alpha=0.35 if falta else 1,
                              facecolor="tab:red", edgecolor="tab:cyan", lw=1))
        if not falta:
            sl = (slice(bb[r_], bb[r_+1]), slice(bb[c_], bb[c_+1])); o = f["orientacion"][sl][m]
            th = 0.5 * math.atan2(np.sin(2 * o).mean(), np.cos(2 * o).mean())
            cx, cy = (bb[c_] + bb[c_+1]) / 2, (bb[r_] + bb[r_+1]) / 2
            a.plot([cx - 18 * math.cos(th), cx + 18 * math.cos(th)], [cy - 18 * math.sin(th), cy + 18 * math.sin(th)], c="yellow", lw=2)
a.set_title("Grilla 5 × 5 de orientación\nrojo: cobertura < 50 %, se imputa con la media de train", fontsize=10)
a.set_xticks([]); a.set_yticks([])

a = ax[2]
X = pickle.load(open("X_edt.pkl", "rb"))["X"]
Z, _ = escalar(X, tr, ["orientacion", "gabor", "minucias"])
mu, val, vec = pca_propio(Z, tr); P = (Z - mu) @ vec[:, :2]
col = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
for k in (1, 2, 3):
    m = tr & (ident == k); a.scatter(P[m, 0], P[m, 1], c=col[k], s=26, label=f"Identidad {k} (train)")
    m = ~tr & (ident == k); a.scatter(P[m, 0], P[m, 1], facecolors="none", edgecolors=col[k], s=26, label=f"Identidad {k} (val y test)")
m = ident == 0; a.scatter(P[m, 0], P[m, 1], marker="x", c="0.45", s=18, lw=0.8, label="intrusos (val y test)")
a.set_title(f"PCA propio (orientación + Gabor + minucias)\nvarianza explicada: {100*val[0]/val.sum():.0f} % y {100*val[1]/val.sum():.0f} %", fontsize=10)
a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.legend(fontsize=7, loc="best")
plt.tight_layout(); plt.savefig("h3_proto.png", dpi=110); print("ok", f["e"]["archivo"])
