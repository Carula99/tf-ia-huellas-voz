import pickle, sys, math, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, "/home/claude/h3"); sys.path.insert(0, "/home/claude/h2/repo2")
import proto_h3 as F
from medir_h3 import filas, escalar, metadatos
from src.config import cargar_config
from src.huellas.generador import generar_base
from src.huellas.variantes import ParametrosVariante, aplicar_variante
from src.huellas.preprocesamiento import preprocesar
part, ident, corte = metadatos(); tr = part == "train"; aut = ident > 0
X = pickle.load(open("X_edt.pkl", "rb"))["X"]
BL = ["orientacion", "gabor", "minucias"]

# (a) ninguna minucia a < 14 px (EDT) y (b) densidad global por identidad con y sin corte
mindist, dens = [], {}
for i, f in enumerate(filas):
    ter, bif = F.minucias(f["esqueleto"], f["mascara"])
    d = F.distancia_al_borde(f["mascara"])
    for y, x in ter + bif: mindist.append(d[y, x])
    if ident[i]:
        valida = d >= F.P["borde_minucia"]
        dens.setdefault((ident[i], corte[i]), []).append(1000 * (len(ter) + len(bif)) / valida.sum())
print(f"(a) distancia mínima de una minucia al borde: {min(mindist):.2f} px")
for k in (1, 2, 3):
    a, b = np.mean(dens[(k, False)]), np.mean(dens[(k, True)])
    print(f"(b) identidad {k}: densidad sin corte {a:.3f}, con corte {b:.3f}, cociente {b/a:.3f}")

# (d) distancia al centroide propio, con y sin corte (escalador de train)
Z, _ = escalar(X, tr, BL)
cent = {k: Z[tr & (ident == k)].mean(0) for k in (1, 2, 3)}
d = np.array([np.linalg.norm(Z[i] - cent[ident[i]]) if ident[i] else np.nan for i in range(len(Z))])
for k in (1, 2, 3):
    m = ident == k
    print(f"(d) identidad {k}: distancia media sin corte {np.nanmean(d[m & ~corte]):.3f}, con corte {np.nanmean(d[m & corte]):.3f}, cociente {np.nanmean(d[m & corte])/np.nanmean(d[m & ~corte]):.3f}; máx de train {d[m & tr].max():.3f}")

# (e) experimento directo: base con corte del 10 % hacia 0/90/180/270 vs sin corte
cfg = cargar_config("/home/claude/h2/repo2/config.yaml")
import medir_h3 as M
# estadísticas del escalador para transformar nuevos
def transformar(bloques):
    partes = []
    for b in BL:
        A = X[b].astype(float); mu = np.nanmean(A[tr], 0); sd = np.nanstd(A[tr], 0, ddof=1)
        ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-9)
        z = (bloques[b][ok] - mu[ok]) / sd[ok]; z[~np.isfinite(z)] = 0
        partes.append(z / math.sqrt(ok.sum()))
    return np.concatenate(partes)
ids = {1: ("presilla", 101), 2: ("verticilo", 202), 3: ("arco_tienda", 303)}
t0 = time.time(); n = 0
for k, (tipo, sem) in ids.items():
    base = generar_base(tipo, sem, cfg)
    for a in (0.0, 90.0, 180.0, 270.0):
        vs = []
        for frac in (0.0, 0.10):
            v = aplicar_variante(base, ParametrosVariante(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, frac, a, 0.0, 0.0), np.random.default_rng(0))
            r = preprocesar(v.imagen, cfg)
            fila = dict(orientacion=r.orientacion, mascara=r.mascara, realzada=r.realzada, esqueleto=r.esqueleto)
            vs.append(transformar(F.vector(fila))); n += 1
        dc = [np.linalg.norm(vs[1] - cent[j]) for j in (1, 2, 3)]
        print(f"(e) identidad {k} corte hacia {a:3.0f}°: asignada a {1+int(np.argmin(dc))}, dist. a su centroide {dc[k-1]:.3f} (sin corte {np.linalg.norm(vs[0]-cent[k]):.3f}), distancia entre las dos {np.linalg.norm(vs[1]-vs[0]):.3f}")
print(f"{(time.time()-t0)/n*1000:.0f} ms por huella (H2 + H3)")
