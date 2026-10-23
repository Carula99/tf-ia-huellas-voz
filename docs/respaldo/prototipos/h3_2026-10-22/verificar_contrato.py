"""Mide, con las definiciones exactas del contrato (proto_h3d), cada criterio de h3_caracteristicas.md."""
import pickle, sys, time, math, json
import numpy as np
sys.path.insert(0, "/home/claude/h3"); sys.path.insert(0, "/home/claude/h2/repo2")
import proto_h3d as F
import proto_h3c as Fc

D = pickle.load(open("/home/claude/h3/h2_cache.pkl", "rb"))
filas = D["filas"]


class H:  # adapta la fila del caché a los atributos de HuellaPreprocesada
    def __init__(self, f):
        self.realzada = f["realzada"]; self.mascara = f["mascara"]; self.esqueleto = f["esqueleto"]
        self.orientacion = f["orientacion"].astype(np.float64)


t0 = time.time()
V, M = [], []
for f in filas:
    h = H(f)
    m = F.extraer_minucias(h)
    M.append(m)
    V.append({"orientacion": F.bloque_orientacion(h), "gabor": F.bloque_gabor(h), "minucias": F.bloque_minucias(h, m)})
dt = (time.time() - t0) / len(filas)
X = {b: np.vstack([v[b] for v in V]) for b in F.BLOQUES}
part = np.array([f["e"]["particion"] for f in filas]); ident = np.array([f["e"]["identidad"] for f in filas])
corte = np.array([f["p"].corte_fraccion > 0 for f in filas])
tr = part == "train"; va = part == "val"; te = part == "test"; aut = ident > 0
print(f"{dt*1000:.0f} ms por huella; largos", {b: X[b].shape[1] for b in F.BLOQUES}, F.largos())

# comparación con la regla de espuelas del prototipo 2
dif = 0; nt_c = nb_c = nt_d = nb_d = 0
for f, m in zip(filas, M):
    mc = Fc.extraer_minucias(H(f))
    nt_c += len(mc.terminaciones); nb_c += len(mc.bifurcaciones); nt_d += len(m.terminaciones); nb_d += len(m.bifurcaciones)
    dif += (len(mc.terminaciones) != len(m.terminaciones)) or (len(mc.bifurcaciones) != len(m.bifurcaciones))
print(f"espuelas: proto2 {nt_c} term + {nb_c} bif; contrato {nt_d} term + {nb_d} bif; huellas con conteo distinto {dif}")

esc = F.ajustar_escalador({b: X[b][tr] for b in F.BLOQUES})
Z = esc.transformar(X)
print("columnas usadas", {b: int(esc.columnas[b].sum()) for b in F.BLOQUES}, "descartadas",
      {b: list(np.nonzero(~esc.columnas[b])[0]) for b in F.BLOQUES})
print("NaN o Inf en Z:", int((~np.isfinite(Z)).sum()))
# imputadas = 0 exacto
cols = []; ini = 0
for b in F.BLOQUES:
    k = int(esc.columnas[b].sum()); cols.append((b, ini, ini + k)); ini += k
imp_ok = all(np.all(Z[:, a:c][np.isnan(X[b][:, esc.columnas[b]])] == 0) for b, a, c in cols)
print("imputadas valen 0:", imp_ok, "| medias de train en Z:", max(abs(Z[tr].mean(0)).max(), 0))
# contribuciones
for b, a, c in cols:
    Zt = Z[tr][:, a:c]; d2 = ((Zt[:, None] - Zt[None]) ** 2).sum(-1); n = len(Zt)
    print(f"  contribución {b}: {d2.sum() / (n * (n - 1)):.3f}")

from sklearn.metrics import silhouette_score
s_tr = F.silhouette(Z[tr], ident[tr]); s_sk = silhouette_score(Z[tr], ident[tr])
print(f"silhouette train {s_tr:.4f} (sklearn {s_sk:.4f}, dif {abs(s_tr-s_sk):.1e})")
for nombre, m in (("val", va & aut), ("test", te & aut), ("val+test", (va | te) & aut)):
    print(f"silhouette {nombre} autorizadas {F.silhouette(Z[m], ident[m]):.3f}")
for b, a, c in cols:
    print(f"  {b:12s} train {F.silhouette(Z[tr][:, a:c], ident[tr]):.3f} val {F.silhouette(Z[va & aut][:, a:c], ident[va & aut]):.3f}"
          f" val+test {F.silhouette(Z[(va | te) & aut][:, a:c], ident[(va | te) & aut]):.3f}")

# PCA contra sklearn
from sklearn.decomposition import PCA as SK
p = F.PCA().ajustar(Z[tr]); sk = SK().fit(Z[tr]); k = p.componentes.shape[0]
P1 = p.transformar(Z); P2 = sk.transform(Z)[:, :k]
sg = np.sign((P1 * P2).sum(0))
print(f"PCA k={k}: var dif {np.abs(p.varianza - sk.explained_variance_[:k]).max():.1e}; proyección dif (salvo signo) "
      f"{np.abs(P1 - P2 * sg).max():.1e}; prop 2 comp {p.proporcion_varianza[:3].round(3)}")

# minucias: borde y pares
dmin = min((F.distancia_al_borde(f["mascara"])[m.terminaciones[:, 1].astype(int), m.terminaciones[:, 0].astype(int)].min()
            if len(m.terminaciones) else 99) for f, m in zip(filas, M))
print("distancia mínima de una terminación al borde:", dmin)

# densidad con corte / sin corte por identidad (120 autorizadas)
dens = np.array([1000 * (len(m.terminaciones) + len(m.bifurcaciones)) / (F.distancia_al_borde(f["mascara"]) >= 14).sum() for f, m in zip(filas, M)])
cent = {k: Z[tr & (ident == k)].mean(0) for k in (1, 2, 3)}
dpropia = np.array([np.linalg.norm(Z[i] - cent[ident[i]]) if ident[i] else np.nan for i in range(len(Z))])
for k in (1, 2, 3):
    sel = ident == k
    print(f"id{k}: con corte {int((sel & corte).sum())}, densidad con/sin {dens[sel & corte].mean() / dens[sel & ~corte].mean():.3f}; "
          f"dist propia con/sin {dpropia[sel & corte].mean() / dpropia[sel & ~corte].mean():.3f}; máx train {dpropia[sel & tr].max():.3f}")
# asignación de todas las autorizadas
Dc = np.stack([np.linalg.norm(Z - cent[k], axis=1) for k in (1, 2, 3)], 1)
asig = Dc.argmin(1) + 1
print("autorizadas asignadas a su identidad:", int((asig[aut] == ident[aut]).sum()), "de", int(aut.sum()))
pickle.dump({"X": X, "Z": Z, "part": part, "ident": ident, "corte": corte, "esc": esc, "cent": cent}, open("/home/claude/h3/X_contrato.pkl", "wb"))
