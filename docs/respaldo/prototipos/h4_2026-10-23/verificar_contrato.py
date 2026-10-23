"""Mide, con las definiciones exactas (proto_h4c), lo que va en la sección 8 del contrato de H4."""
import numpy as np, time, warnings
from itertools import permutations
from sklearn.cluster import KMeans as SK
from sklearn.metrics import confusion_matrix
import proto_h4c as H

warnings.filterwarnings("ignore")
D = np.load("caracteristicas.npz", allow_pickle=False)
A = np.load("muestra_ampliada.npz", allow_pickle=False)
X, part, ident = D["X"], D["particion"], D["identidad"]
tr, va, te = part == "train", part == "val", part == "test"

# --- K-means en el dataset
t0 = time.time()
km = H.KMeans(3, 15, 300, 1e-4, semilla=2026).ajustar(X[tr])
print(f"K-means: {1000*(time.time()-t0):.0f} ms, inercia {km.inercia:.6f}, n_iter {km.n_iter}, por cluster {km.inercia_cluster.round(4)}")
mapa, pureza = H.correspondencia(km.etiquetas, ident[tr], 3)
orden = [j for k in (1, 2, 3) for j in mapa if mapa[j] == k]
C = km.centroides[orden]; ids = np.array([1, 2, 3])
r = H.dispersiones(X[tr], np.array([orden.index(j) for j in km.etiquetas]), C, 0.5)
rbar = np.sqrt(km.inercia / tr.sum())
print(f"mapa {mapa}, pureza {pureza}, r_k {r.round(6)}, r̄ {rbar:.6f}, r_k / r̄ {(r / rbar).round(3)}")
sk = SK(3, init=km.centroides, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd").fit(X[tr])
print("oráculo init=centroides: centroides", np.abs(sk.cluster_centers_ - km.centroides).max(), "inercia rel", abs(sk.inertia_ - km.inercia) / km.inercia)
for s in (0, 1, 7, 2026):
    k2 = H.KMeans(3, 15, 300, 1e-4, semilla=s).ajustar(X[tr]); print("  semilla", s, "inercia", round(k2.inercia, 6))

# --- τ en val y evaluación en test
s_v, p_v = H.puntajes(X[va], C, r, ids)
for crit, f in (("far_max", 0.01), ("eer", 0.05), ("far_max", 0.05)):
    tau = H.calibrar_umbral(s_v, p_v, ident[va], crit, f)
    far_v, frr_v = H.far_frr(s_v, p_v, ident[va], tau)
    s_t, p_t = H.puntajes(X[te], C, r, ids)
    ev = H.evaluar(H.decidir(s_t, p_t, tau), p_t, ident[te])
    s_a, p_a = H.puntajes(A["X"], C, r, ids); ea = H.evaluar(H.decidir(s_a, p_a, tau), p_a, A["identidad"])
    print(f"{crit} {f}: τ* = {tau:.6f} | val FAR {far_v:.4f} FRR {frr_v:.4f} | test ident {ev['identificacion']:.4f} FAR {ev['far']:.4f} FRR {ev['frr']:.4f} "
          f"| ampliada ident {ea['identificacion']:.4f} FAR {ea['far']:.4f} FRR {ea['frr']:.4f}")
    if crit == "far_max" and f == 0.01:
        print("  matriz test", ev["matriz"].tolist())
        a = ident[va] > 0
        print(f"  intervalo val [{s_v[a].max():.6f}, {s_v[~a].min():.6f})")
        print("  val por identidad: máx autorizada", [round(float(s_v[a & (ident[va] == k)].max()), 3) for k in (1, 2, 3)],
              "intrusos asignados", [int(((~a) & (p_v == k)).sum()) for k in (1, 2, 3)])
        tau_ok = tau

# --- K-means contra scikit-learn en datos aleatorios
rng = np.random.default_rng(5)
def datos(n, d, g, sep):
    cent = rng.normal(0, sep, (g, d)); y = rng.integers(0, g, n)
    M = rng.normal(0, 1, (d, d)) * 0.3 + np.eye(d)
    return (cent[y] + rng.normal(0, 1, (n, d)) @ M) + 3.0, y
peor_c = peor_i = 0; peor_pp = 0
for n, d, g, k in ((300, 2, 4, 4), (60, 40, 3, 3), (500, 10, 5, 5), (200, 5, 3, 2), (120, 8, 6, 6)):
    for sep in (2.0, 6.0):
        Xr, _ = datos(n, d, g, sep)
        init = Xr[rng.choice(n, k, replace=False)]
        m = H.KMeans(k, 1, 300, 1e-4).ajustar(Xr, init=init)
        s = SK(k, init=init, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd").fit(Xr)
        dc = np.abs(m.centroides - s.cluster_centers_).max(); di = abs(m.inercia - s.inertia_) / s.inertia_
        eq = np.array_equal(m.etiquetas, s.labels_)
        peor_c = max(peor_c, dc); peor_i = max(peor_i, di)
        mp = H.KMeans(k, 15, 300, 1e-4, semilla=0).ajustar(Xr); sp = SK(k, n_init=15, random_state=0).fit(Xr)
        perm = min(permutations(range(k)), key=lambda q: np.abs(sp.cluster_centers_[list(q)] - mp.centroides).max())
        rel = (mp.inercia - sp.inertia_) / sp.inertia_
        peor_pp = max(peor_pp, abs(rel))
        print(f"  ({n},{d}) g={g} k={k} sep={sep}: init dado cent {dc:.1e} inercia {di:.1e} etiquetas iguales {eq} iter {m.n_iter}/{s.n_iter_} "
              f"| k-means++ inercia rel {rel:+.2e} cent emparejados {np.abs(sp.cluster_centers_[list(perm)] - mp.centroides).max():.1e}")
print(f"peor: centroides {peor_c:.1e}, inercia {peor_i:.1e}, k-means++ inercia {peor_pp:.2e}")

# --- matriz de confusión
yv = rng.integers(0, 4, 50); yp = rng.integers(0, 4, 50)
print("matriz = sklearn:", np.array_equal(H.matriz_confusion(yv, yp, [1, 2, 3, 0]), confusion_matrix(yv, yp, labels=[1, 2, 3, 0])))

# --- ejemplos a mano para la sección 8
y = np.array([1, 2, 3, 1, 0, 0, 0, 0]); p = np.array([1, 2, 3, 1, 1, 2, 3, 1]); s = np.array([0.5, 0.8, 1.0, 0.7, 1.4, 2.0, 2.5, 3.0])
print("ejemplo separable: far_max", H.calibrar_umbral(s, p, y, "far_max", 0.01), "eer", H.calibrar_umbral(s, p, y, "eer", 0.01))
y = np.array([1, 2, 3, 1, 0, 0, 0, 0]); p = np.array([1, 2, 3, 1, 1, 2, 3, 1]); s = np.array([0.5, 0.9, 1.3, 0.7, 1.1, 1.6, 2.0, 3.0])
print("ejemplo con solape: far_max 0.01", H.calibrar_umbral(s, p, y, "far_max", 0.01), "far_max 0.30", H.calibrar_umbral(s, p, y, "far_max", 0.30),
      "eer", H.calibrar_umbral(s, p, y, "eer", 0.01))
y = np.array([1, 2, 3, 1, 0, 0, 0, 0]); p = np.array([1, 3, 3, 1, 1, 2, 3, 1]); s = np.array([0.5, 0.6, 1.0, 0.7, 1.4, 2.0, 2.5, 3.0])
print("ejemplo mal identificada: far_max", H.calibrar_umbral(s, p, y, "far_max", 0.01), "eer", H.calibrar_umbral(s, p, y, "eer", 0.01))
