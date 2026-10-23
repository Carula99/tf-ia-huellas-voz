"""Reproduce el adelanto de H4 y compara el K-means propio con scikit-learn."""
import numpy as np
from sklearn.cluster import KMeans
import proto_h4 as P

D = np.load("caracteristicas.npz", allow_pickle=False)
X, part, ident, intr, tipo = D["X"], D["particion"], D["identidad"], D["intruso"], D["tipo"]
tr, va, te = part == "train", part == "val", part == "test"

C, et, iner, it = P.kmeans(X[tr], 3, 15, 300, 1e-4, semilla=2026)
mapa, pureza, uno = P.correspondencia(et, ident[tr])
print(f"K-means propio: inercia {iner:.4f}, iter {it}, pureza {pureza:.3f}, uno a uno {uno}, mapa {mapa}")
# centroides vs medias por identidad
Cid = np.stack([C[[j for j in mapa if mapa[j] == k][0]] for k in (1, 2, 3)])
M = np.stack([X[tr & (ident == k)].mean(0) for k in (1, 2, 3)])
print("centroides = medias por identidad:", np.abs(Cid - M).max())
# oráculo con init = centroides
sk = KMeans(3, init=C, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd").fit(X[tr])
print("init=C: dif centroides", np.abs(sk.cluster_centers_ - C).max(), "dif inercia", abs(sk.inertia_ - iner))
# oráculo con k-means++ (inicializaciones distintas, centroides emparejados)
sk2 = KMeans(3, init="k-means++", n_init=15, random_state=0).fit(X[tr])
from itertools import permutations
best = min(permutations(range(3)), key=lambda p: np.abs(sk2.cluster_centers_[list(p)] - C).max())
print("k-means++: dif centroides emparejados", np.abs(sk2.cluster_centers_[list(best)] - C).max(), "inercia rel", abs(sk2.inertia_ - iner) / iner)
# arranques distintos: ¿todas las semillas dan la misma partición?
iners = [P.kmeans(X[tr], 3, 1, 300, 1e-4, semilla=s)[2] for s in range(200)]
print("200 arranques sueltos: inercias distintas", np.unique(np.round(iners, 6))[:10], "fracción en el óptimo", np.mean(np.isclose(iners, iner)))

def dist_pred(Z, Cid):
    Dm = np.linalg.norm(Z[:, None, :] - Cid[None], axis=2)
    return Dm.min(1), Dm.argmin(1) + 1, Dm

d, pred, Dm = dist_pred(X, Cid)
for nombre, m in (("train", tr), ("val", va), ("test", te)):
    a = m & (ident > 0); i = m & (ident == 0)
    print(f"{nombre:5s} aut: max {d[a].max():.3f} p95 {np.percentile(d[a],95):.3f} media {d[a].mean():.3f} | "
          + (f"intr: min {d[i].min():.3f} p5 {np.percentile(d[i],5):.3f} media {d[i].mean():.3f}" if i.any() else "")
          + f" | ident ok {np.mean(pred[a]==ident[a]):.3f}")
# EER en val como el adelanto (primer τ con |FAR − FRR| mínimo)
t, far, frr = P.barrido(d[va], pred[va], ident[va])
j = np.argmin(np.abs(far - frr))
eq = np.nonzero(np.abs(far - frr) == np.abs(far - frr).min())[0]
print(f"EER val: τ={t[j]:.4f} FAR {far[j]:.3f} FRR {frr[j]:.3f}; τ empatados: {t[eq].round(4)}")
for tau in (t[j],):
    dec = P.decidir(d[te], pred[te], tau)
    print("test:", P.metricas(dec, ident[te]))
    print(P.matriz_confusion(dec, ident[te]))
# autorizadas de test ordenadas por distancia y sus rechazos
a = te & (ident > 0)
orden = np.argsort(-d[a])
print("autorizadas de test más lejanas:", [(str(D['archivo'][a][k]), round(float(d[a][k]), 3)) for k in orden[:5]])
i = va & (ident == 0)
print("intrusos de val más cercanos:", [(str(D['archivo'][i][k]), str(tipo[i][k]), int(pred[i][k]), round(float(d[i][k]), 3)) for k in np.argsort(d[i])[:6]])
i = te & (ident == 0)
print("intrusos de test más cercanos:", [(str(D['archivo'][i][k]), str(tipo[i][k]), int(pred[i][k]), round(float(d[i][k]), 3)) for k in np.argsort(d[i])[:6]])
a = va & (ident > 0)
print("autorizadas de val más lejanas:", [(str(D['archivo'][a][k]), round(float(d[a][k]), 3)) for k in np.argsort(-d[a])[:5]])
