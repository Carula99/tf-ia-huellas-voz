"""Mide los bloques de H3: separación, contribuciones, PCA propio y un adelanto de H4."""
import pickle, sys, time, math, json
import numpy as np
sys.path.insert(0, "/home/claude/h3"); sys.path.insert(0, "/home/claude/h2/repo2")
import proto_h3 as F

D = pickle.load(open("/home/claude/h3/h2_cache.pkl", "rb"))
filas = D["filas"]
BLOQUES = ["orientacion", "gabor", "minucias", "hu"]


def calcular(p=F.P):
    F._BANCO = None
    t0 = time.time()
    V = [F.vector(f, p) for f in filas]
    dt = (time.time() - t0) / len(filas)
    X = {b: np.vstack([v[b] for v in V]) for b in BLOQUES}
    return X, dt


def metadatos():
    part = np.array([f["e"]["particion"] for f in filas])
    ident = np.array([f["e"]["identidad"] for f in filas])
    corte = np.array([f["p"].corte_fraccion > 0 for f in filas])
    return part, ident, corte


def escalar(X, tr, bloques=BLOQUES, pesos=None):
    """Estandariza con train (nanmean/nanstd), imputa NaN con 0 (la media de train), saca columnas
    constantes en train y divide cada bloque por √d_b."""
    partes, info = [], {}
    for b in bloques:
        A = X[b].astype(float)
        mu = np.nanmean(A[tr], axis=0); sd = np.nanstd(A[tr], axis=0, ddof=1)
        ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-9)
        Z = (A[:, ok] - mu[ok]) / sd[ok]
        Z[~np.isfinite(Z)] = 0.0
        d = ok.sum()
        w = 1 / math.sqrt(d) if pesos is None else pesos[b]
        partes.append(Z * w)
        info[b] = dict(d=int(d), faltan=float(np.isnan(A).mean()), constantes=int((~ok).sum()))
    return np.hstack(partes), info


def silhouette(Z, y):
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(Z, y))


def contribuciones(X, tr, bloques=BLOQUES):
    out = {}
    for b in bloques:
        Z, _ = escalar(X, tr, [b])
        Zt = Z[tr]
        d2 = ((Zt[:, None, :] - Zt[None, :, :]) ** 2).sum(-1)
        n = len(Zt)
        out[b] = float(d2.sum() / (n * (n - 1)))
    return out


def pca_propio(Z, tr):
    mu = Z[tr].mean(0)
    Xc = Z[tr] - mu
    S = Xc.T @ Xc / (len(Xc) - 1)
    val, vec = np.linalg.eigh(S)
    orden = np.argsort(val)[::-1]
    return mu, val[orden], vec[:, orden]


def adelanto_h4(Z, part, ident):
    tr = part == "train"
    cent = {k: Z[tr & (ident == k)].mean(0) for k in (1, 2, 3)}
    def dist(m):
        D = np.stack([np.linalg.norm(Z[m] - cent[k], axis=1) for k in (1, 2, 3)], 1)
        return D.min(1), np.array([1, 2, 3])[D.argmin(1)]
    res = {}
    va = part == "val"; te = part == "test"
    dva, pva = dist(va); dte, pte = dist(te)
    aut_va = ident[va] > 0; aut_te = ident[te] > 0
    # τ: EER en val
    cand = np.sort(np.concatenate([dva, [0, dva.max() + 1]]))
    mejor = None
    for t in cand:
        frr = np.mean(~((dva[aut_va] <= t) & (pva[aut_va] == ident[va][aut_va])))
        far = np.mean(dva[~aut_va] <= t)
        if mejor is None or abs(far - frr) < mejor[0]:
            mejor = (abs(far - frr), t, far, frr)
    _, tau, far_v, frr_v = mejor
    frr_t = np.mean(~((dte[aut_te] <= tau) & (pte[aut_te] == ident[te][aut_te])))
    far_t = np.mean(dte[~aut_te] <= tau)
    ident_t = np.mean(pte[aut_te] == ident[te][aut_te])
    # separación: distancia del intruso más cercano vs autorizado más lejano
    res.update(tau=float(tau), far_val=float(far_v), frr_val=float(frr_v), far_test=float(far_t), frr_test=float(frr_t),
               identificacion_test=float(ident_t),
               aut_max_test=float(dte[aut_te].max()), intr_min_test=float(dte[~aut_te].min()),
               aut_p95_val=float(np.percentile(dva[aut_va], 95)), intr_p5_val=float(np.percentile(dva[~aut_va], 5)))
    return res


if __name__ == "__main__":
    X, dt = calcular()
    part, ident, corte = metadatos()
    tr = part == "train"
    print(f"{dt*1000:.0f} ms por huella")
    Z, info = escalar(X, tr)
    print("bloques:", json.dumps(info))
    print("contribución media a la distancia² en train (≈ 2 cada uno):", {k: round(v, 3) for k, v in contribuciones(X, tr).items()})
    print(f"silhouette train (todo): {silhouette(Z[tr], ident[tr]):.3f}")
    for b in BLOQUES:
        Zb, _ = escalar(X, tr, [b])
        aut = ident > 0
        print(f"  {b:12s} silhouette train {silhouette(Zb[tr], ident[tr]):.3f} | val+test autorizadas {silhouette(Zb[aut & ~tr], ident[aut & ~tr]):.3f}")
    aut = ident > 0
    print(f"silhouette val+test autorizadas (todo): {silhouette(Z[aut & ~tr], ident[aut & ~tr]):.3f}")
    mu, val, vec = pca_propio(Z, tr)
    from sklearn.decomposition import PCA
    sk = PCA().fit(Z[tr])
    k = min(len(val), len(sk.explained_variance_))
    print("PCA: varianza explicada 2/3 comp:", np.round(val[:3] / val.sum(), 3), "| sklearn max dif autovalores", float(np.max(np.abs(val[:k] - sk.explained_variance_[:k]))))
    P2 = (Z - mu) @ vec[:, :2]; Psk = sk.transform(Z)[:, :2]
    print("  proyección vs sklearn (salvo signo):", float(min(np.abs(P2 - Psk).max(), np.abs(P2 + Psk).max(), np.abs(P2 * [1, -1] - Psk).max(), np.abs(P2 * [-1, 1] - Psk).max())))
    print("adelanto H4:", json.dumps(adelanto_h4(Z, part, ident), indent=None))
    pickle.dump({"X": X, "part": part, "ident": ident, "corte": corte}, open("/home/claude/h3/X.pkl", "wb"))
