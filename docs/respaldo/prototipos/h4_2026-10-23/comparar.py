"""Compara puntajes y reglas de umbral: se ajusta con train, τ con val, se mide en la muestra
ampliada (300 autorizadas + 300 intrusos nuevos) y, aparte, en test."""
import numpy as np
import proto_h4 as P
from src_pca import PCA

D = np.load("caracteristicas.npz", allow_pickle=False)
A = np.load("muestra_ampliada.npz", allow_pickle=False)
X, part, ident, tipo = D["X"], D["particion"], D["identidad"], D["tipo"]
tr, va, te = part == "train", part == "val", part == "test"
TIPO_ID = {1: "presilla", 2: "verticilo", 3: "arco_tienda"}

C, et, _, _ = P.kmeans(X[tr], 3, 15, 300, 1e-4, semilla=2026)
mapa, _, _ = P.correspondencia(et, ident[tr])
Cid = np.stack([C[[j for j in mapa if mapa[j] == k][0]] for k in (1, 2, 3)])


def puntaje_factory(nombre):
    """Devuelve f(Z) -> (score (n,), pred (n,))."""
    if nombre == "euclidea":
        def f(Z):
            Dm = np.linalg.norm(Z[:, None] - Cid[None], axis=2); return Dm.min(1), Dm.argmin(1) + 1
        return f
    if nombre.startswith("norm_"):           # d_k / dispersión del cluster k (en train)
        Dtr = np.linalg.norm(X[tr][:, None] - Cid[None], axis=2)
        disp = {"norm_rms": [np.sqrt(np.mean(Dtr[ident[tr] == k, k - 1] ** 2)) for k in (1, 2, 3)],
                "norm_max": [Dtr[ident[tr] == k, k - 1].max() for k in (1, 2, 3)],
                "norm_media": [Dtr[ident[tr] == k, k - 1].mean() for k in (1, 2, 3)]}[nombre]
        disp = np.array(disp)
        def f(Z):
            Dm = np.linalg.norm(Z[:, None] - Cid[None], axis=2) / disp; return Dm.min(1), Dm.argmin(1) + 1
        f.disp = disp
        return f
    if nombre.startswith("pca"):             # euclídea en las m primeras componentes del PCA de train
        m = int(nombre[3:])
        p = PCA(m).ajustar(X[tr]); Ct = p.transformar(Cid)
        def f(Z):
            Dm = np.linalg.norm(p.transformar(Z)[:, None] - Ct[None], axis=2); return Dm.min(1), Dm.argmin(1) + 1
        return f
    if nombre.startswith("maha"):            # PCA m + Mahalanobis con la covarianza dentro de clase (común)
        m = int(nombre[4:])
        p = PCA(m).ajustar(X[tr]); Y = p.transformar(X[tr]); Ct = p.transformar(Cid)
        R = Y - Ct[ident[tr] - 1]; S = R.T @ R / (len(R) - 3); W = np.linalg.inv(np.linalg.cholesky(S)).T
        def f(Z):
            Dm = np.linalg.norm((p.transformar(Z)[:, None] - Ct[None]) @ W, axis=2); return Dm.min(1), Dm.argmin(1) + 1
        return f
    raise ValueError(nombre)


def tau_regla(score, pred, verdad, regla):
    aut = verdad > 0
    ok = pred[aut] == verdad[aut]
    sa, si = score[aut], score[~aut]
    if regla in ("eer_bajo", "eer_medio"):
        t, far, frr = P.barrido(score, pred, verdad)
        g = np.abs(far - frr); j = np.nonzero(g == g.min())[0]
        lo = t[j[0]]
        if regla == "eer_bajo": return lo
        # mitad del intervalo [lo, siguiente score observado) donde vale ese error
        arriba = score[score > t[j[-1]]]
        siguiente = arriba.min() if arriba.size else t[j[-1]]
        return 0.5 * (t[j[-1]] + siguiente) if regla == "eer_medio" else lo
    if regla.startswith("fmed"):             # FAR_val ≤ x y, entre esos τ, el menor FRR; τ en la mitad del intervalo
        x = float(regla[4:]) / 100
        t, far, frr = P.barrido(score, pred, verdad)
        okf = far <= x + 1e-12
        fmin = frr[okf].min(); j = np.nonzero(okf & (frr == fmin))[0]
        arriba = score[score > t[j[-1]]]
        siguiente = arriba.min() if arriba.size else t[j[-1]]
        return 0.5 * (t[j[0]] + siguiente)
    if regla.startswith("far"):              # el mayor τ con FAR_val ≤ x
        x = float(regla[3:]) / 100
        sis = np.sort(si); nmax = int(np.floor(x * len(si) + 1e-9))   # intrusos aceptables
        return np.nextafter(sis[nmax], -np.inf) if nmax < len(sis) else np.inf
    raise ValueError(regla)


def evaluar(f, tau, Z, verdad, tipos=None):
    s, pred = f(Z)
    dec = P.decidir(s, pred, tau)
    r = P.metricas(dec, verdad)
    if tipos is not None:      # FAR con intrusos del mismo tipo que la identidad a la que se acercan
        i = verdad == 0
        mismo = i & (np.array([TIPO_ID[k] for k in pred]) == tipos)
        r["far_mismo"] = float(np.mean(dec[mismo] != 0)); r["n_mismo"] = int(mismo.sum())
    return r


def separabilidad(f, Z, verdad):
    """EER sobre el conjunto (sin elegir τ en otro lado): cuán solapadas están las distribuciones."""
    s, pred = f(Z)
    t, far, frr = P.barrido(s, pred, verdad)
    j = np.argmin(np.maximum(far, frr))
    aut = verdad > 0
    # AUC aprox: P(score aut correcta < score intruso)
    sa = s[aut]; si = s[~aut]
    auc = np.mean(sa[:, None] < si[None, :]) + 0.5 * np.mean(sa[:, None] == si[None, :])
    return max(far[j], frr[j]), auc


if __name__ == "__main__":
    import sys
    Za, va_ = A["X"], A["identidad"]
    print(f"muestra ampliada: {int((va_>0).sum())} autorizadas, {int((va_==0).sum())} intrusos")
    puntajes = ["euclidea", "norm_rms", "norm_media", "norm_max", "pca2", "pca3", "pca5", "pca10", "pca20", "pca40", "pca71",
                "maha2", "maha5", "maha10", "maha20"]
    print("\nSeparabilidad (EER en el propio conjunto y AUC):")
    for n in puntajes:
        f = puntaje_factory(n)
        e_v, a_v = separabilidad(f, X[va], ident[va]); e_a, a_a = separabilidad(f, Za, va_); e_t, a_t = separabilidad(f, X[te], ident[te])
        print(f"  {n:10s} val EER {e_v:.3f} AUC {a_v:.4f} | ampliada EER {e_a:.3f} AUC {a_a:.4f} | test EER {e_t:.3f} AUC {a_t:.4f}")
    print("\nτ en val → ampliada (FRR, FAR, FAR mismo tipo) y test (FRR, FAR):")
    for n in puntajes:
        f = puntaje_factory(n)
        s_v, p_v = f(X[va])
        for regla in ("eer_bajo", "eer_medio", "far2", "far5"):
            tau = tau_regla(s_v, p_v, ident[va], regla)
            ra = evaluar(f, tau, Za, va_, A["tipo"]); rt = evaluar(f, tau, X[te], ident[te], tipo[te]); rv = evaluar(f, tau, X[va], ident[va])
            print(f"  {n:10s} {regla:9s} τ={tau:7.3f} | val FRR {rv['frr']:.3f} FAR {rv['far']:.3f} | ampliada FRR {ra['frr']:.3f} FAR {ra['far']:.3f} "
                  f"(mismo tipo {ra['far_mismo']:.3f}) | test FRR {rt['frr']:.3f} FAR {rt['far']:.3f}")
