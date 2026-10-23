"""Simula muchos pares val/test del tamaño del dataset (8 autorizadas por identidad y 30 intrusos
de 3 variantes en cada uno), sorteados sin repetir desde la muestra ampliada: τ se calibra en el
"val" sorteado y se mide en el "test" sorteado. Los centroides son los del train real."""
import numpy as np
import comparar as K

A = np.load("muestra_ampliada.npz", allow_pickle=False)
Za, ya, intr, tipos = A["X"], A["identidad"], A["intruso"], A["tipo"]
ids_intr = np.unique(intr[intr > 0])


def tau_por_identidad(f, Zv, yv, regla):
    s, p = f(Zv)
    taus = np.zeros(4)
    glob = K.tau_regla(s, p, yv, regla)
    for k in (1, 2, 3):
        m = (p == k)
        if not ((yv[m] == 0).any() and (yv[m] == k).any()):
            taus[k] = glob; continue                  # lo que el sistema asigna a k: sus autorizadas e intrusos
        taus[k] = K.tau_regla(s[m], p[m], np.where(yv[m] == k, k, np.where(yv[m] == 0, 0, -1)), regla)
    return taus


def simular(nombre, regla, n=2000, semilla=0):
    f = K.puntaje_factory(nombre)
    s_all, p_all = f(Za)
    rng = np.random.default_rng(semilla)
    res = []
    for _ in range(n):
        iv, it = [], []
        for k in (1, 2, 3):
            idx = rng.permutation(np.nonzero(ya == k)[0])
            iv += list(idx[:8]); it += list(idx[8:16])
        ii = rng.permutation(ids_intr)
        iv += list(np.nonzero(np.isin(intr, ii[:30]))[0]); it += list(np.nonzero(np.isin(intr, ii[30:60]))[0])
        iv, it = np.array(iv), np.array(it)
        if regla.startswith("id_"):
            taus = tau_por_identidad(f, Za[iv], ya[iv], regla[3:])
            tau_t = taus[p_all[it]]
        else:
            tau_t = K.tau_regla(s_all[iv], p_all[iv], ya[iv], regla)
        dec = K.P.decidir(s_all[it], p_all[it], tau_t)
        r = K.P.metricas(dec, ya[it])
        res.append((r["frr"], r["far"]))
    R = np.array(res)
    ok = (R[:, 0] <= 0.10) & (R[:, 1] <= 0.05)
    return R[:, 0].mean(), R[:, 1].mean(), np.mean(R[:, 0] > 0.10), np.mean(R[:, 1] > 0.05), ok.mean()


if __name__ == "__main__":
    print("método      regla      | FRR medio  FAR medio | P(FRR>10%) P(FAR>5%) | P(cumple las dos)")
    import sys
    nombres = sys.argv[1].split(",")
    reglas = sys.argv[2].split(",")
    for nombre in nombres:
        for regla in reglas:
            fr, fa, pfr, pfa, pok = simular(nombre, regla)
            print(f"{nombre:10s}  {regla:12s} | {fr:6.3f}    {fa:6.3f}   | {pfr:6.3f}     {pfa:6.3f}   | {pok:6.3f}")
