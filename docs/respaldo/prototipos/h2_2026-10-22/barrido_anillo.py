import pickle, sys, math
sys.path.insert(0, "/home/claude/h2/repo"); sys.path.insert(0, "/home/claude/h2/proto")
import numpy as np, cv2
import proto_h2 as H
D = pickle.load(open("/home/claude/h2/proto/cache.pkl", "rb"))
bases, filas = D["bases"], D["filas"]
anillos = [(40,160),(30,120),(30,100),(20,80),(40,240),(60,240),(80,240),(100,240),(0,60)]
def prep(img):
    g = cv2.medianBlur(H.a_gris(img), 3)
    m, _ = H.segmentar(g)
    th, coh, u, v = H.campo(g, m)
    ref, met, _, _ = H.punto_referencia(u, v, th, m)
    return m, coh, u, v, ref
def angs(m, coh, u, v, ref, refv=None):
    r = refv if refv is not None else ref
    out = []
    for a in anillos:
        p = dict(H.P); p["anillo"] = a
        out.append(math.degrees(H.angulo_anillo(u, v, coh, m, r, p)[0]))
    return np.array(out)
tipos = sys.argv[1].split(",")
base_ang = {}
for k, b in bases.items():
    if k[0] in tipos:
        base_ang[k] = angs(*prep(b.imagen))
errs, cortes, tps = [], [], []
for e, v in filas:
    if e["tipo"] not in tipos: continue
    a = angs(*prep(v.imagen))
    err = (a - (base_ang[(e["tipo"], e["semilla"])] - v.parametros.rot_grados) + 90) % 180 - 90
    errs.append(err); cortes.append(v.parametros.corte_fraccion > 0); tps.append(e["tipo"])
errs = np.abs(np.array(errs)); cortes = np.array(cortes); tps = np.array(tps)
for t in tipos:
    print(t)
    for i, a in enumerate(anillos):
        s = tps == t
        e = errs[s, i]; c = cortes[s]
        print(f"  anillo {a}: <=5 {np.mean(e<=5):.3f} | sin corte p95 {np.percentile(e[~c],95):5.1f} | con corte p50 {np.median(e[c]):5.1f} p95 {np.percentile(e[c],95):5.1f} max {e[c].max():5.1f}")
