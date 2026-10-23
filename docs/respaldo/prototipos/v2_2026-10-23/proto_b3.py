import numpy as np, collections
from proto_a import filas, leer, evaluar, FS
ruidos = collections.defaultdict(list)
for r in filas:
    if r["origen"] == "ruido" and r["particion"] == "train":
        ruidos[r["hablante"].split(":")[1]].append(leer(r))
limpios = []
for r in filas:
    if r["clase"] == "otro": continue
    y = leer(r); p, _ = evaluar(y, "crudo")
    limpios.append((r, y, p))
rng = np.random.default_rng(2)
print("cobertura = parte de la palabra limpia que queda; sobra = lo detectado fuera de ella (s)")
for nombre, zs in ruidos.items():
    print("==", nombre)
    for snr in (35, 30, 25, 20, 15, 10):
        cob, sob, inval = [], [], 0
        for r, y, p in limpios:
            z = zs[rng.integers(len(zs))]; off = rng.integers(0, len(z) - len(y) + 1)
            n = z[off:off+len(y)].astype(np.float64)
            i0, i1 = int(round(p[0]*FS)), int(round(p[1]*FS))
            g = np.sqrt(np.mean(y[i0:i1].astype(float)**2) / (np.mean(n**2) * 10**(snr/10)))
            q, _ = evaluar((y + g*n).astype(np.float32), "crudo")
            if q is None or not 0.3 <= q[1]-q[0] <= 2.5:
                inval += 1; continue
            inter = max(0, min(p[1], q[1]) - max(p[0], q[0]))
            cob.append(inter / (p[1]-p[0])); sob.append((q[1]-q[0]) - inter)
        cob = np.array(cob); sob = np.array(sob)
        print(" SNR %2d: no válidos %4.1f%% | cobertura p10 %.2f med %.2f | sobra p90 %.2f s | cob≥0,9 y sobra≤0,1 s: %4.1f%%" % (
            snr, 100*inval/len(limpios), np.percentile(cob,10), np.median(cob), np.percentile(sob,90),
            100*np.mean((cob>=0.9)&(sob<=0.1))*len(cob)/len(limpios)))
