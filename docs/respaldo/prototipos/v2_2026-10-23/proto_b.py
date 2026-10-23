"""Prototipo V2 (B): hasta qué SNR el detector sigue encontrando la palabra entera."""
import sys, collections
import numpy as np
from proto_a import filas, leer, evaluar, iou, FS

ruidos = [leer(r) for r in filas if r["origen"] == "ruido" and r["particion"] == "train"]
cmd = [r for r in filas if r["clase"] != "otro" or (r["origen"] in ("notebook", "tts") and r["detalle"] not in ("no_verbal",))]
rng = np.random.default_rng(1)
limpios = []
for r in cmd:
    y = leer(r)
    p, _ = evaluar(y, "crudo")
    if p is None or not 0.3 <= p[1] - p[0] <= 2.5:
        continue
    limpios.append((r, y, p))
print("clips:", len(limpios))

for snr in (25, 20, 15, 12, 10, 8, 6, 5):
    st = collections.defaultdict(lambda: [0, 0, 0, []])
    for r, y, p in limpios:
        z = ruidos[rng.integers(len(ruidos))]
        off = rng.integers(0, len(z) - len(y) + 1)
        n = z[off:off + len(y)].astype(np.float64)
        i0, i1 = int(round(p[0] * FS)), int(round(p[1] * FS))
        ps = np.mean(y[i0:i1].astype(np.float64) ** 2)
        pn = np.mean(n ** 2)
        g = np.sqrt(ps / (pn * 10 ** (snr / 10)))
        yn = (y + g * n).astype(np.float32)
        q, _ = evaluar(yn, "crudo")
        key = r["clase"] if r["clase"] != "otro" else "otro"
        s = st[key]
        s[0] += 1
        if q is not None and 0.3 <= q[1] - q[0] <= 2.5:
            s[1] += 1
            j = iou(p, q)
            s[3].append(j)
            if j >= 0.8:
                s[2] += 1
    out = []
    for k, (n, v, b, js) in sorted(st.items()):
        out.append("%s válidos %.1f%% IoU≥0,8 %.1f%% IoU p10 %.2f" % (k, 100 * v / n, 100 * b / n, np.percentile(js, 10) if js else 0))
    print("SNR %2d dB | " % snr + " | ".join(out))
