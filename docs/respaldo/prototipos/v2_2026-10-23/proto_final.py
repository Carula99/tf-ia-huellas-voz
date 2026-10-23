"""Prototipo de V2 tal como queda en el contrato: medir validez de las copias, niveles y tiempos."""
import collections, time, zlib, math
import numpy as np
from scipy import signal
from proto_a import filas, leer, FS, cfg
from src.voz.segmentacion import detectar_palabras, envolvente_db

MARGEN_MS, NIVEL, ORDEN = 40.0, -26.0, 4
VEL, SNR, COPIAS = (0.92, 1.08), (10.0, 25.0), 2
C = int(round(3.0 * FS))


def palabra_principal(y):
    segs = detectar_palabras(y, FS, cfg)
    if not segs:
        return None, 0
    d = cfg["voz"]["dataset"]["deteccion"]
    niv = envolvente_db(y, FS, d["trama_ms"], d["avance_ms"])
    h = int(round(d["avance_ms"] * FS / 1000))
    t = np.arange(len(niv)) * h / FS
    mejor, bv = None, -np.inf
    for s in segs:
        m = (t >= s[0]) & (t < s[1])
        v = niv[m].max()
        if v > bv:
            mejor, bv = s, v
    return mejor, len(segs)


def preprocesar(y):
    p, n = palabra_principal(y)
    if p is None:
        return None, "sin_voz", None, n
    if not 0.3 <= p[1] - p[0] <= 2.5:
        return None, "duracion", p, n
    sos = signal.butter(ORDEN, [80, 4000], btype="bandpass", fs=FS, output="sos")
    x = signal.sosfiltfilt(sos, y.astype(np.float64))
    x = np.concatenate(([x[0]], x[1:] - 0.97 * x[:-1]))
    a, b = int(round(p[0] * FS)), int(round(p[1] * FS))
    M = int(round(MARGEN_MS * FS / 1000))
    clip = np.zeros(b - a + 2 * M)
    s0, s1 = max(0, a - M), min(len(x), b + M)
    clip[s0 - (a - M):s1 - (a - M)] = x[s0:s1]
    rms = np.sqrt(np.mean(clip[M:M + b - a] ** 2))
    return (clip * 10 ** (NIVEL / 20) / rms).astype(np.float32), "", p, n


def cambiar_velocidad(y, f):
    num, den = 100, int(round(100 * f))
    g = math.gcd(num, den)
    return signal.resample_poly(y.astype(np.float64), num // g, den // g).astype(np.float32)


def aumentar(y, p0, ruidos, rng):
    f = round(float(rng.uniform(*VEL)), 2)
    snr = float(rng.uniform(*SNR))
    k = int(rng.integers(len(ruidos)))
    u = float(rng.random())
    z = cambiar_velocidad(y, f)
    p = (p0[0] / f, p0[1] / f)
    if len(z) > C:
        c = int(round((p[0] + p[1]) / 2 * FS))
        i0 = min(max(0, c - C // 2), len(z) - C)
        z = z[i0:i0 + C]
        p = (p[0] - i0 / FS, p[1] - i0 / FS)
    r = ruidos[k]
    off = int(math.floor(u * (len(r) - len(z) + 1)))
    n = r[off:off + len(z)].astype(np.float64)
    a, b = int(round(p[0] * FS)), int(round(p[1] * FS))
    ps, pn = np.mean(z[a:b].astype(np.float64) ** 2), np.mean(n ** 2)
    zz = z + n * math.sqrt(ps / (pn * 10 ** (snr / 10)))
    return zz.astype(np.float32), dict(factor=f, snr=snr, ruido=k)


if __name__ == "__main__":
    t0 = time.time()
    ruidos = [leer(r) for r in filas if r["origen"] == "ruido" and r["particion"] == "train"]
    cnt = collections.Counter()
    picos, rel_dur = [], collections.defaultdict(list)
    for r in filas:
        y = leer(r)
        clip, mot, p, n = preprocesar(y)
        cnt[(r["particion"], r["clase"], "k0", clip is not None)] += 1
        if clip is not None:
            picos.append(20 * np.log10(np.abs(clip).max()))
            if r["clase"] != "otro":
                pass  # assert abs(round(p[0], 3) - float(r["palabra_inicio_s"])) < 1e-3 and abs(round(p[1], 3) - float(r["palabra_fin_s"])) < 1e-3, r["archivo"]
        if r["particion"] != "train" or r["origen"] == "ruido" or clip is None:
            continue
        for k in range(1, COPIAS + 1):
            rng = np.random.default_rng([2026, zlib.crc32(r["archivo"].encode("utf-8")), k])
            z, prm = aumentar(y, p, ruidos, rng)
            c2, m2, p2, _ = preprocesar(z)
            cnt[(r["particion"], r["clase"], "copia", c2 is not None)] += 1
            if c2 is not None:
                picos.append(20 * np.log10(np.abs(c2).max()))
                rel_dur[r["clase"]].append((p2[1] - p2[0]) / (p[1] - p[0]) * prm["factor"])
    for k in sorted(cnt):
        print(k, cnt[k])
    print("picos: p50 %.1f p99 %.1f max %.1f dBFS" % (np.median(picos), np.percentile(picos, 99), max(picos)))
    for c, v in rel_dur.items():
        print("duración copia/(original/f)", c, "p10 %.2f med %.2f p90 %.2f" % tuple(np.percentile(v, [10, 50, 90])))
    print("%.0f s" % (time.time() - t0))
