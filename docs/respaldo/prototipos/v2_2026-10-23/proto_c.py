"""Prototipo V2 (C): aumento + características rápidas + K-NN para comparar variantes del aumento.

Las características y el K-NN son un borrador de V3/V4, solo para decidir el aumento de V2.
"""
import sys, collections, zlib, time
import numpy as np, librosa
from scipy import signal
from proto_a import filas, leer, principal, FS, cfg, SOS
from src.voz.segmentacion import detectar_palabras

M = int(round(0.040 * FS))          # margen de 40 ms
C = 3 * FS                          # 3 s
NIVEL = -20.0


# ---------------- V2 ----------------
def detectar(y):
    segs = detectar_palabras(y, FS, cfg)
    if not segs:
        return None, 0
    return principal(y, segs), len(segs)


def cortar(z, a, b):
    """z[a−M : b+M] con ceros donde se sale del audio."""
    out = np.zeros(b - a + 2 * M, dtype=np.float64)
    i0, i1 = a - M, b + M
    s0, s1 = max(0, i0), min(len(z), i1)
    out[s0 - i0:s1 - i0] = z[s0:s1]
    return out


def acondicionar(y):
    z = signal.sosfiltfilt(SOS, y.astype(np.float64))
    z = np.concatenate(([z[0]], z[1:] - 0.97 * z[:-1]))
    return z


def normalizar(z):
    rms = np.sqrt(np.mean(z ** 2))
    return (z * (10 ** (NIVEL / 20) / max(rms, 1e-9))).astype(np.float32)


def preprocesar(y, limites=None):
    """Devuelve (clip, valido, dur). Si limites viene dado, recorta ahí (no detecta)."""
    if limites is None:
        p, _ = detectar(y)
        if p is None or not 0.3 <= p[1] - p[0] <= 2.5:
            return None, False, None
    else:
        p = limites
    a, b = int(round(p[0] * FS)), int(round(p[1] * FS))
    z = acondicionar(y)
    return normalizar(cortar(z, a, b)), True, p[1] - p[0]


# ---------------- aumento ----------------
def velocidad(y, f):
    down = int(round(100 * f))
    g = np.gcd(100, down)
    return signal.resample_poly(y.astype(np.float64), 100 // g, down // g)


def ecualizar(y, gan_db, centros=(150, 500, 1500, 3500)):
    freq = [0.0] + [c / (FS / 2) for c in centros] + [1.0]
    g = 10 ** (np.array([gan_db[0]] + list(gan_db) + [gan_db[-1]]) / 20)
    h = signal.firwin2(257, freq, g)
    return np.convolve(y, h, mode="same")


def sala(rng, rt60, drr):
    n = int(round(rt60 * FS))
    t = np.arange(n) / FS
    cola = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)
    cola[: int(0.003 * FS)] = 0.0
    e = np.sum(cola ** 2)
    cola *= np.sqrt(10 ** (-drr / 10) / e)
    cola[0] = 1.0
    return cola


def recortar_3s(y, p):
    if len(y) <= C:
        return y, p
    c = int(round((p[0] + p[1]) / 2 * FS))
    i0 = min(max(0, c - C // 2), len(y) - C)
    return y[i0:i0 + C], (p[0] - i0 / FS, p[1] - i0 / FS)


def sumar_ruido(y, p, n, snr):
    a, b = int(round(p[0] * FS)), int(round(p[1] * FS))
    ps = np.mean(y[a:b] ** 2)
    pn = np.mean(n ** 2)
    return y + n * np.sqrt(ps / (max(pn, 1e-20) * 10 ** (snr / 10)))


def copia(y, p0, rng, ruidos, modo, snr_rango=(10, 25), con_sala=True, con_eq=None, con_rev=None):
    con_eq = con_sala if con_eq is None else con_eq
    con_rev = con_sala if con_rev is None else con_rev
    f = rng.uniform(0.92, 1.08)
    f = round(f, 2)
    z = velocidad(y, f)
    p = (p0[0] / f, p0[1] / f)
    z, p = recortar_3s(z, p)
    gq = rng.uniform(-6, 6, 4); rt = rng.uniform(0.15, 0.4); dr = rng.uniform(8, 15)
    if con_eq:
        z = ecualizar(z, gq)
    if con_rev:
        z = signal.fftconvolve(z, sala(rng, rt, dr))[: len(z)]
    # palabra en la copia limpia (para el SNR y, en modo "despues", para el recorte)
    q, _ = detectar(z.astype(np.float32))
    if q is None:
        return None, False, None
    ruido = ruidos[rng.integers(len(ruidos))]
    off = rng.integers(0, len(ruido) - len(z) + 1)
    zn = sumar_ruido(z, q, ruido[off:off + len(z)].astype(np.float64), rng.uniform(*snr_rango))
    if modo == "despues":
        if not 0.3 <= q[1] - q[0] <= 2.5:
            return None, False, None
        return preprocesar(zn, limites=q)
    return preprocesar(zn.astype(np.float32))


# ---------------- características (borrador de V3) ----------------
def tramos(m, k=5):
    n = m.shape[1]
    out = []
    for i in range(k):
        a = int(np.floor(i * n / k)); b = max(a + 1, int(np.floor((i + 1) * n / k)))
        s = m[:, a:b]
        out.append(np.concatenate([s.mean(1), s.std(1)]))
    return np.concatenate(out)


def caracteristicas(clip, dur):
    S = librosa.feature.mfcc(y=clip, sr=FS, n_mfcc=13, n_fft=512, win_length=400, hop_length=160,
                             n_mels=40, fmin=80, fmax=4000, center=False)
    w = S[:, M // 160: S.shape[1] - M // 160] if S.shape[1] > 2 * (M // 160) + 5 else S
    w = w - w.mean(1, keepdims=True)          # CMN sobre la palabra
    d = librosa.feature.delta(w, width=5, mode="nearest") if w.shape[1] >= 5 else np.zeros_like(w)
    fr = librosa.util.frame(clip, frame_length=400, hop_length=160)
    fr = fr[:, M // 160: fr.shape[1] - M // 160] if fr.shape[1] > 2 * (M // 160) + 5 else fr
    rms = 20 * np.log10(np.maximum(np.sqrt((fr ** 2).mean(0)), 1e-6))[None, :]
    zcr = (np.abs(np.diff(np.sign(fr), axis=0)) > 0).mean(0)[None, :]
    n = min(w.shape[1], rms.shape[1])
    return [tramos(w[:, :n]), tramos(d[:, :n]), tramos(rms[:, :n]), tramos(zcr[:, :n]), np.array([dur])]


class Modelo:
    def __init__(self, bloques, y, h):
        X = [np.stack(b) for b in zip(*bloques)]
        self.mu = [x.mean(0) for x in X]
        self.sd = [np.where(x.std(0) > 1e-9, x.std(0), 1.0) for x in X]
        self.X = self._t(X)
        self.y = np.array(y)

    def _t(self, X):
        return np.hstack([(x - m) / s / np.sqrt(x.shape[1]) for x, m, s in zip(X, self.mu, self.sd)])

    def predecir(self, bloques, k=5):
        Q = self._t([np.stack(b) for b in zip(*bloques)])
        d = np.sqrt(((Q[:, None, :] - self.X[None]) ** 2).sum(-1))
        idx = np.argsort(d, 1)[:, :k]
        out = []
        for i in range(len(Q)):
            v = collections.Counter()
            for j in idx[i]:
                v[self.y[j]] += 1 / (d[i, j] + 1e-9)
            c, w = v.most_common(1)[0]
            out.append(c if w / sum(v.values()) >= 0.6 else "otro")
        return out


SEMILLA = [2026]
def rng_de(nombre, k):
    return np.random.default_rng([SEMILLA[0], zlib.crc32(nombre.encode()), k])


if __name__ == "__main__":
    t0 = time.time()
    tr = [r for r in filas if r["particion"] == "train"]
    va = [r for r in filas if r["particion"] == "val"]
    ruidos_tr = [leer(r) for r in tr if r["origen"] == "ruido"]
    ruidos_va = [leer(r) for r in va if r["origen"] == "ruido"]
    audios = {r["archivo"]: leer(r) for r in tr + va}

    base = []  # (fila, y, p, feats)
    for r in tr:
        y = audios[r["archivo"]]
        clip, ok, dur = preprocesar(y)
        if ok:
            base.append((r, y, detectar(y)[0], caracteristicas(clip, dur)))
    print("train válidos limpios:", len(base), "de", len(tr), "%.0fs" % (time.time() - t0))

    variantes = {
        "sin aumento": None,
        "ruido después del recorte 10-25": dict(modo="despues", snr_rango=(10, 25)),
        "ruido antes del recorte 10-25": dict(modo="antes", snr_rango=(10, 25)),
        "ruido antes del recorte 20-35": dict(modo="antes", snr_rango=(20, 35)),
        "ruido después 10-25, sin sala": dict(modo="despues", snr_rango=(10, 25), con_sala=False),
    }
    # conjuntos de evaluación: val limpio y val con ruido de val sumado antes (como en vivo)
    evals = {}
    rng = np.random.default_rng(7)
    for nombre, snr in (("val limpio", None), ("val + ruido 30 dB", 30), ("val + ruido 20 dB", 20), ("val + ruido 15 dB", 15)):
        feats, ys, inval = [], [], []
        for r in va:
            y = audios[r["archivo"]].astype(np.float64)
            if snr is not None:
                p, _ = detectar(y.astype(np.float32))
                if p is not None:
                    z = ruidos_va[rng.integers(len(ruidos_va))]
                    off = rng.integers(0, len(z) - len(y) + 1)
                    y = sumar_ruido(y, p, z[off:off + len(y)].astype(np.float64), snr)
            clip, ok, dur = preprocesar(y.astype(np.float32))
            if ok:
                feats.append(caracteristicas(clip, dur)); ys.append(r["clase"]); inval.append(False)
            else:
                feats.append(None); ys.append(r["clase"]); inval.append(True)
        evals[nombre] = (feats, ys)

    for vn, kw in variantes.items():
        bl, ylab, hab = [], [], []
        nval = 0
        for r, y, p, f in base:
            bl.append(f); ylab.append(r["clase"]); hab.append(r["hablante"])
            if kw is None or r["origen"] == "ruido":
                continue
            for k in (1, 2):
                clip, ok, dur = copia(y, p, rng_de(r["archivo"], k), ruidos_tr, **kw)
                if ok:
                    bl.append(caracteristicas(clip, dur)); ylab.append(r["clase"]); hab.append(r["hablante"]); nval += 1
        mod = Modelo(bl, ylab, hab)
        linea = []
        for en, (feats, ys) in evals.items():
            pred = ["otro" if f is None else None for f in feats]
            idx = [i for i, f in enumerate(feats) if f is not None]
            pp = mod.predecir([feats[i] for i in idx])
            for i, c in zip(idx, pp):
                pred[i] = c
            cmd = [i for i, c in enumerate(ys) if c != "otro"]
            otr = [i for i, c in enumerate(ys) if c == "otro"]
            acc = np.mean([pred[i] == ys[i] for i in cmd])
            fa = np.mean([pred[i] != "otro" for i in otr])
            linea.append("%s: acierto %.1f%% otro aceptado %.1f%%" % (en, 100 * acc, 100 * fa))
        print("%-34s copias válidas %4d | " % (vn, nval) + " | ".join(linea), "%.0fs" % (time.time() - t0), flush=True)
