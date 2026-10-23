"""Prototipo V2 (A y B): recorte limpio sobre los 828 clips y robustez del detector con ruido."""
import csv, sys, collections, statistics as st
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal

sys.path.insert(0, "/home/claude/tf-ia-huellas-voz")
from src.config import cargar_config
from src.voz.segmentacion import detectar_palabras, envolvente_db

DS = Path(__file__).parent / "ds"
cfg = cargar_config("/home/claude/tf-ia-huellas-voz/config.yaml")
FS = 16000
filas = list(csv.DictReader(open(DS / "metadatos.csv", encoding="utf-8")))
SOS = signal.butter(4, [80, 4000], btype="bandpass", fs=FS, output="sos")


def leer(r):
    y, fs = sf.read(DS / r["archivo"], dtype="float32")
    assert fs == FS
    return y


def pasabanda(y):
    return signal.sosfiltfilt(SOS, y.astype(np.float64)).astype(np.float32)


def principal(y, segs):
    """Segmento con la trama más fuerte."""
    d = cfg["voz"]["dataset"]["deteccion"]
    niv = envolvente_db(y, FS, d["trama_ms"], d["avance_ms"])
    h = int(d["avance_ms"] * FS / 1000)
    best, bv = None, -1e9
    for s in segs:
        k0, k1 = int(round(s[0] * FS / h)), int(round(s[1] * FS / h))
        v = niv[k0:max(k0 + 1, k1 - 1)].max()
        if v > bv:
            best, bv = s, v
    return best


def iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0


def evaluar(y, modo):
    z = pasabanda(y) if modo == "filtrado" else y
    segs = detectar_palabras(z, FS, cfg)
    if not segs:
        return None, 0
    p = principal(z, segs)
    return p, len(segs)


if __name__ == "__main__":
    # A. limpio
    res = {}
    for modo in ("crudo", "filtrado"):
        cnt = collections.Counter()
        ious, dur = [], collections.defaultdict(list)
        for r in filas:
            y = leer(r)
            p, n = evaluar(y, modo)
            ok = p is not None and 0.3 <= p[1] - p[0] <= 2.5
            cnt[(r["particion"], r["origen"], r["clase"] if r["clase"] != "otro" else "otro", ok)] += 1
            if r["clase"] != "otro" and p is not None and r["palabra_inicio_s"]:
                ious.append(iou(p, (float(r["palabra_inicio_s"]), float(r["palabra_fin_s"]))))
            if ok:
                dur[r["clase"]].append(p[1] - p[0])
        print("==", modo)
        inval = {k: v for k, v in cnt.items() if not k[3]}
        print(" no válidos:", sorted(inval.items()))
        print(" IoU comandos vs V1: min %.3f p1 %.3f mediana %.3f" % (min(ious), np.percentile(ious, 1), np.median(ious)))
        for c, v in dur.items():
            print("  dur", c, "p10 %.2f med %.2f p90 %.2f" % tuple(np.percentile(v, [10, 50, 90])))
