"""Prototipo V2 (D): misma comparación con validación cruzada por hablante dentro de train (más casos)."""
import collections, time
import numpy as np
import proto_c as P
from proto_a import filas, leer

t0 = time.time()
tr = [r for r in filas if r["particion"] == "train"]
va = [r for r in filas if r["particion"] == "val"]
ruidos_tr = [leer(r) for r in tr if r["origen"] == "ruido"]
ruidos_va = [leer(r) for r in va if r["origen"] == "ruido"]
audios = {r["archivo"]: leer(r) for r in tr}

base = []
for r in tr:
    y = audios[r["archivo"]]
    clip, ok, dur = P.preprocesar(y)
    base.append((r, y, P.detectar(y)[0] if ok else None, P.caracteristicas(clip, dur) if ok else None))

# evaluación: clips de la notebook de cada hablante real, limpios y con ruido de val antes del recorte
rng = np.random.default_rng(11)
evals = {}
for nombre, snr in (("limpio", None), ("+ruido 25 dB", 25), ("+ruido 15 dB", 15)):
    e = []
    for r, y, p, f in base:
        if r["origen"] != "notebook":
            continue
        if snr is None:
            e.append((r, f))
            continue
        q, _ = P.detectar(y)
        yy = y.astype(np.float64)
        if q is not None:
            z = ruidos_va[rng.integers(len(ruidos_va))]
            off = rng.integers(0, len(z) - len(y) + 1)
            yy = P.sumar_ruido(yy, q, z[off:off + len(y)].astype(np.float64), snr)
        clip, ok, dur = P.preprocesar(yy.astype(np.float32))
        e.append((r, P.caracteristicas(clip, dur) if ok else None))
    evals[nombre] = e

hablantes = sorted({r["hablante"] for r in tr if r["origen"] == "notebook"})
pliegues = [hablantes[i::6] for i in range(6)]
print("hablantes de la notebook en train:", len(hablantes), "pliegues:", [len(p) for p in pliegues], "%.0fs" % (time.time() - t0))

variantes = {
    "sin aumento": None,
    "después 10-25": dict(modo="despues", snr_rango=(10, 25)),
    "antes 10-25": dict(modo="antes", snr_rango=(10, 25)),
    "antes 15-30": dict(modo="antes", snr_rango=(15, 30)),
    "antes 20-35": dict(modo="antes", snr_rango=(20, 35)),
    "antes 10-25 sin sala": dict(modo="antes", snr_rango=(10, 25), con_sala=False),
}
for vn, kw in variantes.items():
    ent = []  # (hablante, clase, feats, es_copia)
    for r, y, p, f in base:
        if f is None:
            continue
        ent.append((r["hablante"], r["clase"], f))
        if kw is None or r["origen"] == "ruido":
            continue
        for k in (1, 2):
            clip, ok, dur = P.copia(y, p, P.rng_de(r["archivo"], k), ruidos_tr, **kw)
            if ok:
                ent.append((r["hablante"], r["clase"], P.caracteristicas(clip, dur)))
    res = collections.defaultdict(lambda: [0, 0, 0, 0])
    for pl in pliegues:
        fuera = set(pl)
        tr_e = [e for e in ent if e[0] not in fuera]
        mod = P.Modelo([e[2] for e in tr_e], [e[1] for e in tr_e], None)
        for en, e in evals.items():
            casos = [(r, f) for r, f in e if r["hablante"] in fuera]
            idx = [i for i, (r, f) in enumerate(casos) if f is not None]
            pred = ["otro"] * len(casos)
            for i, c in zip(idx, mod.predecir([casos[i][1] for i in idx])):
                pred[i] = c
            for (r, f), c in zip(casos, pred):
                s = res[en]
                if r["clase"] != "otro":
                    s[0] += 1; s[1] += c == r["clase"]
                else:
                    s[2] += 1; s[3] += c != "otro"
    print("%-22s %4d vectores | " % (vn, len(ent)) + " | ".join(
        "%s: acierto %.1f%% (%d) otro aceptado %.1f%% (%d)" % (en, 100 * s[1] / s[0], s[0], 100 * s[3] / s[2], s[2]) for en, s in res.items()),
        "%.0fs" % (time.time() - t0), flush=True)
