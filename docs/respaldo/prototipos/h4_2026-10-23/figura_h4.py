"""Figura del prototipo de H4: curva FAR-FRR en val con la distancia normalizada, distancias de
cada grupo contra τ*, y probabilidad de cumplir las metas en 2000 pares val/test simulados."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import comparar as K, proto_h4 as P

COL = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}; GRIS = "#7f7f7f"; TINTA = "#333333"
f = K.puntaje_factory("norm_rms")
X, y, va, te, tr = K.X, K.ident, K.va, K.te, K.tr
s_v, p_v = f(X[va])
tau = K.tau_regla(s_v, p_v, y[va], "fmed1"); tau_bajo = K.tau_regla(s_v, p_v, y[va], "eer_bajo")
a_v = y[va] > 0; lo, hi = s_v[a_v].max(), s_v[~a_v].min()

from matplotlib.ticker import FuncFormatter
coma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
fig = plt.figure(figsize=(17, 5.4), dpi=150)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 0.9], wspace=0.42)

# (a) FAR y FRR en val
ax = fig.add_subplot(gs[0])
t = np.linspace(0.6, 3.2, 2000)
far = [np.mean(s_v[~a_v] <= x) for x in t]
frr = [np.mean(~((s_v[a_v] <= x) & (p_v[a_v] == y[va][a_v]))) for x in t]
ax.axvspan(lo, hi, color="#e8e8e8", lw=0)
ax.plot(t, np.array(frr) * 100, color="#c0392b", lw=2, label="FRR (autorizadas rechazadas)")
ax.plot(t, np.array(far) * 100, color="#34495e", lw=2, label="FAR (intrusos aceptados)")
ax.axvline(tau, color=TINTA, lw=1.5); ax.axvline(tau_bajo, color=TINTA, lw=1, ls=":")
ax.text(tau + 0.04, 62, f"τ* = {tau:.3f}\n(mitad del intervalo)".replace(".", ","), fontsize=8.5, color=TINTA)
ax.text(tau + 0.04, 44, f"punteada: borde de abajo\n{tau_bajo:.3f} (adelanto)".replace(".", ","), fontsize=8, color=TINTA)
ax.text((lo + hi) / 2, 94, "sin errores en val", fontsize=8, ha="center", color="#555555")
ax.set_xlabel("distancia normalizada  d_k / r_k"); ax.set_ylabel("%"); ax.set_ylim(-2, 102); ax.set_xlim(0.6, 3.2)
ax.set_title("(a) Validación: FAR y FRR según τ", fontsize=10.5, loc="left")
ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.88), frameon=False)
ax.xaxis.set_major_formatter(coma)

# (b) distancias por grupo
ax = fig.add_subplot(gs[1])
s_all, p_all = f(X); sa, pa = f(K.A["X"]); ya = K.A["identidad"]
grupos = [("train · autorizadas", s_all[tr], y[tr]), ("val · autorizadas", s_all[va & (y > 0)], y[va & (y > 0)]),
          ("val · intrusos", s_all[va & (y == 0)], y[va & (y == 0)]), ("test · autorizadas", s_all[te & (y > 0)], y[te & (y > 0)]),
          ("test · intrusos", s_all[te & (y == 0)], y[te & (y == 0)]), ("ampliada · autorizadas (300)", sa[ya > 0], ya[ya > 0]),
          ("ampliada · intrusos (300)", sa[ya == 0], ya[ya == 0])]
rng = np.random.default_rng(1)
for i, (nombre, s, yy) in enumerate(grupos[::-1]):
    jit = rng.uniform(-0.28, 0.28, len(s))
    c = [COL[k] if k > 0 else GRIS for k in yy]
    ax.scatter(np.minimum(s, 3.95), i + jit, s=11, c=c, alpha=0.75 if "ampliada" in nombre else 0.9, lw=0)
ax.axvline(tau, color=TINTA, lw=1.5); ax.axvline(tau_bajo, color=TINTA, lw=1, ls=":")
ax.set_yticks(range(len(grupos))); ax.set_yticklabels([g[0] for g in grupos[::-1]], fontsize=8.5)
ax.set_xlim(0, 4.0); ax.set_xlabel("distancia normalizada al centroide más cercano (los > 4 quedan en 4)")
for k, n in ((1, "identidad 1"), (2, "identidad 2"), (3, "identidad 3")):
    ax.scatter([], [], c=COL[k], s=18, label=n)
ax.scatter([], [], c=GRIS, s=18, label="intrusos")
ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.95, borderaxespad=0.2)
ax.xaxis.set_major_formatter(coma)
ax.set_title("(b) Distancias contra τ* (línea llena) y el borde de abajo (punteada)", fontsize=10.5, loc="left")
ax.grid(axis="x", color="#eeeeee", lw=0.8); ax.set_axisbelow(True)

# (c) simulación
ax = fig.add_subplot(gs[2])
op = [("A  euclídea\nborde de abajo (adelanto)", 0.858, 0.116, 0.026), ("B  euclídea\nmitad del intervalo EER", 0.946, 0.024, 0.030),
      ("C  normalizada\nmitad del intervalo EER", 0.971, 0.004, 0.025), ("D  normalizada\nFAR val ≤ 1 %, mitad", 0.996, 0.004, 0.001)]
yy = np.arange(len(op))[::-1]
ax.barh(yy, [o[1] * 100 for o in op], color=["#bbbbbb", "#bbbbbb", "#6a9fcf", "#1f77b4"], height=0.42)
for yi, o in zip(yy, op):
    ax.text(o[1] * 100 - 1, yi, f"{o[1]*100:.1f} %".replace(".", ","), va="center", ha="right", fontsize=9, color="white", weight="bold")
    ax.text(50.5, yi + 0.36, o[0].replace("\n", " · "), fontsize=8.5, color=TINTA, va="center")
    ax.text(50.5, yi - 0.34, f"P(FRR > 10 %) {o[2]*100:.1f} % · P(FAR > 5 %) {o[3]*100:.1f} %".replace(".", ","), fontsize=7.5, color="#555555", va="center")
ax.set_yticks([]); ax.set_xlim(50, 100); ax.set_ylim(-0.7, len(op) - 0.3)
ax.set_xlabel("% de pares que cumplen FRR ≤ 10 % y FAR ≤ 5 %")
ax.set_title("(c) 2000 pares val/test simulados", fontsize=10.5, loc="left")
for sp in ("top", "right"):
    for a in fig.axes: a.spines[sp].set_visible(False)
fig.savefig("h4_proto.png", bbox_inches="tight", facecolor="white")
print("τ* =", tau, "borde de abajo", tau_bajo)
