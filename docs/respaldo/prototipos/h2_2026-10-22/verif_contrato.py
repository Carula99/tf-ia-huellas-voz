"""Chequeo de los criterios del contrato de H2, leídos al pie de la letra, contra el prototipo."""
import pickle, sys, math, time, collections
sys.path.insert(0, "/home/claude/h2/repo"); sys.path.insert(0, "/home/claude/h2/proto")
import numpy as np, cv2
from scipy import ndimage as ndi
import proto_h2c as B
from src.huellas.variantes import ParametrosVariante, aplicar_variante
D = pickle.load(open("/home/claude/h2/proto/cache.pkl", "rb"))
bases, filas = D["bases"], D["filas"]
ALTO = ANCHO = 250
def pre(img): return B.preprocesar(img)
def err_ang(ang, rot): return ((math.degrees(ang) + rot + 90) % 180) - 90
def mreal(var, M, shape_in):
    m = cv2.warpAffine(var.mascara.astype(np.uint8), M, (ANCHO, ALTO), flags=cv2.INTER_NEAREST) > 0
    d = cv2.warpAffine(np.ones(shape_in, np.uint8), M, (ANCHO, ALTO), flags=cv2.INTER_NEAREST) > 0
    return m, d
def verdad(var, base, r):
    Mi = cv2.invertAffineTransform(r["matriz"])
    yw, xw = np.mgrid[0:ALTO, 0:ANCHO].astype(np.float64)
    px = Mi[0,0]*xw + Mi[0,1]*yw + Mi[0,2]; py = Mi[1,0]*xw + Mi[1,1]*yw + Mi[1,2]
    qx = ndi.map_coordinates(var.mapa_x.astype(np.float64), [py, px], order=1)
    qy = ndi.map_coordinates(var.mapa_y.astype(np.float64), [py, px], order=1)
    c = ndi.map_coordinates(np.cos(2*base.orientacion), [qy, qx], order=1)
    s = ndi.map_coordinates(np.sin(2*base.orientacion), [qy, qx], order=1)
    tv = 0.5*np.arctan2(s, c) - math.radians(var.parametros.rot_grados) - r["angulo"]
    cr = ndi.map_coordinates(base.imagen.astype(np.float64), [qy, qx], order=1) < 128
    return tv, cr
def periodo(img):
    h, w = img.shape; f0, c0 = (h-192)//2, (w-192)//2
    x = img[f0:f0+192, c0:c0+192].astype(float); x -= x.mean(); x *= np.outer(np.hanning(192), np.hanning(192))
    P_ = np.abs(np.fft.fft2(x))**2; f = np.fft.fftfreq(192); F = np.hypot(f[:,None], f[None,:])
    b = (F > 1/20) & (F < 1/5); return 1/((F[b]*P_[b]).sum()/P_[b].sum())
def afin(p, x, y, alto=400, ancho=320):
    cx, cy = (ancho-1)/2, (alto-1)/2; a = math.radians(p.rot_grados); s = p.escala
    return (cx + s*(math.cos(a)*(x-cx) + math.sin(a)*(y-cy)) + p.tx_px, cy + s*(-math.sin(a)*(x-cx) + math.cos(a)*(y-cy)) + p.ty_px)

def evaluar(conjunto, etiqueta):
    peor = collections.defaultdict(lambda: -np.inf); mejor = collections.defaultdict(lambda: np.inf)
    cuenta = collections.defaultdict(list)
    t_total = 0.0
    for e, v in conjunto:
        b = bases[(e["tipo"], e["semilla"])]
        t0 = time.perf_counter(); r = pre(v.imagen); t_total += time.perf_counter() - t0
        tp = e["tipo"]; M = r["matriz"]
        # 5.3
        R2 = M[:, :2]; a = ((r["angulo"] + math.pi / 2) % math.pi) - math.pi / 2
        peor["matriz_rot"] = max(peor["matriz_rot"], np.abs(R2 - [[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]).max())
        peor["matriz_centro"] = max(peor["matriz_centro"], np.abs(M @ [*r["punto_referencia"], 1] - 124.5).max())
        ok_centro = r["mascara"][124:126, 124:126].all()
        cuenta["centro_en_mascara"].append(ok_centro)
        mr, dentro = mreal(v, M, v.imagen.shape)
        A = r["mascara"] & dentro
        iou = (A & mr).sum() / (A | mr).sum(); mejor["iou"] = min(mejor["iou"], iou)
        # 5.4
        cuenta[f"metodo_{tp}"].append(r["metodo"] == {"presilla": "nucleo", "arco_tienda": "nucleo", "verticilo": "nucleos", "arco": "curvatura"}[tp])
        if tp != "arco":
            d = math.dist(r["punto_referencia"], v.punto_referencia)
            cuenta[f"ref8_{tp}"].append(d <= 8); peor[f"ref_{tp}"] = max(peor[f"ref_{tp}"], d)
            ok = all(r["nucleos"] and min(math.dist(c, q) for q in r["nucleos"]) <= 8 for c in v.singularidades.nucleos)
            cuenta[f"nucleos8_{tp}"].append(ok)
        else:
            cuenta["arco_ok"].append(r["metodo"] == "curvatura" and len(r["nucleos"]) == 0)
        # 5.5
        ea = abs(err_ang(r["angulo"], v.parametros.rot_grados))
        cuenta[f"ang5_{tp}"].append(ea <= 5); peor[f"ang_{tp}"] = max(peor[f"ang_{tp}"], ea)
        if v.parametros.corte_fraccion > 0:
            cuenta["ang5_corte"].append(ea <= 5)
        # 5.6
        zona = ndi.binary_erosion(r["mascara"] & mr, structure=np.ones((9, 9), bool), border_value=0)
        tv, cr = verdad(v, b, r)
        mejor["campo"] = min(mejor["campo"], np.cos(2*(r["orientacion"] - tv))[zona].mean())
        mejor["crestas"] = min(mejor["crestas"], (r["binaria"] == cr)[zona].mean())
        E = r["esqueleto"]
        cuenta["esq_en_binaria"].append(not (E & ~r["binaria"]).any())
        peor["bloques22"] = max(peor["bloques22"], int((E[:-1,:-1] & E[1:,:-1] & E[:-1,1:] & E[1:,1:]).sum()))
        dens = E.sum() / r["mascara"].sum(); peor["dens"] = max(peor["dens"], dens); mejor["dens"] = min(mejor["dens"], dens)
        Rz = r["realzada"].astype(float)
        cuenta["fuera255"].append(np.all(r["realzada"][~r["mascara"]] == 255))
        mejor["contraste"] = min(mejor["contraste"], Rz[r["mascara"] & ~r["binaria"]].mean() - Rz[r["binaria"]].mean())
        peor["periodo"] = max(peor["periodo"], abs(periodo(r["realzada"]) - b.periodo_px * v.parametros.escala))
        # formato
        cuenta["formato"].append(r["realzada"].dtype == np.uint8 and r["orientacion"].min() >= 0 and r["orientacion"].max() < math.pi and r["esqueleto"].any())
    print(f"== {etiqueta}: {len(conjunto)} variantes, preprocesar {t_total:.1f} s")
    for k, l in sorted(cuenta.items()):
        print(f"  {k:22s} {np.mean(l):.3f} ({len(l) - sum(l)} fallan de {len(l)})")
    for k in sorted(peor): print(f"  peor {k:18s} {peor[k]:.4g}")
    for k in sorted(mejor): print(f"  mínimo {k:16s} {mejor[k]:.4g}")

muestra = [(e, v) for e, v in filas if e["identidad"] in (1, 2, 3) or 1 <= e["intruso"] <= 12]
evaluar(muestra, "muestra")
evaluar(filas, "dataset completo")
# arco en aceptación
rb = {}; dd = []
for e, v in filas:
    if e["tipo"] != "arco": continue
    k = (e["tipo"], e["semilla"])
    if k not in rb: rb[k] = pre(bases[k].imagen)
    r = pre(v.imagen)
    dd.append(math.dist(afin(v.parametros, *rb[k]["punto_referencia"]), r["punto_referencia"]))
print(f"arco ≤ 30 px: {np.mean(np.array(dd) <= 30):.3f}")
# 5.7
for k in [("presilla",101),("verticilo",202),("arco_tienda",303)]:
    b = bases[k]; r0 = pre(b.imagen)
    for a in (0.0, 90.0, 180.0, 270.0):
        v = aplicar_variante(b, ParametrosVariante(12.0, 1.0, 10.0, -8.0, 0.0, 0.0, 0.10, a, 0.0, 0.0), np.random.default_rng(0))
        r = pre(v.imagen)
        yy, xx = np.mgrid[0:250, 0:250]; z = (np.hypot(xx-124.5, yy-124.5) <= 80) & r["mascara"] & r0["mascara"]
        c = np.cos(2*(r["orientacion"] - r0["orientacion"]))[z].mean()
        print(f"5.7 {k[0]:12s} {a:5.0f}: cos {c:.4f} err ang {abs(err_ang(r['angulo'], 12.0)):.2f} ref {math.dist(r['punto_referencia'], v.punto_referencia):.2f}")
