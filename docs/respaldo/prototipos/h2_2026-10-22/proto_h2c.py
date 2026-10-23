"""Prototipo 3 de H2: segmentación por energía pasa-altos, Poincaré fino, ajuste de Sherlock y Monro,
respaldo de curvatura por centroide, esqueleto de Lee."""
from __future__ import annotations
import math
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import proto_h2 as H
import ajuste_sm2 as A

P = dict(H.P)
P.update(sigma_tensor=3.0, sigma_poinc=2.0, r_poinc=6.0, sigma_ajuste=5.0, holgura=6.0, margen=24.0,
         s_hp=3.0, s_e=6.0, umbral_seg=0.55, sigma_curv=10.0)

def a_gris(imagen):
    imagen = np.asarray(imagen)
    if imagen.dtype != np.uint8 or imagen.ndim not in (2, 3) or (imagen.ndim == 3 and imagen.shape[2] not in (3, 4)):
        raise ValueError("imagen inválida")
    return H.a_gris(imagen)

def segmentar(g, p=P):
    f = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    hp = f - cv2.GaussianBlur(f, (0, 0), p["s_hp"])
    en = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), p["s_e"]))
    ref = np.percentile(en, 95)
    if ref < 5.0:
        raise ValueError("no hay huella")
    masc = (en > p["umbral_seg"] * ref).astype(np.uint8)
    masc = cv2.morphologyEx(masc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(masc, 8)
    if n <= 1:
        raise ValueError("no hay huella")
    masc = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return ndi.binary_fill_holes(masc)

def curvatura_centroide(u, v, masc, p=P):
    s = p["sigma_curv"]
    uu = cv2.GaussianBlur(u.astype(np.float32), (0, 0), s).astype(np.float64)
    vv = cv2.GaussianBlur(v.astype(np.float32), (0, 0), s).astype(np.float64)
    m = np.sqrt(uu ** 2 + vv ** 2) + 1e-9
    uu, vv = uu / m, vv / m
    th = 0.5 * np.arctan2(vv, uu)
    tx, ty = np.cos(th), np.sin(th)
    duy, dux = np.gradient(uu); dvy, dvx = np.gradient(vv)
    k = np.abs(0.5 * (uu * (tx * dvx + ty * dvy) - vv * (tx * dux + ty * duy)))
    dist = cv2.distanceTransform(np.pad(masc, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    k[dist < p["margen"] + 10] = 0
    k = cv2.GaussianBlur(k.astype(np.float32), (0, 0), 6)
    i, j = np.unravel_index(int(np.argmax(k)), k.shape)
    lab, _ = ndi.label(k >= 0.5 * k.max())
    yy, xx = np.nonzero(lab == lab[i, j]); w = k[yy, xx].astype(np.float64)
    return float((xx * w).sum() / w.sum()), float((yy * w).sum() / w.sum())

def preprocesar(imagen, p=P):
    g = cv2.medianBlur(a_gris(imagen), 3)
    masc = segmentar(g, p)
    th, coh, u, v = H.campo(g, masc, p)
    sing, _ = H.singularidades(u, v, masc, p)
    nuc = sorted([b for b in sing["nucleos"] if b[0] >= 0.5], key=lambda b: -b[0])
    if len(nuc) >= 2:
        nucleos, metodo = [(nuc[0][1], nuc[0][2]), (nuc[1][1], nuc[1][2])], "nucleos"
    elif len(nuc) == 1 and nuc[0][0] >= 1.5:
        nucleos, metodo = [(nuc[0][1], nuc[0][2] - 8), (nuc[0][1], nuc[0][2] + 8)], "nucleos"
    elif len(nuc) == 1:
        nucleos, metodo = [(nuc[0][1], nuc[0][2])], "nucleo"
    else:
        nucleos, metodo = [], "curvatura"
    dets = [(b[1], b[2]) for b in sing["deltas"] if b[0] >= 0.5]
    pf = dict(p); pf["sigma_tensor"] = p["sigma_ajuste"]
    th5, coh5, u5, v5 = H.campo(g, masc, pf)
    if nucleos:
        ref0 = tuple(np.mean(np.array(nucleos), axis=0))
    else:
        ref0 = curvatura_centroide(u5, v5, masc, p)
    phi0, _ = H.angulo_anillo(u5, v5, coh5, masc, ref0)
    phi, r, cost = A.ajustar(th5, coh5, masc, nucleos, dets, phi0, p["holgura"])
    if r is not None:
        nucleos = [(float(a), float(b)) for a, b in r.x[1:1 + 2 * len(nucleos)].reshape(-1, 2)]
        ref = tuple(np.mean(np.array(nucleos), axis=0))
    else:
        ref = ref0
    ref = (float(ref[0]), float(ref[1]))
    M = H.matriz_ventana(ref, phi, p["ventana"])
    alto_w, ancho_w = p["ventana"]
    margen = 24
    Mm = M.copy(); Mm[:, 2] += margen
    tam = (ancho_w + 2 * margen, alto_w + 2 * margen)
    fondo = float(np.median(g[~masc])) if (~masc).any() else 255.0
    gw = cv2.warpAffine(g, Mm, tam, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fondo)
    mw = cv2.warpAffine(masc.astype(np.uint8), Mm, tam, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    uw = cv2.warpAffine(u5.astype(np.float32), Mm, tam, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    vw = cv2.warpAffine(v5.astype(np.float32), Mm, tam, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    c2, s2 = math.cos(2 * phi), math.sin(2 * phi)
    thw = (0.5 * np.arctan2(vw * c2 - uw * s2, uw * c2 + vw * s2)) % np.pi
    gc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gw)
    resp = H.gabor_realce(gc, thw, mw, p["periodo"])
    esc = np.percentile(np.abs(resp[mw]), 95) if mw.any() else 1.0
    realz = np.clip(128 + 127 * resp / (esc + 1e-9), 0, 255)
    realz[~mw] = 255
    sl = (slice(margen, margen + alto_w), slice(margen, margen + ancho_w))
    mw_c = mw[sl]
    binaria = (resp[sl] < 0) & mw_c
    esq = skeletonize(binaria, method="lee").astype(bool)
    return dict(realzada=realz[sl].astype(np.uint8), binaria=binaria, esqueleto=esq, mascara=mw_c,
                orientacion=thw[sl].astype(np.float64), punto_referencia=ref, metodo=metodo, angulo=float(phi),
                matriz=M, nucleos=tuple(nucleos), mascara_entrada=masc)
