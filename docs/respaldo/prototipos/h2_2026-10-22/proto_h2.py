"""Prototipo de H2 (no es el código final): segmentación, campo de orientación, singularidades
por índice de Poincaré, punto de referencia, ángulo del anillo, ventana, realce y esqueleto."""
from __future__ import annotations
import math
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, thin

P = dict(
    sigma_suave=1.0,       # suavizado antes de los gradientes
    sigma_tensor=7.0,      # suavizado del tensor de estructura
    sigma_seg=8.0,         # ventana de la varianza local (segmentación)
    umbral_seg=0.30,       # fracción del desvío típico de la huella
    sigma_poinc=3.0,       # suavizado extra del campo doble para Poincaré
    r_poinc=10.0,          # radio del círculo de Poincaré
    paso=2,                # grilla del mapa de índices
    margen=24.0,           # distancia mínima al borde de la máscara para aceptar singularidades
    anillo=(40.0, 160.0),  # radios del anillo del enderezado
    peso_coh=1.0,          # exponente de la coherencia en el promedio
    sigma_curv=10.0,
    ventana=(250, 250),
    periodo=9.0,
)


def a_gris(imagen):
    if imagen.ndim == 3:
        if imagen.shape[2] == 4:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
        else:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen


def segmentar(g, p=P):
    """Máscara por desvío local (pixel a pixel, ventana gaussiana)."""
    f = g.astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), 1.2)
    m = cv2.GaussianBlur(f, (0, 0), p["sigma_seg"])
    v = cv2.GaussianBlur(f * f, (0, 0), p["sigma_seg"]) - m * m
    s = np.sqrt(np.maximum(v, 0))
    ref = np.percentile(s, 95)
    masc = s > p["umbral_seg"] * ref
    masc = cv2.morphologyEx(masc.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(masc, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        masc = lab == k
    else:
        masc = masc.astype(bool)
    masc = ndi.binary_fill_holes(masc)
    return masc, s


def campo(g, masc, p=P):
    """Campo de orientación de las crestas (doble ángulo) y coherencia, por tensor de estructura."""
    f = cv2.GaussianBlur(g.astype(np.float32), (0, 0), p["sigma_suave"])
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    w = masc.astype(np.float32)
    a = (gx * gx - gy * gy) * w
    b = (2 * gx * gy) * w
    e = (gx * gx + gy * gy) * w
    s = p["sigma_tensor"]
    A = cv2.GaussianBlur(a, (0, 0), s); B = cv2.GaussianBlur(b, (0, 0), s); E = cv2.GaussianBlur(e, (0, 0), s)
    # Cresta perpendicular al gradiente: el doble ángulo se invierte.
    u, v = -A, -B
    mod = np.sqrt(u * u + v * v)
    coh = mod / (E + 1e-6)
    theta = (0.5 * np.arctan2(v, u)) % np.pi
    return theta.astype(np.float64), coh.astype(np.float64), (u / (mod + 1e-9)).astype(np.float64), (v / (mod + 1e-9)).astype(np.float64)


def indice_poincare(u, v, xs, ys, r, n=36):
    """Índice de Poincaré del campo doble (u, v) en los puntos (xs, ys)."""
    phi = 2 * np.pi * np.arange(n + 1) / n
    px = xs[:, None] + r * np.cos(phi)[None, :]
    py = ys[:, None] + r * np.sin(phi)[None, :]
    cu = ndi.map_coordinates(u, [py.ravel(), px.ravel()], order=1, mode="nearest").reshape(px.shape)
    cv = ndi.map_coordinates(v, [py.ravel(), px.ravel()], order=1, mode="nearest").reshape(px.shape)
    ang = np.arctan2(cv, cu)  # = 2θ
    d = np.diff(ang, axis=1)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return d.sum(axis=1) / (4 * np.pi)   # doble ángulo: /2 y /2π


def singularidades(u, v, masc, p=P):
    s = p["sigma_poinc"]
    if s > 0:
        uu = cv2.GaussianBlur(u.astype(np.float32), (0, 0), s); vv = cv2.GaussianBlur(v.astype(np.float32), (0, 0), s)
    else:
        uu, vv = u.astype(np.float32), v.astype(np.float32)
    alto, ancho = u.shape
    dist = cv2.distanceTransform(np.pad(masc, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    paso = p["paso"]
    yy, xx = np.mgrid[0:alto:paso, 0:ancho:paso]
    ok = dist[yy, xx] >= p["margen"]
    idx = np.zeros(yy.shape)
    idx[ok] = indice_poincare(uu.astype(np.float64), vv.astype(np.float64), xx[ok].astype(float), yy[ok].astype(float), p["r_poinc"])
    res = {}
    area_disco = math.pi * p["r_poinc"] ** 2 / paso ** 2
    for signo, nombre in ((1, "nucleos"), (-1, "deltas")):
        bin_ = (signo * idx) > 0.25
        lab, n = ndi.label(bin_, structure=np.ones((3, 3)))
        blobs = []
        for k in range(1, n + 1):
            sel = lab == k
            w = signo * idx[sel]
            carga = w.sum() / (0.5 * area_disco)
            cx = float((xx[sel] * w).sum() / w.sum()); cy = float((yy[sel] * w).sum() / w.sum())
            blobs.append((carga, cx, cy))
        res[nombre] = blobs
    return res, idx


def curvatura(u, v, theta, masc, p=P):
    s = p["sigma_curv"]
    uu = cv2.GaussianBlur(u.astype(np.float32), (0, 0), s).astype(np.float64)
    vv = cv2.GaussianBlur(v.astype(np.float32), (0, 0), s).astype(np.float64)
    m = np.sqrt(uu ** 2 + vv ** 2) + 1e-9
    uu, vv = uu / m, vv / m
    th = 0.5 * np.arctan2(vv, uu)
    tx, ty = np.cos(th), np.sin(th)
    duy, dux = np.gradient(uu); dvy, dvx = np.gradient(vv)
    k = 0.5 * (uu * (tx * dvx + ty * dvy) - vv * (tx * dux + ty * duy))
    k = np.abs(k)
    dist = cv2.distanceTransform(np.pad(masc, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    k[dist < p["margen"] + 10] = 0
    k = cv2.GaussianBlur(k.astype(np.float32), (0, 0), 6)
    i, j = np.unravel_index(int(np.argmax(k)), k.shape)
    return float(j), float(i), k


def punto_referencia(u, v, theta, masc, p=P):
    sing, idx = singularidades(u, v, masc, p)
    nuc = [b for b in sing["nucleos"] if b[0] >= 0.5]
    carga = sum(b[0] for b in nuc)
    if not nuc:
        x, y, _ = curvatura(u, v, theta, masc, p)
        return (x, y), "curvatura", sing, 0
    # dos más fuertes si hay más
    nuc = sorted(nuc, key=lambda b: -b[0])
    n_est = int(round(sum(b[0] for b in nuc[:2])))
    if len(nuc) >= 2:
        sel = nuc[:2]
        x = (sel[0][1] + sel[1][1]) / 2; y = (sel[0][2] + sel[1][2]) / 2
        return (x, y), "nucleos", sing, 2
    b = nuc[0]
    if b[0] >= 1.5:
        return (b[1], b[2]), "nucleos", sing, 2
    return (b[1], b[2]), "nucleo", sing, 1


def angulo_anillo(u, v, coh, masc, ref, p=P):
    alto, ancho = u.shape
    yy, xx = np.mgrid[0:alto, 0:ancho]
    r = np.hypot(xx - ref[0], yy - ref[1])
    sel = masc & (r >= p["anillo"][0]) & (r <= p["anillo"][1])
    w = coh[sel] ** p["peso_coh"]
    su = (w * u[sel]).sum(); sv = (w * v[sel]).sum()
    return (0.5 * math.atan2(sv, su)) % math.pi, math.hypot(su, sv) / (w.sum() + 1e-12)


def giro_ventana(phi):
    """El menor de los dos giros que dejan θ∞ horizontal: ((phi + π/2) mod π) − π/2, en [−π/2, π/2)."""
    return ((phi + math.pi / 2) % math.pi) - math.pi / 2


def matriz_ventana(ref, phi, ventana=(250, 250)):
    alto, ancho = ventana
    a = giro_ventana(phi)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, s], [-s, c]])
    centro = np.array([(ancho - 1) / 2, (alto - 1) / 2])
    t = centro - R @ np.array(ref)
    return np.hstack([R, t[:, None]])


def gabor_realce(img, theta_w, masc_w, periodo):
    """Gabor orientado por píxel: banco de 16 orientaciones con interpolación entre las dos vecinas."""
    n = 16
    sigma = 0.5 * periodo
    lado = int(2 * round(3 * sigma) + 1)
    f = img.astype(np.float32)
    f = f - cv2.GaussianBlur(f, (0, 0), 6)
    resp = []
    for k in range(n):
        cresta = np.pi * k / n
        g = cv2.getGaborKernel((lado, lado), sigma, cresta + np.pi / 2, periodo, 1.0, 0, ktype=cv2.CV_32F)
        g -= g.mean()
        resp.append(cv2.filter2D(f, -1, g, borderType=cv2.BORDER_REFLECT))
    resp = np.stack(resp)
    pos = theta_w / np.pi * n
    k0 = np.floor(pos).astype(np.intp) % n; k1 = (k0 + 1) % n
    w = (pos - np.floor(pos)).astype(np.float32)
    r0 = np.take_along_axis(resp, k0[None], 0)[0]; r1 = np.take_along_axis(resp, k1[None], 0)[0]
    return (1 - w) * r0 + w * r1


def preprocesar(imagen, p=P):
    g = a_gris(np.asarray(imagen))
    g = cv2.medianBlur(g, 3)
    masc, _ = segmentar(g, p)
    theta, coh, u, v = campo(g, masc, p)
    ref, metodo, sing, n = punto_referencia(u, v, theta, masc, p)
    phi, fuerza = angulo_anillo(u, v, coh, masc, ref, p)
    M = matriz_ventana(ref, phi, p["ventana"])
    alto_w, ancho_w = p["ventana"]
    margen = 24
    # Ventana con margen para que los filtros no vean el borde.
    Mm = M.copy(); Mm[:, 2] += margen
    tam = (ancho_w + 2 * margen, alto_w + 2 * margen)
    fondo = float(np.median(g[~masc])) if (~masc).any() else 255.0
    gw = cv2.warpAffine(g, Mm, tam, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fondo)
    mw = cv2.warpAffine(masc.astype(np.uint8), Mm, tam, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gc = clahe.apply(gw)
    thw, cohw, uw, vw = campo(gc, mw, p)
    resp = gabor_realce(gc, thw, mw, p["periodo"])
    # crestas oscuras: respuesta positiva del Gabor coseno sobre imagen con crestas oscuras = valle
    cresta = (resp < 0) & mw
    esc = np.percentile(np.abs(resp[mw]), 95) if mw.any() else 1.0
    realz = np.clip(128 + 127 * resp / (esc + 1e-9), 0, 255)
    realz[~mw] = 255
    sl = (slice(margen, margen + alto_w), slice(margen, margen + ancho_w))
    binaria = cresta[sl]
    mw_c = mw[sl]
    # limpiar
    binaria = binaria & cv2.erode(mw_c.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    esq = skeletonize(binaria)
    return dict(realzada=realz[sl].astype(np.uint8), binaria=binaria, esqueleto=esq, mascara=mw_c,
                orientacion=thw[sl], coherencia=cohw[sl], punto_referencia=ref, metodo=metodo, angulo=phi,
                matriz=M, sing=sing, mascara_entrada=masc, theta_entrada=theta, fuerza=fuerza, n=n)
