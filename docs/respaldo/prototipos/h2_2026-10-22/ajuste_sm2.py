"""Ajuste del modelo de Sherlock y Monro: θ∞, núcleos (acotados cerca de lo detectado) y deltas."""
import math
import numpy as np, cv2
from scipy.optimize import least_squares

def muestras(theta, coh, masc, nucleos, paso=6, excl=8.0, erosion=6):
    m = cv2.erode(masc.astype(np.uint8), np.ones((2*erosion+1, 2*erosion+1), np.uint8)).astype(bool)
    yy, xx = np.mgrid[paso//2:theta.shape[0]:paso, paso//2:theta.shape[1]:paso]
    ok = m[yy, xx]
    for cx, cy in nucleos:
        ok &= np.hypot(xx - cx, yy - cy) > excl
    x = xx[ok].astype(float); y = yy[ok].astype(float)
    t = theta[yy[ok], xx[ok]]; w = np.sqrt(np.clip(coh[yy[ok], xx[ok]], 0, 1))
    return x, y, np.cos(2*t), np.sin(2*t), w

def suma_args(p, x, y, n):
    s = np.full_like(x, 2*p[0]); k = 1
    for i in range(n):
        s += np.arctan2(y - p[k+1], x - p[k]); k += 2
    for i in range(n):
        s -= np.arctan2(y - p[k+1], x - p[k]); k += 2
    return s

def ajustar(theta, coh, masc, nucleos, deltas_det, phi0, holgura=6.0, excl=8.0, paso=6):
    n = len(nucleos)
    x, y, c2, s2, w = muestras(theta, coh, masc, nucleos, paso=paso, excl=excl)
    if n == 0:
        return (0.5*math.atan2((w*s2).sum(), (w*c2).sum())) % math.pi, None, 0.0
    ref = np.mean(np.array(nucleos), axis=0)
    R = lambda dx, dy: (ref[0] + dx*math.cos(phi0) - dy*math.sin(phi0), ref[1] + dx*math.sin(phi0) + dy*math.cos(phi0))
    inits = [[R(-80, 90)], [R(80, 90)], [R(0, 30)]] if n == 1 else [[R(-80, 90), R(80, 90)]]
    if len(deltas_det) >= n:
        cerc = sorted(deltas_det, key=lambda d: min(math.dist(d, c) for c in nucleos))[:n]
        inits.insert(0, cerc)
    lo, hi = [-np.inf], [np.inf]
    for cx, cy in nucleos:
        lo += [cx - holgura, cy - holgura]; hi += [cx + holgura, cy + holgura]
    lo += [-np.inf] * 2 * n; hi += [np.inf] * 2 * n
    mejor = None
    for d0 in inits:
        p0 = [phi0] + [v for c in nucleos for v in c] + [v for d in d0 for v in d]
        def res(p):
            s = suma_args(p, x, y, n)
            return np.concatenate([w*(c2 - np.cos(s)), w*(s2 - np.sin(s))])
        r = least_squares(res, p0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=300)
        if mejor is None or r.cost < mejor.cost:
            mejor = r
    return mejor.x[0] % math.pi, mejor, mejor.cost / len(x)
