import pickle, sys, math
sys.path.insert(0, "/home/claude/h2/repo"); sys.path.insert(0, "/home/claude/h2/proto")
import numpy as np, cv2
import proto_h2c as B
D = pickle.load(open("cache.pkl", "rb"))
bases, filas = D["bases"], D["filas"]
def elegir(pred):
    c = [(e, v) for e, v in filas if pred(e)]
    con = [x for x in c if x[1].parametros.corte_fraccion > 0]
    sin = [x for x in c if x[1].parametros.corte_fraccion == 0]
    sin.sort(key=lambda x: -abs(x[1].parametros.rot_grados)); con.sort(key=lambda x: -abs(x[1].parametros.rot_grados))
    return [sin[0], con[0]] if con else sin[:2]
filas_l = [("Identidad 1 · presilla", elegir(lambda e: e["identidad"] == 1)), ("Identidad 2 · verticilo", elegir(lambda e: e["identidad"] == 2)),
           ("Identidad 3 · arco en tienda", elegir(lambda e: e["identidad"] == 3)), ("Intruso 8 · arco", elegir(lambda e: e["intruso"] == 8))]
H_ = 250
def entrada(v, r):
    img = cv2.cvtColor(v.imagen, cv2.COLOR_GRAY2BGR)
    x, y = r["punto_referencia"]; a = r["angulo"]
    for s in (-1, 1):
        cv2.line(img, (int(round(x)), int(round(y))), (int(round(x + s*90*math.cos(a))), int(round(y + s*90*math.sin(a)))), (0, 160, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (int(round(x)), int(round(y))), 7, (0, 0, 230), 2, cv2.LINE_AA)
    if v.punto_referencia is not None:
        tx, ty = v.punto_referencia; cv2.drawMarker(img, (int(round(tx)), int(round(ty))), (0, 170, 0), cv2.MARKER_CROSS, 12, 2)
    M = r["matriz"]; Mi = cv2.invertAffineTransform(M)
    esq = np.array([[0, 0], [249, 0], [249, 249], [0, 249]], float)
    pts = (Mi[:, :2] @ esq.T).T + Mi[:, 2]
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, (200, 120, 0), 1, cv2.LINE_AA)
    return cv2.resize(img, (int(320 * H_ / 400), H_), interpolation=cv2.INTER_AREA)
def rotulo(img, t):
    out = cv2.copyMakeBorder(img, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    return out
renglones = []
for titulo, sel in filas_l:
    celdas = []
    for e, v in sel:
        r = B.preprocesar(v.imagen); p = v.parametros
        corte = f", corte {p.corte_fraccion*100:.0f} %" if p.corte_fraccion > 0 else ""
        ea = ((math.degrees(r["angulo"]) + p.rot_grados + 90) % 180) - 90
        err = f"ref {math.dist(r['punto_referencia'], v.punto_referencia):.1f} px, " if v.punto_referencia else ""
        celdas.append(rotulo(entrada(v, r), f"{e['archivo'].split('/')[1][:-4]}: rot {p.rot_grados:+.0f} gr{corte}"))
        celdas.append(rotulo(cv2.cvtColor(r["realzada"], cv2.COLOR_GRAY2BGR), f"ventana realzada ({err}ang {ea:+.1f} gr)"))
        esq = np.full((250, 250, 3), 255, np.uint8); esq[r["esqueleto"]] = (0, 0, 0); esq[~r["mascara"]] = (235, 235, 235)
        celdas.append(rotulo(esq, f"esqueleto ({r['metodo']})"))
    sep = np.full((celdas[0].shape[0], 8, 3), 255, np.uint8)
    fila = np.hstack(sum([[c, sep] for c in celdas], [])[:-1])
    cab = np.full((26, fila.shape[1], 3), 255, np.uint8)
    cv2.putText(cab, titulo, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    renglones.append(np.vstack([cab, fila, np.full((10, fila.shape[1], 3), 255, np.uint8)]))
w = max(x.shape[1] for x in renglones)
lam = np.vstack([cv2.copyMakeBorder(x, 0, 0, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in renglones])
lam = cv2.copyMakeBorder(lam, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
cv2.imwrite("h2_proto.png", lam, [cv2.IMWRITE_PNG_COMPRESSION, 6]); print(lam.shape)
