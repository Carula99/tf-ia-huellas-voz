import pickle, sys, math, collections
sys.path.insert(0, "/home/claude/h2/repo"); sys.path.insert(0, "/home/claude/h2/proto")
import numpy as np, cv2
import proto_h2 as H, proto_h2c as B
D = pickle.load(open("cache.pkl", "rb"))
bases, filas = D["bases"], D["filas"]
abajo, total, girado_viejo = 0, 0, 0
for e, v in filas:
    if e["tipo"] == "arco": continue
    r = B.preprocesar(v.imagen); M = r["matriz"]
    a_v = math.atan2(M[0, 1], M[0, 0])
    if abs(math.degrees(a_v)) > 90 - 1e-9: print("giro grande", e["archivo"], math.degrees(a_v))
    if math.degrees(r["angulo"]) > 90: girado_viejo += 1
    for dx, dy in v.singularidades.deltas:
        xw, yw = M @ [dx, dy, 1]
        total += 1; abajo += yw > 124.5
print(f"deltas reales abajo del centro de la ventana: {abajo} de {total}; con la matriz vieja habrían salido dadas vuelta {girado_viejo} de {sum(1 for e, v in filas if e['tipo'] != 'arco')} ventanas")
