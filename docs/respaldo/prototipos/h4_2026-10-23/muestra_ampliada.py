"""Muestra ampliada (solo para el prototipo, no es parte del dataset): variantes 41-140 de las
identidades 1-3 y los intrusos 61-160 (semilla 10000 + i, tipo tipos[(i-1) % 4], 3 variantes),
con la receta de H1, preprocesadas con H2, con los bloques de H3 y escaladas con el escalador
de train de caracteristicas.npz."""
import sys, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/home/claude/h4/repo")
from src.config import cargar_config
from src.huellas.generador import generar_base
from src.huellas.variantes import aplicar_variante, sortear_parametros
from src.huellas.preprocesamiento import preprocesar
from src.huellas import caracteristicas as Cr

cfg = cargar_config()
TIPOS = list(cfg["huellas"]["intrusos"]["tipos"])
IDS = [(int(i["id"]), str(i["tipo"]), int(i["semilla"])) for i in cfg["huellas"]["identidades"]]


def trabajo(t):
    """t = (identidad, intruso, tipo, semilla, variantes)."""
    ident, intr, tipo, semilla, variantes = t
    base = generar_base(tipo, semilla, cfg)
    out = []
    for v in variantes:
        rng = np.random.default_rng([semilla, v])
        p = sortear_parametros(rng, cfg)
        var = aplicar_variante(base, p, rng)
        b = Cr.caracteristicas(preprocesar(var.imagen, cfg), cfg)
        out.append(dict(identidad=ident, intruso=intr, tipo=tipo, variante=v, corte=p.corte_fraccion,
                        **{k: b[k] for k in Cr.BLOQUES}))
    return out


if __name__ == "__main__":
    tareas = []
    for k, tipo, sem in IDS:                                   # 100 variantes por identidad, en 10 tandas
        for a in range(41, 141, 10):
            tareas.append((k, 0, tipo, sem, list(range(a, a + 10))))
    for i in range(61, 161):
        tareas.append((0, i, TIPOS[(i - 1) % 4], 10000 + i, [1, 2, 3]))
    t0 = time.time()
    with Pool(8) as pool:
        filas = [f for r in pool.imap(trabajo, tareas) for f in r]
    print(f"{len(filas)} huellas en {time.time() - t0:.0f} s")
    D = np.load("/home/claude/h4/caracteristicas.npz", allow_pickle=False)
    tr = D["particion"] == "train"
    esc = Cr.ajustar_escalador({b: D[b][tr] for b in Cr.BLOQUES})
    B = {k: np.vstack([f[k] for f in filas]) for k in Cr.BLOQUES}
    np.savez("/home/claude/h4/muestra_ampliada.npz",
             identidad=np.array([f["identidad"] for f in filas]), intruso=np.array([f["intruso"] for f in filas]),
             tipo=np.array([f["tipo"] for f in filas]), variante=np.array([f["variante"] for f in filas]),
             corte=np.array([f["corte"] for f in filas]), X=esc.transformar(B), **B)
