"""Prototipo F5 (huellas): variantes nuevas, intrusos nuevos, tiempos, errores y visor PCA."""
import sys, time
from pathlib import Path
import numpy as np
RAIZ = Path(sys.argv[1])
sys.path.insert(0, str(RAIZ))
from src.config import cargar_config
from src.huellas.generador import generar_base
from src.huellas.variantes import sortear_parametros, aplicar_variante
from src.huellas.clasificacion import cargar_modelo, reconocer
from src.huellas.preprocesamiento import preprocesar
from src.core_ml.pca import PCA

cfg = cargar_config()
modelo = cargar_modelo(RAIZ / "models/modelo_huellas.npz")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
DESDE = 1000

def variante(base, semilla, v):
    rng = np.random.default_rng([semilla, v])
    return aplicar_variante(base, sortear_parametros(rng, cfg), rng).imagen

print(f"tau* = {modelo.tau:.4f}")
# identidades
for ident in cfg["huellas"]["identidades"]:
    t = time.perf_counter(); base = generar_base(ident["tipo"], ident["semilla"], cfg); tb = time.perf_counter() - t
    ok = mal = rech = 0; dist = []; tv = []; tr = []
    for n in range(N):
        t = time.perf_counter(); im = variante(base, ident["semilla"], DESDE + n); tv.append(time.perf_counter() - t)
        t = time.perf_counter(); r = reconocer(im, modelo, cfg); tr.append(time.perf_counter() - t)
        dist.append(r.distancia)
        if r.identidad == ident["id"]: ok += 1
        elif r.identidad == 0: rech += 1
        else: mal += 1
    dist = np.array(dist)
    print(f"id{ident['id']}: base {tb:.2f} s; variante {np.mean(tv)*1e3:.0f} ms; reconocer {np.mean(tr)*1e3:.0f} ms; "
          f"{ok}/{N} aceptadas bien, {rech} rechazadas, {mal} con otra identidad; distancia mediana {np.median(dist):.3f}, "
          f"máx {dist.max():.3f}, margen<0.1: {(np.abs(modelo.tau - dist) < 0.1).sum()}")
# intrusos nuevos
intr = cfg["huellas"]["intrusos"]; tipos = intr["tipos"]
acept = 0; dist = []; tb = []
for n in range(N):
    i = DESDE + n
    t = time.perf_counter(); base = generar_base(tipos[(i - 1) % len(tipos)], intr["semilla_base"] + i, cfg); tb.append(time.perf_counter() - t)
    r = reconocer(variante(base, intr["semilla_base"] + i, 1), modelo, cfg)
    dist.append(r.distancia); acept += r.identidad != 0
dist = np.array(dist)
print(f"intrusos nuevos: base {np.mean(tb):.2f} s; {acept}/{N} aceptados; distancia mín {dist.min():.3f}, mediana {np.median(dist):.3f}")
# errores
for nombre, im in [("fondo liso 245", np.full((400, 320), 245, np.uint8)), ("ruido", np.random.default_rng(0).integers(0, 256, (400, 320), dtype=np.uint8)),
                   ("negra", np.zeros((400, 320), np.uint8)), ("chica 50x50", np.full((50, 50), 128, np.uint8)), ("float", np.zeros((400, 320)))]:
    try:
        r = reconocer(im, modelo, cfg); print(f"{nombre}: identidad {r.identidad}, distancia {r.distancia:.3f}")
    except Exception as e:
        print(f"{nombre}: {type(e).__name__}: {e}")
# visor PCA
d = np.load(RAIZ / "data/huellas/caracteristicas.npz", allow_pickle=False)
tr = d["particion"] == "train"
pca = PCA(2).ajustar(d["X"][tr])
print("PCA 2D de train: proporción", np.round(pca.proporcion_varianza, 4), "suma", round(float(pca.proporcion_varianza.sum()), 4))
print("centroides proyectados", np.round(pca.transformar(modelo.centroides), 3).tolist())
