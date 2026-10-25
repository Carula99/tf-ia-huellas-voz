"""Prototipo F5: tiempos de H2 por separado de H3 + H4, y distancias de las primeras variantes e intrusos nuevos."""
import sys, time
from pathlib import Path
import numpy as np
RAIZ = Path(sys.argv[1]); sys.path.insert(0, str(RAIZ))
from src.config import cargar_config
from src.huellas.generador import generar_base
from src.huellas.variantes import sortear_parametros, aplicar_variante
from src.huellas.clasificacion import cargar_modelo, reconocer
from src.huellas.preprocesamiento import preprocesar
from src.huellas.caracteristicas import caracteristicas
import cv2
cfg = cargar_config(); modelo = cargar_modelo(RAIZ / "models/modelo_huellas.npz")
def variante(base, semilla, v):
    rng = np.random.default_rng([semilla, v]); return aplicar_variante(base, sortear_parametros(rng, cfg), rng).imagen
# tiempos con las reservadas
t2, t34 = [], []
for k in (1, 2, 3):
    for v in (39, 40):
        im = cv2.imread(str(RAIZ / f"data/huellas/test/id{k}_v{v:03d}.png"), cv2.IMREAD_UNCHANGED)
        t = time.perf_counter(); h = preprocesar(im, cfg); t2.append(time.perf_counter() - t)
        t = time.perf_counter(); z = modelo.escalador.transformar(caracteristicas(h, cfg)); d, j = modelo.puntajes(z); t34.append(time.perf_counter() - t)
        r = reconocer(im, modelo, cfg)
        assert (r.distancia, r.mas_cercana) == (d, j), "no coincide con reconocer"
        print(f"id{k}_v{v:03d}: distancia {d:.4f} → {r.identidad}")
print(f"H2 (preprocesar): media {np.mean(t2)*1e3:.0f} ms; H3 + H4: media {np.mean(t34)*1e3:.0f} ms")
# identidad de las variantes de prueba igual a la del PNG del dataset (receta de H1)
base = generar_base("presilla", 101, cfg)
print("receta v039 == PNG:", np.array_equal(variante(base, 101, 39), cv2.imread(str(RAIZ / "data/huellas/test/id1_v039.png"), cv2.IMREAD_UNCHANGED)))
for ident in cfg["huellas"]["identidades"]:
    b = generar_base(ident["tipo"], ident["semilla"], cfg)
    ds = [reconocer(variante(b, ident["semilla"], 1000 + n), modelo, cfg) for n in range(10)]
    print(f"id{ident['id']} v1000–v1009:", " ".join(f"{r.distancia:.3f}" for r in ds), "| identidades", {r.identidad for r in ds})
intr = cfg["huellas"]["intrusos"]
ds = []
for n in range(10):
    i = 1000 + n
    b = generar_base(intr["tipos"][(i - 1) % 4], intr["semilla_base"] + i, cfg)
    ds.append(reconocer(variante(b, intr["semilla_base"] + i, 1), modelo, cfg))
print("intrusos 1000–1009:", " ".join(f"{r.distancia:.3f}" for r in ds), "| identidades", {r.identidad for r in ds})
