"""Salidas de H2 (src/huellas/preprocesamiento.py, v0.2) de las 300 imágenes del dataset."""
import sys, time, pickle
sys.path.insert(0, "/home/claude/h2/repo2")
import numpy as np
from src.config import cargar_config
from src.huellas.dataset import plan_dataset
from src.huellas.generador import generar_base
from src.huellas.variantes import sortear_parametros, aplicar_variante
from src.huellas.preprocesamiento import preprocesar

cfg = cargar_config("/home/claude/h2/repo2/config.yaml")
plan = plan_dataset(cfg)
bases, filas = {}, []
t0 = time.time()
for e in plan:
    k = (e["tipo"], e["semilla"])
    if k not in bases:
        bases[k] = generar_base(e["tipo"], e["semilla"], cfg)
    rng = np.random.default_rng([e["semilla"], e["variante"]])
    v = aplicar_variante(bases[k], sortear_parametros(rng, cfg), rng)
    r = preprocesar(v.imagen, cfg)
    filas.append(dict(e=e, p=v.parametros, imagen=v.imagen, realzada=r.realzada, binaria=r.binaria, esqueleto=r.esqueleto,
                      mascara=r.mascara, orientacion=r.orientacion.astype(np.float32), ref=r.punto_referencia,
                      metodo=r.metodo, angulo=r.angulo, matriz=r.matriz))
print(len(filas), f"{time.time()-t0:.0f} s")
pickle.dump({"cfg": cfg, "filas": filas}, open("/home/claude/h3/h2_cache.pkl", "wb"))
