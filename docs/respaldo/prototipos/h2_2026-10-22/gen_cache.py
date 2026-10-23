"""Genera en memoria el dataset de H1 (receta de la sección 4) y lo guarda en un pickle."""
import pickle, sys, time
sys.path.insert(0, "/home/claude/h2/repo")
import numpy as np
from src.config import cargar_config
from src.huellas.generador import generar_base
from src.huellas.variantes import sortear_parametros, aplicar_variante
from src.huellas.dataset import plan_dataset

cfg = cargar_config("/home/claude/h2/repo/config.yaml")
plan = plan_dataset(cfg)
bases, filas = {}, []
t0 = time.time()
for e in plan:
    clave = (e["tipo"], e["semilla"])
    if clave not in bases:
        bases[clave] = generar_base(e["tipo"], e["semilla"], cfg)
    b = bases[clave]
    rng = np.random.default_rng([e["semilla"], e["variante"]])
    p = sortear_parametros(rng, cfg)
    v = aplicar_variante(b, p, rng)
    filas.append((e, v))
print("listo", len(filas), "bases", len(bases), f"{time.time()-t0:.1f} s")
with open("/home/claude/h2/proto/cache.pkl", "wb") as f:
    pickle.dump({"cfg": cfg, "bases": bases, "filas": filas}, f)
