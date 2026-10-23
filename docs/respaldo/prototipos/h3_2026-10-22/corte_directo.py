"""Experimento directo de cortes (criterio 5.6 del contrato de H3) con el escalador y los centroides de train."""
import pickle, sys
import numpy as np
sys.path.insert(0, "/home/claude/h3"); sys.path.insert(0, "/home/claude/h2/repo2")
import proto_h3d as F
from src.config import cargar_config
from src.huellas.generador import generar_base
from src.huellas.variantes import ParametrosVariante, aplicar_variante
from src.huellas.preprocesamiento import preprocesar

cfg = cargar_config("/home/claude/h2/repo2/config.yaml")
R = pickle.load(open("/home/claude/h3/X_contrato.pkl", "rb"))
esc, cent, Z, part, ident = R["esc"], R["cent"], R["Z"], R["part"], R["ident"]
tr = part == "train"
maxtr = {k: np.linalg.norm(Z[tr & (ident == k)] - cent[k], axis=1).max() for k in (1, 2, 3)}
f = max(cfg["huellas"]["variantes"]["corte_periferico"])
for ide in cfg["huellas"]["identidades"]:
    k = ide["id"]; base = generar_base(ide["tipo"], ide["semilla"], cfg)
    for a in (None, 0.0, 90.0, 180.0, 270.0):
        if a is None:
            img = base.imagen; nombre = "base"
        else:
            v = aplicar_variante(base, ParametrosVariante(12.0, 1.0, 10.0, -8.0, 0.0, 0.0, f, a, 0.0, 0.0), np.random.default_rng(0))
            img = v.imagen; nombre = f"corte {a:.0f}°"
        h = preprocesar(img, cfg)
        z = esc.transformar(F.caracteristicas(h))
        d = np.array([np.linalg.norm(z - cent[j]) for j in (1, 2, 3)])
        sin = 1 - h.mascara.mean()
        print(f"id{k} {nombre:10s}: asignada {d.argmin() + 1}, dist {d[k - 1]:.3f} (máx train {maxtr[k]:.3f}, cociente {d[k - 1] / maxtr[k]:.2f}); "
              f"otras {np.delete(d, k - 1).min():.3f}; ventana sin huella {100 * sin:.1f} %")
