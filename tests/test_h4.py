"""Tests de H4 · K-means propio, distancia normalizada, umbral de rechazo, matriz de confusión,
archivo del modelo y reconocimiento de una imagen.

Verifican la sección 6 de docs/contratos/h4_clasificacion.md (versión 1, 23/10/2026). La
sección 7 es manual y se anota en el Documento 2. Los tests de la sección 6.9 llevan
@pytest.mark.aceptacion: se corren con `venv\\Scripts\\python -m pytest -m aceptacion tests/test_h4.py`.

Los datasets salen de H1, H2 y H3, que ya están probados. De H1 se usan solo plan_dataset,
generar_dataset y la receta de su sección 4 (generar_base, sortear_parametros, aplicar_variante
y el campo imagen de la variante); de H2, preprocesar; de H3, BLOQUES, caracteristicas,
Escalador, ajustar_escalador y generar_caracteristicas.

Los oráculos de las secciones 0 y 4 (puntaje, decisión, FAR y FRR, calibración del umbral,
evaluación, correspondencia y dispersiones) están escritos acá, solo a partir del contrato, sin
importar nada de src/huellas/clasificacion.py. Dan los τ* de los tres ejemplos de la sección 8
y, sobre el dataset chico, pureza 1, τ* = 2,327 y una autorizada de test rechazada (sección 8).
El K-means se compara con sklearn.cluster.KMeans y la matriz de confusión con
sklearn.metrics.confusion_matrix.

Lecturas del contrato (las dos primeras, respondidas por Santiago el 23/10):
- Tipos de los escalares: "float" se controla con isinstance(x, float) (np.float64 pasa, un
  array 0-d no). "int" o "un entero" es un int de Python que no sea bool (np.int64 no pasa):
  predecir, ModeloHuellas.decidir y ModeloHuellas.puntajes de un vector, los campos identidad
  y mas_cercana de Reconocimiento, n_iter del K-means y las claves y los valores del dict de
  correspondencia. Los arrays que el contrato pide en int64 o float64 se controlan por su dtype.
- 4.4: far_max_val fuera de (0, 1) da ValueError con los dos criterios (el contrato lo lista
  sin condición, y la Fase 0 también lo valida con criterio "eer").
- 6.2: grupos gaussianos con semilla. "Separados": centros a 30 o más entre sí; "juntos":
  centros N(0, 2²), como en el prototipo. Los datos están elegidos para que scikit-learn no
  tenga que reubicar clusters vacíos (sección 8). El oráculo con init dado se corre con
  max_iter 300, 1 y 2 (tol = 1e-4) y también con tol = 1e-2, donde la parada por
  mov ≤ tol · media(var(X)) corta antes y cambia el resultado. El k-means++ propio se prueba
  con las semillas 0 y 2026 contra random_state=0. Además de lo que pide 6.2, se controlan los
  dtypes de la sección 2 y que distancias sea la distancia euclídea.
- 6.5: el modelo armado a mano tiene un punto que la distancia normalizada asigna a un centroide
  que no es el más cercano en distancia euclídea. far_frr se compara con el oráculo en cada
  puntaje observado, en los puntos medios y fuera del rango, además de los valores que da la
  sección 8.
- 6.7: además del cfg real (con val separable), copias del cfg en memoria con
  piso_dispersion 1.0, criterio "eer" y far_max_val 0.30, con un val con solape: muestran que
  esos valores salen de cfg (con el cfg real y grupos separados el piso no actúa y los dos
  criterios dan el mismo τ*).
- 6.8: "cargar_modelo da un modelo con los mismos valores": igual al archivo (exacto) y a
  ajustar_modelo de las filas de train y val del npz (1e-12); además, guardar_modelo →
  cargar_modelo devuelve exactamente lo guardado. "Sin fuga": el npz se reescribe con otras
  filas de test (en X y en los bloques) y los dos modelos guardados se comparan clave por
  clave con np.array_equal (los NaN en el mismo lugar cuentan como iguales). "Se vuelve a
  escribir": el modelo se reemplaza por bytes que no son un npz y tiene que volver a quedar el
  mismo. El modelo de los tests de reconocimiento y de evaluar_particion es el que devuelve
  cargar_modelo del archivo que escribió entrenar.
- Errores: solo los que nombra la sección 6, como en H3. No se prueban los ValueError de
  puntajes ni de guardar_modelo, ni el FileNotFoundError de cargar_modelo.
- Tolerancias: absolutas salvo donde dice "relativa"; "igual" sin tolerancia se compara a
  1e-12; "idéntico" y np.array_equal, exacto.

src.config y los módulos de src se importan en fixtures o dentro de cada test: si algo todavía
no existe, fallan esos tests y no la colección entera. El dataset chico (33 huellas: PNG,
metadatos.csv y caracteristicas.npz) y el modelo entrenado con él se arman una sola vez
(fixtures de módulo).
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import importlib
import itertools
import math
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans as KMeansSklearn
from sklearn.metrics import confusion_matrix

RAIZ = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Valores del contrato
# ---------------------------------------------------------------------------

K = 3                                   # huellas.kmeans.k
NO_AUTORIZADO = 0                       # sección 0
IDENTIDADES = (1, 2, 3)
BLOQUES_CONTRATO = ("orientacion", "gabor", "minucias")

# Sección 1 / 6.1: camino → (tipo, valor)
CONFIG_H4 = {
    "huellas.kmeans.k": ("int", 3),
    "huellas.kmeans.n_init": ("int", 15),
    "huellas.kmeans.max_iter": ("int", 300),
    "huellas.kmeans.tol": ("float", 1.0e-4),
    "huellas.umbral.criterio": ("str", "far_max"),
    "huellas.umbral.far_max_val": ("float", 0.01),
    "huellas.umbral.piso_dispersion": ("float", 0.5),
}
CLAVE_NUEVA = "huellas.umbral.piso_dispersion"

# 4.6: las 16 claves del archivo del modelo
CLAVES_MODELO = (
    "identidades", "centroides", "dispersion", "tau", "pureza", "inercia", "pesos",
    *(f"{campo}_{b}" for b in BLOQUES_CONTRATO for campo in ("medias", "desvios", "columnas")),
)

# Archivos que no pueden importar scikit-learn (2 y 6.3)
ARCHIVOS_SIN_SKLEARN = ("src/core_ml/kmeans.py", "src/core_ml/metricas.py", "src/huellas/clasificacion.py")

# Sección 8: ejemplos de la calibración (k = 3; 4 autorizadas y 4 intrusos)
EJEMPLOS = {
    "separable": SimpleNamespace(
        verdad=[1, 2, 3, 1, 0, 0, 0, 0],
        prediccion=[1, 2, 3, 1, 1, 2, 3, 1],
        puntaje=[0.5, 0.8, 1.0, 0.7, 1.4, 2.0, 2.5, 3.0],
        tau={("far_max", 0.01): 1.2, ("eer", 0.01): 1.2},
    ),
    "con_solape": SimpleNamespace(
        verdad=[1, 2, 3, 1, 0, 0, 0, 0],
        prediccion=[1, 2, 3, 1, 1, 2, 3, 1],
        puntaje=[0.5, 0.9, 1.3, 0.7, 1.1, 1.6, 2.0, 3.0],
        tau={("far_max", 0.01): 1.0, ("far_max", 0.30): 1.45, ("eer", 0.01): 1.25},
    ),
    "mal_identificada": SimpleNamespace(
        verdad=[1, 2, 3, 1, 0, 0, 0, 0],
        prediccion=[1, 3, 3, 1, 1, 2, 3, 1],
        puntaje=[0.5, 0.6, 1.0, 0.7, 1.4, 2.0, 2.5, 3.0],
        tau={("far_max", 0.01): 1.2, ("eer", 0.01): 1.5},
    ),
}
CASOS_TAU = [(nombre, crit, f, tau) for nombre, e in EJEMPLOS.items() for (crit, f), tau in e.tau.items()]
# (ejemplo, τ, FAR, FRR) que la sección 8 da explícitamente
FAR_FRR_SECCION_8 = [
    ("separable", 1.2, 0.0, 0.0),
    ("con_solape", 1.0, 0.0, 0.25),
    ("mal_identificada", 1.2, 0.0, 0.25),   # la mal identificada (puntaje 0.6) es error con cualquier τ
    ("mal_identificada", 10.0, 1.0, 0.25),
]

# Sección 5: dataset chico
VARIANTES_CHICO = {"train": (1, 2, 3, 4), "val": (25, 26), "test": (33, 34)}
INTRUSOS_CHICO = {"val": (1, 2), "test": (31,)}
CONTEO_CHICO = {"train": 12, "val": 12, "test": 9}
COLUMNAS_CHICO = ("archivo", "particion", "identidad", "intruso", "tipo")

# 6.2: formas de los datos sintéticos (n, d, k)
CASOS_KMEANS = [(300, 2, 4), (60, 40, 3), (500, 10, 5)]
DISPOSICIONES = ("juntos", "separados")
TOL_KMEANS = 1.0e-4
# 6.2: el oráculo con init dado, con max_iter 300, 1 y 2 y con una tol más grande, donde la
# parada por mov ≤ tol · media(var(X)) corta antes y cambia el resultado
ITERACIONES_Y_TOL = [(300, 1.0e-4), (1, 1.0e-4), (2, 1.0e-4), (300, 1.0e-2)]

# Tolerancias
TOL_IGUAL = 1e-12            # "igual" sin tolerancia
TOL_SKLEARN = 1e-6           # 6.2 y 6.9: centroides (absoluta) e inercia (relativa)
TOL_PROPIEDADES = 1e-9       # 6.2: inercia (relativa), medias de los clusters
TOL_DISPERSION = 1e-12       # 6.4
TOL_PUNTAJE = 1e-12          # 6.5
TOL_CENTROIDES_MODELO = 1e-9  # 6.7
TOL_RECONOCER = 1e-9         # 6.8
MARGEN_KMEANSPP = 0.01       # 6.2: la inercia no supera en más del 1 % a la de scikit-learn
PUREZA_MIN = 0.95            # 6.9
TIEMPO_ACEPTACION_S = 300.0  # 6.9: menos de 5 minutos


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------


def importar(nombre: str):
    """Importa un módulo de src con la raíz del repo en sys.path; si falla, el test falla."""
    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))
    try:
        return importlib.import_module(nombre)
    except Exception as error:  # ModuleNotFoundError, SyntaxError, etc.
        motivo = f"{type(error).__name__}: {error}"
    pytest.fail(f"No se pudo importar {nombre}: {motivo}", pytrace=False)


def atributo(modulo, nombre: str):
    if not hasattr(modulo, nombre):
        pytest.fail(f"{modulo.__name__} no define {nombre}.", pytrace=False)
    return getattr(modulo, nombre)


def es_entero(valor) -> bool:
    """int de Python que no sea bool (np.int64 no cuenta)."""
    return isinstance(valor, int) and not isinstance(valor, bool)


def es_float(valor) -> bool:
    """float de Python o np.float64 (que hereda de float); un array 0-d no cuenta."""
    return isinstance(valor, float)


def tipo_de(valor) -> str:
    if isinstance(valor, np.ndarray):
        return f"np.ndarray {valor.dtype} de forma {valor.shape}"
    return type(valor).__name__


def descripcion_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def resumen(problemas: list[str], maximo: int = 20) -> str:
    """Las primeras líneas de una lista de problemas, para los mensajes de error."""
    lineas = [f"  - {p}" for p in problemas[:maximo]]
    if len(problemas) > maximo:
        lineas.append(f"  ... y {len(problemas) - maximo} más")
    return "\n".join(lineas)


def exigir_sin_problemas(problemas: list[str], que: str) -> None:
    assert not problemas, f"{que}:\n{resumen(problemas)}"


def lanza(tipo: type[BaseException], funcion, *args, que: str, **kwargs) -> BaseException:
    """Llama a funcion(*args, **kwargs) y devuelve la excepción `tipo`; falla si no se lanza.

    Una excepción de otro tipo se deja pasar para que se vea su traceback.
    """
    try:
        funcion(*args, **kwargs)
    except tipo as error:
        return error
    pytest.fail(f"{que}: se esperaba {tipo.__name__} y no se lanzó ninguna excepción.", pytrace=False)


def max_dif(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return math.inf
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def dif_relativa(medido: float, esperado: float) -> float:
    return abs(float(medido) - float(esperado)) / max(abs(float(esperado)), 1e-300)


def iguales_con_nan(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def cerca_con_nan(a, b, tol: float) -> bool:
    """Mismos NaN y el resto a `tol` (absoluta)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    ok = ~np.isnan(a)
    return bool(np.all(np.abs(a[ok] - b[ok]) <= tol))


def emparejar(C: np.ndarray, C_ref: np.ndarray) -> float:
    """Menor diferencia máxima entre C y C_ref, sobre las permutaciones de las filas de C."""
    C, C_ref = np.asarray(C, dtype=np.float64), np.asarray(C_ref, dtype=np.float64)
    if C.shape != C_ref.shape:
        return math.inf
    return min(max_dif(C[list(p)], C_ref) for p in itertools.permutations(range(len(C))))


def obtener(cfg: dict, camino: str):
    """Valor de una clave con camino separado por puntos; falla si no está."""
    actual = cfg
    for parte in camino.split("."):
        if not isinstance(actual, dict) or parte not in actual:
            pytest.fail(f"config.yaml no tiene la clave obligatoria {camino!r}.", pytrace=False)
        actual = actual[parte]
    return actual


def leer_config_real() -> dict:
    with open(RAIZ / "config.yaml", encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def sin_clave(datos: dict, camino: str) -> dict:
    """Copia profunda de datos sin la clave `camino`."""
    copia = copy.deepcopy(datos)
    *padres, ultima = camino.split(".")
    actual = copia
    for parte in padres:
        actual = actual.get(parte) if isinstance(actual, dict) else None
    if not isinstance(actual, dict) or ultima not in actual:
        pytest.fail(
            f"El config.yaml real no tiene {camino!r}: no se puede armar la copia sin esa clave.",
            pytrace=False,
        )
    del actual[ultima]
    return copia


def escribir_yaml(ruta: Path, datos) -> Path:
    with open(ruta, "w", encoding="utf-8") as archivo:
        yaml.safe_dump(datos, archivo, allow_unicode=True, sort_keys=False)
    return ruta


def nombra(mensaje: str, camino: str) -> bool:
    """True si `camino` aparece completo en el mensaje (no como parte de otro camino)."""
    return re.search(r"(?<![\w.])" + re.escape(camino) + r"(?!\w)", mensaje) is not None


def cfg_con(cfg: dict, **cambios) -> dict:
    """Copia de cfg con valores de huellas.umbral cambiados (en memoria; config.yaml no se toca)."""
    nuevo = copy.deepcopy(cfg)
    nuevo["huellas"]["umbral"].update(cambios)
    return nuevo


# ---------------------------------------------------------------------------
# Oráculos (secciones 0 y 4), sin importar src/huellas/clasificacion.py
# ---------------------------------------------------------------------------


def oraculo_puntajes(X, centroides, dispersion, identidades) -> tuple[np.ndarray, np.ndarray]:
    """4.3: d_j(x) = ‖x − c_j‖ / r_j; puntaje = min_j d_j y predicción = identidades[argmin] (el primero)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    C = np.asarray(centroides, dtype=np.float64)
    r = np.asarray(dispersion, dtype=np.float64)
    D = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)) / r[None, :]
    j = np.argmin(D, axis=1)
    return D[np.arange(len(X)), j], np.asarray(identidades, dtype=np.int64)[j]


def oraculo_decidir(puntaje, prediccion, tau) -> np.ndarray:
    """4.3: la predicción donde puntaje ≤ τ y NO_AUTORIZADO donde no."""
    puntaje, prediccion = np.asarray(puntaje, dtype=np.float64), np.asarray(prediccion, dtype=np.int64)
    return np.where(puntaje <= tau, prediccion, NO_AUTORIZADO).astype(np.int64)


def oraculo_far_frr(puntaje, prediccion, verdad, tau) -> tuple[float, float]:
    """Sección 0: FAR = intrusos con puntaje ≤ τ; FRR = autorizadas que no quedan aceptadas con su identidad."""
    s = np.asarray(puntaje, dtype=np.float64)
    p, y = np.asarray(prediccion), np.asarray(verdad)
    autorizada = y != NO_AUTORIZADO
    acepta = s <= tau
    far = np.count_nonzero(acepta[~autorizada]) / np.count_nonzero(~autorizada)
    bien = acepta[autorizada] & (p[autorizada] == y[autorizada])
    frr = np.count_nonzero(~bien) / np.count_nonzero(autorizada)
    return float(far), float(frr)


def oraculo_intervalo(puntaje, prediccion, verdad, criterio: str, far_max_val: float) -> tuple[float, float]:
    """4.4: el intervalo [a, b) de τ; τ* es su mitad."""
    s = np.asarray(puntaje, dtype=np.float64)
    p, y = np.asarray(prediccion), np.asarray(verdad)
    autorizada = y != NO_AUTORIZADO
    if criterio == "far_max":
        intrusos = np.sort(s[~autorizada])
        m = math.floor(far_max_val * len(intrusos) + 1e-9)
        b = float(intrusos[m])
        abajo = s[autorizada & (p == y) & (s < b)]
        if abajo.size == 0:
            raise ValueError("oráculo: ninguna autorizada bien identificada por debajo de b")
        return float(abajo.max()), b
    if criterio == "eer":
        candidatos = np.concatenate([[-np.inf], np.unique(s)])   # E es constante en [c_i, c_{i+1})
        E = np.array([max(oraculo_far_frr(s, p, y, t)) for t in candidatos])
        idx = np.flatnonzero(E == E.min())
        assert np.all(np.diff(idx) == 1), "oráculo: el mínimo de E no es un intervalo"
        a = float(candidatos[idx[0]])
        b = float(candidatos[idx[-1] + 1]) if idx[-1] + 1 < len(candidatos) else math.inf
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("oráculo: el mínimo de E llega a ±∞")
        return a, b
    raise ValueError(f"oráculo: criterio {criterio!r}")


def oraculo_calibrar(puntaje, prediccion, verdad, criterio: str, far_max_val: float) -> float:
    a, b = oraculo_intervalo(puntaje, prediccion, verdad, criterio, far_max_val)
    return (a + b) / 2


def oraculo_matriz(verdad, decision, etiquetas) -> np.ndarray:
    posicion = {e: i for i, e in enumerate(etiquetas)}
    M = np.zeros((len(etiquetas), len(etiquetas)), dtype=np.int64)
    for v, d in zip(verdad, decision):
        M[posicion[v], posicion[d]] += 1
    return M


def oraculo_evaluar(decision, prediccion, verdad, k: int) -> dict:
    """4.5."""
    d, p, y = (np.asarray(a, dtype=np.int64) for a in (decision, prediccion, verdad))
    autorizada = y != NO_AUTORIZADO
    return {
        "identificacion": float(np.mean(p[autorizada] == y[autorizada])),
        "far": float(np.mean(d[~autorizada] != NO_AUTORIZADO)),
        "frr": float(np.mean(d[autorizada] != y[autorizada])),
        "matriz": oraculo_matriz(y.tolist(), d.tolist(), [*range(1, k + 1), NO_AUTORIZADO]),
    }


def oraculo_correspondencia(etiquetas, identidades, k: int) -> tuple[dict[int, int], float]:
    """4.2: identidad mayoritaria de cada cluster (la menor si empatan) y pureza."""
    etiquetas, identidades = np.asarray(etiquetas), np.asarray(identidades)
    mapa, aciertos = {}, 0
    for j in range(k):
        valores, cuentas = np.unique(identidades[etiquetas == j], return_counts=True)
        mapa[j] = int(valores[np.argmax(cuentas)])
        aciertos += int(cuentas.max())
    return mapa, aciertos / len(etiquetas)


def oraculo_dispersiones(X, etiquetas, centroides, piso: float) -> np.ndarray:
    """4.2: r_j = max(√(I_j / n_j), piso · r̄), con r̄ = √(Σ I_j / n)."""
    X, etiquetas = np.asarray(X, dtype=np.float64), np.asarray(etiquetas)
    C = np.asarray(centroides, dtype=np.float64)
    inercia = np.array([((X[etiquetas == j] - C[j]) ** 2).sum() for j in range(len(C))])
    filas = np.array([np.count_nonzero(etiquetas == j) for j in range(len(C))])
    r_media = math.sqrt(inercia.sum() / len(X))
    return np.maximum(np.sqrt(inercia / filas), piso * r_media)


def resultado_evaluar_problemas(medido, esperado: dict, k: int) -> list[str]:
    """Compara lo que devuelve evaluar (o evaluar_particion) con el oráculo de 4.5."""
    if not isinstance(medido, dict):
        return [f"devuelve {tipo_de(medido)}, no un dict"]
    problemas = []
    if set(medido) != set(esperado):
        problemas.append(f"claves {sorted(medido)}; se esperan exactamente {sorted(esperado)}")
    for clave in ("identificacion", "far", "frr"):
        if clave in medido:
            v = medido[clave]
            if not es_float(v):
                problemas.append(f"{clave} es {tipo_de(v)}; se espera float")
            elif abs(v - esperado[clave]) > TOL_IGUAL:
                problemas.append(f"{clave} = {v!r}; se espera {esperado[clave]!r}")
    if "matriz" in medido:
        M = medido["matriz"]
        forma = (k + 1, k + 1)
        if not isinstance(M, np.ndarray) or M.dtype != np.int64 or M.shape != forma:
            problemas.append(f"matriz es {tipo_de(M)}; se espera np.ndarray int64 de forma {forma}")
        elif not np.array_equal(M, esperado["matriz"]):
            problemas.append(f"matriz = {M.tolist()}; se espera {esperado['matriz'].tolist()} (orden [1, …, k, 0])")
    return problemas


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------


def grupos(n: int, d: int, k: int, disposicion: str, semilla: int) -> tuple[np.ndarray, np.ndarray]:
    """Grupos gaussianos con semilla: X (n, d) y el grupo de cada fila."""
    rng = np.random.default_rng(semilla)
    if disposicion == "separados":
        centros: list[np.ndarray] = []
        while len(centros) < k:
            c = rng.uniform(-60.0, 60.0, d)
            if all(np.linalg.norm(c - otro) >= 30.0 for otro in centros):
                centros.append(c)
        centros = np.array(centros)
    else:
        centros = rng.normal(0.0, 2.0, (k, d))
    grupo = rng.permutation(np.arange(n) % k)
    mezcla = np.eye(d) + 0.3 * rng.normal(0.0, 1.0, (d, d))
    X = centros[grupo] + rng.normal(0.0, 1.0, (n, d)) @ mezcla + 3.0
    return X, grupo


def semilla_de(n: int, d: int, k: int, disposicion: str) -> int:
    return 1000 * CASOS_KMEANS.index((n, d, k)) + 17 * DISPOSICIONES.index(disposicion) + 5


def init_del_test(X: np.ndarray, k: int, semilla: int) -> np.ndarray:
    """k filas distintas de X elegidas por el test (6.2)."""
    rng = np.random.default_rng(semilla + 7)
    return X[rng.choice(len(X), size=k, replace=False)].copy()


def tres_grupos_de_train(semilla: int = 7) -> tuple[np.ndarray, np.ndarray, dict]:
    """6.7: tres grupos separados en d = 6, con las identidades desordenadas respecto de los clusters.

    Devuelve X_train, identidades_train y, por identidad, (centro, escala de su dispersión).
    """
    rng = np.random.default_rng(semilla)
    d = 6
    centros = {3: np.zeros(d), 1: np.full(d, 40.0), 2: np.array([40.0, -40.0, 0.0, 0.0, 0.0, 0.0])}
    desvios = {3: 1.0, 1: 1.3, 2: 0.8}
    filas, ids = [], []
    for ident in (3, 1, 2):
        filas.append(centros[ident] + desvios[ident] * rng.normal(size=(12, d)))
        ids += [ident] * 12
    X, ids = np.vstack(filas), np.array(ids, dtype=np.int64)
    orden = rng.permutation(len(X))
    escala = {i: desvios[i] * math.sqrt(d) for i in centros}
    return X[orden], ids[orden], {i: (centros[i], escala[i]) for i in centros}


def direccion(rng: np.random.Generator, d: int) -> np.ndarray:
    u = rng.normal(size=d)
    return u / np.linalg.norm(u)


def val_separable(geometria: dict, semilla: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """6.7: 2 autorizadas por identidad cerca de su centro y 4 intrusos lejanos."""
    rng = np.random.default_rng(semilla)
    filas, verdad = [], []
    for ident in IDENTIDADES:
        centro, escala = geometria[ident]
        for t in (0.6, 1.1):
            filas.append(centro + t * escala * direccion(rng, len(centro)))
            verdad.append(ident)
    for ident, t in ((1, 6.0), (2, 7.0), (3, 8.0), (1, 9.0)):
        centro, escala = geometria[ident]
        filas.append(centro + t * escala * direccion(rng, len(centro)))
        verdad.append(NO_AUTORIZADO)
    return np.array(filas), np.array(verdad, dtype=np.int64)


def val_con_solape(geometria: dict, semilla: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """6.7: autorizadas a 0,5, 0,9 y 1,3 escalas de su centro e intrusos a 0,9 (de la identidad 2),
    1,6, 2,0 y 3,0: los tres criterios del test (far_max 0.01, eer y far_max 0.30) dan τ* distintos."""
    rng = np.random.default_rng(semilla)
    filas, verdad = [], []
    for ident in IDENTIDADES:
        centro, escala = geometria[ident]
        for t in (0.5, 0.9, 1.3):
            filas.append(centro + t * escala * direccion(rng, len(centro)))
            verdad.append(ident)
    for ident, t in ((2, 0.9), (1, 1.6), (3, 2.0), (1, 3.0)):
        centro, escala = geometria[ident]
        filas.append(centro + t * escala * direccion(rng, len(centro)))
        verdad.append(NO_AUTORIZADO)
    return np.array(filas), np.array(verdad, dtype=np.int64)


# ---------------------------------------------------------------------------
# Archivos (dataset chico y modelo)
# ---------------------------------------------------------------------------


def escribir_png(ruta: Path, imagen: np.ndarray) -> None:
    """Los mismos bytes que cv2.imwrite, sin problemas con tildes en la ruta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", imagen)
    assert ok, f"cv2.imencode no pudo codificar {ruta.name}"
    buffer.tofile(str(ruta))


def leer_png(ruta: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def escribir_metadatos(carpeta: Path, entradas: list[dict]) -> None:
    with open(carpeta / "metadatos.csv", "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(COLUMNAS_CHICO)
        for e in entradas:
            escritor.writerow([e[col] for col in COLUMNAS_CHICO])


def leer_npz(ruta: Path) -> dict:
    try:
        with np.load(ruta, allow_pickle=False) as datos:
            return {clave: datos[clave] for clave in datos.files}
    except Exception as error:
        pytest.fail(f"No se pudo leer {ruta} con np.load(allow_pickle=False): {descripcion_error(error)}", pytrace=False)


def npz_distintos(a: dict, b: dict) -> list[str]:
    """Claves que faltan en uno de los dos o cuyos arrays no son idénticos (NaN en el mismo lugar)."""
    distintos = sorted(set(a) ^ set(b))
    distintos += [clave for clave in sorted(set(a) & set(b)) if not iguales_con_nan(a[clave], b[clave])]
    return distintos


def escalar(valor) -> float:
    """El valor de un array de un solo elemento del npz."""
    arr = np.asarray(valor)
    if arr.size != 1:
        raise ValueError(f"se esperaba un solo valor; hay {arr.size}")
    return float(arr.reshape(-1)[0])


def receta_h1(h1, cfg: dict, entrada: dict, base):
    """Variante de una entrada del plan con la receta de la sección 4 de H1."""
    rng = np.random.default_rng([entrada["semilla"], entrada["variante"]])
    params = h1.sortear_parametros(rng, cfg)
    return h1.aplicar_variante(base, params, rng)


def elegir_dataset_chico(plan: list[dict]) -> list[dict]:
    """Sección 5: las entradas del dataset chico, en el orden del plan."""
    def entra(e: dict) -> bool:
        if int(e["intruso"]):
            return int(e["intruso"]) in INTRUSOS_CHICO.get(e["particion"], ())
        return int(e["variante"]) in VARIANTES_CHICO.get(e["particion"], ())

    return [e for e in plan if entra(e)]


def problemas_de_escalador(medido, esperado, tol: float) -> list[str]:
    """Dos Escalador de H3 con los mismos dicts (medias y desvios con NaN en el mismo lugar)."""
    problemas = []
    for campo in ("medias", "desvios", "columnas", "pesos"):
        dm, de = getattr(medido, campo, None), getattr(esperado, campo)
        if not isinstance(dm, dict) or set(dm) != set(BLOQUES_CONTRATO):
            problemas.append(f"escalador.{campo} es {tipo_de(dm)} con claves {sorted(dm) if isinstance(dm, dict) else '-'}")
            continue
        for b in BLOQUES_CONTRATO:
            if campo == "columnas":
                ok = np.asarray(dm[b]).dtype == np.bool_ and np.array_equal(dm[b], de[b])
            elif campo == "pesos":
                ok = es_float(dm[b]) and abs(dm[b] - de[b]) <= tol
            else:
                ok = cerca_con_nan(dm[b], de[b], tol)
            if not ok:
                problemas.append(f"escalador.{campo}[{b!r}] distinto ({tipo_de(dm[b])})")
    return problemas


def problemas_de_modelo(medido, esperado, tol: float, escalador_esperado=None) -> list[str]:
    """Compara un ModeloHuellas con otro (o con un SimpleNamespace con los mismos campos)."""
    problemas = []
    ident = getattr(medido, "identidades", None)
    if not isinstance(ident, np.ndarray) or ident.dtype != np.int64 or not np.array_equal(ident, esperado.identidades):
        problemas.append(f"identidades = {ident!r}; se espera int64 {np.asarray(esperado.identidades).tolist()}")
    for campo in ("centroides", "dispersion"):
        v = getattr(medido, campo, None)
        if not isinstance(v, np.ndarray) or v.dtype != np.float64:
            problemas.append(f"{campo} es {tipo_de(v)}; se espera np.ndarray float64")
        elif (dif := max_dif(v, getattr(esperado, campo))) > tol:
            problemas.append(f"{campo} difiere en {dif:.3e} (tolerancia {tol:g})")
    for campo in ("tau", "pureza", "inercia"):
        v = getattr(medido, campo, None)
        if not es_float(v):
            problemas.append(f"{campo} es {tipo_de(v)}; se espera float")
        elif abs(v - getattr(esperado, campo)) > tol:
            problemas.append(f"{campo} = {v!r}; se espera {getattr(esperado, campo)!r} (tolerancia {tol:g})")
    if escalador_esperado is not None:
        problemas += problemas_de_escalador(getattr(medido, "escalador", None), escalador_esperado, tol)
    return problemas


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> dict:
    return atributo(importar("src.config"), "cargar_config")()


@pytest.fixture(scope="module")
def KMeans():
    return atributo(importar("src.core_ml.kmeans"), "KMeans")


@pytest.fixture(scope="module")
def metricas():
    return importar("src.core_ml.metricas")


@pytest.fixture(scope="module")
def cl():
    return importar("src.huellas.clasificacion")


@pytest.fixture(scope="module")
def h1():
    generador = importar("src.huellas.generador")
    variantes = importar("src.huellas.variantes")
    dataset = importar("src.huellas.dataset")
    return SimpleNamespace(
        generar_base=atributo(generador, "generar_base"),
        sortear_parametros=atributo(variantes, "sortear_parametros"),
        aplicar_variante=atributo(variantes, "aplicar_variante"),
        plan_dataset=atributo(dataset, "plan_dataset"),
        generar_dataset=atributo(dataset, "generar_dataset"),
    )


@pytest.fixture(scope="module")
def h2():
    return SimpleNamespace(preprocesar=atributo(importar("src.huellas.preprocesamiento"), "preprocesar"))


@pytest.fixture(scope="module")
def h3():
    car = importar("src.huellas.caracteristicas")
    return SimpleNamespace(
        BLOQUES=atributo(car, "BLOQUES"),
        caracteristicas=atributo(car, "caracteristicas"),
        Escalador=atributo(car, "Escalador"),
        ajustar_escalador=atributo(car, "ajustar_escalador"),
        generar_caracteristicas=atributo(car, "generar_caracteristicas"),
    )


def modelo_a_mano(cl, centroides, dispersion, tau: float):
    """6.5 y 8: ModeloHuellas armado por el test, con escalador=None."""
    ModeloHuellas = atributo(cl, "ModeloHuellas")
    return ModeloHuellas(
        identidades=np.arange(1, len(centroides) + 1, dtype=np.int64),
        centroides=np.asarray(centroides, dtype=np.float64),
        dispersion=np.asarray(dispersion, dtype=np.float64),
        tau=float(tau), pureza=1.0, inercia=0.0, escalador=None,
    )


# ---------------------------------------------------------------------------
# 6.1 · Configuración
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("camino", list(CONFIG_H4))
def test_config_clave_de_h4_con_su_valor_y_su_tipo(cfg, camino):
    tipo, esperado = CONFIG_H4[camino]
    valor = obtener(cfg, camino)
    tipo_ok = {
        "int": es_entero,
        "float": lambda v: type(v) is float,
        "str": lambda v: isinstance(v, str),
    }[tipo](valor)
    assert tipo_ok and valor == esperado, (
        f"6.1 · {camino} = {valor!r} ({type(valor).__name__}); se espera {esperado!r} "
        f"({'int que no sea bool' if tipo == 'int' else tipo})."
    )


def test_cargar_config_exige_piso_dispersion(tmp_path):
    cargar_config = atributo(importar("src.config"), "cargar_config")
    ruta = escribir_yaml(tmp_path / "config.yaml", sin_clave(leer_config_real(), CLAVE_NUEVA))
    error = lanza(ValueError, cargar_config, ruta, que=f"cargar_config de una copia de config.yaml sin {CLAVE_NUEVA}")
    assert nombra(str(error), CLAVE_NUEVA), (
        f"6.1 · el ValueError no nombra el camino completo {CLAVE_NUEVA!r}: {error}"
    )


# ---------------------------------------------------------------------------
# 6.2 · K-means propio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("max_iter", "tol"), ITERACIONES_Y_TOL, ids=[f"max_iter={m}-tol={t:g}" for m, t in ITERACIONES_Y_TOL])
@pytest.mark.parametrize("disposicion", DISPOSICIONES)
@pytest.mark.parametrize(("n", "d", "k"), CASOS_KMEANS)
def test_kmeans_con_el_mismo_init_igual_a_sklearn(KMeans, n, d, k, disposicion, max_iter, tol):
    semilla = semilla_de(n, d, k, disposicion)
    X, _ = grupos(n, d, k, disposicion, semilla)
    init = init_del_test(X, k, semilla)
    propio = KMeans(k, n_init=1, max_iter=max_iter, tol=tol).ajustar(X.copy(), init=init.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = KMeansSklearn(
            n_clusters=k, init=init.copy(), n_init=1, max_iter=max_iter, tol=tol, algorithm="lloyd"
        ).fit(X.copy())
    problemas = []
    if (dif := max_dif(propio.centroides, sk.cluster_centers_)) > TOL_SKLEARN:
        problemas.append(f"centroides difieren en {dif:.3e} (tolerancia {TOL_SKLEARN:g})")
    if (rel := dif_relativa(propio.inercia, sk.inertia_)) > TOL_SKLEARN:
        problemas.append(f"inercia {propio.inercia!r} contra {sk.inertia_!r} (relativa {rel:.3e}, tolerancia {TOL_SKLEARN:g})")
    etiquetas = np.asarray(propio.etiquetas)
    if etiquetas.shape != sk.labels_.shape or not np.array_equal(etiquetas, sk.labels_):
        malas = np.flatnonzero(etiquetas != sk.labels_) if etiquetas.shape == sk.labels_.shape else []
        problemas.append(f"etiquetas distintas de labels_ ({len(malas)} filas; forma {etiquetas.shape})")
    exigir_sin_problemas(
        problemas, f"6.2 · KMeans({k}, max_iter={max_iter}, tol={tol:g}) con init dado en X {X.shape} ({disposicion}) contra scikit-learn"
    )


@pytest.mark.parametrize("semilla", [0, 2026])
@pytest.mark.parametrize("disposicion", DISPOSICIONES)
@pytest.mark.parametrize(("n", "d", "k"), CASOS_KMEANS)
def test_kmeans_pp_con_inercia_como_sklearn(KMeans, n, d, k, disposicion, semilla):
    X, _ = grupos(n, d, k, disposicion, semilla_de(n, d, k, disposicion))
    propio = KMeans(k, n_init=15, max_iter=300, tol=TOL_KMEANS, semilla=semilla).ajustar(X)
    sk = KMeansSklearn(n_clusters=k, n_init=15, random_state=0).fit(X)
    assert propio.inercia <= sk.inertia_ * (1 + MARGEN_KMEANSPP), (
        f"6.2 · k-means++ (semilla {semilla}) en X {X.shape} ({disposicion}): inercia {propio.inercia:.6f}, "
        f"más del {MARGEN_KMEANSPP:.0%} por encima de la de scikit-learn ({sk.inertia_:.6f})."
    )
    if disposicion == "separados":
        dif = emparejar(propio.centroides, sk.cluster_centers_)
        assert dif <= TOL_SKLEARN, (
            f"6.2 · k-means++ (semilla {semilla}) con grupos separados en X {X.shape}: los centroides emparejados "
            f"difieren de los de scikit-learn en {dif:.3e} (tolerancia {TOL_SKLEARN:g})."
        )


@pytest.mark.parametrize("disposicion", DISPOSICIONES)
@pytest.mark.parametrize(("n", "d", "k"), CASOS_KMEANS)
def test_kmeans_propiedades_de_etiquetas_inercia_y_centroides(KMeans, n, d, k, disposicion):
    X, _ = grupos(n, d, k, disposicion, semilla_de(n, d, k, disposicion))
    copia = X.copy()
    modelo = KMeans(k, n_init=15, max_iter=300, tol=TOL_KMEANS, semilla=2026)
    resultado = modelo.ajustar(X)
    problemas = []
    if resultado is not modelo:
        problemas.append("ajustar no devuelve el mismo objeto (self)")
    if not np.array_equal(X, copia):
        problemas.append("ajustar modificó X")
    C, etiquetas = modelo.centroides, modelo.etiquetas
    if not isinstance(C, np.ndarray) or C.dtype != np.float64 or C.shape != (k, d):
        problemas.append(f"centroides es {tipo_de(C)}; se espera float64 de forma {(k, d)}")
    if not isinstance(etiquetas, np.ndarray) or etiquetas.dtype != np.int64 or etiquetas.shape != (n,):
        problemas.append(f"etiquetas es {tipo_de(etiquetas)}; se espera int64 de forma {(n,)}")
    if not es_float(modelo.inercia):
        problemas.append(f"inercia es {tipo_de(modelo.inercia)}; se espera float")
    ic = modelo.inercia_cluster
    if not isinstance(ic, np.ndarray) or ic.dtype != np.float64 or ic.shape != (k,):
        problemas.append(f"inercia_cluster es {tipo_de(ic)}; se espera float64 de forma {(k,)}")
    if not es_entero(modelo.n_iter) or not 1 <= modelo.n_iter <= 300:
        problemas.append(f"n_iter = {modelo.n_iter!r} ({tipo_de(modelo.n_iter)}); se espera un int en [1, max_iter = 300]")
    if problemas:
        exigir_sin_problemas(problemas, f"6.2 · KMeans({k}) en X {X.shape} ({disposicion})")

    D = modelo.distancias(X)
    if not isinstance(D, np.ndarray) or D.shape != (n, k) or (dif := max_dif(D, cdist(X, C))) > TOL_PROPIEDADES:
        problemas.append(f"distancias(X) es {tipo_de(D)}; se espera la euclídea a cada centroide, de forma {(n, k)}")
    elif not np.array_equal(etiquetas, np.argmin(D, axis=1)):
        problemas.append(f"etiquetas no es el argmin de distancias(X) en {int(np.sum(etiquetas != np.argmin(D, axis=1)))} filas")
    d2 = ((X - C[etiquetas]) ** 2).sum(axis=1)
    if (rel := dif_relativa(modelo.inercia, d2.sum())) > TOL_PROPIEDADES:
        problemas.append(f"inercia = {modelo.inercia!r}; la suma de las distancias² es {d2.sum()!r} (relativa {rel:.3e})")
    por_cluster = np.array([d2[etiquetas == j].sum() for j in range(k)])
    malos = [j for j in range(k) if dif_relativa(ic[j], por_cluster[j]) > TOL_PROPIEDADES]
    if malos:
        problemas.append(f"inercia_cluster = {ic.tolist()}; las sumas por cluster son {por_cluster.tolist()}")
    if disposicion == "separados":
        medias = np.array([X[etiquetas == j].mean(axis=0) for j in range(k)])
        if (dif := max_dif(C, medias)) > TOL_PROPIEDADES:
            problemas.append(f"con grupos separados, los centroides difieren de las medias de sus filas en {dif:.3e}")
    p = modelo.predecir(X)
    if not isinstance(p, np.ndarray) or p.dtype != np.int64 or not np.array_equal(p, np.argmin(D, axis=1)):
        problemas.append(f"predecir(X) es {tipo_de(p)}; se espera int64 igual al argmin de distancias(X)")
    for i in (0, n // 2, n - 1):
        pi, di = modelo.predecir(X[i]), modelo.distancias(X[i])
        if not es_entero(pi) or pi != int(np.argmin(D[i])):
            problemas.append(f"predecir(X[{i}]) = {pi!r} ({tipo_de(pi)}); se espera el int {int(np.argmin(D[i]))}")
        if not isinstance(di, np.ndarray) or di.shape != (k,) or max_dif(di, D[i]) > TOL_IGUAL:
            problemas.append(f"distancias(X[{i}]) es {tipo_de(di)}; se espera la fila {i} de distancias(X), de forma {(k,)}")
    exigir_sin_problemas(problemas, f"6.2 · propiedades de KMeans({k}) en X {X.shape} ({disposicion})")


@pytest.mark.parametrize("disposicion", DISPOSICIONES)
def test_kmeans_misma_semilla_mismo_resultado_y_n_init_no_empeora(KMeans, disposicion):
    n, d, k = CASOS_KMEANS[0]
    X, _ = grupos(n, d, k, disposicion, semilla_de(n, d, k, disposicion))
    a = KMeans(k, n_init=15, semilla=7).ajustar(X)
    b = KMeans(k, n_init=15, semilla=7).ajustar(X)
    problemas = []
    for campo in ("centroides", "inercia_cluster"):
        if (dif := max_dif(getattr(a, campo), getattr(b, campo))) > TOL_IGUAL:
            problemas.append(f"{campo} difiere en {dif:.3e} entre dos ajustes con la semilla 7")
    if abs(a.inercia - b.inercia) > TOL_IGUAL:
        problemas.append(f"inercia {a.inercia!r} y {b.inercia!r} con la semilla 7")
    if not np.array_equal(a.etiquetas, b.etiquetas):
        problemas.append("etiquetas distintas entre dos ajustes con la semilla 7")
    uno = KMeans(k, n_init=1, semilla=7).ajustar(X)
    if a.inercia > uno.inercia * (1 + TOL_IGUAL):
        problemas.append(f"con n_init = 15 la inercia ({a.inercia!r}) es mayor que con n_init = 1 ({uno.inercia!r})")
    exigir_sin_problemas(problemas, f"6.2 · semilla del K-means ({disposicion})")


@pytest.mark.parametrize("valor", [0, -1, 2.5, True], ids=["0", "-1", "2.5", "True"])
@pytest.mark.parametrize("parametro", ["k", "n_init", "max_iter"])
def test_kmeans_constructor_rechaza_enteros_invalidos(KMeans, parametro, valor):
    argumentos = {"k": 3, "n_init": 15, "max_iter": 300, "tol": TOL_KMEANS}
    argumentos[parametro] = valor
    lanza(ValueError, KMeans, que=f"6.2 · KMeans con {parametro} = {valor!r}", **argumentos)


def test_kmeans_constructor_rechaza_tol_negativa(KMeans):
    lanza(ValueError, KMeans, 3, que="6.2 · KMeans con tol = -1e-4", tol=-1e-4)


def _x_base() -> np.ndarray:
    return grupos(60, 4, 3, "separados", 99)[0]


CASOS_X_INVALIDO = {
    "1 dimensión": lambda X: X[:, 0].copy(),
    "3 dimensiones": lambda X: X[None, :, :].copy(),
    "con NaN": lambda X: np.where(np.arange(X.size).reshape(X.shape) == 5, np.nan, X),
    "con Inf": lambda X: np.where(np.arange(X.size).reshape(X.shape) == 5, np.inf, X),
    "n < k": lambda X: X[:2].copy(),
    "menos de k filas distintas": lambda X: np.vstack([X[[0]]] * 5 + [X[[1]]] * 5),
}


@pytest.mark.parametrize("caso", list(CASOS_X_INVALIDO))
def test_kmeans_ajustar_rechaza_x_invalido(KMeans, caso):
    X = CASOS_X_INVALIDO[caso](_x_base())
    lanza(ValueError, KMeans(3, n_init=2, semilla=0).ajustar, X, que=f"6.2 · ajustar con X {caso} (forma {X.shape})")


CASOS_INIT_INVALIDO = {
    "k + 1 filas": lambda X: X[:4].copy(),
    "d + 1 columnas": lambda X: np.hstack([X[:3], np.ones((3, 1))]),
    "con NaN": lambda X: np.where(np.arange(12).reshape(3, 4) == 2, np.nan, X[:3]),
}


@pytest.mark.parametrize("caso", list(CASOS_INIT_INVALIDO))
def test_kmeans_ajustar_rechaza_init_invalido(KMeans, caso):
    X = _x_base()
    init = CASOS_INIT_INVALIDO[caso](X)
    lanza(ValueError, KMeans(3, n_init=1).ajustar, X, init=init, que=f"6.2 · ajustar con init {caso} (forma {init.shape})")


@pytest.mark.parametrize("metodo", ["predecir", "distancias"])
def test_kmeans_predecir_y_distancias_exigen_ajustar_y_el_mismo_d(KMeans, metodo):
    X = _x_base()
    modelo = KMeans(3, n_init=2, semilla=0)
    lanza(RuntimeError, getattr(modelo, metodo), X, que=f"6.2 · {metodo} antes de ajustar")
    modelo.ajustar(X)
    lanza(ValueError, getattr(modelo, metodo), X[:, :3], que=f"6.2 · {metodo} con d = 3 (se ajustó con d = 4)")
    lanza(ValueError, getattr(modelo, metodo), X[0, :3], que=f"6.2 · {metodo} de un vector con d = 3")


# ---------------------------------------------------------------------------
# 6.3 · Matriz de confusión
# ---------------------------------------------------------------------------


def _datos_matriz(valores: list, etiquetas: list, n: int, semilla: int) -> tuple[list, list, list]:
    rng = np.random.default_rng(semilla)
    verdad = [valores[i] for i in rng.integers(0, len(valores), n)]
    prediccion = [valores[i] for i in rng.integers(0, len(valores), n)]
    return verdad, prediccion, etiquetas


CASOS_MATRIZ = {
    "enteros [1, 2, 3, 0]": _datos_matriz([1, 2, 3, 0], [1, 2, 3, 0], 60, 1),
    "enteros, una clase sin filas": _datos_matriz([1, 3, 0], [1, 2, 3, 0], 40, 2),
    "enteros en otro orden": _datos_matriz([1, 2, 3, 0], [0, 3, 1, 2], 50, 3),
    "textos": _datos_matriz(["reconocer", "identificar", "salir", "otro"], ["reconocer", "identificar", "salir", "otro"], 45, 4),
    "textos en otro orden, una clase sin filas": _datos_matriz(["salir", "otro", "reconocer"], ["otro", "salir", "identificar", "reconocer"], 30, 5),
}


@pytest.mark.parametrize("como_array", [False, True], ids=["listas", "arrays"])
@pytest.mark.parametrize("caso", list(CASOS_MATRIZ))
def test_matriz_confusion_igual_a_sklearn(metricas, caso, como_array):
    matriz_confusion = atributo(metricas, "matriz_confusion")
    verdad, prediccion, etiquetas = CASOS_MATRIZ[caso]
    if como_array:
        verdad, prediccion = np.array(verdad), np.array(prediccion)
    M = matriz_confusion(verdad, prediccion, etiquetas)
    esperada = confusion_matrix(verdad, prediccion, labels=etiquetas)
    forma = (len(etiquetas), len(etiquetas))
    assert isinstance(M, np.ndarray) and M.dtype == np.int64 and M.shape == forma, (
        f"6.3 · matriz_confusion ({caso}) devuelve {tipo_de(M)}; se espera np.ndarray int64 de forma {forma}."
    )
    assert np.array_equal(M, esperada) and int(M.sum()) == len(verdad), (
        f"6.3 · matriz_confusion ({caso}, etiquetas {etiquetas}) = {M.tolist()}; "
        f"sklearn.metrics.confusion_matrix da {esperada.tolist()} (suma {len(verdad)})."
    )


CASOS_MATRIZ_INVALIDA = {
    "largos distintos": ([1, 2, 0], [1, 2], [1, 2, 0]),
    "etiquetas vacía": ([1, 2, 0], [1, 2, 0], []),
    "etiquetas repetidas": ([1, 2, 0], [1, 2, 0], [1, 2, 2, 0]),
    "valor de verdad fuera de etiquetas": ([1, 4, 0], [1, 2, 0], [1, 2, 0]),
    "valor de predicción fuera de etiquetas": ([1, 2, 0], [1, 2, 5], [1, 2, 0]),
}


@pytest.mark.parametrize("caso", list(CASOS_MATRIZ_INVALIDA))
def test_matriz_confusion_rechaza_entradas_invalidas(metricas, caso):
    matriz_confusion = atributo(metricas, "matriz_confusion")
    verdad, prediccion, etiquetas = CASOS_MATRIZ_INVALIDA[caso]
    lanza(ValueError, matriz_confusion, verdad, prediccion, etiquetas, que=f"6.3 · matriz_confusion con {caso}")


@pytest.mark.parametrize("ruta", ARCHIVOS_SIN_SKLEARN)
def test_archivo_no_usa_scikit_learn(ruta):
    archivo = RAIZ / ruta
    assert archivo.is_file(), f"6.3 · no existe {ruta}."
    texto = archivo.read_text(encoding="utf-8")
    assert "sklearn" not in texto, f"6.3 · {ruta} contiene el texto 'sklearn' (no puede importar scikit-learn)."


# ---------------------------------------------------------------------------
# 6.4 · Correspondencia y dispersión
# ---------------------------------------------------------------------------

# (etiquetas de cluster, identidades, mapa esperado, pureza esperada)
CASOS_CORRESPONDENCIA = {
    "grupos separados": ([0] * 5 + [1] * 4 + [2] * 6, [3] * 5 + [1] * 4 + [2] * 6, {0: 3, 1: 1, 2: 2}, 1.0),
    "un cluster con dos identidades": (
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2], [3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 3], {0: 3, 1: 1, 2: 2}, 10 / 12,
    ),
    "empate: la identidad menor": ([0, 0, 0, 0, 1, 1, 1, 2, 2], [3, 2, 3, 2, 3, 3, 3, 1, 1], {0: 2, 1: 3, 2: 1}, 7 / 9),
}


@pytest.mark.parametrize("como_array", [True, False], ids=["arrays", "listas"])
@pytest.mark.parametrize("caso", list(CASOS_CORRESPONDENCIA))
def test_correspondencia_mapa_y_pureza(cl, caso, como_array):
    correspondencia = atributo(cl, "correspondencia")
    etiquetas, identidades, mapa_esperado, pureza_esperada = CASOS_CORRESPONDENCIA[caso]
    orden = np.random.default_rng(3).permutation(len(etiquetas))     # el orden de las filas no importa
    etiquetas = np.array(etiquetas, dtype=np.int64)[orden]
    identidades = np.array(identidades, dtype=np.int64)[orden]
    assert oraculo_correspondencia(etiquetas, identidades, K) == (mapa_esperado, pureza_esperada)
    if not como_array:
        etiquetas, identidades = etiquetas.tolist(), identidades.tolist()
    mapa, pureza = correspondencia(etiquetas, identidades, K)
    problemas = []
    if not isinstance(mapa, dict) or mapa != mapa_esperado:
        problemas.append(f"mapa = {mapa!r}; se espera {mapa_esperado}")
    elif not all(es_entero(j) and es_entero(i) for j, i in mapa.items()):
        problemas.append(f"mapa tiene claves o valores que no son int: {[(tipo_de(j), tipo_de(i)) for j, i in mapa.items()]}")
    if not es_float(pureza) or abs(pureza - pureza_esperada) > TOL_IGUAL:
        problemas.append(f"pureza = {pureza!r} ({tipo_de(pureza)}); se espera el float {pureza_esperada!r}")
    exigir_sin_problemas(problemas, f"6.4 · correspondencia ({caso})")


CASOS_CORRESPONDENCIA_INVALIDA = {
    "dos clusters a la misma identidad": ([0, 0, 1, 1, 2, 2], [1, 1, 1, 1, 2, 2]),
    "un cluster sin filas": ([0, 0, 1, 1, 0, 1], [1, 1, 2, 2, 1, 2]),
    "largos distintos": ([0, 0, 1, 1, 2, 2], [1, 1, 2, 2, 3]),
}


@pytest.mark.parametrize("caso", list(CASOS_CORRESPONDENCIA_INVALIDA))
def test_correspondencia_rechaza_entradas_invalidas(cl, caso):
    correspondencia = atributo(cl, "correspondencia")
    etiquetas, identidades = CASOS_CORRESPONDENCIA_INVALIDA[caso]
    lanza(ValueError, correspondencia, np.array(etiquetas), np.array(identidades), K, que=f"6.4 · correspondencia con {caso}")


def _grupos_dispersion(semilla: int = 21) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tres grupos de 8, 10 y 12 filas en d = 5 con dispersiones parecidas, y las medias como centroides."""
    rng = np.random.default_rng(semilla)
    filas, etiquetas = [], []
    for j, (n, s) in enumerate(((8, 1.0), (10, 1.4), (12, 0.8))):
        filas.append(20.0 * j + s * rng.normal(size=(n, 5)))
        etiquetas += [j] * n
    X, etiquetas = np.vstack(filas), np.array(etiquetas, dtype=np.int64)
    centroides = np.array([X[etiquetas == j].mean(axis=0) for j in range(3)])
    return X, etiquetas, centroides


def test_dispersiones_formula_sin_piso(cl):
    dispersiones = atributo(cl, "dispersiones")
    X, etiquetas, C = _grupos_dispersion()
    sin_piso = np.sqrt([((X[etiquetas == j] - C[j]) ** 2).sum() / np.count_nonzero(etiquetas == j) for j in range(3)])
    esperado = oraculo_dispersiones(X, etiquetas, C, 0.5)
    assert np.array_equal(esperado, sin_piso), "Control del test: en estos grupos el piso no tendría que actuar."
    r = dispersiones(X, etiquetas, C, 0.5)
    assert isinstance(r, np.ndarray) and r.dtype == np.float64 and r.shape == (3,), (
        f"6.4 · dispersiones devuelve {tipo_de(r)}; se espera np.ndarray float64 de forma (3,)."
    )
    assert max_dif(r, sin_piso) <= TOL_DISPERSION, (
        f"6.4 · dispersiones = {r.tolist()}; se espera √(I_j / n_j) = {sin_piso.tolist()} (el piso no actúa)."
    )
    # I_j se mide contra los centroides que se reciben, no contra las medias de las filas
    corridos = C + np.array([[0.5, -0.3, 0.2, 0.0, 0.1], [0.0, 0.9, -0.4, 0.3, 0.0], [-0.2, 0.0, 0.0, 0.6, 0.4]])
    esperado = oraculo_dispersiones(X, etiquetas, corridos, 0.5)
    r = dispersiones(X, etiquetas, corridos, 0.5)
    assert max_dif(r, esperado) <= TOL_DISPERSION, (
        f"6.4 · con centroides que no son las medias, dispersiones = {np.asarray(r).tolist()}; "
        f"la fórmula de 4.2 da {esperado.tolist()}."
    )


@pytest.mark.parametrize("caso", ["una sola fila", "filas iguales"])
def test_dispersiones_cluster_sin_dispersion_queda_en_el_piso(cl, caso):
    dispersiones = atributo(cl, "dispersiones")
    X, etiquetas, C = _grupos_dispersion()
    if caso == "una sola fila":
        fila = np.array([[100.0, 0.0, -50.0, 25.0, 3.0]])
    else:
        fila = np.repeat(np.array([[100.0, 0.0, -50.0, 25.0, 3.0]]), 4, axis=0)
    X = np.vstack([X, fila])
    etiquetas = np.concatenate([etiquetas, np.full(len(fila), 3, dtype=np.int64)])
    C = np.vstack([C, fila[:1]])       # el centroide es exactamente la fila: I_3 = 0
    piso = 0.5
    r_media = math.sqrt(sum(((X[etiquetas == j] - C[j]) ** 2).sum() for j in range(4)) / len(X))
    esperado = oraculo_dispersiones(X, etiquetas, C, piso)
    r = dispersiones(X, etiquetas, C, piso)
    assert max_dif(r, esperado) <= TOL_DISPERSION and abs(r[3] - piso * r_media) <= TOL_DISPERSION, (
        f"6.4 · con un cluster de {caso}: dispersiones = {np.asarray(r).tolist()}; se espera {esperado.tolist()} "
        f"(el cluster 3 en piso · r̄ = {piso * r_media!r})."
    )


def test_dispersiones_rechaza_sin_dispersion_cluster_vacio_y_piso_invalido(cl):
    dispersiones = atributo(cl, "dispersiones")
    X, etiquetas, C = _grupos_dispersion()
    exactos = np.array([[1.0, 2.0], [1.0, 2.0], [-3.0, 0.5], [-3.0, 0.5], [8.0, 8.0]])
    lanza(ValueError, dispersiones, exactos, np.array([0, 0, 1, 1, 2]), np.array([[1.0, 2.0], [-3.0, 0.5], [8.0, 8.0]]), 0.5,
          que="6.4 · dispersiones con todas las filas iguales a su centroide (r̄ = 0)")
    lanza(ValueError, dispersiones, X, np.where(etiquetas == 2, 1, etiquetas), C, 0.5,
          que="6.4 · dispersiones con un cluster sin filas")
    for piso in (0.0, 1.5):
        lanza(ValueError, dispersiones, X, etiquetas, C, piso, que=f"6.4 · dispersiones con piso = {piso}")


# ---------------------------------------------------------------------------
# 6.5 · Puntaje, decisión y umbral
# ---------------------------------------------------------------------------


def test_decidir_logica_del_plan(cl):
    decidir = atributo(cl, "decidir")
    no_autorizado = atributo(cl, "NO_AUTORIZADO")
    assert es_entero(no_autorizado) and no_autorizado == 0, f"6.5 · NO_AUTORIZADO = {no_autorizado!r}; se espera el int 0."
    puntaje = np.array([0.5, 1.0, 1.5])
    prediccion = np.array([2, 3, 1], dtype=np.int64)
    copia = prediccion.copy()
    d = decidir(puntaje, prediccion, 1.0)
    assert isinstance(d, np.ndarray) and d.dtype == np.int64 and d.shape == (3,), (
        f"6.5 · decidir devuelve {tipo_de(d)}; se espera np.ndarray int64 de forma (3,)."
    )
    assert d.tolist() == [2, 3, 0] and np.array_equal(prediccion, copia), (
        f"6.5 · con τ = 1.0, puntajes [0.5, 1.0, 1.5] y predicciones [2, 3, 1], decidir da {d.tolist()}; "
        "se espera [2, 3, 0] (con el puntaje igual a τ se acepta) sin cambiar la predicción."
    )
    rng = np.random.default_rng(8)
    puntaje = rng.uniform(0.0, 3.0, 200)
    prediccion = rng.integers(1, 4, 200).astype(np.int64)
    tau = float(puntaje[17])
    d = decidir(puntaje, prediccion, tau)
    assert np.array_equal(d, oraculo_decidir(puntaje, prediccion, tau)), (
        "6.5 · decidir de 200 puntajes al azar no es la predicción donde puntaje ≤ τ y 0 donde no."
    )


def _modelos_a_mano():
    """(centroides, dispersiones, X) del test. El primero tiene una fila que la distancia normalizada
    lleva a la identidad 2 aunque el centroide más cercano en euclídea es el 1."""
    C = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 10.0]])
    r = np.array([1.0, 4.0, 2.0])
    X = np.array([[1.4, 0.0], [0.2, -0.1], [2.9, 0.4], [0.0, 8.5], [-3.0, 4.0], [10.0, 10.0], [1.0, 1.0]])
    rng = np.random.default_rng(9)
    C2 = rng.normal(0.0, 3.0, (3, 5))
    r2 = np.array([0.7, 1.9, 1.2])
    X2 = rng.normal(0.0, 3.0, (40, 5))
    return {"2D a mano": (C, r, X), "5D al azar": (C2, r2, X2)}


@pytest.mark.parametrize("caso", ["2D a mano", "5D al azar"])
def test_puntajes_de_un_modelo_armado_a_mano(cl, caso):
    C, r, X = _modelos_a_mano()[caso]
    modelo = modelo_a_mano(cl, C, r, 1.0)
    s_esp, p_esp = oraculo_puntajes(X, C, r, [1, 2, 3])
    if caso == "2D a mano":
        euclidea = int(np.argmin(np.linalg.norm(X[0] - C, axis=1))) + 1
        assert euclidea == 1 and p_esp[0] == 2, "Control del test: la fila 0 tendría que cambiar de centroide."
    s, p = modelo.puntajes(X)
    problemas = []
    if not isinstance(s, np.ndarray) or s.dtype != np.float64 or s.shape != (len(X),):
        problemas.append(f"puntaje es {tipo_de(s)}; se espera float64 de forma ({len(X)},)")
    elif (dif := max_dif(s, s_esp)) > TOL_PUNTAJE:
        problemas.append(f"puntaje difiere de min_j ‖x − c_j‖ / r_j en {dif:.3e}: {s.tolist()} contra {s_esp.tolist()}")
    if not isinstance(p, np.ndarray) or p.dtype != np.int64 or not np.array_equal(p, p_esp):
        problemas.append(f"prediccion = {p!r}; se espera int64 {p_esp.tolist()} (identidades[argmin])")
    for i in range(min(len(X), 7)):
        si, pi = modelo.puntajes(X[i])
        if not es_float(si) or abs(si - s_esp[i]) > TOL_PUNTAJE:
            problemas.append(f"puntajes(X[{i}]) da puntaje {si!r} ({tipo_de(si)}); se espera el float {s_esp[i]!r}")
        if not es_entero(pi) or pi != p_esp[i]:
            problemas.append(f"puntajes(X[{i}]) da predicción {pi!r} ({tipo_de(pi)}); se espera el int {int(p_esp[i])}")
    exigir_sin_problemas(problemas, f"6.5 · ModeloHuellas.puntajes ({caso})")


def test_modelo_decidir_es_decidir_de_sus_puntajes(cl):
    C, r, X = _modelos_a_mano()["2D a mano"]
    tau = 0.5
    modelo = modelo_a_mano(cl, C, r, tau)
    esperado = oraculo_decidir(*oraculo_puntajes(X, C, r, [1, 2, 3]), tau)
    assert 0 < np.count_nonzero(esperado) < len(X), "Control del test: tendría que haber aceptadas y rechazadas."
    d = modelo.decidir(X)
    problemas = []
    if not isinstance(d, np.ndarray) or d.dtype != np.int64 or not np.array_equal(d, esperado):
        problemas.append(f"decidir(X) = {d!r}; se espera int64 {esperado.tolist()}")
    for i in range(len(X)):
        di = modelo.decidir(X[i])
        if not es_entero(di) or di != esperado[i]:
            problemas.append(f"decidir(X[{i}]) = {di!r} ({tipo_de(di)}); se espera el int {int(esperado[i])}")
    exigir_sin_problemas(problemas, f"6.5 · ModeloHuellas.decidir con τ = {tau}")


def _grilla_de_tau(puntaje) -> list[float]:
    s = np.unique(np.asarray(puntaje, dtype=np.float64))
    medios = (s[:-1] + s[1:]) / 2
    return [float(s[0] - 1.0), *map(float, s), *map(float, medios), float(s[-1] + 1.0)]


@pytest.mark.parametrize("ejemplo", list(EJEMPLOS))
def test_far_frr_igual_al_oraculo_en_cada_tau(cl, ejemplo):
    far_frr = atributo(cl, "far_frr")
    e = EJEMPLOS[ejemplo]
    s, p, y = np.array(e.puntaje), np.array(e.prediccion), np.array(e.verdad)
    problemas = []
    for tau in _grilla_de_tau(s):
        far, frr = far_frr(s, p, y, tau)
        far_e, frr_e = oraculo_far_frr(s, p, y, tau)
        if not (es_float(far) and es_float(frr)):
            problemas.append(f"τ = {tau}: devuelve ({tipo_de(far)}, {tipo_de(frr)}); se espera (float, float)")
        elif abs(far - far_e) > TOL_IGUAL or abs(frr - frr_e) > TOL_IGUAL:
            problemas.append(f"τ = {tau}: (FAR, FRR) = ({far!r}, {frr!r}); se espera ({far_e!r}, {frr_e!r})")
    exigir_sin_problemas(problemas, f"6.5 · far_frr en el ejemplo {ejemplo} de la sección 8")


@pytest.mark.parametrize(("ejemplo", "tau", "far_esp", "frr_esp"), FAR_FRR_SECCION_8)
def test_far_frr_valores_de_la_seccion_8(cl, ejemplo, tau, far_esp, frr_esp):
    far_frr = atributo(cl, "far_frr")
    e = EJEMPLOS[ejemplo]
    assert oraculo_far_frr(e.puntaje, e.prediccion, e.verdad, tau) == (far_esp, frr_esp), "Control del oráculo."
    far, frr = far_frr(np.array(e.puntaje), np.array(e.prediccion), np.array(e.verdad), tau)
    assert abs(far - far_esp) <= TOL_IGUAL and abs(frr - frr_esp) <= TOL_IGUAL, (
        f"6.5 · far_frr del ejemplo {ejemplo} en τ = {tau}: ({far!r}, {frr!r}); se espera ({far_esp}, {frr_esp}). "
        "Una autorizada mal identificada cuenta como error de FRR aunque su puntaje sea bajo."
    )


@pytest.mark.parametrize(("ejemplo", "criterio", "far_max_val", "tau_esp"), CASOS_TAU)
def test_calibrar_umbral_ejemplos_de_la_seccion_8(cl, ejemplo, criterio, far_max_val, tau_esp):
    calibrar_umbral = atributo(cl, "calibrar_umbral")
    e = EJEMPLOS[ejemplo]
    assert abs(oraculo_calibrar(e.puntaje, e.prediccion, e.verdad, criterio, far_max_val) - tau_esp) <= TOL_IGUAL, (
        "Control del oráculo: no da el τ* de la sección 8."
    )
    tau = calibrar_umbral(np.array(e.puntaje), np.array(e.prediccion), np.array(e.verdad), criterio, far_max_val)
    a, b = oraculo_intervalo(e.puntaje, e.prediccion, e.verdad, criterio, far_max_val)
    assert es_float(tau) and abs(tau - tau_esp) <= TOL_IGUAL, (
        f"6.5 · calibrar_umbral del ejemplo {ejemplo} con {criterio} ({far_max_val}) = {tau!r} ({tipo_de(tau)}); "
        f"se espera el float {tau_esp} (la mitad de [{a}, {b}))."
    )


@pytest.mark.parametrize("criterio", ["far_max", "eer"])
def test_calibrar_umbral_caso_separable_en_el_punto_medio(cl, criterio):
    calibrar_umbral = atributo(cl, "calibrar_umbral")
    rng = np.random.default_rng(31)
    verdad = np.array([1, 2, 3] * 6 + [0] * 20)
    prediccion = np.where(verdad > 0, verdad, rng.integers(1, 4, verdad.size))
    puntaje = np.where(verdad > 0, rng.uniform(0.2, 1.4, verdad.size), rng.uniform(1.7, 5.0, verdad.size))
    esperado = (puntaje[verdad > 0].max() + puntaje[verdad == 0].min()) / 2
    far_max_val = 0.01                      # < 1 / n_i = 0.05
    tau = calibrar_umbral(puntaje, prediccion, verdad, criterio, far_max_val)
    assert abs(tau - esperado) <= TOL_IGUAL, (
        f"6.5 · caso separable con {criterio}: τ* = {tau!r}; se espera el punto medio entre la autorizada más lejana "
        f"({puntaje[verdad > 0].max()!r}) y el intruso más cercano ({puntaje[verdad == 0].min()!r}): {esperado!r}."
    )


@pytest.mark.parametrize(("ejemplo", "criterio", "far_max_val", "tau_esp"), CASOS_TAU)
def test_calibrar_umbral_no_depende_del_orden_de_las_filas(cl, ejemplo, criterio, far_max_val, tau_esp):
    calibrar_umbral = atributo(cl, "calibrar_umbral")
    e = EJEMPLOS[ejemplo]
    rng = np.random.default_rng(5)
    resultados = []
    for _ in range(4):
        orden = rng.permutation(len(e.verdad))
        resultados.append(calibrar_umbral(np.array(e.puntaje)[orden], np.array(e.prediccion)[orden],
                                          np.array(e.verdad)[orden], criterio, far_max_val))
    assert all(abs(t - tau_esp) <= TOL_IGUAL for t in resultados), (
        f"6.5 · calibrar_umbral del ejemplo {ejemplo} ({criterio}, {far_max_val}) con las filas reordenadas da "
        f"{resultados}; se espera siempre {tau_esp}."
    )


def _invalido_calibrar(caso: str):
    e = EJEMPLOS["con_solape"]
    s, p, y = np.array(e.puntaje), np.array(e.prediccion), np.array(e.verdad)
    autorizada = y > 0
    casos = {
        "sin intrusos": (s[autorizada], p[autorizada], y[autorizada], "far_max", 0.01),
        "sin autorizadas": (s[~autorizada], p[~autorizada], y[~autorizada], "far_max", 0.01),
        "largos distintos": (s[:-1], p, y, "far_max", 0.01),
        "puntaje con NaN": (np.where(np.arange(8) == 2, np.nan, s), p, y, "far_max", 0.01),
        "criterio 'otro'": (s, p, y, "otro", 0.01),
        "far_max con far_max_val 0": (s, p, y, "far_max", 0.0),
        "far_max con far_max_val 1": (s, p, y, "far_max", 1.0),
        "eer con far_max_val 0": (s, p, y, "eer", 0.0),
        "eer con far_max_val 1": (s, p, y, "eer", 1.0),
        "far_max sin autorizadas bien identificadas debajo de b": (
            np.array([2.0, 3.0, 0.4, 1.0, 4.0]), np.array([1, 3, 1, 2, 3]), np.array([1, 2, 0, 0, 0]), "far_max", 0.01,
        ),
    }
    return casos[caso]


@pytest.mark.parametrize("caso", [
    "sin intrusos", "sin autorizadas", "largos distintos", "puntaje con NaN", "criterio 'otro'",
    "far_max con far_max_val 0", "far_max con far_max_val 1", "eer con far_max_val 0", "eer con far_max_val 1",
    "far_max sin autorizadas bien identificadas debajo de b",
])
def test_calibrar_umbral_rechaza_entradas_invalidas(cl, caso):
    calibrar_umbral = atributo(cl, "calibrar_umbral")
    lanza(ValueError, calibrar_umbral, *_invalido_calibrar(caso), que=f"6.5 · calibrar_umbral {caso}")


# ---------------------------------------------------------------------------
# 6.6 · Evaluación
# ---------------------------------------------------------------------------

# Una autorizada rechazada (fila 1), una aceptada con otra identidad (fila 3) y un intruso aceptado (fila 5).
EVALUAR_CASO = SimpleNamespace(
    verdad=[1, 1, 2, 3, 3, 0, 0, 0],
    prediccion=[1, 1, 2, 1, 3, 2, 3, 1],
    decision=[1, 0, 2, 1, 3, 2, 0, 0],
    esperado={
        "identificacion": 4 / 5, "far": 1 / 3, "frr": 2 / 5,
        "matriz": np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 2]], dtype=np.int64),
    },
)


def test_evaluar_caso_a_mano(cl):
    evaluar = atributo(cl, "evaluar")
    c = EVALUAR_CASO
    oraculo = oraculo_evaluar(c.decision, c.prediccion, c.verdad, K)
    assert all(np.array_equal(oraculo[k], c.esperado[k]) for k in oraculo), "Control del oráculo de 4.5."
    resultado = evaluar(np.array(c.decision), np.array(c.prediccion), np.array(c.verdad), K)
    exigir_sin_problemas(resultado_evaluar_problemas(resultado, c.esperado, K), "6.6 · evaluar del caso armado a mano")


@pytest.mark.parametrize("caso", ["sin autorizadas", "sin intrusos"])
def test_evaluar_rechaza_sin_autorizadas_o_sin_intrusos(cl, caso):
    evaluar = atributo(cl, "evaluar")
    c = EVALUAR_CASO
    y = np.array(c.verdad)
    filas = y == 0 if caso == "sin autorizadas" else y > 0
    lanza(ValueError, evaluar, np.array(c.decision)[filas], np.array(c.prediccion)[filas], y[filas], K,
          que=f"6.6 · evaluar {caso}")


# ---------------------------------------------------------------------------
# 6.7 · Modelo (datos sintéticos)
# ---------------------------------------------------------------------------


def esperado_del_modelo(cfg: dict, KMeans, X_train, ids_train, X_val, verdad_val) -> SimpleNamespace:
    """Lo que tiene que dar ajustar_modelo según 4.6, armado con los oráculos y el K-means de src."""
    kmeans = cfg["huellas"]["kmeans"]
    umbral = cfg["huellas"]["umbral"]
    km = KMeans(kmeans["k"], kmeans["n_init"], kmeans["max_iter"], kmeans["tol"], semilla=cfg["semilla_global"]).ajustar(X_train)
    centroides = np.array([X_train[ids_train == i].mean(axis=0) for i in IDENTIDADES])
    dispersion = oraculo_dispersiones(X_train, ids_train - 1, centroides, umbral["piso_dispersion"])
    s, p = oraculo_puntajes(X_val, centroides, dispersion, IDENTIDADES)
    tau = oraculo_calibrar(s, p, verdad_val, umbral["criterio"], umbral["far_max_val"])
    return SimpleNamespace(identidades=np.array(IDENTIDADES, dtype=np.int64), centroides=centroides,
                           dispersion=dispersion, tau=tau, pureza=1.0, inercia=km.inercia)


def problemas_ajustar_modelo(modelo, esperado, cfg: dict, X_train, ids_train, X_val, verdad_val) -> list[str]:
    problemas = problemas_de_modelo(modelo, esperado, TOL_CENTROIDES_MODELO)
    if problemas:
        return problemas
    umbral = cfg["huellas"]["umbral"]
    # dispersión y τ exactos (1e-12) con los centroides del propio modelo
    disp = oraculo_dispersiones(X_train, ids_train - 1, modelo.centroides, umbral["piso_dispersion"])
    if (dif := max_dif(modelo.dispersion, disp)) > TOL_DISPERSION:
        problemas.append(f"dispersion = {modelo.dispersion.tolist()}; 4.2 con piso {umbral['piso_dispersion']} da {disp.tolist()} ({dif:.3e})")
    s, p = oraculo_puntajes(X_val, modelo.centroides, modelo.dispersion, modelo.identidades)
    tau = oraculo_calibrar(s, p, verdad_val, umbral["criterio"], umbral["far_max_val"])
    if abs(modelo.tau - tau) > TOL_IGUAL:
        problemas.append(f"tau = {modelo.tau!r}; calibrar_umbral de val ({umbral['criterio']}, {umbral['far_max_val']}) da {tau!r}")
    if abs(modelo.inercia - esperado.inercia) > TOL_IGUAL:
        problemas.append(f"inercia = {modelo.inercia!r}; KMeans con huellas.kmeans y la semilla global da {esperado.inercia!r}")
    if modelo.pureza != 1.0:
        problemas.append(f"pureza = {modelo.pureza!r}; se espera 1")
    return problemas


def test_ajustar_modelo_con_el_cfg_real(cl, cfg, KMeans):
    ajustar_modelo = atributo(cl, "ajustar_modelo")
    X_train, ids_train, geometria = tres_grupos_de_train()
    X_val, verdad_val = val_separable(geometria)
    esperado = esperado_del_modelo(cfg, KMeans, X_train, ids_train, X_val, verdad_val)
    sin_piso = np.sqrt([((X_train[ids_train == i] - esperado.centroides[i - 1]) ** 2).sum(axis=1).mean() for i in IDENTIDADES])
    s, p = oraculo_puntajes(X_val, esperado.centroides, esperado.dispersion, IDENTIDADES)
    assert max_dif(esperado.dispersion, sin_piso) <= TOL_IGUAL, "Control del test: con el cfg real el piso no tendría que actuar."
    assert abs(oraculo_calibrar(s, p, verdad_val, "eer", 0.01) - esperado.tau) <= TOL_IGUAL, (
        "Control del test: con val separable los dos criterios tendrían que dar el mismo τ*."
    )
    modelo = ajustar_modelo(X_train, ids_train, X_val, verdad_val, cfg)
    exigir_sin_problemas(
        problemas_ajustar_modelo(modelo, esperado, cfg, X_train, ids_train, X_val, verdad_val),
        "6.7 · ajustar_modelo con el cfg real (3 grupos separados, val separable)",
    )


CFG_VARIANTES = {
    "piso_dispersion 1.0": {"piso_dispersion": 1.0},
    "criterio eer": {"criterio": "eer"},
    "far_max_val 0.30": {"far_max_val": 0.30},
}


@pytest.mark.parametrize("variante", list(CFG_VARIANTES))
def test_ajustar_modelo_toma_piso_criterio_y_far_max_val_del_cfg(cl, cfg, KMeans, variante):
    ajustar_modelo = atributo(cl, "ajustar_modelo")
    X_train, ids_train, geometria = tres_grupos_de_train()
    X_val, verdad_val = val_con_solape(geometria)
    otro = cfg_con(cfg, **CFG_VARIANTES[variante])
    real = esperado_del_modelo(cfg, KMeans, X_train, ids_train, X_val, verdad_val)
    esperado = esperado_del_modelo(otro, KMeans, X_train, ids_train, X_val, verdad_val)
    cambia = max_dif(esperado.dispersion, real.dispersion) > 1e-6 or abs(esperado.tau - real.tau) > 1e-6
    assert cambia, f"Control del test: con {variante} el modelo esperado tendría que cambiar respecto del cfg real."
    modelo = ajustar_modelo(X_train, ids_train, X_val, verdad_val, otro)
    exigir_sin_problemas(
        problemas_ajustar_modelo(modelo, esperado, otro, X_train, ids_train, X_val, verdad_val),
        f"6.7 · ajustar_modelo con una copia del cfg con {variante} (val con solape)",
    )


def test_ajustar_modelo_val_solo_cambia_tau(cl, cfg):
    ajustar_modelo = atributo(cl, "ajustar_modelo")
    X_train, ids_train, geometria = tres_grupos_de_train()
    X_val, verdad_val = val_separable(geometria)
    base = ajustar_modelo(X_train, ids_train, X_val, verdad_val, cfg)
    s, _ = oraculo_puntajes(X_val, base.centroides, base.dispersion, base.identidades)
    autorizada = verdad_val > 0
    lejana, cercano = s[autorizada].max(), s[~autorizada].min()
    # el primer intruso pasa a estar entre la autorizada más lejana y el intruso más cercano
    i = int(np.flatnonzero(~autorizada)[0])
    centro = base.centroides[0]
    destino = (lejana + cercano) / 2 * base.dispersion[0]
    X_val2 = X_val.copy()
    X_val2[i] = centro + destino * direccion(np.random.default_rng(4), X_val.shape[1])
    s2, p2 = oraculo_puntajes(X_val2, base.centroides, base.dispersion, base.identidades)
    assert lejana < s2[i] < cercano and s2[i] == s2[~autorizada].min(), "Control del test: el intruso movido tendría que ser el más cercano."
    movido = ajustar_modelo(X_train, ids_train, X_val2, verdad_val, cfg)
    umbral = cfg["huellas"]["umbral"]
    tau2 = oraculo_calibrar(s2, p2, verdad_val, umbral["criterio"], umbral["far_max_val"])
    X_val3, verdad_val3 = val_con_solape(geometria, semilla=29)
    otro_val = ajustar_modelo(X_train, ids_train, X_val3, verdad_val3, cfg)
    problemas = []
    for nombre, m in (("con el intruso movido", movido), ("con otro val", otro_val)):
        for campo in ("centroides", "dispersion"):
            if (dif := max_dif(getattr(m, campo), getattr(base, campo))) > TOL_IGUAL:
                problemas.append(f"{nombre}: {campo} cambia en {dif:.3e} (val no puede tocar el K-means ni las dispersiones)")
        for campo in ("inercia", "pureza"):
            if abs(getattr(m, campo) - getattr(base, campo)) > TOL_IGUAL:
                problemas.append(f"{nombre}: {campo} cambia de {getattr(base, campo)!r} a {getattr(m, campo)!r}")
    if abs(movido.tau - tau2) > TOL_IGUAL or abs(movido.tau - base.tau) <= 1e-6:
        problemas.append(f"con el intruso movido, tau = {movido.tau!r} (antes {base.tau!r}); se espera {tau2!r}")
    exigir_sin_problemas(problemas, "6.7 · de qué depende cada parte del modelo")


def _train_sin_correspondencia() -> tuple[np.ndarray, np.ndarray]:
    """Dos identidades mezcladas en un grupo y la tercera partida en dos grupos lejanos."""
    rng = np.random.default_rng(17)
    d = 4
    mezclado = rng.normal(size=(20, d))
    partido_1 = np.full(d, 60.0) + rng.normal(size=(10, d))
    partido_2 = np.array([60.0, -60.0, 0.0, 0.0]) + rng.normal(size=(10, d))
    X = np.vstack([mezclado, partido_1, partido_2])
    ids = np.array([1, 2] * 10 + [3] * 20, dtype=np.int64)
    return X, ids


@pytest.mark.parametrize("caso", ["identidad 0 en train", "identidad 4 en train", "X_val con otro d", "sin correspondencia uno a uno"])
def test_ajustar_modelo_rechaza_entradas_invalidas(cl, cfg, caso):
    ajustar_modelo = atributo(cl, "ajustar_modelo")
    X_train, ids_train, geometria = tres_grupos_de_train()
    X_val, verdad_val = val_separable(geometria)
    if caso == "identidad 0 en train":
        ids_train = ids_train.copy()
        ids_train[0] = 0
    elif caso == "identidad 4 en train":
        ids_train = ids_train.copy()
        ids_train[0] = 4
    elif caso == "X_val con otro d":
        X_val = X_val[:, :-1]
    else:
        X_train, ids_train = _train_sin_correspondencia()
        X_val = np.vstack([X_train[[0, 1, 25]], X_train[[0, 1, 25]] + 30.0])
        verdad_val = np.array([1, 2, 3, 0, 0, 0], dtype=np.int64)
    lanza(ValueError, ajustar_modelo, X_train, ids_train, X_val, verdad_val, cfg, que=f"6.7 · ajustar_modelo con {caso}")


# ---------------------------------------------------------------------------
# 6.8 · Archivo del modelo y reconocimiento (dataset chico)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chico(tmp_path_factory, cfg, h1, h3) -> SimpleNamespace:
    """Sección 5: PNG, metadatos.csv y caracteristicas.npz del dataset chico, escritos una sola vez."""
    entradas = elegir_dataset_chico(h1.plan_dataset(cfg))
    conteo = {p: sum(e["particion"] == p for e in entradas) for p in CONTEO_CHICO}
    if conteo != CONTEO_CHICO:
        pytest.fail(f"El dataset chico tiene {conteo} filas por partición; se esperan {CONTEO_CHICO} (sección 5).", pytrace=False)
    carpeta = tmp_path_factory.mktemp("h4_chico")
    bases = {}
    for e in entradas:
        clave = (e["tipo"], int(e["semilla"]))
        if clave not in bases:
            bases[clave] = h1.generar_base(e["tipo"], int(e["semilla"]), cfg)
        escribir_png(carpeta / e["archivo"], receta_h1(h1, cfg, e, bases[clave]).imagen)
    escribir_metadatos(carpeta, entradas)
    try:
        h3.generar_caracteristicas(cfg, carpeta)
    except Exception as error:
        pytest.fail(f"generar_caracteristicas (H3) del dataset chico lanzó {descripcion_error(error)}", pytrace=False)
    datos = leer_npz(carpeta / "caracteristicas.npz")
    particion = datos["particion"]
    return SimpleNamespace(
        carpeta=carpeta, entradas=entradas, datos=datos,
        train=particion == "train", val=particion == "val", test=particion == "test",
    )


@pytest.fixture(scope="module")
def entrenado(tmp_path_factory, cl, cfg, chico) -> SimpleNamespace:
    entrenar = atributo(cl, "entrenar")
    salida = tmp_path_factory.mktemp("h4_modelo") / "modelo.npz"
    try:
        ruta = entrenar(cfg, chico.carpeta, salida=salida)
    except Exception as error:
        pytest.fail(f"entrenar(cfg, carpeta, salida) con el dataset chico lanzó {descripcion_error(error)}", pytrace=False)
    if not salida.is_file():
        pytest.fail(f"entrenar no escribió {salida}.", pytrace=False)
    return SimpleNamespace(ruta=ruta, salida=salida, archivo=leer_npz(salida))


@pytest.fixture(scope="module")
def modelo_chico(cl, entrenado):
    cargar_modelo = atributo(cl, "cargar_modelo")
    try:
        return cargar_modelo(entrenado.salida)
    except Exception as error:
        pytest.fail(f"cargar_modelo del archivo de entrenar lanzó {descripcion_error(error)}", pytrace=False)


@pytest.fixture(scope="module")
def escalador_chico(h3, chico):
    datos = chico.datos
    return h3.ajustar_escalador({b: datos[b][chico.train].copy() for b in BLOQUES_CONTRATO})


def test_entrenar_devuelve_la_ruta_y_el_archivo_tiene_las_16_claves(entrenado, chico):
    problemas = []
    if entrenado.ruta is None or Path(entrenado.ruta).resolve() != entrenado.salida.resolve():
        problemas.append(f"devuelve {entrenado.ruta!r}; se espera {entrenado.salida}")
    archivo = entrenado.archivo
    if set(archivo) != set(CLAVES_MODELO):
        problemas.append(f"claves {sorted(archivo)}; se esperan exactamente las 16 de 4.6: {sorted(CLAVES_MODELO)}")
    d = chico.datos["X"].shape[1]
    formas = {"identidades": (K,), "centroides": (K, d), "dispersion": (K,), "pesos": (len(BLOQUES_CONTRATO),)}
    formas.update({f"{campo}_{b}": (chico.datos[b].shape[1],) for b in BLOQUES_CONTRATO for campo in ("medias", "desvios", "columnas")})
    for clave, forma in formas.items():
        if clave in archivo and archivo[clave].shape != forma:
            problemas.append(f"{clave} tiene forma {archivo[clave].shape}; se espera {forma}")
    for clave in ("tau", "pureza", "inercia"):
        if clave in archivo and archivo[clave].size != 1:
            problemas.append(f"{clave} tiene {archivo[clave].size} valores; se espera uno")
    exigir_sin_problemas(problemas, "6.8 · entrenar y el archivo del modelo")


def test_cargar_modelo_da_los_valores_del_archivo(cl, h3, entrenado, modelo_chico):
    archivo = entrenado.archivo
    if set(archivo) != set(CLAVES_MODELO):
        pytest.fail(f"6.8 · el archivo no tiene las 16 claves de 4.6 (tiene {sorted(archivo)}).", pytrace=False)
    m = modelo_chico
    problemas = []
    if not isinstance(m, atributo(cl, "ModeloHuellas")):
        problemas.append(f"cargar_modelo devuelve {tipo_de(m)}, no un ModeloHuellas")
    del_archivo = SimpleNamespace(
        identidades=archivo["identidades"], centroides=archivo["centroides"], dispersion=archivo["dispersion"],
        tau=escalar(archivo["tau"]), pureza=escalar(archivo["pureza"]), inercia=escalar(archivo["inercia"]),
    )
    escalador_archivo = SimpleNamespace(
        medias={b: archivo[f"medias_{b}"] for b in BLOQUES_CONTRATO},
        desvios={b: archivo[f"desvios_{b}"] for b in BLOQUES_CONTRATO},
        columnas={b: archivo[f"columnas_{b}"] for b in BLOQUES_CONTRATO},
        pesos={b: float(archivo["pesos"][i]) for i, b in enumerate(BLOQUES_CONTRATO)},
    )
    if not isinstance(getattr(m, "escalador", None), h3.Escalador):
        problemas.append(f"el escalador del modelo cargado es {tipo_de(getattr(m, 'escalador', None))}, no un Escalador de H3")
    problemas += problemas_de_modelo(m, del_archivo, 0.0, escalador_archivo)
    exigir_sin_problemas(problemas, "6.8 · cargar_modelo del archivo de entrenar")


def test_modelo_entrenado_es_ajustar_modelo_de_train_y_val(cl, cfg, chico, modelo_chico, escalador_chico):
    ajustar_modelo = atributo(cl, "ajustar_modelo")
    datos = chico.datos
    esperado = ajustar_modelo(datos["X"][chico.train], datos["identidad"][chico.train],
                              datos["X"][chico.val], datos["identidad"][chico.val], cfg, escalador_chico)
    exigir_sin_problemas(
        problemas_de_modelo(modelo_chico, esperado, TOL_IGUAL, escalador_chico),
        "6.8 · el modelo de entrenar contra ajustar_modelo de las filas de train y val del npz "
        "y ajustar_escalador de los bloques de train",
    )


def test_tau_del_modelo_es_la_calibracion_de_val(cfg, chico, modelo_chico):
    datos, m = chico.datos, modelo_chico
    umbral = cfg["huellas"]["umbral"]
    s, p = oraculo_puntajes(datos["X"][chico.val], m.centroides, m.dispersion, m.identidades)
    tau = oraculo_calibrar(s, p, datos["identidad"][chico.val], umbral["criterio"], umbral["far_max_val"])
    assert abs(m.tau - tau) <= TOL_IGUAL, (
        f"6.8 · tau del modelo = {m.tau!r}; calibrar_umbral de las filas de val del npz con ese modelo da {tau!r}."
    )


def test_guardar_y_cargar_devuelven_el_mismo_modelo(cl, modelo_chico, tmp_path):
    guardar_modelo, cargar_modelo = atributo(cl, "guardar_modelo"), atributo(cl, "cargar_modelo")
    salida = tmp_path / "copia.npz"
    ruta = guardar_modelo(modelo_chico, salida)
    assert ruta is not None and Path(ruta).resolve() == salida.resolve(), f"6.8 · guardar_modelo devuelve {ruta!r}; se espera {salida}."
    cargado = cargar_modelo(salida)
    exigir_sin_problemas(
        problemas_de_modelo(cargado, modelo_chico, 0.0, modelo_chico.escalador),
        "6.8 · guardar_modelo → cargar_modelo no devuelve exactamente el modelo guardado",
    )


def test_sin_fuga_el_modelo_no_depende_de_las_filas_de_test(cl, cfg, chico, entrenado, tmp_path):
    entrenar = atributo(cl, "entrenar")
    datos = {clave: valor.copy() for clave, valor in chico.datos.items()}
    test = chico.test
    for clave in (*BLOQUES_CONTRATO, "X"):
        datos[clave][test] = chico.datos[clave][test][::-1] * 1.5 + 0.25
    assert not iguales_con_nan(datos["X"][test], chico.datos["X"][test]), "Control del test: las filas de test tendrían que cambiar."
    otra = tmp_path / "otras_filas_de_test"
    otra.mkdir()
    np.savez(otra / "caracteristicas.npz", **datos)
    entrenar(cfg, otra, salida=otra / "modelo.npz")
    distintos = npz_distintos(leer_npz(otra / "modelo.npz"), entrenado.archivo)
    assert not distintos, (
        f"6.8 · sin fuga: con otras filas de test en caracteristicas.npz, el modelo guardado cambia en {distintos} "
        "(se exige np.array_equal): entrenar usa filas de test."
    )


def test_entrenar_sin_sobrescribir_da_file_exists_error_sin_tocar_el_archivo(cl, cfg, chico, entrenado, tmp_path):
    entrenar = atributo(cl, "entrenar")
    salida = tmp_path / "modelo.npz"
    shutil.copyfile(entrenado.salida, salida)
    contenido, fecha = salida.read_bytes(), salida.stat().st_mtime_ns
    lanza(FileExistsError, entrenar, cfg, chico.carpeta, salida=salida, que="6.8 · entrenar con el modelo ya escrito")
    assert salida.read_bytes() == contenido and salida.stat().st_mtime_ns == fecha, (
        "6.8 · después del FileExistsError, el archivo del modelo cambió (contenido o fecha de modificación)."
    )


def test_entrenar_con_sobrescribir_lo_vuelve_a_escribir(cl, cfg, chico, entrenado, tmp_path):
    entrenar = atributo(cl, "entrenar")
    salida = tmp_path / "modelo.npz"
    salida.write_bytes(b"esto no es un npz")
    ruta = entrenar(cfg, chico.carpeta, salida=salida, sobrescribir=True)
    assert ruta is not None and Path(ruta).resolve() == salida.resolve(), f"6.8 · con sobrescribir=True devuelve {ruta!r}; se espera {salida}."
    distintos = npz_distintos(leer_npz(salida), entrenado.archivo)
    assert not distintos, f"6.8 · con sobrescribir=True el modelo reescrito difiere del original en {distintos}."


def test_entrenar_sin_caracteristicas_da_file_not_found(cl, cfg, tmp_path):
    entrenar = atributo(cl, "entrenar")
    vacia = tmp_path / "sin_npz"
    vacia.mkdir()
    lanza(FileNotFoundError, entrenar, cfg, vacia, salida=tmp_path / "modelo.npz", que="6.8 · entrenar sin caracteristicas.npz")


def test_reconocer_cada_png_igual_a_su_fila_de_x(cl, cfg, chico, modelo_chico):
    reconocer = atributo(cl, "reconocer")
    Reconocimiento = atributo(cl, "Reconocimiento")
    X, m = chico.datos["X"], modelo_chico
    problemas = []
    for i, e in enumerate(chico.entradas):
        r = reconocer(leer_png(chico.carpeta / e["archivo"]), m, cfg)
        if not isinstance(r, Reconocimiento):
            problemas.append(f"{e['archivo']}: reconocer devuelve {tipo_de(r)}, no un Reconocimiento")
            continue
        distancia, mas_cercana = m.puntajes(X[i])
        decision = m.decidir(X[i])
        if not es_float(r.distancia) or abs(r.distancia - distancia) > TOL_RECONOCER:
            problemas.append(f"{e['archivo']}: distancia {r.distancia!r} ({tipo_de(r.distancia)}); su fila de X da {distancia!r}")
        if not es_entero(r.mas_cercana) or r.mas_cercana != mas_cercana:
            problemas.append(f"{e['archivo']}: mas_cercana {r.mas_cercana!r} ({tipo_de(r.mas_cercana)}); su fila de X da {mas_cercana!r}")
        if not es_entero(r.identidad) or r.identidad != decision:
            problemas.append(f"{e['archivo']}: identidad {r.identidad!r} ({tipo_de(r.identidad)}); decidir de su fila da {decision!r}")
        if not es_float(r.tau) or r.tau != m.tau:
            problemas.append(f"{e['archivo']}: tau {r.tau!r}; el del modelo es {m.tau!r}")
    exigir_sin_problemas(problemas, f"6.8 · reconocer de los {len(chico.entradas)} PNG del dataset chico")


def test_reconocer_sin_escalador_da_value_error(cl, cfg, chico, modelo_chico):
    reconocer = atributo(cl, "reconocer")
    sin_escalador = dataclasses.replace(modelo_chico, escalador=None)
    imagen = leer_png(chico.carpeta / chico.entradas[0]["archivo"])
    lanza(ValueError, reconocer, imagen, sin_escalador, cfg, que="6.8 · reconocer con un modelo sin escalador")


def test_evaluar_particion_test_es_evaluar_de_las_filas_de_test(cl, cfg, chico, modelo_chico):
    evaluar_particion = atributo(cl, "evaluar_particion")
    datos, m = chico.datos, modelo_chico
    s, p = oraculo_puntajes(datos["X"][chico.test], m.centroides, m.dispersion, m.identidades)
    esperado = oraculo_evaluar(oraculo_decidir(s, p, m.tau), p, datos["identidad"][chico.test], K)
    resultado = evaluar_particion(m, cfg, chico.carpeta, "test")
    exigir_sin_problemas(resultado_evaluar_problemas(resultado, esperado, K),
                         "6.8 · evaluar_particion(modelo, cfg, carpeta, 'test') contra evaluar de las filas de test")


def test_generar_figura_escribe_un_png_legible(cl, cfg, chico, entrenado, tmp_path):
    generar_figura = atributo(cl, "generar_figura")
    salida = tmp_path / "far_frr_h4.png"
    resultado = generar_figura(cfg, chico.carpeta, entrenado.salida, salida)
    assert resultado is not None and Path(resultado).resolve() == salida.resolve(), (
        f"6.8 · generar_figura devolvió {resultado!r}; se espera {salida}."
    )
    contenido = salida.read_bytes() if salida.is_file() else b""
    assert contenido[:8] == b"\x89PNG\r\n\x1a\n", f"6.8 · {salida} no es un PNG (empieza con {contenido[:8]!r})."
    # Decodifica igual que cv2.imread(ruta) y no falla con tildes en la ruta (Windows).
    imagen = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert imagen is not None and imagen.size > 0, f"6.8 · OpenCV no pudo leer {salida}."


# ---------------------------------------------------------------------------
# 6.9 · Aceptación (dataset completo de H1 y H3 en tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def completo(tmp_path_factory, cfg, h1, h3, cl) -> SimpleNamespace:
    entrenar, cargar_modelo = atributo(cl, "entrenar"), atributo(cl, "cargar_modelo")
    carpeta = tmp_path_factory.mktemp("h4_completo")
    inicio = time.perf_counter()
    h1.generar_dataset(cfg, carpeta)
    h3.generar_caracteristicas(cfg, carpeta)
    ruta = entrenar(cfg, carpeta, carpeta / "modelo_huellas.npz")
    tiempo = time.perf_counter() - inicio
    datos = leer_npz(carpeta / "caracteristicas.npz")
    particion = datos["particion"]
    return SimpleNamespace(
        carpeta=carpeta, ruta=ruta, tiempo_s=tiempo, datos=datos, modelo=cargar_modelo(carpeta / "modelo_huellas.npz"),
        train=particion == "train", val=particion == "val", test=particion == "test",
    )


@pytest.mark.aceptacion
def test_aceptacion_todo_tarda_menos_de_5_minutos(completo):
    assert completo.tiempo_s < TIEMPO_ACEPTACION_S, (
        f"6.9 · dataset, características y modelo tardaron {completo.tiempo_s:.1f} s; se exige menos de {TIEMPO_ACEPTACION_S:g} s."
    )


@pytest.mark.aceptacion
def test_aceptacion_pureza_en_train_con_correspondencia_uno_a_uno(completo, cfg, KMeans):
    datos, m = completo.datos, completo.modelo
    X, ident = datos["X"][completo.train], datos["identidad"][completo.train]
    kmeans = cfg["huellas"]["kmeans"]
    km = KMeans(kmeans["k"], kmeans["n_init"], kmeans["max_iter"], kmeans["tol"], semilla=cfg["semilla_global"]).ajustar(X)
    mapa, pureza = oraculo_correspondencia(km.etiquetas, ident, K)
    assert sorted(mapa.values()) == list(IDENTIDADES), (
        f"6.9 · la correspondencia de los clusters de train no es uno a uno: {mapa}."
    )
    assert m.pureza >= PUREZA_MIN and abs(m.pureza - pureza) <= TOL_IGUAL, (
        f"6.9 · pureza del modelo {m.pureza!r} (la de los clusters de train da {pureza!r}); se exige >= {PUREZA_MIN}."
    )


@pytest.mark.aceptacion
def test_aceptacion_kmeans_igual_a_sklearn_en_el_dataset(completo, KMeans):
    X = completo.datos["X"][completo.train]
    sk = KMeansSklearn(n_clusters=K, n_init=15, random_state=0).fit(X)
    propio = KMeans(K, n_init=1, max_iter=300, tol=TOL_KMEANS).ajustar(X, init=sk.cluster_centers_.copy())
    problemas = []
    if (dif := max_dif(propio.centroides, sk.cluster_centers_)) > TOL_SKLEARN:
        problemas.append(f"con init = centroides de scikit-learn, los centroides difieren en {dif:.3e}")
    if (rel := dif_relativa(propio.inercia, sk.inertia_)) > TOL_SKLEARN:
        problemas.append(f"inercia {propio.inercia!r} contra {sk.inertia_!r} (relativa {rel:.3e})")
    if (dif := emparejar(completo.modelo.centroides, sk.cluster_centers_)) > TOL_SKLEARN:
        problemas.append(f"los centroides del modelo, emparejados, difieren de los de scikit-learn en {dif:.3e}")
    exigir_sin_problemas(problemas, f"6.9 · K-means en las {len(X)} filas de train contra scikit-learn (tolerancia {TOL_SKLEARN:g})")


@pytest.mark.aceptacion
def test_aceptacion_metas_en_test(completo, cl, cfg):
    evaluar_particion = atributo(cl, "evaluar_particion")
    metas = cfg["metas"]
    r = evaluar_particion(completo.modelo, cfg, completo.carpeta, "test")
    ok = (r["identificacion"] >= metas["identificacion_min"] and r["far"] <= metas["far_max"]
          and r["frr"] <= metas["frr_max"])
    assert ok, (
        f"6.9 · en test: identificación {r['identificacion']:.4f} (meta >= {metas['identificacion_min']}), "
        f"FAR {r['far']:.4f} (meta <= {metas['far_max']}), FRR {r['frr']:.4f} (meta <= {metas['frr_max']}); "
        f"matriz {np.asarray(r['matriz']).tolist()}."
    )


@pytest.mark.aceptacion
def test_aceptacion_tau_solo_con_las_filas_de_val(completo, cfg):
    datos, m = completo.datos, completo.modelo
    umbral = cfg["huellas"]["umbral"]
    s, p = oraculo_puntajes(datos["X"][completo.val], m.centroides, m.dispersion, m.identidades)
    tau = oraculo_calibrar(s, p, datos["identidad"][completo.val], umbral["criterio"], umbral["far_max_val"])
    assert abs(m.tau - tau) <= TOL_IGUAL, (
        f"6.9 · τ* del modelo = {m.tau!r}; calibrar_umbral con solo las filas de val da {tau!r}."
    )
