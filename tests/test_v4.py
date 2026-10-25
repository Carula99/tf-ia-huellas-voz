"""Tests del contrato V4 · K-NN propio, selección de hiperparámetros por validación cruzada
por persona y umbral de rechazo (`v4_clasificacion.md`, versión 1, 24/10/2026).

Aclaraciones de Santiago del 24/10/2026 que estos tests ya incorporan:
  * Desempate del K-NN (4.1): entre las clases cuyos votos difieren en 1e-12 o menos gana la de
    menor distancia MÍNIMA a sus vecinos; si siguen empatadas, la primera de `clases`.
  * `elegir_mejor` devuelve un dict con exactamente `esquema`, `n_componentes` y `k`.
  * `ModeloVoz.transformar(X, cfg)`, `ModeloVoz.votar(X, cfg)` y `ModeloVoz.decidir(X, cfg)`
    reciben `cfg`: los largos salen de `largos_de_bloque(cfg)` y `cmin`, de `cfg`.
  * Ejemplo 5 de la calibración: con `n_otro_no_validos = 18` entra por el punto 4 de 4.4 y con 16
    por el punto 3 (τ* = 0.85 en los dos). Caso del respaldo `1.2·a`: τ* = 0.77.
  * De V3 se usa solo lo que pegó Santiago (`clip_sintetico`, `largos_de_bloque`, `caracteristicas`,
    `Escalador`, `ajustar_escalador`, `generar_caracteristicas`); de H3, el `PCA`.

Los módulos de `src/` se importan dentro de fixtures: si uno falta o falla, fallan solo los tests
que lo usan. La sección 6.9 lleva la marca `aceptacion` y se saltea si no existe
`data/voz/prepro/caracteristicas.npz`. Todo lo temporal se escribe en `tmp_path`/`tmp_path_factory`.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

RAIZ = Path(__file__).resolve().parents[1]

# Constantes del contrato (6.1 verifica que config.yaml coincide con ellas).
ETIQUETAS = ["reconocer", "identificar", "salir", "otro"]
COMANDOS = ["reconocer", "identificar", "salir"]
OTRO = "otro"
ESQUEMAS = ("contrato", "sin_dur_zcr", "z_columna")
BLOQUES = ("mfcc", "delta", "energia", "zcr", "duracion")
ORDEN_PCA = [20, 30, None]
TOL = 1e-12  # "igual" sin tolerancia explícita (sección 8, Tipos)

CLAVES_TABLA = {"esquema", "n_componentes", "k", "n_tau5", "n_c60", "n_4cl", "tau_cv", "descartada"}
CLAVES_MEJOR = {"esquema", "n_componentes", "k"}
CLAVES_EVALUAR = {"identificacion", "acierto", "frr", "otro_aceptado", "matriz"}
CLAVES_MODELO = (
    {"esquema", "factores", "n_componentes", "k", "eps", "confianza_min", "tau", "entrenamiento",
     "etiquetas", "pca_media", "pca_componentes", "pesos"}
    | {f"{p}_{b}" for p in ("medias", "desvios", "columnas") for b in BLOQUES}
)
CARPETA_REAL = RAIZ / "data" / "voz" / "prepro"


# ===========================================================================
# Fixtures de módulos de src/ (importados acá, nunca en el tope del archivo)
# ===========================================================================

@pytest.fixture(scope="module")
def cfg():
    from src.config import cargar_config
    return cargar_config()


@pytest.fixture(scope="module")
def clf():
    import src.voz.clasificacion as modulo
    return modulo


@pytest.fixture(scope="module")
def knn_mod():
    import src.core_ml.knn as modulo
    return modulo


@pytest.fixture(scope="module")
def v3():
    import src.voz.caracteristicas as modulo
    return modulo


@pytest.fixture(scope="module")
def largos(cfg, v3):
    """Largos de bloque de la config real, en el orden de BLOQUES."""
    dic = v3.largos_de_bloque(cfg)
    return [int(dic[b]) for b in BLOQUES]


@pytest.fixture(scope="module")
def parametros(cfg):
    return {
        "cmin": float(cfg["voz"]["knn"]["confianza_min"]),
        "eps": float(cfg["voz"]["knn"]["eps"]),
        "otro_max": float(cfg["metas"]["voz_otro_aceptado_max"]),
    }


# ===========================================================================
# Utilidades generales
# ===========================================================================

def _cfg_con(cfg: dict, claves: tuple[str, ...], valor) -> dict:
    """Copia profunda de cfg con cfg[claves[0]][claves[1]]... = valor."""
    nuevo = copy.deepcopy(cfg)
    nodo = nuevo
    for c in claves[:-1]:
        nodo = nodo[c]
    nodo[claves[-1]] = valor
    return nuevo


def _es_float(x) -> bool:
    return isinstance(x, float)


def _es_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _cargar_npz(ruta) -> dict:
    with np.load(ruta, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _leer_csv(ruta) -> list[dict]:
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _escribir_csv(ruta, filas: list[dict], columnas: list[str]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        for fila in filas:
            w.writerow({c: fila.get(c, "") for c in columnas})


def _es_valido(valor) -> bool:
    return str(valor).strip() in {"1", "True", "true"}


def _a_int(valor):
    try:
        return int(float(str(valor).strip()))
    except (TypeError, ValueError):
        return None


def _personas(hablantes, cfg) -> np.ndarray:
    """Código de persona según voz.dataset.mismo_hablante (lista de grupos; manda el primero)."""
    mapa = {}
    for grupo in cfg["voz"]["dataset"]["mismo_hablante"]:
        for codigo in grupo:
            mapa[str(codigo)] = str(grupo[0])
    return np.array([mapa.get(str(h), str(h)) for h in hablantes])


def _no_validos_otro(meta: list[dict], particion: str, solo_evaluables: bool) -> int:
    n = 0
    for f in meta:
        if f["particion"] != particion or _es_valido(f["valido"]) or f["clase"] != OTRO:
            continue
        if solo_evaluables and not (f["origen"] == "notebook" and _a_int(f["copia"]) == 0):
            continue
        n += 1
    return n


def _diferencias(a, b, atol: float = 0.0, ruta: str = "obj") -> list[str]:
    """Comparación profunda (dataclasses, dicts, arrays y escalares). Devuelve las diferencias."""
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        if not (dataclasses.is_dataclass(b) and type(a).__name__ == type(b).__name__):
            return [f"{ruta}: tipos distintos ({type(a).__name__} vs {type(b).__name__})"]
        difs = []
        for campo in dataclasses.fields(a):
            difs += _diferencias(getattr(a, campo.name), getattr(b, campo.name, None), atol,
                                 f"{ruta}.{campo.name}")
        return difs
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return [f"{ruta}: uno es dict y el otro no"]
        if set(map(str, a)) != set(map(str, b)):
            return [f"{ruta}: claves {sorted(map(str, a))} vs {sorted(map(str, b))}"]
        difs = []
        for clave in a:
            difs += _diferencias(a[clave], b[clave], atol, f"{ruta}[{clave!r}]")
        return difs
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{ruta}: largos {len(a)} vs {len(b)}"]
        difs = []
        for i, (x, y) in enumerate(zip(a, b)):
            difs += _diferencias(x, y, atol, f"{ruta}[{i}]")
        return difs
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a2, b2 = np.asarray(a), np.asarray(b)
        if a2.shape != b2.shape:
            return [f"{ruta}: formas {a2.shape} vs {b2.shape}"]
        if a2.dtype.kind in "fc" or b2.dtype.kind in "fc":
            a2, b2 = a2.astype(np.float64), b2.astype(np.float64)
            ok = (np.array_equal(a2, b2, equal_nan=True) if atol == 0
                  else bool(np.allclose(a2, b2, rtol=0.0, atol=atol, equal_nan=True)))
        else:
            ok = bool(np.array_equal(a2, b2))
        return [] if ok else [f"{ruta}: arrays distintos"]
    if a is None or b is None:
        return [] if (a is None and b is None) else [f"{ruta}: {a!r} vs {b!r}"]
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return []
        return [] if abs(fa - fb) <= atol else [f"{ruta}: {fa!r} vs {fb!r}"]
    return [] if a == b else [f"{ruta}: {a!r} vs {b!r}"]


def _comparar_npz(a: dict, b: dict) -> list[str]:
    if set(a) != set(b):
        return [f"claves distintas: {sorted(set(a) ^ set(b))}"]
    return [k for k in a if not np.array_equal(a[k], b[k])]


# ===========================================================================
# Oráculos escritos a partir de las secciones 4.1 a 4.5 del contrato
# ===========================================================================

def _knn_oraculo(X_tr, y_tr, X_q, k: int, eps: float):
    """4.1: (distancias (n,k), índices (n,k), predicción, confianza, distancia media)."""
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_q = np.atleast_2d(np.asarray(X_q, dtype=np.float64))
    y_tr = np.asarray(y_tr)
    clases = np.unique(y_tr)
    D = cdist(X_q, X_tr)
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]  # empate de distancias: índice menor
    dist = np.take_along_axis(D, idx, axis=1)
    n = X_q.shape[0]
    pred = np.empty(n, dtype=y_tr.dtype)
    conf = np.empty(n)
    dmed = np.empty(n)
    for i in range(n):
        w = 1.0 / (dist[i] + eps)
        et = y_tr[idx[i]]
        votos, dmin = {}, {}
        for c in clases:
            msk = et == c
            if msk.any():
                votos[c] = float(w[msk].sum())
                dmin[c] = float(dist[i][msk].min())
        vmax = max(votos.values())
        cand = [c for c in clases if c in votos and abs(votos[c] - vmax) <= 1e-12]
        dm = min(dmin[c] for c in cand)
        cand = [c for c in cand if dmin[c] == dm]
        ganador = cand[0]
        pred[i] = ganador
        conf[i] = votos[ganador] / float(w.sum())
        dmed[i] = float(dist[i][et == ganador].mean())
    return dist, idx, pred, conf, dmed


def _calibrar_oraculo(distancia, confianza, prediccion, verdad, cmin: float, otro_max: float,
                      n_otro_no_validos: int = 0) -> float:
    """4.4, con sus ValueError."""
    d = np.asarray(distancia, dtype=np.float64)
    c = np.asarray(confianza, dtype=np.float64)
    p = np.asarray(prediccion).astype(str)
    v = np.asarray(verdad).astype(str)
    if not (len(d) == len(c) == len(p) == len(v)):
        raise ValueError("largos distintos")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(c))):
        raise ValueError("valores no finitos")
    if not (np.isin(p, ETIQUETAS).all() and np.isin(v, ETIQUETAS).all()):
        raise ValueError("etiquetas fuera de ETIQUETAS")
    if not (0.0 < otro_max < 1.0):
        raise ValueError("otro_max fuera de (0, 1)")
    if not _es_int(n_otro_no_validos) or n_otro_no_validos < 0:
        raise ValueError("n_otro_no_validos inválido")
    es_cmd = np.isin(v, COMANDOS)
    es_otro = v == OTRO
    if not es_cmd.any():
        raise ValueError("sin filas de comando")
    acep = (p != OTRO) & (c >= cmin)
    n_otro = int(es_otro.sum()) + n_otro_no_validos
    m = int(math.floor(otro_max * n_otro + 1e-9))
    B = np.sort(d[es_otro & acep])
    C = np.sort(d[es_cmd & acep & (p == v)])
    if len(B) > m:
        b = B[m]
        menores = C[C < b]
        if menores.size == 0:
            raise ValueError("C vacío por debajo de b")
        a = menores[-1]
    else:
        if C.size == 0:
            raise ValueError("C vacío")
        a = C[-1]
        otros = d[es_otro]
        mayores = otros[otros > a]
        b = mayores.min() if mayores.size else 1.2 * a
    return float((a + b) / 2.0)


def _decidir_oraculo(prediccion, confianza, distancia, tau: float, cmin: float) -> np.ndarray:
    p = np.asarray(prediccion).astype(str)
    acep = (p != OTRO) & (np.asarray(confianza) >= cmin) & (np.asarray(distancia) <= tau)
    return np.where(acep, p, OTRO)


def _matriz_oraculo(verdad, decision) -> np.ndarray:
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(np.asarray(verdad).astype(str), np.asarray(decision).astype(str),
                            labels=ETIQUETAS).astype(np.int64)


def _evaluar_oraculo(decision, prediccion, verdad) -> dict:
    d = np.asarray(decision).astype(str)
    p = np.asarray(prediccion).astype(str)
    v = np.asarray(verdad).astype(str)
    es_cmd = np.isin(v, COMANDOS)
    es_otro = v == OTRO
    acierto = float(np.mean(d[es_cmd] == v[es_cmd]))
    return {
        "identificacion": float(np.mean(p[es_cmd] == v[es_cmd])),
        "acierto": acierto,
        "frr": 1.0 - acierto,
        "otro_aceptado": float(np.mean(d[es_otro] != OTRO)),
        "matriz": _matriz_oraculo(v, d),
    }


def _factores_oraculo(esquema: str, largos: list[int]) -> np.ndarray:
    if esquema == "contrato":
        return np.ones(5)
    if esquema == "sin_dur_zcr":
        return np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    if esquema == "z_columna":
        return np.sqrt(np.asarray(largos, dtype=np.float64))
    raise KeyError(esquema)


def _reponderar_oraculo(X, factores, largos: list[int]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    partes, ini = [], 0
    for f, d in zip(factores, largos):
        if f != 0:
            partes.append(X[..., ini:ini + d] * float(f))
        ini += d
    return np.concatenate(partes, axis=-1)


def _normalizar_oraculo(X, piso: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), piso)


def _cv_oraculo(X, etiquetas, personas, evaluable, n_otro_no_validos: int, esquema: str,
                n_componentes, k: int, largos: list[int], par: dict) -> dict:
    """4.3 para UNA combinación, con sklearn.decomposition.PCA y el voto de 4.1."""
    from sklearn.decomposition import PCA as PCASk

    etiquetas = np.asarray(etiquetas).astype(str)
    personas = np.asarray(personas).astype(str)
    evaluable = np.asarray(evaluable, dtype=bool)
    descartada = {"esquema": esquema, "n_componentes": n_componentes, "k": k, "n_tau5": 0,
                  "n_c60": 0, "n_4cl": 0, "tau_cv": float("nan"), "descartada": True}
    pliegues = sorted(set(personas[evaluable & np.isin(etiquetas, COMANDOS)]))
    R = _reponderar_oraculo(X, _factores_oraculo(esquema, largos), largos)
    d_r = R.shape[1]
    dist_t, conf_t, pred_t, verdad_t = [], [], [], []
    for p in pliegues:
        tr = personas != p
        q = evaluable & (personas == p)
        n_tr = int(tr.sum())
        if (n_componentes is not None and n_componentes > min(n_tr - 1, d_r)) or k > n_tr:
            return descartada
        R_tr, R_q = R[tr], R[q]
        if n_componentes is not None:
            pca = PCASk(n_components=n_componentes, svd_solver="full").fit(R_tr)
            R_tr, R_q = pca.transform(R_tr), pca.transform(R_q)
        _, _, pred, conf, dmed = _knn_oraculo(_normalizar_oraculo(R_tr), etiquetas[tr],
                                              _normalizar_oraculo(R_q), k, par["eps"])
        dist_t.append(dmed)
        conf_t.append(conf)
        pred_t.append(pred.astype(str))
        verdad_t.append(etiquetas[q])
    dist_t, conf_t = np.concatenate(dist_t), np.concatenate(conf_t)
    pred_t, verdad_t = np.concatenate(pred_t), np.concatenate(verdad_t)
    try:
        tau = _calibrar_oraculo(dist_t, conf_t, pred_t, verdad_t, par["cmin"], par["otro_max"],
                                n_otro_no_validos)
    except ValueError:
        return descartada
    bien = np.isin(verdad_t, COMANDOS) & (pred_t == verdad_t)
    c60 = bien & (conf_t >= par["cmin"])
    return {"esquema": esquema, "n_componentes": n_componentes, "k": k,
            "n_tau5": int((c60 & (dist_t <= tau)).sum()), "n_c60": int(c60.sum()),
            "n_4cl": int(bien.sum()), "tau_cv": tau, "descartada": False}


def _grilla(cfg) -> list[tuple]:
    pcas = list(cfg["voz"]["knn"]["pca_componentes_candidatas"]) + [None]
    return [(e, n, k) for e in ESQUEMAS for n in pcas for k in cfg["voz"]["knn"]["k_candidatos"]]


def _elegir_oraculo(tabla: list[dict]) -> dict:
    validas = [f for f in tabla if not f["descartada"]]
    if not validas:
        raise ValueError("todas descartadas")
    mejor = min(validas, key=lambda f: (-(f["n_tau5"] + f["n_c60"]), -f["n_4cl"],
                                        ESQUEMAS.index(f["esquema"]),
                                        ORDEN_PCA.index(f["n_componentes"]), f["k"]))
    return {c: mejor[c] for c in ("esquema", "n_componentes", "k")}


# ===========================================================================
# Datos sintéticos (secciones 6.2 a 6.7)
# ===========================================================================

def _datos_gauss(n: int, d: int, c: int, semilla: int, n_consultas: int = 25):
    rng = np.random.default_rng(semilla)
    centros = rng.normal(0.0, 2.0, size=(c, d))
    y_idx = np.arange(n) % c
    rng.shuffle(y_idx)
    X = centros[y_idx] + rng.normal(size=(n, d))
    q_idx = rng.integers(0, c, n_consultas)
    Q = centros[q_idx] + rng.normal(size=(n_consultas, d))
    return X, np.array([f"c{j}" for j in y_idx]), Q


def _escala_bloques(largos: list[int]) -> np.ndarray:
    """Imita el peso 1/√d_b de V3 sobre filas generadas en el espacio z."""
    return np.concatenate([np.full(l, 1.0 / math.sqrt(l)) for l in largos])


def _filas_persona(rng, centros: dict, d: int, clases: list[str], por_clase: int, n_eval: int,
                   ruido: float, ruido_copia: float, desvio_persona: float, fuerza_confusa: float):
    """Filas de una persona en el espacio z. Las primeras n_eval de cada clase son evaluables.

    Los "otro" alternan tres tipos: parecidos a un comando, palabras libres y el centro de "otro".
    """
    offset = rng.normal(0.0, desvio_persona, d)
    Z, y, ev = [], [], []
    for c in clases:
        for j in range(por_clase):
            if c == OTRO:
                tipo = j % 3
                if tipo == 0:
                    cmd = COMANDOS[(j // 3) % 3]
                    base = fuerza_confusa * centros[cmd] + (1 - fuerza_confusa) * centros[OTRO]
                elif tipo == 1:
                    base = rng.normal(0.0, 1.0, d)
                else:
                    base = centros[OTRO]
            else:
                base = centros[c]
            es_ev = j < n_eval
            Z.append(base + offset + rng.normal(0.0, ruido if es_ev else ruido_copia, d))
            y.append(c)
            ev.append(es_ev)
    return Z, y, ev, offset


# Parámetros de los grupos sintéticos (verificados para que los conteos no sean triviales).
_RUIDO_CV = dict(ruido=2.4, ruido_copia=2.7, desvio_persona=0.8, fuerza_confusa=0.65)


def _datos_cv(largos: list[int], semilla: int = 20261024) -> dict:
    """5 personas × 4 clases × 18 filas (6 evaluables por persona y clase), de (360, 281)."""
    rng = np.random.default_rng(semilla)
    d = int(sum(largos))
    esc = _escala_bloques(largos)
    centros = {c: rng.normal(0.0, 1.0, d) for c in ETIQUETAS}
    X, y, P, E = [], [], [], []
    for p in ("s01", "s02", "s03", "s04", "s05"):
        Z, yy, ev, _ = _filas_persona(rng, centros, d, ETIQUETAS, 18, 6, **_RUIDO_CV)
        X += [z * esc for z in Z]
        y += yy
        E += ev
        P += [p] * len(yy)
    return {"X": np.array(X), "etiquetas": np.array(y), "personas": np.array(P),
            "evaluable": np.array(E, dtype=bool), "n_otro_no_validos": 2,
            "centros": centros, "semilla": semilla}


def _agregar_personas_extra(base: dict, largos: list[int]) -> dict:
    """s06: solo filas no evaluables. s07: "otro" evaluables pegados a los comandos y comandos
    no evaluables (sin filas evaluables de comando: no genera pliegue y sus "otro" no se consultan)."""
    rng = np.random.default_rng(base["semilla"] + 99)
    d = int(sum(largos))
    esc = _escala_bloques(largos)
    centros = base["centros"]
    X, y = list(base["X"]), list(base["etiquetas"])
    P, E = list(base["personas"]), list(base["evaluable"])
    Z, yy, _, _ = _filas_persona(rng, centros, d, ETIQUETAS, 5, 0, **_RUIDO_CV)
    X += [z * esc for z in Z]
    y += yy
    E += [False] * len(yy)
    P += ["s06"] * len(yy)
    off = rng.normal(0.0, 0.3, d)
    for j in range(6):
        X.append((centros[COMANDOS[j % 3]] + off + rng.normal(0.0, 0.8, d)) * esc)
        y.append(OTRO)
        E.append(True)
        P.append("s07")
    Z, yy, _, _ = _filas_persona(rng, centros, d, COMANDOS, 4, 0, **_RUIDO_CV)
    X += [z * esc for z in Z]
    y += yy
    E += [False] * len(yy)
    P += ["s07"] * len(yy)
    return {"X": np.array(X), "etiquetas": np.array(y), "personas": np.array(P),
            "evaluable": np.array(E, dtype=bool), "n_otro_no_validos": base["n_otro_no_validos"]}


def _datos_val(largos: list[int], centros: dict, semilla: int = 777) -> dict:
    """val sintético: 2 personas nuevas × 4 clases × 6 filas, más 3 "otro" no válidos."""
    rng = np.random.default_rng(semilla)
    d = int(sum(largos))
    esc = _escala_bloques(largos)
    X, y = [], []
    for _ in ("v01", "v02"):
        Z, yy, _, _ = _filas_persona(rng, centros, d, ETIQUETAS, 6, 6, **_RUIDO_CV)
        X += [z * esc for z in Z]
        y += yy
    return {"X": np.array(X), "etiquetas": np.array(y), "n_otro_no_validos": 3}


def _val_con_otro_pegado_a_un_comando(modelo, cfg, va: dict, par: dict):
    """Agrega a val, como "otro", una copia exacta de la fila de comando que hoy fija `a` en 4.4
    (la más lejana de C por debajo de b): es un "otro" sobre un centro de clase y τ* tiene que bajar."""
    pred, conf, dist = modelo.votar(va["X"], cfg)
    pred, conf, dist = np.asarray(pred).astype(str), np.asarray(conf), np.asarray(dist)
    v = va["etiquetas"]
    acep = (pred != OTRO) & (conf >= par["cmin"])
    idx_c = np.flatnonzero(np.isin(v, COMANDOS) & acep & (pred == v))
    B = np.sort(dist[(v == OTRO) & acep])
    m = int(math.floor(par["otro_max"] * (int((v == OTRO).sum()) + va["n_otro_no_validos"] + 1) + 1e-9))
    tope = B[m] if len(B) > m else np.inf
    candidatas = idx_c[dist[idx_c] < tope]
    assert len(candidatas) >= 2, "precondición del test: val necesita al menos 2 comandos aceptables bajo b"
    i = int(candidatas[np.argmax(dist[candidatas])])
    return np.vstack([va["X"], va["X"][i]]), np.append(v, OTRO)


def _datos_chicos(largos: list[int], semilla: int = 31, personas=("t01", "t02", "t03")) -> dict:
    """Pocas filas bien separadas: 13 por persona (3 por comando y 4 "otro"), todas evaluables.
    Con 3 personas cada pliegue entrena con 26 filas: 30 componentes > 25 se descartan."""
    rng = np.random.default_rng(semilla)
    d = int(sum(largos))
    esc = _escala_bloques(largos)
    centros = {c: rng.normal(0.0, 1.0, d) for c in ETIQUETAS}
    X, y, P = [], [], []
    for p in personas:
        for c, cant in (("reconocer", 3), ("identificar", 3), ("salir", 3), (OTRO, 4)):
            for _ in range(cant):
                X.append((centros[c] + rng.normal(0.0, 0.3, d)) * esc)
                y.append(c)
                P.append(p)
    return {"X": np.array(X), "etiquetas": np.array(y), "personas": np.array(P),
            "evaluable": np.ones(len(y), dtype=bool), "n_otro_no_validos": 0, "centros": centros}


@pytest.fixture(scope="module")
def datos_cv(largos):
    return _datos_cv(largos)


@pytest.fixture(scope="module")
def resultado_cv(clf, cfg, datos_cv):
    d = datos_cv
    return clf.seleccionar_hiperparametros(d["X"], d["etiquetas"], d["personas"], d["evaluable"],
                                           d["n_otro_no_validos"], cfg)


@pytest.fixture(scope="module")
def datos_val(largos, datos_cv):
    return _datos_val(largos, datos_cv["centros"])


@pytest.fixture(scope="module")
def datos_chicos(largos):
    return _datos_chicos(largos)


@pytest.fixture(scope="module")
def modelo_sintetico(clf, cfg, datos_cv, datos_val):
    tr, va = datos_cv, datos_val
    return clf.ajustar_modelo(tr["X"], tr["etiquetas"], tr["personas"], tr["evaluable"],
                              tr["n_otro_no_validos"], va["X"], va["etiquetas"],
                              va["n_otro_no_validos"], cfg)


# ===========================================================================
# Clips y dataset chico (sección 6.8, con la sección 5 de V3)
# ===========================================================================

PARAMETROS_CLASES = {  # (centro, semiancho) de f0 y de la duración; mod_hz fijo
    "reconocer": {"f0": (220.0, 15.0), "mod": 3.0, "dur": (0.75, 0.05)},
    "identificar": {"f0": (520.0, 25.0), "mod": 6.0, "dur": (1.15, 0.06)},
    "salir": {"f0": (1200.0, 40.0), "mod": 1.5, "dur": (0.42, 0.03)},
    "otro": {"f0": (800.0, 40.0), "mod": 4.5, "dur": (0.95, 0.06)},
}


def clip_sintetico(f0_hz: float, mod_hz: float, dur_palabra_s: float, amp: float, semilla: int,
                   fs: int = 16000, M: int = 640) -> np.ndarray:
    """Sección 5 de V3: L + 2·M muestras, envolvente modulada × portadora de 3 armónicos + ruido."""
    L = int(round(dur_palabra_s * fs))
    t = np.arange(L + 2 * M, dtype=np.float64) / fs
    envolvente = 1.0 + 0.4 * np.sin(2 * np.pi * mod_hz * t)
    portadora = (0.6 * np.sin(2 * np.pi * f0_hz * t) + 0.3 * np.sin(2 * np.pi * 2 * f0_hz * t)
                 + 0.1 * np.sin(2 * np.pi * 3 * f0_hz * t))
    ruido = np.random.default_rng(semilla).normal(0.0, 0.02, t.size)
    return (amp * (envolvente * portadora + ruido)).astype(np.float32)


def clip_de_clase(clase: str, semilla: int, fs: int = 16000, M: int = 640) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    p = PARAMETROS_CLASES[clase]
    f0 = rng.uniform(p["f0"][0] - p["f0"][1], p["f0"][0] + p["f0"][1])
    dur = rng.uniform(p["dur"][0] - p["dur"][1], p["dur"][0] + p["dur"][1])
    amp = rng.uniform(0.3, 0.5)
    return clip_sintetico(f0, p["mod"], dur, amp, semilla, fs, M)


def _fs_y_margen(cfg) -> tuple[int, int]:
    fs = int(cfg["voz"]["frecuencia_muestreo"])
    return fs, int(round(float(cfg["voz"]["preprocesamiento"]["margen_ms"]) * fs / 1000))


def _columnas_csv() -> list[str]:
    obligatorias = ["archivo", "particion", "origen", "hablante", "clase", "copia", "valido"]
    try:
        from src.voz.preprocesamiento import COLUMNAS_V2
        columnas = [str(c) for c in COLUMNAS_V2]
    except (ImportError, AttributeError):
        columnas = []
    return columnas + [c for c in obligatorias if c not in columnas]


# particion: (personas, clips por persona y clase, filas no válidas de "otro")
ESTRUCTURA_CHICO = {
    "train": (("q01", "q02", "q03"), 4, 0),
    "val": (("q04", "q05"), 3, 2),
    "test": (("q06", "q07"), 2, 1),
}


@pytest.fixture(scope="module")
def dataset_chico(tmp_path_factory, cfg, v3):
    import soundfile as sf

    carpeta = tmp_path_factory.mktemp("prepro_chico")
    fs, M = _fs_y_margen(cfg)
    filas, semilla = [], 5000
    for particion, (habs, por_clase, n_invalidas) in ESTRUCTURA_CHICO.items():
        (carpeta / particion).mkdir()
        for h in habs:
            for clase in ETIQUETAS:
                for i in range(por_clase):
                    semilla += 1
                    archivo = f"{particion}/nb_{h}_{clase}_{i + 1:02d}_k0.wav"
                    sf.write(carpeta / archivo, clip_de_clase(clase, semilla, fs, M), fs, subtype="FLOAT")
                    filas.append({"archivo": archivo, "particion": particion, "origen": "notebook",
                                  "hablante": h, "clase": clase, "copia": "0", "valido": "1"})
        for j in range(n_invalidas):
            filas.append({"archivo": "", "particion": particion, "origen": "notebook",
                          "hablante": habs[j % len(habs)], "clase": OTRO, "copia": "0", "valido": "0"})
    _escribir_csv(carpeta / "metadatos.csv", filas, _columnas_csv())
    v3.generar_caracteristicas(cfg, carpeta)
    return carpeta


@pytest.fixture(scope="module")
def modelo_chico(tmp_path_factory, clf, cfg, dataset_chico):
    salida = tmp_path_factory.mktemp("modelo_chico") / "modelo.npz"
    ruta = clf.entrenar(cfg, carpeta=dataset_chico, salida=salida)
    return {"salida": salida, "ruta": ruta}


def _arrays_de_particion(npz: dict, particion: str, cfg) -> dict:
    msk = npz["particion"].astype(str) == particion
    return {
        "X": npz["X"][msk],
        "etiquetas": npz["clase"][msk].astype(str),
        "personas": _personas(npz["hablante"][msk], cfg),
        "evaluable": (npz["origen"][msk].astype(str) == "notebook") & (npz["copia"][msk] == 0),
        "mascara": msk,
    }


# ===========================================================================
# Autoverificación de los oráculos (no usa src/): reproducen los ejemplos del contrato
# ===========================================================================

FILAS_A = [  # (verdad, prediccion, confianza, distancia)
    ("reconocer", "reconocer", 0.9, 0.50), ("identificar", "identificar", 0.8, 0.60),
    ("salir", "salir", 1.0, 0.70), ("reconocer", "reconocer", 0.7, 0.55),
    ("otro", "reconocer", 0.9, 1.00), ("otro", "salir", 0.8, 1.30),
    ("otro", "identificar", 0.7, 1.50), ("otro", "otro", 0.9, 0.20),
]
DIST_B = [0.50, 0.90, 1.30, 0.70, 1.10, 1.60, 2.00, 0.20]


def _filas_ejemplo(nombre: str) -> list[tuple]:
    if nombre in ("A", "con_no_validos"):
        return list(FILAS_A)
    if nombre == "B":
        return [(v, p, c, d) for (v, p, c, _), d in zip(FILAS_A, DIST_B)]
    if nombre == "confianza_baja":
        return list(FILAS_A) + [("salir", "salir", 0.5, 0.95)]
    if nombre == "pocos_aceptables":
        return [f for i, f in enumerate(FILAS_A) if i not in (5, 6)]
    if nombre == "respaldo":
        return FILAS_A[:4] + [FILAS_A[7]]
    raise KeyError(nombre)


# (id, filas, otro_aceptado_max, n_otro_no_validos, τ* esperado)
EJEMPLOS_CALIBRACION = [
    ("A_separable", "A", 0.05, 0, 0.85),
    ("B_solape_005", "B", 0.05, 0, 1.00),
    ("B_solape_030", "B", 0.30, 0, 1.45),
    ("confianza_baja", "confianza_baja", 0.05, 0, 0.85),
    ("con_no_validos_16", "con_no_validos", 0.05, 16, 1.00),
    ("pocos_aceptables_16_punto3", "pocos_aceptables", 0.05, 16, 0.85),
    ("pocos_aceptables_18_punto4", "pocos_aceptables", 0.05, 18, 0.85),
    ("respaldo_1_2a", "respaldo", 0.05, 0, 0.77),
]

EJEMPLO_EVALUAR = [  # (verdad, prediccion, decision)
    ("reconocer", "reconocer", "reconocer"), ("identificar", "identificar", "otro"),
    ("salir", "identificar", "identificar"), ("salir", "salir", "salir"),
    ("otro", "salir", "salir"), ("otro", "otro", "otro"),
    ("otro", "identificar", "otro"), ("otro", "otro", "otro"),
]


def _columnas(filas: list[tuple]):
    v, p, c, d = zip(*filas)
    return (np.array(d, dtype=np.float64), np.array(c, dtype=np.float64), np.array(p), np.array(v))


@pytest.mark.parametrize("caso", EJEMPLOS_CALIBRACION, ids=[e[0] for e in EJEMPLOS_CALIBRACION])
def test_oraculo_calibracion_reproduce_ejemplos_del_contrato(caso):
    _, nombre, otro_max, n_nv, esperado = caso
    d, c, p, v = _columnas(_filas_ejemplo(nombre))
    obtenido = _calibrar_oraculo(d, c, p, v, 0.60, otro_max, n_nv)
    assert abs(obtenido - esperado) <= TOL, f"oráculo del test: τ* = {obtenido!r}, esperado {esperado!r}"


def test_oraculos_knn_y_evaluar_reproducen_ejemplos_del_contrato():
    _, _, pred, conf, _ = _knn_oraculo([[-1.0], [1.0]], ["a", "b"], [[0.0]], 2, 1e-9)
    assert pred[0] == "a" and abs(conf[0] - 0.5) <= TOL
    _, _, pred, _, _ = _knn_oraculo([[-2.0], [1.0]], ["a", "b"], [[0.0]], 2, 1e-9)
    assert pred[0] == "b"
    _, _, pred, _, _ = _knn_oraculo([[1.0], [3.0], [-1.5], [1.5]], ["a", "a", "b", "b"], [[0.0]], 4, 1e-15)
    assert pred[0] == "a"
    v, p, d = zip(*EJEMPLO_EVALUAR)
    r = _evaluar_oraculo(d, p, v)
    assert (r["identificacion"], r["acierto"], r["otro_aceptado"]) == (0.75, 0.5, 0.25)
    assert r["matriz"].tolist() == [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 3]]


# ===========================================================================
# 6.1 Configuración
# ===========================================================================

def test_config_claves_v4(cfg):
    knn = cfg["voz"]["knn"]
    assert list(knn["k_candidatos"]) == [3, 5, 7, 9], f"k_candidatos = {knn['k_candidatos']!r}"
    assert _es_float(knn["eps"]) and knn["eps"] == 1.0e-9, f"eps = {knn['eps']!r} (float 1e-9)"
    assert _es_float(knn["confianza_min"]) and knn["confianza_min"] == 0.60, \
        f"confianza_min = {knn['confianza_min']!r} (float 0.60)"
    assert list(knn["pca_componentes_candidatas"]) == [20, 30], \
        f"pca_componentes_candidatas = {knn['pca_componentes_candidatas']!r}"
    assert cfg["metas"]["voz_acierto_min"] == 0.90, f"voz_acierto_min = {cfg['metas']['voz_acierto_min']!r}"
    otro_max = cfg["metas"]["voz_otro_aceptado_max"]
    assert _es_float(otro_max) and otro_max == 0.05, f"voz_otro_aceptado_max = {otro_max!r} (float 0.05)"


def test_config_claves_que_usa_v4_existen(cfg):
    assert list(cfg["voz"]["comandos"]) == COMANDOS, f"voz.comandos = {cfg['voz']['comandos']!r}"
    assert cfg["voz"]["clase_rechazo"] == OTRO, f"voz.clase_rechazo = {cfg['voz']['clase_rechazo']!r}"
    assert "mismo_hablante" in cfg["voz"]["dataset"], "falta voz.dataset.mismo_hablante"
    for clave in ("voz", "modelos", "figuras"):
        assert clave in cfg["rutas"], f"falta rutas.{clave}"


def test_etiquetas_clasificacion(clf, cfg):
    obtenido = clf.etiquetas_clasificacion(cfg)
    assert list(obtenido) == ETIQUETAS, f"etiquetas_clasificacion = {obtenido!r}, esperado {ETIQUETAS!r}"


def test_constantes_del_modulo(clf):
    assert tuple(clf.ESQUEMAS_BLOQUES) == ESQUEMAS, f"ESQUEMAS_BLOQUES = {clf.ESQUEMAS_BLOQUES!r}"
    assert clf.PCA_SIN is None, f"PCA_SIN = {clf.PCA_SIN!r}"


# ===========================================================================
# 6.2 K-NN propio
# ===========================================================================

FORMAS_KNN = [(120, 5, 3, 11), (300, 20, 4, 12), (60, 2, 2, 13)]
IDS_FORMAS = [f"{n}x{d}_{c}clases" for n, d, c, _ in FORMAS_KNN]


@pytest.mark.parametrize("k", [1, 3, 5, 9])
@pytest.mark.parametrize("forma", FORMAS_KNN, ids=IDS_FORMAS)
def test_knn_vecinos_oraculo(knn_mod, forma, k):
    n, d, c, s = forma
    X, y, Q = _datos_gauss(n, d, c, s)
    dist, idx = knn_mod.KNN(k=k, eps=1e-9).ajustar(X, y).vecinos(Q)
    d_o, i_o, *_ = _knn_oraculo(X, y, Q, k, 1e-9)
    assert dist.shape == (len(Q), k) and idx.shape == (len(Q), k), \
        f"formas {dist.shape}, {idx.shape}; esperado {(len(Q), k)}"
    assert dist.dtype == np.float64, f"dtype de distancias {dist.dtype}"
    assert idx.dtype == np.int64, f"dtype de índices {idx.dtype}"
    err = float(np.max(np.abs(dist - d_o)))
    assert err <= TOL, f"distancias: error máximo {err:.3e} contra el oráculo"
    assert np.array_equal(idx, i_o), f"índices distintos en {int((idx != i_o).sum())} posiciones"


@pytest.mark.parametrize("k", [1, 3, 5, 9])
@pytest.mark.parametrize("forma", FORMAS_KNN, ids=IDS_FORMAS)
def test_knn_votar_oraculo(knn_mod, forma, k):
    n, d, c, s = forma
    X, y, Q = _datos_gauss(n, d, c, s)
    modelo = knn_mod.KNN(k=k, eps=1e-9).ajustar(X, y)
    pred, conf, dist = modelo.votar(Q)
    _, _, p_o, c_o, d_o = _knn_oraculo(X, y, Q, k, 1e-9)
    assert pred.shape == (len(Q),) and conf.shape == (len(Q),) and dist.shape == (len(Q),)
    assert conf.dtype == np.float64 and dist.dtype == np.float64, f"dtypes {conf.dtype}, {dist.dtype}"
    assert np.array_equal(pred.astype(str), p_o.astype(str)), \
        f"predicciones distintas: {pred[pred != p_o]} vs {p_o[pred != p_o]}"
    assert float(np.max(np.abs(conf - c_o))) <= TOL, f"confianza: error {np.max(np.abs(conf - c_o)):.3e}"
    assert float(np.max(np.abs(dist - d_o))) <= TOL, f"distancia: error {np.max(np.abs(dist - d_o)):.3e}"
    assert np.array_equal(np.asarray(modelo.predecir(Q)).astype(str), pred.astype(str)), \
        "predecir no coincide con la predicción de votar"


@pytest.mark.parametrize("k", [3, 5, 9])
@pytest.mark.parametrize("forma", FORMAS_KNN, ids=IDS_FORMAS)
def test_knn_contra_sklearn(knn_mod, forma, k):
    from sklearn.neighbors import KNeighborsClassifier

    n, d, c, s = forma
    X, y, Q = _datos_gauss(n, d, c, s)
    propio = knn_mod.KNN(k=k, eps=1e-9).ajustar(X, y)
    sk = KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="brute").fit(X, y)
    p_sk = sk.predict(Q)
    p = np.asarray(propio.predecir(Q)).astype(str)
    assert np.array_equal(p, p_sk.astype(str)), f"distinto a scikit-learn en {int((p != p_sk).sum())} filas"
    d_sk, _ = sk.kneighbors(Q)
    d_p, _ = propio.vecinos(Q)
    err = float(np.max(np.abs(d_p - d_sk)))
    assert err <= 1e-9, f"distancias contra kneighbors: error {err:.3e}"


def test_knn_confianza_en_rango_y_uno_si_vecinos_de_una_clase(knn_mod):
    X, y, Q = _datos_gauss(300, 20, 4, 12)
    _, conf, _ = knn_mod.KNN(k=5).ajustar(X, y).votar(Q)
    assert np.all(conf > 0) and np.all(conf <= 1.0 + TOL), f"confianza fuera de (0, 1]: {conf}"
    # Grupos muy separados: los 5 vecinos de cada consulta son de su misma clase.
    rng = np.random.default_rng(3)
    centros = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    Xs = np.repeat(centros, 10, axis=0) + rng.normal(0, 0.5, (30, 2))
    ys = np.repeat(np.array(["a", "b", "c"]), 10)
    pred, conf, _ = knn_mod.KNN(k=5).ajustar(Xs, ys).votar(centros + 0.1)
    assert list(pred) == ["a", "b", "c"], f"predicción {list(pred)}"
    assert np.all(np.abs(conf - 1.0) <= TOL), f"confianza {conf}, esperado 1.0"


def test_knn_k1_vecino_mas_cercano(knn_mod):
    X, y, Q = _datos_gauss(120, 5, 3, 11)
    pred, conf, dist = knn_mod.KNN(k=1).ajustar(X, y).votar(Q)
    D = cdist(Q, X)
    j = np.argmin(D, axis=1)
    assert np.array_equal(pred.astype(str), y[j]), "con k = 1 la predicción no es la del más cercano"
    assert np.all(np.abs(conf - 1.0) <= TOL), f"con k = 1 la confianza debería ser 1.0: {conf}"
    err = float(np.max(np.abs(dist - D[np.arange(len(Q)), j])))
    assert err <= TOL, f"con k = 1 la distancia debería ser d_1 (error {err:.3e})"


def test_knn_no_depende_del_orden_de_entrenamiento(knn_mod):
    X, y, Q = _datos_gauss(300, 20, 4, 12)
    perm = np.random.default_rng(5).permutation(len(X))
    p1, c1, d1 = knn_mod.KNN(k=5).ajustar(X, y).votar(Q)
    p2, c2, d2 = knn_mod.KNN(k=5).ajustar(X[perm], y[perm]).votar(Q)
    assert np.array_equal(p1, p2), "la predicción cambió al permutar las filas de entrenamiento"
    assert np.max(np.abs(c1 - c2)) <= TOL and np.max(np.abs(d1 - d2)) <= TOL, \
        "la confianza o la distancia cambiaron al permutar las filas de entrenamiento"


def test_knn_vector_igual_a_su_fila(knn_mod):
    X, y, Q = _datos_gauss(120, 5, 3, 11)
    modelo = knn_mod.KNN(k=5).ajustar(X, y)
    P, C, D = modelo.votar(Q)
    dist_l, idx_l = modelo.vecinos(Q)
    for i in (0, 7, 13):
        dv, iv = modelo.vecinos(Q[i])
        assert dv.shape == (5,) and iv.shape == (5,), f"vecinos de un vector: formas {dv.shape}, {iv.shape}"
        assert np.max(np.abs(dv - dist_l[i])) <= TOL and np.array_equal(iv, idx_l[i])
        p, c, d = modelo.votar(Q[i])
        assert isinstance(p, str), f"votar de un vector: la etiqueta es {type(p)}"
        assert _es_float(c) and _es_float(d), f"votar de un vector: tipos {type(c)}, {type(d)}"
        assert p == P[i] and abs(c - C[i]) <= TOL and abs(d - D[i]) <= TOL, \
            f"vector {i}: ({p}, {c}, {d}) vs fila ({P[i]}, {C[i]}, {D[i]})"
        pr = modelo.predecir(Q[i])
        assert isinstance(pr, str) and pr == P[i], f"predecir de un vector: {pr!r} vs {P[i]!r}"


def test_knn_ajustar_devuelve_self_atributos_y_no_modifica(knn_mod):
    X, y, Q = _datos_gauss(120, 5, 3, 11)
    X32 = X.astype(np.float32)
    X_copia, y_copia = X32.copy(), y.copy()
    modelo = knn_mod.KNN(k=3)
    assert modelo.ajustar(X32, y) is modelo, "ajustar debe devolver el mismo objeto"
    assert np.array_equal(X32, X_copia) and np.array_equal(y, y_copia), "ajustar modificó sus argumentos"
    assert modelo.X_entrenamiento.dtype == np.float64 and modelo.X_entrenamiento.shape == X.shape
    assert np.array_equal(modelo.X_entrenamiento, X32.astype(np.float64))
    assert np.array_equal(np.asarray(modelo.etiquetas), y)
    assert np.array_equal(np.asarray(modelo.clases), np.unique(y)), f"clases = {modelo.clases!r}"
    X32[0, 0] += 1000.0  # X_entrenamiento es una copia
    assert modelo.X_entrenamiento[0, 0] != X32[0, 0], "X_entrenamiento no es una copia de X"
    a = modelo.votar(Q)
    b = modelo.votar(Q)
    assert all(np.array_equal(u, v) for u, v in zip(a, b)), "dos llamadas no dan resultados idénticos"


def test_knn_empate_de_votos_gana_el_primero_de_clases(knn_mod):
    pred, conf, _ = knn_mod.KNN(k=2).ajustar(np.array([[-1.0], [1.0]]), np.array(["a", "b"])).votar(np.array([0.0]))
    assert pred == "a", f"votos iguales: ganó {pred!r}, debe ganar 'a' (primero de clases)"
    assert abs(conf - 0.5) <= TOL, f"confianza {conf}, esperado 0.5"
    pred, _, _ = knn_mod.KNN(k=2).ajustar(np.array([[-2.0], [1.0]]), np.array(["a", "b"])).votar(np.array([0.0]))
    assert pred == "b", f"con [[-2], [1]] ganó {pred!r}, debe ganar 'b'"


def test_knn_empate_de_votos_desempata_por_distancia_minima(knn_mod):
    # a: vecinos a 1 y 3 (voto 4/3, mín 1, media 2); b: dos vecinos a 1.5 (voto 4/3, mín 1.5, media 1.5).
    X = np.array([[1.0], [3.0], [-1.5], [1.5]])
    y = np.array(["a", "a", "b", "b"])
    pred, conf, dist = knn_mod.KNN(k=4, eps=1e-15).ajustar(X, y).votar(np.array([0.0]))
    assert pred == "a", f"el desempate es por la distancia MÍNIMA: ganó {pred!r}, esperado 'a'"
    assert abs(dist - 2.0) <= TOL, f"distancia (media de los vecinos de 'a') = {dist}, esperado 2.0"
    assert abs(conf - 0.5) <= 1e-9, f"confianza {conf}, esperado 0.5"


@pytest.mark.parametrize("k", [0, -1, 2.5, True])
def test_knn_k_invalido(knn_mod, k):
    with pytest.raises(ValueError):
        knn_mod.KNN(k=k)


@pytest.mark.parametrize("eps", [0.0, -1e-9, float("nan"), float("inf"), 2e-3])
def test_knn_eps_invalido(knn_mod, eps):
    with pytest.raises(ValueError):
        knn_mod.KNN(k=3, eps=eps)


def _x_invalido(caso):
    X, y, _ = _datos_gauss(30, 4, 3, 1)
    if caso == "1d":
        return X[:, 0], y
    if caso == "3d":
        return X.reshape(30, 2, 2), y
    if caso == "nan":
        X[3, 1] = np.nan
    if caso == "inf":
        X[3, 1] = np.inf
    if caso == "n_menor_k":
        return X[:2], y[:2]
    if caso == "etiquetas_largo":
        return X, y[:-1]
    return X, y


@pytest.mark.parametrize("caso", ["1d", "3d", "nan", "inf", "n_menor_k", "etiquetas_largo"])
def test_knn_ajustar_invalido(knn_mod, caso):
    X, y = _x_invalido(caso)
    with pytest.raises(ValueError):
        knn_mod.KNN(k=3).ajustar(X, y)


@pytest.mark.parametrize("metodo", ["vecinos", "votar", "predecir"])
@pytest.mark.parametrize("caso", ["otro_d", "otro_d_vector", "3d", "nan"])
def test_knn_consulta_invalida(knn_mod, metodo, caso):
    X, y, Q = _datos_gauss(30, 4, 3, 1)
    modelo = knn_mod.KNN(k=3).ajustar(X, y)
    if caso == "otro_d":
        Q = Q[:, :3]
    elif caso == "otro_d_vector":
        Q = Q[0, :3]
    elif caso == "3d":
        Q = Q[:24].reshape(6, 4, 4)
    else:
        Q = Q.copy()
        Q[2, 2] = np.nan
    with pytest.raises(ValueError):
        getattr(modelo, metodo)(Q)


@pytest.mark.parametrize("metodo", ["vecinos", "votar", "predecir"])
def test_knn_antes_de_ajustar(knn_mod, metodo):
    with pytest.raises(RuntimeError):
        getattr(knn_mod.KNN(k=3), metodo)(np.zeros((2, 4)))


@pytest.mark.parametrize("ruta", ["src/core_ml/knn.py", "src/voz/clasificacion.py"])
def test_sin_sklearn_en_src(ruta):
    archivo = RAIZ / ruta
    assert archivo.is_file(), f"no existe {ruta}"
    assert "sklearn" not in archivo.read_text(encoding="utf-8"), f"{ruta} contiene 'sklearn'"


# ===========================================================================
# 6.3 Esquemas de bloques, reponderación y normalización L2
# ===========================================================================

@pytest.mark.parametrize("esquema", ESQUEMAS)
def test_factores_esquema_config_real(clf, cfg, esquema):
    esperado = {
        "contrato": np.ones(5),
        "sin_dur_zcr": np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
        "z_columna": np.sqrt(np.array([130.0, 130.0, 10.0, 10.0, 1.0])),
    }[esquema]
    f = clf.factores_esquema(esquema, cfg)
    assert isinstance(f, np.ndarray) and f.shape == (5,) and f.dtype == np.float64, \
        f"factores: tipo {type(f)}, forma {getattr(f, 'shape', None)}, dtype {getattr(f, 'dtype', None)}"
    assert np.max(np.abs(f - esperado)) <= TOL, f"factores_esquema({esquema!r}) = {f}, esperado {esperado}"


@pytest.mark.parametrize("cambio", [("tramos", 4), ("n_mfcc", 12)])
def test_factores_esquema_otra_config(clf, cfg, cambio):
    clave, valor = cambio
    ruta = ("voz", "tramos") if clave == "tramos" else ("voz", "mfcc", "n_mfcc")
    cfg2 = _cfg_con(cfg, ruta, valor)
    K, n = int(cfg2["voz"]["tramos"]), int(cfg2["voz"]["mfcc"]["n_mfcc"])
    largos2 = [2 * K * n, 2 * K * n, 2 * K, 2 * K, 1]  # 4.3 de V3
    for esquema in ESQUEMAS:
        f = clf.factores_esquema(esquema, cfg2)
        esperado = _factores_oraculo(esquema, largos2)
        assert np.max(np.abs(f - esperado)) <= TOL, \
            f"{clave}={valor}, {esquema}: {f}, esperado {esperado}"


def test_factores_esquema_desconocido(clf, cfg):
    with pytest.raises(ValueError):
        clf.factores_esquema("todo_igual", cfg)


def test_reponderar_bloques_esquemas(clf, cfg, largos):
    X = np.random.default_rng(8).normal(size=(7, 281))
    X_copia = X.copy()
    r = clf.reponderar_bloques(X, clf.factores_esquema("contrato", cfg), cfg)
    assert r.shape == (7, 281) and r.dtype == np.float64 and np.max(np.abs(r - X)) <= TOL, \
        "con 'contrato' X debe quedar igual"
    r = clf.reponderar_bloques(X, clf.factores_esquema("sin_dur_zcr", cfg), cfg)
    assert r.shape == (7, 270), f"'sin_dur_zcr': forma {r.shape}, esperado (7, 270)"
    assert np.max(np.abs(r - X[:, :270])) <= TOL, "'sin_dur_zcr': las primeras 270 columnas deben quedar intactas"
    f = clf.factores_esquema("z_columna", cfg)
    r = clf.reponderar_bloques(X, f, cfg)
    esperado = _reponderar_oraculo(X, _factores_oraculo("z_columna", largos), largos)
    assert r.shape == (7, 281) and np.max(np.abs(r - esperado)) <= TOL, "'z_columna': bloques mal multiplicados"
    assert np.array_equal(X, X_copia), "reponderar_bloques modificó X"


def test_reponderar_bloques_factores_generales_y_vector(clf, cfg, largos):
    X = np.random.default_rng(9).normal(size=(5, 281))
    f = np.array([2.0, 0.0, 0.5, 0.0, 3.0])  # descarta delta y zcr
    r = clf.reponderar_bloques(X, f, cfg)
    esperado = _reponderar_oraculo(X, f, largos)
    assert r.shape == esperado.shape == (5, 141), f"forma {r.shape}, esperado {esperado.shape}"
    assert np.max(np.abs(r - esperado)) <= TOL, "bloques con factor general mal reponderados"
    rv = clf.reponderar_bloques(X[2], f, cfg)
    assert rv.shape == (141,) and np.max(np.abs(rv - r[2])) <= TOL, "un vector no da lo mismo que su fila"


@pytest.mark.parametrize("caso", ["otro_d", "nan", "inf", "factor_negativo"])
def test_reponderar_bloques_invalido(clf, cfg, caso):
    X = np.random.default_rng(10).normal(size=(4, 281))
    f = np.ones(5)
    if caso == "otro_d":
        X = X[:, :280]
    elif caso == "nan":
        X[1, 5] = np.nan
    elif caso == "inf":
        X[1, 5] = np.inf
    else:
        f = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        clf.reponderar_bloques(X, f, cfg)


def test_reponderar_z_columna_varianza_unitaria(clf, cfg, v3, largos, dataset_chico):
    npz = _cargar_npz(dataset_chico / "caracteristicas.npz")
    tr = npz["particion"].astype(str) == "train"
    bloques = {b: npz[b][tr] for b in BLOQUES}
    esc = v3.ajustar_escalador(bloques)
    # Precondición del dataset chico: sin NaN y con todas las columnas usadas (peso 1/√d_b exacto).
    sin_nan = all(not np.isnan(bloques[b]).any() for b in BLOQUES)
    todas = all(bool(np.all(esc.columnas[b])) for b in BLOQUES)
    assert sin_nan and todas, "precondición del test: el dataset chico tiene NaN o columnas sin usar"
    X_tr = npz["X"][tr]
    var_z = np.var(clf.reponderar_bloques(X_tr, clf.factores_esquema("z_columna", cfg), cfg), axis=0, ddof=1)
    err = float(np.max(np.abs(var_z - 1.0)))
    assert err <= 1e-9, f"con 'z_columna' cada columna debe tener varianza 1 (error {err:.3e})"
    var_c = np.var(clf.reponderar_bloques(X_tr, clf.factores_esquema("contrato", cfg), cfg), axis=0, ddof=1)
    esperado = np.concatenate([np.full(l, 1.0 / l) for l in largos])
    err = float(np.max(np.abs(var_c - esperado)))
    assert err <= 1e-9, f"con 'contrato' la varianza debe ser 1/d_b por bloque (error {err:.3e})"


def test_normalizar_l2(clf):
    X = np.random.default_rng(11).normal(size=(6, 9)) * 5.0
    X[3] = 0.0
    r = clf.normalizar_l2(X)
    normas = np.linalg.norm(r, axis=1)
    assert r.shape == X.shape and r.dtype == np.float64
    assert np.max(np.abs(np.delete(normas, 3) - 1.0)) <= TOL, f"normas {normas}"
    assert np.all(r[3] == 0.0) and not np.isnan(r).any(), "una fila de ceros debe quedar en ceros, sin NaN"
    v = clf.normalizar_l2(X[1])
    assert v.shape == (9,) and abs(np.linalg.norm(v) - 1.0) <= TOL and np.max(np.abs(v - r[1])) <= TOL


@pytest.mark.parametrize("caso", ["nan", "inf", "3d", "piso_cero", "piso_negativo"])
def test_normalizar_l2_invalido(clf, caso):
    X = np.ones((3, 4))
    piso = 1e-12
    if caso == "nan":
        X[0, 0] = np.nan
    elif caso == "inf":
        X[0, 0] = np.inf
    elif caso == "3d":
        X = np.ones((2, 2, 3))
    elif caso == "piso_cero":
        piso = 0.0
    else:
        piso = -1.0
    with pytest.raises(ValueError):
        clf.normalizar_l2(X, piso)


def _modelo_a_mano(clf, cfg, largos, par, con_pca: bool):
    from src.core_ml.pca import PCA

    rng = np.random.default_rng(12)
    X_tr = rng.normal(size=(40, 281))
    y_tr = np.array([ETIQUETAS[i % 4] for i in range(40)])
    esquema = "sin_dur_zcr" if con_pca else "z_columna"
    f = _factores_oraculo(esquema, largos)
    R = _reponderar_oraculo(X_tr, f, largos)
    pca = None
    if con_pca:
        pca = PCA(20).ajustar(R)
        media, comp, n_comp = pca.media, pca.componentes, 20
        Z = _normalizar_oraculo((R - media) @ comp.T)
    else:
        media, comp, n_comp = np.zeros(0), np.zeros((0, 0)), None
        Z = _normalizar_oraculo(R)
    modelo = clf.ModeloVoz(esquema=esquema, factores=f, n_componentes=n_comp, pca_media=media,
                           pca_componentes=comp, k=3, eps=par["eps"], confianza_min=par["cmin"],
                           tau=0.8, entrenamiento=Z, etiquetas=y_tr, escalador=None)
    return modelo, pca, f


@pytest.mark.parametrize("con_pca", [True, False], ids=["con_pca", "sin_pca"])
def test_modelo_voz_transformar_cadena(clf, cfg, largos, parametros, con_pca):
    modelo, pca, f = _modelo_a_mano(clf, cfg, largos, parametros, con_pca)
    Xq = np.random.default_rng(13).normal(size=(9, 281))
    R = _reponderar_oraculo(Xq, f, largos)
    esperado = _normalizar_oraculo((R - modelo.pca_media) @ modelo.pca_componentes.T if con_pca else R)
    obtenido = modelo.transformar(Xq, cfg)
    assert obtenido.shape == esperado.shape, f"forma {obtenido.shape}, esperado {esperado.shape}"
    assert np.max(np.abs(obtenido - esperado)) <= 1e-9, "transformar no es la cadena de 4.2"
    if con_pca:
        con_h3 = _normalizar_oraculo(pca.transformar(R))
        assert np.max(np.abs(obtenido - con_h3)) <= 1e-9, "transformar no coincide con PCA.transformar de H3"
    v = modelo.transformar(Xq[4], cfg)
    assert v.shape == (esperado.shape[1],) and np.max(np.abs(v - obtenido[4])) <= 1e-9, \
        "un vector no da lo mismo que su fila"


@pytest.mark.parametrize("con_pca", [True, False], ids=["con_pca", "sin_pca"])
def test_modelo_voz_votar_y_decidir(clf, cfg, largos, parametros, con_pca):
    modelo, _, _ = _modelo_a_mano(clf, cfg, largos, parametros, con_pca)
    Xq = np.random.default_rng(14).normal(size=(9, 281))
    Zq = modelo.transformar(Xq, cfg)
    _, _, p_o, c_o, d_o = _knn_oraculo(modelo.entrenamiento, modelo.etiquetas, Zq, 3, parametros["eps"])
    # τ en la mediana de las distancias, para que haya filas aceptadas y rechazadas.
    modelo = dataclasses.replace(modelo, tau=float(np.median(d_o)))
    pred, conf, dist = modelo.votar(Xq, cfg)
    assert np.array_equal(np.asarray(pred).astype(str), p_o.astype(str)), "ModeloVoz.votar: predicción"
    assert np.max(np.abs(conf - c_o)) <= 1e-9 and np.max(np.abs(dist - d_o)) <= 1e-9, \
        "ModeloVoz.votar: confianza o distancia"
    dec = modelo.decidir(Xq, cfg)
    esperado = _decidir_oraculo(p_o, c_o, d_o, modelo.tau, parametros["cmin"])
    assert list(np.asarray(dec).astype(str)) == list(esperado), f"ModeloVoz.decidir: {dec} vs {esperado}"
    d1 = modelo.decidir(Xq[0], cfg)
    assert isinstance(d1, str) and d1 == esperado[0], f"decidir de un vector: {d1!r}, esperado {esperado[0]!r}"


# ===========================================================================
# 6.4 Selección de hiperparámetros y elegir_mejor
# ===========================================================================

def _fila(esquema, n_comp, k, n_tau5, n_c60, n_4cl, descartada=False, tau=0.5) -> dict:
    return {"esquema": esquema, "n_componentes": n_comp, "k": k, "n_tau5": n_tau5, "n_c60": n_c60,
            "n_4cl": n_4cl, "tau_cv": float("nan") if descartada else tau, "descartada": descartada}


TABLAS_ELEGIR = {
    "mayor_suma": ([_fila("contrato", 20, 3, 100, 100, 230), _fila("z_columna", None, 9, 101, 100, 200)],
                   ("z_columna", None, 9)),
    "seccion_8_empate_411": ([_fila("sin_dur_zcr", 30, 3, 187, 221, 221, tau=0.7354),
                              _fila("sin_dur_zcr", 30, 5, 195, 216, 221, tau=0.7672),
                              _fila("z_columna", 20, 3, 187, 221, 221, tau=0.6519),
                              _fila("z_columna", 20, 5, 190, 221, 224, tau=0.6900),
                              _fila("z_columna", 20, 7, 192, 215, 225, tau=0.7173)],
                             ("z_columna", 20, 5)),
    "misma_suma_mayor_n4cl": ([_fila("contrato", 20, 3, 100, 100, 200), _fila("z_columna", None, 9, 99, 101, 201)],
                              ("z_columna", None, 9)),
    "luego_esquema": ([_fila("z_columna", 20, 3, 50, 60, 70), _fila("contrato", None, 9, 50, 60, 70),
                       _fila("sin_dur_zcr", 20, 3, 50, 60, 70)],
                      ("contrato", None, 9)),
    "luego_pca": ([_fila("sin_dur_zcr", None, 3, 50, 60, 70), _fila("sin_dur_zcr", 30, 3, 50, 60, 70),
                   _fila("sin_dur_zcr", 20, 9, 50, 60, 70)],
                  ("sin_dur_zcr", 20, 9)),
    "pca_30_antes_que_sin": ([_fila("z_columna", None, 3, 50, 60, 70), _fila("z_columna", 30, 7, 50, 60, 70)],
                             ("z_columna", 30, 7)),
    "luego_k_menor": ([_fila("z_columna", 30, 9, 50, 60, 70), _fila("z_columna", 30, 5, 50, 60, 70),
                       _fila("z_columna", 30, 7, 50, 60, 70)],
                      ("z_columna", 30, 5)),
    "descartadas_no_cuentan": ([_fila("contrato", 20, 3, 300, 300, 300, descartada=True),
                                _fila("z_columna", 20, 5, 10, 10, 10)],
                               ("z_columna", 20, 5)),
}


@pytest.mark.parametrize("nombre", list(TABLAS_ELEGIR))
def test_elegir_mejor_criterio_suma_y_desempates(clf, nombre):
    tabla, (e, n, k) = TABLAS_ELEGIR[nombre]
    for orden in (tabla, tabla[::-1]):
        copia = copy.deepcopy(orden)
        mejor = clf.elegir_mejor(copia)
        assert set(mejor) == CLAVES_MEJOR, f"claves {sorted(mejor)}, esperado {sorted(CLAVES_MEJOR)}"
        assert (mejor["esquema"], mejor["n_componentes"], mejor["k"]) == (e, n, k), \
            f"{nombre}: ganó {mejor}, esperado {(e, n, k)}"
        assert _diferencias(copia, orden, TOL) == [], "elegir_mejor modificó la tabla"


@pytest.mark.parametrize("tabla", [[], [_fila("contrato", 20, 3, 0, 0, 0, descartada=True),
                                         _fila("z_columna", None, 5, 0, 0, 0, descartada=True)]],
                         ids=["vacia", "todas_descartadas"])
def test_elegir_mejor_sin_candidatas(clf, tabla):
    with pytest.raises(ValueError):
        clf.elegir_mejor(tabla)


def test_seleccionar_hiperparametros_estructura_tabla(cfg, resultado_cv):
    mejor, tabla = resultado_cv
    grilla = _grilla(cfg)
    assert len(tabla) == 36 == len(grilla), f"{len(tabla)} filas, esperado 36"
    for i, (fila, (e, n, k)) in enumerate(zip(tabla, grilla)):
        assert set(fila) == CLAVES_TABLA, f"fila {i}: claves {sorted(fila)}"
        assert (fila["esquema"], fila["n_componentes"], fila["k"]) == (e, n, k), \
            f"fila {i}: {(fila['esquema'], fila['n_componentes'], fila['k'])}, esperado {(e, n, k)} (orden de 4.3)"
        for c in ("n_tau5", "n_c60", "n_4cl", "k"):
            assert _es_int(fila[c]), f"fila {i}: {c} = {fila[c]!r} debe ser int de Python"
        assert fila["n_componentes"] is None or _es_int(fila["n_componentes"])
        assert _es_float(fila["tau_cv"]) and isinstance(fila["descartada"], (bool, np.bool_))


def test_seleccionar_hiperparametros_mejor_es_elegir_mejor(clf, resultado_cv):
    mejor, tabla = resultado_cv
    assert set(mejor) == CLAVES_MEJOR, f"claves de mejor {sorted(mejor)}"
    assert mejor == clf.elegir_mejor(tabla), f"mejor {mejor} != elegir_mejor(tabla)"
    assert mejor == _elegir_oraculo(tabla), f"mejor {mejor}, el criterio de 4.3 da {_elegir_oraculo(tabla)}"


def test_seleccionar_hiperparametros_conteos_ordenados(resultado_cv):
    _, tabla = resultado_cv
    for f in tabla:
        if f["descartada"]:
            assert math.isnan(f["tau_cv"]) and (f["n_tau5"], f["n_c60"], f["n_4cl"]) == (0, 0, 0), \
                f"descartada con tau_cv {f['tau_cv']} y conteos {(f['n_tau5'], f['n_c60'], f['n_4cl'])}"
        else:
            assert f["n_tau5"] <= f["n_c60"] <= f["n_4cl"], f"conteos desordenados en {f}"
            assert math.isfinite(f["tau_cv"]), f"tau_cv no finito en {f}"


COMBOS_ORACULO = [("sin_dur_zcr", None, 3), ("z_columna", 20, 5), ("contrato", 30, 7)]


@pytest.mark.parametrize("combo", COMBOS_ORACULO, ids=[f"{e}-{n}-k{k}" for e, n, k in COMBOS_ORACULO])
def test_seleccionar_hiperparametros_oraculo(largos, parametros, datos_cv, resultado_cv, combo):
    _, tabla = resultado_cv
    e, n, k = combo
    d = datos_cv
    esperado = _cv_oraculo(d["X"], d["etiquetas"], d["personas"], d["evaluable"],
                           d["n_otro_no_validos"], e, n, k, largos, parametros)
    fila = next(f for f in tabla if (f["esquema"], f["n_componentes"], f["k"]) == combo)
    assert bool(fila["descartada"]) == esperado["descartada"], f"descartada {fila['descartada']} vs oráculo"
    obt = (fila["n_tau5"], fila["n_c60"], fila["n_4cl"])
    esp = (esperado["n_tau5"], esperado["n_c60"], esperado["n_4cl"])
    assert obt == esp, f"{combo}: (n_tau5, n_c60, n_4cl) = {obt}, el oráculo da {esp}"
    if not esperado["descartada"]:
        assert abs(fila["tau_cv"] - esperado["tau_cv"]) <= 1e-9, \
            f"{combo}: tau_cv {fila['tau_cv']!r}, oráculo {esperado['tau_cv']!r}"


def test_seleccionar_hiperparametros_pliegues_por_persona(clf, cfg, largos, parametros, datos_cv):
    """s06 (solo filas no evaluables) y s07 ("otro" evaluables sin comandos) no generan pliegue,
    pero sus filas entrenan a los demás; los "otro" de s07 nunca se consultan."""
    d = _agregar_personas_extra(datos_cv, largos)
    _, tabla = clf.seleccionar_hiperparametros(d["X"], d["etiquetas"], d["personas"], d["evaluable"],
                                               d["n_otro_no_validos"], cfg)
    for combo in (("z_columna", 20, 5), ("sin_dur_zcr", 30, 3)):
        esp = _cv_oraculo(d["X"], d["etiquetas"], d["personas"], d["evaluable"], d["n_otro_no_validos"],
                          *combo, largos, parametros)
        fila = next(f for f in tabla if (f["esquema"], f["n_componentes"], f["k"]) == combo)
        obt = (fila["n_tau5"], fila["n_c60"], fila["n_4cl"])
        assert obt == (esp["n_tau5"], esp["n_c60"], esp["n_4cl"]), \
            f"{combo}: {obt}, oráculo {(esp['n_tau5'], esp['n_c60'], esp['n_4cl'])} (pliegues = personas con comandos evaluables)"
        if not esp["descartada"]:
            assert abs(fila["tau_cv"] - esp["tau_cv"]) <= 1e-9, \
                f"{combo}: tau_cv {fila['tau_cv']!r}, oráculo {esp['tau_cv']!r}"


def test_seleccionar_hiperparametros_descarte_por_pocas_filas(clf, cfg, largos, parametros, datos_chicos):
    d = datos_chicos
    mejor, tabla = clf.seleccionar_hiperparametros(d["X"], d["etiquetas"], d["personas"], d["evaluable"], 0, cfg)
    for fila in tabla:
        if fila["n_componentes"] == 30:
            assert fila["descartada"], f"con 26 filas por pliegue, 30 componentes deben descartarse: {fila}"
        esp = _cv_oraculo(d["X"], d["etiquetas"], d["personas"], d["evaluable"], 0,
                          fila["esquema"], fila["n_componentes"], fila["k"], largos, parametros)
        assert bool(fila["descartada"]) == esp["descartada"], f"descartada distinta del oráculo en {fila}"
        assert (fila["n_tau5"], fila["n_c60"], fila["n_4cl"]) == (esp["n_tau5"], esp["n_c60"], esp["n_4cl"]), \
            f"conteos distintos del oráculo en {fila}"
    assert mejor["n_componentes"] != 30, f"la elegida usa 30 componentes: {mejor}"


def test_seleccionar_hiperparametros_todas_descartadas(clf, cfg):
    X = np.random.default_rng(15).normal(size=(4, 281))
    y = np.array(["reconocer", "otro", "reconocer", "otro"])
    P = np.array(["u1", "u1", "u2", "u2"])
    with pytest.raises(ValueError):  # cada pliegue entrena con 2 filas: k ≥ 3 > 2 en toda la grilla
        clf.seleccionar_hiperparametros(X, y, P, np.ones(4, dtype=bool), 0, cfg)


def test_seleccionar_hiperparametros_determinista_y_no_modifica(clf, cfg, datos_cv, resultado_cv):
    d = datos_cv
    args = [d["X"].copy(), d["etiquetas"].copy(), d["personas"].copy(), d["evaluable"].copy()]
    mejor, tabla = clf.seleccionar_hiperparametros(*args, d["n_otro_no_validos"], cfg)
    for a, original in zip(args, (d["X"], d["etiquetas"], d["personas"], d["evaluable"])):
        assert np.array_equal(a, original), "seleccionar_hiperparametros modificó sus argumentos"
    assert mejor == resultado_cv[0], f"mejor cambió entre corridas: {mejor} vs {resultado_cv[0]}"
    assert len(tabla) == len(resultado_cv[1]), "la tabla cambió de largo entre corridas"
    difs = [x for a, b in zip(tabla, resultado_cv[1]) for x in _diferencias(a, b)]
    assert difs == [], f"tabla distinta entre corridas: {difs[:5]}"


def _args_seleccion_invalidos(caso, d):
    X, y, P, E, n = d["X"].copy(), d["etiquetas"].copy(), d["personas"].copy(), d["evaluable"].copy(), 0
    if caso == "largos_distintos":
        y = y[:-1]
    elif caso == "evaluable_largo":
        E = E[:-1]
    elif caso == "otro_ancho":
        X = X[:, :280]
    elif caso == "x_no_finito":
        X[0, 0] = np.nan
    elif caso == "no_validos_negativo":
        n = -1
    elif caso == "no_validos_no_entero":
        n = 1.5
    elif caso == "etiqueta_fuera":
        y = y.astype("<U12")
        y[0] = "saludo"
    elif caso == "sin_personas_evaluables":
        E = np.zeros(len(E), dtype=bool)
    return X, y, P, E, n


@pytest.mark.parametrize("caso", ["largos_distintos", "evaluable_largo", "otro_ancho", "x_no_finito",
                                  "no_validos_negativo", "no_validos_no_entero", "etiqueta_fuera",
                                  "sin_personas_evaluables"])
def test_seleccionar_hiperparametros_invalido(clf, cfg, datos_chicos, caso):
    X, y, P, E, n = _args_seleccion_invalidos(caso, datos_chicos)
    with pytest.raises(ValueError):
        clf.seleccionar_hiperparametros(X, y, P, E, n, cfg)


# ===========================================================================
# 6.5 Calibración del umbral y decisión
# ===========================================================================

@pytest.mark.parametrize("caso", EJEMPLOS_CALIBRACION, ids=[e[0] for e in EJEMPLOS_CALIBRACION])
def test_calibrar_umbral_ejemplos(clf, cfg, caso):
    _, nombre, otro_max, n_nv, esperado = caso
    cfg_c = cfg if otro_max == 0.05 else _cfg_con(cfg, ("metas", "voz_otro_aceptado_max"), otro_max)
    d, c, p, v = _columnas(_filas_ejemplo(nombre))
    tau = clf.calibrar_umbral(d, c, p, v, cfg_c, n_nv)
    assert _es_float(tau), f"τ* debe ser float, es {type(tau)}"
    assert abs(tau - esperado) <= TOL, f"{caso[0]}: τ* = {tau!r}, esperado {esperado!r}"


@pytest.mark.parametrize("caso", EJEMPLOS_CALIBRACION, ids=[e[0] for e in EJEMPLOS_CALIBRACION])
def test_calibrar_umbral_no_depende_del_orden(clf, cfg, caso):
    _, nombre, otro_max, n_nv, esperado = caso
    cfg_c = cfg if otro_max == 0.05 else _cfg_con(cfg, ("metas", "voz_otro_aceptado_max"), otro_max)
    filas = _filas_ejemplo(nombre)
    for semilla in (1, 2, 3):
        perm = np.random.default_rng(semilla).permutation(len(filas))
        d, c, p, v = _columnas([filas[i] for i in perm])
        tau = clf.calibrar_umbral(d, c, p, v, cfg_c, n_nv)
        assert abs(tau - esperado) <= TOL, f"con las filas reordenadas τ* = {tau!r}, esperado {esperado!r}"


def _args_calibrar_invalidos(caso, cfg):
    d, c, p, v = _columnas(FILAS_A)
    cfg_c, n_nv = cfg, 0
    if caso == "largos_distintos":
        d = d[:-1]
    elif caso == "nan_distancia":
        d[0] = np.nan
    elif caso == "nan_confianza":
        c[1] = np.nan
    elif caso == "verdad_fuera":
        v = v.astype("<U12")
        v[0] = "saludo"
    elif caso == "prediccion_fuera":
        p = p.astype("<U12")
        p[0] = "saludo"
    elif caso == "otro_max_0":
        cfg_c = _cfg_con(cfg, ("metas", "voz_otro_aceptado_max"), 0.0)
    elif caso == "otro_max_1":
        cfg_c = _cfg_con(cfg, ("metas", "voz_otro_aceptado_max"), 1.0)
    elif caso == "no_validos_negativo":
        n_nv = -1
    elif caso == "no_validos_no_entero":
        n_nv = 2.5
    elif caso == "sin_comandos":
        d, c, p, v = _columnas(FILAS_A[4:])
    elif caso == "comandos_con_confianza_baja":
        c[:4] = 0.5
    elif caso == "comandos_por_encima_de_b":
        d[:4] = [2.0, 2.1, 2.2, 2.3]  # b = 1.00 y ningún comando aceptable por debajo
    return d, c, p, v, cfg_c, n_nv


@pytest.mark.parametrize("caso", ["largos_distintos", "nan_distancia", "nan_confianza", "verdad_fuera",
                                  "prediccion_fuera", "otro_max_0", "otro_max_1", "no_validos_negativo",
                                  "no_validos_no_entero", "sin_comandos", "comandos_con_confianza_baja",
                                  "comandos_por_encima_de_b"])
def test_calibrar_umbral_invalido(clf, cfg, caso):
    d, c, p, v, cfg_c, n_nv = _args_calibrar_invalidos(caso, cfg)
    with pytest.raises(ValueError):
        clf.calibrar_umbral(d, c, p, v, cfg_c, n_nv)


def test_decidir_ejemplo_A(clf, cfg):
    d, c, p, _ = _columnas(FILAS_A)
    dec = clf.decidir(p, c, d, 0.85, cfg)
    esperado = ["reconocer", "identificar", "salir", "reconocer", "otro", "otro", "otro", "otro"]
    assert isinstance(dec, np.ndarray) and dec.dtype.kind == "U", f"decidir devuelve {type(dec)} {getattr(dec, 'dtype', '')}"
    assert list(dec) == esperado, f"decisiones {list(dec)}, esperado {esperado}"


def test_decidir_bordes_y_otro(clf, cfg):
    cmin = float(cfg["voz"]["knn"]["confianza_min"])
    p = np.array(["salir", "identificar", "otro", "reconocer", "reconocer"])
    c = np.array([cmin, 0.99, 1.0, np.nextafter(cmin, 0.0), 0.9])
    d = np.array([0.3, 0.85, 0.0, 0.1, np.nextafter(0.85, 1.0)])
    dec = clf.decidir(p, c, d, 0.85, cfg)
    esperado = ["salir", "identificar", "otro", "otro", "otro"]
    assert list(dec) == esperado, \
        f"{list(dec)}, esperado {esperado} (confianza = cmin y distancia = τ se aceptan; 'otro' nunca es comando)"


def test_decidir_escalares(clf, cfg):
    r = clf.decidir("salir", 0.9, 0.2, 0.85, cfg)
    assert isinstance(r, str) and r == "salir", f"decidir con escalares: {r!r}"
    r = clf.decidir("salir", 0.9, 0.9, 0.85, cfg)
    assert isinstance(r, str) and r == "otro", f"decidir con escalares por encima de τ: {r!r}"


def test_decidir_largos_distintos(clf, cfg):
    with pytest.raises(ValueError):
        clf.decidir(np.array(["salir", "otro"]), np.array([0.9]), np.array([0.1, 0.2]), 0.85, cfg)


# ===========================================================================
# 6.6 Evaluación
# ===========================================================================

def test_evaluar_ejemplo(clf, cfg):
    v, p, d = (np.array(x) for x in zip(*EJEMPLO_EVALUAR))
    r = clf.evaluar(d, p, v, cfg)
    assert set(r) == CLAVES_EVALUAR, f"claves {sorted(r)}, esperado {sorted(CLAVES_EVALUAR)}"
    esperado = {"identificacion": 0.75, "acierto": 0.5, "frr": 0.5, "otro_aceptado": 0.25}
    for clave, valor in esperado.items():
        assert _es_float(r[clave]) and abs(r[clave] - valor) <= TOL, f"{clave} = {r[clave]!r}, esperado {valor}"
    assert abs(r["frr"] - (1.0 - r["acierto"])) <= TOL, "frr != 1 − acierto"
    m = r["matriz"]
    assert isinstance(m, np.ndarray) and m.shape == (4, 4) and m.dtype == np.int64, \
        f"matriz: {type(m)}, forma {getattr(m, 'shape', None)}, dtype {getattr(m, 'dtype', None)}"
    esp_m = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 3]]
    assert m.tolist() == esp_m, f"matriz {m.tolist()}, esperado {esp_m}"
    assert np.array_equal(m, _matriz_oraculo(v, d)), "la matriz no coincide con confusion_matrix"


@pytest.mark.parametrize("caso", ["sin_comandos", "sin_otro", "largos_distintos", "verdad_fuera",
                                  "decision_fuera", "prediccion_fuera"])
def test_evaluar_invalido(clf, cfg, caso):
    v, p, d = (np.array(x).astype("<U12") for x in zip(*EJEMPLO_EVALUAR))
    if caso == "sin_comandos":
        v, p, d = v[4:], p[4:], d[4:]
    elif caso == "sin_otro":
        v, p, d = v[:4], p[:4], d[:4]
    elif caso == "largos_distintos":
        d = d[:-1]
    elif caso == "verdad_fuera":
        v[0] = "saludo"
    elif caso == "decision_fuera":
        d[0] = "saludo"
    else:
        p[0] = "saludo"
    with pytest.raises(ValueError):
        clf.evaluar(d, p, v, cfg)


# ===========================================================================
# 6.7 Modelo (datos sintéticos)
# ===========================================================================

def test_ajustar_modelo_hiperparametros_de_la_validacion_cruzada(modelo_sintetico, resultado_cv):
    m = modelo_sintetico
    obtenido = {"esquema": m.esquema, "n_componentes": m.n_componentes, "k": m.k}
    assert obtenido == resultado_cv[0], f"modelo {obtenido}, seleccionar_hiperparametros {resultado_cv[0]}"


def test_ajustar_modelo_entrenamiento_y_proyeccion(cfg, largos, parametros, datos_cv, modelo_sintetico):
    from src.core_ml.pca import PCA

    m, d = modelo_sintetico, datos_cv
    f = _factores_oraculo(m.esquema, largos)
    assert np.max(np.abs(np.asarray(m.factores) - f)) <= TOL, f"factores {m.factores}, esperado {f}"
    R = _reponderar_oraculo(d["X"], f, largos)
    if m.n_componentes is None:
        assert m.pca_media.shape == (0,) and m.pca_componentes.shape == (0, 0), \
            f"sin PCA: formas {m.pca_media.shape}, {m.pca_componentes.shape}"
        esperado = _normalizar_oraculo(R)
    else:
        pca = PCA(m.n_componentes).ajustar(R)
        assert np.max(np.abs(m.pca_media - pca.media)) <= 1e-9, "pca_media distinta de PCA(...).ajustar(...)"
        assert m.pca_componentes.shape == pca.componentes.shape and \
            np.max(np.abs(m.pca_componentes - pca.componentes)) <= 1e-9, "pca_componentes distintas"
        esperado = _normalizar_oraculo(pca.transformar(R))
    assert m.entrenamiento.shape == esperado.shape, f"forma {m.entrenamiento.shape}, esperado {esperado.shape}"
    assert np.max(np.abs(m.entrenamiento - esperado)) <= 1e-9, "entrenamiento no es la cadena de 4.2"
    assert np.max(np.abs(m.entrenamiento - m.transformar(d["X"], cfg))) <= 1e-9, \
        "entrenamiento != transformar(X_train)"
    assert np.array_equal(np.asarray(m.etiquetas).astype(str), d["etiquetas"]), "etiquetas distintas"
    assert np.asarray(m.etiquetas).dtype.kind == "U", f"etiquetas de dtype {np.asarray(m.etiquetas).dtype}"
    assert m.eps == parametros["eps"] and m.confianza_min == parametros["cmin"], \
        f"eps {m.eps!r}, confianza_min {m.confianza_min!r}"
    assert _es_int(m.k) and m.esquema in ESQUEMAS


def test_ajustar_modelo_tau_de_val(cfg, parametros, datos_val, modelo_sintetico):
    m, va = modelo_sintetico, datos_val
    pred, conf, dist = m.votar(va["X"], cfg)
    esperado = _calibrar_oraculo(dist, conf, pred, va["etiquetas"], parametros["cmin"],
                                 parametros["otro_max"], va["n_otro_no_validos"])
    assert _es_float(m.tau), f"tau de tipo {type(m.tau)}"
    assert abs(m.tau - esperado) <= TOL, f"tau = {m.tau!r}, calibrar_umbral de val da {esperado!r}"


def test_ajustar_modelo_cambiar_val_solo_cambia_tau(clf, cfg, parametros, datos_cv, datos_val, modelo_sintetico):
    tr, va, m1 = datos_cv, datos_val, modelo_sintetico
    X_val2, y_val2 = _val_con_otro_pegado_a_un_comando(m1, cfg, va, parametros)
    m2 = clf.ajustar_modelo(tr["X"], tr["etiquetas"], tr["personas"], tr["evaluable"],
                            tr["n_otro_no_validos"], X_val2, y_val2, va["n_otro_no_validos"], cfg)
    assert (m2.esquema, m2.n_componentes, m2.k) == (m1.esquema, m1.n_componentes, m1.k), \
        "cambiar val cambió los hiperparámetros"
    for campo in ("factores", "entrenamiento", "pca_media", "pca_componentes", "etiquetas"):
        assert np.array_equal(getattr(m1, campo), getattr(m2, campo)), f"cambiar val cambió {campo}"
    pred, conf, dist = m2.votar(X_val2, cfg)
    esperado = _calibrar_oraculo(dist, conf, pred, y_val2, parametros["cmin"], parametros["otro_max"],
                                 va["n_otro_no_validos"])
    assert abs(m2.tau - esperado) <= TOL, f"tau con el val nuevo {m2.tau!r}, calibrar_umbral da {esperado!r}"
    assert m2.tau < m1.tau - 1e-9, f"un 'otro' sobre un comando de val debería bajar tau ({m1.tau} → {m2.tau})"


def _args_ajustar_invalidos(caso, d, va):
    a = [d["X"].copy(), d["etiquetas"].copy(), d["personas"].copy(), d["evaluable"].copy(), 0,
         va["X"].copy(), va["etiquetas"].copy(), 0]
    if caso == "x_train_ancho":
        a[0] = a[0][:, :280]
    elif caso == "x_val_ancho":
        a[5] = a[5][:, :280]
    elif caso == "etiquetas_train_largo":
        a[1] = a[1][:-1]
    elif caso == "etiquetas_val_largo":
        a[6] = a[6][:-1]
    elif caso == "etiqueta_train_fuera":
        a[1] = a[1].astype("<U12")
        a[1][0] = "saludo"
    elif caso == "etiqueta_val_fuera":
        a[6] = a[6].astype("<U12")
        a[6][0] = "saludo"
    return a


@pytest.mark.parametrize("caso", ["x_train_ancho", "x_val_ancho", "etiquetas_train_largo",
                                  "etiquetas_val_largo", "etiqueta_train_fuera", "etiqueta_val_fuera"])
def test_ajustar_modelo_invalido(clf, cfg, largos, datos_chicos, caso):
    va = _datos_chicos(largos, semilla=31, personas=("t08", "t09"))
    with pytest.raises(ValueError):
        clf.ajustar_modelo(*_args_ajustar_invalidos(caso, datos_chicos, va), cfg)


# ===========================================================================
# 6.8 Archivo del modelo, evaluación y reconocimiento (dataset chico)
# ===========================================================================

def test_entrenar_devuelve_ruta_y_27_claves(modelo_chico):
    salida, ruta = modelo_chico["salida"], modelo_chico["ruta"]
    assert Path(ruta).resolve() == Path(salida).resolve(), f"devolvió {ruta}, esperado {salida}"
    z = _cargar_npz(salida)
    assert set(z) == CLAVES_MODELO, \
        f"claves sobrantes {sorted(set(z) - CLAVES_MODELO)}, faltantes {sorted(CLAVES_MODELO - set(z))}"
    assert z["n_componentes"].dtype == np.int64, f"n_componentes de dtype {z['n_componentes'].dtype}"
    assert z["pesos"].shape == (5,), f"pesos de forma {z['pesos'].shape}"
    assert all(z[k].dtype != object for k in z), "el archivo contiene objetos de Python"


def test_entrenar_coincide_con_seleccion_calibracion_y_escalador(clf, cfg, v3, parametros, dataset_chico, modelo_chico):
    npz = _cargar_npz(dataset_chico / "caracteristicas.npz")
    meta = _leer_csv(dataset_chico / "metadatos.csv")
    tr, va = _arrays_de_particion(npz, "train", cfg), _arrays_de_particion(npz, "val", cfg)
    nnv_tr, nnv_va = _no_validos_otro(meta, "train", True), _no_validos_otro(meta, "val", False)
    assert (nnv_tr, nnv_va) == (0, 2), f"precondición: no válidos de train y val = {(nnv_tr, nnv_va)}"
    modelo = clf.cargar_modelo(modelo_chico["salida"])
    mejor, _ = clf.seleccionar_hiperparametros(tr["X"], tr["etiquetas"], tr["personas"], tr["evaluable"], nnv_tr, cfg)
    obtenido = {"esquema": modelo.esquema, "n_componentes": modelo.n_componentes, "k": modelo.k}
    assert obtenido == mejor, f"hiperparámetros {obtenido}, seleccionar_hiperparametros {mejor}"
    pred, conf, dist = modelo.votar(va["X"], cfg)
    tau = _calibrar_oraculo(dist, conf, pred, va["etiquetas"], parametros["cmin"], parametros["otro_max"], nnv_va)
    assert abs(modelo.tau - tau) <= TOL, f"tau {modelo.tau!r}, calibrar_umbral de val {tau!r}"
    assert np.max(np.abs(modelo.entrenamiento - modelo.transformar(tr["X"], cfg))) <= 1e-9, \
        "entrenamiento != transformar(X de train)"
    esc = v3.ajustar_escalador({b: npz[b][tr["mascara"]] for b in BLOQUES})
    difs = _diferencias(modelo.escalador, esc, TOL, "escalador")
    assert difs == [], f"escalador distinto de ajustar_escalador de train: {difs[:5]}"


def _copiar_dataset_minimo(origen: Path, destino: Path) -> Path:
    destino.mkdir(parents=True)
    for nombre in ("caracteristicas.npz", "metadatos.csv"):
        shutil.copy2(origen / nombre, destino / nombre)
    return destino


def test_entrenar_sin_fuga_de_test(clf, cfg, tmp_path, dataset_chico, modelo_chico):
    carpeta = _copiar_dataset_minimo(dataset_chico, tmp_path / "prepro_test_cambiado")
    npz = _cargar_npz(carpeta / "caracteristicas.npz")
    msk = npz["particion"].astype(str) == "test"
    rng = np.random.default_rng(99)
    for b in BLOQUES + ("X",):
        npz[b] = npz[b].copy()
        npz[b][msk] = rng.normal(size=npz[b][msk].shape)
    clases = npz["clase"].copy()
    clases[msk] = np.roll(clases[msk], 3)
    npz["clase"] = clases
    hablantes = npz["hablante"].astype("<U12")
    hablantes[msk] = "q99"
    npz["hablante"] = hablantes
    np.savez(carpeta / "caracteristicas.npz", **npz)
    meta = _leer_csv(carpeta / "metadatos.csv")
    columnas = list(meta[0].keys())
    for f in meta:
        if f["particion"] == "test":
            f["hablante"] = "q99"
    meta.append({**{c: "" for c in columnas}, "particion": "test", "origen": "notebook",
                 "hablante": "q98", "clase": OTRO, "copia": "0", "valido": "0"})
    _escribir_csv(carpeta / "metadatos.csv", meta, columnas)
    salida = clf.entrenar(cfg, carpeta=carpeta, salida=tmp_path / "modelo_otro_test.npz")
    difs = _comparar_npz(_cargar_npz(salida), _cargar_npz(modelo_chico["salida"]))
    assert difs == [], f"cambiar las filas de test cambió el modelo en las claves {difs}"


def test_entrenar_reproducible(clf, cfg, tmp_path, dataset_chico, modelo_chico):
    salida = clf.entrenar(cfg, carpeta=dataset_chico, salida=tmp_path / "otra_vez.npz")
    difs = _comparar_npz(_cargar_npz(salida), _cargar_npz(modelo_chico["salida"]))
    assert difs == [], f"dos entrenamientos dan modelos distintos en {difs}"


def test_entrenar_no_sobrescribe_salvo_que_se_pida(clf, cfg, tmp_path, dataset_chico):
    salida = tmp_path / "ya_existe.npz"
    salida.write_bytes(b"no es un modelo")
    mtime = os.stat(salida).st_mtime_ns
    with pytest.raises(FileExistsError):
        clf.entrenar(cfg, carpeta=dataset_chico, salida=salida)
    assert salida.read_bytes() == b"no es un modelo" and os.stat(salida).st_mtime_ns == mtime, \
        "con sobrescribir=False el archivo existente cambió"
    clf.entrenar(cfg, carpeta=dataset_chico, salida=salida, sobrescribir=True)
    assert set(_cargar_npz(salida)) == CLAVES_MODELO, "con sobrescribir=True no se reescribió el modelo"


@pytest.mark.parametrize("falta", ["caracteristicas.npz", "metadatos.csv"])
def test_entrenar_falta_archivo(clf, cfg, tmp_path, dataset_chico, falta):
    carpeta = _copiar_dataset_minimo(dataset_chico, tmp_path / "prepro_incompleto")
    (carpeta / falta).unlink()
    with pytest.raises(FileNotFoundError):
        clf.entrenar(cfg, carpeta=carpeta, salida=tmp_path / "modelo.npz")


def test_guardar_y_cargar_modelo_ida_y_vuelta(clf, tmp_path, modelo_chico):
    m = clf.cargar_modelo(modelo_chico["salida"])
    ruta = clf.guardar_modelo(m, tmp_path / "sub" / "carpeta" / "copia.npz")
    assert Path(ruta).resolve() == (tmp_path / "sub" / "carpeta" / "copia.npz").resolve(), f"devolvió {ruta}"
    m2 = clf.cargar_modelo(ruta)
    difs = _diferencias(m, m2, 0.0, "modelo")
    assert difs == [], f"guardar → cargar no devuelve lo guardado: {difs[:5]}"
    assert _comparar_npz(_cargar_npz(ruta), _cargar_npz(modelo_chico["salida"])) == [], \
        "volver a guardar el modelo cargado da otro archivo"


def test_cargar_modelo_tipos(clf, v3, modelo_chico):
    m = clf.cargar_modelo(modelo_chico["salida"])
    assert isinstance(m, clf.ModeloVoz)
    assert isinstance(m.esquema, str) and m.esquema in ESQUEMAS, f"esquema {m.esquema!r}"
    assert m.n_componentes is None or _es_int(m.n_componentes), f"n_componentes {m.n_componentes!r}"
    assert _es_int(m.k) and _es_float(m.eps) and _es_float(m.tau) and _es_float(m.confianza_min)
    assert np.asarray(m.etiquetas).dtype.kind == "U", f"etiquetas de dtype {np.asarray(m.etiquetas).dtype}"
    assert isinstance(m.escalador, v3.Escalador), f"escalador de tipo {type(m.escalador)}"
    for campo in ("medias", "desvios", "columnas", "pesos"):
        assert set(getattr(m.escalador, campo)) == set(BLOQUES), f"escalador.{campo} con otras claves"


def test_guardar_modelo_sin_pca_vuelve_none(clf, tmp_path, modelo_chico):
    m = clf.cargar_modelo(modelo_chico["salida"])
    sin_pca = dataclasses.replace(m, n_componentes=None, pca_media=np.zeros(0), pca_componentes=np.zeros((0, 0)))
    ruta = clf.guardar_modelo(sin_pca, tmp_path / "sin_pca.npz")
    z = _cargar_npz(ruta)
    assert z["n_componentes"].dtype == np.int64 and int(z["n_componentes"]) == 0, \
        f"sin PCA se guarda n_componentes = 0 (int64): {z['n_componentes']!r}"
    m2 = clf.cargar_modelo(ruta)
    assert m2.n_componentes is None, f"al cargar, n_componentes debe volver a None: {m2.n_componentes!r}"
    assert m2.pca_media.shape == (0,) and m2.pca_componentes.shape == (0, 0)


def test_guardar_modelo_sin_escalador(clf, tmp_path, modelo_chico):
    m = dataclasses.replace(clf.cargar_modelo(modelo_chico["salida"]), escalador=None)
    with pytest.raises(ValueError):
        clf.guardar_modelo(m, tmp_path / "sin_escalador.npz")


def test_cargar_modelo_inexistente(clf, tmp_path):
    with pytest.raises(FileNotFoundError):
        clf.cargar_modelo(tmp_path / "no_existe.npz")


@pytest.mark.parametrize("particion,filas", [("test", 16 + 1), ("val", 24 + 2)])
def test_evaluar_particion(clf, cfg, dataset_chico, modelo_chico, particion, filas):
    modelo = clf.cargar_modelo(modelo_chico["salida"])
    npz = _cargar_npz(dataset_chico / "caracteristicas.npz")
    meta = _leer_csv(dataset_chico / "metadatos.csv")
    msk = npz["particion"].astype(str) == particion
    X = npz["X"][msk]
    pred = np.asarray(modelo.votar(X, cfg)[0]).astype(str)
    dec = np.asarray(modelo.decidir(X, cfg)).astype(str)
    invalidas = [f["clase"] for f in meta if f["particion"] == particion and not _es_valido(f["valido"])]
    verdad = np.concatenate([npz["clase"][msk].astype(str), invalidas])
    pred = np.concatenate([pred, [OTRO] * len(invalidas)])
    dec = np.concatenate([dec, [OTRO] * len(invalidas)])
    esperado = _evaluar_oraculo(dec, pred, verdad)
    r = clf.evaluar_particion(modelo, cfg, dataset_chico, particion)
    assert set(r) == CLAVES_EVALUAR, f"claves {sorted(r)}"
    for clave in ("identificacion", "acierto", "frr", "otro_aceptado"):
        assert abs(r[clave] - esperado[clave]) <= TOL, f"{particion}: {clave} = {r[clave]!r}, esperado {esperado[clave]!r}"
    assert np.array_equal(r["matriz"], esperado["matriz"]), f"{particion}: matriz {r['matriz'].tolist()}"
    assert int(r["matriz"].sum()) == filas, f"{particion}: la matriz suma {int(r['matriz'].sum())}, esperado {filas}"


def test_evaluar_particion_invalida(clf, cfg, dataset_chico, modelo_chico):
    modelo = clf.cargar_modelo(modelo_chico["salida"])
    with pytest.raises(ValueError):
        clf.evaluar_particion(modelo, cfg, dataset_chico, "prueba")


def test_reconocer_silencio(clf, cfg, modelo_chico):
    modelo = clf.cargar_modelo(modelo_chico["salida"])
    fs, _ = _fs_y_margen(cfg)
    r = clf.reconocer(np.zeros(fs, dtype=np.float32), fs, modelo, cfg)
    assert isinstance(r, clf.Reconocimiento), f"devolvió {type(r)}"
    obtenido = (r.comando, r.valido, r.motivo, r.prediccion, r.confianza, r.distancia)
    assert obtenido == ("otro", False, "sin_voz", None, 0.0, float("inf")), f"silencio: {obtenido}"
    assert r.tau == modelo.tau, f"tau {r.tau!r} vs modelo {modelo.tau!r}"


@pytest.mark.parametrize("clase", ETIQUETAS)
def test_reconocer_clip_coincide_con_la_cadena(clf, cfg, v3, modelo_chico, clase):
    from src.voz.preprocesamiento import preprocesar

    modelo = clf.cargar_modelo(modelo_chico["salida"])
    fs, M = _fs_y_margen(cfg)
    silencio = np.zeros(int(0.4 * fs), dtype=np.float32)
    audio = np.concatenate([silencio, clip_de_clase(clase, 9000 + ETIQUETAS.index(clase), fs, M), silencio])
    r = clf.reconocer(audio, fs, modelo, cfg)
    p = preprocesar(audio, fs, cfg)
    assert r.valido == bool(p.valido) and r.tau == modelo.tau
    if not p.valido:
        assert (r.comando, r.prediccion, r.confianza, r.distancia, r.motivo) == \
            ("otro", None, 0.0, float("inf"), p.motivo), f"audio no válido: {r}"
        return
    z = modelo.escalador.transformar(v3.caracteristicas(p.y, cfg))
    pred, conf, dist = modelo.votar(z, cfg)
    comando = clf.decidir(pred, conf, dist, modelo.tau, cfg)
    assert r.motivo == "" and r.prediccion == pred and r.comando == comando, \
        f"reconocer {(r.comando, r.prediccion)}, cadena {(comando, pred)}"
    assert abs(r.confianza - conf) <= TOL and abs(r.distancia - dist) <= TOL, \
        f"confianza/distancia {(r.confianza, r.distancia)} vs {(conf, dist)}"


def test_reconocer_sin_escalador(clf, cfg, modelo_chico):
    modelo = dataclasses.replace(clf.cargar_modelo(modelo_chico["salida"]), escalador=None)
    fs, M = _fs_y_margen(cfg)
    silencio = np.zeros(int(0.4 * fs), dtype=np.float32)
    audio = np.concatenate([silencio, clip_de_clase("salir", 9100, fs, M), silencio])
    with pytest.raises(ValueError):
        clf.reconocer(audio, fs, modelo, cfg)


def test_generar_figura(clf, cfg, tmp_path, dataset_chico, modelo_chico):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    salida = tmp_path / "rechazo_v4.png"
    ruta = clf.generar_figura(cfg, carpeta=dataset_chico, modelo=modelo_chico["salida"], salida=salida)
    assert Path(ruta).resolve() == salida.resolve(), f"devolvió {ruta}, esperado {salida}"
    imagen = plt.imread(salida)
    assert imagen.ndim == 3 and imagen.shape[0] > 0 and imagen.shape[1] > 0, f"PNG ilegible: {imagen.shape}"


# ===========================================================================
# 6.9 Aceptación con los datos reales (data/voz/prepro/)
# ===========================================================================

MATRIZ_VAL = [[20, 0, 0, 3], [0, 19, 1, 2], [0, 0, 18, 4], [0, 0, 1, 33]]
MATRIZ_TEST = [[18, 0, 0, 6], [0, 17, 0, 7], [0, 0, 20, 3], [1, 2, 2, 29]]


@pytest.fixture(scope="module")
def real(tmp_path_factory, request):
    if not (CARPETA_REAL / "caracteristicas.npz").exists():
        pytest.skip("no existe data/voz/prepro/caracteristicas.npz")
    cfg, clf = request.getfixturevalue("cfg"), request.getfixturevalue("clf")
    salida = tmp_path_factory.mktemp("aceptacion_v4") / "modelo_voz.npz"
    t0 = time.perf_counter()
    ruta = clf.entrenar(cfg, carpeta=CARPETA_REAL, salida=salida)
    segundos = time.perf_counter() - t0
    npz = _cargar_npz(CARPETA_REAL / "caracteristicas.npz")
    meta = _leer_csv(CARPETA_REAL / "metadatos.csv")
    return {"salida": Path(ruta), "segundos": segundos, "modelo": clf.cargar_modelo(ruta), "npz": npz,
            "meta": meta, "train": _arrays_de_particion(npz, "train", cfg),
            "val": _arrays_de_particion(npz, "val", cfg)}


@pytest.fixture(scope="module")
def real_cv(real, clf, cfg):
    tr = real["train"]
    nnv = _no_validos_otro(real["meta"], "train", True)
    return clf.seleccionar_hiperparametros(tr["X"], tr["etiquetas"], tr["personas"], tr["evaluable"], nnv, cfg)


@pytest.mark.aceptacion
def test_aceptacion_tiempo_de_entrenamiento(real):
    assert real["segundos"] < 300, f"entrenar tardó {real['segundos']:.1f} s (máximo 300 s)"


@pytest.mark.aceptacion
def test_aceptacion_hiperparametros(real):
    m = real["modelo"]
    assert (m.esquema, m.n_componentes, m.k) == ("z_columna", 20, 5), \
        f"hiperparámetros {(m.esquema, m.n_componentes, m.k)}, esperado ('z_columna', 20, 5)"


@pytest.mark.aceptacion
def test_aceptacion_tabla_cv_empate_411(real, real_cv):
    tr = real["train"]
    cmd_eval = tr["evaluable"] & np.isin(tr["etiquetas"], COMANDOS)
    assert int(cmd_eval.sum()) == 231, f"comandos evaluados {int(cmd_eval.sum())}, esperado 231"
    assert len(set(tr["personas"][cmd_eval])) == 10, f"personas {sorted(set(tr['personas'][cmd_eval]))}"
    _, tabla = real_cv
    filas = {(f["esquema"], f["n_componentes"], f["k"]): f for f in tabla}
    for combo, esperado in ((("z_columna", 20, 5), (190, 221, 224)), (("sin_dur_zcr", 30, 5), (195, 216, 221))):
        f = filas[combo]
        obt = (f["n_tau5"], f["n_c60"], f["n_4cl"])
        assert obt == esperado, f"{combo}: (n_tau5, n_c60, n_4cl) = {obt}, esperado {esperado}"
    assert real_cv[0] == {"esquema": "z_columna", "n_componentes": 20, "k": 5}, f"mejor {real_cv[0]}"


@pytest.mark.aceptacion
def test_aceptacion_tau_solo_de_val(real, cfg, parametros):
    m, va = real["modelo"], real["val"]
    pred, conf, dist = m.votar(va["X"], cfg)
    nnv = _no_validos_otro(real["meta"], "val", False)
    tau = _calibrar_oraculo(dist, conf, pred, va["etiquetas"], parametros["cmin"], parametros["otro_max"], nnv)
    assert abs(m.tau - tau) <= TOL, f"tau {m.tau!r}, calibrar_umbral con solo val {tau!r}"
    assert abs(m.tau - 0.7738) <= 1e-3, f"tau {m.tau!r}, esperado 0.7738 ± 1e-3"


@pytest.mark.aceptacion
def test_aceptacion_val(real, clf, cfg):
    validas = int((real["npz"]["particion"].astype(str) == "val").sum())
    invalidas = sum(1 for f in real["meta"] if f["particion"] == "val" and not _es_valido(f["valido"]))
    assert (validas, invalidas) == (87, 14), f"val: {validas} válidas y {invalidas} no válidas"
    r = clf.evaluar_particion(real["modelo"], cfg, CARPETA_REAL, "val")
    assert r["matriz"].tolist() == MATRIZ_VAL, f"matriz de val {r['matriz'].tolist()}, esperado {MATRIZ_VAL}"
    assert r["otro_aceptado"] <= 1 / 34 + TOL, f"otro aceptado en val {r['otro_aceptado']:.4f} > 1/34"
    assert abs(r["acierto"] - 57 / 67) <= TOL, f"acierto de val {r['acierto']!r}, esperado 57/67"


@pytest.mark.aceptacion
def test_aceptacion_test(real, clf, cfg):
    validas = int((real["npz"]["particion"].astype(str) == "test").sum())
    invalidas = sum(1 for f in real["meta"] if f["particion"] == "test" and not _es_valido(f["valido"]))
    assert (validas, invalidas) == (99, 6), f"test: {validas} válidas y {invalidas} no válidas"
    r = clf.evaluar_particion(real["modelo"], cfg, CARPETA_REAL, "test")
    meta_acierto = float(cfg["metas"]["voz_acierto_min"])
    assert r["identificacion"] >= meta_acierto, f"identificación de test {r['identificacion']:.4f} < {meta_acierto}"
    assert r["matriz"].tolist() == MATRIZ_TEST, f"matriz de test {r['matriz'].tolist()}, esperado {MATRIZ_TEST}"


@pytest.mark.aceptacion
def test_aceptacion_reproducible(real, clf, cfg, tmp_path):
    salida = clf.entrenar(cfg, carpeta=CARPETA_REAL, salida=tmp_path / "modelo_voz_2.npz")
    difs = _comparar_npz(_cargar_npz(salida), _cargar_npz(real["salida"]))
    assert difs == [], f"dos entrenamientos con los datos reales difieren en {difs}"
