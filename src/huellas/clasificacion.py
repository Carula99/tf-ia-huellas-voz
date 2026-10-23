"""H4 · K-means propio, distancia normalizada y umbral de rechazo.

Contrato: docs/contratos/h4_clasificacion.md (versión 1.1). A partir de caracteristicas.npz (H3):

1. K-means propio (src/core_ml/kmeans.py) con k = 3 sobre las filas de train de X; cada cluster
   va a la identidad mayoritaria de sus filas, con correspondencia uno a uno.
2. Puntaje de una huella: la menor distancia normalizada d_k(x) = ‖x − c_k‖ / r_k, con
   r_k = √(inercia del cluster / filas del cluster), que no baja de piso_dispersion · r̄
   (r̄, la dispersión de todo train). La identidad predicha es la de ese centroide.
3. Umbral τ*: con las filas de val, criterio far_max (FAR de val ≤ far_max_val y, entre esos τ,
   el menor FRR) o eer (el menor max(FAR, FRR)); τ* es la mitad del intervalo [a, b) de τ que
   cumple el criterio, no su borde. Se acepta una huella si su puntaje es ≤ τ*.
4. Evaluación a ciegas en test con el modelo congelado: identificación, FAR, FRR y la matriz de
   confusión 4 × 4 (identidades 1 a 3 y "no autorizado").

El modelo (centroides, dispersiones, τ*, pureza, inercia y el escalador de H3) se guarda en un
.npz sin objetos de Python. reconocer() lleva una imagen hasta la decisión, para la Fase 5.

Línea de comandos, desde la raíz del repo (escribe models/modelo_huellas.npz y
docs/figuras/far_frr_h4.png, e imprime las métricas de val y de test):
    venv\\Scripts\\python -m src.huellas.clasificacion [--sobrescribir]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from ..config import RAIZ, cargar_config
from ..core_ml.kmeans import KMeans
from ..core_ml.metricas import matriz_confusion
from .caracteristicas import BLOQUES, NOMBRE_NPZ, Escalador, ajustar_escalador, caracteristicas
from .preprocesamiento import preprocesar

NO_AUTORIZADO: int = 0
CRITERIOS: tuple[str, ...] = ("eer", "far_max")
NOMBRE_MODELO = "modelo_huellas.npz"
NOMBRE_FIGURA = "far_frr_h4.png"


def _es_entero(valor) -> bool:
    return isinstance(valor, (int, np.integer)) and not isinstance(valor, (bool, np.bool_))


def _es_real(valor) -> bool:
    return (isinstance(valor, (int, float, np.integer, np.floating))
            and not isinstance(valor, (bool, np.bool_)) and math.isfinite(valor))


def _vector_1d(valores, nombre: str) -> np.ndarray:
    a = np.asarray(valores)
    if a.ndim != 1:
        raise ValueError(f"{nombre} tiene que ser una secuencia de 1 dimensión; tiene forma {a.shape}.")
    return a


def _distancias(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Distancias euclídeas exactas (n, k)."""
    D = np.empty((X.shape[0], C.shape[0]))
    for j in range(C.shape[0]):
        dif = X - C[j]
        D[:, j] = np.sqrt(np.einsum("ij,ij->i", dif, dif))
    return D


def _puntajes(X, centroides: np.ndarray, dispersion: np.ndarray, identidades: np.ndarray):
    """Sección 4.3: (puntaje, predicción) de (n, d) → arrays, o de un vector (d,) → (float, int)."""
    X = np.asarray(X, dtype=np.float64)
    vector = X.ndim == 1
    X2 = X[None, :] if vector else X
    if X2.ndim != 2 or X2.shape[1] != centroides.shape[1]:
        raise ValueError(f"X tiene forma {X.shape}; se esperan {centroides.shape[1]} columnas.")
    if not np.isfinite(X2).all():
        raise ValueError("X tiene valores NaN o infinitos.")
    D = _distancias(X2, centroides) / dispersion[None, :]
    j = np.argmin(D, axis=1)
    puntaje = D[np.arange(X2.shape[0]), j]
    prediccion = np.asarray(identidades, dtype=np.int64)[j]
    if vector:
        return float(puntaje[0]), int(prediccion[0])
    return puntaje, prediccion


# --------------------------------------------------------------------------- modelo
@dataclass(frozen=True, eq=False)   # eq=False: los arrays no se comparan con ==
class ModeloHuellas:
    """Lo que se congela después de train y val: centroides, dispersiones, τ* y el escalador."""

    identidades: np.ndarray    # int64 (k,): 1, …, k
    centroides: np.ndarray     # float64 (k, d), en el orden de identidades
    dispersion: np.ndarray     # float64 (k,): r_k, con el piso
    tau: float                 # τ*
    pureza: float              # pureza de los clusters en train
    inercia: float             # inercia del K-means en train
    escalador: Escalador | None    # el de H3, ajustado con train (None solo en modelos armados sin él)

    def puntajes(self, X: np.ndarray):
        """(puntaje (n,), prediccion (n,) int64) para (n, d), o (float, int) para un vector (d,)."""
        return _puntajes(X, self.centroides, self.dispersion, self.identidades)

    def decidir(self, X: np.ndarray):
        """La identidad predicha si el puntaje es ≤ τ y NO_AUTORIZADO si no (un int para un vector)."""
        puntaje, prediccion = self.puntajes(X)
        if np.ndim(puntaje) == 0:
            return int(prediccion) if puntaje <= self.tau else NO_AUTORIZADO
        return decidir(puntaje, prediccion, self.tau)


@dataclass(frozen=True)
class Reconocimiento:
    """Resultado de reconocer una imagen."""

    identidad: int         # la decisión: 1 a k, o 0 si se rechaza
    mas_cercana: int       # identidad del centroide más cercano (con distancia normalizada)
    distancia: float       # el puntaje: distancia normalizada a ese centroide
    tau: float             # τ* del modelo


# --------------------------------------------------------------------------- correspondencia y dispersión
def correspondencia(etiquetas, identidades, k: int) -> tuple[dict[int, int], float]:
    """Cada cluster j = 0 … k − 1 va a la identidad con más filas en él (la menor, si empatan).

    Devuelve (mapa, pureza), con pureza = Σ_j (filas de la identidad mayoritaria de j) / n.

    Raises:
        ValueError: si los largos no coinciden, si un cluster no tiene filas o si dos clusters
            van a la misma identidad (la correspondencia tiene que ser uno a uno).
    """
    if not (_es_entero(k) and k >= 1):
        raise ValueError(f"k tiene que ser un entero >= 1; es {k!r}.")
    etiquetas = _vector_1d(etiquetas, "etiquetas")
    identidades = _vector_1d(identidades, "identidades")
    if etiquetas.shape[0] != identidades.shape[0] or etiquetas.shape[0] == 0:
        raise ValueError(f"Hay {etiquetas.shape[0]} etiquetas y {identidades.shape[0]} identidades.")
    fuera = ~np.isin(etiquetas, np.arange(k))
    if fuera.any():
        raise ValueError(f"Hay etiquetas fuera de 0 … {k - 1}: {sorted(set(etiquetas[fuera].tolist()))}.")
    mapa: dict[int, int] = {}
    aciertos = 0
    for j in range(k):
        de_j = identidades[etiquetas == j]
        if de_j.size == 0:
            raise ValueError(f"El cluster {j} no tiene filas: no se le puede asignar una identidad.")
        valores, cuentas = np.unique(de_j, return_counts=True)   # ordenados: con empate, la menor
        mapa[int(j)] = int(valores[np.argmax(cuentas)])
        aciertos += int(cuentas.max())
    if len(set(mapa.values())) != k:
        raise ValueError(f"La correspondencia cluster → identidad no es uno a uno: {mapa}.")
    return mapa, aciertos / etiquetas.shape[0]


def dispersiones(X, etiquetas, centroides, piso: float) -> np.ndarray:
    """r_j = max(√(I_j / n_j), piso · r̄), con I_j = Σ_{etiquetas = j} ‖x − c_j‖² y r̄ = √(Σ_j I_j / n).

    Raises:
        ValueError: si un cluster no tiene filas, si r̄ = 0 o si piso no está en (0, 1].
    """
    if not (_es_real(piso) and 0 < piso <= 1):
        raise ValueError(f"piso tiene que estar en (0, 1]; es {piso!r}.")
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(centroides, dtype=np.float64)
    etiquetas = _vector_1d(etiquetas, "etiquetas")
    if X.ndim != 2 or C.ndim != 2 or C.shape[1] != X.shape[1] or etiquetas.shape[0] != X.shape[0]:
        raise ValueError(f"Formas incompatibles: X {X.shape}, etiquetas {etiquetas.shape}, centroides {C.shape}.")
    k = C.shape[0]
    inercia = np.empty(k)
    filas = np.empty(k)
    for j in range(k):
        de_j = etiquetas == j
        filas[j] = np.count_nonzero(de_j)
        if filas[j] == 0:
            raise ValueError(f"El cluster {j} no tiene filas: no tiene dispersión.")
        inercia[j] = ((X[de_j] - C[j]) ** 2).sum()
    r_media = math.sqrt(inercia.sum() / X.shape[0])
    if r_media == 0:
        raise ValueError("Todas las filas son iguales a su centroide (r̄ = 0): no hay dispersión para normalizar.")
    return np.maximum(np.sqrt(inercia / filas), piso * r_media)


# --------------------------------------------------------------------------- umbral
def _entradas_de_val(puntaje, prediccion, verdad) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = _vector_1d(puntaje, "puntaje").astype(np.float64)
    p = _vector_1d(prediccion, "prediccion")
    y = _vector_1d(verdad, "verdad")
    if not (s.shape[0] == p.shape[0] == y.shape[0]):
        raise ValueError(f"Largos distintos: puntaje {s.shape[0]}, prediccion {p.shape[0]}, verdad {y.shape[0]}.")
    if not np.isfinite(s).all():
        raise ValueError("puntaje tiene valores NaN o infinitos.")
    autorizada = y != NO_AUTORIZADO
    if not autorizada.any():
        raise ValueError("No hay autorizadas (verdad ≠ 0).")
    if autorizada.all():
        raise ValueError("No hay intrusos (verdad = 0).")
    return s, p, y, autorizada


def far_frr(puntaje, prediccion, verdad, tau: float) -> tuple[float, float]:
    """(FAR, FRR) en τ: FAR = intrusos con puntaje ≤ τ; FRR = autorizadas que no quedan aceptadas
    con su propia identidad (puntaje > τ o predicción ≠ verdad)."""
    s, p, y, autorizada = _entradas_de_val(puntaje, prediccion, verdad)
    acepta = s <= tau
    far = np.count_nonzero(acepta[~autorizada]) / np.count_nonzero(~autorizada)
    bien = acepta[autorizada] & (p[autorizada] == y[autorizada])
    frr = np.count_nonzero(~bien) / np.count_nonzero(autorizada)
    return float(far), float(frr)


def _intervalo(puntaje, prediccion, verdad, criterio: str, far_max_val: float) -> tuple[float, float]:
    """Sección 4.4: el intervalo [a, b) de τ que cumple el criterio; τ* es su mitad."""
    if criterio not in CRITERIOS:
        raise ValueError(f"criterio tiene que ser uno de {CRITERIOS}; es {criterio!r}.")
    if not (_es_real(far_max_val) and 0 < far_max_val < 1):
        raise ValueError(f"far_max_val tiene que estar en (0, 1); es {far_max_val!r}.")
    s, p, y, autorizada = _entradas_de_val(puntaje, prediccion, verdad)
    bien_identificada = autorizada & (p == y)
    if criterio == "far_max":
        intrusos = np.sort(s[~autorizada])
        m = math.floor(far_max_val * intrusos.shape[0] + 1e-9)   # intrusos que se pueden aceptar
        b = float(intrusos[m])
        abajo = s[bien_identificada & (s < b)]
        if abajo.size == 0:
            raise ValueError(f"Ninguna autorizada bien identificada tiene puntaje < b = {b}: no se puede calibrar.")
        return float(abajo.max()), b
    # eer: FAR y FRR son constantes en [c_i, c_{i+1}); el mínimo de E = max(FAR, FRR) es un intervalo
    candidatos = np.concatenate([[-np.inf], np.unique(s)])
    n_i, n_a = np.count_nonzero(~autorizada), np.count_nonzero(autorizada)
    E = np.empty(candidatos.shape[0])
    for i, t in enumerate(candidatos):
        acepta = s <= t
        far = np.count_nonzero(acepta[~autorizada]) / n_i
        frr = np.count_nonzero(~(acepta & bien_identificada)[autorizada]) / n_a
        E[i] = max(far, frr)
    minimos = np.flatnonzero(E == E.min())
    a = float(candidatos[minimos[0]])
    b = float(candidatos[minimos[-1] + 1]) if minimos[-1] + 1 < candidatos.shape[0] else math.inf
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("El mínimo de max(FAR, FRR) llega a un extremo (−∞ o +∞): no se puede calibrar.")
    return a, b


def calibrar_umbral(puntaje, prediccion, verdad, criterio: str, far_max_val: float) -> float:
    """τ* = la mitad del intervalo [a, b) de τ que cumple el criterio (far_max o eer) en val.

    Raises:
        ValueError: con largos distintos, puntajes no finitos, sin autorizadas o sin intrusos,
            un criterio desconocido, far_max_val fuera de (0, 1) (con los dos criterios), o si
            el criterio no deja ningún intervalo finito.
    """
    a, b = _intervalo(puntaje, prediccion, verdad, criterio, far_max_val)
    return float((a + b) / 2)


def decidir(puntaje, prediccion, tau: float) -> np.ndarray:
    """La predicción donde puntaje ≤ τ y NO_AUTORIZADO donde no (int64)."""
    puntaje = np.asarray(puntaje, dtype=np.float64)
    prediccion = np.asarray(prediccion, dtype=np.int64)
    if puntaje.shape != prediccion.shape:
        raise ValueError(f"puntaje {puntaje.shape} y prediccion {prediccion.shape} tienen formas distintas.")
    return np.where(puntaje <= tau, prediccion, NO_AUTORIZADO).astype(np.int64)


# --------------------------------------------------------------------------- evaluación
def evaluar(decision, prediccion, verdad, k: int) -> dict:
    """identificacion, far, frr y la matriz de confusión (k + 1) × (k + 1) con el orden [1, …, k, 0].

    Raises:
        ValueError: si los largos no coinciden, si no hay autorizadas o no hay intrusos.
    """
    d = _vector_1d(decision, "decision").astype(np.int64)
    p = _vector_1d(prediccion, "prediccion").astype(np.int64)
    y = _vector_1d(verdad, "verdad").astype(np.int64)
    if not (d.shape[0] == p.shape[0] == y.shape[0]):
        raise ValueError(f"Largos distintos: decision {d.shape[0]}, prediccion {p.shape[0]}, verdad {y.shape[0]}.")
    autorizada = y != NO_AUTORIZADO
    if not autorizada.any() or autorizada.all():
        raise ValueError("Hacen falta autorizadas y también intrusos para evaluar.")
    return {
        "identificacion": float(np.mean(p[autorizada] == y[autorizada])),
        "far": float(np.mean(d[~autorizada] != NO_AUTORIZADO)),
        "frr": float(np.mean(d[autorizada] != y[autorizada])),
        "matriz": matriz_confusion(y.tolist(), d.tolist(), [*range(1, k + 1), NO_AUTORIZADO]),
    }


# --------------------------------------------------------------------------- ajuste del modelo
def ajustar_modelo(X_train, identidades_train, X_val, verdad_val, cfg: dict,
                   escalador: Escalador | None = None) -> ModeloHuellas:
    """K-means con huellas.kmeans y semilla_global, correspondencia, dispersiones con el piso y
    τ* calibrado con val (huellas.umbral). No recibe nada de test."""
    k_cfg, u_cfg = cfg["huellas"]["kmeans"], cfg["huellas"]["umbral"]
    k = int(k_cfg["k"])
    Xt = np.asarray(X_train, dtype=np.float64)
    Xv = np.asarray(X_val, dtype=np.float64)
    if Xt.ndim != 2 or Xv.ndim != 2 or Xt.shape[1] != Xv.shape[1]:
        raise ValueError(f"X_train {Xt.shape} y X_val {Xv.shape} tienen que ser (n, d) con el mismo d.")
    if not (np.isfinite(Xt).all() and np.isfinite(Xv).all()):
        raise ValueError("X_train o X_val tienen valores NaN o infinitos.")
    ids_train = _vector_1d(identidades_train, "identidades_train")
    verdad = _vector_1d(verdad_val, "verdad_val")
    if ids_train.shape[0] != Xt.shape[0] or verdad.shape[0] != Xv.shape[0]:
        raise ValueError("Las identidades no tienen una fila por cada fila de X_train o de X_val.")
    if not np.isin(ids_train, np.arange(1, k + 1)).all():
        raise ValueError(f"identidades_train tiene valores fuera de 1 … {k}.")
    if not np.isin(verdad, np.arange(0, k + 1)).all():
        raise ValueError(f"verdad_val tiene valores fuera de 0 … {k}.")

    km = KMeans(k, k_cfg["n_init"], k_cfg["max_iter"], k_cfg["tol"], semilla=cfg["semilla_global"]).ajustar(Xt)
    mapa, pureza = correspondencia(km.etiquetas, ids_train, k)
    cluster_de = {identidad: j for j, identidad in mapa.items()}
    orden = [cluster_de[i] for i in range(1, k + 1)]          # fila i del modelo = cluster de la identidad i + 1
    nueva = np.empty(k, dtype=np.int64)
    nueva[orden] = np.arange(k)
    etiquetas = nueva[km.etiquetas]
    centroides = km.centroides[orden].copy()
    identidades = np.arange(1, k + 1, dtype=np.int64)
    dispersion = dispersiones(Xt, etiquetas, centroides, u_cfg["piso_dispersion"])
    puntaje, prediccion = _puntajes(Xv, centroides, dispersion, identidades)
    tau = calibrar_umbral(puntaje, prediccion, verdad, u_cfg["criterio"], u_cfg["far_max_val"])
    return ModeloHuellas(identidades=identidades, centroides=centroides, dispersion=dispersion, tau=float(tau),
                         pureza=float(pureza), inercia=float(km.inercia), escalador=escalador)


# --------------------------------------------------------------------------- archivo del modelo
def guardar_modelo(modelo: ModeloHuellas, ruta: str | os.PathLike) -> Path:
    """Escribe el modelo en un .npz (sin objetos de Python) con las 16 claves de 4.6."""
    if modelo.escalador is None:
        raise ValueError("El modelo no tiene escalador: no se puede guardar.")
    esc = modelo.escalador
    datos = {
        "identidades": np.asarray(modelo.identidades, dtype=np.int64),
        "centroides": np.asarray(modelo.centroides, dtype=np.float64),
        "dispersion": np.asarray(modelo.dispersion, dtype=np.float64),
        "tau": np.float64(modelo.tau),
        "pureza": np.float64(modelo.pureza),
        "inercia": np.float64(modelo.inercia),
        "pesos": np.array([esc.pesos[b] for b in BLOQUES], dtype=np.float64),
    }
    for b in BLOQUES:
        datos[f"medias_{b}"] = np.asarray(esc.medias[b], dtype=np.float64)
        datos[f"desvios_{b}"] = np.asarray(esc.desvios[b], dtype=np.float64)
        datos[f"columnas_{b}"] = np.asarray(esc.columnas[b], dtype=np.bool_)
    ruta = Path(os.fspath(ruta))
    buffer = io.BytesIO()            # np.savez agrega ".npz" a una ruta sin esa extensión
    np.savez(buffer, **datos)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(buffer.getvalue())
    return ruta


def cargar_modelo(ruta: str | os.PathLike) -> ModeloHuellas:
    """Lee un modelo escrito por guardar_modelo (allow_pickle=False)."""
    ruta = Path(os.fspath(ruta))
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el modelo {ruta}.")
    with np.load(ruta, allow_pickle=False) as d:
        datos = {clave: d[clave] for clave in d.files}
    escalador = Escalador(
        medias={b: datos[f"medias_{b}"].astype(np.float64) for b in BLOQUES},
        desvios={b: datos[f"desvios_{b}"].astype(np.float64) for b in BLOQUES},
        columnas={b: datos[f"columnas_{b}"].astype(np.bool_) for b in BLOQUES},
        pesos={b: float(datos["pesos"][i]) for i, b in enumerate(BLOQUES)},
    )
    return ModeloHuellas(
        identidades=datos["identidades"].astype(np.int64),
        centroides=datos["centroides"].astype(np.float64),
        dispersion=datos["dispersion"].astype(np.float64),
        tau=float(datos["tau"]), pureza=float(datos["pureza"]), inercia=float(datos["inercia"]),
        escalador=escalador,
    )


def _leer_caracteristicas(carpeta: Path) -> dict:
    ruta = carpeta / NOMBRE_NPZ
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta} (el archivo de características de H3).")
    with np.load(ruta, allow_pickle=False) as d:
        return {clave: d[clave] for clave in d.files}


def entrenar(cfg: dict, carpeta: str | os.PathLike | None = None,
             salida: str | os.PathLike | None = None, sobrescribir: bool = False) -> Path:
    """Ajusta el modelo con train y val de carpeta / caracteristicas.npz y lo guarda en salida.

    Por defecto, carpeta = RAIZ / rutas.huellas y salida = RAIZ / rutas.modelos / modelo_huellas.npz.
    No usa las filas de test. Devuelve la ruta escrita.

    Raises:
        FileExistsError: si salida ya existe y sobrescribir es False (no toca nada).
        FileNotFoundError: si falta caracteristicas.npz.
        ValueError: si no hay filas de train o de val.
    """
    carpeta = RAIZ / cfg["rutas"]["huellas"] if carpeta is None else Path(os.fspath(carpeta))
    salida = RAIZ / cfg["rutas"]["modelos"] / NOMBRE_MODELO if salida is None else Path(os.fspath(salida))
    if salida.exists() and not sobrescribir:
        raise FileExistsError(f"{salida} ya existe; usá --sobrescribir (sobrescribir=True) para volver a escribirlo.")
    datos = _leer_caracteristicas(carpeta)
    particion = datos["particion"]
    train, val = particion == "train", particion == "val"
    if not train.any() or not val.any():
        raise ValueError(f"{carpeta / NOMBRE_NPZ} tiene que tener filas de train y de val.")
    escalador = ajustar_escalador({b: datos[b][train] for b in BLOQUES})
    modelo = ajustar_modelo(datos["X"][train], datos["identidad"][train],
                            datos["X"][val], datos["identidad"][val], cfg, escalador)
    return guardar_modelo(modelo, salida)


def evaluar_particion(modelo: ModeloHuellas, cfg: dict, carpeta: str | os.PathLike | None = None,
                      particion: str = "test") -> dict:
    """evaluar() de las filas de esa partición de caracteristicas.npz, con el modelo congelado."""
    carpeta = RAIZ / cfg["rutas"]["huellas"] if carpeta is None else Path(os.fspath(carpeta))
    datos = _leer_caracteristicas(carpeta)
    filas = datos["particion"] == particion
    if not filas.any():
        raise ValueError(f"{carpeta / NOMBRE_NPZ} no tiene filas de {particion!r}.")
    puntaje, prediccion = modelo.puntajes(datos["X"][filas])
    return evaluar(decidir(puntaje, prediccion, modelo.tau), prediccion, datos["identidad"][filas],
                   len(modelo.identidades))


def reconocer(imagen: np.ndarray, modelo: ModeloHuellas, cfg: dict) -> Reconocimiento:
    """Imagen → H2 → H3 → escalador del modelo → puntaje y decisión."""
    if modelo.escalador is None:
        raise ValueError("El modelo no tiene escalador: no puede llevar una imagen al espacio de X.")
    z = modelo.escalador.transformar(caracteristicas(preprocesar(imagen, cfg), cfg))
    distancia, mas_cercana = modelo.puntajes(z)
    identidad = mas_cercana if distancia <= modelo.tau else NO_AUTORIZADO
    return Reconocimiento(identidad=int(identidad), mas_cercana=int(mas_cercana),
                          distancia=float(distancia), tau=float(modelo.tau))


# --------------------------------------------------------------------------- figura
def _coma(valor: float, decimales: int = 3) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def generar_figura(cfg: dict, carpeta: str | os.PathLike | None = None,
                   modelo: str | os.PathLike | ModeloHuellas | None = None,
                   salida: str | os.PathLike | None = None) -> Path:
    """FAR y FRR de val en función de τ (con τ* y el intervalo [a, b)) y la matriz de confusión de test.

    Por defecto: carpeta = RAIZ / rutas.huellas, modelo = RAIZ / rutas.modelos / modelo_huellas.npz y
    salida = RAIZ / rutas.figuras / far_frr_h4.png. Devuelve la ruta escrita.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    carpeta = RAIZ / cfg["rutas"]["huellas"] if carpeta is None else Path(os.fspath(carpeta))
    if not isinstance(modelo, ModeloHuellas):
        modelo = cargar_modelo(RAIZ / cfg["rutas"]["modelos"] / NOMBRE_MODELO if modelo is None else modelo)
    salida = RAIZ / cfg["rutas"]["figuras"] / NOMBRE_FIGURA if salida is None else Path(os.fspath(salida))
    datos = _leer_caracteristicas(carpeta)
    umbral = cfg["huellas"]["umbral"]
    k = len(modelo.identidades)
    val, test = datos["particion"] == "val", datos["particion"] == "test"
    s_v, p_v = modelo.puntajes(datos["X"][val])
    y_v = datos["identidad"][val]
    a, b = _intervalo(s_v, p_v, y_v, umbral["criterio"], umbral["far_max_val"])
    autorizada = y_v != NO_AUTORIZADO

    figura = Figure(figsize=(13, 5.4))
    FigureCanvasAgg(figura)
    ax1 = figura.add_subplot(1, 2, 1)
    ax2 = figura.add_subplot(1, 2, 2)

    x_max = max(float(np.percentile(s_v, 90)), 1.6 * b)
    t = np.linspace(0.0, x_max, 1500)
    far = [far_frr(s_v, p_v, y_v, x)[0] for x in t]
    frr = [far_frr(s_v, p_v, y_v, x)[1] for x in t]
    ax1.axvspan(a, b, color="0.9", lw=0, label=f"intervalo [a, b) = [{_coma(a)}; {_coma(b)})")
    ax1.plot(t, 100 * np.array(frr), color="#c0392b", lw=2, label="FRR (autorizadas no aceptadas)")
    ax1.plot(t, 100 * np.array(far), color="#34495e", lw=2, label="FAR (intrusos aceptados)")
    ax1.axvline(modelo.tau, color="0.15", lw=1.5, label=f"τ* = {_coma(modelo.tau)} (mitad del intervalo)")
    ax1.plot(s_v[autorizada], np.full(int(autorizada.sum()), -4.0), "|", color="#c0392b", ms=9, alpha=0.7)
    ax1.plot(s_v[~autorizada], np.full(int((~autorizada).sum()), -8.0), "|", color="#34495e", ms=9, alpha=0.7)
    ax1.set_xlim(0.0, x_max)
    ax1.set_ylim(-11, 102)
    ax1.set_xlabel("puntaje: distancia normalizada al centroide más cercano")
    ax1.set_ylabel("%")
    ax1.set_title(f"Validación ({int(autorizada.sum())} autorizadas y {int((~autorizada).sum())} intrusos; "
                  f"criterio {umbral['criterio']})", fontsize=10)
    ax1.legend(fontsize=8, loc="center right")
    ax1.grid(alpha=0.3)

    etiquetas = [f"Identidad {i}" for i in modelo.identidades] + ["No autorizado"]
    if test.any():
        s_t, p_t = modelo.puntajes(datos["X"][test])
        r = evaluar(decidir(s_t, p_t, modelo.tau), p_t, datos["identidad"][test], k)
        M = r["matriz"]
        ax2.imshow(M, cmap="Blues", vmin=0, vmax=max(1, int(M[:k, :k].max())))
        for i in range(k + 1):
            for j in range(k + 1):
                ax2.text(j, i, str(int(M[i, j])), ha="center", va="center", fontsize=12,
                         color="white" if M[i, j] > 0.6 * max(1, int(M[:k, :k].max())) else "black")
        ax2.set_title(f"Test: identificación {_coma(100 * r['identificacion'], 1)} %, "
                      f"FAR {_coma(100 * r['far'], 1)} %, FRR {_coma(100 * r['frr'], 1)} %", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "sin filas de test", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xticks(range(k + 1), etiquetas, fontsize=8)
    ax2.set_yticks(range(k + 1), etiquetas, fontsize=8)
    ax2.set_xlabel("decisión")
    ax2.set_ylabel("real")
    figura.suptitle("H4 · K-means propio, distancia normalizada y umbral calibrado en validación", fontsize=11)
    figura.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.88, wspace=0.25)
    buffer = io.BytesIO()
    figura.savefig(buffer, format="png", dpi=110)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(buffer.getvalue())
    return salida


# --------------------------------------------------------------------------- línea de comandos
def _porcentaje(x: float) -> str:
    return f"{_coma(100 * x, 1)} %"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de huellas de H4 (modelo_huellas.npz) y dibuja la curva FAR–FRR (far_frr_h4.png).")
    parser.add_argument("--sobrescribir", action="store_true", help="vuelve a escribir modelo_huellas.npz si ya existe")
    args = parser.parse_args(argv)
    cfg = cargar_config()
    inicio = time.perf_counter()
    try:
        ruta = entrenar(cfg, sobrescribir=args.sobrescribir)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1
    medio = time.perf_counter()
    figura = generar_figura(cfg, modelo=ruta)
    fin = time.perf_counter()
    modelo = cargar_modelo(ruta)
    umbral = cfg["huellas"]["umbral"]
    print(f"Modelo: {ruta} ({_coma(1000 * (medio - inicio), 0)} ms)")
    print(f"Figura: {figura} ({_coma(fin - medio, 1)} s)")
    print(f"K-means en train: pureza {_coma(modelo.pureza)}, inercia {_coma(modelo.inercia, 4)}")
    print("Dispersión r_k: " + ", ".join(f"identidad {i} {_coma(r, 4)}" for i, r in zip(modelo.identidades, modelo.dispersion)))
    print(f"τ* = {_coma(modelo.tau, 4)} (criterio {umbral['criterio']}, far_max_val {umbral['far_max_val']})")
    for particion in ("val", "test"):
        r = evaluar_particion(modelo, cfg, particion=particion)
        print(f"{particion}: identificación {_porcentaje(r['identificacion'])}, FAR {_porcentaje(r['far'])}, "
              f"FRR {_porcentaje(r['frr'])}")
        if particion == "test":
            print("Matriz de confusión de test (filas: real; columnas: decisión; orden 1, 2, 3, no autorizado):")
            for fila in r["matriz"]:
                print("   " + " ".join(f"{int(v):3d}" for v in fila))
    # Las variantes reservadas para la demo (las últimas de cada identidad, en test)
    datos = _leer_caracteristicas(RAIZ / cfg["rutas"]["huellas"])
    archivos = list(datos["archivo"])
    total, reservadas = int(cfg["huellas"]["variantes_por_identidad"]), int(cfg["huellas"]["reservadas_demo"])
    print("Reservadas para la demo (puntaje contra τ*):")
    for i in modelo.identidades:
        for v in range(total - reservadas + 1, total + 1):
            nombre = f"test/id{i}_v{v:03d}.png"
            if nombre not in archivos:
                continue
            fila = archivos.index(nombre)
            puntaje, prediccion = modelo.puntajes(datos["X"][fila])
            decision = modelo.decidir(datos["X"][fila])
            print(f"  {nombre}: {_coma(puntaje)} → {'identidad ' + str(decision) if decision else 'rechazada'} "
                  f"(margen {_coma(modelo.tau - puntaje)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
