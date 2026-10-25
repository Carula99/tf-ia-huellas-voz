"""Clasificación de comandos de voz con K-NN propio, PCA y umbral de rechazo.

Contrato: docs/contratos/v4_clasificacion.md (versión 1.1).
Prohibido importar scikit-learn en este módulo.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.config import RAIZ, cargar_config
from src.core_ml.knn import KNN
from src.core_ml.pca import PCA
from src.core_ml.metricas import matriz_confusion
from src.voz.caracteristicas import (
    BLOQUES,
    Escalador,
    ajustar_escalador,
    caracteristicas,
    largos_de_bloque,
)
from src.voz.preprocesamiento import preprocesar


ESQUEMAS_BLOQUES: tuple[str, ...] = ("contrato", "sin_dur_zcr", "z_columna")
PCA_SIN: None = None

_ULTIMA_TABLA_CV: list[dict] | None = None


def etiquetas_clasificacion(cfg: dict) -> list[str]:
    """Devuelve las 4 etiquetas en orden: 3 comandos y la clase de rechazo."""
    return list(cfg["voz"]["comandos"]) + [str(cfg["voz"]["clase_rechazo"])]


def factores_esquema(esquema: str, cfg: dict) -> np.ndarray:
    """Devuelve un array (5,) float64 con los factores en el orden de BLOQUES."""
    if esquema not in ESQUEMAS_BLOQUES:
        raise ValueError(f"Esquema desconocido: {esquema!r}. Opciones: {ESQUEMAS_BLOQUES}.")
    largos = largos_de_bloque(cfg)
    if esquema == "contrato":
        return np.ones(5, dtype=np.float64)
    if esquema == "sin_dur_zcr":
        return np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    if esquema == "z_columna":
        return np.array([math.sqrt(float(largos[b])) for b in BLOQUES], dtype=np.float64)
    raise ValueError(f"Esquema no implementado: {esquema!r}")


def reponderar_bloques(X: np.ndarray, factores: Any, cfg: dict) -> np.ndarray:
    """Multiplica cada bloque por su factor y descarta bloques con factor 0."""
    arr = np.asarray(X)
    largos = largos_de_bloque(cfg)
    d_esperado = sum(largos[b] for b in BLOQUES)

    if arr.ndim == 1:
        es_vector = True
        X2 = arr[np.newaxis, :]
    elif arr.ndim == 2:
        es_vector = False
        X2 = arr
    else:
        raise ValueError(f"X debe tener 1 o 2 dimensiones, recibido ndim={arr.ndim}.")

    if X2.shape[1] != d_esperado:
        raise ValueError(f"X debe tener {d_esperado} columnas, recibidas {X2.shape[1]}.")
    if not np.isfinite(X2).all():
        raise ValueError("X contiene valores no finitos (NaN o Inf).")

    f = np.asarray(factores, dtype=np.float64)
    if f.shape != (5,):
        raise ValueError(f"factores debe tener 5 elementos, recibido shape={f.shape}.")
    if np.any(f < 0.0):
        raise ValueError("factores no puede tener valores negativos.")

    partes = []
    ini = 0
    for b_i, b in enumerate(BLOQUES):
        ancho = largos[b]
        factor = float(f[b_i])
        if factor != 0.0:
            partes.append(X2[:, ini : ini + ancho] * factor)
        ini += ancho

    if not partes:
        out = np.empty((X2.shape[0], 0), dtype=np.float64)
    else:
        out = np.concatenate(partes, axis=1, dtype=np.float64)

    return out[0] if es_vector else out


def normalizar_l2(X: np.ndarray, piso: float = 1e-12) -> np.ndarray:
    """Normaliza cada fila o vector a norma L2 unitaria."""
    if isinstance(piso, bool) or not isinstance(piso, (int, float, np.floating)) or piso <= 0.0 or not math.isfinite(float(piso)):
        raise ValueError(f"piso debe ser un número real finito > 0, recibido {piso!r}.")

    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim > 2:
        raise ValueError(f"X no puede tener más de 2 dimensiones, recibido ndim={arr.ndim}.")
    if not np.isfinite(arr).all():
        raise ValueError("X contiene valores no finitos (NaN o Inf).")

    norma = np.linalg.norm(arr, axis=-1, keepdims=True)
    den = np.maximum(norma, float(piso))
    return arr / den


@dataclass(frozen=True)
class ModeloVoz:
    """Modelo de clasificación de comandos de voz entrenado."""

    esquema: str
    factores: np.ndarray
    n_componentes: int | None
    pca_media: np.ndarray
    pca_componentes: np.ndarray
    k: int
    eps: float
    confianza_min: float
    tau: float
    entrenamiento: np.ndarray
    etiquetas: np.ndarray
    escalador: Escalador | None

    def transformar(self, X: np.ndarray, cfg: dict) -> np.ndarray:
        """Aplica reponderación, proyección PCA (si hay) y normalización L2 unitaria."""
        R = reponderar_bloques(X, self.factores, cfg)
        arr = np.asarray(R, dtype=np.float64)
        es_vector = arr.ndim == 1
        X2 = arr[np.newaxis, :] if es_vector else arr

        if self.n_componentes is not None and self.pca_componentes.size > 0:
            proj = (X2 - self.pca_media) @ self.pca_componentes.T
        else:
            proj = X2

        norm = normalizar_l2(proj)
        return norm[0] if es_vector else norm

    def votar(self, X: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[Any, float, float]:
        """Voto de los k vecinos más cercanos sobre la consulta transformada."""
        z = self.transformar(X, cfg)
        knn = KNN(k=self.k, eps=self.eps).ajustar(self.entrenamiento, self.etiquetas)
        return knn.votar(z)

    def decidir(self, X: np.ndarray, cfg: dict) -> np.ndarray | str:
        """Aplica la regla de decisión con el umbral tau calibrado."""
        res = self.votar(X, cfg)
        if isinstance(res, tuple) and len(res) == 3:
            pred, conf, dist = res
            return decidir(pred, conf, dist, self.tau, cfg)
        raise RuntimeError("Respuesta inesperada de votar.")


@dataclass(frozen=True)
class Reconocimiento:
    """Resultado del reconocimiento sobre un audio crudo."""

    comando: str
    prediccion: str | None
    confianza: float
    distancia: float
    tau: float
    valido: bool
    motivo: str


def elegir_mejor(tabla: list[dict]) -> dict:
    """Ordena las combinaciones válidas y devuelve un dict con esquema, n_componentes y k."""
    if not tabla:
        raise ValueError("La tabla de combinaciones está vacía.")
    validas = [f for f in tabla if not f.get("descartada", False)]
    if not validas:
        raise ValueError("Todas las combinaciones están descartadas.")

    esquemas_orden = list(ESQUEMAS_BLOQUES)
    pca_orden = [20, 30, None]

    mejor = min(
        validas,
        key=lambda f: (
            -(f["n_tau5"] + f["n_c60"]),
            -f["n_4cl"],
            esquemas_orden.index(f["esquema"]),
            pca_orden.index(f["n_componentes"]),
            f["k"],
        ),
    )
    return {"esquema": mejor["esquema"], "n_componentes": mejor["n_componentes"], "k": mejor["k"]}


def calibrar_umbral(
    distancia: Any,
    confianza: Any,
    prediccion: Any,
    verdad: Any,
    cfg: dict,
    n_otro_no_validos: int = 0,
) -> float:
    """Calibra el umbral de distancia tau* que limita los 'otro' aceptados a <= otro_max."""
    d = np.asarray(distancia, dtype=np.float64)
    c = np.asarray(confianza, dtype=np.float64)
    p = np.asarray(prediccion).astype(str)
    v = np.asarray(verdad).astype(str)

    if not (len(d) == len(c) == len(p) == len(v)):
        raise ValueError("Los arrays de distancia, confianza, prediccion y verdad deben tener igual largo.")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(c))):
        raise ValueError("distancia y confianza contienen valores no finitos.")

    etiquetas = etiquetas_clasificacion(cfg)
    if not (np.isin(p, etiquetas).all() and np.isin(v, etiquetas).all()):
        raise ValueError("prediccion o verdad contienen etiquetas fuera de ETIQUETAS.")

    otro_max = float(cfg["metas"]["voz_otro_aceptado_max"])
    if not (0.0 < otro_max < 1.0):
        raise ValueError(f"otro_max debe estar en (0, 1), recibido {otro_max}.")

    if isinstance(n_otro_no_validos, bool) or not isinstance(n_otro_no_validos, (int, np.integer)) or n_otro_no_validos < 0:
        raise ValueError(f"n_otro_no_validos debe ser un entero >= 0, recibido {n_otro_no_validos!r}.")

    cmin = float(cfg["voz"]["knn"]["confianza_min"])
    comandos = cfg["voz"]["comandos"]
    clase_rechazo = cfg["voz"]["clase_rechazo"]

    es_cmd = np.isin(v, comandos)
    es_otro = v == clase_rechazo

    if not es_cmd.any():
        raise ValueError("No hay filas de comando en los datos de calibración.")

    acep = (p != clase_rechazo) & (c >= cmin)
    n_otro = int(es_otro.sum()) + int(n_otro_no_validos)
    m = int(math.floor(otro_max * n_otro + 1e-9))

    B = np.sort(d[es_otro & acep])
    C = np.sort(d[es_cmd & acep & (p == v)])

    if len(B) > m:
        b = float(B[m])
        menores = C[C < b]
        if menores.size == 0:
            raise ValueError("No hay comandos correctamente clasificados con confianza suficiente por debajo de b.")
        a = float(menores[-1])
    else:
        if C.size == 0:
            raise ValueError("No hay comandos clasificados correctamente con confianza suficiente.")
        a = float(C[-1])
        otros = d[es_otro]
        mayores = otros[otros > a]
        b = float(mayores.min()) if mayores.size > 0 else 1.2 * a

    return float((a + b) / 2.0)


def decidir(
    prediccion: Any,
    confianza: Any,
    distancia: Any,
    tau: float,
    cfg: dict,
) -> np.ndarray | str:
    """Acepta un comando si prediccion != 'otro', confianza >= cmin y distancia <= tau."""
    cmin = float(cfg["voz"]["knn"]["confianza_min"])
    clase_rechazo = str(cfg["voz"]["clase_rechazo"])

    arr_p = np.asarray(prediccion)
    arr_c = np.asarray(confianza)
    arr_d = np.asarray(distancia)

    if arr_p.shape != arr_c.shape or arr_p.shape != arr_d.shape:
        raise ValueError("prediccion, confianza y distancia deben tener la misma forma.")

    if arr_p.ndim == 0:
        val_p = str(arr_p.item())
        if val_p != clase_rechazo and float(arr_c.item()) >= cmin and float(arr_d.item()) <= tau:
            return val_p
        return clase_rechazo

    acepta = (arr_p != clase_rechazo) & (arr_c >= cmin) & (arr_d <= tau)
    out = np.where(acepta, arr_p, clase_rechazo)
    return out.astype(str)


def evaluar(decision: Any, prediccion: Any, verdad: Any, cfg: dict) -> dict:
    """Calcula identificación, acierto, frr, otro_aceptado y la matriz de confusión 4x4."""
    comandos = cfg["voz"]["comandos"]
    clase_rechazo = str(cfg["voz"]["clase_rechazo"])
    etiquetas = etiquetas_clasificacion(cfg)

    d = np.asarray(decision).astype(str)
    p = np.asarray(prediccion).astype(str)
    v = np.asarray(verdad).astype(str)

    if not (len(d) == len(p) == len(v)):
        raise ValueError("decision, prediccion y verdad deben tener el mismo largo.")
    if not (np.isin(d, etiquetas).all() and np.isin(p, etiquetas).all() and np.isin(v, etiquetas).all()):
        raise ValueError("Existen valores fuera de ETIQUETAS.")

    es_cmd = np.isin(v, comandos)
    es_otro = v == clase_rechazo

    if not es_cmd.any():
        raise ValueError("No hay filas de comando.")
    if not es_otro.any():
        raise ValueError("No hay filas de otro.")

    acierto = float(np.mean(d[es_cmd] == v[es_cmd]))
    return {
        "identificacion": float(np.mean(p[es_cmd] == v[es_cmd])),
        "acierto": acierto,
        "frr": float(1.0 - acierto),
        "otro_aceptado": float(np.mean(d[es_otro] != clase_rechazo)),
        "matriz": matriz_confusion(v, d, etiquetas),
    }


def seleccionar_hiperparametros(
    X: Any,
    etiquetas: Any,
    personas: Any,
    evaluable: Any,
    n_otro_no_validos: int,
    cfg: dict,
) -> tuple[dict, list[dict]]:
    """Selección determinista por validación cruzada dejando una persona afuera en train."""
    arr_X = np.asarray(X, dtype=np.float64)
    if arr_X.ndim != 2:
        raise ValueError(f"X debe tener 2 dimensiones, recibido ndim={arr_X.ndim}.")
    n, d = arr_X.shape

    largos = largos_de_bloque(cfg)
    d_esperado = sum(largos[b] for b in BLOQUES)
    if d != d_esperado:
        raise ValueError(f"X debe tener {d_esperado} columnas, recibidas {d}.")
    if not np.isfinite(arr_X).all():
        raise ValueError("X contiene valores no finitos.")

    arr_et = np.asarray(etiquetas).astype(str)
    arr_p = np.asarray(personas).astype(str)
    arr_ev = np.asarray(evaluable, dtype=bool)

    if not (len(arr_et) == len(arr_p) == len(arr_ev) == n):
        raise ValueError("X, etiquetas, personas y evaluable deben tener el mismo número de filas.")

    etiquetas_validas = etiquetas_clasificacion(cfg)
    if not np.isin(arr_et, etiquetas_validas).all():
        raise ValueError("etiquetas contiene valores fuera de ETIQUETAS.")

    if isinstance(n_otro_no_validos, bool) or not isinstance(n_otro_no_validos, (int, np.integer)) or n_otro_no_validos < 0:
        raise ValueError(f"n_otro_no_validos debe ser un entero >= 0, recibido {n_otro_no_validos!r}.")

    comandos = cfg["voz"]["comandos"]
    cmd_eval = arr_ev & np.isin(arr_et, comandos)
    pliegues = sorted(set(arr_p[cmd_eval]))
    if not pliegues:
        raise ValueError("No hay personas con filas evaluables de comando.")

    cmin = float(cfg["voz"]["knn"]["confianza_min"])
    eps = float(cfg["voz"]["knn"]["eps"])
    k_candidatos = list(cfg["voz"]["knn"]["k_candidatos"])
    pca_candidatos = list(cfg["voz"]["knn"]["pca_componentes_candidatas"]) + [None]

    tabla: list[dict] = []

    for esquema in ESQUEMAS_BLOQUES:
        factores = factores_esquema(esquema, cfg)
        R = reponderar_bloques(arr_X, factores, cfg)
        d_r = R.shape[1]

        for n_comp in pca_candidatos:
            for k in k_candidatos:
                # Comprobar restricciones por pliegue
                descartar = False
                for p in pliegues:
                    tr = arr_p != p
                    n_tr = int(tr.sum())
                    if (n_comp is not None and n_comp > min(n_tr - 1, d_r)) or (k > n_tr):
                        descartar = True
                        break

                if descartar:
                    tabla.append({
                        "esquema": esquema,
                        "n_componentes": n_comp,
                        "k": k,
                        "n_tau5": 0,
                        "n_c60": 0,
                        "n_4cl": 0,
                        "tau_cv": float("nan"),
                        "descartada": True,
                    })
                    continue

                dist_t, conf_t, pred_t, verdad_t = [], [], [], []
                for p in pliegues:
                    tr = arr_p != p
                    q = arr_ev & (arr_p == p)

                    R_tr = R[tr]
                    R_q = R[q]

                    if n_comp is not None:
                        pca = PCA(n_comp).ajustar(R_tr)
                        Z_tr = pca.transformar(R_tr)
                        Z_q = pca.transformar(R_q)
                    else:
                        Z_tr = R_tr
                        Z_q = R_q

                    norm_tr = normalizar_l2(Z_tr)
                    norm_q = normalizar_l2(Z_q)

                    knn = KNN(k=k, eps=eps).ajustar(norm_tr, arr_et[tr])
                    pred_q, conf_q, dist_q = knn.votar(norm_q)

                    dist_t.append(dist_q)
                    conf_t.append(conf_q)
                    pred_t.append(pred_q.astype(str))
                    verdad_t.append(arr_et[q])

                dist_arr = np.concatenate(dist_t)
                conf_arr = np.concatenate(conf_t)
                pred_arr = np.concatenate(pred_t)
                verdad_arr = np.concatenate(verdad_t)

                try:
                    tau_cv = calibrar_umbral(
                        dist_arr, conf_arr, pred_arr, verdad_arr, cfg, n_otro_no_validos=n_otro_no_validos
                    )
                except ValueError:
                    tabla.append({
                        "esquema": esquema,
                        "n_componentes": n_comp,
                        "k": k,
                        "n_tau5": 0,
                        "n_c60": 0,
                        "n_4cl": 0,
                        "tau_cv": float("nan"),
                        "descartada": True,
                    })
                    continue

                bien = np.isin(verdad_arr, comandos) & (pred_arr == verdad_arr)
                c60 = bien & (conf_arr >= cmin)
                n_tau5 = int((c60 & (dist_arr <= tau_cv)).sum())
                n_c60 = int(c60.sum())
                n_4cl = int(bien.sum())

                tabla.append({
                    "esquema": esquema,
                    "n_componentes": n_comp,
                    "k": k,
                    "n_tau5": n_tau5,
                    "n_c60": n_c60,
                    "n_4cl": n_4cl,
                    "tau_cv": tau_cv,
                    "descartada": False,
                })

    mejor = elegir_mejor(tabla)
    global _ULTIMA_TABLA_CV
    _ULTIMA_TABLA_CV = tabla
    return mejor, tabla


def ajustar_modelo(
    X_train: Any,
    etiquetas_train: Any,
    personas_train: Any,
    evaluable_train: Any,
    n_otro_no_validos_train: int,
    X_val: Any,
    etiquetas_val: Any,
    n_otro_no_validos_val: int,
    cfg: dict,
    escalador: Escalador | None = None,
) -> ModeloVoz:
    """Ajusta el modelo con train, calibra tau* con val y no mira test."""
    arr_tr = np.asarray(X_train, dtype=np.float64)
    arr_va = np.asarray(X_val, dtype=np.float64)
    largos = largos_de_bloque(cfg)
    d_esperado = sum(largos[b] for b in BLOQUES)

    if arr_tr.ndim != 2 or arr_tr.shape[1] != d_esperado:
        raise ValueError(f"X_train debe ser 2D con {d_esperado} columnas.")
    if arr_va.ndim != 2 or arr_va.shape[1] != d_esperado:
        raise ValueError(f"X_val debe ser 2D con {d_esperado} columnas.")

    et_tr = np.asarray(etiquetas_train).astype(str)
    et_va = np.asarray(etiquetas_val).astype(str)
    if len(et_tr) != arr_tr.shape[0]:
        raise ValueError("Largo de etiquetas_train no coincide con X_train.")
    if len(et_va) != arr_va.shape[0]:
        raise ValueError("Largo de etiquetas_val no coincide con X_val.")

    etiquetas_validas = etiquetas_clasificacion(cfg)
    if not np.isin(et_tr, etiquetas_validas).all() or not np.isin(et_va, etiquetas_validas).all():
        raise ValueError("Etiquetas fuera de ETIQUETAS.")

    mejor, _ = seleccionar_hiperparametros(
        arr_tr, et_tr, personas_train, evaluable_train, n_otro_no_validos_train, cfg
    )

    esquema = mejor["esquema"]
    n_componentes = mejor["n_componentes"]
    k = mejor["k"]
    factores = factores_esquema(esquema, cfg)

    R_train = reponderar_bloques(arr_tr, factores, cfg)

    if n_componentes is not None:
        pca = PCA(n_componentes).ajustar(R_train)
        pca_media = np.asarray(pca.media, dtype=np.float64)
        pca_comp = np.asarray(pca.componentes, dtype=np.float64)
        Z_train = pca.transformar(R_train)
    else:
        pca_media = np.empty(0, dtype=np.float64)
        pca_comp = np.empty((0, 0), dtype=np.float64)
        Z_train = R_train

    entrenamiento = normalizar_l2(Z_train)

    mod_temp = ModeloVoz(
        esquema=esquema,
        factores=factores,
        n_componentes=n_componentes,
        pca_media=pca_media,
        pca_componentes=pca_comp,
        k=k,
        eps=float(cfg["voz"]["knn"]["eps"]),
        confianza_min=float(cfg["voz"]["knn"]["confianza_min"]),
        tau=0.0,
        entrenamiento=entrenamiento,
        etiquetas=et_tr,
        escalador=escalador,
    )

    pred_va, conf_va, dist_va = mod_temp.votar(arr_va, cfg)
    tau = calibrar_umbral(dist_va, conf_va, pred_va, et_va, cfg, n_otro_no_validos=n_otro_no_validos_val)

    return dataclasses.replace(mod_temp, tau=tau)


def guardar_modelo(modelo: ModeloVoz, ruta: str | os.PathLike) -> Path:
    """Guarda el modelo entrenado en un archivo .npz con exactamente 27 claves."""
    if modelo.escalador is None:
        raise ValueError("El modelo no tiene escalador asociado.")

    path = Path(ruta)
    path.parent.mkdir(parents=True, exist_ok=True)

    datos: dict[str, Any] = {
        "esquema": np.array(str(modelo.esquema)),
        "factores": np.asarray(modelo.factores, dtype=np.float64),
        "n_componentes": np.int64(modelo.n_componentes if modelo.n_componentes is not None else 0),
        "k": np.int64(modelo.k),
        "eps": np.float64(modelo.eps),
        "confianza_min": np.float64(modelo.confianza_min),
        "tau": np.float64(modelo.tau),
        "entrenamiento": np.asarray(modelo.entrenamiento, dtype=np.float64),
        "etiquetas": np.asarray(modelo.etiquetas).astype(str),
        "pca_media": np.asarray(modelo.pca_media, dtype=np.float64),
        "pca_componentes": np.asarray(modelo.pca_componentes, dtype=np.float64),
        "pesos": np.array([float(modelo.escalador.pesos[b]) for b in BLOQUES], dtype=np.float64),
    }

    for b in BLOQUES:
        datos[f"medias_{b}"] = np.asarray(modelo.escalador.medias[b], dtype=np.float64)
        datos[f"desvios_{b}"] = np.asarray(modelo.escalador.desvios[b], dtype=np.float64)
        datos[f"columnas_{b}"] = np.asarray(modelo.escalador.columnas[b])

    np.savez(path, **datos)
    return path


def cargar_modelo(ruta: str | os.PathLike) -> ModeloVoz:
    """Carga un modelo guardado con guardar_modelo."""
    path = Path(ruta)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de modelo: {path}")

    with np.load(path, allow_pickle=False) as data:
        n_comp_raw = int(data["n_componentes"])
        n_comp = None if n_comp_raw == 0 else n_comp_raw

        pesos = {b: float(data["pesos"][i]) for i, b in enumerate(BLOQUES)}
        medias = {b: np.asarray(data[f"medias_{b}"]) for b in BLOQUES}
        desvios = {b: np.asarray(data[f"desvios_{b}"]) for b in BLOQUES}
        columnas = {b: np.asarray(data[f"columnas_{b}"]) for b in BLOQUES}
        escalador = Escalador(medias=medias, desvios=desvios, columnas=columnas, pesos=pesos)

        return ModeloVoz(
            esquema=str(data["esquema"].item() if hasattr(data["esquema"], "item") else data["esquema"]),
            factores=np.asarray(data["factores"], dtype=np.float64),
            n_componentes=n_comp,
            pca_media=np.asarray(data["pca_media"], dtype=np.float64),
            pca_componentes=np.asarray(data["pca_componentes"], dtype=np.float64),
            k=int(data["k"]),
            eps=float(data["eps"]),
            confianza_min=float(data["confianza_min"]),
            tau=float(data["tau"]),
            entrenamiento=np.asarray(data["entrenamiento"], dtype=np.float64),
            etiquetas=np.asarray(data["etiquetas"]).astype(str),
            escalador=escalador,
        )


def _mapear_personas(hablantes: Any, cfg: dict) -> np.ndarray:
    """Mapea cada hablante a su persona de referencia según voz.dataset.mismo_hablante."""
    mapa: dict[str, str] = {}
    for grupo in cfg["voz"]["dataset"]["mismo_hablante"]:
        ref = str(grupo[0])
        for cod in grupo:
            mapa[str(cod)] = ref
    return np.array([mapa.get(str(h), str(h)) for h in hablantes])


def entrenar(
    cfg: dict,
    carpeta: str | os.PathLike | None = None,
    salida: str | os.PathLike | None = None,
    sobrescribir: bool = False,
) -> Path:
    """Entrena y calibra el modelo sobre las particiones train y val de carpeta."""
    c_path = Path(carpeta) if carpeta is not None else RAIZ / cfg["rutas"]["voz"] / "prepro"
    s_path = Path(salida) if salida is not None else RAIZ / cfg["rutas"]["modelos"] / "modelo_voz.npz"

    if s_path.exists() and not sobrescribir:
        raise FileExistsError(f"El archivo {s_path} ya existe y sobrescribir es False.")

    caract_file = c_path / "caracteristicas.npz"
    meta_file = c_path / "metadatos.csv"

    if not caract_file.is_file():
        raise FileNotFoundError(f"Falta el archivo: {caract_file}")
    if not meta_file.is_file():
        raise FileNotFoundError(f"Falta el archivo: {meta_file}")

    with np.load(caract_file, allow_pickle=False) as npz:
        part = npz["particion"].astype(str)
        tr = part == "train"
        va = part == "val"

        if int(tr.sum()) == 0 or int(va.sum()) == 0:
            raise ValueError("No hay filas de train o de val en caracteristicas.npz.")

        X_tr = npz["X"][tr]
        y_tr = npz["clase"][tr].astype(str)
        hab_tr = npz["hablante"][tr]
        orig_tr = npz["origen"][tr].astype(str)
        copia_tr = npz["copia"][tr]

        X_va = npz["X"][va]
        y_va = npz["clase"][va].astype(str)

        personas_tr = _mapear_personas(hab_tr, cfg)
        evaluable_tr = (orig_tr == "notebook") & (copia_tr == 0)

        bloques_tr = {b: npz[b][tr] for b in BLOQUES}
        escalador = ajustar_escalador(bloques_tr)

    # Leer metadatos.csv para contar no válidos
    clase_rechazo = str(cfg["voz"]["clase_rechazo"])
    n_otro_no_validos_train = 0
    n_otro_no_validos_val = 0

    with open(meta_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            particion = fila.get("particion", "")
            valido_str = fila.get("valido", "0")
            es_valido = valido_str in ("1", "True", "true", 1)
            clase = fila.get("clase", "")
            origen = fila.get("origen", "")
            copia = fila.get("copia", "0")

            if not es_valido and clase == clase_rechazo:
                if particion == "train" and origen == "notebook" and str(copia) == "0":
                    n_otro_no_validos_train += 1
                elif particion == "val":
                    n_otro_no_validos_val += 1

    modelo = ajustar_modelo(
        X_tr,
        y_tr,
        personas_tr,
        evaluable_tr,
        n_otro_no_validos_train,
        X_va,
        y_va,
        n_otro_no_validos_val,
        cfg,
        escalador=escalador,
    )

    return guardar_modelo(modelo, s_path)


def evaluar_particion(
    modelo: ModeloVoz,
    cfg: dict,
    carpeta: str | os.PathLike | None = None,
    particion: str = "test",
    sin_umbral: bool = False,
) -> dict:
    """Evalúa el modelo sobre una partición ('train', 'val' o 'test') sumando no válidos."""
    if particion not in ("train", "val", "test"):
        raise ValueError(f"particion debe ser 'train', 'val' o 'test', recibido {particion!r}.")

    c_path = Path(carpeta) if carpeta is not None else RAIZ / cfg["rutas"]["voz"] / "prepro"
    caract_file = c_path / "caracteristicas.npz"
    meta_file = c_path / "metadatos.csv"

    if not caract_file.is_file() or not meta_file.is_file():
        raise FileNotFoundError("Falta caracteristicas.npz o metadatos.csv.")

    with np.load(caract_file, allow_pickle=False) as npz:
        part = npz["particion"].astype(str)
        mask = part == particion
        X_val = npz["X"][mask]
        y_val = npz["clase"][mask].astype(str)

    if X_val.shape[0] > 0:
        pred_val, conf_val, dist_val = modelo.votar(X_val, cfg)
        if sin_umbral:
            dec_val = decidir(pred_val, conf_val, dist_val, float("inf"), cfg)
        else:
            dec_val = decidir(pred_val, conf_val, dist_val, modelo.tau, cfg)
    else:
        pred_val = np.empty(0, dtype="<U12")
        dec_val = np.empty(0, dtype="<U12")

    clases_no_validas: list[str] = []
    with open(meta_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            if fila.get("particion") == particion:
                v_str = fila.get("valido", "0")
                if v_str not in ("1", "True", "true", 1):
                    clases_no_validas.append(fila.get("clase", "otro"))

    n_nv = len(clases_no_validas)
    if n_nv > 0:
        y_nv = np.array(clases_no_validas, dtype="<U12")
        pred_nv = np.full(n_nv, "otro")
        dec_nv = np.full(n_nv, "otro")

        y_tot = np.concatenate([y_val, y_nv])
        pred_tot = np.concatenate([pred_val, pred_nv])
        dec_tot = np.concatenate([dec_val, dec_nv])
    else:
        y_tot = y_val
        pred_tot = pred_val
        dec_tot = dec_val

    return evaluar(dec_tot, pred_tot, y_tot, cfg)


def reconocer(audio: np.ndarray, fs: int, modelo: ModeloVoz, cfg: dict) -> Reconocimiento:
    """Cadena completa de reconocimiento: preprocesamiento -> características -> K-NN -> decisión."""
    if modelo.escalador is None:
        raise ValueError("El modelo no tiene escalador asociado.")

    prep = preprocesar(audio, fs, cfg)
    if not prep.valido:
        return Reconocimiento(
            comando="otro",
            prediccion=None,
            confianza=0.0,
            distancia=float("inf"),
            tau=float(modelo.tau),
            valido=False,
            motivo=str(prep.motivo),
        )

    bloques = caracteristicas(prep.y, cfg)
    z_escalado = modelo.escalador.transformar(bloques)
    pred, conf, dist = modelo.votar(z_escalado, cfg)
    cmd = decidir(pred, conf, dist, modelo.tau, cfg)

    return Reconocimiento(
        comando=str(cmd),
        prediccion=str(pred),
        confianza=float(conf),
        distancia=float(dist),
        tau=float(modelo.tau),
        valido=True,
        motivo="",
    )


def generar_figura(
    cfg: dict,
    carpeta: str | os.PathLike | None = None,
    modelo: str | os.PathLike | None = None,
    salida: str | os.PathLike | None = None,
) -> Path:
    """Genera la figura de rechazo en función de tau y matriz de confusión de test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_path = Path(carpeta) if carpeta is not None else RAIZ / cfg["rutas"]["voz"] / "prepro"
    m_path = Path(modelo) if modelo is not None else RAIZ / cfg["rutas"]["modelos"] / "modelo_voz.npz"
    s_path = Path(salida) if salida is not None else RAIZ / cfg["rutas"]["figuras"] / "rechazo_v4.png"

    s_path.parent.mkdir(parents=True, exist_ok=True)
    mod = cargar_modelo(m_path)

    # Evaluar val en función de tau
    with np.load(c_path / "caracteristicas.npz", allow_pickle=False) as npz:
        mask_va = npz["particion"].astype(str) == "val"
        X_va = npz["X"][mask_va]
        y_va = npz["clase"][mask_va].astype(str)

    pred_va, conf_va, dist_va = mod.votar(X_va, cfg)

    clases_nv_va: list[str] = []
    with open(c_path / "metadatos.csv", mode="r", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            if fila.get("particion") == "val":
                v_str = fila.get("valido", "0")
                if v_str not in ("1", "True", "true", 1):
                    clases_nv_va.append(fila.get("clase", "otro"))

    n_nv_va = len(clases_nv_va)
    y_tot_va = np.concatenate([y_va, np.array(clases_nv_va, dtype="<U12")])
    pred_tot_va = np.concatenate([pred_va, np.full(n_nv_va, "otro")])
    conf_tot_va = np.concatenate([conf_va, np.zeros(n_nv_va, dtype=np.float64)])
    dist_tot_va = np.concatenate([dist_va, np.full(n_nv_va, float("inf"), dtype=np.float64)])

    taus = np.linspace(0.4, 1.4, 100)
    aciertos = []
    otros_ac = []

    for t in taus:
        dec_t = decidir(pred_tot_va, conf_tot_va, dist_tot_va, float(t), cfg)
        ev_t = evaluar(dec_t, pred_tot_va, y_tot_va, cfg)
        aciertos.append(ev_t["acierto"] * 100.0)
        otros_ac.append(ev_t["otro_aceptado"] * 100.0)

    # Evaluar test para la matriz
    res_test = evaluar_particion(mod, cfg, c_path, "test")
    mat_test = res_test["matriz"]
    etiquetas = etiquetas_clasificacion(cfg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(taus, aciertos, label="Acierto comandos (val)", color="#1f77b4", lw=2)
    ax1.plot(taus, otros_ac, label="'otro' aceptado (val)", color="#d62728", lw=2)
    ax1.axvline(mod.tau, color="black", linestyle="--", label=f"τ* = {mod.tau:.4f}")
    ax1.axhline(5.0, color="gray", linestyle=":", label="Meta 'otro' (5 %)")
    ax1.set_xlabel("Umbral de distancia τ")
    ax1.set_ylabel("Porcentaje (%)")
    ax1.set_title("Calibración del umbral de rechazo (val)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    im = ax2.imshow(mat_test, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xticks(range(len(etiquetas)))
    ax2.set_yticks(range(len(etiquetas)))
    ax2.set_xticklabels(etiquetas, rotation=45, ha="right")
    ax2.set_yticklabels(etiquetas)
    ax2.set_xlabel("Decisión")
    ax2.set_ylabel("Etiqueta real")
    ax2.set_title("Matriz de confusión (test)")

    for i in range(mat_test.shape[0]):
        for j in range(mat_test.shape[1]):
            val = mat_test[i, j]
            color = "white" if val > mat_test.max() / 2 else "black"
            ax2.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(s_path, dpi=150)
    plt.close(fig)

    return s_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluación del clasificador de voz V4.")
    parser.add_argument("--sobrescribir", action="store_true", help="Sobrescribe el modelo si ya existe.")
    args = parser.parse_args()

    cfg = cargar_config()
    salida = entrenar(cfg, sobrescribir=args.sobrescribir)
    modelo = cargar_modelo(salida)

    print(f"\nModelo entrenado guardado en: {salida}")
    print(f"Hiperparámetros elegidos: esquema={modelo.esquema}, n_componentes={modelo.n_componentes}, K={modelo.k}")

    # Obtener tabla de validación cruzada
    tabla = _ULTIMA_TABLA_CV
    if tabla is None:
        c_path = RAIZ / cfg["rutas"]["voz"] / "prepro"
        with np.load(c_path / "caracteristicas.npz", allow_pickle=False) as npz:
            tr = npz["particion"].astype(str) == "train"
            X_tr = npz["X"][tr]
            y_tr = npz["clase"][tr].astype(str)
            hab_tr = npz["hablante"][tr]
            orig_tr = npz["origen"][tr].astype(str)
            copia_tr = npz["copia"][tr]
            personas_tr = _mapear_personas(hab_tr, cfg)
            evaluable_tr = (orig_tr == "notebook") & (copia_tr == 0)

        clase_rechazo = str(cfg["voz"]["clase_rechazo"])
        n_nv_tr = 0
        with open(c_path / "metadatos.csv", mode="r", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                if fila.get("particion") == "train" and fila.get("valido", "0") not in ("1", "True", "true", 1):
                    if fila.get("clase") == clase_rechazo and fila.get("origen") == "notebook" and str(fila.get("copia")) == "0":
                        n_nv_tr += 1
        _, tabla = seleccionar_hiperparametros(X_tr, y_tr, personas_tr, evaluable_tr, n_nv_tr, cfg)

    esquemas_orden = list(ESQUEMAS_BLOQUES)
    pca_orden = [20, 30, None]
    validas = [f for f in tabla if not f.get("descartada", False)]
    validas_ord = sorted(
        validas,
        key=lambda f: (
            -(f["n_tau5"] + f["n_c60"]),
            -f["n_4cl"],
            esquemas_orden.index(f["esquema"]),
            pca_orden.index(f["n_componentes"]),
            f["k"],
        ),
    )
    top5 = validas_ord[:5]

    print("\nLas 5 mejores filas de la tabla de la validación cruzada por hablante:")
    print(f" {'suma':>5} | {'n_4cl':>5} | {'n_tau5':>6} | {'n_c60':>5} | {'esquema':<12} | {'PCA':>4} | {'K':>2} | {'tau_cv':>7}")
    print("-" * 65)
    for f in top5:
        suma = f["n_tau5"] + f["n_c60"]
        pca_str = str(f["n_componentes"]) if f["n_componentes"] is not None else "None"
        print(f" {suma:5d} | {f['n_4cl']:5d} | {f['n_tau5']:6d} | {f['n_c60']:5d} | {f['esquema']:<12} | {pca_str:>4} | {f['k']:2d} | {f['tau_cv']:7.4f}")

    print(f"\nUmbral de distancia calibrado en val: tau* = {modelo.tau:.4f}\n")

    eval_val_st = evaluar_particion(modelo, cfg, particion="val", sin_umbral=True)
    eval_val_tau = evaluar_particion(modelo, cfg, particion="val", sin_umbral=False)

    print("--- Evaluación en VAL ---")
    print(f"Sin umbral de distancia (solo confianza >= {modelo.confianza_min:.2f}):")
    print(f"  Identificación: {eval_val_st['identificacion'] * 100:.2f} %")
    print(f"  Acierto:        {eval_val_st['acierto'] * 100:.2f} %")
    print(f"  FRR:            {eval_val_st['frr'] * 100:.2f} %")
    print(f"  Otro aceptado:  {eval_val_st['otro_aceptado'] * 100:.2f} %")
    print(f"Con umbral de distancia tau* = {modelo.tau:.4f}:")
    print(f"  Identificación: {eval_val_tau['identificacion'] * 100:.2f} %")
    print(f"  Acierto:        {eval_val_tau['acierto'] * 100:.2f} %")
    print(f"  FRR:            {eval_val_tau['frr'] * 100:.2f} %")
    print(f"  Otro aceptado:  {eval_val_tau['otro_aceptado'] * 100:.2f} %")
    print("Matriz de confusión (val):")
    print(eval_val_tau["matriz"])

    eval_test_st = evaluar_particion(modelo, cfg, particion="test", sin_umbral=True)
    eval_test_tau = evaluar_particion(modelo, cfg, particion="test", sin_umbral=False)

    print("\n--- Evaluación en TEST ---")
    print(f"Sin umbral de distancia (solo confianza >= {modelo.confianza_min:.2f}):")
    print(f"  Identificación: {eval_test_st['identificacion'] * 100:.2f} %")
    print(f"  Acierto:        {eval_test_st['acierto'] * 100:.2f} %")
    print(f"  FRR:            {eval_test_st['frr'] * 100:.2f} %")
    print(f"  Otro aceptado:  {eval_test_st['otro_aceptado'] * 100:.2f} %")
    print(f"Con umbral de distancia tau* = {modelo.tau:.4f}:")
    print(f"  Identificación: {eval_test_tau['identificacion'] * 100:.2f} %")
    print(f"  Acierto:        {eval_test_tau['acierto'] * 100:.2f} %")
    print(f"  FRR:            {eval_test_tau['frr'] * 100:.2f} %")
    print(f"  Otro aceptado:  {eval_test_tau['otro_aceptado'] * 100:.2f} %")
    print("Matriz de confusión (test):")
    print(eval_test_tau["matriz"])

    fig_path = generar_figura(cfg)
    print(f"\nFigura generada en: {fig_path}")

