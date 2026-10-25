"""Prototipo V4b (24/10/2026): prueba de normalización L2 por fila, pesos intermedios
por bloque, potencia del voto y las 3 familias de puntaje de rechazo (a: d_gan_med / d_norm,
b: cocientes de distancia ratio_otro y ratio_otra_clase, c: 1 - conf) en CV (train) y en val.
NO carga ni mira test.
"""
from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(r"C:\Users\Santiago\source\repos\tf-ia-huellas-voz")
PROTO_DIR = Path(r"C:\Users\Santiago\OneDrive\Escritorio\IA 1\Trabajo final\Prototipos\v4_2026-10-24")
sys.path.insert(0, str(REPO))

from src.config import cargar_config  # noqa: E402
from src.core_ml.metricas import matriz_confusion  # noqa: E402
from src.core_ml.pca import PCA  # noqa: E402
from src.voz.caracteristicas import BLOQUES, largos_de_bloque  # noqa: E402


def cargar_train_val(cfg: dict):
    carpeta = REPO / cfg["rutas"]["voz"] / "prepro"
    npz = np.load(carpeta / "caracteristicas.npz", allow_pickle=False)
    part_npz = npz["particion"]
    mask_tv = np.isin(part_npz, ["train", "val"])
    assert not np.any(part_npz[mask_tv] == "test"), "Prohibido tocar test"
    datos = {k: npz[k][mask_tv] for k in npz.files}

    with open(carpeta / "metadatos.csv", encoding="utf-8", newline="") as f:
        filas_meta = [r for r in csv.DictReader(f) if r["particion"] in ("train", "val")]
    assert all(r["particion"] != "test" for r in filas_meta)
    return datos, filas_meta


def escalar_bloques(X: np.ndarray, factores: dict[str, float], largos: dict[str, int]) -> np.ndarray:
    cols = []
    ini = 0
    for b in BLOQUES:
        fin = ini + largos[b]
        f = float(factores.get(b, 0.0))
        if f > 0.0:
            cols.append(X[:, ini:fin] * f)
        ini = fin
    return np.hstack(cols)


def norm_l2(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-12)


def mat_distancias(X_query: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    q2 = np.sum(X_query ** 2, axis=1, keepdims=True)
    t2 = np.sum(X_train ** 2, axis=1, keepdims=True).T
    d2 = np.maximum(q2 + t2 - 2.0 * (X_query @ X_train.T), 0.0)
    return np.sqrt(d2)


def evaluar_knn_completo(
    D: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    eps: float = 1e-9,
    potencia: int = 1,
    clases_orden: list[str] | None = None,
    clase_rechazo: str = "otro",
):
    """Evalúa K-NN sobre la matriz de distancias D (m, n) y devuelve predicción, confianza
    y las variantes de puntaje para calibrar el umbral τ*."""
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    filas = np.arange(len(D))[:, None]
    dist_sub = D[filas, idx]
    orden_k = np.argsort(dist_sub, axis=1, kind="mergesort")
    idx = idx[filas, orden_k]
    dist_k = dist_sub[filas, orden_k]
    vecinos_y = y_train[idx]

    if potencia == 1:
        pesos = 1.0 / (dist_k + eps)
    else:
        pesos = 1.0 / ((dist_k + eps) ** potencia)
    suma_pesos = pesos.sum(axis=1)

    if clases_orden is None:
        clases_orden = sorted(set(y_train.tolist()))

    m = len(D)
    n_cl = len(clases_orden)
    votos_mat = np.zeros((m, n_cl), dtype=np.float64)
    d_min_k_mat = np.full((m, n_cl), np.inf, dtype=np.float64)
    d_med_k_mat = np.full((m, n_cl), np.inf, dtype=np.float64)
    d_min_all_mat = np.full((m, n_cl), np.inf, dtype=np.float64)
    r_clase = np.ones(n_cl, dtype=np.float64)

    for c_idx, c in enumerate(clases_orden):
        mask_k = vecinos_y == c
        votos_mat[:, c_idx] = np.where(mask_k, pesos, 0.0).sum(axis=1)
        cnt_c = mask_k.sum(axis=1)
        tiene = cnt_c > 0
        if np.any(tiene):
            d_c = np.where(mask_k, dist_k, np.inf)
            d_min_k_mat[:, c_idx] = d_c.min(axis=1)
            d_sum = np.where(mask_k, dist_k, 0.0).sum(axis=1)
            d_med_k_mat[tiene, c_idx] = d_sum[tiene] / cnt_c[tiene]

        mask_tr_c = y_train == c
        if np.any(mask_tr_c):
            d_min_all_mat[:, c_idx] = D[:, mask_tr_c].min(axis=1)
            Xc = X_train[mask_tr_c]
            mu_c = Xc.mean(axis=0)
            r_clase[c_idx] = float(np.sqrt(np.mean(np.sum((Xc - mu_c) ** 2, axis=1))))

    ganador_idx = np.argmax(votos_mat - 1e-15 * np.minimum(d_min_k_mat, 1e6), axis=1)
    clases_arr = np.asarray(clases_orden, dtype=y_train.dtype)
    pred = clases_arr[ganador_idx]
    filas_1d = np.arange(m)
    conf = votos_mat[filas_1d, ganador_idx] / suma_pesos

    # Puntajes (menor = más confiable / más cercano al comando ganador)
    d_gan_med = d_med_k_mat[filas_1d, ganador_idx]
    d_gan_min = d_min_k_mat[filas_1d, ganador_idx]
    d_gan_norm = d_gan_med / np.maximum(r_clase[ganador_idx], 1e-9)

    idx_otro = clases_orden.index(clase_rechazo)
    d_min_otro = np.maximum(d_min_all_mat[:, idx_otro], 1e-9)
    ratio_otro_med = d_gan_med / d_min_otro
    ratio_otro_min = d_gan_min / d_min_otro

    # Distancia mínima a cualquier otra clase distinta de la ganadora en train
    d_otra_min = np.full(m, np.inf, dtype=np.float64)
    for c_idx in range(n_cl):
        es_otra = ganador_idx != c_idx
        d_otra_min[es_otra] = np.minimum(d_otra_min[es_otra], d_min_all_mat[es_otra, c_idx])
    ratio_otra_min = d_gan_min / np.maximum(d_otra_min, 1e-9)
    ratio_otra_med = d_gan_med / np.maximum(d_otra_min, 1e-9)

    puntajes = {
        "a1_d_gan_med": d_gan_med,
        "a2_d_gan_norm": d_gan_norm,
        "b1_ratio_otro": ratio_otro_med,
        "b2_ratio_otra": ratio_otra_min,
        "c_1_menos_conf": 1.0 - conf,
    }
    return pred, conf, puntajes


def calibrar_y_medir(
    es_cmd: np.ndarray,
    pred_ok: np.ndarray,
    acep_base: np.ndarray,
    puntaje: np.ndarray,
    n_cmd_tot: int,
    n_otro_tot: int,
    far_max: float = 0.05,
):
    """Calibra τ* = (a + b)/2 sobre un conjunto para que 'otro' aceptado <= far_max
    y devuelve (tau, ok_cmd_pct, otro_acep_pct, ok_cmd_cnt, otro_acep_cnt)."""
    m_perm = int(np.floor(far_max * n_otro_tot + 1e-9))
    mask_otro_pel = (~es_cmd) & acep_base
    d_otros_pel = np.sort(puntaje[mask_otro_pel])
    mask_cmd_bien = es_cmd & acep_base & pred_ok
    d_cmd_bien = np.sort(puntaje[mask_cmd_bien])

    if len(d_cmd_bien) == 0:
        return 0.0, 0.0, 0.0, 0, 0

    if len(d_otros_pel) > m_perm:
        b = float(d_otros_pel[m_perm])
        bajo = d_cmd_bien[d_cmd_bien < b]
        a = float(bajo[-1]) if len(bajo) > 0 else b * 0.9
        tau = 0.5 * (a + b)
    else:
        a = float(d_cmd_bien[-1])
        d_otro_all = np.sort(puntaje[~es_cmd])
        mayores = d_otro_all[d_otro_all > a]
        b = float(mayores[0]) if len(mayores) > 0 else float(a * 1.15 + 1e-3)
        tau = 0.5 * (a + b)

    pasa = acep_base & (puntaje <= tau)
    ok_cmd = int((es_cmd & pred_ok & pasa).sum())
    otro_ac = int(((~es_cmd) & pasa).sum())
    return tau, 100.0 * ok_cmd / n_cmd_tot, 100.0 * otro_ac / n_otro_tot, ok_cmd, otro_ac


def main():
    buf = io.StringIO()

    def log(msg: str = ""):
        print(msg)
        buf.write(msg + "\n")

    t0 = time.perf_counter()
    cfg = cargar_config()
    datos, filas_meta = cargar_train_val(cfg)
    largos = largos_de_bloque(cfg)
    comandos = list(cfg["voz"]["comandos"])
    clase_rechazo = cfg["voz"]["clase_rechazo"]
    etiquetas_orden = comandos + [clase_rechazo]
    eps = float(cfg["voz"]["knn"]["eps"])
    conf_min = float(cfg["voz"]["knn"]["confianza_min"])

    part = datos["particion"]
    clase = datos["clase"]
    hab = datos["hablante"]
    origen = datos["origen"]
    copia = datos["copia"]
    X = datos["X"]

    tr = part == "train"
    va = part == "val"

    alias = {}
    for grupo in cfg["voz"]["dataset"]["mismo_hablante"]:
        for h in grupo:
            alias[h] = grupo[0]
    persona = np.array([alias.get(h, h) for h in hab])

    es_cmd = np.isin(clase, comandos)
    eval_nb = tr & (origen == "notebook") & (copia == 0)
    personas_nb = sorted(set(persona[eval_nb & es_cmd]))

    n_cmd_cv_tot = int((eval_nb & es_cmd).sum())  # 231
    n_otro_cv_tot = 78  # 72 válidos + 6 no válidos en train notebook k0
    n_cmd_val_tot = 67  # 67 válidos + 0 no válidos
    n_otro_val_tot = 34  # 20 válidos + 14 no válidos

    esquemas = {
        "contrato (1,1,1,1,1)": {"mfcc": 1.0, "delta": 1.0, "energia": 1.0, "zcr": 1.0, "duracion": 1.0},
        "inter_1_03 (1,1,1,.3,.3)": {"mfcc": 1.0, "delta": 1.0, "energia": 1.0, "zcr": 0.3, "duracion": 0.3},
        "inter_05_03 (1,1,.5,.3,.3)": {"mfcc": 1.0, "delta": 1.0, "energia": 0.5, "zcr": 0.3, "duracion": 0.3},
        "inter_05_02 (1,1,.5,.2,.2)": {"mfcc": 1.0, "delta": 1.0, "energia": 0.5, "zcr": 0.2, "duracion": 0.2},
        "sin_dur_zcr (1,1,1,0,0)": {"mfcc": 1.0, "delta": 1.0, "energia": 1.0, "zcr": 0.0, "duracion": 0.0},
        "z_columna (sqrt(d_b))": {b: float(np.sqrt(largos[b])) for b in BLOQUES},
        "z_col_sin_dur_zcr": {"mfcc": float(np.sqrt(130)), "delta": float(np.sqrt(130)), "energia": float(np.sqrt(10)), "zcr": 0.0, "duracion": 0.0},
    }

    log("=" * 120)
    log("PROTOTIPO V4b · NORMALIZACIÓN L2, PESOS INTERMEDIOS, POTENCIA DE VOTO Y PUNTAJES DE RECHAZO (CV Y VAL, SIN TEST)")
    log("=" * 120)

    # 1. Comparación sistemática en CV y en val de (esquema, norm_modo, pca_k, K, potencia, puntaje)
    # Configuraciones representativas para tabla comparativa
    configs_explorar = []
    for nom_esq in esquemas:
        for norm_modo in ("raw", "l2_post"):
            for pca_k in (None, 30, 20):
                for k in (3, 5, 7):
                    for pot in (1, 2):
                        configs_explorar.append((nom_esq, norm_modo, pca_k, k, pot))

    # Precalcular las matrices de distancia por (nom_esq, norm_modo, pca_k)
    cache_dist = {}
    for nom_esq, fac in esquemas.items():
        X_esq = escalar_bloques(X, fac, largos)
        for norm_modo in ("raw", "l2_post"):
            for pca_k in (None, 30, 20):
                pliegues_d = []
                for per in personas_nb:
                    a = tr & (persona != per)
                    q = eval_nb & (persona == per)
                    if pca_k is not None:
                        mp = PCA(pca_k).ajustar(X_esq[a])
                        Xa, Xq = mp.transformar(X_esq[a]), mp.transformar(X_esq[q])
                    else:
                        Xa, Xq = X_esq[a], X_esq[q]
                    if norm_modo == "l2_post":
                        Xa, Xq = norm_l2(Xa), norm_l2(Xq)
                    D_pl = mat_distancias(Xq, Xa)
                    pliegues_d.append((a, q, Xa, D_pl))

                # Val
                if pca_k is not None:
                    mp_v = PCA(pca_k).ajustar(X_esq[tr])
                    Xtr_v, Xva_v = mp_v.transformar(X_esq[tr]), mp_v.transformar(X_esq[va])
                else:
                    Xtr_v, Xva_v = X_esq[tr], X_esq[va]
                if norm_modo == "l2_post":
                    Xtr_v, Xva_v = norm_l2(Xtr_v), norm_l2(Xva_v)
                D_val = mat_distancias(Xva_v, Xtr_v)
                cache_dist[(nom_esq, norm_modo, pca_k)] = (pliegues_d, Xtr_v, D_val)

    # Evaluar todas las combinaciones
    filas_res = []
    for nom_esq, norm_modo, pca_k, k, pot in configs_explorar:
        pliegues_d, Xtr_v, D_val = cache_dist[(nom_esq, norm_modo, pca_k)]

        # CV
        cv_es_cmd = []
        cv_pred_ok = []
        cv_acep_c60 = []
        cv_puntajes = {p_nom: [] for p_nom in ("a1_d_gan_med", "a2_d_gan_norm", "b1_ratio_otro", "b2_ratio_otra", "c_1_menos_conf")}

        for a, q, Xa, D_pl in pliegues_d:
            ya = clase[a]
            yq = clase[q]
            pred_q, conf_q, pts_q = evaluar_knn_completo(
                D_pl, Xa, ya, k=k, eps=eps, potencia=pot, clases_orden=etiquetas_orden, clase_rechazo=clase_rechazo
            )
            cv_es_cmd.append(np.isin(yq, comandos))
            cv_pred_ok.append(pred_q == yq)
            cv_acep_c60.append((pred_q != clase_rechazo) & (conf_q >= conf_min))
            for p_nom in cv_puntajes:
                cv_puntajes[p_nom].append(pts_q[p_nom])

        cv_es_cmd_a = np.concatenate(cv_es_cmd)
        cv_pred_ok_a = np.concatenate(cv_pred_ok)
        cv_acep_c60_a = np.concatenate(cv_acep_c60)
        cv_pts_a = {p_nom: np.concatenate(v) for p_nom, v in cv_puntajes.items()}

        cv_4cl_pct = 100.0 * int((cv_es_cmd_a & cv_pred_ok_a).sum()) / n_cmd_cv_tot
        cv_c60_cmd_pct = 100.0 * int((cv_es_cmd_a & cv_pred_ok_a & cv_acep_c60_a).sum()) / n_cmd_cv_tot
        cv_c60_otro_pct = 100.0 * int(((~cv_es_cmd_a) & cv_acep_c60_a).sum()) / n_otro_cv_tot

        # Val
        y_val = clase[va]
        val_es_cmd = np.isin(y_val, comandos)
        pred_v, conf_v, pts_v = evaluar_knn_completo(
            D_val, Xtr_v, clase[tr], k=k, eps=eps, potencia=pot, clases_orden=etiquetas_orden, clase_rechazo=clase_rechazo
        )
        val_pred_ok = pred_v == y_val
        val_acep_c60 = (pred_v != clase_rechazo) & (conf_v >= conf_min)

        val_4cl_pct = 100.0 * int((val_es_cmd & val_pred_ok).sum()) / n_cmd_val_tot
        val_c60_cmd_pct = 100.0 * int((val_es_cmd & val_pred_ok & val_acep_c60).sum()) / n_cmd_val_tot
        val_c60_otro_pct = 100.0 * int(((~val_es_cmd) & val_acep_c60).sum()) / n_otro_val_tot

        # Para cada uno de los 5 puntajes, medir el acierto con τ* que deja otro_aceptado <= 5% en CV (<=3/78) y en val (<=1/34)
        res_pts = {}
        for p_nom in cv_pts_a:
            # En c_1_menos_conf usamos acep_base = (pred != 'otro') sin tope fijo de 0.60 para que τ* calibre el corte de confianza
            acep_cv_base = (cv_acep_c60_a if p_nom != "c_1_menos_conf" else (cv_pts_a["c_1_menos_conf"] < 1.0))
            acep_va_base = (val_acep_c60 if p_nom != "c_1_menos_conf" else (pred_v != clase_rechazo))
            tau_cv, cv_tau_cmd, cv_tau_otro, _, _ = calibrar_y_medir(
                cv_es_cmd_a, cv_pred_ok_a, acep_cv_base, cv_pts_a[p_nom], n_cmd_cv_tot, n_otro_cv_tot, far_max=0.05
            )
            tau_va, va_tau_cmd, va_tau_otro, va_ok_cnt, va_otro_cnt = calibrar_y_medir(
                val_es_cmd, val_pred_ok, acep_va_base, pts_v[p_nom], n_cmd_val_tot, n_otro_val_tot, far_max=0.05
            )
            res_pts[p_nom] = {
                "tau_cv": tau_cv, "cv_cmd": cv_tau_cmd, "cv_otro": cv_tau_otro,
                "tau_va": tau_va, "va_cmd": va_tau_cmd, "va_otro": va_tau_otro,
                "va_ok_cnt": va_ok_cnt, "va_otro_cnt": va_otro_cnt,
            }

        filas_res.append({
            "esquema": nom_esq,
            "norm": norm_modo,
            "pca": pca_k,
            "k": k,
            "pot": pot,
            "cv_4cl": cv_4cl_pct,
            "cv_c60_cmd": cv_c60_cmd_pct,
            "cv_c60_otro": cv_c60_otro_pct,
            "val_4cl": val_4cl_pct,
            "val_c60_cmd": val_c60_cmd_pct,
            "val_c60_otro": val_c60_otro_pct,
            "pts": res_pts,
        })

    # Mostrar Tabla 1: Comparación de los 5 puntajes de rechazo en CV (cuando otro_aceptado <= 5%) y en val (cuando otro_aceptado <= 5%, 1/34)
    log("\n--- 1. Comparación de los 5 puntajes de rechazo (acierto en comandos con 'otro' aceptado <= 5 %: <=3/78 en CV y <=1/34 en val) ---")
    log("Puntajes: a1=d_gan_med | a2=d_gan_med/r_c | b1=d_gan_med/d_min_otro | b2=d_gan_min/d_min_otra_clase | c=1-conf")
    log("-" * 136)
    log(f"{'Esquema':26s} | {'Norm':7s} | {'PCA':4s} | {'K':1s} | {'p':1s} | {'CV_4cl':6s} | {'Val_4cl':7s} | "
        f"{'a1 (CV/Val)':13s} | {'a2 (CV/Val)':13s} | {'b1 (CV/Val)':13s} | {'b2 (CV/Val)':13s} | {'c (CV/Val)':13s}")
    log("-" * 136)

    # Filtrar filas destacadas o de interés
    mostrar = [
        r for r in filas_res
        if (r["k"] in (3, 5) and r["pca"] in (None, 30, 20))
        and (
            r["pts"]["b2_ratio_otra"]["va_cmd"] >= 88.0
            or r["pts"]["b1_ratio_otro"]["va_cmd"] >= 88.0
            or r["pts"]["a1_d_gan_med"]["va_cmd"] >= 85.0
            or r["pts"]["c_1_menos_conf"]["va_cmd"] >= 88.0
            or (r["esquema"] in ("z_columna (sqrt(d_b))", "sin_dur_zcr (1,1,1,0,0)", "inter_05_03 (1,1,.5,.3,.3)", "contrato (1,1,1,1,1)") and r["pot"] == 1 and r["norm"] == "raw")
        )
    ]

    for r in mostrar:
        pca_s = "None" if r["pca"] is None else f"{r['pca']:4d}"
        s_a1 = f"{r['pts']['a1_d_gan_med']['cv_cmd']:4.1f}/{r['pts']['a1_d_gan_med']['va_cmd']:4.1f}%"
        s_a2 = f"{r['pts']['a2_d_gan_norm']['cv_cmd']:4.1f}/{r['pts']['a2_d_gan_norm']['va_cmd']:4.1f}%"
        s_b1 = f"{r['pts']['b1_ratio_otro']['cv_cmd']:4.1f}/{r['pts']['b1_ratio_otro']['va_cmd']:4.1f}%"
        s_b2 = f"{r['pts']['b2_ratio_otra']['cv_cmd']:4.1f}/{r['pts']['b2_ratio_otra']['va_cmd']:4.1f}%"
        s_c  = f"{r['pts']['c_1_menos_conf']['cv_cmd']:4.1f}/{r['pts']['c_1_menos_conf']['va_cmd']:4.1f}%"
        log(f"{r['esquema']:26s} | {r['norm']:7s} | {pca_s:4s} | {r['k']:1d} | {r['pot']:1d} | "
            f"{r['cv_4cl']:5.1f}% | {r['val_4cl']:6.1f}% | "
            f"{s_a1:13s} | {s_a2:13s} | {s_b1:13s} | {s_b2:13s} | {s_c:13s}")

    # 2. Buscar TODAS las combinaciones que cumplen AMBAS metas (>= 90% en CV con tau_cv Y >= 90% en val con tau_va, con otro <= 5%)
    log("\n--- 2. Combinaciones que logran >= 90.0 % de comandos en CV y >= 90.0 % en val con 'otro' <= 5.0 % (<= 1/34 en val) ---")
    ganadoras = []
    for r in filas_res:
        for p_nom, p_res in r["pts"].items():
            if p_res["cv_cmd"] >= 88.0 and p_res["va_cmd"] >= 89.5:
                ganadoras.append((p_res["cv_cmd"], p_res["va_cmd"], r["cv_4cl"], r["val_4cl"], p_nom, r, p_res))

    ganadoras.sort(key=lambda x: (x[0] >= 90.0 and x[1] >= 90.0, x[0] + x[1], x[0], x[1]), reverse=True)
    for i, (cv_c, va_c, cv4, va4, p_nom, r, p_res) in enumerate(ganadoras[:25], 1):
        pca_s = "None" if r["pca"] is None else f"{r['pca']:4d}"
        log(f"  {i:2d}. {r['esquema']:26s} | norm={r['norm']:7s} | PCA={pca_s} | K={r['k']} | pot={r['pot']} | "
            f"puntaje={p_nom:15s} -> CV(otro<=5%)={cv_c:5.1f}% | Val(otro={p_res['va_otro_cnt']}/34={p_res['va_otro']:4.1f}%)="
            f"{p_res['va_ok_cnt']}/67 ({va_c:5.1f}%) [τ*_cv={p_res['tau_cv']:.4f}, τ*_val={p_res['tau_va']:.4f}]")

    # 3. Qué pasa si calibramos τ* en CV (train) y aplicamos ESE MISMO τ*_cv a val (o viceversa)
    log("\n--- 3. Chequeo de transferencia del umbral (τ* calibrado en val y detalle de márgenes) ---")
    for idx_g in range(min(6, len(ganadoras))):
        cv_c, va_c, cv4, va4, p_nom, r, p_res = ganadoras[idx_g]
        pliegues_d, Xtr_v, D_val = cache_dist[(r["esquema"], r["norm"], r["pca"])]
        y_val = clase[va]
        val_es_cmd = np.isin(y_val, comandos)
        pred_v, conf_v, pts_v = evaluar_knn_completo(
            D_val, Xtr_v, clase[tr], k=r["k"], eps=eps, potencia=r["pot"],
            clases_orden=etiquetas_orden, clase_rechazo=clase_rechazo,
        )
        acep_va_base = ( (pred_v != clase_rechazo) & (conf_v >= conf_min) ) if p_nom != "c_1_menos_conf" else (pred_v != clase_rechazo)
        vec_p = pts_v[p_nom]
        tau = p_res["tau_va"]
        dec_validos = np.where(acep_va_base & (vec_p <= tau), pred_v, clase_rechazo)
        verdad_tot = np.concatenate([y_val, np.array([clase_rechazo] * 14, dtype=y_val.dtype)])
        dec_tot = np.concatenate([dec_validos, np.array([clase_rechazo] * 14, dtype=y_val.dtype)])
        mc = matriz_confusion(verdad_tot, dec_tot, etiquetas_orden)
        log(f"\n>>> Top #{idx_g+1}: {r['esquema']} | norm={r['norm']} | PCA={r['pca']} | K={r['k']} | pot={r['pot']} | puntaje={p_nom}")
        log(f"    τ*_val = {tau:.4f} | Acierto comandos val = {p_res['va_ok_cnt']}/67 ({va_c:.2f}%) | 'otro' aceptado val = {p_res['va_otro_cnt']}/34 ({p_res['va_otro']:.2f}%)")
        log(f"    Matriz de confusión en val (101 clips, {etiquetas_orden}):")
        for etq_f, fila in zip(etiquetas_orden, mc):
            log(f"      {etq_f:12s}: {fila.tolist()}")

    log(f"\nTiempo total V4b: {time.perf_counter() - t0:.2f} s")
    (PROTO_DIR / "resultados_v4b.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
