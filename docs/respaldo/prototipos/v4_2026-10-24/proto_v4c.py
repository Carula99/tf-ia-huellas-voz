"""Prototipo V4c (24/10/2026): inspección de los 'otro' que pasan en val con
z_columna + PCA=20 + l2_post/l2_pre (duración real vs vecinos) y prueba de pesos
intermedios para duración (3, 5, 7) y normalización L2 solo espectral + duración aparte.
NO toca ni carga test.
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


def transformar_espacio(
    X_all: np.ndarray,
    mask_fit: np.ndarray,
    factores: dict[str, float],
    largos: dict[str, int],
    norm_modo: str,
    pca_k: int | None,
) -> np.ndarray:
    """Devuelve X transformado según (factores, norm_modo, pca_k) ajustando PCA solo en mask_fit."""
    if norm_modo in ("raw", "l2_pre", "l2_post"):
        X_e = escalar_bloques(X_all, factores, largos)
        if norm_modo == "l2_pre":
            X_e = norm_l2(X_e)
        if pca_k is not None:
            mp = PCA(pca_k).ajustar(X_e[mask_fit])
            X_e = mp.transformar(X_e)
        if norm_modo == "l2_post":
            X_e = norm_l2(X_e)
        return X_e

    # Modos con bloques espectrales normalizados L2 y duración concatenada aparte:
    # 'l2_esp_pre' (L2 sobre espectral -> concat dur -> PCA si hay)
    # 'l2_esp_post' (PCA sobre espectral -> L2 sobre espectral -> concat dur * w_dur_esf)
    fac_esp = dict(factores)
    w_dur = float(fac_esp.pop("duracion", 0.0))
    fac_esp["duracion"] = 0.0
    X_esp = escalar_bloques(X_all, fac_esp, largos)
    col_dur = X_all[:, -1:]  # ya viene estandarizada con media 0 y sd 1 en train (peso 1/sqrt(1)=1)

    if norm_modo == "l2_esp_pre":
        # Como X_esp normalizado tiene norma 1, escalamos col_dur por w_dur / sqrt(280) para mantener proporción
        X_esp_n = norm_l2(X_esp)
        X_comb = np.hstack([X_esp_n, col_dur * (w_dur / np.sqrt(280.0))])
        if pca_k is not None:
            mp = PCA(pca_k).ajustar(X_comb[mask_fit])
            X_comb = mp.transformar(X_comb)
        return X_comb

    if norm_modo == "l2_esp_post":
        if pca_k is not None:
            mp = PCA(pca_k).ajustar(X_esp[mask_fit])
            X_esp_p = mp.transformar(X_esp)
        else:
            X_esp_p = X_esp
        X_esp_n = norm_l2(X_esp_p)
        # En la esfera unitaria, un peso relativo alpha sobre z_dur aporta alpha^2 * (z_dur1 - z_dur2)^2
        return np.hstack([X_esp_n, col_dur * (w_dur / np.sqrt(280.0))])

    raise ValueError(f"norm_modo desconocido: {norm_modo}")


def knn_detalle(D: np.ndarray, y_train: np.ndarray, k: int, eps: float, clases_orden: list[str], clase_rechazo: str):
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    filas = np.arange(len(D))[:, None]
    dist_sub = D[filas, idx]
    orden_k = np.argsort(dist_sub, axis=1, kind="mergesort")
    idx = idx[filas, orden_k]
    dist_k = dist_sub[filas, orden_k]
    vecinos_y = y_train[idx]
    pesos = 1.0 / (dist_k + eps)
    suma_pesos = pesos.sum(axis=1)

    m = len(D)
    n_cl = len(clases_orden)
    votos_mat = np.zeros((m, n_cl), dtype=np.float64)
    d_min_k_mat = np.full((m, n_cl), np.inf, dtype=np.float64)
    d_med_k_mat = np.full((m, n_cl), np.inf, dtype=np.float64)
    d_min_all_mat = np.full((m, n_cl), np.inf, dtype=np.float64)

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

    ganador_idx = np.argmax(votos_mat - 1e-15 * np.minimum(d_min_k_mat, 1e6), axis=1)
    clases_arr = np.asarray(clases_orden, dtype=y_train.dtype)
    pred = clases_arr[ganador_idx]
    filas_1d = np.arange(m)
    conf = votos_mat[filas_1d, ganador_idx] / suma_pesos
    d_gan_med = d_med_k_mat[filas_1d, ganador_idx]
    d_gan_min = d_min_k_mat[filas_1d, ganador_idx]

    idx_otro = clases_orden.index(clase_rechazo)
    d_min_otro = np.maximum(d_min_all_mat[:, idx_otro], 1e-9)
    ratio_otro = d_gan_med / d_min_otro

    d_otra_min = np.full(m, np.inf, dtype=np.float64)
    for c_idx in range(n_cl):
        es_otra = ganador_idx != c_idx
        d_otra_min[es_otra] = np.minimum(d_otra_min[es_otra], d_min_all_mat[es_otra, c_idx])
    ratio_otra = d_gan_min / np.maximum(d_otra_min, 1e-9)

    return pred, conf, d_gan_med, ratio_otro, ratio_otra, idx, dist_k, vecinos_y


def calibrar_tau(es_cmd, pred_ok, acep_base, puntaje, n_cmd_tot, n_otro_tot, m_perm):
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
    datos, _ = cargar_train_val(cfg)
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
    archivos = datos["archivo"]
    dur_s = datos["duracion"][:, 0]
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

    log("=" * 120)
    log("PROTOTIPO V4c · DETALLE DE 'OTRO' EN VAL Y PESOS INTERMEDIOS DE DURACIÓN EN Z_COLUMNA + L2")
    log("=" * 120)

    # 1. Inspección de los 'otro' que pasan en val con z_columna + PCA=20 + l2_post y l2_pre (K=3 y K=5)
    fac_zcol = {b: float(np.sqrt(largos[b])) for b in BLOQUES}
    y_tr = clase[tr]
    y_va = clase[va]
    arch_tr = archivos[tr]
    arch_va = archivos[va]
    dur_tr = dur_s[tr]
    dur_va = dur_s[va]
    val_es_cmd = np.isin(y_va, comandos)
    val_es_otro = y_va == clase_rechazo

    log("\n--- 1. Detalle de los 'otro' en val para z_columna (w_dur=1.0) + PCA=20 con l2_post y l2_pre ---")
    for norm_m in ("l2_post", "l2_pre"):
        X_t = transformar_espacio(X, tr, fac_zcol, largos, norm_m, pca_k=20)
        D_v = mat_distancias(X_t[va], X_t[tr])
        for k in (3, 5):
            pred, conf, dg_med, r_otro, r_otra, idx_k, dist_k, vec_y = knn_detalle(
                D_v, y_tr, k=k, eps=eps, clases_orden=etiquetas_orden, clase_rechazo=clase_rechazo
            )
            acep_c60 = (pred != clase_rechazo) & (conf >= conf_min)
            ok_c60 = int((val_es_cmd & (pred == y_va) & acep_c60).sum())
            otro_c60 = int((val_es_otro & acep_c60).sum())
            log(f"\n>>> z_columna (w_dur=1) | norm={norm_m} | PCA=20 | K={k} -> "
                f"Sin τ (solo conf>=0.60): cmd={ok_c60}/67 ({100*ok_c60/67:.1f}%), otro={otro_c60}/34 ({100*otro_c60/34:.1f}%)")
            for i_v in np.where(val_es_otro & (pred != clase_rechazo))[0]:
                estado = "PASA_C60" if acep_c60[i_v] else "rech_conf"
                vecs_info = [
                    f"{vec_y[i_v, j]}({arch_tr[idx_k[i_v, j]].split('/')[-1]}, d={dist_k[i_v, j]:.3f}, dur={dur_tr[idx_k[i_v, j]]:.2f}s)"
                    for j in range(k)
                ]
                log(f"  [{estado:9s}] {arch_va[i_v]:30s} | dur_real={dur_va[i_v]:.2f}s | pred={pred[i_v]:11s} "
                    f"conf={conf[i_v]:.3f} d_gan_med={dg_med[i_v]:.4f} r_otro={r_otro[i_v]:.3f}")
                log(f"              Vecinos: {', '.join(vecs_info)}")

            # Mostrar rango de d_gan_med de los comandos bien clasificados
            m_bien = val_es_cmd & (pred == y_va) & acep_c60
            d_bien = np.sort(dg_med[m_bien])
            log(f"  Comandos bien clasificados con conf>=0.60 ({m_bien.sum()}/67): "
                f"p90={np.percentile(d_bien, 90):.4f}, top 5 d_gan_med={[round(float(x), 4) for x in d_bien[-5:]]}")

    # 2. Barrido de peso de duración w_dur en {1, 3, 5, 7} y modos de normalización L2
    log("\n--- 2. Barrido de w_dur en z_columna (1, 3, 5, 7) y modos L2 (l2_post, l2_pre, l2_esp_pre, l2_esp_post) ---")
    log("Se reporta en CV (train: 231 cmd, 78 otro) y en val (67 cmd, 34 otro):")
    log("  - c60: sin umbral de distancia (solo pred!='otro' y conf>=0.60)")
    log("  - τ(d_gan_med <=5%): calibrando τ* con d_gan_med para dejar 'otro' <= 5% (<=3/78 en CV y <=1/34=2.9% en val)")
    log("  - τ(1-conf <=5%): calibrando τ* sobre 1-conf para dejar 'otro' <= 5%")
    log("-" * 138)
    log(f"{'w_dur (%var)':14s} | {'Norm':11s} | {'PCA':4s} | {'K':1s} | {'CV_4cl':6s} | {'Val_4cl':7s} | "
        f"{'CV_c60 (cmd/otro)':18s} | {'Val_c60 (cmd/otro)':20s} | {'τ_d_gan (CV / Val)':19s} | {'τ_conf (CV / Val)':18s}")
    log("-" * 138)

    mejores = []
    for w_dur in (1.0, 3.0, 5.0, 7.0):
        fac = {b: float(np.sqrt(largos[b])) for b in BLOQUES}
        fac["duracion"] = w_dur
        pct_var = 100.0 * (w_dur ** 2) / (280.0 + w_dur ** 2)
        w_str = f"w={w_dur:.0f} ({pct_var:4.1f}%)"

        for norm_m in ("l2_post", "l2_pre", "l2_esp_pre", "l2_esp_post"):
            for pca_k in (20, 30, None):
                if norm_m in ("l2_esp_post",) and pca_k is None:
                    continue
                # Precalcular pliegues CV y val
                pliegues = []
                for per in personas_nb:
                    a = tr & (persona != per)
                    q = eval_nb & (persona == per)
                    X_t = transformar_espacio(X, a, fac, largos, norm_m, pca_k)
                    D_pl = mat_distancias(X_t[q], X_t[a])
                    pliegues.append((a, q, D_pl))

                X_tv = transformar_espacio(X, tr, fac, largos, norm_m, pca_k)
                D_val = mat_distancias(X_tv[va], X_tv[tr])

                for k in (3, 5):
                    cv_es_cmd, cv_pred_ok, cv_acep_c60, cv_dgan, cv_1mconf = [], [], [], [], []
                    for a, q, D_pl in pliegues:
                        ya, yq = clase[a], clase[q]
                        pred_q, conf_q, dg_q, _, _, _, _, _ = knn_detalle(
                            D_pl, ya, k=k, eps=eps, clases_orden=etiquetas_orden, clase_rechazo=clase_rechazo
                        )
                        cv_es_cmd.append(np.isin(yq, comandos))
                        cv_pred_ok.append(pred_q == yq)
                        cv_acep_c60.append((pred_q != clase_rechazo) & (conf_q >= conf_min))
                        cv_dgan.append(dg_q)
                        cv_1mconf.append(np.where(pred_q != clase_rechazo, 1.0 - conf_q, 2.0))

                    cv_es_cmd_a = np.concatenate(cv_es_cmd)
                    cv_pred_ok_a = np.concatenate(cv_pred_ok)
                    cv_acep_c60_a = np.concatenate(cv_acep_c60)
                    cv_dgan_a = np.concatenate(cv_dgan)
                    cv_1mconf_a = np.concatenate(cv_1mconf)

                    cv_4cl = 100.0 * int((cv_es_cmd_a & cv_pred_ok_a).sum()) / 231
                    cv_c60_cmd = 100.0 * int((cv_es_cmd_a & cv_pred_ok_a & cv_acep_c60_a).sum()) / 231
                    cv_c60_otro = 100.0 * int(((~cv_es_cmd_a) & cv_acep_c60_a).sum()) / 78

                    pred_v, conf_v, dg_v, _, _, _, _, _ = knn_detalle(
                        D_val, y_tr, k=k, eps=eps, clases_orden=etiquetas_orden, clase_rechazo=clase_rechazo
                    )
                    val_pred_ok = pred_v == y_va
                    val_acep_c60 = (pred_v != clase_rechazo) & (conf_v >= conf_min)
                    val_1mconf = np.where(pred_v != clase_rechazo, 1.0 - conf_v, 2.0)

                    val_4cl = 100.0 * int((val_es_cmd & val_pred_ok).sum()) / 67
                    val_c60_ok = int((val_es_cmd & val_pred_ok & val_acep_c60).sum())
                    val_c60_cmd = 100.0 * val_c60_ok / 67
                    val_c60_ot_cnt = int((val_es_otro & val_acep_c60).sum())
                    val_c60_otro = 100.0 * val_c60_ot_cnt / 34

                    tau_d_cv, cv_tau_d, _, _, _ = calibrar_tau(cv_es_cmd_a, cv_pred_ok_a, cv_acep_c60_a, cv_dgan_a, 231, 78, m_perm=3)
                    tau_d_va, va_tau_d, va_ot_d, va_ok_d_cnt, va_ot_d_cnt = calibrar_tau(val_es_cmd, val_pred_ok, val_acep_c60, dg_v, 67, 34, m_perm=1)

                    tau_c_cv, cv_tau_c, _, _, _ = calibrar_tau(cv_es_cmd_a, cv_pred_ok_a, cv_1mconf_a < 1.5, cv_1mconf_a, 231, 78, m_perm=3)
                    tau_c_va, va_tau_c, va_ot_c, va_ok_c_cnt, va_ot_c_cnt = calibrar_tau(val_es_cmd, val_pred_ok, pred_v != clase_rechazo, val_1mconf, 67, 34, m_perm=1)

                    pca_s = "None" if pca_k is None else f"{pca_k:4d}"
                    log(f"{w_str:14s} | {norm_m:11s} | {pca_s:4s} | {k:1d} | {cv_4cl:5.1f}% | {val_4cl:6.1f}% | "
                        f"{cv_c60_cmd:5.1f}% / {cv_c60_otro:4.1f}%   | "
                        f"{val_c60_cmd:5.1f}% ({val_c60_ot_cnt}/34={val_c60_otro:4.1f}%) | "
                        f"{cv_tau_d:5.1f}% / {va_tau_d:5.1f}% ({va_ok_d_cnt}/67) | "
                        f"{cv_tau_c:5.1f}% / {va_tau_c:5.1f}% ({va_ok_c_cnt}/67)")

                    mejores.append({
                        "w_dur": w_dur, "norm": norm_m, "pca": pca_k, "k": k,
                        "cv_4cl": cv_4cl, "val_4cl": val_4cl,
                        "cv_c60_cmd": cv_c60_cmd, "cv_c60_otro": cv_c60_otro,
                        "val_c60_cmd": val_c60_cmd, "val_c60_otro": val_c60_otro, "val_c60_ot_cnt": val_c60_ot_cnt,
                        "cv_tau_d": cv_tau_d, "va_tau_d": va_tau_d, "tau_d_va": tau_d_va, "va_ok_d_cnt": va_ok_d_cnt, "va_ot_d_cnt": va_ot_d_cnt,
                        "cv_tau_c": cv_tau_c, "va_tau_c": va_tau_c, "tau_c_va": tau_c_va, "va_ok_c_cnt": va_ok_c_cnt, "va_ot_c_cnt": va_ot_c_cnt,
                    })

    log("\n--- 3. Top 10 configuraciones ordenadas por desempeño combinado en CV y Val (con 'otro' <= 1/34 = 2.9% en Val) ---")
    mejores.sort(key=lambda r: (max(r["va_tau_d"], r["va_tau_c"]) >= 89.5, r["cv_c60_cmd"] + max(r["va_tau_d"], r["va_tau_c"]), r["va_tau_d"]), reverse=True)
    for i, r in enumerate(mejores[:10], 1):
        pca_s = "None" if r["pca"] is None else f"{r['pca']:4d}"
        log(f"  {i:2d}. w_dur={r['w_dur']:.0f} | norm={r['norm']:11s} | PCA={pca_s} | K={r['k']} -> "
            f"CV_4cl={r['cv_4cl']:.1f}%, Val_4cl={r['val_4cl']:.1f}% | "
            f"c60: CV={r['cv_c60_cmd']:.1f}% (otro={r['cv_c60_otro']:.1f}%), Val={r['val_c60_cmd']:.1f}% (otro={r['val_c60_ot_cnt']}/34={r['val_c60_otro']:.1f}%) | "
            f"τ_d_gan: CV={r['cv_tau_d']:.1f}%, Val={r['va_tau_d']:.1f}% ({r['va_ok_d_cnt']}/67, otro={r['va_ot_d_cnt']}/34, τ*={r['tau_d_va']:.4f})")

    log(f"\nTiempo total V4c: {time.perf_counter() - t0:.2f} s")
    (PROTO_DIR / "resultados_v4c.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()

