"""Prototipo de V4 (24/10/2026): clasificador K-NN propio, elección de pesos por bloque,
PCA y K por validación cruzada dejando un hablante afuera en train, y calibración del
umbral de rechazo en val. NO usa ni carga la partición test.
"""
from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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
    """Carga caracteristicas.npz y metadatos.csv filtrando exclusivamente train y val."""
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
    """Multiplica cada bloque de X por su factor (y descarta columnas con factor 0)."""
    cols = []
    ini = 0
    for b in BLOQUES:
        fin = ini + largos[b]
        f = float(factores.get(b, 0.0))
        if f > 0.0:
            cols.append(X[:, ini:fin] * f)
        ini = fin
    return np.hstack(cols)


def mat_distancias(X_query: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Matriz de distancias euclídeas (m, n) en float64."""
    q2 = np.sum(X_query ** 2, axis=1, keepdims=True)
    t2 = np.sum(X_train ** 2, axis=1, keepdims=True).T
    d2 = np.maximum(q2 + t2 - 2.0 * (X_query @ X_train.T), 0.0)
    d = np.sqrt(d2)
    return d


def knn_desde_dist(
    D: np.ndarray,
    y_train: np.ndarray,
    k: int,
    eps: float = 1e-9,
    clases_orden: list[str] | None = None,
):
    """Dado D (m, n) de distancias euclídeas, evalúa K-NN con voto 1 / (d + eps)."""
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    filas = np.arange(len(D))[:, None]
    dist_sub = D[filas, idx]
    orden_k = np.argsort(dist_sub, axis=1, kind="mergesort")
    idx = idx[filas, orden_k]
    dist_k = dist_sub[filas, orden_k]
    vecinos_y = y_train[idx]
    pesos = 1.0 / (dist_k + eps)
    suma_pesos = pesos.sum(axis=1)

    if clases_orden is None:
        clases_orden = sorted(set(y_train.tolist()))

    m = len(D)
    votos_mat = np.zeros((m, len(clases_orden)), dtype=np.float64)
    d_min_mat = np.full((m, len(clases_orden)), np.inf, dtype=np.float64)
    d_med_mat = np.full((m, len(clases_orden)), np.inf, dtype=np.float64)

    for c_idx, c in enumerate(clases_orden):
        mask_c = vecinos_y == c
        votos_mat[:, c_idx] = np.where(mask_c, pesos, 0.0).sum(axis=1)
        cnt_c = mask_c.sum(axis=1)
        tiene = cnt_c > 0
        if np.any(tiene):
            d_c = np.where(mask_c, dist_k, np.inf)
            d_min_mat[:, c_idx] = d_c.min(axis=1)
            d_sum = np.where(mask_c, dist_k, 0.0).sum(axis=1)
            d_med_mat[tiene, c_idx] = d_sum[tiene] / cnt_c[tiene]

    # Ganador por mayor voto (en empate, menor distancia mínima)
    # Restamos una fracción minúscula de d_min para desempatar a favor del más cercano
    ganador_idx = np.argmax(votos_mat - 1e-15 * np.minimum(d_min_mat, 1e6), axis=1)
    clases_arr = np.asarray(clases_orden, dtype=y_train.dtype)
    pred = clases_arr[ganador_idx]
    filas_1d = np.arange(m)
    conf = votos_mat[filas_1d, ganador_idx] / suma_pesos
    d_ganador_media = d_med_mat[filas_1d, ganador_idx]
    d_ganador_min = d_min_mat[filas_1d, ganador_idx]
    d_media_k = dist_k.mean(axis=1)

    return pred, conf, d_ganador_media, d_ganador_min, d_media_k, dist_k, vecinos_y


def knn_propio(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
    eps: float = 1e-9,
    clases_orden: list[str] | None = None,
):
    dif = X_query[:, None, :] - X_train[None, :, :]
    D = np.linalg.norm(dif, axis=2)
    return knn_desde_dist(D, y_train, k=k, eps=eps, clases_orden=clases_orden)


def main():
    buf = io.StringIO()

    def log(msg: str = ""):
        print(msg)
        buf.write(msg + "\n")

    t_inicio = time.perf_counter()
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

    meta_val = [r for r in filas_meta if r["particion"] == "val"]
    meta_val_inv = [r for r in meta_val if r["valido"] in ("0", "False", "false", 0, False)]
    n_val_inv_cmd = sum(1 for r in meta_val_inv if r["clase"] in comandos)
    n_val_inv_otro = sum(1 for r in meta_val_inv if r["clase"] == clase_rechazo)

    log("=" * 96)
    log("PROTOTIPO DE V4 · K-NN PROPIO, PESOS POR BLOQUE EN CV Y UMBRAL EN VAL (SIN TOCAR TEST)")
    log("=" * 96)
    log(f"Filas válidas cargadas: train = {tr.sum()}, val = {va.sum()} (test excluido = 0 filas)")
    log(f"En val metadatos.csv: total = {len(meta_val)} ({va.sum()} válidos + {len(meta_val_inv)} no válidos: "
        f"{n_val_inv_cmd} comandos y {n_val_inv_otro} 'otro')")
    log(f"En val total real: {int(np.isin(clase[va], comandos).sum()) + n_val_inv_cmd} comandos y "
        f"{int((clase[va] == clase_rechazo).sum()) + n_val_inv_otro} 'otro'")

    # 1. Verificación contra scikit-learn
    log("\n--- 1. Chequeo del K-NN propio contra KNeighborsClassifier de scikit-learn ---")
    for k in cfg["voz"]["knn"]["k_candidatos"]:
        pred_p, _, _, _, _, dist_p, _ = knn_propio(X[tr], clase[tr], X[va], k=k, eps=eps)
        sk = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
        sk.fit(X[tr], clase[tr])
        dist_sk, _ = sk.kneighbors(X[va], n_neighbors=k)
        pred_sk = sk.predict(X[va])
        dif_d = float(np.max(np.abs(dist_p - dist_sk)))
        iguales = int((pred_p == pred_sk).sum())
        log(f"  K = {k}: máx |dist_propio - dist_sklearn| = {dif_d:.2e} | predicciones idénticas = {iguales}/{len(pred_p)}")

    # 2. Definición de hablantes y pliegues LOSO en train
    alias = {}
    for grupo in cfg["voz"]["dataset"]["mismo_hablante"]:
        for h in grupo:
            alias[h] = grupo[0]
    persona = np.array([alias.get(h, h) for h in hab])

    es_cmd = np.isin(clase, comandos)
    es_otro = clase == clase_rechazo
    eval_nb_cmd = tr & (origen == "notebook") & (copia == 0) & es_cmd
    eval_nb_otro = tr & (origen == "notebook") & (copia == 0) & es_otro
    eval_real_cmd = tr & (origen != "tts") & (copia == 0) & es_cmd
    personas_nb = sorted(set(persona[eval_nb_cmd]))
    personas_real = sorted(set(persona[eval_real_cmd]))

    meta_tr_nb_inv_otro = sum(
        1 for r in filas_meta
        if r["particion"] == "train" and r["origen"] == "notebook"
        and r["copia"] == "0" and r["clase"] == clase_rechazo
        and r["valido"] in ("0", "False", "false", 0, False)
    )
    log(f"\nPliegues LOSO en train: {len(personas_nb)} personas de notebook "
        f"({eval_nb_cmd.sum()} comandos k0, {eval_nb_otro.sum()} 'otro' válidos k0 + {meta_tr_nb_inv_otro} no válidos = "
        f"{eval_nb_otro.sum() + meta_tr_nb_inv_otro} 'otro' totales)")
    log(f"Total personas reales en train (notebook + whatsapp, sin TTS): {len(personas_real)} "
        f"({eval_real_cmd.sum()} comandos k0)")

    # 3. Esquemas de pesos por bloque sobre X
    esquemas = {
        "contrato (1,1,1,1,1)": {
            "mfcc": 1.0, "delta": 1.0, "energia": 1.0, "zcr": 1.0, "duracion": 1.0,
        },
        "z_columna (sqrt(d_b))": {
            b: float(np.sqrt(largos[b])) for b in BLOQUES
        },
        "sin_dur_zcr (1,1,1,0,0)": {
            "mfcc": 1.0, "delta": 1.0, "energia": 1.0, "zcr": 0.0, "duracion": 0.0,
        },
        "mfcc_delta (1,1,0,0,0)": {
            "mfcc": 1.0, "delta": 1.0, "energia": 0.0, "zcr": 0.0, "duracion": 0.0,
        },
        "z_col_sin_dur_zcr": {
            "mfcc": float(np.sqrt(130)), "delta": float(np.sqrt(130)),
            "energia": float(np.sqrt(10)), "zcr": 0.0, "duracion": 0.0,
        },
    }

    pca_opciones = [None, 30, 20]
    k_opciones = list(cfg["voz"]["knn"]["k_candidatos"])  # [3, 5, 7, 9]

    log("\n--- 2. Grilla completa en CV por hablante (train) ---")
    log("Evaluación sobre los 10 hablantes de notebook en train (231 comandos k0 y 78 'otro' k0)")
    log("Columnas:")
    log("  - CV_id3: identificación entre los 3 comandos (entrenando solo con los comandos de train)")
    log("  - CV_4cl: acierto en comandos entrenando con las 4 clases (sin umbral de distancia ni confianza)")
    log("  - CV_c60_cmd / CV_c60_otro: acierto en comandos y % de 'otro' aceptado con pred!='otro' y conf>=0.60")
    log("  - CV_tau5_cmd: acierto en comandos en CV aplicando también un umbral τ que deja 'otro' aceptado <= 5% (<= 3/78)")
    log("  - CV_18p_id3: identificación en los 332 comandos k0 de las 18 personas reales (nb + whatsapp)")
    log("-" * 118)
    log(f"{'Esquema':25s} | {'PCA':4s} | {'K':2s} | {'CV_id3':7s} | {'CV_4cl':7s} | "
        f"{'CV_c60_cmd':10s} | {'CV_c60_otro':11s} | {'CV_tau5_cmd':11s} | {'CV_18p_id3':10s}")
    log("-" * 118)

    resultados_cv = []
    for nom_esq, fac in esquemas.items():
        X_esq = escalar_bloques(X, fac, largos)
        for pca_k in pca_opciones:
            # Precalcular matrices de distancia para los 10 pliegues de notebook y 18 reales
            pliegues_nb = {}
            for per in personas_nb:
                a = tr & (persona != per)
                q = tr & (origen == "notebook") & (copia == 0) & (persona == per)
                if pca_k is not None:
                    modelo_pca = PCA(pca_k).ajustar(X_esq[a])
                    Xa = modelo_pca.transformar(X_esq[a])
                    Xq = modelo_pca.transformar(X_esq[q])
                else:
                    Xa, Xq = X_esq[a], X_esq[q]
                D_all = mat_distancias(Xq, Xa)
                D_cmd = D_all[:, es_cmd[a]]
                pliegues_nb[per] = (a, q, D_all, D_cmd)

            pliegues_18 = {}
            for per in personas_real:
                a = tr & (persona != per)
                q = eval_real_cmd & (persona == per)
                if pca_k is not None:
                    modelo_pca = PCA(pca_k).ajustar(X_esq[a])
                    Xa = modelo_pca.transformar(X_esq[a])
                    Xq = modelo_pca.transformar(X_esq[q])
                else:
                    Xa, Xq = X_esq[a], X_esq[q]
                D_cmd = mat_distancias(Xq, Xa[es_cmd[a]])
                pliegues_18[per] = (a, q, D_cmd)

            for k in k_opciones:
                ok_id3 = 0
                ok_4cl = 0
                ok_c60 = 0
                otro_acep_c60 = 0
                n_cmd_nb = 0
                n_otro_nb = meta_tr_nb_inv_otro

                cv_es_cmd = []
                cv_pred_ok = []
                cv_acep_c60 = []
                cv_d_gan = []

                for per in personas_nb:
                    a, q, D_all, D_cmd = pliegues_nb[per]
                    ya = clase[a]
                    yq = clase[q]
                    pred_3, _, _, _, _, _, _ = knn_desde_dist(D_cmd, ya[es_cmd[a]], k=k, eps=eps, clases_orden=comandos)
                    pred_4, conf_4, dg_med, _, _, _, _ = knn_desde_dist(D_all, ya, k=k, eps=eps, clases_orden=etiquetas_orden)

                    m_cmd = np.isin(yq, comandos)
                    m_otro = yq == clase_rechazo
                    n_cmd_nb += int(m_cmd.sum())
                    n_otro_nb += int(m_otro.sum())

                    ok_id3 += int((pred_3[m_cmd] == yq[m_cmd]).sum())
                    ok_4cl += int((pred_4[m_cmd] == yq[m_cmd]).sum())

                    acepta_c60 = (pred_4 != clase_rechazo) & (conf_4 >= conf_min)
                    dec_c60 = np.where(acepta_c60, pred_4, clase_rechazo)
                    ok_c60 += int((dec_c60[m_cmd] == yq[m_cmd]).sum())
                    otro_acep_c60 += int((dec_c60[m_otro] != clase_rechazo).sum())

                    cv_es_cmd.append(m_cmd)
                    cv_pred_ok.append(pred_4 == yq)
                    cv_acep_c60.append(acepta_c60)
                    cv_d_gan.append(dg_med)

                ok_18 = 0
                n_18 = 0
                for per in personas_real:
                    a, q, D_cmd = pliegues_18[per]
                    pred_3, _, _, _, _, _, _ = knn_desde_dist(D_cmd, clase[a][es_cmd[a]], k=k, eps=eps, clases_orden=comandos)
                    ok_18 += int((pred_3 == clase[q]).sum())
                    n_18 += int(q.sum())

                cv_es_cmd_arr = np.concatenate(cv_es_cmd)
                cv_pred_ok_arr = np.concatenate(cv_pred_ok)
                cv_acep_c60_arr = np.concatenate(cv_acep_c60)
                cv_d_gan_arr = np.concatenate(cv_d_gan)

                # Umbral en CV para dejar 'otro' aceptado <= 5% (m = floor(0.05 * 78) = 3 de 78)
                d_otro_pel = np.sort(cv_d_gan_arr[(~cv_es_cmd_arr) & cv_acep_c60_arr])
                m_cv = int(np.floor(0.05 * n_otro_nb + 1e-9))  # 3
                if len(d_otro_pel) > m_cv:
                    b_cv = float(d_otro_pel[m_cv])
                else:
                    b_cv = np.inf
                ok_tau5 = int((cv_es_cmd_arr & cv_pred_ok_arr & cv_acep_c60_arr & (cv_d_gan_arr < b_cv)).sum())

                acc_id3 = 100.0 * ok_id3 / n_cmd_nb
                acc_4cl = 100.0 * ok_4cl / n_cmd_nb
                acc_c60 = 100.0 * ok_c60 / n_cmd_nb
                far_c60 = 100.0 * otro_acep_c60 / n_otro_nb
                acc_tau5 = 100.0 * ok_tau5 / n_cmd_nb
                acc_18 = 100.0 * ok_18 / n_18
                pca_str = "None" if pca_k is None else f"{pca_k:4d}"
                log(f"{nom_esq:25s} | {pca_str:4s} | {k:2d} | {acc_id3:5.1f} % | {acc_4cl:5.1f} % | "
                    f"{acc_c60:8.1f} % | {far_c60:9.1f} % | {acc_tau5:9.1f} % | {acc_18:8.1f} %")

                resultados_cv.append({
                    "esquema": nom_esq,
                    "factores": fac,
                    "pca_k": pca_k,
                    "k": k,
                    "acc_id3": acc_id3,
                    "acc_4cl": acc_4cl,
                    "acc_c60": acc_c60,
                    "far_c60": far_c60,
                    "acc_tau5": acc_tau5,
                    "acc_18": acc_18,
                })

    # 4. Ranking de las mejores configuraciones según CV (sin mirar val ni test)
    log("\n--- 3. Top 12 configuraciones en CV por hablante (ordenadas por CV_4cl, CV_c60_cmd y CV_id3) ---")
    orden_cv = sorted(
        resultados_cv,
        key=lambda r: (r["acc_4cl"], r["acc_c60"], r["acc_id3"], r["acc_tau5"], -r["far_c60"], r["acc_18"]),
        reverse=True,
    )
    for i, r in enumerate(orden_cv[:12], 1):
        pca_s = "None" if r["pca_k"] is None else str(r["pca_k"])
        log(f"  {i:2d}. {r['esquema']:25s} PCA={pca_s:4s} K={r['k']} -> "
            f"CV_4cl={r['acc_4cl']:.2f}% | CV_id3={r['acc_id3']:.2f}% | "
            f"CV_c60={r['acc_c60']:.2f}% (otro={r['far_c60']:.1f}%) | "
            f"CV_tau5={r['acc_tau5']:.2f}% | CV_18p={r['acc_18']:.2f}%")

    # 5. Ablación en train CV
    log("\n--- 4. Ablación de la base de entrenamiento en CV (sobre sin_dur_zcr, PCA=None y PCA=30, K=5) ---")
    for pca_k in (None, 30):
        X_esq = escalar_bloques(X, esquemas["sin_dur_zcr (1,1,1,0,0)"], largos)
        for modo_base, filtro_extra in [
            ("Base completa (k0 + k1/k2 + TTS + ruido)", np.ones(len(X), dtype=bool)),
            ("Sin voces sintéticas TTS", origen != "tts"),
            ("Solo originales k0 (sin aumento k1/k2)", copia == 0),
            ("Solo notebook k0 (sin WhatsApp, TTS ni aumento)", (origen == "notebook") & (copia == 0)),
        ]:
            ok_4 = ok_c60 = otro_ac = 0
            for per in personas_nb:
                a = tr & (persona != per) & filtro_extra
                q = tr & (origen == "notebook") & (copia == 0) & (persona == per)
                if pca_k is not None:
                    mp = PCA(pca_k).ajustar(X_esq[a])
                    Xa, Xq = mp.transformar(X_esq[a]), mp.transformar(X_esq[q])
                else:
                    Xa, Xq = X_esq[a], X_esq[q]
                pred_4, conf_4, _, _, _, _, _ = knn_propio(Xa, clase[a], Xq, k=5, eps=eps, clases_orden=etiquetas_orden)
                yq = clase[q]
                m_cmd = np.isin(yq, comandos)
                m_otro = yq == clase_rechazo
                ok_4 += int((pred_4[m_cmd] == yq[m_cmd]).sum())
                acep = (pred_4 != clase_rechazo) & (conf_4 >= conf_min)
                dec = np.where(acep, pred_4, clase_rechazo)
                ok_c60 += int((dec[m_cmd] == yq[m_cmd]).sum())
                otro_ac += int((dec[m_otro] != clase_rechazo).sum())
            log(f"  PCA={str(pca_k):4s} | {modo_base:48s}: CV_4cl={100*ok_4/231:5.1f}% | "
                f"CV_c60_cmd={100*ok_c60/231:5.1f}% | otro_acep={100*otro_ac/78:4.1f}%")

    # 6. Calibración del umbral de rechazo en val para los candidatos principales
    log("\n--- 5. Evaluación en val y calibración del umbral de distancia τ* (solo train -> val) ---")
    log("En val hay 67 comandos (todos válidos en V2) y 34 'otro' (20 válidos en V2 + 14 no válidos en V2).")
    log("Meta del plan: acierto en comandos >= 90.0% (>= 61/67) y 'otro' aceptado <= 5.0% (<= 1/34 = 2.94%).")
    log("-" * 116)

    candidatos_revisar = [
        ("sin_dur_zcr (1,1,1,0,0)", None, 3),
        ("sin_dur_zcr (1,1,1,0,0)", None, 5),
        ("sin_dur_zcr (1,1,1,0,0)", None, 7),
        ("sin_dur_zcr (1,1,1,0,0)", 30, 3),
        ("sin_dur_zcr (1,1,1,0,0)", 30, 5),
        ("sin_dur_zcr (1,1,1,0,0)", 20, 5),
        ("z_columna (sqrt(d_b))", None, 3),
        ("z_columna (sqrt(d_b))", None, 5),
        ("z_columna (sqrt(d_b))", 30, 5),
        ("z_col_sin_dur_zcr", None, 5),
        ("z_col_sin_dur_zcr", 30, 5),
        ("mfcc_delta (1,1,0,0,0)", None, 5),
        ("mfcc_delta (1,1,0,0,0)", 30, 5),
        ("contrato (1,1,1,1,1)", None, 3),
        ("contrato (1,1,1,1,1)", None, 5),
        ("contrato (1,1,1,1,1)", 30, 3),
    ]

    y_val = clase[va]
    val_es_cmd = np.isin(y_val, comandos)
    val_es_otro = y_val == clase_rechazo
    n_val_cmd_tot = int(val_es_cmd.sum()) + n_val_inv_cmd  # 67
    n_val_otro_tot = int(val_es_otro.sum()) + n_val_inv_otro  # 34

    for nom_esq, pca_k, k in candidatos_revisar:
        fac = esquemas[nom_esq]
        X_esq = escalar_bloques(X, fac, largos)
        if pca_k is not None:
            mp = PCA(pca_k).ajustar(X_esq[tr])
            X_tr_p = mp.transformar(X_esq[tr])
            X_va_p = mp.transformar(X_esq[va])
        else:
            X_tr_p, X_va_p = X_esq[tr], X_esq[va]

        pred_va, conf_va, dg_med, dg_min, d_medk, _, _ = knn_propio(
            X_tr_p, clase[tr], X_va_p, k=k, eps=eps, clases_orden=etiquetas_orden
        )

        # Sin rechazo (acierto bruto 4 clases en los 67 comandos de val)
        ok_bruto = int((pred_va[val_es_cmd] == y_val[val_es_cmd]).sum())
        acc_bruto = 100.0 * ok_bruto / n_val_cmd_tot

        # Sin umbral de distancia (solo pred != 'otro' y conf >= 0.60)
        acep_sin_tau = (pred_va != clase_rechazo) & (conf_va >= conf_min)
        dec_sin_tau = np.where(acep_sin_tau, pred_va, clase_rechazo)
        ok_sin_tau = int((dec_sin_tau[val_es_cmd] == y_val[val_es_cmd]).sum())
        acierto_sin_tau = 100.0 * ok_sin_tau / n_val_cmd_tot
        otro_sin_tau = int((dec_sin_tau[val_es_otro] != clase_rechazo).sum())

        mask_otro_peligroso = val_es_otro & acep_sin_tau
        d_otros_peligrosos = np.sort(dg_med[mask_otro_peligroso])
        mask_cmd_bien = val_es_cmd & acep_sin_tau & (pred_va == y_val)
        d_cmd_bien = np.sort(dg_med[mask_cmd_bien])

        res_tau = []
        for m_perm, etq_m in [(0, "0/34"), (1, "1/34")]:
            if len(d_otros_peligrosos) > m_perm:
                b = float(d_otros_peligrosos[m_perm])
                cmd_bajo_b = d_cmd_bien[d_cmd_bien < b]
                a = float(cmd_bajo_b[-1]) if len(cmd_bajo_b) > 0 else b * 0.9
                tau = 0.5 * (a + b)
            else:
                # Si ya cumple <= m_perm falsos aceptados sin cortar por distancia,
                # vemos dónde cae el primer 'otro' o tomamos el máximo de los comandos bien clasificados
                a = float(d_cmd_bien[-1])
                d_otro_todos = np.sort(dg_med[val_es_otro])
                mayores = d_otro_todos[d_otro_todos > a]
                b = float(mayores[0]) if len(mayores) > 0 else float(a * 1.2)
                tau = 0.5 * (a + b)

            acep_tau = acep_sin_tau & (dg_med <= tau)
            dec_tau = np.where(acep_tau, pred_va, clase_rechazo)
            ok_cmd = int((dec_tau[val_es_cmd] == y_val[val_es_cmd]).sum())
            otro_ac = int((dec_tau[val_es_otro] != clase_rechazo).sum())
            res_tau.append((etq_m, tau, a, b, ok_cmd, 100.0 * ok_cmd / n_val_cmd_tot, otro_ac, 100.0 * otro_ac / n_val_otro_tot))

        pca_s = "None" if pca_k is None else f"{pca_k:4d}"
        r0, r1 = res_tau[0], res_tau[1]
        log(f"{nom_esq:24s} PCA={pca_s} K={k} | val_4cl={acc_bruto:5.1f}% | "
            f"c60: cmd={acierto_sin_tau:5.1f}% otro={otro_sin_tau}/34 ({100*otro_sin_tau/34:4.1f}%) | "
            f"τ(0/34)={r0[1]:.3f} -> cmd={r0[4]}/67 ({r0[5]:5.1f}%) | "
            f"τ(1/34)={r1[1]:.3f} -> cmd={r1[4]}/67 ({r1[5]:5.1f}%)")

    # 7. Detalle fino de las distancias en val para las 4 mejores opciones de CV
    log("\n--- 6. Detalle de distancias y matriz de confusión en val (67 comandos + 34 'otro' = 101 clips) ---")
    for nom_esq, pca_k, k in [
        ("sin_dur_zcr (1,1,1,0,0)", None, 5),
        ("sin_dur_zcr (1,1,1,0,0)", 30, 5),
        ("z_columna (sqrt(d_b))", None, 5),
        ("z_col_sin_dur_zcr", None, 5),
    ]:
        fac = esquemas[nom_esq]
        X_esq = escalar_bloques(X, fac, largos)
        if pca_k is not None:
            mp = PCA(pca_k).ajustar(X_esq[tr])
            X_tr_p = mp.transformar(X_esq[tr])
            X_va_p = mp.transformar(X_esq[va])
        else:
            X_tr_p, X_va_p = X_esq[tr], X_esq[va]

        pred_va, conf_va, dg_med, dg_min, d_medk, _, _ = knn_propio(
            X_tr_p, clase[tr], X_va_p, k=k, eps=eps, clases_orden=etiquetas_orden
        )
        acep_sin_tau = (pred_va != clase_rechazo) & (conf_va >= conf_min)

        log(f"\n>>> Configuración: {nom_esq}, PCA={pca_k}, K={k}")
        for idx_v in np.where(val_es_otro)[0]:
            pasa = "PASA_C60" if acep_sin_tau[idx_v] else "rechazado"
            if acep_sin_tau[idx_v] or pred_va[idx_v] != clase_rechazo:
                log(f"  'otro' en val ({pasa:9s}): archivo={datos['archivo'][va][idx_v]:30s} "
                    f"pred={pred_va[idx_v]:11s} conf={conf_va[idx_v]:.3f} d_gan_med={dg_med[idx_v]:.4f} "
                    f"d_gan_min={dg_min[idx_v]:.4f} d_media_k={d_medk[idx_v]:.4f}")

        mask_cmd_bien = val_es_cmd & acep_sin_tau & (pred_va == y_val)
        d_cmd_ord = np.sort(dg_med[mask_cmd_bien])
        log(f"  Comandos bien clasificados con conf>=0.60 en val ({mask_cmd_bien.sum()}/67): "
            f"top 5 mayores d_gan_med = {[round(float(x), 4) for x in d_cmd_ord[-5:]]}")

        for idx_c in np.where(val_es_cmd & (~mask_cmd_bien))[0]:
            log(f"  Comando en val que falla con conf>=0.60: archivo={datos['archivo'][va][idx_c]:30s} "
                f"verdad={y_val[idx_c]:11s} pred={pred_va[idx_c]:11s} conf={conf_va[idx_v if False else idx_c]:.3f} "
                f"d_gan_med={dg_med[idx_c]:.4f}")

        mask_otro_pel = val_es_otro & acep_sin_tau
        d_otros_pel = np.sort(dg_med[mask_otro_pel])
        if len(d_otros_pel) > 1:
            b = float(d_otros_pel[1])
        elif len(d_otros_pel) == 1:
            b = float(d_otros_pel[0])
        else:
            b = float(d_cmd_ord[-1] * 1.15)
        cmd_bajo_b = d_cmd_ord[d_cmd_ord < b]
        a = float(cmd_bajo_b[-1])
        tau = 0.5 * (a + b)
        dec_val_validos = np.where(acep_sin_tau & (dg_med <= tau), pred_va, clase_rechazo)
        verdad_val_completa = np.concatenate([y_val, np.array([clase_rechazo] * n_val_inv_otro, dtype=y_val.dtype)])
        dec_val_completa = np.concatenate([dec_val_validos, np.array([clase_rechazo] * n_val_inv_otro, dtype=y_val.dtype)])
        mc = matriz_confusion(verdad_val_completa, dec_val_completa, etiquetas_orden)
        log(f"  Calibración con d_gan_med: a={a:.4f}, b={b:.4f}, τ*={tau:.4f}")
        log(f"  Matriz de confusión en val (101 clips, orden {etiquetas_orden}):")
        for etq_fila, fila in zip(etiquetas_orden, mc):
            log(f"    {etq_fila:12s}: {fila.tolist()}")

    # 8. Tiempo de inferencia por clip
    t0 = time.perf_counter()
    n_rep = 20
    for _ in range(n_rep):
        knn_propio(X[tr], clase[tr], X[va], k=5, eps=eps, clases_orden=etiquetas_orden)
    dt = (time.perf_counter() - t0) / n_rep
    log(f"\n--- 7. Tiempo del K-NN propio ---")
    log(f"  Lote de {va.sum()} consultas contra {tr.sum()} vecinos (d=281): {1000*dt:.2f} ms ({1000*dt/va.sum():.3f} ms/clip)")
    log(f"  Tiempo total del prototipo: {time.perf_counter() - t_inicio:.2f} s")

    PROTO_DIR.mkdir(parents=True, exist_ok=True)
    (PROTO_DIR / "resultados.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
