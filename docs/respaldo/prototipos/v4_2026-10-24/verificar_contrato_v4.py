"""Verificación exacta de las definiciones del contrato de V4 (24/10/2026).
Mide la grilla exacta de seleccionar_hiperparametros en CV por hablante (train),
calibra τ* en val con el modelo congelado y reporta las métricas exactas para la
sección 8 del contrato docs/contratos/v4_clasificacion.md.
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


ESQUEMAS_BLOQUES: dict[str, tuple[float, ...]] = {
    "contrato": (1.0, 1.0, 1.0, 1.0, 1.0),
    "sin_dur_zcr": (1.0, 1.0, 1.0, 0.0, 0.0),
    "z_columna": (float(np.sqrt(130)), float(np.sqrt(130)), float(np.sqrt(10)), float(np.sqrt(10)), 1.0),
}


def reponderar_bloques(X: np.ndarray, factores: tuple[float, ...] | np.ndarray, largos: dict[str, int]) -> np.ndarray:
    """Multiplica cada bloque de X (en el orden de BLOQUES) por su factor y descarta bloques con factor 0."""
    cols = []
    ini = 0
    for b, f in zip(BLOQUES, factores):
        fin = ini + largos[b]
        f_val = float(f)
        if f_val > 0.0:
            cols.append(X[:, ini:fin] * f_val)
        ini = fin
    return np.hstack(cols)


def normalizar_l2(X: np.ndarray, piso: float = 1e-12) -> np.ndarray:
    """Normaliza cada fila de X a norma L2 unitaria (con piso para evitar división por 0)."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        nrm = float(np.linalg.norm(X_arr))
        return X_arr / max(nrm, piso)
    nrm = np.linalg.norm(X_arr, axis=1, keepdims=True)
    return X_arr / np.maximum(nrm, piso)


class KNN:
    """Clasificador K-NN propio con voto ponderado 1 / (d + eps)."""

    def __init__(self, k: int = 5, eps: float = 1e-9) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k debe ser un int >= 1")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps debe ser un real finito > 0")
        self.k = int(k)
        self.eps = float(eps)
        self.X_train: np.ndarray | None = None
        self.etiquetas: np.ndarray | None = None
        self.clases: np.ndarray | None = None

    def ajustar(self, X: np.ndarray, etiquetas) -> "KNN":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(etiquetas)
        if X_arr.ndim != 2 or X_arr.shape[0] < self.k or X_arr.shape[1] < 1 or not np.all(np.isfinite(X_arr)):
            raise ValueError("X inválido para ajustar KNN")
        if y_arr.ndim != 1 or len(y_arr) != X_arr.shape[0]:
            raise ValueError("etiquetas inválidas para ajustar KNN")
        self.X_train = X_arr.copy()
        self.etiquetas = y_arr.copy()
        self.clases = np.unique(y_arr)
        return self

    def vecinos(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (distancias, indices) de los k vecinos más cercanos, ordenados de menor a mayor distancia."""
        if self.X_train is None:
            raise RuntimeError("KNN no ajustado")
        X_arr = np.asarray(X, dtype=np.float64)
        un_vector = X_arr.ndim == 1
        if un_vector:
            X_arr = X_arr[None, :]
        if X_arr.ndim != 2 or X_arr.shape[1] != self.X_train.shape[1] or not np.all(np.isfinite(X_arr)):
            raise ValueError("X inválido en vecinos")

        dif = X_arr[:, None, :] - self.X_train[None, :, :]
        D = np.linalg.norm(dif, axis=2)
        idx = np.argsort(D, axis=1, kind="mergesort")[:, : self.k]
        filas = np.arange(len(X_arr))[:, None]
        dist_k = D[filas, idx]
        if un_vector:
            return dist_k[0], idx[0].astype(np.int64)
        return dist_k, idx.astype(np.int64)

    def votar(self, X: np.ndarray) -> tuple[np.ndarray | str | int, np.ndarray | float, np.ndarray | float]:
        """Devuelve (prediccion, confianza, distancia_ganadora_media) para cada fila de X."""
        if self.X_train is None or self.etiquetas is None or self.clases is None:
            raise RuntimeError("KNN no ajustado")
        X_arr = np.asarray(X, dtype=np.float64)
        un_vector = X_arr.ndim == 1
        dist_k, idx_k = self.vecinos(X_arr if not un_vector else X_arr[None, :])
        vec_y = self.etiquetas[idx_k]
        pesos = 1.0 / (dist_k + self.eps)
        suma_pesos = pesos.sum(axis=1)

        m = len(dist_k)
        pred = np.empty(m, dtype=self.etiquetas.dtype)
        conf = np.zeros(m, dtype=np.float64)
        d_gan = np.zeros(m, dtype=np.float64)

        for i in range(m):
            mejor_c = None
            mejor_voto = -1.0
            mejor_dmin = np.inf
            for c in self.clases:
                mask_c = vec_y[i] == c
                if not np.any(mask_c):
                    continue
                voto_c = float(pesos[i, mask_c].sum())
                dmin_c = float(dist_k[i, mask_c].min())
                if (voto_c > mejor_voto + 1e-15) or (abs(voto_c - mejor_voto) <= 1e-15 and dmin_c < mejor_dmin):
                    mejor_c = c
                    mejor_voto = voto_c
                    mejor_dmin = dmin_c
            pred[i] = mejor_c
            conf[i] = mejor_voto / float(suma_pesos[i])
            d_gan[i] = float(dist_k[i, vec_y[i] == mejor_c].mean())

        if un_vector:
            p0 = pred[0].item() if hasattr(pred[0], "item") else pred[0]
            return p0, float(conf[0]), float(d_gan[0])
        return pred, conf, d_gan

    def predecir(self, X: np.ndarray):
        pred, _, _ = self.votar(X)
        return pred


def calibrar_umbral_voz(
    distancia: np.ndarray,
    confianza: np.ndarray,
    prediccion: np.ndarray,
    verdad: np.ndarray,
    comandos: list[str],
    clase_rechazo: str,
    confianza_min: float,
    otro_aceptado_max: float,
    n_otro_invalidos: int = 0,
) -> float:
    """Calibra τ* = (a + b) / 2 en val (o en CV) para que la fracción total de 'otro'
    aceptados como comando (sobre los válidos + n_otro_invalidos) sea <= otro_aceptado_max."""
    es_cmd = np.isin(verdad, comandos)
    es_otro = verdad == clase_rechazo
    n_otro_total = int(es_otro.sum()) + int(n_otro_invalidos)
    acep_sin_tau = (prediccion != clase_rechazo) & (confianza >= confianza_min)

    mask_otro_pel = es_otro & acep_sin_tau
    d_otros_pel = np.sort(distancia[mask_otro_pel])
    mask_cmd_bien = es_cmd & acep_sin_tau & (prediccion == verdad)
    d_cmd_bien = np.sort(distancia[mask_cmd_bien])

    m = int(np.floor(otro_aceptado_max * n_otro_total + 1e-9))
    if len(d_otros_pel) > m:
        b = float(d_otros_pel[m])
        cmd_bajo_b = d_cmd_bien[d_cmd_bien < b]
        if len(cmd_bajo_b) == 0:
            raise ValueError("Ningún comando bien clasificado queda por debajo de b")
        a = float(cmd_bajo_b[-1])
        return 0.5 * (a + b)
    else:
        a = float(d_cmd_bien[-1])
        d_otros_todos = np.sort(distancia[es_otro])
        mayores = d_otros_todos[d_otros_todos > a]
        b = float(mayores[0]) if len(mayores) > 0 else float(a * 1.2)
        return 0.5 * (a + b)


def main():
    buf = io.StringIO()

    def log(msg: str = ""):
        print(msg)
        buf.write(msg + "\n")

    cfg = cargar_config()
    carpeta = REPO / cfg["rutas"]["voz"] / "prepro"
    npz = np.load(carpeta / "caracteristicas.npz", allow_pickle=False)
    with open(carpeta / "metadatos.csv", encoding="utf-8", newline="") as f:
        filas_meta = list(csv.DictReader(f))

    part = npz["particion"]
    clase = npz["clase"]
    hab = npz["hablante"]
    origen = npz["origen"]
    copia = npz["copia"]
    X = npz["X"]

    largos = largos_de_bloque(cfg)
    comandos = list(cfg["voz"]["comandos"])
    clase_rechazo = cfg["voz"]["clase_rechazo"]
    etiquetas_orden = comandos + [clase_rechazo]
    eps = float(cfg["voz"]["knn"]["eps"])
    conf_min = float(cfg["voz"]["knn"]["confianza_min"])
    otro_max = float(cfg["metas"]["voz_otro_aceptado_max"])

    tr = part == "train"
    va = part == "val"
    te = part == "test"

    # Contar no válidos por partición
    inv_por_part = {}
    for p_nom in ("train", "val", "test"):
        filas_p = [r for r in filas_meta if r["particion"] == p_nom and r["valido"] in ("0", "False", "false", 0, False)]
        inv_por_part[p_nom] = {
            "total": len(filas_p),
            "cmd": sum(1 for r in filas_p if r["clase"] in comandos),
            "otro": sum(1 for r in filas_p if r["clase"] == clase_rechazo),
            "nb_k0_otro": sum(1 for r in filas_p if r["origen"] == "notebook" and r["copia"] == "0" and r["clase"] == clase_rechazo),
        }

    alias = {}
    for grupo in cfg["voz"]["dataset"]["mismo_hablante"]:
        for h in grupo:
            alias[h] = grupo[0]
    persona = np.array([alias.get(h, h) for h in hab])

    es_cmd = np.isin(clase, comandos)
    eval_nb = tr & (origen == "notebook") & (copia == 0)
    personas_nb = sorted(set(persona[eval_nb & es_cmd]))

    log("=" * 116)
    log("VERIFICACIÓN EXACTA DEL CONTRATO DE V4 · SELECCIÓN EN CV (TRAIN) Y CALIBRACIÓN EN VAL")
    log("=" * 116)
    log(f"Conteos: train válidos={tr.sum()} (no válidos nb k0 otro={inv_por_part['train']['nb_k0_otro']}), "
        f"val válidos={va.sum()} + {inv_por_part['val']['otro']} no válidos = {va.sum() + inv_por_part['val']['otro']}")

    # 1. Grilla de seleccionar_hiperparametros en CV por hablante (train) con l2_post
    log("\n--- 1. Grilla de seleccionar_hiperparametros en CV por hablante (train, 10 hablantes nb: 231 cmd y 78 'otro') ---")
    log(f"{'Esquema':15s} | {'PCA':4s} | {'K':2s} | {'CV_tau5_cmd':11s} | {'CV_c60_cmd':10s} | {'CV_4cl':7s} | {'CV_c60_otro':11s} | {'τ*_cv':8s}")
    log("-" * 88)

    grilla_cv = []
    for nom_esq, fac in ESQUEMAS_BLOQUES.items():
        X_esq = reponderar_bloques(X[tr], fac, largos)
        y_tr = clase[tr]
        per_tr = persona[tr]
        eval_nb_tr = (origen[tr] == "notebook") & (copia[tr] == 0)

        for pca_k in [20, 30, None]:
            # Proyectar y normalizar L2 en cada pliegue
            pliegues = []
            for per in personas_nb:
                m_a = per_tr != per
                m_q = eval_nb_tr & (per_tr == per)
                if pca_k is not None:
                    mp = PCA(pca_k).ajustar(X_esq[m_a])
                    Xa = normalizar_l2(mp.transformar(X_esq[m_a]))
                    Xq = normalizar_l2(mp.transformar(X_esq[m_q]))
                else:
                    Xa = normalizar_l2(X_esq[m_a])
                    Xq = normalizar_l2(X_esq[m_q])
                pliegues.append((Xa, y_tr[m_a], Xq, y_tr[m_q]))

            for k in cfg["voz"]["knn"]["k_candidatos"]:
                preds, confs, dists, verdades = [], [], [], []
                for Xa, ya, Xq, yq in pliegues:
                    knn = KNN(k=k, eps=eps).ajustar(Xa, ya)
                    pr, cf, dg = knn.votar(Xq)
                    preds.append(pr)
                    confs.append(cf)
                    dists.append(dg)
                    verdades.append(yq)

                pred_cv = np.concatenate(preds)
                conf_cv = np.concatenate(confs)
                dist_cv = np.concatenate(dists)
                verd_cv = np.concatenate(verdades)

                tau_cv = calibrar_umbral_voz(
                    dist_cv, conf_cv, pred_cv, verd_cv,
                    comandos=comandos, clase_rechazo=clase_rechazo,
                    confianza_min=conf_min, otro_aceptado_max=otro_max,
                    n_otro_invalidos=inv_por_part["train"]["nb_k0_otro"],
                )
                m_cmd_cv = np.isin(verd_cv, comandos)
                m_otro_cv = verd_cv == clase_rechazo
                acep_c60 = (pred_cv != clase_rechazo) & (conf_cv >= conf_min)
                acep_tau = acep_c60 & (dist_cv <= tau_cv)

                acc_4cl = float((pred_cv[m_cmd_cv] == verd_cv[m_cmd_cv]).mean())
                acc_c60 = float(((pred_cv[m_cmd_cv] == verd_cv[m_cmd_cv]) & acep_c60[m_cmd_cv]).mean())
                acc_tau = float(((pred_cv[m_cmd_cv] == verd_cv[m_cmd_cv]) & acep_tau[m_cmd_cv]).mean())
                far_c60 = float(acep_c60[m_otro_cv].sum() / (m_otro_cv.sum() + inv_por_part["train"]["nb_k0_otro"]))

                pca_s = "None" if pca_k is None else f"{pca_k:4d}"
                log(f"{nom_esq:15s} | {pca_s:4s} | {k:2d} | {100*acc_tau:9.2f} % | {100*acc_c60:8.2f} % | "
                    f"{100*acc_4cl:5.2f} % | {100*far_c60:9.2f} % | {tau_cv:8.4f}")

                grilla_cv.append({
                    "esquema": nom_esq,
                    "factores": fac,
                    "pca_k": pca_k,
                    "k": k,
                    "acc_tau": acc_tau,
                    "acc_c60": acc_c60,
                    "acc_4cl": acc_4cl,
                    "far_c60": far_c60,
                    "tau_cv": tau_cv,
                })

    # Orden de elección en CV: mayor acc_tau, desempata mayor acc_c60, mayor acc_4cl, menor far_c60
    grilla_cv.sort(key=lambda r: (round(r["acc_tau"], 6), round(r["acc_c60"], 6), round(r["acc_4cl"], 6), -round(r["far_c60"], 6)), reverse=True)
    ganador = grilla_cv[0]
    log("\n--- 2. Hiperparámetros elegidos en CV (sin mirar val ni test) ---")
    log(f"  Ganador CV #1: esquema={ganador['esquema']}, pca_k={ganador['pca_k']}, K={ganador['k']} "
        f"(CV_tau5={100*ganador['acc_tau']:.2f}%, CV_c60={100*ganador['acc_c60']:.2f}%, CV_4cl={100*ganador['acc_4cl']:.2f}%)")
    for idx_top in range(1, min(5, len(grilla_cv))):
        r = grilla_cv[idx_top]
        log(f"  Sigue CV #{idx_top+1}: esquema={r['esquema']}, pca_k={r['pca_k']}, K={r['k']} "
            f"(CV_tau5={100*r['acc_tau']:.2f}%, CV_c60={100*r['acc_c60']:.2f}%, CV_4cl={100*r['acc_4cl']:.2f}%)")

    # 3. Ajuste en train y calibración de τ* en val para los mejores de CV
    log("\n--- 3. Ajuste del modelo en train, calibración de τ* en val y evaluación en val y test ---")
    for r in grilla_cv[:4]:
        fac = r["factores"]
        pca_k = r["pca_k"]
        k = r["k"]

        X_tr_b = reponderar_bloques(X[tr], fac, largos)
        X_va_b = reponderar_bloques(X[va], fac, largos)
        X_te_b = reponderar_bloques(X[te], fac, largos)

        if pca_k is not None:
            mp = PCA(pca_k).ajustar(X_tr_b)
            Z_tr = normalizar_l2(mp.transformar(X_tr_b))
            Z_va = normalizar_l2(mp.transformar(X_va_b))
            Z_te = normalizar_l2(mp.transformar(X_te_b))
        else:
            Z_tr = normalizar_l2(X_tr_b)
            Z_va = normalizar_l2(X_va_b)
            Z_te = normalizar_l2(X_te_b)

        knn = KNN(k=k, eps=eps).ajustar(Z_tr, clase[tr])
        pred_va, conf_va, dist_va = knn.votar(Z_va)
        tau = calibrar_umbral_voz(
            dist_va, conf_va, pred_va, clase[va],
            comandos=comandos, clase_rechazo=clase_rechazo,
            confianza_min=conf_min, otro_aceptado_max=otro_max,
            n_otro_invalidos=inv_por_part["val"]["otro"],
        )

        log(f"\n>>> Modelo: esquema={r['esquema']}, PCA={pca_k}, K={k} -> τ* calibrado en val = {tau:.4f}")
        for nom_p, Z_p, y_p, n_inv_otro in [
            ("val", Z_va, clase[va], inv_por_part["val"]["otro"]),
            ("test", Z_te, clase[te], inv_por_part["test"]["otro"]),
        ]:
            pr, cf, dg = knn.votar(Z_p)
            acep_c60 = (pr != clase_rechazo) & (cf >= conf_min)
            acep_tau = acep_c60 & (dg <= tau)
            dec = np.where(acep_tau, pr, clase_rechazo)

            # Agregar los no válidos de V2 como verdad='otro', decision='otro'
            verdad_tot = np.concatenate([y_p, np.array([clase_rechazo] * n_inv_otro, dtype=y_p.dtype)])
            dec_tot = np.concatenate([dec, np.array([clase_rechazo] * n_inv_otro, dtype=y_p.dtype)])
            pred_tot = np.concatenate([pr, np.array([clase_rechazo] * n_inv_otro, dtype=y_p.dtype)])

            m_cmd = np.isin(verdad_tot, comandos)
            m_otro = verdad_tot == clase_rechazo

            id_4cl = float((pred_tot[m_cmd] == verdad_tot[m_cmd]).mean())
            acierto_c60 = float((( np.where(acep_c60, pr, clase_rechazo)[np.isin(y_p, comandos)] ) == y_p[np.isin(y_p, comandos)]).mean())
            otro_c60 = float((acep_c60[y_p == clase_rechazo]).sum() / m_otro.sum())

            acierto = float((dec_tot[m_cmd] == verdad_tot[m_cmd]).mean())
            otro_acep = float((dec_tot[m_otro] != clase_rechazo).mean())
            frr = float((dec_tot[m_cmd] != verdad_tot[m_cmd]).mean())
            mc = matriz_confusion(verdad_tot, dec_tot, etiquetas_orden)

            log(f"  Partición {nom_p:4s} ({len(verdad_tot)} clips = {m_cmd.sum()} cmd + {m_otro.sum()} otro):")
            log(f"    - Sin τ (solo conf>=0.60): acierto_cmd = {100*acierto_c60:.2f}% | otro_aceptado = {100*otro_c60:.2f}% (id_4cl = {100*id_4cl:.2f}%)")
            log(f"    - Con τ* = {tau:.4f}     : acierto_cmd = {100*acierto:.2f}% ({int((dec_tot[m_cmd]==verdad_tot[m_cmd]).sum())}/{m_cmd.sum()}) | "
                f"otro_aceptado = {100*otro_acep:.2f}% ({int((dec_tot[m_otro]!=clase_rechazo).sum())}/{m_otro.sum()}) | FRR = {100*frr:.2f}%")
            log(f"    - Matriz 4x4 {etiquetas_orden}: {mc.tolist()}")

    (PROTO_DIR / "resultados_verificacion.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()

