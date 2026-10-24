"""Prototipo de V3 (extracción de características de voz y PCA propio) · 24/10/2026.

Mide sobre data/voz/prepro/ (los 1908 clips válidos de V2):
1. Dimensiones, cantidad de tramas por palabra y por tramo, ausencia de NaN/Inf y desvíos mínimos.
2. Escalador por bloque (5 bloques, d = 281) y contribución de cada bloque a la distancia cuadrática.
3. Silhouette (propia vs. scikit-learn) global, con PCA (20 y 30) y por bloque, en 3 comandos y 4 clases.
4. PCA propio vs. scikit-learn y varianza explicada acumulada (k = 2, 3, 10, 20, 30, 50).
5. Validación cruzada por hablante en train y evaluación en val con el K-NN del borrador (281d, PCA 20, PCA 30).
6. Tiempos de cómputo por clip y para los 1908 clips.
"""

from __future__ import annotations

import collections
import csv
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA as SkPCA
from sklearn.metrics import silhouette_score as sk_silhouette

RAIZ = Path(r"C:\Users\Santiago\source\repos\tf-ia-huellas-voz")
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import cargar_config
from src.core_ml.metricas import silhouette
from src.core_ml.pca import PCA

BLOQUES = ("mfcc", "delta", "energia", "zcr", "duracion")


def tramos_estadisticas(m: np.ndarray, k: int = 5, ddof: int = 0) -> np.ndarray:
    """Divide la matriz m (filas=variables, cols=tramas) en k tramos y concatena [media, desvio] por tramo."""
    n = m.shape[1]
    out = []
    for i in range(k):
        a = int(np.floor(i * n / k))
        b = max(a + 1, int(np.floor((i + 1) * n / k)))
        s = m[:, a:b]
        out.append(np.concatenate([s.mean(axis=1), s.std(axis=1, ddof=ddof)]))
    return np.concatenate(out).astype(np.float64)


def extraer_caracteristicas(
    y: np.ndarray,
    cfg: dict,
    duracion_s: float | None = None,
    ddof_tramos: int = 0,
    n_fft: int = 512,
    delta_width: int = 5,
    rms_piso: float = 1e-6,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Extrae los 5 bloques crudos de un clip preprocesado de V2."""
    fs = int(cfg["voz"]["frecuencia_muestreo"])
    margen_ms = float(cfg["voz"]["preprocesamiento"]["margen_ms"])
    mfcc_cfg = cfg["voz"]["mfcc"]
    n_mfcc = int(mfcc_cfg["n_mfcc"])
    n_mels = int(mfcc_cfg["n_mels"])
    fmax = float(mfcc_cfg["fmax"])
    fmin = float(cfg["voz"]["pasabanda_hz"][0])
    win_length = int(round(float(mfcc_cfg["ventana_ms"]) * fs / 1000.0))
    hop_length = int(round(float(mfcc_cfg["avance_ms"]) * fs / 1000.0))
    k_tramos = int(cfg["voz"]["tramos"])

    M = int(round(margen_ms * fs / 1000.0))
    m_tramas = M // hop_length

    if duracion_s is None:
        duracion_s = max(0.0, (len(y) - 2 * M) / float(fs))

    y32 = np.asarray(y, dtype=np.float32).reshape(-1)

    S = librosa.feature.mfcc(
        y=y32,
        sr=fs,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        center=False,
    )
    w = S[:, m_tramas : S.shape[1] - m_tramas] if S.shape[1] > 2 * m_tramas + k_tramos else S
    w = w - w.mean(axis=1, keepdims=True)  # CMN sobre la palabra
    d = (
        librosa.feature.delta(w, width=delta_width, mode="nearest")
        if w.shape[1] >= delta_width
        else np.zeros_like(w)
    )

    fr = librosa.util.frame(y32, frame_length=win_length, hop_length=hop_length)
    fr = fr[:, m_tramas : fr.shape[1] - m_tramas] if fr.shape[1] > 2 * m_tramas + k_tramos else fr
    rms = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=0)), rms_piso))[None, :]
    zcr = (np.abs(np.diff(np.sign(fr), axis=0)) > 0).mean(axis=0)[None, :]

    n = min(w.shape[1], rms.shape[1])
    w = w[:, :n]
    d = d[:, :n]
    rms = rms[:, :n]
    zcr = zcr[:, :n]

    min_tramo = min(
        max(1, int(np.floor((i + 1) * n / k_tramos)) - int(np.floor(i * n / k_tramos)))
        for i in range(k_tramos)
    )

    bloques = {
        "mfcc": tramos_estadisticas(w, k=k_tramos, ddof=ddof_tramos),
        "delta": tramos_estadisticas(d, k=k_tramos, ddof=ddof_tramos),
        "energia": tramos_estadisticas(rms, k=k_tramos, ddof=ddof_tramos),
        "zcr": tramos_estadisticas(zcr, k=k_tramos, ddof=ddof_tramos),
        "duracion": np.array([float(duracion_s)], dtype=np.float64),
    }
    info = {
        "tramas_stft_total": int(S.shape[1]),
        "tramas_frame_total": int(fr.shape[1] + (2 * m_tramas if fr.shape[1] > 0 else 0)),
        "tramas_palabra": int(n),
        "min_tramas_tramo": int(min_tramo),
    }
    return bloques, info


def ajustar_y_escalar(bloques_por_part: dict[str, dict[str, np.ndarray]], ddof: int = 1):
    """Ajusta el escalador en train (igual que en H3) y transforma todas las particiones."""
    tr = bloques_por_part["train"]
    medias, desvios, columnas, pesos = {}, {}, {}, {}
    for b in BLOQUES:
        X_b = tr[b]
        col_ok = np.zeros(X_b.shape[1], dtype=bool)
        mu_b = np.zeros(X_b.shape[1], dtype=np.float64)
        sd_b = np.ones(X_b.shape[1], dtype=np.float64)
        for j in range(X_b.shape[1]):
            col = X_b[:, j]
            finitos = col[~np.isnan(col)]
            if len(finitos) >= 2:
                sd = float(np.std(finitos, ddof=ddof))
                if sd > 1e-9:
                    col_ok[j] = True
                    mu_b[j] = float(np.mean(finitos))
                    sd_b[j] = sd
        medias[b] = mu_b
        desvios[b] = sd_b
        columnas[b] = col_ok
        pesos[b] = 1.0 / np.sqrt(float(col_ok.sum()))

    def transformar(bloques_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        partes = []
        por_bloque = {}
        for b in BLOQUES:
            x = bloques_dict[b][:, columnas[b]]
            z = (x - medias[b][columnas[b]]) / desvios[b][columnas[b]]
            z = np.where(np.isnan(z), 0.0, z) * pesos[b]
            partes.append(z)
            por_bloque[b] = z
        return np.hstack(partes), por_bloque

    return medias, desvios, columnas, pesos, transformar


def contribucion_bloque(Z_b: np.ndarray) -> float:
    """Media sobre i != j de ||z_i,b - z_j,b||^2."""
    n = Z_b.shape[0]
    # Identidad algebraica exacta: promedio_{i != j} ||z_i - z_j||^2 = 2 * sum(var(Z_b, ddof=1))
    # Lo verificamos también con la matriz de distancias:
    dif = Z_b[:, None, :] - Z_b[None, :, :]
    d2 = np.sum(dif ** 2, axis=-1)
    return float(np.sum(d2) / (n * (n - 1)))


def knn_predecir(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    k: int = 5,
    eps: float = 1e-9,
    confianza_min: float = 0.60,
) -> list[str]:
    """K-NN con voto ponderado 1/(d + eps) y rechazo por confianza."""
    d = np.sqrt(np.maximum(0.0, ((X_ev[:, None, :] - X_tr[None, :, :]) ** 2).sum(axis=-1)))
    idx = np.argsort(d, axis=1)[:, :k]
    out = []
    for i in range(len(X_ev)):
        v = collections.Counter()
        for j in idx[i]:
            v[str(y_tr[j])] += 1.0 / (float(d[i, j]) + eps)
        c, w = v.most_common(1)[0]
        total = sum(v.values())
        out.append(c if (w / total) >= confianza_min else "otro")
    return out


def main() -> None:
    t0 = time.perf_counter()
    cfg = cargar_config()
    carpeta_prepro = RAIZ / cfg["rutas"]["voz"] / "prepro"
    csv_path = carpeta_prepro / "metadatos.csv"

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        filas_todas = list(csv.DictReader(f))

    filas_validas = [r for r in filas_todas if r["valido"] == "1"]
    print(f"Total filas en metadatos.csv: {len(filas_todas)} | Válidas: {len(filas_validas)}")

    # 1. Extracción y tiempos
    t_ext_0 = time.perf_counter()
    bloques_lista = {b: [] for b in BLOQUES}
    infos = []
    dif_dur_max = 0.0

    tiempos_clip = []
    for r in filas_validas:
        wav_path = carpeta_prepro / r["archivo"]
        y, fs = sf.read(str(wav_path), dtype="float32")
        dur_csv = float(r["duracion_palabra_s"])
        M = int(round(float(cfg["voz"]["preprocesamiento"]["margen_ms"]) * fs / 1000.0))
        dur_muestras = (len(y) - 2 * M) / float(fs)
        dif_dur_max = max(dif_dur_max, abs(dur_csv - dur_muestras))

        tc0 = time.perf_counter()
        bl, inf = extraer_caracteristicas(y, cfg, duracion_s=None, ddof_tramos=0)
        tiempos_clip.append(time.perf_counter() - tc0)

        for b in BLOQUES:
            bloques_lista[b].append(bl[b])
        infos.append(inf)

    t_ext_total = time.perf_counter() - t_ext_0
    print(f"\n--- 1. TIEMPOS Y DIMENSIONES ---")
    print(f"Tiempo total extracción 1908 clips: {t_ext_total:.2f} s")
    print(f"Tiempo medio por clip: {np.mean(tiempos_clip)*1000:.2f} ms (p50: {np.median(tiempos_clip)*1000:.2f} ms, p99: {np.percentile(tiempos_clip, 99)*1000:.2f} ms)")
    print(f"Máxima diferencia entre duracion_palabra_s del CSV y (len(y) - 2*M)/fs: {dif_dur_max:.2e} s")

    tramas_pal = [x["tramas_palabra"] for x in infos]
    min_tramo = [x["min_tramas_tramo"] for x in infos]
    print(f"Tramas por palabra (n): min={min(tramas_pal)}, p10={int(np.percentile(tramas_pal, 10))}, med={int(np.median(tramas_pal))}, max={max(tramas_pal)}")
    print(f"Tramas mínimas en un tramo (n/5): min={min(min_tramo)}, med={int(np.median(min_tramo))}, max={max(min_tramo)}")

    bloques_arr = {b: np.stack(bloques_lista[b], axis=0) for b in BLOQUES}
    for b in BLOQUES:
        arr = bloques_arr[b]
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"Bloque {b:<10}: shape={arr.shape}, NaN={n_nan}, Inf={n_inf}, min={arr.min():.4f}, max={arr.max():.4f}")

    # Separar por partición
    particiones = np.array([r["particion"] for r in filas_validas])
    clases = np.array([r["clase"] for r in filas_validas])
    origenes = np.array([r["origen"] for r in filas_validas])
    copias = np.array([int(r["copia"]) for r in filas_validas])
    hablantes = np.array([r["hablante"] for r in filas_validas])

    bloques_part = {}
    for part in ("train", "val", "test"):
        m = particiones == part
        bloques_part[part] = {b: bloques_arr[b][m] for b in BLOQUES}

    # 2. Escalador y contribuciones
    medias, desvios, columnas, pesos, transformar = ajustar_y_escalar(bloques_part, ddof=1)
    Z_part = {}
    Z_bl_part = {}
    for part in ("train", "val", "test"):
        Z_part[part], Z_bl_part[part] = transformar(bloques_part[part])

    print(f"\n--- 2. ESCALADOR Y CONTRIBUCIONES POR BLOQUE ---")
    print(f"Dimensión total Z: {Z_part['train'].shape[1]}")
    for b in BLOQUES:
        usadas = int(columnas[b].sum())
        total_b = len(columnas[b])
        sd_min = float(desvios[b][columnas[b]].min())
        sd_max = float(desvios[b][columnas[b]].max())
        c_tr = contribucion_bloque(Z_bl_part["train"][b])
        c_va = contribucion_bloque(Z_bl_part["val"][b])
        c_te = contribucion_bloque(Z_bl_part["test"][b])
        print(
            f"Bloque {b:<10}: cols={usadas}/{total_b}, peso={pesos[b]:.6f}, "
            f"sd_train in [{sd_min:.4f}, {sd_max:.4f}], "
            f"contrib train={c_tr:.4f}, val={c_va:.4f}, test={c_te:.4f}"
        )

    # 3. PCA propio vs scikit-learn y varianza explicada
    print(f"\n--- 3. PCA PROPIO VS SCIKIT-LEARN ---")
    pca_full = PCA(n_componentes=None).ajustar(Z_part["train"])
    sk_full = SkPCA(n_components=min(Z_part["train"].shape[0] - 1, Z_part["train"].shape[1]), svd_solver="full").fit(Z_part["train"])

    dif_var = float(np.max(np.abs(pca_full.varianza - sk_full.explained_variance_)))
    dif_ratio = float(np.max(np.abs(pca_full.proporcion_varianza - sk_full.explained_variance_ratio_)))
    for part in ("train", "val", "test"):
        P_prop = pca_full.transformar(Z_part[part])
        P_sk = sk_full.transform(Z_part[part])
        dif_proy = float(np.max(np.abs( np.abs(P_prop) - np.abs(P_sk) )))
        print(f"PCA diff abs proyecciones en {part:<5}: {dif_proy:.2e}")
    print(f"PCA diff varianza: {dif_var:.2e}, diff proporcion_varianza: {dif_ratio:.2e}")

    for k in (2, 3, 5, 10, 20, 30, 50, 100):
        acum = float(np.sum(pca_full.proporcion_varianza[:k]))
        print(f"Varianza explicada acumulada k={k:<3}: {100*acum:.2f}% (individuales primeros 3: {100*pca_full.proporcion_varianza[0]:.2f}%, {100*pca_full.proporcion_varianza[1]:.2f}%, {100*pca_full.proporcion_varianza[2]:.2f}%)")

    # 4. Silhouette
    print(f"\n--- 4. SILHOUETTE (3 COMANDOS vs 4 CLASES) ---")
    pca20 = PCA(n_componentes=20).ajustar(Z_part["train"])
    pca30 = PCA(n_componentes=30).ajustar(Z_part["train"])

    for part in ("train", "val", "test"):
        m_part = particiones == part
        y_p = clases[m_part]
        orig_p = origenes[m_part]
        cop_p = copias[m_part]
        Z_p = Z_part[part]

        # Subconjuntos:
        subconjuntos = [("todos", np.ones(len(y_p), dtype=bool))]
        if part == "train":
            subconjuntos.append(("k0 (originales)", cop_p == 0))
            subconjuntos.append(("notebook k0", (cop_p == 0) & (orig_p == "notebook")))

        for nom_sub, mask_sub in subconjuntos:
            m_3cmd = mask_sub & (y_p != "otro")
            s_281_3 = silhouette(Z_p[m_3cmd], y_p[m_3cmd])
            s_sk_3 = float(sk_silhouette(Z_p[m_3cmd], y_p[m_3cmd]))
            s_p20_3 = silhouette(pca20.transformar(Z_p[m_3cmd]), y_p[m_3cmd])
            s_p30_3 = silhouette(pca30.transformar(Z_p[m_3cmd]), y_p[m_3cmd])
            s_281_4 = silhouette(Z_p[mask_sub], y_p[mask_sub])
            s_p20_4 = silhouette(pca20.transformar(Z_p[mask_sub]), y_p[mask_sub])
            s_p30_4 = silhouette(pca30.transformar(Z_p[mask_sub]), y_p[mask_sub])
            print(
                f"{part:<5} ({nom_sub:<15}) | 3 cmd (n={int(m_3cmd.sum()):4d}): "
                f"281d={s_281_3:.4f} (diff sk={abs(s_281_3-s_sk_3):.1e}), PCA30={s_p30_3:.4f}, PCA20={s_p20_3:.4f} | "
                f"4 clases (n={int(mask_sub.sum()):4d}): 281d={s_281_4:.4f}, PCA30={s_p30_4:.4f}, PCA20={s_p20_4:.4f}"
            )
            # Por bloque en 3 comandos
            bl_str = []
            for b in BLOQUES:
                sb = silhouette(Z_bl_part[part][b][m_3cmd], y_p[m_3cmd])
                bl_str.append(f"{b}={sb:.4f}")
            print(f"      Por bloque (3 cmd): " + ", ".join(bl_str))

    # 5. Chequeo con el K-NN del borrador (CV por hablante en train y evaluación en val)
    print(f"\n--- 5. CHEQUEO CON K-NN DEL BORRADOR (SIN UMBRAL DE DISTANCIA) ---")
    m_tr = particiones == "train"
    y_tr_all = clases[m_tr]
    hab_tr_all = hablantes[m_tr]
    orig_tr_all = origenes[m_tr]
    cop_tr_all = copias[m_tr]

    # Hablantes de notebook en train (son 10 personas, con pares p01/p02, p09/p13, p14/p17 ya unificados en 'hablante')
    habs_nb = sorted(set(hab_tr_all[orig_tr_all == "notebook"]))
    print(f"Hablantes de notebook en train ({len(habs_nb)}): {habs_nb}")

    # Bloques de train para re-ajustar escalador y PCA en cada pliegue sin fuga
    bl_tr_dict = bloques_part["train"]

    for modo_cv, lista_pliegues in [
        ("LOO (10 hablantes)", [[h] for h in habs_nb]),
        ("6 pliegues (proto_d)", [habs_nb[i::6] for i in range(6)]),
    ]:
        print(f"\nValidación cruzada en train ({modo_cv}, evaluando en clips limpios k0 de notebook):")
        for k_knn in (3, 5, 7, 9):
            res_esp = {}
            for espacio in ("281d", "PCA 30", "PCA 20"):
                cmd_tot, cmd_ok, otr_tot, otr_fa = 0, 0, 0, 0
                for pl in lista_pliegues:
                    fuera = set(pl)
                    # Entrenan todos los que no son de los hablantes 'fuera'
                    m_in = np.array([h not in fuera for h in hab_tr_all])
                    # Evalúan los originales k0 de notebook de los hablantes 'fuera'
                    m_ev = np.array([(h in fuera) and (o == "notebook") and (c == 0) for h, o, c in zip(hab_tr_all, orig_tr_all, cop_tr_all)])

                    # Ajustar escalador en m_in
                    _, _, _, _, transf_pl = ajustar_y_escalar({"train": {b: bl_tr_dict[b][m_in] for b in BLOQUES}}, ddof=1)
                    X_in, _ = transf_pl({b: bl_tr_dict[b][m_in] for b in BLOQUES})
                    X_ev, _ = transf_pl({b: bl_tr_dict[b][m_ev] for b in BLOQUES})

                    if espacio == "PCA 30":
                        p = PCA(n_componentes=30).ajustar(X_in)
                        X_in, X_ev = p.transformar(X_in), p.transformar(X_ev)
                    elif espacio == "PCA 20":
                        p = PCA(n_componentes=20).ajustar(X_in)
                        X_in, X_ev = p.transformar(X_in), p.transformar(X_ev)

                    preds = knn_predecir(X_in, y_tr_all[m_in], X_ev, k=k_knn)
                    for real, pred in zip(y_tr_all[m_ev], preds):
                        if real != "otro":
                            cmd_tot += 1
                            cmd_ok += int(pred == real)
                        else:
                            otr_tot += 1
                            otr_fa += int(pred != "otro")
                res_esp[espacio] = f"acierto {100*cmd_ok/cmd_tot:.1f}% ({cmd_ok}/{cmd_tot}), otro_acept {100*otr_fa/otr_tot:.1f}% ({otr_fa}/{otr_tot})"
            print(f"  K={k_knn}: 281d -> {res_esp['281d']} | PCA 30 -> {res_esp['PCA 30']} | PCA 20 -> {res_esp['PCA 20']}")

    # Evaluación en val (incluyendo los 14 no válidos de V2 como 'otro')
    filas_val_todas = [r for r in filas_todas if r["particion"] == "val"]
    n_val_inval_otro = sum(1 for r in filas_val_todas if r["valido"] == "0" and r["clase"] == "otro")
    n_val_inval_cmd = sum(1 for r in filas_val_todas if r["valido"] == "0" and r["clase"] != "otro")
    print(f"\nEvaluación en val (87 válidos + {n_val_inval_otro} 'otro' no válidos de V2 = 101 clips; comandos no válidos={n_val_inval_cmd}):")
    y_va = clases[particiones == "val"]
    for k_knn in (3, 5, 7, 9):
        res_esp = {}
        for espacio in ("281d", "PCA 30", "PCA 20"):
            X_in = Z_part["train"]
            X_ev = Z_part["val"]
            if espacio == "PCA 30":
                X_in, X_ev = pca30.transformar(X_in), pca30.transformar(X_ev)
            elif espacio == "PCA 20":
                X_in, X_ev = pca20.transformar(X_in), pca20.transformar(X_ev)
            preds = knn_predecir(X_in, y_tr_all, X_ev, k=k_knn)
            cmd_tot = int(np.sum(y_va != "otro")) + n_val_inval_cmd
            cmd_ok = sum(1 for r, p in zip(y_va, preds) if r != "otro" and p == r)
            otr_tot = int(np.sum(y_va == "otro")) + n_val_inval_otro
            otr_fa = sum(1 for r, p in zip(y_va, preds) if r == "otro" and p != "otro")
            res_esp[espacio] = f"acierto {100*cmd_ok/cmd_tot:.1f}% ({cmd_ok}/{cmd_tot}), otro_acept {100*otr_fa/otr_tot:.1f}% ({otr_fa}/{otr_tot})"
        print(f"  K={k_knn}: 281d -> {res_esp['281d']} | PCA 30 -> {res_esp['PCA 30']} | PCA 20 -> {res_esp['PCA 20']}")

    print(f"\nTiempo total del script: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
