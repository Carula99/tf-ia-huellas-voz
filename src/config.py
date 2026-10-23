"""Lectura única de config.yaml.

Contrato: docs/contratos/fase0_setup.md, sección 8 (las claves de H1 vienen de
docs/contratos/h1_generador.md, sección 1, y las de H2, de
h2_preprocesamiento.md, sección 1). Todos los módulos leen la
configuración con cargar_config(), así una clave faltante falla siempre igual y
con un mensaje que la nombra.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

RAIZ: Path = Path(__file__).resolve().parents[1]
"""Raíz del repo (la carpeta de config.yaml), calculada desde este archivo."""

CLAVES_OBLIGATORIAS: tuple[str, ...] = (
    "semilla_global",
    "rutas.datos",
    "rutas.huellas",
    "rutas.voz",
    "rutas.modelos",
    "rutas.figuras",
    "huellas.identidades",
    "huellas.variantes_por_identidad",
    "huellas.particion.train",
    "huellas.particion.val",
    "huellas.particion.test",
    "huellas.reservadas_demo",
    "huellas.intrusos.cantidad",
    "huellas.intrusos.variantes_por_intruso",
    "huellas.intrusos.semilla_base",
    "huellas.intrusos.val",
    "huellas.intrusos.test",
    "huellas.intrusos.tipos",
    "huellas.generador.tamano_px",
    "huellas.generador.periodo_base_px",
    "huellas.generador.iteraciones",
    "huellas.generador.orientaciones",
    "huellas.variantes.rotacion_max_grados",
    "huellas.variantes.traslacion_max_px",
    "huellas.variantes.escala_max",
    "huellas.variantes.corte_periferico",
    "huellas.variantes.prob_corte",
    "huellas.variantes.elastica_max_px",
    "huellas.variantes.presion_max",
    "huellas.variantes.ruido_sigma_max",
    "huellas.variantes.sal_pimienta_max",
    "huellas.ventana_px",
    "huellas.periodo_crestas_px",
    "huellas.preprocesamiento.mediana_px",
    "huellas.preprocesamiento.umbral_segmentacion",
    "huellas.preprocesamiento.sigma_tensor_px",
    "huellas.preprocesamiento.sigma_ajuste_px",
    "huellas.preprocesamiento.radio_poincare_px",
    "huellas.preprocesamiento.margen_singularidades_px",
    "huellas.preprocesamiento.holgura_nucleo_px",
    "huellas.preprocesamiento.clahe_limite",
    "huellas.preprocesamiento.clahe_grilla",
    "huellas.kmeans.k",
    "huellas.kmeans.n_init",
    "huellas.kmeans.max_iter",
    "huellas.kmeans.tol",
    "huellas.umbral.criterio",
    "huellas.umbral.far_max_val",
    "voz.frecuencia_muestreo",
    "voz.comandos",
    "voz.clase_rechazo",
    "voz.duracion_palabra_s",
    "voz.push_to_talk_max_s",
    "voz.pasabanda_hz",
    "voz.preenfasis",
    "voz.mfcc.n_mfcc",
    "voz.mfcc.n_mels",
    "voz.mfcc.fmax",
    "voz.mfcc.ventana_ms",
    "voz.mfcc.avance_ms",
    "voz.tramos",
    "voz.knn.k_candidatos",
    "voz.knn.eps",
    "voz.knn.confianza_min",
    "voz.knn.pca_componentes_candidatas",
    "voz.dataset.clip_max_s",
    "voz.dataset.saturacion",
    "voz.dataset.saturacion_frac",
    "voz.dataset.mismo_hablante",
    "voz.dataset.particiones.train",
    "voz.dataset.particiones.val",
    "voz.dataset.particiones.test",
    "voz.dataset.min_por_comando",
    "voz.dataset.deteccion.trama_ms",
    "voz.dataset.deteccion.avance_ms",
    "voz.dataset.deteccion.rango_db",
    "voz.dataset.deteccion.margen_piso_db",
    "voz.dataset.deteccion.dif_palabra_db",
    "voz.dataset.deteccion.nivel_min_db",
    "voz.dataset.deteccion.pausa_min_s",
    "voz.dataset.deteccion.segmento_min_s",
    "voz.dataset.deteccion.margen_borde_s",
    "voz.dataset.whatsapp.margen_s",
    "voz.dataset.whatsapp.factor_doble",
    "voz.dataset.ruido.clips_por_archivo",
    "voz.dataset.ruido.particiones.train",
    "voz.dataset.ruido.particiones.val",
    "voz.dataset.ruido.particiones.test",
    "voz.dataset.tts.voces",
    "voz.dataset.tts.velocidades",
    "voz.dataset.tts.otros",
    "metas.identificacion_min",
    "metas.far_max",
    "metas.frr_max",
    "metas.silhouette_min",
    "metas.voz_acierto_min",
    "metas.voz_otro_aceptado_max",
)
"""Hojas obligatorias de config.yaml, con el camino separado por puntos."""


def _claves_faltantes(datos: dict, claves: tuple[str, ...]) -> list[str]:
    """Caminos de `claves` que no se pueden recorrer en `datos`."""
    faltantes = []
    for camino in claves:
        actual = datos
        for parte in camino.split("."):
            if not isinstance(actual, dict) or parte not in actual:
                faltantes.append(camino)
                break
            actual = actual[parte]
    return faltantes


def cargar_config(ruta: str | os.PathLike | None = None) -> dict:
    """Lee config.yaml (en UTF-8) y devuelve su contenido como un dict nuevo.

    Args:
        ruta: archivo YAML a leer. Si es None, se usa RAIZ / "config.yaml".

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el contenido no es un mapeo o si falta alguna clave
            obligatoria; el mensaje nombra el camino completo de cada una.

    El archivo se lee de nuevo en cada llamada: el dict devuelto no comparte nada
    con el de otras llamadas, así que modificarlo no afecta al resto.
    """
    archivo = RAIZ / "config.yaml" if ruta is None else Path(os.fspath(ruta))
    with open(archivo, encoding="utf-8") as f:
        datos = yaml.safe_load(f)

    if not isinstance(datos, dict):
        raise ValueError(
            f"{archivo} tiene que ser un mapeo YAML (clave: valor); "
            f"se leyó {type(datos).__name__}."
        )

    faltantes = _claves_faltantes(datos, CLAVES_OBLIGATORIAS)
    if faltantes:
        raise ValueError(
            f"A {archivo} le faltan {len(faltantes)} claves obligatorias: " + ", ".join(faltantes)
        )
    return datos
