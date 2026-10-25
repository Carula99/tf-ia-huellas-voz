"""Fase 5 · El controlador de la app, sin Tk.

Contrato: docs/contratos/f5_integracion.md (versión 1.1), secciones 3.3 y 4.6. Arma el agente con la
percepción de la huella (H2 al cargarla; H3 y H4 al decir "Reconocer") y de la voz (V2 a V4), el
micrófono con el push-to-talk, las huellas que ofrece la app y el visor PCA. Después de cada evento
devuelve una Pantalla con todo lo que hay que dibujar: la ventana (ventana.py) solo la dibuja.
Ningún evento lanza: un error deja un aviso y el agente como estaba.

Si se le pasa `registro`, anota cada acción del agente y cada aviso en un CSV (sin audio).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..agente.agente import AUTORIZADO, NO_AUTORIZADO, REGLAS_AVISO, Accion, Agente, Estado, nombres_identidades
from ..agente.captura import Microfono, PushToTalk, SinMicrofono
from ..agente.percepcion import evaluar_huella, leer_imagen, preparar_huella
from ..config import RAIZ
from ..voz.audio import leer_audio
from .muestras import EntradaHuella, GeneradorDemo, catalogo, entrada_archivo
from .visor import VisorPCA, armar_visor

COLORES: tuple[str, ...] = ("neutro", "verde", "rojo")
AVISOS: dict[str, str] = {
    "inicio": "Elegí una huella para empezar",
    "audio_ilegible": "No se pudo leer el audio: elegí otro archivo",
    "sin_microfono": "No hay micrófono disponible: usá «Abrir audio…»",
    "error": "Error inesperado: {tipo}: {mensaje}",
}
COLUMNAS_REGISTRO: tuple[str, ...] = (
    "momento", "percepcion", "detalle", "regla", "estado_antes", "estado_despues", "mensaje",
)
MODELOS: tuple[tuple[str, str], ...] = (
    ("modelo_huellas.npz", "src.huellas.clasificacion"),
    ("modelo_voz.npz", "src.voz.clasificacion"),
)

_EVALUADAS = (Estado.EVALUADA_AUTORIZADA, Estado.EVALUADA_RECHAZADA)


@dataclass(frozen=True, eq=False)   # eq=False: tiene arrays
class Pantalla:
    estado: str                        # el valor del Estado del agente
    huella: str                        # nombre de la huella cargada, o ""
    imagen: np.ndarray | None          # la huella cruda cargada
    esqueleto: np.ndarray | None       # bool (250, 250) de la huella preparada
    cartel: str                        # AUTORIZADO, NO_AUTORIZADO o ""
    color: str                         # uno de COLORES
    identificacion: str                # "Identidad Sintética #N" solo después de la regla 5; si no, ""
    distancia: float | None            # puntaje de la última evaluación (la barra contra τ*)
    tau: float | None
    punto_pca: tuple[float, float] | None
    mensaje: str                       # el texto de la última acción o aviso
    aviso: bool                        # True si lo último fue un aviso
    regla: int                         # regla de la última acción del agente; 0 si fue del controlador
    comando: str                       # el último comando decidido por la voz ("" si todavía no hubo)
    detalle_voz: str                   # diagnóstico del último audio ("" si todavía no hubo)
    microfono: bool                    # hay un micrófono abierto
    grabando: bool                     # el push-to-talk está grabando
    nivel_db: float | None             # medidor de nivel (None sin micrófono)
    cerrar: bool                       # la app tiene que cerrarse


def _coma(valor: float, decimales: int) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",") if math.isfinite(valor) else str(valor)


def detalle_voz(rec: Any) -> str:
    """Lo que dijo V4 de un audio, para la pantalla y el registro."""
    if not rec.valido:
        return f"{rec.comando} (audio no válido para V2: {rec.motivo})"
    return (f"{rec.comando} (el K-NN votó {rec.prediccion}, confianza {_coma(float(rec.confianza), 2)}, "
            f"distancia {_coma(float(rec.distancia), 3)} con τ* {_coma(float(rec.tau), 3)})")


class Controlador:
    """Une el agente con la percepción, el micrófono, las huellas de la app y el visor."""

    def __init__(self, cfg: dict, preparar: Callable, evaluar: Callable, reconocer_voz: Callable, *,
                 visor: VisorPCA | None = None, generador: GeneradorDemo | None = None,
                 entradas: list[EntradaHuella] | None = None, microfono: Microfono | None = None,
                 registro: str | os.PathLike | None = None) -> None:
        self.cfg = cfg
        self.agente = Agente(cfg, preparar, evaluar)
        self.entradas: list[EntradaHuella] = list(entradas) if entradas is not None else []
        self.visor = visor
        self.generador = generador
        self.microfono = microfono
        self._reconocer_voz = reconocer_voz
        self._nombres = nombres_identidades(cfg)
        self._ptt: PushToTalk | None = None
        self._mensaje, self._aviso, self._regla = AVISOS["inicio"], False, 0
        self._comando, self._detalle_voz = "", ""
        self._cerrado = False
        self._registro: Path | None = None
        if registro is not None:
            self._registro = Path(os.fspath(registro))
            self._registro.parent.mkdir(parents=True, exist_ok=True)
            with open(self._registro, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(COLUMNAS_REGISTRO)

    @classmethod
    def desde_archivos(cls, cfg: dict, *, modelos: str | os.PathLike | None = None,
                       huellas: str | os.PathLike | None = None, microfono: bool = True,
                       dispositivo: int | None = None,
                       registro: str | os.PathLike | None = None) -> "Controlador":
        """El controlador de la app con los modelos guardados de H4 y V4.

        Sin caracteristicas.npz no hay visor, y sin metadatos.csv la lista de huellas queda vacía.
        FileNotFoundError, con el comando que lo regenera, si falta un modelo.
        """
        from ..huellas.clasificacion import cargar_modelo as cargar_modelo_huellas
        from ..voz.clasificacion import cargar_modelo as cargar_modelo_voz
        from ..voz.clasificacion import reconocer as reconocer_v4

        modelos = RAIZ / cfg["rutas"]["modelos"] if modelos is None else Path(os.fspath(modelos))
        huellas = RAIZ / cfg["rutas"]["huellas"] if huellas is None else Path(os.fspath(huellas))
        for nombre, modulo in MODELOS:
            if not (modelos / nombre).is_file():
                raise FileNotFoundError(f"No existe el modelo {modelos / nombre}. Se regenera con: "
                                        f"venv\\Scripts\\python -m {modulo} --sobrescribir")
        modelo_h = cargar_modelo_huellas(modelos / MODELOS[0][0])
        modelo_v = cargar_modelo_voz(modelos / MODELOS[1][0])
        try:
            visor = armar_visor(cfg, modelo_h, huellas)
        except (FileNotFoundError, ValueError):
            visor = None
        try:
            entradas = catalogo(cfg, huellas)
        except FileNotFoundError:
            entradas = []
        return cls(cfg, lambda imagen: preparar_huella(imagen, cfg), lambda h: evaluar_huella(h, modelo_h, cfg),
                   lambda y, fs: reconocer_v4(y, fs, modelo_v, cfg), visor=visor, generador=GeneradorDemo(cfg),
                   entradas=entradas, microfono=Microfono(cfg, dispositivo) if microfono else None,
                   registro=registro)

    # ----------------------------------------------------------------------- pantalla
    def _terminado(self) -> bool:
        return self._cerrado or self.agente.estado is Estado.FIN

    def _microfono_abierto(self) -> bool:
        return self.microfono is not None and self.microfono.abierto

    def pantalla(self) -> Pantalla:
        a = self.agente
        resultado = a.resultado if a.estado in _EVALUADAS else None
        punto = None
        if resultado is not None and self.visor is not None and getattr(resultado, "z", None) is not None:
            try:
                punto = tuple(float(c) for c in self.visor.proyectar(resultado.z))
            except ValueError:
                punto = None
        cartel, color = {Estado.EVALUADA_AUTORIZADA: (AUTORIZADO, "verde"),
                         Estado.EVALUADA_RECHAZADA: (NO_AUTORIZADO, "rojo")}.get(a.estado, ("", "neutro"))
        identificacion = ""
        if a.identificada and a.estado is Estado.EVALUADA_AUTORIZADA:
            identificacion = self._nombres[int(a.resultado.identidad)]
        abierto = self._microfono_abierto()
        return Pantalla(
            estado=a.estado.value, huella=a.huella, imagen=a.imagen,
            esqueleto=None if a.preparada is None else getattr(a.preparada, "esqueleto", None),
            cartel=cartel, color=color, identificacion=identificacion,
            distancia=None if resultado is None else float(resultado.distancia),
            tau=None if resultado is None else float(resultado.tau), punto_pca=punto,
            mensaje=self._mensaje, aviso=self._aviso, regla=self._regla,
            comando=self._comando, detalle_voz=self._detalle_voz, microfono=abierto,
            grabando=abierto and self._ptt is not None and self._ptt.estado == "grabando",
            nivel_db=self.microfono.nivel_db() if abierto else None, cerrar=self._terminado(),
        )

    # ----------------------------------------------------------------------- internas
    def _registrar(self, percepcion: str, detalle: str, regla: int, antes: Estado) -> None:
        if self._registro is None:
            return
        with open(self._registro, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="milliseconds"), percepcion, detalle,
                                    regla, antes.value, self.agente.estado.value, self._mensaje])

    def _aplicar(self, accion: Accion, percepcion: str, detalle: str, antes: Estado) -> None:
        self._mensaje, self._aviso, self._regla = accion.texto, accion.regla in REGLAS_AVISO, accion.regla
        if accion.regla == 9:
            self._liberar_microfono()
        self._registrar(percepcion, detalle, accion.regla, antes)

    def _avisar(self, texto: str) -> None:
        self._mensaje, self._aviso, self._regla = texto, True, 0
        self._registrar("aviso", texto, 0, self.agente.estado)

    def _liberar_microfono(self) -> None:
        if self._ptt is not None:
            self._ptt.cancelar()
        if self.microfono is not None:
            self.microfono.cerrar()

    def _evento(self, hacer: Callable[[], Any]) -> Pantalla:
        """Corre un evento: nada después de cerrar y ninguna excepción hacia afuera."""
        if self._terminado():
            return self.pantalla()
        try:
            hacer()
        except Exception as e:
            self._avisar(AVISOS["error"].format(tipo=type(e).__name__, mensaje=str(e)))
        return self.pantalla()

    def _huella(self, imagen: np.ndarray | None, nombre: str) -> None:
        antes = self.agente.estado
        self._aplicar(self.agente.percibir_huella(imagen, nombre), "huella", nombre, antes)

    def _comando_audio(self, y: np.ndarray, fs: int) -> None:
        rec = self._reconocer_voz(y, fs)
        self._comando, self._detalle_voz = str(rec.comando), detalle_voz(rec)
        antes = self.agente.estado
        self._aplicar(self.agente.percibir_comando(self._comando), "comando", self._detalle_voz, antes)

    def _con_microfono(self) -> bool:
        if self._microfono_abierto() and self._ptt is not None:
            return True
        self._avisar(AVISOS["sin_microfono"])
        return False

    # ----------------------------------------------------------------------- eventos
    def elegir(self, entrada: EntradaHuella) -> Pantalla:
        return self._evento(lambda: self._huella(leer_imagen(entrada.ruta), entrada.nombre))

    def abrir_archivo(self, ruta: str | os.PathLike) -> Pantalla:
        return self.elegir(entrada_archivo(ruta))

    def variante_nueva(self, identidad: int) -> Pantalla:
        def hacer():
            entrada, imagen = self.generador.variante_nueva(identidad)
            self._huella(imagen, entrada.nombre)
        return self._evento(hacer)

    def intruso_nuevo(self) -> Pantalla:
        def hacer():
            entrada, imagen = self.generador.intruso_nuevo()
            self._huella(imagen, entrada.nombre)
        return self._evento(hacer)

    def comando_audio(self, y: np.ndarray, fs: int) -> Pantalla:
        return self._evento(lambda: self._comando_audio(y, fs))

    def comando_wav(self, ruta: str | os.PathLike) -> Pantalla:
        def hacer():
            try:
                y, fs = leer_audio(ruta)
            except Exception:
                self._avisar(AVISOS["audio_ilegible"])
                return
            self._comando_audio(y, fs)
        return self._evento(hacer)

    def abrir_microfono(self) -> Pantalla:
        def hacer():
            if self.microfono is None:
                self._avisar(AVISOS["sin_microfono"])
                return
            try:
                self.microfono.abrir()
            except SinMicrofono:
                self._avisar(AVISOS["sin_microfono"])
                return
            if self._ptt is None or self._ptt.bufer is not self.microfono.bufer:
                self._ptt = PushToTalk(self.microfono.bufer, self.microfono.fs, self.cfg)
        return self._evento(hacer)

    def apretar(self) -> Pantalla:
        return self._evento(lambda: self._con_microfono() and self._ptt.apretar())

    def soltar(self) -> Pantalla:
        return self._evento(lambda: self._con_microfono() and self._ptt.soltar())

    def actualizar(self) -> Pantalla:
        """Lo llama la app cada 30 ms: corta en el tope y, con el post-roll grabado, reconoce el clip."""
        def hacer():
            if not (self._microfono_abierto() and self._ptt is not None):
                return
            if self._ptt.tope():
                self._ptt.soltar()
            clip = self._ptt.clip()
            if clip is not None:
                self._comando_audio(clip, self.microfono.fs)
        return self._evento(hacer)

    def cerrar(self) -> Pantalla:
        """Libera el micrófono; desde acá la pantalla pide cerrar y los eventos no hacen nada."""
        self._liberar_microfono()
        self._cerrado = True
        return self.pantalla()
