"""Fase 5 · Captura de voz: micrófono abierto todo el tiempo, búfer circular y push-to-talk.

Contrato: docs/contratos/f5_integracion.md (versión 1.1), secciones 2.3 y 4.3. El micrófono escribe
en un búfer circular desde su propio hilo (el callback de sounddevice), así lo que se dijo justo
antes de apretar el botón ya está grabado. Al soltar, el clip es la ventana [a, b) en índices
absolutos: hasta app.pre_roll_s de antes de apretar, lo que se habló (con el tope de
voz.push_to_talk_max_s) y app.post_roll_s de después de soltar.

sounddevice se importa solo dentro de las funciones que lo usan: el módulo se puede importar sin
PortAudio, y los tests reemplazan el stream por uno falso con el parámetro `fabrica`.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable

import numpy as np

REPOSO, GRABANDO, ESPERANDO = "reposo", "grabando", "esperando"


class SinMicrofono(RuntimeError):
    """No hay micrófono, o no se pudo abrir ni arrancar."""


def _es_entero(valor) -> bool:
    return isinstance(valor, (int, np.integer)) and not isinstance(valor, (bool, np.bool_))


def nivel_db(bloque: np.ndarray, piso: float = 1e-10) -> float:
    """20 · log10(max(rms, piso)) del bloque; un bloque vacío da 20 · log10(piso)."""
    if isinstance(piso, (bool, np.bool_)) or not isinstance(piso, (int, float, np.integer, np.floating)):
        raise ValueError(f"piso tiene que ser un número real positivo; es {piso!r}.")
    if not math.isfinite(piso) or piso <= 0:
        raise ValueError(f"piso tiene que ser un número real positivo; es {piso!r}.")
    x = np.asarray(bloque, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"El bloque tiene que ser de 1 dimensión; tiene forma {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("El bloque tiene valores NaN o infinitos.")
    rms = math.sqrt(float(np.mean(x * x))) if x.size else 0.0
    return 20.0 * math.log10(max(rms, float(piso)))


def listar_microfonos() -> list[tuple[int, str, int]]:
    """(índice, nombre, fs nativa) de cada dispositivo de entrada de sounddevice."""
    import sounddevice as sd

    return [(i, str(d["name"]), int(round(float(d["default_samplerate"]))))
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


# --------------------------------------------------------------------------- búfer circular
class BuferCircular:
    """Las últimas `capacidad` muestras (float32), con índices absolutos: la muestra i es la i-ésima
    que se escribió desde que se creó el búfer. escribir y leer se pueden llamar desde hilos distintos."""

    def __init__(self, capacidad: int) -> None:
        if not _es_entero(capacidad) or capacidad < 1:
            raise ValueError(f"La capacidad tiene que ser un entero ≥ 1; es {capacidad!r}.")
        self.capacidad = int(capacidad)
        self._datos = np.zeros(self.capacidad, dtype=np.float32)
        self._escritas = 0
        self._candado = threading.Lock()

    @property
    def escritas(self) -> int:
        """Muestras escritas desde que se creó el búfer."""
        return self._escritas

    def escribir(self, bloque: np.ndarray) -> None:
        x = np.asarray(bloque)
        if x.ndim != 1:
            raise ValueError(f"El bloque tiene que ser de 1 dimensión; tiene forma {x.shape}.")
        n = x.shape[0]
        if n == 0:
            return
        x = x.astype(np.float32)          # una copia: el que llama puede reusar su array
        with self._candado:
            inicio = self._escritas
            if n > self.capacidad:         # quedan las últimas `capacidad` muestras
                inicio += n - self.capacidad
                x = x[-self.capacidad:]
            p = inicio % self.capacidad
            primera = min(self.capacidad - p, x.shape[0])
            self._datos[p:p + primera] = x[:primera]
            self._datos[:x.shape[0] - primera] = x[primera:]
            self._escritas += n

    def _leer(self, desde: int, hasta: int) -> np.ndarray:
        """Sin el candado y sin controles: [desde, hasta) ya válido."""
        n = hasta - desde
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        p = desde % self.capacidad
        primera = min(self.capacidad - p, n)
        return np.concatenate([self._datos[p:p + primera], self._datos[:n - primera]])

    def leer(self, desde: int, hasta: int) -> np.ndarray:
        """Copia float32 de las muestras absolutas [desde, hasta)."""
        if not (_es_entero(desde) and _es_entero(hasta)):
            raise ValueError(f"desde y hasta tienen que ser enteros; son {desde!r} y {hasta!r}.")
        desde, hasta = int(desde), int(hasta)
        with self._candado:
            if desde > hasta:
                raise ValueError(f"desde ({desde}) es mayor que hasta ({hasta}).")
            if hasta > self._escritas:
                raise ValueError(f"hasta ({hasta}) pasa de las muestras escritas ({self._escritas}).")
            if desde < 0 or desde < self._escritas - self.capacidad:
                raise ValueError(f"Las muestras desde {desde} ya se pisaron (quedan desde "
                                 f"{max(0, self._escritas - self.capacidad)}).")
            return self._leer(desde, hasta)

    def leer_recortado(self, desde: int, hasta: int) -> np.ndarray:
        """Como leer, pero empieza en la muestra más vieja que quede si [desde, hasta) ya se pisó en parte."""
        with self._candado:
            desde = max(int(desde), self._escritas - self.capacidad, 0)
            hasta = min(int(hasta), self._escritas)
            return self._leer(desde, max(desde, hasta))

    def ultimas(self, n: int) -> np.ndarray:
        """Las últimas min(n, escritas, capacidad) muestras."""
        if not _es_entero(n) or n < 0:
            raise ValueError(f"n tiene que ser un entero ≥ 0; es {n!r}.")
        with self._candado:
            k = min(int(n), self._escritas, self.capacidad)
            return self._leer(self._escritas - k, self._escritas)


# --------------------------------------------------------------------------- push-to-talk
class PushToTalk:
    """Graba mientras se mantiene apretado (con tope) y devuelve el clip con pre-roll y post-roll."""

    def __init__(self, bufer: BuferCircular, fs: int, cfg: dict) -> None:
        if not _es_entero(fs) or fs < 1:
            raise ValueError(f"fs tiene que ser un entero ≥ 1; es {fs!r}.")
        fs = int(fs)
        self.pre = int(round(float(cfg["app"]["pre_roll_s"]) * fs))
        self.post = int(round(float(cfg["app"]["post_roll_s"]) * fs))
        self.maximo = int(round(float(cfg["voz"]["push_to_talk_max_s"]) * fs))
        minimo = self.pre + self.maximo + self.post + fs
        if bufer.capacidad < minimo:
            raise ValueError(f"El búfer guarda {bufer.capacidad} muestras; hacen falta al menos {minimo} "
                             "(pre-roll + tope + post-roll + 1 s).")
        self.bufer = bufer
        self.fs = fs
        self.estado = REPOSO
        self._inicio_voz: int | None = None
        self._fin_voz: int | None = None

    def apretar(self) -> bool:
        """En reposo empieza a grabar; si no, no hace nada (la repetición de teclas manda varias pulsaciones)."""
        if self.estado != REPOSO:
            return False
        self._inicio_voz, self._fin_voz = self.bufer.escritas, None
        self.estado = GRABANDO
        return True

    def soltar(self) -> bool:
        """Grabando: termina la voz (con el tope) y espera el post-roll. Si no, no hace nada."""
        if self.estado != GRABANDO:
            return False
        self._fin_voz = min(self.bufer.escritas, self._inicio_voz + self.maximo)
        self.estado = ESPERANDO
        return True

    def tope(self) -> bool:
        return self.estado == GRABANDO and self.bufer.escritas - self._inicio_voz >= self.maximo

    def ventana(self) -> tuple[int, int] | None:
        if self.estado != ESPERANDO:
            return None
        return max(0, self._inicio_voz - self.pre), self._fin_voz + self.post

    def clip(self) -> np.ndarray | None:
        """El clip [a, b) cuando ya se grabó el post-roll (y vuelve a reposo); si no, None."""
        ventana = self.ventana()
        if ventana is None or self.bufer.escritas < ventana[1]:
            return None
        clip = self.bufer.leer_recortado(*ventana)
        self.cancelar()
        return clip

    def cancelar(self) -> None:
        self.estado = REPOSO
        self._inicio_voz = self._fin_voz = None


# --------------------------------------------------------------------------- micrófono
class Microfono:
    """El micrófono abierto todo el tiempo sobre un BuferCircular de app.bufer_s segundos.

    `fabrica` crea el stream de entrada con la firma de sounddevice.InputStream (por defecto, esa)."""

    def __init__(self, cfg: dict, dispositivo: int | None = None, fs: int | None = None,
                 fabrica: Callable[..., Any] | None = None) -> None:
        self.cfg = cfg
        self.dispositivo = dispositivo
        self.fs: int | None = None if fs is None else int(fs)
        self._fabrica = fabrica
        self.bufer: BuferCircular | None = None
        self.abierto = False
        self._stream = None

    def abrir(self) -> None:
        """Abre el stream y arranca a grabar. Cualquier falla se convierte en SinMicrofono."""
        if self.abierto:
            return
        try:
            fabrica = self._fabrica
            if fabrica is None or self.fs is None:
                import sounddevice as sd
                if self.fs is None:
                    info = sd.query_devices(self.dispositivo, "input")
                    self.fs = int(round(float(info["default_samplerate"])))
                if fabrica is None:
                    fabrica = sd.InputStream
            bufer = BuferCircular(int(round(float(self.cfg["app"]["bufer_s"]) * self.fs)))

            def callback(indata, frames, tiempo, estado):
                bufer.escribir(indata[:, 0])

            self._stream = fabrica(samplerate=self.fs, channels=1, dtype="float32",
                                   device=self.dispositivo, callback=callback)
            self.bufer = bufer
            self._stream.start()
        except Exception as e:
            self._cerrar_stream()
            raise SinMicrofono(f"No se pudo abrir el micrófono: {e}") from e
        self.abierto = True

    def _cerrar_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        for metodo in ("stop", "close"):
            try:
                getattr(stream, metodo)()
            except Exception:
                pass

    def cerrar(self) -> None:
        """Libera el micrófono. Se puede llamar varias veces y sin haber abierto."""
        self._cerrar_stream()
        self.abierto = False

    def nivel_db(self) -> float | None:
        """El nivel de los últimos app.medidor_ms (None si no está abierto)."""
        if not self.abierto:
            return None
        n = int(round(float(self.cfg["app"]["medidor_ms"]) / 1000.0 * self.fs))
        return nivel_db(self.bufer.ultimas(n))
