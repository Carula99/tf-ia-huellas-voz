"""Fase 5 · El agente reactivo basado en modelos, como máquina de estados.

Contrato: docs/contratos/f5_integracion.md (versión 1.1), secciones 2.1 y 4.1. Es la tabla de reglas
condición–acción de la sección 2.5 del Documento 2 (reglas 1 a 9), más la regla 10 (una huella que no
se puede leer ni preparar) y la 0 (cualquier percepción en FIN). Recibe percepciones ya interpretadas
(una huella elegida o un comando de voz que clasificó V4) y devuelve una Accion: no sabe nada de la
ventana ni del micrófono.

Estado interno: la huella cargada, lo que devolvió preparar (H2) y el resultado del último
"Reconocer" (H3 + H4). Cargar otra huella (regla 1) o una ilegible (regla 10) borra el resultado
anterior: así "Identificar" nunca muestra la identidad de una huella evaluada antes. La identidad
sale solo en la regla 5, que solo se aplica en EVALUADA_AUTORIZADA.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np


class Estado(str, Enum):
    ESPERANDO_MUESTRA = "ESPERANDO_MUESTRA"
    HUELLA_CARGADA = "HUELLA_CARGADA"
    EVALUADA_AUTORIZADA = "EVALUADA_AUTORIZADA"
    EVALUADA_RECHAZADA = "EVALUADA_RECHAZADA"
    FIN = "FIN"


TIPOS_ACCION: tuple[str, ...] = ("mostrar_huella", "resultado", "identidad", "aviso", "cerrar", "ninguna")
AUTORIZADO: str = "AUTORIZADO"
NO_AUTORIZADO: str = "NO AUTORIZADO"
REGLAS_AVISO: frozenset[int] = frozenset({2, 4, 6, 8, 10})
TEXTOS: dict[int, str] = {
    1: "Huella cargada: decí Reconocer",
    2: "Primero elegí una huella",
    4: "Primero decí Reconocer",
    6: "Huella no autorizada: no se muestra ninguna identidad",
    8: "No entendí, repetí el comando",
    9: "Cerrando la aplicación",
    10: "No se pudo leer la huella: elegí otra",
}

# Los tres comandos (sección 0.1); la clase de rechazo sale de voz.clase_rechazo.
RECONOCER, IDENTIFICAR, SALIR = "reconocer", "identificar", "salir"


@dataclass(frozen=True)
class Accion:
    """Lo que hace el agente ante una percepción (sección 4.1)."""

    regla: int                      # 0 a 10 (la fila de la tabla; 0 = ninguna)
    tipo: str                       # uno de TIPOS_ACCION
    texto: str                      # lo que se muestra
    estado: Estado                  # el estado después de la acción
    autorizado: bool | None = None  # solo reglas 3 y 7
    identidad: int | None = None    # solo regla 5
    nombre: str | None = None       # solo regla 5: "Identidad Sintética #N"
    distancia: float | None = None  # reglas 3, 5 y 7
    tau: float | None = None        # reglas 3, 5 y 7


def nombres_identidades(cfg: dict) -> dict[int, str]:
    """{id: nombre} de huellas.identidades."""
    return {int(ident["id"]): str(ident["nombre"]) for ident in cfg["huellas"]["identidades"]}


def etiquetas_voz(cfg: dict) -> tuple[str, ...]:
    """Los comandos de voz.comandos y, al final, la clase de rechazo."""
    return tuple(str(c) for c in cfg["voz"]["comandos"]) + (str(cfg["voz"]["clase_rechazo"]),)


def _identidad(valor, validas: dict[int, str]) -> int:
    """La identidad de un resultado de evaluar: 0 o una de huellas.identidades; si no, ValueError."""
    if not isinstance(valor, (int, np.integer)) or isinstance(valor, (bool, np.bool_)):
        raise ValueError(f"evaluar devolvió una identidad que no es un entero: {valor!r}.")
    identidad = int(valor)
    if identidad != 0 and identidad not in validas:
        raise ValueError(f"evaluar devolvió la identidad {identidad}, que no es 0 ni una de {sorted(validas)}.")
    return identidad


class Agente:
    """El agente de la tabla 4.1. `preparar(imagen)` es la percepción de la huella al cargarla (H2; puede
    lanzar ValueError si no es una huella) y `evaluar(preparada)`, la de "Reconocer" (H3 + H4: devuelve
    un objeto con identidad, distancia y tau)."""

    def __init__(self, cfg: dict, preparar: Callable[[np.ndarray], Any], evaluar: Callable[[Any], Any]) -> None:
        self._preparar = preparar
        self._evaluar = evaluar
        self._nombres = nombres_identidades(cfg)
        self._etiquetas = etiquetas_voz(cfg)
        self.estado: Estado = Estado.ESPERANDO_MUESTRA
        self.huella: str = ""
        self.imagen: np.ndarray | None = None
        self.preparada: Any | None = None
        self.resultado: Any | None = None
        self.identificada: bool = False

    # ----------------------------------------------------------------------- percepciones
    def percibir_huella(self, imagen: np.ndarray | None, nombre: str) -> Accion:
        """Reglas 1 (huella legible), 10 (ilegible) y 0 (en FIN)."""
        if self.estado is Estado.FIN:
            return Accion(0, "ninguna", "", Estado.FIN)
        if imagen is None:
            return self._ilegible()
        try:
            preparada = self._preparar(imagen)
        except ValueError:
            return self._ilegible()
        self.huella, self.imagen, self.preparada = str(nombre), imagen, preparada
        self.resultado, self.identificada = None, False
        self.estado = Estado.HUELLA_CARGADA
        return Accion(1, "mostrar_huella", TEXTOS[1], self.estado)

    def percibir_comando(self, comando: str) -> Accion:
        """Reglas 2 a 9 y 0 (en FIN). ValueError si el comando no es una de etiquetas_voz(cfg)."""
        if comando not in self._etiquetas:
            raise ValueError(f"Comando desconocido: {comando!r}. Tiene que ser uno de {self._etiquetas}.")
        if self.estado is Estado.FIN:
            return Accion(0, "ninguna", "", Estado.FIN)
        if comando == SALIR:
            self.estado = Estado.FIN
            return Accion(9, "cerrar", TEXTOS[9], Estado.FIN)
        if comando == RECONOCER:
            return self._reconocer()
        if comando == IDENTIFICAR:
            return self._identificar()
        return self._aviso(8)

    # ----------------------------------------------------------------------- reglas
    def _aviso(self, regla: int) -> Accion:
        return Accion(regla, "aviso", TEXTOS[regla], self.estado)

    def _ilegible(self) -> Accion:
        """Regla 10: se descarta también la huella anterior y su resultado."""
        self.huella, self.imagen, self.preparada = "", None, None
        self.resultado, self.identificada = None, False
        self.estado = Estado.ESPERANDO_MUESTRA
        return self._aviso(10)

    def _reconocer(self) -> Accion:
        """Reglas 2, 3 y 7. Si evaluar lanza, o da una identidad inválida, el estado no cambia."""
        if self.estado is Estado.ESPERANDO_MUESTRA:
            return self._aviso(2)
        regla = 3 if self.estado is Estado.HUELLA_CARGADA else 7
        resultado = self._evaluar(self.preparada)
        identidad = _identidad(resultado.identidad, self._nombres)
        distancia, tau = float(resultado.distancia), float(resultado.tau)
        self.resultado, self.identificada = resultado, False
        self.estado = Estado.EVALUADA_AUTORIZADA if identidad != 0 else Estado.EVALUADA_RECHAZADA
        return Accion(regla, "resultado", AUTORIZADO if identidad != 0 else NO_AUTORIZADO, self.estado,
                      autorizado=identidad != 0, distancia=distancia, tau=tau)

    def _identificar(self) -> Accion:
        """Reglas 2, 4, 5 y 6: la identidad sale solo de una huella autorizada."""
        if self.estado is Estado.ESPERANDO_MUESTRA:
            return self._aviso(2)
        if self.estado is Estado.HUELLA_CARGADA:
            return self._aviso(4)
        if self.estado is Estado.EVALUADA_RECHAZADA:
            return self._aviso(6)
        identidad = int(self.resultado.identidad)
        nombre = self._nombres[identidad]
        self.identificada = True
        return Accion(5, "identidad", nombre, self.estado, identidad=identidad, nombre=nombre,
                      distancia=float(self.resultado.distancia), tau=float(self.resultado.tau))
