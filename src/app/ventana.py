"""Fase 5 · La ventana de la app (CustomTkinter): dibuja la Pantalla que devuelve el controlador.

Contrato: docs/contratos/f5_integracion.md (versión 1.1), secciones 3.4 y 4.7; se verifica a mano
(sección 7). No tiene lógica propia: pasa cada evento al controlador y dibuja la Pantalla. Tres
paneles: 1) la huella (la lista, las variantes e intrusos nuevos y "Abrir archivo…"; cruda y
esqueleto), 2) la voz y el agente (push-to-talk con el botón o la barra espaciadora, medidor de
nivel, estado, cartel e identidad) y 3) el PCA de train con la huella evaluada y una barra con su
distancia contra τ*. Importar este módulo no crea ninguna ventana.
"""

from __future__ import annotations

from tkinter import filedialog

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from ..agente.agente import nombres_identidades
from ..config import RAIZ
from .controlador import Controlador, Pantalla

PERIODO_MS = 30              # cada cuánto se llama a controlador.actualizar()
SOLTAR_MS = 40               # la barra espaciadora se suelta si no vuelve a apretarse en este lapso
ALTO_HUELLA_PX = 240         # cruda (400 × 320) y esqueleto (250 × 250) entran lado a lado en el panel
COLOR_IDENTIDAD = {1: "#1f77b4", 2: "#ff7f0e", 3: "#9467bd"}
VERDE, ROJO, NARANJA = "#1e8e3e", "#c5221f", "#b45309"
NIVEL_MIN_DB = -60.0
AYUDA = ("Comandos: «Reconocer», «Identificar» y «Salir». Mantené apretado el botón (o la barra "
         "espaciadora) mientras hablás, hasta 3 s.")


def _coma(valor: float, decimales: int = 3) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def _a_pil(imagen: np.ndarray) -> Image.Image:
    """Gris, BGR o BGRA de OpenCV, o el esqueleto (bool, crestas en negro sobre blanco)."""
    if imagen.dtype == bool:
        return Image.fromarray(np.where(imagen, 0, 255).astype(np.uint8))
    if imagen.ndim == 3:
        return Image.fromarray(np.ascontiguousarray(imagen[..., 2::-1]))
    return Image.fromarray(imagen)


class Ventana(ctk.CTk):
    def __init__(self, controlador: Controlador) -> None:
        super().__init__()
        self.c = controlador
        self._nombres = nombres_identidades(controlador.cfg)
        self.title("TF IA I 2026 · Control de acceso por huella y voz")
        self.geometry("1500x820")
        self.minsize(1200, 700)
        self.grid_columnconfigure((0, 1, 2), weight=1, uniform="panel")
        self.grid_rowconfigure(0, weight=1)
        self._titulo = ctk.CTkFont(size=18, weight="bold")
        self._huella_dibujada = None
        self._pca_dibujado = None
        self._puesto: dict[int, dict] = {}      # lo último que se le puso a cada widget
        self._soltar_pendiente = None
        self._cerrando = False
        self._panel_huella()
        self._panel_voz()
        self._panel_pca()
        self.bind_all("<KeyPress-space>", self._tecla_apretada)
        self.bind_all("<KeyRelease-space>", self._tecla_soltada)
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

    # ----------------------------------------------------------------------- paneles
    def _panel_huella(self) -> None:
        f = ctk.CTkFrame(self)
        f.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        ctk.CTkLabel(f, text="1 · Huella", font=self._titulo).pack(anchor="w", padx=12, pady=(10, 6))
        nombres = [e.nombre for e in self.c.entradas] or ["(no está el dataset de huellas)"]
        self.lista = ctk.CTkOptionMenu(f, values=nombres, command=self._elegir, dynamic_resizing=False)
        self.lista.set("Elegí una huella del dataset…")
        self.lista.pack(fill="x", padx=12, pady=4)
        fila = ctk.CTkFrame(f, fg_color="transparent")
        fila.pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(fila, text="Variante nueva", command=self._variante_nueva).pack(side="left")
        self.identidad = ctk.CTkSegmentedButton(fila, values=[str(k) for k in self._nombres])
        self.identidad.set(str(next(iter(self._nombres))))
        self.identidad.pack(side="left", padx=8)
        fila = ctk.CTkFrame(f, fg_color="transparent")
        fila.pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(fila, text="Intruso nuevo", command=self._intruso_nuevo).pack(side="left")
        ctk.CTkButton(fila, text="Abrir archivo…", command=self._abrir_archivo).pack(side="left", padx=8)
        self.nombre_huella = ctk.CTkLabel(f, text="", font=ctk.CTkFont(size=15, weight="bold"))
        self.nombre_huella.pack(anchor="w", padx=12, pady=(12, 4))
        imagenes = ctk.CTkFrame(f, fg_color="transparent")
        imagenes.pack(fill="both", expand=True, padx=12, pady=4)
        self.cruda, self.esqueleto = [], []
        for columna, (rotulo, destino) in enumerate((("Cruda", "cruda"), ("Esqueleto (H2)", "esqueleto"))):
            ctk.CTkLabel(imagenes, text=rotulo).grid(row=0, column=columna, padx=4)
            etiqueta = ctk.CTkLabel(imagenes, text="", width=0, height=ALTO_HUELLA_PX)   # ancho: el de la imagen
            etiqueta.grid(row=1, column=columna, padx=4, sticky="n")
            setattr(self, destino, etiqueta)

    def _panel_voz(self) -> None:
        f = ctk.CTkFrame(self)
        f.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        ctk.CTkLabel(f, text="2 · Voz y agente", font=self._titulo).pack(anchor="w", padx=12, pady=(10, 6))
        self.boton_hablar = ctk.CTkButton(f, text="Mantener para hablar", height=70,
                                          font=ctk.CTkFont(size=20, weight="bold"))
        self.boton_hablar.pack(fill="x", padx=12, pady=6)
        self.boton_hablar.bind("<ButtonPress-1>", lambda _e: self._dibujar(self.c.apretar()))
        self.boton_hablar.bind("<ButtonRelease-1>", lambda _e: self._dibujar(self.c.soltar()))
        fila = ctk.CTkFrame(f, fg_color="transparent")
        fila.pack(fill="x", padx=12, pady=2)
        self.grabando = ctk.CTkLabel(fila, text="", text_color=ROJO, width=110, anchor="w",
                                     font=ctk.CTkFont(size=15, weight="bold"))
        self.grabando.pack(side="left")
        self.medidor = ctk.CTkProgressBar(fila)
        self.medidor.set(0)
        self.medidor.pack(side="left", fill="x", expand=True, padx=(4, 0))
        ctk.CTkButton(f, text="Abrir audio…", command=self._abrir_audio, fg_color="gray50").pack(anchor="w", padx=12, pady=6)
        ctk.CTkLabel(f, text=AYUDA, wraplength=420, justify="left", text_color="gray40").pack(anchor="w", padx=12)
        self.estado = ctk.CTkLabel(f, text="", font=ctk.CTkFont(size=14))
        self.estado.pack(anchor="w", padx=12, pady=(14, 4))
        self.cartel = ctk.CTkLabel(f, text="", height=110, corner_radius=12, fg_color="transparent",
                                   text_color="white", font=ctk.CTkFont(size=44, weight="bold"))
        self.cartel.pack(fill="x", padx=12, pady=6)
        self.identificacion = ctk.CTkLabel(f, text="", font=ctk.CTkFont(size=26, weight="bold"))
        self.identificacion.pack(padx=12, pady=6)
        self.mensaje = ctk.CTkLabel(f, text="", wraplength=420, font=ctk.CTkFont(size=16))
        self.mensaje.pack(padx=12, pady=6)
        self.detalle = ctk.CTkLabel(f, text="", wraplength=420, justify="left", text_color="gray45",
                                    font=ctk.CTkFont(size=12))
        self.detalle.pack(anchor="w", padx=12, pady=(4, 10))

    def _panel_pca(self) -> None:
        f = ctk.CTkFrame(self)
        f.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        ctk.CTkLabel(f, text="3 · PCA de entrenamiento", font=self._titulo).pack(anchor="w", padx=12, pady=(10, 6))
        self.figura = Figure(figsize=(5, 6.2), dpi=100, layout="constrained")
        rejilla = self.figura.add_gridspec(2, 1, height_ratios=[5, 1.2])
        self.ax_pca = self.figura.add_subplot(rejilla[0])
        self.ax_barra = self.figura.add_subplot(rejilla[1])
        self.lienzo = FigureCanvasTkAgg(self.figura, master=f)
        self.lienzo.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ----------------------------------------------------------------------- eventos
    def _elegir(self, nombre: str) -> None:
        for entrada in self.c.entradas:
            if entrada.nombre == nombre:
                self._dibujar(self.c.elegir(entrada))
                return

    def _generando(self) -> None:
        self._poner(self.mensaje, text="Generando…", text_color=("gray10", "gray90"))
        self.update_idletasks()

    def _variante_nueva(self) -> None:
        self._generando()
        self._dibujar(self.c.variante_nueva(int(self.identidad.get())))

    def _intruso_nuevo(self) -> None:
        self._generando()
        self._dibujar(self.c.intruso_nuevo())

    def _abrir_archivo(self) -> None:
        ruta = filedialog.askopenfilename(title="Abrir una huella", initialdir=RAIZ / self.c.cfg["rutas"]["huellas"],
                                          filetypes=[("Imágenes", "*.png *.jpg *.bmp *.tif"), ("Todos", "*.*")])
        if ruta:
            self._dibujar(self.c.abrir_archivo(ruta))

    def _abrir_audio(self) -> None:
        ruta = filedialog.askopenfilename(title="Abrir un audio con un comando", initialdir=RAIZ / self.c.cfg["rutas"]["voz"],
                                          filetypes=[("Audio", "*.wav *.ogg *.opus *.mp3 *.m4a"), ("Todos", "*.*")])
        if ruta:
            self._dibujar(self.c.comando_wav(ruta))

    def _tecla_apretada(self, _evento=None) -> None:
        # Con la repetición de teclas llegan varias pulsaciones (y, en Linux, sueltas falsas en el medio).
        if self._soltar_pendiente is not None:
            self.after_cancel(self._soltar_pendiente)
            self._soltar_pendiente = None
            return
        self._dibujar(self.c.apretar())

    def _tecla_soltada(self, _evento=None) -> None:
        if self._soltar_pendiente is None:
            self._soltar_pendiente = self.after(SOLTAR_MS, self._soltar_tecla)

    def _soltar_tecla(self) -> None:
        self._soltar_pendiente = None
        self._dibujar(self.c.soltar())

    def _tick(self) -> None:
        if self._cerrando:
            return
        self._dibujar(self.c.actualizar())
        if not self._cerrando:
            self.after(PERIODO_MS, self._tick)

    def _cerrar(self) -> None:
        self._cerrando = True
        self.c.cerrar()
        self.destroy()

    # ----------------------------------------------------------------------- dibujo
    def _poner(self, widget, **opciones) -> None:
        """configure solo si algo cambió: _dibujar corre cada 30 ms y redibujar todo hace parpadear."""
        if self._puesto.get(id(widget)) != opciones:
            self._puesto[id(widget)] = opciones
            widget.configure(**opciones)

    def _dibujar(self, p: Pantalla) -> None:
        if self._cerrando:
            return
        self._dibujar_huella(p)
        self._poner(self.grabando, text="● Grabando" if p.grabando else ("" if p.microfono else "Sin micrófono"))
        self.medidor.set(0 if p.nivel_db is None else min(1.0, max(0.0, 1 - p.nivel_db / NIVEL_MIN_DB)))
        self._poner(self.estado, text=f"Estado del agente: {p.estado}")
        self._poner(self.cartel, text=p.cartel, fg_color={"verde": VERDE, "rojo": ROJO}.get(p.color, "transparent"))
        self._poner(self.identificacion, text=p.identificacion)
        repetido = p.mensaje in (p.cartel, p.identificacion)   # las reglas 3, 5 y 7 ya están en el cartel
        self._poner(self.mensaje, text="" if repetido else p.mensaje,
                    text_color=NARANJA if p.aviso else ("gray10", "gray90"))
        self._poner(self.detalle, text=f"Último audio: {p.detalle_voz}" if p.detalle_voz else "")
        self._dibujar_pca(p)
        if p.cerrar:
            self._cerrando = True
            self.after(600, self.destroy)

    def _dibujar_huella(self, p: Pantalla) -> None:
        clave = (p.huella, id(p.imagen), id(p.esqueleto))
        if clave == self._huella_dibujada:
            return
        self._huella_dibujada = clave
        self.nombre_huella.configure(text=p.huella or "Sin huella cargada")
        for etiqueta, imagen in ((self.cruda, p.imagen), (self.esqueleto, p.esqueleto)):
            if imagen is None:
                etiqueta.configure(image=None, text="")
                etiqueta.imagen = None
                continue
            alto, ancho = imagen.shape[:2]
            tamano = (max(1, round(ALTO_HUELLA_PX * ancho / alto)), ALTO_HUELLA_PX)
            foto = ctk.CTkImage(light_image=_a_pil(imagen), dark_image=_a_pil(imagen), size=tamano)
            etiqueta.configure(image=foto, text="")
            etiqueta.imagen = foto           # que no se la lleve el recolector de basura

    def _dibujar_pca(self, p: Pantalla) -> None:
        clave = (p.punto_pca, p.distancia, p.tau, p.color)
        if clave == self._pca_dibujado:
            return
        self._pca_dibujado = clave
        visor = self.c.visor
        ax, barra = self.ax_pca, self.ax_barra
        ax.clear()
        barra.clear()
        if visor is None:
            ax.text(0.5, 0.5, "Sin datos para el PCA", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            ax.set_axis_on()
            for k, nombre in self._nombres.items():
                puntos = visor.puntos[visor.identidades == k]
                ax.scatter(puntos[:, 0], puntos[:, 1], s=22, alpha=0.75, color=COLOR_IDENTIDAD.get(k), label=nombre)
            ax.scatter(visor.centroides[:, 0], visor.centroides[:, 1], marker="X", s=140, color="black",
                       label="Centroides (K-means)")
            if p.punto_pca is not None:
                color = VERDE if p.color == "verde" else ROJO
                ax.scatter([p.punto_pca[0]], [p.punto_pca[1]], marker="*", s=420, color=color,
                           edgecolors="black", linewidths=0.8, zorder=5, label="Huella evaluada")
            ax.set_xlabel(f"PC1 ({_coma(100 * visor.proporcion[0], 1)} %)")
            ax.set_ylabel(f"PC2 ({_coma(100 * visor.proporcion[1], 1)} %)")
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.3)
        if p.distancia is None or p.tau is None:
            barra.text(0.5, 0.5, "Decí «Reconocer» para ver la distancia contra τ*", ha="center", va="center",
                       transform=barra.transAxes, fontsize=9)
            barra.set_axis_off()
        else:
            barra.set_axis_on()
            color = VERDE if p.color == "verde" else ROJO
            barra.barh([0], [p.distancia], color=color, height=0.6)
            barra.axvline(p.tau, color="black", linestyle="--")
            barra.set_xlim(0, 1.25 * max(p.distancia, p.tau))
            barra.set_yticks([])
            barra.set_title(f"distancia {_coma(p.distancia)} contra τ* = {_coma(p.tau)}", fontsize=10)
        self.lienzo.draw_idle()


def ejecutar(controlador: Controlador) -> None:
    """Crea la ventana, abre el micrófono (si hay) y corre el mainloop; al salir, lo libera."""
    ctk.set_appearance_mode("light")
    ventana = Ventana(controlador)
    try:
        if controlador.microfono is not None:
            ventana._dibujar(controlador.abrir_microfono())
        else:
            ventana._dibujar(controlador.pantalla())
        ventana.after(PERIODO_MS, ventana._tick)
        ventana.mainloop()
    finally:
        controlador.cerrar()
