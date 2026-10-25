"""Arma docs/informe/referencia.docx, la plantilla de estilos que usa pandoc para el informe.

Parte de la plantilla de pandoc y cambia los estilos: A4 con márgenes de 2,5 cm, Calibri,
títulos en azul oscuro, texto justificado, tablas con bordes y la fila de encabezado sombreada,
epígrafes chicos y el código en Consolas. Se corre una vez (o al cambiar algo del estilo):

    pandoc -o plantilla_pandoc.docx --print-default-data-file reference.docx
    python docs/informe/estilo_referencia.py plantilla_pandoc.docx docs/informe/referencia.docx

Necesita python-docx (no está en requirements.txt: es solo para armar el informe).
"""

from __future__ import annotations

import sys

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

AZUL = RGBColor(0x1F, 0x38, 0x64)
GRIS = RGBColor(0x40, 0x40, 0x40)
LETRA = "Calibri"
MONO = "Consolas"


def _letra(estilo, nombre: str, tam: float | None = None, color=None, negrita=None, cursiva=None) -> None:
    f = estilo.font
    f.name = nombre
    rpr = estilo.element.get_or_add_rPr()
    fuentes = rpr.find(qn("w:rFonts"))
    if fuentes is None:
        fuentes = OxmlElement("w:rFonts")
        rpr.append(fuentes)
    for clave in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fuentes.set(qn(clave), nombre)
    for clave in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if fuentes.get(qn(clave)) is not None:
            del fuentes.attrib[qn(clave)]
    if tam is not None:
        f.size = Pt(tam)
    if color is not None:
        f.color.rgb = color
    if negrita is not None:
        f.bold = negrita
    if cursiva is not None:
        f.italic = cursiva


def _parrafo(estilo, antes=None, despues=None, interlineado=None, alineacion=None, izquierda=None) -> None:
    pf = estilo.paragraph_format
    if antes is not None:
        pf.space_before = Pt(antes)
    if despues is not None:
        pf.space_after = Pt(despues)
    if interlineado is not None:
        pf.line_spacing = interlineado
    if alineacion is not None:
        pf.alignment = alineacion
    if izquierda is not None:
        pf.left_indent = Cm(izquierda)


def _bordes_tabla(estilo) -> None:
    """Bordes finos grises en toda la tabla y la primera fila sombreada y en negrita."""
    el = estilo.element
    tblpr = el.find(qn("w:tblPr"))
    if tblpr is None:
        tblpr = OxmlElement("w:tblPr")
        el.append(tblpr)
    viejo = tblpr.find(qn("w:tblBorders"))
    if viejo is not None:
        tblpr.remove(viejo)
    bordes = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{lado}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "A6A6A6")
        bordes.append(b)
    tblpr.append(bordes)
    margenes = OxmlElement("w:tblCellMar")
    for lado, valor in (("left", "80"), ("right", "80"), ("top", "30"), ("bottom", "30")):
        m = OxmlElement(f"w:{lado}")
        m.set(qn("w:w"), valor)
        m.set(qn("w:type"), "dxa")
        margenes.append(m)
    tblpr.append(margenes)

    for viejo in el.findall(qn("w:tblStylePr")):
        el.remove(viejo)
    primera = OxmlElement("w:tblStylePr")
    primera.set(qn("w:type"), "firstRow")
    rpr = OxmlElement("w:rPr")
    rpr.append(OxmlElement("w:b"))
    primera.append(rpr)
    tcpr = OxmlElement("w:tcPr")
    sombra = OxmlElement("w:shd")
    sombra.set(qn("w:val"), "clear")
    sombra.set(qn("w:color"), "auto")
    sombra.set(qn("w:fill"), "E8EEF6")
    tcpr.append(sombra)
    primera.append(tcpr)
    el.append(primera)


class _Estilos:
    """Busca los estilos por su id: python-docx traduce "Heading 1" a "heading 1" y no encuentra el de pandoc."""

    def __init__(self, doc) -> None:
        self._doc = doc
        self._por_id = {e.style_id: e for e in doc.styles}

    def crear(self, nombre: str, tipo=WD_STYLE_TYPE.PARAGRAPH):
        """El estilo si ya está; si no (pandoc crea algunos al vuelo), uno nuevo basado en Normal."""
        if nombre not in self:
            nuevo = self._doc.styles.add_style(nombre, tipo)
            if tipo == WD_STYLE_TYPE.PARAGRAPH:
                nuevo.base_style = self["Normal"]
            self._por_id[nuevo.style_id] = nuevo
        return self[nombre]

    def __getitem__(self, nombre: str):
        return self._por_id[nombre.replace(" ", "")]

    def __contains__(self, nombre: str) -> bool:
        return nombre.replace(" ", "") in self._por_id


def armar(entrada: str, salida: str) -> None:
    doc = Document(entrada)
    for seccion in doc.sections:
        seccion.page_width, seccion.page_height = Cm(21.0), Cm(29.7)
        seccion.left_margin = seccion.right_margin = Cm(2.5)
        seccion.top_margin = seccion.bottom_margin = Cm(2.5)

    est = _Estilos(doc)
    for nombre in ("Normal", "Body Text", "First Paragraph", "Compact"):
        _letra(est[nombre], LETRA, 11)
    for nombre in ("Body Text", "First Paragraph"):
        _parrafo(est[nombre], antes=0, despues=6, interlineado=1.15, alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY)
    _parrafo(est["Compact"], antes=0, despues=2, interlineado=1.1)

    _letra(est["Title"], LETRA, 20, AZUL, negrita=True)
    _parrafo(est["Title"], antes=60, despues=18, alineacion=WD_ALIGN_PARAGRAPH.CENTER)
    for nombre in ("Author", "Date"):
        _letra(est[nombre], LETRA, 11.5, GRIS)
        _parrafo(est[nombre], antes=4, despues=4, alineacion=WD_ALIGN_PARAGRAPH.CENTER)

    for nivel, tam, antes in ((1, 15, 18), (2, 12.5, 12), (3, 11.5, 10)):
        h = est[f"Heading {nivel}"]
        _letra(h, LETRA, tam, AZUL, negrita=True, cursiva=False)
        _parrafo(h, antes=antes, despues=6)
        h.paragraph_format.keep_with_next = True

    for nombre in ("Table Caption", "Image Caption"):
        _letra(est[nombre], LETRA, 9.5, GRIS, cursiva=False)
        _parrafo(est[nombre], antes=4, despues=10, alineacion=WD_ALIGN_PARAGRAPH.JUSTIFY)
    est["Table Caption"].paragraph_format.keep_with_next = True
    for nombre in ("Figure", "Captioned Figure"):
        if nombre in est:
            _parrafo(est[nombre], antes=6, despues=0, alineacion=WD_ALIGN_PARAGRAPH.CENTER)
            est[nombre].paragraph_format.keep_with_next = True

    _letra(est.crear("Source Code"), MONO, 8.5)
    _letra(est.crear("Verbatim Char", WD_STYLE_TYPE.CHARACTER), MONO, 9.5)
    _parrafo(est["Source Code"], antes=4, despues=8, interlineado=1.0)
    _letra(est["Block Text"], LETRA, 10, GRIS)
    _parrafo(est["Block Text"], antes=2, despues=6, izquierda=0.8, alineacion=WD_ALIGN_PARAGRAPH.LEFT)

    _bordes_tabla(est["Table"])
    doc.save(salida)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("uso: python estilo_referencia.py <plantilla de pandoc> <referencia.docx>")
    armar(sys.argv[1], sys.argv[2])
