"""Arma el informe en Word (y en PDF, si está LibreOffice) a partir de docs/informe/informe.md.

    python docs/informe/armar.py [carpeta de salida]

Corre pandoc con la plantilla de estilos referencia.docx (ver estilo_referencia.py) y después
retoca el Word: las tablas con letra de 9,5 pt, la fila de encabezado repetida en cada página
y las filas sin cortar entre páginas; y en las fórmulas, los paréntesis escritos explícitamente,
porque LibreOffice dibuja como barras los delimitadores que no los traen. Por defecto escribe
informe.docx e informe.pdf en docs/informe/salida/ (fuera de Git). Necesita pandoc (3.1 o
más nuevo) y lxml; el PDF, LibreOffice (soffice) con su módulo de fórmulas.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from lxml import etree

CARPETA = Path(__file__).resolve().parent
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
TAM_TABLA = "19"  # medios puntos: 9,5 pt


def _w(etiqueta: str) -> str:
    return f"{{{W}}}{etiqueta}"


def _m(etiqueta: str) -> str:
    return f"{{{M}}}{etiqueta}"


def _hijo(padre, etiqueta: str, primero: bool = False):
    el = padre.find(etiqueta)
    if el is None:
        el = etree.Element(etiqueta)
        if primero:
            padre.insert(0, el)
        else:
            padre.append(el)
    return el


def _retocar_tablas(raiz) -> int:
    tablas = raiz.iter(_w("tbl"))
    n = 0
    for tabla in tablas:
        n += 1
        for i, fila in enumerate(tabla.iter(_w("tr"))):
            trpr = _hijo(fila, _w("trPr"), primero=True)
            _hijo(trpr, _w("cantSplit"))
            if i == 0:
                _hijo(trpr, _w("tblHeader"))
        for corrida in tabla.iter(_w("r")):
            rpr = _hijo(corrida, _w("rPr"), primero=True)
            for etiqueta in ("sz", "szCs"):
                _hijo(rpr, _w(etiqueta)).set(_w("val"), TAM_TABLA)
    return n


def _corrida(texto: str, recta: bool = True):
    r = etree.Element(_m("r"))
    if recta:
        rpr = etree.SubElement(r, _m("rPr"))
        etree.SubElement(rpr, _m("sty")).set(_m("val"), "p")
    etree.SubElement(r, _m("t")).text = texto
    return r


def _retocar_formulas(raiz) -> int:
    """Paréntesis y barras alrededor de algo sin fracciones: como caracteres comunes.

    LibreOffice dibuja como una barra el paréntesis que crece (m:d) cuando lo que encierra es de
    una sola línea; Word los muestra igual de las dos formas. Con una fracción adentro, se deja el
    que crece. Y la norma va con la doble barra (U+2016): LibreOffice dibuja la de pandoc
    (U+2225, «paralelo») inclinada.
    """
    n = 0
    for delim in list(raiz.iter(_m("d"))):
        dpr = delim.find(_m("dPr"))
        abre = dpr.find(_m("begChr")).get(_m("val")) if dpr is not None and dpr.find(_m("begChr")) is not None else "("
        cierra = dpr.find(_m("endChr")).get(_m("val")) if dpr is not None and dpr.find(_m("endChr")) is not None else ")"
        partes = delim.findall(_m("e"))
        if abre not in "(|" or len(partes) != 1 or partes[0].find(".//" + _m("f")) is not None:
            continue
        padre = delim.getparent()
        i = padre.index(delim)
        nuevos = [_corrida(abre), *list(partes[0]), _corrida(cierra)]
        padre.remove(delim)
        for j, el in enumerate(nuevos):
            padre.insert(i + j, el)
        n += 1
    for t in raiz.iter(_m("t")):
        if t.text and "∥" in t.text:
            t.text = t.text.replace("∥", "‖")
    return n


def retocar(docx: Path) -> None:
    temporal = docx.with_suffix(".tmp")
    with zipfile.ZipFile(docx) as entrada, zipfile.ZipFile(temporal, "w", zipfile.ZIP_DEFLATED) as salida:
        for item in entrada.infolist():
            datos = entrada.read(item.filename)
            if item.filename == "word/document.xml":
                raiz = etree.fromstring(datos)
                tablas = _retocar_tablas(raiz)
                delimitadores = _retocar_formulas(raiz)
                datos = etree.tostring(raiz, xml_declaration=True, encoding="UTF-8", standalone=True)
                print(f"{tablas} tablas retocadas; {delimitadores} paréntesis de fórmulas pasados a caracteres comunes")
            salida.writestr(item, datos)
    temporal.replace(docx)


def armar(salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    docx = salida / "informe.docx"
    subprocess.run(
        ["pandoc", "informe.md", "-f", "markdown", "-t", "docx",
         "--reference-doc=referencia.docx", "-o", str(docx)],
        cwd=CARPETA, check=True,
    )
    retocar(docx)
    print(f"Word: {docx}")
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        print("Sin LibreOffice: el PDF se exporta desde Word.")
        return
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(salida), str(docx)],
                   check=True, capture_output=True)
    print(f"PDF: {salida / 'informe.pdf'}")


if __name__ == "__main__":
    armar(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else CARPETA / "salida")
