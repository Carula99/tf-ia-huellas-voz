// Helpers comunes para armar los dos Word del TF (docx-js 9.x)
const fs = require("fs");
const {
  Document, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, ImageRun, Header, Footer, PageNumber,
  VerticalAlign, TableLayoutType, LineRuleType,
} = require("docx");
const AUTO = LineRuleType.AUTO;

const C = {
  text: "1F2933", muted: "5F6B7A", h1: "1F3A5F", h2: "2F6DB5", h3: "1F3A5F",
  headFill: "DCE6F2", border: "B7C3D0", rowAlt: "F7F9FC",
  calloutFill: "EEF4FB", calloutBorder: "2F6DB5",
  warnFill: "FFF6E8", warnBorder: "C07A12",
  codeFill: "F4F5F7", codeBorder: "D5D9DE",
};
const FONT = "Calibri";
const MONO = "Consolas";
const CONTENT_W = 9070; // A4 (11906) - 2 x 1418 de margen

let listInstance = 0;

// Marcado mínimo dentro de un texto: **negrita**, *cursiva*, `código`
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), ...base, bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), ...base, font: MONO, size: 19 }));
    else out.push(new TextRun({ text: t.slice(1, -1), ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const p = (text, o = {}) => new Paragraph({
  children: runs(text, o.run || {}),
  spacing: { after: o.after ?? 120, before: o.before ?? 0, line: 276, lineRule: AUTO },
  alignment: o.align, keepNext: o.keepNext, keepLines: o.keepLines,
  indent: o.indent,
});

const h1 = (t, o = {}) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: o.pageBreak, keepNext: true });
const h2 = (t, o = {}) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], pageBreakBefore: o.pageBreak, keepNext: true });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)], keepNext: true });

// Etiqueta corta en color (Objetivo, Pasos, ...)
const label = (t) => new Paragraph({
  children: [new TextRun({ text: t.toUpperCase(), bold: true, size: 18, color: C.h2, characterSpacing: 20 })],
  spacing: { before: 140, after: 60 }, keepNext: true,
});

function bullets(items, ref = "bullets") {
  const out = [];
  for (const it of items) {
    const [txt, sub] = Array.isArray(it) ? it : [it, null];
    out.push(new Paragraph({ children: runs(txt), numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 264, lineRule: AUTO } }));
    if (sub) for (const s of sub) out.push(new Paragraph({ children: runs(s), numbering: { reference: ref, level: 1 }, spacing: { after: 40, line: 264, lineRule: AUTO } }));
  }
  return out;
}
const checks = (items) => bullets(items, "checks");

function steps(items) {
  const inst = ++listInstance;
  return items.map((t) => new Paragraph({
    children: runs(t), numbering: { reference: "steps", level: 0, instance: inst }, spacing: { after: 60, line: 264, lineRule: AUTO },
  }));
}

function boxPara(text, fill, border, o = {}) {
  return new Paragraph({
    children: runs(text, o.run || {}),
    shading: { type: ShadingType.CLEAR, fill, color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: border, space: 8 } },
    indent: { left: 160, right: 160 },
    spacing: { before: o.before ?? 0, after: o.after ?? 0, line: 264, lineRule: AUTO },
    keepNext: o.keepNext, keepLines: true,
  });
}
// Recuadro de nota: lista de párrafos (el primero puede ir en negrita como título)
function callout(lines, kind = "info") {
  const fill = kind === "warn" ? C.warnFill : C.calloutFill;
  const border = kind === "warn" ? C.warnBorder : C.calloutBorder;
  return lines.map((t, i) => boxPara(t, fill, border, {
    before: i === 0 ? 60 : 0, after: i === lines.length - 1 ? 160 : 40, keepNext: i < lines.length - 1,
  }));
}

function code(lines, o = {}) {
  return lines.map((t, i) => new Paragraph({
    children: [new TextRun({ text: t.length ? t : " ", font: MONO, size: o.size ?? 17, color: "2B2B2B" })],
    shading: { type: ShadingType.CLEAR, fill: C.codeFill, color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 6, color: C.codeBorder, space: 8 } },
    indent: { left: 160, right: 160 },
    spacing: { before: i === 0 ? 60 : 0, after: i === lines.length - 1 ? 160 : 0, line: 240, lineRule: AUTO },
    keepNext: i < lines.length - 1 && !o.allowBreak, keepLines: true,
  }));
}

function cellParas(content, o = {}) {
  const arr = Array.isArray(content) ? content : [content];
  return arr.map((t, i) => {
    if (t instanceof Paragraph) return t;
    const isBullet = typeof t === "string" && t.startsWith("- ");
    const txt = isBullet ? t.slice(2) : t;
    return new Paragraph({
      children: runs(txt, { size: o.size ?? 19, bold: o.bold, color: o.color }),
      numbering: isBullet ? { reference: "cellbullets", level: 0 } : undefined,
      spacing: { after: i === arr.length - 1 ? 0 : 50, line: 252, lineRule: AUTO },
      alignment: o.align, keepNext: o.keepNext,
    });
  });
}

// headers: array de títulos; rows: array de arrays (string o array de strings por celda)
function table(headers, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const brd = { style: BorderStyle.SINGLE, size: 4, color: C.border };
  const borders = { top: brd, bottom: brd, left: brd, right: brd, insideHorizontal: brd, insideVertical: brd };
  const margins = { top: 70, bottom: 70, left: 100, right: 100 };
  const head = new TableRow({
    tableHeader: true, cantSplit: true,
    children: headers.map((h, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: C.headFill, color: "auto" },
      verticalAlign: VerticalAlign.CENTER, margins,
      children: cellParas(h, { bold: true, color: C.h1, size: 19 }),
    })),
  });
  const body = rows.map((r, ri) => new TableRow({
    cantSplit: o.cantSplit ?? true,
    children: r.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: (o.zebra && ri % 2 === 1) ? { type: ShadingType.CLEAR, fill: C.rowAlt, color: "auto" }
        : (o.firstColFill && i === 0 ? { type: ShadingType.CLEAR, fill: o.firstColFill, color: "auto" } : undefined),
      margins, verticalAlign: o.vcenter ? VerticalAlign.CENTER : VerticalAlign.TOP,
      children: cellParas(c, { bold: o.boldFirstCol && i === 0, size: o.size, keepNext: o.keepTogether && ri < rows.length - 1 }),
    })),
  }));
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    layout: TableLayoutType.FIXED, borders, rows: headers ? [head, ...body] : body,
  });
}

// Tabla de 2 columnas sin encabezado (ficha): rows = [[campo, contenido], ...]
function ficha(rows, widths = [2300, CONTENT_W - 2300]) {
  const brd = { style: BorderStyle.SINGLE, size: 4, color: C.border };
  const borders = { top: brd, bottom: brd, left: brd, right: brd, insideHorizontal: brd, insideVertical: brd };
  const margins = { top: 70, bottom: 70, left: 100, right: 100 };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, borders,
    rows: rows.map(([k, v], ri) => new TableRow({
      cantSplit: true,
      children: [
        new TableCell({ width: { size: widths[0], type: WidthType.DXA }, margins,
          shading: { type: ShadingType.CLEAR, fill: C.headFill, color: "auto" },
          children: cellParas(k, { bold: true, color: C.h1, keepNext: ri < rows.length - 1 }) }),
        new TableCell({ width: { size: widths[1], type: WidthType.DXA }, margins, children: cellParas(v, { keepNext: ri < rows.length - 1 }) }),
      ],
    })),
  });
}

function pngSize(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function image(path, widthCm) {
  const { w, h, data } = pngSize(path);
  const wpx = Math.round(widthCm / 2.54 * 96);
  const hpx = Math.round(wpx * h / w);
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new ImageRun({ type: "png", data, transformation: { width: wpx, height: hpx } })],
  });
}
const caption = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: runs(t, { italics: true, size: 18, color: C.muted }),
});

function titleBlock({ kicker, title, subtitle, meta }) {
  return [
    new Paragraph({ spacing: { before: 0, after: 80 }, children: [new TextRun({ text: kicker, bold: true, size: 18, color: C.h2, characterSpacing: 30 })] }),
    new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: title, bold: true, size: 52, color: C.h1 })] }),
    new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: subtitle, size: 26, color: C.muted })] }),
    ...meta.map((m, i) => new Paragraph({
      spacing: { after: i === meta.length - 1 ? 240 : 40 },
      border: i === meta.length - 1 ? { bottom: { style: BorderStyle.SINGLE, size: 8, color: C.h2, space: 8 } } : undefined,
      children: runs(m, { size: 19, color: C.muted }),
    })),
  ];
}

function makeDoc({ title, description, headerText, children }) {
  return new Document({
    creator: "Santiago", title, description,
    styles: {
      default: {
        document: { run: { font: FONT, size: 21, color: C.text, language: { value: "es-AR" } }, paragraph: { spacing: { line: 276, lineRule: AUTO } } },
      },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 32, bold: true, color: C.h1 },
          paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 26, bold: true, color: C.h2 },
          paragraph: { spacing: { before: 280, after: 100 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 22, bold: true, color: C.h3 },
          paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 240 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 800, hanging: 240 } } } },
        ] },
        { reference: "cellbullets", levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 220, hanging: 180 } } } },
        ] },
        { reference: "checks", levels: [
          { level: 0, format: LevelFormat.BULLET, text: "☐", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 420, hanging: 300 } }, run: { font: "Segoe UI Symbol" } } },
          { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 800, hanging: 240 } } } },
        ] },
        { reference: "steps", levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 300 } } } },
        ] },
      ],
    },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1418, right: 1418, header: 567, footer: 567 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: headerText, size: 16, color: C.muted })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Página ", PageNumber.CURRENT, " de ", PageNumber.TOTAL_PAGES], size: 16, color: C.muted })] })] }) },
      children,
    }],
  });
}

module.exports = { C, CONTENT_W, runs, p, h1, h2, h3, label, bullets, checks, steps, callout, code, table, ficha, image, caption, titleBlock, makeDoc };
