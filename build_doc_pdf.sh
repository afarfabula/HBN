#!/usr/bin/env bash
set -euo pipefail

in_path="${1:-doc.md}"
out_path="${2:-doc.pdf}"

python3 - "$in_path" "$out_path" <<'PY'
import os
import sys
from xml.sax.saxutils import escape

import mistune
from mistune.plugins import table as mistune_table
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable
from svglib.svglib import svg2rlg


def _register_main_font():
    candidates = [
        ("NotoSansCJK", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ("WenQuanYiMicroHei", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        ("WenQuanYiZenHei", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for name, path in candidates:
        try:
            if os.path.isfile(path):
                pdfmetrics.registerFont(TTFont(name, path))
                return name
        except Exception:
            pass

    fallback = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(fallback))
    return fallback


MAIN_FONT = _register_main_font()
MONO_FONT = "Courier"


class SvgFlowable(Flowable):
    def __init__(self, drawing, max_width):
        super().__init__()
        self.drawing = drawing
        self.max_width = max_width
        self._scale = 1.0

    def wrap(self, availWidth, availHeight):
        if not getattr(self.drawing, "width", None):
            return 0, 0
        width = min(self.max_width, availWidth)
        self._scale = width / self.drawing.width
        return width, self.drawing.height * self._scale

    def draw(self):
        self.canv.saveState()
        self.canv.scale(self._scale, self._scale)
        renderPDF.draw(self.drawing, self.canv, 0, 0)
        self.canv.restoreState()


def _text_from_inlines(inlines):
    out = []
    for n in inlines or []:
        t = n.get("type")
        if t == "text":
            out.append(escape(n.get("raw", "")))
        elif t == "softbreak":
            out.append(" ")
        elif t == "linebreak":
            out.append("<br/>")
        elif t == "codespan":
            out.append('<font face="{}">{}</font>'.format(MONO_FONT, escape(n.get("raw", ""))))
        elif t == "strong":
            out.append("<b>{}</b>".format(_text_from_inlines(n.get("children", []))))
        elif t == "emphasis":
            out.append("<i>{}</i>".format(_text_from_inlines(n.get("children", []))))
        elif t == "link":
            out.append(_text_from_inlines(n.get("children", [])))
        elif t == "image":
            out.append(escape(n.get("attrs", {}).get("url", "")))
        else:
            if "raw" in n:
                out.append(escape(str(n.get("raw", ""))))
            elif "children" in n:
                out.append(_text_from_inlines(n.get("children", [])))
    return "".join(out).strip()


def _resolve_image_path(md_dir, url):
    if not url:
        return ""
    if "://" in url:
        return ""
    if url.startswith("file://"):
        return url[len("file://") :]
    return os.path.normpath(os.path.join(md_dir, url))


def _build_table(node, styles, available_width):
    head = None
    body = None
    for c in node.get("children", []):
        if c.get("type") == "table_head":
            head = c
        elif c.get("type") == "table_body":
            body = c

    rows = []
    if head:
        r = []
        for cell in head.get("children", []):
            txt = _text_from_inlines(cell.get("children", []))
            r.append(Paragraph(txt, styles["table_header"]))
        rows.append(r)

    if body:
        for row in body.get("children", []):
            if row.get("type") != "table_row":
                continue
            r = []
            for cell in row.get("children", []):
                txt = _text_from_inlines(cell.get("children", []))
                r.append(Paragraph(txt, styles["table_cell"]))
            rows.append(r)

    if not rows:
        return None

    col_count = max(len(r) for r in rows)
    norm_rows = [r + [""] * (col_count - len(r)) for r in rows]
    col_widths = [available_width / col_count] * col_count

    tbl = Table(norm_rows, colWidths=col_widths, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def _build_list(node, styles):
    ordered = bool(node.get("attrs", {}).get("ordered"))
    items = []
    for li in node.get("children", []):
        if li.get("type") != "list_item":
            continue
        blocks = []
        for b in li.get("children", []):
            bt = b.get("type")
            if bt in ("block_text", "paragraph"):
                blocks.append(Paragraph(_text_from_inlines(b.get("children", [])), styles["normal"]))
            elif bt == "list":
                sub = _build_list(b, styles)
                if sub is not None:
                    blocks.append(sub)
            elif bt == "block_code":
                blocks.append(Preformatted(b.get("raw", "").rstrip("\n"), styles["code_block"]))
        if not blocks:
            blocks.append(Paragraph("", styles["normal"]))
        items.append(ListItem(blocks))

    bullet_type = "1" if ordered else "bullet"
    return ListFlowable(items, bulletType=bullet_type, leftIndent=18, bulletFontName=MAIN_FONT)


def _nodes_to_flowables(nodes, styles, md_dir, available_width):
    flowables = []
    for node in nodes or []:
        t = node.get("type")

        if t == "blank_line":
            flowables.append(Spacer(1, 0.2 * cm))
            continue

        if t == "heading":
            level = int(node.get("attrs", {}).get("level", 1))
            level = max(1, min(level, 6))
            txt = _text_from_inlines(node.get("children", []))
            flowables.append(Paragraph(txt, styles[f"h{level}"]))
            flowables.append(Spacer(1, 0.25 * cm))
            continue

        if t == "paragraph":
            children = node.get("children", [])
            if (
                len(children) == 1
                and isinstance(children[0], dict)
                and children[0].get("type") == "image"
            ):
                img = children[0]
                url = img.get("attrs", {}).get("url", "")
                alt = _text_from_inlines(img.get("children", []))
                resolved = _resolve_image_path(md_dir, url)
                if resolved and os.path.isfile(resolved) and resolved.lower().endswith(".svg"):
                    try:
                        drawing = svg2rlg(resolved)
                        flowables.append(SvgFlowable(drawing, max_width=available_width))
                        if alt:
                            flowables.append(Spacer(1, 0.15 * cm))
                            flowables.append(Paragraph(escape(alt), styles["caption"]))
                        flowables.append(Spacer(1, 0.3 * cm))
                        continue
                    except Exception:
                        pass

                fallback = alt or url
                if fallback:
                    flowables.append(Paragraph(escape(fallback), styles["caption"]))
                    flowables.append(Spacer(1, 0.2 * cm))
                continue

            txt = _text_from_inlines(children)
            if txt:
                flowables.append(Paragraph(txt, styles["normal"]))
                flowables.append(Spacer(1, 0.2 * cm))
            continue

        if t == "list":
            lst = _build_list(node, styles)
            if lst is not None:
                flowables.append(lst)
                flowables.append(Spacer(1, 0.2 * cm))
            continue

        if t == "table":
            tbl = _build_table(node, styles, available_width)
            if tbl is not None:
                flowables.append(tbl)
                flowables.append(Spacer(1, 0.3 * cm))
            continue

        if t == "block_code":
            raw = (node.get("raw") or "").rstrip("\n")
            flowables.append(Preformatted(raw, styles["code_block"]))
            flowables.append(Spacer(1, 0.2 * cm))
            continue

    return flowables


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: build_doc_pdf.sh <in.md> <out.pdf>")

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    md_dir = os.path.dirname(os.path.abspath(in_path))
    with open(in_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    md = mistune.create_markdown(renderer="ast", plugins=[mistune_table.table])
    ast = md(md_text)

    base_styles = getSampleStyleSheet()
    styles = {}
    styles["normal"] = ParagraphStyle(
        "BodyCN",
        parent=base_styles["Normal"],
        fontName=MAIN_FONT,
        fontSize=11,
        leading=15,
        spaceAfter=0,
    )
    styles["caption"] = ParagraphStyle(
        "CaptionCN",
        parent=styles["normal"],
        fontSize=9,
        leading=12,
        textColor=colors.grey,
        alignment=1,
    )
    styles["code_block"] = ParagraphStyle(
        "CodeBlock",
        parent=base_styles["Code"],
        fontName=MONO_FONT,
        fontSize=9,
        leading=12,
        backColor=colors.whitesmoke,
        borderColor=colors.lightgrey,
        borderWidth=0.5,
        borderPadding=6,
    )

    styles["table_header"] = ParagraphStyle(
        "TableHeader",
        parent=styles["normal"],
        fontSize=10,
        leading=13,
        alignment=1,
    )
    styles["table_cell"] = ParagraphStyle(
        "TableCell",
        parent=styles["normal"],
        fontSize=10,
        leading=13,
    )

    for level, size in [(1, 18), (2, 15), (3, 13), (4, 12), (5, 11), (6, 11)]:
        styles[f"h{level}"] = ParagraphStyle(
            f"H{level}",
            parent=styles["normal"],
            fontSize=size,
            leading=size + 4,
            spaceAfter=6,
            spaceBefore=10 if level <= 2 else 6,
        )

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title=os.path.basename(in_path),
    )

    story = _nodes_to_flowables(ast, styles, md_dir, doc.width)
    doc.build(story)


if __name__ == "__main__":
    main()
PY

echo "wrote ${out_path}"
