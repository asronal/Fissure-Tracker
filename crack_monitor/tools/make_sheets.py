"""
Step 1: Printable sheets for the crack monitor (OpenCV 5 ArUco markers)

Makes three A4 pages in sheets/, each as SVG and PDF, drawn in exact millimetres:
  markers       markers 1-6, one per crack (60 mm, stick next to the crack)
  test_lines    marker 0 + lines of known width, for measuring our accuracy
  draw_guide    no printer? draw markers 1-6 by hand on 1 cm graph paper

Usage:
    python tools/make_sheets.py
Print the PDF at "Actual size" / Scale 100% (NOT "Fit to page"), then measure the
50 mm check bar with a ruler.
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Marker settings live in marker.py so the printed sheets and the measuring code always agree
from marker import CRACK_MARKER_IDS, DICTIONARY, MARKER_MM, TEST_MARKER_ID  # noqa: E402

CELLS = 6                   # 4x4 data cells + 1-cell black border on each side (60 mm -> 1 cm cells)
TEST_LINE_WIDTHS_MM = [0.2, 0.3, 0.5, 1.0, 2.0, 3.0]
PAGE_W, PAGE_H = 210.0, 297.0
PT_PER_MM = 72 / 25.4       # PDF unit = 1/72 inch

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sheets")


# --- Shapes (all in mm, y going down like on paper) ------------------------------------

def rect(x, y, w, h, grey=0.0):
    return ("rect", x, y, w, h, grey)


def line(x1, y1, x2, y2, grey=0.6, width=0.25):
    return ("line", x1, y1, x2, y2, grey, width)


def text(x, y, s, size=4.2, anchor="start", bold=False):
    return ("text", x, y, s, size, anchor, bold)


def marker_cells(marker_id):
    """6x6 list of booleans, True = black cell."""
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, CELLS, borderBits=1)
    return [[img[r, c] < 128 for c in range(CELLS)] for r in range(CELLS)]


def marker_shapes(marker_id, x, y, size=MARKER_MM, grid=False):
    cell = size / CELLS
    shapes = [rect(x, y, size, size, 0.0)]
    for r, row in enumerate(marker_cells(marker_id)):
        for c, black in enumerate(row):
            if not black:
                shapes.append(rect(x + c * cell, y + r * cell, cell, cell, 1.0))
    if grid:        # grey grid lines so squares are easy to count (hand-drawing guide only)
        for i in range(CELLS + 1):
            shapes.append(line(x + i * cell, y, x + i * cell, y + size))
            shapes.append(line(x, y + i * cell, x + size, y + i * cell))
    return shapes


def check_bar(x, y):
    """A 50 mm bar with 10 mm ticks - measure it with a ruler after printing."""
    shapes = [rect(x, y, 50, 1.5)]
    shapes += [rect(x + i * 10 - 0.15, y - 2, 0.3, 5.5) for i in range(6)]
    shapes.append(text(x + 55, y + 1.8, "must measure exactly 50 mm - if not, reprint at 100% scale", 3.4))
    return shapes


# --- Pages --------------------------------------------------------------------------------

def markers_page():
    shapes = [text(15, 16, "Crack markers 1-6 (60 mm)", 6, bold=True),
              text(15, 23, "Stick one marker flat on the wall next to each crack and leave it there. "
                           "Keep the white border around it.", 3.4)]
    shapes += check_bar(15, 30)
    ids = iter(CRACK_MARKER_IDS)
    for y in (45, 128, 211):
        for x in (25, 115):
            marker_id = next(ids)
            shapes += marker_shapes(marker_id, x, y)
            shapes.append(text(x + MARKER_MM / 2, y + MARKER_MM + 8, f"Crack #{marker_id}", 4.5,
                               anchor="middle", bold=True))
    return shapes


def test_lines_page():
    shapes = [text(15, 16, "Accuracy test sheet (marker 0)", 6, bold=True),
              text(15, 23, "Photograph this page from different distances and angles. "
                           "The app should measure each line's width.", 3.4)]
    shapes += check_bar(15, 30)
    shapes += marker_shapes(TEST_MARKER_ID, 15, 45)
    y = 125
    for width in TEST_LINE_WIDTHS_MM:
        shapes.append(rect(40, y - width / 2, 120, width))
        shapes.append(text(165, y + 1.5, f"{width:g} mm", 4.5))
        y += 25
    return shapes


def draw_guide_page():
    """Small diagrams: which 1 cm squares to colour black for each marker."""
    shapes = [text(15, 16, "No printer? Draw the markers by hand", 6, bold=True),
              text(15, 24, "On 1 cm graph paper, draw a 6 x 6 square block (6 cm x 6 cm). Colour the "
                           "black squares fully with a black marker pen.", 3.4),
              text(15, 29, "Keep at least 1 cm of white paper all around. Check the block is exactly "
                           "6 cm wide with a ruler.", 3.4)]
    guide_mm = 36.0      # diagram size on this page (not the real size)
    ids = iter(CRACK_MARKER_IDS)
    for y in (42, 122, 202):
        for x in (30, 120):
            marker_id = next(ids)
            shapes += marker_shapes(marker_id, x, y, guide_mm, grid=True)
            shapes.append(text(x + guide_mm / 2, y + guide_mm + 8, f"Crack #{marker_id} (draw 6 cm wide)", 4,
                               anchor="middle", bold=True))
    return shapes


# --- SVG ----------------------------------------------------------------------------------

def to_svg(shapes):
    parts = [f'<rect width="{PAGE_W}" height="{PAGE_H}" fill="#fff"/>']
    for s in shapes:
        if s[0] == "rect":
            _, x, y, w, h, grey = s
            parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{_hex(grey)}"/>')
        elif s[0] == "line":
            _, x1, y1, x2, y2, grey, width = s
            parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                         f'stroke="{_hex(grey)}" stroke-width="{width}"/>')
        else:
            _, x, y, string, size, anchor, bold = s
            parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
                         f'font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}" '
                         f'fill="#000">{string}</text>')
    body = "\n".join(parts)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}mm" height="{PAGE_H}mm" '
            f'viewBox="0 0 {PAGE_W} {PAGE_H}" shape-rendering="crispEdges">\n{body}\n</svg>\n')


def _hex(grey):
    v = int(round(grey * 255))
    return f"#{v:02x}{v:02x}{v:02x}"


# --- PDF (written by hand: exact millimetres, no extra libraries) -----------------------

def _text_width_mm(string, size, bold):
    """Rough Helvetica width, only used to centre labels."""
    em = 0
    for ch in string:
        if ch == " ":
            em += 0.28
        elif ch.isupper() or ch == "#":
            em += 0.68 if bold else 0.64
        elif ch.isdigit():
            em += 0.56
        elif ch in "il.,()-":
            em += 0.30
        else:
            em += 0.58 if bold else 0.53
    return em * size


def to_pdf(shapes):
    # Work in mm with y pointing down (like the SVG): scale by PT_PER_MM and flip the page.
    ops = [f"{PT_PER_MM:.6f} 0 0 {-PT_PER_MM:.6f} 0 {PAGE_H * PT_PER_MM:.4f} cm"]
    for s in shapes:
        if s[0] == "rect":
            _, x, y, w, h, grey = s
            ops.append(f"{grey:.3f} g {x:.4f} {y:.4f} {w:.4f} {h:.4f} re f")
        elif s[0] == "line":
            _, x1, y1, x2, y2, grey, width = s
            ops.append(f"{grey:.3f} G {width:.3f} w {x1:.4f} {y1:.4f} m {x2:.4f} {y2:.4f} l S")
        else:
            _, x, y, string, size, anchor, bold = s
            if anchor == "middle":
                x -= _text_width_mm(string, size, bold) / 2
            safe = string.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            # Text matrix flips y back so the letters aren't upside down
            ops.append(f"0 g BT /{'F2' if bold else 'F1'} {size:.3f} Tf 1 0 0 -1 {x:.4f} {y:.4f} Tm "
                       f"({safe}) Tj ET")
    content = "\n".join(ops).encode("latin-1")

    w_pt, h_pt = PAGE_W * PT_PER_MM, PAGE_H * PT_PER_MM
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w_pt:.3f} {h_pt:.3f}] /Contents 4 0 R "
         f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>").encode(),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return pdf


# Kept for the tests and anyone who only wants the SVG text
def markers_sheet():
    return to_svg(markers_page())


def test_lines_sheet():
    return to_svg(test_lines_page())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, page in [("markers", markers_page()), ("test_lines", test_lines_page()),
                       ("draw_guide", draw_guide_page())]:
        with open(os.path.join(OUT_DIR, f"{name}.svg"), "w", encoding="utf-8") as f:
            f.write(to_svg(page))
        with open(os.path.join(OUT_DIR, f"{name}.pdf"), "wb") as f:
            f.write(to_pdf(page))
        print(f"Wrote sheets/{name}.svg and sheets/{name}.pdf")


if __name__ == "__main__":
    main()
