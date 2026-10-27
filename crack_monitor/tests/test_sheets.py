import xml.etree.ElementTree as ET

import cv2
import numpy as np

import make_sheets
from make_sheets import CELLS, CRACK_MARKER_IDS, DICTIONARY, MARKER_MM, TEST_MARKER_ID, marker_cells


def rasterize(cells, px_per_cell=20, quiet=2):
    """Draw the cell pattern as an image with a white margin, like a printed marker."""
    n = len(cells) + 2 * quiet
    img = np.full((n, n), 255, dtype=np.uint8)
    for r, row in enumerate(cells):
        for c, black in enumerate(row):
            if black:
                img[r + quiet, c + quiet] = 0
    return cv2.resize(img, None, fx=px_per_cell, fy=px_per_cell, interpolation=cv2.INTER_NEAREST)


def test_every_marker_decodes_to_its_own_id():
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY),
                                       cv2.aruco.DetectorParameters())
    for marker_id in CRACK_MARKER_IDS + [TEST_MARKER_ID]:
        cells = marker_cells(marker_id)
        assert len(cells) == CELLS and all(len(row) == CELLS for row in cells)
        _, ids, _ = detector.detectMarkers(rasterize(cells))
        assert ids is not None and ids.ravel().tolist() == [marker_id]


def test_sheets_are_a4_in_millimetres():
    for svg in (make_sheets.markers_sheet(), make_sheets.test_lines_sheet()):
        root = ET.fromstring(svg.split("\n", 1)[1])     # skip the <?xml ...?> line
        assert root.get("width") == "210.0mm" and root.get("height") == "297.0mm"
        assert root.get("viewBox") == "0 0 210.0 297.0"  # 1 SVG unit = 1 mm


def test_marker_is_60_mm_so_each_cell_is_1_cm():
    assert MARKER_MM / CELLS == 10.0


def test_pdf_is_one_a4_page_drawn_in_millimetres():
    pdf = make_sheets.to_pdf(make_sheets.markers_page())
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    assert b"/MediaBox [0 0 595.276 841.890]" in pdf               # A4 in points
    assert b"/Count 1" in pdf
    assert b"2.834646 0 0 -2.834646 0" in pdf                      # 1 unit = 1 mm (72 / 25.4 pt)
    xref = int(pdf.rsplit(b"startxref", 1)[1].split()[0])
    assert pdf[xref:xref + 4] == b"xref"                           # viewers can find the objects
