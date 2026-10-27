"""
Step 4: Accuracy on the printed test sheet (sheets/test_lines.pdf)

The sheet has marker 0 and six black lines of known width (0.2-3 mm). This measures each line with
the same OpenCV pipeline the app uses for cracks and compares it with the printed width.

Usage:
    python tools/eval_lines.py samples/lines_*.jpg
Prints a table (and returns rows for tools/evaluate.py).
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crack import find_crack  # noqa: E402
from make_sheets import TEST_LINE_WIDTHS_MM  # noqa: E402
from marker import TEST_MARKER_ID, check_marker, straighten  # noqa: E402
from quality import check_quality, load_image  # noqa: E402

# Where the lines are, in mm from the marker's top-left corner (see make_sheets.test_lines_page:
# marker at sheet (15, 45) mm, lines at sheet y = 125 + 25 i from x = 40 to 160 mm)
LINE_Y_MM = [80.0 + 25.0 * i for i in range(len(TEST_LINE_WIDTHS_MM))]
LINE_X_MM = (32.0, 138.0)      # stay away from the line ends
BAND_MM = 12.0                 # look this far above and below each line (lines are 25 mm apart; the
                               # detector ignores ~one kernel at the crop edge, so the band must be tall)


def measure_lines(photo):
    """Returns (rows, problem). One row per line: printed width, measured max/typical width, uncertainty."""
    marker_result, marker = check_marker(photo)
    if marker is None:
        return [], "no_marker"
    if marker_result["marker_id"] != TEST_MARKER_ID:
        return [], f"marker {marker_result['marker_id']} is not the test sheet (marker 0)"
    view = straighten(photo, marker)
    s, (mx, my, _, _) = view["px_per_mm"], view["marker_box"]
    quality = check_quality(photo, marker)
    rows = []
    for width, y_mm in zip(TEST_LINE_WIDTHS_MM, LINE_Y_MM):
        y0, y1 = int(my + (y_mm - BAND_MM) * s), int(my + (y_mm + BAND_MM) * s)
        x0, x1 = int(mx + LINE_X_MM[0] * s), int(mx + LINE_X_MM[1] * s)
        h, w = view["image"].shape[:2]
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            rows.append({"printed_mm": width, "found": False, "note": "line not fully in the photo"})
            continue
        band, valid = view["image"][y0:y1, x0:x1], view["valid"][y0:y1, x0:x1]
        result, _, _ = find_crack(band, s, valid, (-10_000, -10_000, 1, 1))   # marker far outside the band
        row = {"printed_mm": width, "found": result["crack_found"], "px_per_mm": s,
               "tilt_deg": marker_result["tilt_deg"], "blur_px": quality.get("blur_px")}
        if result["crack_found"]:
            row.update({k: result[k] for k in ("max_width_mm", "typical_width_mm", "uncertainty_mm", "problems")})
            row["error_mm"] = round(result["typical_width_mm"] - width, 3)
        rows.append(row)
    return rows, None


def print_table(name, rows):
    print(f"\n{name}")
    print(f"{'printed':>8} {'typical':>8} {'max':>7} {'+/-':>6} {'error':>7}  note")
    for r in rows:
        if not r["found"]:
            print(f"{r['printed_mm']:>8.1f}  {'not found':>24}  {r.get('note', '')}")
            continue
        flag = "too thin for this distance" if r["problems"] else ""
        print(f"{r['printed_mm']:>8.1f} {r['typical_width_mm']:>8.2f} {r['max_width_mm']:>7.2f} "
              f"{r['uncertainty_mm']:>6.2f} {r['error_mm']:>+7.2f}  {flag}")


if __name__ == "__main__":
    paths = [p for arg in sys.argv[1:] for p in glob.glob(arg)]
    if not paths:
        print("Usage: python tools/eval_lines.py samples/lines_*.jpg")
        sys.exit(1)
    errors = []
    for path in paths:
        rows, problem = measure_lines(load_image(path))
        if problem:
            print(f"\n{path}: {problem}")
            continue
        print_table(f"{path}  ({rows[0].get('px_per_mm', 0):.1f} px/mm, tilt {rows[0].get('tilt_deg', 0):.0f} deg)", rows)
        errors += [abs(r["error_mm"]) for r in rows if r["found"] and not r["problems"]]
    if errors:
        print(f"\nMean absolute error (lines thick enough for the distance): {np.mean(errors):.2f} mm over {len(errors)} lines")
