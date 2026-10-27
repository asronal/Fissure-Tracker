"""
Step 3: Find the crack and measure it in millimetres (OpenCV 5)

Works on the straight-on view from marker.py, where 1 mm = px_per_mm pixels everywhere.

What it does:
  1. Black-hat filter: keeps thin DARK lines, removes the wall's slow light changes
  2. Keeps long, thin shapes only (cracks), ignores the marker and specks, and the rims of wide
     dark areas (stains, shadows): a crack has light wall on both sides
  3. Width edge = half way between the wall and the crack's darkest point ("half-darkness"),
     so blur doesn't make the crack look wider
  4. Centre-line (thinning) -> length; local area / local length -> width along the crack
  5. Reports max width (95th percentile), typical width, length, widest point, and the
     uncertainty (+/- 1 pixel in mm). A crack only a few pixels wide needs a closer photo.

Usage:
    python crack.py path/to/photo.jpg
Prints the result as JSON and saves a marked-up photo to outputs/.
"""

import heapq
import json
import os
import sys

import cv2
import numpy as np

from marker import check_marker, draw_marker, put_label, straighten
from quality import check_quality, load_image

MAX_CRACK_MM = 6.0          # black-hat sees dark lines up to about this wide
MAX_KERNEL_PX = 81
MIN_CONTRAST = 12           # crack must be at least this much darker than the wall around it (0-255)
NOISE_FACTOR = 6.0          # ...and stand out this many times above the wall's texture noise
MIN_CRACK_MM = 10.0         # ignore dark bits shorter than this
MAX_FILL = 0.35             # area / (length x length): cracks are long and thin, blobs are not
MARKER_MARGIN_MM = 5.0      # ignore this much wall around the marker (its shadow/edge)
MIN_WIDTH_PX = 4.0          # thinner than this in the photo = +/-1 px is too coarse -> close-up

# BRE Digest 251 visible-damage categories by crack width (a guide, not a structural assessment)
BRE_CATEGORIES = [
    (0.1, 0, "Negligible (hairline)"),
    (1.0, 1, "Very slight"),
    (5.0, 2, "Slight"),
    (15.0, 3, "Moderate"),
    (25.0, 4, "Severe"),
    (float("inf"), 5, "Very severe"),
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def bre_category(width_mm):
    for limit, number, label in BRE_CATEGORIES:
        if width_mm < limit:
            return number, label
    return BRE_CATEGORIES[-1][1:]


def odd(n):
    n = int(round(n))
    return n if n % 2 else n + 1


def thin(mask):
    """Zhang-Suen thinning -> 1-pixel-wide centre-line (cv2.ximgproc is not in the pip package)."""
    img = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = sum(ring[:8])
            transitions = sum((ring[i] == 0) & (ring[i + 1] == 1) for i in range(8))
            if step == 0:
                side = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                side = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img[1:-1, 1:-1] == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & side
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1]


def step_lengths(skeleton):
    """Length each centre-line pixel stands for: 1 for straight steps, 1.41 for diagonal ones."""
    sk = skeleton.astype(np.float32)
    orth = cv2.filter2D(sk, -1, np.float32([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), borderType=cv2.BORDER_CONSTANT)
    diag = cv2.filter2D(sk, -1, np.float32([[1, 0, 1], [0, 0, 0], [1, 0, 1]]), borderType=cv2.BORDER_CONSTANT)
    links = np.maximum(orth + diag, 1)
    return np.where(skeleton > 0, (orth + np.sqrt(2) * diag) / links, 0).astype(np.float32)


STEPS_8 = [(-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)), (0, -1, 1.0),
           (0, 1, 1.0), (1, -1, np.sqrt(2)), (1, 0, 1.0), (1, 1, np.sqrt(2))]


def _walk(skeleton, start):
    """Shortest walks along the centre-line from `start` (Dijkstra). Returns (farthest pixel, previous-pixel map)."""
    h, w = skeleton.shape
    best = {start: 0.0}
    previous = {start: None}
    heap = [(0.0, start)]
    while heap:
        d, (y, x) = heapq.heappop(heap)
        if d > best[(y, x)]:
            continue
        for dy, dx, step in STEPS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] and d + step < best.get((ny, nx), np.inf):
                best[(ny, nx)] = d + step
                previous[(ny, nx)] = (y, x)
                heapq.heappush(heap, (d + step, (ny, nx)))
    return max(best, key=best.get), previous


def main_paths(skeleton, min_len_px):
    """For each centre-line piece: its longest end-to-end path, as an (N, 2) array of (y, x) pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    paths = []
    for i in range(1, n):
        x0, y0, w, h = stats[i, :4]
        if np.hypot(w, h) < min_len_px:
            continue
        piece = labels[y0:y0 + h, x0:x0 + w] == i
        start = tuple(int(v) for v in np.argwhere(piece)[0])
        a, _ = _walk(piece, start)               # farthest from anywhere = one end
        b, previous = _walk(piece, a)            # farthest from that end = other end
        path = []
        while b is not None:
            path.append(b)
            b = previous[b]
        path = np.array(path) + [y0, x0]
        if path_length(path) >= min_len_px:
            paths.append(path)
    return paths


def path_length(path, smooth=7):
    """Length of a pixel path after smoothing out the 8-neighbour staircase."""
    if len(path) < 2:
        return 0.0
    padded = np.pad(path.astype(np.float64), ((smooth // 2, smooth // 2), (0, 0)), mode="edge")
    kernel = np.ones(smooth) / smooth
    ys = np.convolve(padded[:, 0], kernel, mode="valid")
    xs = np.convolve(padded[:, 1], kernel, mode="valid")
    return float(np.sum(np.hypot(np.diff(ys), np.diff(xs))))


def running_median(values, window):
    """Median over a sliding window: removes single spikes (junctions, bends) but keeps real wide parts."""
    if len(values) < window:
        return np.full(len(values), float(np.median(values)))
    padded = np.pad(values, window // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)


def find_crack(straight, px_per_mm, valid, marker_box):
    """Finds and measures the crack in the straight-on view. Returns (result dict, crack mask, skeleton)."""
    gray = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY) if straight.ndim == 3 else straight
    k = min(MAX_KERNEL_PX, odd(max(7, 2 * MAX_CRACK_MM * px_per_mm)))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    # Only trust wall the photo really shows (not the warp's black border), and skip the marker
    # (borderValue=0: also stay away from the image's own edges, where black-hat is unreliable)
    usable = cv2.erode(valid, np.ones((k, k), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, w, h = marker_box
    m = int(round(MARKER_MARGIN_MM * px_per_mm)) + k // 2
    cv2.rectangle(usable, (x - m, y - m), (x + w + m, y + h + m), 0, cv2.FILLED)
    blackhat[usable == 0] = 0

    # Threshold well above the wall texture's own black-hat "noise"
    wall = blackhat[usable > 0]
    if wall.size == 0:
        return {"crack_found": False, "problems": ["no_crack"]}, None, None
    median = float(np.median(wall))
    spread = 1.4826 * float(np.median(np.abs(wall - median))) + 1.0
    low = max(MIN_CONTRAST, median + NOISE_FACTOR * spread)
    candidates = (blackhat > low).astype(np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Stains and shadows: dark areas too wide for the black-hat. Their curved rims still look like thin dark
    # arcs, so blank out such areas plus a rim around them (wide-scale darkness without thin-line darkness).
    # A crack crossing a stain is then measured only outside it.
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)     # 4x smaller: much faster
    big = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (odd(3 * k / 4),) * 2))
    big = cv2.resize(big, gray.shape[::-1], interpolation=cv2.INTER_LINEAR)
    wide_dark = ((big.astype(np.int16) - gray > low) & (blackhat <= low) & (usable > 0)).astype(np.uint8)
    wide_dark = cv2.morphologyEx(wide_dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # ignore crack-edge specks
    if wide_dark.any():
        candidates[cv2.dilate(wide_dark, np.ones((k // 2 * 2 + 1,) * 2, np.uint8)) > 0] = 0

    # Keep long, thin pieces only
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    min_len_px = MIN_CRACK_MM * px_per_mm
    keep = np.zeros(n, bool)
    for i in range(1, n):
        length = float(np.hypot(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]))
        if length >= min_len_px and stats[i, cv2.CC_STAT_AREA] <= MAX_FILL * length * length:
            keep[i] = True

    region = keep[labels].astype(np.uint8)
    if not region.any():
        return {"crack_found": False, "problems": ["no_crack"], "threshold": round(low, 1)}, None, None

    # Width edge at half the local darkness peak (unbiased by blur for cracks wider than the blur)
    peak = cv2.dilate(blackhat * region, np.ones((k, k), np.uint8))
    crack = (region > 0) & (blackhat.astype(np.float32) >= 0.5 * peak)
    crack = cv2.morphologyEx(crack.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Centre-line (thinning only inside the crack's bounding box - much faster)
    bx, by, bw, bh = cv2.boundingRect(crack)
    skeleton = np.zeros_like(crack)
    skeleton[by:by + bh, bx:bx + bw] = thin(crack[by:by + bh, bx:bx + bw])
    steps = step_lengths(skeleton)
    on_line = skeleton > 0

    # Local width = crack area / centre-line length in a window around each centre-line pixel.
    # The window must be clearly wider than the crack, so size it per piece of crack.
    dist = cv2.distanceTransform(crack, cv2.DIST_L2, 5)
    width_px = np.zeros(crack.shape, np.float32)
    n_pieces, piece = cv2.connectedComponents(crack, connectivity=8)
    for i in range(1, n_pieces):
        here = piece == i
        line_here = on_line & here
        if not line_here.any():
            continue
        window = odd(min(121, max(9, 6 * float(np.percentile(dist[line_here], 90)))))
        area = cv2.boxFilter(here.astype(np.float32), -1, (window, window), normalize=False)
        line = cv2.boxFilter(np.where(here, steps, 0).astype(np.float32), -1, (window, window), normalize=False)
        width_px[line_here] = (area / np.maximum(line, 1e-6))[line_here]

    # Follow each piece's main path (ignores little side twigs) and measure along it
    paths = main_paths(skeleton, min_len_px)
    if not paths:
        return {"crack_found": False, "problems": ["no_crack"], "threshold": round(low, 1)}, None, None

    length_px, all_widths, max_px, widest = 0.0, [], -1.0, None
    for path in paths:
        length_px += path_length(path)
        widths = width_px[path[:, 0], path[:, 1]]
        # Near the two ends the width window sees only half a crack plus its tip, which reads
        # too wide - leave those out
        trim = int(np.ceil(3 * float(np.median(widths)))) + 1
        if len(widths) > 4 * trim:
            widths, path = widths[trim:-trim], path[trim:-trim]
        all_widths.append(widths)
        smoothed = running_median(widths, odd(min(31, max(5, 2 * float(np.median(widths))))))
        k_max = int(np.argmax(smoothed))
        if smoothed[k_max] > max_px:
            max_px, widest = float(smoothed[k_max]), path[k_max]
    typical_px = float(np.median(np.concatenate(all_widths)))
    wy, wx = float(widest[0]), float(widest[1])

    max_mm = max_px / px_per_mm
    category, label = bre_category(max_mm)
    result = {
        "crack_found": True,
        "max_width_mm": round(max_mm, 2),
        "typical_width_mm": round(typical_px / px_per_mm, 2),
        "length_mm": round(length_px / px_per_mm, 1),
        "uncertainty_mm": round(1.0 / px_per_mm, 2),
        "max_width_px": round(max_px, 1),
        "widest_point_straight": [round(wx, 1), round(wy, 1)],
        # Widest point in wall mm, measured from the marker's top-left corner: the same spot
        # keeps the same numbers in every photo of this crack
        "widest_point_mm": [round((wx - x) / px_per_mm, 1), round((wy - y) / px_per_mm, 1)],
        "pieces": len(paths),
        "bre_category": category,
        "bre_label": label,
        "threshold": round(low, 1),
        "problems": ["crack_too_thin_for_this_distance"] if max_px < MIN_WIDTH_PX else [],
    }
    return result, crack, skeleton


def measure_photo(photo):
    """Full chain on one photo: marker -> quality -> straight-on view -> crack. Returns (result, marked-up image)."""
    marker_result, marker = check_marker(photo)
    quality = check_quality(photo, marker)
    result = {"marker": marker_result, "quality": quality}
    marked = draw_marker(photo, marker_result, marker)
    if marker is None:
        return result, marked

    view = straighten(photo, marker)
    crack_result, crack_mask, _ = find_crack(view["image"], view["px_per_mm"], view["valid"], view["marker_box"])
    result["crack"] = crack_result
    if crack_mask is not None:
        marked = draw_crack(marked, crack_mask, view, crack_result)
    return result, marked


def draw_crack(marked, crack_mask, view, crack_result):
    """Paint the crack (red) and its widest point back onto the original photo."""
    h, w = marked.shape[:2]
    to_photo = np.linalg.inv(view["to_straight"])
    mask = cv2.warpPerspective(crack_mask * 255, to_photo, (w, h), flags=cv2.INTER_NEAREST)
    out = marked.copy()
    out[mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(out, 0.6, marked, 0.4, 0)

    point = np.float32([[crack_result["widest_point_straight"]]])
    px, py = cv2.perspectiveTransform(point, to_photo)[0, 0].astype(int)
    radius = max(12, w // 60)
    cv2.circle(out, (px, py), radius, (0, 255, 255), max(2, w // 500), cv2.LINE_AA)
    label = f"max {crack_result['max_width_mm']:.2f} mm (+/-{crack_result['uncertainty_mm']:.2f})"
    put_label(out, label, (px + radius + 6, py), (0, 255, 255))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python crack.py path/to/photo.jpg")
        sys.exit(1)

    path = sys.argv[1]
    result, marked = measure_photo(load_image(path))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name, _ = os.path.splitext(os.path.basename(path))
    out_path = os.path.join(OUTPUT_DIR, f"{name}_crack.jpg")
    cv2.imwrite(out_path, marked)
    result["marked_image"] = out_path
    print(json.dumps(result, indent=2))
