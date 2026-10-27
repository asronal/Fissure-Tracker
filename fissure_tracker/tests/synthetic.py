"""
Synthetic wall photos with a known answer, for testing.

A flat wall (plaster-like texture) with our marker on it, photographed by a virtual
phone camera from a chosen distance and tilt. Because we build the camera ourselves,
we know the true scale and tilt exactly.
"""

import cv2
import numpy as np

from marker import DICTIONARY, FOCAL_FACTOR, MARKER_MM

WALL_MM = (1000.0, 800.0)      # wall area we render (width, height)


def wall_image(px_per_mm, seed=0, wall_mm=WALL_MM):
    """Light grey plaster with some blotches and fine grain (grayscale)."""
    w, h = int(wall_mm[0] * px_per_mm), int(wall_mm[1] * px_per_mm)
    rng = np.random.default_rng(seed)
    blotches = cv2.resize(rng.normal(0, 1, (12, 15)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    grain = rng.normal(0, 4, (h, w)).astype(np.float32)
    return np.clip(200 + 8 * blotches + grain, 0, 255).astype(np.uint8)


def put_marker(wall, marker_id, top_left_mm, px_per_mm):
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    side = int(round(MARKER_MM * px_per_mm))
    x, y = int(round(top_left_mm[0] * px_per_mm)), int(round(top_left_mm[1] * px_per_mm))
    wall[y:y + side, x:x + side] = cv2.aruco.generateImageMarker(dictionary, marker_id, side, borderBits=1)


def camera_view(tilt_deg, distance_mm, look_at_mm, image_size):
    """Homography wall-mm -> photo-px for a camera `distance_mm` away, turned by `tilt_deg` around the vertical axis."""
    w, h = image_size
    f = FOCAL_FACTOR * max(w, h)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    a = np.radians(tilt_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    t = -R @ np.array([look_at_mm[0], look_at_mm[1], 0.0]) + np.array([0, 0, distance_mm])
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def make_photo(marker_id=1, tilt_deg=0.0, distance_mm=300.0, image_size=(1200, 900), marker_id_present=True,
               render_px_per_mm=4.0, blur_sigma=0.0, seed=0, draw=None, wall_mm=WALL_MM,
               marker_offset_mm=(0.0, 0.0), look_at_mm=None):
    """
    Returns (photo BGR, truth). The marker sits in the middle of the wall (moved by `marker_offset_mm`,
    to simulate a marker that was re-stuck); the camera looks at the wall's middle (or `look_at_mm`).
    `draw(wall, px_per_mm)` can paint extra things (e.g. cracks) on the wall before the photo is taken.
    """
    wall = wall_image(render_px_per_mm, seed, wall_mm)
    marker_top_left = (wall_mm[0] / 2 - MARKER_MM / 2 + marker_offset_mm[0],
                       wall_mm[1] / 2 - MARKER_MM / 2 + marker_offset_mm[1])
    if marker_id_present:
        put_marker(wall, marker_id, marker_top_left, render_px_per_mm)
    if draw is not None:
        draw(wall, render_px_per_mm)

    centre = look_at_mm or (wall_mm[0] / 2, wall_mm[1] / 2)
    mm_to_photo = camera_view(tilt_deg, distance_mm, centre, image_size)
    wallpx_to_mm = np.diag([1 / render_px_per_mm, 1 / render_px_per_mm, 1.0])
    photo_scale = FOCAL_FACTOR * max(image_size) / distance_mm      # px per mm, straight on

    # Smooth before shrinking so edges don't get jagged (like a real lens would)
    shrink = render_px_per_mm / photo_scale
    if shrink > 1:
        wall = cv2.GaussianBlur(wall, (0, 0), 0.5 * shrink)
    photo = cv2.warpPerspective(wall, mm_to_photo @ wallpx_to_mm, image_size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    if blur_sigma:
        photo = cv2.GaussianBlur(photo, (0, 0), blur_sigma)

    truth = {"tilt_deg": tilt_deg, "px_per_mm": photo_scale, "mm_to_photo": mm_to_photo,
             "marker_top_left_mm": marker_top_left}
    return cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR), truth


def crack_path(start_mm, length_mm, seed=0, step_mm=4.0, wiggle_mm=1.5):
    """A wiggly, mostly horizontal crack line: list of (x, y) points in wall mm."""
    rng = np.random.default_rng(seed)
    x, y = start_mm
    points = [(x, y)]
    for _ in range(int(length_mm / step_mm)):
        x += step_mm
        y += float(rng.normal(0, wiggle_mm))
        points.append((x, y))
    return points


def crack_drawer(points_mm, width_mm, darkness=70, wide_from=None, wide_to=None, wide_mm=None):
    """
    Paints a crack of `width_mm` (and optionally a wider section between point indexes
    wide_from..wide_to) for make_photo(draw=...). Returns (draw function, truth dict).
    The true widths are what really gets painted (whole pixels at the render scale).
    """
    truth = {}

    def draw(wall, px_per_mm):
        shift = 4                                            # sub-pixel point positions

        def paint(points, w_mm):
            thickness = max(1, int(round(w_mm * px_per_mm)))
            pts = (np.array(points) * px_per_mm * (1 << shift)).astype(np.int32)
            cv2.polylines(wall, [pts], False, darkness, thickness, cv2.LINE_AA, shift)
            return thickness / px_per_mm

        truth["width_mm"] = paint(points_mm, width_mm)
        truth["max_width_mm"] = truth["width_mm"]
        if wide_mm is not None:
            truth["max_width_mm"] = paint(points_mm[wide_from:wide_to + 1], wide_mm)
            xs, ys = zip(*points_mm[wide_from:wide_to + 1])
            truth["wide_centre_mm"] = (float(np.mean(xs)), float(np.mean(ys)))
        xs, ys = np.array(points_mm).T
        truth["length_mm"] = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))

    return draw, truth


def sheet_photo(tilt_deg=0.0, distance_mm=300.0, image_size=(2250, 3000), line_widths_mm=None):
    """
    A photo of the printed accuracy test sheet (marker 0 + lines of known width, laid out like
    sheets/test_lines.pdf). Returns (photo, truth widths as actually painted).
    """
    from make_sheets import TEST_LINE_WIDTHS_MM

    widths = line_widths_mm or TEST_LINE_WIDTHS_MM
    wall_mm, render = (500.0, 600.0), 12.0
    marker = (wall_mm[0] / 2 - MARKER_MM / 2, wall_mm[1] / 2 - MARKER_MM / 2)
    sheet = (marker[0] - 15.0, marker[1] - 45.0)            # sheet's top-left corner on the wall
    truth = []

    def draw(wall, px_per_mm):
        def px(v):
            return int(round(v * px_per_mm))
        wall[px(sheet[1]):px(sheet[1] + 297), px(sheet[0]):px(sheet[0] + 210)] = 235      # white paper
        put_marker(wall, 0, marker, px_per_mm)
        truth.clear()
        for i, width in enumerate(widths):
            rows = max(1, px(width))
            top = px(sheet[1] + 125 + 25 * i) - rows // 2
            wall[top:top + rows, px(sheet[0] + 40):px(sheet[0] + 160)] = 25                # black ink
            truth.append(rows / px_per_mm)

    look_at = (sheet[0] + 105.0, sheet[1] + 148.0)
    photo, _ = make_photo(marker_id=0, tilt_deg=tilt_deg, distance_mm=distance_mm, image_size=image_size,
                          render_px_per_mm=render, draw=draw, wall_mm=wall_mm, marker_id_present=False,
                          look_at_mm=look_at)
    return photo, truth
