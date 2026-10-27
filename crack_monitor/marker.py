"""
Step 2: Find the printed marker -> scale, tilt and a straight-on view (OpenCV 5)

What it does:
  1. Finds the ArUco marker (DICT_4X4_50) we stuck next to the crack. Its ID tells us WHICH crack.
  2. The marker is exactly 60 mm wide, so its size in pixels gives the scale (pixels per mm).
  3. Measures how slanted the photo is (tilt): 0 degrees = phone held parallel to the wall.
  4. Warps the photo to a straight-on view with a fixed scale, so later steps can measure in mm.

Problems it reports (the agent uses these to decide what to ask for next):
  no_marker          - marker not in the photo
  marker_too_small   - photo taken from too far away
  too_tilted         - phone not parallel to the wall

Usage:
    python marker.py path/to/photo.jpg
Prints the result as JSON and saves a marked-up photo + the straight-on view to outputs/.
"""

import json
import os
import sys

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_MM = 60.0                    # printed size: 6 cells of 10 mm
CRACK_MARKER_IDS = [1, 2, 3, 4, 5, 6]
TEST_MARKER_ID = 0                  # the accuracy test sheet
VALID_IDS = [TEST_MARKER_ID] + CRACK_MARKER_IDS

MIN_MARKER_SIDE_PX = 80             # smaller = photo taken from too far away
MAX_TILT_DEG = 30.0                 # more = measurements get unreliable

# We don't know the phone's lens, so tilt uses a typical phone main camera
# (~26 mm equivalent, ~70 degrees wide): focal length ~0.72 x the longest image side.
FOCAL_FACTOR = 0.72

MAX_STRAIGHT_PX_PER_MM = 20.0       # never enlarge more than this in the straight-on view
MAX_STRAIGHT_SIDE_PX = 4000         # keep the straight-on image a sensible size
MAX_EXTENT_MM = 500.0               # only keep wall up to this far from the marker

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Marker corners on the wall in mm (ArUco order: top-left, top-right, bottom-right, bottom-left)
MARKER_CORNERS_MM = np.float32([[0, 0], [MARKER_MM, 0], [MARKER_MM, MARKER_MM], [0, MARKER_MM]])

_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # sub-pixel corners = better mm scale
DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), _params)


def find_markers(img):
    """All our markers in the photo: list of {"id", "corners" (4x2 float32)}."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    if ids is None:
        return []
    return [{"id": int(i), "corners": c.reshape(4, 2).astype(np.float32)}
            for c, i in zip(corners, ids.ravel()) if int(i) in VALID_IDS]


def side_lengths(corners):
    return [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]


def camera_matrix(img_shape):
    h, w = img_shape[:2]
    f = FOCAL_FACTOR * max(h, w)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def tilt_degrees(corners, img_shape):
    """
    Angle between the camera's viewing direction and the wall's straight-on direction.
    The homography from the flat marker (mm) to the photo (px) contains the camera rotation:
    K^-1 * H = [r1 r2 t] -> wall normal = r1 x r2.
    """
    H = cv2.getPerspectiveTransform(MARKER_CORNERS_MM, corners)
    M = np.linalg.inv(camera_matrix(img_shape)) @ H
    r1 = M[:, 0] / np.linalg.norm(M[:, 0])
    r2 = M[:, 1] / np.linalg.norm(M[:, 1])
    normal = np.cross(r1, r2)
    normal /= np.linalg.norm(normal)
    return float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))


def check_marker(img):
    """Returns (result dict, the chosen marker or None)."""
    markers = find_markers(img)
    result = {"marker_found": bool(markers), "marker_ids_seen": sorted(m["id"] for m in markers), "problems": []}
    if not markers:
        result["problems"].append("no_marker")
        return result, None

    # Several of our markers in one photo: use the biggest (closest) one
    marker = max(markers, key=lambda m: cv2.contourArea(m["corners"]))
    sides = side_lengths(marker["corners"])
    tilt = tilt_degrees(marker["corners"], img.shape)

    result.update({
        "marker_id": marker["id"],
        "is_test_sheet": marker["id"] == TEST_MARKER_ID,
        "marker_side_px": round(float(np.mean(sides)), 1),
        "px_per_mm": round(float(np.mean(sides)) / MARKER_MM, 3),
        "px_per_mm_min": round(min(sides) / MARKER_MM, 3),     # worst direction (squashed by tilt)
        "tilt_deg": round(tilt, 1),
    })
    if min(sides) < MIN_MARKER_SIDE_PX:
        result["problems"].append("marker_too_small")
    if tilt > MAX_TILT_DEG:
        result["problems"].append("too_tilted")
    return result, marker


def straighten(img, marker, px_per_mm=None):
    """
    Warp the photo to a straight-on view of the wall where 1 mm = px_per_mm pixels everywhere.
    Returns {"image", "valid" (mask: 255 where the photo has data), "px_per_mm",
             "to_straight" (3x3, photo px -> straight px), "marker_box" (x, y, w, h)}.
    """
    if px_per_mm is None:
        # Use the resolution the photo really has at the marker - don't invent detail
        px_per_mm = min(MAX_STRAIGHT_PX_PER_MM, float(np.mean(side_lengths(marker["corners"]))) / MARKER_MM)

    photo_to_mm = cv2.getPerspectiveTransform(marker["corners"], MARKER_CORNERS_MM)

    # Which part of the wall does the photo show? (in mm, marker top-left = 0,0)
    h, w = img.shape[:2]
    photo_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    wall = cv2.perspectiveTransform(photo_corners, photo_to_mm).reshape(-1, 2)
    centre = MARKER_MM / 2
    x0, y0 = np.clip(wall.min(axis=0), centre - MAX_EXTENT_MM, centre + MAX_EXTENT_MM)
    x1, y1 = np.clip(wall.max(axis=0), centre - MAX_EXTENT_MM, centre + MAX_EXTENT_MM)

    # Shrink the scale if the straight-on image would be too big
    px_per_mm = min(px_per_mm, MAX_STRAIGHT_SIDE_PX / max(x1 - x0, y1 - y0))

    mm_to_straight = np.array([[px_per_mm, 0, -x0 * px_per_mm],
                               [0, px_per_mm, -y0 * px_per_mm],
                               [0, 0, 1]], dtype=np.float64)
    to_straight = mm_to_straight @ photo_to_mm
    size = (int(np.ceil((x1 - x0) * px_per_mm)), int(np.ceil((y1 - y0) * px_per_mm)))
    straight = cv2.warpPerspective(img, to_straight, size, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), to_straight, size,
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    mx, my = -x0 * px_per_mm, -y0 * px_per_mm
    side = MARKER_MM * px_per_mm
    return {"image": straight, "valid": valid, "px_per_mm": round(px_per_mm, 3), "to_straight": to_straight,
            "marker_box": (int(round(mx)), int(round(my)), int(round(side)), int(round(side)))}


def draw_marker(img, result, marker):
    out = img.copy()
    if marker is not None:
        cv2.polylines(out, [marker["corners"].astype(np.int32)], True, (0, 200, 0), max(2, img.shape[1] // 400))
        label = f"#{marker['id']}  {result['px_per_mm']:.1f} px/mm  tilt {result['tilt_deg']:.0f} deg"
        x, y = marker["corners"][0].astype(int)
        put_label(out, label, (x, max(20, y - 10)), (0, 220, 0))
    return out


def put_label(img, text, origin, colour):
    """Text with a dark outline, readable on light and dark walls."""
    scale = max(0.6, img.shape[1] / 1500)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python marker.py path/to/photo.jpg")
        sys.exit(1)

    path = sys.argv[1]
    photo = cv2.imread(path, cv2.IMREAD_COLOR)
    if photo is None:
        sys.exit(f"Could not read image: {path}")

    result, marker = check_marker(photo)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name, _ = os.path.splitext(os.path.basename(path))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_marker.jpg"), draw_marker(photo, result, marker))
    if marker is not None:
        view = straighten(photo, marker)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_straight.jpg"), view["image"])
        result["straight_px_per_mm"] = view["px_per_mm"]
    print(json.dumps(result, indent=2))
