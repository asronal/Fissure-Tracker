"""
Step 5: Line this photo up with the last photo of the same crack (ORB, then OpenCV 5 ALIKED + LightGlue)

Both photos are first turned into straight-on views of the wall in millimetres (marker.straighten),
so the same wall point should land on the same mm position in both photos - as long as the marker
hasn't moved. Features on the wall (the marker itself is masked out) are matched between the two views,
and RANSAC fits the shift + rotation between them.

Two ways to find and match the features, tried in this order:
  orb               OpenCV's classic ORB + ratio test: fast (~0.3 s) and precise to ~0.5 mm when the two
                    photos are similar. Also always used for the plain-wall check.
  aliked_lightglue  OpenCV 5's learned features: ALIKED keypoints + descriptors, matched by the LightGlue
                    network (both ONNX models run in OpenCV's own DNN engine). Used only when ORB finds too
                    few matches: it copes better with a different angle, light or a curled sheet, but the
                    model works on a 640 x 640 input, so it is less precise (~1-2 mm) and slower (~3 s).
                    Needs the model files in models/ (or MODEL_DIR): run tools/get_models.py once (52 MB).

Problems it reports (the agent uses these to decide what to ask for next):
  not_enough_matches  - the photos show different parts of the wall -> "stand where you stood last time"
  marker_moved        - the wall shifted against the marker -> the marker was moved or re-stuck,
                        so old and new widths can't be compared directly (needs a human)
  plain_wall          - too little wall texture to compare (e.g. smooth paint): rely on the marker alone
  scale_too_different - e.g. a close-up vs a normal photo: wall texture looks different at such
                        different scales, so no comparison is made (not the owner's fault)

Usage:
    python align.py new_photo.jpg old_photo.jpg [orb|aliked_lightglue]     (default: ORB, then ALIKED + LightGlue)
Prints the result as JSON and saves the matched features side by side to outputs/.
"""

import json
import os
import sys

import cv2
import numpy as np

from marker import MARKER_MM, check_marker, side_lengths, straighten
from quality import load_image

ALIGN_PX_PER_MM = 3.0       # both photos are compared at this common resolution
ORB_FEATURES = 4000
ORB_THRESHOLD = 7           # plaster texture is faint: lower than ORB's default of 20
CLAHE_CLIP = 3.0            # local contrast boost so faint wall texture gives features
RATIO = 0.8                 # Lowe's ratio test: best match must be clearly better than the second best
RANSAC_MM = 1.5             # a match further off than this (in mm) is an outlier
MIN_INLIERS = 25            # fewer = the photos don't show enough of the same wall
MAX_SHIFT_MM = 3.0          # wall shifted more than this against the marker = marker was moved
MARKER_MARGIN_MM = 8.0      # ignore the marker and this much around it (it always "matches")
MIN_FEATURES = 150          # fewer wall features in a photo = wall too plain to compare
MAX_SCALE_RATIO = 2.0       # photos whose px/mm differ more than this aren't compared
LIGHTGLUE_SCORE = 0.2       # LightGlue match confidence below this is dropped

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HERE, "models"))
ALIKED_MODEL = "aliked-n16rot-top1k-640.onnx"      # same files and checksums as OpenCV's own tests
LIGHTGLUE_MODEL = "aliked_lightglue.onnx"

OUTPUT_DIR = os.path.join(HERE, "outputs")
ORB = cv2.ORB.create(nfeatures=ORB_FEATURES, fastThreshold=ORB_THRESHOLD)
CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)
_deep = {}


def deep_models():
    """(ALIKED, LightGlue), loaded once per process; None if the model files aren't there."""
    if "models" not in _deep:
        paths = [os.path.join(MODEL_DIR, name) for name in (ALIKED_MODEL, LIGHTGLUE_MODEL)]
        _deep["models"] = None
        if all(os.path.exists(p) for p in paths):
            _deep["models"] = (cv2.ALIKED_create(paths[0]), cv2.LightGlueMatcher_create(paths[1], LIGHTGLUE_SCORE))
    return _deep["models"]


def wall_view(photo, marker):
    """Straight-on grey view at ALIGN_PX_PER_MM, a mask of usable wall, and where the marker's corner is."""
    # A close-up has far more pixels per mm than the common scale: smooth it first, or shrinking it
    # turns the fine wall texture into noise that matches nothing (aliasing)
    shrink = float(np.mean(side_lengths(marker["corners"]))) / MARKER_MM / ALIGN_PX_PER_MM
    if shrink > 1.3:
        photo = cv2.GaussianBlur(photo, (0, 0), 0.5 * shrink)
    view = straighten(photo, marker, ALIGN_PX_PER_MM)
    gray = CLAHE.apply(cv2.cvtColor(view["image"], cv2.COLOR_BGR2GRAY))
    mask = cv2.erode(view["valid"], np.ones((15, 15), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    s = view["px_per_mm"]
    # Exact position of the marker's top-left corner (0, 0 mm) in the straight view
    origin = cv2.perspectiveTransform(marker["corners"][:1].reshape(-1, 1, 2), view["to_straight"]).reshape(2)
    m = MARKER_MARGIN_MM * s
    x0, y0 = int(origin[0] - m), int(origin[1] - m)
    side = int(MARKER_MM * s + 2 * m)
    mask[max(0, y0):max(0, y0 + side), max(0, x0):max(0, x0 + side)] = 0
    return {"gray": gray, "mask": mask, "origin": origin, "px_per_mm": s}


def to_mm(keypoints, view):
    """Keypoint positions in wall mm (marker top-left corner = 0, 0)."""
    pts = np.float32([k.pt for k in keypoints])
    return (pts - view["origin"]) / view["px_per_mm"]


def orb_matches(kp_new, des_new, kp_old, des_old):
    good = []
    if des_new is not None and des_old is not None and len(kp_old) >= 2:
        for pair in MATCHER.knnMatch(des_new, des_old, k=2):
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
                good.append(pair[0])
    return kp_new, kp_old, good


def deep_matches(new, old):
    """ALIKED keypoints + descriptors on both views, matched by LightGlue (which also uses where the
    keypoints are, not only the descriptors)."""
    aliked, lightglue = deep_models()
    found = []
    for view in (new, old):
        # Grey out everything that isn't wall: the network sees well past each keypoint, so it would
        # otherwise "recognise" the marker (identical in every photo) and match a different wall
        gray = np.where(view["mask"] > 0, view["gray"], np.uint8(cv2.mean(view["gray"], view["mask"])[0]))
        colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)                  # the network expects 3 channels
        kp, des = aliked.detectAndCompute(colour, view["mask"])
        found.append((kp, des, colour.shape[1::-1]))
    (kp_new, des_new, size_new), (kp_old, des_old, size_old) = found
    if len(kp_new) < 2 or len(kp_old) < 2:
        return kp_new, kp_old, []
    lightglue.setPairInfo(np.float32([k.pt for k in kp_new]), np.float32([k.pt for k in kp_old]), size_new, size_old)
    return kp_new, kp_old, list(lightglue.match(des_new, des_old))


def align_views(new, old, method="orb"):
    """Match wall features between two wall views with one method. Returns (result, drawing data)."""
    kp_new, des_new = ORB.detectAndCompute(new["gray"], new["mask"])
    kp_old, des_old = ORB.detectAndCompute(old["gray"], old["mask"])
    texture = [len(kp_new), len(kp_old)]          # ORB keypoints = how much wall texture there is
    result = {"method": method, "features": texture, "matches": 0, "inliers": 0, "shift_mm": None,
              "shift_xy_mm": None, "rotation_deg": None, "aligned": False, "problems": []}
    if method == "orb":
        kp_new, kp_old, good = orb_matches(kp_new, des_new, kp_old, des_old)
    elif min(texture) >= MIN_FEATURES:            # no point running the network on a blank wall
        kp_new, kp_old, good = deep_matches(new, old)
        result["features"] = [len(kp_new), len(kp_old)]
    else:
        kp_new, kp_old, good = [], [], []
    result["matches"] = len(good)
    inlier_matches = []
    if len(good) >= 3:
        mm_new = to_mm([kp_new[m.queryIdx] for m in good], new)
        mm_old = to_mm([kp_old[m.trainIdx] for m in good], old)
        # Shift + rotation (+ a scale that should stay ~1, both views are in mm)
        transform, inliers = cv2.estimateAffinePartial2D(mm_new, mm_old, method=cv2.RANSAC,
                                                         ransacReprojThreshold=RANSAC_MM, maxIters=5000)
        if transform is not None:
            inliers = inliers.ravel().astype(bool)
            inlier_matches = [m for m, keep in zip(good, inliers) if keep]
            result["inliers"] = int(inliers.sum())
            centre = np.array([MARKER_MM / 2, MARKER_MM / 2, 1.0])
            shift = transform @ centre - centre[:2]           # how far the wall moved, at the marker
            result["shift_xy_mm"] = [round(float(shift[0]), 2), round(float(shift[1]), 2)]
            result["shift_mm"] = round(float(np.hypot(*shift)), 2)
            result["rotation_deg"] = round(float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))), 2)
            result["new_to_old_mm"] = np.round(transform, 5).tolist()

    if min(texture) < MIN_FEATURES:
        result["problems"].append("plain_wall")
    elif result["inliers"] < MIN_INLIERS:
        result["problems"].append("not_enough_matches")
    elif result["shift_mm"] > MAX_SHIFT_MM:
        result["problems"].append("marker_moved")
    else:
        result["aligned"] = True
    return result, (kp_new, kp_old, inlier_matches)


def align(new_photo, old_photo, method=None):
    """Compare a new photo with an older photo of the same marker. Both must show the marker.
    method: "orb" or "aliked_lightglue" to force one; default = ORB, and ALIKED + LightGlue when ORB
    finds too few matches (and the models are there)."""
    _, marker_new = check_marker(new_photo)
    _, marker_old = check_marker(old_photo)
    if marker_new is None or marker_old is None:
        return {"method": method or "orb", "aligned": False, "problems": ["no_marker"]}, None
    scales = [float(np.mean(side_lengths(m["corners"]))) / MARKER_MM for m in (marker_new, marker_old)]
    if max(scales) / min(scales) > MAX_SCALE_RATIO:
        return {"method": method or "orb", "aligned": False, "px_per_mm": [round(v, 2) for v in scales],
                "problems": ["scale_too_different"]}, None
    new, old = wall_view(new_photo, marker_new), wall_view(old_photo, marker_old)
    result, drawn = align_views(new, old, method or "orb")
    if method is None and result["problems"] == ["not_enough_matches"] and deep_models() is not None:
        orb_inliers = result["inliers"]
        result, drawn = align_views(new, old, "aliked_lightglue")
        result["orb_inliers"] = orb_inliers           # the trace shows why the network was used
    kp_new, kp_old, matches = drawn
    drawing = cv2.drawMatches(new["gray"], kp_new, old["gray"], kp_old, matches[:300], None,
                              matchColor=(0, 200, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result, drawing


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python align.py new_photo.jpg old_photo.jpg [orb|aliked_lightglue]")
        sys.exit(1)

    result, drawing = align(load_image(sys.argv[1]), load_image(sys.argv[2]), (sys.argv[3:] or [None])[0])
    if drawing is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_matches.jpg"), drawing)
    print(json.dumps(result, indent=2))
