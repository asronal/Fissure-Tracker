"""
Step 1: Photo quality check - is the photo sharp and well lit enough to measure? (OpenCV 5)

A plain wall has almost no texture, so a sharp photo of it can still look "blurry" to a
whole-photo blur score. Instead we use the printed marker as a reference:
  - Sharpness  -> how many pixels its black/white edges are smeared over (sharp photo ~2 px)
  - Lighting   -> its white paper and black ink levels (too dark / washed out / no contrast)
Without a marker we can only check overall brightness.

Usage:
    python quality.py path/to/photo.jpg
"""

import json
import sys

import cv2
import numpy as np

MAX_SIDE = 3000         # keep detail for thin cracks, but don't process giant 48 MP photos
BLUR_MAX_PX = 4.0       # marker edges smeared over more pixels than this = blurry
WHITE_MIN = 70          # marker's white paper darker than this = too dark
BLACK_MAX = 120         # marker's black ink lighter than this = washed out / glare
CONTRAST_MIN = 60       # white - black smaller than this = can't see edges well
PHOTO_DARK = 40         # no marker: mean brightness limits
PHOTO_BRIGHT = 235

TIPS = {
    "blurry": "The photo is blurry. Hold the phone still and tap on the crack to focus.",
    "too_dark": "The photo is too dark. Turn on a light or use daylight; avoid your own shadow on the wall.",
    "overexposed": "The photo is too bright or has glare. Move so the light doesn't reflect off the marker.",
    "low_contrast": "The marker is hard to see clearly. Avoid glare and make sure it is in good light.",
}


def prepare_image(img):
    # Transparent PNG (4 channels): put it on a white background.
    # Otherwise OpenCV turns transparent pixels black.
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype("float32")
        alpha = img[:, :, 3:4].astype("float32") / 255.0
        img = (bgr * alpha + 255.0 * (1 - alpha)).astype("uint8")
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Shrink only really big photos (longest side MAX_SIDE)
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return prepare_image(img)


def decode_image(data):
    """Same as load_image, but for raw file bytes (an uploaded photo)."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not read image bytes")
    return prepare_image(img)


def marker_levels_and_blur(gray, corners):
    """White/black levels inside the marker and how many pixels its edges are smeared over."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    inside = gray[mask > 0]
    black, white = float(np.percentile(inside, 10)), float(np.percentile(inside, 90))
    contrast = white - black

    # Steepest brightness change across the marker's edges. Sobel/8 = brightness change per pixel.
    # A perfectly sharp edge changes by contrast/2 per pixel -> "2 px"; blur spreads it wider.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    magnitude = cv2.magnitude(gx, gy)
    ring = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    steepest = float(np.percentile(magnitude[ring > 0], 99.5))
    blur_px = contrast / steepest if steepest > 0 else float("inf")
    return white, black, contrast, blur_px


def check_quality(img, marker=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    problems = []

    if marker is None:
        brightness = float(gray.mean())
        if brightness < PHOTO_DARK:
            problems.append("too_dark")
        if brightness > PHOTO_BRIGHT:
            problems.append("overexposed")
        return {"measured_on": "photo", "brightness": round(brightness, 1),
                "ok": not problems, "problems": problems}

    white, black, contrast, blur_px = marker_levels_and_blur(gray, marker["corners"])
    if blur_px > BLUR_MAX_PX:
        problems.append("blurry")
    if white < WHITE_MIN:
        problems.append("too_dark")
    elif black > BLACK_MAX:
        problems.append("overexposed")
    elif contrast < CONTRAST_MIN:
        problems.append("low_contrast")

    return {"measured_on": "marker", "blur_px": round(blur_px, 2), "white_level": round(white, 1),
            "black_level": round(black, 1), "ok": not problems, "problems": problems}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quality.py path/to/photo.jpg")
        sys.exit(1)

    from marker import check_marker

    image = load_image(sys.argv[1])
    _, found = check_marker(image)
    print(json.dumps(check_quality(image, found), indent=2))
