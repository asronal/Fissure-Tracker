"""
Step 9: Evaluation - how well does it measure, line up, and decide?

Everything here uses synthetic photos with a known answer (a virtual camera looking at a rendered wall),
plus any real test-sheet photos found in samples/lines_*.jpg. Writes docs/evaluation.md and
docs/evaluation.json.

    python tools/evaluate.py          (about 1-2 minutes)
"""

import glob
import itertools
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "tools")]

from agent import LocalNotifier, handle_photo  # noqa: E402
from align import align, deep_models  # noqa: E402
from crack import measure_photo  # noqa: E402
from eval_lines import measure_lines  # noqa: E402
from quality import load_image, prepare_image  # noqa: E402
from store import LocalStore  # noqa: E402
from synthetic import crack_drawer, crack_path, make_photo, sheet_photo  # noqa: E402
from test_agent import shot  # noqa: E402

WALL, IMAGE = (420.0, 320.0), (2400, 1800)     # big enough to keep the crack in frame at 10 px/mm
CENTRE = (WALL[0] / 2, WALL[1] / 2)
DOCS = os.path.join(ROOT, "docs")


def crack_photo(width_mm, px_per_mm, tilt_deg=0.0, seed=1, darkness=70, draw=None, **kw):
    points = crack_path((CENTRE[0] - 50, CENTRE[1] + 45), 100, seed=seed)
    crack_draw, truth = crack_drawer(points, width_mm, darkness=darkness)
    photo, _ = make_photo(tilt_deg=tilt_deg, distance_mm=0.72 * max(IMAGE) / px_per_mm, image_size=IMAGE,
                          render_px_per_mm=12, wall_mm=WALL, draw=draw or crack_draw, **kw)
    return photo, truth


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


# --- 1. Crack width and length ---------------------------------------------------------------------

def width_accuracy():
    rows = []
    for px_per_mm in (4.0, 6.0, 10.0):
        for tilt in (0.0, 25.0):
            for width in (0.3, 0.5, 1.0, 2.0, 3.0):
                photo, truth = crack_photo(width, px_per_mm, tilt)
                crack = measure_photo(photo)[0].get("crack", {})
                row = {"px_per_mm": px_per_mm, "tilt_deg": tilt, "true_width_mm": round(truth["max_width_mm"], 3),
                       "true_length_mm": round(truth["length_mm"], 1), "found": crack.get("crack_found", False)}
                if row["found"]:
                    row.update(max_width_mm=crack["max_width_mm"], typical_width_mm=crack["typical_width_mm"],
                               uncertainty_mm=crack["uncertainty_mm"],
                               length_mm=crack["length_mm"], too_thin=bool(crack["problems"]),
                               error_mm=round(crack["max_width_mm"] - truth["max_width_mm"], 3))
                rows.append(row)
    return rows


def summarise_widths(rows):
    lines = []
    for px_per_mm in (4.0, 6.0, 10.0):
        mine = [r for r in rows if r["px_per_mm"] == px_per_mm]
        ok = [r for r in mine if r["found"] and not r["too_thin"]]
        errors = np.abs([r["error_mm"] for r in ok]) if ok else np.array([np.nan])
        bias = np.mean([r["error_mm"] for r in ok]) if ok else np.nan
        typical = np.abs([r["typical_width_mm"] - r["true_width_mm"] for r in ok]) if ok else np.array([np.nan])
        within = sum(abs(r["error_mm"]) <= r["uncertainty_mm"] for r in ok)
        len_err = [abs(r["length_mm"] - r["true_length_mm"]) / r["true_length_mm"] * 100 for r in ok]
        thin = [r["true_width_mm"] for r in mine if r["found"] and r["too_thin"]]
        lines.append([f"{px_per_mm:g}", f"±{1 / px_per_mm:.2f}", len(ok), f"{np.mean(errors):.2f}", f"{bias:+.2f}",
                      f"{np.max(errors):.2f}", f"{within}/{len(ok)}", f"{np.mean(typical):.2f}", f"{np.mean(len_err):.1f}%" if len_err else "-",
                      ", ".join(f"{w:g}" for w in sorted(set(thin))) or "none",
                      sum(not r["found"] for r in mine)])
    return table(["px/mm", "stated uncertainty (mm)", "cracks measured", "max width: mean abs error (mm)",
                  "mean bias (mm)", "max abs error (mm)", "within uncertainty", "typical width: mean abs error (mm)",
                  "mean length error",
                  "flagged too thin -> close-up (true mm)", "missed"], lines)


# --- 2. Printed test sheet ------------------------------------------------------------------------

def test_sheet():
    views = [(d, t) for d in (220.0, 300.0, 400.0) for t in (0.0, 25.0)]
    per_line = {}
    for distance, tilt in views:
        photo, truth = sheet_photo(tilt_deg=tilt, distance_mm=distance)
        rows, _ = measure_lines(photo)
        for row, true_width in zip(rows, truth):
            entry = per_line.setdefault(row["printed_mm"], {"errors": [], "thin": 0, "missed": 0, "true": true_width})
            if not row["found"]:
                entry["missed"] += 1
            elif row["problems"]:
                entry["thin"] += 1
            else:
                entry["errors"].append(row["typical_width_mm"] - true_width)
    out = []
    for printed, e in per_line.items():
        errs = np.abs(e["errors"])
        out.append([f"{printed:g}", f"{e['true']:.2f}", f"{len(e['errors'])}/{len(views)}",
                    f"{errs.mean():.2f}" if len(errs) else "-", f"{errs.max():.2f}" if len(errs) else "-", e["thin"], e["missed"]])
    return table(["printed width (mm)", "true painted (mm)", "measured", "mean abs error (mm)", "max abs error (mm)",
                  "too thin -> close-up", "missed"], out), len(views)


def real_sheet_photos():
    paths = sorted(glob.glob(os.path.join(ROOT, "samples", "lines_*.jp*g")))
    out = []
    for path in paths:
        rows, problem = measure_lines(load_image(path))
        for row in rows:
            if row["found"] and not row["problems"]:
                out.append([os.path.basename(path), f"{row['printed_mm']:g}", f"{row['typical_width_mm']:.2f}",
                            f"{row['error_mm']:+.2f}", f"±{row['uncertainty_mm']:.2f}"])
        if problem:
            out.append([os.path.basename(path), "-", "-", problem, "-"])
    return table(["photo", "printed (mm)", "measured (mm)", "error (mm)", "uncertainty"], out) if out else None


# --- 3. Repeatability ---------------------------------------------------------------------------

def repeatability():
    views = [(6.0, t) for t in (-20.0, -10.0, 0.0, 10.0, 20.0)] + [(4.0, 0.0), (8.0, 0.0), (6.0, 15.0)]
    values, truth = [], None
    for px_per_mm, tilt in views:
        photo, t = crack_photo(1.0, px_per_mm, tilt)
        truth = t["max_width_mm"]
        values.append(measure_photo(photo)[0]["crack"]["max_width_mm"])
    return {"true_mm": round(truth, 3), "views": len(views), "mean_mm": round(float(np.mean(values)), 3),
            "std_mm": round(float(np.std(values)), 3), "min_mm": min(values), "max_mm": max(values)}


# --- 4. Alignment ---------------------------------------------------------------------------------

def alignment():
    pairs = [("same view", {}, {}), ("closer + 20° to the side", dict(tilt_deg=20, distance_mm=250), {}),
             ("-25° vs +15°", dict(tilt_deg=-25, distance_mm=350), dict(tilt_deg=15)),
             ("450 mm vs 250 mm", dict(distance_mm=450), dict(distance_mm=250)),
             ("another wall", dict(seed=7), {})]
    views = []
    for name, new, old in pairs:
        r, _ = align(make_photo(**new)[0], make_photo(**old)[0])
        views.append([name, r["method"], r.get("inliers", "-"), "-" if r.get("shift_mm") is None else r["shift_mm"], ", ".join(r["problems"]) or "aligned"])
    moved = []
    for offset in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        r, _ = align(make_photo(marker_offset_mm=(offset, 0.0))[0], make_photo()[0])
        moved.append([f"{offset:g}", f"{r['shift_mm']:.2f}", "marker_moved" if "marker_moved" in r["problems"] else "-"])
    started = time.time()
    for _ in range(3):
        align(make_photo(tilt_deg=10)[0], make_photo()[0])
    return (table(["photo pair", "method", "matched wall features", "wall shift (mm)", "result"], views),
            table(["marker really moved (mm)", "measured shift (mm)", "flagged (> 3 mm)"], moved),
            round((time.time() - started) / 3, 2))


def real_alignment():
    """ORB vs OpenCV 5 ALIKED + LightGlue on every pair of real photos taken in the same round (the sheet
    stayed taped in place within a round, so the right answer is always "lined up")."""
    if deep_models() is None:
        return None
    rounds = [("taped flat", sorted(glob.glob(os.path.join(ROOT, "samples", "lines_*.jp*g")))),
              ("curled sheet", sorted(glob.glob(os.path.join(ROOT, "samples", "first_try_curled", "lines_*.jp*g"))))]
    rows, seconds = [], {"orb": [], "aliked_lightglue": []}
    for name, paths in rounds:
        photos = [load_image(p) for p in paths]
        counts = {m: {"aligned": 0, "marker_moved": 0, "not_enough_matches": 0} for m in ("orb", "aliked_lightglue", "auto")}
        pairs = 0
        for a, b in itertools.combinations(photos, 2):
            results = {}
            for method in ("orb", "aliked_lightglue"):
                started = time.time()
                results[method], _ = align(a, b, method)
                seconds[method].append(time.time() - started)
            if results["orb"]["problems"] == ["scale_too_different"]:
                continue
            pairs += 1
            orb = results["orb"]
            results["auto"] = results["aliked_lightglue"] if orb["problems"] == ["not_enough_matches"] else orb
            for method, r in results.items():
                counts[method]["aligned" if r["aligned"] else r["problems"][0]] += 1
        if pairs:
            for method, label in (("orb", "ORB"), ("aliked_lightglue", "ALIKED + LightGlue"),
                                  ("auto", "**ORB, then ALIKED + LightGlue (the app)**")):
                c = counts[method]
                rows.append([f"{name} ({len(paths)} photos)", pairs, label, c["aligned"], c["marker_moved"], c["not_enough_matches"]])
    return (table(["photos", "pairs", "method", "lined up (right)", "false 'marker moved'", "not enough matches"], rows),
            {m: round(float(np.mean(v)), 1) for m, v in seconds.items()})


# --- 5. Agent scenarios: different OpenCV results -> different actions ------------------------------

SCENARIOS = [
    ("First photo of a crack", [], dict()),
    ("Same crack a week later (other angle)", [dict()], dict(tilt_deg=15)),
    ("Blurry photo", [], dict(px_per_mm=6, blur_sigma=3)),
    ("Marker not in the photo", [], dict(marker=False)),
    ("0.3 mm crack from far away", [], dict(width_mm=0.3)),
    ("Crack grew 1.0 -> 1.8 mm", [dict(width_mm=1.0)], dict(width_mm=1.8)),
    ("...then the requested close-up", "CONTINUE", dict(width_mm=1.8, px_per_mm=10)),
    ("2.5 mm crack, precise photo", [], dict(width_mm=2.5, px_per_mm=6)),
    ("Marker re-stuck 9 mm away", [dict()], dict(offset=(8.0, 5.0))),
    ("Photo of a different wall", [dict()], dict(seed=7)),
]


def agent_scenarios():
    rows, times, case = [], [], None
    for name, history, photo in SCENARIOS:
        if history != "CONTINUE":
            store = LocalStore(tempfile.mkdtemp())
            for h in history:
                handle_photo(store, shot(**h), notifier=LocalNotifier())
        started = time.time()
        case = handle_photo(store, shot(**photo), case["case_id"] if history == "CONTINUE" else None, LocalNotifier())
        times.append(time.time() - started)
        last = case["photos"][-1]["photo_id"]
        tools = [s["tool"] + (f"({s['input']['kind']})" if s["input"].get("kind") else "") for s in case["trace"]
                 if s["photo_id"] == last]
        rows.append([name, " → ".join(tools), case["status"], case["owner_message"][:110]])
    return table(["situation", "tool calls (rules backup)", "result", "message to the owner"], rows), float(np.mean(times))


# --- 6. Failure cases -----------------------------------------------------------------------------

def failure_cases():
    out = []

    def pencil(wall, s):      # a straight drawn line, not a crack
        cv2.line(wall, (int((CENTRE[0] - 50) * s), int((CENTRE[1] + 45) * s)),
                 (int((CENTRE[0] + 50) * s), int((CENTRE[1] + 50) * s)), 90, max(1, int(0.5 * s)))
    crack = measure_photo(crack_photo(0, 6.0, draw=pencil)[0])[0]["crack"]
    out.append(["Straight pencil line (no crack)", "measured as a 'crack' "
                f"{crack.get('max_width_mm', '-')} mm" if crack["crack_found"] else "ignored",
                "Limitation: colour/shape can't tell a drawn line from a crack. The owner is told what was measured "
                "and the marked-up photo shows it."])

    def stain(wall, s):       # a dark blotch
        cv2.circle(wall, (int(CENTRE[0] * s), int((CENTRE[1] + 60) * s)), int(12 * s), 110, -1)
    crack = measure_photo(crack_photo(0, 6.0, draw=stain)[0])[0]["crack"]
    out.append(["Round dark stain (24 mm)", "ignored" if not crack["crack_found"] else "measured as a crack",
                "Wide dark areas and their rims are blanked out before looking for cracks." if not crack["crack_found"]
                else "Stain rims can look like thin dark arcs."])

    points = crack_path((CENTRE[0] - 50, CENTRE[1] + 45), 100, seed=1)
    crack_draw, _ = crack_drawer(points, 1.0)

    def crack_and_stain(wall, s):
        cv2.circle(wall, (int(CENTRE[0] * s), int((CENTRE[1] + 46) * s)), int(12 * s), 110, -1)
        crack_draw(wall, s)
    crack = measure_photo(crack_photo(0, 6.0, draw=crack_and_stain)[0])[0]["crack"]
    out.append(["1.0 mm crack running through that stain",
                f"found, {crack['max_width_mm']} mm wide, {crack['length_mm']:.0f} of ~100 mm long" if crack["crack_found"] else "missed",
                "The part inside the stain is not measured (limitation)."])

    photo, _ = crack_photo(0.4, 6.0, darkness=175)     # wall is ~200: only 25 grey levels darker
    crack = measure_photo(photo)[0]["crack"]
    out.append(["Faint 0.4 mm hairline (25 grey levels darker than the wall)",
                f"found, {crack['max_width_mm']} mm" + (" (flagged too thin -> close-up)" if crack["problems"] else "")
                if crack["crack_found"] else "missed",
                "Faint, thin cracks read too wide: the agent's close-up loop exists for this."])

    result = measure_photo(crack_photo(1.0, 6.0, tilt_deg=40)[0])[0]["marker"]
    out.append(["Photo taken 40° from the side", ", ".join(result["problems"]) or "accepted",
                "Agent asks for a straight-on retake."])

    plain = make_photo(seed=3)[0]
    plain[:] = np.where(plain < 128, plain, 205).astype(np.uint8)
    out.append(["Plain wall with no texture (alignment)", ", ".join(align(plain, plain.copy())[0]["problems"]),
                "Alignment skipped, widths still compared; not blamed on the owner."])
    out.append(["Close-up vs normal photo (alignment)",
                ", ".join(align(make_photo(distance_mm=90)[0], make_photo(distance_mm=300)[0])[0]["problems"]),
                "Agent lines up with the latest photo of similar scale instead."])
    return table(["case", "what happened", "notes"], out)


# --- 7. Speed -------------------------------------------------------------------------------------

def speed():
    photo, _ = crack_photo(1.0, 6.0)
    started = time.time()
    for _ in range(3):
        measure_photo(photo)
    small = (time.time() - started) / 3
    big = prepare_image(cv2.resize(photo, (4000, 3000)))     # a 12 MP phone photo (shrunk to 3000 px)
    started = time.time()
    measure_photo(big)
    return round(small, 2), round(time.time() - started, 2), big.shape[1::-1]


def main():
    os.makedirs(DOCS, exist_ok=True)
    started = time.time()
    print("1/7 crack width...")
    widths = width_accuracy()
    print("2/7 test sheet...")
    sheet_table, sheet_views = test_sheet()
    real = real_sheet_photos()
    print("3/7 repeatability...")
    repeat = repeatability()
    print("4/7 alignment...")
    align_table, moved_table, align_s = alignment()
    print("4b/7 alignment on real photos (ORB vs ALIKED + LightGlue)...")
    real_align = real_alignment()
    print("5/7 agent scenarios...")
    agent_table, agent_s = agent_scenarios()
    print("6/7 failure cases...")
    failures = failure_cases()
    print("7/7 speed...")
    small_s, big_s, big_size = speed()

    md = f"""# Evaluation

Generated by `python tools/evaluate.py` on {time.strftime('%d %b %Y')} in {time.time() - started:.0f} s.

**How:** almost everything here uses **synthetic photos with a known answer**: a rendered plaster wall with a
crack of known width, photographed by a virtual phone camera (focal length 0.72 × image width) from a known
distance and angle. This tests the geometry and the measurement honestly, but real photos add lens distortion,
uneven light and real wall texture. Real-photo results are listed separately below.

Uncertainty = ±1 pixel in the straight-on view (1 / px per mm). "px/mm" is how many pixels one millimetre
of wall covers in the photo: about 4 at 30 cm with a 1600 px-wide photo, about 10 for a close-up.

## 1. Crack width and length (synthetic, {len(widths)} photos)

Wiggly cracks of 0.3–3 mm, photographed straight on and 25° from the side, at three resolutions (photo
2400 × 1800 px). Cracks narrower than 4 pixels are flagged so the agent asks for a close-up.

**Max width reads slightly high** (positive bias): it is the largest of many local width estimates along the
crack, and noise pushes the largest one up. The bias is consistent between photos (see repeatability), so it
cancels out when this week's maximum is compared with last week's. The typical (median) width is close to
the truth.

{summarise_widths(widths)}

## 2. Printed test sheet (synthetic photos of `sheets/test_lines.pdf`, {sheet_views} views)

Distances 220 / 300 / 400 mm × straight on and 25°, photo 2250 × 3000 px. Each line is measured with the
same pipeline as cracks (typical width along the line).

{sheet_table}

### Real photos of the printed sheet

{real or "_No real photos yet: put them in `samples/lines_*.jpg` and run `python tools/eval_lines.py samples/lines_*.jpg`._"}

## 3. Repeatability

The same 1.0 mm crack ({repeat['true_mm']:.2f} mm as rendered) from {repeat['views']} different viewpoints
(−20° to +20°, 4–8 px/mm): mean **{repeat['mean_mm']:.2f} mm**, standard deviation **{repeat['std_mm']:.2f} mm**,
range {repeat['min_mm']:.2f}–{repeat['max_mm']:.2f} mm.

## 4. Lining up with last week's photo (ORB, then OpenCV 5 ALIKED + LightGlue)

The app tries ORB first and uses OpenCV 5's ALIKED + LightGlue networks only when ORB finds too few
matches. On these synthetic walls ORB always succeeds or rightly fails, so the method column shows ORB
except where the network was tried.

{align_table}

Marker moved between photos (the wall shifts against the marker, so old and new widths aren't comparable):

{moved_table}

About {align_s:.1f} s per photo pair on a laptop.

### Real photos: ORB vs ALIKED + LightGlue

{real_align[0] if real_align else "_ALIKED + LightGlue models not downloaded: run `python tools/get_models.py`._"}

{f"Time per pair on a laptop: ORB {real_align[1]['orb']} s, ALIKED + LightGlue {real_align[1]['aliked_lightglue']} s." if real_align else ""}
ORB is precise when the photos are similar; ALIKED + LightGlue finds the wall when the angle or the curled
sheet defeats ORB, but works on a 640 × 640 input, so its shift is less precise and taken alone it raises
more false "marker moved" flags. Using it only as ORB's backup keeps the best of both.

## 5. Agent decisions: different OpenCV results lead to different tool calls

Each row is a real run of the agent (rules backup; the AI model gets the same tools and the same safety rules).

{agent_table}

Average time per photo for the whole agent turn: {agent_s:.1f} s (laptop, rules backup).

## 6. Failure cases

{failures}

## 7. Speed (laptop CPU)

- Measuring one 2400 × 1800 photo (marker, quality, straight-on view, crack): **{small_s:.2f} s**
- A 12 MP phone photo (shrunk to {big_size[0]} × {big_size[1]}): **{big_s:.2f} s**

## Limits of this evaluation

- Synthetic photos have no lens distortion and an assumed focal length; the tilt estimate depends on it.
- The synthetic wall texture is random grain; real walls have pores, paint and dirt.
- Real crack growth takes weeks: growth was simulated by drawing a wider crack.
- The decision thresholds (2 mm alert, 0.2 mm growth) are project settings, not engineering standards.
"""
    with open(os.path.join(DOCS, "evaluation.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(DOCS, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump({"widths": widths, "repeatability": repeat}, f, indent=1)
    print(f"Wrote docs/evaluation.md ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
