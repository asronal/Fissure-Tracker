# Fissure Tracker: technical report on a wall-crack growth monitor with an OpenCV 5 agent

**OpenCV AI Competition 2026 (powered by AWS): Agentic Vision · Team: Asron, Anuj**

> Fill in before submitting: live endpoint `https://hx64dq2waacw4s76cojqxw3d3q0kjien.lambda-url.us-east-1.on.aws/` · Bedrock model live: `[yes / rules backup only]`

## 1. Problem

Cracks in walls are common. Most are cosmetic, but a crack that **keeps widening** can mean settlement or
structural movement. The usual professional advice is to *monitor* it: measure its width every few weeks and
call a structural engineer if it grows (BRE Digest 251 grades damage mainly by crack width).

In practice:
- **Owners don't measure.** Crack-width gauges and tell-tales cost money and need care; a phone photo is
  easy but has no scale, and two photos from different angles can't be compared.
- **Engineers get called too late or too often.** Without numbers, owners either ignore a crack or call
  someone for every hairline.

**Goal:** turn a weekly phone photo into a measured, comparable reading in millimetres, and involve an engineer
only when the measurements justify it, with the engineer's approval before the owner is told to act.

## 2. Users

| User | Needs |
|---|---|
| Building owner / tenant / caretaker | Takes the weekly photo; needs clear instructions when a photo isn't good enough, and a plain answer |
| Structural engineer | Wants to be contacted only for cracks that matter, with the evidence (photos, numbers, history) in one link |

## 3. How it works

1. **Marker.** The owner prints a 60 mm ArUco marker (IDs 1–6, one per crack) and sticks it next to the
   crack. It gives the photo a scale (mm per pixel), a viewing angle, and an identity, so the app knows which
   crack this is without any typing. No printer: the marker can be drawn on 1 cm graph paper.
2. **Photo.** Once a week, a phone browser opens the web page and takes a photo of the crack with the marker;
   the page shrinks it to ≤ 2560 px before upload.
3. **Agent turn.** The agent calls OpenCV 5 tools and decides one of: retake, close-up, save, or engineer review.
4. **Case page.** The owner sees the result, the marked-up photos and every agent step. If the agent asked for
   a photo, the owner sends it on the same page.
5. **Engineer.** If needed, the engineer gets an SNS e-mail with a one-time link, reviews the evidence and
   presses *Inspection needed* or *No action needed*; the owner sees the answer.

![Architecture](architecture.png)

## 4. OpenCV 5 pipeline

All image analysis uses `opencv-python-headless==5.0.0.93` (numpy 2.4.6). There is no GPU, and a photo takes ~0.4 s.

### 4.1 Marker, scale and viewing angle (`marker.py`)
- `cv2.aruco.ArucoDetector` with `DICT_4X4_50` and sub-pixel corner refinement. If several markers are seen,
  the biggest wins. **Marker ID = crack ID.**
- Scale: mean side length / 60 mm → px per mm.
- Tilt: from the homography between the marker corners and a flat square. With an assumed phone focal length
  (0.72 × the long image side), `K⁻¹H = [r1 r2 t]` and the wall normal = r1 × r2. (OpenCV 5 removed
  `estimatePoseSingleMarkers`.)
- Straight-on view: `getPerspectiveTransform` + `warpPerspective` to a view where every millimetre of wall is
  the same number of pixels, never enlarged beyond the photo's real resolution. A validity mask marks
  where the photo really has data.
- Problems reported: `no_marker`, `marker_too_small` (< 80 px), `too_tilted` (> 30°).

### 4.2 Photo quality judged on the marker (`quality.py`)
The marker is a known black-and-white target, so quality is measured there rather than guessed from the whole
photo. **Blur** = contrast / steepest edge gradient (Sobel), i.e. how many pixels an edge smears over: sharp ≈
2 px, rejected > 4 px. **Light** = the marker's white and black levels (too dark, glare, low contrast).
Each problem has a specific tip for the owner.

### 4.3 Crack width and length (`crack.py`)
1. **Black-hat** (`morphologyEx`, rectangular kernel ≈ 2 × the widest expected crack) keeps thin dark lines
   and removes slow lighting changes.
2. **Threshold from the wall's own noise:** median + 6 × robust spread of the black-hat response (minimum 12 grey
   levels).
3. **Wide dark areas** (stains, shadows) are found by comparing with a 3× larger closing; they and a rim around
   them are blanked, because a stain's curved edge otherwise looks like a thin dark arc.
4. **Long and thin only:** `connectedComponentsWithStats`, length ≥ 10 mm and area ≤ 0.35 × length².
5. **Width edge at half-darkness:** a pixel belongs to the crack if its darkness is ≥ half the local peak. This
   makes the width independent of blur for cracks wider than the blur.
6. **Centre-line:** Zhang-Suen thinning (own implementation; `ximgproc` isn't in the pip build). The main path
   of each piece is found by a Dijkstra double sweep, and its smoothed length is the crack length.
7. **Width along the crack** = local area / local centre-line length in a window. **Max width** = maximum of a
   running median with the ends trimmed. **Typical width** = median.
8. **Uncertainty** = ±1 px in the straight-on view. A crack narrower than 4 px is flagged
   `crack_too_thin_for_this_distance`: it isn't guessed.
9. The widest point is reported in mm from the marker's corner, so it is the same spot in every photo, and is
   drawn on the photo.

### 4.4 Lining up with last week (`align.py`)
Both photos are straightened into the marker's mm frame at a common 3 px/mm, where the same wall point
should land on the same mm position **if the marker hasn't moved**.
- CLAHE brings out faint plaster texture.
- **First try, ORB:** features with fast threshold 7 (the marker itself masked out, because it always
  "matches"), `BFMatcher` + Lowe ratio test. Fast (0.3 s) and precise when the two photos are similar.
- **If ORB finds too few matches, OpenCV 5's learned features take over:** `cv2.ALIKED` keypoints and
  descriptors, matched by `cv2.LightGlueMatcher`, which also uses where the keypoints are
  (`setPairInfo`). Both are ONNX networks run by OpenCV 5's new DNN engine, no PyTorch. They cope with a
  different angle, light or a curled sheet, but the model works on a 640 × 640 input, so its shift is less
  precise (~1–2 mm) and it takes ~3 s. Everything outside the wall mask is greyed out before the network
  sees it: otherwise it "recognises" the marker's surroundings and lines up a different wall.
- Either way, `estimateAffinePartial2D` with RANSAC (1.5 mm threshold) in mm gives the shift and rotation.
- The trace says which method was used and why ("ORB found only 15 matches, so OpenCV 5's ALIKED +
  LightGlue networks were used"); `/health` reports whether the networks are loaded.

**Outputs:**
- inlier count
- wall shift and rotation at the marker
- `not_enough_matches` (< 25 inliers: probably a different spot)
- `marker_moved` (> 3 mm)
- `plain_wall` (too little texture: alignment skipped, not the owner's fault)
- `scale_too_different` (close-up vs normal photo: the agent compares with the latest photo of similar scale instead)

**Why ORB first and not the network first** (measured on our real photos, §6.2): alone, ALIKED + LightGlue
raised 6 false "marker moved" flags on 21 pairs of the flat sheet, where ORB raised 1. On the curled sheet,
ORB found too few matches on 4 of 13 pairs and the network rescued 3 of them. ORB first, network as backup,
got the best of both: 20/21 and 12/13 lined up.

**On AWS** the two models (52 MB) don't fit in the Lambda layer next to OpenCV, so `deploy.py` puts them in
the private bucket and the function copies them to `/tmp` once per container (cold start ~4 s).

## 5. The agent

![Agent workflow](agent_workflow.png)

### 5.1 Tools
| Tool | Kind | Returns / does |
|---|---|---|
| `check_photo` | perception | marker ID, px/mm, tilt, blur, light, problems (also: wrong marker for this case, test-sheet marker) |
| `measure_crack` | perception | max and typical width, uncertainty, length, BRE category, widest point and where it is ("about 5 cm below the marker's centre") |
| `compare_with_history` | perception | alignment result, last reading, growth ± combined uncertainty, `growing`, `narrower_than_before`, trend |
| `ask_for_photo(kind, message)` | action, ends the turn | case waits for a retake or close-up |
| `save_reading(message)` | action, ends the turn | reading stored as history |
| `request_engineer_review(reason, message)` | action, ends the turn | reading stored as `needs_review`; SNS e-mail with a one-time link |

### 5.2 Loop
Amazon Bedrock **Converse API with tool use**, model `us.amazon.nova-2-lite-v1:0`, `maxTokens` 700, temperature 0.
The model gets the case context (which photo of the case this is, what was asked for before) and calls tools
one at a time. Tool results go back as JSON text; the model's own text is kept as a "thought" in the trace.
Limits: 8 tool calls and 45 s per photo.

### 5.3 Safety rules in code (failure handling and human control)
The model is free to choose, but some choices are blocked in code and returned to the model as an error,
which it must react to:
- `check_photo` must come first.
- `save_reading` only after a clean measurement and a history comparison, and **never** when the numbers
  need an engineer or a confirming close-up.
- After 4 photos in a case, `ask_for_photo` is refused and the case goes to the engineer.
- Unknown tools or bad arguments are refused; model-written messages are cleaned and length-limited.

If Bedrock fails (no network, quota, timeout, bad answer) or the model doesn't finish within its limits, a
**rules backup** calls the same tools in a fixed order and applies the same decisions. The trace records the
fallback and its reason. The owner always gets an answer.

### 5.4 Decision thresholds (project settings, not building codes)
- Engineer review if max width ≥ 2.0 mm, growth > max(0.2 mm, combined uncertainty), narrower than before by
  the same margin (repair? moved marker? bad photo?), or marker moved > 3 mm.
- Close-up first if the measurement uncertainty is > 0.2 mm and no close-up was asked for yet in this case.

### 5.5 Why this is agentic
The vision results decide the **next tool call**, not just the wording of an answer. One photo of a crack can
lead to six different paths (table in section 6.4). A close-up request is aimed at a location measured by OpenCV.
The loop continues across several photos in one case, and a human approval step gates the outcome.
[trace_example.md](trace_example.md) shows a full case:
1. A coarse growth reading triggers a close-up instead of a save.
2. The close-up confirms the growth.
3. That triggers the engineer review.

## 6. Evaluation

Full tables: [evaluation.md](evaluation.md), produced by `tools/evaluate.py`.

### 6.1 Synthetic photos with a known answer
A virtual phone camera photographs a rendered plaster wall with cracks or printed lines of known width,
from known distances and angles.
- **Printed test lines 1–3 mm** (220–400 mm away, 0° and 25°): mean absolute error 0.04–0.05 mm, max 0.09 mm.
  The 0.2–0.5 mm lines were correctly flagged as too thin at those distances.
- **Wiggly cracks 0.3–3 mm** (4, 6, 10 px/mm, 0° and 25°):
  - Typical width within 0.05 mm on average.
  - Max width has a consistent positive bias (+0.13 mm), partly from how anti-aliased synthetic cracks are drawn.
  - Repeatability over 8 viewpoints: standard deviation **0.02 mm**. So week-to-week growth (same method both weeks) is measured fairly.
  - Length within 2–5%.
- **Alignment:**
  - −25° vs +15° and 250 vs 450 mm still align (wall shift < 0.3 mm).
  - Another wall gets 2 inliers with ORB and 0 with ALIKED + LightGlue (after greying out the marker area).
  - A moved marker is measured within 0.1 mm and flagged from 4 mm.

### 6.2 Real photos
`tools/eval_lines.py` measures photos of the printed test sheet (`sheets/test_lines.pdf`).
Seven iPhone photos (3024 × 4032 px) of the test sheet taped to a rough plaster wall, in the evening under
artificial light, 10–35° to the side. The sheet was printed at 94% scale by mistake; this doesn't change the
test, because the marker and the lines shrink together. Full table: `docs/evaluation.md`.

| Result | Lines |
|---|---|
| Measured (thick enough for the distance) | 13 lines, 0.3–3 mm: mean error **0.05 mm**, worst 0.11 mm, every one inside its stated uncertainty (±0.12–0.16 mm) |
| The two flattest views (10° and 17°) | 0.5, 1, 2 and 3 mm all within **0.07 mm** (3 mm read 2.94 and 3.01) |
| Too thin for the distance (0.2 mm, some 0.3 mm) | flagged, not measured; in the app this asks for a close-up |
| Not found | lower lines (1–3 mm, 10–20 cm from the marker) in the three photos taken at 21–35° |
| Not measurable | two close-ups, where the lines run off the photo |

**Why lines were missed:** the wall is very uneven, so even taped at four corners the paper sat on bumps and
the marker was at a slightly different angle from the lower part of the page. The marker corners were found
correctly, but the straightened view drifted by up to about 1 cm at 15–20 cm from the marker, so the tool
looked for the line in the wrong place. A first set of six photos, with the sheet taped only at the top and
curling, failed the same way and isn't counted. **Lesson for real use:** the marker must lie flat on the same
surface as the crack and close to it (a few cm, as the instructions say); the measurement is only as good as
that plane.

**Alignment on real photos (ORB vs OpenCV 5 ALIKED + LightGlue).** Every pair of photos from the same round
(the sheet stayed in place, so the right answer is always "lined up"):

| Photos | Pairs | ORB | ALIKED + LightGlue | ORB, then ALIKED + LightGlue (the app) |
|---|---|---|---|---|
| Taped flat (7) | 21 | 20 lined up, 1 false "moved" | 15 lined up, 6 false "moved" | **20 lined up, 1 false "moved"** |
| Curled sheet (6) | 13 | 9 lined up, 4 not enough matches | 9 lined up, 3 false "moved", 1 not enough | **12 lined up, 1 not enough** |

### 6.3 Failure cases
| Case | Result |
|---|---|
| Straight pencil line | measured as a "crack" (limitation; the marked-up photo shows what was measured) |
| 24 mm dark stain | ignored |
| Crack running through the stain | found with the right width; the part inside the stain isn't measured |
| Faint 0.4 mm hairline | found but reads wide, and flagged too thin → close-up |
| 40° photo angle | rejected → straight-on retake |
| Paper not flat on a rough wall (real photos) | lines 10–20 cm from the marker are missed (§6.2) |
| Plain wall, close-up vs normal photo | alignment skipped with a reason; widths still compared |

### 6.4 Agent scenarios (same photo pipeline, different OpenCV results → different tool calls)
| Situation | Tool calls | Result |
|---|---|---|
| First photo | check → measure → compare → save_reading | baseline saved |
| A week later, other angle | check → measure → compare → save_reading | stable |
| Blurry | check → ask_for_photo(retake) | "hold still and tap to focus" |
| No marker | check → ask_for_photo(retake) | "include the marker" |
| 0.3 mm crack from far | check → measure → ask_for_photo(closeup) | close-up at the widest point |
| Grew 1.0 → 1.8 mm (coarse) | check → measure → compare → ask_for_photo(closeup) | confirm first |
| …the close-up | check → measure → compare → request_engineer_review | engineer e-mailed |
| 2.5 mm, precise photo | … → request_engineer_review | engineer e-mailed |
| Marker re-stuck elsewhere | … → request_engineer_review | engineer e-mailed |
| Different wall | check → measure → compare → ask_for_photo(retake) | "stand where you stood" |

### 6.5 Speed and cost
- 0.4–0.6 s per photo for OpenCV on a laptop CPU; ~0.7 s per agent turn with the rules backup.
- On AWS, roughly 1–2 cents per photo, mostly Bedrock; the other services stay within the free tier at demo scale.

## 7. AWS

| Service | Role |
|---|---|
| **AWS Lambda** (Python 3.13, 2 GB, 90 s) + **Function URL** | the whole web app and agent; OpenCV 5 + numpy as a Lambda layer (76 MB zipped, 209 MB unzipped) |
| **Amazon Bedrock** (Nova 2 Lite, Converse tool use) | the agent's decisions |
| **Amazon DynamoDB** (on demand) | cases with full traces, readings (history per crack) |
| **Amazon S3** (private) | original, marked-up and match photos; 1-hour pre-signed links |
| **Amazon SNS** | engineer e-mail with the review link |
| **IAM** | one role: its photo folder, its table, its topic, the one Bedrock model |
| **CloudWatch Logs** (14 days), **AWS Budgets** ($20, counted before credits) | operations and cost control |

**Responsible cloud use:**
- No servers or GPUs; everything scales to zero.
- One-command deploy and one-command delete (`deploy/deploy.py`, `deploy/destroy.py`).
- The account ID is never printed or used in names.
- Least-privilege IAM.
- Upload cap, model step and time caps, and a budget alert.

## 8. Responsible use and limitations

- **Not an engineering assessment.** The app measures; an engineer decides. The owner is never told a crack is
  "safe": only that it hasn't changed, or that an engineer is looking at it. Thresholds are settings.
- **Privacy:**
  - No accounts or personal data.
  - Private bucket and short-lived photo links.
  - Access code for uploads.
  - Unguessable case IDs.
  - A one-time engineer link that only a button press (POST) can use, so mail scanners can't approve.
- **Data rights:** all evaluation images are synthetic; the only real photos are our own. The ALIKED
  (BSD-3-Clause) and LightGlue (Apache-2.0) networks are the ONNX exports OpenCV's own tests use; they are
  downloaded by `tools/get_models.py` and not redistributed in the repository.
- **Limitations:**
  - Max width reads ~0.1 mm high, and cracks under ~0.3 mm need a close-up.
  - Drawn lines and straight shadows look like cracks.
  - Cracks inside dark stains are only measured outside them.
  - Tilt assumes a typical phone lens.
  - Alignment needs wall texture.
  - Real growth takes weeks, so the demo simulates it.

## 9. Future work

- ALIKED + LightGlue at full resolution (tiles, or a larger ONNX export) so the network is as precise as ORB
  and could go first.
- Lens calibration from a printed ChArUco sheet for better tilt and width at the image edges.
- A fixed Raspberry Pi 4 + Camera Module 3 station per crack: the agent schedules captures, refocuses and
  photographs more often when a crack grows.
- A learned crack segmenter as an extra tool, used when the classic pipeline is unsure.
- Temperature and season notes: cracks open and close slightly with the seasons.

## 10. Team

Asron (project lead, AWS) and Anuj (computer vision / AI).
