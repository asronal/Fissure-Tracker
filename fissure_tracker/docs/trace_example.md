# Agent trace example: crack #1

Generated with `python tools/make_demo.py --ai` (synthetic photos; crack #1 slowly widening over five weeks).
Each step is a tool call; `decided_by` says who chose it. **OpenCV's output changes the next tool call:** in the
earlier weeks `compare_with_history` showed no real change and the agent called `save_reading`; this week it
reports growth from 1.49 to 2.14 mm, so the agent does not save the reading.

The same photos are shown twice, because the two deciders take different (both safe) paths:

1. **Amazon Nova 2 Lite** (the live model): growth beyond the uncertainty → asks a structural engineer straight away.
2. **Rules backup** (used when the model is unavailable): the measurement is coarse, so it first asks for a
   close-up at the widest point, and asks the engineer only after the close-up confirms the growth.

## 1. Amazon Nova 2 Lite

### 1. `check_photo` — decided by us.amazon.nova-2-lite-v1:0 (photo p20261027T171401474-f9d5a6)

```json
{
 "marker_found": true,
 "problems": [],
 "marker_id": 1,
 "px_per_mm": 3.99,
 "tilt_deg": 5.0,
 "crack_marker_expected": null,
 "blur_px": 2.11,
 "white_level": 255.0,
 "black_level": 0.0
}
```

### 2. `measure_crack` — decided by us.amazon.nova-2-lite-v1:0 (photo p20261027T171401474-f9d5a6)

```json
{
 "crack_found": true,
 "max_width_mm": 2.14,
 "typical_width_mm": 2.0,
 "length_mm": 100.4,
 "uncertainty_mm": 0.25,
 "widest_point_mm": [
  45.9,
  78.2
 ],
 "bre_category": 2,
 "bre_label": "Slight",
 "problems": [],
 "widest_point_where": "about 2 cm to the right and 5 cm below the marker's centre"
}
```

### 3. `compare_with_history` — decided by us.amazon.nova-2-lite-v1:0 (photo p20261027T171401474-f9d5a6)

```json
{
 "readings": 5,
 "last_reading": {
  "time": "2026-10-20T04:30:04.456+00:00",
  "max_width_mm": 1.49,
  "uncertainty_mm": 0.25
 },
 "trend": [
  {
   "date": "2026-09-22",
   "max_width_mm": 1.12
  },
  {
   "date": "2026-09-29",
   "max_width_mm": 1.23
  },
  {
   "date": "2026-10-06",
   "max_width_mm": 1.26
  },
  {
   "date": "2026-10-13",
   "max_width_mm": 1.29
  },
  {
   "date": "2026-10-20",
   "max_width_mm": 1.49
  }
 ],
 "alignment": {
  "method": "orb",
  "orb_inliers": null,
  "inliers": 45,
  "shift_mm": 0.15,
  "rotation_deg": 0.04
 },
 "problems": [],
 "growth_mm": 0.65,
 "growth_uncertainty_mm": 0.35,
 "growing": true,
 "narrower_than_before": false
}
```

### 4. `request_engineer_review` — decided by us.amazon.nova-2-lite-v1:0 (photo p20261027T171401474-f9d5a6)
Input: `{"message": "An engineer will check this crack for you soon.", "reason": "The crack grew from 1.49 mm to 2.14 mm (0.65 mm growth) with uncertainty 0.35 mm. Max width is now 2.14 mm."}`

```json
{
 "ok": true,
 "status": "waiting_for_engineer"
}
```

**Final status:** waiting_for_engineer  
**Message to the owner:** An engineer will check this crack for you soon.

## 2. Rules backup, same photos

### 1. `check_photo` — decided by rules (photo p20261023T173254651-eb9ffb)

```json
{
 "marker_found": true,
 "problems": [],
 "marker_id": 1,
 "px_per_mm": 3.99,
 "tilt_deg": 5.0,
 "crack_marker_expected": null,
 "blur_px": 2.11,
 "white_level": 255.0,
 "black_level": 0.0
}
```

### 2. `measure_crack` — decided by rules (photo p20261023T173254651-eb9ffb)

```json
{
 "crack_found": true,
 "max_width_mm": 2.14,
 "typical_width_mm": 2.0,
 "length_mm": 100.4,
 "uncertainty_mm": 0.25,
 "widest_point_mm": [
  45.9,
  78.2
 ],
 "bre_category": 2,
 "bre_label": "Slight",
 "problems": [],
 "widest_point_where": "about 2 cm to the right and 5 cm below the marker's centre"
}
```

### 3. `compare_with_history` — decided by rules (photo p20261023T173254651-eb9ffb)

```json
{
 "readings": 5,
 "last_reading": {
  "time": "2026-10-16T04:30:00.343+00:00",
  "max_width_mm": 1.49,
  "uncertainty_mm": 0.25
 },
 "trend": [
  {
   "date": "2026-09-18",
   "max_width_mm": 1.12
  },
  {
   "date": "2026-09-25",
   "max_width_mm": 1.23
  },
  {
   "date": "2026-10-02",
   "max_width_mm": 1.26
  },
  {
   "date": "2026-10-09",
   "max_width_mm": 1.29
  },
  {
   "date": "2026-10-16",
   "max_width_mm": 1.49
  }
 ],
 "alignment": {
  "inliers": 45,
  "shift_mm": 0.15,
  "rotation_deg": 0.04
 },
 "problems": [],
 "growth_mm": 0.65,
 "growth_uncertainty_mm": 0.35,
 "growing": true,
 "narrower_than_before": false
}
```

### 4. `ask_for_photo` — decided by rules (photo p20261023T173254651-eb9ffb)
Input: `{"kind": "closeup", "message": "The crack may have changed. Please take a close-up of the widest part (about 2 cm to the right and 5 cm below the marker's centre), with the phone about 10-15 cm from the wall and the marker still in the picture."}`

```json
{
 "ok": true,
 "status": "waiting_for_photo"
}
```

### 5. `check_photo` — decided by rules (photo p20261023T173255396-8d7928)

```json
{
 "marker_found": true,
 "problems": [],
 "marker_id": 1,
 "px_per_mm": 9.997,
 "tilt_deg": 0.0,
 "crack_marker_expected": 1,
 "blur_px": 2.33,
 "white_level": 255.0,
 "black_level": 0.0
}
```

### 6. `measure_crack` — decided by rules (photo p20261023T173255396-8d7928)

```json
{
 "crack_found": true,
 "max_width_mm": 2.08,
 "typical_width_mm": 1.99,
 "length_mm": 103.1,
 "uncertainty_mm": 0.1,
 "widest_point_mm": [
  61.0,
  76.1
 ],
 "bre_category": 2,
 "bre_label": "Slight",
 "problems": [],
 "widest_point_where": "about 3 cm to the right and 5 cm below the marker's centre"
}
```

### 7. `compare_with_history` — decided by rules (photo p20261023T173255396-8d7928)

```json
{
 "readings": 5,
 "last_reading": {
  "time": "2026-10-16T04:30:00.343+00:00",
  "max_width_mm": 1.49,
  "uncertainty_mm": 0.25
 },
 "trend": [
  {
   "date": "2026-09-18",
   "max_width_mm": 1.12
  },
  {
   "date": "2026-09-25",
   "max_width_mm": 1.23
  },
  {
   "date": "2026-10-02",
   "max_width_mm": 1.26
  },
  {
   "date": "2026-10-09",
   "max_width_mm": 1.29
  },
  {
   "date": "2026-10-16",
   "max_width_mm": 1.49
  }
 ],
 "alignment": {
  "inliers": null,
  "shift_mm": null,
  "rotation_deg": null
 },
 "problems": [
  "scale_too_different"
 ],
 "growth_mm": 0.59,
 "growth_uncertainty_mm": 0.27,
 "growing": true,
 "narrower_than_before": false
}
```

### 8. `request_engineer_review` — decided by rules (photo p20261023T173255396-8d7928)
Input: `{"reason": "Crack #1: the crack widened from 1.49 to 2.08 mm (+/-0.27 mm).", "message": "This crack needs a closer look, so I've asked a structural engineer to review it. You'll see their answer on this page."}`

```json
{
 "ok": true,
 "status": "waiting_for_engineer"
}
```

**Final status:** waiting_for_engineer  
**Message to the owner:** This crack needs a closer look, so I've asked a structural engineer to review it. You'll see their answer on this page.
