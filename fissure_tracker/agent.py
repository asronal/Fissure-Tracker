"""
Step 6: The agent - decides what happens with each new crack photo

The owner sends a photo. An AI model (Amazon Nova on Amazon Bedrock) calls our OpenCV 5 tools one at
a time, reads their results and picks the next step:

  check_photo              marker + photo quality            (marker.py, quality.py)
  measure_crack            width, length, widest point in mm (crack.py)
  compare_with_history     line up with the last photo, growth since then (align.py + stored readings)
  ask_for_photo            ends the turn: owner must send a retake or a close-up
  save_reading             ends the turn: reading saved as history ("stable, check next week")
  request_engineer_review  ends the turn: an engineer gets an e-mail with an approve link

So what the camera sees changes what the agent does next. Two safety nets:
  - Policy checks in code: the model can't save a reading that needs an engineer, can't skip the photo
    check, and can't ask the owner for photos forever. A blocked call is returned to the model as an error.
  - Rules backup: if Bedrock is unavailable (no internet, quota, timeout) the same tools are called by a
    fixed policy (rules_policy), so the owner always gets an answer.
Every step is written to the case's trace with who decided it (the model id, "rules" or "engineer").

Usage (no AWS needed with AGENT_MODE=rules):
    python agent.py photo.jpg [case_id]
"""

import hmac
import json
import os
import secrets
import sys
import time

import cv2
import numpy as np

from align import MAX_SCALE_RATIO, align
from crack import draw_crack, find_crack
from marker import MARKER_MM, check_marker, draw_marker, straighten
from quality import TIPS, check_quality, decode_image
from store import LocalStore, new_id, now_iso

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_TOKENS = 700            # always set explicitly (Bedrock reserves quota for the maximum)
MAX_TOOL_CALLS = 8          # per photo
TIME_BUDGET_S = 45          # stop asking the model after this and let the rules finish
MAX_PHOTOS_PER_CASE = 4     # then an engineer looks at it instead of asking the owner again

# Decision thresholds - project settings, not engineering rules
ALERT_WIDTH_MM = 2.0              # this wide or wider -> engineer review
MIN_GROWTH_MM = 0.2               # smaller changes are ignored
CONFIRM_UNCERTAINTY_MM = 0.2      # before alerting, get a close-up if the measurement is coarser than this

END_TOOLS = ("ask_for_photo", "save_reading", "request_engineer_review")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

SYSTEM_PROMPT = f"""You are the inspection agent of a wall-crack monitoring service.
A building owner sticks a printed marker (a 60 mm black-and-white square) next to a crack and sends a
photo of it every week. You decide what happens with each new photo by calling tools.

How to work:
1. Call check_photo first. If it reports problems, call ask_for_photo (kind "retake") and tell the owner
   exactly how to fix the photo.
2. Call measure_crack. If no crack is found, ask for a retake. If the crack is too thin for the photo's
   resolution, ask for a close-up (kind "closeup") and say where the widest point is relative to the marker.
3. Call compare_with_history. If the photos don't match (not_enough_matches), ask the owner to stand where
   they stood last time. plain_wall or scale_too_different only mean the photos couldn't be lined up (not
   the owner's fault): carry on and use the width comparison.
4. Decide:
   - request_engineer_review if the crack is {ALERT_WIDTH_MM} mm wide or wider, if it grew by more than the
     measurement uncertainty (and at least {MIN_GROWTH_MM} mm), if it is clearly narrower than before
     (narrower_than_before), or if the marker moved (marker_moved).
     Before alerting an engineer about width or growth, if uncertainty_mm is above {CONFIRM_UNCERTAINTY_MM}
     and no close-up was asked for in this case yet, first ask for a close-up of the widest point.
   - otherwise save_reading.
Every turn must end with exactly one of: ask_for_photo, save_reading, request_engineer_review.
Messages to the owner: plain, friendly English, at most 2 short sentences, no jargon. Never give structural
or repair advice yourself; that is the engineer's job. Never invent numbers: use the tool results."""


def _schema(properties=None, required=()):
    return {"json": {"type": "object", "properties": properties or {}, "required": list(required)}}


TOOL_SPECS = [
    {"toolSpec": {"name": "check_photo",
                  "description": "Find the marker (which crack, scale, tilt) and check photo quality on the marker.",
                  "inputSchema": _schema()}},
    {"toolSpec": {"name": "measure_crack",
                  "description": "Measure the crack next to the marker: max and typical width in mm, length, "
                                 "widest point (mm from the marker's top-left corner, x right, y down), uncertainty.",
                  "inputSchema": _schema()}},
    {"toolSpec": {"name": "compare_with_history",
                  "description": "Line this photo up with the last accepted photo of the same crack and "
                                 "compare widths: growth, uncertainty, marker moved, trend.",
                  "inputSchema": _schema()}},
    {"toolSpec": {"name": "ask_for_photo",
                  "description": "End the turn and ask the owner for another photo.",
                  "inputSchema": _schema({
                      "kind": {"type": "string", "enum": ["retake", "closeup"]},
                      "message": {"type": "string", "description": "What the owner should do, specific."}},
                      ["kind", "message"])}},
    {"toolSpec": {"name": "save_reading",
                  "description": "End the turn: save this measurement as the crack's latest reading.",
                  "inputSchema": _schema({
                      "message": {"type": "string", "description": "Short result for the owner."}},
                      ["message"])}},
    {"toolSpec": {"name": "request_engineer_review",
                  "description": "End the turn: e-mail a structural engineer, who approves the next step.",
                  "inputSchema": _schema({
                      "reason": {"type": "string", "description": "For the engineer: what changed, with numbers."},
                      "message": {"type": "string", "description": "For the owner: what happens next."}},
                      ["reason", "message"])}},
]


# --- Notifications (local list here; e-mail through Amazon SNS on AWS) ---------------------------

class LocalNotifier:
    def __init__(self):
        self.sent = []

    def notify_engineer(self, subject, body):
        self.sent.append({"subject": subject, "body": body})


# --- The tools ----------------------------------------------------------------------------------

def _clean(text, limit=400):
    return " ".join(str(text or "").split())[:limit]


def _jpeg(img, quality=88):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def where_on_wall(point_mm):
    """'about 4 cm to the right and 6 cm below the marker's centre' for a point in marker mm."""
    dx, dy = point_mm[0] - MARKER_MM / 2, point_mm[1] - MARKER_MM / 2
    parts = []
    if abs(dx) >= 10:
        parts.append(f"{abs(dx) / 10:.0f} cm to the {'right' if dx > 0 else 'left'}")
    if abs(dy) >= 10:
        parts.append(f"{abs(dy) / 10:.0f} cm {'below' if dy > 0 else 'above'}")
    return ("about " + " and ".join(parts) + " the marker's centre") if parts else "right next to the marker"


class Tools:
    """The agent's tools for one photo in one case. Results are cached, every call is traced."""

    def __init__(self, store, case, photo_id, img, notifier, base_url=""):
        self.store, self.case, self.photo_id, self.img = store, case, photo_id, img
        self.notifier, self.base_url = notifier, base_url
        self.results = {}
        self.marker = None
        self.ended = False
        self.decided_by = "rules"

    # -- trace --
    def trace(self, tool, args, output, thought=None, decided_by=None):
        step = {"step": len(self.case["trace"]) + 1, "time": now_iso(), "photo_id": self.photo_id,
                "decided_by": decided_by or self.decided_by, "tool": tool, "input": args, "output": output}
        if thought:
            step["thought"] = _clean(thought, 600)
        self.case["trace"].append(step)

    def run(self, name, args, thought=None):
        """Run one tool (with the policy check) and trace it. Returns what the model gets back."""
        args = args if isinstance(args, dict) else {}
        blocked = self.policy_check(name, args)
        if blocked:
            output = {"error": blocked}
        elif name in self.results:
            output = self.results[name]
        else:
            output = getattr(self, "_" + name)(args)
            if name not in END_TOOLS and "error" not in output:
                self.results[name] = output
        self.trace(name, args, output, thought)
        return output

    # -- policy: what the model is not allowed to do --
    def policy_check(self, name, args):
        if name not in [t["toolSpec"]["name"] for t in TOOL_SPECS]:
            return f"unknown tool {name}"
        if self.ended:
            return "this photo has already been decided"
        if name != "check_photo" and "check_photo" not in self.results:
            return "call check_photo first"
        if name == "ask_for_photo":
            if args.get("kind") not in ("retake", "closeup"):
                return "kind must be 'retake' or 'closeup'"
            if len(self.case["photos"]) >= MAX_PHOTOS_PER_CASE:
                return (f"the owner has already sent {MAX_PHOTOS_PER_CASE} photos in this case; "
                        "use request_engineer_review instead")
        if name == "save_reading":
            if self.results["check_photo"]["problems"]:
                return "the photo has problems; it can't be saved as a reading"
            crack = self.results.get("measure_crack")
            if not crack or not crack.get("crack_found") or crack["problems"]:
                return "call measure_crack first; only a clean measurement can be saved"
            if "compare_with_history" not in self.results:
                return "call compare_with_history first"
            reason, confirm_first = self.assessment()
            if confirm_first:
                return f"{reason}: ask for a close-up of the widest point to confirm, or request an engineer review"
            if reason:
                return f"{reason}: this needs request_engineer_review, it can't be saved as normal"
        return None

    def assessment(self):
        """(reason an engineer is needed or None, whether to get a close-up first)."""
        crack = self.results.get("measure_crack") or {}
        history = self.results.get("compare_with_history") or {}
        reason = None
        if "marker_moved" in history.get("problems", []):
            reason = f"the marker moved by about {history['alignment']['shift_mm']} mm since the last photo"
        elif history.get("growing"):
            reason = (f"the crack widened from {history['last_reading']['max_width_mm']} to "
                      f"{crack['max_width_mm']} mm (+/-{history['growth_uncertainty_mm']} mm)")
        elif history.get("narrower_than_before"):
            reason = (f"the crack measured narrower than last time ({history['last_reading']['max_width_mm']} -> "
                      f"{crack['max_width_mm']} mm): repaired, marker moved, or a measurement problem?")
        elif crack.get("crack_found") and crack["max_width_mm"] >= ALERT_WIDTH_MM:
            reason = f"the crack is {crack['max_width_mm']} mm wide (alert level {ALERT_WIDTH_MM} mm)"
        closeup_asked = any(p.get("asked_for") == "closeup" for p in self.case["photos"])
        confirm_first = bool(reason and "marker" not in reason and crack.get("uncertainty_mm", 0) > CONFIRM_UNCERTAINTY_MM
                             and not closeup_asked and len(self.case["photos"]) < MAX_PHOTOS_PER_CASE)
        return reason, confirm_first

    # -- perception tools (OpenCV 5) --
    def _check_photo(self, args):
        result, self.marker = check_marker(self.img)
        quality = check_quality(self.img, self.marker)
        problems = result["problems"] + quality["problems"]
        expected = self.case.get("marker_id")
        if self.marker is not None:
            if result["is_test_sheet"]:
                problems.append("test_sheet_marker")
            elif expected is not None and result["marker_id"] != expected:
                problems.append("different_marker")
            elif expected is None:
                self.case["marker_id"] = result["marker_id"]
        marked = draw_marker(self.img, result, self.marker)
        self.store.save_photo(self.photo_id, _jpeg(marked), "marked")
        out = {"marker_found": result["marker_found"], "problems": problems}
        if self.marker is not None:
            out.update({k: result[k] for k in ("marker_id", "px_per_mm", "tilt_deg")})
            out["crack_marker_expected"] = expected
        out.update({k: quality[k] for k in ("blur_px", "white_level", "black_level") if k in quality})
        return out

    def _measure_crack(self, args):
        if self.marker is None:
            return {"crack_found": False, "problems": ["no_marker"]}
        view = straighten(self.img, self.marker)
        result, crack_mask, _ = find_crack(view["image"], view["px_per_mm"], view["valid"], view["marker_box"])
        if crack_mask is not None:
            marked = decode_image(self.store.load_photo(self.photo_id, "marked"))     # photo with the marker drawn
            self.store.save_photo(self.photo_id, _jpeg(draw_crack(marked, crack_mask, view, result)), "marked")
        keep = ("crack_found", "max_width_mm", "typical_width_mm", "length_mm", "uncertainty_mm",
                "widest_point_mm", "bre_category", "bre_label", "problems")
        out = {k: result[k] for k in keep if k in result}
        if result.get("crack_found"):
            out["widest_point_where"] = where_on_wall(result["widest_point_mm"])
        return out

    def _compare_with_history(self, args):
        crack = self.results.get("measure_crack")
        if not crack:
            return {"error": "call measure_crack first"}
        marker_id = self.case.get("marker_id")
        history = [r for r in self.store.readings(marker_id) if r["case_id"] != self.case["case_id"]]
        if not history:
            return {"readings": 0, "note": "first reading of this crack - nothing to compare with", "problems": []}
        last = history[-1]
        # Line up with the latest photo taken from a similar distance (a close-up can't be lined up
        # with a normal photo); widths are always compared with the latest reading
        scale = self.results["check_photo"]["px_per_mm"]
        similar = [r for r in history
                   if max(scale, r.get("px_per_mm", 0)) / max(1e-6, min(scale, r.get("px_per_mm", 0))) <= MAX_SCALE_RATIO]
        reference = similar[-1] if similar else last
        old = decode_image(self.store.load_photo(reference["photo_id"]))
        alignment, drawing = align(self.img, old)
        if drawing is not None:
            self.store.save_photo(self.photo_id, _jpeg(drawing, 80), "matches")
        out = {"readings": len(history),
               "last_reading": {k: last[k] for k in ("time", "max_width_mm", "uncertainty_mm")},
               "trend": [{"date": r["time"][:10], "max_width_mm": r["max_width_mm"]} for r in history[-5:]],
               "alignment": {k: alignment.get(k) for k in ("method", "orb_inliers", "inliers", "shift_mm", "rotation_deg")},
               "problems": alignment["problems"]}
        if crack.get("crack_found"):
            growth = crack["max_width_mm"] - last["max_width_mm"]
            uncertainty = float(np.hypot(crack["uncertainty_mm"], last["uncertainty_mm"]))
            change = max(MIN_GROWTH_MM, uncertainty)
            lined_up = "not_enough_matches" not in alignment["problems"]
            out.update({"growth_mm": round(growth, 2), "growth_uncertainty_mm": round(uncertainty, 2),
                        "growing": bool(growth > change and lined_up),
                        # Cracks rarely close up: a big drop suggests a repair, a moved marker or a bad photo
                        "narrower_than_before": bool(growth < -change and lined_up)})
        return out

    # -- action tools (end the turn) --
    def _end(self, status, message):
        self.ended = True
        self.case["status"] = status
        self.case["owner_message"] = _clean(message, 300)
        return {"ok": True, "status": status}

    def _ask_for_photo(self, args):
        self.case["request"] = {"kind": args["kind"], "message": _clean(args.get("message"), 300)}
        return self._end("waiting_for_photo", args.get("message"))

    def _store_reading(self, status=None):
        crack = self.results["measure_crack"]
        history = self.results.get("compare_with_history", {})
        reading = {"marker_id": self.case["marker_id"], "case_id": self.case["case_id"],
                   "photo_id": self.photo_id, "time": now_iso(), "px_per_mm": self.results["check_photo"]["px_per_mm"],
                   "status": status or ("baseline" if not history.get("readings") else "stable")}
        reading.update({k: crack[k] for k in ("max_width_mm", "typical_width_mm", "length_mm",
                                              "uncertainty_mm", "widest_point_mm", "bre_category")})
        self.store.add_reading(reading)
        self.case["reading"] = reading
        return reading

    def _save_reading(self, args):
        self._store_reading()
        return self._end("closed", args.get("message"))

    def _request_engineer_review(self, args):
        crack = self.results.get("measure_crack") or {}
        if crack.get("crack_found") and not crack["problems"] and not self.results["check_photo"]["problems"]:
            self._store_reading(status="needs_review")
        token = secrets.token_urlsafe(24)
        self.case["review"] = {"token": token, "reason": _clean(args.get("reason"), 500),
                               "requested": now_iso(), "decision": None}
        link = f"{self.base_url}/review/{self.case['case_id']}?token={token}"
        self.notifier.notify_engineer(
            f"Crack #{self.case.get('marker_id')} needs a review",
            f"{self.case['review']['reason']}\n\nPhotos, measurements and the agent's steps:\n{link}\n\n"
            "Open the link to confirm an inspection or mark it as not needed.")
        return self._end("waiting_for_engineer", args.get("message"))



# --- Deciding: rules backup ---------------------------------------------------------------------

def rules_policy(tools):
    """The fixed policy, used when the model is unavailable. Same tools, same order every time."""
    photo = tools.run("check_photo", {})
    problems = photo.get("problems", [])
    if problems:
        return ask(tools, "retake", retake_message(problems, tools.case))

    crack = tools.run("measure_crack", {})
    if not crack.get("crack_found"):
        return ask(tools, "retake",
                   "I can't find a crack next to the marker. Take the photo again with the crack and "
                   "the whole marker in the picture, from about 20-30 cm.")
    if "crack_too_thin_for_this_distance" in crack["problems"]:
        return ask(tools, "closeup", closeup_message(crack))

    history = tools.run("compare_with_history", {})
    if "not_enough_matches" in history.get("problems", []):
        return ask(tools, "retake",
                   "This photo shows the wall differently from last time. Please stand where you stood "
                   "last time, with the marker in the same place in the picture.")

    reason, confirm_first = tools.assessment()
    if confirm_first:
        return ask(tools, "closeup", "The crack may have changed. " + closeup_message(crack))
    if reason:
        return tools.run("request_engineer_review", {
            "reason": f"Crack #{tools.case['marker_id']}: {reason}.",
            "message": "This crack needs a closer look, so I've asked a structural engineer to review it. "
                       "You'll see their answer on this page."})
    status = "First reading saved" if not history.get("readings") else "No change since last time"
    return tools.run("save_reading", {"message": f"{status}: the crack is {crack['max_width_mm']} mm wide at "
                                                 f"its widest. Please take the next photo in a week."})


def ask(tools, kind, message):
    """Ask the owner for another photo - or, once the photo limit is reached, hand the case to an engineer."""
    output = tools.run("ask_for_photo", {"kind": kind, "message": message})
    if "error" not in output:
        return output
    return tools.run("request_engineer_review", {
        "reason": f"Crack #{tools.case.get('marker_id')}: no usable photo after {len(tools.case['photos'])} "
                  f"tries. Last problem: {message}",
        "message": "I couldn't get a clear enough photo, so I've asked a structural engineer to look at this case."})


def retake_message(problems, case):
    if "no_marker" in problems:
        return "I can't see the marker. Take the photo again with the whole black-and-white square next to the crack."
    if "different_marker" in problems:
        return f"This case is about crack #{case.get('marker_id')}. Please photograph the crack with marker #{case.get('marker_id')}."
    if "test_sheet_marker" in problems:
        return "That is the test-sheet marker. Please photograph a crack with one of the crack markers (1-6)."
    if "marker_too_small" in problems:
        return "You are too far away. Move closer, about 20-30 cm from the wall, and take the photo again."
    if "too_tilted" in problems:
        return "The photo is taken at a strong angle. Stand straight in front of the wall, phone parallel to it."
    return " ".join(TIPS[p] for p in problems if p in TIPS) or "Please take the photo again."


def closeup_message(crack):
    return (f"Please take a close-up of the widest part ({crack['widest_point_where']}), with the phone about "
            "10-15 cm from the wall and the marker still in the picture.")


# --- Deciding: the AI model on Amazon Bedrock ---------------------------------------------------

def make_llm():
    """Bedrock runtime client, or None when AGENT_MODE=rules."""
    if os.environ.get("AGENT_MODE", "auto") == "rules":
        return None
    import boto3
    from botocore.config import Config

    return boto3.client("bedrock-runtime", region_name=REGION,
                        config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2, "mode": "standard"}))


def case_context(case):
    n = len(case["photos"])
    text = f"New photo in case {case['case_id']} (photo {n} of this case)."
    if case.get("marker_id") is not None:
        text += f" This case is about crack marker #{case['marker_id']}."
    if n > 1 and case.get("request"):
        text += f" Earlier you asked the owner for a {case['request']['kind']}: \"{case['request']['message']}\""
    return text + " Decide what to do with this photo."


def model_policy(tools, llm, model_id=MODEL_ID):
    """Let the model choose the tools. Returns True if it ended the turn itself."""
    tools.decided_by = model_id
    messages = [{"role": "user", "content": [{"text": case_context(tools.case)}]}]
    started, calls = time.time(), 0
    while calls < MAX_TOOL_CALLS and time.time() - started < TIME_BUDGET_S:
        response = llm.converse(modelId=model_id, system=[{"text": SYSTEM_PROMPT}], messages=messages,
                                inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0},
                                toolConfig={"tools": TOOL_SPECS})
        message = response["output"]["message"]
        messages.append(message)
        thought = " ".join(b["text"] for b in message["content"] if "text" in b)
        uses = [b["toolUse"] for b in message["content"] if "toolUse" in b]
        if not uses:
            messages.append({"role": "user", "content": [{"text": "Please continue by calling one of the tools."}]})
            calls += 1
            continue
        results = []
        for use in uses:
            calls += 1
            output = tools.run(use["name"], use.get("input") or {}, thought)
            thought = None
            # Errors (e.g. a blocked call) go back as text: the "status" field only works with some models
            results.append({"toolResult": {"toolUseId": use["toolUseId"], "content": [{"text": json.dumps(output)}]}})
        if tools.ended:
            return True
        messages.append({"role": "user", "content": results})
    return False


# --- One photo, start to finish -----------------------------------------------------------------

def new_case():
    return {"case_id": new_id("c"), "created": now_iso(), "status": "new", "marker_id": None,
            "photos": [], "trace": []}


def handle_photo(store, photo_bytes, case_id=None, notifier=None, llm=None, model_id=MODEL_ID, base_url=""):
    """
    Run the agent on one uploaded photo. Continues `case_id` if that case is waiting for a photo,
    otherwise starts a new case. Returns the case (also saved in the store).
    """
    case = store.get_case(case_id) if case_id else None
    if case is None or case["status"] != "waiting_for_photo":
        case = new_case()
    notifier = notifier or LocalNotifier()

    photo_id = new_id("p")
    store.save_photo(photo_id, photo_bytes)
    asked_for = case.get("request", {}).get("kind", "first") if case["photos"] else "first"
    case["photos"].append({"photo_id": photo_id, "time": now_iso(), "asked_for": asked_for})

    try:
        img = decode_image(photo_bytes)
    except ValueError:
        img = None
    tools = Tools(store, case, photo_id, img, notifier, base_url)
    if img is None:
        tools.trace("check_photo", {}, {"problems": ["unreadable_image"]})
        tools._end("waiting_for_photo", "I couldn't open that file. Please send a normal photo (JPEG or PNG).")
        case["request"] = {"kind": "retake", "message": case["owner_message"]}
    else:
        ended = False
        if llm is not None:
            try:
                ended = model_policy(tools, llm, model_id)
                if not ended:
                    tools.trace("fallback", {}, {"reason": "the model did not finish within its step/time limit"},
                                decided_by="system")
            except Exception as e:   # no internet, no quota, timeout, bad answer: the rules take over
                tools.trace("fallback", {}, {"reason": f"{type(e).__name__}: {_clean(e, 200)}"}, decided_by="system")
        if not ended:
            tools.decided_by = "rules"
            rules_policy(tools)

    case["updated"] = now_iso()
    store.put_case(case)
    return case


def record_engineer_decision(store, case_id, token, decision, note=""):
    """The engineer's answer from the review page. Returns the case, or None if the link is wrong/used."""
    case = store.get_case(case_id)
    review = (case or {}).get("review") or {}
    if not review or review.get("decision") or not hmac.compare_digest(str(token), review.get("token", "")):
        return None
    if decision not in ("inspect", "not_needed"):
        return None
    review.update({"decision": decision, "note": _clean(note, 500), "decided": now_iso()})
    message = ("A structural engineer reviewed this crack and recommends an inspection. They will contact you; "
               "until then, keep taking a photo every week." if decision == "inspect" else
               "A structural engineer reviewed this crack: no action is needed now. Keep taking a photo every week.")
    if review["note"]:
        message += f" Engineer's note: {review['note']}"
    case["owner_message"] = message
    case["status"] = "closed"
    case["trace"].append({"step": len(case["trace"]) + 1, "time": now_iso(), "decided_by": "engineer",
                          "tool": "engineer_decision", "input": {"decision": decision}, "output": {"note": review["note"]}})
    store.put_case(case)
    return case


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python agent.py photo.jpg [case_id]")
        sys.exit(1)
    with open(sys.argv[1], "rb") as f:
        data = f.read()
    local = LocalStore(os.path.join(OUTPUT_DIR, "store"))
    note = LocalNotifier()
    result = handle_photo(local, data, sys.argv[2] if len(sys.argv) == 3 else None, note, make_llm())
    for step in result["trace"]:
        print(f"{step['step']:>2}. [{step['decided_by']}] {step['tool']} {json.dumps(step['input'])}\n"
              f"      -> {json.dumps(step['output'])[:300]}")
    print(f"\nCase {result['case_id']}: {result['status']}\nOwner: {result['owner_message']}")
    for mail in note.sent:
        print(f"\nE-mail to engineer: {mail['subject']}\n{mail['body']}")
