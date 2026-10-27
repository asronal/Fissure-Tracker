"""
The agent: different OpenCV results must lead to different tool calls (the Agentic Vision Award's core
requirement), the policy checks must hold, and the rules backup must take over when the model fails.
"""

import copy
import functools

import cv2
import pytest

from agent import (MAX_PHOTOS_PER_CASE, MAX_TOOL_CALLS, LocalNotifier, handle_photo, record_engineer_decision)
from store import LocalStore
from synthetic import crack_drawer, crack_path, make_photo

WALL = (420.0, 320.0)
CENTRE = (WALL[0] / 2, WALL[1] / 2)
IMAGE = (1600, 1200)
MODEL = "test-model"


@functools.lru_cache(maxsize=None)
def shot(width_mm=1.0, px_per_mm=4.0, tilt_deg=0.0, blur_sigma=0.0, marker=True, offset=(0.0, 0.0), seed=0):
    """JPEG of a crack next to marker #1. Cached: rendering is the slow part."""
    points = crack_path((CENTRE[0] - 50, CENTRE[1] + 45), 100, seed=1 + seed)    # another wall, another crack
    draw, _ = crack_drawer(points, width_mm)
    photo, _ = make_photo(tilt_deg=tilt_deg, distance_mm=0.72 * max(IMAGE) / px_per_mm, image_size=IMAGE,
                          render_px_per_mm=12, wall_mm=WALL, draw=draw, blur_sigma=blur_sigma,
                          marker_id_present=marker, marker_offset_mm=offset, seed=seed)
    return cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


@pytest.fixture
def store(tmp_path):
    return LocalStore(str(tmp_path))


@pytest.fixture
def mail():
    return LocalNotifier()


def steps(case):
    """(tool, kind) for the last photo of the case, e.g. [("check_photo", None), ("ask_for_photo", "retake")]."""
    last = case["photos"][-1]["photo_id"]
    return [(s["tool"], s["input"].get("kind")) for s in case["trace"] if s.get("photo_id") == last]


CHECK, MEASURE, COMPARE = ("check_photo", None), ("measure_crack", None), ("compare_with_history", None)


# --- Different OpenCV results -> different next steps (rules backup) -------------------------

def test_first_photo_becomes_the_baseline(store, mail):
    case = handle_photo(store, shot(), notifier=mail)
    assert steps(case) == [CHECK, MEASURE, COMPARE, ("save_reading", None)]
    assert case["status"] == "closed" and case["reading"]["status"] == "baseline"
    assert case["marker_id"] == 1
    assert store.load_photo(case["photos"][0]["photo_id"], "marked") is not None


def test_unchanged_crack_is_saved_as_stable(store, mail):
    handle_photo(store, shot(), notifier=mail)
    case = handle_photo(store, shot(tilt_deg=15), notifier=mail)
    assert steps(case)[-1] == ("save_reading", None)
    assert case["reading"]["status"] == "stable" and mail.sent == []


def test_blurry_photo_gets_a_specific_retake(store, mail):
    case = handle_photo(store, shot(px_per_mm=6, blur_sigma=3), notifier=mail)
    assert steps(case) == [CHECK, ("ask_for_photo", "retake")]
    assert "blurry" in case["owner_message"] and case["status"] == "waiting_for_photo"


def test_missing_marker_gets_a_retake(store, mail):
    case = handle_photo(store, shot(marker=False), notifier=mail)
    assert steps(case) == [CHECK, ("ask_for_photo", "retake")]
    assert "marker" in case["owner_message"]


def test_thin_crack_far_away_gets_a_close_up_request(store, mail):
    case = handle_photo(store, shot(width_mm=0.3, px_per_mm=4), notifier=mail)
    assert steps(case) == [CHECK, MEASURE, ("ask_for_photo", "closeup")]
    assert "cm" in case["owner_message"]       # says where the widest point is


def test_growth_is_confirmed_with_a_close_up_then_goes_to_an_engineer(store, mail):
    handle_photo(store, shot(width_mm=1.0), notifier=mail)
    case = handle_photo(store, shot(width_mm=1.8), notifier=mail)
    assert steps(case) == [CHECK, MEASURE, COMPARE, ("ask_for_photo", "closeup")]
    case = handle_photo(store, shot(width_mm=1.8, px_per_mm=10), case_id=case["case_id"], notifier=mail)
    assert steps(case) == [CHECK, MEASURE, COMPARE, ("request_engineer_review", None)]
    assert case["status"] == "waiting_for_engineer" and len(case["photos"]) == 2
    assert len(mail.sent) == 1 and "widened" in mail.sent[0]["body"]
    assert store.readings(1)[-1]["status"] == "needs_review"


def test_wide_crack_goes_straight_to_an_engineer_when_the_measurement_is_precise(store, mail):
    case = handle_photo(store, shot(width_mm=2.5, px_per_mm=6), notifier=mail)
    assert steps(case)[-1] == ("request_engineer_review", None)
    assert "alert level" in case["review"]["reason"]


def test_moved_marker_goes_to_an_engineer(store, mail):
    handle_photo(store, shot(), notifier=mail)
    case = handle_photo(store, shot(offset=(8.0, 5.0)), notifier=mail)
    assert steps(case)[-1] == ("request_engineer_review", None)
    assert "marker moved" in case["review"]["reason"]


def test_photo_of_another_wall_asks_to_stand_where_they_stood(store, mail):
    handle_photo(store, shot(), notifier=mail)
    case = handle_photo(store, shot(seed=7), notifier=mail)
    assert steps(case) == [CHECK, MEASURE, COMPARE, ("ask_for_photo", "retake")]
    assert "stood last time" in case["owner_message"]


def test_photo_limit_hands_the_case_to_an_engineer(store, mail):
    case = handle_photo(store, shot(marker=False), notifier=mail)
    for _ in range(MAX_PHOTOS_PER_CASE - 1):
        case = handle_photo(store, shot(marker=False), case_id=case["case_id"], notifier=mail)
    assert len(case["photos"]) == MAX_PHOTOS_PER_CASE
    assert steps(case) == [CHECK, ("ask_for_photo", "retake"), ("request_engineer_review", None)]
    assert "error" in case["trace"][-2]["output"]          # the blocked ask is in the trace
    assert case["status"] == "waiting_for_engineer"


def test_unreadable_upload(store, mail):
    case = handle_photo(store, b"not an image", notifier=mail)
    assert case["status"] == "waiting_for_photo" and "couldn't open" in case["owner_message"]


def test_engineer_decision_needs_the_right_token_and_works_once(store, mail):
    case = handle_photo(store, shot(width_mm=2.5, px_per_mm=6), notifier=mail)
    token = case["review"]["token"]
    assert record_engineer_decision(store, case["case_id"], "wrong", "inspect") is None
    done = record_engineer_decision(store, case["case_id"], token, "inspect", "I'll visit on Monday")
    assert done["status"] == "closed" and "Monday" in done["owner_message"]
    assert done["trace"][-1]["decided_by"] == "engineer"
    assert record_engineer_decision(store, case["case_id"], token, "not_needed") is None


# --- The AI model loop (scripted fake model, no AWS) -----------------------------------------

class FakeModel:
    """Answers Converse calls from a script of tool calls; an Exception in the script is raised."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def converse(self, **request):
        self.calls.append(copy.deepcopy(request))          # the agent keeps adding to its list
        step = self.script.pop(0) if self.script else ("check_photo", {})
        if isinstance(step, Exception):
            raise step
        name, args = step
        return {"output": {"message": {"role": "assistant", "content": [
            {"text": f"Next I'll call {name}."},
            {"toolUse": {"toolUseId": f"t{len(self.calls)}", "name": name, "input": args}}]}}}


def test_model_drives_the_tools_and_is_named_in_the_trace(store, mail):
    model = FakeModel([("check_photo", {}), ("measure_crack", {}), ("compare_with_history", {}),
                       ("save_reading", {"message": "Saved."})])
    case = handle_photo(store, shot(), notifier=mail, llm=model, model_id=MODEL)
    assert steps(case) == [CHECK, MEASURE, COMPARE, ("save_reading", None)]
    assert {s["decided_by"] for s in case["trace"]} == {MODEL}
    assert case["trace"][0]["thought"].startswith("Next I'll call")
    request = model.calls[0]
    assert request["inferenceConfig"]["maxTokens"] > 0 and request["toolConfig"]["tools"]
    # the model sees the real OpenCV numbers in the tool results
    assert '"max_width_mm"' in model.calls[2]["messages"][-1]["content"][0]["toolResult"]["content"][0]["text"]


def test_policy_blocks_saving_a_growing_crack(store, mail):
    handle_photo(store, shot(width_mm=1.0), notifier=mail)
    model = FakeModel([("check_photo", {}), ("measure_crack", {}), ("compare_with_history", {}),
                       ("save_reading", {"message": "All fine."}),                 # not allowed
                       ("ask_for_photo", {"kind": "closeup", "message": "Close-up please."})])
    case = handle_photo(store, shot(width_mm=1.8), notifier=mail, llm=model, model_id=MODEL)
    blocked = [s for s in case["trace"] if s["tool"] == "save_reading"][0]
    assert "close-up" in blocked["output"]["error"]
    assert case["status"] == "waiting_for_photo" and case["request"]["kind"] == "closeup"
    assert '"error"' in model.calls[-1]["messages"][-1]["content"][0]["toolResult"]["content"][0]["text"]


def test_model_cannot_skip_the_photo_check(store, mail):
    model = FakeModel([("save_reading", {"message": "Done."}), ("check_photo", {}), ("measure_crack", {}),
                       ("compare_with_history", {}), ("save_reading", {"message": "Done."})])
    case = handle_photo(store, shot(), notifier=mail, llm=model, model_id=MODEL)
    assert case["trace"][0]["output"]["error"] == "call check_photo first"
    assert case["status"] == "closed"


def test_bedrock_failure_falls_back_to_the_rules(store, mail):
    model = FakeModel([RuntimeError("ThrottlingException: Too many tokens per day")])
    case = handle_photo(store, shot(marker=False), notifier=mail, llm=model, model_id=MODEL)
    assert case["trace"][0]["tool"] == "fallback" and "Throttling" in case["trace"][0]["output"]["reason"]
    assert case["trace"][-1]["decided_by"] == "rules"
    assert case["status"] == "waiting_for_photo"


def test_model_that_never_finishes_is_stopped(store, mail):
    model = FakeModel([("check_photo", {})] * 20)
    case = handle_photo(store, shot(), notifier=mail, llm=model, model_id=MODEL)
    assert len(model.calls) == MAX_TOOL_CALLS
    assert any(s["tool"] == "fallback" for s in case["trace"])
    assert case["status"] == "closed" and case["trace"][-1]["decided_by"] == "rules"
