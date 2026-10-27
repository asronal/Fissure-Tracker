import numpy as np
import pytest

import align as align_module
from align import MIN_INLIERS, align
from synthetic import make_photo


def test_same_view_lines_up_exactly():
    photo, _ = make_photo()
    result, drawing = align(photo, photo.copy())
    assert result["aligned"] and result["problems"] == []
    assert result["shift_mm"] < 0.1
    assert drawing is not None


@pytest.mark.parametrize("new, old", [
    (dict(tilt_deg=20, distance_mm=250), dict()),                  # closer and from the side
    (dict(tilt_deg=-25, distance_mm=350), dict(tilt_deg=15)),      # other side
    (dict(distance_mm=450), dict(distance_mm=250)),                # much further away
])
def test_different_viewpoints_still_line_up(new, old):
    result, _ = align(make_photo(**new)[0], make_photo(**old)[0])
    assert result["aligned"], result
    assert result["inliers"] >= MIN_INLIERS
    assert result["shift_mm"] < 0.5
    assert abs(result["rotation_deg"]) < 0.5


def test_moved_marker_is_detected_and_measured():
    result, _ = align(make_photo(marker_offset_mm=(8, 5))[0], make_photo()[0])
    assert result["problems"] == ["marker_moved"] and not result["aligned"]
    assert result["shift_xy_mm"] == pytest.approx([8, 5], abs=0.3)   # the wall moved by the same amount


def test_small_marker_shift_is_tolerated():
    result, _ = align(make_photo(marker_offset_mm=(2, 0))[0], make_photo()[0])
    assert result["aligned"]
    assert result["shift_xy_mm"] == pytest.approx([2, 0], abs=0.3)


def test_different_wall_does_not_match():
    result, _ = align(make_photo(seed=7)[0], make_photo()[0])
    assert result["problems"] == ["not_enough_matches"] and not result["aligned"]


def test_photo_without_marker():
    result, drawing = align(make_photo(marker_id_present=False)[0], make_photo()[0])
    assert result["problems"] == ["no_marker"] and drawing is None


def test_close_up_is_not_compared_with_a_normal_photo():
    result, _ = align(make_photo(distance_mm=90)[0], make_photo(distance_mm=300)[0])
    assert result["problems"] == ["scale_too_different"]


def test_plain_wall_is_not_blamed_on_the_owner():
    plain = make_photo(seed=3)[0]
    plain[:] = np.where(plain < 128, plain, 205).astype(np.uint8)    # keep the marker, wipe the wall texture
    result, _ = align(plain, plain.copy())
    assert result["problems"] == ["plain_wall"]


# --- OpenCV 5 ALIKED + LightGlue (skipped when the models haven't been downloaded) ---

needs_models = pytest.mark.skipif(align_module.deep_models() is None, reason="run tools/get_models.py first")


@needs_models
def test_aliked_lightglue_lines_up_the_same_view():
    photo, _ = make_photo()
    result, _ = align(photo, photo.copy(), "aliked_lightglue")
    assert result["method"] == "aliked_lightglue" and result["aligned"], result
    assert result["shift_mm"] < 0.1


@needs_models
def test_aliked_lightglue_does_not_match_another_wall():
    # The marker is identical in every photo: it must not be what the network matches
    result, _ = align(make_photo(seed=7)[0], make_photo()[0], "aliked_lightglue")
    assert not result["aligned"] and result["problems"] == ["not_enough_matches"]


@needs_models
def test_aliked_lightglue_takes_over_when_orb_finds_too_few_matches(monkeypatch):
    monkeypatch.setattr(align_module, "orb_matches", lambda kp_new, des_new, kp_old, des_old: (kp_new, kp_old, []))
    photo, _ = make_photo()
    result, _ = align(photo, photo.copy())
    assert result["method"] == "aliked_lightglue" and result["orb_inliers"] == 0 and result["aligned"]


def test_orb_only_without_the_models(monkeypatch):
    monkeypatch.setitem(align_module._deep, "models", None)
    monkeypatch.setattr(align_module, "orb_matches", lambda kp_new, des_new, kp_old, des_old: (kp_new, kp_old, []))
    photo, _ = make_photo()
    result, _ = align(photo, photo.copy())
    assert result["method"] == "orb" and result["problems"] == ["not_enough_matches"]
