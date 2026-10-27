import numpy as np
import pytest

from marker import MARKER_MM, check_marker, find_markers, straighten
from synthetic import make_photo


def test_straight_on_photo_gives_scale_and_id():
    photo, truth = make_photo(marker_id=3, tilt_deg=0)
    result, marker = check_marker(photo)
    assert result["marker_found"] and result["marker_id"] == 3
    assert result["px_per_mm"] == pytest.approx(truth["px_per_mm"], rel=0.02)
    assert result["tilt_deg"] < 3
    assert result["problems"] == []


@pytest.mark.parametrize("tilt", [0, 15, 25, 40])
def test_tilt_is_measured(tilt):
    photo, _ = make_photo(tilt_deg=tilt)
    result, _ = check_marker(photo)
    assert result["tilt_deg"] == pytest.approx(tilt, abs=3)
    assert ("too_tilted" in result["problems"]) == (tilt > 30)


def test_no_marker():
    photo, _ = make_photo(marker_id_present=False)
    result, marker = check_marker(photo)
    assert marker is None and result["problems"] == ["no_marker"]


def test_far_away_marker_is_too_small():
    photo, _ = make_photo(distance_mm=1300, image_size=(800, 600), render_px_per_mm=2)
    result, _ = check_marker(photo)
    assert result["marker_found"]
    assert "marker_too_small" in result["problems"]


def test_test_sheet_marker_is_recognised():
    photo, _ = make_photo(marker_id=0)
    result, _ = check_marker(photo)
    assert result["is_test_sheet"] is True


@pytest.mark.parametrize("tilt", [0, 25])
def test_straight_view_has_a_square_marker_at_the_chosen_scale(tilt):
    photo, _ = make_photo(tilt_deg=tilt)
    _, marker = check_marker(photo)
    view = straighten(photo, marker, px_per_mm=3.0)

    # Find the marker again in the straightened image: it must be square and 60 mm = 180 px
    again = find_markers(view["image"])
    assert len(again) == 1
    corners = again[0]["corners"]
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    assert np.mean(sides) == pytest.approx(MARKER_MM * 3.0, rel=0.02)
    assert max(sides) / min(sides) < 1.03
    x, y, w, h = view["marker_box"]
    assert corners[0] == pytest.approx([x, y], abs=3)
