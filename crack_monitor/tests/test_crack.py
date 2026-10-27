import cv2
import numpy as np
import pytest

from crack import bre_category, measure_photo, path_length, thin
from quality import check_quality
from marker import check_marker
from synthetic import crack_drawer, crack_path, make_photo

WALL = (420.0, 320.0)
CENTRE = (WALL[0] / 2, WALL[1] / 2)
IMAGE = (1600, 1200)


def photo_of_crack(width_mm, px_per_mm=6.0, tilt_deg=0.0, seed=1, **wide):
    points = crack_path((CENTRE[0] - 60, CENTRE[1] + 70), 120, seed=seed)
    draw, truth = crack_drawer(points, width_mm, **wide)
    distance = 0.72 * max(IMAGE) / px_per_mm
    photo, camera = make_photo(tilt_deg=tilt_deg, distance_mm=distance, image_size=IMAGE,
                               render_px_per_mm=12, wall_mm=WALL, draw=draw)
    return photo, truth, camera


# --- Measuring cracks with a known answer --------------------------------------------

@pytest.mark.parametrize("width", [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("tilt", [0, 25])
def test_width_and_length_match_the_truth(width, tilt):
    photo, truth, _ = photo_of_crack(width, tilt_deg=tilt)
    result, _ = measure_photo(photo)
    crack = result["crack"]
    tolerance = 1.5 * crack["uncertainty_mm"]           # about 1.5 pixels
    assert crack["crack_found"]
    assert crack["max_width_mm"] == pytest.approx(truth["max_width_mm"], abs=tolerance)
    assert crack["typical_width_mm"] == pytest.approx(truth["width_mm"], abs=tolerance)
    assert crack["length_mm"] == pytest.approx(truth["length_mm"], rel=0.03)
    assert crack["pieces"] == 1


def test_finds_the_widest_part():
    photo, truth, camera = photo_of_crack(0.6, wide_from=12, wide_to=17, wide_mm=1.5)
    crack = measure_photo(photo)[0]["crack"]
    assert crack["max_width_mm"] == pytest.approx(truth["max_width_mm"], abs=1.5 * crack["uncertainty_mm"])
    mx, my = camera["marker_top_left_mm"]
    wide_x, wide_y = truth["wide_centre_mm"]
    assert np.hypot(crack["widest_point_mm"][0] - (wide_x - mx), crack["widest_point_mm"][1] - (wide_y - my)) < 12


def test_thin_crack_from_far_away_asks_for_a_closer_photo():
    photo, _, _ = photo_of_crack(0.3, px_per_mm=4.0)
    crack = measure_photo(photo)[0]["crack"]
    assert crack["crack_found"]
    assert "crack_too_thin_for_this_distance" in crack["problems"]


def test_plain_wall_has_no_crack():
    photo, _ = make_photo(distance_mm=192, image_size=IMAGE, render_px_per_mm=12, wall_mm=WALL)
    result, _ = measure_photo(photo)
    assert result["crack"]["crack_found"] is False
    assert result["crack"]["problems"] == ["no_crack"]


def stain_photo(with_crack):
    points = crack_path((CENTRE[0] - 60, CENTRE[1] + 70), 120, seed=1)
    crack, _ = crack_drawer(points, 1.0)

    def draw(wall, s):
        cv2.circle(wall, (int(CENTRE[0] * s), int((CENTRE[1] + 70) * s)), int(12 * s), 110, -1)   # dark stain
        if with_crack:
            crack(wall, s)
    return make_photo(distance_mm=192, image_size=IMAGE, render_px_per_mm=12, wall_mm=WALL, draw=draw)[0]


def test_stain_rim_is_not_a_crack_but_a_crack_across_a_stain_is():
    assert measure_photo(stain_photo(False))[0]["crack"]["crack_found"] is False
    crack = measure_photo(stain_photo(True))[0]["crack"]
    assert crack["crack_found"] and crack["max_width_mm"] == pytest.approx(1.0, abs=0.35)
    assert crack["length_mm"] < 110          # the part inside the stain isn't measured


def test_no_marker_means_no_measurement():
    photo, _ = make_photo(marker_id_present=False)
    result, marked = measure_photo(photo)
    assert "no_marker" in result["marker"]["problems"]
    assert "crack" not in result
    assert marked.shape == photo.shape


def test_marked_up_image_is_drawn():
    photo, _, _ = photo_of_crack(1.0)
    _, marked = measure_photo(photo)
    assert marked.shape == photo.shape
    assert np.any(np.all(marked == (0, 255, 255), axis=2))      # yellow circle at the widest point


# --- Building blocks --------------------------------------------------------------------

def test_thinning_gives_a_one_pixel_line():
    mask = np.zeros((40, 100), np.uint8)
    mask[15:22, 10:90] = 1                                         # 7 px thick bar
    skeleton = thin(mask)
    assert skeleton[:, 20:80].sum(axis=0).max() == 1               # one pixel per column in the middle
    assert 60 <= skeleton.sum() <= 82


def test_path_length_of_a_diagonal_is_not_staircased():
    path = np.array([(i, i) for i in range(101)] + [(100, 100 + i) for i in range(1, 101)])
    assert path_length(path) == pytest.approx(100 * np.sqrt(2) + 100, rel=0.02)


@pytest.mark.parametrize("width, category", [(0.05, 0), (0.5, 1), (3.0, 2), (10.0, 3), (20.0, 4), (30.0, 5)])
def test_bre_categories(width, category):
    assert bre_category(width)[0] == category


# --- Photo quality (uses the marker as reference) -----------------------------------------

@pytest.mark.parametrize("blur, expected", [(0, []), (0.7, []), (2.0, ["blurry"])])
def test_blur_is_judged_on_the_marker(blur, expected):
    photo, _ = make_photo(distance_mm=200, blur_sigma=blur)
    _, marker = check_marker(photo)
    assert check_quality(photo, marker)["problems"] == expected


def test_dark_and_washed_out_photos():
    photo, _ = make_photo(distance_mm=200)
    _, marker = check_marker(photo)
    assert check_quality((photo * 0.2).astype(np.uint8), marker)["problems"] == ["too_dark"]
    washed = cv2.addWeighted(photo, 0.4, photo, 0, 150)
    assert check_quality(washed, marker)["problems"] == ["overexposed"]
    assert check_quality(np.full((100, 100, 3), 10, np.uint8))["problems"] == ["too_dark"]
