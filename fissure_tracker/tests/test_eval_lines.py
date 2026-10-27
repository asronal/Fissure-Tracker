from eval_lines import measure_lines
from synthetic import make_photo, sheet_photo


def test_printed_test_sheet_lines_are_measured_within_their_uncertainty():
    photo, truth = sheet_photo(tilt_deg=-20, distance_mm=220)
    rows, problem = measure_lines(photo)
    assert problem is None and len(rows) == len(truth)
    for row, true_width in zip(rows, truth):
        assert row["found"]
        if not row["problems"]:                         # thick enough for this distance
            assert abs(row["typical_width_mm"] - true_width) <= row["uncertainty_mm"] + 0.02


def test_crack_marker_is_not_mistaken_for_the_test_sheet():
    rows, problem = measure_lines(make_photo(marker_id=2)[0])
    assert rows == [] and "not the test sheet" in problem
