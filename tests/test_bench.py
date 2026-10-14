"""The reproducible GUI-grounding benchmark: deterministic screens, correct
scoring, and the standard run above its floor. No model."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from irix import bench


def test_make_screen_is_deterministic():
    a, ga = bench.make_screen(seed=7)
    b, gb = bench.make_screen(seed=7)
    assert np.array_equal(a, b) and ga == gb
    c, _ = bench.make_screen(seed=8)
    assert not np.array_equal(a, c)


def test_ground_truth_boxes_are_on_screen():
    frame, gt = bench.make_screen(seed=1)
    h, w = frame.shape[:2]
    assert len(gt) == 8
    for item in gt:
        x, y, bw, bh = item["bbox"]
        assert 0 <= x and 0 <= y and x + bw <= w and y + bh <= h


def test_evaluate_scores_crisp_screens_well():
    samples = [bench.make_screen(seed=i, font_px=16, blur=False) for i in range(2)]
    out = bench.evaluate(samples)
    assert out["ok"] and out["n"] == 16
    assert out["box_recall"] >= 0.9, out
    assert out["grounding_acc"] >= 0.8, out          # crisp is the easy case


def test_run_reports_source_and_latency():
    out = bench.run(n_screens=2)
    assert out["ok"] and out["screens"] == 2
    assert "synthetic" in out["source"]
    assert out["parse_s_mean"] is not None


def test_load_jsonl_groups_by_image(tmp_path):
    p = tmp_path / "gt.jsonl"
    p.write_text('{"image": "a.png", "label": "OK", "bbox": [1,2,3,4]}\n'
                 '{"image": "a.png", "label": "No", "bbox": [5,6,7,8]}\n'
                 '{"image": "b.png", "label": "Go", "bbox": [9,9,9,9]}\n')
    samples = bench.load_jsonl(str(p))
    assert len(samples) == 2
    assert len(dict(samples)["a.png"]) == 2
