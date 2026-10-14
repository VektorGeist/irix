"""The autonomous PLAY loop, closed-loop on the synthetic moving world:
perceive (describe_scene) -> decide (structured-output reflex) -> act.
Deterministic — no model, no game needed."""
import pytest

cv2 = pytest.importorskip("cv2")

from irix import play


def test_play_loop_avoids_the_obstacle():
    out = play.run(ticks=16)
    assert out["ok"], {k: v for k, v in out.items() if k != "log"}
    assert not out["collided"] and not out["off_road"]
    assert out["reflex_fired"]                    # the swerve actually happened
    whys = [t["why"] for t in out["log"]]
    assert any(w.startswith("reflex") for w in whys)
    assert any(w.startswith("lane-keep") for w in whys)


def test_play_loop_builds_world_memory():
    out = play.run(ticks=6)
    assert out["world_mapped_m2"] and out["world_mapped_m2"] > 0
    assert out["fps"] > 0


def test_policy_reads_structured_output_only():
    # the brain contract: the reflex never sees pixels — only describe_scene's dict
    r = play.Reflex()
    steer, brake, why = r({"occupancy": {"clear_ahead_m": 5.0, "occupied": [
        {"x_m": 0.5, "y_m": 5.0, "range_m": 5.0, "bearing_deg": 5.7}]}})
    assert steer < 0 and brake and why.startswith("reflex")   # blocker right -> go left
    steer2, _, _ = r({"occupancy": {"clear_ahead_m": 5.0, "occupied": []}})
    assert steer2 == steer                                     # direction stays latched
