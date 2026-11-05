import numpy as np

from demo import pick_backend
from src.infer.overlay import Overlay


def _det(tid, x):
    return {"track_id": tid, "class_name": "car", "confidence": 0.9, "x1": x, "y1": 50, "x2": x + 40, "y2": 90}


def test_overlay_draws_boxes_trail_banner_and_hud():
    ov = Overlay({"trail_seconds": 1.0, "banner_seconds": 1.0}, fps=10)
    frame = np.zeros((200, 300, 3), np.uint8)
    for i in range(1, 6):
        out = ov.draw(frame, [_det(7, 10 + 10 * i)], ["crowding"] if i == 3 else [], i, 12.0)
    assert out.shape == frame.shape and out.any()
    assert not frame.any()  # input frame is not modified
    assert len(ov.trails[7]) == 5
    assert tuple(out[5, 5]) == (0, 0, 200)  # banner still visible 2 frames after the event (banner_seconds=1 at 10 fps)


def test_overlay_drops_trail_of_vanished_track():
    ov = Overlay({}, fps=10)
    frame = np.zeros((100, 100, 3), np.uint8)
    ov.draw(frame, [_det(1, 5)], [], 1, 10.0)
    ov.draw(frame, [], [], 10, 10.0)
    assert 1 not in ov.trails


def test_pick_backend_explicit_cpu():
    assert pick_backend("cpu", None) == ("cpu", "yolo11s.onnx")
