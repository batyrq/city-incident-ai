import logging
from types import SimpleNamespace

from src.infer import runtime


class _Det:
    names = {0: "person"}

    def track(self, **kw):
        raise RuntimeError("boom")

    def predict(self, **kw):
        return [SimpleNamespace(boxes=None)]


def test_tracker_failure_is_logged_once_and_recorded(caplog):
    runtime._reset_tracker_fallback()
    spec = runtime.DemoSpec(mode="pretrained_yolo11s", model_type="yolo11s", weights="x")
    with caplog.at_level(logging.WARNING, logger=runtime.logger.name):
        for _ in range(3):
            runtime._infer_frame(_Det(), object(), spec, {}, track=True)
    assert len([r for r in caplog.records if "Tracker failed" in r.getMessage()]) == 1
    assert runtime._TRACKER_FALLBACK["frames"] == 3
    assert "boom" in runtime._TRACKER_FALLBACK["first_error"]
