from __future__ import annotations

import argparse
from pprint import pformat
from pathlib import Path
from typing import Any

from src.infer.runtime import run_inference_stream
from src.utils.io import load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTSP inference pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--alerts-dir", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompts", default=None)
    args = parser.parse_args()
    config = _apply_overrides(load_yaml(args.config), args)
    results = []
    for stream in config.get("streams", []):
        stream_cfg = dict(config)
        stream_cfg["run_name"] = stream_cfg.get("run_name") or stream["camera_id"]
        stream_cfg["report_path"] = str(Path("reports") / f"{stream_cfg['run_name']}_{stream['camera_id']}_summary.json")
        results.append(run_inference_stream(stream["uri"], stream["camera_id"], stream_cfg, args.model))
    report_path = config.get("report_path", "reports/infer_rtsp_summary.json")
    write_json(results[0] if len(results) == 1 else results, report_path)
    for result in results:
        print(_format_console_summary(result))


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.run_name:
        config["run_name"] = args.run_name
    if args.report_path:
        config["report_path"] = args.report_path
    if args.alerts_dir:
        config["alerts_root"] = args.alerts_dir
    if args.mode:
        config["mode"] = args.mode
    if args.model_type:
        config["model_type"] = args.model_type
    if args.prompts:
        config["prompts"] = [item.strip() for item in args.prompts.split(",") if item.strip()]
    if args.model:
        config["model_weights"] = args.model
    return config


def _format_console_summary(result: dict[str, Any]) -> str:
    payload = {
        "status": result.get("status"),
        "run_name": result.get("run_name"),
        "mode": result.get("mode"),
        "model_type": result.get("model_type"),
        "source_path": result.get("source_path"),
        "frames_processed": result.get("frames_processed"),
        "events_detected": result.get("events_detected", {}),
        "output_video_path": result.get("output_video_path"),
        "skipped_reason": result.get("skipped_reason"),
    }
    return pformat(payload, sort_dicts=False)


if __name__ == "__main__":
    main()
