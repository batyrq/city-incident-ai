from __future__ import annotations

import argparse
from pprint import pformat
from pathlib import Path
from typing import Any

from src.infer.runtime import run_demo_video
from src.utils.io import load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pretrained-first local video inference pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-name", default=None, help="Override the run name used for reports and alerts.")
    parser.add_argument("--report-path", default=None, help="Override the summary JSON path.")
    parser.add_argument("--alerts-dir", default=None, help="Override alerts output root directory.")
    parser.add_argument("--mode", default=None, help="Override mode: pretrained_yolo11s or open_vocab.")
    parser.add_argument("--model-type", default=None, help="Override model type: yolo11s, yolo-world, or yoloe.")
    parser.add_argument("--model", default=None, help="Optional checkpoint/weights override.")
    parser.add_argument("--prompts", default=None, help="Comma-separated prompts for open-vocabulary mode.")
    parser.add_argument("--classes", default=None, help="Comma-separated class names to keep (e.g. person,car), or 'all' for no filter.")
    args = parser.parse_args()

    config = _apply_overrides(load_yaml(args.config), args)
    source = Path(args.source)
    results = []
    if source.is_dir():
        for video_path in sorted(source.glob("*")):
            per_video_config = dict(config)
            base_run_name = config.get("run_name", source.stem)
            per_video_config["run_name"] = f"{base_run_name}_{video_path.stem}"
            per_video_config["report_path"] = str(Path("reports") / f"{per_video_config['run_name']}_summary.json")
            results.append(run_demo_video(str(video_path), per_video_config, model_path=args.model))
    else:
        results.append(run_demo_video(str(source), config, model_path=args.model))
    report_path = Path(config.get("report_path", f"reports/{config.get('run_name', source.stem)}_summary.json"))
    if len(results) == 1:
        write_json(results[0], report_path)
    else:
        batch_path = report_path.with_name(f"{report_path.stem}_batch.json")
        write_json(results, batch_path)
        write_json({"batch_report_path": str(batch_path), "runs": results}, report_path)
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
    if args.classes:
        config["classes"] = [item.strip() for item in args.classes.split(",") if item.strip()]
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
