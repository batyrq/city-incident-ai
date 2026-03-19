from __future__ import annotations

from pathlib import Path

from src.utils.env import has_package


def generate_masks_with_sam2(config: dict, image_dir: str, output_dir: str) -> dict:
    if not has_package("sam2"):
        return {"enabled": False, "status": "skipped", "reason": "sam2 is not installed; auxiliary assist disabled gracefully."}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return {
        "enabled": bool(config.get("enabled", False)),
        "status": "ready",
        "reason": "Implement promptable mask proposal flow here when SAM2 package/checkpoint are available.",
        "image_dir": image_dir,
        "output_dir": output_dir,
    }
