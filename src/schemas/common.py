from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DecisionLiteral = Literal[
    "train_full",
    "train_smoke_test",
    "use_pretrained_only",
    "skip_branch",
]


class TrainingDecision(BaseModel):
    branch: str
    decision: DecisionLiteral
    reasons: list[str] = Field(default_factory=list)
    stats_snapshot: dict = Field(default_factory=dict)
    status: str = "skipped"


class DatasetRecord(BaseModel):
    sample_id: str
    source_dataset: str
    task_type: str
    image_path: str | None = None
    label_path: str | None = None
    mask_path: str | None = None
    video_path: str | None = None
    video_id: str | None = None
    frame_id: str | None = None
    camera_id: str | None = None
    event_id: str | None = None
    width: int | None = None
    height: int | None = None
    classes_raw: list[str] = Field(default_factory=list)
    classes_mapped: list[str] = Field(default_factory=list)
    is_negative: bool = False
    weather: str | None = None
    time_of_day: str | None = None
    split_group: str = ""
    roi_tags: list[str] = Field(default_factory=list)
