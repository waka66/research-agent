"""一次执行结束时已固化的研究成果摘要，不保存报告正文。"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

SummaryUnavailableReason = Literal[
    "not_captured",
    "no_checkpoint",
    "checkpoint_unavailable",
    "snapshot_invalid",
    "process_interrupted",
]


class ResultSummary(BaseModel):
    schema_version: Literal[1] = 1
    scope: Literal["research_checkpoint_at_execution_end"] = "research_checkpoint_at_execution_end"
    available: bool
    captured_at: datetime
    checkpoint_id: str | None = None
    reason: SummaryUnavailableReason | None = None
    graph_completed: bool | None = None
    planned_task_count: int | None = None
    completed_task_count: int | None = None
    finding_count: int | None = None
    source_count: int | None = None
    point_status_counts: dict[str, int] | None = None
    contested_point_count: int | None = None
    report_generated: bool | None = None
    render_mode: Literal["structured_report", "evidence_only"] | None = None
    markdown_chars: int | None = None


def unavailable_summary(reason: SummaryUnavailableReason) -> dict:
    return ResultSummary(
        available=False,
        captured_at=datetime.now(UTC),
        reason=reason,
    ).model_dump(mode="json")
