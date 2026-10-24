"""历史查询的响应与筛选类型，与研究的业务 state 分开。"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from research_agent.schemas.summary import ResultSummary

ExecutionStatus = Literal["pending", "running", "completed", "failed", "cancelled", "interrupted"]
RecordKind = Literal["node", "tool"]
RecordStatus = Literal[
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "skipped",
    "degraded",
    "partial_success",
    "interrupted",
    "unknown",
]


class ExecutionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    execution_id: UUID
    run_id: UUID
    mode: Literal["start", "resume"]
    status: ExecutionStatus
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: dict[str, Any] | None
    records_complete: bool
    result_summary: ResultSummary | None = None


class RecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    record_id: UUID
    execution_id: UUID
    parent_record_id: UUID | None
    kind: RecordKind
    name: str
    task_id: str | None
    round: int | None
    status: RecordStatus
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None
    payload: dict[str, Any]


class ExecutionPage(BaseModel):
    run_id: UUID
    items: list[ExecutionResponse]
    next_cursor: str | None


class RecordPage(BaseModel):
    execution: ExecutionResponse
    items: list[RecordResponse]
    next_cursor: str | None
