"""单次 execution 的运行报告；数量均基于已保存记录。"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel

from research_agent.schemas.history import ExecutionResponse, RecordKind, RecordStatus


class UsageTotals(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class RecordReference(BaseModel):
    record_id: UUID
    parent_record_id: UUID | None
    kind: RecordKind
    name: str
    task_id: str | None
    round: int | None


class SlowNode(RecordReference):
    status: RecordStatus
    duration_ms: float


class NodeTiming(BaseModel):
    name: str
    call_count: int
    measured_count: int
    inclusive_duration_ms_sum: float | None
    max_duration_ms: float | None


class TimingSummary(BaseModel):
    duration_ms: float | None
    elapsed_ms: float | None
    basis: Literal["finished", "in_progress", "not_started", "unknown_interruption_time"]
    node_timings: list[NodeTiming]
    slow_nodes: list[SlowNode]


class RecordCounts(BaseModel):
    node_count: int
    tool_count: int
    status_counts: dict[str, int]
    unfinished_count: int
    complete: bool


class ModelSummary(BaseModel):
    call_count: int
    retry_count: int
    failed_attempt_count: int
    cancelled_attempt_count: int
    missing_usage_attempts: int
    usage: UsageTotals | None
    usage_complete: bool


class SearchSummary(BaseModel):
    tool_call_count: int
    cache_hits: int
    cache_misses: int
    cache_unknown: int
    remote_requested_count: int
    remote_request_unknown_count: int
    succeeded_count: int
    failed_count: int
    cancelled_count: int
    complete: bool


class AttemptIssue(BaseModel):
    attempt: int
    status: str
    error: dict[str, Any] | None


class RecordIssue(RecordReference):
    status: RecordStatus
    error: dict[str, Any] | None
    warnings: list[str]
    failed_attempts: list[AttemptIssue]
    parents: list[RecordReference]


class ExecutionReport(BaseModel):
    schema_version: Literal[1] = 1
    as_of: datetime
    is_final: bool
    execution: ExecutionResponse
    records: RecordCounts
    timing: TimingSummary
    models: ModelSummary
    search: SearchSummary
    issues: list[RecordIssue]
    warnings: list[str]
