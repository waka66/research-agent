"""确定性统计，不访问数据库、checkpoint、模型或工具。"""

import math
from collections import Counter, defaultdict
from datetime import UTC, datetime

from research_agent.schemas.history import ExecutionResponse, RecordResponse
from research_agent.schemas.report import (
    AttemptIssue,
    ExecutionReport,
    ModelSummary,
    NodeTiming,
    RecordCounts,
    RecordIssue,
    RecordReference,
    SearchSummary,
    SlowNode,
    TimingSummary,
)

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
UNFINISHED_RECORD_STATUSES = {"running", "unknown", "interrupted"}
PROBLEM_STATUSES = {"failed", "cancelled", "degraded", "partial_success", "unknown", "interrupted"}
TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def _tokens(value):
    if not isinstance(value, dict) or not all(
        type(value.get(key)) is int and value[key] >= 0 for key in TOKEN_KEYS
    ):
        return None
    return {key: value[key] for key in TOKEN_KEYS}


def _model_summary(nodes, complete):
    calls = retries = failures = cancelled = missing = 0
    known = []
    usages_complete = True
    for node in nodes:
        attempts = node.payload.get("attempts") or []
        if not attempts:
            continue  # 不汇总没有 attempts 的父节点/未来的汇总字段，避免重复算子节点用量。
        calls += len(attempts)
        retries += max(len(attempts) - 1, 0)
        failures += sum(item.get("status") == "failed" for item in attempts)
        cancelled += sum(item.get("status") == "cancelled" for item in attempts)
        missing += sum(_tokens(item.get("usage")) is None for item in attempts)
        usage = _tokens(node.payload.get("usage"))
        if usage is not None:
            known.append(usage)
        usages_complete &= (
            node.payload.get("usage_complete") is True
            and usage is not None
            and all(_tokens(item.get("usage")) is not None for item in attempts)
        )
    usage = {key: sum(item[key] for item in known) for key in TOKEN_KEYS} if known else None
    if calls == 0 and complete:
        usage = dict.fromkeys(TOKEN_KEYS, 0)
    return ModelSummary(
        call_count=calls,
        retry_count=retries,
        failed_attempt_count=failures,
        cancelled_attempt_count=cancelled,
        missing_usage_attempts=missing,
        usage=usage,
        usage_complete=complete and usages_complete,
    )


def _search_summary(records, complete):
    tools = [r for r in records if r.kind == "tool" and r.name == "web_search_tool"]
    cache = Counter(r.payload.get("cache_status") for r in tools)
    remote_unknown = sum(type(r.payload.get("remote_requested")) is not bool for r in tools)
    cache_unknown = len(tools) - cache["hit"] - cache["miss"]
    return SearchSummary(
        tool_call_count=len(tools),
        cache_hits=cache["hit"],
        cache_misses=cache["miss"],
        cache_unknown=cache_unknown,
        remote_requested_count=sum(r.payload.get("remote_requested") is True for r in tools),
        remote_request_unknown_count=remote_unknown,
        succeeded_count=sum(r.status == "succeeded" for r in tools),
        failed_count=sum(r.status == "failed" for r in tools),
        cancelled_count=sum(r.status == "cancelled" for r in tools),
        complete=complete and not remote_unknown and not cache_unknown,
    )


def _timing_summary(execution, nodes, now):
    duration = elapsed = None
    if execution.status == "interrupted":
        basis = "unknown_interruption_time"
    elif execution.started_at is None:
        basis = "not_started"
    elif execution.finished_at is not None:
        basis = "finished"
        value = (execution.finished_at - execution.started_at).total_seconds() * 1000
        duration = elapsed = round(value, 2) if value >= 0 else None
    else:
        basis = "in_progress"
        value = (now - execution.started_at).total_seconds() * 1000
        elapsed = round(value, 2) if value >= 0 else None
    groups = defaultdict(list)
    measured = []
    for node in nodes:
        groups[node.name].append(node)
        if (
            node.duration_ms is not None
            and math.isfinite(node.duration_ms)
            and node.duration_ms >= 0
        ):
            measured.append(node)
    timings = []
    for name, group in sorted(groups.items()):
        values = [node.duration_ms for node in measured if node.name == name]
        timings.append(
            NodeTiming(
                name=name,
                call_count=len(group),
                measured_count=len(values),
                inclusive_duration_ms_sum=round(sum(values), 2) if values else None,
                max_duration_ms=max(values) if values else None,
            )
        )
    slow = sorted(measured, key=lambda r: (-r.duration_ms, r.started_at, r.record_id))[:5]
    return TimingSummary(
        duration_ms=duration,
        elapsed_ms=elapsed,
        basis=basis,
        node_timings=timings,
        slow_nodes=[
            SlowNode(
                **RecordReference.model_validate(r.model_dump()).model_dump(),
                status=r.status,
                duration_ms=r.duration_ms,
            )
            for r in slow
        ],
    )


def _record_issues(records):
    index = {r.record_id: r for r in records}
    issues = []
    links_complete = True
    for record in records:
        attempts = record.payload.get("attempts") or []
        failed = [
            AttemptIssue(
                attempt=item.get("attempt", number),
                status=item["status"],
                error=item.get("error"),
            )
            for number, item in enumerate(attempts, 1)
            if item.get("status") in {"failed", "cancelled"}
        ]
        warnings = record.payload.get("warnings") or []
        parents, visited, parent = [], {record.record_id}, record.parent_record_id
        while parent is not None:
            if parent in visited or parent not in index:
                links_complete = False
                break
            visited.add(parent)
            item = index[parent]
            parents.append(RecordReference.model_validate(item.model_dump()))
            parent = item.parent_record_id
        if record.status not in PROBLEM_STATUSES and not failed and not warnings:
            continue
        issues.append(
            RecordIssue(
                **RecordReference.model_validate(record.model_dump()).model_dump(),
                status=record.status,
                error=record.payload.get("error"),
                warnings=warnings,
                failed_attempts=failed,
                parents=list(reversed(parents)),
            )
        )
    # 先展示内层记录，再展示上层传播；不推断唯一根因，也不计算“独立故障次数”。
    issues.sort(key=lambda item: (-len(item.parents), item.record_id))
    return issues, links_complete


def build_execution_report(execution, records, *, now: datetime | None = None) -> ExecutionReport:
    execution = ExecutionResponse.model_validate(execution)
    records = [RecordResponse.model_validate(record) for record in records]
    if any(record.execution_id != execution.execution_id for record in records):
        raise ValueError("运行报告不能合并不同 execution 的记录")
    now = now or datetime.now(UTC)
    is_final = execution.status in TERMINAL_STATUSES
    nodes = [record for record in records if record.kind == "node"]
    unfinished = sum(record.status in UNFINISHED_RECORD_STATUSES for record in records)
    issues, links_complete = _record_issues(records)
    complete = (
        is_final
        and execution.status != "interrupted"
        and execution.records_complete
        and not unfinished
        and links_complete
    )
    models = _model_summary(nodes, complete)
    search = _search_summary(records, complete)
    warnings = []
    if not is_final:
        warnings.append("执行进行中，统计仅代表当前已保存记录")
    if not execution.records_complete or unfinished:
        warnings.append("过程记录未齐全，数量与消耗可能不完整")
    if not links_complete:
        warnings.append("部分父调用关系无法还原")
    if not models.usage_complete:
        warnings.append("token 仅汇总已返回的用量，不能当作全部实际消耗")
    if not search.complete:
        warnings.append("搜索统计不完整，未知的缓存/远程调用信息未按零处理")
    if execution.result_summary is None:
        warnings.append("本次执行没有已保存的成果快照")
    elif not execution.result_summary.available:
        warnings.append(f"成果快照不可用：{execution.result_summary.reason}")
    if execution.status == "interrupted":
        warnings.append("中断的发现时间不代表实际结束时间，无法给出精确执行耗时")
    return ExecutionReport(
        as_of=now,
        is_final=is_final,
        execution=execution,
        records=RecordCounts(
            node_count=len(nodes),
            tool_count=len(records) - len(nodes),
            status_counts=dict(Counter(record.status for record in records)),
            unfinished_count=unfinished,
            complete=complete,
        ),
        timing=_timing_summary(execution, nodes, now),
        models=models,
        search=search,
        issues=issues,
        warnings=warnings,
    )
