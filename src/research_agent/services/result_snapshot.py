"""收尾时读取一次 checkpoint，之后历史查询不再读取可变的研究状态。"""

import asyncio
import logging
from datetime import UTC, datetime
from uuid import UUID

from research_agent.schemas.research import ResearchPlan, ResearchResult
from research_agent.schemas.summary import ResultSummary, unavailable_summary

logger = logging.getLogger(__name__)
SNAPSHOT_TIMEOUT_SECONDS = 5.0


def build_result_summary(values: dict, *, checkpoint_id: str, graph_completed: bool) -> dict:
    """统计已完成子任务的成果；相同任务取最后一份结果，证据与来源全局去重。"""
    plan = values.get("research_plan")
    plan = ResearchPlan.model_validate(plan) if plan is not None else None
    by_task = {}
    for value in values.get("research_results") or []:
        result = ResearchResult.model_validate(value)
        by_task[result.task_id] = result
    findings = {finding.id for result in by_task.values() for finding in result.findings}
    sources = {
        source.norm_url or source.url or source.id
        for result in by_task.values()
        for source in result.sources
    }
    points = {point.id: point for result in by_task.values() for point in result.points}
    answer = values.get("final_answer")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("checkpoint 的 final_answer 必须是字符串或 None")
    generated = answer is not None
    return ResultSummary(
        available=True,
        captured_at=datetime.now(UTC),
        checkpoint_id=checkpoint_id,
        graph_completed=graph_completed,
        planned_task_count=len(plan.tasks) if plan is not None else None,
        completed_task_count=len(by_task),
        finding_count=len(findings),
        source_count=len(sources),
        point_status_counts={
            status: sum(point.status == status for point in points.values())
            for status in ("open", "done", "thin", "missing")
        },
        contested_point_count=sum(point.contested for point in points.values()),
        report_generated=generated,
        render_mode=("structured_report" if values.get("report") is not None else "evidence_only")
        if generated
        else None,
        markdown_chars=len(answer) if generated else None,
    ).model_dump(mode="json")


async def capture_result_summary(agent, run_id: UUID) -> dict:
    if agent is None:
        return unavailable_summary("not_captured")
    try:
        async with asyncio.timeout(SNAPSHOT_TIMEOUT_SECONDS):
            snapshot = await agent.aget_state(
                {"configurable": {"thread_id": str(run_id)}},
                subgraphs=False,
            )
    except Exception:
        logger.exception("读取成果快照失败：%s", run_id)
        return unavailable_summary("checkpoint_unavailable")
    try:
        checkpoint_id = (snapshot.config or {}).get("configurable", {}).get("checkpoint_id")
        if not checkpoint_id:
            return unavailable_summary("no_checkpoint")
        return build_result_summary(
            snapshot.values or {},
            checkpoint_id=checkpoint_id,
            graph_completed=not snapshot.next,
        )
    except Exception:
        logger.exception("构建成果快照失败：%s", run_id)
        return unavailable_summary("snapshot_invalid")
