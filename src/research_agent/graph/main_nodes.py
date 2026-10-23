import asyncio
import logging

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from research_agent.graph.error import WorkerFatalError
from research_agent.graph.main_prompt import PLANNER_SYSTEM_PROMPT, SYNTH_SYSTEM_PROMPT
from research_agent.graph.observability import model_attempt, node_log, record_model_response
from research_agent.graph.render import render_report
from research_agent.graph.state import ResearchState
from research_agent.graph.worker_workflow import worker_graph
from research_agent.llm import llm, llm_synth
from research_agent.schemas.research import ResearcherInput, ResearchPlan, ResearchReport
from research_agent.validators.report import validate_report

logger = logging.getLogger("research_agent/graph/main_nodes.py")


async def Planner(state: ResearchState, config: RunnableConfig):
    """把研究问题拆分为独立任务；调用失败或解析失败直接向上传播。"""
    with node_log(logger, "Planner", state, config) as record:
        planner_model = llm.with_structured_output(
            ResearchPlan,
            method="function_calling",
            include_raw=True,
        )
        with model_attempt(record, 1) as attempt:
            response = await planner_model.ainvoke(
                [
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=state["question"]),
                ]
            )
            research_plan = record_model_response(attempt, response)
            if research_plan is None:
                raise ValueError(
                    f"Planner 结构化输出失败：{response.get('parsing_error')}"
                ) from response.get("parsing_error")
        record["output_summary"] = {"task_count": len(research_plan.tasks)}
        return {"research_plan": research_plan, "status": "running"}


def dispatch_research_tasks(state: ResearchState):
    """
    分发主节点的任务给子节点，让子节点并行运行
    """

    return [Send("research_node", {"task": task}) for task in state["research_plan"].tasks[:1]]


async def research_node(state: ResearcherInput, config: RunnableConfig):
    """运行一个研究子图；耗时包含该任务内所有搜索、提取和规划。"""
    with node_log(logger, "research_node", state, config) as record:
        worker_result = await worker_graph.ainvoke(
            {
                "task": state["task"],
                "points": {},
                "findings": {},
                "sources": {},
                "pending_queries": [],
                "queries_tried": [],
                "out_of_scope": [],
                "round": 0,
                "no_progress": 0,
                "stop_now": False,
                "stop_reason": "",
                "new_results": [],
                "round_findings": [],
                "search_calls": 0,
                "error_calls": 0,
            },
            config=config,
        )
        result = worker_result["result"]
        record["output_summary"] = {
            "point_count": len(result.points),
            "finding_count": len(result.findings),
            "source_count": len(result.sources),
            "round_count": worker_result.get("round", 0),
            "stop_reason": worker_result.get("stop_reason"),
        }
        return {"research_results": [result]}


MAX_SUPPORTS_PER_POINT = 3
MAX_CONTRASTS_PER_POINT = 3


def _evidence_block(result) -> tuple[str, int, int]:
    """按点分组渲染证据。只给 claim，不给 quote；context 不送。

    返回 (文本, 用到的证据数, 被裁掉的证据数)。
    result 是 ResearchResult（有 .points 和 .findings）。
    """
    by_point: dict[str, list] = {}
    for f in result.findings:
        for pid in f.point_ids:
            by_point.setdefault(pid, []).append(f)

    lines, used, skipped = [], 0, 0
    for p in result.points:
        fs = by_point.get(p.id, [])
        sup = [f for f in fs if f.stance == "supports"][:MAX_SUPPORTS_PER_POINT]
        con = [f for f in fs if f.stance == "contradicts"][:MAX_CONTRASTS_PER_POINT]
        skipped += len(fs) - len(sup) - len(con)
        if not sup and not con:
            continue
        lines.append(f"■ {p.id}（{p.status}）：{p.text}")
        for f in sup:
            lines.append(f"  + [{f.id}] {f.claim}")
        for f in con:
            lines.append(f"  - [{f.id}] {f.claim}")
        used += len(sup) + len(con)
    return "\n".join(lines), used, skipped


async def Synthesizer(state: ResearchState, config: RunnableConfig):
    with node_log(logger, "Synthesizer", state, config) as record:
        results = state["research_results"]
        findings_by_id = {f.id: f for r in results for f in r.findings}
        evidence_parts, thin, missing, contested = [], [], [], []
        used = skipped = 0
        for result in results:
            block, count, discarded = _evidence_block(result)
            used += count
            skipped += discarded
            if block:
                evidence_parts.append(block)
            for point in result.points:
                if point.status == "thin":
                    thin.append(point.text)
                elif point.status == "missing":
                    missing.append(point.text)
                if point.contested:
                    contested.append(point.text)

        def lines(items):
            return "\n".join(f"- {text}" for text in items) or "（无）"

        prompt = (
            f"研究计划：{state['research_plan']}\n\n"
            f"【证据（按点分组，+ 支持 / - 反驳）】\n{chr(10).join(evidence_parts)}\n\n"
            f"【仅单一来源的点】\n{lines(thin)}\n\n"
            f"【没查到的点（必须写成缺口）】\n{lines(missing)}\n\n"
            f"【存在冲突的点（两边都要写）】\n{lines(contested)}"
        )
        summary = {
            "research_result_count": len(results),
            "used_evidence_count": used,
            "omitted_evidence_count": skipped,
            "thin_point_count": len(thin),
            "missing_point_count": len(missing),
            "contested_point_count": len(contested),
            "prompt_chars": len(prompt),
        }
        record["output_summary"] = summary
        model = llm_synth.with_structured_output(
            ResearchReport, method="function_calling", include_raw=True
        )
        messages = [SystemMessage(content=SYNTH_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        report = None
        for attempt in (1, 2):
            response_received = False
            try:
                with model_attempt(record, attempt) as attempt_record:
                    response = await asyncio.wait_for(model.ainvoke(messages), timeout=180)
                    response_received = True
                    parsed = record_model_response(attempt_record, response)
                    if parsed is not None:
                        fatal, warnings = validate_report(parsed, findings_by_id)
                        if warnings:
                            record.setdefault("warnings", []).extend(warnings)
                        if fatal:
                            attempt_record["status"] = "failed"
                            attempt_record["error"] = {
                                "type": "ReportValidationError",
                                "message": "; ".join(fatal),
                            }
                            messages = messages + [
                                HumanMessage(
                                    content=(
                                        "上一次报告未通过校验，原因：\n"
                                        + "\n".join(f"- {reason}" for reason in fatal)
                                        + "\n请只使用我给出的证据 id，"
                                        "没证据的方面写进缺口。重新输出。"
                                    )
                                )
                            ]
                        else:
                            report = parsed
            except TimeoutError as exc:
                raise WorkerFatalError(f"Synthesizer 超时（提示词 {len(prompt)} 字）") from exc
            except Exception:
                if response_received:
                    # 校验或日志处理的程序错误不按模型请求失败重试。
                    raise
                continue
            if report is not None:
                break

        if report is None:
            record["status"] = "degraded"
            summary["render_mode"] = "evidence_only"
            summary["reason"] = "no_valid_report_after_retries"
            return {"report": None, "status": "partial"}
        summary.update(
            {
                "render_mode": "structured_report",
                "key_conclusion_count": len(report.key_conclusions),
                "section_count": len(report.sections),
            }
        )
        return {"report": report}


def _question_text(state) -> str:
    """state["question"] 实际是 [HumanMessage]，取其中的文本。"""
    q = state.get("question")
    if isinstance(q, str):
        return q
    if isinstance(q, (list, tuple)):
        parts = []
        for m in q:
            content = getattr(m, "content", m)
            if isinstance(content, str):
                parts.append(content)
        return " ".join(parts).strip()
    return ""


async def generator(state: ResearchState, config: RunnableConfig):
    with node_log(logger, "generator", state, config) as record:
        question = _question_text(state)
        results = state.get("research_results") or []
        report = state.get("report")
        markdown = render_report(report, results, question)
        record["output_summary"] = {
            "render_mode": "structured_report" if report is not None else "evidence_only",
            "research_result_count": len(results),
            "markdown_chars": len(markdown),
        }
        if report is None:
            record["status"] = "degraded"
            record["output_summary"]["reason"] = "no_structured_report"
        return {"final_answer": markdown, "status": "completed"}
