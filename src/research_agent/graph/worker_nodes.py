import asyncio
import hashlib
import logging

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from research_agent.graph.error import WorkerFatalError
from research_agent.graph.observability import model_attempt, node_log, record_model_response
from research_agent.graph.state import ResearchWorkerState
from research_agent.graph.tools import web_search_tool
from research_agent.graph.worker_prompt import (
    EXTRACT_SYSTEM_PROMPT,
    PLAN_NEXT_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
)
from research_agent.llm import llm
from research_agent.schemas.research import (
    Query,
    ResearchResult,
    ResearchWorkerPlan,
    Source,
    make_extract_schema,
    make_next_schema,
)
from research_agent.validators.findings import validate_extract
from research_agent.validators.points import (
    add_discovered,
    close_finished_points,
    evidence_count,
    finalize_points,
    new_domain_evidence,
    required_evidence,
    seed_points,
    should_stop,
)
from research_agent.validators.sources import norm_url, registrable_domain

logger = logging.getLogger("research_agent/graph/worker_nodes.py")


async def research_plan(state: ResearchWorkerState, config: RunnableConfig):
    """生成初始研究点和查询；已有研究点时跳过播种。"""
    with node_log(
        logger, "research_plan", state, config, round_number=state.get("round", 0)
    ) as record:
        task = state["task"]
        if state.get("points"):
            record["status"] = "skipped"
            record["output_summary"] = {"reason": "existing_points"}
            return {}
        plan_model = llm.with_structured_output(
            ResearchWorkerPlan,
            method="function_calling",
            include_raw=True,
        )
        with model_attempt(record, 1) as attempt:
            response = await plan_model.ainvoke(
                [
                    SystemMessage(content=PLAN_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"任务 id：{task.id}\n研究问题：{task.question}\n"
                            f"研究目标：{task.objective}"
                        )
                    ),
                ]
            )
            parsed = record_model_response(attempt, response)
            if parsed is None:
                parsing_error = response.get("parsing_error")
                raise WorkerFatalError(
                    f"research_plan 解析失败，task_id={task.id}：{parsing_error}"
                ) from parsing_error
        points, queries, warnings = seed_points(parsed, task)
        if warnings:
            record["warnings"] = warnings
        record["output_summary"] = {
            "point_count": len(points),
            "query_count": len(queries),
            "warning_count": len(warnings),
        }
        return {
            "points": points,
            "pending_queries": queries,
            "queries_tried": [],
            "out_of_scope": parsed.out_of_scope or [],
            "round": 0,
        }


MAX_SEARCH_CALLS = 6  # 最大搜索次数
MAX_ERROR_CALLS = 2  # 最大错误次数


async def search_node(state: ResearchWorkerState, config: RunnableConfig):
    """并发搜索、合并来源，并记录本轮查询的执行状态与产出。"""
    with node_log(
        logger, "search_node", state, config, round_number=state.get("round", 0) + 1
    ) as record:
        record["query_errors"] = []
        search_calls = state.get("search_calls", 0)
        error_calls = state.get("error_calls", 0)
        sources = dict(state.get("sources") or {})
        tried = list(state.get("queries_tried") or [])
        queries = list(state.get("pending_queries") or [])

        # 保留现有预算口径：search_calls 累计成功的工具调用，失败另计。
        # 工具可能命中缓存，因此这些次数不能直接当作远程请求数或费用。
        remaining = MAX_SEARCH_CALLS - search_calls
        runnable = queries[: max(remaining, 0)]
        summary = {
            "query_count": len(runnable),
            "succeeded_query_count": 0,
            "failed_query_count": 0,
            "new_source_count": 0,
            "total_source_count": len(sources),
        }
        record["output_summary"] = summary

        if not runnable:
            record["status"] = "skipped"
            summary["reason"] = (
                "search_budget_exhausted" if remaining <= 0 else "no_pending_queries"
            )
            return {
                "pending_queries": [],
                "new_results": [],
                "search_result": [{"query": "", "success": False, "search_result": []}],
                "error_calls": error_calls + (1 if not queries else 0),
            }

        # 给工具日志传递业务上下文；复制 metadata，避免并行研究任务互相覆盖。
        tool_config = {
            **config,
            "metadata": {
                **(config.get("metadata") or {}),
                "research_task_id": record["task_id"],
                "research_round": record["round"],
            },
        }
        # 单条工具异常允许其他查询完成，取消则仍然向外传播。
        outcomes = await asyncio.gather(
            *(web_search_tool.ainvoke(q.text, config=tool_config) for q in runnable),
            return_exceptions=True,
        )

        seen = {s.norm_url for s in sources.values() if s.norm_url}
        new_ids: list[str] = []
        ok_queries: list[str] = []

        for q, outcome in zip(runnable, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    raise outcome
                record["query_errors"].append(
                    {
                        "query": q.text,
                        "type": type(outcome).__name__,
                        "message": str(outcome),
                    }
                )
                summary["failed_query_count"] += 1
                error_calls += 1
                continue

            search_calls += 1
            ok_queries.append(q.text)
            summary["succeeded_query_count"] += 1

            for item in (outcome or {}).get("search_result") or []:
                url = item.get("url") or ""
                key = norm_url(url)
                if not key or key in seen:
                    continue
                seen.add(key)

                sid = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
                if sid in sources:
                    continue

                sources[sid] = Source(
                    id=sid,
                    url=url,
                    norm_url=key,
                    domain=registrable_domain(url),
                    title=item.get("title") or "",
                    summary=item.get("summary") or "",
                )
                new_ids.append(sid)
                summary["new_source_count"] = len(new_ids)
                summary["total_source_count"] = len(sources)

        if not ok_queries:
            # 全部搜索失败仍返回错误计数，由后续 update_node 裁决是否停止。
            record["status"] = "failed"
            record["error"] = {
                "type": "AllSearchQueriesFailed",
                "message": "本轮所有搜索查询失败，详情见 query_errors",
            }
        elif record["query_errors"]:
            record["status"] = "partial_success"
        else:
            # 查询成功但没有结果也是 succeeded，成果由 new_source_count 表达。
            record["status"] = "succeeded"

        return {
            "sources": sources,
            "new_results": [sources[i] for i in new_ids],
            "pending_queries": [],
            "queries_tried": tried + ok_queries,
            "search_calls": search_calls,
            "error_calls": error_calls,
        }


MAX_EXTRACT_RESULTS = 12  # 单轮最多让模型读这么多条结果，控制输入成本


def _render_points(points, open_ids: list[str]) -> str:
    lines = []
    for pid in open_ids:
        p = points[pid]
        line = f"- {p.id}（{p.kind}）：{p.text}"
        if p.accepts:
            line += f"\n    判据：{p.accepts}"
        lines.append(line)
    return "\n".join(lines)


def _render_results(results) -> str:
    return "\n\n".join(
        f"[{s.id}] {s.domain}\n{s.url}\n标题：{s.title}\n摘要：{s.summary}" for s in results
    )


async def extract_node(state: ResearchWorkerState, config: RunnableConfig):
    with node_log(
        logger, "extract_node", state, config, round_number=state.get("round", 0) + 1
    ) as record:
        task = state["task"]
        points = state.get("points") or {}
        new_results = state.get("new_results") or []
        findings = dict(state.get("findings") or {})
        open_ids = [pid for pid, point in points.items() if point.status == "open"]
        summary = {
            "open_point_count": len(open_ids),
            "input_source_count": len(new_results),
            "new_finding_count": 0,
            "total_finding_count": len(findings),
        }
        record["output_summary"] = summary
        if not open_ids or not new_results:
            record["status"] = "skipped"
            summary["reason"] = "no_open_points" if not open_ids else "no_new_results"
            return {"round_findings": []}

        batch = new_results[:MAX_EXTRACT_RESULTS]
        summary["processed_source_count"] = len(batch)
        summary["omitted_source_count"] = len(new_results) - len(batch)
        if summary["omitted_source_count"]:
            record["warnings"] = [
                f"超过单轮提取上限，跳过 {summary['omitted_source_count']} 条来源"
            ]
        by_id = {item.id: item for item in batch}
        schema = make_extract_schema(open_ids, list(by_id))
        system = EXTRACT_SYSTEM_PROMPT.format(
            out_of_scope="、".join(state.get("out_of_scope") or []) or "（无）"
        )
        human = (
            f"任务：{task.question}\n目标：{task.objective}\n\n"
            f"【点清单】（只含仍然开放的点）\n{_render_points(points, open_ids)}\n\n"
            f"【本轮搜索结果】\n{_render_results(batch)}"
        )
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        model = llm.with_structured_output(schema, method="function_calling", include_raw=True)
        parsed = None
        for attempt in (1, 2):
            try:
                with model_attempt(record, attempt) as attempt_record:
                    response = await model.ainvoke(messages, timeout=90)
                    parsed = record_model_response(attempt_record, response)
            except TimeoutError as exc:
                raise WorkerFatalError("extract 调用超时 90s") from exc
            except Exception:
                continue
            if parsed is not None:
                break
            error = attempt_record["error"]["message"]
            messages = messages + [
                HumanMessage(
                    content=(
                        f"上一次输出未通过校验：{error}\n"
                        f"点 id 只能从 {open_ids} 里选，"
                        f"结果 id 只能从 {list(by_id)} 里选。请重新输出。"
                    )
                )
            ]
        if parsed is None:
            raise WorkerFatalError(
                f"extract {task.id} 连续两次未取得有效结构化输出，round={state.get('round')}"
            )

        accepted, dropped = validate_extract(parsed.findings, by_id, set(open_ids))
        new_findings = []
        for finding in accepted:
            if finding.id not in findings:
                findings[finding.id] = finding
                new_findings.append(finding)
        if dropped:
            record.setdefault("warnings", []).extend(dropped)
        summary.update(
            {
                "candidate_finding_count": len(parsed.findings),
                "accepted_finding_count": len(accepted),
                "dropped_finding_count": len(dropped),
                "duplicate_finding_count": len(accepted) - len(new_findings),
                "new_finding_count": len(new_findings),
                "total_finding_count": len(findings),
            }
        )
        return {"findings": findings, "round_findings": new_findings}


MAX_NO_PROGRESS = 2  # 连续几轮没有新证据就停
MAX_ROUNDS = 6  # 轮数硬上限


async def update_node(state: ResearchWorkerState, config: RunnableConfig):
    with node_log(
        logger, "update_node", state, config, round_number=state.get("round", 0) + 1
    ) as record:
        points = {pid: point.model_copy() for pid, point in (state.get("points") or {}).items()}
        findings = state.get("findings") or {}
        sources = state.get("sources") or {}
        round_findings = state.get("round_findings") or []
        round_no = state.get("round", 0) + 1
        closed = close_finished_points(points, findings, sources)
        round_ids = {finding.id for finding in round_findings}
        previous = {fid: finding for fid, finding in findings.items() if fid not in round_ids}
        gained = new_domain_evidence(round_findings, previous, sources)
        no_progress = 0 if gained > 0 else state.get("no_progress", 0) + 1
        # 使用最终的 no_progress 裁决，日志与返回给路由的状态保持一致。
        stop, reason = should_stop(
            points,
            round_no,
            no_progress,
            state.get("search_calls", 0),
            state.get("error_calls", 0) >= MAX_ERROR_CALLS,
        )
        record["output_summary"] = {
            "closed_point_count": len(closed),
            "open_point_count": sum(point.status == "open" for point in points.values()),
            "new_point_domain_pair_count": gained,
            "no_progress": no_progress,
            "stop_now": stop,
            "stop_reason": reason,
        }
        return {
            "points": points,
            "round": round_no,
            "no_progress": no_progress,
            "stop_now": stop,
            "stop_reason": reason,
        }


MAX_QUERIES_PER_ROUND = 2
MAX_TRIED_IN_PROMPT = 12


async def plan_next(state: ResearchWorkerState, config: RunnableConfig):
    with node_log(logger, "plan_next", state, config, round_number=state.get("round", 0)) as record:
        task = state["task"]
        points = state.get("points") or {}
        findings = state.get("findings") or {}
        sources = state.get("sources") or {}
        round_no = state.get("round", 0)
        open_ids = [pid for pid, point in points.items() if point.status == "open"]
        if not open_ids:
            raise WorkerFatalError("plan_next 被调用但没有 open 点：should_stop / 路由逻辑不一致")

        lines = []
        for pid in open_ids:
            point = points[pid]
            line = (
                f"- {point.id}（{point.kind}）证据 "
                f"{evidence_count(pid, findings, sources)}/{required_evidence(point)}：{point.text}"
            )
            if point.accepts:
                line += f"\n    判据：{point.accepts}"
            lines.append(line)
        tried = list(state.get("queries_tried") or [])[-MAX_TRIED_IN_PROMPT:]
        tried_text = "\n".join(f"- {text}" for text in tried) or "（无）"
        human = (
            f"研究问题：{task.question}\n研究目标：{task.objective}\n\n"
            "【仍然开放的点】\n"
            + "\n".join(lines)
            + f"\n\n【已经试过的查询】\n{tried_text}\n\n【不属于本任务的主题】\n"
            + ("、".join(state.get("out_of_scope") or []) or "（无）")
        )
        schema = make_next_schema(open_ids)
        model = llm.with_structured_output(schema, method="function_calling", include_raw=True)
        messages = [SystemMessage(content=PLAN_NEXT_SYSTEM_PROMPT), HumanMessage(content=human)]
        parsed = None
        for attempt in (1, 2):
            try:
                with model_attempt(record, attempt) as attempt_record:
                    response = await model.ainvoke(messages)
                    parsed = record_model_response(attempt_record, response)
            except Exception:
                continue
            if parsed is not None:
                break
            error = attempt_record["error"]["message"]
            messages = messages + [
                HumanMessage(
                    content=(
                        f"上一次输出未通过校验：{error}\n点 id 只能从 {open_ids} 里选。请重新输出。"
                    )
                )
            ]
        if parsed is None:
            raise WorkerFatalError(
                f"plan_next {task.id} 连续两次未取得有效结构化输出，round={round_no}"
            )

        tried_norm = {"".join(text.split()).lower() for text in (state.get("queries_tried") or [])}
        queries = []
        duplicate_count = 0
        for query in parsed.queries:
            key = "".join(query.text.split()).lower()
            if not key or key in tried_norm:
                duplicate_count += 1
                continue
            queries.append(
                Query(text=query.text.strip(), targets=list(dict.fromkeys(query.targets)))
            )
            tried_norm.add(key)
            if len(queries) >= MAX_QUERIES_PER_ROUND:
                break
        new_points, notes = add_discovered(points, parsed.new_points, task.id, round_no)
        covered = {pid for query in queries for pid in query.targets}
        uncovered = [pid for pid in open_ids if pid not in covered]
        zero_uncovered = [pid for pid in uncovered if evidence_count(pid, findings, sources) == 0]
        if notes:
            record["point_decisions"] = notes
        if uncovered:
            record.setdefault("warnings", []).append(f"未被下一轮查询覆盖的开放点：{uncovered}")
        record["output_summary"] = {
            "open_point_count": len(open_ids),
            "candidate_query_count": len(parsed.queries),
            "query_count": len(queries),
            "discarded_query_count": len(parsed.queries) - len(queries),
            "duplicate_or_empty_query_count": duplicate_count,
            "new_point_count": len(set(new_points) - set(points)),
            "uncovered_point_ids": uncovered,
            "zero_evidence_uncovered_point_ids": zero_uncovered,
            "stop_now": not queries,
            "stop_reason": "no_queries" if not queries else "",
            "give_up_reason": parsed.give_up_reason if not parsed.queries else None,
            "no_query_reason": (
                "model_no_queries" if not parsed.queries else "all_queries_filtered"
            )
            if not queries
            else None,
        }
        return {
            "pending_queries": queries,
            "points": new_points,
            "stop_now": not queries,
            "stop_reason": "no_queries" if not queries else "",
        }


async def research_results(state: ResearchWorkerState, config: RunnableConfig):
    with node_log(
        logger, "research_results", state, config, round_number=state.get("round", 0)
    ) as record:
        task = state["task"]
        findings = state.get("findings") or {}
        sources = state.get("sources") or {}
        points = finalize_points(state.get("points") or {}, findings, sources)
        order = {pid: index for index, pid in enumerate(points)}
        ordered = sorted(
            findings.values(),
            key=lambda finding: (
                min((order.get(pid, 999) for pid in finding.point_ids), default=999),
                finding.id,
            ),
        )
        cited = {finding.source_id for finding in ordered}
        result = ResearchResult(
            task_id=task.id,
            findings=ordered,
            points=list(points.values()),
            sources=[
                item.model_copy(update={"summary": ""})
                for item in sources.values()
                if item.id in cited
            ],
        )
        counts = {
            status: sum(point.status == status for point in points.values())
            for status in ("done", "thin", "missing")
        }
        record["output_summary"] = {
            "point_status_counts": counts,
            "contested_point_count": sum(point.contested for point in points.values()),
            "finding_count": len(ordered),
            "source_count": len(result.sources),
            "stop_reason": state.get("stop_reason"),
        }
        return {"result": result, "points": points}
