from langchain.messages import HumanMessage, ToolMessage, AnyMessage, SystemMessage
from langgraph.types import Send
import asyncio
import hashlib
import logging
import json
import time
from datetime import UTC, datetime

from langchain_core.runnables import RunnableConfig

from research_agent.graph.error import WorkerFatalError
from research_agent.llm import llm
from research_agent.graph.state import ResearchWorkerState
from research_agent.schemas.research import (ResearchResult, ResearchWorkerPlan,Query, Source, make_extract_schema, Finding, make_next_schema)
from research_agent.graph.tools import web_search_tool
from research_agent.validators.findings import validate_extract
from research_agent.validators.sources import norm_url, registrable_domain
from research_agent.graph.worker_prompt import (PLAN_SYSTEM_PROMPT, EXTRACT_SYSTEM_PROMPT, PLAN_NEXT_SYSTEM_PROMPT)
from research_agent.validators.points import (seed_points, close_finished_points,evidence_count,required_evidence,
                                         should_stop,add_discovered, finalize_points, new_domain_evidence)


logger = logging.getLogger("research_agent/graph/worker_nodes.py")


async def research_plan(state: ResearchWorkerState, config: RunnableConfig):
    """
    子计划节点

    接收父graph传过来的`任务`

    产出：
    points 点
    pending_queries 待查询的1-2个问题
    out_of_scope 超出范围的东西，预付搜索过多
    round 已完成的搜索轮数
    """

    record = {
        "run_id": config["configurable"]["thread_id"],
        "node_name": "research_plan",
        "started_at": datetime.now(UTC).isoformat(),
        "duration_ms": None,
        "status": "failed",
        "error": None,
    }
    start = time.perf_counter()

    task = state["task"]
    record["task_id"] = task.id

    try:
        # 幂等：重试 / 恢复时不重新播种，避免点被换掉
        if state.get("points"):
            record["status"] = "skipped"
            record["output_summary"] = {
                "reason": "existing_points"
            }
            return {}

        plan_model = llm.with_structured_output(
            ResearchWorkerPlan,
            method="function_calling",
            include_raw=True,
        )
        response = await plan_model.ainvoke(
            [
                SystemMessage(content=PLAN_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"任务 id：{task.id}\n"
                        f"研究问题：{task.question}\n"
                        f"研究目标：{task.objective}"
                    )
                ),
            ]
        )

        record["usage"] = response["raw"].usage_metadata

        parsed = response.get("parsed")

        if parsed is None:
            parsing_error = response.get("parsing_error")
            raise WorkerFatalError(
                f"research_plan 解析失败，task_id={task.id}：{parsing_error}"
            ) from parsing_error

        #检验与格式化LLM的输出
        points, queries, warnings = seed_points(parsed, task)   #分别为字典，列表，列表
        if warnings:
            logger.warning(
                "[seed_points] task=%s 产出 %d 条警告：points=%d queries=%d",
                task.id, len(warnings), len(points), len(queries),
            )
            for w in warnings:
                logger.warning("[seed_points] task=%s %s", task.id, w)
        else:
            logger.info(
                "research_plan运行成功，这是任务：%s 无警告：产生点：%d个 种子问题%d个",
                task.id, len(points), len(queries),
            )

        record["status"] = "succeeded"
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
    except asyncio.CancelledError:
        # 取消单独记录，并继续向外传播
        record["status"] = "cancelled"
        raise

    except Exception as exc:
        # ④ 记录失败原因，但不吞掉异常
        record["status"] = "failed"
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise

    finally:
        # ⑤ 正常返回、异常退出、取消都会经过这里
        record["duration_ms"] = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        level = (
            logging.ERROR
            if record["status"] == "failed"
            else logging.INFO
        )
        logger.log(level, json.dumps(record, ensure_ascii=False))



MAX_SEARCH_CALLS = 6    #最大搜索次数
MAX_ERROR_CALLS = 2     #最大错误次数
async def search_node(state: ResearchWorkerState, config: RunnableConfig):
    """并发搜索、合并来源，并记录本轮查询的执行状态与产出。"""
    record = {
        "run_id": config["configurable"]["thread_id"],
        "node_name": "search_node",
        "task_id": state["task"].id,
        "round": state.get("round", 0) + 1,
        "started_at": datetime.now(UTC).isoformat(),
        "duration_ms": None,
        "status": "failed",
        "error": None,
        "query_errors": [],
    }
    start = time.perf_counter()

    try:
        search_calls = state.get("search_calls", 0)
        error_calls = state.get("error_calls", 0)
        sources = dict(state.get("sources") or {})
        tried = list(state.get("queries_tried") or [])
        queries = list(state.get("pending_queries") or [])

        # 保留现有预算口径：search_calls 累计成功的工具调用，失败另计。
        # 工具可能命中缓存，因此这些次数不能直接当作远程请求数或费用。
        remaining = MAX_SEARCH_CALLS - search_calls
        runnable = queries[:max(remaining, 0)]
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

        # 单条工具异常允许其他查询完成，取消则仍然向外传播。
        outcomes = await asyncio.gather(
            *(web_search_tool.ainvoke(q.text) for q in runnable),
            return_exceptions=True,
        )

        seen = {s.norm_url for s in sources.values() if s.norm_url}
        new_ids: list[str] = []
        ok_queries: list[str] = []

        for q, outcome in zip(runnable, outcomes):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    raise outcome
                record["query_errors"].append({
                    "query": q.text,
                    "type": type(outcome).__name__,
                    "message": str(outcome),
                })
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
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        record["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
        level = {
            "failed": logging.ERROR,
            "partial_success": logging.WARNING,
        }.get(record["status"], logging.INFO)
        logger.log(level, json.dumps(record, ensure_ascii=False))



MAX_EXTRACT_RESULTS = 12        # 单轮最多让模型读这么多条结果，控制输入成本
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
        f"[{s.id}] {s.domain}\n{s.url}\n标题：{s.title}\n摘要：{s.summary}"
        for s in results
    )
async def extract_node(state: ResearchWorkerState):
    """
    提取节点

    获得搜索结果
    判断是否还有开放的点和是否有新增加的搜索结果，两者都无就退出
    调用LLM，获得findings列表
    对findings列表进行校验，进行验证和去重等处理

    输出累计的findings列表和当前新增的findings列表
    """
    task = state["task"]
    points = state.get("points") or {}
    new_results = state.get("new_results") or []
    findings = dict(state.get("findings") or {})

    open_ids = [pid for pid, p in points.items() if p.status == "open"]
    if not open_ids or not new_results:
        print(f"----------extract {task.id} 跳过：open 点={len(open_ids)} 新结果={len(new_results)}----------")
        return {"round_findings": []}

    # 单轮成本闸门
    batch = new_results[:MAX_EXTRACT_RESULTS]
    if len(new_results) > MAX_EXTRACT_RESULTS:
        print(f"  ! 本轮新结果 {len(new_results)} 条，超过 {MAX_EXTRACT_RESULTS}，"
              f"后 {len(new_results) - MAX_EXTRACT_RESULTS} 条不会被抽取（也不会有第二次机会）")
    by_id = {s.id: s for s in batch}

    schema = make_extract_schema(open_ids, [s.id for s in batch])   #拼装好的pydantic对象

    system = EXTRACT_SYSTEM_PROMPT.format(
        out_of_scope="、".join(state.get("out_of_scope") or []) or "（无）"
    )
    human = (
        f"任务：{task.question}\n"
        f"目标：{task.objective}\n\n"
        f"【点清单】（只含仍然开放的点）\n{_render_points(points, open_ids)}\n\n"
        f"【本轮搜索结果】\n{_render_results(batch)}"
    )

    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    model = llm.with_structured_output(schema, method="function_calling", include_raw=True)

    parsed = None
    for attempt in (1, 2):   #循环2次，成功结束，失败重试
        try:
            LLM_TIMEOUT = 90
            t0 = time.perf_counter()
            response = await model.ainvoke(messages, timeout=LLM_TIMEOUT)
            print(f"  [extract] 耗时 {time.perf_counter() - t0:.1f}s")
        except asyncio.TimeoutError:
            raise WorkerFatalError(f"extract 调用超时 {LLM_TIMEOUT}s")
        except Exception as e:                        # 工具调用本身失败
            print(f"  ! extract 第 {attempt} 次调用异常：{type(e).__name__}: {e}")
            continue

        parsed = response.get("parsed") if isinstance(response, dict) else response
        if parsed is not None:
            break

        err = response.get("parsing_error") if isinstance(response, dict) else "unknown"
        print(f"  ! extract 第 {attempt} 次结构化输出不合法：{err}")
        messages = messages + [HumanMessage(content=(
            f"上一次输出未通过校验：{err}\n"
            f"点 id 只能从 {open_ids} 里选，结果 id 只能从 {list(by_id)} 里选。请重新输出。"
        ))]

    if parsed is None:
        raise WorkerFatalError(
            f"extract {task.id} 连续两次结构化输出失败，已终止（round={state.get('round')}，"
            f"open={open_ids}，results={[s.id for s in batch]}）"
        )

    accepted, dropped = validate_extract(parsed.findings, by_id, set(open_ids))   #校验器，返回经过验证的finding和出现的错误

    new_findings: list[Finding] = []
    for f in accepted:
        if f.id in findings:          # 跨轮重复：同一来源里同一句话以前抽过
            continue
        findings[f.id] = f
        new_findings.append(f)

    # ---- 日志：三个必须看的数字 ----
    print(f"----------extract {task.id}----------")
    print(f"  候选 {len(parsed.findings)} → 校验通过 {len(accepted)} → 跨轮新增 {len(new_findings)}"
          f"（累计 {len(findings)}）")
    for d in dropped:
        print(f"  x {d}")

    per_point: dict[str, int] = {}
    for f in new_findings:
        if f.stance == "supports":
            for pid in f.point_ids:
                per_point[pid] = per_point.get(pid, 0) + 1
    for pid in open_ids:
        print(f"    {pid} 本轮 +{per_point.get(pid, 0)}  supports")
    for f in new_findings:
        print(f"  + {f.id} [{f.stance}] {f.point_ids} {f.claim[:50]}")

    return {"findings": findings, "round_findings": new_findings}



MAX_NO_PROGRESS = 2      # 连续几轮没有新证据就停
MAX_ROUNDS = 6           # 轮数硬上限
async def update_node(state: ResearchWorkerState):
    """
    更新节点

    记录这是第几轮
    拿到本轮新增的finding
    判断`点`是否关闭完
    打印信息，方便观测

    返回连续几轮没有新证据
    是否继续
    停止原因
    """
    # 复制一份再改：LangGraph 会共享嵌套对象，原地改动会污染 checkpointer 里的历史状态
    points = {pid: p.model_copy() for pid, p in (state.get("points") or {}).items()}
    findings = state.get("findings") or {}
    sources = state.get("sources") or {}
    round_findings = state.get("round_findings") or []
    search_calls = state.get("search_calls", 0)

    round_no = state.get("round", 0) + 1
    no_progress = 0 if round_findings else state.get("no_progress", 0) + 1
    errors_exhausted = state.get("error_calls", 0) >= MAX_ERROR_CALLS

    closed = close_finished_points(points, findings, sources)     #这个函数返回这轮关闭的点
    stop, reason = should_stop(points, round_no, no_progress, search_calls, errors_exhausted)    #返回是否结束，关闭原因

    round_ids = {f.id for f in round_findings}
    prev = {fid: f for fid, f in findings.items() if fid not in round_ids}   #不要重复的
    gained = new_domain_evidence(round_findings, prev, sources)              #计算本轮新增的点

    no_progress = 0 if gained > 0 else state.get("no_progress", 0) + 1

    # ---------- 日志：这是这一步唯一的产出价值 ----------
    print(f"----------update {state['task'].id} round={round_no}----------")
    print(f"  本轮新增 (点,域名) 组合 {gained} 个，无进展连击 {no_progress}/{MAX_NO_PROGRESS}")
    for p in points.values():
        n = evidence_count(p.id, findings, sources)
        req = required_evidence(p)
        flag = " ←本轮关闭" if p.id in closed else ""
        print(f"  {p.id:>10} {p.status:<7} {n}/{req} domains"
              f"{'  contested' if p.contested else ''}{flag}")
    print(f"  裁决：{'停止' if stop else '继续'} ({reason or 'continue'})"
          f"  round={round_no}/{MAX_ROUNDS}")

    return {
        "points": points,
        "round": round_no,
        "no_progress": no_progress,
        "stop_now": stop,
        "stop_reason": reason,
    }




MAX_QUERIES_PER_ROUND = 2
MAX_TRIED_IN_PROMPT = 12
async def plan_next(state: ResearchWorkerState):
    """
    提出下一个问题的节点

    到了这个节点，LLM根据还未关闭的点提出1-2个问题，开始下一轮循环
    如果LLM认为没必要再搜索就结束
    这个节点还可以提出新的`点`
    """
    task = state["task"]
    points = state.get("points") or {}
    findings = state.get("findings") or {}
    sources = state.get("sources") or {}
    round_no = state.get("round", 0)

    #检查bug正常情况不触发
    open_ids = [pid for pid, p in points.items() if p.status == "open"]
    assert open_ids, "plan_next 被调用但没有 open 点：should_stop / 路由逻辑不一致"
    if not open_ids:
        return {
            "pending_queries": [],
            "no_queries": True,
            "stop_now": True,
            "stop_reason": "all_done",
        }

    # ---------- 渲染材料 ----------
    lines = []
    for pid in open_ids:
        p = points[pid]
        n = evidence_count(pid, findings, sources)
        req = required_evidence(p)
        line = f"- {p.id}（{p.kind}）证据 {n}/{req}：{p.text}"
        if p.accepts:
            line += f"\n    判据：{p.accepts}"
        lines.append(line)

    tried = list(state.get("queries_tried") or [])[-MAX_TRIED_IN_PROMPT:]
    tried_text = "\n".join(f"- {t}" for t in tried) or "（无）"

    human = (
        f"研究问题：{task.question}\n"
        f"研究目标：{task.objective}\n\n"
        f"【仍然开放的点】\n" + "\n".join(lines) + "\n\n"
        f"【已经试过的查询】\n{tried_text}\n\n"
        f"【不属于本任务的主题】\n"
        + ("、".join(state.get("out_of_scope") or []) or "（无）")
    )

    # ---------- 调用（失败重试 1 次，把错误塞回去） ----------
    schema = make_next_schema(open_ids)
    model = llm.with_structured_output(schema, method="function_calling", include_raw=True)
    messages = [SystemMessage(content=PLAN_NEXT_SYSTEM_PROMPT), HumanMessage(content=human)]

    parsed = None
    for attempt in (1, 2):
        try:
            response = await model.ainvoke(messages)
        except Exception as e:
            print(f"  ! plan_next 第 {attempt} 次调用异常：{type(e).__name__}: {e}")
            continue
        parsed = response.get("parsed") if isinstance(response, dict) else response
        if parsed is not None:
            break
        err = response.get("parsing_error") if isinstance(response, dict) else "unknown"
        print(f"  ! plan_next 第 {attempt} 次输出不合法：{err}")
        messages = messages + [HumanMessage(content=(
            f"上一次输出未通过校验：{err}\n"
            f"点 id 只能从 {open_ids} 里选。请重新输出。"
        ))]

    if parsed is None:
        print(f"----------plan_next {task.id} 放弃本轮（无有效输出）----------")
        raise WorkerFatalError(
            f"extract {task.id} 结构化输出失败，已终止（round={state.get('round')}，"
        )
        #return {"pending_queries": [], "no_queries": False}   # 不因一次解析失败就终止研究

    # ---------- 查询：去重、限流 ----------
    tried_norm = {"".join(t.split()).lower() for t in (state.get("queries_tried") or [])}
    queries: list[Query] = []
    for q in parsed.queries:
        key = "".join(q.text.split()).lower()
        if not key or key in tried_norm:
            print(f"  x 查询重复/为空，丢弃：{q.text}")
            continue
        queries.append(Query(text=q.text.strip(), targets=list(dict.fromkeys(q.targets))))
        tried_norm.add(key)
        if len(queries) >= MAX_QUERIES_PER_ROUND:
            print(f"  x 超出每轮 {MAX_QUERIES_PER_ROUND} 条上限，丢弃：{q.text}")
            break

    # ---------- 新点：代码批准 ----------
    new_points, notes = add_discovered(points, parsed.new_points, task.id, round_no)

    # ---------- 覆盖检查（这是"点永远不收敛"的直接探针） ----------
    covered = {pid for q in queries for pid in q.targets}
    uncovered = [pid for pid in open_ids if pid not in covered]
    zero_uncovered = [pid for pid in uncovered if evidence_count(pid, findings, sources) == 0]

    # ---------- 日志 ----------
    print(f"----------plan_next {task.id} round={round_no}----------")
    for q in queries:
        print(f"  Q {q.text}  → {q.targets}")
    for note in notes:
        print(f"  * {note}")
    for pid in points:
        if pid not in (state.get("points") or {}) and pid in new_points:
            print(f"  + 新点 {pid} [{new_points[pid].kind}] {new_points[pid].text[:40]}")
    if uncovered:
        print(f"  ! 未被任何查询覆盖的开放点：{uncovered}")
    if zero_uncovered:
        print(f"  !! 这些点是 0 证据、本轮又没人搜：{zero_uncovered}（它们一定会变成 missing）")
    if not queries:
        print(f"  ==> 模型主动放弃：{parsed.give_up_reason or '（未给理由）'}")

    return {
        "pending_queries": queries,
        "points": new_points,
        "stop_now": not queries,
        "stop_reason": "no_queries" if not queries else "",
    }


async def research_results(state: ResearchWorkerState):
    """
    总结节点

    纯代码，把finding，`点`，去掉原文的结果（证据）返回给父图
    """
    task = state["task"]
    findings = state.get("findings") or {}
    sources = state.get("sources") or {}

    points = finalize_points(state.get("points") or {}, findings, sources)    #把目前还开放的点关闭附上各自的状态，然后收集起来

    # 按点的顺序排，保证同一份 state 每次输出完全一致
    order = {pid: i for i, pid in enumerate(points)}
    ordered = sorted(
        findings.values(),
        key=lambda f: (min((order.get(p, 999) for p in f.point_ids), default=999), f.id),
    )

    cited = {f.source_id for f in ordered}
    result = ResearchResult(
        task_id=task.id,
        findings=ordered,
        points=list(points.values()),
        sources=[
            s.model_copy(update={"summary": ""})  # 不需要原文
            for s in sources.values() if s.id in cited
        ],
    )

    counts = {s: 0 for s in ("done", "thin", "missing")}
    for p in points.values():
        counts[p.status] = counts.get(p.status, 0) + 1

    print(f"----------results {task.id}----------")
    print(f"  轮数={state.get('round', 0)}  停因={state.get('stop_reason') or '-'}")
    print(f"  done={counts['done']} thin={counts['thin']} missing={counts['missing']}"
          f"  findings={len(ordered)} 条，来源 {len({f.source_id for f in ordered})} 个")
    for p in points.values():
        if p.status != "done":
            print(f"  [{p.status}] {p.id} {p.text[:60]}")
        if p.contested:
            print(f"  [contested] {p.id} {p.text[:60]}")

    # 把收尾后的 points 也写回 state：checkpointer / 状态接口能看到终态
    return {"result": result, "points": points}