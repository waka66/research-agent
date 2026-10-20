from langchain.messages import HumanMessage, ToolMessage, AnyMessage, SystemMessage
from langgraph.types import Send
import asyncio

from research_agent.graph.error import WorkerFatalError
from research_agent.graph.main_prompt import SYNTH_SYSTEM_PROMPT
from research_agent.graph.render import render_report
from research_agent.llm import llm, llm_synth
from research_agent.graph.state import ResearchState
from research_agent.schemas.research import ResearchPlan, ResearchReport, ResearcherInput
from research_agent.graph.worker_workflow import worker_graph
from research_agent.validators.report import validate_report



async def Planner(state: ResearchState):
    planner_model = llm.with_structured_output(
        ResearchPlan,
        method="function_calling",
        include_raw=True,    # 调试用的，保留原始输出
    )

    plan: ResearchPlan =await planner_model.ainvoke(
        [
            SystemMessage(content="""
                你是 Research Agent 的 Planner 节点。

                你的职责：
                1. 分析用户的研究问题
                2. 将问题拆分为多个可独立执行的研究任务--为了控制成本，如果任务简单请尽量少拆，或者直接不拆直接返回一个任务
                3. 每个任务必须能够通过搜索和资料分析独立完成
                4. 任务之间尽量不存在依赖关系，以便系统并行执行
                5. 不负责最终总结
                6. 不创建“综合以上任务”“形成最终报告”之类的任务
                7. 不进行网页搜索
                8. 不向用户询问确认
                9. task的id以T1,T2,T3...的形式编号

                只输出研究任务计划。
                """)
        ] + state["question"]
    )
    print("----------plan-----------")
    print(plan)

    research_plan = plan['parsed']


    # 把 Pydantic 对象序列化成 JSON 字符串，再放进 AIMessage
    return {
        "research_plan": research_plan,
        "status": "running"
    }



def dispatch_research_tasks(state: ResearchState):
    return [
        Send(
            "research_node",
            {"task": task}
        )
        for task in state["research_plan"].tasks
    ]


async def research_node(state: ResearcherInput):

    worker_result = await worker_graph.ainvoke({
        "task": state["task"],
        "points": {}, "findings": {}, "sources": {},
        "pending_queries": [], "queries_tried": [], "out_of_scope": [],
        "round": 0, "no_progress": 0, "stop_now": False, "stop_reason": "",
        "new_results": [], "round_findings": [],
        "search_calls": 0, "error_calls": 0,
    })
    print("----------result-----------")
    print(worker_result)


    return {
        "research_results": [
            worker_result["result"]
        ]
    }



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
async def Synthesizer(state: ResearchState):
    results = state["research_results"]

    # 证据 id -> Finding，给校验器用（必须是 dict，不是 list）
    findings_by_id = {}
    for r in results:
        for f in r.findings:
            findings_by_id[f.id] = f

    evidence_parts, thin, missing, contested = [], [], [], []
    used = skipped = 0

    for r in results:
        block, u, s = _evidence_block(r)        # ←←← 调用点在这里
        used += u
        skipped += s
        if block:
            evidence_parts.append(block)
        for p in r.points:
            if p.status == "thin":
                thin.append(p.text)
            elif p.status == "missing":
                missing.append(p.text)
            if p.contested:
                contested.append(p.text)

    def _lines(items):
        return "\n".join(f"- {t}" for t in items) or "（无）"

    prompt = (
        f"研究计划：{state['research_plan']}\n\n"
        f"【证据（按点分组，+ 支持 / - 反驳）】\n{chr(10).join(evidence_parts)}\n\n"
        f"【仅单一来源的点】\n{_lines(thin)}\n\n"
        f"【没查到的点（必须写成缺口）】\n{_lines(missing)}\n\n"
        f"【存在冲突的点（两边都要写）】\n{_lines(contested)}"
    )

    print(f"  [synthesizer] 证据 {used} 条（裁掉 {skipped} 条）/ 提示词 {len(prompt)} 字")

    model = llm_synth.with_structured_output(
        ResearchReport, method="function_calling", include_raw=True
    )
    messages = [SystemMessage(content=SYNTH_SYSTEM_PROMPT), HumanMessage(content=prompt)]

    report = None
    for attempt in (1, 2):
        try:
            response = await asyncio.wait_for(model.ainvoke(messages), timeout=180)
        except asyncio.TimeoutError:
            raise WorkerFatalError(f"Synthesizer 超时（提示词 {len(prompt)} 字）")
        except Exception as e:
            print(f"  ! Synthesizer 第 {attempt} 次异常：{type(e).__name__}: {e}")
            continue

        parsed = response.get("parsed") if isinstance(response, dict) else response
        if parsed is None:
            print(f"  ! Synthesizer 第 {attempt} 次结构化输出失败")
            continue

        fatal, warn = validate_report(parsed, findings_by_id)
        for w in warn:
            print(f"  ~ {w}")
        if not fatal:
            report = parsed
            break
        print(f"  ! 报告校验 fatal（第 {attempt} 次）：")
        for h in fatal:
            print(f"     - {h}")
        messages = messages + [HumanMessage(content=(
                "上一次报告未通过校验，原因：\n" + "\n".join(f"- {h}" for h in fatal)
                + "\n请只使用我给出的证据 id，没证据的方面写进缺口。重新输出。"
        ))]

    if report is None:
        print("  ==> Synthesizer 两轮不合格，降级为纯代码渲染（无导语、无分组）")
        return {"report": None, "status": "partial"}

    print(f"----------Synthesizer {len(results)} 个结果 -> "
          f"{len(report.key_conclusions)} 个关键结论 / {len(report.sections)} 节----------")
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
async def generator(state: ResearchState):
    question = _question_text(state)
    print(f"  [generator] question={question[:60]!r}")
    markdown = render_report(
        state.get("report"),
        state.get("research_results") or [],
        question,
    )
    print(f"----------generator----------\n{markdown[:400]}")
    return {"final_answer": markdown, "status": "completed"}
