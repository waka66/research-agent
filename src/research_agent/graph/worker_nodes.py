from langchain.messages import HumanMessage, ToolMessage, AnyMessage, SystemMessage
from langgraph.types import Send
from typing import Literal
import uuid



from research_agent.llm import llm, llm2
from research_agent.graph.state import ResearchWorkerState
from research_agent.schemas.research import ResearchResult, ResearchDecision, ResearchWorkerPlan, ResearchResultDraft, ResearchFinding
from research_agent.graph.tools import web_search_tool
from research_agent.validators.findings import validate_findings

model_with_tools = llm.bind_tools([web_search_tool])

async def research_plan(state: ResearchWorkerState):
    task = state["task"]
    plan_model = llm.with_structured_output(
        ResearchWorkerPlan,
        method="function_calling",
        include_raw=True,
    )
    response = await plan_model.ainvoke(
        [
            SystemMessage(
                content="""
                    你是 Research Agent的 Researcher_plan节点。
                    你的任务是针对给定的研究任务给出检查清单(要回答的具体问题/要素)。
                    再根据研究任务进行一次提问，这个问题会进入下一个节点进行搜索，第一次的搜索结果将给下一个节点。
                    下一个节点将根据检查清单（TaskList）不断检索优化完成本任务的目标
                    所以要注意第一次提问的全面性
                    """
            )
        ]+
        [
            HumanMessage(content=f"任务id：{task.id} 研究问题：{task.question} 研究目标：{task.objective}")
        ]
    )
    print(f"----------research{task.id}-----------")
    print(response)

    return {
        "current_search_query": response['parsed'].query,
        "TaskList": response['parsed'].TaskList
    }

async def research_judge(state: ResearchWorkerState):
    last_search_result = state["search_result"][-1]
    history = state.get("research_history") or []
    last_history = history[-1] if history else None

    if last_history:
        progress_text = (
            f"key_facts={last_history.key_facts}, "
            f"missing_topics={last_history.missing_topics}"
        )
    else:
        progress_text = "（这是第一次判断，暂无研究进展）"

    research_model = llm.with_structured_output(
        ResearchDecision,
        method="function_calling",
        include_raw=True,
    )

    Decision: ResearchDecision = await research_model.ainvoke(
        [
            SystemMessage(content=f"""
                    你是 Research Agent 的 judge 节点。
                    你的职责：我会把研究进展和验收清单给你，根据目前的验收清单中任务完成情况判断是否需要继续搜索，
                    对之前的搜索进行总结与归纳形成新的进展。
                    如果证据足够，则返回sufficient = True，否则返回sufficient = False，并提供next_query
                    请合理规划next_query，搜索次数有限，
                    key_facts 只保留能够支持后续研究或最终回答的重要事实。
                    筛选出上一次搜索结果中，与当前研究问题无关或者不能被作为证据的网页的id加入irrelevant_ids。
                    如果不同的搜索结果之间表达内容重复，请只保留一个，另一个加入irrelevant_ids。
                    """)
        ]+
        [
            HumanMessage(content=f"""
                        验收清单{state["TaskList"]},
                        上一次的搜索结果：{last_search_result},
                        是否发生错误：{state["error_result"]},
                        研究进展：{progress_text},
                        只调用工具，不要在 content 里输出任何文字。
                        不要解释，不要前后缀，直接通过工具调用返回结构化参数。"""
                         )
        ]
    )
    print(f"----------judge{state['task'].id}-----------")
    print(Decision)
    filtered_search_results = []
    if Decision['parsed'].irrelevant_ids:
        for item in last_search_result["search_result"]:
            if item["id"] not in Decision['parsed'].irrelevant_ids:
                filtered_search_results.append(item)


    return {
        "filtered_search_results": filtered_search_results,
        "research_history": [Decision['parsed']],
        "current_search_query": Decision['parsed'].next_query
    }


async def search_node(state: ResearchWorkerState):
    history = state.get("research_history") or []
    last = history[-1] if history else None
    question = (
        last.next_query if last and last.next_query
        else state["current_search_query"]
    )

    search_calls = state.get("search_calls", 0)
    error_calls = state.get("error_calls", 0)
    urls = []
    for item in state["filtered_search_results"]:
        for url in item["url"]:
            urls.append(url)

    search_result = []
    response = None

    try:
        response = await web_search_tool.ainvoke(question)
        error_result = None
        for new_search in response["search_result"]:
            if new_search["url"] not in urls:
                search_result.append(new_search)
        response["search_result"] = search_result
        search_calls += 1

    except Exception as e:
        error_result=f"工具调用失败：{type(e).__name__}: {e}"
        error_calls += 1


    return {
        "search_result": [response] if response is not None else [],
        "error_result": error_result,
        "search_calls": search_calls,
        "error_calls": error_calls,
    }


MAX_SEARCH_CALLS = 6
MAX_ERROR_CALLS = 2
def should_continue(state: ResearchWorkerState) -> Literal["search_node", "research_results"]:


    if state["research_history"][-1].sufficient:
        return "research_results"

    if state.get("search_calls", 0) >= MAX_SEARCH_CALLS:
        return "research_results"

    if state.get("error_calls", 0) >= MAX_ERROR_CALLS:
        return "research_results"

    if not (state["research_history"][-1].next_query or state["current_search_query"]):
        return "research_results"

    return "search_node"


async def research_results(state: ResearchWorkerState):
    research_model = llm2.with_structured_output(
        ResearchResultDraft,
        method="function_calling",
    )
    draft: ResearchResultDraft = await research_model.ainvoke(
        [
            SystemMessage(content=f"""
                    你是 Research Agent 的 results 节点。
                    1. 不重复撰写搜索结果中已经表达清楚的自然语言结论。
                    2. 优先保留与研究问题直接相关的事实、原始证据和来源。
                    3. 不把多个来源的不同数字混成一个看似确定的结论。
                    4. 不确定的信息、相互矛盾的信息以及评测条件差异必须显式保留。
                    5. 不为了满足固定数量而生成更多发现，也不为了让结果显得完整而推测缺失数据。
                    如果输入已经包含足够完整的结构化数据，节点可以直接返回这些数据的整理结果，而不是重新生成长篇摘要。
                    quote是原文片段，不要自行添加，
                    source_id一定真实存在，且quote和source_id是一一对应的
                    claim 尽量是 quote 的忠实改写,不加评论、不写对比、不写"与 X 无关"这类跨条目的话
                    """)
        ]+
        [
            HumanMessage(content=f"""任务id：{state['task'].id}，总的研究问题：{state['task'].question}，总的研究目标：{state['task'].objective}
                        搜索结果：{state['filtered_search_results']}""")
        ]
    )

    all_sources = state.get("filtered_search_results") or []

    valid_drafts = validate_findings(draft.findings, all_sources)

    findings = [
        ResearchFinding(id=str(uuid.uuid4()), **f.model_dump())  #model_dump() 是 Pydantic v2 的方法，把对象转成 dict
        for f in valid_drafts                                    #**等价于把 dict 里的每个 key-value 当成关键字参数传进去
    ]

    result = ResearchResult(
        task_id=state['task'].id,
        findings=findings,
        item_status=draft.item_status,
    )

    return {"result": result}