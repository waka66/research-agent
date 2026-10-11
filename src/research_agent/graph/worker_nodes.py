from langchain.messages import HumanMessage, ToolMessage, AnyMessage, SystemMessage
from langgraph.types import Send
from typing import Literal



from research_agent.llm import llm
from research_agent.graph.state import ResearchWorkerState
from research_agent.schemas.research import ResearchResult, ResearchDecision
from research_agent.graph.tools import web_search_tool

model_with_tools = llm.bind_tools([web_search_tool])

async def research_plan(state: ResearchWorkerState):
    task = state["task"]
    response =await model_with_tools.ainvoke(
        [
            SystemMessage(
                content="""
                    你是 Research Agent 的 Researcher 节点。
                    你的任务是针对给定研究任务收集资料。
                    先根据研究任务进行第一次整体搜索
                    你只能调用一次工具，所以要注意搜索的全面性
                    """
            )
        ]+
        [
            HumanMessage(content=f"任务id：{task.id} 研究问题：{task.question} 研究目标：{task.objective}")
        ]
    )
    print(f"----------research{task.id}-----------")
    print(response)
    if not response.tool_calls:
        raise RuntimeError("Research Plan 未生成搜索工具调用")
    next_query = response.tool_calls[0]['args']['query']
    print(next_query)


    return {
        "current_search_query": next_query
    }

async def research_judge(state: ResearchWorkerState):
    last_search_result = state["search_result"][-1]
    research_model = llm.with_structured_output(
        ResearchDecision,
        method="function_calling",
        include_raw=True,
    )

    Decision: ResearchDecision = await research_model.ainvoke(
        [
            SystemMessage(content=f"""
                    你是 Research Agent 的 judge 节点。
                    你的职责：我会把整个research_history给你，根据目前的任务完成情况判断是否需要继续搜索，对之前的搜索进行总结与归纳。
                    如果证据足够，则返回sufficient = True，否则返回sufficient = False，并提供next_query
                    请合理规划next_query，搜索次数有限，
                    key_facts 只保留能够支持后续研究或最终回答的重要事实。
                    筛选出上一次搜索结果中，与当前研究问题无关或者不能被作为证据的网页的id加入irrelevant_ids。
                    如果不同的搜索结果之间表达内容重复，请只保留一个，另一个加入irrelevant_ids。
                    """)
        ]+
        [
            HumanMessage(content=f"""
                        总的研究问题：{state['task'].question} 总的研究目标：{state['task'].objective}，
                        上一次的搜索结果：{last_search_result}，研究历史：{state['research_history']}
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

    try:
        search_result = await web_search_tool.ainvoke(question)
        search_calls += 1
    except Exception as e:
        search_result=f"工具调用失败：{type(e).__name__}: {e}"
        error_calls += 1

    return {
        "search_result": [search_result],
        "search_calls": search_calls,
        "error_calls": error_calls,
    }


MAX_SEARCH_CALLS = 8
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
    research_model = llm.with_structured_output(
        ResearchResult,
        method="function_calling",
    )
    result: ResearchResult = await research_model.ainvoke(
        [
            SystemMessage(content=f"""
                    你是 Research Agent 的 results 节点。
                    1. 不重复撰写搜索结果中已经表达清楚的自然语言结论。
                    2. 优先保留与研究问题直接相关的事实、原始证据和来源。
                    3. 不把多个来源的不同数字混成一个看似确定的结论。
                    4. 不确定的信息、相互矛盾的信息以及评测条件差异必须显式保留。
                    5. 不为了满足固定数量而生成更多发现，也不为了让结果显得完整而推测缺失数据。
                    如果输入已经包含足够完整的结构化数据，节点可以直接返回这些数据的整理结果，而不是重新生成长篇摘要。
                    """)
        ]+
        [
            HumanMessage(content=f"""总的研究问题：{state['task'].question} 总的研究目标：{state['task'].objective}
                        搜索结果：{state['filtered_search_results']}""")
        ]
    )
    return {"result": result}