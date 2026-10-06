from langchain.messages import HumanMessage, ToolMessage, AnyMessage, SystemMessage
from langgraph.types import Send
from typing import Literal


from research_agent.llm import llm
from research_agent.graph.state import ResearchWorkerState
from research_agent.schemas.research import ResearchResult
from research_agent.graph.tools import model_with_tools, tools_by_name









async def research_llm(state: ResearchWorkerState):
    task = state["task"]
    response =await model_with_tools.ainvoke(
        [
            SystemMessage(
                content="""
                    你是 Research Agent 的 Researcher 节点。
                    你的任务是针对给定研究任务收集资料。
                    每一轮最多调用3次工具，总共最多调用6次。
                    在搜集完资料后，说明完成情况，无需生成完整报告
                    """
            )
        ]+
        [
            HumanMessage(content=f"任务id：{task.id} 研究问题：{task.question} 研究目标：{task.objective}")
        ]+state["messages"]
    )
    #print(f"----------research{task.id}-----------")
    #print(response)

    return {
        "messages": [response],
    }



async def tool_node(state: ResearchWorkerState):

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]

    try:
        observation =await tool.ainvoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    except Exception as e:
        result.append(
            ToolMessage(
                content=f"工具调用失败：{type(e).__name__}: {e}",
                tool_call_id=tool_call["id"],
            )
        )
    return {
        "messages": result,
        "search_calls": state.get("search_calls", 0) + len(result),
    }
MAX_SEARCH_CALLS = 6
def should_continue(state: ResearchWorkerState) -> Literal["tool_node", "research_results"]:
    messages = state["messages"]
    last_message = messages[-1]

    search_calls = state.get("search_calls", 0)
    if search_calls >= MAX_SEARCH_CALLS:
        return "research_results"

    if last_message.tool_calls:
        return "tool_node"

    return "research_results"

async def research_results(state: ResearchWorkerState):
    research_model = llm.with_structured_output(
        ResearchResult,
        method="function_calling"
    )
    result: ResearchResult = await research_model.ainvoke(
        [
            SystemMessage(content=f"""
                    你是 Research Agent 的 results 节点。
                    你的职责：根据上下文生成结构化的研究结果
                    """)
        ]+
        [
            HumanMessage(content="请根据以上信息生成结果")
        ]
        + state["messages"]
    )
    return {"result": result}