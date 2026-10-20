from langgraph.graph import StateGraph, START, END
from typing import Literal

from research_agent.graph.state import ResearchWorkerState
from research_agent.graph.worker_nodes import research_plan, search_node, research_results, extract_node, update_node, plan_next


def route_after_update(state) -> Literal["plan_next", "research_results"]:
    """终止权在这里。只看代码写进去的 stop_now，不看任何 LLM 字段。"""
    return "research_results" if state.get("stop_now") else "plan_next"


def route_after_plan_next(state) -> Literal["search_node", "research_results"]:
    """plan_next 自己放弃时直接收尾，不再空跑一轮。"""
    return "research_results" if state.get("stop_now") else "search_node"


#子图
builder = StateGraph(ResearchWorkerState)

builder.add_node("research_plan", research_plan)
builder.add_node("search_node", search_node)
builder.add_node("extract_node", extract_node)
builder.add_node("update_node", update_node)
builder.add_node("plan_next", plan_next)
builder.add_node("research_results", research_results)

builder.add_edge(START, "research_plan")
builder.add_edge("research_plan", "search_node")
builder.add_edge("search_node", "extract_node")
builder.add_edge("extract_node", "update_node")
builder.add_conditional_edges("update_node", route_after_update,
                              {"plan_next": "plan_next", "research_results": "research_results"})
builder.add_conditional_edges("plan_next", route_after_plan_next,
                              {"search_node": "search_node", "research_results": "research_results"})
builder.add_edge("research_results", END)

worker_graph = builder.compile()
