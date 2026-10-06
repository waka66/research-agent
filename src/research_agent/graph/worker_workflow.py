from langgraph.graph import StateGraph, START, END

from research_agent.graph.state import ResearchWorkerState
from research_agent.graph.worker_nodes import research_llm, tool_node, research_results, should_continue


#子图
researcher_builder = StateGraph(ResearchWorkerState)

researcher_builder.add_node("research_llm", research_llm)
researcher_builder.add_node("tool_node", tool_node)
researcher_builder.add_node("research_results", research_results)

researcher_builder.add_edge(START, "research_llm")
researcher_builder.add_conditional_edges("research_llm", should_continue)
researcher_builder.add_edge("tool_node", "research_llm")
researcher_builder.add_edge("research_results", END)
worker_graph = researcher_builder.compile()
