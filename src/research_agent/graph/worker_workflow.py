from langgraph.graph import StateGraph, START, END

from research_agent.graph.state import ResearchWorkerState
from research_agent.graph.worker_nodes import research_plan, search_node, research_results, should_continue, research_judge


#子图
researcher_builder = StateGraph(ResearchWorkerState)

researcher_builder.add_node("research_plan", research_plan)
researcher_builder.add_node("research_judge", research_judge)
researcher_builder.add_node("search_node", search_node)
researcher_builder.add_node("research_results", research_results)

researcher_builder.add_edge(START, "research_plan")
researcher_builder.add_edge("research_plan", "search_node")
researcher_builder.add_edge("search_node", "research_judge")
researcher_builder.add_conditional_edges("research_judge", should_continue)
researcher_builder.add_edge("research_results", END)
worker_graph = researcher_builder.compile()
