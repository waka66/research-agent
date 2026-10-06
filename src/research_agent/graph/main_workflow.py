from langgraph.graph import StateGraph, START, END

from research_agent.graph.state import ResearchState
from research_agent.graph.main_nodes import Planner, research_node, Synthesizer, dispatch_research_tasks, generator



#主图
agent_builder = StateGraph(ResearchState)

agent_builder.add_node("Planner", Planner)
agent_builder.add_node("research_node", research_node)
agent_builder.add_node("Synthesizer", Synthesizer)
agent_builder.add_node("generator", generator)


agent_builder.add_edge(START, "Planner")
agent_builder.add_conditional_edges("Planner", dispatch_research_tasks, ["research_node"])
agent_builder.add_edge("research_node", "Synthesizer")
agent_builder.add_edge("Synthesizer", "generator")
agent_builder.add_edge("generator", END)
