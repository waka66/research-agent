from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated
import operator

from research_agent.schemas.research import ResearchPlan, ResearchResult, ResearchReport, ResearchTask, ResearchDecision






class ResearchState(TypedDict):
    question: str
    status: str
    research_plan: ResearchPlan | None
    research_results: Annotated[
        list[ResearchResult],
        operator.add
    ]
    report: ResearchReport | None
    final_answer: str | None


class ResearchWorkerState(TypedDict):
    task: ResearchTask
    current_search_query: str | None
    result: ResearchResult
    search_result: list[str]
    filtered_search_results: Annotated[list[str], operator.add]
    search_calls: int
    error_calls: int
    research_history: Annotated[list[ResearchDecision], operator.add]