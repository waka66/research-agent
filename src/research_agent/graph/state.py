from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated
import operator

from research_agent.schemas.research import ResearchPlan, ResearchResult, ResearchReport, ResearchTask






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
    messages: Annotated[list[AnyMessage], add_messages]
    result: ResearchResult | None
    search_calls: int