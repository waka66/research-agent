from pydantic import BaseModel, Field
from uuid import UUID
from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict






#接口相关
class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)


class ResearchRunResponse(BaseModel):
    run_id: UUID
    status: str

class ResearchStatusResponse(BaseModel):
    run_id: UUID
    question: str
    status: str
    plan: str
    results: str
    final_answer: str
    error: str



#graph相关
class ResearchTask(BaseModel):
    id: str
    question: str
    objective: str
class ResearchPlan(BaseModel):
    goal: str
    tasks: list[ResearchTask]

class ResearcherInput(TypedDict):
    task: ResearchTask

class ResearchResult(BaseModel):
    task_id: str
    summary: str
    sources: list[str]



class ResearchFinding(BaseModel):
    claim: str
    evidence: str
    source_url: str
class ResearchResult(BaseModel):
    task_id: str
    summary: str
    findings: list[ResearchFinding]



class ReportSection(BaseModel):
    title: str
    content: str
class ReportClaim(BaseModel):
    claim: str
    citation_ids: list[str]
class ResearchReport(BaseModel):
    title: str
    summary: str
    sections: list[ReportSection]
    claims: list[ReportClaim]



class ResearchDecision(BaseModel):
    key_facts: list[str]
    covered_topics: list[str]
    missing_topics: list[str]

    sufficient: bool = Field(default=False)
    next_query: str | None
    irrelevant_ids: list[str]