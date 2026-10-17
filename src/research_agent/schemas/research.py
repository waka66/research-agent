from pydantic import BaseModel, Field
from uuid import UUID
from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Literal






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


class ResearchFindingDraft(BaseModel):
    checklist_item_id: str
    claim: str
    quote: str
    source_id: str
    stance: Literal["supports", "contradicts", "context"]  #正，反
class ResearchResultDraft(BaseModel):
    findings: list[ResearchFindingDraft]

class ResearchFinding(BaseModel):
    id: str
    checklist_item_id: str    # 回答 research_plan 里的哪一项
    claim: str   #一句话的原子断言,不含代词,主体、时间、数值、范围都写全
    quote: str   #原文的逐字摘录,1 到 2 句,自身完整(含主语、数字、时间限定)
    source_id: str
    stance: Literal["supports", "contradicts", "context"]
class ResearchResult(BaseModel):
    task_id: str
    findings: list[ResearchFinding]
    item_status: dict[str, Literal["answered", "partial", "missing"]]



class Point(BaseModel):
    text: str
    finding_ids: list[str] = []
    kind: Literal["fact", "inference", "transition"]
    importance: Literal["key", "supporting"] = "supporting"
class Section(BaseModel):
    heading: str
    points: list[Point]
class ResearchReport(BaseModel):
    title: str
    key_conclusions: list[Point]
    sections: list[Section]


class ResearchDecision(BaseModel):
    key_facts: list[str]
    missing_topics: list[str]

    sufficient: bool = Field(default=False)
    next_query: str | None
    irrelevant_ids: list[str]

class ResearchWorkerPlan(BaseModel):
    TaskList: list[str]
    query: str


