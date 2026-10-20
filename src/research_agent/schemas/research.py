from pydantic import BaseModel, Field, create_model
from uuid import UUID
from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Literal
import hashlib






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



#research_plan节点相关的
PointKind = Literal["official_fact", "multi_source_fact", "comparison"]  #唯一来源，多个来源，
class PlannedPoint(BaseModel):
    """LLM 眼里的"点"的唯一形态：只有文本，没有 id。
    注意：新增点（plan_next）也复用它，否则两种点的高低门槛会不一致。"""

    text: str = Field(
        min_length=4,
        description="一个能用 1~3 句话回答的判据式表述，必须自带主语、不含代词",
    )
    kind: PointKind = Field(
        default="multi_source_fact",
        description="决定这个点需要几个独立来源才算完成",
    )
    accepts: str = Field(
        default="",
        description="一句话：什么样的证据算回答了它。用来和相邻点区分",
    )
    priority: Literal["key", "supporting"] = "supporting"   #回答研究问题的主体内容/对核心内容的补充、例证、背景或细节
class SeedQuery(BaseModel):
    text: str = Field(min_length=2)
    target_indices: list[int] = Field(
        default_factory=list,
        description="想回答的点的 0-based 下标；留空表示目标是所有点",
    )
class ResearchWorkerPlan(BaseModel):
    points: list[PlannedPoint]
    queries: list[SeedQuery]
    out_of_scope: list[str] = Field(default_factory=list)

#与state有关
class PointState(BaseModel):
    id: str                                                    # 代码生成 f"{task_id}-p{n}"
    text: str
    kind: PointKind = "multi_source_fact"
    accepts: str = ""
    priority: Literal["key", "supporting"] = "supporting"
    status: Literal["open", "done", "thin", "missing"] = "open"
    misses: int = 0
    origin: Literal["plan", "discovered"] = "plan"
    round_added: int = 0
    contested: bool = False  #拿到过 contradicts 证据
    single_domain: bool = False
class Query(BaseModel):
    """运行时查询：targets 已经是真实 point id。plan 落种和 plan_next 都产出它。"""
    text: str
    targets: list[str] = Field(default_factory=list)


class Source(BaseModel):
    id: str                       # 传给 findings.source_id，也是 sources 字典的 key
    url: str                      # 原始 URL（给报告用）
    norm_url: str                 # 归一化 URL（去重口径）
    domain: str                   # 可注册域名（算"独立来源"时按它去重）
    title: str = ""
    summary: str = ""             # Tavily 的 content，截断 1500 字，quote 校验的对象


class Finding(BaseModel):
    """固化后的证据。id 由代码生成，之后任何人不得改写这条记录的语义。"""
    id: str
    point_ids: list[str] = Field(min_length=1)
    claim: str
    quote: str
    source_id: str
    stance: Literal["supports", "contradicts", "context"]
def make_extract_schema(open_ids: list[str], result_ids: list[str]):
    """extract 的 LLM 输出 schema。两个 id 空间都锁成枚举，模型无法越界。"""
    PointId = Literal[tuple(open_ids)]
    ResultId = Literal[tuple(result_ids)]

    class RawFinding(BaseModel):
        point_ids: list[PointId] = Field(
            min_length=1,
            description="这条证据回答了清单里的哪几个点，只能用给定的点 id",
        )
        result_id: ResultId = Field(description="quote 来自哪一条结果")
        quote: str = Field(min_length=4, description="从结果摘要里逐字复制的原文")
        claim: str = Field(min_length=4, description="quote 支持的一句原子断言")
        stance: Literal["supports", "contradicts", "context"]

    class ExtractOutput(BaseModel):
        findings: list[RawFinding]

    return ExtractOutput

def make_next_schema(open_ids: list[str]):
    """plan_next 的输出 schema。点 id 锁成枚举——这是清单不可被改写的硬保证。"""
    PointId = Literal[tuple(open_ids)]

    class NextQuery(BaseModel):
        text: str = Field(min_length=2, description="一条短的自然语言查询")
        targets: list[PointId] = Field(
            min_length=1,
            description="这条查询想回答哪些点，只能用给定的点 id",
        )

    class NextStep(BaseModel):
        queries: list[NextQuery] = Field(
            default_factory=list, max_length=3,
            description="1~2 条；认为没必要再搜时留空",
        )
        new_points: list[PlannedPoint] = Field(
            default_factory=list,
            description="清单整体漏掉的方面。复用 PlannedPoint：新点也必须带 kind/accepts",
        )
        give_up_reason: str = Field(
            default="", description="queries 留空时必填",
        )

    return NextStep


class ResearchResult(BaseModel):
    task_id: str
    findings: list[Finding]
    points: list[PointState]
    sources: list[Source]

