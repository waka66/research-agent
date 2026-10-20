from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated
import operator

from research_agent.schemas.research import (ResearchPlan, ResearchResult, ResearchReport,ResearchTask,
                                             PointState, Query, Source, Finding)






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

    # ---- 代码拥有的账本 ----
    points: dict[str, PointState]
    findings: dict[str, Finding]    # 累计证据，只增不改；id -> Finding
    sources: dict[str, Source]      # 累计来源表，只增不删；search_node 是唯一写者

    # ---- 循环控制 ----
    pending_queries: list[Query]   # 待执行的查询（search_node 消费后清空）
    queries_tried: list[str]       # 真正执行过的查询（search_node 才写）
    round: int                     # 已完成的搜索轮数，plan 时 = 0
    no_progress: int               # 连续几轮没有新证据
    stop_now: bool                 # update 的裁决（这一步只记录，下一步才用来路由）
    stop_reason: str               # all_done / no_progress / max_rounds / errors

    # ---- 单轮载体 ----
    new_results: list[Source]        # 本轮新来源，只活一轮，给 extract 用
    round_findings: list[Finding]    # 本轮新增，只活一轮，给 update 判"有没有进展"用

    # ---- 预算与产出 ----
    search_calls: int
    error_calls: int
    out_of_scope: list[str]
    result: ResearchResult
