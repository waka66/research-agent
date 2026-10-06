from langchain.messages import HumanMessage, ToolMessage, AnyMessage, SystemMessage
from langgraph.types import Send



from research_agent.llm import llm
from research_agent.graph.state import ResearchState
from research_agent.schemas.research import ResearchPlan, ResearchReport, ResearcherInput
from research_agent.graph.worker_workflow import worker_graph






async def Planner(state: ResearchState):
    planner_model = llm.with_structured_output(
        ResearchPlan,
        method="function_calling",
        # include_raw=True,    # 调试用的，保留原始输出
    )

    plan: ResearchPlan =await planner_model.ainvoke(
        [
            SystemMessage(content="""
                你是 Research Agent 的 Planner 节点。

                你的职责：
                1. 分析用户的研究问题
                2. 将问题拆分为多个可独立执行的研究任务
                3. 每个任务必须能够通过搜索和资料分析独立完成
                4. 任务之间尽量不存在依赖关系，以便系统并行执行
                5. 不负责最终总结
                6. 不创建“综合以上任务”“形成最终报告”之类的任务
                7. 不进行网页搜索
                8. 不向用户询问确认
                9. task的id以T1,T2,T3...的形式编号

                只输出研究任务计划。
                """)
        ] + state["question"]
    )
    #print("----------plan-----------")
    #print(plan)

    # 把 Pydantic 对象序列化成 JSON 字符串，再放进 AIMessage
    return {
        "research_plan": plan,
        "status": "running"
    }



def dispatch_research_tasks(state: ResearchState):
    return [
        Send(
            "research_node",
            {"task": task}
        )
        for task in state["research_plan"].tasks
    ]


async def research_node(state: ResearcherInput):

    worker_result = await worker_graph.ainvoke({
        "task": state["task"],
        "messages": [],
        "result": None,
        "search_calls": 0,
    })
    #print("----------result-----------")
    #print(worker_result["result"])

    return {
        "research_results": [
            worker_result["result"]
        ],
        "status": "running"
    }


async def Synthesizer(state: ResearchState):
    synthesizer_model = llm.with_structured_output(
        ResearchReport,
        method="function_calling"
    )
    result: ResearchReport = await synthesizer_model.ainvoke(
        [
            SystemMessage(content="""
                你是 Research Agent 的 synthesizer 节点。
                你的职责：根据多个研究结果生成综合报告
                """)
        ]+
        [
            HumanMessage(content=f"研究计划：{state['research_plan']}\n研究结果：{state['research_results']}")
        ]
    )
    return {
        "report": result,
        "status": "running"
    }


async def generator(state: ResearchState):
    result = await llm.ainvoke(
        [
            SystemMessage(content="""
                你是 Research Agent 的 generator 节点。
                你的职责：根据综合报告生成最终报告,需要markdown格式，最终由前端格式化
                """)
        ]+
        [
            HumanMessage(content=f"综合报告：{state['report']}")
        ]
    )
    return {
        "final_answer": result,
        "status": "completed"
    }
