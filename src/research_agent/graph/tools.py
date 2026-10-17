from langchain.tools import tool
from tavily import AsyncTavilyClient

import os

from research_agent.llm import llm






TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
client = AsyncTavilyClient(TAVILY_API_KEY)
@tool
async def web_search_tool(query: str) -> dict:
    """联网搜索互联网公开信息，返回标准化标题、摘要、来源URL
    Args:
        query: 搜索关键词/问题
    """
    # 官方标准调用，深度高级检索
    response =await client.search(
        query=query,
        search_depth="advanced"  #advanced  两倍消耗，basic普通
    )
    result_list = []
    # 提取核心字段，规整输出
    max_length = 1500
    for item in response.get("results", []):
        content = lambda item, max_length: item.get("content")[:max_length] if item.get("content") else ""
        result_list.append({
            "title": item.get("title"),
            "summary": content(item, max_length),
            "url": item.get("url"),
            "id": item.get("id")
        })
    return {
        "success": True,
        "query": query,
        "search_result": result_list,
    }


tools = [web_search_tool]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)