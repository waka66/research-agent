import os
import logging

from langchain.tools import tool
from tavily import AsyncTavilyClient

from research_agent.graph import cache

logger = logging.getLogger("research_agent/graph/tools.py")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
client = AsyncTavilyClient(TAVILY_API_KEY)

SEARCH_DEPTH = os.getenv("TAVILY_DEPTH", "basic")
MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
MAX_CONTENT = 1500

# 影响结果的参数全部进缓存命名空间：改了配置，旧缓存自动失效
CACHE_NS = f"{SEARCH_DEPTH}-{MAX_RESULTS}-{MAX_CONTENT}"


def _normalize(query: str, raw: dict) -> dict:
    return {
        "success": True,
        "query": query,
        "search_result": [
            {
                "title": item.get("title") or "",
                "summary": (item.get("content") or "")[:MAX_CONTENT],
                "url": item.get("url") or "",
                "id": item.get("id"),
            }
            for item in raw.get("results", [])
        ],
    }
@tool
async def web_search_tool(query: str) -> dict:
    """联网搜索互联网公开信息，返回标准化标题、摘要、来源URL"""
    cached = cache.get(query, CACHE_NS)
    if cached is not None:
        print("缓存命中")
        return cached

    if cache.OFFLINE:
        raise RuntimeError(f"离线模式且缓存未命中：{query}")

    raw = await client.search(
        query=query,
        search_depth=SEARCH_DEPTH,
        max_results=MAX_RESULTS,
    )
    payload = _normalize(query, raw)
    cache.put(query, payload, CACHE_NS)
    return payload
