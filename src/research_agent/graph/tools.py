import asyncio
import logging
import os
import time

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from tavily import AsyncTavilyClient

from research_agent.graph import cache
from research_agent.graph.observability import tool_log

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
async def web_search_tool(query: str, config: RunnableConfig) -> dict:
    """联网搜索互联网公开信息，返回标准化标题、摘要、来源URL"""
    with tool_log(logger, "web_search_tool", config) as record:
        record.update(
            {
                "query": query,
                "cache_status": "unknown",
                "cache_namespace": CACHE_NS,
                "remote_requested": False,
                "remote_duration_ms": None,
                "result_count": None,
            }
        )
        stage = "cache_read"
        try:
            cached = cache.get(query, CACHE_NS)
            record["cache_status"] = "hit" if cached is not None else "miss"
            if cached is not None:
                record["result_count"] = len(cached.get("search_result") or [])
                record["status"] = "succeeded"
                return cached

            stage = "offline_check"
            if cache.OFFLINE:
                raise RuntimeError(f"离线模式且缓存未命中：{query}")

            stage = "remote_search"
            # 只表示进入了一次 SDK 调用，不代表供应商收到请求或产生计费。
            record["remote_requested"] = True
            remote_start = time.perf_counter()
            try:
                raw = await client.search(
                    query=query,
                    search_depth=SEARCH_DEPTH,
                    max_results=MAX_RESULTS,
                )
            finally:
                # 超时、异常和取消也保留已等待的时间。
                record["remote_duration_ms"] = round((time.perf_counter() - remote_start) * 1000, 2)

            stage = "normalize"
            payload = _normalize(query, raw)
            record["result_count"] = len(payload["search_result"])
            stage = "cache_write"
            cache.put(query, payload, CACHE_NS)
            record["status"] = "succeeded"
            return payload
        except asyncio.CancelledError:
            record["cancelled_stage"] = stage
            raise
        except Exception:
            # 公共记录会补上异常类型和信息，这里只补充工具特有的阶段。
            record["error"] = {"stage": stage}
            raise
