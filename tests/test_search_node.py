"""search_node 的离线行为测试：不初始化模型、搜索客户端或数据库。

运行：python -m unittest discover -s tests -v
"""

import asyncio
import importlib.util
import json
import logging
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.graph import END, START, StateGraph

from research_agent.graph.state import ResearchWorkerState
from research_agent.schemas.research import Query, ResearchTask, Source


def load_worker_nodes():
    # 只替换有外部服务初始化的两个模块；节点、模型结构、URL 校验均使用真实代码。
    llm_module = ModuleType("research_agent.llm")
    llm_module.llm = None
    tool_module = ModuleType("research_agent.graph.tools")
    tool_module.web_search_tool = SimpleNamespace(ainvoke=AsyncMock())
    path = (
        Path(__file__).resolve().parents[1]
        / "src/research_agent/graph/worker_nodes.py"
    )
    spec = importlib.util.spec_from_file_location("_search_worker_nodes_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "research_agent.llm": llm_module,
        "research_agent.graph.tools": tool_module,
    }):
        spec.loader.exec_module(module)
    return module


nodes = load_worker_nodes()


class SearchNodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {"configurable": {"thread_id": "run-test"}}
        self.state = {
            "task": ResearchTask(id="T2", question="测试问题", objective="测试目标"),
            "round": 2,
            "pending_queries": [Query(text="查询甲"), Query(text="查询乙")],
            "sources": {},
            "queries_tried": ["历史查询"],
            "search_calls": 1,
            "error_calls": 0,
        }
        self.tool = AsyncMock()
        tool_patch = patch.object(nodes.web_search_tool, "ainvoke", self.tool)
        log_patch = patch.object(nodes.logger, "log")
        tool_patch.start()
        self.log = log_patch.start()
        self.addCleanup(tool_patch.stop)
        self.addCleanup(log_patch.stop)

    def read_record(self, status, level):
        self.log.assert_called_once()
        actual_level, payload = self.log.call_args.args
        record = json.loads(payload)
        self.assertEqual(actual_level, level)
        self.assertEqual(record["status"], status)
        self.assertEqual(record["node_name"], "search_node")
        self.assertEqual(record["run_id"], "run-test")
        self.assertEqual(record["task_id"], "T2")
        self.assertEqual(record["round"], 3)
        self.assertGreaterEqual(record["duration_ms"], 0)
        self.assertTrue(record["started_at"].endswith("+00:00"))
        return record

    async def test_success_deduplicates_sources_without_mutating_input(self):
        old = Source(
            id="old", url="https://example.com/old", norm_url="https://example.com/old",
            domain="example.com",
        )
        self.state["sources"] = {old.id: old}
        before = deepcopy(self.state)
        self.tool.side_effect = [
            {"search_result": [
                {"url": "https://example.com/old?utm_source=test"},
                {"url": "https://example.com/new", "title": "新资料"},
            ]},
            {"search_result": [
                {"url": "https://example.com/new?utm_source=test"},
                {"url": ""},
            ]},
        ]

        result = await nodes.search_node(self.state, self.config)

        record = self.read_record("succeeded", logging.INFO)
        self.assertEqual(record["output_summary"], {
            "query_count": 2, "succeeded_query_count": 2, "failed_query_count": 0,
            "new_source_count": 1, "total_source_count": 2,
        })
        self.assertEqual(len(result["new_results"]), 1)
        self.assertEqual(result["search_calls"], 3)
        self.assertEqual(result["queries_tried"], ["历史查询", "查询甲", "查询乙"])
        self.assertEqual(result["pending_queries"], [])
        self.assertEqual(self.state, before)
        self.assertNotIn("status", result)

    async def test_empty_search_results_are_successful(self):
        self.tool.return_value = {"search_result": []}
        result = await nodes.search_node(self.state, self.config)
        record = self.read_record("succeeded", logging.INFO)
        self.assertEqual(record["output_summary"]["new_source_count"], 0)
        self.assertEqual(result["error_calls"], 0)

    async def test_partial_failure_preserves_successful_results(self):
        self.tool.side_effect = [
            TimeoutError("搜索超时"),
            {"search_result": [{"url": "https://example.com/new"}]},
        ]
        result = await nodes.search_node(self.state, self.config)
        record = self.read_record("partial_success", logging.WARNING)
        self.assertEqual(record["query_errors"], [
            {"query": "查询甲", "type": "TimeoutError", "message": "搜索超时"},
        ])
        self.assertEqual(record["output_summary"]["failed_query_count"], 1)
        self.assertEqual(result["search_calls"], 2)
        self.assertEqual(result["error_calls"], 1)
        self.assertEqual(result["queries_tried"], ["历史查询", "查询乙"])
        self.assertEqual(len(result["new_results"]), 1)

    async def test_all_query_failures_return_counters_for_stop_decision(self):
        self.tool.side_effect = [TimeoutError("超时"), RuntimeError("离线缓存未命中")]
        result = await nodes.search_node(self.state, self.config)
        record = self.read_record("failed", logging.ERROR)
        self.assertEqual(record["error"]["type"], "AllSearchQueriesFailed")
        self.assertEqual(len(record["query_errors"]), 2)
        self.assertEqual(record["output_summary"]["succeeded_query_count"], 0)
        self.assertEqual(result["search_calls"], 1)
        self.assertEqual(result["error_calls"], 2)
        self.assertEqual(result["new_results"], [])

    async def test_budget_exhausted_skips_without_calling_tool(self):
        self.state["search_calls"] = nodes.MAX_SEARCH_CALLS
        result = await nodes.search_node(self.state, self.config)
        record = self.read_record("skipped", logging.INFO)
        self.assertEqual(record["output_summary"]["reason"], "search_budget_exhausted")
        self.tool.assert_not_called()
        self.assertEqual(result["error_calls"], 0)

    async def test_no_queries_preserves_existing_error_counter_policy(self):
        self.state["pending_queries"] = []
        result = await nodes.search_node(self.state, self.config)
        record = self.read_record("skipped", logging.INFO)
        self.assertEqual(record["output_summary"]["reason"], "no_pending_queries")
        self.tool.assert_not_called()
        self.assertEqual(result["error_calls"], 1)

    async def test_remaining_budget_limits_dispatch(self):
        self.state["search_calls"] = nodes.MAX_SEARCH_CALLS - 1
        self.tool.return_value = {"search_result": []}
        result = await nodes.search_node(self.state, self.config)
        record = self.read_record("succeeded", logging.INFO)
        self.tool.assert_awaited_once_with("查询甲")
        self.assertEqual(record["output_summary"]["query_count"], 1)
        self.assertEqual(result["search_calls"], nodes.MAX_SEARCH_CALLS)

    async def test_unexpected_processing_error_is_logged_and_reraised(self):
        self.tool.return_value = {"search_result": [None]}
        with self.assertRaises(AttributeError):
            await nodes.search_node(self.state, self.config)
        record = self.read_record("failed", logging.ERROR)
        self.assertEqual(record["error"]["type"], "AttributeError")
        self.assertEqual(self.state["sources"], {})

    async def test_cancellation_propagates_and_cancels_parallel_calls(self):
        both_started = asyncio.Event()
        started = []
        cancelled = []

        async def wait_for_cancel(query):
            started.append(query)
            if len(started) == 2:
                both_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(query)
                raise

        self.tool.side_effect = wait_for_cancel
        task = asyncio.create_task(nodes.search_node(self.state, self.config))
        try:
            await asyncio.wait_for(both_started.wait(), timeout=2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.read_record("cancelled", logging.INFO)
        self.assertCountEqual(cancelled, ["查询甲", "查询乙"])

    async def test_child_cancellation_is_not_treated_as_query_failure(self):
        self.tool.side_effect = [asyncio.CancelledError(), {"search_result": []}]
        with self.assertRaises(asyncio.CancelledError):
            await nodes.search_node(self.state, self.config)
        record = self.read_record("cancelled", logging.INFO)
        self.assertEqual(record["query_errors"], [])

    async def test_langgraph_supplies_config_to_node(self):
        self.tool.return_value = {"search_result": []}
        builder = StateGraph(ResearchWorkerState)
        builder.add_node("search_node", nodes.search_node)
        builder.add_edge(START, "search_node")
        builder.add_edge("search_node", END)
        result = await builder.compile().ainvoke(self.state, config=self.config)
        self.read_record("succeeded", logging.INFO)
        self.assertEqual(result["search_calls"], 3)


if __name__ == "__main__":
    unittest.main()
