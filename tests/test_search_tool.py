"""使用临时缓存和模拟搜索客户端验证工具日志，不依赖 API key 或外部服务。"""

import asyncio
import importlib.util
import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


def load_tools():
    path = Path(__file__).resolve().parents[1] / "src/research_agent/graph/tools.py"
    spec = importlib.util.spec_from_file_location("_search_tools_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch("tavily.AsyncTavilyClient", return_value=SimpleNamespace(search=AsyncMock())):
        spec.loader.exec_module(module)
    return module


tools = load_tools()


class SearchToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = {
            "configurable": {"thread_id": "run-tool-test"},
            "metadata": {"research_task_id": "T3", "research_round": 2},
        }
        self.query = "FastAPI BackgroundTasks"
        self.payload = {
            "success": True, "query": self.query,
            "search_result": [{"url": "https://example.com/doc", "title": "资料"}],
        }
        self.remote = AsyncMock(return_value={
            "results": [{"url": "https://example.com/doc", "content": "搜索内容"}],
        })
        for target, attribute, value in [
            (tools.cache, "CACHE_DIR", self.directory),
            (tools.cache, "OFFLINE", False),
            (tools, "CACHE_NS", "basic-5-1500"),
            (tools.client, "search", self.remote),
        ]:
            replacement = patch.object(target, attribute, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        log_patch = patch.object(tools.logger, "log")
        self.log = log_patch.start()
        self.addCleanup(log_patch.stop)

    def read_record(self, status, level=logging.INFO):
        self.log.assert_called_once()
        actual_level, payload = self.log.call_args.args
        record = json.loads(payload)
        self.assertEqual(actual_level, level)
        self.assertEqual(record["status"], status)
        self.assertEqual(record["tool_name"], "web_search_tool")
        self.assertEqual(record["run_id"], "run-tool-test")
        self.assertEqual(record["task_id"], "T3")
        self.assertEqual(record["round"], 2)
        self.assertGreaterEqual(record["duration_ms"], 0)
        return record

    async def test_cache_hit_returns_payload_without_remote_request(self):
        tools.cache.put(self.query, self.payload, tools.CACHE_NS)
        result = await tools.web_search_tool.ainvoke(self.query, config=self.config)
        record = self.read_record("succeeded")
        self.assertEqual(result, self.payload)
        self.assertEqual(record["cache_status"], "hit")
        self.assertFalse(record["remote_requested"])
        self.assertIsNone(record["remote_duration_ms"])
        self.assertEqual(record["result_count"], 1)
        self.remote.assert_not_awaited()

    async def test_miss_writes_cache_and_repeated_query_hits(self):
        first = await tools.web_search_tool.ainvoke(self.query, config=self.config)
        record = self.read_record("succeeded")
        self.assertEqual(record["cache_status"], "miss")
        self.assertTrue(record["remote_requested"])
        self.assertGreaterEqual(record["remote_duration_ms"], 0)
        self.assertEqual(first["search_result"][0]["summary"], "搜索内容")
        self.assertEqual(tools.cache.get(self.query, tools.CACHE_NS), first)
        self.log.reset_mock()

        # 大小写和空白不会改变当前缓存键。
        second = await tools.web_search_tool.ainvoke(
            "  fastapi   backgroundtasks ", config=self.config,
        )
        record = self.read_record("succeeded")
        self.assertEqual(record["cache_status"], "hit")
        self.assertEqual(first, second)
        self.remote.assert_awaited_once()

    async def test_namespace_change_does_not_reuse_old_cache(self):
        tools.cache.put(self.query, self.payload, "different-settings")
        await tools.web_search_tool.ainvoke(self.query, config=self.config)
        self.assertEqual(self.read_record("succeeded")["cache_status"], "miss")
        self.remote.assert_awaited_once()

    async def test_offline_miss_fails_before_remote_search(self):
        with patch.object(tools.cache, "OFFLINE", True):
            with self.assertRaisesRegex(RuntimeError, "离线模式且缓存未命中"):
                await tools.web_search_tool.ainvoke(self.query, config=self.config)
        record = self.read_record("failed", logging.ERROR)
        self.assertEqual(record["cache_status"], "miss")
        self.assertEqual(record["error"]["stage"], "offline_check")
        self.assertFalse(record["remote_requested"])
        self.remote.assert_not_awaited()

    async def test_offline_hit_still_succeeds(self):
        tools.cache.put(self.query, self.payload, tools.CACHE_NS)
        with patch.object(tools.cache, "OFFLINE", True):
            await tools.web_search_tool.ainvoke(self.query, config=self.config)
        self.assertEqual(self.read_record("succeeded")["cache_status"], "hit")
        self.remote.assert_not_awaited()

    async def test_remote_error_keeps_duration_and_does_not_write_cache(self):
        failure = TimeoutError("搜索超时")
        self.remote.side_effect = failure
        with self.assertRaises(TimeoutError) as raised:
            await tools.web_search_tool.ainvoke(self.query, config=self.config)
        record = self.read_record("failed", logging.ERROR)
        self.assertIs(raised.exception, failure)
        self.assertEqual(record["error"]["stage"], "remote_search")
        self.assertEqual(record["error"]["type"], "TimeoutError")
        self.assertTrue(record["remote_requested"])
        self.assertGreaterEqual(record["remote_duration_ms"], 0)
        self.assertIsNone(record["result_count"])
        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_corrupt_cache_identifies_read_failure(self):
        path = self.directory / f"{tools.cache._key(self.query, tools.CACHE_NS)}.json"
        path.write_text("not json", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            await tools.web_search_tool.ainvoke(self.query, config=self.config)
        record = self.read_record("failed", logging.ERROR)
        self.assertEqual(record["error"]["stage"], "cache_read")
        self.assertEqual(record["cache_status"], "unknown")
        self.remote.assert_not_awaited()

    async def test_normalization_error_is_distinct_from_remote_error(self):
        self.remote.return_value = {"results": [None]}
        with self.assertRaises(AttributeError):
            await tools.web_search_tool.ainvoke(self.query, config=self.config)
        record = self.read_record("failed", logging.ERROR)
        self.assertEqual(record["error"]["stage"], "normalize")
        self.assertIsNone(record["result_count"])

    async def test_cache_write_failure_keeps_received_result_count(self):
        with patch.object(tools.cache, "put", side_effect=OSError("缓存写入失败")):
            with self.assertRaises(OSError):
                await tools.web_search_tool.ainvoke(self.query, config=self.config)
        record = self.read_record("failed", logging.ERROR)
        self.assertEqual(record["error"]["stage"], "cache_write")
        self.assertEqual(record["result_count"], 1)
        self.assertTrue(record["remote_requested"])

    async def test_empty_results_are_successful_with_zero_count(self):
        self.remote.return_value = {"results": []}
        result = await tools.web_search_tool.ainvoke(self.query, config=self.config)
        self.assertEqual(self.read_record("succeeded")["result_count"], 0)
        self.assertEqual(result["search_result"], [])

    async def test_cancellation_logs_remote_wait_and_propagates(self):
        started = asyncio.Event()

        async def wait_for_cancel(**kwargs):
            started.set()
            await asyncio.Event().wait()

        self.remote.side_effect = wait_for_cancel
        task = asyncio.create_task(tools.web_search_tool.ainvoke(self.query, config=self.config))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        record = self.read_record("cancelled")
        self.assertEqual(record["cancelled_stage"], "remote_search")
        self.assertGreaterEqual(record["remote_duration_ms"], 0)
        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_direct_tool_call_needs_no_config_or_model_visible_config_argument(self):
        self.assertEqual(set(tools.web_search_tool.args), {"query"})
        await tools.web_search_tool.ainvoke(self.query)
        self.log.assert_called_once()
        record = json.loads(self.log.call_args.args[1])
        self.assertEqual(record["status"], "succeeded")
        self.assertIsNone(record["run_id"])
        self.assertIsNone(record["task_id"])


if __name__ == "__main__":
    unittest.main()
