"""记录生命周期与并行/取消/数据库故障，使用内存存储，不调用模型。"""

import asyncio
import json
import logging
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from research_agent.graph import observability as obs


class MemoryRecordStore:
    def __init__(self):
        self.rows = {}
        self.start_error = None
        self.finish_error = None
        self.start_entered = asyncio.Event()
        self.finish_entered = asyncio.Event()
        self.start_gate = None
        self.finish_gate = None

    async def start_record(self, record):
        self.start_entered.set()
        if self.start_gate:
            await self.start_gate.wait()
        if self.start_error:
            raise self.start_error
        self.rows[record["record_id"]] = deepcopy(record)

    async def finish_record(self, record):
        self.finish_entered.set()
        if self.finish_gate:
            await self.finish_gate.wait()
        if self.finish_error:
            raise self.finish_error
        if record["record_id"] not in self.rows:
            raise LookupError("记录不存在")
        self.rows[record["record_id"]] = deepcopy(record)


class RecordPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = MemoryRecordStore()
        self.execution_id = uuid4()
        self.config = {
            "configurable": {"thread_id": str(uuid4())},
            "metadata": {"execution_id": str(self.execution_id)},
        }
        self.logger = Mock()
        replacement = patch.object(obs.persistence_logger, "exception")
        self.persistence_error = replacement.start()
        self.addCleanup(replacement.stop)

    def records(self):
        return list(self.store.rows.values())

    async def test_running_is_visible_before_body_and_details_survive_finish(self):
        with obs.recording_execution(self.execution_id, self.store) as context:
            async with obs.node_log(self.logger, "Planner", {}, self.config) as record:
                saved = self.store.rows[record["record_id"]]
                self.assertEqual(saved["status"], "running")
                self.assertIsNone(saved["finished_at"])
                with obs.model_attempt(record, 1) as attempt:
                    attempt["usage"] = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
                record["output_summary"] = {"task_count": 2}
            self.assertTrue(context.records_complete)
        (saved,) = self.records()
        self.assertEqual(saved["status"], "succeeded")
        self.assertEqual(saved["output_summary"], {"task_count": 2})
        self.assertEqual(saved["usage"]["total_tokens"], 30)
        self.assertEqual(len(saved["attempts"]), 1)
        self.assertTrue(saved["usage_complete"])
        self.assertIsNotNone(saved["finished_at"])
        self.assertGreaterEqual(saved["duration_ms"], 0)
        self.logger.log.assert_called_once()
        self.assertEqual(json.loads(self.logger.log.call_args.args[1]), saved)

    async def test_parallel_branches_and_tools_keep_separate_parents(self):
        gate = asyncio.Barrier(2)

        async def branch(task_id):
            async with obs.node_log(
                self.logger,
                "search_node",
                {"task": SimpleNamespace(id=task_id)},
                self.config,
                round_number=2,
            ) as node:
                await gate.wait()
                tool_config = {
                    **self.config,
                    "metadata": {
                        **self.config["metadata"],
                        "research_task_id": task_id,
                        "research_round": 2,
                    },
                }
                async with obs.tool_log(self.logger, "web_search_tool", tool_config) as tool:
                    self.assertEqual(tool["parent_record_id"], node["record_id"])
                    tool.update(cache_status="hit", remote_requested=False)

        with obs.recording_execution(self.execution_id, self.store):
            async with obs.node_log(self.logger, "research_node", {}, self.config) as root:
                await asyncio.gather(branch("T1"), branch("T2"))
        rows = self.records()
        self.assertEqual(len(rows), 5)
        nodes = {r["task_id"]: r for r in rows if r["name"] == "search_node"}
        for item in rows:
            if item["name"] == "search_node":
                self.assertEqual(item["parent_record_id"], root["record_id"])
            if item["kind"] == "tool":
                self.assertEqual(item["parent_record_id"], nodes[item["task_id"]]["record_id"])
                self.assertEqual(item["round"], 2)

    async def test_repeated_nodes_get_new_ids_and_resume_scope_is_independent(self):
        old = None
        for execution_id in (self.execution_id, uuid4()):
            config = {**self.config, "metadata": {"execution_id": str(execution_id)}}
            with obs.recording_execution(execution_id, self.store):
                for _ in range(2):
                    async with obs.node_log(self.logger, "Planner", {}, config):
                        pass
            if old is None:
                old = deepcopy(self.store.rows)
        self.assertEqual(len(self.store.rows), 4)
        self.assertTrue(all(self.store.rows[key] == value for key, value in old.items()))
        self.assertTrue(all(r["parent_record_id"] is None for r in self.records()))

    async def test_skip_degradation_and_partial_success_are_preserved(self):
        with obs.recording_execution(self.execution_id, self.store):
            for status in ("skipped", "degraded", "partial_success"):
                async with obs.node_log(self.logger, "test_node", {}, self.config) as record:
                    record["status"] = status
        self.assertEqual(
            [r["status"] for r in self.records()],
            [
                "skipped",
                "degraded",
                "partial_success",
            ],
        )

    async def test_write_failure_preserves_original_business_exception(self):
        error = TimeoutError("原始模型超时")
        self.store.finish_error = ConnectionError("记录数据库不可用")
        with obs.recording_execution(self.execution_id, self.store) as context:
            with self.assertRaises(TimeoutError) as raised:
                async with obs.node_log(self.logger, "extract_node", {}, self.config):
                    raise error
            self.assertIs(raised.exception, error)
            self.assertFalse(context.records_complete)
        payload = json.loads(self.logger.log.call_args.args[1])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error"]["message"], "原始模型超时")
        self.assertEqual(self.logger.log.call_args.args[0], logging.ERROR)

    async def test_missing_parent_does_not_make_children_reference_nonexistent_row(self):
        with obs.recording_execution(self.execution_id, self.store) as context:
            self.store.start_error = ConnectionError("开始写入失败")
            async with obs.node_log(self.logger, "research_node", {}, self.config):
                self.store.start_error = None
                async with obs.node_log(self.logger, "search_node", {}, self.config) as child:
                    self.assertIsNone(child["parent_record_id"])
            self.assertFalse(context.records_complete)
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.records()[0]["status"], "succeeded")

    async def test_parallel_write_failure_marks_shared_execution_incomplete(self):
        original = self.store.finish_record

        async def finish(record):
            if record["name"] == "bad":
                raise ConnectionError("一个分支记录失败")
            await original(record)

        async def branch(name):
            async with obs.node_log(self.logger, name, {}, self.config):
                pass

        self.store.finish_record = finish
        with obs.recording_execution(self.execution_id, self.store) as context:
            await asyncio.gather(branch("bad"), branch("good"))
            self.assertFalse(context.records_complete)

    async def test_business_cancellation_closes_nested_records(self):
        entered = asyncio.Event()

        async def run():
            with obs.recording_execution(self.execution_id, self.store):
                async with obs.node_log(self.logger, "search_node", {}, self.config):
                    async with obs.tool_log(self.logger, "web_search_tool", self.config):
                        entered.set()
                        await asyncio.Event().wait()

        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual([r["status"] for r in self.records()], ["cancelled", "cancelled"])

    async def test_cancel_during_start_waits_for_commit_and_never_enters_body(self):
        self.store.start_gate = asyncio.Event()
        entered_body = False

        async def run():
            nonlocal entered_body
            with obs.recording_execution(self.execution_id, self.store):
                async with obs.node_log(self.logger, "Planner", {}, self.config):
                    entered_body = True

        task = asyncio.create_task(run())
        await self.store.start_entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        self.store.start_gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(entered_body)
        self.assertEqual(self.records()[0]["status"], "cancelled")

    async def test_cancel_during_finish_does_not_abandon_record_commit(self):
        self.store.finish_gate = asyncio.Event()

        async def run():
            with obs.recording_execution(self.execution_id, self.store):
                async with obs.node_log(self.logger, "generator", {}, self.config):
                    pass

        task = asyncio.create_task(run())
        await self.store.finish_entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        self.store.finish_gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.records()[0]["status"], "succeeded")

    async def test_database_write_timeout_does_not_block_or_fail_business(self):
        self.store.start_gate = asyncio.Event()
        self.store.finish_gate = asyncio.Event()
        with (
            patch.object(obs, "RECORD_WRITE_TIMEOUT_SECONDS", 0.01),
            obs.recording_execution(self.execution_id, self.store) as context,
        ):
            async with obs.node_log(self.logger, "Planner", {}, self.config) as record:
                record["output_summary"] = {"task_count": 1}
            self.assertFalse(context.records_complete)
        self.assertEqual(json.loads(self.logger.log.call_args.args[1])["status"], "succeeded")

    async def test_mismatched_execution_is_not_written_to_another_execution(self):
        config = {**self.config, "metadata": {"execution_id": str(uuid4())}}
        with obs.recording_execution(self.execution_id, self.store) as context:
            async with obs.node_log(self.logger, "Planner", {}, config):
                pass
            self.assertFalse(context.records_complete)
        self.assertEqual(self.store.rows, {})

    async def test_without_service_scope_only_logs_and_context_does_not_leak(self):
        with obs.recording_execution(self.execution_id, self.store):
            async with obs.node_log(self.logger, "Planner", {}, self.config):
                pass
        async with obs.node_log(self.logger, "standalone", {}, self.config) as record:
            self.assertIsNone(record["parent_record_id"])
        self.assertEqual(len(self.store.rows), 1)
