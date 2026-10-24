"""首次启动的 API/SSE 生命周期测试，不连接数据库或模型服务。"""

import asyncio
import importlib.util
import json
import sys
import unittest
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI

from research_agent.db.errors import ExecutionConflictError
from research_agent.schemas.research import ResearchPlan

ROOT = Path(__file__).resolve().parents[1] / "src/research_agent"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryHistory:
    def __init__(self):
        self.rows = {}
        self.create_error = None
        self.finish_error = None

    async def create_start_execution(self, run_id, execution_id, question):
        if self.create_error:
            raise self.create_error
        self.rows[execution_id] = {
            "run_id": run_id,
            "execution_id": execution_id,
            "question": question,
            "mode": "start",
            "status": "pending",
            "created_at": datetime.now(UTC),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "records_complete": True,
            "result_summary": None,
        }

    async def mark_execution_running(self, execution_id):
        row = self.rows[execution_id]
        assert row["status"] == "pending"
        row.update(status="running", started_at=datetime.now(UTC))

    async def create_resume_execution(self, run_id, execution_id, expected_execution_id):
        if self.create_error:
            raise self.create_error
        latest = await self.get_latest_execution(run_id)
        if latest is None:
            raise LookupError(run_id)
        if latest["execution_id"] != expected_execution_id or latest["status"] not in {
            "failed",
            "cancelled",
            "interrupted",
        }:
            raise ExecutionConflictError("执行状态已变化")
        await self.create_start_execution(run_id, execution_id, latest["question"])
        self.rows[execution_id]["mode"] = "resume"

    async def finish_execution(
        self,
        execution_id,
        status,
        error=None,
        *,
        records_complete=True,
        result_summary=None,
    ):
        if self.finish_error:
            raise self.finish_error
        row = self.rows[execution_id]
        row["records_complete"] = row["records_complete"] and records_complete
        if row["status"] == status:
            return False
        assert row["status"] in {"pending", "running"}
        row.update(status=status, error=error, finished_at=datetime.now(UTC))
        row["result_summary"] = deepcopy(result_summary)
        return True

    async def get_execution(self, execution_id):
        return deepcopy(self.rows.get(execution_id))

    async def get_latest_execution(self, run_id):
        rows = [row for row in self.rows.values() if row["run_id"] == run_id]
        return deepcopy(max(rows, key=lambda row: row["created_at"])) if rows else None


class FakeAgent:
    def __init__(self):
        self.config = None
        self.inputs = None
        self.gate = None
        self.started = asyncio.Event()
        self.failure = None
        self.tail_failure = None
        self.snapshot_error = None
        self.snapshot = {"status": "running"}
        self.checkpoint_id = "saved-checkpoint"
        self.next_nodes = ("research_node",)

    async def astream(self, inputs, config, stream_mode):
        self.inputs, self.config = inputs, config
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        yield {"Planner": {"research_plan": ResearchPlan(goal="研究", tasks=[])}}
        if self.failure:
            raise self.failure
        yield {"research_node": {"research_results": []}}
        yield {"Synthesizer": {"report": None, "status": "partial"}}
        yield {"generator": {"final_answer": "# 测试报告\n中文内容", "status": "completed"}}
        if self.tail_failure:
            raise self.tail_failure

    async def aget_state(self, config, subgraphs):
        if self.snapshot_error:
            raise self.snapshot_error
        return SimpleNamespace(
            values=self.snapshot,
            config={"configurable": {"checkpoint_id": self.checkpoint_id}},
            next=self.next_nodes,
        )


class ExecutionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # patch.dict(sys.modules) 会移除范围内新导入的模块；提前加载，避免数据库测试
        # 随后加载另一套 SQLAlchemy PostgreSQL 类型，破坏 JSONB 参数推断。
        self.record_store = importlib.import_module("research_agent.db.records")
        self.snapshot_module = importlib.import_module("research_agent.services.result_snapshot")
        self.history = MemoryHistory()
        self.agent = FakeAgent()
        history_module = ModuleType("research_agent.db.history")
        for name in (
            "create_start_execution",
            "create_resume_execution",
            "mark_execution_running",
            "finish_execution",
            "get_execution",
            "get_latest_execution",
        ):
            setattr(history_module, name, getattr(self.history, name))
        agent_module = ModuleType("research_agent.graph.agent")
        agent_module.get_agent = lambda: self.agent
        with patch.dict(
            sys.modules,
            {
                "research_agent.db.history": history_module,
                "research_agent.graph.agent": agent_module,
            },
        ):
            self.service = load_file("_execution_service_test", "services/research_service.py")
            with patch.dict(
                sys.modules,
                {
                    "research_agent.services.research_service": self.service,
                },
            ):
                self.api = load_file("_execution_api_test", "api/research.py")
        self.app = FastAPI()
        self.app.include_router(self.api.router)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://test",
        )
        for logger in (self.service.logger, self.api.logger):
            replacement = patch.object(logger, "exception")
            replacement.start()
            self.addCleanup(replacement.stop)
        replacement = patch.object(self.api.logger, "error")
        replacement.start()
        self.addCleanup(replacement.stop)

    async def asyncTearDown(self):
        for task in self.api.runs.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.api.runs.values(), return_exceptions=True)
        await self.client.aclose()

    async def start(self):
        result = await self.client.post("/research/", json={"question": "研究测试问题"})
        self.assertEqual(result.status_code, 202, result.text)
        payload = result.json()
        self.assertEqual(payload["status"], "pending")
        return UUID(payload["run_id"]), UUID(payload["execution_id"])

    async def wait_finished(self, run_id):
        await asyncio.wait_for(asyncio.shield(self.api.runs[run_id]), timeout=2)

    async def events(self, run_id):
        result = await self.client.get(f"/research/{run_id}/stream")
        self.assertEqual(result.status_code, 200, result.text)
        events = []
        for block in result.text.strip().split("\n\n"):
            event, data = block.split("\n", 1)
            events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
        return events

    async def test_success_tracks_execution_and_encodes_json_sse(self):
        run_id, execution_id = await self.start()
        await self.wait_finished(run_id)
        row = await self.history.get_execution(execution_id)
        self.assertEqual(row["status"], "completed")
        self.assertIsNotNone(row["started_at"])
        self.assertIsNotNone(row["finished_at"])
        self.assertEqual(self.agent.config["configurable"]["thread_id"], str(run_id))
        self.assertEqual(self.agent.config["metadata"]["execution_id"], str(execution_id))
        events = await self.events(run_id)
        self.assertEqual(
            [event for event, _ in events],
            [
                "plan_ready",
                "research_completed",
                "synthesis_completed",
                "completed",
            ],
        )
        self.assertEqual(events[-1][1]["execution_id"], str(execution_id))
        self.assertIn("中文内容", events[-1][1]["final_answer"])

    async def test_failed_graph_is_persisted_and_closes_sse(self):
        self.agent.failure = TimeoutError("模型调用超时")
        run_id, execution_id = await self.start()
        await self.wait_finished(run_id)
        row = await self.history.get_execution(execution_id)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"]["type"], "TimeoutError")
        events = await self.events(run_id)
        self.assertEqual(events[-1][0], "failed")
        self.assertEqual(events[-1][1]["error"]["message"], "模型调用超时")
        self.assertNotIn("completed", [event for event, _ in events])

    async def test_failure_after_generator_does_not_send_completed(self):
        self.agent.tail_failure = RuntimeError("检查点收尾失败")
        run_id, execution_id = await self.start()
        await self.wait_finished(run_id)
        self.assertEqual(self.history.rows[execution_id]["status"], "failed")
        self.assertEqual((await self.events(run_id))[-1][0], "failed")

    async def test_final_status_is_saved_before_terminal_event(self):
        history = self.history
        states = []

        class InspectQueue(asyncio.Queue):
            async def put(queue_self, item):
                if item[0] in {"completed", "failed", "cancelled"}:
                    states.append(history.rows[execution_id]["status"])
                await super().put(item)

        run_id, execution_id = uuid4(), uuid4()
        await history.create_start_execution(run_id, execution_id, "测试")
        await self.service.run_research(self.agent, run_id, execution_id, "测试", InspectQueue())
        self.assertEqual(states, ["completed"])

    async def test_running_cancellation_is_persisted_once(self):
        self.agent.gate = asyncio.Event()
        run_id, execution_id = await self.start()
        await self.agent.started.wait()
        result = await self.client.post(f"/research/{run_id}/cancel")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "cancelled")
        self.assertEqual(self.history.rows[execution_id]["status"], "cancelled")
        self.assertEqual([event for event, _ in await self.events(run_id)], ["cancelled"])

    async def test_cancellation_before_coroutine_starts_finishes_pending_record(self):
        run_id, execution_id = uuid4(), uuid4()
        await self.history.create_start_execution(run_id, execution_id, "测试")
        queue = asyncio.Queue()
        task = asyncio.create_task(
            self.service.run_research(
                self.agent,
                run_id,
                execution_id,
                "测试",
                queue,
            )
        )
        self.api.runs[run_id] = task
        self.api.run_queues[run_id] = queue
        self.api.run_execution_ids[run_id] = execution_id
        result = await self.api.research_cancel(run_id)
        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(self.history.rows[execution_id]["started_at"])
        self.assertEqual(queue.qsize(), 1)
        self.assertEqual(queue.get_nowait()[0], "cancelled")

    async def test_cancel_completed_task_reports_actual_status(self):
        run_id, _ = await self.start()
        await self.wait_finished(run_id)
        result = await self.client.post(f"/research/{run_id}/cancel")
        self.assertEqual(result.json()["status"], "completed")

    async def test_status_prefers_execution_over_stale_graph_state(self):
        self.agent.failure = RuntimeError("失败")
        run_id, execution_id = await self.start()
        await self.wait_finished(run_id)
        result = await self.client.get(f"/research/{run_id}")
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["status"], "failed")
        self.assertEqual(result.json()["execution_id"], str(execution_id))
        self.assertIsNone(result.json()["plan"])
        self.assertEqual(result.json()["results"], [])

    async def test_status_serializes_structured_plan_and_checkpoint_failure(self):
        self.agent.snapshot = {"research_plan": ResearchPlan(goal="规划", tasks=[])}
        run_id, _ = await self.start()
        await self.wait_finished(run_id)
        result = await self.client.get(f"/research/{run_id}")
        self.assertEqual(result.json()["plan"]["goal"], "规划")
        self.agent.snapshot_error = RuntimeError("检查点暂不可用")
        result = await self.client.get(f"/research/{run_id}")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "completed")
        self.assertFalse(result.json()["checkpoint_available"])

    async def test_terminal_status_can_be_queried_and_replayed_without_memory_queue(self):
        run_id, _ = await self.start()
        await self.wait_finished(run_id)
        self.api.run_queues.clear()
        self.api.runs.clear()
        self.assertEqual(
            (await self.client.get(f"/research/{run_id}")).json()["status"], "completed"
        )
        self.assertEqual([event for event, _ in await self.events(run_id)], ["completed"])

    async def test_missing_run_is_404_and_lost_active_stream_is_409(self):
        run_id, execution_id = uuid4(), uuid4()
        self.assertEqual((await self.client.get(f"/research/{run_id}")).status_code, 404)
        self.assertEqual((await self.client.get(f"/research/{run_id}/stream")).status_code, 404)
        await self.history.create_start_execution(run_id, execution_id, "测试")
        self.assertEqual((await self.client.get(f"/research/{run_id}/stream")).status_code, 409)

    async def test_create_database_failure_does_not_start_graph(self):
        self.history.create_error = ConnectionError("数据库不可用")
        result = await self.client.post("/research/", json={"question": "测试"})
        self.assertEqual(result.status_code, 503)
        self.assertEqual(self.api.runs, {})
        self.assertFalse(self.agent.started.is_set())

    async def test_terminal_persistence_failure_is_not_reported_as_completed(self):
        self.history.finish_error = ConnectionError("数据库不可用")
        run_id, execution_id = await self.start()
        with self.assertRaises(ConnectionError):
            await self.wait_finished(run_id)
        events = await self.events(run_id)
        self.assertEqual(events[-1][0], "failed")
        self.assertFalse(events[-1][1]["persisted"])
        self.assertEqual(events[-1][1]["status"], "unknown")
        self.assertEqual(self.history.rows[execution_id]["status"], "running")

    async def test_cancel_during_completion_does_not_interrupt_commit(self):
        run_id, execution_id = uuid4(), uuid4()
        await self.history.create_start_execution(run_id, execution_id, "测试")
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.history.finish_execution

        async def blocked_finish(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        queue = asyncio.Queue()
        with patch.object(self.service, "finish_execution", blocked_finish):
            task = asyncio.create_task(
                self.service.run_research(
                    self.agent,
                    run_id,
                    execution_id,
                    "测试",
                    queue,
                )
            )
            await entered.wait()
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.history.rows[execution_id]["status"], "completed")
        events = []
        while not queue.empty():
            events.append(queue.get_nowait()[0])
        self.assertEqual(events[-1], "completed")

    async def failed_start(self):
        self.agent.failure = TimeoutError("首次失败")
        run_id, execution_id = await self.start()
        await self.wait_finished(run_id)
        self.agent.failure = None
        return run_id, execution_id

    async def test_snapshot_is_saved_before_terminal_event_and_does_not_follow_later_state(self):
        self.agent.snapshot = {"final_answer": "首次成果", "report": None}
        run_id, execution_id = await self.start()
        await self.wait_finished(run_id)
        summary = self.history.rows[execution_id]["result_summary"]
        self.assertTrue(summary["available"])
        self.assertEqual(summary["markdown_chars"], 4)
        self.assertEqual(summary["render_mode"], "evidence_only")
        self.agent.snapshot = {"final_answer": "后来恢复产生了不同的成果"}
        self.assertEqual(self.history.rows[execution_id]["result_summary"], summary)
        self.assertEqual((await self.events(run_id))[-1][0], "completed")

    async def test_snapshot_failure_preserves_original_execution_failure(self):
        self.agent.failure = TimeoutError("原始模型超时")
        self.agent.snapshot_error = ConnectionError("检查点暂时不可用")
        with patch.object(self.snapshot_module.logger, "exception"):
            run_id, execution_id = await self.start()
            await self.wait_finished(run_id)
        row = self.history.rows[execution_id]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"], {"type": "TimeoutError", "message": "原始模型超时"})
        self.assertEqual(row["result_summary"]["reason"], "checkpoint_unavailable")
        self.assertTrue(row["records_complete"])

    async def test_repeated_cancel_during_snapshot_waits_for_snapshot_and_terminal_commit(self):
        self.agent.gate = asyncio.Event()
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.agent.aget_state

        async def blocked(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        with patch.object(self.agent, "aget_state", blocked):
            run_id, execution_id = await self.start()
            await self.agent.started.wait()
            request = asyncio.create_task(self.client.post(f"/research/{run_id}/cancel"))
            await entered.wait()
            self.api.runs[run_id].cancel()
            await asyncio.sleep(0)
            self.api.runs[run_id].cancel()
            release.set()
            result = await asyncio.wait_for(request, timeout=2)
        self.assertEqual(result.json()["status"], "cancelled")
        self.assertTrue(self.history.rows[execution_id]["result_summary"]["available"])

    async def test_resume_uses_database_without_memory_and_preserves_old_execution(self):
        run_id, old_id = await self.failed_start()
        old = await self.history.get_execution(old_id)
        self.api.run_queues.clear()
        self.api.runs.clear()
        self.api.run_execution_ids.clear()
        result = await self.client.post(f"/research/{run_id}/resume")
        self.assertEqual(result.status_code, 202, result.text)
        new_id = UUID(result.json()["execution_id"])
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(result.json()["run_id"], str(run_id))
        await self.wait_finished(run_id)
        self.assertEqual(await self.history.get_execution(old_id), old)
        new = await self.history.get_execution(new_id)
        self.assertEqual((new["mode"], new["status"]), ("resume", "completed"))
        self.assertIsNone(self.agent.inputs)
        self.assertEqual(self.agent.config["configurable"]["thread_id"], str(run_id))
        self.assertEqual(self.agent.config["metadata"]["execution_id"], str(new_id))
        self.assertEqual((await self.events(run_id))[-1][1]["execution_id"], str(new_id))
        status = (await self.client.get(f"/research/{run_id}")).json()
        self.assertEqual((status["mode"], status["execution_id"]), ("resume", str(new_id)))

    async def test_resume_missing_run_and_non_resumable_statuses(self):
        self.assertEqual(
            (await self.client.post(f"/research/{uuid4()}/resume")).status_code,
            404,
        )
        run_id, _ = await self.start()
        await self.wait_finished(run_id)
        self.assertEqual(
            (await self.client.post(f"/research/{run_id}/resume")).status_code,
            409,
        )
        run_id, execution_id = uuid4(), uuid4()
        await self.history.create_start_execution(run_id, execution_id, "未完成")
        for status in ("pending", "running"):
            self.history.rows[execution_id]["status"] = status
            self.assertEqual(
                (await self.client.post(f"/research/{run_id}/resume")).status_code,
                409,
            )

    async def test_resume_checks_saved_checkpoint_and_pending_nodes(self):
        run_id, _ = await self.failed_start()
        self.agent.checkpoint_id = None
        result = await self.client.post(f"/research/{run_id}/resume")
        self.assertEqual(result.status_code, 409)
        self.assertIn("没有已保存", result.json()["detail"])
        self.agent.checkpoint_id = "saved"
        self.agent.next_nodes = ()
        result = await self.client.post(f"/research/{run_id}/resume")
        self.assertEqual(result.status_code, 409)
        self.assertIn("没有待执行", result.json()["detail"])
        self.assertEqual(len(self.history.rows), 1)

    async def test_resume_checkpoint_failure_and_database_failure_do_not_create_task(self):
        run_id, old_id = await self.failed_start()
        old_task = self.api.runs[run_id]
        self.agent.snapshot_error = ConnectionError("检查点不可用")
        self.assertEqual(
            (await self.client.post(f"/research/{run_id}/resume")).status_code,
            503,
        )
        self.agent.snapshot_error = None
        self.history.create_error = ConnectionError("数据库不可用")
        self.assertEqual(
            (await self.client.post(f"/research/{run_id}/resume")).status_code,
            503,
        )
        self.assertIs(self.api.runs[run_id], old_task)
        self.assertEqual(list(self.history.rows), [old_id])

    async def test_resume_state_changed_during_checkpoint_read_returns_conflict(self):
        run_id, _ = await self.failed_start()

        async def conflict(*args):
            raise ExecutionConflictError("执行记录已变化")

        with patch.object(self.api, "create_resume_execution", conflict):
            result = await self.client.post(f"/research/{run_id}/resume")
        self.assertEqual(result.status_code, 409)
        self.assertEqual(len(self.history.rows), 1)

    async def test_concurrent_resume_accepts_only_one_and_creates_fresh_queue(self):
        run_id, _ = await self.failed_start()
        old_queue = self.api.run_queues[run_id]
        self.agent.gate = asyncio.Event()
        results = await asyncio.gather(
            *(self.client.post(f"/research/{run_id}/resume") for _ in range(2)),
        )
        self.assertEqual(sorted(result.status_code for result in results), [202, 409])
        self.assertEqual(len(self.history.rows), 2)
        self.assertIsNot(self.api.run_queues[run_id], old_queue)
        result = await self.client.post(f"/research/{run_id}/cancel")
        self.assertEqual(result.json()["status"], "cancelled")
        latest = await self.history.get_latest_execution(run_id)
        self.assertEqual((latest["mode"], latest["status"]), ("resume", "cancelled"))

    async def test_failed_resume_can_be_resumed_again(self):
        run_id, old_id = await self.failed_start()
        self.agent.failure = ValueError("恢复仍失败")
        result = await self.client.post(f"/research/{run_id}/resume")
        first_resume = UUID(result.json()["execution_id"])
        await self.wait_finished(run_id)
        self.assertEqual(self.history.rows[first_resume]["status"], "failed")
        self.agent.failure = None
        result = await self.client.post(f"/research/{run_id}/resume")
        second_resume = UUID(result.json()["execution_id"])
        await self.wait_finished(run_id)
        self.assertEqual(self.history.rows[second_resume]["status"], "completed")
        self.assertEqual(self.history.rows[old_id]["status"], "failed")
        self.assertEqual(len(self.history.rows), 3)

    async def test_interrupted_execution_can_resume_and_shutdown_cancels_new_execution(self):
        run_id, old_id = await self.failed_start()
        self.history.rows[old_id]["status"] = "interrupted"
        self.api.runs.clear()
        self.api.run_queues.clear()
        self.agent.gate = asyncio.Event()
        result = await self.client.post(f"/research/{run_id}/resume")
        new_id = UUID(result.json()["execution_id"])
        await self.api.shutdown_research_tasks()
        self.assertTrue(self.api.runs[run_id].done())
        self.assertEqual(self.history.rows[new_id]["status"], "cancelled")
        self.assertEqual(self.history.rows[old_id]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
