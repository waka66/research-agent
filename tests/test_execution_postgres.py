"""可选 PostgreSQL 联调：真实历史表与 HTTP/SSE，使用模拟图，不调用模型。

先设置 RUN_POSTGRES_TESTS=1，再运行本文件对应的 unittest。
每项测试使用独立随机 schema，结束后只删除该 schema。
"""

import asyncio
import importlib
import os
import re
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI
from psycopg.conninfo import make_conninfo
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema
from test_execution_lifecycle import FakeAgent, load_file

from research_agent.db.errors import ExecutionConflictError


@unittest.skipUnless(os.getenv("RUN_POSTGRES_TESTS") == "1", "需要显式开启 PostgreSQL 联调")
class PostgresExecutionTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def asyncSetUp(self):
        self.history = importlib.import_module("research_agent.db.history")
        models = importlib.import_module("research_agent.db.models")
        session_module = importlib.import_module("research_agent.db.session")
        from research_agent.config import settings

        self.Run, self.Execution = models.ResearchRun, models.RunExecution
        self.schema = f"test_execution_{uuid4().hex}"
        self.admin_engine = create_async_engine(session_module.database_url, poolclass=NullPool)
        self.engine = create_async_engine(
            session_module.database_url,
            poolclass=NullPool,
            connect_args={"options": f"-csearch_path={self.schema}"},
        )
        self.checkpoint_uri = make_conninfo(
            settings.db_uri,
            options=f"-csearch_path={self.schema}",
        )
        self.addAsyncCleanup(self.cleanup_database)
        async with self.admin_engine.begin() as connection:
            await connection.execute(CreateSchema(self.schema))
        async with self.engine.begin() as connection:
            await connection.run_sync(models.Base.metadata.create_all)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        replacement = patch.object(self.history, "SessionFactory", self.sessions)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.run_ids = set()
        self.agent = FakeAgent()
        history_module = ModuleType("research_agent.db.history")

        async def tracked_create(run_id, execution_id, question):
            self.run_ids.add(run_id)
            await self.history.create_start_execution(run_id, execution_id, question)

        history_module.create_start_execution = tracked_create
        for name in (
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
            self.service = load_file("_postgres_service_test", "services/research_service.py")
            with patch.dict(
                sys.modules,
                {
                    "research_agent.services.research_service": self.service,
                },
            ):
                self.api = load_file("_postgres_api_test", "api/research.py")
        app = FastAPI()
        app.include_router(self.api.router)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        replacement = patch.object(self.service.logger, "exception")
        replacement.start()
        self.addCleanup(replacement.stop)

    async def asyncTearDown(self):
        await self.api.shutdown_research_tasks()
        await asyncio.gather(*self.api.runs.values(), return_exceptions=True)
        await self.client.aclose()

    async def cleanup_database(self):
        # 名称由本测试生成，必须匹配固定格式；不触及 public 或用户 schema。
        if not re.fullmatch(r"test_execution_[0-9a-f]{32}", self.schema):
            raise ValueError("拒绝清理非测试 schema")
        try:
            await self.engine.dispose()
            async with self.admin_engine.begin() as connection:
                await connection.execute(DropSchema(self.schema, cascade=True, if_exists=True))
        finally:
            await self.admin_engine.dispose()

    async def start(self):
        result = await self.client.post(
            "/research/", json={"question": "[integration-test] 执行生命周期"}
        )
        self.assertEqual(result.status_code, 202, result.text)
        payload = result.json()
        return UUID(payload["run_id"]), UUID(payload["execution_id"])

    async def test_completed_is_committed_and_readable_through_http(self):
        run_id, execution_id = await self.start()
        await asyncio.wait_for(self.api.runs[run_id], timeout=10)
        row = await self.history.get_execution(execution_id)
        self.assertEqual(row["status"], "completed")
        self.assertIsNotNone(row["started_at"])
        self.assertGreaterEqual(row["finished_at"], row["started_at"])
        status = await self.client.get(f"/research/{run_id}")
        self.assertEqual(status.status_code, 200, status.text)
        self.assertEqual(status.json()["execution_id"], str(execution_id))
        stream = await self.client.get(f"/research/{run_id}/stream")
        self.assertIn("event: completed", stream.text)
        self.assertIn('"final_answer"', stream.text)

    async def test_failure_error_round_trips_jsonb(self):
        self.agent.failure = TimeoutError("模拟模型超时")
        run_id, execution_id = await self.start()
        await asyncio.wait_for(self.api.runs[run_id], timeout=10)
        row = await self.history.get_execution(execution_id)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"], {"type": "TimeoutError", "message": "模拟模型超时"})
        stream = await self.client.get(f"/research/{run_id}/stream")
        self.assertIn("event: failed", stream.text)
        self.assertNotIn("event: completed", stream.text)

    async def test_cancelled_is_committed_and_duplicate_finish_is_idempotent(self):
        self.agent.gate = asyncio.Event()
        run_id, execution_id = await self.start()
        await asyncio.wait_for(self.agent.started.wait(), timeout=10)
        result = await self.client.post(f"/research/{run_id}/cancel")
        self.assertEqual(result.status_code, 200, result.text)
        row = await self.history.get_execution(execution_id)
        self.assertEqual(row["status"], "cancelled")
        self.assertFalse(await self.history.finish_execution(execution_id, "cancelled"))
        self.assertEqual(
            (await self.history.get_execution(execution_id))["finished_at"], row["finished_at"]
        )

    async def test_start_inserts_roll_back_together(self):
        first_run, second_run, execution_id = uuid4(), uuid4(), uuid4()
        self.run_ids.update({first_run, second_run})
        await self.history.create_start_execution(
            first_run, execution_id, "[integration-test] 原记录"
        )
        with self.assertRaises(IntegrityError):
            await self.history.create_start_execution(
                second_run, execution_id, "[integration-test] 回滚验证"
            )
        async with self.sessions() as session:
            self.assertIsNone(await session.get(self.Run, second_run))
            rows = await session.scalars(
                select(self.Execution).where(self.Execution.run_id == first_run)
            )
            self.assertEqual(len(rows.all()), 1)

    async def failed_start(self):
        self.agent.failure = TimeoutError("[integration-test] 首次失败")
        run_id, execution_id = await self.start()
        await asyncio.wait_for(self.api.runs[run_id], timeout=10)
        self.agent.failure = None
        return run_id, execution_id

    async def test_concurrent_http_resume_is_serialized_by_database(self):
        run_id, old_id = await self.failed_start()
        old = await self.history.get_execution(old_id)
        arrived, ready = 0, asyncio.Event()

        async def simultaneous_checkpoints(*args):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=10)
            return await self.agent.aget_state({}, False)

        self.agent.gate = asyncio.Event()
        with patch.object(self.api, "research_status", simultaneous_checkpoints):
            responses = await asyncio.gather(
                *(self.client.post(f"/research/{run_id}/resume") for _ in range(2)),
            )
        self.assertEqual(sorted(result.status_code for result in responses), [202, 409])
        await self.client.post(f"/research/{run_id}/cancel")
        self.assertEqual(await self.history.get_execution(old_id), old)
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(self.Execution).where(self.Execution.run_id == run_id),
                )
            ).all()
            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(row.mode for row in rows), ["resume", "start"])

    async def test_unique_index_rejects_active_execution_even_without_application_lock(self):
        run_id, execution_id = uuid4(), uuid4()
        self.run_ids.add(run_id)
        await self.history.create_start_execution(run_id, execution_id, "[integration-test] 约束")
        with self.assertRaises(IntegrityError):
            async with self.sessions.begin() as session:
                session.add(
                    self.Execution(
                        run_id=run_id,
                        execution_id=uuid4(),
                        mode="resume",
                        status="pending",
                    )
                )
        await self.history.finish_execution(execution_id, "cancelled")
        resume_id = uuid4()
        await self.history.create_resume_execution(run_id, resume_id, execution_id)
        await self.history.finish_execution(resume_id, "cancelled")
        # 即使第一个恢复已很快结束，拿旧 execution 校验的并发请求也不能再创建一次。
        with self.assertRaises(ExecutionConflictError):
            await self.history.create_resume_execution(run_id, uuid4(), execution_id)

    async def test_restart_interruption_preserves_terminal_rows_and_allows_resume(self):
        run_id, execution_id = uuid4(), uuid4()
        self.run_ids.add(run_id)
        await self.history.create_start_execution(
            run_id,
            execution_id,
            "[integration-test] 进程中断",
        )
        await self.history.mark_execution_running(execution_id)
        completed_run, completed_id = await self.start()
        await asyncio.wait_for(self.api.runs[completed_run], timeout=10)
        before = await self.history.get_execution(completed_id)
        # 测试严格限定自身 UUID，不接管用户真实运行的记录。
        count = await self.history.interrupt_active_executions(self.run_ids)
        self.assertEqual(count, 1)
        interrupted = await self.history.get_execution(execution_id)
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(interrupted["error"]["type"], "ProcessInterrupted")
        self.assertIsNotNone(interrupted["finished_at"])
        self.assertEqual(await self.history.get_execution(completed_id), before)
        result = await self.client.post(f"/research/{run_id}/resume")
        self.assertEqual(result.status_code, 202, result.text)
        await asyncio.wait_for(self.api.runs[run_id], timeout=10)
        self.assertEqual(await self.history.get_execution(execution_id), interrupted)

    async def test_runtime_lock_prevents_second_instance_and_releases_on_exit(self):
        from research_agent.config import settings
        from research_agent.db.runtime import single_instance_runtime

        # 独立测试锁，不与正在运行的应用争抢服务锁。
        key = uuid4().int % (2**63)
        async with single_instance_runtime(settings.db_uri, lock_key=key):
            with self.assertRaisesRegex(RuntimeError, "单实例"):
                async with single_instance_runtime(settings.db_uri, lock_key=key):
                    self.fail("第二个实例不应获得同一把锁")
        async with single_instance_runtime(settings.db_uri, lock_key=key):
            pass

    async def test_real_parent_and_child_graph_resume_after_recreating_checkpointer(self):
        """真实 workflow + PG 检查点；搜索已成功、提取超时，恢复不得重新搜索。"""
        import hashlib

        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from test_node_observability import (
            NodeObservabilityTests,
            main,
            response,
            worker,
        )
        from test_node_observability import load_file as load_graph

        from research_agent.schemas.research import ResearchPlan, ResearchWorkerPlan

        fixture = NodeObservabilityTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        planner = AsyncMock(return_value=response(fixture.main_state["research_plan"]))
        worker_plan = AsyncMock(
            return_value=response(
                ResearchWorkerPlan(
                    points=[{"text": fixture.point.text, "kind": "official_fact"}],
                    queries=[{"text": "查询", "target_indices": [0]}],
                )
            )
        )
        source_id = hashlib.sha1(fixture.source.norm_url.encode()).hexdigest()[:12]
        extract = AsyncMock(
            side_effect=[
                TimeoutError("[integration-test] 提取超时"),
                response(
                    SimpleNamespace(
                        findings=[
                            SimpleNamespace(
                                **{**vars(fixture.raw_finding), "result_id": source_id},
                            )
                        ]
                    )
                ),
            ]
        )

        def model_for(schema, **kwargs):
            if schema is ResearchPlan:
                return SimpleNamespace(ainvoke=planner)
            if schema is ResearchWorkerPlan:
                return SimpleNamespace(ainvoke=worker_plan)
            return SimpleNamespace(ainvoke=extract)

        worker.llm.with_structured_output.side_effect = model_for
        main.llm_synth.with_structured_output.return_value = SimpleNamespace(
            ainvoke=AsyncMock(return_value=response(None, error=ValueError("模拟降级"))),
        )
        with patch.dict(sys.modules, {"research_agent.graph.worker_nodes": worker}):
            child = load_graph("worker_workflow")
        with patch.dict(sys.modules, {"research_agent.graph.main_nodes": main}):
            workflow = load_graph("main_workflow")
        search = AsyncMock(
            return_value={
                "search_result": [
                    {
                        "url": fixture.source.url,
                        "title": fixture.source.title,
                        "summary": fixture.source.summary,
                    }
                ]
            }
        )
        run_id = None
        try:
            with (
                patch.object(main, "worker_graph", child.worker_graph),
                patch.object(worker.web_search_tool, "ainvoke", search),
            ):
                async with AsyncPostgresSaver.from_conn_string(self.checkpoint_uri) as saver:
                    await saver.setup()
                    self.agent = workflow.agent_builder.compile(checkpointer=saver)
                    run_id, old_id = await self.start()
                    await asyncio.wait_for(self.api.runs[run_id], timeout=10)
                    old = await self.history.get_execution(old_id)
                    self.assertEqual(old["status"], "failed")
                    self.assertEqual(old["error"]["type"], "WorkerFatalError")
                    snapshot = await self.service.research_status(self.agent, run_id)
                    self.assertEqual(snapshot.next, ("research_node",))
                self.api.runs.clear()
                self.api.run_queues.clear()
                self.api.run_execution_ids.clear()
                # 新连接、新编译的图，依靠 PG 恢复，不依靠原图/事件队列内存。
                async with AsyncPostgresSaver.from_conn_string(self.checkpoint_uri) as saver:
                    self.agent = workflow.agent_builder.compile(checkpointer=saver)
                    result = await self.client.post(f"/research/{run_id}/resume")
                    self.assertEqual(result.status_code, 202, result.text)
                    new_id = UUID(result.json()["execution_id"])
                    await asyncio.wait_for(self.api.runs[run_id], timeout=10)
                    new = await self.history.get_execution(new_id)
                    self.assertEqual((new["mode"], new["status"]), ("resume", "completed"))
                    self.assertEqual(await self.history.get_execution(old_id), old)
                    snapshot = await self.service.research_status(self.agent, run_id)
                    self.assertEqual(snapshot.next, ())
                    self.assertIn("证据清单", snapshot.values["final_answer"])
                    self.assertEqual(len(snapshot.values["research_results"]), 1)
                    planner.assert_awaited_once()
                    worker_plan.assert_awaited_once()
                    search.assert_awaited_once()
                    self.assertEqual(extract.await_count, 2)
                    resumed_logs = [
                        record
                        for _, record in fixture.records
                        if record["execution_id"] == str(new_id)
                    ]
                    self.assertTrue(resumed_logs)
                    self.assertNotIn("search_node", [r["node_name"] for r in resumed_logs])
        finally:
            if run_id is not None:
                async with AsyncPostgresSaver.from_conn_string(self.checkpoint_uri) as saver:
                    await saver.adelete_thread(str(run_id))


if __name__ == "__main__":
    unittest.main()
