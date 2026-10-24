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
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
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

from research_agent.api.executions import router as history_router
from research_agent.db.errors import ExecutionConflictError
from research_agent.graph import observability as obs


@unittest.skipUnless(os.getenv("RUN_POSTGRES_TESTS") == "1", "需要显式开启 PostgreSQL 联调")
class PostgresExecutionTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def asyncSetUp(self):
        self.history = importlib.import_module("research_agent.db.history")
        self.record_store = importlib.import_module("research_agent.db.records")
        models = importlib.import_module("research_agent.db.models")
        session_module = importlib.import_module("research_agent.db.session")
        from research_agent.config import settings

        self.Run, self.Execution = models.ResearchRun, models.RunExecution
        self.Record = models.ExecutionRecord
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
        replacement = patch.object(self.record_store, "SessionFactory", self.sessions)
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
        app.include_router(history_router)
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

    async def saved_records(self, execution_id):
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(self.Record)
                    .where(self.Record.execution_id == execution_id)
                    .order_by(self.Record.started_at, self.Record.record_id),
                )
            ).all()
            return [
                {column.name: getattr(row, column.name) for column in self.Record.__table__.columns}
                for row in rows
            ]

    async def create_record_execution(self):
        run_id, execution_id = uuid4(), uuid4()
        self.run_ids.add(run_id)
        await self.history.create_start_execution(
            run_id, execution_id, "[integration-test] 过程记录"
        )
        await self.history.mark_execution_running(execution_id)
        config = {
            "configurable": {"thread_id": str(run_id)},
            "metadata": {"execution_id": str(execution_id)},
        }
        return run_id, execution_id, config

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
                    self.assertTrue(old["records_complete"])
                    self.assertTrue(old["result_summary"]["available"])
                    self.assertEqual(old["result_summary"]["completed_task_count"], 0)
                    old_report_response = await self.client.get(f"/executions/{old_id}/report")
                    self.assertEqual(old_report_response.status_code, 200, old_report_response.text)
                    old_report = old_report_response.json()
                    old_report.pop("as_of")
                    self.assertEqual(old_report["models"]["usage"]["total_tokens"], 40)
                    self.assertFalse(old_report["models"]["usage_complete"])
                    old_records = await self.saved_records(old_id)
                    self.assertTrue(old_records)
                    self.assertTrue(any(r["status"] == "failed" for r in old_records))
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
                    self.assertEqual(await self.saved_records(old_id), old_records)
                    old_again = (await self.client.get(f"/executions/{old_id}/report")).json()
                    old_again.pop("as_of")
                    self.assertEqual(old_again, old_report)
                    new_report_response = await self.client.get(f"/executions/{new_id}/report")
                    self.assertEqual(new_report_response.status_code, 200, new_report_response.text)
                    new_report = new_report_response.json()
                    self.assertEqual(new_report["models"]["usage"]["total_tokens"], 60)
                    self.assertTrue(new_report["models"]["usage_complete"])
                    summary = new_report["execution"]["result_summary"]
                    self.assertEqual(summary["completed_task_count"], 1)
                    self.assertEqual((summary["finding_count"], summary["source_count"]), (1, 1))
                    self.assertEqual(summary["render_mode"], "evidence_only")
                    self.assertTrue(summary["report_generated"])
                    new_records = await self.saved_records(new_id)
                    self.assertTrue(new["records_complete"])
                    self.assertEqual(
                        len(new_records),
                        len(
                            resumed := [
                                r for _, r in fixture.records if r["execution_id"] == str(new_id)
                            ]
                        ),
                    )
                    self.assertNotIn("search_node", [r["name"] for r in new_records])
                    parents = {r["record_id"]: r for r in new_records}
                    child = next(r for r in new_records if r["name"] == "extract_node")
                    self.assertEqual(parents[child["parent_record_id"]]["name"], "research_node")
                    self.assertEqual(child["payload"]["usage"]["total_tokens"], 20)
                    self.assertEqual(len(child["payload"]["attempts"]), 1)
                    snapshot = await self.service.research_status(self.agent, run_id)
                    self.assertEqual(snapshot.next, ())
                    self.assertIn("证据清单", snapshot.values["final_answer"])
                    self.assertEqual(len(snapshot.values["research_results"]), 1)
                    planner.assert_awaited_once()
                    worker_plan.assert_awaited_once()
                    search.assert_awaited_once()
                    self.assertEqual(extract.await_count, 2)
                    self.assertTrue(resumed)
                    self.assertNotIn("search_node", [r["node_name"] for r in resumed])
        finally:
            if run_id is not None:
                async with AsyncPostgresSaver.from_conn_string(self.checkpoint_uri) as saver:
                    await saver.adelete_thread(str(run_id))

    async def test_running_node_and_real_tool_cache_records_are_committed(self):
        from pathlib import Path

        from test_search_tool import tools

        _, execution_id, config = await self.create_record_execution()
        config["metadata"].update(research_task_id="T1", research_round=1)
        remote = AsyncMock(
            return_value={
                "results": [
                    {
                        "url": "https://example.com/doc",
                        "content": "测试资料",
                    }
                ]
            }
        )
        with (
            TemporaryDirectory() as directory,
            patch.object(tools.cache, "CACHE_DIR", Path(directory)),
            patch.object(tools.cache, "OFFLINE", False),
            patch.object(tools.client, "search", remote),
            patch.object(tools.logger, "log"),
            obs.recording_execution(execution_id, self.record_store) as context,
        ):
            async with obs.node_log(
                Mock(),
                "search_node",
                {"task": SimpleNamespace(id="T1")},
                config,
                round_number=1,
            ) as node:
                (running,) = await self.saved_records(execution_id)
                self.assertEqual(running["status"], "running")
                self.assertIsNone(running["finished_at"])
                await tools.web_search_tool.ainvoke("集成测试查询", config=config)
                await tools.web_search_tool.ainvoke("集成测试查询", config=config)
                node["output_summary"] = {"succeeded_query_count": 2}
            self.assertTrue(context.records_complete)
        rows = await self.saved_records(execution_id)
        self.assertEqual(len(rows), 3)
        root, first_tool, second_tool = rows
        self.assertEqual(root["payload"]["output_summary"], {"succeeded_query_count": 2})
        for item in (first_tool, second_tool):
            self.assertEqual(item["kind"], "tool")
            self.assertEqual(item["parent_record_id"], root["record_id"])
            self.assertEqual((item["task_id"], item["round"]), ("T1", 1))
            self.assertEqual(item["status"], "succeeded")
        self.assertEqual(first_tool["payload"]["cache_status"], "miss")
        self.assertTrue(first_tool["payload"]["remote_requested"])
        self.assertEqual(second_tool["payload"]["cache_status"], "hit")
        self.assertFalse(second_tool["payload"]["remote_requested"])
        remote.assert_awaited_once()
        await self.history.finish_execution(execution_id, "completed")

    async def test_record_write_failure_keeps_business_success_but_marks_incomplete(self):
        base_agent = self.agent

        class TracedAgent(FakeAgent):
            async def astream(inner, inputs, config, stream_mode):
                async with obs.node_log(Mock(), "Planner", {}, config) as record:
                    record["output_summary"] = {"task_count": 1}
                async for item in base_agent.astream(inputs, config, stream_mode):
                    yield item

        self.agent = TracedAgent()
        with (
            patch.object(
                self.record_store, "finish_record", side_effect=ConnectionError("模拟写入失败")
            ),
            patch.object(obs.persistence_logger, "exception"),
        ):
            run_id, execution_id = await self.start()
            await asyncio.wait_for(self.api.runs[run_id], timeout=10)
        row = await self.history.get_execution(execution_id)
        self.assertEqual(row["status"], "completed")
        self.assertFalse(row["records_complete"])
        (record,) = await self.saved_records(execution_id)
        self.assertEqual(record["status"], "unknown")
        self.assertIsNone(record["duration_ms"])
        self.assertIsNone(record["finished_at"])
        self.assertEqual(record["payload"]["error"]["type"], "RecordIncomplete")
        status = await self.client.get(f"/research/{run_id}")
        self.assertFalse(status.json()["records_complete"])
        self.assertIn(
            "event: completed", (await self.client.get(f"/research/{run_id}/stream")).text
        )
        # 重复结束不能把“不完整”恢复成 True。
        await self.history.finish_execution(execution_id, "completed")
        self.assertFalse((await self.history.get_execution(execution_id))["records_complete"])
        report = (await self.client.get(f"/executions/{execution_id}/report")).json()
        self.assertFalse(report["records"]["complete"])
        self.assertIsNone(report["models"]["usage"])

    async def test_cancelled_execution_has_cancelled_node_and_tool_records(self):
        entered = asyncio.Event()

        class WaitingAgent(FakeAgent):
            async def astream(inner, inputs, config, stream_mode):
                async with obs.node_log(Mock(), "search_node", {}, config):
                    async with obs.tool_log(Mock(), "web_search_tool", config):
                        entered.set()
                        await asyncio.Event().wait()
                yield {}  # 保持异步生成器接口；取消后不会执行。

        self.agent = WaitingAgent()
        run_id, execution_id = await self.start()
        await asyncio.wait_for(entered.wait(), timeout=10)
        result = await self.client.post(f"/research/{run_id}/cancel")
        self.assertEqual(result.json()["status"], "cancelled")
        records = await self.saved_records(execution_id)
        self.assertEqual([r["status"] for r in records], ["cancelled", "cancelled"])
        self.assertTrue(all(r["finished_at"] is not None for r in records))
        self.assertTrue((await self.history.get_execution(execution_id))["records_complete"])

    async def test_orphan_records_are_interrupted_without_invented_finish_time(self):
        from datetime import UTC, datetime

        _, execution_id, config = await self.create_record_execution()
        payload = {
            "record_id": str(uuid4()),
            "execution_id": str(execution_id),
            "parent_record_id": None,
            "kind": "node",
            "name": "Planner",
            "node_name": "Planner",
            "run_id": config["configurable"]["thread_id"],
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
            "duration_ms": None,
            "error": None,
        }
        await self.record_store.start_record(payload)
        await self.history.interrupt_active_executions(self.run_ids)
        (row,) = await self.saved_records(execution_id)
        self.assertEqual(row["status"], "interrupted")
        self.assertEqual(row["payload"]["status"], "interrupted")
        self.assertEqual(row["payload"]["error"]["type"], "ProcessInterrupted")
        report = (await self.client.get(f"/executions/{execution_id}/report")).json()
        self.assertIsNone(report["timing"]["duration_ms"])
        self.assertEqual(report["execution"]["result_summary"]["reason"], "process_interrupted")
        self.assertIn("observed_at", row["payload"])
        self.assertIsNone(row["finished_at"])
        self.assertIsNone(row["duration_ms"])
        self.assertFalse((await self.history.get_execution(execution_id))["records_complete"])
        await self.history.interrupt_active_executions(self.run_ids)
        self.assertEqual((await self.saved_records(execution_id))[0], row)

    async def test_parent_foreign_key_cannot_reference_another_execution(self):
        _, first_id, first_config = await self.create_record_execution()
        _, second_id, second_config = await self.create_record_execution()
        with obs.recording_execution(first_id, self.record_store):
            async with obs.node_log(Mock(), "Planner", {}, first_config) as first:
                pass
        # 取得完整合法 payload，只更改关联，让数据库检验跨 execution 的父节点。
        with obs.recording_execution(second_id, self.record_store):
            async with obs.node_log(Mock(), "Planner", {}, second_config) as second:
                pass
        bad = {
            **second,
            "record_id": str(uuid4()),
            "parent_record_id": first["record_id"],
            "status": "running",
            "finished_at": None,
            "duration_ms": None,
        }
        with self.assertRaises(IntegrityError):
            await self.record_store.start_record(bad)

    async def test_parallel_database_writes_preserve_branch_parents(self):
        _, execution_id, config = await self.create_record_execution()
        barrier = asyncio.Barrier(3)

        async def branch(task_id):
            async with obs.node_log(
                Mock(),
                "search_node",
                {"task": SimpleNamespace(id=task_id)},
                config,
            ) as node:
                await barrier.wait()
                tool_config = {
                    **config,
                    "metadata": {
                        **config["metadata"],
                        "research_task_id": task_id,
                    },
                }
                async with obs.tool_log(Mock(), "web_search_tool", tool_config) as tool:
                    self.assertEqual(tool["parent_record_id"], node["record_id"])

        with obs.recording_execution(execution_id, self.record_store) as context:
            async with obs.node_log(Mock(), "research_node", {}, config) as root:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(branch(task_id) for task_id in ("T1", "T2", "T3")),
                    ),
                    timeout=10,
                )
            self.assertTrue(context.records_complete)
        rows = await self.saved_records(execution_id)
        self.assertEqual(len(rows), 7)
        nodes = {r["task_id"]: r for r in rows if r["name"] == "search_node"}
        for item in rows:
            if item["name"] == "search_node":
                self.assertEqual(item["parent_record_id"], UUID(root["record_id"]))
            if item["kind"] == "tool":
                self.assertEqual(item["parent_record_id"], nodes[item["task_id"]]["record_id"])
        await self.history.finish_execution(execution_id, "completed")

    async def test_execution_history_keyset_survives_new_resume_between_pages(self):
        from datetime import UTC, datetime

        run_id = uuid4()
        self.run_ids.add(run_id)
        first_id, second_id, third_id = UUID(int=1), UUID(int=2), UUID(int=3)
        await self.history.create_start_execution(run_id, first_id, "[integration-test] 历史分页")
        await self.history.finish_execution(first_id, "failed", {"message": "首次失败"})
        await self.history.create_resume_execution(run_id, second_id, first_id)
        await self.history.finish_execution(second_id, "cancelled")
        # 人为构造同一时间，验证 UUID 是稳定的第二排序键。
        async with self.sessions.begin() as session:
            for execution_id in (first_id, second_id):
                row = await session.get(self.Execution, execution_id)
                row.created_at = datetime(2020, 1, 1, tzinfo=UTC)
        self.agent.snapshot_error = AssertionError("历史接口不能读取 checkpoint")
        self.api.runs.clear()
        self.api.run_queues.clear()
        path = f"/research/{run_id}/executions"
        result = await self.client.get(path, params={"limit": 1})
        self.assertEqual(result.status_code, 200, result.text)
        page = result.json()
        self.assertEqual([r["execution_id"] for r in page["items"]], [str(second_id)])
        self.assertEqual(page["items"][0]["mode"], "resume")
        self.assertEqual(page["items"][0]["status"], "cancelled")
        await self.history.create_resume_execution(run_id, third_id, second_id)
        result = await self.client.get(path, params={"limit": 1, "cursor": page["next_cursor"]})
        self.assertEqual(result.status_code, 200, result.text)
        page = result.json()
        self.assertEqual([r["execution_id"] for r in page["items"]], [str(first_id)])
        self.assertEqual(page["items"][0]["error"]["message"], "首次失败")
        self.assertIsNone(page["next_cursor"])
        fresh = (await self.client.get(path)).json()
        self.assertEqual(
            [r["execution_id"] for r in fresh["items"]],
            [
                str(third_id),
                str(second_id),
                str(first_id),
            ],
        )
        self.assertEqual(fresh["items"][0]["status"], "pending")

    async def test_record_query_paginates_ties_filters_and_isolates_executions(self):
        from datetime import UTC, datetime

        _, execution_id, config = await self.create_record_execution()
        config["metadata"].update(research_task_id="T1", research_round=0)
        with obs.recording_execution(execution_id, self.record_store):
            async with obs.node_log(
                Mock(),
                "search_node",
                {"task": SimpleNamespace(id="T1")},
                config,
                round_number=0,
            ):
                for _ in range(2):
                    async with obs.tool_log(Mock(), "web_search_tool", config) as record:
                        record.update(cache_status="hit", remote_requested=False, result_count=1)
            with self.assertRaises(TimeoutError):
                async with obs.node_log(
                    Mock(),
                    "extract_node",
                    {"task": SimpleNamespace(id="T2")},
                    config,
                ):
                    raise TimeoutError("[integration-test] 原始提取超时")
        _, other_execution, other_config = await self.create_record_execution()
        with obs.recording_execution(other_execution, self.record_store):
            async with obs.node_log(Mock(), "Planner", {}, other_config):
                pass
        fixed_time = datetime(2020, 1, 1, tzinfo=UTC)
        async with self.sessions.begin() as session:
            rows = (
                await session.scalars(
                    select(self.Record).where(self.Record.execution_id == execution_id),
                )
            ).all()
            for row in rows:
                row.started_at = fixed_time
                row.payload = {**row.payload, "started_at": fixed_time.isoformat()}
        all_rows = await self.saved_records(execution_id)
        expected = [str(r["record_id"]) for r in all_rows]
        path = f"/executions/{execution_id}/records"
        self.agent.snapshot_error = AssertionError("历史接口不能读取 checkpoint")
        first = await self.client.get(path, params={"limit": 2})
        self.assertEqual(first.status_code, 200, first.text)
        first_page = first.json()
        second = await self.client.get(
            path, params={"limit": 2, "cursor": first_page["next_cursor"]}
        )
        self.assertEqual(second.status_code, 200, second.text)
        second_page = second.json()
        self.assertEqual(
            [r["record_id"] for r in first_page["items"] + second_page["items"]],
            expected,
        )
        self.assertIsNone(second_page["next_cursor"])
        self.assertEqual(len(expected), 4)
        filters = {
            "kind": "tool",
            "status": "succeeded",
            "name": "web_search_tool",
            "task_id": "T1",
            "round": 0,
            "limit": 1,
        }
        page = (await self.client.get(path, params=filters)).json()
        tool = page["items"][0]
        self.assertEqual(tool["payload"]["cache_status"], "hit")
        self.assertFalse(tool["payload"]["remote_requested"])
        self.assertIsNotNone(tool["parent_record_id"])
        self.assertEqual(tool["round"], 0)
        next_page = (
            await self.client.get(
                path,
                params={**filters, "cursor": page["next_cursor"]},
            )
        ).json()
        self.assertIsNone(next_page["next_cursor"])
        self.assertNotEqual(next_page["items"][0]["record_id"], tool["record_id"])
        failure = (await self.client.get(path, params={"status": "failed"})).json()
        self.assertEqual(len(failure["items"]), 1)
        self.assertEqual(
            failure["items"][0]["payload"]["error"]["message"], "[integration-test] 原始提取超时"
        )

    async def test_history_empty_unknown_and_missing_record_states(self):
        self.assertEqual(
            (await self.client.get(f"/research/{uuid4()}/executions")).status_code,
            404,
        )
        self.assertEqual(
            (await self.client.get(f"/executions/{uuid4()}/records")).status_code,
            404,
        )
        empty_run = uuid4()
        self.run_ids.add(empty_run)
        async with self.sessions.begin() as session:
            session.add(self.Run(run_id=empty_run, question="[integration-test] 无执行的研究"))
        empty = await self.client.get(f"/research/{empty_run}/executions")
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json()["items"], [])
        _, execution_id, _ = await self.create_record_execution()
        async with self.sessions.begin() as session:
            execution = await session.get(self.Execution, execution_id)
            execution.records_complete = False
        result = await self.client.get(f"/executions/{execution_id}/records")
        self.assertEqual(result.status_code, 200, result.text)
        page = result.json()
        self.assertEqual(page["items"], [])
        self.assertIsNone(page["next_cursor"])
        self.assertFalse(page["execution"]["records_complete"])

    async def test_report_reads_all_records_and_saved_snapshot_without_live_graph(self):
        from datetime import UTC, datetime, timedelta

        from research_agent.services.result_snapshot import build_result_summary

        _, execution_id, _ = await self.create_record_execution()
        _, other_id, _ = await self.create_record_execution()
        instant = datetime.now(UTC)
        async with self.sessions.begin() as session:
            for scope, count, token_count in ((execution_id, 105, 3), (other_id, 1, 999)):
                for _ in range(count):
                    record_id = uuid4()
                    tokens = {
                        "input_tokens": 1,
                        "output_tokens": token_count - 1,
                        "total_tokens": token_count,
                    }
                    session.add(
                        self.Record(
                            record_id=record_id,
                            execution_id=scope,
                            parent_record_id=None,
                            kind="node",
                            name="Planner",
                            status="succeeded",
                            started_at=instant,
                            finished_at=instant + timedelta(milliseconds=1),
                            duration_ms=1,
                            payload={
                                "record_id": str(record_id),
                                "execution_id": str(scope),
                                "attempts": [
                                    {"attempt": 1, "status": "succeeded", "usage": tokens}
                                ],
                                "usage": tokens,
                                "usage_complete": True,
                            },
                        )
                    )
        summary = build_result_summary(
            {"final_answer": "首次正文"},
            checkpoint_id="first-checkpoint",
            graph_completed=True,
        )
        await self.history.finish_execution(execution_id, "completed", result_summary=summary)
        before = await self.history.get_execution(execution_id)
        later_summary = build_result_summary(
            {"final_answer": "后来不同的正文"},
            checkpoint_id="later-checkpoint",
            graph_completed=True,
        )
        self.assertFalse(
            await self.history.finish_execution(
                execution_id,
                "completed",
                result_summary=later_summary,
            )
        )
        self.assertEqual(await self.history.get_execution(execution_id), before)
        self.api.runs.clear()
        self.api.run_queues.clear()
        self.agent.snapshot_error = AssertionError("历史报告不能读取 checkpoint")
        page = await self.client.get(f"/executions/{execution_id}/records", params={"limit": 1})
        self.assertEqual(len(page.json()["items"]), 1)
        result = await self.client.get(f"/executions/{execution_id}/report")
        self.assertEqual(result.status_code, 200, result.text)
        report = result.json()
        self.assertEqual(report["records"]["node_count"], 105)
        self.assertEqual(report["models"]["call_count"], 105)
        self.assertEqual(report["models"]["usage"]["total_tokens"], 315)
        self.assertTrue(report["models"]["usage_complete"])
        self.assertEqual(report["execution"]["result_summary"], summary)
        self.assertEqual(await self.history.get_execution(execution_id), before)

    async def test_result_summary_and_terminal_status_roll_back_together(self):
        from research_agent.services.result_snapshot import build_result_summary

        _, execution_id, _ = await self.create_record_execution()
        summary = build_result_summary({}, checkpoint_id="cp-test", graph_completed=False)
        with patch.object(
            self.history, "close_open_records", side_effect=ConnectionError("事务失败")
        ):
            with self.assertRaises(ConnectionError):
                await self.history.finish_execution(
                    execution_id, "completed", result_summary=summary
                )
        row = await self.history.get_execution(execution_id)
        self.assertEqual(row["status"], "running")
        self.assertIsNone(row["result_summary"])
        self.assertIsNone(row["finished_at"])
        await self.history.finish_execution(execution_id, "completed", result_summary=summary)
        self.assertEqual(
            (await self.history.get_execution(execution_id))["result_summary"], summary
        )

    async def test_snapshot_failure_does_not_replace_business_outcome(self):
        from research_agent.services import result_snapshot

        self.agent.snapshot_error = ConnectionError("模拟检查点故障")
        with patch.object(result_snapshot.logger, "exception"):
            self.agent.failure = TimeoutError("原始模型错误")
            run_id, failed_id = await self.start()
            await asyncio.wait_for(self.api.runs[run_id], timeout=10)
            self.agent.failure = None
            run_id, success_id = await self.start()
            await asyncio.wait_for(self.api.runs[run_id], timeout=10)
        failed = await self.history.get_execution(failed_id)
        success = await self.history.get_execution(success_id)
        self.assertEqual(failed["error"], {"type": "TimeoutError", "message": "原始模型错误"})
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(success["status"], "completed")
        for row in (failed, success):
            self.assertEqual(row["result_summary"]["reason"], "checkpoint_unavailable")
            self.assertFalse(row["result_summary"]["available"])
            response = await self.client.get(f"/executions/{row['execution_id']}/report")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["is_final"])


if __name__ == "__main__":
    unittest.main()
