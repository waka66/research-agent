"""可选 PostgreSQL 联调：真实历史表与 HTTP/SSE，使用模拟图，不调用模型。

先设置 RUN_POSTGRES_TESTS=1，再运行本文件对应的 unittest。
仅删除本测试生成的 UUID 对应的数据。
"""

import asyncio
import importlib
import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from test_execution_lifecycle import FakeAgent, load_file


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
        self.Run, self.Execution = models.ResearchRun, models.RunExecution
        self.engine = create_async_engine(session_module.database_url, poolclass=NullPool)
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
        try:
            for task in self.api.runs.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self.api.runs.values(), return_exceptions=True)
            await self.client.aclose()
            if self.run_ids:
                async with self.engine.begin() as connection:
                    await connection.execute(
                        delete(self.Execution).where(
                            self.Execution.run_id.in_(self.run_ids),
                        )
                    )
                    await connection.execute(
                        delete(self.Run).where(
                            self.Run.run_id.in_(self.run_ids),
                        )
                    )
        finally:
            await self.engine.dispose()

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


if __name__ == "__main__":
    unittest.main()
