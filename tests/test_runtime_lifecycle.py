"""服务接管与关闭顺序：不访问真实数据库，也不加载模型。"""

import sys
import unittest
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from test_execution_lifecycle import load_file


class RuntimeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.lock_error = None
        self.setup_error = None

        @asynccontextmanager
        async def ownership(uri):
            if self.lock_error:
                raise self.lock_error
            self.events.append("lock_acquired")
            try:
                yield
            finally:
                self.events.append("lock_released")

        async def setup():
            self.events.append("checkpoint_setup")
            if self.setup_error:
                raise self.setup_error

        @asynccontextmanager
        async def checkpoints(uri):
            self.events.append("checkpoint_open")
            try:
                yield SimpleNamespace(setup=setup)
            finally:
                self.events.append("checkpoint_closed")

        async def interrupt():
            self.events.append("interrupt_old")
            return 0

        async def shutdown():
            self.events.append("tasks_stopped")

        async def dispose():
            self.events.append("engine_disposed")

        replacements = {}
        for name, fields in {
            "research_agent.config": {"settings": SimpleNamespace(db_uri="unused")},
            "research_agent.db.runtime": {"single_instance_runtime": ownership},
            "research_agent.db.history": {"interrupt_active_executions": interrupt},
            "research_agent.db.session": {"engine": SimpleNamespace(dispose=dispose)},
            "research_agent.graph.main_workflow": {"agent_builder": Mock()},
        }.items():
            module = ModuleType(name)
            module.__dict__.update(fields)
            replacements[name] = module
        with patch.dict(sys.modules, replacements):
            self.module = load_file("_runtime_lifecycle_test", "graph/agent.py")
        replacement = patch.object(
            self.module.AsyncPostgresSaver,
            "from_conn_string",
            checkpoints,
        )
        replacement.start()
        self.addCleanup(replacement.stop)
        self.app = FastAPI()
        self.app.state.shutdown_research = shutdown

    async def test_ownership_precedes_takeover_and_shutdown_precedes_unlock(self):
        async with self.module.lifespan(self.app):
            self.events.append("serving")
            self.assertIsNotNone(self.app.state.agent)
        self.assertEqual(
            self.events,
            [
                "lock_acquired",
                "checkpoint_open",
                "checkpoint_setup",
                "interrupt_old",
                "serving",
                "tasks_stopped",
                "checkpoint_closed",
                "lock_released",
                "engine_disposed",
            ],
        )

    async def test_second_instance_does_not_interrupt_existing_executions(self):
        self.lock_error = RuntimeError("已有实例")
        with self.assertRaisesRegex(RuntimeError, "已有实例"):
            async with self.module.lifespan(self.app):
                self.fail("不应开始服务")
        self.assertEqual(self.events, ["engine_disposed"])

    async def test_checkpoint_startup_failure_releases_resources_without_takeover(self):
        self.setup_error = ConnectionError("检查点不可用")
        with self.assertRaises(ConnectionError):
            async with self.module.lifespan(self.app):
                self.fail("不应开始服务")
        self.assertEqual(
            self.events,
            [
                "lock_acquired",
                "checkpoint_open",
                "checkpoint_setup",
                "checkpoint_closed",
                "lock_released",
                "engine_disposed",
            ],
        )

    async def test_shutdown_failure_still_closes_resources(self):
        self.app.state.shutdown_research = AsyncMock(side_effect=RuntimeError("收尾异常"))
        with self.assertRaisesRegex(RuntimeError, "收尾异常"):
            async with self.module.lifespan(self.app):
                pass
        self.assertEqual(
            self.events[-3:],
            [
                "checkpoint_closed",
                "lock_released",
                "engine_disposed",
            ],
        )
