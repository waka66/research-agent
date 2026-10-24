"""历史查询 HTTP 契约：分页、筛选、错误码；不访问数据库或检查点。"""

import base64
import json
import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI

from research_agent.api import executions as api
from research_agent.api.pagination import decode_cursor, encode_cursor


def execution_row(run_id, execution_id):
    return {
        "run_id": run_id,
        "execution_id": execution_id,
        "mode": "start",
        "status": "failed",
        "created_at": datetime.now(UTC),
        "started_at": None,
        "finished_at": None,
        "error": {"type": "TimeoutError", "message": "超时"},
        "records_complete": False,
    }


def record_row(execution_id, record_id):
    return {
        "record_id": record_id,
        "execution_id": execution_id,
        "parent_record_id": uuid4(),
        "kind": "tool",
        "name": "web_search_tool",
        "task_id": "T1",
        "round": 0,
        "status": "succeeded",
        "started_at": datetime.now(UTC),
        "finished_at": None,
        "duration_ms": 12.5,
        "payload": {"cache_status": "hit", "usage": None},
    }


class HistoryQueryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.run_id, self.execution_id = uuid4(), uuid4()
        self.execution = execution_row(self.run_id, self.execution_id)
        self.list_executions = AsyncMock(return_value=[])
        self.list_records = AsyncMock(return_value=[])
        for name, value in (
            ("list_executions", self.list_executions),
            ("list_records", self.list_records),
            ("get_execution", AsyncMock(return_value=self.execution)),
        ):
            replacement = patch.object(api, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        replacement = patch.object(api.logger, "exception")
        replacement.start()
        self.addCleanup(replacement.stop)
        app = FastAPI()  # 没有 agent、checkpointer、队列或后台任务。
        app.include_router(api.router)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_execution_page_returns_next_cursor_and_passes_position(self):
        first = execution_row(self.run_id, UUID(int=2))
        second = execution_row(self.run_id, UUID(int=1))
        self.list_executions.return_value = [first, second]
        path = f"/research/{self.run_id}/executions"
        response = await self.client.get(path, params={"limit": 1})
        self.assertEqual(response.status_code, 200, response.text)
        page = response.json()
        self.assertEqual(
            [row["execution_id"] for row in page["items"]], [str(first["execution_id"])]
        )
        self.assertFalse(page["items"][0]["records_complete"])
        self.assertEqual(page["items"][0]["error"]["message"], "超时")
        position = decode_cursor(page["next_cursor"], scope=f"executions:{self.run_id}", filters={})
        self.assertEqual(position.item_id, first["execution_id"])
        self.list_executions.return_value = [second]
        response = await self.client.get(path, params={"limit": 1, "cursor": page["next_cursor"]})
        self.assertIsNone(response.json()["next_cursor"])
        self.list_executions.assert_awaited_with(
            self.run_id,
            limit=1,
            before=(position.timestamp, position.item_id),
        )

    async def test_existing_empty_run_and_missing_run_are_distinct(self):
        path = f"/research/{self.run_id}/executions"
        response = await self.client.get(path)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [])
        self.assertIsNone(response.json()["next_cursor"])
        self.list_executions.return_value = None
        self.assertEqual((await self.client.get(path)).status_code, 404)

    async def test_record_page_preserves_payload_and_all_filters(self):
        records = [record_row(self.execution_id, uuid4()) for _ in range(2)]
        self.list_records.return_value = records
        path = f"/executions/{self.execution_id}/records"
        params = {
            "limit": 1,
            "kind": "tool",
            "status": "succeeded",
            "name": "web_search_tool",
            "task_id": "T1",
            "round": 0,
        }
        response = await self.client.get(path, params=params)
        self.assertEqual(response.status_code, 200, response.text)
        page = response.json()
        self.assertEqual(page["execution"]["run_id"], str(self.run_id))
        self.assertFalse(page["execution"]["records_complete"])
        self.assertEqual(page["items"][0]["payload"], {"cache_status": "hit", "usage": None})
        self.assertEqual(page["items"][0]["parent_record_id"], str(records[0]["parent_record_id"]))
        self.list_records.assert_awaited_with(
            self.execution_id,
            limit=1,
            after=None,
            kind="tool",
            status="succeeded",
            name="web_search_tool",
            task_id="T1",
            round_number=0,
        )
        self.list_records.return_value = records[1:]
        response = await self.client.get(path, params={**params, "cursor": page["next_cursor"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["next_cursor"])
        self.assertEqual(self.list_records.call_args.kwargs["after"][1], records[0]["record_id"])

    async def test_empty_records_keep_execution_metadata_and_missing_execution_is_404(self):
        path = f"/executions/{self.execution_id}/records"
        response = await self.client.get(path)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [])
        self.assertFalse(response.json()["execution"]["records_complete"])
        self.list_records.reset_mock()
        with patch.object(api, "get_execution", AsyncMock(return_value=None)):
            self.assertEqual((await self.client.get(path)).status_code, 404)
        self.list_records.assert_not_awaited()

    async def test_invalid_parameters_are_422_before_database_access(self):
        path = f"/executions/{self.execution_id}/records"
        for params in (
            {"limit": 0},
            {"limit": 101},
            {"kind": "llm"},
            {"status": "completed"},
            {"name": ""},
            {"task_id": ""},
            {"round": -1},
            {"cursor": "not a cursor"},
            {"cursor": ""},
            {"cursor": "x" * 2049},
        ):
            with self.subTest(params=params):
                self.assertEqual((await self.client.get(path, params=params)).status_code, 422)
        self.assertEqual((await self.client.get("/executions/not-a-uuid/records")).status_code, 422)
        self.list_records.assert_not_awaited()
        api.get_execution.assert_not_awaited()

    async def test_cursor_is_bound_to_scope_and_filters(self):
        path = f"/executions/{self.execution_id}/records"
        for scope, filters in (
            (f"records:{uuid4()}", {}),
            (f"executions:{self.execution_id}", {}),
            (f"records:{self.execution_id}", {"kind": "tool"}),
        ):
            cursor = encode_cursor(datetime.now(UTC), uuid4(), scope=scope, filters=filters)
            self.assertEqual(
                (await self.client.get(path, params={"cursor": cursor})).status_code, 422
            )
        api.get_execution.assert_not_awaited()

    async def test_naive_timestamp_or_missing_cursor_fields_are_rejected(self):
        path = f"/research/{self.run_id}/executions"
        for payload in (
            {
                "timestamp": "2026-10-24T12:00:00",
                "item_id": str(uuid4()),
                "scope": f"executions:{self.run_id}",
                "filters": {},
            },
            {"timestamp": datetime.now(UTC).isoformat()},
        ):
            cursor = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
            self.assertEqual(
                (await self.client.get(path, params={"cursor": cursor})).status_code, 422
            )
        self.list_executions.assert_not_awaited()

    async def test_database_errors_are_503_without_exposing_internal_details(self):
        self.list_executions.side_effect = ConnectionError("private database detail")
        result = await self.client.get(f"/research/{self.run_id}/executions")
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("private database detail", result.text)
        self.list_records.side_effect = ConnectionError("private database detail")
        result = await self.client.get(f"/executions/{self.execution_id}/records")
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("private database detail", result.text)
        self.list_records.assert_awaited_once()

    async def test_report_reads_complete_dataset_without_agent_or_page_limit(self):
        rows = [record_row(self.execution_id, uuid4()) for _ in range(101)]
        for row in rows:
            row["parent_record_id"] = None
        with patch.object(
            api, "load_execution_report", AsyncMock(return_value=(self.execution, rows))
        ) as loader:
            result = await self.client.get(f"/executions/{self.execution_id}/report")
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["records"]["tool_count"], 101)
        self.assertIsNone(result.json()["models"]["usage"])
        loader.assert_awaited_once_with(self.execution_id)

    async def test_report_missing_database_error_and_invalid_identity(self):
        path = f"/executions/{self.execution_id}/report"
        with patch.object(api, "load_execution_report", AsyncMock(return_value=None)):
            self.assertEqual((await self.client.get(path)).status_code, 404)
        with patch.object(
            api, "load_execution_report", AsyncMock(side_effect=ConnectionError("private detail"))
        ):
            result = await self.client.get(path)
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("private detail", result.text)
        self.assertEqual((await self.client.get("/executions/not-a-uuid/report")).status_code, 422)
