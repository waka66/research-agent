"""成果快照的去重、阶段成果与读取故障处理。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from research_agent.schemas.research import ResearchResult
from research_agent.services import result_snapshot as snapshots


def task_result(task_id, status="done", *, contested=False):
    return {
        "task_id": task_id,
        "points": [
            {
                "id": f"{task_id}-p0",
                "text": "待核实的问题",
                "status": status,
                "contested": contested,
            }
        ],
        "findings": [
            {
                "id": "shared-finding",
                "point_ids": [f"{task_id}-p0"],
                "claim": "可验证结论",
                "quote": "原始引文",
                "source_id": "s1",
                "stance": "supports",
            }
        ],
        "sources": [
            {
                "id": "s1",
                "url": f"https://example.com/doc?task={task_id}",
                "norm_url": "https://example.com/doc",
                "domain": "example.com",
            }
        ],
    }


class ResultSnapshotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        replacement = patch.object(snapshots.logger, "exception")
        replacement.start()
        self.addCleanup(replacement.stop)

    def test_deduplicates_tasks_findings_sources_and_preserves_quality_counts(self):
        values = {
            "research_plan": {
                "goal": "测试",
                "tasks": [
                    {"id": name, "question": "问题", "objective": "目标"}
                    for name in ("T1", "T2", "T3")
                ],
            },
            "research_results": [
                task_result("T1", "missing"),
                ResearchResult.model_validate(task_result("T1", "done")),
                task_result("T2", "thin", contested=True),
            ],
            "final_answer": "报告正文",
            "report": None,
        }
        summary = snapshots.build_result_summary(values, checkpoint_id="cp-1", graph_completed=True)
        self.assertEqual(summary["planned_task_count"], 3)
        self.assertEqual(summary["completed_task_count"], 2)
        self.assertEqual((summary["finding_count"], summary["source_count"]), (1, 1))
        self.assertEqual(
            summary["point_status_counts"], {"open": 0, "done": 1, "thin": 1, "missing": 0}
        )
        self.assertEqual(summary["contested_point_count"], 1)
        self.assertEqual(summary["markdown_chars"], 4)
        self.assertEqual(summary["render_mode"], "evidence_only")
        self.assertTrue(summary["graph_completed"])

    def test_partial_checkpoint_counts_only_completed_worker_results(self):
        summary = snapshots.build_result_summary({}, checkpoint_id="cp-1", graph_completed=False)
        self.assertEqual(summary["completed_task_count"], 0)
        self.assertEqual(summary["finding_count"], 0)
        self.assertIsNone(summary["planned_task_count"])
        self.assertFalse(summary["report_generated"])
        self.assertIsNone(summary["markdown_chars"])
        self.assertIsNone(summary["render_mode"])

    def test_structured_render_mode_and_empty_report_text_are_preserved(self):
        summary = snapshots.build_result_summary(
            {
                "final_answer": "",
                "report": {"title": "报告", "sections": [], "key_conclusions": []},
            },
            checkpoint_id="cp-1",
            graph_completed=True,
        )
        self.assertEqual(summary["render_mode"], "structured_report")
        self.assertTrue(summary["report_generated"])
        self.assertEqual(summary["markdown_chars"], 0)

    async def test_missing_checkpoint_is_unavailable_instead_of_zero_results(self):
        agent = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(config={})))
        summary = await snapshots.capture_result_summary(agent, uuid4())
        self.assertFalse(summary["available"])
        self.assertEqual(summary["reason"], "no_checkpoint")
        self.assertIsNone(summary["finding_count"])

    async def test_read_failure_does_not_escape_capture(self):
        agent = SimpleNamespace(aget_state=AsyncMock(side_effect=ConnectionError("检查点不可用")))
        summary = await snapshots.capture_result_summary(agent, uuid4())
        self.assertEqual(summary["reason"], "checkpoint_unavailable")
        self.assertIsNone(summary["report_generated"])

    async def test_invalid_snapshot_is_explicitly_unavailable(self):
        agent = SimpleNamespace(
            aget_state=AsyncMock(
                return_value=SimpleNamespace(
                    config={"configurable": {"checkpoint_id": "cp-1"}},
                    next=(),
                    values={"research_results": [{"task_id": "incomplete"}]},
                )
            )
        )
        summary = await snapshots.capture_result_summary(agent, uuid4())
        self.assertEqual(summary["reason"], "snapshot_invalid")

    async def test_capture_timeout_is_bounded(self):
        async def blocked(*args, **kwargs):
            await asyncio.Event().wait()

        with patch.object(snapshots, "SNAPSHOT_TIMEOUT_SECONDS", 0.01):
            summary = await snapshots.capture_result_summary(
                SimpleNamespace(aget_state=blocked), uuid4()
            )
        self.assertEqual(summary["reason"], "checkpoint_unavailable")

    async def test_no_agent_is_recorded_as_not_captured(self):
        summary = await snapshots.capture_result_summary(None, uuid4())
        self.assertEqual(summary["reason"], "not_captured")
