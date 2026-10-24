"""报告统计使用可人工核算的样本，不访问任何外部服务。"""

import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from research_agent.services.execution_report import build_execution_report

NOW = datetime(2026, 10, 24, tzinfo=UTC)


def usage(input_tokens, output_tokens):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class ExecutionReportTests(unittest.TestCase):
    def setUp(self):
        self.execution = {
            "execution_id": uuid4(),
            "run_id": uuid4(),
            "mode": "start",
            "status": "completed",
            "created_at": NOW,
            "started_at": NOW,
            "finished_at": NOW + timedelta(seconds=3),
            "error": None,
            "records_complete": True,
            "result_summary": None,
        }

    def record(
        self,
        name="Planner",
        *,
        kind="node",
        parent=None,
        status="succeeded",
        duration=100,
        payload=None,
    ):
        return {
            "record_id": uuid4(),
            "execution_id": self.execution["execution_id"],
            "parent_record_id": parent,
            "kind": kind,
            "name": name,
            "task_id": None,
            "round": None,
            "status": status,
            "started_at": NOW,
            "finished_at": NOW + timedelta(milliseconds=100),
            "duration_ms": duration,
            "payload": payload or {},
        }

    def model(self, attempts, *, complete=True, total=None, **kwargs):
        return self.record(
            payload={
                "attempts": [
                    {"attempt": number, **item} for number, item in enumerate(attempts, 1)
                ],
                "usage": total,
                "usage_complete": complete,
            },
            **kwargs,
        )

    def report(self, records):
        return build_execution_report(self.execution, records, now=NOW + timedelta(seconds=5))

    def test_retry_usage_is_counted_once_and_parent_rollup_is_excluded(self):
        parent = self.record("research_node", payload={"usage": usage(999, 999)})
        child = self.model(
            [
                {"status": "failed", "usage": usage(10, 5), "error": {"message": "解析失败"}},
                {"status": "succeeded", "usage": usage(10, 5)},
            ],
            total=usage(20, 10),
            parent=parent["record_id"],
        )
        result = self.report([parent, child])
        self.assertEqual(result.models.usage.total_tokens, 30)
        self.assertEqual((result.models.call_count, result.models.retry_count), (2, 1))
        self.assertEqual(result.models.failed_attempt_count, 1)
        self.assertTrue(result.models.usage_complete)
        self.assertEqual(result.issues[0].status, "succeeded")
        self.assertEqual(len(result.issues[0].failed_attempts), 1)

    def test_failed_unknown_attempt_does_not_turn_into_zero_consumption(self):
        node = self.model(
            [
                {"status": "failed", "usage": None},
                {"status": "succeeded", "usage": usage(10, 20)},
            ],
            total=usage(10, 20),
            complete=False,
        )
        result = self.report([node])
        self.assertEqual(result.models.usage.total_tokens, 30)
        self.assertEqual(result.models.missing_usage_attempts, 1)
        self.assertFalse(result.models.usage_complete)
        unknown = self.model([{"status": "failed", "usage": None}], status="failed", complete=False)
        result = self.report([unknown])
        self.assertIsNone(result.models.usage)
        self.assertEqual(result.models.call_count, 1)

    def test_missing_records_make_otherwise_known_usage_incomplete(self):
        self.execution["records_complete"] = False
        node = self.model([{"status": "succeeded", "usage": usage(10, 20)}], total=usage(10, 20))
        result = self.report([node])
        self.assertFalse(result.records.complete)
        self.assertFalse(result.models.usage_complete)
        self.assertEqual(result.models.usage.total_tokens, 30)
        self.assertEqual(result.models.missing_usage_attempts, 0)
        self.assertIsNone(self.report([]).models.usage)

    def test_no_model_calls_is_zero_only_when_final_records_are_complete(self):
        result = self.report([self.record("generator")])
        self.assertEqual(result.models.usage.total_tokens, 0)
        self.assertTrue(result.models.usage_complete)
        self.execution.update(status="running", finished_at=None)
        result = self.report([self.record(status="running", duration=None)])
        self.assertIsNone(result.models.usage)
        self.assertFalse(result.is_final)
        self.assertEqual(result.timing.elapsed_ms, 5000)
        self.assertIsNone(result.timing.duration_ms)

    def test_nested_and_parallel_durations_are_not_added_to_execution_time(self):
        parent = self.record("research_node", duration=3000)
        children = [
            self.record("search_node", parent=parent["record_id"], duration=value)
            for value in (2000, 1500)
        ]
        result = self.report([parent, *children])
        self.assertEqual(result.timing.duration_ms, 3000)
        group = next(row for row in result.timing.node_timings if row.name == "search_node")
        self.assertEqual(group.inclusive_duration_ms_sum, 3500)
        self.assertEqual((group.call_count, group.measured_count), (2, 2))
        self.assertEqual(result.timing.slow_nodes[0].record_id, parent["record_id"])

    def test_unknown_node_duration_is_not_zero_and_slow_nodes_are_bounded(self):
        result = self.report(
            [
                self.record("unknown_node", status="unknown", duration=None),
                *[self.record(f"node_{i}", duration=i * 10) for i in range(7)],
            ]
        )
        group = next(row for row in result.timing.node_timings if row.name == "unknown_node")
        self.assertIsNone(group.inclusive_duration_ms_sum)
        self.assertFalse(result.records.complete)
        self.assertEqual(len(result.timing.slow_nodes), 5)
        self.assertEqual(result.timing.slow_nodes[0].duration_ms, 60)

    def test_search_distinguishes_cache_tool_failures_and_remote_attempts(self):
        tools = [
            self.record(
                "web_search_tool",
                kind="tool",
                status=status,
                payload={
                    "cache_status": cache,
                    "remote_requested": remote,
                },
            )
            for status, cache, remote in (
                ("succeeded", "hit", False),
                ("failed", "miss", True),
                ("succeeded", "miss", True),
            )
        ]
        result = self.report([*tools, self.record("other_tool", kind="tool")])
        self.assertEqual(result.records.tool_count, 4)
        self.assertEqual(result.search.tool_call_count, 3)
        self.assertEqual((result.search.cache_hits, result.search.cache_misses), (1, 2))
        self.assertEqual(result.search.remote_requested_count, 2)
        self.assertEqual((result.search.succeeded_count, result.search.failed_count), (2, 1))
        self.assertTrue(result.search.complete)

    def test_unknown_search_metadata_is_not_assumed_to_mean_cache_miss_or_no_request(self):
        result = self.report([self.record("web_search_tool", kind="tool", status="cancelled")])
        self.assertEqual(result.search.cache_unknown, 1)
        self.assertEqual(result.search.cache_misses, 0)
        self.assertEqual(result.search.remote_request_unknown_count, 1)
        self.assertFalse(result.search.complete)

    def test_failure_chain_is_preserved_and_deepest_record_is_shown_first(self):
        payload = {"error": {"type": "TimeoutError", "message": "搜索超时"}}
        root = self.record("research_node", status="failed", payload=payload)
        node = self.record(
            "search_node", status="failed", parent=root["record_id"], payload=payload
        )
        tool = self.record(
            "web_search_tool",
            kind="tool",
            status="failed",
            parent=node["record_id"],
            payload=payload,
        )
        self.execution.update(status="failed", error=payload["error"])
        result = self.report([root, node, tool])
        self.assertEqual(result.issues[0].record_id, tool["record_id"])
        self.assertEqual(
            [row.name for row in result.issues[0].parents], ["research_node", "search_node"]
        )
        self.assertEqual(result.records.status_counts["failed"], 3)  # 失败记录数，不是独立故障数。

    def test_degraded_output_and_warnings_are_visible_without_changing_business_status(self):
        result = self.report(
            [
                self.record(
                    "generator",
                    status="degraded",
                    payload={
                        "warnings": ["使用证据清单"],
                    },
                )
            ]
        )
        self.assertEqual(result.execution.status, "completed")
        self.assertEqual(result.issues[0].status, "degraded")
        self.assertEqual(result.issues[0].warnings, ["使用证据清单"])

    def test_interruption_detection_time_is_not_used_as_actual_duration(self):
        self.execution.update(status="interrupted", finished_at=NOW + timedelta(days=3))
        result = self.report([])
        self.assertIsNone(result.timing.duration_ms)
        self.assertIsNone(result.timing.elapsed_ms)
        self.assertEqual(result.timing.basis, "unknown_interruption_time")
        self.assertFalse(result.models.usage_complete)
        self.assertIsNone(result.models.usage)

    def test_cancellation_before_start_has_no_fake_execution_duration(self):
        self.execution.update(status="cancelled", started_at=None)
        result = self.report([])
        self.assertEqual(result.timing.basis, "not_started")
        self.assertIsNone(result.timing.duration_ms)

    def test_other_execution_records_are_rejected(self):
        record = self.record()
        record["execution_id"] = uuid4()
        with self.assertRaises(ValueError):
            self.report([record])

    def test_report_is_deterministic_and_does_not_mutate_inputs(self):
        rows = [self.record()]
        original = deepcopy((self.execution, rows))
        first = self.report(rows).model_dump(mode="json")
        self.assertEqual(first, self.report(rows).model_dump(mode="json"))
        self.assertEqual((self.execution, rows), original)

    def test_reasoning_token_details_are_not_added_again(self):
        tokens = {**usage(10, 20), "output_token_details": {"reasoning": 15}}
        node = self.model([{"status": "succeeded", "usage": tokens}], total=tokens)
        self.assertEqual(self.report([node]).models.usage.total_tokens, 30)

    def test_missing_parent_even_on_successful_record_prevents_complete_claim(self):
        result = self.report([self.record(parent=uuid4())])
        self.assertFalse(result.records.complete)
        self.assertFalse(result.models.usage_complete)
