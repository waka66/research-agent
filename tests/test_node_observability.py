"""剩余图节点的离线测试：使用真实校验/渲染逻辑和模拟模型。"""

import asyncio
import importlib.util
import json
import logging
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from research_agent.graph.error import WorkerFatalError
from research_agent.schemas.research import (
    Finding,
    Point,
    PointState,
    Query,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
    ResearchTask,
    ResearchWorkerPlan,
    Section,
)

ROOT = Path(__file__).resolve().parents[1] / "src/research_agent/graph"


def load_file(name):
    spec = importlib.util.spec_from_file_location(
        f"_observability_{name}_test", ROOT / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_nodes():
    llm = ModuleType("research_agent.llm")
    llm.llm = Mock()
    llm.llm_synth = Mock()
    tools = ModuleType("research_agent.graph.tools")
    tools.web_search_tool = SimpleNamespace(ainvoke=AsyncMock())
    workflow = ModuleType("research_agent.graph.worker_workflow")
    workflow.worker_graph = SimpleNamespace(ainvoke=AsyncMock())
    with patch.dict(
        sys.modules,
        {
            "research_agent.llm": llm,
            "research_agent.graph.tools": tools,
            "research_agent.graph.worker_workflow": workflow,
        },
    ):
        return load_file("worker_nodes"), load_file("main_nodes")


worker, main = load_nodes()


def response(parsed, tokens=10, error=None):
    usage = (
        None
        if tokens is None
        else {
            "input_tokens": tokens,
            "output_tokens": tokens,
            "total_tokens": 2 * tokens,
        }
    )
    return {"parsed": parsed, "raw": SimpleNamespace(usage_metadata=usage), "parsing_error": error}


class NodeObservabilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {"configurable": {"thread_id": "run-nodes"}}
        self.task = ResearchTask(id="T1", question="什么时候执行任务？", objective="确认执行时机")
        self.point = PointState(id="T1-p0", text="后台任务执行时机", kind="official_fact")
        self.source = SimpleNamespace(
            id="s1",
            url="https://example.com/doc",
            norm_url="https://example.com/doc",
            domain="example.com",
            title="文档",
            summary="后台任务在响应发送之后执行。",
        )
        # 使用真实 Source，校验与渲染不会被模拟。
        from research_agent.schemas.research import Source

        self.source = Source(**vars(self.source))
        self.raw_finding = SimpleNamespace(
            point_ids=[self.point.id],
            result_id="s1",
            quote=self.source.summary,
            claim="后台任务在响应发送之后执行。",
            stance="supports",
        )
        self.parsed_extract = SimpleNamespace(findings=[self.raw_finding])
        self.finding = Finding(
            id="f1",
            point_ids=[self.point.id],
            source_id="s1",
            claim="后台任务在响应发送之后执行。",
            quote=self.source.summary,
            stance="supports",
        )
        self.state = {
            "task": self.task,
            "points": {self.point.id: self.point},
            "new_results": [self.source],
            "sources": {"s1": self.source},
            "findings": {},
            "round_findings": [],
            "round": 0,
            "no_progress": 0,
            "search_calls": 0,
            "error_calls": 0,
            "queries_tried": [],
            "stop_reason": "",
        }
        self.result = ResearchResult(
            task_id="T1",
            points=[self.point],
            findings=[self.finding],
            sources=[self.source],
        )
        self.report = ResearchReport(
            title="报告",
            key_conclusions=[
                Point(
                    text=self.finding.claim,
                    finding_ids=["f1"],
                    kind="fact",
                )
            ],
            sections=[
                Section(
                    heading="执行时机",
                    points=[
                        Point(
                            text=self.finding.claim,
                            finding_ids=["f1"],
                            kind="fact",
                        )
                    ],
                )
            ],
        )
        self.main_state = {
            "question": self.task.question,
            "research_plan": ResearchPlan(goal="研究", tasks=[self.task]),
            "research_results": [self.result],
            "report": self.report,
        }
        self.model = SimpleNamespace(ainvoke=AsyncMock())
        self.records = []

        def capture(level, payload):
            self.records.append((level, json.loads(payload)))

        for target, attribute, value in [
            (worker.llm, "with_structured_output", Mock(return_value=self.model)),
            (main.llm_synth, "with_structured_output", Mock(return_value=self.model)),
            (main.llm, "with_structured_output", Mock(return_value=self.model)),
            (worker.logger, "log", capture),
            (main.logger, "log", capture),
        ]:
            replacement = patch.object(target, attribute, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def record(self, name, status="succeeded"):
        matches = [(level, item) for level, item in self.records if item["node_name"] == name]
        self.assertEqual(len(matches), 1)
        level, record = matches[0]
        self.assertEqual(record["status"], status)
        self.assertEqual(record["run_id"], "run-nodes")
        self.assertGreaterEqual(record["duration_ms"], 0)
        if status == "failed":
            self.assertEqual(level, logging.ERROR)
        if status == "degraded":
            self.assertEqual(level, logging.WARNING)
        return record

    async def test_planner_records_one_attempt_without_adding_retry(self):
        self.model.ainvoke.return_value = response(self.main_state["research_plan"])
        result = await main.Planner(self.main_state, self.config)
        record = self.record("Planner")
        self.model.ainvoke.assert_awaited_once()
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(record["usage"]["total_tokens"], 20)
        self.assertTrue(record["usage_complete"])
        self.assertEqual(record["output_summary"], {"task_count": 1})
        self.assertEqual(result["status"], "running")

    async def test_planner_parse_failure_keeps_usage_and_propagates(self):
        self.model.ainvoke.return_value = response(None, error=ValueError("计划解析失败"))
        with self.assertRaisesRegex(ValueError, "Planner 结构化输出失败"):
            await main.Planner(self.main_state, self.config)
        record = self.record("Planner", "failed")
        self.model.ainvoke.assert_awaited_once()
        self.assertEqual(record["attempts"][0]["status"], "failed")
        self.assertEqual(record["usage"]["total_tokens"], 20)

    async def test_planner_request_failure_keeps_usage_unknown(self):
        self.model.ainvoke.side_effect = RuntimeError("请求失败")
        with self.assertRaisesRegex(RuntimeError, "请求失败"):
            await main.Planner(self.main_state, self.config)
        record = self.record("Planner", "failed")
        self.model.ainvoke.assert_awaited_once()
        self.assertIsNone(record["usage"])
        self.assertFalse(record["usage_complete"])

    async def test_research_plan_skip_has_no_model_attempt(self):
        result = await worker.research_plan(self.state, self.config)
        record = self.record("research_plan", "skipped")
        self.assertEqual(result, {})
        self.assertEqual(record["output_summary"]["reason"], "existing_points")
        self.assertNotIn("attempts", record)
        self.model.ainvoke.assert_not_awaited()

    async def test_research_plan_warnings_are_in_same_structured_record(self):
        self.state["points"] = {}
        parsed = ResearchWorkerPlan(
            points=[{"text": self.point.text}, {"text": self.point.text}],
            queries=[{"text": "查询", "target_indices": [0]}],
        )
        self.model.ainvoke.return_value = response(parsed)
        result = await worker.research_plan(self.state, self.config)
        record = self.record("research_plan")
        self.model.ainvoke.assert_awaited_once()
        self.assertEqual(record["task_id"], self.task.id)
        self.assertEqual(record["round"], 0)
        self.assertEqual(len(result["points"]), 1)
        self.assertEqual(record["output_summary"]["warning_count"], len(record["warnings"]))
        self.assertIn("重复", record["warnings"][0])
        self.assertEqual(self.records[0][0], logging.WARNING)
        self.assertEqual(len(record["attempts"]), 1)

    async def test_research_plan_parse_failure_keeps_usage(self):
        self.state["points"] = {}
        self.model.ainvoke.return_value = response(None, error=ValueError("计划解析失败"))
        with self.assertRaisesRegex(WorkerFatalError, "research_plan 解析失败"):
            await worker.research_plan(self.state, self.config)
        record = self.record("research_plan", "failed")
        self.assertEqual(record["usage"]["total_tokens"], 20)
        self.assertEqual(record["attempts"][0]["status"], "failed")

    async def test_research_plan_validation_error_is_not_swallowed(self):
        self.state["points"] = {}
        self.model.ainvoke.return_value = response(ResearchWorkerPlan(points=[], queries=[]))
        with patch.object(worker, "seed_points", side_effect=ValueError("没有有效研究点")):
            with self.assertRaisesRegex(ValueError, "没有有效研究点"):
                await worker.research_plan(self.state, self.config)
        record = self.record("research_plan", "failed")
        self.assertEqual(record["attempts"][0]["status"], "succeeded")
        self.assertEqual(record["usage"]["total_tokens"], 20)

    async def test_planning_nodes_record_cancellation_and_propagate(self):
        self.state["points"] = {}
        self.model.ainvoke.side_effect = asyncio.CancelledError()
        for name, function, state in [
            ("Planner", main.Planner, self.main_state),
            ("research_plan", worker.research_plan, self.state),
        ]:
            with self.subTest(node=name):
                self.records.clear()
                with self.assertRaises(asyncio.CancelledError):
                    await function(state, self.config)
                record = self.record(name, "cancelled")
                self.assertEqual(record["attempts"][0]["status"], "cancelled")

    async def test_extract_skips_without_calling_model(self):
        self.state["new_results"] = []
        result = await worker.extract_node(self.state, self.config)
        record = self.record("extract_node", "skipped")
        self.assertEqual(record["output_summary"]["reason"], "no_new_results")
        self.assertEqual(result, {"round_findings": []})
        self.model.ainvoke.assert_not_awaited()

    async def test_extract_retry_keeps_failed_parse_usage(self):
        self.model.ainvoke.side_effect = [
            response(None, tokens=10, error=ValueError("解析错误")),
            response(self.parsed_extract, tokens=20),
        ]
        result = await worker.extract_node(self.state, self.config)
        record = self.record("extract_node")
        self.assertEqual([item["status"] for item in record["attempts"]], ["failed", "succeeded"])
        self.assertEqual(record["usage"]["total_tokens"], 60)
        self.assertTrue(record["usage_complete"])
        self.assertEqual(record["output_summary"]["new_finding_count"], 1)
        self.assertEqual(len(result["round_findings"]), 1)
        self.assertEqual(self.state["findings"], {})

    async def test_extract_failure_records_both_attempts_and_propagates(self):
        self.model.ainvoke.side_effect = [RuntimeError("接口失败"), RuntimeError("再次失败")]
        with self.assertRaises(WorkerFatalError):
            await worker.extract_node(self.state, self.config)
        record = self.record("extract_node", "failed")
        self.assertEqual(len(record["attempts"]), 2)
        self.assertIsNone(record["usage"])
        self.assertFalse(record["usage_complete"])

    async def test_extract_timeout_does_not_retry(self):
        self.model.ainvoke.side_effect = TimeoutError("超时")
        with self.assertRaises(WorkerFatalError):
            await worker.extract_node(self.state, self.config)
        self.assertEqual(len(self.record("extract_node", "failed")["attempts"]), 1)

    async def test_extract_cancellation_propagates(self):
        self.model.ainvoke.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await worker.extract_node(self.state, self.config)
        record = self.record("extract_node", "cancelled")
        self.assertEqual(record["attempts"][0]["status"], "cancelled")

    async def test_update_stop_uses_final_domain_progress_and_preserves_input(self):
        self.point.kind = "multi_source_fact"
        old = self.finding.model_copy(update={"id": "old"})
        self.state.update(
            {
                "findings": {"old": old, "f1": self.finding},
                "round_findings": [self.finding],
                "no_progress": 1,
            }
        )
        before = deepcopy(self.state)
        result = await worker.update_node(self.state, self.config)
        record = self.record("update_node")
        self.assertEqual(result["no_progress"], 2)
        self.assertTrue(result["stop_now"])
        self.assertEqual(result["stop_reason"], "no_progress")
        self.assertEqual(record["output_summary"]["no_progress"], result["no_progress"])
        self.assertEqual(self.state, before)

    async def test_update_closes_point_when_evidence_is_sufficient(self):
        self.state["findings"] = {"f1": self.finding}
        self.state["round_findings"] = [self.finding]
        result = await worker.update_node(self.state, self.config)
        self.assertEqual(result["stop_reason"], "all_done")
        self.assertEqual(self.record("update_node")["output_summary"]["closed_point_count"], 1)
        self.assertEqual(self.point.status, "open")

    async def test_plan_next_retry_exposes_incomplete_usage(self):
        parsed = SimpleNamespace(
            queries=[Query(text="新查询", targets=[self.point.id])],
            new_points=[],
            give_up_reason="",
        )
        self.model.ainvoke.side_effect = [RuntimeError("接口错误"), response(parsed)]
        result = await worker.plan_next(self.state, self.config)
        record = self.record("plan_next")
        self.assertEqual(len(record["attempts"]), 2)
        self.assertEqual(record["usage"]["total_tokens"], 20)
        self.assertFalse(record["usage_complete"])
        self.assertFalse(result["stop_now"])

    async def test_plan_next_distinguishes_filtered_queries_from_model_give_up(self):
        self.state["queries_tried"] = ["重复查询"]
        self.model.ainvoke.return_value = response(
            SimpleNamespace(
                queries=[Query(text="重复查询", targets=[self.point.id])],
                new_points=[],
                give_up_reason="",
            )
        )
        result = await worker.plan_next(self.state, self.config)
        record = self.record("plan_next")
        self.assertTrue(result["stop_now"])
        self.assertEqual(record["output_summary"]["no_query_reason"], "all_queries_filtered")

    async def test_plan_next_logs_model_give_up(self):
        self.model.ainvoke.return_value = response(
            SimpleNamespace(
                queries=[],
                new_points=[],
                give_up_reason="没有更多可用查询",
            )
        )
        await worker.plan_next(self.state, self.config)
        record = self.record("plan_next")
        self.assertEqual(record["output_summary"]["no_query_reason"], "model_no_queries")
        self.assertEqual(record["output_summary"]["give_up_reason"], "没有更多可用查询")

    async def test_plan_next_all_failed_is_fatal(self):
        self.model.ainvoke.return_value = response(None, error=ValueError("解析失败"))
        with self.assertRaises(WorkerFatalError):
            await worker.plan_next(self.state, self.config)
        record = self.record("plan_next", "failed")
        self.assertEqual(len(record["attempts"]), 2)

    async def test_results_logs_quality_and_removes_source_summary(self):
        self.state["stop_reason"] = "no_queries"
        self.state["findings"] = {"f1": self.finding}
        result = await worker.research_results(self.state, self.config)
        record = self.record("research_results")
        self.assertEqual(record["output_summary"]["point_status_counts"]["done"], 1)
        self.assertEqual(record["output_summary"]["stop_reason"], "no_queries")
        self.assertEqual(result["result"].sources[0].summary, "")
        self.assertNotEqual(self.source.summary, "")

    async def test_research_node_passes_config_and_logs_child_summary(self):
        graph = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value={
                    "result": self.result,
                    "round": 2,
                    "stop_reason": "all_done",
                }
            )
        )
        with patch.object(main, "worker_graph", graph):
            result = await main.research_node({"task": self.task}, self.config)
        graph.ainvoke.assert_awaited_once()
        self.assertIs(graph.ainvoke.await_args.kwargs["config"], self.config)
        self.assertEqual(result["research_results"], [self.result])
        self.assertEqual(self.record("research_node")["output_summary"]["round_count"], 2)

    async def test_research_node_propagates_child_failure(self):
        graph = SimpleNamespace(ainvoke=AsyncMock(side_effect=WorkerFatalError("子图失败")))
        with patch.object(main, "worker_graph", graph):
            with self.assertRaises(WorkerFatalError):
                await main.research_node({"task": self.task}, self.config)
        self.assertEqual(self.record("research_node", "failed")["error"]["message"], "子图失败")

    async def test_synthesizer_validation_retry_keeps_both_usage_records(self):
        invalid = self.report.model_copy(deep=True)
        invalid.key_conclusions[0].finding_ids = ["does-not-exist"]
        self.model.ainvoke.side_effect = [response(invalid), response(self.report)]
        result = await main.Synthesizer(self.main_state, self.config)
        record = self.record("Synthesizer")
        self.assertEqual(record["attempts"][0]["error"]["type"], "ReportValidationError")
        self.assertEqual(record["usage"]["total_tokens"], 40)
        self.assertEqual(result["report"], self.report)

    async def test_synthesizer_degrades_after_two_failed_attempts(self):
        self.model.ainvoke.return_value = response(None, error=ValueError("解析失败"))
        result = await main.Synthesizer(self.main_state, self.config)
        record = self.record("Synthesizer", "degraded")
        self.assertEqual(result, {"report": None, "status": "partial"})
        self.assertEqual(len(record["attempts"]), 2)
        self.assertEqual(record["output_summary"]["render_mode"], "evidence_only")

    async def test_synthesizer_validator_bug_is_not_hidden_by_fallback(self):
        self.model.ainvoke.return_value = response(self.report)
        with patch.object(main, "validate_report", side_effect=RuntimeError("校验器程序错误")):
            with self.assertRaisesRegex(RuntimeError, "校验器程序错误"):
                await main.Synthesizer(self.main_state, self.config)
        self.assertEqual(len(self.record("Synthesizer", "failed")["attempts"]), 1)

    async def test_generator_logs_markdown_size(self):
        result = await main.generator(self.main_state, self.config)
        record = self.record("generator")
        self.assertEqual(record["output_summary"]["markdown_chars"], len(result["final_answer"]))
        self.assertIn("参考来源", result["final_answer"])

    async def test_generator_fallback_is_observable(self):
        self.main_state["report"] = None
        result = await main.generator(self.main_state, self.config)
        self.assertIn("证据清单", result["final_answer"])
        self.assertEqual(
            self.record("generator", "degraded")["output_summary"]["render_mode"], "evidence_only"
        )

    async def test_generator_failure_propagates(self):
        with patch.object(main, "render_report", side_effect=ValueError("渲染失败")):
            with self.assertRaises(ValueError):
                await main.generator(self.main_state, self.config)
        self.record("generator", "failed")

    async def test_complete_graph_with_simulated_providers(self):
        def model_for(schema, **kwargs):
            if schema is ResearchPlan:
                parsed = self.main_state["research_plan"]
            elif schema is ResearchWorkerPlan:
                parsed = ResearchWorkerPlan(
                    points=[{"text": self.point.text, "kind": "official_fact"}],
                    queries=[{"text": "查询", "target_indices": [0]}],
                )
            else:
                # search_node 使用 URL hash 生成来源 ID，按真实 ID 构造模型返回。
                import hashlib

                source_id = hashlib.sha1(self.source.norm_url.encode()).hexdigest()[:12]
                parsed = SimpleNamespace(
                    findings=[
                        SimpleNamespace(
                            **{**vars(self.raw_finding), "result_id": source_id},
                        )
                    ]
                )
            return SimpleNamespace(ainvoke=AsyncMock(return_value=response(parsed)))

        worker.llm.with_structured_output.side_effect = model_for
        # 复用真实 workflow 定义，检查新签名在完整父子图里能正确注入 config。
        with patch.dict(sys.modules, {"research_agent.graph.worker_nodes": worker}):
            workflow = load_file("worker_workflow")
        with patch.dict(sys.modules, {"research_agent.graph.main_nodes": main}):
            main_workflow = load_file("main_workflow")
        search = AsyncMock(
            return_value={
                "search_result": [
                    {
                        "url": self.source.url,
                        "title": self.source.title,
                        "summary": self.source.summary,
                    }
                ]
            }
        )
        # 报告综合使用证据清单降级路径，验证完整流程仍然能渲染并结束。
        main.llm_synth.with_structured_output.return_value = SimpleNamespace(
            ainvoke=AsyncMock(return_value=response(None, error=ValueError("模拟降级"))),
        )
        with (
            patch.object(main, "worker_graph", workflow.worker_graph),
            patch.object(
                worker.web_search_tool,
                "ainvoke",
                search,
            ),
        ):
            result = await main_workflow.agent_builder.compile().ainvoke(
                {"question": self.task.question},
                config=self.config,
            )
        self.assertEqual(result["status"], "completed")
        self.assertIn("证据清单", result["final_answer"])
        names = {record["node_name"] for _, record in self.records}
        self.assertEqual(
            names,
            {
                "Planner",
                "research_node",
                "research_plan",
                "search_node",
                "extract_node",
                "update_node",
                "research_results",
                "Synthesizer",
                "generator",
            },
        )
        self.assertTrue(all(record["run_id"] == "run-nodes" for _, record in self.records))


if __name__ == "__main__":
    unittest.main()
