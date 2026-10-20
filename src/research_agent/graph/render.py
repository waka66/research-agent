import re

from research_agent.schemas.research import ResearchReport, ResearchResult


def _assign_citations(results: list[ResearchResult]):
    """把 source_id 映射成 [n]，顺序 = 正文首次出现顺序（确定性）。"""
    findings, sources, points = {}, {}, {}
    for r in results:
        for f in r.findings:
            findings[f.id] = f
        for s in r.sources:
            sources[s.id] = s
        for p in r.points:
            points[p.id] = p
    return findings, sources, points


def render_report(
    report: ResearchReport | None,
    results: list[ResearchResult],
    question: str,
) -> str:
    """纯代码渲染。report 为 None 时降级输出"证据 + 缺口"。"""
    findings, sources, points = _assign_citations(results)

    cite_no: dict[str, int] = {}

    def cite(source_id: str) -> str:
        if source_id not in cite_no:
            cite_no[source_id] = len(cite_no) + 1
        return f"[{cite_no[source_id]}]"

    def render_point(p) -> list[str]:
        ids = [i for i in p.finding_ids if i in findings]
        marks = "".join(dict.fromkeys(cite(findings[i].source_id) for i in ids))
        prefix = "（推断）" if p.kind == "inference" else ""
        lines = [f"- {prefix}{p.text} {marks}".rstrip()]
        for i in ids:
            f, src = findings[i], sources.get(findings[i].source_id)
            if p.kind == "fact":                      # 只有 fact 展开原文
                lines.append(f'  > "{f.quote}"')
                if src:
                    lines.append(f"  > —— {src.title} · {src.domain}")
        return lines

    out: list[str] = []
    title = report.title if report else f"调研报告：{question[:40]}"
    out += [f"# {title}", ""]

    # ---- 核心结论 ----
    if report and report.key_conclusions:
        out += ["## 核心结论", ""]
        for p in report.key_conclusions:
            out += render_point(p)
        out += [""]

    # ---- 正文分节 ----
    if report:
        for idx, section in enumerate(report.sections, 1):
            out += [f"## {idx}. {section.heading}", ""]
            for p in section.points:
                out += render_point(p)
            out += [""]
    else:
        out += ["## 证据清单（未生成结构，按点列出）", ""]
        for f in findings.values():
            out += [f'- {f.claim} {cite(f.source_id)}', f'  > "{f.quote}"', ""]

    # ---- 争议（代码生成，两边一定都在）----
    by_point: dict[str, list] = {}
    for f in findings.values():
        for pid in f.point_ids:
            by_point.setdefault(pid, []).append(f)

    contested = [p for p in points.values() if p.contested or any(
        f.stance == "contradicts" for f in by_point.get(p.id, [])
    )]
    if contested:
        out += ["## 存在冲突的证据", ""]
        for p in contested:
            out += [f"**{p.text}**", ""]
            for f in by_point.get(p.id, []):
                tag = "支持" if f.stance == "supports" else (
                    "反驳" if f.stance == "contradicts" else "背景")
                out.append(f'- [{tag}] {f.claim} {cite(f.source_id)}')
                out.append(f'  > "{f.quote}"')
            out += [""]

    # ---- 缺口（代码生成，来自点状态，不经过 LLM）----
    thin = [p for p in points.values() if p.status == "thin"]
    missing = [p for p in points.values() if p.status == "missing"]
    if thin or missing:
        out += ["## 证据缺口与局限", ""]
        for p in thin:
            out.append(f"- **仅单一来源**：{p.text}")
        for p in missing:
            out.append(f"- **未找到证据（missing）**：{p.text}")
        out += ["", "以上问题在本轮检索中未获得足够证据，报告不对其作结论。", ""]

    # ---- 参考来源（编号与正文一致）----
    if cite_no:
        out += ["## 参考来源", ""]
        for sid, n in sorted(cite_no.items(), key=lambda kv: kv[1]):
            s = sources.get(sid)
            if s:
                out.append(f"{n}. [{s.title}]({s.url}) · {s.domain}")
            else:
                out.append(f"{n}. （来源缺失：{sid}）")

    return "\n".join(out).strip() + "\n"