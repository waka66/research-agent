"""纯代码渲染：LLM 只给骨架，引用/原文/编号/缺口全部由代码产生。"""

import re

from research_agent.schemas.research import ResearchReport, ResearchResult


def _collect(results: list[ResearchResult]):
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
    findings, sources, points = _collect(results)

    # ---- 引用编号：按 norm_url 合并，顺序 = 正文首次出现 ----
    url_to_no: dict[str, int] = {}
    sid_to_no: dict[str, int] = {}
    refs: dict[int, object] = {}

    def cite(source_id: str) -> str:
        if source_id in sid_to_no:
            return f"[{sid_to_no[source_id]}]"
        src = sources.get(source_id)
        key = (getattr(src, "norm_url", "") or "") or source_id
        if key not in url_to_no:
            url_to_no[key] = len(url_to_no) + 1
            refs[url_to_no[key]] = src
        sid_to_no[source_id] = url_to_no[key]
        return f"[{sid_to_no[source_id]}]"

    def render_point(p) -> list[str]:
        ids = [i for i in p.finding_ids if i in findings]
        marks = "".join(dict.fromkeys(cite(findings[i].source_id) for i in ids))
        prefix = "（推断）" if p.kind == "inference" else ""
        lines = [f"- {prefix}{p.text} {marks}".rstrip()]
        if p.kind == "fact":
            seen: set[str] = set()
            for i in ids:
                f = findings[i]
                if f.quote in seen:
                    continue
                seen.add(f.quote)
                src = sources.get(f.source_id)
                lines.append(f'  > "{f.quote}"')
                if src:
                    lines.append(f"  > —— {src.title} · {src.domain}")
        return lines

    out: list[str] = []
    title = report.title if report else f"调研报告：{question[:40]}"
    out += [f"# {title}", ""]

    if report and report.key_conclusions:
        out += ["## 核心结论", ""]
        for p in report.key_conclusions:
            out += render_point(p)
        out += [""]

    if report:
        for idx, section in enumerate(report.sections, 1):
            out += [f"## {idx}. {section.heading}", ""]
            for p in section.points:
                out += render_point(p)
            out += [""]
    else:
        out += ["## 证据清单（未生成结构，按点列出）", ""]
        for f in findings.values():
            out += [f"- {f.claim} {cite(f.source_id)}", f'  > "{f.quote}"', ""]

    # ---- 冲突：只列 contradicts ----
    by_point: dict[str, list] = {}
    for f in findings.values():
        for pid in f.point_ids:
            by_point.setdefault(pid, []).append(f)

    conflicts = [
        p for p in points.values()
        if any(f.stance == "contradicts" for f in by_point.get(p.id, []))
    ]
    if conflicts:
        out += [
            "## 存在冲突的证据", "",
            "以下材料对这些点的判断与正文引用的说法不一致，原句保留供核对。", "",
        ]
        for p in conflicts:
            out += [f"### 「{p.text}」", ""]
            for f in by_point[p.id]:
                if f.stance != "contradicts":
                    continue
                out.append(f"- {f.claim} {cite(f.source_id)}")
                out.append(f'  > "{f.quote}"')
                src = sources.get(f.source_id)
                if src:
                    out.append(f"  > —— {src.title} · {src.domain}")
            out += [""]

    # ---- 缺口 ----
    thin = [p for p in points.values() if p.status == "thin"]
    missing = [p for p in points.values() if p.status == "missing"]
    if thin or missing:
        out += ["## 证据缺口与局限", ""]
        for p in thin:
            out.append(f"- **仅单一来源**：{p.text}")
        for p in missing:
            out.append(f"- **未找到证据**：{p.text}")
        out += ["", "以上问题未获得足够证据，本报告不对其作结论。", ""]

    # ---- 参考来源（按编号，已按 URL 合并）----
    if refs:
        out += ["## 参考来源", ""]
        for no, s in sorted(refs.items()):
            if s is None:
                out.append(f"{no}. （来源信息缺失）")
            else:
                out.append(f"{no}. [{s.title}]({s.url}) · {s.domain}")

    return "\n".join(out).strip() + "\n"