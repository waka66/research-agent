from research_agent.schemas.research import (PointState,Query,ResearchTask,ResearchWorkerPlan)

# 关闭门槛：同域名只算一个来源
MIN_EVIDENCE_BY_KIND: dict[str, int] = {
    "official_fact": 1,       # 官方文档/源码里的单一权威事实
    "multi_source_fact": 2,   # 需要 2 个不同域名互证
    "comparison": 2,          # 两侧各自的来源
}

MAX_SEED_POINTS = 7
MAX_SEED_QUERIES = 2
def normalize(text: str) -> str:
    return "".join((text or "").split()).lower()  #把所有大写字母转成小写,可以用于比较
def min_evidence(point: PointState) -> int:
    return MIN_EVIDENCE_BY_KIND.get(point.kind, 2)

#校验research_plan的产物
def seed_points(
    plan: ResearchWorkerPlan,
    task: ResearchTask,
) -> tuple[dict[str, PointState], list[Query], list[str]]:
    """
    把 LLM 生成的只读 plan 翻译成代码可控的"账本"。

    背景：plan 是 LLM 的产物，不可信——可能有空文本、重复点、越界索引、
    超过上限的数量、无效的 target_indices。本函数负责一次性清洗、兜底、
    赋稳定 id，产出后续节点可以放心消费的两个结构。

    输入：
        plan: ResearchWorkerPlan — LLM 产出的计划（只读，不改）
        task: ResearchTask — 任务信息，提供 id 和兜底文本

    输出（三元组）：
        points:  dict[pid -> PointState]
                 所有合法的"研究点"，每个带稳定 id f"{task.id}-p{n}"，
                 round_added=0 表示本轮的种子。空/重复/超限的点已剔除；
                 若一个都没剩，用 task.objective 兜底一个 key 点。
        queries: list[Query]
                 所有合法的"查询"，每条带 text 和 targets（指向哪些 pid）。
                 target_indices 已按 index_to_id 翻译成 pid，无效项忽略；
                 若某个 query 没有有效 target，兜底为"搜索所有点"。
                 若一个 query 都没剩，用 task.question 兜底。
        warnings: list[str]
                  清洗过程中产生的所有警告，供上层日志或 UI 展示。

    不变式：
        - 返回的 points 一定非空（有兜底）
        - 返回的 queries 一定非空（有兜底）
        - queries 里的每个 target 都是 points 里真实存在的 pid
        - 同一 text（normalize 后）在 points / queries 里各只出现一次
    """

    warnings: list[str] = []
    points: dict[str, PointState] = {}
    seen_text: set[str] = set()
    index_to_id: dict[int, str] = {}

    # ---- 点 ----
    for idx, pp in enumerate(plan.points):  #enumerate 是 Python 内置函数，遍历时同时拿到"索引"和"元素"
        text = " ".join((pp.text or "").split())  #split()把所有连续空白（换行、tab、多空格）压成单个空格，去掉首尾空白
        if not text:
            warnings.append(f"points[{idx}] 为空，已丢弃")
            continue

        key = normalize(text)
        if key in seen_text:
            warnings.append(f"points[{idx}] 与已有点重复，已丢弃：{text[:40]}")
            continue

        if len(points) >= MAX_SEED_POINTS:
            warnings.append(f"点数超过上限 {MAX_SEED_POINTS}，points[{idx}] 及之后全部丢弃")
            break

        pid = f"{task.id}-p{len(points)}"          # id 只在这里产生
        points[pid] = PointState(
            id=pid,
            text=text,
            kind=pp.kind,
            accepts=" ".join((pp.accepts or "").split()),
            priority=pp.priority,
            round_added=0,
        )
        seen_text.add(key)
        index_to_id[idx] = pid

    # 空清单 = should_stop 立刻为 True = 空报告，必须兜底
    if not points:
        pid = f"{task.id}-p0"
        points[pid] = PointState(
            id=pid,
            text=" ".join((task.objective or task.question).split()),
            kind="multi_source_fact",
            priority="key",
            round_added=0,
        )
        warnings.append("plan 没有产出有效点，已用 task.objective 兜底 1 个点")

    all_ids = list(points)   #取points的所有 key，得到"所有 point 的 id 列表"

    # ---- 查询 ----
    queries: list[Query] = []
    seen_query: set[str] = set()

    for idx, sq in enumerate(plan.queries):
        text = " ".join((sq.text or "").split())
        if not text:
            continue
        if normalize(text) in seen_query:
            warnings.append(f"queries[{idx}] 重复，已丢弃")
            continue
        if len(queries) >= MAX_SEED_QUERIES:
            warnings.append(f"查询数超过上限 {MAX_SEED_QUERIES}，queries[{idx}] 及之后丢弃")
            break

        targets: list[str] = []
        for k in sq.target_indices:
            if k not in index_to_id:      # 越界 / 指向被丢弃的点（这里兜不了 1-based 的错位）
                warnings.append(f"queries[{idx}] 的 target_indices={k} 无效，已忽略")
                continue
            pid = index_to_id[k]
            if pid not in targets:
                targets.append(pid)

        if not targets:                    # 首轮兜底：目标是所有点
            targets = all_ids

        queries.append(Query(text=text, targets=targets))
        seen_query.add(normalize(text))

    if not queries:
        queries = [Query(text=" ".join(task.question.split()), targets=all_ids)]
        warnings.append("plan 没有产出有效查询，已用 task.question 兜底")

    return points, queries, warnings


#-------------------------------
#update_node的校验规则
MAX_NO_PROGRESS = 2      # 连续几轮没有新证据就停
MAX_ROUNDS = 6           # 轮数硬上限（你的规则本身上界是 N+2，这是防代码 bug 的兜底）

# 出现过 contradicts 的点，关闭门槛是否 +1 个独立来源。
# 默认 0：见讨论——继续找"支持方"的来源并不能消解冲突，真正该做的是把冲突记下来。
CONTESTED_EXTRA_EVIDENCE = 0


def evidence_domains(pid: str, findings: dict, sources: dict) -> set[str]:
    """这个点有多少个【不同发布方】的 supports 证据。"""
    return {
        sources[f.source_id].domain
        for f in findings.values()
        if pid in f.point_ids
        and f.stance == "supports"
        and f.source_id in sources
    }


def evidence_count(pid: str, findings: dict, sources: dict) -> int:
    return len(evidence_domains(pid, findings, sources))


def is_contested(pid: str, findings: dict) -> bool:
    return any(
        pid in f.point_ids and f.stance == "contradicts"
        for f in findings.values()
    )


def required_evidence(point: PointState) -> int:
    n = min_evidence(point)
    if point.contested:
        n += CONTESTED_EXTRA_EVIDENCE
    return n


def close_finished_points(points: dict, findings: dict, sources: dict) -> list[str]:
    """把够了的点标成 done。**只改 status/contested，不动 text。** 返回本轮关闭的 id。"""
    closed: list[str] = []
    for p in points.values():
        if p.status != "open":
            continue
        p.contested = is_contested(p.id, findings)
        if evidence_count(p.id, findings, sources) >= required_evidence(p):
            p.status = "done"
            closed.append(p.id)
    return closed


def should_stop(
    points: dict,
    round_no: int,
    no_progress: int,
    errors_exhausted: bool = False,
) -> tuple[bool, str]:
    """终止裁决。纯算术，不含任何 LLM 判断。"""
    if not any(p.status == "open" for p in points.values()):
        return True, "all_done"

    if no_progress >= MAX_NO_PROGRESS:
        return True, "no_progress"
    if round_no >= MAX_ROUNDS:
        return True, "max_rounds"
    if errors_exhausted:
        return True, "errors"
    return False, ""


def finalize_points(points: dict, findings: dict, sources: dict) -> dict:
    """收尾：把仍然 open 的点落成终态。【下一步 results 才会调用】"""
    out = {pid: p.model_copy() for pid, p in points.items()}
    for p in out.values():
        if p.status != "open":
            continue
        p.contested = is_contested(p.id, findings)
        n = evidence_count(p.id, findings, sources)
        p.status = "done" if n >= min_evidence(p) else ("thin" if n >= 1 else "missing")
        doms = evidence_domains(p.id, findings, sources)
        if len(doms) == 1 and evidence_count(p.id, findings, sources) >= 2:
            p.single_domain = True  # 来源充足但只有一个发布方
    return out

#---------------------------------------------

NEW_POINT_MAX_ROUNDS = 2        # 只在前两轮接受新点
MAX_DISCOVERED_POINTS = 2       # 发现来的点最多 2 个
MAX_POINTS_TOTAL = 9            # 总点数额度


def _next_point_index(points: dict) -> int:
    """从现有 id 里推下一个序号。不用 len()，避免以后有点被删/被跳过时撞号。"""
    max_idx = -1
    for pid in points:
        tail = pid.rsplit("-p", 1)[-1]
        if tail.isdigit():
            max_idx = max(max_idx, int(tail))
    return max_idx + 1


def add_discovered(
    points: dict, new_points: list, task_id: str, round_no: int
) -> tuple[dict, list[str]]:
    """代码批准新点。返回 (新 points 字典, 采纳/拒绝说明)。"""
    out = dict(points)
    notes: list[str] = []

    if not new_points:
        return out, notes
    if round_no > NEW_POINT_MAX_ROUNDS:
        return out, [f"第 {round_no} 轮不再接受新点（上限 {NEW_POINT_MAX_ROUNDS} 轮）"]

    existing = {normalize(p.text) for p in out.values()}
    discovered = sum(1 for p in out.values() if p.origin == "discovered")

    for np in new_points:
        text = " ".join((np.text or "").split())
        if not text:
            continue
        if len(out) >= MAX_POINTS_TOTAL:
            notes.append(f"总数已达 {MAX_POINTS_TOTAL}，丢弃：{text[:30]}")
            break
        if discovered >= MAX_DISCOVERED_POINTS:
            notes.append(f"发现点已达 {MAX_DISCOVERED_POINTS}，丢弃：{text[:30]}")
            break
        key = normalize(text)
        if key in existing:
            notes.append(f"与已有点重复，丢弃：{text[:30]}")
            continue

        pid = f"{task_id}-p{_next_point_index(out)}"
        out[pid] = PointState(
            id=pid,
            text=text,
            kind=np.kind,
            accepts=" ".join((np.accepts or "").split()),
            priority=np.priority,
            origin="discovered",
            round_added=round_no,
        )
        existing.add(key)
        discovered += 1
        notes.append(f"采纳 {pid}：{text[:30]}")

    return out, notes


def new_domain_evidence(round_findings: list, prev_findings: dict, sources: dict) -> int:
    """本轮真正新增的（点, 域名）组合数。同域名重复不算进展。"""

    def pairs(items):
        return {
            (pid, sources[f.source_id].domain)
            for f in items
            if f.stance == "supports" and f.source_id in sources
            for pid in f.point_ids
        }

    return len(pairs(round_findings) - pairs(prev_findings.values()))






