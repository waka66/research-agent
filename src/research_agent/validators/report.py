# validators/report.py
from research_agent.validators.findings import (
    _extract_hard_tokens, _quote_contains_token,
)

# fact 里不许出现的词——出现就说明这是推断，不是事实
_HEDGE_WORDS = ("可能", "应该", "似乎", "大概", "也许", "推测", "估计", "或许")


def _all_points(report):
    yield from report.key_conclusions
    for section in report.sections:
        yield from section.points


def _is_year_like(token: str) -> bool:
    return token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2099
def validate_report(report, findings: dict) -> tuple[list[str], list[str]]:
    """fatal 才阻断重试；warn 只打印，不阻断。"""
    fatal, warn = [], []

    for p in _all_points(report):
        # ---- fatal：结构性错误，一定能判 ----
        bad = [i for i in p.finding_ids if i not in findings]
        if bad:
            fatal.append(f"引用了不存在的证据 id：{bad}")
        p.finding_ids = [i for i in p.finding_ids if i in findings]

        if p.kind == "fact" and not p.finding_ids:
            fatal.append(f"fact 无证据：{p.text[:40]}")
        if p.importance == "key" and not p.finding_ids:
            fatal.append(f"关键结论无证据：{p.text[:40]}")

        # ---- warn：启发式检查，先观察 ----
        if p.kind == "fact":
            hedges = [w for w in _HEDGE_WORDS if w in p.text]
            if hedges:
                warn.append(f"fact 里有推断词 {hedges}：{p.text[:50]}")

        if p.finding_ids:
            quotes = " ".join(findings[i].quote for i in p.finding_ids)
            for t in _extract_hard_tokens(p.text):
                if _is_year_like(t):          # 年份不算硬信息
                    continue
                if not _quote_contains_token(quotes, t):
                    warn.append(f"「{p.text[:30]}」里的 {t!r} 不在被引用的原文中")

    if not report.key_conclusions:
        warn.append("没有关键结论")
    if not report.sections or any(not s.points for s in report.sections):
        fatal.append("结构不完整：有空 section")

    return fatal, warn