import logging
import re

from research_agent.schemas.research import ResearchFindingDraft

logger = logging.getLogger(__name__)


# 从 claim 里提取"硬 token"：数字（含小数）+ 英文标识符（含下划线）
_TOKEN_PATTERN = re.compile(r"""
    \d+(?:\.\d+)?            # 数字，如 300、1.2
    |
    [A-Za-z][A-Za-z0-9_]*    # 英文标识符，如 supports_ttl、PutOp、TTL
""", re.VERBOSE)

# 常见英文虚词，不参与校验（claim 是英文时避免误杀）
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its",
    "as", "by", "with", "from", "into", "via",
    "if", "then", "else", "when", "while",
}


def _extract_hard_tokens(text: str) -> set[str]:
    """从 claim 里提取必须出现在 quote 里的硬信息"""
    tokens: set[str] = set()
    for m in _TOKEN_PATTERN.finditer(text):
        tok = m.group()
        if tok.isdigit():
            tokens.add(tok)
        elif tok.lower() not in _STOPWORDS and len(tok) >= 2:
            tokens.add(tok)
    return tokens


def _quote_contains_token(quote: str, token: str) -> bool:
    """判断 quote 里是否包含某个 token"""
    if token.isdigit():
        # 数字用前后非数字边界，避免 300 命中 3000
        return re.search(rf"(?<!\d){re.escape(token)}(?!\d)", quote) is not None
    # 英文忽略大小写
    return token.lower() in quote.lower()


def _normalize(text: str) -> str:
    """去空白 + 转小写，用于宽容匹配"""
    return "".join(text.split()).lower()


def validate_findings(
    findings: list[ResearchFindingDraft],
    sources: list[dict],
) -> list[ResearchFindingDraft]:
    """
    校验 LLM 生成的 findings，任意一条规则不满足就整条丢弃。

    规则：
    1. source_id 必须存在于 sources 里
    2. quote 必须能在 source_id 对应的原文里找到（严格或宽松）
    3. claim 里的数字/英文术语/代码标识符必须出现在 quote 里
    """
    source_map = {
        s["id"]: s.get("summary") or ""
        for s in sources
        if s.get("id")
    }

    valid: list[ResearchFindingDraft] = []

    for f in findings:
        # ---- 规则 1：source_id 存在 ----
        if f.source_id not in source_map:
            logger.warning(
                "丢弃（source_id 不存在）: source_id=%s claim=%s",
                f.source_id, f.claim[:50],
            )
            continue

        original = source_map[f.source_id]

        # ---- 规则 2：quote 在原文里 ----
        if f.quote not in original and _normalize(f.quote) not in _normalize(original):
            logger.warning(
                "丢弃（quote 不在原文）: source_id=%s quote=%s",
                f.source_id, f.quote[:80],
            )
            continue

        # ---- 规则 3：claim 的硬信息必须在 quote 里 ----
        hard_tokens = _extract_hard_tokens(f.claim)
        missing = [t for t in hard_tokens if not _quote_contains_token(f.quote, t)]
        if missing:
            logger.warning(
                "丢弃（claim 硬信息不在 quote）: source_id=%s missing=%s claim=%s",
                f.source_id, missing, f.claim[:60],
            )
            continue

        valid.append(f)

    return valid