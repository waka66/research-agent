import logging
import re
import hashlib

from research_agent.schemas.research import Finding, Source

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
#---------------------------------------
#校验extract_node的finding
MIN_QUOTE_CHARS = 8         #finding的quote最短长度
MAX_QUOTE_CHARS = 400       #finding的quote最长长度
def finding_id(source_id: str, quote: str) -> str:
    """稳定 id：同一来源里的同一句话，任何一轮抽出来都是同一个 id → 天然跨轮去重。"""
    return hashlib.sha1(f"{source_id}|{_normalize(quote)}".encode("utf-8")).hexdigest()[:12]#将证据ID+格式化的原文拼一起用hash生成ID，一共40位只取前12位


def validate_extract(
    raw_findings: list,                      # 动态生成的 RawFinding，无法静态标注类型
    sources: dict[str, Source],              # 本轮 new_results：id -> Source，清单ID和搜索结果ID组成的字典
    open_ids: set[str],                      #目前开放的点
) -> tuple[list[Finding], list[str]]:
    """把 LLM 的抽取结果变成账本。返回 (通过的 findings, 丢弃原因)。"""
    accepted: list[Finding] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for r in raw_findings:
        pids = list(dict.fromkeys(getattr(r, "point_ids", []) or []))   #先拿到point_ids，再合并成字典，自动去重
        if not pids:
            dropped.append("point_ids 为空")
            continue
        bad = [p for p in pids if p not in open_ids]
        if bad:
            dropped.append(f"point_ids 不在开放清单里 {bad}")
            continue

        src = sources.get(getattr(r, "result_id", None))
        if src is None:
            dropped.append(f"result_id 不在本轮新结果里 {getattr(r, 'result_id', None)}")
            continue

        quote = (r.quote or "").strip()
        if len(quote) < MIN_QUOTE_CHARS:
            dropped.append(f"quote 太短 {quote!r}")
            continue
        if len(quote) > MAX_QUOTE_CHARS:
            dropped.append(f"quote 太长（{len(quote)} 字），疑似整段复制")
            continue

        original = src.summary or ""
        if quote not in original and _normalize(quote) not in _normalize(original):
            dropped.append(f"quote 不在原文: {quote[:40]}")
            continue

        hard = _extract_hard_tokens(r.claim or "")
        missing = [t for t in hard if not _quote_contains_token(quote, t)]
        if missing:
            dropped.append(f"claim 的硬信息不在 quote {missing}: {(r.claim or '')[:40]}")
            continue

        fid = finding_id(src.id, quote)
        if fid in seen:
            dropped.append(f"本轮重复：{quote[:30]}")
            continue
        seen.add(fid)

        accepted.append(
            Finding(id=fid, point_ids=pids, claim=(r.claim or "").strip(),
                    quote=quote, source_id=src.id, stance=r.stance)
        )

    return accepted, dropped