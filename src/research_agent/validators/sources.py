"""来源表的规则：URL 归一化 + 发布方归属。纯代码、可单测。"""

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# 两段式后缀：这些情况下 eTLD+1 要取三段
_TWO_PART_SUFFIXES = {
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "co.uk", "ac.uk", "org.uk", "co.jp", "ne.jp", "or.jp",
    "com.hk", "com.tw", "com.au", "co.kr", "co.in",
}

# 追踪参数，不影响页面身份
_DROP_KEYS = {"ref", "ref_src", "fbclid", "gclid", "spm", "from", "source"}


def norm_url(url: str) -> str:
    """去重口径。只做保守合并，不会把两个不同页面并成一个。"""
    if not url:
        return ""
    p = urlparse(url.strip())      #urlparse把 URL 拆成 6 段
    netloc = (p.netloc or "").lower()    #netloc：域名+端口
    if netloc.startswith("www."):       #去掉www.
        netloc = netloc[4:]
    query = [
        (k, v)
        for k, v in parse_qsl(p.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _DROP_KEYS
    ]
    path = (p.path or "/").rstrip("/") or "/"
    # scheme 统一成 https：同一页面的 http / https 视为同一个来源
    return urlunparse(("https", netloc, path, "", urlencode(query), ""))


def registrable_domain(url: str) -> str:
    """算"独立来源"时用的域名：docs.x.com 与 blog.x.com 归为同一个 x.com。"""
    host = (urlparse(url).netloc or "").lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2 or all(p.isdigit() for p in parts):   # 裸域 / IP
        return ".".join(parts)
    if ".".join(parts[-2:]) in _TWO_PART_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])