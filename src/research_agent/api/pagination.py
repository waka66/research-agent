"""两类历史列表共用的游标，只负责解析和范围校验，不访问数据库。"""

import base64
import binascii
from uuid import UUID

from fastapi import HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError


class PageCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: AwareDatetime
    item_id: UUID
    scope: str
    filters: dict[str, str]


def encode_cursor(timestamp, item_id: UUID, *, scope: str, filters: dict[str, str]) -> str:
    payload = PageCursor(timestamp=timestamp, item_id=item_id, scope=scope, filters=filters)
    return base64.urlsafe_b64encode(payload.model_dump_json().encode()).decode().rstrip("=")


def decode_cursor(value: str | None, *, scope: str, filters: dict[str, str]) -> PageCursor | None:
    if value is None:
        return None
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        cursor = PageCursor.model_validate_json(raw)
        if cursor.scope != scope or cursor.filters != filters:
            raise ValueError("游标与当前查询不匹配")
    except (ValueError, binascii.Error, ValidationError) as exc:
        raise HTTPException(
            status_code=422, detail="无效游标，或游标与当前研究、执行及筛选条件不匹配"
        ) from exc
    return cursor
