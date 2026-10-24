"""节点/工具记录的短事务；不共享 Session，不跨越业务调用持有事务。"""

import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import bindparam, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from research_agent.db.models import ExecutionRecord
from research_agent.db.session import SessionFactory


async def start_record(record: dict) -> None:
    if record["status"] != "running":
        raise ValueError("start_record 只接受 running 记录")
    payload = json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))
    async with SessionFactory.begin() as session:
        session.add(
            ExecutionRecord(
                record_id=UUID(record["record_id"]),
                execution_id=UUID(record["execution_id"]),
                parent_record_id=UUID(value) if (value := record["parent_record_id"]) else None,
                kind=record["kind"],
                name=record["name"],
                task_id=record.get("task_id"),
                round=record.get("round"),
                status="running",
                started_at=datetime.fromisoformat(record["started_at"]),
                payload=payload,
            )
        )


async def finish_record(record: dict) -> None:
    if record["status"] in {"running", "interrupted", "unknown"}:
        raise ValueError("finish_record 只接受节点自身的结束状态")
    payload = json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))
    async with SessionFactory.begin() as session:
        result = await session.execute(
            update(ExecutionRecord)
            .where(
                ExecutionRecord.record_id == UUID(record["record_id"]),
                ExecutionRecord.execution_id == UUID(record["execution_id"]),
                ExecutionRecord.status == "running",
            )
            .values(
                status=record["status"],
                finished_at=datetime.fromisoformat(record["finished_at"]),
                duration_ms=record["duration_ms"],
                payload=payload,
            )
            .returning(ExecutionRecord.record_id)
        )
        if result.scalar_one_or_none() is None:
            raise LookupError("没有可更新的 running 记录")


async def close_open_records(
    session: AsyncSession,
    execution_ids: list[UUID],
    *,
    interrupted: bool,
) -> int:
    """修复遗留 running；只记录发现时间，不伪造实际完成时间和耗时。"""
    if not execution_ids:
        return 0
    status = "interrupted" if interrupted else "unknown"
    payload = {
        "status": status,
        "observed_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "duration_ms": None,
        "error": {
            "type": "ProcessInterrupted" if interrupted else "RecordIncomplete",
            "message": "服务进程退出，记录未正常结束"
            if interrupted
            else "执行已结束，记录未正常收尾",
        },
    }
    result = await session.execute(
        update(ExecutionRecord)
        .where(
            ExecutionRecord.execution_id.in_(execution_ids),
            ExecutionRecord.status == "running",
        )
        .values(
            status=status,
            finished_at=None,
            duration_ms=None,
            payload=ExecutionRecord.payload.op("||")(
                bindparam("record_repair_payload", payload, type_=JSONB),
            ),
        )
        .returning(ExecutionRecord.record_id)
    )
    return len(result.scalars().all())
