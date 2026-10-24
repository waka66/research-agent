"""当前单进程服务的所有权：持锁期间其他实例不能接管遗留执行。"""

from contextlib import asynccontextmanager

from psycopg import AsyncConnection

RUNTIME_LOCK_KEY = 0x5245534541524348


@asynccontextmanager
async def single_instance_runtime(db_uri: str, *, lock_key: int = RUNTIME_LOCK_KEY):
    # 使用独立长连接持有 session advisory lock，连接关闭或进程退出后自动释放。
    async with await AsyncConnection.connect(db_uri, autocommit=True) as connection:
        cursor = await connection.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
        (acquired,) = await cursor.fetchone()
        if not acquired:
            raise RuntimeError("同一数据库已有研究服务运行；当前进程内 SSE 仅支持单实例")
        yield
