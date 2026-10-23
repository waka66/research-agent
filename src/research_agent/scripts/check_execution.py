import asyncio
from uuid import uuid4

from research_agent.db.history import (
    create_start_execution,
    finish_execution,
    get_execution,
    mark_execution_running,
)
from research_agent.db.session import engine


async def main():
    run_id = uuid4()
    execution_id = uuid4()

    try:
        await create_start_execution(
            run_id,
            execution_id,
            "[手动验证] execution 状态变化",
        )
        print("创建后：", await get_execution(execution_id))

        await mark_execution_running(execution_id)
        print("开始后：", await get_execution(execution_id))

        await finish_execution(
            execution_id,
            status="failed",
            error={
                "type": "TestError",
                "message": "用于验证失败记录，不是真实模型错误",
            },
        )
        print("结束后：", await get_execution(execution_id))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    # 当前 Windows + psycopg 异步连接使用 SelectorEventLoop。
    with asyncio.Runner(
        loop_factory=asyncio.SelectorEventLoop,
    ) as runner:
        runner.run(main())