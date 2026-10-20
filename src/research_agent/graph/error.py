class WorkerFatalError(RuntimeError):
    """调试期：任何不该继续的错误直接终止整个 run，不留半成品。"""