from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI

from research_agent.config import settings


llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=settings.api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.6,
    request_timeout=300.0,
    max_retries=2,
)

llm2 = ChatOpenAI(
    model="glm-5.3-flashx",
    api_key=settings.api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=1,                   # 官方推荐值
    request_timeout=300.0,
    max_retries=2,
    model_kwargs={                   # 透传额外参数
        "reasoning_effort": "max",
    },
)