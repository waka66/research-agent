from langchain_openai import ChatOpenAI

from research_agent.config import settings


llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=settings.api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.6,
)