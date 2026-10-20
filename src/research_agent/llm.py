from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI

from research_agent.config import settings


llm = ChatOpenAI(
    model="glm-5.3-flash",
    api_key=settings.api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.6,
    request_timeout=90.0,
    max_retries=0,
    streaming=True,
)

llm_synth = ChatOpenAI(
    model="glm-5.3-flashx",
    api_key=settings.api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.3,
    request_timeout=120.0,
    max_retries=0,
    streaming=True,
    model_kwargs={"reasoning_effort": "low"},
)