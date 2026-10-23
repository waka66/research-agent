from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import logging

from research_agent.api import research
from research_agent.graph.agent import lifespan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(lifespan=lifespan)

# CORS 中间件,
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "research-agent成功运行"}


app.include_router(research.router)