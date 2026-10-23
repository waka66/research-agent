from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from research_agent.config import settings


database_url = make_url(settings.db_uri).set(
    drivername="postgresql+psycopg"
)

engine = create_async_engine(
    database_url,
    pool_pre_ping=True,
)

SessionFactory = async_sessionmaker(
    engine,
    expire_on_commit=False,
)