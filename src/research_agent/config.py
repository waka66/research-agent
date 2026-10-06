from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# 计算项目根目录的绝对路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    app_name: str = "research-agent"
    debug: bool = False

    api_key: str

    db_uri: str


    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",  # 使用绝对路径
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()