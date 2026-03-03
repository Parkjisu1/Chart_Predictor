"""Application settings loaded from .env via Pydantic."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class BybitSettings(BaseSettings):
    api_key: str = Field(default="", alias="BYBIT_API_KEY")
    api_secret: str = Field(default="", alias="BYBIT_API_SECRET")
    testnet: bool = Field(default=True, alias="BYBIT_TESTNET")


class TelegramSettings(BaseSettings):
    bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")


class ClaudeSettings(BaseSettings):
    cli_path: str = Field(default="claude", alias="CLAUDE_CLI_PATH")
    timeout: int = Field(default=120, alias="CLAUDE_CLI_TIMEOUT")


class TradingSettings(BaseSettings):
    initial_capital: float = Field(default=100_000, alias="INITIAL_CAPITAL")
    max_leverage: int = Field(default=5, alias="MAX_LEVERAGE")
    default_symbols: str = Field(default="BTCUSDT,ETHUSDT", alias="DEFAULT_SYMBOLS")

    @property
    def symbols(self) -> list[str]:
        return [s.strip() for s in self.default_symbols.split(",")]


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    db_path: str = Field(default="data/chart_predictor.db", alias="DB_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    bybit: BybitSettings = Field(default_factory=BybitSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    claude: ClaudeSettings = Field(default_factory=ClaudeSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)

    @property
    def db_full_path(self) -> Path:
        return Path(self.db_path)


def get_settings() -> Settings:
    return Settings()
