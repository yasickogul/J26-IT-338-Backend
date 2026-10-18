from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str = ""
    database_url_direct: str = ""
    jwt_secret: str = "change-me"
    jwt_expire_minutes: int = 60
    cors_origins: str = "http://localhost:3000"

    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_api_key: str = ""
    embedding_model: str = ""
    embedding_dim: int = 768

    file_storage: str = "local"
    file_storage_path: str = "./data/uploads"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
