from app.core.config import settings


def get_llm(temperature: float = 0.0):
    """Return a LangChain chat model. Edit this ONE place to change provider."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # pip install langchain-anthropic
        return ChatAnthropic(model=settings.llm_model, api_key=settings.llm_api_key,
                             temperature=temperature)
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI  # pip install langchain-openai
        return ChatOpenAI(model=settings.llm_model, api_key=settings.llm_api_key,
                          temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
