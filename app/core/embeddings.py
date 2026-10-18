from app.core.config import settings


def get_embedder():
    """Return a LangChain Embeddings object. ALL components must use this same embedder."""
    from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface sentence-transformers
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
