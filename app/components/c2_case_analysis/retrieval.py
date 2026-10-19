"""Qdrant retrieval (notebook cells 11, 12, 19).

The embedding model is chosen from the dimension of the existing Qdrant collection
(384 -> all-MiniLM-L6-v2, 768 -> bge-base-en-v1.5, 1024 -> bge-large-en-v1.5), exactly like the
notebook. BGE models get the query-instruction prefix on queries only.
"""
import logging
import threading

from . import config

logger = logging.getLogger(__name__)

EMBED_MODEL_MAP = {
    384: "sentence-transformers/all-MiniLM-L6-v2",
    768: "BAAI/bge-base-en-v1.5",
    1024: "BAAI/bge-large-en-v1.5",
}

_state = {}
_lock = threading.Lock()


def get_qdrant_client():
    from qdrant_client import QdrantClient

    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY are not set.")
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)


def init_retrieval():
    """Connect to Qdrant, read the collection config and load the matching embedding model.
    Safe to call repeatedly; runs once."""
    if _state.get("ready"):
        return _state
    with _lock:
        if _state.get("ready"):
            return _state

        import torch
        from qdrant_client import models
        from sentence_transformers import SentenceTransformer

        qdrant = get_qdrant_client()
        name = config.QDRANT_COLLECTION_NAME
        if not qdrant.collection_exists(name):
            raise RuntimeError(
                f"Qdrant collection '{name}' does not exist. "
                "Run `python -m scripts.build_knowledge_base` first."
            )

        vector_name = None
        vectors_cfg = qdrant.get_collection(name).config.params.vectors
        if isinstance(vectors_cfg, dict):  # named vectors
            vector_name = next(iter(vectors_cfg))
            index_dim = vectors_cfg[vector_name].size
        else:
            index_dim = vectors_cfg.size

        if index_dim not in EMBED_MODEL_MAP:
            raise ValueError(
                f"No embedding model mapped for collection dimension {index_dim}. "
                f"Supported dimensions: {list(EMBED_MODEL_MAP.keys())}."
            )

        # Payload indexes so the source_type / section filters are fast (safe to re-run)
        for field in ("source_type", "section"):
            try:
                qdrant.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.info("Payload index for '%s' skipped: %s", field, e)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = EMBED_MODEL_MAP[index_dim]
        embed_model = SentenceTransformer(model_name, device=device)
        logger.info("Retrieval ready: collection=%s dim=%s model=%s device=%s",
                    name, index_dim, model_name, device)

        _state.update(
            qdrant=qdrant,
            vector_name=vector_name,
            index_dim=index_dim,
            embed_model=embed_model,
            embed_model_name=model_name,
            query_prefix=(
                "Represent this sentence for searching relevant passages: "
                if "bge" in model_name.lower() else ""
            ),
            ready=True,
        )
    return _state


def embed_passages(texts, batch_size=32):
    s = init_retrieval()
    return s["embed_model"].encode(
        texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
    ).tolist()


def embed_query(text):
    s = init_retrieval()
    q = f"{s['query_prefix']}{text}"
    return s["embed_model"].encode([q], normalize_embeddings=True)[0].tolist()


def _build_filter(source_filter=None, section_filter=None):
    from qdrant_client import models

    conditions = []
    if source_filter:
        conditions.append(
            models.FieldCondition(key="source_type", match=models.MatchAny(any=source_filter))
        )
    if section_filter:
        conditions.append(
            models.FieldCondition(key="section", match=models.MatchAny(any=section_filter))
        )
    return models.Filter(must=conditions) if conditions else None


def _run_query(vec, top_k, source_filter=None, section_filter=None):
    s = init_retrieval()
    result = s["qdrant"].query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=vec,
        using=s["vector_name"],
        query_filter=_build_filter(source_filter, section_filter),
        limit=top_k,
        score_threshold=config.MIN_RELEVANCE_SCORE,
        with_payload=True,
    )
    return [{"score": p.score, "metadata": p.payload or {}} for p in result.points]


def retrieve_similar_cases(issue_text, top_k=None):
    top_k = top_k or config.RETRIEVAL_TOP_K_CASES
    vec = embed_query(issue_text)
    return _run_query(vec, top_k, source_filter=["case_law"], section_filter=["summary"])


def retrieve_legal_evidence(issue_text, top_k=None):
    top_k = top_k or config.RETRIEVAL_TOP_K_LAW
    vec = embed_query(issue_text)
    return _run_query(
        vec, top_k,
        source_filter=["statute_plain", "statute_scanned", "case_law"],
    )


def format_context_block(passages):
    lines = []
    for i, p in enumerate(passages):
        meta = p["metadata"]
        label = meta.get("case_title") or meta.get("file_name") or meta.get("source_type", "source")
        extra = []
        if meta.get("law"):
            extra.append(f"law: {meta['law']}")
        if meta.get("cited_case"):
            extra.append(f"cites: {meta['cited_case']}")
        if meta.get("decided_date"):
            extra.append(f"decided: {meta['decided_date']}")
        extra_str = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"[{i+1}] Source: {label}{extra_str}\n{meta.get('text', '')}")
    return "\n\n".join(lines)
