"""Validated CompNet artifact ingestion. CLI is dry-run only; no model loading."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

MODEL = "sentence-transformers/LaBSE"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expected_inputs(chunks):
    """Exact reviewed CompNet preparation, without tokenizer/model execution."""
    result = []
    for chunk in chunks:
        if chunk["chunk_id"] != "CON-001-0004":
            result.append(dict(deepcopy(chunk), embedding_text=chunk["text"]))
            continue
        original = chunk["text"]
        a = original.index("(a) where such person")
        b = original.index("(b) where such person", a)
        intro = original[:a]
        meta = chunk["metadata"]
        heading = f"{meta['title']} | {meta['section_label']} | inserted subsection (4A) | "
        for suffix, passage, label in (("a", original[a:b], "(a)"), ("b", original[b:], "(b)")):
            result.append(dict(
                chunk_id=chunk["chunk_id"] + "-" + suffix,
                original_chunk_id=chunk["chunk_id"], text=intro + passage,
                embedding_text=heading + intro + passage, metadata=deepcopy(meta),
                paragraph=label, parent_context=dict(
                    section_label=meta["section_label"], inserted_subsection="(4A)",
                    introduction=intro, original_chunk_id=chunk["chunk_id"]),
            ))
    return result


def validate_artifacts(*, manifest_path, input_path, corpus_path, vector_path):
    """Validate explicit files; manifest paths are never followed implicitly."""
    manifest, prepared, corpus = map(_json, (manifest_path, input_path, corpus_path))
    if manifest.get("model_name") != MODEL or manifest.get("dimension") != 768:
        raise ValueError("Manifest must identify LaBSE/768")
    if prepared.get("model_name") != MODEL or prepared.get("dimension") != 768:
        raise ValueError("Prepared inputs must identify LaBSE/768")
    if (manifest.get("input_file_sha256") != _sha(input_path)
            or manifest.get("original_corpus_sha256") != _sha(corpus_path)):
        raise ValueError("Manifest hashes do not match supplied artifacts")
    chunks = corpus["chunks"] if isinstance(corpus, dict) else corpus
    inputs = prepared["inputs"]
    ids = [x["chunk_id"] for x in inputs]
    originals = [x["chunk_id"] for x in chunks]
    if (not ids or len(set(ids)) != len(ids) or len(set(originals)) != len(originals)
            or any(not isinstance(x, str) or not x.strip() for x in ids + originals)):
        raise ValueError("Chunk identifiers must be unique nonblank strings")
    if inputs != _expected_inputs(chunks):
        raise ValueError("Prepared inputs differ from reviewed corpus transformation")
    if manifest.get("count") != len(inputs) or prepared.get("derived_input_count") != len(inputs):
        raise ValueError("Artifact counts disagree")
    with np.load(vector_path, allow_pickle=False) as archive:
        vectors, vector_ids = archive["embeddings"], archive["chunk_ids"]
    if (vectors.shape != (len(inputs), 768) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)):
        raise ValueError("Vectors must be finite normalized float32 LaBSE/768")
    if vector_ids.tolist() != ids:
        raise ValueError("Vector IDs/order differ from prepared inputs")
    documents, positions, rows = {}, Counter(), []
    for item, vector in zip(inputs, vectors, strict=True):
        meta = item["metadata"]
        if not isinstance(meta, dict):
            raise ValueError("Metadata must be an object")
        for key in ("source_id", "title", "section_label", "language", "legal_domain",
                    "document_type", "source_file"):
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                raise ValueError(f"Missing citation metadata: {key}")
        for field in ("text", "embedding_text"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Invalid {field}")
        if item.get("parent_context") is not None and not isinstance(item["parent_context"], dict):
            raise ValueError("Invalid parent_context")
        if item.get("paragraph") is not None and not isinstance(item["paragraph"], str):
            raise ValueError("Invalid paragraph")
        source = meta["source_id"]
        doc_id = uuid5(NAMESPACE_URL, "shared-corpus:" + source)
        source_type = {"Principal Enactment": "act", "Amendment Act": "act",
                       "act": "act", "judgment": "judgment", "regulation": "regulation"}.get(meta["document_type"])
        if source_type is None:
            raise ValueError("Unmapped document_type")
        doc = dict(id=doc_id, source_id=source, source_type=source_type,
                   title=meta["title"], citation=meta.get("act_number"),
                   language=meta["language"])
        if source in documents and documents[source] != doc:
            raise ValueError("Conflicting document metadata for source_id")
        documents[source] = doc
        rows.append(dict(
            document_id=doc_id, chunk_id=item["chunk_id"],
            original_chunk_id=item.get("original_chunk_id", item["chunk_id"]),
            chunk_index=positions[source], section=meta["section_label"],
            legal_text=item["text"], embedding_text=item["embedding_text"],
            metadata=json.dumps(meta, ensure_ascii=False, allow_nan=False),
            parent_context=(json.dumps(item["parent_context"], ensure_ascii=False, allow_nan=False)
                            if item.get("parent_context") is not None else None),
            paragraph=item.get("paragraph"), embedding_model=MODEL,
            embedding="[" + ",".join(repr(float(v)) for v in vector) + "]",
        ))
        positions[source] += 1
    summary = dict(documents=len(documents), chunks=len(rows), model=MODEL, dimension=768,
                   vector_sha256=_sha(vector_path), input_sha256=_sha(input_path),
                   corpus_sha256=_sha(corpus_path), dry_run=True)
    return list(documents.values()), rows, summary


async def import_artifacts(*, session: AsyncSession, **paths):
    """Explicit first-load operation only. No updates/upserts; idle session required.

    Caller supplies the authorized target session. Validation precedes DB access.
    Own one transaction; locks prevent concurrent imports racing empty checks.
    Existing corpus data causes rejection, not replacement. Session remains open.
    """
    documents, rows, summary = validate_artifacts(**paths)
    if session.in_transaction():
        raise ValueError("Import requires an idle dedicated session")
    async with session.begin():
        await session.execute(text(
            "LOCK TABLE shared.legal_documents, shared.legal_chunks IN SHARE ROW EXCLUSIVE MODE"))
        count = await session.execute(text(
            "SELECT (SELECT count(*) FROM shared.legal_documents) + "
            "(SELECT count(*) FROM shared.legal_chunks)"))
        if count.scalar_one() != 0:
            raise ValueError("Refusing to import into a nonempty corpus")
        await session.execute(text("""
            INSERT INTO shared.legal_documents
                (id, source_id, source_type, title, citation, language)
            VALUES (:id, :source_id, :source_type, :title, :citation, :language)
        """), documents)
        await session.execute(text("""
            INSERT INTO shared.legal_chunks
                (document_id, chunk_id, original_chunk_id, chunk_index, section,
                 text, embedding_text, metadata, parent_context, paragraph, embedding_model, embedding)
            VALUES (:document_id, :chunk_id, :original_chunk_id, :chunk_index, :section,
                    :legal_text, :embedding_text, CAST(:metadata AS jsonb),
                    CAST(:parent_context AS jsonb), :paragraph, :embedding_model, CAST(:embedding AS vector))
        """), rows)
    return dict(summary, dry_run=False)


def main():
    parser = argparse.ArgumentParser(description="Offline corpus validation; never writes to a database")
    for name in ("manifest", "input", "corpus", "vector"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    paths = {name + "_path": getattr(args, name) for name in ("manifest", "input", "corpus", "vector")}
    _, _, summary = validate_artifacts(**paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
