"""Validate and explicitly import copied C3 corpus artifacts.

The command-line entry point is validation-only. Database import is available
only to an authorized caller that supplies an idle C3 database session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .reviewed_corpus_artifacts import (
    validate_artifacts as validate_reviewed_artifacts,
)

MODEL_NAME = "sentence-transformers/LaBSE"
IMPORT_AUTHORIZATION = "IMPORT_C3_CORPUS"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "c3_misinformation"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside_data_root(path: str | Path, *, data_root: Path) -> Path:
    root = data_root.resolve(strict=True)
    requested = Path(path)
    candidate = requested if requested.is_absolute() else root / requested
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("C3 corpus files must be inside the configured data directory") from None

    current = candidate
    while True:
        if current.is_symlink():
            raise ValueError("C3 corpus files and parent directories must not be symlinks")
        if current.resolve() == root:
            break
        if current.parent == current:
            raise ValueError("Unable to validate C3 corpus path")
        current = current.parent
    if not resolved.is_file():
        raise ValueError("C3 corpus artifact path must identify a file")
    return resolved


def _source_file(
    source_file: str,
    *,
    data_root: Path,
) -> tuple[str, str]:
    if not isinstance(source_file, str) or not source_file.strip():
        raise ValueError("Each source requires nonblank metadata.source_file")
    if "\\" in source_file:
        raise ValueError("metadata.source_file must use a relative POSIX path")
    relative = PurePosixPath(source_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("metadata.source_file must be relative and cannot traverse directories")
    if relative.parts and relative.parts[0] == "legal_documents":
        requested = Path(*relative.parts)
    else:
        requested = Path("legal_documents", *relative.parts)
    resolved = _inside_data_root(requested, data_root=data_root)
    local_reference = resolved.relative_to(data_root.resolve(strict=True)).as_posix()
    return local_reference, _sha256(resolved)


def _common_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    common = dict(items[0])
    for metadata in items[1:]:
        common = {
            key: value
            for key, value in common.items()
            if key in metadata and metadata[key] == value
        }
    return common


def validate_c3_artifacts(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    manifest_path: str | Path,
    input_path: str | Path,
    corpus_path: str | Path,
    vector_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate copied reviewed artifacts without database or model access."""
    root = Path(data_root).resolve(strict=True)
    paths = {
        name: _inside_data_root(path, data_root=root)
        for name, path in {
            "manifest_path": manifest_path,
            "input_path": input_path,
            "corpus_path": corpus_path,
            "vector_path": vector_path,
        }.items()
    }
    reviewed_documents, reviewed_rows, reviewed_summary = validate_reviewed_artifacts(
        **paths,
    )
    if reviewed_summary.get("model") != MODEL_NAME or reviewed_summary.get("dimension") != 768:
        raise ValueError("C3 corpus artifacts must use sentence-transformers/LaBSE with 768 dimensions")

    metadata_by_source: dict[str, list[dict[str, Any]]] = {}
    chunks: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for row in reviewed_rows:
        metadata = json.loads(row["metadata"])
        source_id = metadata["source_id"]
        metadata_by_source.setdefault(source_id, []).append(metadata)
        if row["chunk_id"] in seen_chunk_ids:
            raise ValueError("Duplicate C3 chunk identifier")
        seen_chunk_ids.add(row["chunk_id"])
        chunks.append({
            "source_id": source_id,
            "chunk_id": row["chunk_id"],
            "original_chunk_id": row["original_chunk_id"],
            "chunk_index": row["chunk_index"],
            "section": row["section"],
            "legal_text": row["legal_text"],
            "embedding_text": row["embedding_text"],
            "metadata": row["metadata"],
            "parent_context": row["parent_context"],
            "paragraph": row["paragraph"],
            "embedding_model": row["embedding_model"],
            "embedding": row["embedding"],
        })

    sources: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for document in reviewed_documents:
        source_id = document["source_id"]
        if source_id in seen_source_ids:
            raise ValueError("Duplicate C3 source identifier")
        seen_source_ids.add(source_id)
        source_metadata = _common_metadata(metadata_by_source[source_id])
        legal_domain = source_metadata.get("legal_domain")
        document_type = source_metadata.get("document_type")
        language = source_metadata.get("language")
        for name, value in (
            ("legal_domain", legal_domain),
            ("document_type", document_type),
            ("language", language),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Source {source_id} has inconsistent or missing {name}")
        local_path, source_sha256 = _source_file(
            source_metadata.get("source_file"), data_root=root,
        )
        source_url = source_metadata.get("source_url")
        citation = document.get("citation")
        if citation is not None and (
            not isinstance(citation, str) or not citation.strip()
        ):
            raise ValueError(f"Source {source_id} has invalid citation")
        if source_url is not None and (
            not isinstance(source_url, str) or not source_url.strip()
        ):
            raise ValueError(f"Source {source_id} has invalid source_url")
        sources.append({
            "source_id": source_id,
            "title": document["title"],
            "legal_domain": legal_domain,
            "document_type": document_type,
            "language": language,
            "citation": citation,
            "source_url": source_url,
            "local_source_path": local_path,
            "source_file_sha256": source_sha256,
            "metadata": json.dumps(source_metadata, ensure_ascii=False, allow_nan=False),
        })

    if {item["source_id"] for item in chunks} != seen_source_ids:
        raise ValueError("C3 source and chunk identifiers are not aligned")
    summary = {
        **reviewed_summary,
        "sources": len(sources),
        "chunks": len(chunks),
        "source_files_validated": len(sources),
        "dry_run": True,
    }
    return sources, chunks, summary


async def import_c3_artifacts(
    *,
    session: AsyncSession,
    authorization: str,
    **paths: Any,
) -> dict[str, Any]:
    """Perform one explicitly authorized first import; never merge or overwrite."""
    if authorization != IMPORT_AUTHORIZATION:
        raise PermissionError("Explicit C3 corpus import authorization is required")
    sources, chunks, summary = validate_c3_artifacts(**paths)
    if session.in_transaction():
        raise ValueError("C3 corpus import requires an idle dedicated session")
    async with session.begin():
        await session.execute(text(
            "LOCK TABLE c3.legal_sources, c3.legal_chunks IN SHARE ROW EXCLUSIVE MODE"
        ))
        existing = await session.execute(text(
            "SELECT (SELECT count(*) FROM c3.legal_sources) + "
            "(SELECT count(*) FROM c3.legal_chunks)"
        ))
        if existing.scalar_one() != 0:
            raise ValueError("Refusing to import into a nonempty C3 legal corpus")
        await session.execute(text("""
            INSERT INTO c3.legal_sources
                (source_id, title, legal_domain, document_type, language, citation,
                 source_url, local_source_path, source_file_sha256, metadata)
            VALUES
                (:source_id, :title, :legal_domain, :document_type, :language,
                 :citation, :source_url, :local_source_path, :source_file_sha256,
                 CAST(:metadata AS jsonb))
        """), sources)
        await session.execute(text("""
            INSERT INTO c3.legal_chunks
                (source_id, chunk_id, original_chunk_id, chunk_index, section,
                 text, embedding_text, metadata, parent_context, paragraph,
                 embedding_model, embedding)
            VALUES
                (:source_id, :chunk_id, :original_chunk_id, :chunk_index, :section,
                 :legal_text, :embedding_text, CAST(:metadata AS jsonb),
                 CAST(:parent_context AS jsonb), :paragraph, :embedding_model,
                 CAST(:embedding AS vector))
        """), chunks)
    return {**summary, "dry_run": False}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate copied C3 legal-corpus artifacts; never accesses a database",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    for name in ("manifest", "input", "corpus", "vector"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    _, _, summary = validate_c3_artifacts(
        data_root=args.data_root,
        manifest_path=args.manifest,
        input_path=args.input,
        corpus_path=args.corpus,
        vector_path=args.vector,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATA_ROOT",
    "IMPORT_AUTHORIZATION",
    "import_c3_artifacts",
    "validate_c3_artifacts",
]
