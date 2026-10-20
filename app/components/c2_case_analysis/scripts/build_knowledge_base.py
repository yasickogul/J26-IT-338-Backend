"""One-off / admin script: build (or refresh) the Qdrant knowledge base.
Reproduces notebook cells 11, 12, 15, 16 and 17.

    cd J26-IT-338-Backend
    python -m app.components.c2_case_analysis.scripts.build_knowledge_base --plain-dir /path/to/law-docs-054 \
                                           --scanned-dir /path/to/law-docs-scanned

The web app does NOT need this if the `law-rag-kb` collection already exists in Qdrant Cloud.
Re-running is safe: point IDs are deterministic UUIDs (uuid5), so records are overwritten, not duplicated.

NOTE: chunking uses the fixed chunker from app/chunking.py. If your existing collection was built with
the notebook's version, it may contain many near-duplicate tail chunks; rebuilding cleans that up.
"""
import argparse
import uuid
from pathlib import Path

from qdrant_client import models
from tqdm.auto import tqdm

from .. import config, retrieval
from ..chunking import chunk_for_retrieval
from ..ocr import extract_text_from_pdf
from ..utils import is_civil, safe_list

NEW_COLLECTION_DIM = 768   # used ONLY if the collection has to be created
UPSERT_BATCH = 64


def build_pdf_records(directory, source_label):
    records = []
    pdf_paths = list(Path(directory).rglob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs in {directory}")
    for path in tqdm(pdf_paths, desc=f"Extracting {source_label}"):
        try:
            text = extract_text_from_pdf(str(path))
        except Exception as e:
            print(f"Failed to extract {path.name}: {e}")
            continue
        for i, chunk in enumerate(chunk_for_retrieval(text)):
            records.append({
                "id": f"{source_label}-{path.stem}-{i}",
                "text": chunk,
                "metadata": {
                    "source_type": source_label,
                    "file_name": path.name,
                    "chunk_index": i,
                },
            })
    return records


def build_case_law_records(frame):
    records = []
    for _, r in frame.iterrows():
        case_no = str(r.get("case_no", "")).strip()
        base_meta = {
            "source_type": "case_law",
            "case_no": case_no,
            "case_title": str(r.get("case_title", "")).strip(),
            "court": str(r.get("court", "")).strip(),
            "decided_date": str(r.get("decided_date", "")).strip(),
        }

        full_summary = str(r.get("full_summary", "") or "")
        for i, chunk in enumerate(chunk_for_retrieval(full_summary)):
            records.append({
                "id": f"caselaw-{case_no}-summary-{i}",
                "text": chunk,
                "metadata": {**base_meta, "section": "summary"},
            })

        for i, item in enumerate(safe_list(r["laws_referred"])):
            text = f"{item.get('law','')} {item.get('chapter_or_section','')}: {item.get('explanation','')}".strip()
            if text:
                records.append({
                    "id": f"caselaw-{case_no}-law-{i}",
                    "text": text,
                    "metadata": {**base_meta, "section": "law_reference", "law": item.get("law", "")},
                })

        for i, item in enumerate(safe_list(r["citations"])):
            text = f"{item.get('title','')}: {item.get('reason_cited','')}".strip()
            if text:
                records.append({
                    "id": f"caselaw-{case_no}-citation-{i}",
                    "text": text,
                    "metadata": {**base_meta, "section": "citation", "cited_case": item.get("title", "")},
                })

        for i, item in enumerate(safe_list(r["judgements"])):
            text = f"{item.get('title','')}: {item.get('description','')}".strip()
            if text:
                records.append({
                    "id": f"caselaw-{case_no}-judgement-{i}",
                    "text": text,
                    "metadata": {**base_meta, "section": "judgement"},
                })
    return records


def load_civil_case_frame():
    from datasets import load_dataset

    raw_ds = load_dataset("navodPeiris/sri-lanka-case-law", split="SLR", token=config.HF_TOKEN or None)
    case_df = raw_ds.to_pandas()
    case_df["is_civil"] = case_df.apply(is_civil, axis=1)
    civil_case_df = case_df[case_df["is_civil"]].reset_index(drop=True)
    print(f"Civil cases for KB: {len(civil_case_df)} / {len(case_df)}")
    return civil_case_df


def to_point_id(record_id):
    """Qdrant IDs must be UUIDs or unsigned ints, so derive a deterministic UUID from our string ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))


def upsert_records(records, batch_size=UPSERT_BATCH):
    state = retrieval.init_retrieval()
    qdrant, vector_name = state["qdrant"], state["vector_name"]
    for i in tqdm(range(0, len(records), batch_size), desc="Embedding + upserting"):
        batch = records[i:i + batch_size]
        vectors = retrieval.embed_passages([r["text"] for r in batch])
        points = []
        for r, vec in zip(batch, vectors):
            payload = dict(r["metadata"])
            payload["text"] = r["text"][:4000]  # keep raw text for retrieval display
            payload["record_id"] = r["id"]      # original readable ID
            points.append(models.PointStruct(
                id=to_point_id(r["id"]),
                vector={vector_name: vec} if vector_name else vec,
                payload=payload,
            ))
        qdrant.upsert(collection_name=config.QDRANT_COLLECTION_NAME, points=points, wait=True)


def main():
    ap = argparse.ArgumentParser(description="Build the law-rag-kb Qdrant collection.")
    ap.add_argument("--plain-dir", help="Folder of text-based statute PDFs (source_type=statute_plain)")
    ap.add_argument("--scanned-dir", help="Folder of scanned statute PDFs (source_type=statute_scanned)")
    ap.add_argument("--skip-case-law", action="store_true", help="Do not index the civil case-law dataset")
    args = ap.parse_args()

    qdrant = retrieval.get_qdrant_client()
    name = config.QDRANT_COLLECTION_NAME
    if not qdrant.collection_exists(name):
        qdrant.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=NEW_COLLECTION_DIM, distance=models.Distance.COSINE),
        )
        print(f"Created collection '{name}' (dim={NEW_COLLECTION_DIM}, cosine)")

    retrieval.init_retrieval()  # reads dim, picks the embedding model, creates payload indexes

    records = []
    if args.plain_dir:
        records += build_pdf_records(args.plain_dir, "statute_plain")
    if args.scanned_dir:
        records += build_pdf_records(args.scanned_dir, "statute_scanned")
    if not args.skip_case_law:
        records += build_case_law_records(load_civil_case_frame())

    print(f"Total records to index: {len(records)}")
    upsert_records(records)
    print("Upsert complete.")
    print("Points in collection:", qdrant.count(name, exact=True).count)


if __name__ == "__main__":
    main()
