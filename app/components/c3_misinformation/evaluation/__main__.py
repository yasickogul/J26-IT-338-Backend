"""Run a fully offline evaluation over supplied queries and recorded outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from .contracts import EvaluationInputError
from .runner import build_report


def _read_json(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    try:
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"{path} is not valid UTF-8 JSON") from exc


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Offline C3 RAG Evaluation",
        "",
        f"- Dataset: `{report['evaluation_scope']['dataset_name']}`",
        f"- Queries: {report['evaluation_scope']['query_count']}",
        f"- Prediction records: {report['retrieval']['prediction_count']}",
        f"- Retrieval results evaluated: {report['retrieval']['evaluated_retrieval_count']}",
        f"- Missing retrieval predictions: {len(report['retrieval']['missing_retrieval_query_ids'])}",
        f"- Valid recorded RAG outputs: {report['rag']['valid_count']}",
        f"- Invalid recorded RAG outputs: {report['rag']['invalid_count']}",
        "",
        "## Retrieval",
        "",
        "| K | Target hits / evaluated | Target Hit@K | Target MRR@K | Complete labelled-ID conditions / evaluated |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k, metric in report["retrieval"]["metrics"].items():
        hit = metric["target_hit_at_k"]
        success = metric["labelled_condition_id_coverage_at_k"]
        mrr = metric["target_mrr_at_k"]
        lines.append(
            f"| {k} | {hit['hits']} / {metric['denominator']} | "
            f"{hit['rate']} | {mrr} | "
            f"{success['queries_with_complete_condition_ids']} / {metric['denominator']} |"
        )
    lines.extend([
        "",
        "These metrics evaluate retrieval labels only. They do not establish legal correctness.",
        "Citation support requires manual review. Unsupported metrics and their reasons are listed in the JSON report.",
        "",
    ])
    return "\n".join(lines)


def _write_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True, help="Reviewed query-label JSON")
    parser.add_argument("--predictions", type=Path, required=True, help="Recorded prediction JSON")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; must not exist")
    parser.add_argument("--markdown", type=Path, help="Optional new Markdown report path; must not exist")
    parser.add_argument("--k", type=int, action="append", dest="ks",
                        help="Metric K; repeat to select multiple values (default: 1, 3, 5)")
    args = parser.parse_args(argv)
    ks = args.ks or [1, 3, 5]

    input_paths = {args.queries.resolve(), args.predictions.resolve()}
    output_paths = {args.output.resolve()}
    if args.markdown:
        output_paths.add(args.markdown.resolve())
    if input_paths & output_paths:
        parser.error("output paths must not overwrite input files")
    if len(output_paths) != 1 + bool(args.markdown):
        parser.error("JSON and Markdown output paths must be distinct")
    if any(path.exists() for path in output_paths):
        parser.error("output paths must not already exist")

    try:
        query_dataset, query_hash = _read_json(args.queries)
        predictions_document, predictions_hash = _read_json(args.predictions)
        report = build_report(
            query_dataset,
            predictions_document,
            query_sha256=query_hash,
            predictions_sha256=predictions_hash,
            ks=ks,
        )
        json_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        markdown_text = _markdown(report) if args.markdown else None
        _write_exclusive(args.output, json_text)
        if args.markdown and markdown_text is not None:
            _write_exclusive(args.markdown, markdown_text)
    except (OSError, EvaluationInputError, TypeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    print(f"JSON_REPORT={args.output}")
    if args.markdown:
        print(f"MARKDOWN_REPORT={args.markdown}")
    print(f"QUERIES={report['evaluation_scope']['query_count']}")
    print(f"RETRIEVALS_EVALUATED={report['retrieval']['evaluated_retrieval_count']}")
    print(f"MISSING_RETRIEVALS={len(report['retrieval']['missing_retrieval_query_ids'])}")
    print("LIVE_INFERENCE_OR_DATABASE=not_used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
