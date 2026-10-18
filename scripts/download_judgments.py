#!/usr/bin/env python3
"""Download Court of Appeal judgment PDFs listed in the Dataset CSVs.

Usage (from J26-IT-338-Backend/):
  python -m scripts.download_judgments --limit 50
  python -m scripts.download_judgments --limit 50 --include-uncertain
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import httpx

from app.shared.utils.text import sanitize_case_number

BACKEND_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = BACKEND_ROOT.parent
DEFAULT_CIVIL_CSV = APP_ROOT / "Dataset" / "court_of_appeal_civil_cases.csv"
DEFAULT_UNCERTAIN_CSV = APP_ROOT / "Dataset" / "court_of_appeal_civil_cases_needs_review.csv"
DEFAULT_OUT_DIR = BACKEND_ROOT / "data" / "judgments" / "raw"


def load_rows(csv_paths: list[Path], *, civil_only: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_type = (row.get("case_type") or "").strip().lower()
                if civil_only and case_type != "civil":
                    continue
                case_number = (row.get("case_number") or "").strip()
                pdf_url = (row.get("pdf_url") or "").strip()
                if not case_number or not pdf_url:
                    continue
                if case_number in seen:
                    continue
                seen.add(case_number)
                rows.append(row)
    return rows


def download_one(
    client: httpx.Client,
    *,
    url: str,
    dest: Path,
) -> str:
    """Return status: downloaded | skipped | failed."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"

    last_error = ""
    for attempt in range(2):  # try once, retry once, then skip
        try:
            resp = client.get(url, follow_redirects=True)
            if resp.status_code == 404:
                return "failed"
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            if dest.stat().st_size == 0:
                dest.unlink(missing_ok=True)
                last_error = "empty body"
                continue
            return "downloaded"
        except Exception as exc:  # noqa: BLE001 — log & continue
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                time.sleep(1.0)
    print(f"  FAIL {dest.name}: {last_error}")
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CoA judgment PDFs")
    parser.add_argument("--limit", type=int, default=50, help="Max PDFs to process (default 50)")
    parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Also include court_of_appeal_civil_cases_needs_review.csv",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Override civil CSV path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--delay-min", type=float, default=1.0)
    parser.add_argument("--delay-max", type=float, default=2.0)
    args = parser.parse_args()

    csv_paths = [args.csv or DEFAULT_CIVIL_CSV]
    if args.include_uncertain:
        csv_paths.append(DEFAULT_UNCERTAIN_CSV)

    print("CSV paths:")
    for p in csv_paths:
        print(f"  {p}  ({'exists' if p.exists() else 'MISSING'})")

    # When including uncertain file, still keep civil rows from main + uncertain from second.
    # Main file is civil-only; needs_review is uncertain — load both without civil_only filter
    # when --include-uncertain, otherwise civil-only from main CSV.
    if args.include_uncertain:
        rows = load_rows([csv_paths[0]], civil_only=True)
        uncertain = load_rows([csv_paths[1]], civil_only=False)
        # keep uncertain rows (case_type uncertain) only from second file
        uncertain = [r for r in uncertain if (r.get("case_type") or "").lower() == "uncertain"]
        seen = {r["case_number"].strip() for r in rows}
        for r in uncertain:
            cn = r["case_number"].strip()
            if cn not in seen:
                rows.append(r)
                seen.add(cn)
    else:
        rows = load_rows(csv_paths, civil_only=True)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    print(f"Processing {len(rows)} rows → {out_dir}")

    headers = {
        "User-Agent": "J26-IT-338-CorpusBot/0.1 (research; contact=local-dev)",
        "Accept": "application/pdf,*/*",
    }
    with httpx.Client(timeout=60.0, headers=headers) as client:
        for i, row in enumerate(rows, start=1):
            case_number = row["case_number"].strip()
            url = row["pdf_url"].strip()
            dest = out_dir / f"{sanitize_case_number(case_number)}.pdf"
            print(f"[{i}/{len(rows)}] {case_number}")
            status = download_one(client, url=url, dest=dest)
            counts[status] += 1
            if status == "downloaded":
                print(f"  OK → {dest.name}")
            elif status == "skipped":
                print(f"  SKIP (exists) {dest.name}")
            if i < len(rows):
                time.sleep(random.uniform(args.delay_min, args.delay_max))

    print("\n=== Download summary ===")
    print(f"  downloaded : {counts['downloaded']}")
    print(f"  skipped    : {counts['skipped']}")
    print(f"  failed     : {counts['failed']}")
    print(f"  total      : {sum(counts.values())}")
    return 0 if counts["failed"] < len(rows) else 1


if __name__ == "__main__":
    # Allow `python scripts/download_judgments.py` from backend root.
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    raise SystemExit(main())
