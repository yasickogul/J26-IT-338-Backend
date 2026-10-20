"""Validate and import C4 court-case CSV records into Neon."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse, urlunparse

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from ..models import CourtCase as CourtCaseModel

EXPECTED_COLUMNS = (
    "date_decided", "case_number", "parties", "judge", "keywords",
    "legislation", "case_type", "pdf_url",
)
_CASE_NUMBER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .()_-]*/[A-Za-z0-9 .()_-]+(?:/.*)?$")
_HEADER_RE = re.compile(r"[^a-z0-9]+")


class CSVCaseRecord(TypedDict, total=False):
    date_decided: date | None
    case_number: str
    source_document_key: str
    parties: str | None
    judge: str | None
    keywords: str | None
    legislation: str | None
    case_type: str | None
    pdf_url: str | None
    source_csv: str


@dataclass(slots=True)
class InvalidRow:
    source_file: str
    row_number: int
    errors: list[str]
    missing_fields: list[str]
    case_number: str | None = None


@dataclass(slots=True)
class ValidationReport:
    files: list[str] = field(default_factory=list)
    detected_columns: dict[str, list[str]] = field(default_factory=dict)
    total: int = 0
    valid: int = 0
    invalid: int = 0
    duplicates: int = 0
    missing_values: int = 0
    invalid_rows: list[InvalidRow] = field(default_factory=list)
    duplicate_source_documents: dict[str, list[str]] = field(default_factory=dict)
    repeated_case_numbers: dict[str, list[str]] = field(default_factory=dict)
    missing_by_field: Counter[str] = field(default_factory=Counter)
    invalid_by_field: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_by_field"] = dict(self.missing_by_field)
        payload["invalid_by_field"] = dict(self.invalid_by_field)
        return payload


def _normalize_header(value: str | None) -> str:
    return _HEADER_RE.sub("_", (value or "").strip().lstrip("\ufeff")).strip("_").lower()


def _parse_date(value: str) -> date:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError("expected YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY or YYYY/MM/DD")


def _normalize_url(value: str) -> str:
    parsed = urlparse(value.strip())
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.params, parsed.query, ""))


def build_source_document_key(case_number: str, decision_date: date | None, pdf_url: str | None) -> str:
    """Return a stable identity for one source judgment document."""
    material = "|".join((case_number.strip().upper(), decision_date.isoformat() if decision_date else "", _normalize_url(pdf_url or "")))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        raw_headers = reader.fieldnames or []
        headers = [_normalize_header(header) for header in raw_headers]
        if not raw_headers:
            raise ValueError("CSV has no header row")
        if len(headers) != len(set(headers)):
            raise ValueError("CSV contains duplicate column names after normalization")
        rows = [{_normalize_header(k): (v or "").strip() for k, v in row.items() if k is not None} for row in reader]
    return rows, headers


def _candidate_files(paths: Sequence[str | Path] | None) -> list[Path]:
    if paths:
        candidates = [Path(path) for path in paths]
    else:
        component_dir = Path(__file__).resolve().parents[1] / "data" / "csv"
        backend_dir = Path(__file__).resolve().parents[4] / "data" / "csv"
        search_dir = component_dir if component_dir.exists() else backend_dir
        candidates = sorted(search_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No CSV files found")
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CSV file(s) not found: {', '.join(missing)}")
    return candidates


def validate_csv_files(paths: Sequence[str | Path] | None = None) -> tuple[ValidationReport, list[CSVCaseRecord]]:
    """Validate CSV files and retain every distinct source document."""
    files = _candidate_files(paths)
    report = ValidationReport(files=[str(path) for path in files])
    records: list[CSVCaseRecord] = []
    seen_keys: dict[str, list[str]] = defaultdict(list)
    seen_case_numbers: dict[str, list[str]] = defaultdict(list)

    for path in files:
        rows, headers = _read_rows(path)
        report.detected_columns[path.name] = headers
        for row_number, row in enumerate(rows, start=2):
            report.total += 1
            errors: list[str] = []
            missing = [field for field in EXPECTED_COLUMNS if not row.get(field)]
            for field in missing:
                report.missing_by_field[field] += 1
            report.missing_values += int(bool(missing))
            case_number = row.get("case_number", "")
            if not case_number:
                errors.append("case_number is required")
                report.invalid_by_field["case_number"] += 1
            elif not _CASE_NUMBER_RE.fullmatch(case_number):
                errors.append("invalid case_number format")
                report.invalid_by_field["case_number"] += 1

            decision_date: date | None = None
            if row.get("date_decided"):
                try:
                    decision_date = _parse_date(row["date_decided"])
                except ValueError as exc:
                    errors.append(f"invalid date_decided: {exc}")
                    report.invalid_by_field["date_decided"] += 1
            if row.get("pdf_url"):
                parsed = urlparse(row["pdf_url"])
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    errors.append("invalid pdf_url: expected an absolute http(s) URL")
                    report.invalid_by_field["pdf_url"] += 1

            if errors:
                report.invalid += 1
                report.invalid_rows.append(InvalidRow(path.name, row_number, errors, missing, case_number or None))
                continue

            key = build_source_document_key(case_number, decision_date, row.get("pdf_url") or None)
            seen_case_numbers[case_number].append(key)
            if key in seen_keys:
                report.duplicates += 1
                seen_keys[key].append(path.name)
                continue
            seen_keys[key].append(path.name)
            records.append(CSVCaseRecord(
                date_decided=decision_date,
                case_number=case_number,
                source_document_key=key,
                parties=row.get("parties") or None,
                judge=row.get("judge") or None,
                keywords=row.get("keywords") or None,
                legislation=row.get("legislation") or None,
                case_type=row.get("case_type") or None,
                pdf_url=row.get("pdf_url") or None,
                source_csv=path.name,
            ))

    report.valid = len(records)
    report.duplicate_source_documents = {key: names for key, names in seen_keys.items() if len(names) > 1}
    report.repeated_case_numbers = {number: keys for number, keys in seen_case_numbers.items() if len(keys) > 1}
    return report, records


async def apply_migration(engine: AsyncEngine, migration_path: str | Path) -> None:
    """Apply the C4 forward-only SQL migration in one transaction."""
    sql = "\n".join(
        line for line in Path(migration_path).read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    )
    statements = [part.strip() for part in sql.split(";") if part.strip()]
    async with engine.begin() as connection:
        for statement in statements:
            await connection.execute(text(statement))


async def import_validated_cases(session: AsyncSession, records: Iterable[CSVCaseRecord], *, batch_size: int = 250) -> dict[str, int]:
    """Upsert source documents in batches; caller owns the transaction."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    rows = list(records)
    stats = {"inserted": 0, "updated": 0, "skipped": 0, "failed": 0}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        keys = [row["source_document_key"] for row in batch]
        existing = set((await session.execute(select(CourtCaseModel.source_document_key).where(CourtCaseModel.source_document_key.in_(keys)))).scalars().all())
        statement = insert(CourtCaseModel).values(batch)
        excluded = statement.excluded
        statement = statement.on_conflict_do_update(
            index_elements=[CourtCaseModel.source_document_key],
            set_={
                "case_number": func.coalesce(func.nullif(excluded.case_number, ""), CourtCaseModel.case_number),
                "date_decided": func.coalesce(excluded.date_decided, CourtCaseModel.date_decided),
                "parties": func.coalesce(excluded.parties, CourtCaseModel.parties),
                "judge": func.coalesce(excluded.judge, CourtCaseModel.judge),
                "keywords": func.coalesce(excluded.keywords, CourtCaseModel.keywords),
                "legislation": func.coalesce(excluded.legislation, CourtCaseModel.legislation),
                "case_type": func.coalesce(excluded.case_type, CourtCaseModel.case_type),
                "pdf_url": func.coalesce(excluded.pdf_url, CourtCaseModel.pdf_url),
                "source_csv": func.coalesce(excluded.source_csv, CourtCaseModel.source_csv),
                "updated_at": func.now(),
            },
        )
        await session.execute(statement)
        stats["updated"] += len(existing)
        stats["inserted"] += len(batch) - len(existing)
    return stats


async def execute_import(paths: Sequence[str | Path] | None, migration_path: str | Path, batch_size: int) -> dict[str, Any]:
    from app.core.database import DirectSessionLocal, direct_engine

    report, records = validate_csv_files(paths)
    if direct_engine is None or DirectSessionLocal is None:
        raise RuntimeError("DATABASE_URL/DATABASE_URL_DIRECT is not configured")
    await apply_migration(direct_engine, migration_path)
    async with DirectSessionLocal() as session:
        async with session.begin():
            import_stats = await import_validated_cases(session, records, batch_size=batch_size)
    import_stats["skipped"] = report.duplicates
    import_stats["validated"] = report.valid
    import_stats["validation_invalid"] = report.invalid
    return {"validation": report.as_dict(), "import": import_stats}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or import C4 court-case CSV files")
    parser.add_argument("csv", nargs="*", help="CSV files; defaults to the C4/backend data directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true", help="apply migration and import using configured Neon")
    parser.add_argument("--migration", default=str(Path(__file__).resolve().parents[1] / "migrations" / "versions" / "001_create_c_argumentation.sql"))
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    if args.dry_run:
        report, _records = validate_csv_files(args.csv or None)
        print(json.dumps(report.as_dict(), default=str, indent=2))
        return 0 if report.invalid == 0 else 1
    print(json.dumps(asyncio.run(execute_import(args.csv or None, args.migration, args.batch_size)), default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
