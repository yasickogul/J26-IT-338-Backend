`-- Read-only C4 verification queries. These do not modify Neon.

SELECT COUNT(*) AS total_imported_cases
FROM c_argumentation.court_cases;

SELECT source_csv, COUNT(*) AS imported_cases
FROM c_argumentation.court_cases
GROUP BY source_csv
ORDER BY source_csv;

SELECT COUNT(*) AS cases_with_pdf_urls
FROM c_argumentation.court_cases
WHERE pdf_url IS NOT NULL AND btrim(pdf_url) <> '';

SELECT source_document_key, case_number, date_decided, pdf_url, COUNT(*) AS duplicate_count
FROM c_argumentation.court_cases
GROUP BY source_document_key, case_number, date_decided, pdf_url
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, case_number;

-- Invalid rows remain in the dry-run report and are not imported.
SELECT COUNT(*) AS imported_rows_with_missing_case_number
FROM c_argumentation.court_cases
WHERE case_number IS NULL OR btrim(case_number) = '';
