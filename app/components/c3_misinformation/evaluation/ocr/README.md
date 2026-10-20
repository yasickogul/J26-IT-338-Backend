# OCR benchmark fixture contract

No reference images or ground-truth transcriptions are currently committed, so
no English, Sinhala, Tamil, or mixed-language OCR accuracy result is reported.

Place reviewed evaluation material under this directory using the following
layout when the project obtains permission to use it:

```text
ocr/
  fixtures/
    images/          # JPEG/PNG reference inputs
    manifest.jsonl   # one reviewed ground-truth record per image
  results/           # generated reports; do not commit sensitive raw images
```

Each `manifest.jsonl` record should contain:

```json
{
  "image": "images/example.png",
  "languages": ["eng", "sin", "tam"],
  "ground_truth": "reviewed transcription",
  "expected_reading_order": ["line one", "line two"],
  "required_tokens": ["30", "Section 10", "not"],
  "license_or_consent": "documented source"
}
```

Do not add fabricated legal content or inferred transcriptions. Remove or
anonymize personal information before accepting a fixture.

For every engine/configuration combination, record:

- engine and model/trained-data versions;
- operating system and Python version;
- preprocessing settings and OCR language configuration;
- extracted text, reading order, and provider confidence;
- character error rate and word error rate against reviewed ground truth;
- required-token recall for numbers, dates, statutory references, conditions,
  and negations;
- per-language results and mixed-script results;
- runtime and peak memory for each image.

Compare Tesseract, EasyOCR, and PaddleOCR only after confirming that the exact
installed models support every evaluated script. Select an engine only from the
recorded dataset and results; configuration strings or advertised language lists
are not accuracy evidence.
