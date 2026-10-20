# C3 image claim extraction setup

## Status and architecture

The C3 OCR adapter uses Tesseract with the `eng`, `sin`, and `tam` trained-data
packs. OCR confidence is derived from Tesseract word recognition data and does
not represent legal truth, verification accuracy, evidence quality, or Gemini
confidence. No configuration is universally more accurate, so preprocessing,
scale, deskewing, page segmentation, and language selection remain configurable.

Tesseract is a provisional project choice because its local runtime exposes
English, Sinhala, and Tamil trained-data packs and word confidence values. The
SRDS v2 summary supplied to the implementation proposes empirical selection
between EasyOCR/PaddleOCR and lists Tesseract as an alternative. No repository
copy of the official SRDS was available for independent verification, and no
benchmark proves Tesseract more accurate. The project must compare engines on a
reviewed multilingual legal-image dataset and obtain academic approval or revise
the SRDS before treating this engine choice as final. No additional OCR engine
is installed by this implementation.

Observed development images indicated that 2x scaling sometimes recovered
important English legal words, PSM 6 sometimes favored continuous text, PSM 11
sometimes favored separated infographic text, and loading unrelated language
packs could add mixed-script characters. These observations are not benchmark
results and are not generalized accuracy claims.

The C3-owned processing path is:

```text
multipart image
  -> byte limit and JPEG/PNG media-type check
  -> decoded content/type and pixel-limit validation
  -> EXIF orientation, grayscale, optional denoising/deskewing/contrast/scaling
  -> Tesseract OCR with configured language packs and PSM
  -> aggregate and per-line OCR confidence
  -> existing clean_text()
  -> existing detect_language()
  -> existing extract_legal_claim()
  -> structured response
```

This endpoint performs extraction only. It does not classify truth, run legal
verification, write to the database, or retain uploaded image bytes.

## Runtime installation

Install the declared Python dependencies in the project virtual environment:

```bash
python -m pip install -r requirements.txt
```

Install the Tesseract executable separately for the deployment operating system,
then install or copy the official `eng.traineddata`, `sin.traineddata`, and
`tam.traineddata` files into that installation's `tessdata` directory.

On Windows, the executable can be installed with:

```powershell
winget install --id UB-Mannheim.TesseractOCR --exact
```

On Ubuntu or Debian, the executable and language packs can be installed with:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-sin tesseract-ocr-tam
```

Confirm the runtime can find all configured languages:

```powershell
tesseract --list-langs
```

The expected list must include `eng`, `sin`, and `tam`. Configure a nonstandard
executable or `TESSDATA_PREFIX` at the host level according to pytesseract and
Tesseract documentation. The application never downloads models automatically.

Operational settings:

- `C3_OCR_MAX_IMAGE_BYTES` (default `10485760`)
- `C3_OCR_MAX_DECODED_PIXELS` (default `40000000`)
- `C3_OCR_MAX_PROCESSED_PIXELS` (default `40000000`)
- `C3_OCR_IMAGE_SCALE` (`1` or `2`; default `1`)
- `C3_OCR_PSM` (`6` or `11`; default `6`)
- `C3_OCR_LANGUAGES` (default `eng,sin,tam`)
- `C3_OCR_LOW_CONFIDENCE_THRESHOLD` (default `0.60`)
- `C3_OCR_IMPROVE_CONTRAST` (default `true`)
- `C3_OCR_REDUCE_NOISE` (default `false`)
- `C3_OCR_DESKEW` (default `false`)
- `C3_OCR_DESKEW_MAX_ANGLE_DEGREES` (`1` to `10`; default `5.0`)

`C3_OCR_LANGUAGES` accepts one or more comma-separated supported packs, such as
`eng`, `sin`, `tam`, or `eng,sin,tam`. Use the smallest appropriate set because
loading unrelated packs can introduce mixed-script characters. PSM 6 targets a
uniform text block; PSM 11 targets sparse text and separated infographic
sections. A 2x scale can help small text but quadruples the processed pixel count
and is rejected when it exceeds `C3_OCR_MAX_PROCESSED_PIXELS`.

The byte, decoded-pixel, and processed-pixel defaults are operational safety
limits. JPEG and PNG content is inspected independently of the filename,
bounded before processing, protected by Pillow decompression-bomb handling,
orientation-corrected using EXIF, and converted to grayscale for OCR. Uploaded
bytes and processed images are not persisted.

Deskewing is separate from EXIF orientation correction and is disabled by
default. When enabled, it downsamples only for angle estimation to at most 512
pixels on the longest side, tests deterministic 0.5-degree candidates within
the configured bound, and requires at least an 8% horizontal-projection score
improvement. Blank, low-information, over-inked, low-improvement, and
boundary-angle estimates are left unchanged. Accepted corrections use bicubic
rotation with an expanded white canvas to avoid destructive cropping, followed
by another processed-pixel-limit check. Projection heuristics can still fail on
graphics, curved baselines, vertical text, and complex layouts; this must be
evaluated with reviewed fixtures.

If Tesseract is installed outside `PATH`, configure it before the application
starts. One Windows option is:

```powershell
$env:PATH = "C:\Program Files\Tesseract-OCR;$env:PATH"
$env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"
tesseract --list-langs
```

`TESSDATA_PREFIX` must point to the directory containing the configured
`.traineddata` files. Merely configuring `eng,sin,tam` in the application does
not establish that those packs exist or recognize those languages accurately.

## API

The component router exposes:

```text
POST /api/v1/c3/claims/extract-image
Content-Type: multipart/form-data
field name: image
accepted image media types: image/jpeg, image/png
```

Example PowerShell request:

```powershell
curl.exe -X POST http://localhost:8003/api/v1/c3/claims/extract-image `
  -F "image=@C:\path\claim.png;type=image/png"
```

The JSON response contains raw OCR text, explicit OCR status, cleaned text,
extracted legal claim, aggregate and per-line OCR confidence, configured OCR
languages, the low-confidence flag, language detection output, extraction
status, provider methods, and a safe error object.
Extraction outcomes remain `extracted`, `multiple_claims`, `ambiguous`,
`no_legal_assertion`, or `failed`.

Safe failure codes include invalid/oversized/corrupted images, excessive decoded
dimensions, preprocessing failures, unavailable OCR runtime or language packs,
empty OCR output, invalid provider responses, and existing claim-provider
failure categories. Raw third-party exceptions are not returned.

Example response fields (values are illustrative):

```json
{
  "ocr_raw_text": "Section 10 applies.",
  "ocr_status": "completed",
  "cleaned_text": "Section 10 applies.",
  "extracted_legal_claim": "Section 10 applies.",
  "ocr_confidence": 0.91,
  "ocr_line_confidences": [
    {"line_number": 1, "text": "Section 10 applies.", "confidence": 0.91}
  ],
  "ocr_languages": ["eng"],
  "low_confidence": false,
  "detected_language": null,
  "language_detection_status": "uncertain",
  "extraction_status": "extracted",
  "ocr_method": "tesseract",
  "claim_extraction_method": "gemini:model-name",
  "error": null
}
```

Aggregate confidence is the mean of valid Tesseract word confidences. Each line
confidence is the mean of valid words on that line. A line, or the aggregate,
uses `null` when no valid confidence is available. `low_confidence` is true when
aggregate confidence is unavailable or below the configured threshold. Raw OCR
text remains separate from conservative whitespace and Unicode cleaning; no
stage guesses or repairs legal facts.

OCR cleaning applies NFC normalization, conservative whitespace normalization,
meaningful line preservation, and removal of non-whitespace control characters.
It removes an exact duplicate only across an explicit page/form-feed boundary;
ordinary repeated lines are retained because repetition may be legally
meaningful. URLs, punctuation, dates, amounts, section numbers, legislation
titles, negations, conditions, Sinhala, and Tamil are preserved. There are no
automatic spelling or character-substitution heuristics because no substitution
has yet been demonstrated safe for legal OCR; uncertain OCR remains unchanged.

OCR pack selection and detected language are separate metadata. Native Sinhala
and Tamil scripts remain deterministic. English and reliable mixed-language
cases use the existing lazy `FastTextLanguageIdentifier` with the local path in
`COMPNET_FASTTEXT_MODEL_PATH`. The model and Python bindings are loaded only on
first statistical prediction and then reused; nothing is downloaded. Missing
or unusable models preserve OCR output and return an uncertain/detector-error
language status rather than crashing extraction. FastText scores are retained
as uncalibrated diagnostics and no invented threshold is applied.

## Development image inspection

The repository-root `manual_image_test.py` uses the same environment defaults as
the API. OCR-only inspection is the default and does not initialize or call
Gemini:

```bash
python manual_image_test.py /path/to/claim.png --languages eng --scale 2 --psm 11
```

Omit command-line overrides to use the `C3_OCR_*` environment settings. Run the
full OCR-to-claim path only intentionally; it requires Gemini configuration and
may make a billable provider request:

```bash
python manual_image_test.py /path/to/claim.png --full
```

Set `GEMINI_API_KEY` only in the runtime secret store or local untracked
environment file. Never place the key in source control or command output.

The full path performs conservative claim extraction, not legal verification.
Multiple independent assertions return `multiple_claims`; they are not combined
or silently reduced to one claim. Gemini failures retain successful OCR output
and return a safe claim-extraction error.

Environment settings are read when the cached API extractor is first created;
restart the application after changing them. Direct service calls receive
configuration as explicit arguments and do not automatically read environment
variables.

## Verification and limitations

Run the offline implementation tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  app/components/c3_misinformation/tests/test_image_claim_service.py `
  app/components/c3_misinformation/tests/test_image_api.py
```

The unit tests use local image decoding/preprocessing and mocked OCR/claim
providers. They do not measure OCR accuracy. English, Sinhala, Tamil, and mixed
OCR require evaluation against a reviewed reference-image dataset with ground
truth. Known limitations include small or blurred text, decorative fonts,
complex layouts, mixed scripts, low contrast, and incorrect language selection.
Compare 1x/2x and PSM 6/11 by image type rather than assuming one setting wins.

The separate validation stage should run the focused image service and API
contract tests, invalid-configuration cases, real English/Sinhala/Tamil/mixed
fixtures, required-token recall, confidence checks, decompression/resource-limit
cases, and mocked Gemini success/failure paths. The current environment's known
Starlette/TestClient/httpx2 timeout must be resolved or isolated before endpoint
tests can be considered complete. No live Gemini request is required for those
tests.

Deskew validation must include known-angle, no-text, graphics-heavy, boundary,
canvas-expansion, and processed-pixel-limit fixtures. Language validation must
cover configured/unconfigured FastText models, English, native Sinhala/Tamil,
reliable mixed text, and noisy mixed-script OCR without treating model score as
a calibrated confidence.

The existing text claim lifecycle can accept `cleaned_text` from a successful
image response through `POST /api/v1/c3/claims/analyze`. That second request
applies the existing token, language, ownership, persistence, and worker rules.
There is currently no schema that associates the original image and raw OCR text
with that claim, so the image endpoint does not create a claim record itself.

The SRDS-specified image lifecycle adapter remains blocked. Although the C3
claim table permits `input_type='image'`, the repository exposes only
`create_text_claim`, the worker leases only queued text claims, and no approved
C3 contract stores image identity/bytes/path, raw OCR text, OCR confidence, or
OCR provenance. Treating OCR text as an ordinary text claim would discard that
required provenance and misrepresent persistence. Implementing
`POST /api/claims/analyze-image` therefore requires a team-approved C3-owned
media/OCR storage contract, retention policy, image-claim repository operation,
worker input contract, migration decision, and exact router prefix. No dummy ID,
duplicate worker, database write, route, or invented storage path is added here.

See the [OCR benchmark fixture contract](evaluation/ocr/README.md) for the
real-image benchmark procedure.
