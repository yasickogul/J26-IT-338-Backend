# C3 image claim extraction setup

## Status and architecture

The OCR adapter is provisional and replaceable. It uses Tesseract because the
project architecture identifies the `eng`, `sin`, and `tam` trained-data packs,
and Tesseract exposes word confidence values. No accuracy winner is claimed;
real multilingual benchmark images have not yet been run.

The C3-owned processing path is:

```text
multipart image
  -> byte limit and JPEG/PNG media-type check
  -> decoded content/type and pixel-limit validation
  -> EXIF orientation, grayscale, optional contrast/noise processing
  -> replaceable OCR provider
  -> existing clean_text()
  -> existing detect_language()
  -> existing extract_legal_claim()
  -> structured response
```

This endpoint performs extraction only. It does not classify truth, run legal
verification, write to the database, or retain uploaded image bytes.

## Runtime installation

The repository intentionally does not install optional OCR dependencies. In the
project virtual environment, install the Python adapters manually:

```powershell
python -m pip install Pillow pytesseract
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
- `C3_OCR_LANGUAGES` (default `eng,sin,tam`)
- `C3_OCR_LOW_CONFIDENCE_THRESHOLD` (default `0.60`)
- `C3_OCR_IMPROVE_CONTRAST` (default `true`)
- `C3_OCR_REDUCE_NOISE` (default `false`)

The byte and decoded-pixel defaults are operational safety limits chosen by this
implementation; they are not numeric limits specified or approved by the SRDS.

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
extracted legal claim, OCR confidence and configured languages, the low-confidence flag, language
detection status, extraction status, provider methods, and a safe error object.
Extraction outcomes remain `extracted`, `multiple_claims`, `ambiguous`,
`no_legal_assertion`, or `failed`.

Safe failure codes include invalid/oversized/corrupted images, excessive decoded
dimensions, preprocessing failures, unavailable OCR runtime or language packs,
empty OCR output, invalid provider responses, and existing claim-provider
failure categories. Raw third-party exceptions are not returned.

## Verification and limitations

Run the offline implementation tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  app/components/c3_misinformation/tests/test_image_claim_service.py `
  app/components/c3_misinformation/tests/test_image_api.py
```

The unit tests use real local image decoding/preprocessing and mocked OCR/claim
providers. They do not measure OCR accuracy. English, Sinhala, Tamil, and mixed
OCR remain unverified until Tesseract and all language packs are installed and a
reviewed reference-image dataset with ground truth is supplied.

The existing text claim lifecycle can accept `cleaned_text` from a successful
image response through `POST /api/v1/c3/claims/analyze`. That second request
applies the existing token, language, ownership, persistence, and worker rules.
There is currently no schema that associates the original image and raw OCR text
with that claim, so the image endpoint does not create a claim record itself.

See the [OCR benchmark fixture contract](evaluation/ocr/README.md) for the
real-image benchmark procedure.
