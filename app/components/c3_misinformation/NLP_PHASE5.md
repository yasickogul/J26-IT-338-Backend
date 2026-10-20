# C3 Phase 5 Multilingual NLP

This document describes the implemented C3 text-cleaning, language-detection,
and claim-extraction boundaries. It does not describe legal verification as an
NLP extraction result. The repository does not contain the official SRDS v2
document, so the matrix below is based on the Phase 5 requirements supplied to
the implementation task and still requires comparison with the controlled SRDS.

## Architecture

Typed claims follow this staged path:

```text
raw submitted text
  -> conservative clean_text (typed mode)
  -> LaBSE tokenizer admission for raw and cleaned text
  -> optional language detection
  -> durable raw/cleaned claim record
  -> optional conservative claim extraction in the worker
  -> production RAG input admission (including an extracted claim)
  -> existing embedding, retrieval, and evidence-based verification
```

OCR claims follow this extraction-only path:

```text
OCR raw text
  -> conservative clean_text (OCR mode)
  -> language detection
  -> conservative claim extraction
  -> image extraction response
```

The image endpoint does not persist or enqueue claims. Its extracted text must
pass the existing LaBSE tokenizer admission rule if a future approved image
lifecycle sends it to embedding/RAG. No image persistence contract is implied.

## Text cleaning (COMP-12 / FR-05)

`pipeline/text_preprocessing.py` provides the shared `clean_text` function.
It validates the input type, applies Unicode NFC, normalizes line endings and
inconsistent whitespace, removes invalid control/surrogate characters, and
preserves meaningful line breaks. It preserves URLs, punctuation, numbers,
amounts, dates, statutory references, legislation names, negations,
conditions, exceptions, and English/Sinhala/Tamil text.

Typed mode does not deduplicate repeated lines. OCR mode is explicitly selected
with `ocr_mode=True`; it only removes an exact line repeated across a form-feed
page boundary. It does not perform spelling correction, infer missing words, or
rewrite uncertain OCR. Raw input remains separately available in the API,
database lifecycle, image result, or extraction result that owns it.

## Language detection (COMP-13 / FR-06)

`pipeline/language_detection.py` combines Unicode script evidence with the
existing local FastText adapter:

- Sinhala-only and Tamil-only native-script text can be detected without a
  statistical model.
- English Latin-script identification requires a configured statistical result.
- Meaningful native-script mixtures require at least two words per represented
  native script.
- Native plus English text requires both structural evidence and an English
  statistical identification for its Latin segment.
- Romanized Sinhala/Tamil remains uncertain because a statistical label alone
  does not prove which romanization was intended.
- Short, corrupted, conflicting, unsupported, model-unavailable, and malformed
  model results produce explicit non-success outcomes.

`FastTextLanguageIdentifier` reads `COMPNET_FASTTEXT_MODEL_PATH`, loads the
model lazily, and is shared as a process singleton across typed, image, and
optional multilingual-RAG workflows. It never downloads a model. Its model
name and uncalibrated score are retained in the detection result. Missing model
files or Python bindings produce safe uncertainty, not a fabricated prediction.

The OCR pack setting (`C3_OCR_LANGUAGES`) controls which Tesseract trained-data
packs OCR may use; it is not a detected text-language result.

## Claim extraction (COMP-14 / FR-03)

`pipeline/claim_extraction.py` retains the existing provider interface and
`GeminiClaimExtractionProvider`. It sends input as untrusted data and requests
one strict JSON object. The only successful outcome containing claim text is
`extracted`. `multiple_claims`, `ambiguous`, and `no_legal_assertion` contain no
selected claim. Provider/configuration/schema failures return `failed` with a
safe category and never fabricate a fallback claim.

Trusted extraction instructions prohibit legal verification and require the
provider to preserve negations, conditions, exceptions, dates, numbers,
amounts, names, and statutory references. Questions, hypotheticals,
speculation, attributed reports, unendorsed quotations, contradictions, and
instructions embedded in input must not be converted into confident claims.
Multiple independent assertions are not merged or arbitrarily selected.

The Gemini adapter supplies strict schema validation, bounded request-token
counting, safe finish-reason handling, and provider error containment. Claim
extraction does not retrieve evidence and is not proof that the claim is true.

## Translation compatibility

Translation is optional and is not inserted into claim extraction. The existing
translation contract can create a separate English retrieval query for a
detected Sinhala or Tamil claim when a provider is explicitly injected.
Original claim text remains authoritative. English is skipped; mixed,
uncertain, and unsupported input is not guessed. Provider failure leaves the
original retrieval query in place and does not manufacture a translation.

The configured worker does not automatically enable a translation provider.
Deployment must deliberately compose the existing multilingual adapter and
provider if an English-only retrieval corpus requires this compatibility path.

## Token admission and NLP-to-RAG handoff

`pipeline/claim_input_tokens.py` loads only locally cached
`sentence-transformers/LaBSE` tokenizer assets. It includes special tokens,
uses the cached `max_seq_length`, disables truncation, and rejects the first
over-limit input explicitly. It never downloads model or tokenizer assets.

Typed submission validates raw and cleaned text before persistence. The
production asynchronous RAG service validates its final claim and any distinct
translated retrieval query, so a worker-extracted claim cannot proceed to
embedding when it exceeds the established LaBSE contract. Missing tokenizer
assets become preparation failures. Vector dimensions, embeddings, retrieval,
and database schemas are outside Phase 5 and remain unchanged.

When worker extraction is enabled, only an `extracted` result is passed to RAG.
Multiple, ambiguous, no-assertion, and failed outcomes terminate processing
safely. When the flag is disabled, the prior cleaned-text worker behavior is
preserved for backward compatibility.

## Configuration

| Setting | Default | Purpose |
|---|---:|---|
| `COMPNET_LANGUAGE_DETECTION_ENABLED` | `false` | Enables typed-submission and optional multilingual-RAG language detection. |
| `COMPNET_CLAIM_EXTRACTION_ENABLED` | `false` | Enables conservative extraction in the queued text-claim worker. |
| `COMPNET_FASTTEXT_MODEL_PATH` | unset | Local FastText model path; no automatic download. |
| `GEMINI_API_KEY` | unset | Required only for a deliberately invoked Gemini extraction/translation/provider flow. |

The extraction-only image endpoint always reports language status and performs
claim extraction because those are part of that endpoint's existing response
contract. Its FastText and Gemini dependencies still fail safely when absent.
The two `COMPNET_*_ENABLED` defaults remain unchanged to preserve typed-claim
and worker behavior.

Required Python packages are declared in the root requirements files:

- `fasttext-wheel` for configured statistical language identification;
- `google-genai` for hosted Gemini provider calls;
- `transformers` and `huggingface-hub` for local-only LaBSE tokenizer loading.

The FastText model, LaBSE tokenizer/model cache, and Tesseract trained data are
deployment assets, not Python packages, and are never downloaded at runtime.

## Failure behavior

- Empty/whitespace typed input is rejected before persistence.
- Invalid Unicode surrogate code points and unnecessary controls are removed;
  unsupported scripts remain available to language detection.
- Missing/malformed FastText produces uncertainty or detector error.
- Missing LaBSE tokenizer assets prevents token admission.
- Gemini configuration, timeout/provider, blocked, truncated, and invalid JSON
  responses do not produce extracted claims.
- Multiple/no-assertion/ambiguous extraction never proceeds as a valid claim
  when worker extraction is enabled.
- No provider exception, API key, prompt, or private input is returned as a
  public error detail.

## SRDS Phase 5 compliance matrix

| Requirement | Status | Evidence and remaining work |
|---|---|---|
| FR-03 — Extract legal claim | IMPLEMENTED | Conservative structured outcomes, Gemini boundary, worker and OCR integration, and no-verification separation are implemented. Accuracy evaluation remains pending. |
| FR-05 — Clean extracted text | IMPLEMENTED | NFC, whitespace/line handling, safe character removal, opt-in exact OCR duplicate removal, and raw-text preservation are implemented. No speculative correction is performed. |
| FR-06 — Detect language | PARTIAL | English/Sinhala/Tamil/mixed logic and FastText integration are implemented; deployment needs a compatible local model and empirical multilingual evaluation. |
| COMP-12 — Text cleaning | IMPLEMENTED | Shared deterministic `clean_text` contract supports typed and OCR modes. |
| COMP-13 — Language detection | PARTIAL | Script and statistical contracts are complete; statistical operation depends on a provisioned model and bindings. |
| COMP-14 — Claim extraction | IMPLEMENTED | Provider-independent contract and Gemini implementation are integrated; provider access and real-world evaluation remain deployment concerns. |

These labels cover implementation, not benchmark accuracy or final legal
verification. At least 20 hand-labelled inputs per target language, mixed and
romanized cases, OCR corruption, extraction safety, and comparative error
analysis remain future evaluation work.

