# Phase 6 — requirements and feature audit

Audit basis: the 76-page [Component 4 SRDS v2.0](../fyp/Component4_Legal_Misinformation_SRDS_V2.pdf), [C2 research component specification](./C2_RESEARCH_COMPONENT_SPEC%20(1).md), [backend architecture](./BACKEND_ARCHITECTURE.md), and actual frontend/backend code. No application-wide SRS or SRDS, nor an official C1/C4 feature-level requirements document, was found in the two repositories. Therefore **whole-application requirements compliance cannot be certified**. C3 IDs below are the real SRDS IDs; no IDs have been invented for other modules.

Status key: **PASS** = acceptance behavior demonstrated within stated test scope; **FAIL** = an implemented path demonstrably misses an acceptance criterion; **NOT IMPLEMENTED** = required path absent; **BLOCKED** = implementation or quality cannot be proven safely with available requirements, environment, data, or authorization. A page rendering, a mocked provider, or a passing unit test is not proof of legal correctness.

## Feature inventory

| Feature / route | Frontend | Backend | Integration and classification |
|---|---|---|---|
| Login / workspace selection (`/`) | Role picker with email/password inputs | No authentication endpoint | **Prototype**: role stored locally; credentials are not checked. Browser navigation PASS; real authentication NOT IMPLEMENTED. |
| Workspace shell / navigation | Role-aware links, theme toggle, language selector, search and notifications | No workspace API | Navigation PASS; language selector, search and notifications are UI-only. |
| Overview (`/overview`) | Sample dashboard cards/statistics | None | Prototype/simulation; not real user metrics. |
| Civil case analysis (`/civil-case-analysis`, `/new`) | Five-step wizard, validation, draft save/reopen, simulated submit | Registered C1 intake/extraction endpoints; registered C2 endpoint only `/ping` | Wizard UI PASS, backend analysis integration FAIL; no request is sent. |
| Legal document analysis (`/legal-document-analysis`, `/new`) | Sample dashboard and five-step simulated wizard | C1 document intake exists, but no frontend call | Prototype/simulation; upload is not wired to backend. |
| C3 misinformation (`/legal-misinformation`) | Real text submit, polling, verdict/evidence UI, OCR extraction, session-history reopen | C3 claim/image/result/history/source APIs and worker | Isolated text claim workflow PASS with deterministic AI mock; genuine RAG/legal quality BLOCKED; image-linked claim FAIL. |
| Analysis history (`/analysis-history`) | Static sample entries | C3 owner-scoped history endpoint | Shared page is a prototype; C3-local history is connected by new C3 UI, browser-mocked test PASS, isolated full-history round trip BLOCKED. |
| Precedent library (`/precedent-library`) | Static judgments and relevance percentages | No registered retrieval endpoint for frontend | Prototype/simulation. |
| Research insights (`/research-insights`) | Illustrative charts/data | No analytics endpoint | Prototype/simulation. |
| Settings (`/settings`) | Local checkboxes/fields and inert save button | No settings endpoint | Prototype/simulation. |
| Settlement simulator (`/settlement-simulator`) | Alias of C4 demo argumentation UI | Registered C4 endpoint only `/ping` | Demo/simulation, no real agent run or persisted output. |
| C1 Stage 2 API | No connected frontend flow | `/api/v1/c1/intake`, `/intake/document`, `/intake/{id}`, `/extract-text`, health | Backend route exists; end-to-end behavior not tested here. Owner: C1. |
| C2 case intelligence / case analysis API | Wizard simulated | Registered `/api/v1/c2/ping`; separate unregistered `c2_case_analysis` router contains analysis routes | Integration NOT IMPLEMENTED in registered app. Owner: C2. |
| C4 argumentation API | Demo UI | Registered `/api/v1/c_argumentation/ping` | Integration NOT IMPLEMENTED. Owner: C4. |

Only the C3 frontend component calls `fetch`; the other workspace modules have no frontend API call. The C3 API prefix is `/api/v1/c3`. Its runtime request is `{text}`; submission returns `claim_id` and `processing`; result lookup is cookie-owner-scoped. The SRDS Section 21 sketches `/api/...` and `{error_code,message}`, whereas the implemented C3 API uses `/api/v1/c3/...` and FastAPI `{detail:{code,...}}`. The current C3 frontend understands the implemented shape; these are **document/API contract deviations** to reconcile deliberately, not a reason to rewrite a working path.

## C3 functional requirement traceability

Evidence abbreviations: **P5** = previous Phase 5 isolated PostgreSQL/worker/browser run with deterministic mocked AI (successful and failed text claims); **B377** = current 377 passing C3 backend tests and 389 subtests; **E12** = current 12 passing Playwright tests, one intentionally skipped; **Code** = inspection only. E12 browser API routes are mocked, not database-backed. “API” below refers to actual C3 endpoints, not the planned SRDS paths.

| Requirement ID | Feature | Frontend | Backend | Integration | Test evidence | Status |
|---|---|---|---|---|---|---|
| FR-01 | Text claim submit | Text form | `/claims/analyze` persists claim | P5 text path | P5, B377, E12 | PASS |
| FR-02 | Image claim submit/result | OCR-to-text then text submit | `/claims/extract-image` does **not** persist image-linked claim | Image association missing | Code, B377 OCR mocks | FAIL |
| FR-03 | Normalized legal claim extraction | Extracted OCR text shown, not final claim | Worker extraction opt-in, default off | Not assured for every claim | Code, B377 | FAIL |
| FR-04 | EN/SI/TA OCR and confidence | File checks, preview, OCR result and low-confidence warning | OCR preprocessing/provider boundary exists | Genuine three-script OCR quality not run | B377 mocks | BLOCKED |
| FR-05 | Text cleaning | Text displayed | `clean_text` used on submit/OCR | Cleaned text persisted | B377 | PASS |
| FR-06 | Detect language | OCR language shown; text flow does not show label | Detection flag defaults off | Not guaranteed on text claims | Code, B377 | FAIL |
| FR-07 | 768-d multilingual embedding | N/A | Local LaBSE loaded in Phase 3; 768-d contract | Embedding interface verified, not live retrieval quality | Phase 3 model check, B377 | PASS |
| FR-08 | Top-K pgvector legal retrieval | Evidence UI | Retrieval adapter and 348-vector corpus artifact | No controlled live run with imported full corpus and genuine LaBSE query | B377, corpus dry-run | BLOCKED |
| FR-09 | Bounded RAG prompt | N/A | Prompt builder/schema tests | Mocked evidence path | B377 | PASS |
| FR-10 | Structured LLM reasoning/retry | Result renderer | Gemini adapter/parser and failure tests | Paid/hosted model not invoked; actual reasoning unverified | B377 | BLOCKED |
| FR-11 | True verdict | Label supported | Contract supports True | Genuine verdict accuracy unverified | B377, E12 mock | BLOCKED |
| FR-12 | False verdict | Label supported | Contract supports False | Genuine verdict accuracy unverified | B377 | BLOCKED |
| FR-13 | Misleading verdict | Label supported | Contract supports Misleading | Genuine verdict accuracy unverified | B377, E12 mock | BLOCKED |
| FR-14 | Numeric confidence 0–1 | No confidence display | `ClaimRagResponse` has no confidence field | No confidence calculation | Code | NOT IMPLEMENTED |
| FR-15 | Evidence-grounded explanation | Explanation rendered | RAG response permits explanation | Semantic grounding/legal accuracy not reviewed | B377, E12 mock | BLOCKED |
| FR-16 | Verdict and extracted claim | Verdict plus submitted/stored text | Result API persists verdict; extracted claim in history | Extracted claim not explicitly displayed with final verdict | Code, E12 | FAIL |
| FR-17 | Explanation in selected language | Workspace language selector is local UI state | Optional translation provider | Selected language not passed to C3 or used to render explanation | Code | FAIL |
| FR-18 | Legal references | Citation IDs/source names rendered | Source metadata/citations in result | Mocked rendering PASS; actual cited legal sources unverified | B377, E12 | BLOCKED |
| FR-19 | Evidence snippet linked to source | Snippets rendered without source links | Source metadata can include URL | User cannot follow displayed source reference | Code | FAIL |
| FR-20 | Source metadata | Title/citation shown | Metadata exists in evidence packet | Act/year/section/date not consistently exposed in UI | Code | FAIL |
| FR-21 | Admin source add/update/deactivate | No admin screen | No authenticated admin source API | Absent | Code | NOT IMPLEMENTED |
| FR-22 | Admin trigger re-index | No admin control | Offline importer/dry-run, no admin update API | Absent | Code | NOT IMPLEMENTED |
| FR-23 | Admin logs/health view | No admin view | No admin log API | Absent | Code | NOT IMPLEMENTED |
| FR-24 | Session claim history and reopen | C3-local history/reopen now implemented | Cookie-owner-scoped `/history` and `/results/{id}` | Browser mocked refresh/reopen PASS; full DB-backed refresh path not rerun | B377, E12 | BLOCKED |
| FR-25 | Graceful pipeline errors | Polling, failed-state and network error display | Structured submission/result/OCR errors and worker failure state | P5 failed provider path; new browser tests | P5, B377, E12 | PASS |
| FR-26 | Persisted submission-to-result system logs | No admin trail | No C3 `system_logs` write path found | Required audit trail absent | Code | NOT IMPLEMENTED |

## C3 nonfunctional and acceptance traceability

| Requirement ID | Feature | Frontend | Backend | Integration | Test evidence | Status |
|---|---|---|---|---|---|---|
| NFR-01 | Text p95 ≤15s on reference hardware | Poll timeout 120s | Worker asynchronous | No reference-hardware benchmark | None | BLOCKED |
| NFR-02 | Image p95 ≤30s | OCR flow | No image-linked terminal workflow | Target unattainable as specified | Code | FAIL |
| NFR-03 | ≥95% well-formed completion | Error UI | Worker retry/failure tests | No representative completion-rate study | B377 | BLOCKED |
| NFR-04 | Demonstration uptime | Route served | Health endpoints | No documented uptime window | None | BLOCKED |
| NFR-05 | Three-person usability | Usable form | N/A | No participant study | None | BLOCKED |
| NFR-06 | Automated accessibility/no critical issues | Keyboard labels and alt text present | N/A | No axe/Lighthouse run | Code only | BLOCKED |
| NFR-07 | Layered maintainability | C3 component isolated | C3 module documented | Ten-minute developer-location criterion not measured | Code | BLOCKED |
| NFR-08 | Moderate corpus growth/index | N/A | pgvector schema/adapters | No scale/index migration trial | Code | BLOCKED |
| NFR-09 | MIME/size and OWASP security | JPEG/PNG 10MB check | OCR upload validation | No OWASP baseline scan | B377 | BLOCKED |
| NFR-10 | Anonymous/minimal data | Prototype role, C3 cookie | Anonymous owner hash | Core text flow needs no account | P5, B377 | PASS |
| NFR-11 | Explanation and reference for evidence | Renderer has fallback text | Response explanation nullable | 100% schema guarantee not proven | Code, B377 | BLOCKED |
| NFR-12 | Reconstruct processing from logs | N/A | `system_logs` trail absent | Absent | Code | NOT IMPLEMENTED |
| NFR-13 | FK integrity | N/A | Claim/result schema | Phase 5 isolated DB proved terminal persistence, not full corpus integrity | P5, B377 | BLOCKED |
| NFR-14 | Chrome/Safari desktop/mobile | Responsive styles | N/A | Chromium only; Safari/mobile not run | E12 | BLOCKED |
| AC-01 | Text submit to result | C3 path | Claim API/worker | P5 isolated deterministic success | P5 | PASS |
| AC-02 | Image submit to result | OCR only | No image-linked claim | Absent end-to-end | Code | FAIL |
| AC-03 | EN/SI/TA submit and result | Language selector inert | Detection opt-in; translation experimental | Real three-language path not demonstrated | B377 mocks | FAIL |
| AC-04 | Relevant evidence in three domains | Evidence UI | Nine documents / 348 prepared vectors available | No imported full corpus retrieval evaluation | Corpus dry-run | BLOCKED |
| AC-05 | Defined verdict category for every processed claim | Renderer supports categories/abstention | Contract and failure state | Representative actual-model runs not done | B377 | BLOCKED |
| AC-06 | Reference whenever evidence exists | Citations rendered | Structural citation checks | Semantic support not established | B377 | BLOCKED |
| AC-07 | Persist claims/results/history | C3 history added | Owner-scoped endpoints | Claims/results P5; full DB-backed history/reopen pending | P5, B377, E12 | BLOCKED |
| AC-08 | Full functional/integration suite and high defects resolved | UI tests partial | C3 unit/contract tests pass | High requirements gaps remain | B377, E12 | FAIL |
| AC-09 | Domain/language/class AI metrics | N/A | Offline evaluator exists | No approved real prediction set or reported metrics | Evaluation readiness note | BLOCKED |
| AC-10 | Double-annotated reviewed dataset | N/A | 18-query English pilot not held-out verdict dataset | Legal review and agreement not confirmed | Evaluation readiness note | BLOCKED |
| AC-11 | No-legal-advice disclaimer | Visible C3 notice | N/A | Browser rendered | E12 | PASS |
| AC-12 | Admin source to retrievable chunks | No admin UI | No admin update API | Absent | Code | NOT IMPLEMENTED |

## C2 and other-component traceability (no fabricated IDs)

The [C2 specification](./C2_RESEARCH_COMPONENT_SPEC%20(1).md) calls for a Sri Lankan civil-judgment dataset, semantic/hybrid retrieval, percentage-based relevance, fact-level similarities/differences/evidence gaps, and grounded cited explanation. The current C2 wizard only saves a local draft and logs simulated intake; the registered C2 router only responds to `/ping`. The separate `c2_case_analysis` router is not registered. Therefore these C2 objectives are **FAIL/NOT IMPLEMENTED at frontend-to-registered-API integration**, regardless of internal prototypes. Its historical-judgment data quality and retrieval accuracy are **BLOCKED** without an approved C2 test environment and owner review. C1 has real Stage 2 intake/extraction routes but no frontend call, so integration **FAIL**; its backend processing quality is **BLOCKED**. C4 is a demo UI plus `/ping`, so live argumentation **NOT IMPLEMENTED**. There is no official feature-level SRS/SRDS for C1/C4 in scope, so their requirement compliance is **BLOCKED**. The common dashboard, precedent library, insights and settings show illustrative data; they are prototypes, not live integrations.

## Changes, tests and handoff

- Preserved all three pre-existing modified C3 backend tests and the existing C3 frontend page/component. Phase 6 changed only the C3 frontend component and added C3 Playwright plus async HTTP history tests; no shared or teammate code was changed.
- Added a session-history button using the existing owner-scoped C3 history API. A user can reopen a completed claim after refresh, or resume polling a queued/processing claim by selecting it. Automatic resume on refresh is not expressly required by FR-24; manual reopen satisfies its intent provisionally. The new UI deliberately labels historical text as stored claim text because the history API supplies `extracted_claim`, not guaranteed original text. This remains **BLOCKED** for a full database-backed browser refresh test.
- New mocked C3 browser cases cover one submission/result/evidence, polling transport failure versus persisted failure, and history reopen after refresh. Async HTTP tests cover C3 history session scoping, pagination, and storage-error responses. Full frontend suite: **12 PASS, 1 SKIPPED** (C2 API error state intentionally absent). Frontend TypeScript check and production build: **PASS**. C3 backend: **377 PASS, 389 subtests PASS**. P5 isolated DB/worker result persistence remains prior evidence, not rerun in Phase 6.
- Real OCR-script accuracy, Sinhala/Tamil translation, corpus retrieval quality, Gemini verdict quality/cost, citation semantics, p95 latency, usability, accessibility, OWASP scanning, and browser matrix remain **BLOCKED**. The 9-source/348-vector artifact has a verified dry-run importer but has not been shown here as a live, complete production-equivalent corpus.
- **Shared owner (@lead) handoff:** [migrations/env.py](./migrations/env.py) imports `Base` from [app/shared/corpus/models.py](./app/shared/corpus/models.py), which exports `SharedBase`. A fresh bounded collection of `app/shared/corpus/tests/test_corpus.py` failed at its own `Base` import before any test ran; root migration setup has the same import mismatch. Required owner fix: align both shared imports with the actual shared base and rerun migration/test collection against an isolated DB. Do not patch this in C3 work. This report is the owner-facing handoff; no external message or issue was posted.
- **C1 owner:** connect document/case intake UI to C1 API only after agreed product contract and isolated tests. **C2 owner:** register intended analysis API and connect wizard/precedent results; remove misleading simulated percentages or clearly retain prototype labels. **C4 owner:** implement or continue labeling demo-only argumentation; clarify expected API contract. **Shared/frontend owner:** decide real authentication, common history, search/notifications/settings and language-control ownership.

**Phase 7 readiness: BLOCKED for full application acceptance.** C3 text-claim infrastructure is ready for controlled continuation, but the official C3 image/history/live-RAG/admin/logging and evaluation gaps above need owners, contracts, data and authorization. Whole-app SRS/SRDS must be supplied before comprehensive compliance can be certified. Do not use Neon or paid providers to close these gaps without explicit approval.
