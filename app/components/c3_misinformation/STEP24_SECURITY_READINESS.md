# C3 Step 24 Security Readiness

**Scope:** offline review of the C3 claim submission and result API, anonymous
claim-session contract, and worker boundary. This document records code-level
controls and remaining deployment decisions; it is not a production security
certification.

## Implemented controls

- Anonymous tokens are generated with `secrets.token_urlsafe(32)`. The token
  format is validated before use. C3 stores a domain-separated SHA-256 digest,
  not the browser token.
- Result lookup passes the token digest to an owner-filtered repository query.
  Missing and foreign claim IDs both produce `404 result_not_found`; absent or
  malformed session tokens produce `401 invalid_claim_session`. Unknown but
  syntactically valid tokens do not identify a claim and cannot read another
  session's claim.
- Submission reuses a valid token only when the repository confirms that it
  owns at least one claim. Missing, malformed, or unknown tokens are replaced.
  The response cookie is set only after the caller-owned transaction exits
  successfully. Persistence or commit failure returns a sanitized error and
  does not acknowledge the claim.
- Cookies use `HttpOnly`, `SameSite=Lax`, and `Path=/`, without a `Domain`,
  `Max-Age`, or `Expires` attribute. HTTPS requests use
  `__Host-compnet_claim_session` with `Secure`; HTTP uses
  `compnet_claim_session` without `Secure` for local development. Setting
  `COMPNET_SECURE_COOKIES=1` forces the secure host-only cookie name and
  `Secure` attribute even when the request scope is HTTP.
- The submission schema accepts a strict string field named `text` and rejects
  extra fields. Empty/whitespace-only input and input cleaned to empty are
  rejected. Existing NFC/whitespace cleanup is retained. Both original and
  cleaned text pass the source-compatible local LaBSE tokenizer admission
  check without truncation. Tokenizer and optional language-detection work is
  moved off the event loop. Missing tokenizer assets fail closed with a
  sanitized `503`.
- Submission responds `202` only after transaction success. The acknowledgment
  status `"processing"` does not mutate the stored `"queued"` state. The
  request handler does not start a worker, retrieve evidence, or invoke Gemini.
- Result responses are owner-scoped, use `Cache-Control: no-store`, omit
  internal failure categories, and return the stored result snapshot unchanged.
  Database/provider exception details are not included in API error bodies.
  Malformed lifecycle records are converted to a sanitized
  `500 result_state_unavailable`.
- The existing C3 routes are mounted by the group application beneath
  `/api/v1/c3`; their OpenAPI paths and request schema were checked offline.

## Source and SRDS basis

The read-only CompNet implementation uses the same 32-byte URL-safe token,
domain-separated hash, ownership check, cookie names/attributes, cleanup,
tokenizer-based admission, `202` acknowledgment, owner-scoped result lookup,
and sanitized `404` behavior. Its claim route declares that authentication is
not required, with an optional bearer token only for account association; the
current C3 anonymous cookie is an ownership capability, not account identity.

SRDS v2.0 FR-01 requires non-empty free text, a length limit, a claim record,
and a processing acknowledgment. FR-01 says the limit is enforced per NFR-01,
but NFR-01 specifies text-analysis p95 response time (15 seconds on reference
hardware), not an input character, byte, or token maximum. Existing source
behavior derives admission from the local tokenizer's model sequence limit.
No additional numeric request-size limit is specified or introduced here.

The source SRDS is marked Baseline / Draft for Review. These mappings establish
the current compatibility basis, not approval of unresolved deployment policy.

## Ownership and API boundaries

The anonymous session hash is checked as part of the database query, so a
claim ID alone is not sufficient to read its lifecycle or result. The API does
not expose owner hashes, worker lease tokens, database exceptions, or failure
categories. The separate worker owns processing; API submission persists a
queued claim and acknowledges it only after commit.

Group `app/core/security.py` currently contains only a JWT-dependency TODO.
Group `app/main.py` configures credentialed CORS using configured origins and
registers C3 under `/api/v1/c3`. No group authentication or routing files were
changed as part of this step.

## Remaining decisions and risks

### CSRF, origin, CORS, and JWT

`SameSite=Lax` is a useful cookie restriction, but is not a complete CSRF
policy. C3 does not currently validate `Origin`/`Referer` or require a CSRF
token. Submission is cookie-independent for authorization (it creates a claim
owned by the submitted session), but a browser can still be induced to make
unwanted submissions in some deployment contexts. CORS controls whether
cross-origin JavaScript can read responses; it should not be treated as the
only CSRF control.

Before production exposure, the group must decide and document:

1. Which frontend origins may submit anonymous claims, including whether
   sibling subdomains are trusted.
2. Whether to enforce an allowlisted `Origin` on browser POST requests, add a
   CSRF token, or adopt another group-approved mechanism.
3. Whether/when JWT identity should associate claims with accounts. The
   anonymous C3 session must not be treated as a JWT substitute.

These choices affect group deployment/authentication behavior, so no
group-wide policy was invented or changed here.

### TLS termination and cookie scheme detection

Cookie security uses the ASGI request URL scheme, or the explicit
`COMPNET_SECURE_COOKIES=1` override. When TLS terminates at a reverse proxy,
the group deployment must verify that the application receives the correct
HTTPS scheme only from a trusted proxy. An untrusted or missing forwarded
scheme can otherwise select the non-secure development cookie on an externally
HTTPS deployment. Production should force `COMPNET_SECURE_COOKIES=1` and
configure trusted proxy handling; the `__Host-` cookie attributes require
HTTPS, `Path=/`, and no `Domain`.

### Request-size/resource exhaustion

No approved byte/character body limit was found. Tokenizer admission happens
after FastAPI has received and parsed the request body, so it does not bound
network/body buffering or all CPU cost for extremely large JSON strings. The
tokenizer check prevents over-sequence-limit claims from being persisted or
processed, but is not a transport-level denial-of-service control.

The product/API owner must clarify whether FR-01's "length limit" means
characters, UTF-8 bytes, tokenizer tokens, or another measure, and set a
limit. Until then, configure and verify an infrastructure request-body limit
at the approved ingress/reverse proxy without changing claim acceptance in C3.

### Verification not covered offline

FastAPI test clients verify generated headers and route behavior, not browser
cookie enforcement, proxy header trust, cross-site behavior, or deployed CORS
semantics. Mocked repository tests do not establish real PostgreSQL ownership,
transaction rollback, or JSONB behavior. Real PostgreSQL/pgvector verification
remains postponed until after Step 29, as directed.

## Recommended deployment checks

1. Require HTTPS, set `COMPNET_SECURE_COOKIES=1`, and verify the trusted proxy
   scheme configuration and generated `__Host-` cookie in a real browser.
2. Agree on frontend origin, CSRF, and JWT-association policies before exposing
   cookie-authenticated browser workflows beyond the approved origin set.
3. Approve an input size unit/limit and enforce a transport-level body cap at
   ingress; keep the tokenizer model limit as a separate semantic admission
   check.
4. After Step 29, verify migrations, owner-scoped reads, commit/rollback,
   cookie-to-hash persistence, and stored result snapshots against an isolated
   disposable PostgreSQL database. Never use production/shared databases for
   that verification.
5. Run browser and deployment tests for credentialed CORS, same-site sibling
   origins, CSRF behavior, proxy TLS termination, cache behavior, and token
   rotation before production rollout.
