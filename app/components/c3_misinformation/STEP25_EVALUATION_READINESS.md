# C3 Step 25 Evaluation Readiness

## Scope and available data

The read-only CompNet source provides
`/home/sano_01/fyp/data/embedding_eval_queries_v1_1.json`, a revised English
retrieval pilot. Its declared scope is 18 constructed questions. Query IDs and
target/background IDs remain in source order. The file states that target
chunks are reviewed direct-answer passages, background chunks are context and
not automatic direct-answer hits, and labels are not exhaustive. It also says
legal review is not confirmed and the pilot is not a held-out claim-verdict
evaluation.

Q08 retains its direct target `CON-003-0068`. Its alternative path is the
separate required set `CON-003-0170` **and** `CON-003-0172`: the first passage
describes the clause and the second provides the section opening that makes it
void. The two labels are not flattened into extra interchangeable direct
targets.

The existing CompNet retrieval runners report target Hit@1 and Hit@5. The
embedding script computes full-ranking MRR only with its full corpus ranking;
the PostgreSQL top-five verifiers explicitly leave full MRR unreported. The
Q08 packet verifier checks packet IDs, ranks, score preservation, parent
relations, and empty evidence. The RAG verifiers validate structured response
schemas and citation-ID resolution, and explicitly require manual review for
semantic citation support.

The C3 evaluator consumes the original query JSON and caller-supplied
recorded-prediction JSON. It does not copy or modify the source dataset and
does not run retrieval or model inference.

## Retrieval measures and denominators

The report includes:

- **Target Hit@K**: queries with at least one designated target ID in the first
  K results divided by queries with a supplied retrieval ranking.
- **Target MRR@K**: reciprocal rank of the first designated target when it
  appears within K, otherwise zero, averaged over supplied retrieval rankings.
  This is explicitly bounded MRR@K, not full-ranking MRR.
- **Labelled-condition ID coverage@K**: a query has all IDs for any individual
  target singleton or any separate alternative set in the first K results.
  Every ID in an alternative set is required. Per-query reports list each
  condition, missing IDs, completion rank, and each completed condition. This
  retrieval-only measure does **not** assert structural relationships,
  context resolution, legal sufficiency, or full evidence-set success.

K values are explicitly recorded in the report and default to 1, 3, and 5.
Queries with a recorded empty result list count in the denominator as misses.
Queries with no retrieval record are listed as missing and excluded from
retrieval metric denominators; query coverage and evaluated counts are
reported separately.

Precision@K is not reported because retrieved labels are not exhaustive.
Recall@K is not reported because complete relevant corpus sets are not known.
nDCG@K is unsupported because there are no complete graded judgments.
Full-ranking MRR is not derived from truncated predictions. No claim-verdict
accuracy, legal correctness, or semantic citation-support score is produced.
The report names these unsupported metrics and their reasons.

No cosine distance is converted to an RRF or hybrid score. Recorded C3 ranks
must be contiguous one-based order, unique by chunk ID, and ordered by
ascending cosine distance then chunk ID. Distances must be finite in [0, 2],
similarities finite in [-1, 1], and similarity must equal `1 - distance`
within numeric tolerance. Invalid data is rejected, not repaired.

## Recorded RAG checks

For each supplied completed RAG record, the evaluator checks the serialized
C3 `RagResponse` shape (including question-versus-claim fields), matching mode,
required explanation presence, packet/query consistency, evidence ID and
chunk-ID consistency (`chunk:<chunk_id>`), unique packet IDs, exact packet /
serialized-passage ID set agreement, context and relationship references,
unique citation IDs, and that every citation resolves in the packet.

It checks that an empty evidence packet has the existing application-generated
abstention shape, including no citations, no answer/verdict, and
`insufficient_evidence=true`. For non-empty packets, an insufficient response
must not cite evidence and must not make a claim verdict or question answer.
Unavailable, failed, and not-attempted RAG results are counted separately from
invalid results. An optional `expected_request_text` enables an exact recorded
request-text comparison; otherwise input preservation is not inferred.

Structural citation resolution does not establish that evidence supports a
response. legal sufficiency, semantic support, verdict correctness, and explanation
quality require independent human review.

## Input formats

The reviewed query file is the CompNet v1.1 JSON schema. A prediction file has
this top-level shape:

```json
{
  "schema_version": "1.0",
  "predictions": [
    {
      "query_id": "Q01",
      "retrieval_results": [
        {
          "rank": 1,
          "chunk_id": "recorded-chunk-id",
          "distance": 0.2,
          "similarity": 0.8
        }
      ]
    }
  ]
}
```

Prediction IDs must be unique and belong to the supplied query file. A
retrieval record may have an empty `retrieval_results` list. Omitting a query
record means its output is missing. A prediction may also include a `rag`
object with `status` (`completed`, `unavailable`, `not_attempted`, or `failed`).
A completed record contains:

```json
{
  "mode": "question",
  "request_text": "exact submitted question",
  "expected_request_text": "optional expected text for exact comparison",
  "retrieval_query": "exact query represented by the packet",
  "evidence_packet": {"query": "exact query represented by the packet", "evidence": []},
  "response": {}
}
```

`response` is the serialized C3 `RagResponse`. Supply both the retrieval
ranking and RAG record when measuring both stages. Invalid query and retrieval
structures stop scoring with an error naming the offending record. Invalid
RAG responses are reported as invalid per-query checks rather than converted
to successful outputs.

## Offline command

Run with explicit input and new output paths:

```sh
/home/sano_01/fyp/backend/.venv/bin/python -m \
  app.components.c3_misinformation.evaluation \
  --queries /home/sano_01/fyp/data/embedding_eval_queries_v1_1.json \
  --predictions /path/to/recorded-predictions.json \
  --output /path/to/new-report.json \
  --markdown /path/to/new-report.md \
  --k 1 --k 3 --k 5
```

Run from the target repository root with `PYTHONPATH=.`. Output paths must not
exist; the runner uses exclusive creation and does not overwrite earlier
reports. The JSON report records both input SHA-256 hashes, dataset identity
and review status, K/ranking settings, metric definitions, per-query detail,
missing outputs, invalid RAG output summaries, and unsupported metrics. It
contains no secrets or session tokens. The runner performs no network,
database, Gemini, LaBSE, or model access.

## Reproducibility and future evaluation

Use frozen query-label revisions and retain prediction files as immutable
inputs. Keep prediction provenance with the report externally: code/model
version, embedding identity and dimension, corpus snapshot/hash, filters,
top-K, prompt/schema revision, provider/model configuration, and run
environment. The report hashes query and prediction bytes; those hashes make
the evaluated input files identifiable but do not attest to how predictions
were generated.

Future LaBSE/Gemini evaluations should first record real retrieval and RAG
outputs under an approved, read-only/isolated setup, then run this offline
scorer. Model calls, database access, and human annotation stay outside the
scorer. To claim verdict or semantic-quality results, prepare a versioned
claim set with independent legal review, adjudication, explicit verdict and
evidence labels, provenance/seen status, leakage-resistant splits, declared
denominators, and a review protocol. The existing 18 constructed questions
and current development reports are not a held-out legal-accuracy benchmark.

## Limitations and current evidence

This step verifies evaluation code with synthetic predictions, not the quality
of LaBSE, Gemini, live PostgreSQL retrieval, or legal conclusions. The existing
CompNet report at
`/home/sano_01/fyp/reports/embedding_retrieval/20261014T123708804391Z/results.json`
records a prior run on the older query file with Hit@1 17/18 and Hit@5 18/18;
it explicitly does not calculate full-ranking MRR. This is historical source
evidence only, not a Step 25 rerun, not scored against v1.1 alternative
conditions, and not a new C3 result.

The reviewed labels do not establish corpus-wide relevance or legal-review
approval. Without recorded real predictions in this execution, no new
retrieval-quality or model-performance number is claimed.
