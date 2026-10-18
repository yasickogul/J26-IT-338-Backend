# Individual Research Component Specification — Explainable Civil Case Analysis

> This is the authoritative spec for Component 2. Read this together with `BACKEND_ARCHITECTURE.md` §6.2 (folder structure, endpoints, tables). Where this file and the architecture file's wording differ, this file wins on *scope and behaviour*; the architecture file wins on *folder/file layout* (reuse the existing `app/components/c2_case_intelligence/` skeleton).

## Component title
Explainable Civil Case Analysis

## Core problem
Lawyers and legal researchers may need to examine large numbers of previous judgments to determine which cases are relevant to a current civil dispute. Manual research is time-consuming, and conventional keyword search does not adequately explain the legal relevance of each retrieved judgment.

## Research objective
Develop an explainable civil case retrieval system for Sri Lankan law. The system takes the facts of a current civil case, finds the most relevant previous cases, ranks them using a relevance score, and explains the similarities and differences between the cases.

## Four principal sub-objectives
1. Build and preprocess a Sri Lankan civil-case dataset using publicly accessible judgments from the Court of Appeal website (courtofappeal.lk).
2. Develop semantic civil precedent retrieval using legal embeddings so the system identifies relevant cases based on meaning and legal context, not only exact keyword matches.
3. Develop a Legal Precedent Relevance Score that combines multiple legal similarity factors to rank retrieved cases and communicate their relative relevance as a percentage-based score.
4. Develop fact-level similarity/difference and evidence-gap analysis to identify similarities, differences, and potentially relevant evidence or information not provided.

These four objectives are the core of the research. Keep them central in implementation, evaluation, and documentation.

## Expected system behaviour

**Input** — the user enters the facts of a current civil case:
- Type of dispute
- Important facts and circumstances
- Legal issue or claim
- Available evidence
- Requested remedy (where applicable)

**System behaviour:**
1. Processes and structures the input.
2. Searches the prepared Sri Lankan civil-case dataset.
3. Retrieves candidate judgments using legal text representations and suitable retrieval methods.
4. Ranks the candidate cases using a Legal Precedent Relevance Score.
5. Displays the most relevant previous judgments, each with a percentage-based relevance score.
6. Explains why each case was recommended and which factors contributed to the score.
7. Compares important facts and evidence between the current case and each retrieved judgment.
8. Identifies potential evidence gaps or missing information for further consideration.
9. Provides source references and supporting judgment passages wherever possible.

## 4. Important distinction: relevance is not outcome prediction

This is primarily an **explainable civil case retrieval and analysis system**, not an outcome-prediction system.

**Central pipeline:**
```
Current Civil Case → Fact and Claim Processing → Relevant Previous Cases
→ Legal Relevance Ranking → Percentage-Based Relevance Score
→ Fact Comparison → Evidence-Gap Analysis → Explainable Results
```

Do not redefine this as a generic legal chatbot or a win/loss predictor. If outcome analysis comes up, treat it as secondary/out of scope unless explicitly approved otherwise.

**Relevance score interpretation:** the percentage is a *percentage-based relevance score*, not a probability of winning, not a measure of how legally binding a precedent is, and not a guarantee the court reaches the same decision. It reflects relative relevance under the system's defined factors and must be validated.

**Possible scoring approach:**
```
R(A, B) = Σ (weight_i × similarity_i(A, B))
```
Candidate factors: legal issue similarity, claim similarity, factual similarity, evidence similarity, remedy similarity, relevant precedent relationships/court hierarchy (where data allows). Weights are not assumed equal — determine and validate them.

## 5. Technical architecture (candidate technologies, subject to final design)

| Tech | Role |
|---|---|
| Python | Data processing, NLP, retrieval, scoring, comparison, evaluation |
| FastAPI | REST endpoints: submit case info, return retrieved judgments, scores, comparisons, explanations |
| PyMuPDF | Extract text/metadata from PDF judgments |
| Legal-BERT (or another suitable legal embedding model) | Vectorizes legal text for semantic retrieval — evaluate fit for Sri Lankan legal language, don't assume a foreign-jurisdiction model performs well as-is |
| BM25 | Keyword-based retrieval for legal terms, phrases, citations |
| PostgreSQL + pgvector or FAISS | Store/index embeddings, retrieve semantically similar documents — don't assume both are required |
| RAG + LLM | Natural-language explanations from retrieved passages and structured comparison results — **the LLM must never invent the relevance percentage**; it is computed by the defined scoring algorithm, and the explanation must be grounded in the actual factors/evidence the system used |

**Possible processing pipeline:**
1. Collect publicly accessible Sri Lankan judgments.
2. Extract judgment text and metadata.
3. Clean and preprocess.
4. Prepare the civil-case dataset.
5. Generate legal embeddings.
6. Store embeddings + metadata.
7. Process the user's current civil-case input.
8. Retrieve candidates (lexical + semantic).
9. Rank candidates with the relevance score.
10. Compare current case vs. retrieved judgments.
11. Identify evidence gaps.
12. Generate explanations grounded in retrieved passages.
13. Return results via FastAPI.

## 6. Dataset and data quality requirements

Source: courtofappeal.lk, publicly accessible judgments relevant to scope.

Cover: lawful/appropriate collection, download + text extraction, PDF parsing quality, OCR for scanned docs, cleaning/normalization, duplicate detection, case metadata extraction, civil-case identification/scope verification, language/document-quality issues, storage/indexing, relevance annotation where feasible, train/dev/test separation where model training is involved, avoiding dev/eval leakage.

Do not assume every document on the site is a civil judgment or has complete structured metadata — handle these limitations explicitly. Do not invent dataset sizes, labels, performance numbers, or findings.

## 7. Evaluation

Evaluate against the component's own objectives, not generic classification accuracy alone.

- **A. Dataset preparation:** text-extraction quality, duplicate rate, metadata completeness, coverage of intended case types.
- **B. Precedent retrieval/ranking:** Precision@k, Recall@k, nDCG@k, comparison vs. a lexical/semantic baseline, expert-labelled relevance judgements where feasible.
- **C. Relevance scoring:** agreement with human/expert relevance assessments, ranking quality, whether the score meaningfully separates relevant from less-relevant precedents.
- **D. Fact-level analysis:** Precision/Recall/F1 for extracted/matched facts where labelled data exists; whether key similarities/differences are correctly identified.
- **E. Evidence-gap analysis & explanations:** expert assessment of factual correctness, evidence support/traceability, clarity/usefulness of explanations, analysis of unsupported claims or hallucinations.

Use Accuracy/Precision/Recall/F1/Confusion Matrix where a specific classification sub-task calls for it — not as sufficient evidence of good precedent retrieval on their own.
