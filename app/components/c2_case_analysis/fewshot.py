"""Few-shot examples (notebook cell 4 + `build_fewshot_block` in cell 7).

The notebook sampled 2 real examples per category from the civil subset of the
`navodPeiris/sri-lanka-case-law` (SLR split) dataset with random.seed(42). We reproduce the
same sampling, then cache the result in data/fewshot.json so the dataset is only downloaded once.
"""
import json
import logging
import random
import threading

from . import config
from .utils import is_civil, safe_list

logger = logging.getLogger(__name__)

FEWSHOT_CATEGORIES = ["Argument", "Decision", "Law Reference", "Judicial Precedent"]

_lock = threading.Lock()
_cache = None


def sample_examples(frame, field, title_key, desc_key, rng, n=2, min_len=30, max_len=300):
    pool = []
    for _, r in frame.iterrows():
        for item in safe_list(r[field]):
            text = f"{item.get(title_key,'')}. {item.get(desc_key,'')}".strip()
            if min_len <= len(text) <= max_len:
                pool.append(text)
    if not pool:
        return []
    return rng.sample(pool, min(n, len(pool)))


def _build_from_dataset():
    from datasets import load_dataset

    raw_ds = load_dataset("navodPeiris/sri-lanka-case-law", split="SLR", token=config.HF_TOKEN or None)
    df = raw_ds.to_pandas()
    df["is_civil"] = df.apply(is_civil, axis=1)
    civil_df = df[df["is_civil"]].reset_index(drop=True)
    logger.info("Civil cases for few-shot sampling: %d / %d", len(civil_df), len(df))

    rng = random.Random(42)  # same sequence as random.seed(42) + random.sample in the notebook
    return {
        "Argument": sample_examples(civil_df, "arguments", "title", "description", rng, n=2),
        "Decision": sample_examples(civil_df, "judgements", "title", "description", rng, n=2),
        "Law Reference": sample_examples(civil_df, "laws_referred", "law", "explanation", rng, n=2),
        "Judicial Precedent": sample_examples(civil_df, "citations", "title", "reason_cited", rng, n=2),
    }


def load_fewshot():
    """Returns {category: [example, ...]}. Cached in memory and on disk."""
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        if config.FEWSHOT_CACHE.exists():
            _cache = json.loads(config.FEWSHOT_CACHE.read_text(encoding="utf-8"))
            return _cache
        try:
            data = _build_from_dataset()
            config.FEWSHOT_CACHE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            _cache = data
        except Exception as e:  # keep the app usable; prompt simply has no style examples
            logger.warning("Could not build few-shot examples (%s). Continuing without them.", e)
            _cache = {cat: [] for cat in FEWSHOT_CATEGORIES}
            # not written to disk, so the next restart retries
    return _cache


def build_fewshot_block(fewshot=None):
    fewshot = fewshot if fewshot is not None else load_fewshot()
    lines = []
    for cat, examples in fewshot.items():
        if examples:
            lines.append(f"\nExamples of \"{cat}\":")
            for e in examples:
                lines.append(f'  - "{e}"')
    return "\n".join(lines)
