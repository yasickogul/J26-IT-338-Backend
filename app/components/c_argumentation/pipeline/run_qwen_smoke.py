"""One-process isolated Qwen3 smoke test; does not initialize embeddings or RAG."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
from pathlib import Path
import time

from .local_qwen_adapter import LocalQwenAdapter


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def main() -> None:
    path = Path(__file__).resolve().parents[4] / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    started = time.perf_counter()
    adapter = LocalQwenAdapter(path, n_ctx=2048, n_threads=4)
    _ = adapter.model
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = adapter.generate(
        "Explain the difference between a legal claim and supporting evidence in two sentences.",
        max_tokens=96,
        temperature=0.1,
    )
    inference_seconds = time.perf_counter() - started
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    measured = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    print(f"MODEL_PATH={path}")
    print(f"MODEL_BYTES={path.stat().st_size}")
    print(f"LOAD_SECONDS={load_seconds:.3f}")
    print(f"INFERENCE_SECONDS={inference_seconds:.3f}")
    print(f"PEAK_WORKING_SET_BYTES={counters.PeakWorkingSetSize if measured else 'unavailable'}")
    print(f"RESPONSE={result.text.replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
