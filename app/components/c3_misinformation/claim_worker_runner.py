"""Explicitly invoked C3 claim-worker runner; importing starts no work."""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any

from .claim_worker import process_one_claim
from .claim_worker_contract import WorkerOutcome
from .worker_dependencies import (
    WorkerDependencies,
    WorkerRuntimeConfig,
    create_configured_worker_dependencies,
)


_RUNNER_LOCK = Lock()
WorkerProcess = Callable[[], Awaitable[WorkerOutcome | None]]


@dataclass(frozen=True)
class WorkerRunnerConfig:
    poll: bool = False
    poll_interval: float = 5.0
    lease_seconds: int = 3600

    def __post_init__(self) -> None:
        if type(self.poll) is not bool:
            raise TypeError("poll must be a boolean")
        if not 0.25 <= self.poll_interval <= 300:
            raise ValueError("poll_interval must be between 0.25 and 300 seconds")
        if type(self.lease_seconds) is not int or not 60 <= self.lease_seconds <= 86400:
            raise ValueError("lease_seconds must be an integer from 60 to 86400")


async def _wait_for_shutdown(shutdown: asyncio.Event, timeout: float) -> bool:
    try:
        await asyncio.wait_for(shutdown.wait(), timeout=timeout)
    except TimeoutError:
        return False
    return True


async def run_worker(
    *,
    process: WorkerProcess,
    shutdown: asyncio.Event,
    poll: bool = False,
    poll_interval: float = 5.0,
) -> WorkerOutcome | None:
    """Run once or drain available work, sleeping only on an empty poll."""
    if not callable(process):
        raise TypeError("process must be callable")
    if not isinstance(shutdown, asyncio.Event):
        raise TypeError("shutdown must be an asyncio.Event")
    config = WorkerRunnerConfig(poll=poll, poll_interval=poll_interval)
    if not _RUNNER_LOCK.acquire(blocking=False):
        raise RuntimeError("A C3 worker runner is already active in this process.")

    last_outcome: WorkerOutcome | None = None
    try:
        while not shutdown.is_set():
            last_outcome = await process()
            if not config.poll:
                break
            if last_outcome is None:
                if await _wait_for_shutdown(shutdown, config.poll_interval):
                    break
        return last_outcome
    finally:
        _RUNNER_LOCK.release()


def _worker_process(
    dependencies: WorkerDependencies,
    config: WorkerRunnerConfig,
) -> WorkerProcess:
    async def process() -> WorkerOutcome | None:
        return await process_one_claim(
            analysis=dependencies.analyzer,
            session_factory=dependencies.session_factory,
            lease_seconds=config.lease_seconds,
            top_k=dependencies.config.top_k,
        )

    return process


async def run_configured_worker(
    *,
    config: WorkerRunnerConfig = WorkerRunnerConfig(),
    dependencies: WorkerDependencies | None = None,
    shutdown: asyncio.Event | None = None,
) -> WorkerOutcome | None:
    """Run with the existing group sessionmaker and C3 dependency composition."""
    if dependencies is None:
        dependencies = create_configured_worker_dependencies(
            config=WorkerRuntimeConfig(),
        )
    stop = shutdown or asyncio.Event()
    return await run_worker(
        process=_worker_process(dependencies, config),
        shutdown=stop,
        poll=config.poll,
        poll_interval=config.poll_interval,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process C3 queued claims separately from FastAPI.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="process at most one claim (default)")
    mode.add_argument("--poll", action="store_true", help="continuously poll for queued claims")
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="empty-queue polling interval in seconds (0.25 to 300; default 5)",
    )
    parser.add_argument(
        "--lease-seconds", type=int, default=3600,
        help="renewable claim lease in seconds (60 to 86400; default 3600)",
    )
    return parser


async def _run_cli(config: WorkerRunnerConfig) -> int:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    registered: list[signal.Signals] = []
    previous_handlers: dict[signal.Signals, Any] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            loop.add_signal_handler(signum, stop.set)
            registered.append(signum)
        outcome = await run_configured_worker(config=config, shutdown=stop)
        if not config.poll:
            print(
                "No queued claim." if outcome is None
                else f"Claim processing: {outcome.status}.",
            )
        return 0
    finally:
        for signum in registered:
            loop.remove_signal_handler(signum)
            signal.signal(signum, previous_handlers[signum])


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(argv)
    try:
        config = WorkerRunnerConfig(
            poll=options.poll,
            poll_interval=options.interval,
            lease_seconds=options.lease_seconds,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        return asyncio.run(_run_cli(config))
    except Exception:
        print("C3 claim worker stopped after an error.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WorkerRunnerConfig",
    "main",
    "run_configured_worker",
    "run_worker",
]
