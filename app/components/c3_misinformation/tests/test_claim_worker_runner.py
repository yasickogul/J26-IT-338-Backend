"""Offline tests for explicit C3 worker runner modes and shutdown."""

import asyncio
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import AsyncMock, patch
from uuid import UUID

from app.components.c3_misinformation import claim_worker_runner as runner
from app.components.c3_misinformation.claim_worker_contract import WorkerOutcome
from app.components.c3_misinformation.worker_dependencies import (
    WorkerDependencies,
    WorkerRuntimeConfig,
)


CLAIM_ID = UUID("11111111-1111-4111-8111-111111111111")
OUTCOME = WorkerOutcome(CLAIM_ID, "completed")


class ClaimWorkerRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_shot_returns_outcome_or_empty_queue(self):
        process = AsyncMock(return_value=OUTCOME)
        self.assertEqual(
            await runner.run_worker(process=process, shutdown=asyncio.Event()),
            OUTCOME,
        )
        process.assert_awaited_once()

        empty = AsyncMock(return_value=None)
        self.assertIsNone(await runner.run_worker(process=empty, shutdown=asyncio.Event()))
        empty.assert_awaited_once()

    async def test_pre_requested_shutdown_exits_without_processing(self):
        shutdown = asyncio.Event()
        shutdown.set()
        process = AsyncMock()
        self.assertIsNone(await runner.run_worker(process=process, shutdown=shutdown, poll=True))
        process.assert_not_awaited()

    async def test_poll_drains_claims_then_waits_interruptibly_on_empty_queue(self):
        shutdown = asyncio.Event()
        process = AsyncMock(side_effect=[OUTCOME, None])
        wait = AsyncMock(side_effect=lambda _event, timeout: shutdown.set() or True)
        with patch.object(runner, "_wait_for_shutdown", new=wait):
            result = await runner.run_worker(
                process=process,
                shutdown=shutdown,
                poll=True,
                poll_interval=2.5,
            )
        self.assertIsNone(result)
        self.assertEqual(process.await_count, 2)
        wait.assert_awaited_once_with(shutdown, 2.5)

    async def test_shutdown_during_claim_finishes_current_work_then_exits(self):
        shutdown = asyncio.Event()

        async def process():
            shutdown.set()
            return OUTCOME

        self.assertEqual(
            await runner.run_worker(process=process, shutdown=shutdown, poll=True),
            OUTCOME,
        )

    async def test_cancellation_propagates_and_releases_single_runner_guard(self):
        started = asyncio.Event()

        async def blocked():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            runner.run_worker(process=blocked, shutdown=asyncio.Event()),
        )
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(
            await runner.run_worker(
                process=AsyncMock(return_value=OUTCOME),
                shutdown=asyncio.Event(),
            ),
            OUTCOME,
        )

    async def test_concurrent_runner_is_rejected_and_errors_propagate(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked():
            started.set()
            await release.wait()
            return OUTCOME

        task = asyncio.create_task(
            runner.run_worker(process=blocked, shutdown=asyncio.Event()),
        )
        await started.wait()
        with self.assertRaisesRegex(RuntimeError, "already active"):
            await runner.run_worker(
                process=AsyncMock(return_value=None),
                shutdown=asyncio.Event(),
            )
        release.set()
        self.assertEqual(await task, OUTCOME)

        failure = RuntimeError("worker database failure")

        async def fail():
            raise failure

        with self.assertRaisesRegex(RuntimeError, "worker database failure"):
            await runner.run_worker(process=fail, shutdown=asyncio.Event())

    async def test_validates_source_interval_and_lease_boundaries(self):
        for kwargs in (
            {"poll_interval": 0.249}, {"poll_interval": 300.01},
            {"lease_seconds": 59}, {"lease_seconds": 86401},
            {"lease_seconds": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                runner.WorkerRunnerConfig(**kwargs)
        runner.WorkerRunnerConfig(poll_interval=0.25, lease_seconds=60)
        runner.WorkerRunnerConfig(poll_interval=300, lease_seconds=86400)

    async def test_worker_process_injects_composed_analyzer_and_lease_config(self):
        dependencies = WorkerDependencies(
            session_factory=object(),
            query_embedder=object(),
            evidence_packet_builder=object(),
            gemini_provider=object(),
            rag_service=object(),
            analyzer=object(),
            config=WorkerRuntimeConfig(top_k=11),
        )
        config = runner.WorkerRunnerConfig(lease_seconds=120)
        with patch.object(
            runner, "process_one_claim", new=AsyncMock(return_value=OUTCOME),
        ) as process_one:
            result = await runner._worker_process(dependencies, config)()
        self.assertEqual(result, OUTCOME)
        process_one.assert_awaited_once_with(
            analysis=dependencies.analyzer,
            session_factory=dependencies.session_factory,
            lease_seconds=120,
            top_k=11,
        )

    async def test_configured_runner_is_explicit_and_uses_injected_dependencies(self):
        dependencies = WorkerDependencies(
            session_factory=object(),
            query_embedder=object(),
            evidence_packet_builder=object(),
            gemini_provider=object(),
            rag_service=object(),
            analyzer=object(),
            config=WorkerRuntimeConfig(),
        )
        with patch.object(
            runner, "process_one_claim", new=AsyncMock(return_value=None),
        ) as process_one:
            result = await runner.run_configured_worker(
                dependencies=dependencies,
                config=runner.WorkerRunnerConfig(),
            )
        self.assertIsNone(result)
        process_one.assert_awaited_once()

    def test_cli_preserves_source_arguments_and_sanitizes_runner_failures(self):
        with patch.object(runner, "_run_cli", new=AsyncMock(return_value=0)) as run_cli:
            self.assertEqual(
                runner.main(["--poll", "--interval", "2.5", "--lease-seconds", "120"]),
                0,
            )
        config = run_cli.await_args.args[0]
        self.assertEqual(
            config,
            runner.WorkerRunnerConfig(poll=True, poll_interval=2.5, lease_seconds=120),
        )

        with patch.object(
            runner, "_run_cli",
            new=AsyncMock(side_effect=RuntimeError("private claim and credential")),
        ), redirect_stderr(io.StringIO()) as error_output:
            self.assertEqual(runner.main(["--once"]), 1)
        self.assertNotIn("private claim", error_output.getvalue())

    def test_import_does_not_start_worker(self):
        with patch.object(runner, "process_one_claim") as process:
            self.assertTrue(callable(runner.main))
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
