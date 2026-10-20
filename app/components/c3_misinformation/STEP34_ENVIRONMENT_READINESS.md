# Step 34 — C3 dependency restoration and offline verification

## Audit and approval

Branch `pirakash` was verified with an empty staging area. Existing staged,
unstaged, and untracked work was preserved. The approved target interpreter is
`/home/sano_01/J26-IT-338-Backend/.venv/bin/python`.

The audit identified two immediate offline-test dependencies:

- `jsonschema==4.26.0`: imported at module load by the Gemini adapter and
  therefore required by RAG, worker, and multilingual integration modules.
- `numpy==2.4.1`: imported by LaBSE embedding/retrieval and shared-corpus
  tests.

These exact constraints are documented by the original read-only CompNet
backend requirements. Large or provider/model packages (`google-genai`,
`sentence-transformers`, `transformers`, `huggingface-hub`, and `fasttext`)
were intentionally not selected for this offline remediation because they are
lazy/future runtime dependencies and may pull large artifacts or model tools.

## Authorized changes and installation result

The requested installation was attempted only in the group environment:

```text
./.venv/bin/python -m pip install "jsonschema==4.26.0" "numpy==2.4.1"
```

The first automated attempt failed because PyPI name resolution was
unavailable (`Name or service not known`). After WSL networking was manually
recovered by the user, the same approved installation completed successfully
in the group environment. No system interpreter, offline installation, or
package workaround was used.

With approval, the exact constraints were added to the target
[requirements.txt](../../requirements.txt). No application, migration, API,
worker, model, or database code was changed.

## Verification results

The initial pre-install verification had the expected import blockers. The
post-install results are recorded below.

The absence of an error from `pip check` does not prove requirements-file
completeness; pip check only evaluates installed distribution metadata.

## Remaining blockers and next action

Restore approved WSL DNS/network access or provide an approved reachable
package index, then rerun the exact installation command above in the group
`.venv`. After successful installation, verify imports, rerun the blocked
focused suites, and perform a single bounded C3 discovery attempt. Do not
install large ML/provider packages until a separate runtime requirement is
approved. PostgreSQL/Neon migration execution remains out of scope.

No database connection, migration execution, model loading, provider call, or
Git staging/history operation occurred in Step 34.

## Step 34B WSL DNS follow-up

Read-only diagnostics confirmed that connectivity has **not** been restored:

- `/etc/resolv.conf` is a dangling symlink to `/mnt/wsl/resolv.conf`;
- `/mnt/wsl` and its resolver file are absent;
- `getent ahostsv4/ahostsv6` fails for both `pypi.org` and
  `files.pythonhosted.org`;
- bounded IPv4 and IPv6 HTTPS requests fail at hostname resolution;
- `ip route` and `ip -4 address` cannot expose an initialized interface in
  the current WSL context;
- no proxy, custom pip index, or credential-bearing pip configuration was
  detected.

After the user manually restarted/recovered WSL networking, the approved
packages were installed successfully in the group environment. No DNS rewrite,
`/etc/wsl.conf` edit, Windows networking change, or insecure index
configuration was performed.

Installed versions:

- `jsonschema==4.26.0`
- `numpy==2.4.1`

`pip check` passed, and the C3 RAG, claim-worker, worker-runner, and worker-
dependency modules now import successfully.

## Post-install verification

Passed:

- repository/lifecycle/model tests: **33 passed**;
- claim worker: **12 passed**;
- claim worker runner: **11 passed**;
- multilingual adapter: **7 passed**;
- worker RAG adapter: **3 passed**;
- async HTTP integration: **7 passed**;
- Step 25 evaluation: **15 passed**;
- shared corpus: **10 passed**;
- `pip check`, package-version checks, C3 imports, and `git diff --check`.

The RAG service suite ran 10 test bodies successfully but timed out on the
threaded-provider boundary/shutdown path (bounded exit 124). Worker-dependency
tests ran 10 test bodies successfully and likewise timed out on the final
shutdown-sensitive case (bounded exit 124). These are not reported as fully
passed.

The complete C3 discovery suite was attempted once with a 60-second bound and
exited 124. It remains blocked by the known TestClient/AnyIO and executor
shutdown behavior; no test hang was counted as success.

Alembic in the group environment reports one head, `c3_0001`, with history
`shared_0001 → c3_0001`. Offline migration commands completed without a
database connection.
