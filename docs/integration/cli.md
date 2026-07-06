# CLI integration

> **Status: stub.** This page will be expanded with the full `spe` command
> reference. For now it points at the authoritative sources.

Sparrow Engine ships a command-line binary. The CPU build is `spe`; the GPU
build is `spe-gpu`. They expose the **same subcommands**; only the execution
backend differs.

```bash
spe --help          # CPU
spe-gpu --help      # GPU
```

Key points for integrators (e.g. batch jobs, shell pipelines, CI):

- The CLI and the Python package expose the **same function set** with the same
  conventions (a project rule — Local and Web must not diverge).
- Output is machine-parseable (JSON / ndjson) where a batch consumer needs it.
- Exit codes are non-zero on failure; errors go to stderr as structured logs.
- The `spe` and `spe-gpu` binaries are never co-located; pick the flavor that
  matches the host.

Until this page is filled in, run `spe --help` (and `spe <subcommand> --help`)
for the authoritative command list, and see the top-level
[`../user-manual.md`](../user-manual.md) for worked examples.
