"""Audit pass: fill in what OpenRouter knows about past runs.

`python -m bench.reconcile` walks result rows that carry a generation id
but no billed cost, asks OpenRouter's generation endpoint about each one,
and reports what it would write. `--apply` performs the writes.

Dry run is the default because this is the one path in the bench that
edits history. Seeing the diff before taking it is worth one extra
command, and a backfill that silently rewrote a hundred rows would be
indistinguishable from a bug.

Deliberately a separate command rather than something that runs at boot
or inside a request: reconciliation is N serial upstream calls, and
putting that in the request path would make a comparison wait on an audit
it did not ask for. The rows it touches are the audit columns only
(store.RECONCILABLE_COLUMNS), never text, errors, timings, or the local
estimate.

Safe to run against a live bench. store.connect opens the same database
in WAL mode, which is what lets a reader and a writer coexist.
"""

import argparse
import asyncio
import os
import sqlite3
import sys
from typing import Any, TextIO

import httpx

from bench import store
from bench.models import fetch_generation, keepalive_socket_options

# One request at a time with a pause between them. The pass is not urgent
# (nothing is waiting on it) and a burst of lookups against an endpoint
# that exists to serve dashboards is the kind of thing that earns a rate
# limit for the whole key, including the runs that do matter.
POLITE_DELAY_S = 0.5


def _describe(record: dict[str, Any]) -> str:
    """What this record would write, in one line, or why it would not."""
    if record["error"] is not None:
        return f"skipped: {record['error']}"
    parts = [
        f"{column}={record[column]!r}"
        for column in store.RECONCILABLE_COLUMNS
        if record[column] is not None
    ]
    return ", ".join(parts) if parts else "nothing to write"


def _has_writes(record: dict[str, Any]) -> bool:
    if record["error"] is not None:
        return False
    return any(record[column] is not None for column in store.RECONCILABLE_COLUMNS)


async def reconcile(
    conn: sqlite3.Connection,
    client: httpx.AsyncClient,
    apply: bool = False,
    limit: int | None = None,
    delay_s: float = POLITE_DELAY_S,
    out: TextIO = sys.stdout,
) -> dict[str, int]:
    """Walk the pending rows, report, and optionally write. Returns counts.

    conn and client are both injected, the same pattern the endpoints use,
    which is what lets the script-level test run the whole pass against a
    seeded database and a stubbed endpoint without a subprocess or a
    network.
    """
    pending = store.results_awaiting_reconciliation(conn, limit)
    counts = {"pending": len(pending), "written": 0, "skipped": 0}
    if not pending:
        print("nothing to reconcile", file=out)
        return counts

    mode = "applying" if apply else "dry run, nothing will be written"
    print(f"{len(pending)} result(s) to reconcile ({mode})", file=out)
    for index, row in enumerate(pending):
        record = await fetch_generation(client, row["generation_id"])
        print(
            f"  result {row['id']} ({row['model']}, {row['generation_id']}): "
            f"{_describe(record)}",
            file=out,
        )
        if _has_writes(record):
            if apply:
                store.apply_reconciliation(conn, row["id"], record)
                counts["written"] += 1
        else:
            counts["skipped"] += 1
        # Serial with a pause, and no pause after the last row: a trailing
        # sleep only delays the summary.
        if delay_s and index < len(pending) - 1:
            await asyncio.sleep(delay_s)

    if apply:
        print(f"wrote {counts['written']} row(s)", file=out)
    else:
        writable = counts["pending"] - counts["skipped"]
        print(f"would write {writable} row(s); re-run with --apply", file=out)
    return counts


async def _run(args: argparse.Namespace, api_key: str) -> dict[str, int]:
    conn = store.connect(os.environ.get("BENCH_DB", "./bench.db"))
    try:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            transport=httpx.AsyncHTTPTransport(
                socket_options=keepalive_socket_options()
            ),
        ) as client:
            return await reconcile(
                conn, client, apply=args.apply, limit=args.limit, delay_s=args.delay
            )
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.reconcile",
        description=(
            "Fill in billed cost, provider, quantization and the provider's "
            "own finish reason for past runs, from OpenRouter's generation "
            "endpoint. Reports what it would write unless --apply is given."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the writes (default is a dry run that writes nothing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="reconcile at most this many rows, oldest first",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=POLITE_DELAY_S,
        help=f"seconds to wait between lookups (default {POLITE_DELAY_S})",
    )
    args = parser.parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # The same refusal the app makes at boot, for the same reason: a
        # pass that authenticates as nobody would report every row as a 401
        # and look like the records had expired.
        print(
            "OPENROUTER_API_KEY is not set. Export it before reconciling.",
            file=sys.stderr,
        )
        return 2
    asyncio.run(_run(args, api_key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
