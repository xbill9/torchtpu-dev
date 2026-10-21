#!/usr/bin/env python3
"""Execute built notebooks on a real Colab TPU runtime and fail on any error.

This is the gate. A notebook that has not run end to end on the accelerator it
claims is not publishable, and reading the rendered markdown will not tell you
that a cell raises.

It drives the Colab CLI (https://github.com/googlecolab/google-colab-cli):

    colab new -s <session> --tpu v5e1
    colab exec -s <session> -f notebooks/<nb>.ipynb   # -> <nb>_output.ipynb
    colab stop -s <session>

`colab exec` writes the executed copy beside the input as ``*_output.ipynb``.
We scan that for ``output_type == "error"`` and for any stderr stream, then
keep it under ``runs/<date>/`` as the evidence that the notebook ran.

Authentication defaults to ``--auth adc``, which reuses the gcloud Application
Default Credentials already on the box. The CLI's own default is ``oauth2``,
which launches an InstalledAppFlow consent in a browser and blocks on a pasted
code -- unusable from a script or an agent. ADC must be a *user* credential
(``gcloud auth application-default login``); a service account cannot drive
Colab, because runtimes belong to a Google user and bill against that user's
compute units.

Usage:
    python3 tools/verify_on_tpu.py                 # all notebooks, each on its own runtime
    python3 tools/verify_on_tpu.py 01_torchtpu_execution_modes
    python3 tools/verify_on_tpu.py --tpu v6e1      # override the accelerator
    python3 tools/verify_on_tpu.py --keep          # leave the runtime up to poke at
    python3 tools/verify_on_tpu.py --auth oauth2   # fall back to the browser flow
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"
CONFIG = ROOT / "notebooks.json"
RUNS = ROOT / "runs"

# Cells whose stderr is expected and is not a failure. JAX writes several
# benign notices to stderr on first device init, and Colab's TPU image trips
# the hugepages one on every single run -- measured 2026-09-20 on a v5e-1.
# Anything not listed here fails the notebook, which is the point.
STDERR_ALLOWLIST = (
    "Platform 'TPU'",
    "tcmalloc",
    "WARNING:absl",
    "Transparent hugepages are not enabled",
    "cloud_tpu_init.py",
    # huggingface_hub under `colab exec`: Colab secrets are only served to the
    # UI, so the HF_TOKEN lookup times out and downloads go unauthenticated.
    # Notebook 02 needs no token, so both are benign. Seen 2026-10-09.
    "HF_TOKEN",
    # Download progress bars from huggingface_hub and datasets.
    "Fetching ",
    "Downloading bytes",
    "Reconstructing",
    "it/s]",
)

# `colab exec --timeout` is a deadline per cell: the cell's whole run, not the
# gap between outputs. The CLI default of 30 s fails notebook 02 on its first
# download. Its slowest cells are the 26 GB download (5-15 min) and each
# checkpoint's load-and-measure cell (load alone 320-396 s on v5e-1).
EXEC_TIMEOUT_S = 1800


def colab(auth: str, *args: str, **kw) -> subprocess.CompletedProcess:
    """Invoke the Colab CLI. `--auth` is global, so it precedes the subcommand."""
    cmd = ["colab", "--auth", auth, *args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    # stdin is closed so an auth prompt fails fast instead of hanging a CI run
    # waiting for a code nobody is there to paste.
    return subprocess.run(cmd, text=True, stdin=subprocess.DEVNULL, **kw)


def scan(executed: pathlib.Path) -> list[str]:
    """Return a list of problems found in an executed notebook."""
    nb = json.loads(executed.read_text())
    problems = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "error":
                tb = "".join(output.get("traceback", []))[:600]
                problems.append(f"cell {i}: {output.get('ename')}: {output.get('evalue')}\n{tb}")
            elif kind == "stream" and output.get("name") == "stderr":
                text = "".join(output.get("text", []))
                if text.strip() and not any(a in text for a in STDERR_ALLOWLIST):
                    problems.append(f"cell {i}: unexpected stderr:\n{text[:400]}")
    return problems


def check_adc() -> bool:
    """Fail before provisioning if ADC is missing or is the wrong credential type."""
    adc = (
        pathlib.Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
        or pathlib.Path.home() / ".config/gcloud/application_default_credentials.json"
    )
    if not adc.is_file():
        adc = pathlib.Path.home() / ".config/gcloud/application_default_credentials.json"

    if not adc.is_file():
        print(
            "no Application Default Credentials found.\n    gcloud auth application-default login",
            file=sys.stderr,
        )
        return False

    try:
        kind = json.loads(adc.read_text()).get("type")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"could not read ADC at {adc}: {exc}", file=sys.stderr)
        return False

    if kind != "authorized_user":
        print(
            f"ADC at {adc} is '{kind}', not 'authorized_user'.\n"
            "Colab runtimes belong to a Google user and bill against that user's\n"
            "compute units, so a service account cannot provision one. Run:\n"
            "    gcloud auth application-default login",
            file=sys.stderr,
        )
        return False
    return True


def provision(auth: str, session: str, tpu: str, attempts: int) -> bool:
    """Allocate a TPU runtime, retrying only failures that retrying can fix.

    Two failures look similar from the outside and must not be treated alike:

      "Backend rejected accelerator 'V6E1'"  -- no entitlement. Permanent.
      "Service Unavailable" (HTTP 503)       -- refused, nothing allocated.
                                                Safe to retry; often clears.
      ReadTimeout / ConnectionError          -- SILENCE, not refusal. The
                                                runtime may exist. Never retry.
      "TooManyAssignments"                   -- you already hold runtimes,
                                                probably orphans. Go look.

    Only the 503 is retried. See the comment in the loop for what retrying a
    timeout cost on 2026-09-20.

    Observed the same day: v5e-1 returned 503 while a CPU runtime provisioned
    normally, which is how you tell capacity apart from an auth problem.
    """
    for attempt in range(1, attempts + 1):
        result = colab(auth, "new", "-s", session, "--tpu", tpu, capture_output=True)
        text = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            print("  READY")
            return True

        if "rejected accelerator" in text:
            print(
                f"  {tpu} is not entitled on this account -- retrying will not help",
                file=sys.stderr,
            )
            return False

        # A timeout is NOT retryable, and this is the expensive lesson.
        #
        # A 503 is a refusal: nothing was built. A ReadTimeout is silence --
        # the assign request may well have succeeded and left a runtime the CLI
        # never got told about. Retrying then builds another. Doing that twice
        # on 2026-09-20 produced three orphaned v5e-1 runtimes that
        # `colab sessions` shows as `[?]`, that `colab stop -s` cannot address
        # because it resolves names from local metadata, and that then tripped
        # TooManyAssignmentsError on every further attempt. They bill until
        # they idle out or a human kills them in the Colab UI.
        if any(m in text for m in ("ReadTimeout", "Read timed out", "ConnectionError")):
            print(
                "  timed out -- NOT retrying: the runtime may exist despite the timeout.\n"
                "  Check for orphans and clear them before trying again:\n"
                f"      colab --auth {auth} sessions        # `[?]` rows are orphans\n"
                "      https://colab.research.google.com/  Runtime > Manage sessions",
                file=sys.stderr,
            )
            return False

        if "TooManyAssignments" in text:
            print(
                "  assignment cap reached -- you are already holding runtimes.\n"
                f"      colab --auth {auth} sessions\n"
                "      https://colab.research.google.com/  Runtime > Manage sessions",
                file=sys.stderr,
            )
            return False

        if "Service Unavailable" not in text and "503" not in text:
            print(f"  attempt {attempt}/{attempts}: failed", file=sys.stderr)
            print(text.strip()[-500:], file=sys.stderr)
            return False

        # Only a clean 503 reaches here: refused, nothing allocated, safe to retry.
        print(f"  attempt {attempt}/{attempts}: no capacity", file=sys.stderr)
        if attempt == attempts:
            print(
                f"  no {tpu} capacity after {attempts} attempts. A CPU runtime\n"
                f"  provisioning normally would confirm it is capacity, not auth:\n"
                f"      colab --auth {auth} new -s probe && colab --auth {auth} stop -s probe",
                file=sys.stderr,
            )
            return False

        wait = 30 * attempt
        print(f"  waiting {wait}s", file=sys.stderr, flush=True)
        time.sleep(wait)
    return False


def verify(stem: str, tpu: str, keep: bool, auth: str, attempts: int = 3) -> bool:
    notebook = OUT / f"{stem}.ipynb"
    if not notebook.exists():
        print(f"not built: {notebook} — run `make build` first", file=sys.stderr)
        return False

    session = f"verify-{stem}"[:40]
    executed = OUT / f"{stem}_output.ipynb"
    executed.unlink(missing_ok=True)

    print(f"\n=== {stem} on --tpu {tpu} (auth: {auth}) ===", flush=True)
    if not provision(auth, session, tpu, attempts):
        return False

    try:
        colab(auth, "status", "-s", session)
        result = colab(auth, "exec", "-s", session, "-f", str(notebook), "--timeout", str(EXEC_TIMEOUT_S))
    finally:
        # Always in a finally: a runtime left up keeps spending compute units,
        # so a crash between here and teardown costs real money.
        if keep:
            print(f"  (--keep: session {session} left running; `colab stop -s {session}`)")
        else:
            colab(auth, "stop", "-s", session)

    if not executed.exists():
        print(f"no executed notebook at {executed}", file=sys.stderr)
        return False

    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    archive = RUNS / stamp
    archive.mkdir(parents=True, exist_ok=True)
    shutil.move(str(executed), archive / executed.name)
    executed = archive / executed.name

    problems = scan(executed)
    if problems:
        print(f"\nFAILED: {stem} ({len(problems)} problem(s))", file=sys.stderr)
        for p in problems:
            print(f"\n  {p}", file=sys.stderr)
        print(f"\nexecuted copy: {executed.relative_to(ROOT)}", file=sys.stderr)
        return False

    if result.returncode != 0:
        print(f"FAILED: colab exec returned {result.returncode}", file=sys.stderr)
        return False

    print(f"OK: {stem} ran clean on {tpu} -> {executed.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stems", nargs="*", help="notebook stems; default is all")
    parser.add_argument("--tpu", help="override accelerator; default comes from notebooks.json")
    parser.add_argument("--keep", action="store_true", help="do not stop the runtime")
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="provisioning attempts before giving up on transient capacity errors",
    )
    parser.add_argument(
        "--auth",
        choices=("adc", "oauth2"),
        default="adc",
        help="Colab auth strategy (default: adc, which reuses gcloud ADC)",
    )
    args = parser.parse_args()

    if not shutil.which("colab"):
        print(
            "colab CLI not found. Install it with:\n    pip install google-colab-cli",
            file=sys.stderr,
        )
        return 1

    if args.auth == "adc" and not check_adc():
        return 1

    cfg = json.loads(CONFIG.read_text())
    entries = {pathlib.Path(name).stem: meta for name, meta in cfg["notebooks"].items()}
    stems = args.stems or list(entries)

    failed = []
    for stem in stems:
        if stem not in entries:
            print(f"unknown notebook: {stem}", file=sys.stderr)
            return 1
        tpu = args.tpu or entries[stem].get("tpu", "v5e1")
        if not verify(stem, tpu, args.keep, args.auth, args.attempts):
            failed.append(stem)

    print(f"\n{len(stems) - len(failed)}/{len(stems)} notebooks verified")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
