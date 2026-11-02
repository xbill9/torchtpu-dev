#!/usr/bin/env python3
"""Refresh the bundled skill snapshots from the repo-root sources.

Regenerates:
  .claude/skills/<skill>/mcp/server.py              from  server.py
  .claude/skills/<skill>/mcp/project-setup.sh       from  project-setup.sh
  .claude/skills/<skill>/mcp/payload/{rust/...,rust_openai_server.py}
                                                    from  rust/ and rust_openai_server.py
                                                    (what the startup script builds and serves)
  .claude/skills/<skill>/references/tpu-guide.md
                                                            from  tpu.md
                                                            (base64 screenshots stripped)

SKILL.md and mcp/startup_script_rust_template.sh are hand-maintained and left alone.
"""

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Skill name carries the rig directory so sibling rigs do not collide on one
# ~/.claude/skills/<name>; the Makefile passes the same value.
SKILL_NAME = os.getenv("SKILL_NAME", f"{ROOT.name}-management")
SKILL = ROOT / ".claude" / "skills" / SKILL_NAME

IMAGE_DEF = re.compile(r"^\[image\d+\]: <data:image/[^;]+;base64,")

NOTE_TEMPLATE = """\
> **Note:** This is a text-only copy of the repository's `tpu.md` ({size} KB original). The
> embedded base64 screenshots for the Cloud Console walkthrough have been stripped;
> `![][imageN]` markers show where they appeared. See the original `tpu.md` at the repo
> root for the images.
"""


def build_guide(src: Path, dest: Path) -> None:
    # tpu.md is private and gitignored, so a fresh clone legitimately lacks it.
    # Keep whatever guide snapshot is already committed rather than aborting the
    # whole refresh — otherwise every other snapshot silently goes stale too.
    if not src.exists():
        print(
            f"warning: {src.name} not found — keeping the existing "
            f"{dest.relative_to(ROOT)} (regenerate it where {src.name} is present)"
        )
        return
    lines = src.read_text().splitlines()
    kept = [line for line in lines if not IMAGE_DEF.match(line)]
    while kept and not kept[-1].strip():
        kept.pop()
    note = NOTE_TEMPLATE.format(size=round(src.stat().st_size / 1024))
    # Insert the note right after the document's title line.
    body = [kept[0], "", note.rstrip()] + kept[1:]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(body) + "\n")
    print(f"wrote {dest.relative_to(ROOT)} ({dest.stat().st_size} bytes)")


# The files server.py packs into the startup script (its _PAYLOAD_FILES). An installed skill
# runs mcp/server.py, which finds them under mcp/payload/.
PAYLOAD_FILES = (
    "rust/Cargo.toml",
    "rust/Cargo.lock",
    "rust/PYO3.md",
    "rust/build.rs",
    "rust/rust-toolchain.toml",
    "rust/src/lib.rs",
    "rust/src/aten.rs",
    "rust/src/aten.cpp",
    "rust/src/bindings.rs",
    "rust/src/gemma.rs",
    "rust/src/weights.rs",
    "rust/python/torchtpu_rs.pyi",
    "rust_openai_server.py",
)


def copy_payload() -> None:
    dest_root = SKILL / "mcp" / "payload"
    shutil.rmtree(dest_root, ignore_errors=True)
    for rel in PAYLOAD_FILES:
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    print(f"copied {len(PAYLOAD_FILES)} payload files -> {dest_root.relative_to(ROOT)}")


def main() -> int:
    if not (SKILL / "SKILL.md").exists():
        print(f"error: {SKILL} not found — run from the repo root", file=sys.stderr)
        return 1
    shutil.copyfile(ROOT / "server.py", SKILL / "mcp" / "server.py")
    print(f"copied server.py -> {(SKILL / 'mcp' / 'server.py').relative_to(ROOT)}")
    shutil.copy(ROOT / "project-setup.sh", SKILL / "mcp" / "project-setup.sh")  # copy() preserves +x
    print(f"copied project-setup.sh -> {(SKILL / 'mcp' / 'project-setup.sh').relative_to(ROOT)}")
    shutil.copyfile(ROOT / "requirements.txt", SKILL / "mcp" / "requirements.txt")
    print(f"copied requirements.txt -> {(SKILL / 'mcp' / 'requirements.txt').relative_to(ROOT)}")
    copy_payload()
    build_guide(ROOT / "tpu.md", SKILL / "references" / "tpu-guide.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
