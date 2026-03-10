#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Update hmfit_gui_qt.version.VERSION in-place for CI builds.")
    ap.add_argument("--version", required=True, help="Version string to write.")
    ap.add_argument(
        "--file",
        default=str(Path("hmfit_gui_qt") / "version.py"),
        help="Target version.py path (default: hmfit_gui_qt/version.py).",
    )
    args = ap.parse_args()

    version = str(args.version or "").strip()
    if not version:
        raise SystemExit("Version cannot be empty.")

    target = Path(args.file).resolve()
    text = target.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(^VERSION\s*=\s*")[^"]+(")',
        rf'\g<1>{version}\g<2>',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(f"Could not update VERSION in {target}")

    target.write_text(updated, encoding="utf-8")
    print(f"[ok] Updated {target} to VERSION={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
