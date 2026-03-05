#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def newest_appimage(search_dir: Path) -> Path | None:
    candidates = list(search_dir.rglob("*.AppImage"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build HM Fit AppImage without GUI (no tkinter). Runs build in temp dir to avoid repo pollution."
    )
    p.add_argument("--source", default=".", help="Path to HM Fit repo (default: .)")
    p.add_argument("--ref", default="", help="Optional git ref (tag/branch/commit) if your .sh supports it")
    p.add_argument(
        "--dest",
        default=str((Path.home() / "HMFit_builds").resolve()),
        help="Destination (directory OR full .AppImage file path). Default: ~/HMFit_builds",
    )
    p.add_argument(
        "--build-root",
        default=str((Path.home() / "BUILD_HMFIT_PYSIDE6").resolve()),
        help="Build root directory (temp build folder is created inside). Default: ~/BUILD_HMFIT_PYSIDE6",
    )
    p.add_argument(
        "--sh",
        default=str((Path(__file__).resolve().parent / "build_hmfit_pyside6_appimage.sh")),
        help="Path to build_hmfit_pyside6_appimage.sh",
    )
    p.add_argument("--keep-build", action="store_true", help="Keep temp build directory (for debugging).")
    args = p.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        print(f"[error] source not found: {source}", file=sys.stderr)
        return 2

    sh_path = Path(args.sh).expanduser().resolve()
    if not sh_path.is_file():
        print(f"[error] .sh not found: {sh_path}", file=sys.stderr)
        return 2

    dest = Path(args.dest).expanduser().resolve()
    dest_is_file = dest.suffix.lower() == ".appimage"

    # Ensure destination directory exists
    if dest_is_file:
        dest.parent.mkdir(parents=True, exist_ok=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)

    build_root = Path(args.build_root).expanduser().resolve()
    build_root.mkdir(parents=True, exist_ok=True)

    # Temp build dir outside repo to avoid "ensuciar"
    build_dir = Path(tempfile.mkdtemp(prefix="hmfit_appimage_", dir=str(build_root))).resolve()
    out_dir = build_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["bash", str(sh_path), "--source", str(source), "--out", str(out_dir)]
    if args.ref.strip():
        # Only pass if you want; if your .sh doesn't support it, remove this.
        cmd += ["--ref", args.ref.strip()]

    print(f"[build] build_dir: {build_dir}")
    print(f"[build] out_dir:   {out_dir}")
    print(f"[build] running:   {' '.join(cmd)}")

    try:
        # Key point: cwd=build_dir so any relative paths the .sh creates land here, not in the repo
        subprocess.run(cmd, check=True, cwd=str(build_dir))

        app = newest_appimage(out_dir) or newest_appimage(build_dir)
        if not app or not app.is_file():
            print("[error] Build finished but no .AppImage found in output.", file=sys.stderr)
            print(f"[hint] Look inside: {build_dir}", file=sys.stderr)
            return 3

        # Decide final output path
        if dest_is_file:
            final_path = dest
        else:
            final_path = dest / app.name

        shutil.copy2(str(app), str(final_path))
        # Make sure it's executable
        final_path.chmod(final_path.stat().st_mode | 0o111)

        print(f"[ok] AppImage saved to: {final_path}")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"[error] Build failed with code {e.returncode}", file=sys.stderr)
        print(f"[hint] temp build dir: {build_dir}", file=sys.stderr)
        return int(e.returncode)

    finally:
        if args.keep_build:
            print(f"[build] keeping build dir: {build_dir}")
        else:
            shutil.rmtree(str(build_dir), ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())