#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


IGNORE_COPY = shutil.ignore_patterns(
    ".git", ".idea", "__pycache__", ".pytest_cache",
    "dist", "build", "dist_appimage", ".flatpak-builder",
    ".hmfvenv", ".venv", "node_modules"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def die(msg: str, code: int = 2) -> int:
    print(f"[error] {msg}", file=sys.stderr)
    return code


def which_ok(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def find_manifest(repo: Path) -> Path | None:
    # Prefer root, then fallback
    c1 = repo / "org.hmfit.HMFit.yml"
    if c1.is_file():
        return c1
    c2 = repo / "packaging" / "linux" / "flatpak" / "org.hmfit.HMFit.yml"
    if c2.is_file():
        return c2
    hits = list(repo.rglob("org.hmfit.HMFit.yml"))
    return hits[0] if hits else None


def extract_appid(manifest: Path) -> str | None:
    for line in manifest.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("app-id:"):
            return s.split(":", 1)[1].strip().strip("'\"")
        if s.startswith("id:"):
            return s.split(":", 1)[1].strip().strip("'\"")
    return None


def patch_manifest(original: Path, out_path: Path, src_rel: str, wheels_rel: str) -> None:
    """
    Patch:
      - pip install -> offline pip install from wheels dir
      - sources: dir path: .  -> path: <src_rel>
    Keep everything else.
    """
    txt = original.read_text(encoding="utf-8", errors="ignore").splitlines()

    patched: list[str] = []
    for line in txt:
        # Replace the pip command line (match loosely)
        if "pip3 install" in line and "requirements_qt.txt" in line:
            indent = line.split("p", 1)[0]  # keep YAML indentation
            patched.append(
                indent
                + f"python3 -m pip install --prefix=/app --no-index --find-links={wheels_rel} "
                  f"--no-cache-dir -r requirements_qt.txt"
            )
            continue

        # Replace sources dir path
        if line.strip() == "path: .":
            # Only replace if it's under a dir source
            patched.append(line.replace("path: .", f"path: {src_rel}"))
            continue

        patched.append(line)

    out_path.write_text("\n".join(patched) + "\n", encoding="utf-8")


def sdk_installed(runtime: str) -> bool:
    for cmd in (
        ["flatpak", "info", "--user", runtime],
        ["flatpak", "info", runtime],
    ):
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            continue
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Build HM Fit Flatpak bundle without polluting the repo (offline pip).")
    ap.add_argument("--source", default=".", help="Path to HM Fit repo (default: .)")
    ap.add_argument("--manifest", default="", help="Path to org.hmfit.HMFit.yml (auto-detect if empty)")
    ap.add_argument("--appid", default="", help="Flatpak app-id (auto from manifest if empty)")
    ap.add_argument("--dest", default=str((Path.home() / "HMFit_builds").resolve()),
                    help="Destination dir OR full path ending with .flatpak (default: ~/HMFit_builds)")
    ap.add_argument("--branch", default="stable",
                    help="Flatpak branch to export/build-bundle (default: stable)")
    ap.add_argument("--repo-dir", default="",
                    help="Optional OSTree repo dir to populate instead of a temporary repo")
    ap.add_argument("--work-root", default=str((Path.home() / "BUILD_HMFIT_FLATPAK").resolve()),
                    help="Workspace root (default: ~/BUILD_HMFIT_FLATPAK)")
    ap.add_argument("--install", action="store_true", help="Install to --user after build")
    ap.add_argument("--run", action="store_true", help="Run after install (implies --install)")
    ap.add_argument("--keep-work", action="store_true", help="Keep workspace for debugging")
    args = ap.parse_args()

    repo = Path(args.source).expanduser().resolve()
    if not repo.is_dir():
        return die(f"source not found: {repo}")

    if not which_ok("flatpak") or not which_ok("flatpak-builder"):
        return die("Need flatpak + flatpak-builder. On Arch: sudo pacman -S flatpak flatpak-builder")

    sdk_runtime = "org.kde.Sdk//6.7"
    print(f"[check] Verifying Flatpak SDK is installed: {sdk_runtime}")
    if not sdk_installed(sdk_runtime):
        return die(
            f"Missing Flatpak SDK {sdk_runtime}. Install it with:\n"
            f"  flatpak install --user flathub {sdk_runtime}"
        )

    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else (find_manifest(repo) or Path())
    if not manifest.is_file():
        return die("Manifest not found. Provide --manifest /path/to/org.hmfit.HMFit.yml")

    appid = args.appid.strip() or (extract_appid(manifest) or "")
    if not appid:
        return die("Could not determine app-id. Provide --appid org.hmfit.HMFit")

    dest = Path(args.dest).expanduser().resolve()
    dest_is_file = dest.suffix.lower() == ".flatpak"
    if dest_is_file:
        dest.parent.mkdir(parents=True, exist_ok=True)
        bundle_path = dest
    else:
        dest.mkdir(parents=True, exist_ok=True)
        bundle_path = dest / f"{appid}.flatpak"

    branch = str(args.branch or "stable").strip() or "stable"

    work_root = Path(args.work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="hmfit_flatpak_", dir=str(work_root))).resolve()
    src_dir = work_dir / "src"
    build_dir = work_dir / "build"
    repo_dir = Path(args.repo_dir).expanduser().resolve() if args.repo_dir else (work_dir / "repo")
    wheels_dir = src_dir / "packaging" / "linux" / "flatpak" / "wheels"

    try:
        print(f"[info] repo:      {repo}")
        print(f"[info] manifest:  {manifest}")
        print(f"[info] appid:     {appid}")
        print(f"[info] branch:    {branch}")
        print(f"[info] work_dir:  {work_dir}")
        print(f"[info] repo_dir:   {repo_dir}")
        print(f"[info] bundle ->  {bundle_path}")

        # 1) Copy repo into workspace (clean)
        print("[step] Copying repo into workspace (no repo pollution)")
        shutil.copytree(repo, src_dir, ignore=IGNORE_COPY, dirs_exist_ok=True)
        wheels_dir.mkdir(parents=True, exist_ok=True)

        # 2) Download wheels USING SDK python (correct ABI) with network enabled
        # We mount the workspace path so the SDK sandbox can read/write it.
        print("[step] Downloading wheels inside org.kde.Sdk//6.7 (with network)")
        sdk_cmd = (
            "python3 -m ensurepip --upgrade >/dev/null 2>&1 || true; "
            "python3 -m pip install -q --upgrade pip >/dev/null 2>&1 || true; "
            "python3 -m pip download --only-binary=:all: -r requirements_qt.txt -d packaging/linux/flatpak/wheels"
        )
        run([
            "flatpak", "run",
            "--command=sh",
            "--share=network",
            f"--filesystem={src_dir}",
            "org.kde.Sdk//6.7",
            "-lc",
            f"cd /run/host{src_dir} 2>/dev/null || cd {src_dir}; {sdk_cmd}"
        ], cwd=work_dir)

        # 3) Patch manifest to use offline pip + sources path=src
        patched_manifest = work_dir / "org.hmfit.HMFit.patched.yml"
        patch_manifest(
            original=manifest,
            out_path=patched_manifest,
            src_rel="src",
            wheels_rel="packaging/linux/flatpak/wheels",
        )

        # 4) Build (offline)
        print("[step] Building flatpak (offline pip)")
        run([
            "flatpak-builder",
            "--user",
            "--force-clean",
            "--disable-rofiles-fuse",
            f"--repo={repo_dir}",
            f"--default-branch={branch}",
            str(build_dir),
            str(patched_manifest),
        ], cwd=work_dir)

        if args.run:
            args.install = True

        # 5) Bundle for testers
        print("[step] Creating .flatpak bundle for testers")
        run([
            "flatpak", "build-bundle",
            "--runtime-repo=https://flathub.org/repo/flathub.flatpakrepo",
            str(repo_dir),
            str(bundle_path),
            appid,
            branch,
        ], cwd=work_dir)

        print(f"[ok] Flatpak bundle created: {bundle_path}")

        if args.install:
            print("[step] Installing bundle in user Flatpak installation")
            run([
                "flatpak", "install", "--user", "-y",
                "--or-update",
                "--bundle",
                str(bundle_path),
            ], cwd=work_dir)

        if args.run:
            run(["flatpak", "run", appid], cwd=work_dir)

        return 0

    except subprocess.CalledProcessError as e:
        print(f"[error] Build failed with code {e.returncode}", file=sys.stderr)
        if args.keep_work:
            print(f"[hint] workspace kept at: {work_dir}", file=sys.stderr)
        else:
            print("[hint] re-run with --keep-work to inspect wheels/manifest/build logs", file=sys.stderr)
        return int(e.returncode)

    finally:
        if args.keep_work:
            print(f"[info] Keeping workspace: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
