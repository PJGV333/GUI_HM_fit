#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from tkinter import Tk, filedialog, messagebox, simpledialog


def main() -> int:
    root = Tk()
    root.withdraw()
    root.update()

    messagebox.showinfo(
        "HM Fit AppImage Builder",
        "Select the HM Fit project folder (GUI_HM_fit).\n"
        "The build will run in a temporary directory (no repo pollution).",
    )

    project_dir = filedialog.askdirectory(title="Select HM Fit project folder")
    if not project_dir:
        return 0

    project_dir = os.path.abspath(project_dir)

    # Optional: git ref (tag/branch/commit). Leave empty to use local working tree copy.
    ref = simpledialog.askstring(
        "Git ref (optional)",
        "Enter git tag/branch/commit (leave empty to use the current local folder state):",
    )
    if ref is not None:
        ref = ref.strip()
    if not ref:
        ref = ""

    default_name = "hmfit_pyside6-x86_64.AppImage"
    out_file = filedialog.asksaveasfilename(
        title="Save AppImage as...",
        defaultextension=".AppImage",
        filetypes=[("AppImage", "*.AppImage"), ("All files", "*.*")],
        initialfile=default_name,
    )
    if not out_file:
        return 0

    out_file = os.path.abspath(out_file)

    # Locate the shell script next to this file; fallback: let user pick it.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sh_path = os.path.join(script_dir, "build_hmfit_pyside6_appimage.sh")
    if not os.path.isfile(sh_path):
        messagebox.showwarning(
            "Shell script not found",
            "Could not find build_hmfit_pyside6_appimage.sh next to this .py.\n"
            "Please select the .sh script manually.",
        )
        sh_path = filedialog.askopenfilename(
            title="Select build_hmfit_pyside6_appimage.sh",
            filetypes=[("Shell script", "*.sh"), ("All files", "*.*")],
        )
        if not sh_path:
            return 1

    sh_path = os.path.abspath(sh_path)

    # Temporary build root (will be removed at the end)
    build_root = tempfile.mkdtemp(prefix="hmfit_appimage_build_")

    cmd = ["bash", sh_path, "--source", project_dir, "--build-root", build_root, "--out-file", out_file]
    if ref:
        cmd += ["--ref", ref]

    try:
        messagebox.showinfo(
            "Build started",
            "Build started.\n\nA terminal will show progress.\n"
            "If it fails, you will see an error dialog at the end.",
        )
        # Run in foreground; user sees terminal output
        subprocess.run(cmd, check=True)
        messagebox.showinfo("Build finished", f"AppImage created:\n{out_file}")
        return 0

    except subprocess.CalledProcessError as e:
        messagebox.showerror(
            "Build failed",
            f"Build failed with exit code {e.returncode}.\n\n"
            "Check the terminal output for details.",
        )
        return e.returncode

    finally:
        # Remove build root (script already removes its mktemp build dir; this removes the outer root)
        shutil.rmtree(build_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
