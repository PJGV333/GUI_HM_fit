import os
import sys
import pytest

def test_removed_directories():
    """Ensure that legacy framework directories have been deleted."""
    removed_dirs = [
        "hmfit_tauri",
        "backend_fastapi",
        "hmfit_wx",
        "hmfit_wx_legacy",
        "hmfit_legacy_math",
        "web_frontend",
        "hmfit_kivy",
    ]
    for d in removed_dirs:
      path = os.path.join(os.path.dirname(__file__), "..", d)
      assert not os.path.exists(path), f"Directory {d} should have been removed."

def test_removed_root_stubs():
    """Ensure that root compatibility stubs have been deleted."""
    stubs = [
        "NR_conc_algoritm.py",
        "LM_conc_algoritm.py",
        "np_backend.py",
        "errors.py",
        "core_ad_probe.py",
        "noncoop_utils.py",
    ]
    for s in stubs:
      path = os.path.join(os.path.dirname(__file__), "..", s)
      assert not os.path.exists(path), f"Root stub {s} should have been removed."

def test_no_forbidden_imports():
    """
    Ensure that no code in hmfit_core or hmfit_gui_qt imports 
    legacy frameworks or root stubs.
    """
    import subprocess
    
    forbidden = ["fastapi", "uvicorn", "wx", "tauri"]
    # Also forbid importing from the old root names (without hmfit_core prefix)
    forbidden_stubs = ["NR_conc_algoritm", "LM_conc_algoritm", "np_backend", "errors", "core_ad_probe", "noncoop_utils"]
    
    search_paths = ["hmfit_core", "hmfit_gui_qt"]
    
    for lib in forbidden + forbidden_stubs:
        for folder in search_paths:
            try:
                # -r: recursive, -E: regex, -l: list filenames
                # Pattern: import lib or from lib
                cmd = ["grep", "-rlE", f"^(import {lib}|from {lib})", folder]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    files = result.stdout.strip().split("\n")
                    pytest.fail(f"Forbidden import '{lib}' found in files: {files}")
            except FileNotFoundError:
                pass

def test_no_sys_path_manipulation():
    """
    Ensure we are not using sys.path.append('..') or similar hacks 
    that were cleaned up during refactoring.
    """
    import subprocess
    folder = "hmfit_core"
    try:
        cmd = ["grep", "-r", "sys.path.append", folder]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Check if it's adding local paths
            lines = result.stdout.strip().split("\n")
            # We allow some sys.path if absolutely necessary, but not the ../ ones we removed
            for line in lines:
                if ".." in line or "os.path.dirname" in line:
                     pytest.fail(f"Legacy sys.path manipulation found: {line}")
    except FileNotFoundError:
        pass
