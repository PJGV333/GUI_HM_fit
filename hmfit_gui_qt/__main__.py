from __future__ import annotations

import faulthandler
import os

_CRASH_LOG_FH = None


def _enable_crash_log() -> None:
    global _CRASH_LOG_FH
    if _CRASH_LOG_FH is not None:
        return

    crash_path = os.path.join(os.getcwd(), "hmfit_crash.log")
    try:
        _CRASH_LOG_FH = open(crash_path, "w", encoding="utf-8", buffering=1)
    except Exception:
        _CRASH_LOG_FH = None
        return

    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    try:
        faulthandler.enable(_CRASH_LOG_FH, all_threads=True)
    except Exception:
        pass


def main() -> int:
    _enable_crash_log()
    from .main import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
