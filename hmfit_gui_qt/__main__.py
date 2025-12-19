from __future__ import annotations


def main() -> int:
    from .main import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())

