#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmfit_gui_qt.update_utils import (
    UPDATE_CHANNEL_BETA,
    UPDATE_CHANNEL_STABLE,
    default_update_channel_for_version,
    is_prerelease_version,
    normalize_update_channel,
)
from hmfit_gui_qt.version import VERSION


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit HM Fit release metadata for GitHub Actions.")
    ap.add_argument("--version", default=VERSION, help="Release version (default: hmfit_gui_qt.version.VERSION)")
    ap.add_argument("--tag", default="", help="Optional git tag. If omitted, v<version> is used.")
    ap.add_argument(
        "--channel",
        default="",
        choices=(UPDATE_CHANNEL_STABLE, UPDATE_CHANNEL_BETA, ""),
        help="Optional release channel override (stable or beta).",
    )
    ap.add_argument("--github-output", default="", help="Optional path to the GITHUB_OUTPUT file.")
    args = ap.parse_args()

    version = str(args.version or VERSION).strip()
    if not version:
        raise SystemExit("Version cannot be empty.")

    tag = str(args.tag or "").strip() or f"v{version}"
    prerelease_from_version = is_prerelease_version(version)
    derived_channel = default_update_channel_for_version(version)
    channel = normalize_update_channel(args.channel or derived_channel, fallback=derived_channel)

    if channel == UPDATE_CHANNEL_STABLE and prerelease_from_version:
        raise SystemExit(
            "Stable releases must use a final version without prerelease suffix, for example 1.2.3."
        )
    if channel == UPDATE_CHANNEL_BETA and not prerelease_from_version:
        raise SystemExit(
            "Beta releases must use a prerelease version, for example 1.2.3-beta.1."
        )

    prerelease = channel == UPDATE_CHANNEL_BETA

    payload = {
        "version": version,
        "tag": tag,
        "channel": channel,
        "prerelease": prerelease,
        "release_name": f"HM Fit {version}",
        "asset_prefix": f"HMFit-{version}",
    }

    if args.github_output:
        output_path = Path(args.github_output)
        with output_path.open("a", encoding="utf-8") as fh:
            for key, value in payload.items():
                fh.write(f"{key}={json.dumps(value) if isinstance(value, bool) else value}\n")
    else:
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
