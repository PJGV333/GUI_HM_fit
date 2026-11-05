"""Print pytest JUnit XML counts as a Markdown table (for $GITHUB_STEP_SUMMARY).

Usage: python tools/ci/junit_summary.py report.xml [label]
Counts: collected, passed, failed (failures + errors), xfailed, xpassed, skipped.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def counts(path: str) -> dict[str, int]:
    root = ET.parse(path).getroot()
    out = dict(collected=0, passed=0, failed=0, xfailed=0, xpassed=0, skipped=0)
    for case in root.iter("testcase"):
        out["collected"] += 1
        tags = {child.tag: child for child in case}
        skipped = tags.get("skipped")
        if "failure" in tags or "error" in tags:
            msg = (tags.get("failure") if "failure" in tags else tags.get("error")).get("message", "")
            out["xpassed" if "XPASS" in msg else "failed"] += 1
        elif skipped is not None:
            kind = skipped.get("type", "")
            out["xfailed" if kind == "pytest.xfail" else "skipped"] += 1
        else:
            out["passed"] += 1
    return out


def main() -> None:
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    c = counts(path)
    print(f"### pytest: {label}\n")
    print("| collected | passed | failed | xfailed | xpassed | skipped |")
    print("|---:|---:|---:|---:|---:|---:|")
    print("| {collected} | {passed} | {failed} | {xfailed} | {xpassed} | {skipped} |".format(**c))


if __name__ == "__main__":
    main()
