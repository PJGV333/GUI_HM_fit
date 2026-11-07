# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract the literature benchmark cases from the Roelens/Vacca 2009 SI PDFs.

The PDFs are NOT part of the repository (copyright of their publishers/authors);
the author keeps them locally and passes their directory with ``--pdf-dir``.
Only the numbers extracted from them are versioned, under
``tests/data/literature/<case>/``.

Two independent passes are run and compared automatically:

* pass A: ``pdfplumber`` (pdfminer.six) words with their x/y coordinates; the
  two-column limiting-shift blocks are assigned by x position.
* pass B: poppler's ``pdftotext -layout``; the limiting-shift blocks are assigned
  by the chemical rule (a nucleus can only have a shift in a species that
  contains its reagent), with the character column used only to resolve
  value-without-error ambiguities.

Any disagreement between the passes aborts the extraction. Invariants that are
checked on top of that: printed counts (spectra, resonance values, nuclei), the
HypNMR degrees-of-freedom identity (sigma/RMS)^2 = m/(m-p), repeated tables
(r09b-r-qx vs r09a-r1-5sp; the shared 0.15 mM titration of r09a-p2-058/-106)
and agreement with the rounded values of the article tables.

Usage::

    python -m validation.literature_extract --pdf-dir "/path/to/pdfs" [--write]

Requires ``pdfplumber`` and ``pdftotext`` (poppler-utils); neither is needed to
run the benchmark itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data" / "literature"

SI_A = "chem_200802298_sm_miscellaneous_information.pdf"  # R09a SI
SI_B = "chem_200900342_sm_miscellaneous_information.pdf"  # R09b SI

REFS = {
    "R09a": {
        "citation": ("S. Roelens, A. Vacca, C. Venturi, \"Binding of Ionic Species: A General "
                     "Approach To Measuring Binding Constants and Assessing Affinities\", "
                     "Chem. Eur. J. 2009, 15, 2635-2644"),
        "doi": "10.1002/chem.200802298",
        "si_file": SI_A,
        "article_file": ("Chemistry A European J - 2009 - Roelens - Binding of Ionic Species  A "
                         "General Approach To Measuring Binding Constants and.pdf"),
    },
    "R09b": {
        "citation": ("S. Roelens, A. Vacca, O. Francesconi, C. Venturi, \"Ion-Pair Binding: Is "
                     "Binding Both Binding Better?\", Chem. Eur. J. 2009, 15, 8296-8302"),
        "doi": "10.1002/chem.200900342",
        "si_file": SI_B,
        "article_file": "roelens2009.pdf",
    },
}

NOTICE = ("Data belong to their authors (see SOURCES.md); used only for the internal "
          "validation of HM fit.")

# Fitting cases. data_page/results_pages are the printed SI page labels ("S4" ...).
# nuclei: (name, reagent) in table order. ``datasets``: "by_salt" splits titrations
# when the salt total changes; "single" keeps one titration.
CASES: dict[str, dict[str, Any]] = {
    "r09a-r1-11-me": dict(source="R09a", data_page="S4", results_pages=["S5"], reagents=["R", "G"],
                          table_nuclei=[("NH-2", "R"), ("NH-1", "R"), ("Me", "G")], use_nuclei=["Me"],
                          datasets="single", results_block=0,
                          description="Receptor 1 + TMAC, 1:1 two-reagent model (R, G), Me signal only",
                          notes=["Me is assigned to G: the paper treats TMAC as the chloride guest."]),
    "r09a-r1-11-3n": dict(source="R09a", data_page="S4", results_pages=["S6"], reagents=["R", "G"],
                          table_nuclei=[("NH-2", "R"), ("NH-1", "R"), ("Me", "G")],
                          use_nuclei=["NH-2", "NH-1", "Me"], datasets="single", results_block=0,
                          description="Same data, 1:1 model with 3 nuclei (deliberately poor fit)",
                          notes=["Deliberately inadequate model (the paper uses it to show that a 1:1 "
                                 "two-reagent model does not describe the system)."]),
    "r09a-r1-3sp": dict(source="R09a", data_page="S7", results_pages=["S8"], reagents=["R", "Q", "X"],
                        table_nuclei=[("NH-2", "R"), ("NH-1", "R"), ("Me", "Q")],
                        use_nuclei=["NH-2", "NH-1", "Me"], datasets="by_salt", results_block=0,
                        description="Receptor 1 + TMAC, three reagents, species RQX, RX, QX; "
                                    "titrations at 0.131 and 1.06 mM",
                        notes=["Intermediate model of the paper (standard deviations judged unsatisfactory)."]),
    "r09a-r1-5sp": dict(source="R09a", data_page="S7", results_pages=["S9"], reagents=["R", "Q", "X"],
                        table_nuclei=[("NH-2", "R"), ("NH-1", "R"), ("Me", "Q")],
                        use_nuclei=["NH-2", "NH-1", "Me"], datasets="by_salt", results_block=0,
                        article_table="R09a Table 1",
                        description="Receptor 1 + TMAC, species RQX, RX, R2, R2X, QX (final model)",
                        notes=["Final reference model for receptor 1 (R09a Table 1)."]),
    "r09a-qx-dil": dict(source="R09a", data_page="S10", results_pages=["S11"], reagents=["Q", "X"],
                        table_nuclei=[("Me", "Q")], use_nuclei=["Me"], datasets="single",
                        results_block=0,
                        description="TMAC dilution, Q + X = QX, Me signal",
                        notes=["Independent value of log beta(QX) (log beta = 4.05 +/- 0.01 in the text)."]),
    "r09a-p2-058": dict(source="R09a", data_page="S12", results_pages=["S13", "S14"],
                        reagents=["P", "Q", "X"],
                        table_nuclei=[("NH-2", "P"), ("CH-B", "P"), ("CH-C", "P"), ("NH-1", "P"),
                                      ("CH2-Bn", "P"), ("CH2N", "P"), ("Me", "Q")],
                        use_nuclei=["NH-2", "CH-B", "CH-C", "NH-1", "CH2-Bn", "CH2N", "Me"],
                        datasets="by_salt", results_block=0, article_table="R09a Table 2",
                        known_article_mismatch={"P2X.sd": (0.0353, 0.03)},
                        description="Receptor 2 (P) + TMAC, titrations at 0.15 and 0.58 mM, "
                                    "7 nuclei, 6 species",
                        notes=["beta(P2) is held constant (log beta = 0.1000) by the authors.",
                               "Missing values in the table ('-') are left empty.",
                               "R09a Table 2 prints SD(log beta P2X) = 0.03; the SI results page "
                               "prints 0.0353, which rounds to 0.04. Both values are reported; the "
                               "benchmark uses the SI value."]),
    "r09a-p2-106": dict(source="R09a", data_page="S17", results_pages=["S18", "S19"],
                        reagents=["P", "Q", "X"],
                        table_nuclei=[("NH-2", "P"), ("CH-B", "P"), ("CH-C", "P"), ("NH-1", "P"),
                                      ("CH2-Bn", "P"), ("CH2N", "P"), ("Me", "Q")],
                        use_nuclei=["NH-2", "CH-B", "CH-C", "NH-1", "CH2-Bn", "CH2N", "Me"],
                        datasets="by_salt", results_block=0,
                        description="Receptor 2 (P) + TMAC, titrations at 0.15 and 1.06 mM",
                        notes=["Shares the 0.15 mM titration with r09a-p2-058.",
                               "beta(P2) is held constant (log beta = 0.1000) by the authors."]),
    "r09b-r-qx": dict(source="R09b", data_page="S9", results_pages=["S10"], reagents=["R", "Q", "X"],
                      table_nuclei=[("NH-2", "R"), ("NH-1", "R"), ("Me", "Q")],
                      use_nuclei=["NH-2", "NH-1", "Me"], datasets="by_salt", results_block=0,
                      article_table="R09b Table 1",
                      description="Repeats r09a-r1-5sp from the R09b SI (extraction cross-check)",
                      notes=["Data and results reported from R09a (ref. 42 of R09b)."]),
    "r09b-b-qx": dict(source="R09b", data_page="S13", results_pages=["S14"], reagents=["B", "Q", "X"],
                      table_nuclei=[("Me", "Q")], use_nuclei=["Me"], datasets="by_salt",
                      results_block=0, article_table="R09b Table 2",
                      description="Receptor B + TMAC, Me only, beta(QX) constant, 2 titrations",
                      notes=["45 resonance values in 50 spectra (5 missing Me values).",
                             "beta(QX) held constant at log beta = 4.0897 (from R09b Table 1)."]),
    "r09b-d-qx": dict(source="R09b", data_page="S15", results_pages=["S16", "S17"],
                      reagents=["D", "Q", "X"],
                      table_nuclei=[("NH-1", "D"), ("CH2O-1", "D"), ("CH2O-2", "D"), ("CH2N", "D"),
                                    ("NH-2", "D"), ("Me", "Q")],
                      use_nuclei=["NH-1", "CH2O-1", "CH2O-2", "CH2N", "NH-2", "Me"],
                      datasets="by_salt", results_block=0, article_table="R09b Table 3",
                      known_count_mismatch={"n_values": (236, 235)},
                      description="Ditopic receptor D + TMAC, 6 nuclei, beta(QX) and beta(D2) constant",
                      notes=["The six DQ limiting shifts are printed without error: they were held "
                             "fixed (see the degrees-of-freedom identity, p = 33).",
                             "The D2 limiting shifts are refined although beta(D2) is constant.",
                             "The SI table holds 236 resonance values but the results page prints "
                             "235 (and the degrees-of-freedom identity holds with m = 235): one "
                             "tabulated value was not used by the original fit and the SI does not "
                             "say which."]),
}

# Speciation tables (stage D). columns as printed.
SPEC_CASES: dict[str, dict[str, Any]] = {
    "r09b-spec-r": dict(source="R09b", pages=["S20"], results_case="r09b-r-qx", reagents=["R", "Q", "X"],
                        analysed="R", bound="X", ancillary="Q",
                        description="R + QX: BC50 (X vs R) and species distribution, T_Q = T_X"),
    "r09b-spec-b": dict(source="R09b", pages=["S21"], results_case="r09b-b-qx", reagents=["B", "Q", "X"],
                        analysed="B", bound="Q", ancillary="X",
                        description="B + QX: BC50 (Q vs B) and species distribution, T_X = T_Q"),
    "r09b-spec-d-x": dict(source="R09b", pages=["S22"], results_case="r09b-d-qx", reagents=["D", "Q", "X"],
                          analysed="D", bound="X", ancillary="Q",
                          description="D + QX: BC50 (X vs D) and species distribution, T_Q = T_X"),
    "r09b-spec-d-q": dict(source="R09b", pages=["S23"], results_case="r09b-d-qx", reagents=["D", "Q", "X"],
                          analysed="D", bound="Q", ancillary="X",
                          description="D + QX: BC50 (Q vs D) and species distribution, T_X = T_Q"),
}

# Article tables (rounded), transcribed twice by hand from the article PDFs and
# compared below with the full-precision SI results: log beta and its SD.
ARTICLE_TABLES = {
    "R09a Table 1": {"RQX": ("6.121", "0.010"), "RX": ("2.946", "0.008"), "R2": ("0.783", "0.027"),
                     "R2X": ("4.513", "0.046"), "QX": ("4.090", "0.005"), "_sigma": "0.00027", "_rms": "0.00025"},
    "R09a Table 2": {"PQX": ("7.19", "0.02"), "PX": ("3.58", "0.01"), "P2": ("0.1", None), "P2X": ("6.26", "0.03"),
                     "P2QX": ("9.47", "0.04"), "QX": ("4.06", "0.01"), "_sigma": "0.00054", "_rms": "0.00049"},
    "R09b Table 1": {"RQX": ("6.121", "0.010"), "RX": ("2.946", "0.008"), "R2": ("0.783", "0.027"),
                     "R2X": ("4.513", "0.046"), "QX": ("4.090", "0.005"), "_sigma": "0.00027", "_rms": "0.00025"},
    "R09b Table 2": {"BQX": ("4.767", "0.021"), "BQ": ("1.373", "0.007"), "QX": ("4.090", "0.005"),
                     "_sigma": "0.00055", "_rms": "0.00051"},
    "R09b Table 3": {"DQX": ("5.371", "0.017"), "DQ2X2": ("12.061", "0.018"), "D2": ("0.189", "0.006"),
                     "DQ": ("0.983", "0.022"), "DX": ("2.132", "0.020"), "QX": ("4.090", "0.005"),
                     "_sigma": "0.00016", "_rms": "0.00015"},
}

CONC_RE = re.compile(r"^\d\.\d+E[+-]\d\d$")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:E[+-]\d+)?$")


def _fill_labels(labels: dict[str, int], n_pages: int) -> dict[str, int]:
    """Pages whose printed label could not be read get it from the constant
    index offset of their neighbours (fails loudly if the offset is not constant)."""
    offsets = {i - int(k[1:]) for k, i in labels.items()}
    if len(offsets) != 1:
        raise ValueError(f"inconsistent SI page labels: {labels}")
    off = offsets.pop()
    out = dict(labels)
    for i in range(n_pages):
        out.setdefault(f"S{i - off}", i)
    return out


# ----------------------------------------------------------------------------- text sources
class PdfplumberSource:
    """Pass A: words with coordinates, grouped into lines by their top position."""

    def __init__(self, path: Path):
        import pdfplumber

        self._pdf = pdfplumber.open(str(path))
        self._labels = {}
        for i, page in enumerate(self._pdf.pages):
            lines = (page.extract_text() or "").strip().splitlines()
            for ln in lines[-3:]:
                if re.fullmatch(r"S\d+", ln.strip()):
                    self._labels[ln.strip()] = i
        self._labels = _fill_labels(self._labels, len(self._pdf.pages))

    def lines(self, label: str) -> list[list[dict]]:
        page = self._pdf.pages[self._labels[label]]
        # x_tolerance=1: the default (3) glues tightly set columns of the speciation tables
        words = page.extract_words(keep_blank_chars=False, use_text_flow=False, x_tolerance=1)
        rows: list[list[dict]] = []
        for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
            if rows and abs(rows[-1][0]["top"] - w["top"]) < 2.5:
                rows[-1].append(w)
            else:
                rows.append([w])
        return [sorted(r, key=lambda w: w["x0"]) for r in rows]


class PdftotextSource:
    """Pass B: poppler pdftotext -layout, one page at a time."""

    def __init__(self, path: Path):
        self._path = path
        out = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True,
                             capture_output=True, text=True).stdout
        self._pages = out.split("\f")
        self._labels = {}
        for i, txt in enumerate(self._pages):
            tail = [ln.strip() for ln in txt.strip().splitlines()[-3:]]
            for ln in tail:
                if re.fullmatch(r"S\d+", ln):
                    self._labels[ln] = i
        self._labels = _fill_labels(self._labels, len(self._pages))

    def text_lines(self, label: str) -> list[str]:
        return self._pages[self._labels[label]].splitlines()


# ----------------------------------------------------------------------------- data tables
def _parse_rows(token_rows: list[list[str]], ncomp: int, nnuc: int) -> list[list[str | None]]:
    rows = []
    for toks in token_rows:
        if not toks or not CONC_RE.match(toks[0]):
            continue
        if len(toks) != ncomp + nnuc:
            raise ValueError(f"data row with {len(toks)} tokens, expected {ncomp + nnuc}: {toks}")
        vals: list[str | None] = []  # printed strings: keep the printed precision
        for i, t in enumerate(toks):
            if t == "-":
                if i < ncomp:
                    raise ValueError(f"missing concentration in row {toks}")
                vals.append(None)
            else:
                if not NUM_RE.match(t):
                    raise ValueError(f"non-numeric token {t!r} in row {toks}")
                vals.append(t)
        rows.append(vals)
    return rows


def data_table_A(src: PdfplumberSource, spec) -> list[list[float | None]]:
    toks = [[w["text"] for w in line] for line in src.lines(spec["data_page"])]
    return _parse_rows(toks, len(spec["reagents"]), len(spec["table_nuclei"]))


def data_table_B(src: PdftotextSource, spec) -> list[list[float | None]]:
    toks = [ln.split() for ln in src.text_lines(spec["data_page"])]
    return _parse_rows(toks, len(spec["reagents"]), len(spec["table_nuclei"]))


# ----------------------------------------------------------------------------- results pages
def _stoich_label(label: str, reagents: list[str]) -> tuple[int, ...]:
    if label in reagents:
        return tuple(int(r == label) for r in reagents)
    parts = tuple(int(x) for x in label.split(","))
    if len(parts) != len(reagents):
        raise ValueError(f"species label {label} incompatible with reagents {reagents}")
    return parts


def species_name(stoich: tuple[int, ...], reagents: list[str]) -> str:
    out = ""
    for n, r in zip(stoich, reagents):
        if n == 1:
            out += r
        elif n > 1:
            out += f"{r}{n}"
    return out


def _scalars(text: str) -> dict[str, Any]:
    def grab(pat, cast=float):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None

    out = {
        "n_spectra": grab(r"no\. of spectra\s+(\d+)", int),
        "n_values": grab(r"no\. of resonance values\s+(\d+)", int),
        "n_nuclei": grab(r"no\. of resonant nuclei\s+(\d+)", int),
        "chi_squared": grab(r"Chi-squared\s*=\s*([\d.]+)"),
        "sigma": grab(r"sigma\s*=\s*([\d.]+)"),
        "rms_weighted_residual": grab(r"RMS weighted residual\s*=\s*([\d.]+)"),
    }
    m = re.search(r"HypNMR version:\s*([\d.]+)", text)
    out["program"] = "HypNMR"
    out["version"] = m.group(1) if m else None
    m = re.search(r"Hqdll version:\s*([\d/]+)", text)
    out["hqdll_version"] = m.group(1) if m else None
    return out


def _betas(lines: list[str], reagents: list[str]) -> list[dict[str, Any]]:
    out = []
    n = len(reagents)
    for ln in lines:
        toks = ln.split()
        if not toks or toks[0] != "Beta":
            continue
        stoich = tuple(int(t) for t in toks[1:1 + n])
        status = toks[1 + n]
        nums = [t for t in toks[2 + n:] if NUM_RE.match(t)]
        rec = {"stoich": list(stoich), "name": species_name(stoich, reagents)}
        if status == "refined":
            rec.update(status="refined", beta=float(nums[0]), rel_sd=float(nums[1]),
                       log_beta=float(nums[2]), sd=float(nums[3]))
        elif status == "constant":
            rec.update(status="fixed", beta=float(nums[0]), log_beta=float(nums[1]),
                       sd=float(nums[2]) if len(nums) > 2 else None)
        else:
            raise ValueError(f"unknown beta status in {ln!r}")
        out.append(rec)
    return out


def _correlation(lines: list[str]) -> dict[str, Any] | None:
    start = next((i for i, ln in enumerate(lines) if "Correlation coefficients*1000" in ln), None)
    if start is None:
        return None
    params = {}
    for ln in lines[start:]:
        for m in re.finditer(r"(\d+)\s+beta\s+([\d,]+)", ln):
            params[int(m.group(1))] = [int(x) for x in m.group(2).split(",")]
    npar = max(params) if params else 0
    lower: dict[int, list[int]] = {}
    for ln in lines[start + 1:]:
        toks = ln.split()
        if not toks or not re.fullmatch(r"\d+", toks[0]):
            continue
        i = int(toks[0])
        if i > npar or i in lower:
            continue
        ints = []
        for t in toks[1:]:
            if re.fullmatch(r"-?\d+", t) and len(ints) < i - 1:
                ints.append(int(t))
            else:
                break
        if len(ints) == i - 1 and (i > 1 or len(toks) == 1 or "beta" in ln):
            lower[i] = ints
    rows = [lower.get(i) for i in range(1, npar + 1)]
    if any(r is None for r in rows):
        raise ValueError(f"incomplete correlation matrix: {lower}")
    return {"params": [params[i] for i in range(1, npar + 1)], "lower_x1000": rows}


def _assign_chem(block: list[str], nums: list[str], nuc_reagent: str, reagents: list[str],
                 col_hint: list[int] | None) -> dict[str, dict]:
    """Pass B rule: species able to carry the nucleus are those containing its reagent."""
    ridx = reagents.index(nuc_reagent)
    able = [lab for lab in block if _stoich_label(lab, reagents)[ridx] > 0]
    vals = [float(x) for x in nums]
    out: dict[str, dict] = {}
    if len(vals) == 2 * len(able):
        for k, lab in enumerate(able):
            out[lab] = {"value": vals[2 * k], "error": vals[2 * k + 1]}
    elif len(vals) == len(able) == 1 and len(block) == 1:
        out[able[0]] = {"value": vals[0], "error": None}
    elif len(able) == 2 and len(vals) == 3 and col_hint is not None:
        # one species printed without error: the character columns tell which one
        # value and error of one species are closer to each other than to the other column
        first_has_pair = (col_hint[1] - col_hint[0]) < (col_hint[2] - col_hint[1])
        if first_has_pair:
            out[able[0]] = {"value": vals[0], "error": vals[1]}
            out[able[1]] = {"value": vals[2], "error": None}
        else:
            out[able[0]] = {"value": vals[0], "error": None}
            out[able[1]] = {"value": vals[1], "error": vals[2]}
    elif len(able) == 1 and len(vals) == 1:
        out[able[0]] = {"value": vals[0], "error": None}
    elif len(vals) == 0:
        pass
    else:
        raise ValueError(f"cannot assign {nums} to {block} for reagent {nuc_reagent}")
    return out


def _shift_blocks_B(lines: list[str], reagents: list[str], nuc_reagent: dict[str, str]):
    shifts: dict[str, dict[str, dict]] = {}
    block: list[str] | None = None
    label_re = re.compile(r"^(?:[A-Z]|\d+(?:,\d+)+)$")
    in_section = False
    for ln in lines:
        if "Individual chemical shifts" in ln:
            in_section = True
            continue
        if not in_section:
            continue
        if "Correlation coefficients" in ln or "Titration Plot" in ln:
            break
        toks = ln.split()
        if toks and all(label_re.match(t) for t in toks) and not re.fullmatch(r"S\d+", ln.strip()):
            block = toks
            continue
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9\-]*)\s*\+\s*(.*)$", ln)
        if m and block is not None and m.group(1) in nuc_reagent:
            nums = m.group(2).split()
            hint = [mm.start() for mm in re.finditer(r"\S+", m.group(2))]  # character columns
            for lab, rec in _assign_chem(block, nums, nuc_reagent[m.group(1)], reagents, hint).items():
                name = species_name(_stoich_label(lab, reagents), reagents)
                shifts.setdefault(m.group(1), {})[name] = rec
    return shifts


def _shift_blocks_A(lines: list[list[dict]], reagents: list[str], nuc_names: set[str]):
    """Pass A rule: x position. Column 0 numbers end at x1 < 260 pt, column 1 beyond;
    within a column the value ends left of the error."""
    shifts: dict[str, dict[str, dict]] = {}
    block = None
    label_re = re.compile(r"^(?:[A-Z]|\d+(?:,\d+)+)$")
    in_section = False
    for words in lines:
        texts = [w["text"] for w in words]
        joined = " ".join(texts)
        if "Individual chemical shifts" in joined:
            in_section = True
            continue
        if not in_section:
            continue
        if "Correlation" in joined or "Titration" in joined:
            break
        if texts and all(label_re.match(t) for t in texts) and not re.fullmatch(r"S\d+", joined):
            block = [(w["text"], 0.5 * (w["x0"] + w["x1"])) for w in words]
            continue
        if block is None or not texts or texts[0] not in nuc_names:
            continue
        nums = [w for w in words if NUM_RE.match(w["text"])]
        for w in nums:
            col = 0 if w["x1"] < 260 else 1
            if col >= len(block):
                raise ValueError(f"number {w['text']} outside block {block}")
            lab = block[col][0]
            name = species_name(_stoich_label(lab, reagents), reagents)
            # value columns end at ~186.7 / ~330.7 pt, error columns at ~240.7 / ~384.7 pt
            is_err = (w["x1"] > 200 and col == 0) or (w["x1"] > 345 and col == 1)
            rec = shifts.setdefault(texts[0], {}).setdefault(name, {"value": None, "error": None})
            rec["error" if is_err else "value"] = float(w["text"])
    return shifts


def results_B(src: PdftotextSource, spec) -> dict[str, Any]:
    lines: list[str] = []
    for lab in spec["results_pages"]:
        lines += src.text_lines(lab)
    text = "\n".join(lines)
    nuc_reagent = dict(spec["table_nuclei"])
    out = _scalars(text)
    out["betas"] = _betas(lines, spec["reagents"])
    out["shifts"] = _shift_blocks_B(lines, spec["reagents"], nuc_reagent)
    out["correlation"] = _correlation(lines)
    return out


def results_A(src: PdfplumberSource, spec) -> dict[str, Any]:
    lines: list[list[dict]] = []
    for lab in spec["results_pages"]:
        lines += src.lines(lab)
    flat = [" ".join(w["text"] for w in ln) for ln in lines]
    out = _scalars("\n".join(flat))
    out["betas"] = _betas(flat, spec["reagents"])
    out["shifts"] = _shift_blocks_A(lines, spec["reagents"], {n for n, _ in spec["table_nuclei"]})
    out["correlation"] = _correlation(flat)
    return out


# ----------------------------------------------------------------------------- speciation tables
def spec_table(lines_tokens: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    header = None
    rows = []
    for toks in lines_tokens:
        if toks and toks[0].startswith("xbound("):
            header = toks
            continue
        if header and toks and re.fullmatch(r"\d\.\d\d", toks[0]) and len(toks) == len(header):
            if not all(NUM_RE.match(t) for t in toks):
                raise ValueError(f"non-numeric speciation row {toks}")
            rows.append(list(toks))
    if header is None:
        raise ValueError("speciation header not found")
    return header, rows


# ----------------------------------------------------------------------------- checks & output
def _compare(a, b, path="") -> list[str]:
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: only in {'A' if k in a else 'B'}")
            else:
                diffs += _compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += _compare(x, y, f"{path}[{i}]")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def split_datasets(rows, spec) -> list[int]:
    if spec["datasets"] == "single":
        return [1] * len(rows)
    salt_idx = 1  # second reagent column is the salt cation (G/Q) in every case
    ids, cur, last = [], 0, None
    for r in rows:
        if r[salt_idx] != last:
            cur += 1
            last = r[salt_idx]
        ids.append(cur)
    return ids


def build_case(cid: str, spec, rows, res) -> tuple[dict[str, Any], list[list[Any]]]:
    reagents = spec["reagents"]
    tab_names = [n for n, _ in spec["table_nuclei"]]
    use = spec["use_nuclei"]
    nuc_reagent = dict(spec["table_nuclei"])
    dsid = split_datasets(rows, spec)
    nr = len(reagents)
    csv_rows = []
    for d, r in zip(dsid, rows):
        vals = [r[nr + tab_names.index(n)] for n in use]
        csv_rows.append([d] + r[:nr] + vals)
    # counts actually used by the case
    n_values = sum(v is not None for row in csv_rows for v in row[1 + nr:])
    species = []
    for b in res["betas"]:
        rec = {"name": b["name"], "stoich": b["stoich"], "status": b["status"]}
        if b["status"] == "fixed":
            rec["log_beta"] = b["log_beta"]
            rec["fixed_value_note"] = "printed log beta used as the constant value"
        species.append(rec)
    all_species = [species_name(tuple(int(i == j) for j in range(nr)), reagents) for i in range(nr)]
    all_species += [b["name"] for b in res["betas"]]
    stoich_of = {species_name(tuple(int(i == j) for j in range(nr)), reagents):
                 [int(i == j) for j in range(nr)] for i in range(nr)}
    stoich_of.update({b["name"]: b["stoich"] for b in res["betas"]})
    nuclei = []
    p_shift = 0
    for n in use:
        ridx = reagents.index(nuc_reagent[n])
        status = {}
        for sp in all_species:
            if stoich_of[sp][ridx] == 0:
                continue
            rec = res["shifts"].get(n, {}).get(sp)
            if rec is None:
                status[sp] = {"status": "absent"}
            elif rec["error"] is None:
                status[sp] = {"status": "fixed", "value": rec["value"]}
            else:
                status[sp] = {"status": "refined"}
                p_shift += 1
        nuclei.append({"name": n, "reagent": nuc_reagent[n], "limiting_shifts": status})
    p_beta = sum(b["status"] == "refined" for b in res["betas"])
    p = p_beta + p_shift
    m = res["n_values"]
    ratio = (res["sigma"] / res["rms_weighted_residual"]) ** 2
    reference = {k: res[k] for k in ("program", "version", "hqdll_version", "n_spectra", "n_values",
                                     "n_nuclei", "chi_squared", "sigma", "rms_weighted_residual")}
    reference["log_beta"] = {b["name"]: {k: b.get(k) for k in ("status", "beta", "rel_sd", "log_beta", "sd")}
                             for b in res["betas"]}
    reference["limiting_shifts"] = res["shifts"]
    reference["correlation_x1000"] = res["correlation"]
    if res["correlation"] is not None:
        reference["correlation_x1000"]["names"] = [species_name(tuple(s), reagents)
                                                   for s in res["correlation"]["params"]]
    src = REFS[spec["source"]]
    case = {
        "id": cid,
        "technique": "nmr",
        "kind": "fit",
        "description": spec["description"],
        "source": {"article": spec["source"], "citation": src["citation"], "doi": src["doi"],
                   "si_pages": [spec["data_page"]] + spec["results_pages"],
                   "data_table_page": spec["data_page"], "results_pages": spec["results_pages"],
                   "article_table": spec.get("article_table"), "notice": NOTICE},
        "reagents": reagents,
        "species": species,
        "nuclei": nuclei,
        "counts": {"n_spectra": len(csv_rows), "n_values": n_values, "n_nuclei": len(use),
                   "n_datasets": max(dsid)},
        "dof_identity": {"m": m, "p_beta": p_beta, "p_shifts": p_shift, "p": p,
                         "m_over_m_minus_p": m / (m - p), "sigma_over_rms_squared": ratio,
                         "rel_diff": ratio / (m / (m - p)) - 1.0},
        "reference": reference,
        "notes": spec.get("notes", []),
        "extraction": "validation/literature_extract.py (pdfplumber vs pdftotext, compared)",
    }
    return case, csv_rows


def check_case(cid, spec, case) -> list[str]:
    errs = []
    ref = case["reference"]
    c = case["counts"]
    for k in ("n_spectra", "n_values", "n_nuclei"):
        if c[k] != ref[k]:
            known = spec.get("known_count_mismatch", {}).get(k)
            if known == (c[k], ref[k]):
                print(f"  note {cid}: {k} in the table = {c[k]}, printed = {ref[k]} (documented)")
            else:
                errs.append(f"{cid}: count {k} {c[k]} != printed {ref[k]}")
    dof = case["dof_identity"]
    # sigma and RMS are printed with ~8 significant digits: the identity must hold to 1e-6
    if abs(dof["rel_diff"]) > 1e-6:
        errs.append(f"{cid}: dof identity fails: (sigma/rms)^2={dof['sigma_over_rms_squared']:.8f} "
                    f"vs m/(m-p)={dof['m_over_m_minus_p']:.8f} with p={dof['p']}")
    tab = spec.get("article_table")
    if tab:
        at = ARTICLE_TABLES[tab]
        for name, rec in ref["log_beta"].items():
            if name not in at:
                errs.append(f"{cid}: {name} missing from {tab}")
                continue
            lb_s, sd_s = at[name]
            nd = len(lb_s.split(".")[1])
            lb = float(lb_s)
            sd = float(sd_s) if sd_s is not None else None
            # the article value must be the SI value rounded (half-up) to the printed digits
            if abs(rec["log_beta"] - lb) > 0.5 * 10 ** -nd + 1e-9:
                errs.append(f"{cid}: {name} log beta {rec['log_beta']} vs {tab} {lb}")
            if sd is not None and rec.get("sd") is not None and abs(rec["sd"] - sd) > 0.5 * 10 ** -nd + 1e-9:
                known = spec.get("known_article_mismatch", {}).get(f"{name}.sd")
                if known == (rec["sd"], sd):
                    print(f"  note {cid}: {name} SD {rec['sd']} (SI) vs {sd} ({tab}) (documented)")
                else:
                    errs.append(f"{cid}: {name} SD {rec['sd']} vs {tab} {sd}")
        for key, refkey in (("_sigma", "sigma"), ("_rms", "rms_weighted_residual")):
            nd = len(at[key].split(".")[1])
            if abs(ref[refkey] - float(at[key])) > 0.5 * 10 ** -nd + 1e-9:
                errs.append(f"{cid}: {refkey} {ref[refkey]} vs {tab} {at[key]}")
    return errs


def write_csv(path: Path, header, rows):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else str(v)) for v in r])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf-dir", required=True, type=Path)
    ap.add_argument("--write", action="store_true", help="write tests/data/literature/<case>/")
    args = ap.parse_args(argv)
    srcA = {k: PdfplumberSource(args.pdf_dir / v["si_file"]) for k, v in REFS.items()}
    srcB = {k: PdftotextSource(args.pdf_dir / v["si_file"]) for k, v in REFS.items()}
    errors: list[str] = []
    built: dict[str, tuple] = {}
    for cid, spec in CASES.items():
        rowsA = data_table_A(srcA[spec["source"]], spec)
        rowsB = data_table_B(srcB[spec["source"]], spec)
        resA = results_A(srcA[spec["source"]], spec)
        resB = results_B(srcB[spec["source"]], spec)
        d = _compare(rowsA, rowsB, f"{cid}:data") + _compare(resA, resB, f"{cid}:results")
        errors += d
        case, csv_rows = build_case(cid, spec, rowsA, resA)
        errors += check_case(cid, spec, case)
        built[cid] = (case, csv_rows)
        dof = case["dof_identity"]
        print(f"{cid:15s} spectra={case['counts']['n_spectra']:3d} values={case['counts']['n_values']:3d} "
              f"nuclei={case['counts']['n_nuclei']} m/(m-p)={dof['m']}/{dof['m'] - dof['p']} "
              f"(p={dof['p_beta']}+{dof['p_shifts']}) identity rel.diff={dof['rel_diff']:+.1e} "
              f"A/B diffs={len(d)}")
    # second capture of the hand-transcribed article tables: every number must be
    # present as a token in the article text extracted by both tools
    import pdfplumber
    for art, key in (("R09a", ("R09a Table 1", "R09a Table 2")), ("R09b", ("R09b Table 1", "R09b Table 2", "R09b Table 3"))):
        path = args.pdf_dir / REFS[art]["article_file"]
        with pdfplumber.open(str(path)) as pdf:
            textA = " ".join((pg.extract_text() or "") for pg in pdf.pages)
        textB = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True,
                               capture_output=True, text=True).stdout
        tokA, tokB = set(re.findall(r"\d+\.\d+", textA)), set(re.findall(r"\d+\.\d+", textB))
        for tab in key:
            for name, val in ARTICLE_TABLES[tab].items():
                for sv in (val if isinstance(val, tuple) else (val,)):
                    if sv is None:
                        continue
                    for lab, toks in (("pdfplumber", tokA), ("pdftotext", tokB)):
                        if sv not in toks:
                            errors.append(f"{tab} {name}: {sv} not found in the article text ({lab})")
    print("article tables: transcription found in both text extractions" )
    # repeated tables must be identical
    def rows_of(cid):
        return built[cid][1]
    if _compare(rows_of("r09b-r-qx"), rows_of("r09a-r1-5sp")):
        errors.append("r09b-r-qx data differ from r09a-r1-5sp")
    ra = dict(built["r09a-r1-5sp"][0]["reference"]); rb = dict(built["r09b-r-qx"][0]["reference"])
    for k in ("version", "hqdll_version"):
        ra.pop(k, None); rb.pop(k, None)
    d = _compare(ra, rb, "r09b-r-qx vs r09a-r1-5sp results")
    errors += d
    s1 = [r[1:] for r in rows_of("r09a-p2-058") if r[0] == 1]
    s2 = [r[1:] for r in rows_of("r09a-p2-106") if r[0] == 1]
    if _compare(s1, s2):
        errors.append("0.15 mM titration differs between r09a-p2-058 and r09a-p2-106")
    first20 = [r[1:] for r in rows_of("r09a-r1-5sp") if r[0] == 1]
    two = [[r[1], r[2], r[2]] + r[3:] for r in rows_of("r09a-r1-11-3n")]
    if _compare(first20, two):
        errors.append("r09a-r1-11 table differs from the first titration of r09a-r1-5sp")
    print("repeated tables: r09b-r-qx == r09a-r1-5sp (data, results); p2 0.15 mM shared; "
          "r1 2-reagent == first titration of 3-reagent table" if not errors else "")
    # speciation tables
    spec_built = {}
    for sid, sp in SPEC_CASES.items():
        hA, rA = spec_table([[w["text"] for w in ln] for lab in sp["pages"] for ln in srcA[sp["source"]].lines(lab)])
        hB, rB = spec_table([ln.split() for lab in sp["pages"] for ln in srcB[sp["source"]].text_lines(lab)])
        d = _compare([hA, rA], [hB, rB], sid)
        errors += d
        spec_built[sid] = (hA, rA)
        print(f"{sid:15s} rows={len(rA)} cols={len(hA)} A/B diffs={len(d)}")
    if errors:
        print("\nEXTRACTION ERRORS:")
        for e in errors:
            print("  " + e)
        return 1
    if args.write:
        for cid, (case, csv_rows) in built.items():
            out = DATA_DIR / cid
            out.mkdir(parents=True, exist_ok=True)
            header = ["dataset"] + [f"T_{r}" for r in case["reagents"]] + [n["name"] for n in case["nuclei"]]
            write_csv(out / "data.csv", header, csv_rows)
            (out / "case.json").write_text(json.dumps(case, indent=1) + "\n")
        for sid, (hdr, rows) in spec_built.items():
            sp = SPEC_CASES[sid]
            out = DATA_DIR / sid
            out.mkdir(parents=True, exist_ok=True)
            write_csv(out / "table.csv", hdr, rows)
            src = REFS[sp["source"]]
            case = {"id": sid, "technique": "speciation", "kind": "speciation",
                    "description": sp["description"],
                    "source": {"article": sp["source"], "citation": src["citation"], "doi": src["doi"],
                               "si_pages": sp["pages"], "constants_from": sp["results_case"],
                               "notice": NOTICE},
                    "reagents": sp["reagents"], "analysed": sp["analysed"], "bound": sp["bound"],
                    "ancillary": sp["ancillary"], "columns": hdr, "n_rows": len(rows),
                    "extraction": "validation/literature_extract.py (pdfplumber vs pdftotext, compared)"}
            (out / "case.json").write_text(json.dumps(case, indent=1) + "\n")
        print(f"written to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
