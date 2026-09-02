"""Gate the reporting, not just the experiments.

Every ``check_*.py`` in this project guards an experiment. Nothing guarded the
manuscript, and it showed: three consecutive independent reviews returned the
same class of blocking finding -- a number in the paper disagreeing with the
artefact it came from, or the same quantity quoted with two different values in
two sections. Every one of those was mechanically detectable.

Two failure modes, both of which have actually occurred here:

1. **Computing before the artefact is complete.** A sweep's ``results.json``
   files are written per shard, so a statistic taken while shards are still
   running silently uses a subset. The confirmatory rate control was reported
   at 66 seeds, then 68, before settling at the released 70.
2. **A quantity quoted twice and updated once.** When a figure is corrected in
   one section and left stale in another, both survive review by anyone reading
   linearly.

This checks the load-bearing figures against ``runs/`` and cross-checks that
repeated quantities agree. It does not attempt to parse every number in the
paper -- a gate that tries to do everything is a gate nobody runs.

    python scripts/check_paper_numbers.py
"""

from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "PAPER_RAL.md"
sys.path.insert(0, str(ROOT / "scripts"))


def ratematch() -> dict[str, object]:
    """The confirmatory rate control, over every seed the release contains."""
    by: dict[int, dict] = collections.defaultdict(dict)
    n_records = 0
    for f in sorted(ROOT.glob("runs/ratematch_*/results.json")):
        rows = json.loads(f.read_text())["results"]
        n_records += len(rows)
        for r in rows:
            by[r["seed"]][r["kind"]] = r
    complete = sorted(s for s, v in by.items() if len(v) == 3)
    out = {"records": n_records, "seeds": len(by), "complete": len(complete)}
    for arm in ("perframe", "perframe_rate"):
        d = np.array([by[s][arm]["penalty"] - by[s]["modular"]["penalty"] for s in complete])
        wins = int((d > 0).sum())
        out[arm] = {
            "mean": float(d.mean()),
            "wins": wins,
            "n": len(d),
            "d": float(d.mean() / d.std(ddof=1)),
            "t_p": float(stats.ttest_1samp(d, 0).pvalue),
            "sign_p": float(stats.binomtest(wins, len(d), 0.5).pvalue),
        }
    return out


def realised_rates() -> dict[str, float]:
    """Final-epoch validation rates per arm, over every history in the release.

    The manuscript quoted 2.293 nats/frame for the modular arm here; the
    complete release gives 2.312. The earlier figure came from a subset written
    while shards were still running, which is the same failure this file's
    docstring describes for the penalty figures -- so the rates are gated too.
    """
    import os
    import re as _re
    import statistics as st

    def last(f: Path) -> dict:
        h = json.loads(f.read_text())
        rows = h["epochs"] if isinstance(h, dict) and "epochs" in h else h
        return rows[-1] if isinstance(rows, list) else rows

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for f in ROOT.glob("runs/ratematch_*/*/history.json"):
        m = _re.match(r"s\d+_(.+)_b\d+_(naive|informed)$", f.parent.name)
        if m:
            buckets[m.group(1)].append(last(f))
    out: dict[str, float] = {}
    for kind, rows in buckets.items():
        if kind.startswith("perframe"):
            out[kind] = st.mean(r["val_kl"] for r in rows)
        else:
            out[kind] = st.mean(r["val_usage_entropy"] - r["val_assign_entropy"] for r in rows)
        out[kind + "_n"] = len(rows)
    return out


def floor_at_256() -> float:
    """Baseline penalty on a bundle whose true penalty is zero by construction."""
    f = ROOT / "runs" / "pc_easy_ratematched" / "results.json"
    rows = json.loads(f.read_text())["results"]
    vals = [r["penalty"] for r in rows if r["budget"] == 256 and r["kind"] == "perframe"]
    return float(np.mean(vals))


#: Pairs of runs the manuscript sets side by side. Any argument that differs
#: between two compared sweeps has to be named in the text: a reader told "the
#: same configuration" and handed two numbers will read the difference as noise.
#: The rate control and the original sweep differ in ``min_per_composition``
#: (4 against 5), which sets the round-robin swap depth and so changes the
#: informed set while leaving the naive set identical -- a review caught that
#: after the paper had already called the two runs identically configured.
COMPARED: tuple[tuple[str, str], ...] = (
    ("runs/ratematch_a", "runs/oakink_official_category"),
)

#: Arguments that cannot differ between compared runs without being said.
MATERIAL_ARGS = (
    "granularity", "held_compositions", "min_chains", "min_per_composition",
    "budgets", "epochs", "window", "stride", "latent_dim", "n_primitives",
)


def compared_run_args() -> list[str]:
    """Differences between runs the manuscript compares directly."""
    notes: list[str] = []
    for left, right in COMPARED:
        try:
            a = json.loads((ROOT / left / "results.json").read_text())["args"]
            b = json.loads((ROOT / right / "results.json").read_text())["args"]
        except FileNotFoundError:
            continue
        for key in MATERIAL_ARGS:
            if a.get(key) != b.get(key):
                notes.append(f"{left} and {right} differ in {key}: "
                             f"{a.get(key)!r} against {b.get(key)!r}")
    return notes



def gate_exit_codes() -> list[str]:
    """Do the released gate artefacts agree with their own exit codes?

    A gate that prints TOO THIN on every row and then exits 0 is worse than no
    gate: the manuscript cites its rows while a reader who runs it gets the
    opposite verdict. That happened here -- check_informed_coverage.py decided
    each row on breadth and depth but tested only depth in its summary, so the
    DexYCB run the paper withdraws would have passed if run as released. It
    survived twenty review rounds because every check read the rows.

    This re-runs nothing. It reads each released gate artefact and asserts that
    a file containing failure rows also contains a failure verdict.
    """
    import re as _re

    FAIL_ROW = _re.compile(r"TOO THIN|FAIL\b|DEAD |PINNED ")
    FAIL_SUMMARY = _re.compile(r"ARTEFACT RISK|FAIL:|does not|unusable")
    PASS_SUMMARY = _re.compile(r"adequate|PASS:|Balanced enough|no layout problems")
    notes: list[str] = []
    # Both globs: gate artefacts have lived in ``runs/`` as well as
    # ``runs/gates/``, and the two files that actually exhibited this bug were
    # in the former -- so a check that looked only in ``runs/gates`` could not
    # see the defect it exists for.
    files = sorted(ROOT.glob("runs/gates/*.txt")) + sorted(ROOT.glob("runs/cover_*.txt"))         + sorted(ROOT.glob("runs/leak_*.txt"))
    for f in files:
        body = f.read_text(errors="replace")
        rows_fail = bool(FAIL_ROW.search(body))
        says_fail = bool(FAIL_SUMMARY.search(body))
        says_pass = bool(PASS_SUMMARY.search(body))
        if rows_fail and says_pass and not says_fail:
            notes.append(f"{f.name}: rows report failures and the summary says it passed")
        if not rows_fail and says_fail:
            notes.append(f"{f.name}: summary reports failure with no failing row")
    return notes


def repeated_figures(text: str) -> list[tuple[str, list[str]]]:
    """Signed effect sizes that appear under more than one heading."""
    section = None
    seen: dict[str, list[str]] = collections.defaultdict(list)
    for line in text.split("\n"):
        head = re.match(r"^#{2,3} (.+)", line)
        if head:
            section = head.group(1)[:28]
            continue
        for tok in re.findall(r"[+\u2212-]0\.0\d{3,4}", line):
            seen[tok.replace("\u2212", "-")].append(section or "?")
    return [(k, sorted(set(v))) for k, v in sorted(seen.items()) if len(set(v)) > 1]


def section_pointers(text: str) -> list[str]:
    """Every cross-reference must name a section that exists.

    A cut that removes or renumbers a subsection leaves pointers behind, and
    the pointer still reads plausibly, so nothing downstream complains. One
    review found two of these in a single paragraph, both pointing at a
    section that had been rewritten to defer elsewhere.
    """
    bad = []
    body = text.split("## References")[0]
    heads = re.findall(r"^#{2,3} (?:([IVX]+)\.|([A-Z])\.) ", body, re.M)
    sections = {a for a, b in heads if a}
    subsections = {b for a, b in heads if b}
    for roman, letter in set(re.findall(r"\u00a7([IVX]+)-([A-Z])", body)):
        if letter not in subsections:
            bad.append(f"\u00a7{roman}-{letter} is referenced but no such subsection exists")
    for roman in set(re.findall(r"\u00a7([IVX]+)(?![-\w])", body)):
        if roman not in sections:
            bad.append(f"\u00a7{roman} is referenced but no such section exists")
    return bad


def table_pointers(text: str) -> list[str]:
    """Every "Table N" must have a captioned table N, in document order.

    LaTeX numbers tables by position, so deleting one silently renumbers every
    reference after it and the manuscript still compiles.
    """
    bad = []
    body = text.split("## References")[0]
    captions = re.findall(r"\*\*TABLE ([IVX]+):", body)
    want = ["I", "II", "III", "IV", "V", "VI"][:len(captions)]
    if captions != want:
        bad.append(f"tables are captioned {captions} but LaTeX will number them {want}")
    for ref in set(re.findall(r"Table ([IVX]+)", body)):
        if ref not in captions:
            bad.append(f"the text refers to Table {ref}, which has no captioned table")
    return bad


#: Words the manuscript uses for small counts, so a claim written out in
#: English can be compared against a table it is counting.
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10}

#: "a fifth" names the fifth axis, so it asserts a count of five.
ORDINALS = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


def advantage_counts(text: str) -> list[str]:
    """A sentence counting Table II's rows must agree with Table II.

    The draft said "positive advantage on five of the seven axes" above a table
    showing six positive advantages. The claim is one line from its own
    refutation and survived three rounds of review.
    """
    body = text.split("## References")[0]
    # A wins/n fraction in parentheses is Table II's signature; Table III
    # also parenthesises a number, but a z rather than a fraction.
    rows = [ln for ln in body.split("\n")
            if ln.startswith("|")
            and re.search(r"\|\s*\**[+−-]?0\.\d+ \(\d+/\d+\)", ln)]
    if not rows:
        return ["Table II's rows were not found; the advantage check did not run"]
    positive = significant = 0
    for ln in rows:
        cells = [c.strip().strip("*") for c in ln.strip("|").split("|")]
        adv = cells[1].split(" ")[0].replace("\u2212", "-")
        p = cells[3].replace("<", "").strip("*")
        try:
            if float(adv) > 0:
                positive += 1
            if float(p) < 0.05:
                significant += 1
        except ValueError:
            continue
    bad = []
    flat = " ".join(body.split())
    m = re.search(r"advantage is positive on (\w+) of the (\w+)(?: \w+)? axes", flat)
    if not m:
        bad.append("no sentence states how many axes carry a positive advantage")
        return bad
    said_pos, said_tot = (WORDS.get(g.lower(), -1) for g in m.groups())
    if said_pos != positive:
        bad.append(f"the text says {said_pos} axes carry a positive advantage; "
                   f"Table II shows {positive}")
    if said_tot != len(rows):
        bad.append(f"the text says {said_tot} axes; Table II has {len(rows)} rows")

    # Section IV declares a t-test AND a sign test. Table II carries only the
    # t-test, so a bare count of "significant" axes silently means the weaker
    # standard; the manuscript has to say which it means.
    pair = re.search(r"clears \u00a7IV's declared pair of tests on (\w+)", flat)
    alone = re.search(r"a (\w+), [^.]*?, clears the \*t\*-test alone", flat)
    if not pair or not alone:
        bad.append("the axis count does not say which test it is counting; "
                   "\u00a7IV declares a t-test and a sign test together")
        return bad
    said_pair = WORDS.get(pair.group(1).lower(), -1)
    said_alone = ORDINALS.get(alone.group(1).lower(), -1)
    if said_alone != said_pair + 1:
        bad.append(f"the text says {said_pair} clear both tests and calls the "
                   f"next one the {alone.group(1)}; those do not line up")
    if said_alone != significant:
        bad.append(f"the text implies {said_alone} axes clear the t-test; "
                   f"Table II shows {significant} at p < 0.05")
    return bad


def main() -> int:
    text = PAPER.read_text(encoding="utf-8")
    rm = ratematch()
    bad: list[str] = []
    bad += section_pointers(text)
    bad += table_pointers(text)
    bad += advantage_counts(text)

    print(f"rate control: {rm['records']} records over {rm['seeds']} seeds, "
          f"{rm['complete']} with every arm")
    if rm["records"] != rm["seeds"] * 3:
        bad.append(f"INCOMPLETE: {rm['records']} records for {rm['seeds']} seeds; "
                   "a shard is still writing and any statistic taken now is a subset")

    for arm, label in (("perframe", "unmatched"), ("perframe_rate", "rate-matched")):
        a = rm[arm]
        print(f"  {label:12s} {a['mean']:+.5f}  {a['wins']}/{a['n']}  d={a['d']:.2f}  "
              f"t p={a['t_p']:.4f}  sign p={a['sign_p']:.3f}")
        if f"{a['mean']:+.5f}".replace("+", "+") not in text:
            bad.append(f"{label} mean {a['mean']:+.5f} does not appear in the paper")
        if f"{a['wins']} of {a['n']}" not in text:
            bad.append(f"{label} win count '{a['wins']} of {a['n']}' does not appear")

    fl = floor_at_256()
    print(f"  false-positive floor (budget 256) {fl:+.5f}")
    if f"{fl:+.5f}" not in text:
        bad.append(f"floor {fl:+.5f} does not appear in the paper")

    rates = realised_rates()
    print("")
    print("realised per-frame rates (final-epoch validation):")
    for kind in ("perframe", "perframe_rate", "modular"):
        if kind not in rates:
            continue
        print(f"  {kind:14s} n={int(rates[kind + '_n']):3d}  {rates[kind]:.4f} nats/frame")
        if f"{rates[kind]:.3f}" not in text:
            bad.append(f"{kind} rate {rates[kind]:.3f} does not appear in the paper")
    if "perframe_rate" in rates and "modular" in rates:
        matched = rates["modular"] / rates["perframe_rate"]
        unmatched = rates["modular"] / rates["perframe"]
        print(f"  ratio matched {matched:.2f}   unmatched {unmatched:.2f}")
        for label, value in (("matched", matched), ("unmatched", unmatched)):
            if f"{value:.2f}" not in text and f"{value:.1f}" not in text:
                bad.append(f"{label} rate ratio {value:.2f} does not appear in the paper")

    diffs = compared_run_args()
    print("")
    print("configuration differences between runs the paper compares:")
    for note in diffs or ["none"]:
        print(f"  {note}")
    for note in diffs:
        key = note.split(" differ in ")[1].split(":")[0]
        if key not in text:
            bad.append(f"{note}; {key!r} is never named in the paper")

    gates = gate_exit_codes()
    print("")
    print("released gate artefacts whose rows and verdict disagree:")
    for note in gates or ["none"]:
        print(f"  {note}")
    bad.extend(gates)

    print("\nfigures quoted under more than one heading:")
    for tok, secs in repeated_figures(text):
        print(f"  {tok:>10s}  {secs}")

    if bad:
        print("\nFAIL")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("\nEvery load-bearing figure checked appears in the manuscript, and the "
          "release is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
