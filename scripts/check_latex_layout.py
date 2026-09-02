"""Static checks on the generated .tex, for the problems a compiler reports quietly.

No LaTeX toolchain is installed here, so the usual answer -- compile it and look
-- is unavailable. These are the failures that can be found by reading the
source, and they are the ones that survive a proofread: an Overfull \\hbox is a
warning among hundreds, and a table with a missing ampersand fails loudly but
only once the file is already on Overleaf.

    python scripts/check_latex_layout.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Rough capacity of one IEEEtran column at \footnotesize, in characters. A row
#: longer than this is a candidate for overflow, not a certainty: the estimate
#: cannot know the font metrics. It is a flag to look at, not a failure.
COLUMN_CHARS = 62
FULL_WIDTH_CHARS = 130


def visible(cell: str) -> int:
    """Characters a reader sees: markup and math delimiters do not take width."""
    s = re.sub(r"\\(textbf|emph|texttt|textit)\{", "", cell)
    s = re.sub(r"\\[a-zA-Z]+\{?|[{}$]", "", s)
    return len(s.strip())


def count_columns(spec: str) -> int:
    """Columns in a tabular preamble, where ``p{2.75in}`` is one, not eight."""
    n, i = 0, 0
    while i < len(spec):
        ch = spec[i]
        if ch in "lrc":
            n += 1
        elif ch in "pmb" and i + 1 < len(spec) and spec[i + 1] == "{":
            n += 1
            depth = 0
            while i < len(spec):
                if spec[i] == "{":
                    depth += 1
                elif spec[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
        i += 1
    return n


def check_tables(tex: str) -> list[str]:
    problems = []
    pattern = re.compile(r"\\begin\{(table\*?)\}(.*?)\\end\{\1\}", re.S)
    for i, m in enumerate(pattern.finditer(tex), 1):
        env, body = m.group(1), m.group(2)
        # One level of nesting is allowed, because ``p{2.75in}`` puts braces
        # inside the preamble and ``[^}]+`` stops at the first one -- which read
        # ``lp{5.30in}rrrll`` as two columns and reported every row as broken.
        spec_match = re.search(
            r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}", body)
        if not spec_match:
            problems.append(f"table {i}: no tabular environment")
            continue
        ncol = count_columns(spec_match.group(1))
        rows = [r for r in body.splitlines() if r.rstrip().endswith("\\\\")]

        for j, row in enumerate(rows):
            got = row.count("&") + 1
            if got != ncol:
                problems.append(
                    f"table {i} row {j + 1}: {got} cells for {ncol} columns "
                    f"-- LaTeX will refuse to build this")

        # A ``p{}`` column wraps, so the row's total length says nothing about
        # whether the table fits; only fixed-width preambles can overflow.
        if "p{" in spec_match.group(1):
            continue

        budget = FULL_WIDTH_CHARS if env == "table*" else COLUMN_CHARS
        widest = max((sum(visible(c) for c in r.split("&")) + 2 * (ncol - 1)
                      for r in rows), default=0)
        if widest > budget:
            problems.append(
                f"table {i} ({env}, {ncol} cols): widest row about {widest} "
                f"chars against roughly {budget} available -- likely to overflow"
                + ("; consider table*" if env == "table" else ""))
    return problems


def check_structure(tex: str) -> list[str]:
    problems = []
    pairs = [("\\begin{document}", "\\end{document}"),
             ("\\begin{abstract}", "\\end{abstract}"),
             ("\\begin{thebibliography}", "\\end{thebibliography}")]
    for open_, close in pairs:
        if tex.count(open_) != tex.count(close):
            problems.append(f"unbalanced: {tex.count(open_)} {open_} against "
                            f"{tex.count(close)} {close}")

    for env in ("itemize", "enumerate", "tabular", "figure", "table", "table*"):
        o, c = tex.count(f"\\begin{{{env}}}"), tex.count(f"\\end{{{env}}}")
        if o != c:
            problems.append(f"unbalanced {env}: {o} begin, {c} end")

    if "\\maketitle" not in tex:
        problems.append("no \\maketitle")
    if re.search(r"\\includegraphics[^{]*\{([^}]+)\}", tex):
        for g in re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", tex):
            if not (ROOT / "docs" / g).exists():
                problems.append(f"figure file missing: docs/{g}")

    # Unescaped specials outside math and \texttt survive as silent errors.
    # The already-escaped forms have to go first: stripping ``\[a-zA-Z]+`` alone
    # leaves the ``%`` of every ``\%`` behind and reports the whole paper as
    # broken.
    # Tabular bodies are removed first: every ``&`` in them is a column
    # separator doing its job, and counting those reported the whole paper as
    # broken on its own tables.
    # A ``%`` alone at end of line is the idiom that swallows the newline, as in
    # ``\resizebox{...}{!}{%``. It is deliberate, not a missing escape.
    stripped = re.sub(r"%[ \t]*$", "", tex, flags=re.M)
    stripped = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", stripped,
                      flags=re.S)
    stripped = re.sub(r"\\[%&#_${}]", "", stripped)
    stripped = re.sub(r"\$[^$]*\$|\\texttt\{[^}]*\}|\\[a-zA-Z]+", "", stripped)
    for ch in ("%", "&", "#"):
        loose = [m.start() for m in re.finditer(re.escape(ch), stripped)]
        if loose:
            problems.append(f"{len(loose)} unescaped '{ch}' outside math")

    left = sorted(set(re.findall(r"[^\x00-\x7f]", tex)))
    if left:
        problems.append("non-ASCII remains: "
                        + ", ".join(f"U+{ord(c):04X}" for c in left))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default="docs/paper.tex")
    args = ap.parse_args()

    tex = (ROOT / args.tex).read_text(encoding="utf-8")
    problems = check_structure(tex) + check_tables(tex)

    n_tab = len(re.findall(r"\\begin\{table\}", tex))
    n_tabs = len(re.findall(r"\\begin\{table\*\}", tex))
    print(f"{args.tex}: {len(tex.splitlines())} lines, "
          f"{tex.count(chr(92) + 'section{')} sections, "
          f"{n_tab} single-column tables, {n_tabs} full-width, "
          f"{tex.count(chr(92) + 'includegraphics')} figures")

    if not problems:
        print("\nno layout problems found by static inspection.")
        print("This does not replace compiling: line breaking, float placement "
              "and page count cannot be checked without LaTeX.")
        return 0
    print(f"\n{len(problems)} to look at:")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
