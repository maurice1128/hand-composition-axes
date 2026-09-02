#!/usr/bin/env python
"""Estimate the compiled page count of docs/paper.tex.

This is a proxy, not a compile. It lays the document out in units of column
lines: body text wraps at a measured characters-per-line, headings and float
environments consume a known number of lines, and a full-width ``table*``
costs twice what its rows would cost in one column. The output is a range,
because the two quantities it cannot know -- where floats land and how badly
lines break -- both work in the same direction and only ever cost pages.

Accuracy is roughly +-10%. That is enough to answer "is this 8 pages or 12",
which is the question, and not enough to answer "is this 8.0 or 8.4". Treat a
result inside half a page of the limit as unresolved and compile it.

The layout constants are IEEEtran, 10pt, US letter, two columns:

  column width      3.5 in     -> ~55 characters per line at 10pt Times
  text height       9.25 in    -> 58 lines per column at a 11.5pt baseline
  columns per page  2          -> 116 line-slots per page

The constants above are IEEEtran's. **They are optimistic by about 15% under
`IEEEconf`, which is what RA-L asks for.** Measured once against a real compile:
this script said 9.5 pages where pdflatex produced 11. Treat its output as a
lower bound and compile before believing any margin.

    pdflatex -interaction=nonstopmode paper.tex     (run twice, for ef)

The limit is six pages. Two more are allowed at $175 each, and RA-L states that
supplemental text or figures are "grounds for editorial rejection without
review", so nothing can be moved out of the page budget into an appendix.

Usage:
    python scripts/estimate_pages.py [--tex docs/paper.tex] [--limit 8]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Characters that fit on one IEEEtran column line at 10pt.
CHARS_PER_LINE = 55
#: Lines in one column of an IEEEtran text block.
LINES_PER_COLUMN = 58
COLUMNS_PER_PAGE = 2
LINES_PER_PAGE = LINES_PER_COLUMN * COLUMNS_PER_PAGE

#: Line costs that are not body text. Each is the vertical space the
#: environment occupies, expressed in column lines.
COST = {
    "section": 3,        # \section, with its space above and below
    "subsection": 2.5,   # \subsection
    "paragraph_gap": 0.6,  # \parskip between paragraphs, amortised
    "table_frame": 4,    # \begin{table} .. \toprule .. \bottomrule .. caption
    "table_row": 1.15,   # one tabular row at \footnotesize, plus leading
    "figure_body": 17,   # a column-width scatter plot
    "bib_frame": 2,
}


def text_lines(s: str) -> float:
    """Column lines a run of body text occupies once wrapped."""
    # Macros contribute width very unevenly: \cite{ref12} prints as "[12]" and
    # \emph{p} as "p". Counting their source characters overstates the text by
    # roughly a fifth, which is a whole page over a document this size.
    s = re.sub(r"\\cite\{[^}]*\}", "[00]", s)
    s = re.sub(r"\\(?:emph|textbf|texttt)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = s.replace("{", "").replace("}", "")
    return max(1.0, len(s) / CHARS_PER_LINE)


def measure(tex: str) -> dict:
    """Walk the document once, adding up column lines by kind."""
    lines = tex.split("\n")
    i = 0
    acc = {"body": 0.0, "floats": 0.0, "bib": 0.0, "headings": 0.0}
    tables = {"single": 0, "wide": 0}
    in_body = False
    para: list[str] = []

    def flush() -> None:
        nonlocal para
        if para:
            acc["body"] += text_lines(" ".join(para)) + COST["paragraph_gap"]
            para = []

    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        if stripped.startswith(r"\begin{document}"):
            in_body = True
            i += 1
            continue
        if not in_body:
            i += 1
            continue

        m = re.match(r"\\begin\{(table\*?|figure\*?)\}", stripped)
        if m:
            flush()
            env = m.group(1)
            wide = env.endswith("*")
            rows = 0
            cap: list[str] = []
            in_cap = False
            depth = 0
            while i < len(lines) and not lines[i].strip().startswith(
                    r"\end{" + env + "}"):
                t = lines[i].strip()
                if t.endswith(r"\\"):
                    rows += 1
                if t.startswith(r"\caption{"):
                    in_cap = True
                if in_cap:
                    cap.append(t)
                    depth += t.count("{") - t.count("}")
                    if depth <= 0:
                        in_cap = False
                i += 1
            i += 1
            if env.startswith("figure"):
                cost = COST["figure_body"] + text_lines(" ".join(cap))
            else:
                cost = COST["table_frame"] + rows * COST["table_row"] \
                    + text_lines(" ".join(cap))
                tables["wide" if wide else "single"] += 1
            # A starred float spans both columns, so each of its lines costs
            # two column-line slots.
            acc["floats"] += cost * (2 if wide else 1)
            continue

        if stripped.startswith(r"\section{"):
            flush()
            acc["headings"] += COST["section"]
            i += 1
            continue
        if stripped.startswith(r"\subsection{"):
            flush()
            acc["headings"] += COST["subsection"]
            i += 1
            continue
        if stripped.startswith(r"\bibitem"):
            flush()
            j = i + 1
            entry = [stripped]
            while j < len(lines) and not lines[j].strip().startswith(
                    (r"\bibitem", r"\end{thebibliography}")):
                entry.append(lines[j].strip())
                j += 1
            acc["bib"] += text_lines(" ".join(entry)) + COST["paragraph_gap"]
            i = j
            continue
        if stripped.startswith("\\") and not stripped.startswith(
                (r"\emph", r"\textbf", r"\texttt", r"\cite", r"\footnote")):
            flush()
            i += 1
            continue
        if not stripped:
            flush()
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush()
    acc["bib"] += COST["bib_frame"]
    return {"acc": acc, "tables": tables}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default="docs/paper.tex")
    ap.add_argument("--limit", type=float, default=6.0,
                    help="page limit to report against; RA-L is 6 free, 8 paid")
    args = ap.parse_args()

    tex = (ROOT / args.tex).read_text(encoding="utf-8")
    r = measure(tex)
    acc, tables = r["acc"], r["tables"]
    total = sum(acc.values())
    # The title block, author block and abstract sit in a one-column span at
    # the top of page 1 and displace roughly a third of a page.
    total += LINES_PER_PAGE * 0.33

    low = total / LINES_PER_PAGE
    # Floats never pack perfectly; the slack is what a compile would reveal.
    high = low * 1.12

    print(f"{args.tex}")
    print(f"  body text        {acc['body']:7.0f} column lines")
    print(f"  headings         {acc['headings']:7.0f}")
    print(f"  floats           {acc['floats']:7.0f}   "
          f"({tables['single']} single-column, {tables['wide']} full-width, "
          f"each full-width line costing two slots)")
    print(f"  bibliography     {acc['bib']:7.0f}")
    print(f"  title block      {LINES_PER_PAGE * 0.33:7.0f}")
    print(f"  {'':17}{'-' * 7}")
    print(f"  total            {total:7.0f} column lines"
          f"  at {LINES_PER_PAGE} per page")
    print()
    print(f"  estimated pages  {low:.1f} to {high:.1f}   (limit {args.limit:.0f})")

    if low > args.limit:
        over = (low - args.limit) / low
        print(f"\n  OVER by at least {low - args.limit:.1f} pages. Cutting "
              f"{over * 100:.0f}% of everything, or the equivalent in floats, "
              f"would reach the limit.")
        if args.limit == 6.0:
            paid = max(0.0, low - 6.0)
            charged = min(2, int(paid) + (1 if paid % 1 else 0))
            print(f"  RA-L allows two paid pages at $175 each; at this length "
                  f"that is {charged} page(s), ${charged * 175}, "
                  f"and {'within' if low <= 8.0 else 'still over'} the "
                  f"eight-page maximum.")
        return 1
    if high > args.limit:
        print("\n  UNRESOLVED: the range straddles the limit. Compile it.")
        return 1
    print("\n  Fits, with the caveat that this is a proxy and not a compile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
