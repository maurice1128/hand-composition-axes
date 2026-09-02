"""Convert the Markdown draft into an IEEEtran submission.

The paper is written and edited in Markdown because that is where the numbers
get checked against `runs/`, and rewriting it into LaTeX by hand would mean two
sources of truth from the first revision onwards. Every trimming pass, every
reviewer response and every re-run that moves a number would then have to be
applied twice, and the second application is the one that gets forgotten.

So the Markdown stays authoritative and this regenerates the .tex. It is a
converter for *this* document, not a general one: it handles the constructs the
draft actually uses and raises on anything it does not recognise, because a
converter that silently drops a table is worse than one that stops.

No LaTeX toolchain is installed here, so the output is not compiled. Upload
`paper.tex` with `docs/figures/` to Overleaf, which ships IEEEtran, or run
pdflatex locally if you have it.

    python scripts/build_latex.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREAMBLE = r"""\documentclass[letterpaper, 10pt, conference]{ieeeconf}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{textcomp}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{url}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em T\kern-.1667em
    \lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{%(title)s}
\author{Mu-Hua Wang\\
National Yang Ming Chiao Tung University\\
{\tt\small mauricewang1128@gmail.com}}
\maketitle
"""

#: Characters that are legal in Markdown and fatal or wrong in LaTeX. Order
#: matters: the backslash must be escaped before anything that introduces one.
ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
    ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"), ("$", r"\$"),
]

#: Unicode the draft uses freely and IEEEtran's fonts do not all carry.
UNICODE = {
    "\u2014": "---", "\u2013": "--", "\u2018": "`", "\u2019": "'",
    "\u201c": "``", "\u201d": "''", "\u2026": r"\ldots{}",
    "\u00d7": r"$\times$", "\u00b1": r"$\pm$", "\u00b0": r"$^{\circ}$",
    "\u2192": r"$\rightarrow$", "\u2264": r"$\leq$", "\u2265": r"$\geq$",
    "\u03c1": r"$\rho$", "\u03b7": r"$\eta$", "\u03b2": r"$\beta$",
    "\u2705": "", "\u23f3": "", "\u2717": "", "\u2713": "",
    "\u00b2": r"$^{2}$", "\u00b3": r"$^{3}$", "\u2075": r"$^{5}$",
    "\u2261": r"$\equiv$", "\u2248": r"$\approx$", "\u00a0": " ",
    "\u2212": "-",              # true minus, not a hyphen
    "\u00a7": r"\S{}",          # braces, or LaTeX reads the next letter as part of it
    "\u03bc": r"$\mu$", "\u03c3": r"$\sigma$", "\u0394": r"$\Delta$", "\u03b1": r"$\alpha$",
    "\u2260": r"$\neq$", "\u2022": r"$\bullet$", "\u2032": r"$'$",
}


def inline(text: str) -> str:
    """Markdown inline markup to LaTeX, escaping everything else."""
    # Math and code are extracted first so their contents are not escaped.
    held: list[str] = []

    def hold(s: str) -> str:
        held.append(s)
        return f"\x00{len(held) - 1}\x00"

    # Before the citation rule, which would read "Fig. 1" as a bibliography key.
    text = re.sub(r"\bFig\. 1\b", lambda m: hold(r"Fig.~\ref{fig:screen}"), text)
    # Before the citation rule: ``[^name]`` must not be read as ``[1]``.
    text = re.sub(r"\[\^([\w-]+)\]",
                  lambda m: hold(FOOTNOTES[m.group(1)]), text)
    text = re.sub(r"\$([^$]+)\$", lambda m: hold(f"${m.group(1)}$"), text)
    text = re.sub(r"`([^`]+)`",
                  lambda m: hold(r"\texttt{" + re.sub(r"([_&%#{}])", r"\\\1",
                                                      m.group(1)) + "}"), text)
    # Citations: [1] or [1], [2] -> \cite{ref1,ref2}
    text = re.sub(r"\[(\d+)\](,\s*\[(\d+)\])*",
                  lambda m: hold(r"\cite{"
                                 + ",".join("ref" + n for n in
                                            re.findall(r"\d+", m.group(0)))
                                 + "}"), text)
    # Markdown links: keep the text, drop the target (all targets are local).
    text = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Unicode expands to LaTeX, so it has to be protected the same way math is:
    # substituting first and escaping afterwards turned every ``10^2`` into
    # ``10\$\textasciicircum{}\{2\}\$``.
    for ch, rep in UNICODE.items():
        if ch in text:
            text = text.replace(ch, hold(rep) if rep else "")

    for ch, rep in ESCAPES:
        text = text.replace(ch, rep)

    # Straight quotes are Markdown's; LaTeX wants the directional pair.
    text = re.sub(r'"([^"]*)"', r"``\1''", text)

    # Non-greedy, and bold before italic, so ``***r* = 0.863**`` -- bold wrapping
    # an italic, which the draft uses for every headline number -- resolves
    # instead of leaking asterisks into the output.
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*(.+?)\*(?!\*)", r"\\emph{\1}", text)

    for i, s in enumerate(held):
        text = text.replace(f"\x00{i}\x00", s)
    return text


#: Beyond this many columns a table cannot fit IEEEtran's 3.5-inch column and
#: has to span both. Chosen from the draft's own tables: the five-column results
#: tables fit at \footnotesize, the seven-column screen table does not.
MAX_SINGLE_COLUMN = 5


#: A ``**TABLE N: caption**`` paragraph immediately before a Markdown table
#: becomes that table's ``\caption``. Tables without one stay uncaptioned, and
#: therefore unnumbered, which is how the two inline tables in III-A and III-B
#: are meant to read. Numbering is LaTeX's, so the order of captioned tables in
#: the file is what "Table II" in the prose resolves to.
PENDING_CAPTION: list[str] = []

#: Figure labels, in the order they reach the output.
FIGURES: list[str] = []

#: The numerals of captioned tables, in the order they reach the output. LaTeX
#: numbers by position, so this has to read I, II, III, ... for the prose's
#: "Table II" to point where it says.
CAPTION_ORDER: list[str] = []


def table(rows: list[str]) -> str:
    """A Markdown pipe table to a booktabs tabular inside a table float.

    Wide tables become ``table*``, which spans both columns. Left in ``table``
    they overflow the column and LaTeX reports it only as an Overfull \\hbox
    warning, which is easy to scroll past and shows up in the PDF as a table
    running off the page edge.
    """
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]        # cells[1] is the ---|--- rule
    ncol = len(header)
    wide = ncol > MAX_SINGLE_COLUMN
    env = "table*" if wide else "table"

    # Column types are inferred from the cells rather than assumed. Every column
    # was previously `l` or `r`, neither of which wraps, so the gate table --
    # whose second column holds whole sentences -- ran off the page edge with
    # nothing but an Overfull \hbox to say so.
    def numeric(col: int) -> bool:
        vals = [r[col] for r in body if col < len(r) and r[col].strip()]
        return bool(vals) and sum(
            # The draft writes negatives with a true minus, so a class holding
            # only the ASCII hyphen classified every negative column as text.
            bool(re.match(r"^[\s*_+\-−±]*[\d.]", v))
            for v in vals) >= 0.7 * len(vals)

    def longest(col: int) -> int:
        return max([len(re.sub(r"[*_`]", "", r[col]))
                    for r in body + [header] if col < len(r)] or [0])

    # IEEEtran gives a column about 3.5in and a full-width float about 7.16in;
    # these leave a margin for the inter-column gaps booktabs does not count.
    total = 6.9 if wide else 3.3
    text_cols = [c for c in range(ncol) if not numeric(c) and longest(c) > 22]
    spec_parts = []
    for c in range(ncol):
        if c in text_cols:
            share = longest(c) / sum(longest(t) for t in text_cols)
            numeric_room = 0.55 * sum(1 for k in range(ncol) if k not in text_cols)
            spec_parts.append(f"p{{{share * (total - numeric_room):.2f}in}}")
        else:
            spec_parts.append("r" if numeric(c) else "l")
    spec = "".join(spec_parts)
    # Widths above are an estimate and a compile showed every table
    # overflowing its column by 15 to 87pt. \resizebox makes the fit exact:
    # a table that would overflow is scaled down instead, and one that fits is
    # left alone, so the estimate only has to be close.
    fit = r"\columnwidth" if not wide else r"\textwidth"
    out = [r"\begin{" + env + "}[t]", r"\centering", r"\footnotesize",
           r"\resizebox{" + fit + r"}{!}{%",
           r"\begin{tabular}{" + spec + "}", r"\toprule",
           " & ".join(inline(h) for h in header) + r" \\", r"\midrule"]
    for row in body:
        row = (row + [""] * ncol)[:ncol]
        out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}"]
    if PENDING_CAPTION:
        cap = PENDING_CAPTION.pop()
        num, _, text = cap.partition(":")
        numeral = num.split()[-1].strip()
        CAPTION_ORDER.append(numeral)
        out.append(r"\caption{" + inline(text.strip()) + "}")
        out.append(r"\label{tab:" + numeral.lower() + "}")
    out.append(r"\end{" + env + "}")
    return "\n".join(out)


#: ``[^name]`` -> a finished ``\footnote{...}``. Filled by :func:`footnotes`
#: and consumed inside :func:`inline`, which must hold the result rather than
#: substitute it: a backslash placed into the text before escaping runs comes
#: out as ``\textbackslash{}footnote\{``.
FOOTNOTES: dict[str, str] = {}


def footnotes(md: str) -> str:
    """Fold Markdown footnote definitions into their markers.

    IEEEtran has no footnote-definition syntax, so ``[^name]`` becomes a
    ``\footnote{}`` carrying the text of its ``[^name]:`` block and the block
    itself is deleted. An undefined marker is a hard error rather than a silent
    passthrough, because such a marker reaches the PDF as literal brackets,
    which is how three of them shipped in an earlier draft.
    """
    defs: dict[str, str] = {}

    def take(m):
        name = re.match(r"^\[\^([\w-]+)\]:", m.group(0)).group(1)
        defs[name] = " ".join(m.group(0).split(":", 1)[1].split())
        return ""

    # A definition runs to the first blank line, so it may wrap over lines.
    md = re.sub(r"^\[\^[\w-]+\]:[^\n]*(?:\n(?![ \t]*$)[^\n]*)*", take, md,
                flags=re.M)

    FOOTNOTES.clear()
    for name, body in defs.items():
        FOOTNOTES[name] = r"\footnote{" + inline(body) + "}"

    markers = set(re.findall(r"\[\^([\w-]+)\]", md))
    undefined = markers - set(defs)
    if undefined:
        raise SystemExit("footnote markers with no definition: "
                         + ", ".join(sorted(undefined)))
    unused = set(defs) - markers
    if unused:
        raise SystemExit("footnote definitions never referenced: "
                         + ", ".join(sorted(unused)))
    return md


def convert(md: str) -> str:
    md = footnotes(md)
    lines = md.splitlines()

    out: list[str] = []
    i, in_refs = 0, False

    while i < len(lines):
        line = lines[i]

        # A figure block may be hard-wrapped, so gather to its closing brace.
        if line.startswith("!["):
            block = [line]
            while i + 1 < len(lines) and not block[-1].rstrip().endswith("}"):
                i += 1
                block.append(lines[i])
            m_fig = re.match(r"^!\[(.*)\]\((figures/[^)]+)\)\{#(fig:[\w-]+)\}$",
                             " ".join(" ".join(block).split()))
            if not m_fig:
                raise SystemExit("malformed figure block: " + " ".join(block)[:70])
            cap, path, key = m_fig.groups()
            out += [r"\begin{figure}[t]", r"\centering",
                    r"\includegraphics[width=\columnwidth]{" + path + "}",
                    r"\caption{" + inline(cap) + "}",
                    r"\label{" + key + "}", r"\end{figure}"]
            FIGURES.append(key)
            i += 1
            continue

        # A caption paragraph is held, not emitted: the next table takes it.
        m_cap = re.match(r"^\*\*(TABLE [IVX]+:.*?)\*\*\s*$", line)
        if m_cap:
            PENDING_CAPTION.append(m_cap.group(1))
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and set(
                lines[i + 1].replace("|", "").replace(":", "").strip()) <= {"-", " "}:
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(table(block))
            continue

        if line.startswith("## "):
            title = re.sub(r"^#+\s*(?:[IVX]+\.\s*)?", "", line)
            if title.strip().lower().startswith("abstract"):
                out.append(r"\begin{abstract}")
                out.append("\x01ABSTRACT\x01")
            elif title.strip().lower().startswith("references"):
                in_refs = True
                out.append(r"\begin{thebibliography}{99}")
            else:
                out.append(r"\section{" + inline(title) + "}")
            i += 1
            continue

        if line.startswith("### "):
            title = re.sub(r"^#+\s*(?:[A-Z]\.\s*)?", "", line)
            out.append(r"\subsection{" + inline(title) + "}")
            i += 1
            continue

        if line.startswith("# "):
            i += 1
            continue

        if line.strip() in ("---", "***"):
            i += 1
            continue

        # The references section holds bibliography entries and nothing else.
        # The draft keeps two notes to ourselves there -- which sources are
        # verified, and which are still to check -- and a submission must not
        # carry either.
        if in_refs and line.strip() and not re.match(r"^\[\d+\]", line):
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("["):
                i += 1
            continue

        if in_refs and re.match(r"^\[\d+\]", line):
            num = re.match(r"^\[(\d+)\]", line).group(1)
            body = [line[line.index("]") + 1:].strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("["):
                body.append(lines[i].strip())
                i += 1
            out.append(r"\bibitem{ref" + num + "} " + inline(" ".join(body)))
            continue

        if re.match(r"^\d+\.\s", line) or line.startswith("- "):
            env = "enumerate" if re.match(r"^\d+\.\s", line) else "itemize"
            # Each item is gathered whole before conversion. Converting line by
            # line split every ``**bold phrase**`` that wrapped, and the halves
            # matched nothing, so the asterisks reached the output verbatim.
            items: list[list[str]] = []
            while i < len(lines) and (re.match(r"^\d+\.\s", lines[i])
                                      or lines[i].startswith("- ")
                                      or lines[i].startswith("   ")):
                if lines[i].strip():
                    if lines[i].startswith("   ") and items:
                        items[-1].append(lines[i].strip())
                    else:
                        items.append([re.sub(r"^(\d+\.|-)\s*", "",
                                             lines[i].strip())])
                i += 1
            out.append(r"\begin{" + env + "}")
            for parts in items:
                out.append(r"\item " + inline(" ".join(parts)))
            out.append(r"\end{" + env + "}")
            continue

        if not line.strip():
            out.append("")
            i += 1
            continue

        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(
                ("#", "|", "-", "[", "*")) and not re.match(r"^\d+\.\s", lines[i]):
            para.append(lines[i])
            i += 1
        out.append(inline(" ".join(p.strip() for p in para)))

    if in_refs:
        out.append(r"\end{thebibliography}")
    return "\n".join(out)


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def check_captions(md: str) -> None:
    """Captions must be numbered in document order, and all be referenced.

    LaTeX derives a table's number from its position, so a caption written out
    of order silently renumbers every reference after it. Nothing else in this
    pipeline can see that, and it would surface only in the compiled PDF.
    """
    if PENDING_CAPTION:
        raise SystemExit("table caption with no table following it: "
                         + "; ".join(PENDING_CAPTION))
    want = ROMAN[:len(CAPTION_ORDER)]
    if CAPTION_ORDER != want:
        raise SystemExit("captioned tables appear as %s but LaTeX will number "
                         "them %s" % (CAPTION_ORDER, want))
    body = md.split("## References")[0]
    for n in re.findall(r"\bTable ([IVX]+)\b", body):
        if n not in CAPTION_ORDER:
            raise SystemExit("prose refers to Table %s, which has no caption" % n)
    print("  %d captioned tables, numbered %s, every reference resolves"
          % (len(CAPTION_ORDER), ", ".join(CAPTION_ORDER)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", default="docs/PAPER_RAL.md")
    ap.add_argument("--out", default="docs/paper.tex")
    args = ap.parse_args()

    md = (ROOT / args.md).read_text(encoding="utf-8")
    title = md.splitlines()[0].lstrip("# ").strip()

    # The provenance header and the target line are notes to ourselves.
    md = re.sub(r"^\*\*Target:.*?\n---\n", "", md, flags=re.S | re.M)

    body = convert(md)
    check_captions(md)

    # The abstract is one paragraph in IEEEtran; close it before Section I.
    body = body.replace("\x01ABSTRACT\x01", "")
    first_section = body.index(r"\section{")
    head, tail = body[:first_section], body[first_section:]
    if r"\begin{abstract}" in head:
        head = head.rstrip() + "\n" + r"\end{abstract}" + "\n"

    fig = r"""
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/screen_scatter.pdf}
\caption{A statistic computed on raw data factors, without training anything,
against the compositional penalty each axis was measured to carry by a full
paired sweep. Every axis with a measurable penalty lies right of zero and none
lies left of it, so the statistic separates presence from absence; the scatter
within the right-hand group is what it does not predict. DexYCB is absent
because its sweep could not have measured what it claimed (\S{}V-B).}
\label{fig:screen}
\end{figure}
"""
    tail = tail.replace(r"\section{Results}", fig + "\n" + r"\section{Results}", 1)
    if r"\ref{fig:screen}" not in tail:
        raise SystemExit("the figure is emitted but nothing refers to it; "
                         "write \"Fig. 1\" into the manuscript or drop the float")

    tex = (PREAMBLE % {"title": title}) + head + tail + "\n\\end{document}\n"
    out = ROOT / args.out
    out.write_text(tex, encoding="utf-8")

    words = len(re.findall(r"\w+", md))
    print(f"wrote {out}")
    print(f"  {words:,} words of Markdown -> {len(tex.splitlines()):,} lines of LaTeX")
    print(f"  {tex.count(chr(92) + 'section')} sections, "
          f"{tex.count(chr(92) + 'begin{table}')} tables, "
          f"{tex.count(chr(92) + 'bibitem')} references")
    # A reference listed but never cited, or cited but never listed, is the kind
    # of error that survives every proofread and is caught instantly here.
    defined = set(re.findall(r"\\bibitem\{(ref\d+)\}", tex))
    cited = {k for group in re.findall(r"\\cite\{([^}]+)\}", tex)
             for k in group.split(",")}
    order = lambda s: int(s[3:])
    if defined - cited:
        print(f"  listed but never cited: "
              f"{sorted(defined - cited, key=order)}")
    if cited - defined:
        print(f"  cited but not listed: {sorted(cited - defined, key=order)}")
    if defined == cited:
        print(f"  {len(defined)} references, all cited and all listed")

    leftover = set(re.findall(r"[^\x00-\x7f]", tex))
    if leftover:
        codes = ", ".join(f"U+{ord(c):04X}" for c in sorted(leftover))
        print(f"  non-ASCII left in output: {codes} -- add each to UNICODE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
