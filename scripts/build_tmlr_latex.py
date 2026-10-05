"""Convert docs/tmlr/PAPER_TMLR.md into a TMLR-style LaTeX submission and compile it.

TMLR requires "PDF files generated using the TMLR LaTeX stylefile and template" (author guide), so the
Word-built reading copy cannot be submitted. The Markdown stays the single source of truth (every number in it is
checked by scripts/verify_tmlr_draft.py); this script regenerates the .tex from it and compiles with pdflatex +
bibtex (MiKTeX). It is a converter for THIS document: it handles the constructs the manuscript uses and raises on
anything it does not recognise, because silently dropping a table is worse than stopping.

Usage:
    python scripts/build_tmlr_latex.py OUT_DIR [--named] [--style DIR]
        OUT_DIR   build directory (receives main.tex, references.bib, figures/, the style files, main.pdf)
        --named   preprint option with the author block (NOT for submission; TMLR is double-blind)
        --style   directory holding tmlr.sty, tmlr.bst, fancyhdr.sty, math_commands.tex
                  (default: scratchpad tmlr-style-file-main, downloaded 2026-10-05 from github.com/JmlrOrg/tmlr-style-file)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "tmlr"
DEFAULT_STYLE = Path(r"C:\Users\maurice\AppData\Local\Temp\claude\C--Users-maurice-Desktop-hand-IK"
                     r"\e3e1e6df-a744-4919-821f-e8aca52a8fb9\scratchpad\tmlr\tmlr-style-file-main")

NAMED = "--named" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(args[0]) if args else ROOT / "docs" / "tmlr" / "latex"
STYLE = Path(sys.argv[sys.argv.index("--style") + 1]) if "--style" in sys.argv else DEFAULT_STYLE

md = (SRC / "PAPER_TMLR.md").read_text(encoding="utf-8")
md = re.sub(r"<!--.*?-->", "", md, flags=re.S)

# ----------------------------------------------------------------------------------------------------- inline text
SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}


def inline(s: str) -> str:
    """Markdown inline text -> LaTeX. Citations and cross-references are converted here too."""
    # protect citation keys and refs before escaping
    s = re.sub(r"\b([A-Z][\w-]+(?:-[A-Z][\w-]+)?) et al\. \[@(\w+)\]", r"\\citet{\2}", s)
    s = re.sub(r"\[@([^\]]+)\]", lambda m: "\\citep{" + ",".join(k.strip().lstrip("@") for k in m.group(1).split(";")) + "}", s)
    s = re.sub(r"\bTable (A?\d+)\b", r"Table~\\ref{tab:\1}", s)
    s = re.sub(r"\bFigure (\d+)\b", r"Figure~\\ref{fig:\1}", s)
    s = re.sub(r"\bEquation \(1\)", r"Equation~\\ref{eq:penalty}", s)
    # escape LaTeX specials outside the macros just inserted
    parts = re.split(r"(\\(?:citet|citep|ref)\{[^}]*\}|~)", s)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(p)
            continue
        p = p.replace("\\", r"\textbackslash{}")
        for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
            p = p.replace(a, b)
        p = p.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
        out.append(p)
    s = "".join(out)
    # bold / italic
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\\emph{\1}", s)
    # quotes
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    # unicode
    s = re.sub(r"10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "$10^{-" + "".join(SUP[c] for c in m.group(1)) + "}$", s)
    s = re.sub(r"([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "$^{" + "".join(SUP[c] for c in m.group(1)) + "}$", s)
    s = s.replace("Eₙ", "$E_n$").replace("Eᵢ", "$E_i$")
    for a, b in (("×", r"$\times$"), ("π", r"$\pi$"), ("°", r"$^\circ$"), ("‡", r"$\ddagger$"), ("−", r"$-$"),
                 ("–", "--"), ("—", "---"), ("‘", "`"), ("’", "'"), ("“", "``"), ("”", "''"), ("≥", r"$\geq$"),
                 ("≤", r"$\leq$"), ("±", r"$\pm$"), ("ë", r"\"e"), ("ä", r"\"a"), ("é", r"\'e"), ("β", r"$\beta$")):
        s = s.replace(a, b)
    # non-breaking spaces in "p = 0.03", "95 % CI", "40 seeds" are left to LaTeX's own line breaking
    bad = sorted({c for c in s if ord(c) > 127})
    if bad:
        raise SystemExit(f"unhandled non-ASCII {bad!r} in: {s[:80]}")
    return s


# -------------------------------------------------------------------------------------------------------- tables
COLSPEC = {  # caption number -> column spec; default is auto (text-ish first column left, rest centred)
    "1": r"p{0.24\textwidth}rrclcc",
    "2": r"lp{0.15\textwidth}rrrrrrrrrr",
    "3": r"p{0.40\textwidth}rrcc",
    "4": r"p{0.22\textwidth}ccc",
    "A1": r"p{0.10\textwidth}p{0.24\textwidth}p{0.18\textwidth}p{0.30\textwidth}",
    "D": r"p{0.40\textwidth}cc p{0.32\textwidth}",
}
SIZE = {"1": r"\scriptsize", "2": r"\scriptsize", "3": r"\small", "4": r"\small", "A1": r"\footnotesize", "D": r"\footnotesize"}


def table(rows: list[list[str]], num: str) -> str:
    ncol = len(rows[0])
    spec = COLSPEC.get(num) or ("l" + "c" * (ncol - 1))
    body = []
    # Header cells of two or more words stack (makecell) so a wide header does not widen a narrow number column.
    def hcell(c: str, col: int) -> str:
        words = c.split()
        if len(words) >= 2 and num in ("1", "2") and not (num == "2" and col == 1) and c != "95 % CI":
            k = next((j for j, w in enumerate(words) if w.startswith("(")), None)  # "Penalty (%)" -> Penalty / (%)
            k = k if k else (len(words) + 1) // 2                                   # else two halves
            lines = (" ".join(words[:k]), " ".join(words[k:]))
            return r"\shortstack{" + r" \\ ".join(r"\textbf{" + inline(w) + "}" for w in lines) + "}"
        return r"\textbf{" + inline(c) + "}"
    head = " & ".join(hcell(c, k) for k, c in enumerate(rows[0])) + r" \\"
    for r in rows[1:]:
        r = r + [""] * (ncol - len(r))
        body.append(" & ".join(inline(c) for c in r[:ncol]) + r" \\")
    sep = "2pt" if num == "2" else "3pt"
    if num == "D":  # the sweep list is long and unnumbered: a longtable may break across pages, a tabular may not
        return "\n".join(["{" + SIZE["D"], rf"\setlength{{\tabcolsep}}{{{sep}}}", r"\begin{longtable}[c]{" + spec + "}",
                          r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot", *body,
                          r"\end{longtable}}"])
    tab = "\n".join([r"\begin{tabular}{" + spec + "}", r"\toprule", head, r"\midrule", *body, r"\bottomrule",
                     r"\end{tabular}"])
    return "\n".join([r"\begin{center}", SIZE.get(num, r"\small"), rf"\setlength{{\tabcolsep}}{{{sep}}}", tab, r"\end{center}"])


# ------------------------------------------------------------------------------------------------------- blocks
blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", md) if b.strip()]

title = None
sections: list[tuple[str, str, list[str]]] = []  # (level, heading, latex lines)
cur: list[str] | None = None
pending_fig: tuple[str, str] | None = None
pending_tab: tuple[str, str] | None = None
i = 0


def para(text: str) -> str:
    return inline(" ".join(text.split()))


while i < len(blocks):
    b = blocks[i]
    i += 1
    if b.startswith("# "):
        title = b[2:].strip()
        continue
    if b.startswith("## ") or b.startswith("### "):
        level = "sub" if b.startswith("### ") else ""
        h = b.lstrip("#").strip()
        cur = []
        sections.append((level, h, cur))
        continue
    assert cur is not None, b[:60]
    m = re.match(r"^!\[Figure (\d+)\]\((figures/[^)]+)\.png\)$", b)
    if m:
        pending_fig = (m.group(1), m.group(2))
        continue
    if pending_fig:
        n, path = pending_fig
        pending_fig = None
        cap = " ".join(b.split())
        assert cap.startswith(f"*Figure {n}.") and cap.endswith("*"), cap[:50]
        cap = cap[len(f"*Figure {n}."):-1].strip()
        width = r"\textwidth" if n in ("1", "2") else r"0.62\textwidth"
        place = "!h" if n == "1" else "t"  # the teaser stays under the Introduction heading, not above the title
        cur += [rf"\begin{{figure}}[{place}]", r"\centering", rf"\includegraphics[width={width}]{{{path}.pdf}}",
                r"\caption{" + inline(cap) + "}", rf"\label{{fig:{n}}}", r"\end{figure}", ""]
        continue
    m = re.match(r"^Table (A?\d+)\.\s*(.*)$", b, re.S)
    if m:
        pending_tab = (m.group(1), " ".join(m.group(2).split()))
        continue
    if b.startswith("|"):
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in b.splitlines()
                if not re.match(r"^\|[-| ]+\|$", ln)]
        if pending_tab:
            n, cap = pending_tab
            pending_tab = None
            place = "!h" if n == "A1" else "t"  # keep the appendix table inside its appendix, not above the last reference
            cur += [rf"\begin{{table}}[{place}]", r"\caption{" + inline(cap) + "}", rf"\label{{tab:{n}}}", table(rows, n),
                    r"\end{table}", ""]
        else:
            cur += [table(rows, "D"), ""]
        continue
    if re.match(r"^[A-Z]\w* = .*\(\d+\)$", b):
        assert b.startswith("P = 100 (Eₙ − Eᵢ) / Eₙ"), b
        cur += [r"\begin{equation}", r"P = 100\,(E_n - E_i)/E_n,", r"\label{eq:penalty}", r"\end{equation}", ""]
        continue
    if re.match(r"^\d+\. ", b):
        items = re.split(r"\n(?=\d+\. )", b)
        cur += [r"\begin{enumerate}"] + [r"\item " + para(re.sub(r"^\d+\. ", "", it)) for it in items] + [r"\end{enumerate}", ""]
        continue
    cur += [para(b), ""]

assert title and not pending_fig and not pending_tab

# ------------------------------------------------------------------------------------------------- assemble .tex
by_name = {h: (lvl, body) for lvl, h, body in sections}
order_main = [h for lvl, h, _ in sections if h not in ("Abstract", "References", "Broader impact statement",
                                                        "Declaration of AI assistance") and not h.startswith("Appendix")]
appendices = [h for lvl, h, _ in sections if h.startswith("Appendix")]

tex = [r"\documentclass[10pt]{article}",
       r"\usepackage[preprint]{tmlr}" if NAMED else r"\usepackage{tmlr}",
       r"\input{math_commands.tex}", r"\usepackage{hyperref}", r"\usepackage{url}", r"\usepackage{graphicx}",
       r"\usepackage{booktabs}", r"\usepackage{longtable}", r"\usepackage{amsmath}", r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}",
       r"\usepackage{microtype}",
       r"\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue}", "",
       r"\title{" + inline(title) + "}", ""]
if NAMED:
    tex += [r"\author{\name Mu-Hua Wang \email may.be13@nycu.edu.tw \\",
            r"      \addr Department of Biomedical Engineering\\ National Yang Ming Chiao Tung University}"]
else:
    tex += [r"\author{\name Anonymous authors \email anonymous@example.org \\ \addr Paper under double-blind review}"]
tex += [r"\def\month{MM}", r"\def\year{YYYY}", r"\def\openreview{\url{https://openreview.net/forum?id=XXXX}}", "",
        r"\begin{document}", r"\maketitle", "", r"\begin{abstract}"]
tex += by_name["Abstract"][1] + [r"\end{abstract}", ""]
for h in order_main:
    lvl, body = by_name[h]
    name = re.sub(r"^\d+(\.\d+)?\.?\s+", "", h)
    tex += [("\\subsection{" if lvl else "\\section{") + inline(name) + "}", ""] + body
for h in ("Broader impact statement", "Declaration of AI assistance"):
    tex += [r"\subsubsection*{" + h[0].upper() + h[1:] + "}", ""] + by_name[h][1]
tex += [r"\bibliographystyle{tmlr}", r"\bibliography{references}", "", r"\appendix",
        r"\renewcommand{\thetable}{A\arabic{table}}", r"\setcounter{table}{0}", ""]
for h in appendices:
    name = re.sub(r"^Appendix [A-Z]\.\s*", "", h)
    tex += [r"\section{" + inline(name) + "}", ""] + by_name[h][1]
tex += [r"\end{document}", ""]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "main.tex").write_text("\n".join(tex), encoding="utf-8")
for f in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty", "math_commands.tex"):
    shutil.copy2(STYLE / f, OUT / f)
bib = (SRC / "references.bib").read_text(encoding="utf-8")
bib = bib.replace("β", r"$\beta$").replace("‘", "`").replace("’", "'")  # pdflatex cannot take these from bibtex output
(OUT / "references.bib").write_text(bib, encoding="utf-8")
(OUT / "figures").mkdir(exist_ok=True)
for f in (SRC / "figures").glob("*.pdf"):
    shutil.copy2(f, OUT / "figures" / f.name)


def run(cmd):
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True, errors="replace")
    return r.returncode, r.stdout + r.stderr


for f in ("main.pdf", "main.aux", "main.bbl", "main.blg", "main.out", "main.log"):  # a stale .bbl re-raises old errors
    (OUT / f).unlink(missing_ok=True)
log = ""
for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
            ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
    rc, out = run(cmd)
    log += f"\n===== {' '.join(cmd)} (exit {rc})\n" + out
(OUT / "build.log").write_text(log, encoding="utf-8")
last = log.split("===== pdflatex")[-1]  # only the final pass matters for undefined references and overfull boxes
errors = re.findall(r"^! .*", log, re.M)
warn = re.findall(r"LaTeX Warning: (Citation|Reference) .* undefined", last)
over = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)", last)
print("pdf:", (OUT / "main.pdf").exists(), "| errors:", len(errors), "| undefined refs/cites:", len(warn),
      "| overfull boxes:", len(over), "max", max(map(float, over), default=0), "pt")
for e in errors[:5]:
    print("  ", e)
