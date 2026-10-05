# Layout audit of the TMLR submission PDF (anonymous, 17 pages)

Date: 2026-10-05. File audited: `scratchpad/tmlr_anon.pdf` (page images at 110 dpi, zoom crops at
200-400 dpi under `scratchpad/final_pages/z_*.png`). Every page was viewed; the text layer was also
read with pymupdf for margins, fonts and non-ASCII glyphs.

## Global checks (all pages)

| Check | Result |
|---|---|
| Page size | US Letter, 612 x 792 pt, all 17 pages |
| Text block | x 72 to about 543 pt on every page (1 in side margins); three lines reach 545.5 pt (p4, p5, p9), which is trailing-space slack of a justified line, not a visible overrun |
| Page numbers | present, centred at the foot, 1 to 17, sequential; none inside a table |
| Fonts | Times New Roman regular / bold / italic / bold-italic only, all embedded as subsets; no fallback font |
| Tofu / replacement characters | none (0 x U+FFFD). Rendered correctly: superscript exponents (10⁻⁵ ... 10⁻¹²), ×, −, –, ‡ (3), β, π, °, ², ³, ⁴, ‘ ’, ä, ë, subscript ₙ ᵢ |
| Equation (1) | p5: `P = 100 (Eₙ − Eᵢ) / Eₙ, (1)`; subscripts render, centred, numbered, true minus sign |
| Title block | p1: title, "Anonymous authors", italic "Paper under double-blind review", "Abstract" heading centred |
| Section numbering | 1 Introduction, 2 Related work, 3 Method (3.1-3.8), 4 Results (4.1-4.4), 5 Discussion, 6 Conclusion; unnumbered Broader impact statement, References, Appendix A-D, Declaration of AI assistance. Consistent |
| Figures referenced before they appear | Fig. 1 appears on p1 and is first cited on p2 (teaser, intended); Fig. 2 cited on p7, appears p8; Fig. 3 cited on p10 above itself. Table 1 cited p6, appears p7; Table 2 cited p6, appears p8; Table A1 cited p3 |
| Tables split across pages | none |
| Figure captions | all below the figure, same page. Table captions all above the table, same page |
| Figure resolution | Fig. 1 1440 px over 6.4 in (225 dpi); Fig. 2 1279 px over 6.4 in (200 dpi); Fig. 3 759 px over 3.8 in (200 dpi). All readable at 200 dpi zoom; Fig. 2 is the softest (small tick labels) but legible |

## Per-page findings

- **p1** — Clean. Title block, abstract, Figure 1 and caption, first paragraph of Section 1. Figure fully inside the text block (75.6-536.4 pt), caption below.
- **p2** — Clean. Ends with "2. Related work" heading followed by six body lines (not orphaned).
- **p3** — Clean. "3. Method" and "3.1" headings each followed by body text.
- **p4** — Clean.
- **p5** — Clean. Equation (1) correct. One line of slack at 545.5 pt is invisible.
- **p6** — Clean. Ends with "4.1 Calibration" heading plus three lines of body (acceptable).
- **p7** — **DEFECT (layout, major).** Page is only 60 % used: text ends at y = 475 pt, leaving about 245 pt (3.4 in) blank below the paragraph "The floor of the measurement ...". Figure 2 was pushed whole to p8 because it needs about 255 pt (198 pt image + 3-line caption + spacing), 10 pt more than the gap. Also: Table 1's **Condition** column is narrow, so every multi-word condition wraps onto two lines ("OakInk-Image category × / intent, permuted", "GRAB shape × fine intent, / permuted", "TACO action × tool, / permuted cells"); readable but untidy. Rules: top, header and bottom present. **Minor:** the horizontal rules carry 1.4 pt vertical stubs at every column boundary (cell-border end caps from the Word export), visible as tiny ticks at 400 dpi (`z_p07_tab1_rule.png`); same on Tables 2, 3, 4, A1, D. Negative numbers in the table use hyphen-minus ("-11.53", "-8.62") while Equation (1) uses a true minus.
- **p8** — Figure 2 at the top of the page (no text above it; acceptable), caption below, legend and axis labels readable. **DEFECT (Table 2, moderate):** the **95 % CI** column is too narrow, so every interval wraps onto two lines ("+19.2 to / +25.2", "+13.2 to / +17.2", ... "+9.0 to / +15.4"), which doubles the row height for 8 of 10 rows. The **Axis** column also wraps: "functional class × / intent ‡ W" and "annotated transitions ‡ / W", leaving a lone "W" on its own line. "-0.0 to +4.6" shows a signed zero. Numbers in the other columns are on one line. Rules present.
- **p9** — Table 3 and Table 4 both with caption above, rules present, no wrapping. Table 3 has the minor border-stub artefact (`z_p09_tab4_rule.png` shows the ticks on Table 4 too). Otherwise clean.
- **p10** — Figure 3 centred, inside the block (169-443 pt), caption below, legend readable. "4.4" heading followed by body. Clean.
- **p11** — Discussion ends at y = 661 pt, about 60 pt (0.8 in) blank at the foot because "6. Conclusion" starts p12 (keep-with-next). Acceptable; noted only.
- **p12** — Conclusion, Broader impact statement, References [1]-[12]. Hanging indents consistent. [11]'s DOI breaks at an existing hyphen: "doi:10.1038/s42256-023-/00729-y" (no character lost). Clean otherwise.
- **p13** — References [13]-[30]. DOIs broken at existing hyphens in [15] ("978-3-/030-11018-5_8") and [17] ("978-3-030-92659-/5_12"); no character lost, but a reader cannot tell whether the line-end hyphen belongs to the DOI. About 0.6 in blank at the foot (next entry did not fit). Clean otherwise.
- **p14** — References [31]-[39] (β in "β-VAE", "van Merriënboer", "Schärli" all render), then Appendix A heading and Table A1 (caption above, rules present, no bad wraps; cells are centred multi-line text, readable). Clean.
- **p15** — Appendix A text, Appendix B (five run-in paragraphs), "Appendix C" heading followed by two lines (acceptable). Clean.
- **p16** — Appendix C text, Appendix D heading, intro paragraph and the sweep table (caption-less, but it is introduced by the paragraph; rules present). Row "GRAB shape × fine intent with contact channels, two inputs, stride / 32" wraps with "32" alone on a line (minor). Table ends at y = 683 pt, leaving about 37 pt free.
- **p17** — **DEFECT (orphan, moderate).** The last page holds only the heading "Declaration of AI assistance" and five lines of text (ends at y = 163 pt): about 12 % of the page used, 88 % blank. The section needs roughly 85 pt, 48 pt more than p16 has left.

Pages with no defect at all: 1, 2, 3, 4, 5, 6, 10, 14, 15.

## Required fixes (page, what to change)

1. **p7 / p8 — Figure 2 gap.** Reduce Figure 2's height by about 5 % (image 198 pt → about 185 pt, or set its width to 6.0 in instead of 6.4 in) or trim its caption by one line, so that the figure fits in the 245 pt left on p7 directly after the paragraph that cites it. This removes the 3.4 in blank on p7 and lets Section 4.2 and Table 2 move up on p8.
2. **p8 — Table 2 column widths.** Widen the **95 % CI** column so each interval sits on one line (take width from Windows, Held cells, n and SD, which are over-wide for their contents), and widen **Axis** so "functional class × intent ‡ W" and "annotated transitions ‡ W" do not wrap, or move the W / ‡ markers to a separate narrow "Pre-reg." column. Change "-0.0 to +4.6" to "0.0 to +4.6".
3. **p16 / p17 — orphan Declaration.** Bring "Declaration of AI assistance" onto p16: either (a) move it to follow the Broader impact statement on p12 (its usual place in TMLR papers; references then reflow and the slack on p13 and p16 absorbs the shift), or (b) reduce the Appendix D table's cell padding / row spacing by about 50 pt and keep the heading with its paragraph. Either way the PDF should end on page 16 with no page holding a single short section.
4. **p7 — Table 1 Condition column.** Widen Condition (shrink SD, Seeds positive, p) so rows are single-line; cosmetic but it halves the table height and helps fix 1.
5. **All tables (p7, p8, p9, p14, p16) — border stubs.** In the docx generator set the inside vertical cell borders to none (not merely white/thin) so the horizontal rules no longer show 1.4 pt tick marks at every column boundary. Cosmetic.
6. **References p12 [11], p13 [15], [17] — DOIs broken at a hyphen.** No character is lost, so this is optional: either allow the URL/DOI runs to break only after "/" (set no-hyphenation on those runs and insert zero-width spaces after slashes) or accept as is.
7. **Optional consistency.** Use the true minus sign (U+2212) for negative values in Tables 1, 2 and D and in the running text ("-3.76 %", "-1.21"), matching Equation (1). Not a defect in rendering.

Not required: Figure resolutions (200-225 dpi) are acceptable for a PDF submission; the 0.8 in gap at the foot of p11 and the figure-only top of p8 are normal pagination.
