# Group F: reference-list and in-text citation FORMAT check

Checked 2026-10-03 against the rendered PDF
`C:\Users\maurice\Desktop\paper彙整\paper_0929\TMLR_手部組合泛化_WANG_MU_HUA_2026-10-02.pdf` (16 pages; text
extracted with pymupdf), `docs/tmlr/refcheck/rendered_reference_list.txt` (38 entries), `docs/tmlr/references.bib`
(48 entries) and `docs/tmlr/PAPER_TMLR.md`, with the renderer `scripts/tmlr_md_to_docx.py` read to attribute each
defect to the .bib data or to the renderer. Content (whether a reference is the right one) was not checked here.

## 0. What passes

- **Numbering by first appearance:** the 38 numbers first appear in the body in the order 1, 2, ..., 38 (page 1:
  [1]; page 2: [2]-[11]; page 3: [10], [12]-[34]; page 4: [35]-[37]; page 6: [38]). The renderer's `cite()` assigns
  numbers in order of first appearance in the Markdown, so this holds by construction.
- **Order inside brackets:** every multi-citation bracket is ascending ([2, 3], [4, 5], [6, 7, 8], [15, 16, 17],
  [23, 24], [36, 37]). `cite()` sorts the numbers, so this also holds by construction.
- **No sentence starts with a bare citation number.** Every bracket follows a word, an author name ("Hupkes et al.
  [11]", "Tse et al. [10]", "Finegan-Dollak et al. [12]", "Atzmon et al. [5]", "Sun et al. [21]", "Engstrom et al.
  [25]") or a closing parenthesis of a dataset name.
- **One space between name and bracket:** all six "Author et al. [n]" cases have exactly one space; no bracket is
  glued to the preceding word anywhere in the body (the two apparent double spaces, "sources  [18]" on page 3 and
  "GRAB  [32]" / "OakInk2  [33]" / "TACO  [9]" / "parameters  [34]" on page 3, are line breaks in the PDF text
  extraction, not typesetting).
- **No "?" or unresolved citation** anywhere in the PDF; no `[@key]` survives; the renderer reports `missing []`.
- **Every listed entry is cited and every cited number is listed:** 38 cited, 38 listed, no gaps.
- **Author truncation is consistent:** the renderer prints the first six authors and "et al." when there are more
  than six (entries 1, 3, 7, 8, 9, 10, 11, 12, 14, 29, 31, 33, 35, 37); entries with six or fewer authors list all.
  This is a single rule applied uniformly (Vancouver style).
- **Venue names are full names, not "Proc." abbreviations**, and the same venue is always spelled the same way
  (CVPR x8 identical; ICLR x4 identical; ICML x2 identical; PNAS x2 identical; ECCV x2 in the same form).
- **The 10 uncited .bib entries** (hupkes2020compositionality, nikolaus2019compositional, montero2021role,
  montero2022lost, schott2022visual, fu2025gigahands, christiansen2013measuring, demsar2006statistical,
  bouthillier2021accounting, dietterich1998approximate) are **absent from the rendered list and from the PDF text**
  (grep for each key's surname: no hits). The renderer only emits keys in `order`, i.e. keys actually cited, so an
  uncited entry can never leak into the list. Nothing else to flag about them.
- **arXiv-only entries ([1], [7], [8]):** Crossref title queries (2026-10-03) return no published version for
  "LAMP: Latent Motion Prior-Guided Real-World Learning for Dexterous Hand Manipulation", "RoboHiMan: A
  Hierarchical Evaluation Paradigm ..." or "Diagnosing Compositional Generalization in Sequential Robot Tasks"
  (nearest hits are unrelated papers). Citing arXiv for these three is currently correct.

## 1. Reference list: per-entry defects

Two defects apply to **every** entry and are renderer-level (see 4b):

- **D1. No DOI, URL or arXiv id is printed for any entry that has a venue.** `fmt_ref` prints only
  authors, title, venue, year, volume:pages. The .bib holds DOIs for 24 of the 38 cited entries and OpenReview URLs
  for the four ICLR papers; all are dropped. Black's checklist asks for references "as complete as possible
  (DOI/arXiv id)".
- **D2. Issue numbers are dropped** (`number` is never read): 36(6), 4(9), 21(3), 33(2), 115(11), 11(62), 183(4),
  110(48), 21(9), 34(1-2), 39(7), 5(10). Consistent, so acceptable as a style, but it makes [34] ambiguous (below).
- **D3. En dashes become hyphens.** The renderer does `val.replace("--", "-")`, so every page range ("2873-2882") and
  every "Computer Vision -- ECCV" and "Encoder--Decoder" renders with a hyphen. Typographically a page range and the
  Springer series title take an en dash (–).

Per entry (rendered line quoted; "ok" means nothing beyond D1-D3):

| # | Rendered (abridged) | Defects |
|---|---|---|
| 1 | `Yang X, Ma Z, Yu H, Chen Y, Yang Y, Chai X, et al. LAMP: Latent Motion Prior-Guided Real-World Learning for Dexterous Hand Manipulation. arXiv:2607.06323, 2026.` | ok; still preprint-only (Crossref). |
| 2 | `Lake B, Baroni M. Generalization without Systematicity: ... Proceedings of the 35th International Conference on Machine Learning (ICML), 2018, 80:2873-2882.` | "80" is the PMLR volume but "PMLR" / "Proceedings of Machine Learning Research" is not printed (`series`/`publisher` dropped), so "80:" reads as an ICML volume. Same in [25]. |
| 3 | `Keysers D, Schärli N, Scales N, Buisman H, Furrer D, Kashubin S, et al. Measuring Compositional Generalization: ... International Conference on Learning Representations (ICLR), 2020.` | ok (ICLR has no pages); OpenReview URL dropped (D1). |
| 4 | `Materzynska J, ... Something-Else: Compositional Action Recognition With Spatial-Temporal Interaction Networks. ... (CVPR), 2020, 1049-1059.` | Title case: "With" capitalised here and in [19], but "with" lower-case in [37] ("with a Constrained") -- inconsistent minor-word capitalisation across the list (see T1 below). |
| 5 | `Atzmon Y, Kreuk F, Shalit U, Chechik G. A Causal View of Compositional Zero-Shot Recognition. Advances in Neural Information Processing Systems, 2020, 33:1462-1473.` | Only proceedings venue printed without its acronym "(NeurIPS)" while CVPR/ICLR/ICML/EMNLP/CoNLL/ICCV carry one (V1). |
| 6 | `Gao J, Xie A, Xiao T, Finn C, Sadigh D. Efficient Data Collection for Robotic Manipulation via Compositional Generalization. Robotics: Science and Systems XX, 2024.` | No pages or paper number and the DOI `10.15607/RSS.2024.XX.013` in the .bib is dropped (D1); with D1 this entry has no locator at all. No "(RSS)" acronym (V1). |
| 7 | `Chen Y, Chen Z, Chan NT, Chen J, Yin J, Shi J, et al. RoboHiMan: ... arXiv:2510.13149, 2025.` | ok; preprint-only (Crossref). |
| 8 | `Wang Y, Wu CE, Sun L, Wang P, Ji X, Liang B, et al. Diagnosing Compositional Generalization in Sequential Robot Tasks. arXiv:2607.29687, 2026.` | ok; preprint-only (Crossref). |
| 9 | `Liu Y, Yang H, Si X, Liu L, Li Z, Zhang Y, et al. TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding. ... (CVPR), 2024, 21740-21751.` | "Tool-ACtion-Object" looks like a stray capital but is the authors' own stylisation (the acronym TACO); the .bib protects it with `{ACtion}`. Intentional; a referee may still read it as a typo. Leave or add a brace-protected note; no change recommended. |
| 10 | `Tse THE, Feng R, Zheng L, Park J, Gao Y, Kim J, et al. Collaborative Learning for 3D Hand-Object Reconstruction ... Proceedings of the AAAI Conference on Artificial Intelligence, 2025, 39:7437-7445.` | "Tse THE" is the correct initials of "Tze Ho Elden" but reads as the word "THE"; acceptable in Vancouver style. Issue 7 dropped (D2). |
| 11 | `Hupkes D, Giulianelli M, Dankers V, Artetxe M, Elazar Y, Pimentel T, et al. A Taxonomy and Review of Generalization Research in NLP. Nature Machine Intelligence, 2023, 5:1161-1174.` | ok. |
| 12 | `Finegan-Dollak C, Kummerfeld JK, Zhang L, Ramanathan K, Sadasivam S, Zhang R, et al. Improving Text-to-SQL Evaluation Methodology. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2018, 351-360.` | No "(ACL)" acronym while sibling ACL-Anthology venues [35] and [21] carry "(EMNLP)" / "(CoNLL)" (V1). |
| 13 | `Bansal A, Sikka K, Sharma G, Chellappa R, Divakaran A. Zero-Shot Object Detection. Computer Vision - ECCV 2018, 2018, 397-414.` | **Duplicated year** ("ECCV 2018, 2018") because the year is part of the Springer booktitle and the renderer appends `year` again. Hyphen for en dash (D3). No LNCS volume (the .bib has `series` but no `volume`), so the pages 397-414 have no volume to attach to. |
| 14 | `Zhang L, Chang X, Liu J, Luo M, Wang S, Ge Z, et al. ZSTAD: Zero-Shot Temporal Activity Detection. ... (CVPR), 2020, 876-885.` | ok. |
| 15 | `Roitberg A, Martinez M, Haurilet M, Stiefelhagen R. Towards a Fair Evaluation of Zero-Shot Action Recognition Using External Data. Computer Vision - ECCV 2018 Workshops, 2019, 97-105.` | Venue says 2018, year says 2019. This is correct (LNCS 11130 appeared in 2019) but reads as an error next to [13]/[32] where the two years agree; recommend adding the LNCS volume so the 2019 is visibly the book's year. Hyphen for en dash (D3). |
| 16 | `Brattoli B, Tighe J, Zhdanov F, Perona P, Chalupka K. Rethinking Zero-Shot Video Classification: End-to-End Training for Realistic Applications. ... (CVPR), 2020, 4613-4623.` | ok. |
| 17 | `Gowda SN, Sevilla-Lara L, Kim K, Keller F, Rohrbach M. A New Split for Evaluating True Zero-Shot Action Recognition. Pattern Recognition (DAGM GCPR 2021), 2021, 191-205.` | Springer-LNCS venue written in a different pattern from [13]/[15]/[32] ("Title (CONF YEAR)" vs "Computer Vision - ECCV YEAR"); no LNCS volume (13024). Pick one pattern for the four LNCS entries. |
| 18 | `Kapoor S, Narayanan A. Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. Patterns, 2023, 4:100804.` | ok (article number as pages is standard for Patterns); issue 9 dropped (D2). |
| 19 | `Punnakkal AR, Chandrasekaran A, Athanasiou N, Quiros-Ramirez A, Black MJ. BABEL: Bodies, Action and Behavior With English Labels. ... (CVPR), 2021, 722-731.` | "With" capitalised (CVF style) vs "with" in [37] (T1). |
| 20 | `Moltisanti D, Wray M, Mayol-Cuevas W, Damen D. Trespassing the Boundaries: Labeling Temporal Bounds for Object Interactions in Egocentric Video. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, 2886-2894.` | ok (2017 ICCV was IEEE-only; the "IEEE" vs "IEEE/CVF" difference against the CVPR entries is historically correct). |
| 21 | `Sun K, Williams A, Hupkes D. The Validity of Evaluation Results: ... Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL), 2023, 274-293.` | ok. |
| 22 | `Lipsitch M, Tchetgen Tchetgen E, Cohen T. Negative Controls: ... Epidemiology, 2010, 21:383-388.` | ok ("Tchetgen Tchetgen E" is the correct surname). |
| 23 | `Schuemie MJ, Ryan PB, DuMouchel W, Suchard MA, Madigan D. Interpreting Observational Studies: Why Empirical Calibration is Needed to Correct p-Values. Statistics in Medicine, 2014, 33:209-218.` | "is" lower-case in an otherwise Title-Case title (T1). "p-Values": the "p" should be italic in the journal's own rendering; plain "p" is acceptable. |
| 24 | `Schuemie MJ, Hripcsak G, Ryan PB, Madigan D, Suchard MA. Empirical Confidence Interval Calibration ... Proceedings of the National Academy of Sciences, 2018, 115:2571-2577.` | ok. |
| 25 | `Engstrom L, Ilyas A, Santurkar S, Tsipras D, Steinhardt J, Madry A. Identifying Statistical Bias in Dataset Replication. Proceedings of the 37th International Conference on Machine Learning (ICML), 2020, 119:2922-2932.` | Same as [2]: "119" is a PMLR volume with no "PMLR" printed. |
| 26 | `Ojala M, Garriga GC. Permutation Tests for Studying Classifier Performance. Journal of Machine Learning Research, 2010, 11:1833-1863.` | ok; JMLR URL dropped (D1). |
| 27 | `Bernardi S, Benna MK, Rigotti M, Munuera J, Fusi S, Salzman CD. The Geometry of Abstraction in the Hippocampus and Prefrontal Cortex. Cell, 2020, 183:954-967.e21.` | ok ("954-967.e21" is Cell's own pagination). |
| 28 | `Petigura EA, Howard AW, Marcy GW. Prevalence of Earth-Size Planets Orbiting Sun-Like Stars. Proceedings of the National Academy of Sciences, 2013, 110:19273-19278.` | ok. |
| 29 | `Jiang L, Schlesinger F, Davis CA, Zhang Y, Li R, Salit M, et al. Synthetic Spike-in Standards for RNA-seq Experiments. Genome Research, 2011, 21:1543-1551.` | ok ("Spike-in", "RNA-seq" follow the journal's title). |
| 30 | `Adebayo J, Muelly M, Abelson H, Kim B. Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation. International Conference on Learning Representations (ICLR), 2022.` | "Post hoc ... may be" is the authors' own capitalisation on OpenReview, but it is the least Title-Case entry in the list (T1). |
| 31 | `Yang L, Li K, Zhan X, Wu F, Xu A, Liu L, et al. OakInk: A Large-Scale Knowledge Repository for Understanding Hand-Object Interaction. ... (CVPR), 2022, 20953-20962.` | ok. |
| 32 | `Taheri O, Ghorbani N, Black MJ, Tzionas D. GRAB: A Dataset of Whole-Body Human Grasping of Objects. Computer Vision - ECCV 2020, 2020, 581-600.` | **Duplicated year** ("ECCV 2020, 2020"), hyphen for en dash, no LNCS volume (12349), same as [13]. |
| 33 | `Zhan X, Yang L, Zhao Y, Mao K, Xu H, Lin Z, et al. OAKINK2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion. ... (CVPR), 2024, 445-456.` | "OAKINK2" is the official CVF title, but the body text writes "OakInk2" throughout (page 3, Table 1, Appendix B). Not wrong; note the mismatch in case a referee asks. |
| 34 | `Romero J, Tzionas D, Black MJ. Embodied Hands: Modeling and Capturing Hands and Bodies Together. ACM Transactions on Graphics, 2017, 36:245:1-245:17.` | **"36:245:1-245:17" is a triple-colon locator** produced by the renderer's `vol:pages` rule on ACM's "245:1--245:17" article pagination; without the issue number (6) it is hard to parse. The .bib `note = {Proc. SIGGRAPH Asia 2017}` is dropped. Render as "36(6):245:1-245:17" or "36(6), Article 245". |
| 35 | `Cho K, van Merriënboer B, Gulcehre C, Bahdanau D, Bougares F, Schwenk H, et al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014, 1724-1734.` | "using" lower-case (T1); "Encoder--Decoder" rendered with a hyphen (D3); the ACL Anthology title uses an en dash. |
| 36 | `Kingma DP, Welling M. Auto-Encoding Variational Bayes. International Conference on Learning Representations (ICLR), 2014.` | ok (published version is ICLR 2014). |
| 37 | `Higgins I, Matthey L, Pal A, Burgess C, Glorot X, Botvinick M, et al. beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. International Conference on Learning Representations (ICLR), 2017.` | The published title is "β-VAE" (Greek beta); "beta-VAE" is a transliteration. The body text also says "beta-VAE", so the two are at least consistent with each other. "with" lower-case (T1). |
| 38 | ``Welch BL. The Generalization of `Student's' Problem when Several Different Population Variances are Involved. Biometrika, 1947, 34:28-35.`` | **Stray character:** a LaTeX left quote (backtick `) survives before "Student's" and a straight apostrophe closes it, because the renderer strips braces but does not convert LaTeX quotes. Should be ‘Student's’ (or "Student's"). "when ... are" lower-case (T1). Issue "1-2" dropped (D2). |

Cross-entry consistency findings referred to above:

- **T1. Title capitalisation is mixed.** Most titles are in Title Case with every word capitalised, including
  minor words ("With" in [4] and [19], "Without" in [2]); others leave minor words lower-case ("using" [35],
  "is" [23], "with" [37], "may be" [30], "when ... are" [38]). Each follows its publisher's own rendering, so the
  .bib is faithful to the sources, but as a list it is inconsistent. Choose one rule (Title Case for all, or
  sentence case for all -- sentence case is the usual TMLR/natbib look) and apply it in the .bib or in the renderer.
- **V1. Venue acronym in parentheses is present for CVPR, ICCV, ICLR, ICML, EMNLP, CoNLL, ECCV and GCPR but absent
  for NeurIPS [5], RSS [6], ACL [12] and AAAI [10]** (AAAI's name contains the acronym). Either add the acronym to the
  four or drop it from all.
- **V2. The four Springer LNCS entries ([13], [15], [17], [32]) are the only ones without a volume**, and the only
  ones whose venue string contains a year, which produces the duplicated year in [13] and [32].
- **Braces in the .bib titles** are present for GRAB, MANO-free entries, OakInk, OAKINK2, TACO/ACtion, RNN, SQL, NLP,
  ZSTAD, BABEL/English, 3D/RGB, VAE, Bayes, Earth/Sun, Student's, RoboHiMan, LAMP. Missing brace protection (would
  matter only if the list were ever produced by BibTeX with a lower-casing style; the Python renderer does not
  lower-case): "Something-Else", "Text-to-SQL" (only SQL is braced), "ECCV"/"CVPR" inside booktitles (BibTeX does not
  lower-case booktitles, so harmless), "RNA-seq" (RNA braced), "p-Values", "Hand-Object", "Zero-Shot", "beta-VAE"
  (only VAE braced), "Post hoc". Nothing to fix for the current renderer; see 4a for a BibTeX-safe set if the paper
  moves to LaTeX.

## 2. In-text citations: violations

None of the rules in the brief is violated in the body text. Every bracket was checked (49 occurrences, pages 1-6);
all are ascending inside the bracket, preceded by one space, and never sentence-initial. Citation numbers are
assigned by first appearance and the first appearances run 1 to 38 without a jump.

Observations that are not violations but worth a glance:

- Page 2, line "Hupkes et al. [11] review the practice." and the five other "Author et al. [n]" sentences: the
  pattern matches Black's "Name [n]" form with one space. Good.
- Page 3, "GRAB [32] (1,048 trajectories ...)", "OakInk2 [33] (609 ...)", "TACO [9] (2,317 ...)": a citation
  immediately followed by a parenthesis is slightly dense; acceptable.
- Page 3, "the spike-in of genomics [29] and, in machine learning, the planted artefact [30]": fine.
- No citation appears in the abstract, in any table, figure caption, or appendix (checked the Markdown: all 50
  `[@...]` tokens lie in Sections 1-3 and 3.5).
- Reference [9] (TACO) is cited six times and [1], [2], [4], [6], [7], [8], [10], [31] more than once; numbering
  stays stable (the renderer looks the key up in `order`).

## 3. The 10 uncited .bib entries

hupkes2020compositionality, nikolaus2019compositional, montero2021role, montero2022lost, schott2022visual,
fu2025gigahands, christiansen2013measuring, demsar2006statistical, bouthillier2021accounting,
dietterich1998approximate: **confirmed absent** from the rendered list (38 entries, keys 1-38 all cited) and from
the PDF text (no occurrence of Nikolaus, Montero, Schott, GigaHands, Christiansen, Demšar, Bouthillier or
Dietterich; the only Hupkes hits are the 2023 taxonomy, entry [11]). The renderer emits only cited keys, so they
cannot appear by accident. Nothing else flagged.

## 4. Concrete fixes

### 4a. `.bib` data changes (exact field and value)

1. `welch1947generalization`: `title = {The Generalization of {`}{Student's}' Problem ...}` currently renders a
   backtick. Change to `title = {The Generalization of {\textquoteleft}{Student's}{\textquoteright} Problem when Several Different Population Variances are Involved}`
   or, simpler for this Python renderer, `title = {The Generalization of ‘{Student's}’ Problem when Several Different Population Variances are Involved}`
   (Unicode quotes). Alternatively leave the .bib and fix the renderer (4b-5).
2. `taheri2020grab`: add `volume = {12349}` (LNCS) and change `booktitle = {Computer Vision -- ECCV 2020}` to
   `booktitle = {Computer Vision -- ECCV 2020, Lecture Notes in Computer Science}` only if the renderer is not
   changed to print `series`; otherwise keep booktitle and rely on 4b-4 to suppress the duplicated year.
3. `bansal2018zero`: add `volume = {11205}` (LNCS; ECCV 2018 Part XIV).
4. `roitberg2018towards`: add `volume = {11130}` (LNCS; ECCV 2018 Workshops Part I). The 2019 book year then
   reads as intended next to "ECCV 2018 Workshops".
5. `gowda2021new`: add `volume = {13024}` (LNCS) and rewrite `booktitle = {Pattern Recognition -- DAGM GCPR 2021}`
   so the four LNCS entries share one pattern ("Series title -- CONF YEAR").
6. `higgins2017beta`: change `title = {beta-{VAE}: ...}` to `title = {{$\beta$-VAE}: Learning Basic Visual Concepts with a Constrained Variational Framework}`
   (or Unicode `{β-VAE}` for the Python renderer) to match the published title. If changed, change the two body
   occurrences of "beta-VAE" (Section 3.2 and the verifier's string if it checks that sentence) the same way, or
   leave both as "beta-VAE" deliberately.
7. `atzmon2020causal`: `booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}`;
   `gao2024efficient`: `booktitle = {Robotics: Science and Systems XX (RSS)}`;
   `finegan2018improving`: `booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL, Volume 1: Long Papers)}`.
   This makes the venue-acronym convention uniform (V1). (Or drop every "(ACRONYM)" instead; either way, one rule.)
8. `gao2024efficient`: add `pages = {013}` (RSS paper number) or leave and rely on 4b-1 printing the DOI, so the
   entry has at least one locator.
9. Title capitalisation (T1), only if the list is to be uniformly Title Case: `cho2014learning` "using" -> "Using";
   `schuemie2014interpreting` "is" -> "Is"; `higgins2017beta` "with" -> "With"; `adebayo2022post` "Post hoc
   Explanations may be" -> "Post Hoc Explanations May Be"; `welch1947generalization` "when ... are" -> "When ...
   Are". If instead sentence case is chosen, no .bib change: do it in the renderer (4b-6) with the braces marking
   what must stay capitalised. The .bib would then also need braces around `{Something-Else}`, `{Text-to-SQL}`,
   `{Hand-Object}`, `{Zero-Shot}`, `{RNA-seq}`, `{p}-Values`, `{Earth}`/`{Sun}` (already), `{Tool-ACtion-Object}`,
   `{Spatial-Temporal}`, `{Encoder--Decoder}`, `{Egocentric}` (no), `{Kepler}` (uncited), i.e. every proper noun
   and acronym, before a lower-casing pass is safe.
10. Optional, for cross-checking only: `zhan2024oakink2` title "OAKINK2" is the official CVF spelling; the body
    uses "OakInk2". No change required, but if the author prefers the body spelling, set `title = {{OakInk2}: ...}`
    and note that it departs from the CVF listing.

### 4b. Renderer changes (`scripts/tmlr_md_to_docx.py`, `fmt_ref`)

1. **Print a persistent identifier.** After the tail, append `doi:<doi>` when `doi` exists, else `url` when it
   exists, else `arXiv:<eprint>` when `eprint` exists and no venue was printed. Today the arXiv id is printed only
   for venue-less entries; DOIs (24 of 38 cited entries) and OpenReview/JMLR URLs are never printed.
2. **Print the issue number.** Render `volume(number):pages` when `number` exists, e.g. "36(6):245:1-245:17",
   "21(3):383-388", "34(1-2):28-35". This also disambiguates [34].
3. **Keep en dashes.** Replace `val.replace("--", "-")` with `val.replace("--", "\u2013")` (and the same in
   `pages`), so page ranges and "Computer Vision – ECCV 2020" / "Encoder–Decoder" render with an en dash.
4. **Do not repeat a year that is already inside the venue string.** If `venue` ends with the four-digit `year`
   (or contains it as a whole word), omit the separate year field; [13] and [32] then read
   "Computer Vision – ECCV 2018, 397–414" and "..., 12349:581–600" once 4a-2/3 add the volume.
5. **Convert LaTeX quotes.** In the field cleaner, map `` ` `` -> ‘, ``` `` ``` -> “, `''` -> ” and a lone `'`
   following a word character inside quotes -> ’, before braces are stripped. Fixes [38] regardless of 4a-1.
6. **One capitalisation rule for titles.** Either (a) leave titles as the .bib gives them and fix the five
   lower-case minor words in the .bib (4a-9), or (b) implement sentence case: lower-case every word after the first
   except those protected by braces in the .bib, applying it before the `re.sub(r"[{}]", "", val)` strip (which
   currently discards the brace information that such a rule needs). (b) requires the extra braces listed in 4a-9.
7. **Print `series` and `publisher` for PMLR / LNCS entries**, or at least `series`, so "80:2873-2882" in [2] and
   "119:2922-2932" in [25] read as "PMLR 80:2873–2882" and the LNCS volumes added in 4a-2..5 read as
   "LNCS 12349:581–600".
8. **Print `note`** when present ("Proc. SIGGRAPH Asia 2017" for [34]), or delete the field from the .bib so the
   data file does not promise what the list does not show.
9. Optional safety: have the renderer refuse to run if any cited key is missing from `entries` (today it only
   prints `missing [...]` after saving) so a `[?]`-equivalent can never reach the PDF unnoticed.
