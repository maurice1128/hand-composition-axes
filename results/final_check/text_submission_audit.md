# Final text and submission-checklist audit — TMLR manuscript

Date: 2026-10-05. Scope: `docs/tmlr/PAPER_TMLR.md` (source) and the anonymous Word-built PDF
`scratchpad/tmlr_anon.pdf` (17 pages; text in `scratchpad/final_pages/anon_text.txt`). Nothing was edited except
this report. Numbers were not re-verified here (that is `scripts/verify_tmlr_draft.py`'s job, 129 checks / 0 failures);
this audit covers language, cross-references, terminology, scope of claims, the TMLR checklist and the abstract.
TMLR requirements were read from `jmlr.org/tmlr/author-guide.html`, `editorial-policies.html`,
`reviewer-guide.html`, `acceptance-criteria.html` and `ethics.html` on 2026-10-05.

Severity key: **MUST** (fix before upload), **SHOULD** (fix if the author agrees; none changes a number), *note*.

---

## A. Language

Overall: British spelling is consistent (generalisation 4, labelled 7, normalised 2, centring, artefact, grey,
modelled); the only American forms are inside quoted reference titles ("Generalization", "Labeling", "Behavior",
"Modeling"), which is correct. No doubled words ("Tchetgen Tchetgen" is a surname). No contractions. "we" is used
sparingly and consistently (We test, We adopt, We use, we do not propose, we wrote down, we measure). No
"showcase / delve / allows to / leverage / novel / state-of-the-art / significantly". Capitalisation of OakInk-Image,
OakInk2, GRAB, TACO, Welch, Bonferroni, Figure/Table/Section/Appendix is consistent. Thousands separators
("4,591", "2,283,059", "1,048") and "× 10⁻⁷", "p = ", "% " spacing, "s", "fps" are consistent.

### A1. Tense (SHOULD)
Main text is present tense throughout; Appendix B drops into the past and mixes tenses inside one paragraph.
- "leave the penalty in place. The penalty was +13.70 % and +19.64 % respectively" (App. B, Retargeting) →
  "leave the penalty in place: +13.70 % and +19.64 % respectively, against +15.19 % unmodified".
- "gave +2.22 %, not different from the control" (App. B, Stride) → "gives".
- "returned +2.30 % ... returned +8.93 % ... did not differ ... was not above them" (App. B, Budget) → "returns ...
  returns ... does not differ ... is not above them".
Appendix C is a history and may stay in the past.

### A2. Minus sign (SHOULD)
Equation (1) uses a true minus ("Eₙ − Eᵢ"); everywhere else a hyphen-minus is used: "10t³ - 15t⁴ + 6t⁵",
"(1 - w)a + wb", "[-1, 1]", "-3.39", "-0.1 to +2.8", "-11.53 (-14.45 to -8.62)". Use "−" throughout (LaTeX math mode
does this automatically for the equations; tables need `$-$` or `\textminus`).

### A3. Line breaks inside "p = " in the PDF (SHOULD; disappears in LaTeX if `p~=~` is used)
- App. B: "(p \n= 0.39 and p = 0.0025)" and "(r = +0.25, p \n= 0.12)". Use a non-breaking space in "p = " and "r = ".

### A4. "-0.0" (SHOULD)
Table 2, OakInk2 scene × primitive, 95 % CI: "-0.0 to +4.6" → "0.0 to +4.6".

### A5. "beta" as a word (SHOULD)
3.2: "whose KL weight beta [37] rises linearly to 1.0" → "whose KL weight β [37]"; the cited title is "β-VAE".
"KL" is never expanded (Kullback–Leibler); "RMS" (3.5) is never expanded. Expand each once.

### A6. Awkward or ambiguous sentences (SHOULD)
- Abstract: "calibrated with a synthetic dataset whose true penalty is zero (+2.6 % of the naive error)" reads as
  "zero = +2.6 %". → "whose true penalty is zero and which reads +2.6 % of the naive error".
- Intro: "three kinds of control tell how large a gain must be to mean anything" → "show how large".
- 3.1: "On OakInk-Image hand-overs, GRAB's passes and OakInk2 primitives performed by the left hand, a label can
  describe the other hand" → "In OakInk-Image hand-overs, in GRAB's passes and in OakInk2 primitives performed by
  the left hand, a label can describe the other hand".
- 3.4: "must reach more than 0.8 of the held-out cells" → "more than 80 % of the held-out cells".
- 3.5: "permuting one factor leaves at most 3 usable cells against 17" → "against the real grid's 17".
- 4.4: "(the paper reports 131 triplets over 20 object categories)" → "(the TACO paper [9] reports ...)".
- App. A, Table A1, OakInk2: "the frame of each image frame is kept" → "the motion-capture frame coinciding with
  each image frame is kept".
- App. A: "Degrees of freedom pinned at a limit or never moving number 1 of 27 on TACO" ("number" as a verb) →
  "One of 27 degrees of freedom is pinned at a limit or never moves on TACO, and between 1 and 6 on the other
  datasets".
- App. B: "The informed arm is most exposed on TACO (0.46) and GRAB (0.30 and 0.31), which fall on opposite sides"
  → "... which fall on opposite sides of the null".
- Table A1: "about 2.5K sequences" → "about 2,500 sequences" (number style used elsewhere).

### A7. Hyphenation (note, no action)
All variants found are the correct adjective/noun pairs: "held-out combination" vs "held-out-combination split",
"first-clip windows" vs "first clip", "motion-capture frame" vs "motion capture", "per-frame RMS" vs "per frame",
"naive-arm figures" vs "naive arm", "stride-4 value" vs "stride 4", "category-by-intent grid" vs "category by intent".
Tables use "×" ("category × intent") and prose uses "by" ("category by intent"); consistent within each medium.

### A8. Table layout in the Word PDF (note; re-set by the LaTeX template)
Table 2's "95 % CI" column wraps every cell ("+19.2 to / +25.2") and the "‡ W" markers wrap onto their own line
("annotated transitions ‡ / W"). Table 1 repeats each permuted row twice (once per comparison); consider one row with
two comparison columns.

---

## B. Cross-references

Every referenced object exists and points at the right content:

| Reference | Where | Target content | OK |
|---|---|---|---|
| Figure 1 | Intro ×2, caption | teaser, (a)(b)(c) match the image | yes |
| Figure 2 | 4.1 ×2, 4.4 | per-seed real vs permuted grids; blue/grey match the image | yes |
| Figure 3 | 4.3, caption | window classes, red = misaligned matches image | yes |
| Table 1 | 4.1 ×2, 4.4 | calibration rows | yes |
| Table 2 | 3.8 ×2, 4.2 ×3, App. B, App. C ×2, App. D | penalty by axis | yes |
| Table 3 | 4.3 ×2, Table 4 caption, App. C | budget-64 versions | yes (see B2) |
| Table 4 | 4.3, caption, Fig. 3 caption | window classes | yes |
| Table A1 | 3.1, App. A | published vs used counts | yes |
| Sections 3.2, 3.4, 3.6, 3.8, 4.1, 4.2, 4.3; "Sections 4.1, 4.3 and 4.4" | various | all exist and match | yes |
| Appendix A (3.1 ×2), B (4.2 ×2, 5 ×3, App. C, App. D ×2), C (3.8, App. D), D (Intro) | | all exist and match | yes (see B1) |
| Equation (1) | 3.3 | defined, never cited by number | fine |

All three figures and all five tables are referenced in the body.

### B1. "Appendix D lists every sweep" (MUST — also a contradiction, see D1)
Intro: "Appendix D lists every sweep run under the protocol". Appendix D's own lead: "Sweeps on real data not
reported in the main text ... Sweeps of a second prior architecture ... and development runs on synthetic data are
listed in the supplement." So Appendix D lists neither the main-text sweeps nor the synthetic or second-architecture
ones. Fix the Intro: "Appendix D lists every sweep on real data that is not reported in the main text, and the
supplement lists the rest; the per-seed results ..." — or retitle Appendix D "Sweeps not reported in the main text".

### B2. "Table 3" cited for a number it does not hold (SHOULD)
4.3: "It lies at most 2.9 points above the control (difference +0.9, 95 % CI -1.1 to 2.9; Table 3)". Table 3 gives
the penalty, SD, seeds positive and p, not the difference or its CI. Either drop "; Table 3" or add the
difference-from-control column to Table 3.

### B3. Numbering order (note)
Tables are numbered by position, not by first mention: Table A1 (3.1) and Table 2 (3.8) are cited before Table 1
(4.1). Figures 1, 2, 3 are in order of first mention. Journal practice accepts position numbering; no change needed.

---

## C. Terminology

### C1. "zero-truth control" vs "synthetic control" (SHOULD)
Section 4.1 switches to "synthetic control" four times ("the permuted grid matches the synthetic control", "reads
below the synthetic control", "The synthetic control is therefore one reference", "the declared reference ... is the
synthetic control") while 3.5, 3.7, 3.8, Tables 1–3 and the Discussion say "zero-truth control". Because Table 1 also
has a "Planted synthetic" row, "synthetic control" is ambiguous at exactly the point where the two synthetic rows
are being compared. Use "zero-truth control" in all four places; keep "synthetic" only for the dataset description
in 3.5 and the "Planted synthetic" row.

### C2. Undefined-before-use terms (SHOULD)
- "axis": used in 3.1 ("for one axis the capture subject joins the fine label") and 3.3 ("OakInk2's transitions
  axis") before any definition. Add to 3.3: "An axis is one two-factor grid of a dataset."
- "stride": 3.2 says "windows cut every 4 frames"; the word "stride" first appears in 3.5 ("window stride of 32")
  and Table 1. Add "(the stride)" in 3.2.
- "exposure": defined only in Table 2's caption, but used in 4.1 ("its informed arm is the more exposed ... 0.38
  against 0.30"). Define it once in 3.3.
- "cell / combination / composition": 3.3 defines a composition as a cell; "held-out combination", "held-out
  cell" and "held-out composition" are then used interchangeably. One sentence in 3.3 stating the equivalence
  would remove any doubt.

### C3. "segment" carries three senses (note)
Minimum-jerk pose-to-pose segment (3.5), GRAB "contact segment" (5, App. B), and OakInk2 "primitive segments"
(App. D). The main text avoids "segment" for OakInk2 ("recordings cut at their annotated primitive boundaries"),
which is good; consider "primitive clips" in App. D for consistency with that phrasing.

### C4. "shape-by-intent" vs the two GRAB axes (SHOULD)
3.5: "an offset is drawn for each shape-by-intent cell ... 53 of 70 cells"; 4.2: "its held-out shape-by-intent cells
show no penalty". Table 2 and App. A name the axes "shape × fine intent" and "shape × intent class". Say which grid
the planted offset was drawn on ("shape-by-fine-intent", 70 cells) and in 4.2 write "its held-out shape-by-intent
cells, on both intent groupings, show no penalty".

### C5. Consistent (no action)
aligned/misaligned (3.6, Table 3, Table 4, Fig. 3, 4.3, 5); naive arm / informed arm; penalty (Eq. 1); budget
(trajectories per arm, 3.3); trajectory / recording (OakInk2) / clip (OakInk-Image source) / window (32 frames);
fine label (defined per dataset in 3.1); "W" and "‡" (Table 2 caption); "Welch tests" (3.8) and "Welch's test"
(table captions) are both acceptable.

---

## D. Claims versus scope, and contradictions

### D1. Appendix D does not list every sweep (MUST) — see B1.

### D2. The Conclusion drops the one-dataset scope (MUST)
Conclusion: "A held-out combination therefore tests a motion model only where each label covers the whole
trajectory". This is the "only when ... whole trajectory" claim that all five council-2 reviewers said overclaims one
manipulation on one dataset, and that the title change removed. The Discussion is correctly scoped ("necessary for
the penalty on OakInk-Image, but it is not sufficient"); the Conclusion is not. Proposed: "On OakInk-Image, a
held-out combination therefore tests a motion model only where each label covers the whole trajectory; more
generally, whether a model appears to generalise ...".

### D3. "written down before training" is asserted for sweeps that have no pre-registration (MUST)
3.8: "The settings and readings of every sweep in Sections 4.1, 4.3 and 4.4 ... were written down before training".
Section 4.1 reports the zero-truth synthetic (n = 20, budgets 256 and 64) and planted synthetic (n = 18) sweeps.
No file in `runs/PREREG_*.md` has either as its subject (they appear only as comparison references inside other
pre-registrations; `PREREG_sham_grid.md` covers the permuted grids, `PREREG_grab_diagnostics.md` the GRAB plant,
`PREREG_taco_prediction.md` addendum 3 the TACO permuted cells), and Appendix D itself calls them "development
runs on synthetic data". Narrow the sentence: "of the permuted-grid and planted-GRAB sweeps of Section 4.1, of every
sweep in Sections 4.3 and 4.4, and of the rows of Table 2 marked W". Otherwise the "W" column of Table 2 and this
sentence disagree about what pre-registration covers.

### D4. TACO's labels "describe" vs "name" one action (SHOULD)
Abstract and Discussion: "whose labels each name one action / one tool-use action". Conclusion: "on TACO, whose
labels each describe one action". The paper's whole point is that naming is not describing, and 3.1 only says the
right hand "usually" holds the tool. Use "name" in the Conclusion.

### D5. Primitive-bank priors attributed an assumption without a citation (SHOULD)
Intro: "Priors with a continuous latent space [1] and priors built from a bank of primitives are both used on the
assumption that what is learned separately recombines". The bank-of-primitives clause has no citation. Either cite
one or write "are used in the hope that".

### D6. "none of these works checks whether it does" (SHOULD)
Related work, para 1: Tse et al. [10] are described two sentences earlier as "removing every training sequence
that contains the held-out object", which is a content-level check. Write "none of these works measures whether
it does" (matching the paragraph-2 sentence "None of these measures the case ...").

### D7. "Four checks gate every sweep" (note)
3.4: the constituent-leak check applies only to joined-clip trajectories and is failed "by design" by the misaligned
version; the frame-leak check runs on five seeds of derived datasets only. "Four checks gate the sweeps" is accurate.

### D8. Missing p for one stated comparison (note)
4.2: "GRAB at a budget of 64 ... not distinguishable from the zero-truth control at that budget (difference +0.5,
95 % CI -2.2 to 3.1)" gives no p where every neighbouring comparison does (p = 0.73 in
`runs/DIAGNOSTICS_RESULTS.md` section 24). Add it, or the sentence looks selectively reported.

### D9. Checked and consistent (no action)
350 of 609 recordings with several primitives (Intro) = 609 − 259 single-primitive (App. B); 613 frames at 30 fps =
20 s; 254 frames = 8.5 s; 72 frames = 2.4 s; 148 frames = 4.9 s; "both above zero" for the two zero-truth values
(t ≈ 5.0 and 5.7); Bonferroni: all four primary p-values × 4 stay below 10⁻⁵; 82 % loss = 12.11 / 14.79; +2.7 =
5.64 − 2.96; +1.7 = 17.75 − 16.07; 4.1 = 5.64 − 1.54; 8.0 = 10.63 − 2.59; Table 1 differences match the prose
roundings; "necessary but not sufficient" (5) agrees with "cannot test a necessary one" (5) and with Abstract
"shown here by manipulation on one dataset".

---

## E. TMLR submission checklist

Quoted requirements are from the TMLR pages as fetched on 2026-10-05.

| # | Requirement (quoted) | Source | Status |
|---|---|---|---|
| E1 | "Submissions must be PDF files generated using the TMLR LaTeX stylefile and template" (github.com/JmlrOrg/tmlr-style-file) | author guide | **NOT MET.** The PDF is Word-built (Producer "Microsoft® Word", Letter 8.5 × 11 in, single column, Times New Roman, numeric "[n]" references). The TMLR template is a specific LaTeX class with author-year `natbib` citations and the running header "Under review as submission to TMLR". The guide names no alternative format. The manuscript must be converted to `tmlr.sty` (CLAUDE.md: downloading the style file needs the author's OK). |
| E2 | "TMLR uses a double blind review process and submissions must be anonymized." | author guide; editorial policies: "identities of authors and invited reviewers are withheld" | **MET in the PDF text.** Title block reads "Anonymous authors / Paper under double-blind review"; no acknowledgements, affiliation, e-mail, GitHub or website URL, "our previous work", or self-identifying phrase anywhere, including the 39 references and the AI declaration (searched for author name, institution, project name, repository). *Two residual leaks outside the text:* (a) PDF metadata Creator/Producer is the Traditional-Chinese Word string "適用於 Microsoft 365 的 Microsoft® Word" (a locale hint; Author field is "python-docx", harmless) — scrub metadata or, better, rebuild in LaTeX, which removes it; (b) see E5 for the supplement. |
| E3 | Length: "Submissions may be any length, but a paper's length should be justified by its content" (author guide); "reviews must be submitted within 2 weeks of their assignment by the AE for submissions up to 12 pages ... 4 weeks for submissions over 12 pages" (reviewer guide) | | **Borderline.** In this layout the main text runs from page 1 to part of page 12 (Conclusion and Broader impact end on p. 12; References begin on p. 12; Appendices A–D pp. 14–16; AI declaration p. 17). That is about 11.5 main-text pages. The TMLR template (10 pt, wider text block) will re-flow it; re-count after conversion. If it lands over 12, expect the 4-week review clock. |
| E4 | "If their work carries a significant risk of harm, authors are required to include a Statement of Broader Impact" (author guide; "otherwise optional", editorial policies); ethics page: "expected, when applicable, to include a discussion about potential negative societal impacts" and, for human-derived data, consent / PII / bias | | **MET.** "Broader impact statement" present on p. 12. *Suggestion:* add one sentence that all four datasets are public releases of consenting participants used under their licences and that no new human data were collected, since the ethics page lists human-derived data explicitly. |
| E5 | "Authors may submit up to 100MB of supplementary material, such data, source code or illustrative videos; all supplementary materials must be in PDF or ZIP format" and must remain anonymised | author guide | **NOT YET MET.** The paper promises a supplement four times (Intro: "the per-seed results, gate transcripts, written readings and a short explanatory video are supplementary material"; 3.8: "documents supplied with the results"; App. D: "listed in the supplement"; AI declaration: "both supplied"). No TMLR supplement ZIP exists (only the RA-L-era `submit_ral/multimedia_supplement.zip`). It must be assembled as one ZIP ≤ 100 MB. **Anonymity:** 44 of the 146 gate transcripts in `runs/gates/` carry the absolute path `C:\Users\maurice\Desktop\hand_IK\...` (command lines and Python tracebacks, e.g. `cover_oakink_category_subject.txt` line 1, `composition_leak_taco_action_tool_h3.txt` lines 11–14). Strip the user path before packaging. `runs/PREREG_*.md`, `runs/DIAGNOSTICS_RESULTS.md`, `runs/*/results.json` and `docs/tmlr/supplement_all_sweeps.md` are clean. The video (`docs/figures/explainer.mp4`, 2.9 MB) shows title A and no name. Do not include or link the public GitHub release; it is under the author's name. |
| E6 | "We encourage authors to, whenever possible, upload materials that improve reproducibility ... data or code" | author guide | Encouraged, not required. The AI declaration states the scripts are supplied, so include `scripts/verify_tmlr_draft.py` and the per-seed `results.json` files in the E5 ZIP, with paths scrubbed. |
| E7 | Reproducibility statement | — | Not required by TMLR. Section 3.8 plus the AI declaration serve. MET. |
| E8 | Author contributions / Acknowledgements | author guide (not required at submission) | Correctly absent in the anonymous version; the TMLR template adds them at camera-ready. MET. |
| E9 | "LLMs may be used as general-purpose assistive tools ... authors are fully responsible for content"; "LLMs are not eligible for authorship" (editorial policies); acceptance criteria caution against papers "AI generated with little or no human involvement" | | **MET.** The "Declaration of AI assistance" exceeds the requirement (no mandatory disclosure). It states the LLM "drafted text under author direction" — the authors should be comfortable that reviewers read this against the "little or no human involvement" caution; the sentence "The authors set the research questions, decided which claims to make and are responsible for the content" is the right one to keep. Place the declaration before the references or as the last main-text section rather than after Appendix D, where it is easy to miss. |
| E10 | "TMLR does not accept submissions that have any overlap with previously published work"; "only accepts original contributions that don't reuse the authors' own prior work ... expanded versions of conference papers" | TMLR front page; editorial policies | **MET.** RA-L 26-5425 was returned and withdrawn, never published. Preprints are allowed ("Authors may post to preprint servers ... provided the ... double-blind status remains protected"). *Risk only:* the public GitHub release and the project web page carry the author's name and the previous title; reviewers "are expected not to take any actions that would violate this double blind state, e.g. searching", so this is permitted, but do not cite or link them. |
| E11 | OpenReview profile complete with affiliations, conflicts, publication history | author guide | Outside the PDF; the author's task at upload. |

---

## F. Abstract

- Word count: **209** (TMLR sets no limit; the council target was about 180–200).
- One claim: yes — "Partial labels therefore hide compositional difficulty, shown here by manipulation on one
  dataset." The three numbers before it all serve that claim.
- Acronyms: "CI" (undefined; standard), "TACO", "OakInk-Image" (dataset names). Write "95 % confidence interval"
  once if space allows; otherwise acceptable.
- One clarity fix (A6): "(+2.6 % of the naive error)" directly after "whose true penalty is zero" reads as a
  contradiction on first pass → "whose true penalty is zero and which reads +2.6 % of the naive error".

---

## Summary

### Must-fix before upload
1. **Format (E1):** rebuild in the TMLR LaTeX template; a Word PDF does not satisfy "Submissions must be PDF files
   generated using the TMLR LaTeX stylefile and template". Needs the author's OK to download the style file.
2. **Supplement (E5):** assemble the promised ZIP (per-seed results, gate transcripts, pre-registrations, verifier
   script, video, the "supplement" sweep list) ≤ 100 MB, and strip `C:\Users\maurice\...` from the 44 gate
   transcripts first.
3. **D3 / 3.8:** "every sweep in Sections 4.1 ..." claims pre-registration for the zero-truth and planted synthetic
   sweeps, which have none; narrow the sentence.
4. **D2 / Conclusion:** "tests a motion model only where each label covers the whole trajectory" needs the
   "On OakInk-Image" scope the Discussion already has.
5. **B1 / Intro vs Appendix D:** "Appendix D lists every sweep run under the protocol" is contradicted by
   Appendix D's own first sentence.

### Should-fix (no number changes)
Appendix B tense (A1); minus sign (A2); non-breaking "p = " (A3); "-0.0" (A4); β / KL / RMS (A5); the nine
sentences in A6; "synthetic control" → "zero-truth control" in 4.1 (C1); define axis, stride, exposure (C2);
"shape-by-intent" on which GRAB grid (C4); "Table 3" cited for a CI it does not contain (B2); "describe" → "name"
for TACO in the Conclusion (D4); citation or softening for primitive-bank priors (D5); "checks" → "measures" (D6);
add p = 0.73 for GRAB at budget 64 (D8); one data-ethics sentence in the Broader impact statement (E4);
abstract parenthesis (F).

### TMLR requirements not met
- **E1** LaTeX template (PDF is Word-built, different layout and citation style).
- **E5** Supplementary ZIP not yet assembled; gate transcripts it will contain carry a de-anonymising user path.
- **E3** is not a violation but is borderline: about 11.5 main-text pages in this layout; re-count after the LaTeX
  conversion, since over 12 pages moves the review to the 4-week clock.
- **E2** residual: PDF metadata carries a Traditional-Chinese Word locale string; removed automatically by the
  LaTeX rebuild.
