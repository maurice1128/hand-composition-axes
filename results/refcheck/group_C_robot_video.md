# Reference check, group C (robot learning + video action recognition)

Date: 2026-10-03. Keys: gao2024efficient, wang2026diagnosing, chen2025robohiman, punnakkal2021babel,
moltisanti2017trespassing, gowda2021new, brattoli2020rethinking.

Method: every paper's full text was downloaded (curl) and read with pymupdf; bibliographic fields were
checked against the paper's own first page, the publisher record (RSS proceedings page, Crossref/IEEE
record, Springer chapter page) and the arXiv API. Quotes below are from the opened files; page numbers are
the printed footers of the opened version. The scratch copies are in
`...\scratchpad\refcheck\{gao2024_rss,wang2026,chen2025,babel_cvf,moltisanti_cvf,gowda2021,brattoli_cvf}.pdf/.txt`.

The earlier `references_audit.md` was not relied on.

---

### gao2024efficient
- Full text opened: https://www.roboticsproceedings.org/rss20/p013.pdf (20 pp., RSS XX, Delft, July 15-19
  2024; also https://arxiv.org/pdf/2403.05110, identical text). Publisher record:
  https://www.roboticsproceedings.org/rss20/p013.html; DOI 10.15607/RSS.2024.XX.013 resolves to that PDF.
- Bib fields: OK. Authors (Gao, Xie, Xiao, Finn, Sadigh), title, year and DOI match the RSS record exactly.
  RSS's own BibTeX uses `booktitle = {Proceedings of Robotics: Science and Systems}`; Crossref's container
  title is "Robotics: Science and Systems XX". Either is acceptable; the entry's form is Crossref's.
  Optional additions: `address = {Delft, Netherlands}`, `month = jul`.
- Definitive version cited? Yes (RSS 2024 proceedings with DOI; arXiv id kept as eprint, which is fine).
- Claims:
  - [SUPPORTED] "The test is established in ... robot manipulation [@gao2024efficient; ...]" — evidence:
    "we study the proficiency of visual imitation learning policies when evaluated on unseen combinations
    of environmental factors" (Sec. I, p. 1).
  - [SUPPORTED] "unseen combinations of environmental factors [@gao2024efficient] ... evaluate policies" —
    evidence: "transferred to entirely new environments that encompass unseen combinations of environmental
    factors ... success rate of 77.5%" (Abstract, p. 1); "Our work instead assesses compositional
    generalization, where policies must reason about unseen combinations of factor values" (Sec. II, p. 2).
    The factors are object position, object type, table texture, table height, distractors, camera pose
    (Sec. I).
- Required fix: none.

### wang2026diagnosing
- Full text opened: https://arxiv.org/pdf/2607.29687 (19 pp., v1 submitted 2026-07-31; CoRL-style
  "Abstract:" template, no venue footer).
- Bib fields: OK. All eight authors and their order match the PDF (Yixiao Wang, Cheng-En Wu, Lingfeng Sun,
  Pengcheng Wang, Xiang Ji, Boyuan Liang, Guojian Zhan, Masayoshi Tomizuka); title and year match.
- Definitive version cited? Yes, as far as one exists. Crossref title and bibliographic searches
  (2026-10-03) return no published version; the arXiv record carries no journal-ref or DOI. The `@misc`
  arXiv form is correct for now. Re-check before camera-ready.
- Claims:
  - [SUPPORTED] "The test is established in ... robot manipulation [... @wang2026diagnosing]" — evidence:
    "the robot must execute new instruction tuples whose individual components are familiar, but whose
    combinations were rarely or never observed during training" (Sec. 1, p. 1); "which instruction tuples
    must be covered to generalize to held-out recombinations?" (Sec. 2, p. 2).
  - [PARTLY SUPPORTED] "unseen combinations ... of instructions and contexts [@wang2026diagnosing]
    evaluate policies" — what the paper holds out is combinations of *instruction components* (object,
    container, button; "instruction tuples"); contexts are not a composed factor but a consequence:
    "Unseen values of l−i can still generate novel contexts" (Sec. 3.3, p. 5) and "poor skill routing
    under unseen instruction combinations?" (Sec. 5.3, p. 8). "Context–action shift" is one of the three
    terms in their decomposition of the gap (Abstract), not a held-out axis. The phrase "combinations of
    instructions and contexts" therefore misdescribes the split, though mildly.
- Required fix (sentence): replace "and of instructions and contexts [@wang2026diagnosing]" with
  "and of instruction components [@wang2026diagnosing]" (or "of instruction tuples and the contexts they
  induce [@wang2026diagnosing]" if the context aspect is wanted).

### chen2025robohiman
- Full text opened: https://arxiv.org/pdf/2510.13149 (21 pp., v1 2025-10-15, every page headed
  "Preprint. Under review."; ICLR template).
- Bib fields: OK. All nine authors and their order match the PDF (Yangtao Chen, Zixuan Chen, Nga Teng Chan,
  Junting Chen, Junhui Yin, Jieqi Shi, Yang Gao, Yong-Lu Li, Jing Huo); title and year match. The arXiv
  comment states the first two authors contributed equally (no bib change needed).
- Definitive version cited? Yes, as far as one exists. Crossref (2026-10-03) returns no published version
  of this title; the nearest hit by the same group is a different paper (DeCo, IEEE RA-L 2026,
  10.1109/LRA.2026.3666363), not RoboHiMan. Re-check before camera-ready.
- Claims:
  - [SUPPORTED] "The test is established in ... robot manipulation [... @chen2025robohiman ...]" —
    evidence: "we propose RoboHiMan, a hierarchical evaluation paradigm for compositional generalization
    in long-horizon manipulation" (Abstract, p. 1).
  - [SUPPORTED] "unseen combinations ... of atomic skills [@chen2025robohiman] ... evaluate policies" —
    evidence: "How well can models trained solely on atomic skills generalize to long-horizon
    compositional tasks in the presence of various perturbations?" (Sec. 1, p. 1); "HiMan-Bench evaluates
    whether models can compose atomic skills to accomplish long-horizon tasks" (Sec. 1, p. 3). Caveat, not
    contradicting the sentence: training levels L2-L4 add some compositional demonstrations ("additional
    training data with compositional examples improves performance, a substantial gap persists", Sec. 1,
    p. 3), so the composition is fully unseen only at level L1.
- Required fix: none.

### punnakkal2021babel
- Full text opened: https://openaccess.thecvf.com/content/CVPR2021/papers/Punnakkal_BABEL_Bodies_Action_and_Behavior_With_English_Labels_CVPR_2021_paper.pdf
  (10 pp., footers 722-731).
- Bib fields: OK. Authors, title, venue, year match the paper and Crossref (10.1109/CVPR46437.2021.00078,
  pages 722-731, which here coincide with the CVF footers). Optional: add
  `doi = {10.1109/CVPR46437.2021.00078}`.
- Definitive version cited? Yes (CVPR 2021 proceedings).
- Claims:
  - [PARTLY SUPPORTED] "a single sequence label leaves most of a motion-capture sequence undescribed
    [@punnakkal2021babel]" — the paper makes the argument and gives one worked example, not a
    dataset-wide statistic. Supporting: "labeling the entire sequence with the single label is imprecise,
    and problematic. First, many actions such as turn and grasp are ignored and remain unlabeled"
    (Sec. 3, p. 723); "we examine a typical sequence (see the Sup. Mat.), and find that only 20% of the
    duration of the sequence labeled as pick up and place object corresponds to this action ... walking
    towards and away from the object – actions that remain unlabeled – account for 40% of the duration"
    (Sec. 3, pp. 723-724). Dataset-wide facts that bear on "most": "a single mocap sequence has 5.01
    segments on average, with 3.4 unique actions" (Sec. 1, p. 723), but also "For 47.7% of the sequences,
    both annotators agree that they contain multiple actions" (Sec. 3.1, p. 724), i.e. for about half of
    AMASS sequences one label was judged to cover the whole sequence. So "most of a sequence undescribed"
    is BABEL's illustrative example (80% of one sequence), not a general measurement; as a general claim
    it overstates what the paper measured.
- Required fix (sentence): "For motion data, a single sequence label can leave most of a motion-capture
  sequence undescribed: in BABEL's worked example the labelled action covers 20 % of the sequence
  [@punnakkal2021babel], ..." (or, weaker and fully general: "a single sequence label leaves other
  actions in the sequence unlabelled [@punnakkal2021babel]").

### moltisanti2017trespassing
- Full text opened: https://openaccess.thecvf.com/content_ICCV_2017/papers/Moltisanti_Trespassing_the_Boundaries_ICCV_2017_paper.pdf
  (9 pp., footers 2886-2894).
- Bib fields: authors, title, venue, year OK. Pages: the entry's 2886-2894 are the CVF open-access
  footers; the IEEE Xplore record (DOI 10.1109/ICCV.2017.314, Crossref) paginates the same paper
  2905-2913. Either numbering is defensible but the entry should carry the DOI, and the page numbers
  should match whichever version is cited (if the DOI is added, IEEE pagination is the consistent choice).
- Definitive version cited? Yes (ICCV 2017 proceedings); DOI missing.
- Claims:
  - [SUPPORTED, one wording caveat] "the temporal bounds of hand-object interactions change recognition
    accuracy [@moltisanti2017trespassing]" — evidence: "We demonstrate that the recognition rate drops by
    2-10% when temporal bounds are modified albeit within an Intersection-over-Union of more than 0.5"
    (Sec. 1 contributions, p. 2886); "As IoU changes from > 0.9 to > 0.5, we observe a drop of more than
    20%" on CMU (Sec. 4, p. 2890); re-annotation with Rubicon Boundaries gives "a 4% increase in overall
    accuracy" (Abstract). Scope: the paper is about *object interactions in egocentric video* (BEOID,
    GTEA Gaze+, CMU kitchen; actions such as pour, open door, cut pepper); it never uses the term
    "hand-object interaction", though its bounds are tied to the hands ("the first and last frames when the
    hands are visible", Sec. 3.1, p. 2888). The gloss "hand-object interactions" is fair but not the
    paper's term.
- Required fix: bib: add `doi = {10.1109/ICCV.2017.314}` (and, if IEEE pagination is adopted,
  `pages = {2905--2913}`). Sentence (optional, for precision): "the temporal bounds of object interactions
  in egocentric video change recognition accuracy [@moltisanti2017trespassing]".

### gowda2021new
- Full text opened: https://arxiv.org/pdf/2107.13029 (v2, 2021-09-13, "Accepted to GCPR 2021"; 15 pp.,
  LNCS layout, page header "A New Split for Evaluating True Zero-Shot Action Recognition"). Publisher
  record: Crossref 10.1007/978-3-030-92659-5_12 and https://link.springer.com/chapter/10.1007/978-3-030-92659-5_12
  (full Springer text is paywalled; the arXiv v2 is the accepted version).
- Bib fields: authors (Gowda, Sevilla-Lara, Kim, Keller, Rohrbach), title, pages 191-205, year, DOI all
  match Crossref/Springer. Discrepancies: the LNCS `volume = {13024}` is missing; Springer's book title is
  "Pattern Recognition: 43rd DAGM German Conference, DAGM GCPR 2021, Bonn, Germany, September 28 - October
  1, 2021, Proceedings" (the entry's "Pattern Recognition (DAGM GCPR 2021)" is an acceptable short form).
- Definitive version cited? Yes (Springer LNCS chapter with DOI).
- Claims:
  - [PARTLY SUPPORTED] "classes treated as unseen reach pre-training under other names [... @gowda2021new],
    an overlap defined by class identity" — "under other names" is directly supported: "typing (class in
    UCF101) that the model detects as using computer (class in Kinetics), applying eye makeup ... detects
    as filling eyebrows" (Sec. 5.2, p. 6); "classes which are supposed to be unseen, are present during
    supervised pre-training" (Abstract). The overlap is indeed decided class by class (whole classes are
    moved to the seen set). But Gowda's criterion is *not* names alone: "we compute visual and semantic
    similarities, and discard those classes that are too similar. To calculate visual similarity, we use an
    I3D model pre-trained on Kinetics-400 and evaluate all video samples" (Sec. 5.2, p. 6), explicitly to
    catch overlaps "often not recognized in terms of semantic similarities"; they criticise Roitberg et al.
    for looking "only ... at the semantic similarity of labels" (Sec. 2, p. 3). So if "defined by class
    identity" is read as "by class name", that describes Brattoli and Roitberg, not Gowda; if it is read
    as "at the granularity of classes rather than of clips", it holds for all three.
- Required fix: bib: add `volume = {13024}`. Sentence: keep, or sharpen to "an overlap decided class by
  class (by name similarity, or by name and visual similarity in @gowda2021new)" if the name-versus-content
  contrast is load-bearing in that paragraph.

### brattoli2020rethinking
- Full text opened: https://openaccess.thecvf.com/content_CVPR_2020/papers/Brattoli_Rethinking_Zero-Shot_Video_Classification_End-to-End_Training_for_Realistic_Applications_CVPR_2020_paper.pdf
  (11 pp., footers 4613-4623).
- Bib fields: authors (Brattoli, Tighe, Zhdanov, Perona, Chalupka), title, venue, year OK. Pages: the
  entry's 4613-4623 are the CVF footers; the IEEE Xplore record (DOI 10.1109/CVPR42600.2020.00467,
  Crossref) paginates 4612-4622. Same situation as Moltisanti: add the DOI and keep the pagination
  consistent with the version cited.
- Definitive version cited? Yes (CVPR 2020 proceedings); DOI missing.
- Claims:
  - [SUPPORTED] "classes treated as unseen reach pre-training under other names [... @brattoli2020rethinking
    ...], an overlap defined by class identity" — evidence: "The simple solution – to remove source class
    names from target classes or vice-versa – does not work, because two classes with slightly different
    names can easily refer to the same concept" (Sec. 3.3, p. 4616); overlap is defined purely on names:
    "d(c1, c2) = cos(W2V(c1), W2V(c2))" with threshold τ = 0.05 (Eq. 4, p. 4616), and "Remove from Kinetics
    700 all the classes whose distance to any class in UCF ∪ HMDB is smaller than τ" (Sec. 4.2, p. 4617).
    Also "Zhu et al. have 23 classes in their training datasets that overlap with the test dataset"
    (Sec. 2, p. 4614).
- Required fix: bib: add `doi = {10.1109/CVPR42600.2020.00467}` (and `pages = {4612--4622}` if IEEE
  pagination is adopted).

---

## Summary

| key | full text opened | bib OK | claims (supported / partly / not) |
|---|---|---|---|
| gao2024efficient | Y (RSS proceedings PDF) | Y | 2 / 0 / 0 |
| wang2026diagnosing | Y (arXiv 2607.29687) | Y (arXiv-only, no published version found) | 1 / 1 / 0 |
| chen2025robohiman | Y (arXiv 2510.13149) | Y (arXiv-only, no published version found) | 2 / 0 / 0 |
| punnakkal2021babel | Y (CVF CVPR 2021) | Y (DOI optional) | 0 / 1 / 0 |
| moltisanti2017trespassing | Y (CVF ICCV 2017) | N (DOI missing; pages are CVF, IEEE is 2905-2913) | 1 / 0 / 0 |
| gowda2021new | Y (arXiv v2 = accepted GCPR version) | N (LNCS volume 13024 missing) | 0 / 1 / 0 |
| brattoli2020rethinking | Y (CVF CVPR 2020) | N (DOI missing; pages are CVF, IEEE is 4612-4622) | 1 / 0 / 0 |

## Required fixes

Bib entries:
1. `moltisanti2017trespassing`: add `doi = {10.1109/ICCV.2017.314}`; pages 2886-2894 are the CVF numbering,
   IEEE Xplore has 2905-2913 — choose one convention for all CVF/IEEE entries.
2. `brattoli2020rethinking`: add `doi = {10.1109/CVPR42600.2020.00467}`; pages 4613-4623 are CVF, IEEE
   has 4612-4622.
3. `gowda2021new`: add `volume = {13024}`.
4. `punnakkal2021babel` (optional): add `doi = {10.1109/CVPR46437.2021.00078}`.
5. `wang2026diagnosing`, `chen2025robohiman`: no published version exists as of 2026-10-03; re-run the
   Crossref title query before camera-ready.

Sentences:
1. Wang: "and of instructions and contexts [@wang2026diagnosing]" -> "and of instruction components
   [@wang2026diagnosing]".
2. BABEL: "a single sequence label leaves most of a motion-capture sequence undescribed
   [@punnakkal2021babel]" -> "a single sequence label can leave most of a motion-capture sequence
   undescribed: in BABEL's worked example the labelled action covers 20 % of the sequence
   [@punnakkal2021babel]". The current wording presents one illustrative sequence as a general measurement,
   and BABEL's annotators judged only 47.7 % of sequences to contain multiple actions.
3. Moltisanti (optional precision): "hand-object interactions" -> "object interactions in egocentric
   video".
4. Gowda (optional): "an overlap defined by class identity" holds for all three papers only in the sense
   "decided class by class"; Gowda also uses I3D visual similarity, so do not describe it as name-only.
