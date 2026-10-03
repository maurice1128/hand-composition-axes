# Reference check, group B: compositional-generalisation references

Checked 2026-10-03 against full text (PDF downloaded and read with pymupdf), not metadata. Keys: finegan2018improving,
lake2018generalization, keysers2020measuring, hupkes2023taxonomy, sun2023validity, atzmon2020causal,
materzynska2020something, tse2025collaborative. Citing sentences are those listed in `docs/tmlr/citation_contexts.md`;
the truncated "Hupkes et al." and "Sun et al." sentences were completed from `docs/tmlr/PAPER_TMLR.md` lines 77-103.

### finegan2018improving
- Full text opened: https://aclanthology.org/P18-1033.pdf (10 pages, proceedings pp. 351-360)
- Bib fields: OK. First page: "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics
  (Long Papers), pages 351-360, Melbourne, Australia, July 15-20, 2018". Authors on the paper: Catherine Finegan-Dollak,
  Jonathan K. Kummerfeld, Li Zhang, Karthik Ramanathan, Sesh Sadasivam, Rui Zhang, Dragomir Radev; order and spelling
  match. ACL Anthology's booktitle string is "(Volume 1: Long Papers)", as in the .bib. DOI 10.18653/v1/P18-1033 is
  the Anthology DOI.
- Definitive version cited? yes (ACL 2018 proceedings).
- Claims:
  - [SUPPORTED] "show that splitting text-to-SQL data by question leaves the SQL queries being predicted in both
    training and test, and propose splitting by query instead." — evidence: "Traditional question-based splits allow
    queries to appear in both train and test. Our query-based split ensures each query is in only one." (Figure 1
    caption, p. 351); "we propose a complementary new division, where no SQL query is allowed to appear in more than
    one set; we call this the query split." (Section 5, p. 355). Also: "If at least one copy of every SQL query
    appears in training, then the task evaluated is classification, not true semantic parsing" (Section 5, p. 355).
- Required fix: none.

### lake2018generalization
- Full text opened: https://proceedings.mlr.press/v80/lake18a/lake18a.pdf (PMLR 80, 10 pages); metadata from
  https://proceedings.mlr.press/v80/lake18a.html (firstpage 2873, lastpage 2882, authors Brenden Lake, Marco Baroni).
- Bib fields: OK. Paper header: "Generalization without Systematicity: On the Compositional Skills of
  Sequence-to-Sequence Recurrent Networks", Brenden Lake, Marco Baroni; footer "Proceedings of the 35th International
  Conference on Machine Learning, Stockholm, Sweden, PMLR 80, 2018". Pages 2873-2882 confirmed by the PMLR record.
- Definitive version cited? yes (ICML 2018 / PMLR 80, not the arXiv version).
- Claims:
  - [SUPPORTED] "The test is established in language [lake2018generalization; ...]" — evidence: "At test time, the
    model has to execute all composed commands for the action that it only saw in the primitive context (e.g., 'jump
    twice'...)" (Section 4, Experiment 3, p. 2877). SCAN holds out compositions of a seen primitive.
  - [SUPPORTED] "SCAN [...] and CFQ [...] do so for language" (withhold combinations whose parts were seen) — same
    passage, Experiment 3; also Experiment 2 (length split) and Experiment 1 (random split).
  - [SUPPORTED] "models trained with 'turn left' seen only as a bare command still generalise to its compositions,
    which the authors attribute to its output action appearing in training inside other commands, whereas they fail
    on 'jump', whose action appears nowhere else" — evidence, all from Section 4, Experiment 3, pp. 2877-2878:
    - the experiment: "the model is exposed to the primitive command only denoting a certain basic action (e.g.,
      'jump')... At test time, the model has to execute all composed commands for the action that it only saw in the
      primitive context"; the two variants are "turn left" and "jump", each over-sampled to about 10 % of training
      presentations (footnote 10: "Without over-sampling, performance was consistently worse").
    - the numbers: "For 'turn left', many models generalize very well to composed commands. The best performance is
      achieved by a GRU network with attention ... (90.3% accuracy). The overall-best model achieved 90.0% accuracy."
      "for 'jump,' models are almost completely incapable to generalize to composed commands. The best performance was
      1.2% accuracy ... The overall-best model reached 0.08% accuracy."
    - the authors' explanation: "although models are only exposed to the primitive command during training, they will
      see the action it denotes (LTURN) many times, as it is used to accomplish many directed actions. For example, a
      training item is: 'walk left and jump left', with ground-truth interpretation: LTURN WALK LTURN JUMP. Apparently,
      seeing action sequences containing LTURN suffices ... On the other hand, the action denoted by 'jump' (JUMP) only
      occurs with this primitive command in training".
    - Note for precision: the authors hedge ("Apparently ... probably because the model receives direct evidence about
      how LTURN is used in context"), so "attribute" is fair; "appears nowhere else" paraphrases "only occurs with this
      primitive command in training" correctly. The manuscript does not state the numbers; if it ever does, use 90.3 %
      (best) / 90.0 % (overall-best model) for "turn left" and 1.2 % / 0.08 % for "jump".
- Required fix: none.

### keysers2020measuring
- Full text opened: https://arxiv.org/pdf/1912.09713 (38 pages, header "Published as a conference paper at ICLR
  2020"). OpenReview (https://openreview.net/forum?id=SygcCnNKwr and /pdf?id=SygcCnNKwr) returned a browser-verification
  page to curl, so the ICLR-branded PDF was not fetched directly; the arXiv file is the camera-ready with the ICLR
  header and the same 14 authors.
- Bib fields: OK. Authors on the paper: Daniel Keysers, Nathanael Schärli, Nathan Scales, Hylke Buisman, Daniel
  Furrer, Sergii Kashubin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, Dmitry Tsarkov, Xiao
  Wang, Marc van Zee, Olivier Bousquet; order and spelling match the .bib (14 of 14). Title matches. ICLR 2020 has no
  pages or volume.
- Definitive version cited? yes (ICLR 2020 with OpenReview URL).
- Claims:
  - [SUPPORTED] "The test is established in language [...; keysers2020measuring]" — evidence: "We introduce a novel
    method to systematically construct such benchmarks by maximizing compound divergence while guaranteeing a small
    atom divergence between train and test sets" (Abstract, p. 1).
  - [SUPPORTED] "CFQ varies how far the test distribution diverges from training across several splits" — evidence:
    "we compute the accuracy of each model configuration on a series of divergence-based splits that we produce with
    target compound divergences that span the range between zero and the maximum achievable in 0.1 increments ... For
    each target divergence, we produce at least 3 different splits" (Section 6, p. 7); "a surprisingly strong negative
    correlation between compound divergence and accuracy" (Abstract); R^2 0.81-0.88 (Section 6, p. 8). Note that the
    varied quantity is compound divergence at fixed low atom divergence (<= 0.02), not an arbitrary notion of distance;
    the manuscript's "how far the test distribution diverges" is an acceptable gloss.
- Required fix: none.

### hupkes2023taxonomy
- Full text opened: https://arxiv.org/pdf/2210.03050 (v2, 86 pages, arXiv title "State-of-the-art generalisation
  research in NLP: A taxonomy and review"). Publisher record read from https://www.nature.com/articles/s42256-023-00729-y
  (page marked Open Access; metadata: Nature Machine Intelligence 5(10), 1161-1174, Oct 2023, article type "Analysis",
  20 authors).
- Bib fields: OK. All 20 authors in the Nature record match the .bib in order and spelling (Hupkes, Giulianelli,
  Dankers, Artetxe, Elazar, Pimentel, Christodoulopoulos, Lasri, Saphra, Sinclair, Ulmer, Schottmann, Batsuren, Sun,
  Sinha, Khalatbari, Ryskina, Frieske, Cotterell, Jin). The .bib uses the published title ("A Taxonomy and Review of
  Generalization Research in NLP"), which differs from the arXiv title; that is correct per Black's rule. Volume 5,
  number 10, pages 1161-1174, DOI 10.1038/s42256-023-00729-y all confirmed.
- Definitive version cited? yes (Nature Machine Intelligence).
- Claims:
  - [PARTLY SUPPORTED] "Hupkes et al. review the practice." (antecedent: measuring compositional generalisation by
    withholding combinations whose parts were seen) — evidence: "we present our analysis of the current state of
    generalisation research, grounded on a review of 449 papers and a total of 619 generalisation experiments"
    (Section 1, p. 3); "The first prominent type of generalisation addressed in the literature is compositional
    generalisation" (Section 2.2, p. 7). The paper reviews generalisation research in NLP across six types and five
    axes; compositional generalisation is one type, and the review does not specifically survey the held-out-combination
    split as a practice. "Review the practice" is defensible only if "the practice" is read as generalisation testing
    in NLP; as written, the nearest antecedent is the held-out-combination test.
- Required fix (optional, wording only): "Hupkes et al. [@hupkes2023taxonomy] place such tests in a taxonomy of
  generalisation research in NLP." No bib change.

### sun2023validity
- Full text opened: https://aclanthology.org/2023.conll-1.19.pdf (20 pages, proceedings pp. 274-293)
- Bib fields: OK. First page: "Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL),
  pages 274-293, December 6-7, 2023" (Singapore per the Anthology record). Authors Kaiser Sun, Adina Williams,
  Dieuwke Hupkes; title matches. DOI 10.18653/v1/2023.conll-1.19 is the Anthology DOI.
- Definitive version cited? yes (CoNLL 2023).
- Claims:
  - [PARTLY SUPPORTED] "Sun et al. show that benchmarks built from the same data disagree." — The paper's headline
    result is that compositional benchmarks in general rank models inconsistently: "On average, the concurrence
    between dataset splits is low: a mere 0.22 ... Of the 153 pairs of dataset split we compare ... only 10 pairs
    surpass this threshold" (Section 4.2, p. 279). But on the specific point the sentence makes, the paper's emphasis
    is the opposite: sharing the data source is what makes benchmarks agree MORE. "whether datasets are sampled from
    the same source is more predictive of the resulting model ranking than whether they maintain the same
    interpretation of compositionality" (Abstract, p. 274); "concurrence between experiments that share the same
    source of data averages at 0.38, whereas different data but the same splitting strategy results in an average
    concurrence of 0.32" (Section 4.4, p. 280-281); "Predictably, datasets concur most with themselves" (Section 4.4,
    p. 280). It is true that 0.38 is still below the paper's 0.7 threshold for high concurrence, so same-source splits
    do disagree in absolute terms, but the sentence as written attributes to Sun et al. a finding they frame the other
    way round (same data = relatively more agreement; same splitting strategy on different data = less).
- Required fix (sentence): replace "and Sun et al. [@sun2023validity] show that benchmarks built from the same data
  disagree." with "and Sun et al. [@sun2023validity] show that compositional benchmarks rank the same models
  inconsistently, even when they share a data source." No bib change.

### atzmon2020causal
- Full text opened: https://proceedings.neurips.cc/paper/2020/file/1010cedf85f6a7e24b087e63235dc12e-Paper.pdf
  (12 pages, footer "34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada");
  bibtex from https://proceedings.neurips.cc/paper/2020/file/1010cedf85f6a7e24b087e63235dc12e-Bibtex.bib; arXiv
  2006.14610 (24 pages, with supplement) also opened.
- Bib fields: OK. NeurIPS bibtex: Atzmon, Yuval and Kreuk, Felix and Shalit, Uri and Chechik, Gal; "A causal view of
  compositional zero-shot recognition"; Advances in Neural Information Processing Systems, volume 33, pages 1462-1473,
  2020. All match. (Optional additions the NeurIPS record carries: publisher Curran Associates, Inc.; editors
  Larochelle, Ranzato, Hadsell, Balcan, Lin.)
- Definitive version cited? yes (NeurIPS 33).
- Claims:
  - [SUPPORTED] "The test is established in ... vision [materzynska2020something; atzmon2020causal]" — evidence:
    "People easily recognize new visual categories that are new combinations of known components" (Abstract, p. 1);
    the paper evaluates attribute-object compositional zero-shot recognition on unseen attribute-object pairs (Zappos,
    AO-CLEVr; Section 7).
  - [SUPPORTED] "Atzmon et al. audit label quality on an observational compositional benchmark" — evidence: "The
    MIT-states dataset was labeled automatically using early technology of image search engine based on text
    surrounding images. As a result, labels are often incorrect. We quantified label quality using human raters, and
    found that they are too noisy to be useful for proper evaluations ... Only 32% of raters selected the correct
    attribute for their first choice ... only 47% ... top-2" (Section 7.1, p. 7); "We conclude that this level of ~70%
    label noise is too noisy for evaluating noun-attribute compositionality" (Section 7.1, p. 7; details in
    Supplement H). The manuscript does not state the 70 % figure anywhere (checked `PAPER_TMLR.md`); if it is ever
    added, the exact wording is "~70% label noise" from a rater study with top-1 32 % and top-2 47 %.
- Required fix: none.

### materzynska2020something
- Full text opened: https://openaccess.thecvf.com/content_CVPR_2020/papers/Materzynska_Something-Else_Compositional_Action_Recognition_With_Spatial-Temporal_Interaction_Networks_CVPR_2020_paper.pdf
  (11 pages; CVF record lists pp. 1049-1059).
- Bib fields: OK against the CVF open-access record (authors Joanna Materzynska, Tete Xiao, Roei Herzig, Huijuan Xu,
  Xiaolong Wang, Trevor Darrell, in that order; CVPR 2020; pp. 1049-1059). One note: the IEEE Xplore version (DOI
  10.1109/CVPR42600.2020.00113, Crossref) paginates the same paper 1046-1056. The .bib's booktitle names the IEEE/CVF
  proceedings and uses the CVF pagination, which is internally consistent; do not add the IEEE DOI without switching
  the pages to 1046-1056.
- Definitive version cited? yes (CVPR 2020).
- Claims:
  - [SUPPORTED] "The test is established in ... vision [materzynska2020something; ...]" and "In vision, Something-Else
    holds out verb-object combinations in video" — evidence: "In our setting, the combinations of a verb (action) and
    nouns in the training set do not exist in the testing set" (Section 4, "Compositional Action Recognition", p. 1053);
    "we combine action group 1 with object group A, and action group 2 with object group B, to form the training set,
    termed as 1A + 2B. The validation set is built by flipping the combination into 1B + 2A" (Section 4, p. 1053).
    Both verbs and objects are seen in training, only their pairings are new, so this is a genuine parts-seen holdout.
- Required fix: none.

### tse2025collaborative
- Full text opened: https://ojs.aaai.org/index.php/AAAI/article/download/32800/34955 (AAAI-25 camera-ready, 9 pages,
  "The Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)", pages 7437-7445); https://arxiv.org/pdf/2501.07100
  (v1, 16 pages with supplement) also opened. Crossref record for 10.1609/aaai.v39i7.32800: Proceedings of the AAAI
  Conference on Artificial Intelligence 39(7), 7437-7445, issued 2025-04-11.
- Bib fields: OK except one diacritic. Authors on the paper: Tze Ho Elden Tse, Runyang Feng, Linfang Zheng, Jiho Park,
  Yixing Gao, Jihie Kim, Aleš Leonardis, Hyung Jin Chang; order matches. The .bib has "Leonardis, Ales" (no caron);
  the paper prints "Aleš Leonardis" (Crossref also drops the caron). Title, volume 39, number 7, pages 7437-7445, year,
  DOI all match.
- Definitive version cited? yes (AAAI 2025 proceedings).
- Claims:
  - [PARTLY SUPPORTED] "The test is established in ... hand-object interaction, where ... verb-noun splits exist for
    H2O and FPHA [@tse2025collaborative]" — the splits exist: "We extend H2O and FPHA datasets with compositional
    splits" (Abstract, p. 7437). But the sentence's frame is "withholding combinations whose parts were seen", and
    Tse et al.'s split removes the noun from training altogether: "we remove the sequences that contain the
    predefined object category from the train split, such that the combinations of a verb (action) and nouns do not
    overlap in the testing set ... we evaluate the model using N_obj-fold cross validation" (Section "Compositional
    action recognition", p. 7442); the task is "recognising a seen action when facing new objects" (Introduction,
    p. 7438). So the noun part is unseen, unlike Something-Else where both parts are seen. The splits are verb-noun
    splits by the authors' own name, but they are seen-verb / unseen-object splits, not parts-seen combination splits.
  - [SUPPORTED] "Tse et al. build compositional verb-noun splits on H2O and FPHA for action recognition." — evidence:
    "we first extend the two existing 3D annotated egocentric hand-object datasets, H2O and FPHA, by introducing new
    compositional splits" (Introduction, p. 7438); the evaluated task is "compositional action recognition" (Table 2,
    p. 7443), with hand pose (MEPE) reported under the same split as a secondary measure. It is a recognition paper
    (plus 3D hand-object reconstruction), not forecasting or generation, as the sentence says.
  - Also verified (asked for by the audit brief): yes, it removes training sequences containing the held-out object
    category, quoted above; S2 additionally removes two object categories at random on H2O.
- Required fix:
  - bib: `Leonardis, Ales` -> `Leonardis, Ale{\v{s}}`.
  - sentence (optional, for precision in the "parts were seen" frame): "... and Tse et al. [@tse2025collaborative]
    build verb-noun splits on H2O and FPHA for action recognition, removing every training sequence that contains the
    held-out object." The Related-work sentence at line 83-84 can stay as is.

## Summary

| key | full text opened | bib OK | claims (supported / partly / not) |
|---|---|---|---|
| finegan2018improving | Y (ACL Anthology PDF) | Y | 1 / 0 / 0 |
| lake2018generalization | Y (PMLR PDF) | Y | 3 / 0 / 0 |
| keysers2020measuring | Y (arXiv camera-ready with ICLR header; OpenReview blocked curl) | Y | 2 / 0 / 0 |
| hupkes2023taxonomy | Y (arXiv v2 full text; Nature record for metadata) | Y | 0 / 1 / 0 |
| sun2023validity | Y (ACL Anthology PDF) | Y | 0 / 1 / 0 |
| atzmon2020causal | Y (NeurIPS proceedings PDF + arXiv) | Y | 2 / 0 / 0 |
| materzynska2020something | Y (CVF open access PDF) | Y (note IEEE pagination differs) | 2 / 0 / 0 |
| tse2025collaborative | Y (AAAI OJS PDF + arXiv) | Y except "Ales" -> "Aleš" | 1 / 1 / 0 |

## Required fixes

1. `sun2023validity`, sentence (PAPER_TMLR.md line 101): replace "show that benchmarks built from the same data
   disagree" with "show that compositional benchmarks rank the same models inconsistently, even when they share a
   data source". The paper reports that same-source benchmarks agree more, not less, than different-source ones (0.38
   vs 0.32 concurrence), though both are below its 0.7 threshold.
2. `tse2025collaborative`, bib: `Leonardis, Ales` -> `Leonardis, Ale{\v{s}}`.
3. Optional precision: (a) `hupkes2023taxonomy` "review the practice" -> "place such tests in a taxonomy of
   generalisation research in NLP"; (b) `tse2025collaborative` in the first Related-work sentence sits under
   "combinations whose parts were seen" but its split removes the held-out object from training entirely; add
   "removing every training sequence that contains the held-out object" or move it out of that frame.
4. No change to `finegan2018improving`, `lake2018generalization`, `keysers2020measuring`, `atzmon2020causal`,
   `materzynska2020something` (if an IEEE DOI is ever added to the last, change pages to 1046-1056).

Disagreements with `references_audit.md`: none on bibliographic facts; that audit's reading of Sun et al. ("benchmarks
and splitting strategies rank the same models differently") is correct and is what the manuscript sentence should say,
rather than "built from the same data disagree".
