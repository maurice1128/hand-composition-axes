# Reference check, 2026-10-03 — summary of the six group reports

Scope: the 38 references cited in `docs/tmlr/PAPER_TMLR.md`, each opened in FULL TEXT (not metadata), every citing
sentence checked against the source, every bib field checked against the paper's first page and the publisher
record; plus a format check of the rendered list and the in-text citations. Group reports:
`group_A_datasets.md`, `group_B_compositional.md`, `group_C_robot_video.md`, `group_D_leakage_epi.md`,
`group_E_calibration.md`, `group_F_format.md`. Citing sentences per key: `citation_contexts.md`.

## Coverage

| | count |
|---|---|
| cited references | 38 |
| full text opened | 37 |
| full text NOT accessible without library login | 1: Welch 1947 (Biometrika 34:28-35). Bib fields checked against Crossref. Open it at https://nycu.primo.exlibrisgroup.com/discovery/search?query=any,contains,The%20generalization%20of%20Student%27s%20problem%20when%20several%20different%20population%20variances%20are%20involved&vid=886UST_NYCU:886UST_NYCU&search_scope=MyInst_and_CI&tab=Everything |
| bib entries with a wrong field | 2 (author names: Abelson, Leonardis) |
| bib entries missing a field | 8 (DOIs, LNCS volumes) |
| citing sentences checked | 54 |
| NOT SUPPORTED | 2 (Sun 2023 direction reversed; OakInk2 "120 Hz" attributed to the paper) |
| PARTLY SUPPORTED | 12 |
| SUPPORTED / not a claim | 40 |

## A. Sentences that are wrong (must fix)

| # | key | what the manuscript says | what the source says | fix |
|---|---|---|---|---|
| A1 | sun2023validity | "benchmarks built from the same data disagree" | same-source benchmarks agree MORE (0.38) than different-source ones (0.32); all pairs agree poorly (10 of 153 above 0.7) | "show that compositional benchmarks rank the same models inconsistently, even when they share a data source" |
| A2 | zhan2024oakink2 | "motion capture subsampled from 120 Hz to 30 fps"; Table A1 "motion capture at 120 Hz" as the PUBLISHED figure | the paper says only "We synchronize all sensors at 30 fps" (Sec. 3.2.1, 4.1); "120" appears nowhere. Re-checked on the released files today: every recording holds 4.00 motion-capture frames per image frame (40 recordings, ratio 4.000-4.010), so 120 Hz is a correct INFERENCE from the data, not a published number | 3.1: "motion capture subsampled by 4 to 30 fps (the release holds four motion-capture frames per 30 fps image frame)"; Table A1 published column: "627 sequences (363 primitive, 264 complex task), sensors synchronised at 30 fps"; difference column: "every image frame's motion-capture frame is kept (4 per image frame in the release)" |
| A3 | punnakkal2021babel | "a single sequence label leaves most of a motion-capture sequence undescribed" | that is BABEL's one worked example (labelled action covers 20 % of the sequence; Sec. 3); dataset-wide, 47.7 % of sequences were judged to contain multiple actions | "a single sequence label can leave most of a motion-capture sequence undescribed: in BABEL's worked example the labelled action covers 20 % of the sequence" |
| A4 | yang2026lamp | 3.2 "we adopt its design"; Intro attributes a recombination assumption to LAMP | LAMP's prior is an MLP over an 8-step history with a 2-D latent predicting the next hand target (App. B.2); ours is a bidirectional GRU window autoencoder with a 12-D per-frame latent; shared only in idea (KL-regularised Gaussian latent over hand motion). LAMP never states the recombination assumption | 3.2: "The prior is a latent motion prior in the sense of LAMP, a Gaussian latent over a window of hand motion trained with a reconstruction term and a KL penalty; we do not propose this idea. LAMP's prior is a multilayer perceptron over an 8-step history with a 2-dimensional latent that predicts the next hand target; ours is recurrent over a 32-frame window with a 12-dimensional latent per frame." Intro: "Priors with a continuous latent space ... are used on the assumption that ..." |
| A5 | higgins2017beta / kingma2014auto | "trains as a beta-VAE ... Beta rises linearly to 1.0 ... free-bits floor of 0.02 nats" | Higgins defines beta-VAE as beta > 1 ("beta = 1" is the plain VAE); the free-bits floor is from Kingma et al. 2016 (IAF paper), cited by neither entry | "trains as a VAE [@kingma2014auto] whose KL weight beta [@higgins2017beta] rises linearly to 1.0 over the first 12 epochs, with a free-bits floor [@kingma2016improved] of 0.02 nats per dimension" + new bib entry kingma2016improved (NeurIPS 29, 2016) |
| A6 | yang2022oakink | Table A1 published "792 clips, 32 object categories" | the paper gives 230,064 frames, 100 objects, 32 categories, 12 subjects, up to 5 intents; "792 multi-view video clips" is from the GitHub README only | Table A1: "32 object categories, 100 objects (paper); 792 clips (release README)" |

## B. Sentences that are imprecise (recommended)

| # | key | current | fix |
|---|---|---|---|
| B1 | bansal2018zero | "removes training images that contain unseen objects" | add "on MS-COCO (not possible on Visual Genome)" — Sec. 4.1 says they "can not do this for VG" |
| B2 | roitberg2018towards | "reach pre-training under other names" | "reach the training or pre-training data under other names" — Roitberg trains on the external set; "pre-training" is Gowda's case |
| B3 | wang2026diagnosing | "unseen combinations of instructions and contexts" | "unseen combinations of instruction components" — it holds out instruction tuples (object/container/button); "context" is a term of its error decomposition |
| B4 | bernardi2020geometry | "permutation nulls ... test generalisation across conditions in neuroscience" | "shuffle and random-geometry nulls calibrate cross-condition generalisation in neuroscience" — CCGP is calibrated with a geometric random model, the label shuffle is used for decoding/parallelism |
| B5 | tse2025collaborative | listed under "combinations whose parts were seen" | its split removes every training sequence containing the held-out object (seen verb, unseen object): "build verb-noun splits on H2O and FPHA for action recognition, removing every training sequence that contains the held-out object" |
| B6 | gowda2021new + brattoli2020rethinking | "an overlap defined by class identity" | Brattoli is name-based (Word2Vec on class names); Gowda adds I3D visual similarity and criticises name-only overlap. "an overlap decided class by class, by name or by visual similarity" |
| B7 | hupkes2023taxonomy | "review the practice" | "place such tests in a taxonomy of generalisation research" (449 NLP papers; compositional is one of six types) |
| B8 | moltisanti2017trespassing | "temporal bounds of hand-object interactions" | the paper says "object interactions in egocentric video"; the 2-10 % drop at IoU > 0.5 is confirmed |
| B9 | yang2022oakink | "a single grasp lasting a median of 2.4 s" | OakInk clip = "one intent-oriented interaction" (starts flat on the table, includes pick-up): "a single intent-oriented interaction" |
| B10 | liu2024taco | Table A1 "131 triplets, about 2.5K" (paper) vs "17 tools, 9 objects" (3.1, from the release) | state that 15 actions / 17 tools / 9 objects / 151 triplets are release counts; the paper reports 131 triplets, 20 object categories, 2.5K sequences |

## C. Bib entries

| # | key | fix | source |
|---|---|---|---|
| C1 | adebayo2022post | author "Abelson, Harold" -> "Abelson, Hal" | ICLR 2022 byline, arXiv 2212.04629 |
| C2 | tse2025collaborative | "Leonardis, Ales" -> "Leonardis, Ale{\v{s}}" | AAAI 2025 byline |
| C3 | moltisanti2017trespassing | add doi 10.1109/ICCV.2017.314 (pages: CVF 2886-2894 / IEEE 2905-2913; keep one convention) | IEEE |
| C4 | brattoli2020rethinking | add doi 10.1109/CVPR42600.2020.00467 (CVF 4613-4623 / IEEE 4612-4622) | IEEE |
| C5 | punnakkal2021babel | add doi 10.1109/CVPR46437.2021.00078 | IEEE |
| C6 | taheri2020grab | add volume 12349 (LNCS) | Springer |
| C7 | bansal2018zero | add volume 11205 (LNCS) | Springer |
| C8 | roitberg2018towards | add volume 11132 (LNCS; ECCV 2018 Workshops Part IV, checked on the Springer book page today) | Springer |
| C9 | gowda2021new | add volume 13024 (LNCS); booktitle "Pattern Recognition -- DAGM GCPR 2021" | Springer |
| C10 | higgins2017beta | title "beta-{VAE}" -> "{$\beta$-VAE}" (published title) | OpenReview |
| C11 | welch1947generalization | title quotes: LaTeX `` `{Student's}' `` renders with a stray backtick | renderer or bib |
| C12 | new entry kingma2016improved | Kingma, Salimans, Jozefowicz, Chen, Sutskever, Welling. Improved variational inference with inverse autoregressive flow. NeurIPS 29, 2016 | for A5 |
| C13 | zhang2020zstad | venue confirmed CVPR 2020, pp. 876-885 (the old "AAAI 2020" note was wrong) — bib already correct | CVF + Crossref |
| C14 | wang2026diagnosing, chen2025robohiman, yang2026lamp | arXiv-only; Crossref finds no published version today; re-check before camera-ready | Crossref |

## D. Rendered reference list (renderer `scripts/tmlr_md_to_docx.py`)

In-text citations pass every rule (49 brackets ascending, one space, none sentence-initial, numbered 1-38 by first
appearance, no "?", 38 cited = 38 listed, the 10 uncited .bib entries absent). The list itself needs:
- D1 print a persistent identifier: DOI if present, else URL, else arXiv id (24 DOIs and 4 URLs are never printed now)
- D2 print the issue: volume(number):pages — fixes "36:245:1-245:17" for MANO and restores 21(3), 33(2), 34(1-2), 39(7)
- D3 keep en dashes ("--" -> "–") in page ranges and titles
- D4 suppress the year when the venue string already ends with it ("ECCV 2018, 2018", "ECCV 2020, 2020")
- D5 convert LaTeX quotes before stripping braces (fixes Welch)
- D6 print series/publisher for PMLR and LNCS entries ("PMLR 80:2873-2882", "LNCS 12349:...")
- D7 print `note` when present, or delete the field (MANO "Proc. SIGGRAPH Asia 2017")
- D8 one title-capitalisation rule across entries (five entries differ: Cho, Schuemie 2014, Higgins, Adebayo, Welch)
- D9 venue acronyms uniform: add "(NeurIPS)", "(RSS)", "(ACL)" or drop all

Numbers in the manuscript are untouched by every item above; none of the fixes changes a result.
