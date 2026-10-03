# Reference check, group A: datasets, the prior and its building blocks

Checked 2026-10-03 against the full text of each paper (PDF opened and text-extracted; file copies in the session scratchpad `refcheck/`). Inputs: `docs/tmlr/citation_contexts.md` (citing sentences) and `docs/tmlr/PAPER_TMLR.md` Appendix A, Table A1. The earlier metadata-only `references_audit.md` was not relied on; where this check disagrees with it, that is said.

Keys: yang2026lamp, yang2022oakink, zhan2024oakink2, taheri2020grab, liu2024taco, romero2017embodied, higgins2017beta, kingma2014auto, cho2014learning.

---

### yang2026lamp
- Full text opened: https://arxiv.org/pdf/2607.06323 (v1, 7 Jul 2026, 17 pages). arXiv abstract page lists only "17 pages, 11 figures": no journal reference or venue, so arXiv is the only version.
- Bib fields: OK. Authors (Yang, Ma, Yu, Chen, Yang, Chai, Chen, Yu), title, year and arXiv id match the first page exactly. primaryClass cs.RO matches the arXiv stamp.
- Definitive version cited? Yes (no published version exists as of today).
- Claims:
  - [PARTLY SUPPORTED] "Priors with a continuous latent space [LAMP] and priors built from a bank of primitives both assume that what is learned separately recombines" — The continuous-latent part is supported: "We introduce LMPM, a history-conditioned latent motion prior that learns a continuous, decodable" latent (Sec. 1, contributions, p. 2) and "decodes continuous latent commands into executable high-dimensional hand targets" (Abstract). The recombination assumption is not LAMP's: the paper never discusses composition or recombination of separately learned motions (no occurrence of "compos", "recombin" anywhere in the text; its "generalization" is spatial, Sec. 5). The sentence attributes an assumption to LAMP that LAMP does not state. Acceptable only if read as the manuscript's own interpretation.
  - [PARTLY SUPPORTED, overstated] "The prior follows LAMP [LAMP]; we adopt its design and do not propose one." — What is shared is the idea of a KL-regularised Gaussian latent over a hand-motion history that decodes to hand targets: "q_phi(z_t|H_t) = N(mu_phi(H_t), diag(sigma^2_phi(H_t))), h_hat_{t+1} = D_theta(z_t)" and "a reconstruction objective and a KL-regularized bottleneck" with weight beta (Sec. 3.1, Eqs. 2-3, p. 3). The *design* is not the same: LAMP's "encoder maps an 8-step hand-target history to a Gaussian latent prior in R2, and its decoder maps a latent sample to the next 6-D absolute hand target. Both encoder and decoder are MLPs with hidden width 256. We train with a KL weight of 10-3" (Appendix B.2, p. 12; Table: "Input history 8 hand targets ... MLP encoder/decoder, width 256, dz = 2"). The manuscript's prior is a two-layer bidirectional GRU over a 32-frame window with a 12-D latent per frame, a unidirectional GRU decoder, beta annealed to 1.0 and free bits (Sec. 3.2). Architecture, latent size, input length, output (window reconstruction vs next-step prediction) and KL weight all differ, so "we adopt its design" is not accurate; "we follow its idea" is.
- Other errors about this paper in the manuscript: none found.
- Required fix: replace the Sec. 3.2 sentence with: "The prior is a latent motion prior in the sense of LAMP [@yang2026lamp], a Gaussian latent over a window of hand motion trained with a reconstruction term and a KL penalty; we do not propose this idea. LAMP's prior is a multilayer perceptron over an 8-step history with a 2-dimensional latent that predicts the next hand target; ours is recurrent over a 32-frame window with a 12-dimensional latent per frame, as follows." The CLAUDE.md rule ("we adopt, never we propose") is preserved; only the false equivalence of designs is removed. In the Introduction sentence, consider "Priors with a continuous latent space [@yang2026lamp] ... are used on the assumption that ..." so the assumption is not attributed to LAMP.

---

### yang2022oakink
- Full text opened: https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_OakInk_A_Large-Scale_Knowledge_Repository_for_Understanding_Hand-Object_Interaction_CVPR_2022_paper.pdf (10 pages, footer pages 20953-20962); also https://arxiv.org/pdf/2203.15709 (17 pages, with appendix) for the counts not in the main text.
- Bib fields: OK. Authors (Lixin Yang, Kailin Li, Xinyu Zhan, Fei Wu, Anran Xu, Liu Liu, Cewu Lu), CVPR 2022, pages 20953-20962 all match the CVF copy. Cosmetic: the paper's own header writes "Large-scale"; the CVF index and the bib write "Large-Scale". Either is acceptable.
- Definitive version cited? Yes (CVPR).
- Claims:
  - [NOT A CLAIM ABOUT THE PAPER / PARTLY] "OakInk-Image [@yang2022oakink] (770 trajectories at 30 fps, median 72 frames)" — the 770 and 72 are the manuscript's own counts. The paper gives no sequence count and no frame rate anywhere (CVF and arXiv versions searched for "fps", "Hz", "frames per second": none). What it states: "230,064 RGB-D frames capturing 12 subjects performing up to 5 intent-oriented hand interactions with objects in a pool of total 100 instances from 32 categories" (Sec. 1, p. 1) and "230K image frames ... 100 real-world objects of 32 categories" (Sec. 3.4, p. 5). The 30 fps therefore rests on the release, not the paper (see Table A1 below).
  - [SUPPORTED] "the object's category, functional class and affordance come from the dataset's own taxonomy [@yang2022oakink]" — "a taxonomy that groups Oak base objects into two levels ... manipulation tools (maniptool) and function tools (functool)" (top level = functional class), "WordNet categories as the sub-level classification. The total number of categories in the Oak base is 32" (Sec. 3.1, p. 3), and "The affordance is represented by a set of attributes ... total 30 phrases" (Sec. 3.1, pp. 3-4). Intents "use, hold, lift-up, hand-out, and receive" (Sec. 3.2.2, p. 4) and 12 subjects (Sec. 3.2.2) support the manuscript's intent and subject factors; "hand-out" paired with "receive" supports "OakInk-Image hand-overs" (manuscript Sec. 3.1).
- Other errors about this paper in the manuscript:
  - Table A1, "Published: 792 clips, 32 object categories". The "32 object categories" is in the paper (Sec. 3.1). The "792 clips" is NOT in the paper (neither CVF nor arXiv/appendix); it comes from the dataset's GitHub README (https://github.com/oakink/OakInk, "It contains 792 multi-view video clips (230K images)"). Labelling it "Published" without saying where misattributes it to the paper. The paper's published count is "230,064 RGB-D frames" / "230K image frames".
  - Manuscript Sec. 1 says "An OakInk-Image trajectory is a single grasp lasting a median of 2.4 s". The paper's protocol: subjects "start from a hand pose lying flat on the table, pick up the assigned object, and finish the action with a given intent" (Sec. 3.2.2, p. 4), so a clip is one intent-oriented interaction including the reach and pick-up, not literally a single grasp. Minor; "a single intent-oriented interaction" would be exact.
  - The manuscript uses 33 categories against the paper's 32 (Table A1 says "categories follow the dataset's metaV2 taxonomy file"). This is disclosed, but the reader is not told why the release has one more; one clause ("metaV2 adds one category to the paper's 32") would close it.
- Required fix: Table A1, OakInk-Image "Published" cell: "230K RGB-D frames, 100 objects, 32 categories, 12 subjects, up to 5 intents (paper); 792 multi-view clips (dataset README)". Optionally add a footnote that the frame rate is taken from the release.

---

### zhan2024oakink2
- Full text opened: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhan_OAKINK2_A_Dataset_of_Bimanual_Hands-Object_Manipulation_in_Complex_Task_CVPR_2024_paper.pdf (12 pages, footer 445-456); also https://arxiv.org/pdf/2403.19417 (26 pages, main text plus supplement).
- Bib fields: OK. Authors (Zhan, Yang, Zhao, Mao, Xu, Lin, Li, Lu), title, CVPR 2024, pages 445-456 all match.
- Definitive version cited? Yes (CVPR).
- Claims:
  - [NOT SUPPORTED on the capture rate; counts OK] "OakInk2 [@zhan2024oakink2] (609 trajectories, motion capture subsampled from 120 Hz to 30 fps, median 613 frames)" — The paper states the opposite of 120 Hz: "We synchronize all sensors at 30 fps and calibrate the transformation between these two systems" (Sec. 3.2.1, p. 5, said of the MoCap system and the RGB cameras together) and "synchronized at 30 fps, with resolution 848 x 480" (Sec. 4.1, p. 6). The string "120" does not occur anywhere in the CVF or the 26-page arXiv text. The counts behind "609" are supported: "627 sequences of bimanual dexterous hand-object interaction in total. 363 of these are for Primitives and 264 are for Complex Tasks. In total, OAKINK2 contains 4.01M image frames" (Sec. 4.2, p. 6).
- Other errors about this paper in the manuscript:
  - Table A1, "Published: 627 sequences (363 primitive, 264 complex task), motion capture at 120 Hz". The 627/363/264 are exactly the paper's. "Motion capture at 120 Hz" is not published anywhere I could open (paper, supplement, project page). The project notes (CLAUDE.md, 2026-10-01) say the 120 Hz was inferred from "the annotation's image frames", i.e. from the release files, not from the paper. This matters beyond attribution: the manuscript's "subsampled by 4" to 30 fps, the "median 20 s" recording length, the "1.07 s" window on OakInk2 and the "median 613 frames" all depend on the raw annotation being 120 Hz. If the released pose annotation is at the paper's 30 fps, the bundle is 7.5 fps, the window is 4.3 s and the median recording is 82 s, which is what the project's loader originally recorded and what one council reviewer flagged. The manuscript's Sec. "What changed" sentence ("the annotation's image frames show 120 Hz") should state the evidence from the release (e.g. the timestamp spacing or the frame-count-to-duration ratio of a named file), because the paper cannot be cited for it.
- Required fix: (1) In Sec. 3.1 write "motion capture released at 120 Hz and subsampled to 30 fps here (the paper reports its synchronised sensors at 30 fps; the rate used is that of the released pose files)" only if the release rate has been re-verified from the data; otherwise revert to 30 fps and recompute the OakInk2 durations. (2) Table A1 "Published" cell: "627 sequences (363 primitive, 264 complex task), 4.01M frames, sensors synchronised at 30 fps" and move the 120 Hz statement to the "Difference" column with its source.

---

### taheri2020grab
- Full text opened: https://arxiv.org/pdf/2008.11200 (v1, 37 pages: main paper plus supplementary). Publisher record: Crossref 10.1007/978-3-030-58548-8_34 (Computer Vision - ECCV 2020, LNCS, Springer International Publishing, pages 581-600).
- Bib fields: OK. Authors (Taheri, Ghorbani, Black, Tzionas), title, ECCV 2020, LNCS, Springer, pages 581-600, DOI all match Crossref. Optional: add `volume = {12349}` (LNCS volume) for completeness.
- Definitive version cited? Yes (ECCV/LNCS).
- Claims:
  - [SUPPORTED for what is attributable; 1,048 and 254 are the manuscript's own] "GRAB [@taheri2020grab] (1,048 trajectories at 30 fps, median 254 frames, from subject archives s1 to s7 and s10)" — the paper: "The dataset contains 1334 sequences and over 1.6M frames of MoCap" (Sec. 3.5, p. 10; Table 1: Use 579, Pass 414, Lift 274, Off-hand 67, total 1334); "a Vicon system with 54 infrared 'Vantage 16' cameras that capture 16 MP at 120 fps" (Sec. 3.1, p. 7); "10 different people (5 male and 5 female) interacting with 51 objects" (Sec. 1, p. 3). So 120 Hz to 30 fps by subsampling 4 is consistent, and using 8 of 10 subjects is a disclosed subset. Sequence counts per subject are not in the paper, so 1,048 cannot be checked against it.
- Other errors about this paper in the manuscript:
  - Table A1 "1,334 sequences, 10 subjects, 120 Hz": all three are the paper's own numbers (Sec. 3.5 Table 1; Sec. 3.4; Sec. 3.1). Correct.
  - Manuscript Sec. 1: "A GRAB trajectory is a whole interaction of 8.5 s including approach and release" — protocol supports "including approach and release": "(iii) the subject starts from a T-pose and approaches the object, (iv) performs the instructed task, and (v) leaves the object and returns to a T-pose" (Sec. 3.4, p. 10). The 8.5 s is the manuscript's measurement.
  - Manuscript Sec. 3.1: "the fine label is the object and the intent verb of the sequence name; ... the intent class groups the verbs". The paper's published intents are four: "'use' and 'pass' (to someone) ... as well as 'lift' and 'off-hand pass'" (Sec. 3.4, p. 9). The finer verbs (e.g. drink, fly) come from the release's file names, not from the paper; this is consistent with the manuscript's wording but the paper should not be read as the source of the fine verbs. "GRAB's passes" (manuscript Sec. 3.1, on left-hand motion) matches the "off-hand pass" intent: "the subject grasps the object with the offhand, passes it to the dominant hand" (Supp. Protocol Details). The manuscript's "9 grasp-relevant shape classes" is its own grouping, not GRAB's.
- Required fix: none required. Optional: add LNCS volume 12349 to the bib entry.

---

### liu2024taco
- Full text opened: https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_TACO_Benchmarking_Generalizable_Bimanual_Tool-ACtion-Object_Understanding_CVPR_2024_paper.pdf (12 pages, footer 21740-21751).
- Bib fields: OK. Authors (Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, Li Yi), title with "Tool-ACtion-Object", CVPR 2024, pages 21740-21751 all match.
- Definitive version cited? Yes (CVPR).
- Claims:
  - [SUPPORTED] "TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco]" — "Test set 3 (S3): Interaction-level generalization. The interaction triplet is novel, while the tool categories and geometries are included in the training set" (Sec. 5.1, p. 7). "Every tool" corresponds to the paper's "tool categories and geometries"; exact.
  - [SUPPORTED] "Held-out-triplet results for hand-object motion forecasting come from one partition and one run per split [@liu2024taco], so neither the size of this floor nor the spread over splits is known." — One fixed partition: "The training set and the four test sets follow a data ratio of 4:1:1:1.5:2.5" (Sec. 5.1, p. 7). Table 4 ("Results on generalizable interaction forecasting", p. 8) reports a single value per method and test set; no standard deviation, confidence interval, seed count or repeated split appears anywhere in the paper (searched for "+-", "std", "seed", "runs"). The claim is supported by the absence of any variance report.
  - [SUPPORTED] "TACO evaluates motion forecasting on held-out triplets [@liu2024taco]" — Sec. 5.3 "Generalizable Hand-object Motion Forecasting" evaluates on S1-S4; "methods consistently exhibit significant performance declines with the right hand and the tool, regardless of whether applied to novel tool geometries (S2, S4) or interaction triplets (S3, S4)" (Sec. 5.3, p. 7).
  - [SUPPORTED, with the paper's actual numbers noted] "TACO [@liu2024taco] (2,317 trajectories at 30 fps, median 148 frames)" — frame rate: "The frequency of our cameras and mocap system is 30 Hz" (Sec. 3.1, p. 3). Count: the paper says "TACO contains 2.5K motion sequences" (Abstract; Sec. 3.3, p. 4); 2,317 is the released count, and Table A1 reconciles it ("about 2.5K ... 2,317 ... the released data, all of which are used"). Consistent.
  - [SUPPORTED] "On TACO the right hand usually holds the tool [@liu2024taco]" — "Consider using the right hand to manipulate a tool and the left hand to operate a target object" (Sec. 5.4, problem formulation, p. 7); "the tool and the hand holding it play dominant roles ... significant performance declines with the right hand and the tool" (Sec. 5.3, p. 7). The paper states it as its formulation rather than as a per-sequence statistic, so "usually" is the right hedge.
- Other errors about this paper in the manuscript:
  - Table A1 "131 triplets, about 2.5K sequences": matches "131 <tool category, action label, target object category> triplets" (Sec. 1, p. 1; Sec. 3.3, p. 5) and "2.5K motion sequences" (Abstract). Correct. The paper also publishes "20 object categories, 196 fine-grained object 3D models, 14 participants, and 15 actions" (Sec. 3.3, p. 4); the manuscript's "15 actions" (Sec. 4.2) agrees. The manuscript's "17 tools, 9 objects" (Sec. 4.2) are counted from the release's directory names; the paper counts 20 object categories over tools and targets together, so the two are not comparable and should not be presented as the paper's numbers.
  - Manuscript Sec. 1: "A TACO trajectory is described by its authors as one tool-use action" — the paper formulates "a behavior as using a tool category to perform an action on a target object category" (Sec. 3.3, p. 5) and labels each sequence with one triplet; supported.
- Required fix: none required.

---

### romero2017embodied
- Full text opened: https://arxiv.org/pdf/2201.02610 (author-posted copy of the ACM TOG article, 19 pages: article 245:1-245:17 plus two appendix pages 245:18-19). Publisher record: Crossref 10.1145/3130800.3130883 (ACM Transactions on Graphics 36(6), pages 1-17, 2017).
- Bib fields: OK. Authors (Romero, Tzionas, Black), title, ACM TOG vol. 36 no. 6, article 245, pages 245:1-245:17, year 2017, DOI all match the first page ("0730-0301/2017/11-ART245", "Vol. 36, No. 6, Article 245. Publication date: November 2017"). The note "Proc. SIGGRAPH Asia 2017" is correct. The eprint 2201.02610 is a 2022 arXiv posting of the 2017 paper; harmless but unnecessary given the DOI.
- Definitive version cited? Yes (ACM TOG).
- Claims:
  - [SUPPORTED / NOT A CLAIM ABOUT THE PAPER] "Each includes MANO hand parameters [@romero2017embodied]." — the citation is the origin of MANO: "we develop a new model called MANO (hand Model with Articulated and Non-rigid defOrmations) ... learned from around 1000 high-resolution 3D scans of hands of 31 subjects" (Abstract, p. 1). That the four datasets include MANO parameters is a claim about those datasets, which their papers support (OakInk Sec. 3.2.3; OakInk2 Sec. 4.1 "pose and shape for ... hands"; GRAB Sec. 3.2 "MANO [60] for the hands"; TACO Sec. 3.2 "recover MANO [53] hand meshes").
- Other errors about this paper in the manuscript: Appendix A refers to "MANO's mean pose"; MANO does define a mean pose (the release's `hands_mean`), consistent with the paper's pose parameterisation. None found.
- Required fix: none. Optional: drop the eprint fields, since the DOI identifies the definitive version.

---

### higgins2017beta
- Full text opened: https://openreview.net/references/pdf?id=Sy2fzU9gl (13 pages; header "Under review as a conference paper at ICLR 2017"; the OpenReview forum lists it as an ICLR 2017 poster). The `openreview.net/pdf?id=` and `attachment?...` endpoints returned an HTML error page from this environment; the `references/pdf` endpoint served the PDF.
- Bib fields: OK. Authors (Higgins, Matthey, Pal, Burgess, Glorot, Botvinick, Mohamed, Lerchner), title, ICLR 2017, OpenReview id all match the first page. No arXiv version exists.
- Definitive version cited? Yes (ICLR 2017 via OpenReview, the only version).
- Claims:
  - [PARTLY SUPPORTED] "The prior trains as a beta-VAE [@kingma2014auto; @higgins2017beta] for 120 epochs ... with ... a free-bits floor of 0.02 nats per dimension." — beta-VAE is "a modification of the variational autoencoder (VAE) framework. We introduce an adjustable hyperparameter beta that balances latent channel capacity and independence constraints with reconstruction accuracy" (Abstract, p. 1). Two caveats. (a) The manuscript's beta "rises linearly to 1.0" and stays there, and the beta-VAE paper defines its contribution as "beta > 1 qualitatively outperforms VAE (beta = 1)" (Abstract): a model scored at beta = 1 is, in the paper's own terms, a VAE with KL warm-up, not a beta-VAE. (b) The free-bits floor is not from either cited paper (it is from Kingma et al. 2016, "Improved variational inference with inverse autoregressive flow"); the sentence does not attribute it explicitly, but a reader will. 
- Other errors about this paper in the manuscript: none.
- Required fix: either cite beta-VAE only for the KL weight schedule ("a VAE [@kingma2014auto] whose KL weight beta [@higgins2017beta] rises linearly to 1.0 over the first 12 epochs, with a free-bits floor [cite Kingma et al. 2016] of 0.02 nats per dimension") or keep "beta-VAE" and state that the final beta is 1. Add the free-bits citation or make clear it is a training detail not from the cited papers.

---

### kingma2014auto
- Full text opened: https://arxiv.org/pdf/1312.6114 (14 pages; the ICLR 2014 paper).
- Bib fields: OK. Authors (Kingma, Welling), title, ICLR 2014, arXiv 1312.6114 match. ICLR 2014 has no page numbers.
- Definitive version cited? Yes (ICLR 2014; arXiv is the canonical copy).
- Claims:
  - [SUPPORTED] "The prior trains as a beta-VAE [@kingma2014auto; @higgins2017beta] ..." — as the origin of the VAE and reparameterisation: "a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized" and "fitting an approximate inference model (also called a recognition model) to the intractable posterior" (Abstract, p. 1). The caveats under higgins2017beta (beta = 1; free bits) apply to the sentence, not to this citation.
- Other errors about this paper in the manuscript: none.
- Required fix: none.

---

### cho2014learning
- Full text opened: https://aclanthology.org/D14-1179.pdf (11 pages). Publisher record: Crossref 10.3115/v1/D14-1179.
- Bib fields: OK. Authors (Cho, van Merrienboer, Gulcehre, Bahdanau, Bougares, Schwenk, Bengio), title, EMNLP 2014, pages 1724-1734, Doha, ACL, DOI all match the header "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724-1734, October 25-29, 2014, Doha, Qatar".
- Definitive version cited? Yes (ACL Anthology / EMNLP).
- Claims:
  - [SUPPORTED] "Two-layer bidirectional gated recurrent units [@cho2014learning] of 256 units encode ..." — the gated unit is introduced here: "First, the reset gate r_j is computed by ..." and "Similarly, the update gate z_j is computed by ..." (Sec. 2.3, p. 1726), "The update gate z selects whether the hidden state is to be updated ... The reset gate r decides whether the previous hidden state is ignored" (Sec. 2.3, p. 1727). The paper does not use the name "GRU" (coined later) and does not propose bidirectionality or stacking; the citation is for the unit only, which is the normal use.
- Other errors about this paper in the manuscript: none.
- Required fix: none.

---

## Summary

| key | full text opened | bib OK | claims (supported / partly / not) |
|---|---|---|---|
| yang2026lamp | Y (arXiv 2607.06323) | Y | 0 / 2 / 0 (the "adopt its design" sentence is the one that must change) |
| yang2022oakink | Y (CVF + arXiv) | Y | 1 / 1 / 0 (counts sentence: 30 fps and 770 are not in the paper); Table A1 "792 clips" is from the GitHub README, not the paper |
| zhan2024oakink2 | Y (CVF + arXiv incl. supp.) | Y | 0 / 0 / 1 (paper says all sensors synchronised at 30 fps; "120 Hz" appears nowhere); Table A1 "motion capture at 120 Hz" same problem |
| taheri2020grab | Y (arXiv 2008.11200 + Crossref) | Y (optional: add LNCS volume 12349) | 1 / 0 / 0 |
| liu2024taco | Y (CVF) | Y | 5 / 0 / 0 |
| romero2017embodied | Y (arXiv copy of TOG + Crossref) | Y | 1 / 0 / 0 |
| higgins2017beta | Y (OpenReview references/pdf) | Y | 0 / 1 / 0 (final beta = 1; free bits uncited) |
| kingma2014auto | Y (arXiv 1312.6114) | Y | 1 / 0 / 0 |
| cho2014learning | Y (ACL Anthology) | Y | 1 / 0 / 0 |

Disagreements with `references_audit.md`: it reads LAMP as "the prior design the project adopts"; the full text shows a different architecture (MLP, 8-step history, 2-D latent, next-step decoding), so only the idea is adopted. It did not check any dataset statistic; the OakInk2 120 Hz and OakInk 792-clip figures do not come from the cited papers.

## Required fixes (ordered by consequence)

1. **OakInk2 capture rate (Sec. 3.1, Table A1, "What changed" paragraph).** The paper states "We synchronize all sensors at 30 fps" (Sec. 3.2.1) and "synchronized at 30 fps" (Sec. 4.1); no 120 Hz appears in the paper, its supplement or its project page. Re-verify the released pose annotation's rate from the data (timestamp spacing or frames per recorded second of a named file). If it is 120 Hz, say so with that evidence and write "released at 120 Hz (the paper reports its synchronised sensors at 30 fps)"; if it is 30 fps, the subsample-by-4 bundle is 7.5 fps and the OakInk2 durations (median 20 s, 1.07 s window, median 613 frames at 30 fps) must be recomputed. Table A1 "Published" cell: "627 sequences (363 primitive, 264 complex task), 4.01M frames, sensors synchronised at 30 fps".
2. **LAMP (Sec. 3.2).** Replace "The prior follows LAMP [@yang2026lamp]; we adopt its design and do not propose one." with "The prior is a latent motion prior in the sense of LAMP [@yang2026lamp], a Gaussian latent over a window of hand motion trained with a reconstruction term and a KL penalty; we do not propose this idea. LAMP's prior is a multilayer perceptron over an 8-step history with a 2-dimensional latent that predicts the next hand target; ours is recurrent over a 32-frame window with a 12-dimensional latent per frame, as follows." In Sec. 1, make the recombination assumption the manuscript's, not LAMP's.
3. **OakInk Table A1.** "Published" cell: "230K RGB-D frames, 100 objects, 32 categories, 12 subjects, up to 5 intents (paper); 792 multi-view clips (dataset README)". The paper gives no frame rate; if 30 fps is kept in Sec. 3.1 it is the release's rate.
4. **beta-VAE sentence (Sec. 3.2).** Either "a VAE [@kingma2014auto] whose KL weight beta [@higgins2017beta] rises linearly to 1.0 over the first 12 epochs, with a free-bits floor [Kingma et al. 2016] of 0.02 nats per dimension", or keep "beta-VAE" and state that the final beta is 1 (the cited paper defines beta-VAE by beta > 1). Add a citation for free bits (Kingma, Salimans, Jozefowicz, Chen, Sutskever, Welling, "Improved variational inference with inverse autoregressive flow", NeurIPS 2016) or drop the term.
5. Optional, no claim at stake: OakInk Sec. 1 "a single grasp" -> "a single intent-oriented interaction" (the protocol starts flat on the table and includes the pick-up); GRAB bib add `volume = {12349}`; romero2017embodied drop the eprint fields; TACO Sec. 4.2 make clear that "17 tools, 9 objects" are counted from the release (the paper publishes "20 object categories, 196 object instances, 15 actions, 14 participants").
