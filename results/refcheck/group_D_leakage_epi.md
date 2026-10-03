# Reference check, group D: leakage and epidemiology controls

Keys: roitberg2018towards, bansal2018zero, zhang2020zstad, kapoor2023leakage, lipsitch2010negative,
schuemie2014interpreting, schuemie2018empirical. Checked 2026-10-03 against the full text of each paper (PDF or PMC
full HTML), not against metadata. Bibliographic fields were compared with the paper's own first page and the Crossref
record for the DOI. Scratch copies of every file opened are in
`C:\Users\maurice\AppData\Local\Temp\claude\C--Users-maurice-Desktop-hand-IK\e3e1e6df-a744-4919-821f-e8aca52a8fb9\scratchpad\refcheck\`.

Citing sentences (all in Section 2, "Related work", of `docs/tmlr/PAPER_TMLR.md`, lines 93-108):

- S-A: "Zero-shot detection removes training images that contain unseen objects [@bansal2018zero], and zero-shot temporal
  activity detection removes training clips that contain an unseen activity [@zhang2020zstad]; both state this as
  protocol rather than measure what happens without it."
- S-B: "In zero-shot action recognition, classes treated as unseen reach pre-training under other names
  [@roitberg2018towards; @brattoli2020rethinking; @gowda2021new], an overlap defined by class identity."
- S-C: "A survey of leakage in machine-learning-based science does not single out partial label coverage among its sources
  [@kapoor2023leakage]."
- S-D: "A condition whose true effect is zero is the negative control of epidemiology [@lipsitch2010negative] and the basis
  of empirical calibration of estimates [@schuemie2014interpreting; @schuemie2018empirical]; in machine learning,
  Engstrom et al. [@engstrom2020identifying] measure the bias of a replicated test set in the same spirit."

---

### roitberg2018towards
- Full text opened: https://link.springer.com/content/pdf/10.1007/978-3-030-11018-5_8.pdf (full 9-page chapter, pp. 97-105,
  served without login; the CVF ECCVW 2018 open-access URL returned 404; no arXiv copy exists; Semantic Scholar lists the
  open-access status as CLOSED). Crossref: https://api.crossref.org/works/10.1007/978-3-030-11018-5_8
- Bib fields: OK. Authors (Roitberg, Martinez, Haurilet, Stiefelhagen), title, booktitle "Computer Vision -- ECCV 2018
  Workshops", LNCS, Springer, pages 97-105, year 2019 (the chapter's footer reads "ECCV 2018 Workshops, LNCS 11132,
  pp. 97-105, 2019"; Crossref published-online 2019-01-23) and DOI all match. Optional additions, not errors:
  `volume = {11132}`, `editor = {Leal-Taix{\'e}, Laura and Roth, Stefan}`.
- Definitive version cited? Yes (Springer LNCS chapter is the only published version).
- Claims:
  - [PARTLY SUPPORTED] S-B "classes treated as unseen reach pre-training under other names ... an overlap defined by class
    identity" — evidence for "under other names" and "class identity": "External datasets often contain slightly diverging
    variants or specializations of the target actions (e.g., drinking beer and drink)" (Sect. 2.2, p. 100) and "actions from
    external dataset are far closer, often even identical, to the target classes" (Sect. 3, p. 102; Fig. 2). Their fix
    filters source *labels* by word-vector similarity, so the overlap is indeed defined at the class-name level (Table 1,
    "Exclude exact labels" vs "Exclude similar labels (ours)"). What the paper does NOT say: "pre-training". In Roitberg et
    al. the external dataset (Kinetics) is the zero-shot *source training set* on which the I3D visual model is trained
    (Sect. 3: "As our visual recognition model, we use I3D ... trained using SGD"); the word "pre-training" does not occur
    in the paper. The "pre-training" wording fits Gowda et al. (TruZe, Kinetics pre-training) but is borrowed for Roitberg.
- Required fix: soften the verb so it covers all three cited papers, e.g. replace "reach pre-training under other names"
  with "reach the training or pre-training data under other names". Bib entry: no change needed.

### bansal2018zero
- Full text opened: https://openaccess.thecvf.com/content_ECCV_2018/papers/Ankan_Bansal_Zero-Shot_Object_Detection_ECCV_2018_paper.pdf
  (ECCV 2018 camera-ready, 17 pp.; identical text to arXiv 1804.04340, also opened at https://arxiv.org/pdf/1804.04340).
  Crossref: https://api.crossref.org/works/10.1007/978-3-030-01246-5_24
- Bib fields: OK. Authors (Bansal, Sikka, Sharma, Chellappa, Divakaran, in that order), title, booktitle "Computer Vision --
  ECCV 2018", LNCS, Springer, pages 397-414, year 2018, DOI and arXiv id all match the first page and Crossref. Optional:
  `volume = {11205}` (LNCS volume for ECCV 2018 Part XIII; not verified from the chapter itself, so leave out unless checked).
- Definitive version cited? Yes (Springer LNCS proceedings chapter).
- Claims:
  - [PARTLY SUPPORTED] S-A "Zero-shot detection removes training images that contain unseen objects" — evidence: "For MSCOCO,
    to avoid taking unseen categories as background boxes, we remove all images from the training set which contain any
    object from unseen categories." (Sect. 4.1, Preparing Datasets, p. 8 of the CVF PDF). Contradicting qualifier in the same
    paragraph: "However, we can not do this for VG because the large number of test categories and dense labeling results in
    most images being eliminated from the training set." So the removal is done for one of the two datasets only.
  - [PARTLY SUPPORTED] S-A "... state this as protocol rather than measure what happens without it" — The removal is
    introduced as a precaution, not ablated: there is no run of MS-COCO with and without the removal. But Bansal et al. do
    *interpret* the no-removal case: on VG, where removal was impossible, "background boxes might actually belong to the seen
    (train) or unseen (test) categories. This leads to the SB model learning sub-optimal visual embeddings", whereas on
    MS-COCO "the SB model increases the recall ... This is because we remove all images that contain any object from unseen
    classes" (Sect. 4.3, Quantitative Results, p. 10). That is a cross-dataset observation confounded by everything else that
    differs between VG and MS-COCO, not a controlled measurement, so "rather than measure" is defensible; "state this as
    protocol" slightly understates that they discuss its consequence.
- Required fix: qualify the sentence so it does not claim the removal was general. Replacement for the first clause of S-A:
  "Zero-shot detection removes training images that contain unseen objects where the label density allows it (on MS-COCO
  but not Visual Genome) [@bansal2018zero], ...". If space is tight, the minimum change is "removes training images that
  contain unseen objects on MS-COCO [@bansal2018zero]". Bib entry: no change needed.

### zhang2020zstad
- Full text opened: https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_ZSTAD_Zero-Shot_Temporal_Activity_Detection_CVPR_2020_paper.pdf
  (CVPR 2020 camera-ready, 10 pp., running page numbers 876-885 visible in the extracted text); arXiv 2003.05583v1 also
  opened at https://arxiv.org/pdf/2003.05583 (same author list, same text). Crossref:
  https://api.crossref.org/works/10.1109/CVPR42600.2020.00096
- Bib fields: OK. Authors (Zhang, Chang, Liu, Luo, Wang, Ge, Hauptmann), title, pages 876-885, year 2020, DOI and arXiv id
  all match the CVF header and the Crossref record.
- **Venue conflict settled: CVPR 2020, not AAAI 2020.** Crossref for DOI 10.1109/CVPR42600.2020.00096 gives container
  "2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", event Seattle, 13-19 June 2020, publisher
  IEEE, pages 876-885; the CVF open-access PDF carries the CVPR 2020 paper id and page numbers. No AAAI record exists for
  this title. The bib entry is correct as written; the earlier "AAAI 2020" note was wrong.
- Definitive version cited? Yes (IEEE/CVF proceedings).
- Claims:
  - [SUPPORTED] S-A "zero-shot temporal activity detection removes training clips that contain an unseen activity" —
    evidence: "we remove any clips containing unseen activities from the training video clips. In addition, we ensure that
    each clip in the testing set contains at least one unseen activity." (Sect. 4, implementation details, p. 881).
  - [SUPPORTED] S-A "state this as protocol rather than measure what happens without it" — evidence: the removal is
    justified only by the setting's definition, "Note that the ZSL setting demands that the training dataset does not
    contain any instances labeled with testing classes. Therefore, we remove ..." (p. 881); no experiment in Tables 1-4
    varies this removal.
- Required fix: none.

### kapoor2023leakage
- Full text opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/ (Patterns 4(9):100804, full text, open access);
  arXiv 2207.07048v1 also opened at https://arxiv.org/pdf/2207.07048 (the preprint's survey count is 329 papers, the
  published version's 294; the taxonomy is the same eight types in both). Crossref:
  https://api.crossref.org/works/10.1016/j.patter.2023.100804
- Bib fields: OK. Authors Kapoor and Narayanan, journal Patterns, volume 4, number 9, article 100804, year 2023, DOI and
  arXiv id all match Crossref and the PMC record (first published online 2023-08-04, issue September 2023).
- Definitive version cited? Yes (the Patterns article).
- Claims:
  - [SUPPORTED] S-C "A survey of leakage in machine-learning-based science does not single out partial label coverage among
    its sources" — The published taxonomy ("we introduce a detailed taxonomy of eight types of leakage", Summary) is, verbatim
    from the "Taxonomy of data leakage" section: [L1.1] No test set; [L1.2] Pre-processing on training and test set; [L1.3]
    Feature selection on training and test set; [L1.4] Duplicates in datasets; [L2] Model uses features that are not
    legitimate; [L3.1] Temporal leakage; [L3.2] Nonindependence between training and test samples; [L3.3] Sampling bias in
    test distribution. None concerns what a label covers; the word "label" occurs in the published text only inside two
    reference-list entries, and "coverage" does not occur. Note for accuracy: Kapoor and Narayanan call the work "a survey
    of literature" plus a taxonomy, so "survey" is their own word.
- Required fix: none.

### lipsitch2010negative
- Full text opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC3053408/ (NIH author manuscript, full text; "Published in
  final edited form as: Epidemiology. 2010 May;21(3):383-388"). Crossref: https://api.crossref.org/works/10.1097/EDE.0b013e3181d61eeb
- Bib fields: OK. Authors Marc Lipsitch, Eric Tchetgen Tchetgen, Ted Cohen (order and spelling match the PMC author block;
  "Tchetgen Tchetgen" is the correct double surname and is given as the family name in Crossref); title matches the PMC
  header (Crossref's title field is truncated to "Negative Controls" but the full title is on the paper); Epidemiology
  21(3):383-388, May 2010, DOI all match. Not an error, but worth knowing: the article has a published correction
  (Epidemiology 21(4):589) noted on the PMC page.
- Definitive version cited? Yes (journal version; the PMC copy is the accepted manuscript of the same article).
- Claims:
  - [SUPPORTED] S-D "A condition whose true effect is zero is the negative control of epidemiology" — evidence: "In the
    absence of a causal association, any measured association with these probe variables would suggest recall bias"
    (Negative control exposures, MS example) and "a negative control outcome (N) should be an outcome such that the set of
    common causes of exposure A and outcome Y should be as identical as possible to the set of common causes of A and N"
    (Fig. 2 discussion); abstract: negative controls "help to identify and resolve confounding as well as other sources of
    error". The defining property is a known absence of causal effect with the same sources of bias, which is what the
    sentence says. The paper names the concept's origin in laboratory practice ("a routine precaution taken in the design of
    biological laboratory experiments", Abstract), so "of epidemiology" is attribution of the epidemiological use, which is
    fair for this paper.
- Required fix: none.

### schuemie2014interpreting
- Full text opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC4285234/ (Statistics in Medicine 33(2):209-218, full text,
  CC BY-NC open access). Crossref: https://api.crossref.org/works/10.1002/sim.5925
- Bib fields: OK. Authors Schuemie, Ryan, DuMouchel, Suchard, Madigan (order matches the paper and Crossref), title
  (Crossref: "Interpreting observational studies: why empirical calibration is needed to correct p-values"), journal,
  volume 33, issue 2, pages 209-218, year 2014 (issue date 30 Jan 2014; online 30 Jul 2013), DOI all match.
- Definitive version cited? Yes.
- Claims:
  - [SUPPORTED] S-D "... the basis of empirical calibration of estimates [@schuemie2014interpreting ...]" — The paper is
    p-value calibration from negative controls: "we applied the same three designs to sets of negative controls: drugs that
    are not believed to cause the outcome of interest ... we fitted distributions to the effect estimates. Using these
    distributions, we compute calibrated p-values" (Abstract). The manuscript's phrase "calibration of estimates" is
    slightly loose for this paper, which calibrates *p-values* (the 2018 paper is the one that calibrates the estimate's
    confidence interval), but the negative-control basis is exactly what the sentence attributes to it.
- Required fix: none strictly required. Optional precision: "the basis of empirical calibration of p-values and confidence
  intervals [@schuemie2014interpreting; @schuemie2018empirical]".

### schuemie2018empirical
- Full text opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC5856503/ (PNAS 115(11):2571-2577, full text; the PNAS PDF URL
  is behind a Cloudflare challenge for curl, so PMC was used). Crossref: https://api.crossref.org/works/10.1073/pnas.1708282114
- Bib fields: OK. Authors Schuemie, Hripcsak, Ryan, Madigan, Suchard (order matches the paper and Crossref), title, journal
  "Proceedings of the National Academy of Sciences", volume 115, number 11, pages 2571-2577, year 2018 (issue 13 Mar 2018),
  DOI all match. Note: the paper is part of the "Sackler Colloquium on Improving the Reproducibility of Scientific Research";
  no bib field needed.
- Definitive version cited? Yes.
- Claims:
  - [SUPPORTED] S-D "... the basis of empirical calibration of estimates [... @schuemie2018empirical]" — evidence: "we
    extend this work to calibration of confidence intervals (CIs). CIs require positive controls, which we synthesize by
    modifying negative controls" (Abstract); "We describe a statistical procedure for CI calibration that uses negative
    controls as well as positive controls ... we use synthetic positive controls constructed by modifying real negative
    controls" (Introduction). So the 2018 paper calibrates CIs with negative AND synthetic positive controls. The citing
    sentence attributes only the zero-effect (negative-control) basis to it, which is true; it does not mention the
    synthetic positive controls, which is an omission rather than an error. Given that the manuscript's own planted
    control is a synthetic positive control, Schuemie 2018 is also a direct precedent for Section 2's "Planting an effect of
    known size" sentence, where it is currently not cited.
- Required fix: none required. Optional: add `[@schuemie2018empirical]` to the planted-effect sentence ("Planting an effect
  of known size ... the spike-in of genomics [@jiang2011synthetic], the synthetic positive control of observational
  epidemiology [@schuemie2018empirical] and, in machine learning, the planted artefact [@adebayo2022post]").

---

## Summary

| key | full text opened | bib OK | claims (supported / partly / not) |
|---|---|---|---|
| roitberg2018towards | Y (Springer chapter PDF) | Y | 0 / 1 / 0 |
| bansal2018zero | Y (CVF ECCV 2018 PDF; arXiv) | Y | 0 / 2 / 0 (one sentence, two sub-claims) |
| zhang2020zstad | Y (CVF CVPR 2020 PDF; arXiv) | Y (venue is CVPR 2020, AAAI note was wrong) | 2 / 0 / 0 |
| kapoor2023leakage | Y (PMC full text; arXiv) | Y | 1 / 0 / 0 |
| lipsitch2010negative | Y (PMC author manuscript) | Y | 1 / 0 / 0 |
| schuemie2014interpreting | Y (PMC full text) | Y | 1 / 0 / 0 |
| schuemie2018empirical | Y (PMC full text) | Y | 1 / 0 / 0 |

## Required fixes (text only; no bib entry needs changing)

1. S-A, bansal2018zero: the removal of training images containing unseen objects is done for MS-COCO only; Bansal et al.
   state they could not do it for Visual Genome. Replace "Zero-shot detection removes training images that contain unseen
   objects [@bansal2018zero]" with "Zero-shot detection removes training images that contain unseen objects where the label
   density allows it (on MS-COCO but not Visual Genome) [@bansal2018zero]" (or, shorter, "... unseen objects on MS-COCO
   [@bansal2018zero]").
2. S-B, roitberg2018towards: Roitberg et al. train the zero-shot visual model on the external dataset as its source classes;
   the paper never speaks of "pre-training". Replace "reach pre-training under other names" with "reach the training or
   pre-training data under other names".

## Optional precision (not errors)

- S-D: "empirical calibration of estimates" could read "empirical calibration of p-values and confidence intervals", which
  is what the 2014 and 2018 papers respectively do.
- Schuemie 2018 uses synthetic positive controls built from negative controls; it could also be cited in the planted-effect
  sentence of Section 2.
- roitberg2018towards: `volume = {11132}` and the editors (Leal-Taixé and Roth) may be added to the bib entry.
