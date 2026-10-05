# Reference check, group E (calibration controls and statistics)

Keys: ojala2010permutation, bernardi2020geometry, petigura2013prevalence, jiang2011synthetic,
engstrom2020identifying, adebayo2022post, welch1947generalization.

Method: every full text below was downloaded with curl and read with pymupdf (PDFs) or from the
PMC HTML (Cell, PNAS, Genome Research). Bibliographic fields were compared against the paper's own
first page and against the publisher record (Crossref API for DOIs, JMLR and PMLR index pages,
arXiv API). The earlier metadata-only audit (`references_audit.md`) was not relied on.
Scratch copies: `...\scratchpad\refcheck\{ojala10a,engstrom20a,petigura_arxiv,adebayo_arxiv}.pdf`,
`{bernardi,jiang,petigura}_pmc.html`.

### ojala2010permutation
- Full text opened: https://jmlr.org/papers/volume11/ojala10a/ojala10a.pdf (31 pages; index page
  https://jmlr.org/papers/v11/ojala10a.html)
- Bib fields: OK. First page reads "Journal of Machine Learning Research 11 (2010) 1833-1863 ...
  Published 6/10", authors Markus Ojala, Gemma C. Garriga; index page gives "11(62):1833-1863, 2010".
  Authors, title, journal, volume, number, pages, year and URL all match.
- Definitive version cited? Yes (JMLR; no DOI exists for this volume).
- Claims:
  - [SUPPORTED] "Permutation nulls that preserve structure are standard for classifier evaluation" —
    evidence: "we study two simple permutation tests. The first test ... null distribution is estimated
    by permuting the labels in the data. This test has been used extensively in classification"
    (Abstract, p. 1833); the structure-preserving one is "Test 2 (Permute data columns per class) ...
    applying independent permutations to the columns of X within each class" and "Test 2 is inspired by
    the restricted randomizations from statistics (see, e.g., Good, 2000)" (Section 3, p. 1838).
    Caveat: Ojala and Garriga present Test 2 as their own proposal, grounded in the older statistical
    idea of restricted randomization; "standard" is defensible for permutation nulls in classifier
    evaluation as a whole (Test 1), slightly generous for the within-class variant specifically. No
    change required; if the author wants to be exact, "established" would be safer than "standard".
- Required fix: none.

### bernardi2020geometry
- Full text opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC8451959/ (NIH author manuscript, full
  text incl. STAR Methods "Random models"); publisher record
  https://api.crossref.org/works/10.1016/j.cell.2020.09.031
- Bib fields: OK. Crossref: title "The Geometry of Abstraction in the Hippocampus and Prefrontal
  Cortex", Cell 183(4):954-967.e21, authors Silvia Bernardi, Marcus K. Benna, Mattia Rigotti, Jérôme
  Munuera, Stefano Fusi, C. Daniel Salzman; issued 2020-11 (online 14 Oct 2020, issue 12 Nov 2020).
  All six authors, order and spelling match. (The author-manuscript title drops "the" before
  "hippocampus"; the published title, which the bib uses, has it.)
- Definitive version cited? Yes (Cell).
- Claims:
  - [PARTLY SUPPORTED] "Permutation nulls that preserve structure ... test generalisation across
    conditions in neuroscience" — the "generalisation across conditions" part is exact: "We call the
    performance of this decoder 'cross-condition generalization performance' (CCGP), because it reflects
    the ability of a decoder to generalize to task conditions not used for training" (Results, "The
    cross-condition generalization performance"). The null, however, is not a plain structure-preserving
    permutation. Bernardi et al. use two random models: (i) "a shuffle of the data, in which we assign a
    new, random condition label to each trial for each neuron independently (... preserves the total
    number of trials for each condition)", used for decoding accuracy and the parallelism score (STAR
    Methods, "Random models: Shuffle of the data"; Fig. 3, 5 captions "from a shuffle of the data
    (decoding accuracy and PS)"); and (ii) for CCGP specifically, a "geometric random model" that moves
    cluster centres "to new random locations that are sampled from an isotropic Gaussian distribution"
    and keeps the within-condition noise clouds by "permuting the axes separately for each condition.
    We basically shuffled the neuron labels in a different way for each condition" (STAR Methods,
    "Geometric random model"; Fig. 3, 5 captions "from a geometric random model (CCGP)"). So the
    CCGP null does contain a structure-preserving permutation (of neuron axes, preserving noise
    structure and signal variance), but its signal part is random resampling, not permutation, and the
    label-permutation null is applied to decoding and PS rather than to CCGP. The paper also notes that
    permuting condition labels only within conditions is NOT what they do ("as opposed to the random
    permutation across all trials performed in the shuffle"; "Random models and the analysis of
    different dichotomies"). The sentence is right in spirit (count-preserving shuffle and
    structure-preserving random models calibrate a cross-condition generalisation score) and loose in
    letter.
- Required fix (recommended wording, sentence in Section on calibration controls):
  "Permutation nulls that preserve structure are standard for classifier evaluation
  [@ojala2010permutation], and shuffle and random-geometry nulls calibrate cross-condition
  generalisation in neuroscience [@bernardi2020geometry]."
  Bib entry: no change.

### petigura2013prevalence
- Full text opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC3845182/ (PNAS version, full text) and
  https://arxiv.org/pdf/1311.6806 (author version, 60 pages incl. SI; first page carries
  "www.pnas.org/cgi/doi/10.1073/pnas.1319909110 PNAS Early Edition"); publisher record
  https://api.crossref.org/works/10.1073/pnas.1319909110
- Bib fields: OK. Crossref: "Prevalence of Earth-size planets orbiting Sun-like stars", PNAS 110(48):
  19273-19278, authors Erik A. Petigura, Andrew W. Howard, Geoffrey W. Marcy, issued 2013-11-04 (print
  2013-11-26). All match. (The arXiv journal_ref field says "19175-19176", which is the "In This Issue"
  page, not the article; the bib's 19273-19278 is correct.) Note: PNAS published a correction, 110(48):
  19652 (2013); it does not affect the cited method.
- Definitive version cited? Yes (PNAS, not arXiv).
- Claims:
  - [SUPPORTED] "Planting an effect of known size and measuring its recovery is the injection-recovery
    test of astronomy" — evidence: "The second correction is computed by the injection and recovery of
    synthetic (mock) planet-caused dimmings into real Kepler photometry. We injected 40,000 transit-like
    synthetic dimmings having randomly selected planetary and orbital properties into the actual
    photometry ... We measured survey completeness, C, ... determining the fraction of injected synthetic
    planets that were discovered by TERRA" (Results, "Planet Occurrence" / Fig. 1 caption; Significance
    statement: "We measured the detectability of these planets by injecting synthetic planet-caused
    dimmings into Kepler brightness measurements"). The injected signals have known (randomly drawn,
    recorded) period and radius, so "known size" holds.
- Required fix: none.

### jiang2011synthetic
- Full text opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC3166838/ (full text; Genome Research's
  own PDF link returned an HTML interstitial); publisher record
  https://api.crossref.org/works/10.1101/gr.121095.111
- Bib fields: OK. Crossref: "Synthetic spike-in standards for RNA-seq experiments", Genome Research
  21(9):1543-1551, 2011 (online 2011-08-04, print Sept 2011); authors Lichun Jiang, Felix Schlesinger,
  Carrie A. Davis, Yu Zhang, Renhua Li, Marc Salit, Thomas R. Gingeras, Brian Oliver. All eight authors,
  order and spelling match.
- Definitive version cited? Yes (Genome Research).
- Claims:
  - [SUPPORTED] "Planting an effect of known size and measuring its recovery is ... the spike-in of
    genomics" — evidence: "We used a newly developed pool of 96 synthetic RNAs with various lengths, and
    GC content covering a 2^20 concentration range as spike-in controls to measure sensitivity, accuracy,
    and biases in RNA-seq experiments" (Abstract); "Our objective was twofold: First, determine how
    RNA-seq performs on known inputs and, second, evaluate spike-in controls" (Introduction); the
    standards are the "Phase IV test set of ERCC RNA standards" (Introduction). Known input
    concentrations, measured read-density recovery (Fig. 1C,D).
- Required fix: none.

### engstrom2020identifying
- Full text opened: https://proceedings.mlr.press/v119/engstrom20a/engstrom20a.pdf (11 pages; index
  page https://proceedings.mlr.press/v119/engstrom20a.html)
- Bib fields: OK. PMLR citation metadata: title "Identifying Statistical Bias in Dataset Replication",
  authors Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Jacob Steinhardt,
  Aleksander Madry, pages 2922-2932, International Conference on Machine Learning, PMLR, 2020 (v119).
  All match. The paper's own byline prints "Aleksander Mądry" (with ogonek); the PMLR index and the bib
  both use "Madry", which is acceptable. arXiv 2005.09619 confirmed as the same paper.
- Definitive version cited? Yes (ICML / PMLR 119, not arXiv).
- Claims:
  - [SUPPORTED] "[Engstrom et al.] measure the bias of a replicated test set in the same spirit" —
    evidence: "we present unintuitive yet significant ways in which standard approaches to dataset
    replication introduce statistical bias ... We study ImageNet-v2 ... after remeasuring selection
    frequencies and correcting for statistical bias, only an estimated 3.6%±1.5% of the original
    11.7%±1.0% accuracy drop remains unaccounted for" (Abstract, p. 1); mechanism: "mean-zero noise in
    selection frequency readings leads to bias in the selection frequencies of the replicated dataset"
    (Section 1). They both identify and quantify the bias, so "measure" is accurate.
- Required fix: none.

### adebayo2022post
- Full text opened: https://arxiv.org/pdf/2212.04629 (32 pages; first page reads "Published as a
  conference paper at ICLR 2022"; arXiv API journal_ref "ICLR 2022 conference paper"). OpenReview
  (https://openreview.net/pdf?id=xNOVfCCvDpM and its API) returned a bot-challenge page, so the
  OpenReview copy itself could not be fetched; the arXiv copy is the camera-ready with the ICLR header.
- Bib fields: one discrepancy. The paper's byline is "Julius Adebayo, Michael Muelly, Hal Abelson,
  Been Kim" (p. 1; arXiv metadata identical). The bib has "Abelson, Harold". "Hal" is the name under
  which this author publishes and the name printed on the paper; change to "Abelson, Hal". Title,
  venue (ICLR 2022), year, OpenReview id and arXiv id 2212.04629 are correct. (Note for the record:
  arXiv 2011.05429 is a different paper, "Debugging Tests for Model Explanations", Adebayo, Muelly,
  Liccardi, Kim, NeurIPS 2020; the bib correctly does NOT point at it.)
- Definitive version cited? Yes (ICLR 2022 via booktitle + OpenReview URL; the arXiv id is a
  supplementary pointer).
- Claims:
  - [SUPPORTED] "Planting an effect of known size and measuring its recovery is ... in machine
    learning, the planted artefact" — evidence: "We design an empirical methodology that uses
    semi-synthetic datasets along with pre-specified spurious artifacts to obtain models that verifiably
    rely on these spurious training signals" (Abstract); "We consider 3 (2 visible and 1 non-visible)
    kinds of spurious signals (See Figure 1): i) a localized tag; ii) a distinctive stripped pattern;
    and iii) Gaussian blur applied to the image background" (Section 2, Experimental Methodology);
    "Tag for the 'MGH' hospital tag added to the pre-puberty class" (Section 3, Setup); recovery is
    scored by the "Known Spurious Signal Detection Measure (K-SSD)" (Section 1.1). Caveat: the planted
    artefact is known in identity and location rather than in "size"; its strength is quantified
    after the fact by their "spurious score". The analogy in the sentence (plant a known effect, test
    whether the instrument recovers it) holds; "of known size" describes the astronomy and genomics
    cases precisely and the Adebayo case only approximately. No change strictly required.
- Required fix: bib entry, replace `author = {Adebayo, Julius and Muelly, Michael and Abelson, Harold
  and Kim, Been}` with `author = {Adebayo, Julius and Muelly, Michael and Abelson, Hal and Kim, Been}`.

### welch1947generalization
- Full text opened: YES, 2026-10-05, all 8 pages (pp. 28-35), through the NYCU library (Primo -> LibKey -> OpenAthens
  -> Oxford Academic, https://academic.oup.com/biomet/article/34/1-2/28/210174, PDF 34-1-2-28.pdf) in the author's
  logged-in Chrome. Earlier attempts without the library (OUP, JSTOR, Semantic Scholar, Internet Archive) found no
  open copy. First page: "THE GENERALIZATION OF 'STUDENT'S' PROBLEM WHEN SEVERAL DIFFERENT POPULATION VARIANCES ARE
  INVOLVED, By B. L. Welch, B.A., Ph.D.", page [28]; last page 35 ends with the references; running head "Biometrika 34".
- Claim check: "Two-sided Welch tests [@welch1947generalization] on per-seed values compare penalties" - SUPPORTED. Sec. 1
  sets up "samples of n1 and n2 ... from two normal populations with ... standard deviations sigma1 and sigma2" and asks
  for h with Pr[(y - eta) < h(s1^2, ..., P)] = P (eq. 2); Sec. 4 gives the criterion v = (y - eta)/sqrt(lambda1 s1^2 +
  lambda2 s2^2) "follows approximately the 'Student' t-distribution with degrees of freedom" f = (lambda1 sigma1^2 +
  lambda2 sigma2^2)^2 / (lambda1^2 sigma1^4/f1 + lambda2^2 sigma2^4/f2) (eqs. 25-26, p. 32), with the sample estimate
  in eq. 29. That is the unequal-variances t-test the manuscript uses.
- Bib fields: OK against the publisher record
  (https://api.crossref.org/works/10.1093/biomet/34.1-2.28): author B. L. Welch, Biometrika 34(1-2):
  28-35, 1947, DOI 10.1093/biomet/34.1-2.28. Crossref's title string is an OCR artefact
  ("...POPULATION VARLANCES ARE INVOLVED"); the bib's "Variances" is the correct spelling. The bib's
  quotation marks around Student's (`Student's') render as the single quotes the original uses.
- Definitive version cited? Yes (Biometrika).
- Claims:
  - [SUPPORTED, on title and publisher record only; full text not opened] "Two-sided Welch tests
    [@welch1947generalization] on per-seed values compare penalties ..." — the paper is the standard
    citation for the unequal-variance t test's approximate degrees of freedom (the Welch-Satterthwaite
    solution to "Student's problem when several different population variances are involved", per the
    title). This is a methods attribution, not a claim about the paper's findings, and the title and
    bibliographic record are consistent with it. For completeness: Welch's 1938 Biometrika paper
    ("The significance of the difference between two means when the population variances are unequal",
    29:350-362) is the two-sample origin; the 1947 paper is the general-k derivation and is the one
    SciPy and most textbooks cite for `ttest_ind(equal_var=False)`. Either is acceptable; 1947 is fine.
- Required fix: none (full-text confirmation left to the author if the library copy is wanted).

## Summary

| key | full text opened | bib OK | claims (supported / partly / not) |
|---|---|---|---|
| ojala2010permutation | Y | Y | 1 / 0 / 0 |
| bernardi2020geometry | Y | Y | 0 / 1 / 0 |
| petigura2013prevalence | Y | Y | 1 / 0 / 0 |
| jiang2011synthetic | Y | Y | 1 / 0 / 0 |
| engstrom2020identifying | Y | Y | 1 / 0 / 0 |
| adebayo2022post | Y | N (Abelson given name) | 1 / 0 / 0 |
| welch1947generalization | N (paywalled; verified on publisher record) | Y | 1 / 0 / 0 |

## Required fixes

1. `adebayo2022post` bib entry: `Abelson, Harold` -> `Abelson, Hal` (the name printed on the ICLR
   paper and in arXiv metadata).
2. Citing sentence for `bernardi2020geometry` (recommended, not mandatory): replace
   "Permutation nulls that preserve structure are standard for classifier evaluation
   [@ojala2010permutation] and test generalisation across conditions in neuroscience
   [@bernardi2020geometry]."
   with
   "Permutation nulls that preserve structure are standard for classifier evaluation
   [@ojala2010permutation], and shuffle and random-geometry nulls calibrate cross-condition
   generalisation in neuroscience [@bernardi2020geometry]."
   Reason: Bernardi et al. calibrate CCGP with a geometric random model (random cluster centres plus
   per-condition axis permutation of the noise clouds), and use the count-preserving label shuffle for
   decoding accuracy and the parallelism score, not for CCGP.
3. `welch1947generalization`: no fix; full text could not be opened without a library login (URL above).
